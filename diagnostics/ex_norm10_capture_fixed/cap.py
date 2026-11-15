#!/usr/bin/env python3
"""
Gemma 2B GGUF → RMSNorm activation capture (standalone, 4 bugs fixed)
- GGUF + torch + numpy + gguf 만 필수 (transformers / sentencepiece 선택)
- lzero_npu_system_sw 파이프라인 미사용
- 출력: diagnostics/norm10_large_calibration/ 에만 저장
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gguf import GGUFReader
from gguf.quants import dequantize

# ══════════════════════════════════════════════════════════════
# 0. 유틸리티
# ══════════════════════════════════════════════════════════════

# FIXED (bug 4): np.percentile 사용 (int 근사 제거)
def quantile_sorted(sorted_arr: np.ndarray, q: float) -> float:
    """정렬된 1-D 배열에서 q 분위수 반환 (numpy percentile 기반, 정확)"""
    if len(sorted_arr) == 0:
        return 0.0
    # q는 (0,1) 범위의 분위수 → percentile은 0..100
    return float(np.percentile(sorted_arr, q * 100.0))


def summarize_1d(arr: np.ndarray) -> Dict[str, float]:
    """1-D float32 배열의 absmax, p99, p999, p9999, mean, std 계산"""
    if arr.size == 0:
        return {"absmax": 0.0, "p99": 0.0, "p999": 0.0, "p9999": 0.0,
                "mean": 0.0, "std": 0.0}
    abs_arr = np.abs(arr)
    sorted_abs = np.sort(abs_arr)
    return {
        "absmax": float(sorted_abs[-1]),
        "p99":    quantile_sorted(sorted_abs, 0.99),
        "p999":   quantile_sorted(sorted_abs, 0.999),
        "p9999":  quantile_sorted(sorted_abs, 0.9999),
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
    }


# FIXED (bug 1): cap 로직 - s_out <= cap_value 일 때 floor 로 고정하지 않음
# GeLU 때와 동일: s_out > 0.08 → 0.08, 아니면 max(s_out, 0.01)
def cap_scale(s_out: float, cap_value: float = 0.08, floor_value: float = 0.01) -> float:
    if s_out > cap_value:
        return cap_value
    return max(s_out, floor_value)


# ══════════════════════════════════════════════════════════════
# FIXED (bug 2): RoPE 구현 (Gemma 2B: head_dim=256, base=10000)
# ══════════════════════════════════════════════════════════════

def compute_rope_cache(seq_len: int, head_dim: int, base: float, device: torch.device):
    """[seq_len, head_dim] 크기의 (cos, sin) 반환. HF Gemma와 동일한 non-interleaved 방식."""
    half = head_dim // 2
    # inv_freq: [head_dim/2]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    # positions: [seq_len]
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    # freqs: [seq_len, head_dim/2]
    freqs = torch.outer(t, inv_freq)
    # emb: [seq_len, head_dim] (= [freqs, freqs] concat)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[..., D] → [..., D], rotate_half(x) = cat([-x2, x1])"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: [B, H, S, D]
    cos, sin: [S, D] → broadcast to [1, 1, S, D]
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ══════════════════════════════════════════════════════════════
# 1. GGUF 로더
# ══════════════════════════════════════════════════════════════

class GGUFWeightLoader:
    def __init__(self, gguf_path: str):
        self.reader = GGUFReader(gguf_path)
        self.tensor_map = {t.name: t for t in self.reader.tensors}

    def get(self, name: str, dtype=torch.float32) -> torch.Tensor:
        if name not in self.tensor_map:
            raise KeyError(
                f"Tensor not found: {name}. 예시: {list(self.tensor_map)[:5]}"
            )
        t = self.tensor_map[name]
        if t.tensor_type.name == "F32":
            arr = np.asarray(t.data)
        else:
            arr = dequantize(t.data, t.tensor_type)
        arr = np.asarray(arr)
        # make writable copy if needed
        if not arr.flags.writeable:
            arr = arr.copy()
        if dtype==torch.float16:
            arr = arr.astype(np.float16)
        elif dtype==torch.bfloat16:
            arr = arr.astype(np.float32)  # numpy no bf16, will convert via torch
        else:
            arr = arr.astype(np.float32)
        t_torch = torch.from_numpy(arr)
        if dtype==torch.bfloat16:
            t_torch = t_torch.to(dtype)
        return t_torch

    def names(self) -> List[str]:
        return list(self.tensor_map.keys())


# ══════════════════════════════════════════════════════════════
# 2. CapturingRMSNorm
# ══════════════════════════════════════════════════════════════

class CapturingRMSNorm(nn.Module):
    """Gemma RMSNorm: weight * (x / sqrt(mean(x^2) + eps)), forward 진입 시 입력 tensor 저장"""

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps
        self._captured: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._captured.append(x.detach())  # 저장 (device 유지)
        var = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm

    def pop_captured(self) -> List[torch.Tensor]:
        out = self._captured
        self._captured = []
        return out


# ══════════════════════════════════════════════════════════════
# 3. Gemma 2B 수동 모델 (RoPE 적용)
# ══════════════════════════════════════════════════════════════

class Gemma2BForCapture(nn.Module):
    """
    Gemma 2B: 18 layers, hidden=2048, inter=16384, heads=8, kv_heads=1, head_dim=256
    """

    def __init__(self, loader: GGUFWeightLoader,
                 num_layers: int = 18, hidden_dim: int = 2304,
                 num_heads: int = 8, num_kv_heads: int = 4, head_dim: int = 256,
                 intermediate_dim: int = 9216, vocab_size: int = 256000,
                 eps: float = 1e-6, rope_base: float = 10000.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_dim = intermediate_dim
        self.vocab_size = vocab_size
        self.eps = eps
        self.rope_base = rope_base
        self._rope_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}

        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.embed_tokens.weight.data.copy_(loader.get("token_embd.weight"))

        # Layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                "attn_norm":   CapturingRMSNorm(hidden_dim, eps),  # = input_layernorm
                "ffn_norm":    CapturingRMSNorm(hidden_dim, eps),  # = post_attention_layernorm
                "attn_q":      nn.Linear(hidden_dim, num_heads    * head_dim, bias=False),
                "attn_k":      nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False),
                "attn_v":      nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False),
                "attn_output": nn.Linear(num_heads    * head_dim, hidden_dim,  bias=False),
                "ffn_gate":    nn.Linear(hidden_dim, intermediate_dim, bias=False),
                "ffn_up":      nn.Linear(hidden_dim, intermediate_dim, bias=False),
                "ffn_down":    nn.Linear(intermediate_dim, hidden_dim, bias=False),
            })
            layer["attn_norm"].weight.data.copy_(loader.get(f"blk.{i}.attn_norm.weight"))
            layer["ffn_norm"].weight.data.copy_(loader.get(f"blk.{i}.ffn_norm.weight"))
            layer["attn_q"].weight.data.copy_(loader.get(f"blk.{i}.attn_q.weight"))
            layer["attn_k"].weight.data.copy_(loader.get(f"blk.{i}.attn_k.weight"))
            layer["attn_v"].weight.data.copy_(loader.get(f"blk.{i}.attn_v.weight"))
            layer["attn_output"].weight.data.copy_(loader.get(f"blk.{i}.attn_output.weight"))
            layer["ffn_gate"].weight.data.copy_(loader.get(f"blk.{i}.ffn_gate.weight"))
            layer["ffn_up"].weight.data.copy_(loader.get(f"blk.{i}.ffn_up.weight"))
            layer["ffn_down"].weight.data.copy_(loader.get(f"blk.{i}.ffn_down.weight"))
            self.layers.append(layer)

        # Final norm
        self.final_norm = CapturingRMSNorm(hidden_dim, eps)  # = final_norm
        self.final_norm.weight.data.copy_(loader.get("output_norm.weight"))

        # LM head
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if "output.weight" in loader.tensor_map:
            self.lm_head.weight.data.copy_(loader.get("output.weight"))
        else:
            self.lm_head.weight = self.embed_tokens.weight

    # ── RoPE 캐시 ─────────────────────────────────────
    def _get_rope(self, needed_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (needed_len, str(device))
        if key not in self._rope_cache:
            cos, sin = compute_rope_cache(needed_len, self.head_dim, self.rope_base, device)
            self._rope_cache[key] = (cos, sin)
        return self._rope_cache[key]

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        input_ids: [B, S]
        start_pos: RoPE 위치 오프셋 (prefill은 0, decode는 실제 위치)
        return: logits [B, S, vocab]
        """
        B, S = input_ids.shape
        device = input_ids.device
        x = self.embed_tokens(input_ids)  # [B, S, H]

        # RoPE cos/sin: 필요한 최대 위치 = start_pos + S
        cos, sin = self._get_rope(start_pos + S, device)
        cos = cos[start_pos:start_pos + S]  # [S, D]
        sin = sin[start_pos:start_pos + S]

        for layer in self.layers:
            # ── Pre-attention RMSNorm (captured before RoPE) ──
            normed = layer["attn_norm"](x)   # [B, S, H]

            # Projections
            q = layer["attn_q"](normed).view(B, S, self.num_heads,    self.head_dim).transpose(1, 2)
            k = layer["attn_k"](normed).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = layer["attn_v"](normed).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # FIXED (bug 2): RoPE on q and k
            q, k = apply_rope(q, k, cos, sin)

            # GQA repeat
            if self.num_kv_heads != self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)

            # Causal self-attention
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, H, S, D]
            attn = attn.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
            attn = layer["attn_output"](attn)
            x = x + attn

            # ── Pre-FFN RMSNorm (captured after attention) ──
            normed = layer["ffn_norm"](x)

            # MLP (GeGLU)
            gate = F.gelu(layer["ffn_gate"](normed))
            up   = layer["ffn_up"](normed)
            mlp  = layer["ffn_down"](gate * up)
            x = x + mlp

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


# ══════════════════════════════════════════════════════════════
# 4. 대화 데이터 (oasst1 스타일 64개)
# ══════════════════════════════════════════════════════════════

def build_oasst1_style_prompts(num_conversations: int = 64,
                               prefill_lengths: List[int] = [32, 128, 512, 1024],
                               decode_lengths: List[int] = [16, 24, 32]) -> List[Dict]:
    base = [
        "Hello, how can I help you today?",
        "I'm looking for information about artificial intelligence.",
        "Could you explain the difference between machine learning and deep learning?",
        "What are the benefits of using renewable energy sources?",
        "Can you recommend a good book on computer science?",
        "How does photosynthesis work in plants?",
        "What is the capital of France?",
        "Tell me a fun fact about space exploration.",
        "How do I improve my programming skills?",
        "What are the main causes of climate change?",
        "Can you help me write a cover letter for a job application?",
        "What is the history of the internet?",
        "How do vaccines work in the human body?",
        "What are some tips for learning a new language?",
        "Explain the concept of blockchain technology.",
        "What is the difference between HTTP and HTTPS?",
        "How do I bake a chocolate cake?",
        "What are the health benefits of meditation?",
        "Can you summarize the plot of Hamlet?",
        "What is the significance of the Turing test?",
    ]
    random.seed(42)
    out = []
    for conv_id in range(num_conversations):
        pl = prefill_lengths[conv_id % len(prefill_lengths)]
        dl = decode_lengths[conv_id % len(decode_lengths)]
        target_chars = pl * 4
        parts = []
        while sum(len(s) for s in parts) < target_chars:
            parts.append(random.choice(base))
        prompt = " ".join(parts)[:target_chars]
        out.append({
            "id": conv_id, "prompt_text": prompt,
            "prefill_len": pl, "decode_len": dl,
        })
    return out


# ══════════════════════════════════════════════════════════════
# FIXED (bug 3): 토크나이저 — transformers → sentencepiece → char fallback
# ══════════════════════════════════════════════════════════════

class GemmaTokenizer:
    def __init__(self):
        self.mode = None
        self.vocab_size = None
        self.tok = None
        self.sp = None

        # 1) transformers.AutoTokenizer
        try:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained("google/gemma-2b-it")
            self.vocab_size = len(self.tok)
            self.mode = "transformers"
            print(f"[*] Tokenizer: transformers (vocab={self.vocab_size})")
            return
        except Exception as e:
            print(f"[!] transformers tokenizer unavailable: {e}")

        # 2) sentencepiece + .model 파일
        try:
            import sentencepiece as spm
            candidates = ["models/gemma-2b-it.model", "models/tokenizer.model",
                          "tokenizer.model"]
            for p in candidates:
                if os.path.exists(p):
                    self.sp = spm.SentencePieceProcessor(model_file=p)
                    self.vocab_size = self.sp.get_piece_size()
                    self.mode = "sentencepiece"
                    print(f"[*] Tokenizer: sentencepiece from {p} (vocab={self.vocab_size})")
                    return
            print("[!] sentencepiece installed but no .model file found")
        except ImportError:
            print("[!] sentencepiece not installed")

        # 3) Char-level fallback (경고)
        print("[!] WARNING: char-level fallback tokenizer — statistics will NOT match real Gemma distribution!")
        self.mode = "char"
        self.vocab_size = 256

    def encode(self, text: str, max_length: int) -> List[int]:
        if self.mode == "transformers":
            ids = self.tok.encode(text, add_special_tokens=True)
            return ids[:max_length]
        elif self.mode == "sentencepiece":
            ids = self.sp.encode(text, out_type=int)
            return ids[:max_length]
        else:
            return [ord(c) % 256 for c in text[:max_length]]


# ══════════════════════════════════════════════════════════════
# 5. 캡처 실행 (prefill + decode, growing prefix)
# ══════════════════════════════════════════════════════════════

def _collect_norm_captures(model: Gemma2BForCapture,
                           group: str,
                           records: List[Dict],
                           row_counter: int,
                           take_all_positions: bool) -> int:
    """모든 Norm 모듈에서 캡처된 tensor를 record로 변환. row_counter 갱신."""
    for layer_idx, layer in enumerate(model.layers):
        for norm_type in ["attn_norm", "ffn_norm"]:
            for t in layer[norm_type].pop_captured():
                # t: [1, S, H] (B=1 가정)
                t_2d = t.squeeze(0).cpu().numpy().astype(np.float32)  # [S, H]
                rows = range(t_2d.shape[0]) if take_all_positions else [t_2d.shape[0] - 1]
                for r in rows:
                    stats = summarize_1d(t_2d[r])
                    records.append({
                        "layer_name": f"blk.{layer_idx}.{norm_type}",
                        "norm_type": norm_type,
                        "group": group,
                        "row_id": row_counter,
                        **stats,
                    })
                    row_counter += 1

    for t in model.final_norm.pop_captured():
        t_2d = t.squeeze(0).cpu().numpy().astype(np.float32)
        rows = range(t_2d.shape[0]) if take_all_positions else [t_2d.shape[0] - 1]
        for r in rows:
            stats = summarize_1d(t_2d[r])
            records.append({
                "layer_name": "output_norm",
                "norm_type": "final_norm",
                "group": group,
                "row_id": row_counter,
                **stats,
            })
            row_counter += 1
    return row_counter


def run_capture(model: Gemma2BForCapture,
                tokenizer: GemmaTokenizer,
                conversations: List[Dict],
                device: torch.device,
                max_seq_len: int = 2048) -> List[Dict]:
    records: List[Dict] = []
    row_counter = 0
    model.eval().to(device)

    for conv in conversations:
        input_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        if not input_ids:
            continue

        # ── Prefill ──
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _ = model(input_tensor, start_pos=0)

        # prefill: 모든 위치 캡처
        row_counter = _collect_norm_captures(model, "prefill", records, row_counter,
                                             take_all_positions=True)

        # ── Decode (growing prefix; greedy next token) ──
        current_ids = list(input_ids)
        for step in range(conv["decode_len"]):
            if len(current_ids) >= max_seq_len:
                break
            cur = torch.tensor([current_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(cur, start_pos=0)

            # decode: 마지막 위치만 캡처 (이번에 생성된 새 토큰의 activation)
            row_counter = _collect_norm_captures(model, "decode", records, row_counter,
                                                 take_all_positions=False)

            # greedy next token
            next_tok = int(logits[0, -1].argmax().item())
            current_ids.append(next_tok)

    return records


# ══════════════════════════════════════════════════════════════
# 6. Summary CSV (per-layer)
# ══════════════════════════════════════════════════════════════

def build_summary_csv(records: List[Dict], output_path: str):
    """
    per-layer 집계: absmax = max(rows), p99/p999/p9999 = max(rows) (보수적), mean/std = mean(rows)
    outlier_ratio = p999/absmax
    s_in_candidate = p999/127
    s_out_candidate = cap_scale(p999/511)  # > 0.08 → 0.08, 아니면 max(s_out, 0.01)
    """
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        groups[r["layer_name"]].append(r)

    fieldnames = [
        "layer_name", "absmax", "p99", "p999", "p9999", "mean", "std",
        "outlier_ratio", "s_in_candidate", "s_out_candidate",
    ]

    rows_out = []
    for layer_name in sorted(groups.keys(), key=_sort_key):
        recs = groups[layer_name]
        absmax = max(r["absmax"] for r in recs)
        p99    = max(r["p99"]    for r in recs)
        p999   = max(r["p999"]   for r in recs)
        p9999  = max(r["p9999"]  for r in recs)
        mean   = float(np.mean([r["mean"] for r in recs]))
        std    = float(np.mean([r["std"]  for r in recs]))

        outlier_ratio = p999 / absmax if absmax > 0 else 0.0
        s_in_candidate = p999 / 127.0
        s_out_raw = p999 / 511.0
        s_out_candidate = cap_scale(s_out_raw, cap_value=0.08, floor_value=0.01)

        rows_out.append({
            "layer_name": layer_name,
            "absmax": absmax, "p99": p99, "p999": p999, "p9999": p9999,
            "mean": mean, "std": std,
            "outlier_ratio": outlier_ratio,
            "s_in_candidate": s_in_candidate,
            "s_out_candidate": s_out_candidate,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"[✓] Summary CSV saved: {output_path} ({len(rows_out)} layers)")


def _sort_key(name: str):
    """output_norm을 마지막에, 나머지는 blk 인덱스 순서로"""
    if name == "output_norm":
        return (999, 0, name)
    try:
        parts = name.split(".")
        idx = int(parts[1])
        suffix = ".".join(parts[2:])
        sub = {"attn_norm": 0, "ffn_norm": 1}.get(suffix, 2)
        return (idx, sub, name)
    except Exception:
        return (998, 0, name)


# ══════════════════════════════════════════════════════════════
# 7. 메인
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Gemma 2B RMSNorm activation capture")
    ap.add_argument("--gguf", type=str, default="models/gemma-2b-q8_0.gguf")
    ap.add_argument("--output_dir", type=str, default="diagnostics/norm10_large_calibration")
    ap.add_argument("--num_conversations", type=int, default=64)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float32",
                    choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading GGUF: {args.gguf}")
    loader = GGUFWeightLoader(args.gguf)
    print(f"[*] Tensors in GGUF: {len(loader.names())}")

    print("[*] Building Gemma 2B model...")
    model = Gemma2BForCapture(loader)
    model.eval()

    print("[*] Preparing tokenizer...")
    tokenizer = GemmaTokenizer()

    convs = build_oasst1_style_prompts(
        num_conversations=args.num_conversations,
        prefill_lengths=[32, 128, 512, 1024],
        decode_lengths=[16, 24, 32],
    )
    total_prefill = sum(c["prefill_len"] for c in convs)
    print(f"[*] {len(convs)} conversations, total prefill ≈ {total_prefill} tokens")
    if total_prefill < 8000:
        print("[!] WARNING: prefill rows < 8k. Increase --num_conversations or prefill lengths.")

    device = torch.device(args.device)
    print(f"[*] Capture on {device} ...")
    records = run_capture(model, tokenizer, convs, device)
    print(f"[*] Captured {len(records)} rows (prefill + decode)")

    jsonl_path = out_dir / "norm_capture_64.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[✓] JSONL saved: {jsonl_path}")

    csv_path = out_dir / "norm_summary.csv"
    build_summary_csv(records, str(csv_path))

    print("\n[✓] Done.")
    print(f"    - {jsonl_path}")
    print(f"    - {csv_path}")


if __name__ == "__main__":
    main()