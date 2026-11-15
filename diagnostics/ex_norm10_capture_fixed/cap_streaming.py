#!/usr/bin/env python3
"""
Streaming Gemma 2B RMSNorm activation capture.

목표: peak RSS < 2GB (15GB 머신에서도 안전).
  - 레이어당 가중치 9개만 dequantize 후 즉시 사용·해제
  - embedding은 초기에 1회 로드 (≈1.18GB f16)
  - final_norm은 마지막에만 로드 (크기 무시 가능)
  - lm_head 미로드: decode 위치는 "prefill 뒤에 유효 랜덤 토큰을 append한
    단일 causal forward"로 캡처 (autoregressive 통계와 사실상 동일)

버그 4개 수정 유지:
  1) quantile: np.percentile
  2) cap_scale: >0.08 → 0.08, else max(s_out, 0.01)
  3) RoPE: compute_rope_cache / rotate_half / apply_rope
  4) tokenizer: transformers → sentencepiece → char (경고)

출력: <output_dir>/norm_capture_64.jsonl, <output_dir>/norm_summary.csv
"""

import argparse
import csv
import gc
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import gguf
from gguf import GGUFReader
from gguf.quants import dequantize


# ══════════════════════════════════════════════════════════════
# 0. 유틸리티 (기존 버그 수정 유지)
# ══════════════════════════════════════════════════════════════

def quantile_sorted(sorted_arr: np.ndarray, q: float) -> float:
    if len(sorted_arr) == 0:
        return 0.0
    return float(np.percentile(sorted_arr, q * 100.0))  # FIXED bug 4


def summarize_1d(arr: np.ndarray) -> Dict[str, float]:
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


def cap_scale(s_out: float, cap_value: float = 0.08, floor_value: float = 0.01) -> float:
    # FIXED bug 1: > cap → cap, else max(s_out, floor)
    if s_out > cap_value:
        return cap_value
    return max(s_out, floor_value)


def compute_rope_cache(seq_len, head_dim, base, device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)                       # [S, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)                # [S, D]
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # FIXED bug 2
    seq_len = q.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0).to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ══════════════════════════════════════════════════════════════
# 1. Config (사용자 확인값)
# ══════════════════════════════════════════════════════════════

class Config:
    hidden_dim       = 2304
    intermediate_dim = 9216
    num_heads        = 8
    num_kv_heads     = 4
    head_dim         = 256
    vocab_size       = 256000
    eps              = 1e-6
    rope_base        = 10000.0
    num_layers       = 18          # auto-detect로 재설정

    @property
    def q_dim(self):  return self.num_heads * self.head_dim       # 2048
    @property
    def kv_dim(self): return self.num_kv_heads * self.head_dim    # 1024


def detect_num_layers(tensor_names) -> int:
    """GGUF 텐서 이름에서 blk.{i}의 최대 i + 1을 반환."""
    mx = -1
    for n in tensor_names:
        if n.startswith("blk."):
            try:
                i = int(n.split(".")[1])
                mx = max(mx, i)
            except (ValueError, IndexError):
                pass
    return mx + 1


# ══════════════════════════════════════════════════════════════
# 2. Streaming Gemma forward
# ══════════════════════════════════════════════════════════════

class StreamingGemma:
    """
    한 번에 한 레이어의 가중치만 dequantize → forward → 즉시 해제.
    activation 캡처도 레이어 직후 즉시 f32 numpy로 변환 후 torch tensor 폐기.
    """

    def __init__(self, gguf_path: str, dtype: torch.dtype = torch.float16, device="cpu"):
        self.reader = GGUFReader(gguf_path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.dtype = dtype
        self.device = torch.device(device)
        self.cfg = Config()
        self.cfg.num_layers = detect_num_layers(self.tensors.keys())
        self._np_dtype = np.float16 if dtype == torch.float16 else np.float32
        self._embedding: Optional[torch.Tensor] = None
        self._rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        print(f"[*] detected num_layers = {self.cfg.num_layers}")
        print(f"[*] vocab_size = {self.cfg.vocab_size}, hidden = {self.cfg.hidden_dim}, "
              f"q_dim = {self.cfg.q_dim}, kv_dim = {self.cfg.kv_dim}")

    # ── 가중치 로드 (dequantize + f16 변환 후 원본 즉시 해제) ──
    def _load_weight(self, name: str) -> torch.Tensor:
        if name not in self.tensors:
            raise KeyError(f"Tensor not found: {name}")

        t = self.tensors[name]
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            arr = t.data
        else:
            arr = dequantize(t.data, t.tensor_type)   # float32 반환

        # F16으로 직접 변환 + 복사 (원본 dequantize 결과 즉시 해제 가능)
        arr_cast = arr.astype(self._np_dtype, copy=True)
        del arr
        gc.collect()

        return torch.from_numpy(arr_cast).to(self.device)

    def _load_linear(self, name: str, out_features: int, in_features: int) -> torch.Tensor:
        w = self._load_weight(name)
        if w.shape == (out_features, in_features):
            return w
        if w.shape == (in_features, out_features):
            return w.T.contiguous()
        raise ValueError(f"{name}: shape {w.shape}, expected ({out_features}, {in_features})")

    def _load_norm(self, name: str) -> torch.Tensor:
        w = self._load_weight(name)
        if w.numel() != self.cfg.hidden_dim:
            raise ValueError(f"{name}: numel {w.numel()}, expected {self.cfg.hidden_dim}")
        return w.reshape(self.cfg.hidden_dim)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._embedding is None:
            w = self._load_weight("token_embd.weight")
            if w.shape == (self.cfg.vocab_size, self.cfg.hidden_dim):
                self._embedding = w
            elif w.shape == (self.cfg.hidden_dim, self.cfg.vocab_size):
                self._embedding = w.T.contiguous()
            else:
                raise ValueError(f"embedding shape {w.shape} not compatible")
        return F.embedding(input_ids.to(self.device), self._embedding)

    # ── RMSNorm (f32 internally for precision) ──
    def _rmsnorm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        var = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(var + self.cfg.eps)
        return (weight.float() * x_norm).to(x.dtype)

    # ── 단일 레이어 forward + activation 캡처 ──
    def _layer_forward(self, x: torch.Tensor, layer_idx: int,
                       rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        cfg = self.cfg
        B, S, H = x.shape
        p = f"blk.{layer_idx}"

        # attention 블록 가중치 (9개)
        attn_norm_w = self._load_norm(f"{p}.attn_norm.weight")
        q_w = self._load_linear(f"{p}.attn_q.weight",      cfg.q_dim,  cfg.hidden_dim)
        k_w = self._load_linear(f"{p}.attn_k.weight",      cfg.kv_dim, cfg.hidden_dim)
        v_w = self._load_linear(f"{p}.attn_v.weight",      cfg.kv_dim, cfg.hidden_dim)
        o_w = self._load_linear(f"{p}.attn_output.weight", cfg.hidden_dim, cfg.q_dim)

        # ★ 캡처: attn_norm의 INPUT (f32 CPU로 즉시 이동)
        cap_attn = x.detach().float().cpu().numpy()

        normed = self._rmsnorm(x, attn_norm_w)
        q = F.linear(normed, q_w).view(B, S, cfg.num_heads,    cfg.head_dim).transpose(1, 2)
        k = F.linear(normed, k_w).view(B, S, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = F.linear(normed, v_w).view(B, S, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, rope_cos, rope_sin)

        if cfg.num_kv_heads != cfg.num_heads:
            rep = cfg.num_heads // cfg.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, cfg.q_dim)
        attn = F.linear(attn, o_w)
        x = x + attn

        # attn 가중치 즉시 해제
        del attn_norm_w, q_w, k_w, v_w, o_w, normed, q, k, v, attn
        gc.collect()

        # FFN 블록 가중치
        ffn_norm_w = self._load_norm(f"{p}.ffn_norm.weight")
        gate_w = self._load_linear(f"{p}.ffn_gate.weight", cfg.intermediate_dim, cfg.hidden_dim)
        up_w   = self._load_linear(f"{p}.ffn_up.weight",   cfg.intermediate_dim, cfg.hidden_dim)
        down_w = self._load_linear(f"{p}.ffn_down.weight", cfg.hidden_dim, cfg.intermediate_dim)

        # ★ 캡처: ffn_norm의 INPUT
        cap_ffn = x.detach().float().cpu().numpy()

        normed = self._rmsnorm(x, ffn_norm_w)
        gate = F.gelu(F.linear(normed, gate_w))
        up   = F.linear(normed, up_w)
        mlp  = F.linear(gate * up, down_w)
        x = x + mlp

        del ffn_norm_w, gate_w, up_w, down_w, normed, gate, up, mlp
        gc.collect()

        return x, cap_attn, cap_ffn

    # ── 전체 forward (single pass, layer-streaming) ──
    def forward_capture(self, input_ids: torch.Tensor) -> List[Tuple[str, str, np.ndarray]]:
        """
        반환: list of (layer_name, norm_type, np.ndarray[S, H])
          layer_name: "blk.{i}" 또는 "output_norm"
          norm_type:  "attn_norm" | "ffn_norm" | "final_norm"
        """
        cfg = self.cfg
        B, S = input_ids.shape

        x = self._embed(input_ids)                              # [B, S, H]

        if self._rope_cache is None or self._rope_cache[0].shape[0] < S:
            self._rope_cache = compute_rope_cache(S, cfg.head_dim, cfg.rope_base, self.device)
        cos, sin = self._rope_cache

        out: List[Tuple[str, str, np.ndarray]] = []
        for i in range(cfg.num_layers):
            x, cap_attn, cap_ffn = self._layer_forward(x, i, cos, sin)
            # [B,S,H] → [S,H] (batch=1 가정)
            if cap_attn.ndim == 3: cap_attn = cap_attn[0]
            if cap_ffn.ndim  == 3: cap_ffn  = cap_ffn[0]
            out.append((f"blk.{i}", "attn_norm", cap_attn))
            out.append((f"blk.{i}", "ffn_norm",  cap_ffn))

        # final_norm: 가중치는 존재 확인만 하고 로드 후 즉시 해제 (크기 무시)
        _ = self._load_norm("output_norm.weight")
        del _
        gc.collect()

        cap_final = x.detach().float().cpu().numpy()
        if cap_final.ndim == 3: cap_final = cap_final[0]
        out.append(("output_norm", "final_norm", cap_final))

        # embedding 해제 (다음 대화에서 재로드 — 메모리 우선)
        del self._embedding
        self._embedding = None
        del x
        gc.collect()

        return out


# ══════════════════════════════════════════════════════════════
# 3. 데이터 / 토크나이저
# ══════════════════════════════════════════════════════════════

def build_conversations(n_convs: int,
                        prefill_lengths: List[int],
                        decode_lengths: List[int],
                        seed: int = 42) -> List[Dict]:
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
    rng = random.Random(seed)
    out = []
    for i in range(n_convs):
        pl = prefill_lengths[i % len(prefill_lengths)]
        dl = decode_lengths[i % len(decode_lengths)]
        target = pl * 4
        parts, total = [], 0
        while total < target:
            s = rng.choice(base)
            parts.append(s)
            total += len(s) + 1
        prompt = " ".join(parts)[:target]
        out.append({"id": i, "prompt_text": prompt,
                    "prefill_len": pl, "decode_len": dl})
    return out


class GemmaTokenizer:
    """transformers → sentencepiece → char (경고). FIXED bug 3."""
    def __init__(self):
        self.mode = None
        self.tok = None
        self.sp = None
        try:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained("google/gemma-2b-it")
            self.mode = "transformers"
            print(f"[*] Tokenizer: transformers (vocab={len(self.tok)})")
            return
        except Exception as e:
            print(f"[!] transformers unavailable: {e}")
        try:
            import sentencepiece as spm
            for p in ("models/gemma-2b-it.model", "models/tokenizer.model", "tokenizer.model"):
                if os.path.exists(p):
                    self.sp = spm.SentencePieceProcessor(model_file=p)
                    self.mode = "sentencepiece"
                    print(f"[*] Tokenizer: sentencepiece from {p}")
                    return
        except ImportError:
            pass
        print("[!] WARNING: char-level fallback — stats will NOT match real Gemma!")
        self.mode = "char"

    def encode(self, text: str, max_length: int) -> List[int]:
        if self.mode == "transformers":
            return self.tok.encode(text, add_special_tokens=True)[:max_length]
        if self.mode == "sentencepiece":
            return self.sp.encode(text, out_type=int)[:max_length]
        return [ord(c) % 256 for c in text[:max_length]]


# ══════════════════════════════════════════════════════════════
# 4. 캡처 루프
# ══════════════════════════════════════════════════════════════

def run_capture(model: StreamingGemma,
                tokenizer: GemmaTokenizer,
                conversations: List[Dict],
                model_vocab_size: int,
                seed: int = 42) -> List[Dict]:
    records: List[Dict] = []
    row_counter = 0
    rng = random.Random(seed)

    for conv in conversations:
        prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        if not prompt_ids:
            continue
        prefill_len = len(prompt_ids)

        # decode: 유효 랜덤 토큰 append (lm_head 없이도 causal attention 통계 확보)
        decode_ids = [rng.randint(0, model_vocab_size - 1) for _ in range(conv["decode_len"])]
        full_ids = prompt_ids + decode_ids

        input_tensor = torch.tensor([full_ids], dtype=torch.long)
        captures = model.forward_capture(input_tensor)   # [(layer_name, norm_type, [S,H])]

        for layer_name, norm_type, arr in captures:
            S = arr.shape[0]
            for pos in range(S):
                group = "prefill" if pos < prefill_len else "decode"
                stats = summarize_1d(arr[pos].astype(np.float32))
                records.append({
                    "layer_name": layer_name,
                    "norm_type":  norm_type,
                    "group":      group,
                    "row_id":     row_counter,
                    **stats,
                })
                row_counter += 1

        # 메모리 압박 완화
        del captures, arr
        gc.collect()

    return records


# ══════════════════════════════════════════════════════════════
# 5. Summary CSV
# ══════════════════════════════════════════════════════════════

def _sort_key(name: str):
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


def build_summary_csv(records: List[Dict], output_path: str):
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        groups[r["layer_name"]].append(r)

    fieldnames = ["layer_name", "absmax", "p99", "p999", "p9999", "mean", "std",
                  "outlier_ratio", "s_in_candidate", "s_out_candidate"]

    rows_out = []
    for layer_name in sorted(groups.keys(), key=_sort_key):
        recs = groups[layer_name]
        absmax = max(r["absmax"] for r in recs)
        p99    = max(r["p99"]    for r in recs)
        p999   = max(r["p999"]   for r in recs)
        p9999  = max(r["p9999"]  for r in recs)
        mean   = float(np.mean([r["mean"] for r in recs]))
        std    = float(np.mean([r["std"]  for r in recs]))

        outlier_ratio   = p999 / absmax if absmax > 0 else 0.0
        s_in_candidate  = p999 / 127.0
        s_out_candidate = cap_scale(p999 / 511.0)

        rows_out.append({
            "layer_name": layer_name, "absmax": absmax,
            "p99": p99, "p999": p999, "p9999": p9999,
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


# ══════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Streaming Gemma 2B RMSNorm capture")
    ap.add_argument("--gguf", type=str,
                    default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--output_dir", type=str,
                    default="diagnostics/ex_norm10_capture_fixed")
    ap.add_argument("--num_conversations", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "float32"])
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    print(f"[*] gguf={args.gguf}")
    print(f"[*] dtype={args.dtype}, device={args.device}")

    model = StreamingGemma(args.gguf, dtype=dtype, device=args.device)

    print("[*] Preparing tokenizer...")
    tokenizer = GemmaTokenizer()

    convs = build_conversations(
        args.num_conversations,
        prefill_lengths=[32, 128, 512, 1024],
        decode_lengths=[16, 24, 32],
    )
    tot_pf = sum(c["prefill_len"] for c in convs)
    tot_dc = sum(c["decode_len"] for c in convs)
    print(f"[*] {len(convs)} conversations, total prefill={tot_pf}, decode={tot_dc}")
    if tot_pf + tot_dc < 8000:
        print("[!] WARNING: total tokens < 8k. Increase --num_conversations.")

    print("[*] Capturing (streaming, 1 layer at a time)...")
    records = run_capture(model, tokenizer, convs, model.cfg.vocab_size)

    jsonl_path = out_dir / "norm_capture_64.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[✓] JSONL saved: {jsonl_path} ({len(records)} rows)")

    csv_path = out_dir / "norm_summary.csv"
    build_summary_csv(records, str(csv_path))

    # peak RSS (Linux)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmPeak", "VmHWM")):
                    print(f"[mem] {line.strip()}")
    except Exception:
        pass

    print("[✓] Done.")


if __name__ == "__main__":
    main()