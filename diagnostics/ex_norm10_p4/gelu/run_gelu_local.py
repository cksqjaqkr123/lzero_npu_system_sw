#!/usr/bin/env python3
"""
GELU Local 캡처 (P4용, RMSNorm 방식과 별도)
- 위치: diagnostics/ex_norm10_p4/gelu/
- 목적: GELU는 RMSNorm이랑 달라서 s_in/s_out을 따로 재야 함
  RMSNorm: y = x / RMS(x) * gamma, BOS가 RMS를 깸
  GELU: y = GELU(gate) * up, gate = Linear(ffn_norm(x)), RMS 없음, gamma 없음
- 측정:
  1) gate 입력 분포: absmax / p999 / p9999 / mean / std, BOS vs non-BOS
  2) gate 양자화 → GELU 출력 NMSE (s_in sweep)
  3) GELU 출력 양자화 NMSE (s_out sweep)
- 출력: gelu_local.csv, gelu_local.json
"""
import argparse, gc, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations, compute_rope_cache, apply_rope

def quant_int10(x, s):
    # s: scale, x_fp / s -> RNE -> clip [-511, 511] (INT10 signed, 10-bit? 512 levels)
    # 여기서는 RMSNorm이랑 같은 방식: q = clamp(RNE(x/s), -511, 511)
    q = torch.round(x / s).clamp(-511, 511)
    return q * s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_p4/gelu")
    ap.add_argument("--n_conv", type=int, default=4)
    ap.add_argument("--prefill", type=int, default=128)
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    tokenizer = GemmaTokenizer()
    convs = build_conversations(args.n_conv, prefill_lengths=[args.prefill]*args.n_conv, decode_lengths=[8]*args.n_conv)
    # gate 수집: layer -> list of (is_bos, gate_values)
    from collections import defaultdict
    gate_by_layer = defaultdict(list)
    gate_bos_by_layer = defaultdict(list)
    for ci, conv in enumerate(convs):
        prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        B, S = input_tensor.shape
        cfg = model.cfg
        emb_w = model._load_weight("token_embd.weight")
        if emb_w.shape == (cfg.hidden_dim, cfg.vocab_size):
            emb_w = emb_w.T.contiguous()
        x = F.embedding(input_tensor.to(model.device), emb_w)
        if model._rope_cache is None or model._rope_cache[0].shape[0] < S:
            model._rope_cache = compute_rope_cache(S, cfg.head_dim, cfg.rope_base, model.device)
        cos, sin = model._rope_cache
        print(f"[*] conv{ci} S={S} capturing gate...")
        for i in range(cfg.num_layers):
            p = f"blk.{i}"
            attn_norm_w = model._load_norm(f"{p}.attn_norm.weight")
            q_w = model._load_linear(f"{p}.attn_q.weight", cfg.q_dim, cfg.hidden_dim)
            k_w = model._load_linear(f"{p}.attn_k.weight", cfg.kv_dim, cfg.hidden_dim)
            v_w = model._load_linear(f"{p}.attn_v.weight", cfg.kv_dim, cfg.hidden_dim)
            o_w = model._load_linear(f"{p}.attn_output.weight", cfg.hidden_dim, cfg.q_dim)
            normed = model._rmsnorm(x, attn_norm_w)
            q = F.linear(normed, q_w).view(B, S, cfg.num_heads, cfg.head_dim).transpose(1, 2)
            k = F.linear(normed, k_w).view(B, S, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
            v = F.linear(normed, v_w).view(B, S, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
            q, k = apply_rope(q, k, cos, sin)
            if cfg.num_kv_heads != cfg.num_heads:
                rep = cfg.num_heads // cfg.num_kv_heads
                k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(B, S, cfg.q_dim)
            attn = F.linear(attn, o_w); x = x + attn
            del attn_norm_w, q_w, k_w, v_w, o_w, normed, q, k, v, attn; gc.collect()
            # FFN: gate 캡처
            ffn_norm_w = model._load_norm(f"{p}.ffn_norm.weight")
            gate_w = model._load_linear(f"{p}.ffn_gate.weight", cfg.intermediate_dim, cfg.hidden_dim)
            up_w = model._load_linear(f"{p}.ffn_up.weight", cfg.intermediate_dim, cfg.hidden_dim)
            down_w = model._load_linear(f"{p}.ffn_down.weight", cfg.hidden_dim, cfg.intermediate_dim)
            normed = model._rmsnorm(x, ffn_norm_w)
            gate = F.linear(normed, gate_w)  # [B, S, intermediate]
            up = F.linear(normed, up_w)
            # gate 수집 (FP, CPU numpy)
            g = gate[0].detach().float().cpu().numpy()  # [S, intermediate]
            # BOS (pos 0) vs non-BOS 분리
            gate_bos_by_layer[i].append(g[0:1].copy())
            gate_by_layer[i].append(g[1:].copy())
            # FFN 나머지 진행
            mlp = F.linear(F.gelu(gate) * up, down_w); x = x + mlp
            del ffn_norm_w, gate_w, up_w, down_w, normed, gate, up, mlp; gc.collect()
        del emb_w, x; gc.collect()
    print(f"[*] gate 수집 완료, Local NMSE 계산 중...")
    import csv
    rows = []
    summary = {}
    for layer in sorted(gate_by_layer.keys()):
        non_bos = np.concatenate(gate_by_layer[layer], axis=0)  # [N, intermediate]
        bos = np.concatenate(gate_bos_by_layer[layer], axis=0)    # [n_conv, intermediate]
        # 통계: BOS vs non-BOS
        for tag, arr in [("non_bos", non_bos), ("bos", bos)]:
            flat = arr.flatten()
            absmax = float(np.max(np.abs(flat)))
            p999 = float(np.percentile(np.abs(flat), 99.9))
            p9999 = float(np.percentile(np.abs(flat), 99.99))
            rows.append({
                "layer": f"blk.{layer}",
                "tag": tag,
                "n_tokens": int(arr.shape[0]),
                "absmax": absmax,
                "p999": p999,
                "p9999": p9999,
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
                "bos_vs_nonbos_absmax_ratio": float(np.max(np.abs(bos)) / max(np.max(np.abs(non_bos)), 1e-9)) if tag=="bos" else None,
            })
        # s_in sweep: gate 양자화 → GELU 출력 NMSE (non-BOS만)
        # GELU 출력 = GELU(gate) (up는 아직 안 곱함, GELU 자체만)
        sample = non_bos[np.random.choice(non_bos.shape[0], min(200, non_bos.shape[0]), replace=False)]
        gelu_fp = F.gelu(torch.from_numpy(sample)).numpy()
        for cand, s_val in [
            ("p999", float(np.percentile(np.abs(non_bos), 99.9)) / 511),
            ("p9999", float(np.percentile(np.abs(non_bos), 99.99)) / 511),
            ("absmax", float(np.max(np.abs(non_bos))) / 511),
        ]:
            if s_val == 0:
                continue
            q_gate = np.round(sample / s_val).clip(-511, 511) * s_val
            gelu_q = F.gelu(torch.from_numpy(q_gate)).numpy()
            nmse = float(np.sum((gelu_q - gelu_fp)**2) / max(np.sum(gelu_fp**2), 1e-12))
            rows.append({
                "layer": f"blk.{layer}",
                "tag": f"gelu_nmse_sin_{cand}",
                "s_in": s_val,
                "nmse": nmse,
                "n_tokens": int(sample.shape[0]),
            })
        # s_out sweep: GELU 출력 자체를 양자화
        gelu_absmax = float(np.max(np.abs(gelu_fp)))
        gelu_p999 = float(np.percentile(np.abs(gelu_fp), 99.9))
        gelu_p9999 = float(np.percentile(np.abs(gelu_fp), 99.99))
        for cand, s_val in [
            ("p999", gelu_p999 / 511),
            ("p9999", gelu_p9999 / 511),
            ("absmax", gelu_absmax / 511),
        ]:
            if s_val == 0:
                continue
            q_out = np.round(gelu_fp / s_val).clip(-511, 511) * s_val
            nmse = float(np.sum((q_out - gelu_fp)**2) / max(np.sum(gelu_fp**2), 1e-12))
            rows.append({
                "layer": f"blk.{layer}",
                "tag": f"gelu_nmse_sout_{cand}",
                "s_out": s_val,
                "nmse": nmse,
                "gelu_absmax": gelu_absmax,
                "gelu_p999": gelu_p999,
                "gelu_p9999": gelu_p9999,
            })
        summary[f"blk.{layer}"] = {
            "non_bos_absmax": float(np.max(np.abs(non_bos))),
            "bos_absmax": float(np.max(np.abs(bos))),
            "bos_ratio": float(np.max(np.abs(bos)) / max(np.max(np.abs(non_bos)), 1e-9)),
            "non_bos_p999": float(np.percentile(np.abs(non_bos), 99.9)),
            "non_bos_p9999": float(np.percentile(np.abs(non_bos), 99.99)),
        }
        print(f"  blk.{layer}: nonBOS absmax={summary[f'blk.{layer}']['non_bos_absmax']:.3f} BOS absmax={summary[f'blk.{layer}']['bos_absmax']:.3f} ratio={summary[f'blk.{layer}']['bos_ratio']:.2f}")
    with open(out_dir/"gelu_local.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer","tag","n_tokens","absmax","p999","p9999","mean","std","bos_vs_nonbos_absmax_ratio","s_in","s_out","nmse","gelu_absmax","gelu_p999","gelu_p9999"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(out_dir/"gelu_local.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] gelu_local.csv / gelu_local.json 저장 ({out_dir})")
    print(f"[*] 다음: BOS ratio가 2~3배면 RMSNorm처럼 BOS 문제 있음. 1배 근처면 GELU는 BOS 문제 없음.")

if __name__=="__main__":
    main()
