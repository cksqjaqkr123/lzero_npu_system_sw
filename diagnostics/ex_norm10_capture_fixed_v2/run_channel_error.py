#!/usr/bin/env python3
"""
Worst Norm들의 채널별 오차 분해.
- 대상: blk.0 attn, blk.1 attn, blk.5 attn, blk.5 ffn (Group A) + blk.2 attn (비교용)
- 각 Norm마다 best s_in 사용, s_out은 p999(y)/511 고정
- 출력: channel_error.csv (Norm x 2304 채널, err2 기여도 순)
"""
import argparse, csv, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations

EPS = 1e-6
TARGETS = {
    ("blk.0", "attn_norm"): "s_in_p999",
    ("blk.1", "attn_norm"): "s_in_p9999",
    ("blk.5", "attn_norm"): "s_in_p999",
    ("blk.5", "ffn_norm"): "s_in_p999",
    ("blk.2", "attn_norm"): "s_in_p9999",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--input_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    ap.add_argument("--num_conversations", type=int, default=8)
    args = ap.parse_args()
    in_dir = Path(args.input_dir); out_dir = Path(args.output_dir)

    cand = {}
    with open(in_dir / "scale_candidates_v2.csv") as f:
        for r in csv.DictReader(f):
            cand[(r["layer_name"], r["norm_type"])] = {
                "s_in_p999": float(r["s_in_p999"]),
                "s_in_p9999": float(r["s_in_p9999"]),
                "s_in_absmax": float(r["s_in_absmax"]),
                "s_out": float(r["s_out_p999"]),
            }
    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    tokenizer = GemmaTokenizer()
    convs = build_conversations(args.num_conversations,
        prefill_lengths=[32, 128, 512, 1024], decode_lengths=[16, 24, 32])

    def gguf_name(ln, nt):
        if ln == "output_norm": return "output_norm.weight"
        return f"blk.{ln.split('.')[1]}.{nt}.weight"
    gamma_cache = {}
    for k in TARGETS:
        gamma_cache[k] = model._load_norm(gguf_name(k[0], k[1])).float().cpu().numpy().astype(np.float32)

    # Norm별, 채널별 누적 err2 / y2 / clip_out_count
    err2 = {k: np.zeros(2304, dtype=np.float64) for k in TARGETS}
    y2 = {k: np.zeros(2304, dtype=np.float64) for k in TARGETS}
    clip = {k: np.zeros(2304, dtype=np.int64) for k in TARGETS}
    n_tok = {k: 0 for k in TARGETS}
    rng = random.Random(42)

    for ci, conv in enumerate(convs):
        prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        if not prompt_ids: continue
        prefill_len = len(prompt_ids)
        decode_ids = [rng.randint(0, model.cfg.vocab_size - 1) for _ in range(conv["decode_len"])]
        full_ids = prompt_ids + decode_ids
        captures = model.forward_capture(torch.tensor([full_ids], dtype=torch.long))
        for layer_name, norm_type, arr in captures:
            k = (layer_name, norm_type)
            if k not in TARGETS: continue
            gamma = gamma_cache[k]
            c = cand[k]
            s_in = c[TARGETS[k]]; s_out = c["s_out"]
            for pos in range(arr.shape[0]):
                if pos >= prefill_len or pos == 0: continue
                x = arr[pos].astype(np.float32)
                rms_fp = np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS)
                y_fp = (x / rms_fp * gamma).astype(np.float64)
                q = np.clip(np.round(x / s_in), -127, 127)
                x_hat = q * s_in
                rms_q = np.sqrt(np.mean(x_hat.astype(np.float64) ** 2) + EPS)
                y_hat = (x_hat / rms_q * gamma).astype(np.float64)
                q_out = np.clip(np.round(y_hat / s_out), -512, 511)
                y_out = q_out * s_out
                err2[k] += (y_out - y_fp) ** 2
                y2[k] += y_fp ** 2
                clip[k] += (np.abs(y_hat / s_out) > 511.5).astype(np.int64)
                n_tok[k] += 1
        print(f"[*] conv {ci} done")
        del captures
        import gc; gc.collect()

    rows = []
    for k in TARGETS:
        total_err = err2[k].sum(); total_y = y2[k].sum()
        gamma = gamma_cache[k]
        # 기여도 순으로 정렬
        order = np.argsort(-err2[k])
        print(f"\n=== {k[0]} {k[1]}  NMSE={total_err/total_y:.4f}  tokens={n_tok[k]} ===")
        print(f"  Top 10 채널 (err 기여도):")
        cum = 0.0
        for rank, ch in enumerate(order[:10]):
            contrib = err2[k][ch] / total_err if total_err > 0 else 0
            cum += contrib
            print(f"    #{rank+1:2d} ch={ch:4d} gamma={gamma[ch]:.3f} "
                  f"err_share={contrib*100:5.1f}% cum={cum*100:5.1f}% "
                  f"clip={clip[k][ch]} y2={y2[k][ch]:.1f}")
        for ch in range(2304):
            rows.append({
                "layer_name": k[0], "norm_type": k[1], "channel": ch,
                "gamma": float(gamma[ch]),
                "err2": float(err2[k][ch]), "y2": float(y2[k][ch]),
                "err_share": float(err2[k][ch] / total_err) if total_err > 0 else 0.0,
                "clip_out_count": int(clip[k][ch]), "n_tokens": n_tok[k],
            })
    with open(out_dir / "channel_error.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[✓] channel_error.csv 저장 ({len(rows)} 행)")

if __name__ == "__main__":
    main()