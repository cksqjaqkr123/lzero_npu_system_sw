#!/usr/bin/env python3
"""
P3 RoPE BOS 체크: RoPE도 BOS처럼 첫 토큰이 문제인지
- blk.0 attn의 Q,K를 RoPE 전/후로 나눠서, BOS(0번) vs 나머지 비교
- BOS의 Q,K가 outlier인지, RoPE 후 양자화 오차가 BOS에서 큰지
"""
import argparse, gc, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations, compute_rope_cache, apply_rope

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    tokenizer = GemmaTokenizer()
    convs = build_conversations(1, prefill_lengths=[128], decode_lengths=[16])
    conv = convs[0]
    prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    B, S = input_tensor.shape
    print(f"[*] S={S}")
    cfg = model.cfg
    emb_w = model._load_weight("token_embd.weight")
    if emb_w.shape == (cfg.hidden_dim, cfg.vocab_size):
        emb_w = emb_w.T.contiguous()
    x = F.embedding(input_tensor.to(model.device), emb_w)
    if model._rope_cache is None or model._rope_cache[0].shape[0] < S:
        model._rope_cache = compute_rope_cache(S, cfg.head_dim, cfg.rope_base, model.device)
    cos, sin = model._rope_cache
    # blk.0만
    p = "blk.0"
    attn_norm_w = model._load_norm(f"{p}.attn_norm.weight")
    q_w = model._load_linear(f"{p}.attn_q.weight", cfg.q_dim, cfg.hidden_dim)
    k_w = model._load_linear(f"{p}.attn_k.weight", cfg.kv_dim, cfg.hidden_dim)
    normed = model._rmsnorm(x, attn_norm_w)
    q = F.linear(normed, q_w).view(B, S, cfg.num_heads, cfg.head_dim).transpose(1, 2)
    k = F.linear(normed, k_w).view(B, S, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
    q_pre = q.detach().float().cpu().numpy()
    k_pre = k.detach().float().cpu().numpy()
    q_rope, k_rope = apply_rope(q, k, cos, sin)
    q_post = q_rope.detach().float().cpu().numpy()
    k_post = k_rope.detach().float().cpu().numpy()
    rows = []
    for name, arr_pre, arr_post in [("Q", q_pre, q_post), ("K", k_pre, k_post)]:
        # arr shape: [1, heads, S, head_dim]
        bos_pre = arr_pre[0, :, 0, :]
        rest_pre = arr_pre[0, :, 1:, :]
        bos_post = arr_post[0, :, 0, :]
        rest_post = arr_post[0, :, 1:, :]
        row = {
            "tensor": name,
            "bos_pre_max": float(np.abs(bos_pre).max()),
            "rest_pre_max": float(np.abs(rest_pre).max()),
            "rest_pre_p999": float(np.percentile(np.abs(rest_pre), 99.9)),
            "bos_post_max": float(np.abs(bos_post).max()),
            "rest_post_max": float(np.abs(rest_post).max()),
            "rest_post_p999": float(np.percentile(np.abs(rest_post), 99.9)),
            "bos_pre_mean_abs": float(np.abs(bos_pre).mean()),
            "rest_pre_mean_abs": float(np.abs(rest_pre).mean()),
            "bos_post_mean_abs": float(np.abs(bos_post).mean()),
            "rest_post_mean_abs": float(np.abs(rest_post).mean()),
            # BOS가 RoPE로 바뀌었는지: cos=1, sin=0이면 BOS는 안 바뀌어야 함
            "bos_rope_diff_max": float(np.abs(bos_post - bos_pre).max()),
            "rest_rope_diff_max": float(np.abs(rest_post - rest_pre).max()),
        }
        rows.append(row)
        print(f"[{name}] BOS pre max {row['bos_pre_max']:.3f} rest pre max {row['rest_pre_max']:.3f} p999 {row['rest_pre_p999']:.3f}")
        print(f"[{name}] BOS post max {row['bos_post_max']:.3f} rest post max {row['rest_post_max']:.3f} p999 {row['rest_post_p999']:.3f}")
        print(f"[{name}] BOS RoPE diff max {row['bos_rope_diff_max']:.6f} (0이면 BOS는 RoPE 안 탐) rest diff {row['rest_rope_diff_max']:.3f}")
        # INT10 양자화 오차: s = absmax/511 vs p999/511
        for label, s_val in [("absmax", row['rest_pre_max']/511), ("p999", row['rest_pre_p999']/511)]:
            q_bos = np.clip(np.rint(bos_pre / s_val), -512, 511) * s_val
            q_rest = np.clip(np.rint(rest_pre / s_val), -512, 511) * s_val
            nmse_bos = float(np.sum((q_bos - bos_pre)**2) / np.sum(bos_pre**2))
            nmse_rest = float(np.sum((q_rest - rest_pre)**2) / np.sum(rest_pre**2))
            print(f"  [{name} {label} s={s_val:.6f}] BOS NMSE {nmse_bos:.4f} rest NMSE {nmse_rest:.4f}")
    with open(out_dir/"rope_bos_check.json","w") as f:
        json.dump(rows, f, indent=2)
    print(f"[✓] rope_bos_check.json 저장")

if __name__=="__main__":
    main()
