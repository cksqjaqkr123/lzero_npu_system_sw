# P6 BOS absmax calib - BOS 0번 토큰만 수집, s = absmax / 511 (클리핑 방지)
import argparse, json, gc
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations, compute_rope_cache, apply_rope

def log(*a, **k):
    print(*a, flush=True, **k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--n_conv", type=int, default=32)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--prefill", type=int, default=256)
    ap.add_argument("--out", default="diagnostics/ex_norm10_p6_bos_int10/calib/calib_bos_absmax_32.json")
    args = ap.parse_args()
    log(f"[CALIB_BOS_ABSMAX] n_conv={args.n_conv} prefill={args.prefill}")
    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    tokenizer = GemmaTokenizer()
    all_convs = build_conversations(args.start_idx + args.n_conv, prefill_lengths=[args.prefill]*(args.start_idx + args.n_conv), decode_lengths=[16]*(args.start_idx + args.n_conv))
    convs = all_convs[args.start_idx:]
    sin_vals = {}
    sout_vals = {}
    for ci, conv in enumerate(convs):
        prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        cfg = model.cfg
        B, S = input_tensor.shape
        emb_w = model._load_weight("token_embd.weight")
        if emb_w.shape == (cfg.hidden_dim, cfg.vocab_size):
            emb_w = emb_w.T.contiguous()
        x = F.embedding(input_tensor.to(model.device), emb_w)
        if model._rope_cache is None or model._rope_cache[0].shape[0] < S:
            model._rope_cache = compute_rope_cache(S, cfg.head_dim, cfg.rope_base, model.device)
        cos, sin = model._rope_cache
        for i in range(cfg.num_layers):
            p = f"blk.{i}"
            attn_norm_w = model._load_norm(f"{p}.attn_norm.weight")
            key_in = f"{p}.attn_norm"
            x_np = x[0, 0:1, :].detach().float().cpu().numpy()
            sin_vals.setdefault(key_in, []).append(np.abs(x_np).ravel())
            normed = model._rmsnorm(x, attn_norm_w)
            normed_np = normed[0, 0:1, :].detach().float().cpu().numpy()
            sout_vals.setdefault(key_in, []).append(np.abs(normed_np).ravel())
            q_w = model._load_linear(f"{p}.attn_q.weight", cfg.q_dim, cfg.hidden_dim)
            k_w = model._load_linear(f"{p}.attn_k.weight", cfg.kv_dim, cfg.hidden_dim)
            v_w = model._load_linear(f"{p}.attn_v.weight", cfg.kv_dim, cfg.hidden_dim)
            o_w = model._load_linear(f"{p}.attn_output.weight", cfg.hidden_dim, cfg.q_dim)
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
            del attn_norm_w, q_w, k_w, v_w, o_w, normed, q, k, v, attn
            ffn_norm_w = model._load_norm(f"{p}.ffn_norm.weight")
            key_ffn = f"{p}.ffn_norm"
            x_np2 = x[0, 0:1, :].detach().float().cpu().numpy()
            sin_vals.setdefault(key_ffn, []).append(np.abs(x_np2).ravel())
            normed2 = model._rmsnorm(x, ffn_norm_w)
            normed2_np = normed2[0, 0:1, :].detach().float().cpu().numpy()
            sout_vals.setdefault(key_ffn, []).append(np.abs(normed2_np).ravel())
            gate_w = model._load_linear(f"{p}.ffn_gate.weight", cfg.intermediate_dim, cfg.hidden_dim)
            up_w = model._load_linear(f"{p}.ffn_up.weight", cfg.intermediate_dim, cfg.hidden_dim)
            down_w = model._load_linear(f"{p}.ffn_down.weight", cfg.hidden_dim, cfg.intermediate_dim)
            gate = F.gelu(F.linear(normed2, gate_w)); up = F.linear(normed2, up_w)
            mlp = F.linear(gate * up, down_w); x = x + mlp
            del ffn_norm_w, gate_w, up_w, down_w, normed2, gate, up, mlp
        final_w = model._load_norm("output_norm.weight")
        x_np3 = x[0, 0:1, :].detach().float().cpu().numpy()
        sin_vals.setdefault("output_norm", []).append(np.abs(x_np3).ravel())
        x_final = model._rmsnorm(x, final_w)
        x_final_np = x_final[0, 0:1, :].detach().float().cpu().numpy()
        sout_vals.setdefault("output_norm", []).append(np.abs(x_final_np).ravel())
        del final_w, emb_w, x_final, x
        gc.collect()
        log(f"[*] calib BOS conv{ci} done")
    out = {}
    for key in sorted(sin_vals.keys()):
        all_in = np.concatenate(sin_vals[key])
        all_out = np.concatenate(sout_vals[key])
        absmax_in = float(np.max(all_in))
        absmax_out = float(np.max(all_out))
        p9999_in = float(np.quantile(all_in, 0.9999))
        p9999_out = float(np.quantile(all_out, 0.9999))
        out[key] = {"s_in": absmax_in / 511.0, "s_out": absmax_out / 511.0, "absmax_in": absmax_in, "absmax_out": absmax_out, "p9999_in": p9999_in, "p9999_out": p9999_out, "n": int(all_in.size)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    log(f"[DONE] saved to {args.out} keys={len(out)}")

if __name__ == "__main__":
    main()
