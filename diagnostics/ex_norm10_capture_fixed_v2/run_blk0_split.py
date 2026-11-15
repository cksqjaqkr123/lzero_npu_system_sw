#!/usr/bin/env python3
"""
P1 세부: blk.0 attn vs ffn 중 누가 범인인지, s_out absmax가 E2E를 살리는지.
- A: blk.0 attn만 양자화
- B: blk.0 ffn만 양자화
- C: blk.0 둘 다 + attn s_out=absmax (0.15898), ffn은 p9999 유지
- D: blk.0 둘 다 + p9999 (기존 k=1과 동일, 대조군)
"""
import argparse, gc, json, random
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations, compute_rope_cache, apply_rope
from run_e2e_smoke import quant_rmsnorm

def forward_blk0_split(model, input_ids, params, mode):
    cfg = model.cfg
    B, S = input_ids.shape
    emb_w = model._load_weight("token_embd.weight")
    if emb_w.shape == (cfg.hidden_dim, cfg.vocab_size):
        emb_w = emb_w.T.contiguous()
    x = F.embedding(input_ids.to(model.device), emb_w)
    if model._rope_cache is None or model._rope_cache[0].shape[0] < S:
        model._rope_cache = compute_rope_cache(S, cfg.head_dim, cfg.rope_base, model.device)
    cos, sin = model._rope_cache
    for i in range(cfg.num_layers):
        p = f"blk.{i}"
        is_blk0 = (i == 0)
        # attn
        attn_norm_w = model._load_norm(f"{p}.attn_norm.weight")
        q_w = model._load_linear(f"{p}.attn_q.weight", cfg.q_dim, cfg.hidden_dim)
        k_w = model._load_linear(f"{p}.attn_k.weight", cfg.kv_dim, cfg.hidden_dim)
        v_w = model._load_linear(f"{p}.attn_v.weight", cfg.kv_dim, cfg.hidden_dim)
        o_w = model._load_linear(f"{p}.attn_output.weight", cfg.hidden_dim, cfg.q_dim)
        quant_attn = False
        s_in_a, s_out_a = None, None
        if is_blk0 and mode in ("attn_only", "both_p9999", "both_absmax"):
            quant_attn = True
            s_in_a = params["blk.0.attn_norm"]["s_in"]
            s_out_a = 0.15898 if mode == "both_absmax" else params["blk.0.attn_norm"]["s_out"]
        normed = quant_rmsnorm(x, attn_norm_w, s_in_a, s_out_a) if quant_attn else model._rmsnorm(x, attn_norm_w)
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
        # ffn
        ffn_norm_w = model._load_norm(f"{p}.ffn_norm.weight")
        gate_w = model._load_linear(f"{p}.ffn_gate.weight", cfg.intermediate_dim, cfg.hidden_dim)
        up_w = model._load_linear(f"{p}.ffn_up.weight", cfg.intermediate_dim, cfg.hidden_dim)
        down_w = model._load_linear(f"{p}.ffn_down.weight", cfg.hidden_dim, cfg.intermediate_dim)
        quant_ffn = is_blk0 and mode in ("ffn_only", "both_p9999", "both_absmax")
        if quant_ffn:
            normed = quant_rmsnorm(x, ffn_norm_w, params["blk.0.ffn_norm"]["s_in"], params["blk.0.ffn_norm"]["s_out"])
        else:
            normed = model._rmsnorm(x, ffn_norm_w)
        gate = F.gelu(F.linear(normed, gate_w)); up = F.linear(normed, up_w)
        mlp = F.linear(gate * up, down_w); x = x + mlp
        del ffn_norm_w, gate_w, up_w, down_w, normed, gate, up, mlp; gc.collect()
    final_w = model._load_norm("output_norm.weight")
    x_final = model._rmsnorm(x, final_w)
    del final_w; gc.collect()
    logits = F.linear(x_final, emb_w)
    del emb_w, x_final, x; gc.collect()
    return logits[0].detach().float().cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--params", default="diagnostics/ex_norm10_capture_fixed_v2/norm_selected_parameters.json")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    args = ap.parse_args()
    params = json.load(open(args.params))
    out_dir = Path(args.output_dir)
    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    tokenizer = GemmaTokenizer()
    convs = build_conversations(1, prefill_lengths=[128], decode_lengths=[16])
    rng = random.Random(42)
    conv = convs[0]
    prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
    full_ids = prompt_ids + [rng.randint(0, model.cfg.vocab_size-1) for _ in range(32)]
    input_tensor = torch.tensor([full_ids], dtype=torch.long)
    print(f"[*] S={len(full_ids)} FP baseline...")
    # FP baseline: mode="fp" (아무것도 양자화 안 함) — forward_blk0_split에서 mode가 attn/ffn에 안 걸리면 FP
    logits_fp = forward_blk0_split(model, input_tensor, params, "fp")
    S = logits_fp.shape[0]
    eval_pos = list(range(max(0, S-9), S-1))
    rows = []
    for mode in ["attn_only", "ffn_only", "both_p9999", "both_absmax"]:
        print(f"[*] mode={mode} forward...")
        logits_q = forward_blk0_split(model, input_tensor, params, mode)
        agree = sum(1 for pos in eval_pos if np.argmax(logits_fp[pos]) == np.argmax(logits_q[pos]))
        fp = logits_fp[eval_pos].astype(np.float64); qq = logits_q[eval_pos].astype(np.float64)
        fp_s = fp - fp.max(axis=-1, keepdims=True); qq_s = qq - qq.max(axis=-1, keepdims=True)
        p = np.exp(fp_s)/np.exp(fp_s).sum(axis=-1, keepdims=True)
        qprob = np.exp(qq_s)/np.exp(qq_s).sum(axis=-1, keepdims=True)
        kl = float(np.sum(p*(np.log(p+1e-12)-np.log(qprob+1e-12)), axis=-1).mean())
        nmse = float(np.sum((qq-fp)**2)/np.sum(fp**2))
        rows.append({"mode": mode, "top1": agree/len(eval_pos), "agree": agree, "n": len(eval_pos), "kl": kl, "logits_nmse": nmse})
        print(f"  {mode:12s} Top1 {agree}/{len(eval_pos)}={agree/len(eval_pos):.3f} KL={kl:.4f} NMSE={nmse:.4f}")
        del logits_q; gc.collect()
    with open(out_dir/"blk0_split.json","w") as f:
        json.dump(rows, f, indent=2)
    print(f"[✓] blk0_split.json 저장")

if __name__=="__main__":
    main()
