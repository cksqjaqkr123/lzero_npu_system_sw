#!/usr/bin/env python3
"""
P1 propagation: 누적 삽입으로 first divergence 찾기.
- Norm 0..k까지만 양자화, 나머지는 FP. k를 키워가며 Top1/KL이 어디서 떨어지는지 본다.
- 느려서 coarse-to-fine: 먼저 층 단위(0,1,2,3,5,10,15,20,26)로, 그 다음 터지는 층 안을 Norm 단위로.
- 대화는 E2E에서 0%였던 conv1 패턴을 재현하는 1개 대화만 (빠르게).
"""
import argparse, gc, json, random
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations, compute_rope_cache, apply_rope
from run_e2e_smoke import quant_rmsnorm, EPS

def forward_logits_prefix(model, input_ids, params, k_layers):
    """k_layers: 앞 k개 층(blk.0..blk.k-1)만 양자화, 나머지는 FP. final_norm은 k==26일 때만 양자화."""
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
        quant_this_layer = (i < k_layers)
        p = f"blk.{i}"
        attn_norm_w = model._load_norm(f"{p}.attn_norm.weight")
        q_w = model._load_linear(f"{p}.attn_q.weight", cfg.q_dim, cfg.hidden_dim)
        k_w = model._load_linear(f"{p}.attn_k.weight", cfg.kv_dim, cfg.hidden_dim)
        v_w = model._load_linear(f"{p}.attn_v.weight", cfg.kv_dim, cfg.hidden_dim)
        o_w = model._load_linear(f"{p}.attn_output.weight", cfg.hidden_dim, cfg.q_dim)
        if quant_this_layer:
            normed = quant_rmsnorm(x, attn_norm_w, params[f"blk.{i}.attn_norm"]["s_in"], params[f"blk.{i}.attn_norm"]["s_out"])
        else:
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
        ffn_norm_w = model._load_norm(f"{p}.ffn_norm.weight")
        gate_w = model._load_linear(f"{p}.ffn_gate.weight", cfg.intermediate_dim, cfg.hidden_dim)
        up_w = model._load_linear(f"{p}.ffn_up.weight", cfg.intermediate_dim, cfg.hidden_dim)
        down_w = model._load_linear(f"{p}.ffn_down.weight", cfg.hidden_dim, cfg.intermediate_dim)
        if quant_this_layer:
            normed = quant_rmsnorm(x, ffn_norm_w, params[f"blk.{i}.ffn_norm"]["s_in"], params[f"blk.{i}.ffn_norm"]["s_out"])
        else:
            normed = model._rmsnorm(x, ffn_norm_w)
        gate = F.gelu(F.linear(normed, gate_w)); up = F.linear(normed, up_w)
        mlp = F.linear(gate * up, down_w); x = x + mlp
        del ffn_norm_w, gate_w, up_w, down_w, normed, gate, up, mlp; gc.collect()
    final_w = model._load_norm("output_norm.weight")
    if k_layers >= cfg.num_layers:
        x_final = quant_rmsnorm(x, final_w, params["output_norm.final_norm"]["s_in"], params["output_norm.final_norm"]["s_out"])
    else:
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
    ap.add_argument("--ks", default="0,1,2,3,5,10,15,20,26", help="comma-separated k_layers")
    args = ap.parse_args()
    params = json.load(open(args.params))
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    tokenizer = GemmaTokenizer()
    convs = build_conversations(1, prefill_lengths=[128], decode_lengths=[16])
    rng = random.Random(42)
    conv = convs[0]
    prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
    full_ids = prompt_ids + [rng.randint(0, model.cfg.vocab_size-1) for _ in range(32)]
    input_tensor = torch.tensor([full_ids], dtype=torch.long)
    print(f"[*] S={len(full_ids)} FP baseline...")
    logits_fp = forward_logits_prefix(model, input_tensor, params, 0)
    S = logits_fp.shape[0]
    eval_pos = list(range(max(0, S-9), S-1))
    rows = []
    for k in [int(x) for x in args.ks.split(",")]:
        print(f"[*] k_layers={k} forward...")
        logits_q = forward_logits_prefix(model, input_tensor, params, k)
        agree = sum(1 for pos in eval_pos if np.argmax(logits_fp[pos]) == np.argmax(logits_q[pos]))
        fp = logits_fp[eval_pos].astype(np.float64); qq = logits_q[eval_pos].astype(np.float64)
        fp_s = fp - fp.max(axis=-1, keepdims=True); qq_s = qq - qq.max(axis=-1, keepdims=True)
        p = np.exp(fp_s)/np.exp(fp_s).sum(axis=-1, keepdims=True)
        qprob = np.exp(qq_s)/np.exp(qq_s).sum(axis=-1, keepdims=True)
        kl = float(np.sum(p*(np.log(p+1e-12)-np.log(qprob+1e-12)), axis=-1).mean())
        nmse = float(np.sum((qq-fp)**2)/np.sum(fp**2))
        rows.append({"k_layers": k, "top1": agree/len(eval_pos), "agree": agree, "n": len(eval_pos), "kl": kl, "logits_nmse": nmse})
        print(f"  k={k:2d} Top1 {agree}/{len(eval_pos)}={agree/len(eval_pos):.3f} KL={kl:.4f} NMSE={nmse:.4f}")
        del logits_q; gc.collect()
    with open(out_dir/"propagation_coarse.json","w") as f:
        json.dump(rows, f, indent=2)
    print(f"[✓] propagation_coarse.json 저장")

if __name__=="__main__":
    main()
