#!/usr/bin/env python3
"""
s_out LUT 클러스터링 적용 후 E2E 재검증: 1500개 위치 (프롬프트 8개 × ~190)
- sout_lut.json의 클러스터 대표 s_out을 쓰고, s_in은 기존 absmax(BOS 제외) 그대로
- BOS bypass 적용 (모든 Norm BOS만 FP)
- 1500개 프롬프트가 아니라 1500개 위치 기준: 8개 프롬프트 × ~190위치 = ~1520
- 정말 1500개 프롬프트를 원하면 --n_conv 1500으로 (시간: 1500×2 forward, 수십 시간, 비추천)
"""
import argparse, gc, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations, compute_rope_cache, apply_rope
from run_e2e_bos_bypass import quant_bos_fp

def forward_e2e_lut(model, input_ids, params_sin, lut_map, bos_fp=True):
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
        attn_norm_w = model._load_norm(f"{p}.attn_norm.weight")
        q_w = model._load_linear(f"{p}.attn_q.weight", cfg.q_dim, cfg.hidden_dim)
        k_w = model._load_linear(f"{p}.attn_k.weight", cfg.kv_dim, cfg.hidden_dim)
        v_w = model._load_linear(f"{p}.attn_v.weight", cfg.kv_dim, cfg.hidden_dim)
        o_w = model._load_linear(f"{p}.attn_output.weight", cfg.hidden_dim, cfg.q_dim)
        s_in = params_sin[f"blk.{i}.attn_norm"]["s_in"]
        s_out = lut_map[f"blk.{i}.attn_norm"]
        normed = quant_bos_fp(x, attn_norm_w, s_in, s_out, bos_fp)
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
        s_in = params_sin[f"blk.{i}.ffn_norm"]["s_in"]
        s_out = lut_map[f"blk.{i}.ffn_norm"]
        normed = quant_bos_fp(x, ffn_norm_w, s_in, s_out, bos_fp)
        gate = F.gelu(F.linear(normed, gate_w)); up = F.linear(normed, up_w)
        mlp = F.linear(gate * up, down_w); x = x + mlp
        del ffn_norm_w, gate_w, up_w, down_w, normed, gate, up, mlp; gc.collect()
    final_w = model._load_norm("output_norm.weight")
    s_in = params_sin["output_norm.final_norm"]["s_in"]
    s_out = lut_map["output_norm.final_norm"]
    normed = quant_bos_fp(x, final_w, s_in, s_out, bos_fp)
    del final_w; gc.collect()
    logits = F.linear(normed, emb_w)
    del emb_w, normed, x; gc.collect()
    return logits[0].detach().float().cpu().numpy()

def forward_fp(model, input_ids):
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
        ffn_norm_w = model._load_norm(f"{p}.ffn_norm.weight")
        gate_w = model._load_linear(f"{p}.ffn_gate.weight", cfg.intermediate_dim, cfg.hidden_dim)
        up_w = model._load_linear(f"{p}.ffn_up.weight", cfg.intermediate_dim, cfg.hidden_dim)
        down_w = model._load_linear(f"{p}.ffn_down.weight", cfg.hidden_dim, cfg.intermediate_dim)
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
    ap.add_argument("--params", default="/tmp/p9999_absmax.json")
    ap.add_argument("--lut", default="diagnostics/ex_norm10_capture_fixed_v2/sout_lut.json")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    ap.add_argument("--n_conv", type=int, default=8)
    ap.add_argument("--prefill", type=int, default=256)
    args = ap.parse_args()
    params = json.load(open(args.params))
    lut_data = json.load(open(args.lut))
    # lut_map: norm -> rep_s_out (클러스터 대표값)
    lut_map = {}
    for c in lut_data["clusters"]:
        for m in c["members"]:
            lut_map[m["norm"]] = c["rep_s_out"]
    print(f"[*] LUT: {lut_data['n_norms']} Norms -> {lut_data['n_clusters']} clusters, threshold={lut_data['threshold']}")
    print(f"[*] E2E: {args.n_conv} conv, prefill {args.prefill} (목표 ~1500 위치)")
    out_dir = Path(args.output_dir)
    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    tokenizer = GemmaTokenizer()
    convs = build_conversations(args.n_conv, prefill_lengths=[args.prefill]*args.n_conv, decode_lengths=[16]*args.n_conv)
    all_agree = 0; all_n = 0; all_kl = []; all_nmse = []
    rows = []
    for ci, conv in enumerate(convs):
        prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        S = len(prompt_ids)
        print(f"[*] conv{ci} S={S} FP...")
        logits_fp = forward_fp(model, input_tensor)
        eval_pos = list(range(1, S))
        print(f"[*] conv{ci} LUT BOS_FP forward... eval n={len(eval_pos)}")
        logits_q = forward_e2e_lut(model, input_tensor, params, lut_map, True)
        agree = sum(1 for pos in eval_pos if np.argmax(logits_fp[pos]) == np.argmax(logits_q[pos]))
        fp = logits_fp[eval_pos].astype(np.float64); qq = logits_q[eval_pos].astype(np.float64)
        fp_s = fp - fp.max(axis=-1, keepdims=True); qq_s = qq - qq.max(axis=-1, keepdims=True)
        p = np.exp(fp_s)/np.exp(fp_s).sum(axis=-1, keepdims=True)
        qprob = np.exp(qq_s)/np.exp(qq_s).sum(axis=-1, keepdims=True)
        kl = float(np.sum(p*(np.log(p+1e-12)-np.log(qprob+1e-12)), axis=-1).mean())
        nmse = float(np.sum((qq-fp)**2)/np.sum(fp**2))
        rows.append({"conv": ci, "mode": "LUT_BOS_FP", "top1": agree/len(eval_pos), "agree": agree, "n": len(eval_pos), "kl": kl, "logits_nmse": nmse, "prefill": args.prefill, "n_clusters": lut_data["n_clusters"]})
        print(f"  conv{ci} LUT Top1 {agree}/{len(eval_pos)}={agree/len(eval_pos):.3f} KL={kl:.4f} NMSE={nmse:.4f}")
        all_agree += agree; all_n += len(eval_pos); all_kl.append(kl); all_nmse.append(nmse)
        del logits_fp, logits_q; gc.collect()
    print(f"[SUMMARY] LUT ({lut_data['n_clusters']} clusters) Top1 {all_agree}/{all_n}={all_agree/all_n:.3f} KL avg {sum(all_kl)/len(all_kl):.4f} NMSE avg {sum(all_nmse)/len(all_nmse):.4f}")
    print(f"[비교] 기존 per-Norm s_out (53개): Top1 86.5% KL 0.109 (1606개)")
    with open(out_dir/"e2e_sout_lut_1500.json","w") as f:
        json.dump(rows, f, indent=2)
    print(f"[✓] e2e_sout_lut_1500.json 저장 (n={all_n})")

if __name__=="__main__":
    main()
