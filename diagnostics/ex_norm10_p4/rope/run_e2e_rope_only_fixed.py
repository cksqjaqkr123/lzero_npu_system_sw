#!/usr/bin/env python3
"""
RoPE-only E2E FIXED — GELU-only랑 똑같은 구조
- 목적: RoPE만 양자화했을 때 E2E 확인. RMSNorm FP, GELU FP, Linear FP.
- RoPE 양자화: Q/K를 RoPE 전에 양자화 → RoPE는 FP 회전 (magnitude 보존이라 s_out = s_in)
- s 계산: dynamic per-layer absmax / p9999 (sequence마다 계산, calibration 오차 없는 상한)
  - absmax: max(|Q|) / 511, max(|K|) / 511
  - p9999: quantile(|Q|, 0.9999) / 511
- BOS bypass 없음 (RoPE BOS diff 0.0으로 확인됨, Q/K에서 BOS는 아웃라이어 아님)
- GELU 때랑 똑같이: no_grad + flush + partial jsonl + pilot 모드
사용법:
  PYTHONPATH=diagnostics/ex_norm10_capture_fixed python3 -u diagnostics/ex_norm10_p4/rope/run_e2e_rope_only_fixed.py --n_conv 2 --prefill 128 --mode absmax 2>&1 | tee diagnostics/ex_norm10_p4/rope/e2e_rope_only.log
  PYTHONPATH=diagnostics/ex_norm10_capture_fixed python3 -u diagnostics/ex_norm10_p4/rope/run_e2e_rope_only_fixed.py --n_conv 8 --prefill 256 --mode both 2>&1 | tee -a diagnostics/ex_norm10_p4/rope/e2e_rope_only.log
"""
import argparse, gc, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations, compute_rope_cache, apply_rope

def log(*a, **k):
    print(*a, flush=True, **k)

def quant_dynamic(x, mode):
    # x: [B, H, S, D], per-layer scale (전체 sequence에서)
    if mode == "absmax":
        s = x.abs().max() / 511
    elif mode == "p9999":
        s = torch.quantile(x.abs().flatten().float(), 0.9999) / 511
    else: # p999
        s = torch.quantile(x.abs().flatten().float(), 0.999) / 511
    s = s.clamp(min=1e-8)
    return torch.round(x / s).clamp(-511, 511) * s

@torch.no_grad()
def forward_rope_only(model, input_ids, mode):
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
        # RoPE만 양자화: RoPE 전에 Q/K 양자화, RoPE는 FP
        q = quant_dynamic(q, mode)
        k = quant_dynamic(k, mode)
        q, k = apply_rope(q, k, cos, sin)
        if cfg.num_kv_heads != cfg.num_heads:
            rep = cfg.num_heads // cfg.num_kv_heads
            k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, cfg.q_dim)
        attn = F.linear(attn, o_w); x = x + attn
        del attn_norm_w, q_w, k_w, v_w, o_w, normed, q, k, v, attn
        ffn_norm_w = model._load_norm(f"{p}.ffn_norm.weight")
        gate_w = model._load_linear(f"{p}.ffn_gate.weight", cfg.intermediate_dim, cfg.hidden_dim)
        up_w = model._load_linear(f"{p}.ffn_up.weight", cfg.intermediate_dim, cfg.hidden_dim)
        down_w = model._load_linear(f"{p}.ffn_down.weight", cfg.hidden_dim, cfg.intermediate_dim)
        normed = model._rmsnorm(x, ffn_norm_w)
        gate = F.gelu(F.linear(normed, gate_w)); up = F.linear(normed, up_w)
        mlp = F.linear(gate * up, down_w); x = x + mlp
        del ffn_norm_w, gate_w, up_w, down_w, normed, gate, up, mlp
    final_w = model._load_norm("output_norm.weight")
    x_final = model._rmsnorm(x, final_w)
    del final_w
    logits = F.linear(x_final, emb_w)
    out = logits[0].detach().float().cpu().numpy()
    del emb_w, x_final, x, logits
    gc.collect()
    return out

@torch.no_grad()
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
        del attn_norm_w, q_w, k_w, v_w, o_w, normed, q, k, v, attn
        ffn_norm_w = model._load_norm(f"{p}.ffn_norm.weight")
        gate_w = model._load_linear(f"{p}.ffn_gate.weight", cfg.intermediate_dim, cfg.hidden_dim)
        up_w = model._load_linear(f"{p}.ffn_up.weight", cfg.intermediate_dim, cfg.hidden_dim)
        down_w = model._load_linear(f"{p}.ffn_down.weight", cfg.hidden_dim, cfg.intermediate_dim)
        normed = model._rmsnorm(x, ffn_norm_w)
        gate = F.gelu(F.linear(normed, gate_w)); up = F.linear(normed, up_w)
        mlp = F.linear(gate * up, down_w); x = x + mlp
        del ffn_norm_w, gate_w, up_w, down_w, normed, gate, up, mlp
    final_w = model._load_norm("output_norm.weight")
    x_final = model._rmsnorm(x, final_w)
    del final_w
    logits = F.linear(x_final, emb_w)
    out = logits[0].detach().float().cpu().numpy()
    del emb_w, x_final, x, logits
    gc.collect()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_p4/rope")
    ap.add_argument("--n_conv", type=int, default=2)
    ap.add_argument("--prefill", type=int, default=128)
    ap.add_argument("--mode", default="absmax", choices=["absmax", "p9999", "p999", "both"])
    args = ap.parse_args()
    log(f"[START] RoPE-only E2E n_conv={args.n_conv} prefill={args.prefill} mode={args.mode}")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "e2e_rope_only_partial.jsonl"
    modes = ["absmax", "p9999"] if args.mode == "both" else [args.mode]
    log(f"[MODEL] loading {args.gguf} ...")
    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    log(f"[MODEL] loaded")
    tokenizer = GemmaTokenizer()
    convs = build_conversations(args.n_conv, prefill_lengths=[args.prefill]*args.n_conv, decode_lengths=[16]*args.n_conv)
    log(f"[CONV] {len(convs)} convs built")
    for ci, conv in enumerate(convs):
        prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        S = len(prompt_ids)
        log(f"[*] conv{ci} S={S} FP start...")
        logits_fp = forward_fp(model, input_tensor)
        log(f"[*] conv{ci} FP done")
        eval_pos = list(range(1, S))
        for mode in modes:
            log(f"[*] conv{ci} RoPE-only {mode} start...")
            logits_q = forward_rope_only(model, input_tensor, mode)
            agree = sum(1 for pos in eval_pos if np.argmax(logits_fp[pos]) == np.argmax(logits_q[pos]))
            fp = logits_fp[eval_pos].astype(np.float64); qq = logits_q[eval_pos].astype(np.float64)
            fp_s = fp - fp.max(axis=-1, keepdims=True); qq_s = qq - qq.max(axis=-1, keepdims=True)
            p = np.exp(fp_s)/np.exp(fp_s).sum(axis=-1, keepdims=True)
            qprob = np.exp(qq_s)/np.exp(qq_s).sum(axis=-1, keepdims=True)
            kl = float(np.sum(p*(np.log(p+1e-12)-np.log(qprob+1e-12)), axis=-1).mean())
            nmse = float(np.sum((qq-fp)**2)/np.sum(fp**2))
            rec = {"conv": ci, "mode": mode, "top1": agree/len(eval_pos), "agree": agree, "n": len(eval_pos), "kl": kl, "logits_nmse": nmse}
            log(f"  [DONE] conv{ci} RoPE {mode} Top1 {agree}/{len(eval_pos)}={agree/len(eval_pos):.3f} KL={kl:.4f} NMSE={nmse:.4f}")
            with open(jsonl_path, "a") as f: f.write(json.dumps(rec)+"\n")
            del logits_q; gc.collect()
        del logits_fp; gc.collect()
    log(f"[DONE] partial saved to {jsonl_path}")
    totals = {}
    for line in open(jsonl_path):
        r = json.loads(line)
        if r["mode"] not in totals: totals[r["mode"]] = {"agree":0,"n":0,"kl":[]}
        totals[r["mode"]]["agree"] += r["agree"]; totals[r["mode"]]["n"] += r["n"]; totals[r["mode"]]["kl"].append(r["kl"])
    for mode, t in totals.items():
        log(f"[SUMMARY] RoPE-only {mode} Top1 {t['agree']}/{t['n']}={t['agree']/t['n']:.3f} KL avg {sum(t['kl'])/len(t['kl']):.4f}")
    log(f"[비교] RMSNorm-only BOS_FP p9999: Top1 86.5% KL 0.109 | GELU-only absmax: Top1 97.2% KL 0.006")
    with open(out_dir/"e2e_rope_only.json","w") as f:
        json.dump({"totals": {k: {"agree": v["agree"], "n": v["n"], "top1": v["agree"]/v["n"], "kl_avg": sum(v["kl"])/len(v["kl"])} for k,v in totals.items()}}, f, indent=2)

if __name__=="__main__":
    main()
