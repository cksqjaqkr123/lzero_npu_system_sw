#!/usr/bin/env python3
"""
E2E smoke: FP_FULL vs IO_ORACLE (freeze된 s_in/s_out 적용)
- 3 conversations x 8 targets = 24 targets (md smoke와 동일 규모)
- Top1 agreement, KL, logits NMSE
- lm_head는 token_embd.weight 재사용 (Gemma tied embedding)
"""
import argparse, gc, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations, compute_rope_cache, apply_rope

EPS = 1e-6

def quant_rmsnorm(x_torch, gamma_torch, s_in, s_out):
    # x: [B,S,H] torch, gamma: [H]
    x_np = x_torch.detach().float().cpu().numpy()
    g_np = gamma_torch.detach().float().cpu().numpy().astype(np.float32)
    q = np.clip(np.round(x_np / s_in), -127, 127)
    x_hat = q * s_in
    rms = np.sqrt(np.mean(x_hat.astype(np.float64) ** 2, axis=-1, keepdims=True) + EPS)
    y_hat = (x_hat / rms * g_np).astype(np.float64)
    q_out = np.clip(np.round(y_hat / s_out), -512, 511)
    y_out = (q_out * s_out).astype(np.float32)
    return torch.from_numpy(y_out).to(x_torch.device).to(x_torch.dtype)

def forward_logits(model, input_ids, mode, params):
    cfg = model.cfg
    B, S = input_ids.shape
    # embedding + lm_head weight는 동일 (tied) — 한 번만 로드해서 재사용
    emb_w = model._load_weight("token_embd.weight")  # [vocab, H] or [H, vocab]
    if emb_w.shape == (cfg.hidden_dim, cfg.vocab_size):
        emb_w = emb_w.T.contiguous()
    # embedding
    x = F.embedding(input_ids.to(model.device), emb_w)
    if model._rope_cache is None or model._rope_cache[0].shape[0] < S:
        model._rope_cache = compute_rope_cache(S, cfg.head_dim, cfg.rope_base, model.device)
    cos, sin = model._rope_cache

    for i in range(cfg.num_layers):
        p = f"blk.{i}"
        # --- attn ---
        attn_norm_w = model._load_norm(f"{p}.attn_norm.weight")
        q_w = model._load_linear(f"{p}.attn_q.weight", cfg.q_dim, cfg.hidden_dim)
        k_w = model._load_linear(f"{p}.attn_k.weight", cfg.kv_dim, cfg.hidden_dim)
        v_w = model._load_linear(f"{p}.attn_v.weight", cfg.kv_dim, cfg.hidden_dim)
        o_w = model._load_linear(f"{p}.attn_output.weight", cfg.hidden_dim, cfg.q_dim)
        if mode == "FP":
            normed = model._rmsnorm(x, attn_norm_w)
        else:
            key = f"blk.{i}.attn_norm"
            normed = quant_rmsnorm(x, attn_norm_w, params[key]["s_in"], params[key]["s_out"])
        q = F.linear(normed, q_w).view(B, S, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = F.linear(normed, k_w).view(B, S, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = F.linear(normed, v_w).view(B, S, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if cfg.num_kv_heads != cfg.num_heads:
            rep = cfg.num_heads // cfg.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, cfg.q_dim)
        attn = F.linear(attn, o_w)
        x = x + attn
        del attn_norm_w, q_w, k_w, v_w, o_w, normed, q, k, v, attn
        gc.collect()
        # --- ffn ---
        ffn_norm_w = model._load_norm(f"{p}.ffn_norm.weight")
        gate_w = model._load_linear(f"{p}.ffn_gate.weight", cfg.intermediate_dim, cfg.hidden_dim)
        up_w = model._load_linear(f"{p}.ffn_up.weight", cfg.intermediate_dim, cfg.hidden_dim)
        down_w = model._load_linear(f"{p}.ffn_down.weight", cfg.hidden_dim, cfg.intermediate_dim)
        if mode == "FP":
            normed = model._rmsnorm(x, ffn_norm_w)
        else:
            key = f"blk.{i}.ffn_norm"
            normed = quant_rmsnorm(x, ffn_norm_w, params[key]["s_in"], params[key]["s_out"])
        gate = F.gelu(F.linear(normed, gate_w))
        up = F.linear(normed, up_w)
        mlp = F.linear(gate * up, down_w)
        x = x + mlp
        del ffn_norm_w, gate_w, up_w, down_w, normed, gate, up, mlp
        gc.collect()

    # --- final norm ---
    final_w = model._load_norm("output_norm.weight")
    if mode == "FP":
        x_final = model._rmsnorm(x, final_w)
    else:
        key = "output_norm.final_norm"
        x_final = quant_rmsnorm(x, final_w, params[key]["s_in"], params[key]["s_out"])
    del final_w
    gc.collect()
    # --- lm_head (tied) ---
    logits = F.linear(x_final, emb_w)  # [B,S,vocab]
    del emb_w, x_final, x
    gc.collect()
    return logits[0].detach().float().cpu().numpy()  # [S, vocab]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--params", default="diagnostics/ex_norm10_capture_fixed_v2/norm_selected_parameters.json")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    ap.add_argument("--num_conversations", type=int, default=3)
    ap.add_argument("--targets_per_conv", type=int, default=8)
    args = ap.parse_args()

    params = json.load(open(args.params))
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    tokenizer = GemmaTokenizer()
    # smoke: 3 convs, prefill 128 고정 (md smoke와 유사)
    convs = build_conversations(args.num_conversations, prefill_lengths=[128], decode_lengths=[16])
    rng = random.Random(42)

    total_agree = 0; total_n = 0
    kl_list = []; nmse_num = 0.0; nmse_den = 0.0
    per_conv = []
    for ci, conv in enumerate(convs):
        prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        if not prompt_ids: continue
        # 8 targets: prompt + 8개 다음 토큰 위치를 평가 (teacher-forced, 랜덤 decode 사용)
        decode_ids = [rng.randint(0, model.cfg.vocab_size - 1) for _ in range(32)]
        full_ids = prompt_ids + decode_ids
        # targets: 마지막 targets_per_conv개 위치만 평가 (마지막 위치는 next 없음)
        input_tensor = torch.tensor([full_ids], dtype=torch.long)
        print(f"[*] conv {ci} S={len(full_ids)} FP forward...")
        logits_fp = forward_logits(model, input_tensor, "FP", params)
        print(f"[*] conv {ci} IO_ORACLE forward...")
        logits_q = forward_logits(model, input_tensor, "IO_ORACLE", params)
        # 평가 위치: prefill 뒤쪽 + decode 앞쪽, 마지막 위치 제외
        S = logits_fp.shape[0]
        eval_pos = list(range(max(0, S - args.targets_per_conv - 1), S - 1))
        agree = 0
        for pos in eval_pos:
            if np.argmax(logits_fp[pos]) == np.argmax(logits_q[pos]):
                agree += 1
        # KL, NMSE는 eval_pos에 대해서 pooled
        fp = logits_fp[eval_pos].astype(np.float64)
        qq = logits_q[eval_pos].astype(np.float64)
        # softmax 안정화
        fp_s = fp - fp.max(axis=-1, keepdims=True)
        qq_s = qq - qq.max(axis=-1, keepdims=True)
        p = np.exp(fp_s) / np.exp(fp_s).sum(axis=-1, keepdims=True)
        qprob = np.exp(qq_s) / np.exp(qq_s).sum(axis=-1, keepdims=True)
        kl = np.sum(p * (np.log(p + 1e-12) - np.log(qprob + 1e-12)), axis=-1).mean()
        nmse_num += float(np.sum((qq - fp) ** 2))
        nmse_den += float(np.sum(fp ** 2))
        total_agree += agree; total_n += len(eval_pos)
        kl_list.append(float(kl))
        per_conv.append({"conv": ci, "n": len(eval_pos), "agree": agree, "top1": agree/len(eval_pos), "kl": float(kl)})
        print(f"  conv {ci}: Top1 {agree}/{len(eval_pos)}={agree/len(eval_pos):.3f} KL={kl:.4f}")
        del logits_fp, logits_q
        gc.collect()

    summary = {
        "mode": "FP_FULL vs IO_ORACLE (s_in/s_out freeze)",
        "total_top1": total_agree/total_n if total_n else 0,
        "total_n": total_n,
        "mean_kl": float(np.mean(kl_list)) if kl_list else 0,
        "logits_nmse_pooled": nmse_num/nmse_den if nmse_den else 0,
        "per_conv": per_conv,
    }
    with open(out_dir/"e2e_smoke_summary.json","w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== E2E Smoke ===")
    print(f"Top1 agreement: {total_agree}/{total_n} = {summary['total_top1']:.4f}")
    print(f"Mean KL: {summary['mean_kl']:.5f}")
    print(f"Logits NMSE (pooled): {summary['logits_nmse_pooled']:.6f}")
    print(f"[✓] e2e_smoke_summary.json 저장")

if __name__=="__main__":
    main()
