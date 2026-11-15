#!/usr/bin/env python3
"""
p999 vs p9999 robustness: 대화만 바꿨는데 우열이 뒤집히는지.
- blk.0 attn만 양자화, s_out을 p999 / p9999 / absmax로 바꿔서
- 대화 2개 (conv0, conv1)에 대해 KL/NMSE/Top1 측정
- 만약 conv0에서는 p999가 좋고 conv1에서는 p9999가 좋으면 -> scale tuning은 fragile
"""
import argparse, gc, json, random
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations, compute_rope_cache, apply_rope
from run_e2e_smoke import quant_rmsnorm

def forward_blk0_attn(model, input_ids, params, s_out_blk0_attn):
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
        attn_norm_w = model._load_norm(f"{p}.attn_norm.weight")
        q_w = model._load_linear(f"{p}.attn_q.weight", cfg.q_dim, cfg.hidden_dim)
        k_w = model._load_linear(f"{p}.attn_k.weight", cfg.kv_dim, cfg.hidden_dim)
        v_w = model._load_linear(f"{p}.attn_v.weight", cfg.kv_dim, cfg.hidden_dim)
        o_w = model._load_linear(f"{p}.attn_output.weight", cfg.hidden_dim, cfg.q_dim)
        if is_blk0:
            normed = quant_rmsnorm(x, attn_norm_w, params["blk.0.attn_norm"]["s_in"], s_out_blk0_attn)
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
    # 2개 대화, 각각 다른 prompt
    convs = build_conversations(2, prefill_lengths=[128, 128], decode_lengths=[16, 16])
    # s_out 후보: p999 0.02733, p9999 (selected), absmax 0.15898
    s_out_p9999 = params["blk.0.attn_norm"]["s_out"]
    candidates = [
        ("p999", 0.02733),
        ("p9999", s_out_p9999),
        ("absmax", 0.15898),
    ]
    print(f"[*] s_out p9999 (selected) = {s_out_p9999}")
    rows = []
    for ci, conv in enumerate(convs):
        rng = random.Random(42 + ci)  # 대화마다 다른 decode 토큰
        prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        full_ids = prompt_ids + [rng.randint(0, model.cfg.vocab_size-1) for _ in range(32)]
        input_tensor = torch.tensor([full_ids], dtype=torch.long)
        print(f"[*] conv{ci} S={len(full_ids)} FP baseline...")
        logits_fp = forward_blk0_attn(model, input_tensor, params, s_out_p9999)  # FP는 s_out 안 쓰지만, dummy로 호출하면 안 됨 -> 아래에서 FP용 따로 처리
        # FP baseline은 quant 없이: forward_blk0_attn에서 s_out을 써도 blk0 attn을 양자화하므로, FP를 얻으려면 별도 경로가 필요.
        # 간단히: s_out을 None으로 하면 FP로 처리하도록 forward를 다시 호출하지 않고, 여기서는 FP를 quant 함수 우회로 계산.
        # -> 위 호출은 잘못됐으니, FP는 아래와 같이 직접 계산: 여기서는 logits_fp를 버리고 FP forward를 다시 정의
        del logits_fp; gc.collect()
        # FP forward: blk0 attn도 FP로 (s_out=None이면 quant_rmsnorm을 안 타도록 별도 함수 호출)
        # 편의를 위해 forward_blk0_attn에 s_out=None이면 FP로 동작하게 패치된 버전이 아니므로, 여기서는 quant_rmsnorm 대신 _rmsnorm을 쓰는 인라인 FP를 재사용
        # -> 간단히: params의 s_out을 무시하고, 아래 loop에서 FP를 따로 계산하기 위해 forward_blk0_attn을 s_out=None으로 호출할 수 있게 임시로 처리
        # 실제로는 forward_blk0_attn이 s_out=None이면 에러가 나므로, FP를 위해 별도 호출을 만든다:
        # (아래에서 FP를 계산하는 헬퍼를 다시 정의)
        def forward_fp():
            # FP 전용: blk0 attn도 FP
            return forward_blk0_attn_fp(model, input_tensor)
        logits_fp = forward_fp()
        S = logits_fp.shape[0]
        eval_pos = list(range(max(0, S-9), S-1))
        for name, s_out in candidates:
            print(f"[*] conv{ci} s_out={name} ({s_out:.5f}) forward...")
            logits_q = forward_blk0_attn(model, input_tensor, params, s_out)
            agree = sum(1 for pos in eval_pos if np.argmax(logits_fp[pos]) == np.argmax(logits_q[pos]))
            fp = logits_fp[eval_pos].astype(np.float64); qq = logits_q[eval_pos].astype(np.float64)
            fp_s = fp - fp.max(axis=-1, keepdims=True); qq_s = qq - qq.max(axis=-1, keepdims=True)
            p = np.exp(fp_s)/np.exp(fp_s).sum(axis=-1, keepdims=True)
            qprob = np.exp(qq_s)/np.exp(qq_s).sum(axis=-1, keepdims=True)
            kl = float(np.sum(p*(np.log(p+1e-12)-np.log(qprob+1e-12)), axis=-1).mean())
            nmse = float(np.sum((qq-fp)**2)/np.sum(fp**2))
            rows.append({"conv": ci, "s_out_name": name, "s_out": s_out, "top1": agree/len(eval_pos), "agree": agree, "n": len(eval_pos), "kl": kl, "logits_nmse": nmse})
            print(f"  conv{ci} {name:8s} Top1 {agree}/{len(eval_pos)}={agree/len(eval_pos):.3f} KL={kl:.4f} NMSE={nmse:.4f}")
            del logits_q; gc.collect()
        del logits_fp; gc.collect()
    with open(out_dir/"sout_robustness.json","w") as f:
        json.dump(rows, f, indent=2)
    print(f"[✓] sout_robustness.json 저장")

def forward_blk0_attn_fp(model, input_ids):
    """FP 전용: 모든 Norm FP"""
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

if __name__=="__main__":
    main()
