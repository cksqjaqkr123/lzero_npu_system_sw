#!/usr/bin/env python3
"""
P6 BOS INT10 - clean rebuild
- BOS FP bypass 없음. BOS도 INT10, pos 0만 전용 스케일 사용
- 일반 토큰: calib_32_scales.json (p9999 / 511)
- BOS 토큰: calib_bos_32.json (BOS p9999 / 511)
- GELU: s_in per-layer best (NMSE 최소, p999면 p9999로 통일), s_out absmax
- RoPE: s_q, s_k = p9999 / 511, BOS도 동일 스케일로 INT10
- RMSNorm: s_in, s_out 둘 다 양자화, BOS만 전용 스케일로 덮어씀
사용법:
  PYTHONPATH=diagnostics/ex_norm10_capture_fixed python3 -u diagnostics/ex_norm10_p6_bos_int10/run/run_tail_sweep.py --rms_json diagnostics/ex_norm10_p6_bos_int10/calib/calib_32_scales.json --rope_calib diagnostics/ex_norm10_p4/e2e/rope_calib.json --n_conv 1 --prefill 256 --tail_k 26 --output_dir diagnostics/ex_norm10_p6_bos_int10/results/smoke_1conv
"""
import argparse, gc, json, csv
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations, compute_rope_cache, apply_rope

def log(*a, **k):
    print(*a, flush=True, **k)

def build_gelu_lut(s_in, s_out):
    idx = np.arange(-511, 512, dtype=np.float64)
    deq = idx * float(s_in)
    gelu = 0.5 * deq * (1 + np.tanh(np.sqrt(2/np.pi)*(deq + 0.044715*deq**3)))
    q = np.round(gelu / float(s_out)).clip(-511, 511)
    return q.astype(np.int64)

def apply_gelu_lut(gate, s_in, s_out, lut):
    code = torch.round(gate / float(s_in)).clamp(-511, 511).to(torch.int64) + 511
    lut_t = torch.from_numpy(lut).to(code.device)
    out_code = lut_t[code]
    return (out_code.float() * float(s_out)).to(gate.dtype)

def quant_int10(x, s):
    out = torch.round(x / float(s)).clamp(-511, 511) * float(s)
    return out.to(x.dtype)

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

@torch.no_grad()
def forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales, int10_layers=None, bos_scales=None):
    if int10_layers is None:
        int10_layers = set(range(26))
    if bos_scales is None:
        bos_scales = {}
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
        sc = rms_scales[f"{p}.attn_norm"]
        if i in int10_layers:
            x_q = quant_int10(x, sc["s_in"])
            if bos_scales:
                _bos = bos_scales.get(f"{p}.attn_norm", sc)
                x_q[:, 0:1, :] = quant_int10(x[:, 0:1, :], _bos["s_in"])
            normed = model._rmsnorm(x_q, attn_norm_w)
            normed_q = quant_int10(normed, sc["s_out"])
            if bos_scales:
                _bos = bos_scales.get(f"{p}.attn_norm", sc)
                normed_q[:, 0:1, :] = quant_int10(normed[:, 0:1, :], _bos["s_out"])
        else:
            normed_q = model._rmsnorm(x, attn_norm_w)
            x_q = x
            normed = normed_q
        q = F.linear(normed_q, q_w).view(B, S, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = F.linear(normed_q, k_w).view(B, S, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = F.linear(normed_q, v_w).view(B, S, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        rs = rope_scales[f"blk.{i}"]
        q = quant_int10(q, rs["s_q"])
        k = quant_int10(k, rs["s_k"])
        q, k = apply_rope(q, k, cos, sin)
        if cfg.num_kv_heads != cfg.num_heads:
            rep = cfg.num_heads // cfg.num_kv_heads
            k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, cfg.q_dim)
        attn = F.linear(attn, o_w); x = x + attn
        del attn_norm_w, q_w, k_w, v_w, o_w, normed, normed_q, x_q, q, k, v, attn
        ffn_norm_w = model._load_norm(f"{p}.ffn_norm.weight")
        gate_w = model._load_linear(f"{p}.ffn_gate.weight", cfg.intermediate_dim, cfg.hidden_dim)
        up_w = model._load_linear(f"{p}.ffn_up.weight", cfg.intermediate_dim, cfg.hidden_dim)
        down_w = model._load_linear(f"{p}.ffn_down.weight", cfg.hidden_dim, cfg.intermediate_dim)
        sc2 = rms_scales[f"{p}.ffn_norm"]
        if i in int10_layers:
            x_q2 = quant_int10(x, sc2["s_in"])
            if bos_scales:
                _bos = bos_scales.get(f"{p}.ffn_norm", sc2)
                x_q2[:, 0:1, :] = quant_int10(x[:, 0:1, :], _bos["s_in"])
            normed2 = model._rmsnorm(x_q2, ffn_norm_w)
            normed2_q = quant_int10(normed2, sc2["s_out"])
            if bos_scales:
                _bos = bos_scales.get(f"{p}.ffn_norm", sc2)
                normed2_q[:, 0:1, :] = quant_int10(normed2[:, 0:1, :], _bos["s_out"])
        else:
            normed2_q = model._rmsnorm(x, ffn_norm_w)
            x_q2 = x
            normed2 = normed2_q
        gate = F.linear(normed2_q, gate_w)
        up = F.linear(normed2_q, up_w)
        gs = gelu_scales[f"blk.{i}"]
        lut = gelu_luts[f"blk.{i}"]
        gelu_q = apply_gelu_lut(gate, gs["s_in"], gs["s_out"], lut)
        mlp = F.linear(gelu_q * up, down_w); x = x + mlp
        del ffn_norm_w, gate_w, up_w, down_w, normed2, normed2_q, x_q2, gate, up, gelu_q, mlp
    final_w = model._load_norm("output_norm.weight")
    scf = rms_scales["output_norm"]
    x_qf = quant_int10(x, scf["s_in"])
    if bos_scales:
        _bos = bos_scales.get("output_norm", scf)
        x_qf[:, 0:1, :] = quant_int10(x[:, 0:1, :], _bos["s_in"])
    x_final = model._rmsnorm(x_qf, final_w)
    x_final_q = quant_int10(x_final, scf["s_out"])
    if bos_scales:
        _bos = bos_scales.get("output_norm", scf)
        x_final_q[:, 0:1, :] = quant_int10(x_final[:, 0:1, :], _bos["s_out"])
    del final_w
    logits = F.linear(x_final_q, emb_w)
    out = logits[0].detach().float().cpu().numpy()
    del emb_w, x_final, x_final_q, x_qf, x, logits
    gc.collect()
    return out

@torch.no_grad()
def calibrate_rope(model, tokenizer, n_calib, prefill):
    log(f"[CALIB] RoPE calibration n={n_calib} prefill={prefill}")
    convs = build_conversations(n_calib, prefill_lengths=[prefill]*n_calib, decode_lengths=[16]*n_calib)
    cfg = model.cfg
    all_q = {f"blk.{i}": [] for i in range(cfg.num_layers)}
    all_k = {f"blk.{i}": [] for i in range(cfg.num_layers)}
    for ci, conv in enumerate(convs):
        prompt_text = conv["prompt_text"] + " [calibration]"
        prompt_ids = tokenizer.encode(prompt_text, max_length=prefill)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
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
            del attn_norm_w, q_w, k_w, v_w, o_w, normed, v, attn
            all_q[f"blk.{i}"].append(q[:, :, 1:, :].detach().float().cpu().numpy().reshape(-1))
            all_k[f"blk.{i}"].append(k[:, :, 1:, :].detach().float().cpu().numpy().reshape(-1))
            del q, k
            ffn_norm_w = model._load_norm(f"{p}.ffn_norm.weight")
            gate_w = model._load_linear(f"{p}.ffn_gate.weight", cfg.intermediate_dim, cfg.hidden_dim)
            up_w = model._load_linear(f"{p}.ffn_up.weight", cfg.intermediate_dim, cfg.hidden_dim)
            down_w = model._load_linear(f"{p}.ffn_down.weight", cfg.hidden_dim, cfg.intermediate_dim)
            normed = model._rmsnorm(x, ffn_norm_w)
            gate = F.gelu(F.linear(normed, gate_w)); up = F.linear(normed, up_w)
            mlp = F.linear(gate * up, down_w); x = x + mlp
            del ffn_norm_w, gate_w, up_w, down_w, normed, gate, up, mlp
        del emb_w, x
        gc.collect()
        log(f"  [CALIB] conv{ci} done S={S}")
    rope_scales = {}
    for i in range(cfg.num_layers):
        q_all = np.concatenate(all_q[f"blk.{i}"])
        k_all = np.concatenate(all_k[f"blk.{i}"])
        s_q = float(np.quantile(np.abs(q_all), 0.9999) / 511)
        s_k = float(np.quantile(np.abs(k_all), 0.9999) / 511)
        rope_scales[f"blk.{i}"] = {"s_q": s_q, "s_k": s_k, "q_p9999": float(np.quantile(np.abs(q_all), 0.9999)), "k_p9999": float(np.quantile(np.abs(k_all), 0.9999))}
        log(f"  [CALIB] blk.{i} s_q={s_q:.6f} s_k={s_k:.6f}")
    return rope_scales

def load_rms_scales(rms_json_path):
    data = json.load(open(rms_json_path))
    rms_scales = {}
    if isinstance(data, dict) and "norms" in data:
        items = data["norms"]
        for n in items:
            name = n.get("name") or n.get("norm") or n.get("id")
            if name: rms_scales[name] = {"s_in": float(n["s_in"]), "s_out": float(n["s_out"])}
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict) and "s_in" in v:
                rms_scales[k] = {"s_in": float(v["s_in"]), "s_out": float(v["s_out"])}
    elif isinstance(data, list):
        for n in data:
            name = n.get("name")
            if name: rms_scales[name] = {"s_in": float(n["s_in"]), "s_out": float(n["s_out"])}
    log(f"[DATA] RMSNorm keys sample: {list(rms_scales.keys())[:5]} ... total {len(rms_scales)}")
    if "output_norm" not in rms_scales:
        for k in list(rms_scales.keys()):
            if "output" in k.lower() or "final" in k.lower():
                rms_scales["output_norm"] = rms_scales[k]
                log(f"[DATA] output_norm 매핑: {k} -> output_norm s_in={rms_scales[k]['s_in']:.6f} s_out={rms_scales[k]['s_out']:.6f}")
                break
    if "output_norm" not in rms_scales:
        for k in list(rms_scales.keys()):
            if not k.startswith("blk."):
                rms_scales["output_norm"] = rms_scales[k]
                log(f"[DATA] output_norm 매핑 (non-blk): {k} -> output_norm")
                break
    return rms_scales

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--rms_json", default="diagnostics/ex_norm10_p6_bos_int10/calib/calib_32_scales.json")
    ap.add_argument("--gelu_csv", default="diagnostics/ex_norm10_p4/gelu/gelu_local.csv")
    ap.add_argument("--rope_calib", default=None, help="기존 rope_calib.json 재사용")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_p6_bos_int10/results")
    ap.add_argument("--n_calib", type=int, default=4)
    ap.add_argument("--calib_prefill", type=int, default=128)
    ap.add_argument("--n_conv", type=int, default=2)
    ap.add_argument("--prefill", type=int, default=256)
    ap.add_argument("--tail_k", type=int, default=26)
    args = ap.parse_args()
    log(f"[START] P6 BOS INT10 Combined E2E n_conv={args.n_conv} prefill={args.prefill}")
    int10_layers = set(range(26 - args.tail_k, 26))
    log(f"[TAIL] tail_k={args.tail_k} int10_layers={sorted(int10_layers)}")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rms_scales = load_rms_scales(args.rms_json)
    log(f"[DATA] RMSNorm from {args.rms_json}")
    bos_scales = {}
    BOS_PATH = "diagnostics/ex_norm10_p6_bos_int10/calib/calib_bos_absmax_1conv.json"
    try:
        bos_scales = load_rms_scales(BOS_PATH)
        log(f"[DATA] BOS scales from {BOS_PATH} keys={len(bos_scales)}")
        log(f"[DATA] BOS sample blk.0.attn_norm s_in={bos_scales['blk.0.attn_norm']['s_in']:.6f} s_out={bos_scales['blk.0.attn_norm']['s_out']:.6f} vs general s_in={rms_scales['blk.0.attn_norm']['s_in']:.6f} s_out={rms_scales['blk.0.attn_norm']['s_out']:.6f}")
    except Exception as e:
        log(f"[WARN] BOS load fail {e}, BOS will use general scales (still INT10, no FP)")
        bos_scales = {}
    sin_vals, sout_vals = {}, {}
    with open(args.gelu_csv) as f:
        for row in csv.DictReader(f):
            layer, tag = row.get("layer",""), row.get("tag","")
            if tag.startswith("gelu_nmse_sin_"):
                cand = tag.replace("gelu_nmse_sin_", "")
                sin_vals.setdefault(layer, {})[cand] = {"nmse": float(row["nmse"]), "s": float(row["s_in"]) if row.get("s_in") else None}
            elif tag.startswith("gelu_nmse_sout_"):
                cand = tag.replace("gelu_nmse_sout_", "")
                sout_vals.setdefault(layer, {})[cand] = {"nmse": float(row["nmse"]), "s": float(row["s_out"]) if row.get("s_out") else None}
    gelu_scales, gelu_luts = {}, {}
    for layer in sorted(sin_vals.keys()):
        d = sin_vals[layer]
        best = min(d, key=lambda k: d[k]["nmse"])
        if best == "p999": best = "p9999"
        s_in = d[best]["s"]
        s_out = sout_vals[layer]["absmax"]["s"]
        gelu_scales[layer] = {"s_in": s_in, "s_out": s_out, "s_in_mode": best}
        gelu_luts[layer] = build_gelu_lut(s_in, s_out)
    log(f"[DATA] GELU LUT 26개 준비 완료")
    log(f"[MODEL] loading {args.gguf} ...")
    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    log(f"[MODEL] loaded")
    tokenizer = GemmaTokenizer()
    if args.rope_calib and Path(args.rope_calib).exists():
        rope_scales = json.load(open(args.rope_calib))
        log(f"[CALIB] 기존 {args.rope_calib} 재사용, calibration 스킵")
    else:
        rope_scales = calibrate_rope(model, tokenizer, args.n_calib, args.calib_prefill)
        with open(out_dir/"rope_calib.json","w") as f:
            json.dump(rope_scales, f, indent=2)
        log(f"[CALIB] rope_calib.json 저장")
    convs = build_conversations(args.n_conv, prefill_lengths=[args.prefill]*args.n_conv, decode_lengths=[16]*args.n_conv)
    log(f"[CONV] {len(convs)} validation convs")
    jsonl_path = out_dir/"e2e_combined_strict_partial.jsonl"
    for ci, conv in enumerate(convs):
        prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        S = len(prompt_ids)
        log(f"[*] conv{ci} S={S} FP start...")
        logits_fp = forward_fp(model, input_tensor)
        log(f"[*] conv{ci} FP done, Combined Strict BOS INT10 start...")
        logits_q = forward_combined_strict(model, input_tensor, rms_scales, gelu_luts, gelu_scales, rope_scales, int10_layers=int10_layers, bos_scales=bos_scales)
        eval_pos = list(range(1, S))
        agree = sum(1 for pos in eval_pos if np.argmax(logits_fp[pos]) == np.argmax(logits_q[pos]))
        fp = logits_fp[eval_pos].astype(np.float64); qq = logits_q[eval_pos].astype(np.float64)
        fp_s = fp - fp.max(axis=-1, keepdims=True); qq_s = qq - qq.max(axis=-1, keepdims=True)
        p = np.exp(fp_s)/np.exp(fp_s).sum(axis=-1, keepdims=True)
        qprob = np.exp(qq_s)/np.exp(qq_s).sum(axis=-1, keepdims=True)
        kl = float(np.sum(p*(np.log(p+1e-12)-np.log(qprob+1e-12)), axis=-1).mean())
        nmse = float(np.sum((qq-fp)**2)/np.sum(fp**2))
        rec = {"conv": ci, "top1": agree/len(eval_pos), "agree": agree, "n": len(eval_pos), "kl": kl, "logits_nmse": nmse}
        log(f"  [DONE] conv{ci} Combined Strict BOS INT10 Top1 {agree}/{len(eval_pos)}={agree/len(eval_pos):.3f} KL={kl:.4f} NMSE={nmse:.4f}")
        with open(jsonl_path, "a") as f: f.write(json.dumps(rec)+"\n")
        del logits_fp, logits_q; gc.collect()
    log(f"[DONE] partial saved to {jsonl_path}")
    totals = {"agree":0,"n":0,"kl":[]}
    for line in open(jsonl_path):
        r = json.loads(line)
        totals["agree"] += r["agree"]; totals["n"] += r["n"]; totals["kl"].append(r["kl"])
    log(f"[SUMMARY] Combined Strict BOS INT10 Top1 {totals['agree']}/{totals['n']}={totals['agree']/totals['n']:.3f} KL avg {sum(totals['kl'])/len(totals['kl']):.4f}")

if __name__=="__main__":
    main()
