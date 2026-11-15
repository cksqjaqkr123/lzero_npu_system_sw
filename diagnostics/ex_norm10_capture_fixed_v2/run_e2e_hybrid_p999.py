#!/usr/bin/env python3
"""
Hybrid p999 테스트: p999가 쓸만한 레이어만 p999로, 나머지는 p9999로
- 기준: Local NMSE ratio = NMSE(p999) / NMSE(p9999) < threshold (기본 2.0)
  -> p999가 p9999보다 2배 이내로만 나쁜 레이어만 p999로 씀
- 클러스터링은 안 함 (per-Norm 그대로) — p999 효과만 분리해서 보기 위해
  클러스터링 오차(86.5%->82.9%)와 섞이면 뭐 때문인지 모름
- 비교: per-Norm p9999 전부 (86.5%, KL 0.109) vs Hybrid
"""
import argparse, csv, gc, json
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
import torch.nn.functional as F
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations, compute_rope_cache, apply_rope
from run_e2e_bos_bypass import quant_bos_fp
from run_e2e_sout_lut_1500 import forward_fp, forward_e2e_lut

def load_hybrid(sweep_path, ratio_threshold):
    rows = []
    with open(sweep_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    grouped = defaultdict(dict)
    for row in rows:
        norm = f"{row['layer_name'].strip()}.{row['norm_type'].strip()}"
        grouped[norm][row['s_out_candidate'].strip()] = (float(row['s_out']), float(row['nmse']))
    hybrid = {}
    selected = []
    for norm, d in grouped.items():
        s999, nmse999 = d["p999"]
        s9999, nmse9999 = d["p9999"]
        ratio = nmse999 / max(nmse9999, 1e-12)
        if ratio < ratio_threshold:
            hybrid[norm] = s999
            selected.append({"norm": norm, "choice": "p999", "s_out": s999, "nmse_p999": nmse999, "nmse_p9999": nmse9999, "ratio": ratio})
        else:
            hybrid[norm] = s9999
            # 선택 안 된 건 기록 안 함 (너무 많음)
    selected = sorted(selected, key=lambda x: x["ratio"])
    return hybrid, selected, grouped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--params", default="/tmp/p9999_absmax.json")
    ap.add_argument("--sweep", default="diagnostics/ex_norm10_capture_fixed_v2/sout_sweep_all.csv")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    ap.add_argument("--ratio_threshold", type=float, default=2.0)
    ap.add_argument("--n_conv", type=int, default=8)
    ap.add_argument("--prefill", type=int, default=256)
    args = ap.parse_args()
    hybrid, selected, grouped = load_hybrid(args.sweep, args.ratio_threshold)
    print(f"[*] Hybrid 기준: ratio < {args.ratio_threshold} -> p999 선택 {len(selected)}개 / 전체 {len(grouped)}개")
    for s in selected:
        print(f"  {s['norm']}: p999 s_out={s['s_out']:.6f} NMSE p999={s['nmse_p999']:.5f} vs p9999={s['nmse_p9999']:.5f} ratio={s['ratio']:.2f}")
    print(f"[*] 나머지 {len(grouped)-len(selected)}개는 p9999")
    params = json.load(open(args.params))
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
        print(f"[*] conv{ci} Hybrid BOS_FP forward... eval n={len(eval_pos)}")
        logits_q = forward_e2e_lut(model, input_tensor, params, hybrid, True)
        agree = sum(1 for pos in eval_pos if np.argmax(logits_fp[pos]) == np.argmax(logits_q[pos]))
        fp = logits_fp[eval_pos].astype(np.float64); qq = logits_q[eval_pos].astype(np.float64)
        fp_s = fp - fp.max(axis=-1, keepdims=True); qq_s = qq - qq.max(axis=-1, keepdims=True)
        p = np.exp(fp_s)/np.exp(fp_s).sum(axis=-1, keepdims=True)
        qprob = np.exp(qq_s)/np.exp(qq_s).sum(axis=-1, keepdims=True)
        kl = float(np.sum(p*(np.log(p+1e-12)-np.log(qprob+1e-12)), axis=-1).mean())
        nmse = float(np.sum((qq-fp)**2)/np.sum(fp**2))
        rows.append({"conv": ci, "mode": "Hybrid_p999_BOS_FP", "top1": agree/len(eval_pos), "agree": agree, "n": len(eval_pos), "kl": kl, "logits_nmse": nmse, "prefill": args.prefill, "ratio_threshold": args.ratio_threshold, "n_p999": len(selected)})
        print(f"  conv{ci} Hybrid Top1 {agree}/{len(eval_pos)}={agree/len(eval_pos):.3f} KL={kl:.4f} NMSE={nmse:.4f}")
        all_agree += agree; all_n += len(eval_pos); all_kl.append(kl); all_nmse.append(nmse)
        del logits_fp, logits_q; gc.collect()
    print(f"[SUMMARY] Hybrid (p999 {len(selected)}개 + p9999 {len(grouped)-len(selected)}개, 클러스터링 없음) Top1 {all_agree}/{all_n}={all_agree/all_n:.3f} KL avg {sum(all_kl)/len(all_kl):.4f} NMSE avg {sum(all_nmse)/len(all_nmse):.4f}")
    print(f"[비교] per-Norm p9999 전부 (53개, 클러스터링 없음): Top1 86.5% KL 0.109")
    print(f"[비교] 클러스터 LUT p9999 (19개): Top1 82.9% KL 0.184")
    out = {"ratio_threshold": args.ratio_threshold, "selected_p999": selected, "rows": rows, "total": {"agree": all_agree, "n": all_n, "top1": all_agree/all_n, "kl_avg": sum(all_kl)/len(all_kl)}}
    with open(out_dir/"e2e_hybrid_p999.json","w") as f:
        json.dump(out, f, indent=2)
    print(f"[✓] e2e_hybrid_p999.json 저장 (n={all_n})")

if __name__=="__main__":
    main()
