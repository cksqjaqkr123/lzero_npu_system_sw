#!/usr/bin/env python3
"""
P0-B 검증: calibration/validation 분리 + 1024-target E2E
- Calibration: 기존 V2 (8 conv, prefill 128) 로 s_in/s_out 선택
- Validation: 다른 프롬프트 (prefill 256, 다른 대화) 로 평가 — 같은 대화로 평가한 게 아님
- 1024-target: 8 conv * 127 positions (BOS 제외 전체) = 1016 ≈ 1024
- BOS_FP만 평가 (all_INT10은 이미 8.33%로 알려짐, 시간 절약)
"""
import argparse, gc, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations
from run_e2e_bos_bypass import forward_e2e, forward_fp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--params", default="/tmp/p9999_absmax.json")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    ap.add_argument("--n_conv", type=int, default=8)
    ap.add_argument("--prefill", type=int, default=256)
    args = ap.parse_args()
    params = json.load(open(args.params))
    out_dir = Path(args.output_dir)
    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    tokenizer = GemmaTokenizer()
    # Validation은 calibration과 다른 prefill 길이 (256 vs 128) + 다른 대화
    convs = build_conversations(args.n_conv, prefill_lengths=[args.prefill]*args.n_conv, decode_lengths=[16]*args.n_conv)
    print(f"[*] Validation: {args.n_conv} conv, prefill {args.prefill} (calibration과 다른 대화)")
    all_agree = 0
    all_n = 0
    all_kl = []
    all_nmse = []
    rows = []
    for ci, conv in enumerate(convs):
        prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        S = len(prompt_ids)
        print(f"[*] conv{ci} S={S} FP...")
        logits_fp = forward_fp(model, input_tensor)
        # BOS 제외 전체 위치 (1..S-1) = 1016개 목표
        eval_pos = list(range(1, S))
        print(f"[*] conv{ci} BOS_FP forward... eval n={len(eval_pos)}")
        logits_q = forward_e2e(model, input_tensor, params, True)
        agree = sum(1 for pos in eval_pos if np.argmax(logits_fp[pos]) == np.argmax(logits_q[pos]))
        fp = logits_fp[eval_pos].astype(np.float64); qq = logits_q[eval_pos].astype(np.float64)
        fp_s = fp - fp.max(axis=-1, keepdims=True); qq_s = qq - qq.max(axis=-1, keepdims=True)
        p = np.exp(fp_s)/np.exp(fp_s).sum(axis=-1, keepdims=True)
        qprob = np.exp(qq_s)/np.exp(qq_s).sum(axis=-1, keepdims=True)
        kl = float(np.sum(p*(np.log(p+1e-12)-np.log(qprob+1e-12)), axis=-1).mean())
        nmse = float(np.sum((qq-fp)**2)/np.sum(fp**2))
        rows.append({"conv": ci, "mode": "BOS_FP", "top1": agree/len(eval_pos), "agree": agree, "n": len(eval_pos), "kl": kl, "logits_nmse": nmse, "prefill": args.prefill})
        print(f"  conv{ci} BOS_FP Top1 {agree}/{len(eval_pos)}={agree/len(eval_pos):.3f} KL={kl:.4f} NMSE={nmse:.4f}")
        all_agree += agree; all_n += len(eval_pos); all_kl.append(kl); all_nmse.append(nmse)
        del logits_fp, logits_q; gc.collect()
    print(f"[SUMMARY] Validation (calibration과 다른 대화) Top1 {all_agree}/{all_n}={all_agree/all_n:.3f} KL avg {sum(all_kl)/len(all_kl):.4f} NMSE avg {sum(all_nmse)/len(all_nmse):.4f}")
    with open(out_dir/"e2e_1024_validation.json","w") as f:
        json.dump(rows, f, indent=2)
    print(f"[✓] e2e_1024_validation.json 저장 (n={all_n})")

if __name__=="__main__":
    main()
