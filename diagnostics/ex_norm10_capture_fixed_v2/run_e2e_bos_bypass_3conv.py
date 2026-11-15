#!/usr/bin/env python3
"""
E2E BOS bypass 3대화 검증: 1개 대화가 아니라 3개에서도 90%가 나오는지
- 각 대화: 프롬프트만, 마지막 32개
- BOS_FP vs all_INT10
"""
import argparse, gc, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations, compute_rope_cache, apply_rope
from run_e2e_bos_bypass import forward_e2e, forward_fp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--params", default="/tmp/p9999_absmax.json")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    args = ap.parse_args()
    params = json.load(open(args.params))
    out_dir = Path(args.output_dir)
    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    tokenizer = GemmaTokenizer()
    convs = build_conversations(3, prefill_lengths=[128, 128, 128], decode_lengths=[16, 16, 16])
    all_rows = []
    for ci, conv in enumerate(convs):
        prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        S = len(prompt_ids)
        print(f"[*] conv{ci} S={S} FP baseline...")
        logits_fp = forward_fp(model, input_tensor)
        eval_pos = list(range(max(0, S-33), S-1))
        for bos_fp, name in [(False, "all_INT10"), (True, "BOS_FP")]:
            print(f"[*] conv{ci} {name} forward...")
            logits_q = forward_e2e(model, input_tensor, params, bos_fp)
            agree = sum(1 for pos in eval_pos if np.argmax(logits_fp[pos]) == np.argmax(logits_q[pos]))
            fp = logits_fp[eval_pos].astype(np.float64); qq = logits_q[eval_pos].astype(np.float64)
            fp_s = fp - fp.max(axis=-1, keepdims=True); qq_s = qq - qq.max(axis=-1, keepdims=True)
            p = np.exp(fp_s)/np.exp(fp_s).sum(axis=-1, keepdims=True)
            qprob = np.exp(qq_s)/np.exp(qq_s).sum(axis=-1, keepdims=True)
            kl = float(np.sum(p*(np.log(p+1e-12)-np.log(qprob+1e-12)), axis=-1).mean())
            nmse = float(np.sum((qq-fp)**2)/np.sum(fp**2))
            row = {"conv": ci, "mode": name, "bos_fp": bos_fp, "top1": agree/len(eval_pos), "agree": agree, "n": len(eval_pos), "kl": kl, "logits_nmse": nmse}
            all_rows.append(row)
            print(f"  conv{ci} {name:12s} Top1 {agree}/{len(eval_pos)}={agree/len(eval_pos):.3f} KL={kl:.4f} NMSE={nmse:.4f}")
            del logits_q; gc.collect()
        del logits_fp; gc.collect()
    # 전체 평균
    for name in ["all_INT10", "BOS_FP"]:
        rows = [r for r in all_rows if r["mode"] == name]
        avg_top1 = sum(r["agree"] for r in rows) / sum(r["n"] for r in rows)
        avg_kl = sum(r["kl"] for r in rows) / len(rows)
        avg_nmse = sum(r["logits_nmse"] for r in rows) / len(rows)
        print(f"[SUMMARY] {name}: Top1 {avg_top1:.3f} KL {avg_kl:.4f} NMSE {avg_nmse:.4f} (3대화 평균)")
    with open(out_dir/"e2e_bos_bypass_3conv.json","w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"[✓] e2e_bos_bypass_3conv.json 저장")

if __name__=="__main__":
    main()
