#!/usr/bin/env python3
"""64 conv 결과 합치기: results/final_64conv/conv_*.json -> summary.json"""
import argparse, json, glob, os
ap = argparse.ArgumentParser()
ap.add_argument("--input_dir", default="diagnostics/ex_norm10_p6_bos_int10/results/final_64conv")
ap.add_argument("--out", default=None)
args = ap.parse_args()
files = sorted(glob.glob(os.path.join(args.input_dir, "conv_*.json")))
tot_match=tot_n=0
kls=[]; nmses=[]
per=[]
for f in files:
    d=json.load(open(f))
    # result.json 포맷이 다르면 키 이름 맞게 조정: n_match/total, top1, kl, nmse
    m = d.get("n_match", d.get("top1_match", 0))
    n = d.get("total", d.get("n_total", d.get("n", 0)))
    kl = d.get("kl", d.get("mean_kl", 0))
    nmse = d.get("nmse", d.get("mean_nmse", 0))
    tot_match+=m; tot_n+=n; kls.append(kl); nmses.append(nmse)
    per.append({"conv": os.path.basename(f), "top1": m/n if n else 0, "kl": kl, "nmse": nmse})
summary={"n_conv": len(files), "total_match": tot_match, "total_n": tot_n,
 "top1": tot_match/tot_n if tot_n else 0,
 "mean_kl": sum(kls)/len(kls) if kls else 0,
 "mean_nmse": sum(nmses)/len(nmses) if nmses else 0,
 "per_conv": per}
out = args.out or os.path.join(args.input_dir, "summary.json")
json.dump(summary, open(out,"w"), indent=2)
print(f"{len(files)} conv, Top1 {tot_match}/{tot_n}={summary['top1']*100:.1f}% KL {summary['mean_kl']:.4f} NMSE {summary['mean_nmse']:.4f} -> {out}")
