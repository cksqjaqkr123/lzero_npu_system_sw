#!/usr/bin/env python3
"""
GELU 의사결정 — RMSNorm 방식 그대로
- 입력: gelu_local.csv (run_gelu_local.py 결과)
- RMSNorm 때랑 똑같이: per-layer NMSE 비교 → p999 / p9999 / absmax 중 best 선택 → 분포 집계
- 출력: gelu_decision.json + 콘솔 요약
- RMSNorm 때: s_out은 p999 0개, p9999 53개. GELU도 같은 방식으로 보면 됨.
"""
import argparse, csv, json
from collections import defaultdict, Counter
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="diagnostics/ex_norm10_p4/gelu/gelu_local.csv")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_p4/gelu")
    args = ap.parse_args()
    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    # s_in: gate 양자화 → GELU 출력 NMSE
    sin_by_layer = defaultdict(dict)
    sout_by_layer = defaultdict(dict)
    for r in rows:
        tag = r.get("tag", "")
        layer = r.get("layer", "")
        try:
            nmse = float(r["nmse"]) if r.get("nmse") else None
        except:
            nmse = None
        if nmse is None:
            continue
        if tag.startswith("gelu_nmse_sin_"):
            cand = tag.replace("gelu_nmse_sin_", "")
            sin_by_layer[layer][cand] = nmse
        elif tag.startswith("gelu_nmse_sout_"):
            cand = tag.replace("gelu_nmse_sout_", "")
            sout_by_layer[layer][cand] = nmse
    print(f"[*] GELU Local rows={len(rows)} layers={len(sin_by_layer)}")
    print(f"\n=== s_in (gate 양자화 → GELU 출력) ===")
    cnt_in = Counter()
    for layer in sorted(sin_by_layer.keys()):
        d = sin_by_layer[layer]
        best = min(d, key=lambda k: d[k])
        cnt_in[best] += 1
        print(f"  {layer}: " + " / ".join([f"{k}={d[k]:.5f}" for k in sorted(d.keys())]) + f"  → best={best}")
    print(f"  [분포] {dict(cnt_in)}")
    print(f"\n=== s_out (GELU 출력 양자화) ===")
    cnt_out = Counter()
    for layer in sorted(sout_by_layer.keys()):
        d = sout_by_layer[layer]
        best = min(d, key=lambda k: d[k])
        cnt_out[best] += 1
        print(f"  {layer}: " + " / ".join([f"{k}={d[k]:.5f}" for k in sorted(d.keys())]) + f"  → best={best}")
    print(f"  [분포] {dict(cnt_out)}")
    # RMSNorm이랑 같은 결론 형식으로
    print(f"\n=== 결론 (RMSNorm 방식) ===")
    print(f"  s_in best 분포: {dict(cnt_in)}")
    print(f"  s_out best 분포: {dict(cnt_out)}")
    if cnt_in.get("p999", 0) == 0 and cnt_out.get("p999", 0) == 0:
        print(f"  → p999 0개, RMSNorm이랑 똑같이 p999 버리고 p9999/absmax로 간다")
    else:
        print(f"  → p999가 이긴 레이어가 있음, Hybrid 검토 필요")
    out = {
        "s_in_best": dict(cnt_in),
        "s_out_best": dict(cnt_out),
        "per_layer_sin": {k: dict(v) for k, v in sin_by_layer.items()},
        "per_layer_sout": {k: dict(v) for k, v in sout_by_layer.items()},
    }
    with open(Path(args.output_dir)/"gelu_decision.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[✓] gelu_decision.json 저장")

if __name__=="__main__":
    main()
