#!/usr/bin/env python3
"""
P0-B s_out LUT 클러스터링: Layer별로 p999 / p9999 중 최적 고르고, 비슷한 값 묶어서 용량 줄이기
- 입력: sout_sweep_all.csv (159행 = 53 Norm × 3 candidate, 컬럼: layer_name, norm_type, s_out_candidate, s_in_used, s_in, s_out, nmse)
- 출력: sout_lut.json (클러스터별 대표 s_out + 소속 Norm)
"""
import argparse, json
from pathlib import Path
import numpy as np
import csv

def load_sweep(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def cluster_sout(values, threshold=0.05):
    # values: list of (norm_name, s_out, choice)
    # threshold: 상대 오차 5% 이내면 같은 클러스터
    vals = sorted(values, key=lambda x: x[1])
    clusters = []
    for name, s, choice in vals:
        placed = False
        for c in clusters:
            rep = c["rep_s_out"]
            if abs(s - rep) / max(rep, 1e-9) <= threshold:
                c["members"].append({"norm": name, "s_out": s, "choice": choice})
                # 대표값은 평균으로 업데이트
                all_s = [m["s_out"] for m in c["members"]]
                c["rep_s_out"] = float(np.mean(all_s))
                c["min_s_out"] = float(min(all_s))
                c["max_s_out"] = float(max(all_s))
                placed = True
                break
        if not placed:
            clusters.append({
                "cluster_id": len(clusters),
                "rep_s_out": float(s),
                "min_s_out": float(s),
                "max_s_out": float(s),
                "members": [{"norm": name, "s_out": s, "choice": choice}]
            })
    return clusters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="diagnostics/ex_norm10_capture_fixed_v2/sout_sweep_all.csv")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    rows = load_sweep(args.sweep)
    print(f"[*] {args.sweep} rows={len(rows)}")
    # 각 Norm별로 p999 vs p9999 중 NMSE 작은 쪽 선택 (absmax는 제외, 회의 결정은 p999/p9999 둘 중 하나)
    # CSV 컬럼 가정: norm, nmse_p999, nmse_p9999, s_out_p999, s_out_p9999
    # 실제 컬럼이 다르면 아래에서 자동 감지
    per_norm = {}
    # 실제 CSV 형식: layer_name, norm_type, s_out_candidate, s_in_used, s_in, s_out, nmse
    # 159행 = 53 Norm × 3 candidate (p999/p9999/absmax). Norm별로 그룹핑해서 p999 vs p9999 비교.
    from collections import defaultdict
    grouped = defaultdict(dict)
    for row in rows:
        layer = row.get("layer_name", "").strip()
        ntype = row.get("norm_type", "").strip()
        cand = row.get("s_out_candidate", "").strip()
        if not layer or not ntype or not cand:
            print(f"[!] 컬럼 확인 필요: {list(row.keys())} row={row}")
            continue
        norm = f"{layer}.{ntype}"
        try:
            grouped[norm][cand] = (float(row["s_out"]), float(row["nmse"]))
        except Exception as e:
            print(f"[!] 파싱 실패 {row} {e}")
            continue
    for norm, d in grouped.items():
        if "p999" not in d or "p9999" not in d:
            print(f"[!] {norm}에 p999/p9999 중 하나 없음: {list(d.keys())}")
            continue
        s_999, nmse_999 = d["p999"]
        s_9999, nmse_9999 = d["p9999"]
        # absmax 참고용 (회의 결정은 p999/p9999만, 비교용으로만 기록)
        abs_info = d.get("absmax", (None, None))
        if nmse_9999 <= nmse_999:
            per_norm[norm] = (s_9999, "p9999", nmse_9999, nmse_999)
        else:
            per_norm[norm] = (s_999, "p999", nmse_999, nmse_9999)
    print(f"[*] per-norm 선택 완료: {len(per_norm)}개")
    from collections import Counter
    cnt = Counter(v[1] for v in per_norm.values())
    print(f"[*] 선택 분포: {dict(cnt)}")
    values = [(k, v[0], v[1]) for k, v in per_norm.items()]
    clusters = cluster_sout(values, threshold=args.threshold)
    print(f"[*] 클러스터링 threshold={args.threshold} → {len(clusters)}개 LUT (원래 {len(values)}개 → {len(values)-len(clusters)}개 절감, {len(clusters)/len(values)*100:.1f}% 용량)")
    for c in clusters:
        print(f"  Cluster {c['cluster_id']}: rep_s_out={c['rep_s_out']:.6f} range=[{c['min_s_out']:.6f},{c['max_s_out']:.6f}] members={len(c['members'])} {', '.join([m['norm'] for m in c['members']][:5])}{'...' if len(c['members'])>5 else ''}")
    out = {
        "threshold": args.threshold,
        "n_norms": len(values),
        "n_clusters": len(clusters),
        "reduction": f"{len(values)} -> {len(clusters)}",
        "per_norm_choice": {k: {"s_out": v[0], "choice": v[1], "nmse_best": v[2], "nmse_other": v[3]} for k, v in per_norm.items()},
        "clusters": clusters
    }
    with open(out_dir/"sout_lut.json","w") as f:
        json.dump(out, f, indent=2)
    print(f"[✓] sout_lut.json 저장")

if __name__=="__main__":
    main()
