import json, csv, collections, statistics
from pathlib import Path

SRC = Path("diagnostics/ex_norm10_capture_fixed_v2/norm_v2_per_token.jsonl")
OUT = Path("diagnostics/ex_norm10_p5/bos_lut/bos_range_per_norm.csv")

bos_vals = collections.defaultdict(list)
gen_vals = collections.defaultdict(list)
# conv별 BOS 값도 저장 (나중에 어떤 conv가 튀는지 보려고)
bos_by_conv = collections.defaultdict(dict)

n = 0
with open(SRC) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # group이 prefill만 있는 경우도 있고, decode도 있으면 BOS는 prefill pos0에만 있음
        nid = f"{obj['layer_name']}_{obj['norm_type']}"
        v = float(obj["x_absmax"])
        if obj.get("is_bos") == 1:
            bos_vals[nid].append(v)
            bos_by_conv[nid][obj["conv"]] = v
        else:
            gen_vals[nid].append(v)
        n += 1
        if n % 200000 == 0:
            print(f"... {n} lines")

print(f"total lines: {n}")
print(f"Norms: {len(bos_vals)}")

rows = []
for nid in sorted(bos_vals.keys()):
    vals = sorted(bos_vals[nid])
    gvals = sorted(gen_vals.get(nid, []))
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    mn, mx = vals[0], vals[-1]
    ratio = mx / mn if mn > 0 else float("inf")
    gen_mean = statistics.mean(gvals) if gvals else 0
    def pct(data, p):
        if not data:
            return 0
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data)-1)]
    rows.append({
        "norm_id": nid,
        "n_bos": len(vals),
        "bos_mean": mean,
        "bos_std": std,
        "bos_min": mn,
        "bos_max": mx,
        "max_min_ratio": ratio,
        "bos_p1": pct(vals, 1),
        "bos_p99": pct(vals, 99),
        "gen_mean": gen_mean,
        "bos_over_gen": mean / gen_mean if gen_mean else 0,
        "cv": std / mean if mean else 0,
        "bos_by_conv": ",".join(f"{c}:{bos_by_conv[nid].get(c,0):.3f}" for c in sorted(bos_by_conv[nid])),
    })

rows.sort(key=lambda r: r["max_min_ratio"], reverse=True)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"saved -> {OUT}")
print("\n--- max/min 큰 상위 10 (고정 어려운 Norm) ---")
for r in rows[:10]:
    print(f"{r['norm_id']}: max/min={r['max_min_ratio']:.2f} cv={r['cv']:.3f} bos/gen={r['bos_over_gen']:.2f} vals={r['bos_by_conv']}")
print("\n--- max/min 작은 하위 10 (고정 쉬운 Norm) ---")
for r in rows[-10:]:
    print(f"{r['norm_id']}: max/min={r['max_min_ratio']:.2f} cv={r['cv']:.3f} bos/gen={r['bos_over_gen']:.2f}")

ratios = [r["max_min_ratio"] for r in rows]
print(f"\n전체 {len(rows)}개: max/min 평균 {statistics.mean(ratios):.2f}, 중앙값 {statistics.median(ratios):.2f}, 최대 {max(ratios):.2f}")
print(f"<1.5: {sum(1 for x in ratios if x<1.5)}개, 1.5~3: {sum(1 for x in ratios if 1.5<=x<3)}개, >3: {sum(1 for x in ratios if x>=3)}개")