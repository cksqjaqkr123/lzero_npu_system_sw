import csv, json
from pathlib import Path

SRC = Path("diagnostics/ex_norm10_p5/bos_lut/bos_range_per_norm.csv")
OUT = Path("diagnostics/ex_norm10_p5/bos_lut/bos_lut.json")

# INT10: 2^9 - 1 = 511
QMAX = 511

lut = {}
with open(SRC) as f:
    for row in csv.DictReader(f):
        nid = row["norm_id"]
        bos_absmax = float(row["bos_max"])  # min==max라서 max가 곧 고정값
        lut[nid] = {
            "bos_x_absmax": bos_absmax,
            "bos_s_in": bos_absmax / QMAX,
            "bos_over_gen": float(row["bos_over_gen"]),
            "note": "BOS is deterministic, pos0 causal, fixed across all convs"
        }

with open(OUT, "w") as f:
    json.dump(lut, f, indent=2)

print(f"saved {OUT} : {len(lut)} entries")
for k in sorted(lut)[:5]:
    print(k, lut[k])
print("...")
for k in sorted(lut)[-3:]:
    print(k, lut[k])
