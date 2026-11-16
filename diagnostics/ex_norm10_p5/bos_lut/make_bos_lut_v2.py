import json, collections
from pathlib import Path

SRC = Path("diagnostics/ex_norm10_capture_fixed_v2/norm_v2_per_token.jsonl")
OUT = Path("diagnostics/ex_norm10_p5/bos_lut/bos_lut.json")
QMAX = 511

bos_x = collections.defaultdict(list)
bos_y = collections.defaultdict(list)
with open(SRC) as f:
    for line in f:
        o = json.loads(line)
        if o.get("is_bos") != 1:
            continue
        nid = f"{o['layer_name']}_{o['norm_type']}"
        bos_x[nid].append(float(o["x_absmax"]))
        bos_y[nid].append(float(o["y_absmax"]))

lut = {}
for nid in sorted(bos_x.keys()):
    # BOS는 전부 동일값이라서 [0]이 곧 고정값
    x_abs = bos_x[nid][0]
    y_abs = bos_y[nid][0]
    lut[nid] = {
        "bos_x_absmax": x_abs,
        "bos_s_in": x_abs / QMAX,
        "bos_y_absmax": y_abs,
        "bos_s_out": y_abs / QMAX,
    }

Path(OUT).write_text(json.dumps(lut, indent=2))
print(f"saved {OUT}: {len(lut)} entries")
for k in sorted(lut)[:2]:
    print(k, lut[k])
