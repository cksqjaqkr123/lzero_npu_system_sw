import csv, json
from pathlib import Path
in_dir = Path("diagnostics/ex_norm10_capture_fixed_v2")

# oracle에서 best s_in
oracle = list(csv.DictReader(open(in_dir/"oracle_v2.csv")))
best_sin = {}
for r in oracle:
    k = (r["layer_name"], r["norm_type"])
    if k not in best_sin or float(r["nmse"]) < float(best_sin[k]["nmse"]):
        best_sin[k] = r

# sout sweep에서 s_out p9999
sout = list(csv.DictReader(open(in_dir/"sout_sweep_all.csv")))
sout_p9999 = {(r["layer_name"], r["norm_type"]): r for r in sout if r["s_out_candidate"]=="p9999"}

out = {}
for k, r in best_sin.items():
    s = sout_p9999[k]
    out[f"{k[0]}.{k[1]}"] = {
        "s_in": float(r["s_in"]),
        "s_in_policy": r["s_in_candidate"],
        "s_out": float(s["s_out"]),
        "s_out_policy": "p9999",
        "local_nmse_p9999": float(s["nmse"]),
        "gamma_policy": "weight-as-gamma, F32",
    }
with open(in_dir/"norm_selected_parameters.json","w") as f:
    json.dump(out, f, indent=2)
print(f"[✓] norm_selected_parameters.json 저장 ({len(out)} Norm)")
for k in sorted(out)[:5]:
    print(k, out[k])
