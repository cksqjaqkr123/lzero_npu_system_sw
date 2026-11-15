#!/usr/bin/env python3
"""
P0-B 수정 v2: s_in은 absmax, s_out은 p9999
- s_in = x_absmax / 127  (blk.0 attn: 2.03/127=0.016, sweep에서 좋았던 값)
- s_out = y_p9999 / 511 (output_only에서 KL 0.09로 좋았던 값)
"""
import argparse, csv, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="diagnostics/ex_norm10_capture_fixed_v2/scale_candidates_v2.csv")
    ap.add_argument("--old_params", default="diagnostics/ex_norm10_capture_fixed_v2/norm_selected_parameters.json")
    ap.add_argument("--out", default="/tmp/p9999_absmax.json")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    print(f"[*] csv rows={len(rows)}")
    out = {}
    for r in rows:
        layer = r["layer_name"]
        ntype = r["norm_type"]
        name = f"{layer}.{ntype}"
        if "output" in layer.lower() and "final" not in name.lower():
            name = "output_norm.final_norm"
        try:
            x_absmax = float(r["x_absmax"])
            y_p9999 = float(r["y_p9999"])
        except Exception as e:
            print(f"[!] skip {name}: {e}")
            continue
        s_in = x_absmax / 127.0
        s_out = y_p9999 / 511.0
        out[name] = {
            "s_in": s_in,
            "s_in_policy": "absmax",
            "s_out": s_out,
            "s_out_policy": "p9999",
            "x_absmax": x_absmax,
            "y_p9999": y_p9999,
            "gamma_policy": "weight-as-gamma, F32",
        }
        print(f"  {name:25s} x_absmax={x_absmax:.4f} -> s_in={s_in:.6f}  y_p9999={y_p9999:.4f} -> s_out={s_out:.6f}")
    try:
        old = json.load(open(args.old_params))
        for k, v in old.items():
            if k in out and isinstance(v, dict):
                for kk in ["local_nmse_p9999", "local_nmse", "gamma_mean"]:
                    if kk in v:
                        out[k][kk] = v[kk]
    except Exception as e:
        print(f"[*] old params 복사 스킵: {e}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[✓] {args.out} 저장, {len(out)} Norms")

if __name__=="__main__":
    main()
