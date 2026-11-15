#!/usr/bin/env python3
"""
P0-B 수정: s_in을 전부 p9999로 재생성
- 입력: scale_candidates_v2.csv (x_p9999, y_p9999 포함)
- 출력: norm_selected_parameters_p9999.json
  s_in = x_p9999 / 127, s_out = y_p9999 / 511, 둘 다 p9999 정책
"""
import argparse, csv, json
from pathlib import Path

def find_col(cols, keywords):
    for c in cols:
        lc = c.lower()
        if all(k in lc for k in keywords):
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="diagnostics/ex_norm10_capture_fixed_v2/scale_candidates_v2.csv")
    ap.add_argument("--old_params", default="diagnostics/ex_norm10_capture_fixed_v2/norm_selected_parameters.json")
    ap.add_argument("--out", default="diagnostics/ex_norm10_capture_fixed_v2/norm_selected_parameters_p9999.json")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    print(f"[*] csv rows={len(rows)} cols={list(rows[0].keys()) if rows else []}")
    # 컬럼 찾기: x_p9999, y_p9999
    cols = list(rows[0].keys())
    x_col = find_col(cols, ["x", "p9999"]) or find_col(cols, ["p9999", "in"]) or find_col(cols, ["x_p9999"])
    y_col = find_col(cols, ["y", "p9999"]) or find_col(cols, ["p9999", "out"]) or find_col(cols, ["y_p9999"])
    name_col = find_col(cols, ["layer"]) or find_col(cols, ["norm"]) or "layer_name"
    print(f"[*] x_p9999 col={x_col}, y_p9999 col={y_col}, name col={name_col}")
    if not x_col or not y_col:
        print("[!] p9999 컬럼을 못 찾았어. csv 헤더를 확인해줘.")
        return
    out = {}
    for r in rows:
        name = r[name_col]
        # name이 blk.0.attn_norm 형태인지, blk.0 + attn_norm 형태인지 확인
        # scale_candidates_v2.csv는 layer_name이 blk.0, norm_type이 attn_norm일 수 있음
        if "norm_type" in r and "layer" not in name:
            # name이 blk.0만 있으면 norm_type 붙이기
            if "." not in name or name.count(".") == 1:
                name = f"{r['layer_name']}.{r['norm_type']}" if "layer_name" in r else name
        # output_norm 처리: blk가 아니라 output_norm.final_norm
        if "output" in name.lower() and "final" not in name.lower():
            name = "output_norm.final_norm"
        try:
            x_p9999 = float(r[x_col])
            y_p9999 = float(r[y_col])
        except:
            print(f"[!] skip {name} (parse fail)")
            continue
        s_in = x_p9999 / 127.0
        s_out = y_p9999 / 511.0
        # 기존 params에서 gamma 등 유지
        out[name] = {
            "s_in": s_in,
            "s_in_policy": "p9999",
            "s_out": s_out,
            "s_out_policy": "p9999",
            "x_p9999": x_p9999,
            "y_p9999": y_p9999,
            "gamma_policy": "weight-as-gamma, F32",
        }
        print(f"  {name:25s} x_p9999={x_p9999:.4f} -> s_in={s_in:.6f}  y_p9999={y_p9999:.4f} -> s_out={s_out:.6f}")
    # 기존 params가 있으면 local_nmse 등 복사 (없으면 스킵)
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
