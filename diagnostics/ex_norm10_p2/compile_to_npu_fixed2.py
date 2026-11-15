#!/usr/bin/env python3
"""
P2 FIXED v2 — sout_lut.json(19-cluster) 절대 안 씀
- 지난번 실수: sout_lut.json을 후보에 넣어서 19-cluster 파일을 잡아버림
- per-Norm 53개로 확정한 거라 sout_lut.json은 쓰면 안 됨
- sout_lut_per_norm.json만 찾고, 없으면 norm_selected_parameters.json(53개 scale)으로 폴백
- sout_lut_per_norm.json 구조가 {name: float} / {name: [lut]}여도 안 터지게 방어
"""
import json, csv, glob
from pathlib import Path
import numpy as np

def log(*a, **k):
    print(*a, flush=True, **k)

def build_gelu_lut(s_in, s_out):
    idx = np.arange(-511, 512, dtype=np.float64)
    deq = idx * float(s_in)
    gelu = 0.5 * deq * (1 + np.tanh(np.sqrt(2/np.pi)*(deq + 0.044715*deq**3)))
    q = np.round(gelu / float(s_out)).clip(-511, 511)
    return q.astype(int).tolist()

def find_per_norm():
    # sout_lut_per_norm.json만 찾음 — sout_lut.json(19-cluster)은 후보에서 제외
    for p in [
        "diagnostics/ex_norm10_capture_fixed_v2/sout_lut_per_norm.json",
        "diagnostics/ex_norm10_p4/lut/sout_lut_per_norm.json",
        "diagnostics/ex_norm10_capture_fixed_v2/lut/sout_lut_per_norm.json",
    ]:
        if Path(p).exists():
            return Path(p)
    for m in glob.glob("diagnostics/**/sout_lut_per_norm.json", recursive=True):
        return Path(m)
    return None

def main():
    base = Path("diagnostics/ex_norm10_p2")
    lut_dir = base / "lut"; lut_dir.mkdir(parents=True, exist_ok=True)
    comp_dir = base / "compiler"; comp_dir.mkdir(parents=True, exist_ok=True)
    rms_entries = []
    sout_path = find_per_norm()
    if sout_path:
        log(f"[RMSNorm] per-Norm 파일 찾음: {sout_path}")
        sout = json.load(open(sout_path))
        # 구조 방어: {name: {s_in,s_out,lut}} / {"norms": [...]} / {name: [lut]} / {name: float}
        items = []
        if isinstance(sout, dict) and "norms" in sout:
            items = sout["norms"]
        elif isinstance(sout, dict):
            for k, v in sout.items():
                if isinstance(v, dict) and ("s_in" in v or "s_out" in v or "lut" in v):
                    items.append({"name": k, **v})
                elif isinstance(v, list):
                    # {name: [lut]} — s는 norm_selected에서 나중에 채움
                    items.append({"name": k, "lut": v})
                else:
                    log(f"[SKIP] {k}: dict/list가 아님 (type={type(v).__name__}) — sout_lut.json(19-cluster) 형태일 수 있음, 건너뜀")
        elif isinstance(sout, list):
            items = sout
        else:
            log(f"[WARN] sout 파일 구조가 예상과 다름: {type(sout)}")
        for it in items:
            if not isinstance(it, dict):
                continue
            name = it.get("name") or it.get("norm")
            if not name:
                continue
            rms_entries.append({
                "name": name, "type": "rmsnorm",
                "s_in": float(it["s_in"]) if "s_in" in it else None,
                "s_out": float(it["s_out"]) if "s_out" in it else None,
                "lut": it.get("lut") or it.get("table"),
                "is_bos_bypass": True,
            })
        log(f"[RMSNorm] {len(rms_entries)}개 파싱됨")
        # s가 None인 게 있으면 norm_selected에서 채움
        if any(e["s_in"] is None or e["s_out"] is None for e in rms_entries):
            log(f"[RMSNorm] s가 비어있는 항목 있음 — norm_selected_parameters.json에서 채움")
            sel = None
            for p in ["diagnostics/ex_norm10_capture_fixed_v2/norm_selected_parameters.json"]:
                if Path(p).exists():
                    sel = json.load(open(p)); break
            if sel and isinstance(sel, dict) and "norms" in sel:
                sel_map = {n.get("name"): n for n in sel["norms"]}
                for e in rms_entries:
                    if e["s_in"] is None and e["name"] in sel_map:
                        e["s_in"] = float(sel_map[e["name"]]["s_in"])
                        e["s_out"] = float(sel_map[e["name"]]["s_out"])
    else:
        log(f"[RMSNorm] sout_lut_per_norm.json 없음 — norm_selected_parameters.json으로 폴백 (53개 scale)")
        sel_path = Path("diagnostics/ex_norm10_capture_fixed_v2/norm_selected_parameters.json")
        if not sel_path.exists():
            for m in glob.glob("diagnostics/**/norm_selected_parameters.json", recursive=True):
                sel_path = Path(m); break
        log(f"[RMSNorm] 폴백 파일: {sel_path}")
        sel = json.load(open(sel_path))
        if isinstance(sel, dict) and "norms" in sel:
            items = sel["norms"]
        elif isinstance(sel, dict):
            items = [{"name": k, **v} for k, v in sel.items() if isinstance(v, dict)]
        else:
            items = sel
        for n in items:
            name = n.get("name") or n.get("norm")
            rms_entries.append({"name": name, "type": "rmsnorm", "s_in": float(n["s_in"]), "s_out": float(n["s_out"]), "lut": None, "is_bos_bypass": True})
        log(f"[RMSNorm] 폴백 {len(rms_entries)}개 (lut=None, scale만)")
    # output_norm
    names = {e["name"] for e in rms_entries}
    if "output_norm" not in names:
        for e in list(rms_entries):
            if e["name"] and ("output" in e["name"].lower() or "final" in e["name"].lower()):
                rms_entries.append({"name": "output_norm", "type": "rmsnorm", "s_in": e["s_in"], "s_out": e["s_out"], "lut": e["lut"], "is_bos_bypass": True})
                log(f"[RMSNorm] output_norm 매핑: {e['name']} -> output_norm")
                break
    # GELU
    gelu_entries = []
    sin_vals, sout_vals = {}, {}
    gelu_csv = Path("diagnostics/ex_norm10_p4/gelu/gelu_local.csv")
    if not gelu_csv.exists():
        for m in glob.glob("diagnostics/**/gelu_local.csv", recursive=True):
            gelu_csv = Path(m); break
    log(f"[GELU] csv: {gelu_csv}")
    with open(gelu_csv) as f:
        for row in csv.DictReader(f):
            layer, tag = row.get("layer",""), row.get("tag","")
            if tag.startswith("gelu_nmse_sin_"):
                cand = tag.replace("gelu_nmse_sin_", "")
                sin_vals.setdefault(layer, {})[cand] = {"nmse": float(row["nmse"]), "s": float(row["s_in"]) if row.get("s_in") else None}
            elif tag.startswith("gelu_nmse_sout_"):
                cand = tag.replace("gelu_nmse_sout_", "")
                sout_vals.setdefault(layer, {})[cand] = {"nmse": float(row["nmse"]), "s": float(row["s_out"]) if row.get("s_out") else None}
    for layer in sorted(sin_vals.keys()):
        d = sin_vals[layer]
        best = min(d, key=lambda k: d[k]["nmse"])
        if best == "p999": best = "p9999"
        s_in = d[best]["s"]
        s_out = sout_vals[layer]["absmax"]["s"]
        lut = build_gelu_lut(s_in, s_out)
        gelu_entries.append({"name": f"{layer}.gelu", "type": "gelu", "s_in": float(s_in), "s_out": float(s_out), "s_in_mode": best, "lut": lut, "is_bos_bypass": False})
        log(f"[GELU] {layer} s_in={s_in:.6f} ({best}) s_out={s_out:.6f}")
    # RoPE
    rope_entries = []
    rope_path = Path("diagnostics/ex_norm10_p4/e2e/rope_calib.json")
    if not rope_path.exists():
        for m in glob.glob("diagnostics/**/rope_calib.json", recursive=True):
            rope_path = Path(m); break
    if rope_path.exists():
        rope = json.load(open(rope_path))
        for k in sorted(rope.keys()):
            v = rope[k]
            rope_entries.append({"name": f"{k}.rope", "type": "rope", "s_q": float(v["s_q"]), "s_k": float(v["s_k"]), "s_out_equals_s_in": True, "is_bos_bypass": False})
        log(f"[RoPE] {len(rope_entries)} from {rope_path}")
    out = {"rmsnorm": rms_entries, "gelu": gelu_entries, "rope": rope_entries,
           "policy": {"rmsnorm": "s_in=absmax, s_out=p9999, is_bos=True", "gelu": "s_in=per-layer best, s_out=absmax, is_bos=False", "rope": "s_q/s_k=p9999, s_out=s_in, is_bos=False", "e2e": "Combined Strict Top1 87.0% KL 0.1066 (1606 positions, static scale, LUT only)"}}
    with open(lut_dir/"npu_lut_tables.json","w") as f:
        json.dump(out, f, indent=2)
    log(f"[✓] {lut_dir/'npu_lut_tables.json'} 저장: RMSNorm {len(rms_entries)} + GELU {len(gelu_entries)} + RoPE {len(rope_entries)}")
    h_lines = ["// Auto-generated — DO NOT EDIT", "#pragma once", "#include <stdint.h>", "", "// if (is_bos) { out = fp_rmsnorm(x); } else { out = lut[quant(x, s_in)] * s_out; }", ""]
    for e in gelu_entries:
        name = e["name"].replace(".", "_").replace("-", "_")
        h_lines.append(f"// GELU {e['name']} s_in={e['s_in']:.6f} s_out={e['s_out']:.6f}")
        h_lines.append(f"static const int16_t GELU_LUT_{name}[1023] = {{ {', '.join(map(str, e['lut'][:10]))} /* ... */ }};")
    with open(comp_dir/"npu_lut.h","w") as f: f.write("\n".join(h_lines))
    log(f"[✓] {comp_dir/'npu_lut.h'} 저장")
    snippet = '''float rmsnorm_quant(float x, float s_in, float s_out, bool is_bos, int16_t* lut) {
    if (is_bos) return rmsnorm_fp(x);
    int code = (int)roundf(x / s_in);
    code = code < -511 ? -511 : (code > 511 ? 511 : code);
    return (float)lut[code + 511] * s_out;
}
float gelu_quant(float gate, float s_in, float s_out, int16_t* lut) {
    int code = (int)roundf(gate / s_in);
    code = code < -511 ? -511 : (code > 511 ? 511 : code);
    return (float)lut[code + 511] * s_out;
}
'''
    with open(comp_dir/"is_bos_snippet.c","w") as f: f.write(snippet)
    log(f"[✓] {comp_dir/'is_bos_snippet.c'} 저장")

if __name__=="__main__":
    main()
