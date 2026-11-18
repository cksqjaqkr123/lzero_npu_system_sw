#!/usr/bin/env python3
"""LUT 메모리 배치 CSV 생성 - 인수자용 확정본
서버에서: python3 generate_lut_csv.py
입력: lut_delivery/norm/*.json, gelu/*.json, rope/*.json, npu/*.json
출력: lut_delivery/csv/*.csv (주소까지 확정)
"""
import json, csv, pathlib
base = pathlib.Path("diagnostics/ex_norm10_p6_bos_int10/lut_delivery")
out_dir = base / "csv"
out_dir.mkdir(parents=True, exist_ok=True)

def align(addr, a):
    return (addr + a - 1) // a * a

# --- Norm 스케일 메모리 맵 ---
norm_gen = json.load(open(base / "norm/norm_selected_parameters.json"))
norm_bos = json.load(open(base / "norm/calib_bos_absmax_1conv.json"))
# 순서: blk.0.attn, blk.0.ffn, ..., blk.25.ffn, output_norm
def norm_order(d):
    keys = [k for k in d.keys() if k.startswith("blk.")]
    # blk.0.attn_norm, blk.0.ffn_norm 순서로 정렬
    def keyfn(k):
        # blk.12.attn_norm -> (12, 0), blk.12.ffn_norm -> (12, 1)
        parts = k.split(".")
        layer = int(parts[1])
        typ = 0 if "attn" in k else 1
        return (layer, typ)
    keys = sorted(keys, key=keyfn)
    others = [k for k in d.keys() if not k.startswith("blk.")]
    return keys + others

addr = 0
rows = []
for name in norm_order(norm_gen):
    g = norm_gen[name]
    b = norm_bos.get(name, g)
    # 일반 s_in, s_out, BOS s_in, s_out을 4개씩 묶어서 16B 블록으로
    base_addr = align(addr, 4)
    rows.append({
        "norm": name, "kind": "general", "s_in": g["s_in"], "s_out": g["s_out"],
        "s_in_addr": base_addr, "s_out_addr": base_addr+4, "block_addr": base_addr, "block_size": 16
    })
    rows.append({
        "norm": name, "kind": "BOS", "s_in": b["s_in"], "s_out": b["s_out"],
        "s_in_addr": base_addr+8, "s_out_addr": base_addr+12, "block_addr": base_addr, "block_size": 16
    })
    addr = base_addr + 16

with open(out_dir / "norm_scales_memory.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["norm","kind","s_in","s_out","s_in_addr","s_out_addr","block_addr","block_size"])
    w.writeheader(); w.writerows(rows)
print(f"norm_scales_memory.csv: {len(rows)} rows, total {addr} B, next_addr={align(addr,4)}")

# --- GELU LUT 메모리 맵 ---
# gelu_decision.json 또는 npu_lut_tables.json에서 LUT를 읽음. 없으면 gelu_local.csv에서 s_in/s_out만으로 구조만 만듦
gelu_rows = []
try:
    gelu = json.load(open(base / "gelu/gelu_decision.json"))
    # gelu_decision 구조가 다양하므로, 키에 layer가 있으면 그걸 씀
    print("gelu_decision keys:", list(gelu.keys())[:5])
except Exception as e:
    print("gelu_decision 로드 실패", e)
    gelu = {}

# LUT 실제 값은 npu_lut_tables.json에 있을 수 있음. 여기서는 주소 맵만 확정 (값은 별도 json 참조)
addr = align(addr, 2048)  # LUT는 2KB 정렬
summary = []
for layer in range(26):
    lut_base = addr
    # 1023 entries + 1 패딩 = 1024 x 2B = 2048 B
    summary.append({"lut_name": f"gelu_lut_layer_{layer}", "base_addr": lut_base, "entries": 1023, "padded_entries": 1024, "entry_size": 2, "total_bytes": 2048, "s_in_ref": f"gelu_decision layer {layer} s_in", "s_out_ref": f"gelu_decision layer {layer} s_out"})
    # 상세 CSV는 1023줄이라서 따로 파일로 (layer별)
    # 여기서는 대표로 layer 0만 상세 생성, 나머지는 주소만 summary에
    addr += 2048
    if layer == 0:
        # 상세: input_code -511..511, output_code는 LUT json에서 가져오거나, 없으면 placeholder (실제 값은 npu_lut_tables.json 참조)
        try:
            npu = json.load(open(base / "npu/npu_lut_tables.json"))
            # npu 구조가 다르면 fallback
            lut_vals = None
            if isinstance(npu, dict):
                # 흔한 키 탐색
                for k in ["gelu", "gelu_lut", "gelu_luts", "lut"]:
                    if k in npu:
                        lut_vals = npu[k]
                        break
            print("npu_lut_tables keys:", list(npu.keys())[:10] if isinstance(npu, dict) else type(npu))
        except Exception as e:
            print("npu lut 로드 실패", e)
            lut_vals = None
        # 상세 CSV는 구조만 (값은 npu_lut_tables.json에서 복사해야 함을 명시)
        with open(out_dir / "gelu_lut_layer0_detail.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["layer","index","input_code","output_code","addr"])
            w.writeheader()
            for idx in range(1023):
                code = idx - 511
                w.writerow({"layer": 0, "index": idx, "input_code": code, "output_code": "SEE npu_lut_tables.json", "addr": lut_base + idx*2})
        print("gelu_lut_layer0_detail.csv written (output_code는 npu_lut_tables.json에서 채워야 함)")

with open(out_dir / "gelu_lut_memory_map.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["lut_name","base_addr","entries","padded_entries","entry_size","total_bytes","s_in_ref","s_out_ref"])
    w.writeheader(); w.writerows(summary)
print(f"gelu_lut_memory_map.csv: 26 LUTs, total {26*2048} B, next_addr={addr}")

# --- RoPE 메모리 맵 ---
try:
    rope = json.load(open(base / "rope/rope_calib.json"))
    print("rope keys:", list(rope.keys())[:5] if isinstance(rope, dict) else type(rope))
except Exception as e:
    print("rope 로드 실패", e)
    rope = {}
addr = align(addr, 4)
rope_rows = []
for layer in range(26):
    # rope_calib 구조가 layer별 dict일 수도, flat일 수도 있음. 여기서는 주소만 확정
    base_addr = addr
    rope_rows.append({"layer": layer, "s_q_addr": base_addr, "s_k_addr": base_addr+4, "block_size": 8, "s_q_ref": f"rope_calib layer {layer} s_q", "s_k_ref": f"rope_calib layer {layer} s_k"})
    addr += 8
with open(out_dir / "rope_scales_memory.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["layer","s_q_addr","s_k_addr","block_size","s_q_ref","s_k_ref"])
    w.writeheader(); w.writerows(rope_rows)
print(f"rope_scales_memory.csv: 26 rows, total {26*8} B")

# --- 전체 요약 ---
with open(out_dir / "lut_memory_map_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["section","base_addr","total_bytes","entries","note"])
    w.writeheader()
    w.writerow({"section": "norm_scales (53 norms x 4 scales x 4B)", "base_addr": 0, "total_bytes": 53*16, "entries": 53*4, "note": "FP32, 4개씩 16B 블록, 일반+BOS 묶음"})
    w.writerow({"section": "gelu_luts (26 x 2048B)", "base_addr": align(53*16, 2048), "total_bytes": 26*2048, "entries": 26*1023, "note": "INT16, 1023+1 패딩=1024, 2KB 정렬"})
    w.writerow({"section": "rope_scales (26 x 8B)", "base_addr": align(53*16, 2048)+26*2048, "total_bytes": 26*8, "entries": 52, "note": "FP32, s_q/s_k"})
print("lut_memory_map_summary.csv written")
print(f"전체 LUT+스케일 크기: {53*16 + 26*2048 + 26*8} B ≈ {(53*16 + 26*2048 + 26*8)/1024:.1f} KB (RoPE cos/sin 별도)")
