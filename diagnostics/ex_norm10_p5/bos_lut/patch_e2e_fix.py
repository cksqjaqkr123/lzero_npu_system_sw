from pathlib import Path
src = Path("diagnostics/ex_norm10_p4/e2e/run_e2e_combined_strict_fixed.py")
dst = Path("diagnostics/ex_norm10_p5/bos_lut/run_e2e_bos_lut.py")
t = src.read_text()

# 1. 함수 시그니처
t = t.replace(
    "def forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales):",
    "def forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales, bos_lut=None):",
    1
)

# 2. attn
t = t.replace(
    '        x_q = quant_int10(x, sc["s_in"])\n        x_q[:, 0, :] = x[:, 0, :]',
    '        x_q = quant_int10(x, sc["s_in"])\n        x_q[:, 0, :] = quant_int10(x[:, 0, :], bos_lut[f"{p}.attn_norm"]["bos_s_in"]) if bos_lut else x[:, 0, :]',
    1
)
t = t.replace(
    '        normed_q = quant_int10(normed, sc["s_out"])\n        normed_q[:, 0, :] = normed[:, 0, :]',
    '        normed_q = quant_int10(normed, sc["s_out"])\n        normed_q[:, 0, :] = quant_int10(normed[:, 0, :], bos_lut[f"{p}.attn_norm"]["bos_s_out"]) if bos_lut else normed[:, 0, :]',
    1
)

# 3. ffn — 원본 변수명 확인: x_q2 / normed2_q 라고 가정, 없으면 아래에서 확인
# 파일에 실제로 있는지 먼저 체크
if 'x_q2 = quant_int10' in t:
    t = t.replace(
        '        x_q2 = quant_int10(x, sc2["s_in"])\n        x_q2[:, 0, :] = x[:, 0, :]',
        '        x_q2 = quant_int10(x, sc2["s_in"])\n        x_q2[:, 0, :] = quant_int10(x[:, 0, :], bos_lut[f"{p}.ffn_norm"]["bos_s_in"]) if bos_lut else x[:, 0, :]',
        1
    )
    t = t.replace(
        '        normed2_q = quant_int10(normed2, sc2["s_out"])\n        normed2_q[:, 0, :] = normed2[:, 0, :]',
        '        normed2_q = quant_int10(normed2, sc2["s_out"])\n        normed2_q[:, 0, :] = quant_int10(normed2[:, 0, :], bos_lut[f"{p}.ffn_norm"]["bos_s_out"]) if bos_lut else normed2[:, 0, :]',
        1
    )
    print("ffn patched")
else:
    print("WARN: ffn 패턴 없음 — 파일에서 ffn 부분 grep 필요")

# 4. final — 변수명도 파일 확인
if 'x_qf = quant_int10' in t:
    t = t.replace(
        '    x_qf = quant_int10(x, scf["s_in"])\n    x_qf[:, 0, :] = x[:, 0, :]',
        '    x_qf = quant_int10(x, scf["s_in"])\n    x_qf[:, 0, :] = quant_int10(x[:, 0, :], bos_lut["output_norm_final_norm"]["bos_s_in"]) if bos_lut else x[:, 0, :]',
        1
    )
    t = t.replace(
        '    x_final_q = quant_int10(x_final, scf["s_out"])\n    x_final_q[:, 0, :] = x_final[:, 0, :]',
        '    x_final_q = quant_int10(x_final, scf["s_out"])\n    x_final_q[:, 0, :] = quant_int10(x_final[:, 0, :], bos_lut["output_norm_final_norm"]["bos_s_out"]) if bos_lut else x_final[:, 0, :]',
        1
    )
    print("final patched")
else:
    print("WARN: final 패턴 없음")

# 5. bos_lut 로드 — try/except 끝난 뒤, # GELU 바로 앞에 딱 1번만
old = "    # GELU\n    sin_vals, sout_vals = {}, {}"
new = "    bos_lut = json.loads(Path(\"diagnostics/ex_norm10_p5/bos_lut/bos_lut.json\").read_text())\n    # GELU\n    sin_vals, sout_vals = {}, {}"
assert old in t, "GELU 마커 없음"
t = t.replace(old, new, 1)

# 6. forward 호출에 bos_lut 전달 — 1번만
old_call = "forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales)"
if old_call in t:
    t = t.replace(old_call, "forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales, bos_lut=bos_lut)", 1)
    print("call patched")
else:
    print("WARN: forward 호출 패턴 없음 — grep 필요")
    # 다른 형태일 수 있으니 찾기
    import re
    for m in re.finditer(r"forward_combined_strict.*", t):
        print(m.group(0))

dst.write_text(t)
print(f"patched -> {dst}")
