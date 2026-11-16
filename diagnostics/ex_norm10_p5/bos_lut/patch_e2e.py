from pathlib import Path
src = Path("diagnostics/ex_norm10_p4/e2e/run_e2e_combined_strict_fixed.py")
dst = Path("diagnostics/ex_norm10_p5/bos_lut/run_e2e_bos_lut.py")
t = src.read_text()

# 1. bos_lut 로드 추가: forward_combined_strict 시그니처에 bos_lut 파라미터 추가
t = t.replace(
    "def forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales):",
    "def forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales, bos_lut=None):"
)

# 2. attn_norm: x_q bypass -> BOS LUT
t = t.replace(
    '        x_q = quant_int10(x, sc["s_in"])\n        x_q[:, 0, :] = x[:, 0, :]',
    '        x_q = quant_int10(x, sc["s_in"])\n        x_q[:, 0, :] = quant_int10(x[:, 0, :], bos_lut[f"{p}.attn_norm"]["bos_s_in"]) if bos_lut else x[:, 0, :]'
)
t = t.replace(
    '        normed_q = quant_int10(normed, sc["s_out"])\n        normed_q[:, 0, :] = normed[:, 0, :]',
    '        normed_q = quant_int10(normed, sc["s_out"])\n        normed_q[:, 0, :] = quant_int10(normed[:, 0, :], bos_lut[f"{p}.attn_norm"]["bos_s_out"]) if bos_lut else normed[:, 0, :]'
)

# 3. ffn_norm: 변수명이 x_q2 / normed2_q 임 — 파일에서 확인하고 똑같이
t = t.replace(
    '        x_q2 = quant_int10(x, sc2["s_in"])\n        x_q2[:, 0, :] = x[:, 0, :]',
    '        x_q2 = quant_int10(x, sc2["s_in"])\n        x_q2[:, 0, :] = quant_int10(x[:, 0, :], bos_lut[f"{p}.ffn_norm"]["bos_s_in"]) if bos_lut else x[:, 0, :]'
)
t = t.replace(
    '        normed2_q = quant_int10(normed2, sc2["s_out"])\n        normed2_q[:, 0, :] = normed2[:, 0, :]',
    '        normed2_q = quant_int10(normed2, sc2["s_out"])\n        normed2_q[:, 0, :] = quant_int10(normed2[:, 0, :], bos_lut[f"{p}.ffn_norm"]["bos_s_out"]) if bos_lut else normed2[:, 0, :]'
)

# 4. final_norm: 변수명이 x_qf / x_final_q 임
t = t.replace(
    '    x_qf = quant_int10(x, scf["s_in"])\n    x_qf[:, 0, :] = x[:, 0, :]',
    '    x_qf = quant_int10(x, scf["s_in"])\n    x_qf[:, 0, :] = quant_int10(x[:, 0, :], bos_lut["output_norm_final_norm"]["bos_s_in"]) if bos_lut else x[:, 0, :]'
)
t = t.replace(
    '    x_final_q = quant_int10(x_final, scf["s_out"])\n    x_final_q[:, 0, :] = x_final[:, 0, :]',
    '    x_final_q = quant_int10(x_final, scf["s_out"])\n    x_final_q[:, 0, :] = quant_int10(x_final[:, 0, :], bos_lut["output_norm_final_norm"]["bos_s_out"]) if bos_lut else x_final[:, 0, :]'
)

# 5. bos_lut 로드 + forward 호출에 전달 — main 부분에 추가
#    rms_scales 로드하는 근처에 bos_lut 로드 삽입
t = t.replace(
    "    rms_scales =",
    "    bos_lut = json.loads(Path(\"diagnostics/ex_norm10_p5/bos_lut/bos_lut.json\").read_text())\n    rms_scales ="
)
# forward_combined_strict 호출 부분 찾기 (이름이 다르면 아래 grep으로 확인)
# 흔한 형태: forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales)
t = t.replace(
    "forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales)",
    "forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales, bos_lut=bos_lut)"
)

dst.write_text(t)
print(f"patched -> {dst}")
# 패치 됐는지 확인
import re
for m in re.finditer(r".*bos_lut.*", t):
    print(m.group(0).strip())
