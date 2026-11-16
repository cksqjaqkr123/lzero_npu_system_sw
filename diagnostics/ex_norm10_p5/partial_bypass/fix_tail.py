from pathlib import Path
src = Path("diagnostics/ex_norm10_p5/bos_lut/run_e2e_bos_lut.py")
dst = Path("diagnostics/ex_norm10_p5/partial_bypass/run_tail_sweep.py")
t = src.read_text()

# forward_combined_strict만 분리해서 패치
marker = "def forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales, bos_lut=None):"
assert marker in t, "forward_combined_strict 없음"
head, tail = t.split(marker, 1)
# tail = forward_combined_strict 이후 전부
# tail 안에서만 패치
tail = tail.replace(
    "def forward_combined_strict",
    "DEF_PLACEHOLDER", 1
) if False else tail  # 더미

# 1. 시그니처는 전체에서 1번만 (이건 forward_combined_strict 거라서 안전)
t = t.replace(marker, "def forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales, bos_lut=None, int10_layers=None):", 1)

# 2. int10_layers 기본값 — forward_combined_strict 안에서만
# forward_combined_strict 시작 부분 찾기: 그 함수 다음에 오는 cfg 라인만
idx = t.find("def forward_combined_strict")
# 그 뒤에서 첫 번째 cfg 라인만
cfg_pat = "    cfg = model.cfg\n    B, S = input_ids.shape"
pos = t.find(cfg_pat, idx)
assert pos != -1
t = t[:pos] + cfg_pat + "\n    if int10_layers is None:\n        int10_layers = set(range(cfg.num_layers))" + t[pos+len(cfg_pat):]

# 3. 루프 플래그 — forward_combined_strict 안에서만
loop_pat = '    for i in range(cfg.num_layers):\n        p = f"blk.{i}"'
pos2 = t.find(loop_pat, idx)
assert pos2 != -1
t = t[:pos2] + loop_pat + '\n        is_int10 = (i in int10_layers)' + t[pos2+len(loop_pat):]

# 4. RMSNorm만 조건으로 — forward_combined_strict 이후에서만, 각각 1번씩
# attn
t = t.replace('        x_q = quant_int10(x, sc["s_in"])', '        x_q = quant_int10(x, sc["s_in"]) if is_int10 else x', 1)
# 이 replace가 forward_fp를 건드렸을 수 있음 — forward_fp는 x_q가 없으니 안전
# normed_q도 마찬가지
t = t.replace('        normed_q = quant_int10(normed, sc["s_out"])', '        normed_q = quant_int10(normed, sc["s_out"]) if is_int10 else normed', 1)
if 'x_q2 = quant_int10' in t:
    t = t.replace('        x_q2 = quant_int10(x, sc2["s_in"])', '        x_q2 = quant_int10(x, sc2["s_in"]) if is_int10 else x', 1)
    t = t.replace('        normed2_q = quant_int10(normed2, sc2["s_out"])', '        normed2_q = quant_int10(normed2, sc2["s_out"]) if is_int10 else normed2', 1)
if 'x_qf = quant_int10' in t:
    t = t.replace('    x_qf = quant_int10(x, scf["s_in"])', '    x_qf = quant_int10(x, scf["s_in"]) if (25 in int10_layers) else x', 1)
    t = t.replace('    x_final_q = quant_int10(x_final, scf["s_out"])', '    x_final_q = quant_int10(x_final, scf["s_out"]) if (25 in int10_layers) else x_final', 1)

# 5. --tail_k
t = t.replace(
    '    ap.add_argument("--prefill", type=int, default=128)',
    '    ap.add_argument("--prefill", type=int, default=128)\n    ap.add_argument("--tail_k", type=int, default=None)',
    1
)
# 6. main에 int10_layers 계산
old2 = '    bos_lut = json.loads(Path("diagnostics/ex_norm10_p5/bos_lut/bos_lut.json").read_text())'
new2 = old2 + '\n    int10_layers = set(range(26 - args.tail_k, 26)) if args.tail_k is not None else set(range(26))\n    print(f"[TAIL] tail_k={args.tail_k} int10_layers={sorted(int10_layers)}")'
t = t.replace(old2, new2, 1)

# 7. 호출에 int10_layers 전달
import re
calls = list(re.finditer(r"forward_combined_strict\(.*?\)", t))
print(f"calls: {len(calls)}")
for m in calls:
    print(m.group(0)[:250])
    # 첫 번째 호출이 main에 있는 호출일 것 — bos_lut=bos_lut 포함된 것만
    if "bos_lut=bos_lut" in m.group(0) and "int10_layers" not in m.group(0):
        old = m.group(0)
        new = old.replace("bos_lut=bos_lut", "bos_lut=bos_lut, int10_layers=int10_layers")
        t = t.replace(old, new, 1)
        print("call patched")
        break

dst.write_text(t)
print(f"patched -> {dst}")
