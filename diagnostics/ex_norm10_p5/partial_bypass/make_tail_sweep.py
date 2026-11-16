from pathlib import Path
src = Path("diagnostics/ex_norm10_p5/bos_lut/run_e2e_bos_lut.py")
dst = Path("diagnostics/ex_norm10_p5/partial_bypass/run_tail_sweep.py")
t = src.read_text()

# 1. 시그니처
t = t.replace(
    "def forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales, bos_lut=None):",
    "def forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales, bos_lut=None, int10_layers=None):",
    1
)
# int10_layers 기본값 = 전부 INT10
t = t.replace(
    "    cfg = model.cfg\n    B, S = input_ids.shape",
    "    cfg = model.cfg\n    B, S = input_ids.shape\n    if int10_layers is None:\n        int10_layers = set(range(cfg.num_layers))",
    1
)
# 2. 루프에 is_int10 플래그
t = t.replace(
    '    for i in range(cfg.num_layers):\n        p = f"blk.{i}"',
    '    for i in range(cfg.num_layers):\n        p = f"blk.{i}"\n        is_int10 = (i in int10_layers)',
    1
)
# 3. RMSNorm만 조건으로 — GELU/RoPE는 항상 INT10으로 두고 Norm만 분리 (과제 2는 Norm 민감도가 핵심)
t = t.replace('        x_q = quant_int10(x, sc["s_in"])', '        x_q = quant_int10(x, sc["s_in"]) if is_int10 else x', 1)
t = t.replace('        normed_q = quant_int10(normed, sc["s_out"])', '        normed_q = quant_int10(normed, sc["s_out"]) if is_int10 else normed', 1)
# ffn
if 'x_q2 = quant_int10' in t:
    t = t.replace('        x_q2 = quant_int10(x, sc2["s_in"])', '        x_q2 = quant_int10(x, sc2["s_in"]) if is_int10 else x', 1)
    t = t.replace('        normed2_q = quant_int10(normed2, sc2["s_out"])', '        normed2_q = quant_int10(normed2, sc2["s_out"]) if is_int10 else normed2', 1)
    print("ffn patched")
# final은 루프 밖이라서 is_int10이 없음 — tail_k에 따라 final도 결정
# final은 blk.25에 포함된다고 보고, tail_k에 따라 처리 (아래 main에서 final_int10 플래그로)
# 여기서는 일단 그대로 두고, main에서 int10_layers에 25가 있으면 final도 INT10, 아니면 FP로 처리하게
# final 부분은 별도로 패치: x_qf / x_final_q
if 'x_qf = quant_int10' in t:
    t = t.replace('    x_qf = quant_int10(x, scf["s_in"])', '    x_qf = quant_int10(x, scf["s_in"]) if (25 in int10_layers) else x', 1)
    t = t.replace('    x_final_q = quant_int10(x_final, scf["s_out"])', '    x_final_q = quant_int10(x_final, scf["s_out"]) if (25 in int10_layers) else x_final', 1)
    print("final patched")

# 4. --tail_k 인자
t = t.replace(
    '    ap.add_argument("--prefill", type=int, default=128)',
    '    ap.add_argument("--prefill", type=int, default=128)\n    ap.add_argument("--tail_k", type=int, default=None, help="마지막 k개 레이어만 INT10, 나머지 FP")',
    1
)
# 5. forward 호출 부분에 int10_layers 전달 — 호출 패턴 출력해서 확인
import re
calls = list(re.finditer(r"forward_combined_strict\(.*?\)", t))
print(f"found {len(calls)} calls")
for m in calls:
    print("CALL:", m.group(0)[:250])

# 호출이 1개면 바로 교체, 여러 개면 첫 번째만
if calls:
    old = calls[0].group(0)
    # 기존 호출에 bos_lut=bos_lut가 있으면 뒤에 int10_layers 추가
    if "bos_lut=bos_lut" in old:
        new = old.replace("bos_lut=bos_lut", "bos_lut=bos_lut, int10_layers=int10_layers")
        t = t.replace(old, new, 1)
        print("call patched")
    else:
        print("WARN: bos_lut 호출 형태가 다름")

# 6. main에서 int10_layers 계산 — bos_lut 로드 바로 뒤에 삽입
old2 = '    bos_lut = json.loads(Path("diagnostics/ex_norm10_p5/bos_lut/bos_lut.json").read_text())'
new2 = old2 + '\n    int10_layers = set(range(26 - args.tail_k, 26)) if args.tail_k is not None else set(range(26))\n    print(f"[TAIL] tail_k={args.tail_k} int10_layers={sorted(int10_layers)}")'
if old2 in t:
    t = t.replace(old2, new2, 1)
    print("main patched")
else:
    print("WARN: bos_lut 로드 위치 없음")

dst.write_text(t)
print(f"patched -> {dst}")
