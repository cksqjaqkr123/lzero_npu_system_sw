#!/usr/bin/env python3
"""P4 run_e2e_combined_strict_fixed.py -> P6 BOS INT10 + conv_idx + tail_k + flush 패치
서버에서: python3 patch_p4_to_p6.py
"""
import pathlib, re, sys
path = pathlib.Path("diagnostics/ex_norm10_p6_bos_int10/run_tail_sweep.py")
text = path.read_text()
orig = text

# 1) forward_combined_strict 시그니처에 bos_scales 추가
text = text.replace(
    "def forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales):",
    "def forward_combined_strict(model, input_ids, rms_scales, gelu_luts, gelu_scales, rope_scales, bos_scales=None):"
)

# 2) BOS FP 3곳 -> BOS INT10 (변수명에 맞게)
# attn_norm: x_q, sc, p
text = text.replace(
    "        x_q = quant_int10(x, sc[\"s_in\"])\n        x_q[:, 0, :] = x[:, 0, :]",
    "        x_q = quant_int10(x, sc[\"s_in\"])\n        _bos = bos_scales.get(f\"{p}.attn_norm\", sc) if bos_scales else sc\n        x_q[:, 0, :] = quant_int10(x[:, 0, :], _bos[\"s_in\"])"
)
# ffn_norm: x_q2, sc2, p
text = text.replace(
    "        x_q2 = quant_int10(x, sc2[\"s_in\"])\n        x_q2[:, 0, :] = x[:, 0, :]",
    "        x_q2 = quant_int10(x, sc2[\"s_in\"])\n        _bos2 = bos_scales.get(f\"{p}.ffn_norm\", sc2) if bos_scales else sc2\n        x_q2[:, 0, :] = quant_int10(x[:, 0, :], _bos2[\"s_in\"])"
)
# output_norm: x_qf, scf
text = text.replace(
    "    x_qf = quant_int10(x, scf[\"s_in\"])\n    x_qf[:, 0, :] = x[:, 0, :]",
    "    x_qf = quant_int10(x, scf[\"s_in\"])\n    _bosf = bos_scales.get(\"output_norm\", scf) if bos_scales else scf\n    x_qf[:, 0, :] = quant_int10(x[:, 0, :], _bosf[\"s_in\"])"
)

# 3) normed_q, normed2_q, x_final_q도 BOS 전용 s_out으로 - 현재는 일반 s_out만 쓰므로, BOS pos0만 다시 계산
# P4 코드가 normed 전체를 s_out으로 양자화한 뒤 BOS를 FP로 덮었는데, 지금은 BOS를 INT10으로 해야 하므로
# normed_q[:,0]를 BOS s_out으로 다시 양자화하는 패치 (이미 x_q에서 BOS를 INT10으로 했으므로 normed 자체는 BOS 포함)
# 여기서는 s_out BOS 처리가 필요하면 아래 줄을 넣음 - 원본에 normed_q 라인이 있으면 BOS pos만 교체
# (원본이 normed_q = quant_int10(normed, sc["s_out"]) 형태면)
text = text.replace(
    "        normed_q = quant_int10(normed, sc[\"s_out\"])",
    "        normed_q = quant_int10(normed, sc[\"s_out\"])\n        _bos_o = bos_scales.get(f\"{p}.attn_norm\", sc) if bos_scales else sc\n        normed_q[:, 0, :] = quant_int10(normed[:, 0, :], _bos_o[\"s_out\"])"
)
text = text.replace(
    "        normed2_q = quant_int10(normed2, sc2[\"s_out\"])",
    "        normed2_q = quant_int10(normed2, sc2[\"s_out\"])\n        _bos_o2 = bos_scales.get(f\"{p}.ffn_norm\", sc2) if bos_scales else sc2\n        normed2_q[:, 0, :] = quant_int10(normed2[:, 0, :], _bos_o2[\"s_out\"])"
)
text = text.replace(
    "    x_final_q = quant_int10(x_final, scf[\"s_out\"])",
    "    x_final_q = quant_int10(x_final, scf[\"s_out\"])\n    _bos_of = bos_scales.get(\"output_norm\", scf) if bos_scales else scf\n    x_final_q[:, 0, :] = quant_int10(x_final[:, 0, :], _bos_of[\"s_out\"])"
)

# 4) argparse에 conv_idx, tail_k, bos_json 하드코딩용 인자 추가
text = text.replace(
    '    ap.add_argument("--prefill", type=int, default=128)',
    '    ap.add_argument("--prefill", type=int, default=128)\n    ap.add_argument("--conv_idx", type=int, default=None)\n    ap.add_argument("--tail_k", type=int, default=26)\n    ap.add_argument("--bos_json", default="diagnostics/ex_norm10_p6_bos_int10/calib/calib_bos_absmax_1conv.json")'
)

# 5) rms 로딩: /tmp/p9999_absmax.json 우선 로딩을 버리고 rms_json을 직접 쓰기 (P4 폴백이 p9999를 끌어오는 문제)
text = text.replace(
    "    # RMSNorm\n    try:\n        rms_scales = load_rms_scales(args.rms_p9999)\n        log(f\"[DATA] RMSNorm from {args.rms_p9999}\")\n    except Exception as e:\n        log(f\"[WARN] p9999_absmax.json 실패 {e}, {args.rms_json}으로 폴백\")\n        rms_scales = load_rms_scales(args.rms_json)",
    "    # RMSNorm - P6: rms_json을 직접 사용 (p9999 폴백 금지)\n    rms_scales = load_rms_scales(args.rms_json)\n    log(f\"[DATA] RMSNorm from {args.rms_json}\")"
)
# 만약 위 패턴이 안 맞으면, rms_p9999 줄만 주석 처리
if "rms_p9999" in text and "RMSNorm from" in text and "P6: rms_json" not in text:
    text = text.replace('rms_scales = load_rms_scales(args.rms_p9999)', 'rms_scales = load_rms_scales(args.rms_json)  # P6 fix')

# 6) BOS 스케일 로딩 (main에서, rms 로딩 직후)
if "bos_scales" not in text.split("def main")[1].split("log(f\"[DATA] RMSNorm")[0]:
    text = text.replace(
        '    log(f"[DATA] RMSNorm from {args.rms_json}")',
        '    log(f"[DATA] RMSNorm from {args.rms_json}")\n    # BOS 전용 스케일 (하드코딩 기본값, FP 금지)\n    bos_scales = {}\n    try:\n        bos_scales = json.load(open(args.bos_json))\n        log(f"[DATA] BOS scales from {args.bos_json} keys={len(bos_scales)}")\n    except Exception as e:\n        log(f"[WARN] BOS scales 로드 실패 {e}, 일반 스케일로 폴백 (BOS도 INT10)")'
    )

# 7) forward 호출에 bos_scales 전달
text = text.replace(
    "logits_q = forward_combined_strict(model, input_tensor, rms_scales, gelu_luts, gelu_scales, rope_scales)",
    "logits_q = forward_combined_strict(model, input_tensor, rms_scales, gelu_luts, gelu_scales, rope_scales, bos_scales)"
)

# 8) conv_idx로 1개만 자르기
text = text.replace(
    "    convs = build_conversations(args.n_conv, prefill_lengths=[args.prefill]*args.n_conv, decode_lengths=[16]*args.n_conv)",
    "    convs_all = build_conversations(args.n_conv if args.conv_idx is None else max(args.conv_idx+1, args.n_conv), prefill_lengths=[args.prefill]*(args.n_conv if args.conv_idx is None else max(args.conv_idx+1, args.n_conv)), decode_lengths=[16]*(args.n_conv if args.conv_idx is None else max(args.conv_idx+1, args.n_conv)))\n    convs = convs_all[args.conv_idx:args.conv_idx+1] if args.conv_idx is not None else convs_all\n    log(f\"[CONV] conv_idx={args.conv_idx} -> {len(convs)} convs\")"
)

# 9) eval_pos를 tail_k로 (P4는 전체, P6는 뒤 26개)
text = text.replace(
    "        eval_pos = list(range(1, S))",
    "        eval_pos = list(range(max(1, S-args.tail_k), S))  # P6: tail_k만 평가"
)

# 10) 저장 시 flush + fsync + result.json (드라이버가 보는 파일)
text = text.replace(
    '        with open(jsonl_path, "a") as f: f.write(json.dumps(rec)+"\\n")',
    '        with open(jsonl_path, "a") as f:\n            f.write(json.dumps(rec)+"\\n"); f.flush(); import os as _os; _os.fsync(f.fileno())\n        # 드라이버용 단일 결과 파일 (conv_idx 모드)\n        try:\n            with open(out_dir/"result.json", "w") as _rf:\n                json.dump({"conv": ci, "n_match": rec["agree"], "total": rec["n"], "top1": rec["top1"], "kl": rec["kl"], "nmse": rec["logits_nmse"]}, _rf, indent=2)\n        except Exception as _e:\n            log(f"[WARN] result.json 저장 실패 {_e}")'
)

# 11) conv 루프에서 ci가 conv_idx일 때 rec conv 번호를 원래 idx로 (드라이버가 conv_XXX.json으로 복사)
# 이미 convs를 자르면 ci는 0부터 시작하므로, rec conv를 args.conv_idx로 보정
if "rec = {\"conv\": ci" in text:
    text = text.replace(
        'rec = {"conv": ci,',
        'rec = {"conv": (args.conv_idx if args.conv_idx is not None else ci),'
    )

path.write_text(text)
print(f"patched {path}")
print(f"changed: {orig != text}")
# 검증
t = path.read_text()
checks = [
    ("BOS FP 잔존", "x_q[:, 0, :] = x[:, 0, :]" in t or "x_q2[:, 0, :] = x[:, 0, :]" in t or "x_qf[:, 0, :] = x[:, 0, :]" in t),
    ("BOS INT10 있음", "quant_int10(x[:, 0, :]" in t),
    ("conv_idx 있음", "--conv_idx" in t),
    ("tail_k 있음", "--tail_k" in t),
    ("bos_scales 전달", "bos_scales)" in t),
]
for k, v in checks:
    print(k, "FAIL" if (k=="BOS FP 잔존" and v) or (k!="BOS FP 잔존" and not v) else "OK", v)
print("py_compile check:")
import py_compile
py_compile.compile(str(path), doraise=True)
print("compile OK")
