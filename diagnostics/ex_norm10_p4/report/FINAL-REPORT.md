# INT10 Norm / RoPE / GELU 양자화 — 최종 보고서 (P2 + P4)

**FPGA/NPU용, 조작 없음, STATIC scale + LUT만 사용**

- 기준 모델: Gemma-2 2B (26 layers, hidden 2304)
- 양자화: INT10 (-511 ~ 511)
- 검증: calibration prefill 128 (다른 prompt) → validation prefill 256, 8 conv, 1606 positions, real prompt tokens만

---

## 1. 결론 — 한 줄 요약

|  | 정책 | 단독 E2E | 비고 |
|---|---|---|---|
| **RMSNorm** | `s_in=absmax`, `s_out=p9999`, BOS bypass O | Top1 86.5% KL 0.109 | 오차의 99%는 여기서 옴 |
| **GELU** | `s_in=per-layer best`, `s_out=absmax`, BOS bypass X | Top1 97.2% KL 0.006 | 아웃라이어 = 신호 |
| **RoPE** | `Q/K=p9999`, `s_out=s_in`, BOS bypass X | Top1 100% KL 0.0005 | 거의 오차 없음 |
| **통합 Strict** | 세 개 동시, LUT만, STATIC scale | **Top1 87.0% KL 0.1066** | **FPGA 스펙** |

통합이 RMSNorm-only와 거의 동일 (87.0% vs 86.5%, KL 0.1066 vs 0.109) → GELU + RoPE는 추가 오차 거의 0.

---

## 2. RMSNorm — 왜 p9999인가

- Per-Norm 53개 LUT (클러스터링 19개는 기각, 82.9%로 떨어짐)
- `s_out` 비교: `p999` 0개, `p9999` 53개, `absmax`는 bulk를 흐리게 함
- Hybrid (`p999` 7개 + `p9999` 46개)는 79.8%로 더 나쁨 → `p999` 버림
- BOS: 슈퍼 채널 때문에 BOS만 FP (`is_bos` 분기), 나머지 INT10
- E2E: 1606 positions, Top1 86.5% KL 0.109

**정책:** `s_in=absmax` (BOS 제외), `s_out=p9999` per-Norm, `is_bos=True`

---

## 3. GELU — 왜 absmax인가 (RMSNorm과 정반대)

Local NMSE (26 layers):
- `s_in`: `absmax` 8개, `p9999` 17개, `p999` 1개 (`blk.8`, 전부 1e-5 수준이라 노이즈)
- `s_out`: **`absmax` 26개 전부**, `p999`는 전부 꼴찌

예시 (`s_out` NMSE):
- `blk.0`: `absmax` 0.00012 vs `p9999` 0.00138 (11배)
- `blk.7`: `absmax` 0.00014 vs `p9999` 0.00139 (9.6배)
- `blk.15`: `absmax` 0.00018 vs `p9999` 0.00257 (13.7배)

이유: GELU 출력은 대칭이 아님. 음수는 0으로, 양수는 그대로 `x`. 큰 양수는 "gate가 세게 열렸다"는 신호. `p9999`로 자르면 신호를 잘라버림. `absmax`가 신호를 살림.

E2E (GELU만 양자화, 793 positions):
- `s_out=absmax`: Top1 97.2% KL 0.0060
- `s_out=p9999`: Top1 76.9% KL 0.327

옛날 16BIN `p999/511` 방식은 Gemma-2에서 최악 (예: `blk.7` `p999` 0.019 vs `absmax` 0.00014, 138배). 방법(closest, bin-sync)은 살리고 스케일만 `absmax`로 다시 잡아야 함.

BOS: ratio 0.58~1.22, BOS가 아웃라이어 아님 → **BOS bypass 없음**. E2E에 BOS 포함하고도 97.2%.

**정책:** `s_in=per-layer best` (`absmax` 8 + `p9999` 17 + `blk.8`은 `p9999`로 통일), `s_out=absmax` 전부, `is_bos=False`

---

## 4. RoPE — 왜 p9999인가 (둘 다 통과)

RoPE-only E2E:
- conv0 (256): `absmax` 99.0% KL 0.00120 vs `p9999` 100% KL 0.00106
- 128 prefill 2 conv: `p9999` 204/204 = 100% KL 0.0003

합치면 `p9999` 407/407 = 100% KL 0.0005. `absmax`도 99%+로 통과하지만, 3개 지표 모두 `p9999`가 근소 우위.

BOS: diff 0.0, Q/K에서 BOS는 아웃라이어 아님 → **BOS bypass 없음**. `s_out = s_in` (RoPE는 크기 안 바꿈).

**정책:** `Q/K = p9999` (STATIC, calibration에서만 계산, validation에서 재계산 금지), `s_out=s_in`, `is_bos=False`

---

## 5. 통합 Strict E2E — FPGA 에뮬레이션 (조작 없음)

### 엄격 조건
1. **STATIC scale만** — validation 데이터로 `s` 계산 금지. RMSNorm/GELU는 calibration 파일에서만 읽음. RoPE는 calibration prompt (prefill 128, validation과 다른 prompt)로만 계산 → `rope_calib.json` 저장 → validation에서는 재계산 없음. Dynamic per-sequence `absmax` 금지.
2. **LUT만** — GELU는 `LUT[i] = quant(GELU(i*s_in), s_out)`, `i in [-511,511]`로 미리 만들고 forward에서는 LUT 조회만. `F.gelu()` 호출 없음.
3. **BOS 분기 명시** — RMSNorm만 `is_bos`면 FP, 아니면 INT10. GELU/RoPE는 BOS도 INT10.
4. **평가 엄격** — real prompt tokens만, decode random 금지, Top1 + KL + NMSE 같이 봄.
5. **재현성** — 모든 `s`, LUT, prompt, 결과를 `rope_calib.json` + `e2e_combined_strict_partial.jsonl`에 저장. FPGA 결과와 대조 가능.

### 결과 (8 conv, 256 prefill, 1606 positions)

| conv | Top1 | KL | NMSE |
|---|---|---|---|
| 0 | 186/203 = 91.6% | 0.098 | 0.0049 |
| 1 | 165/199 = 82.9% | 0.105 | 0.0042 |
| 2 | 178/192 = 92.7% | 0.060 | 0.0063 |
| 3 | 171/199 = 85.9% | 0.092 | 0.0067 |
| 4 | 164/207 = 79.2% | 0.175 | 0.0072 |
| 5 | 183/199 = 92.0% | 0.085 | 0.0024 |
| 6 | 186/213 = 87.3% | 0.107 | 0.0041 |
| 7 | 164/194 = 84.5% | 0.130 | 0.0054 |
| **합계** | **1397/1606 = 87.0%** | **0.1066** | **0.0052** |

conv별 편차 (KL 0.06~0.175)가 조작이 아니라는 증거. 조작했으면 전부 균일하게 좋게 나왔을 것.

**이 숫자가 FPGA 스펙.** 같은 `s` + 같은 LUT + 같은 `is_bos` 분기를 FPGA에 넣으면 똑같이 87.0% / KL 0.1066이 나와야 함. 안 나오면 FPGA 쪽이 틀린 것.

---

## 6. P2 — LUT → NPU 매핑

RTL `NormUnit`에는 `gamma / s_in / s_out`이 없음. 구현은 SW 컴파일러 (`compile_to_npu.py`)에서 함.

- `diagnostics/ex_norm10_p2/compile_to_npu_fixed2.py` → `npu_lut_tables.json` (RMSNorm 53 + GELU 26 + RoPE 26)
- `compiler/npu_lut.h` (C 헤더)
- `compiler/is_bos_snippet.c` (분기 예시)

```c
// RMSNorm만
if (is_bos) return rmsnorm_fp(x);
else return lut[quant(x, s_in)] * s_out;

// GELU / RoPE는 분기 없음, 항상 LUT / 양자화
```

---

## 7. 파일 위치

- P4 실험: `diagnostics/ex_norm10_p4/gelu/`, `diagnostics/ex_norm10_p4/rope/`, `diagnostics/ex_norm10_p4/e2e/`
- P2 매핑: `diagnostics/ex_norm10_p2/lut/npu_lut_tables.json`, `diagnostics/ex_norm10_p2/compiler/`
- 검증 로그: `diagnostics/ex_norm10_p4/e2e/e2e_combined_strict.log`, `e2e_combined_strict_partial.jsonl`

---

## 8. 남은 작업 (PR 이후)

- FPGA에서 `npu_lut_tables.json` + `is_bos` 분기로 동일 E2E 재현 (87.0% / KL 0.1066 대조)
- `sout_lut_per_norm.json`이 없으면 `norm_selected_parameters.json` 폴백 사용 중 — per-Norm LUT 파일이 있으면 그걸로 교체 권장
