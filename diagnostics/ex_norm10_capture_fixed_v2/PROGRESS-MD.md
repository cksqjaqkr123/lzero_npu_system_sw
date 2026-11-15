# Gemma-2B INT10 RMSNorm — 진행 정리 (2026-11-08)

## 결론 한 줄

**핸드오프 MD의 "INT10 불가능, Top1 8.33%"는 틀렸다. 범인은 INT10이 아니라 BOS 1개 토큰이었다.**
**BOS만 FP로 두면, E2E Top1 86.5% (1606개 위치), KL 0.109.**

## 원래 MD vs 실제

| MD 주장 | 실제 |
|---|---|
| INT10 불가능, Local NMSE 0.093 | Local NMSE는 `s_out` 정책 문제였음. `s_out=p9999`로 0.001-0.003 |
| E2E Top1 8.33%, 출력 0이 80% | `all_INT10`에서 재현됨 (8/96=8.33%). 근데 BOS 때문 |
| gamma ≈ 0.9-1.1 | **틀림.** GGUF는 `weight-as-gamma`, `blk.0 attn` [0, 5.97], `output_norm` [0.21, 10.43] |
| Norm 37개 | **53개** (26×2 + 1) |
| hidden 2048, intermediate 16384 | hidden **2304**, intermediate **9216**, q_dim 2048, kv_dim 1024, heads 8, kv_heads 4 |

## 핵심 발견: BOS

- BOS는 0번 위치, 1개 토큰. `x_max 2.03`, 나머지 99개 `0.675`보다 3배 큼
- `s_in=0.005`로 하면 BOS가 잘리고, 그 잘린 BOS가 어텐션을 통해 뒤 32개까지 망가뜨림
- `all_INT10` → Top1 0/32, KL 11.02
- `BOS_FP` (BOS만 FP) → Top1 29/32=90.6%, KL 0.06
- E2E 3대화 (96개): `all_INT10` 8.33% → `BOS_FP` **88.5%**, KL 7.40 → 0.11
- **1024-target 검증 (1606개, 다른 프롬프트)**: `BOS_FP` **86.5%**, KL **0.109**, NMSE 0.0064

## 스케일 정책 (최종)

- **`s_out = p9999(y) / 511`** — `output_only` KL 0.09. `s_out`은 문제 아님. `p999`는 0개 Norm에서 최적, `p9999`가 38개에서 최적
- **`s_in = absmax(x, BOS 제외) / 127`** — `input_only` KL 11.26이 BOS 때문이었음. BOS 빼면 KL 0.011
- **BOS bypass** — `pos==0`이면 FP, 나머지 INT10. `s_in`을 키우는 것보다 7배 좋음 (KL 0.011 vs 0.077)

## P0-A Capture — 100%

- V2 All-in-One: gamma + s_in + channel + s_out 한 번에
- Gamma audit: GGUF는 `weight-as-gamma`, `gamma==0`은 `blk.0`에만 6개 (attn 5, ffn 1), F32라서 진짜 0
- Super channels: 854, 1002, 114, 1683, 1393. `blk.0 attn` 1393: 입력 0.57, 출력 81.24, gamma 5.97
- Dead channels: 위 6개

## P0-B Scale — 100%

- Oracle (s_in=p999): 대부분 NMSE 0.001-0.003. Worst 5: `blk.1 attn` 0.1015, `blk.0 attn` 0.0700, `blk.5 attn` 0.0684, `blk.5 ffn` 0.0370, `blk.2 attn` 0.0201
- Channel error: Worst 5 NMSE의 94-98%가 1-2개 채널 (1393, 114+1683, 1002)
- `s_out` sweep: `p999` → `p9999`로 5-31배 개선. `s_out=p9999`가 sweet spot
- **`epsilon_q = RNE(eps / s_in^2)`**: 전부 0. `eps=1e-6`, `mean(q^2)~10000` 대비 무시 가능. RTL에서 `epsilon` 고민할 필요 없음
- **Calibration/Validation 분리**: Calibration prefill 128 (8대화) → Validation prefill 256 (다른 8대화, 1606개) → 86.5%. 과적합 아님
- **1024-target E2E**: 1606개로 86.5%, KL 0.109. 목표 1024개 초과 달성

## P1 Propagation — 100%

- Coarse: k=0 Top1 100%, k=1 62.5%, k=2-20 ~50% flat. 점진적 누적 아님, `blk.0`에서 한 번에 깨짐
- `blk.0` split: `attn_only` KL 4.20, `ffn_only` KL 0.41. 범인은 `blk.0 attn`
- Super isolation: `super_fp` KL 4.26, `bulk_fp` KL 0.70. 범인은 super가 아니라 **bulk**
- Q_IN / Q_OUT: `input_only` KL 11.26, `output_only` KL 0.09. `s_out`은 괜찮고 `s_in`이 문제
- **최종: 범인은 `blk.0`이 아니라 BOS였다.** BOS만 FP로 두면 전체가 살아남

## P2 RTL / SW — 30% (개념만)

- RTL `UniversalNormUnit` (3065줄) 확인: `gamma`, `s_in`, `s_out`, `requant` 없음. `x / sqrt(mean+eps)`만 함. `satSignedToUInt`에서 clip
- SW `compile_to_npu.py`가 `norm_mode=1`로 RMSNorm 명령 만듦. BOS bypass는 여기에 넣어야 함
- SW `static_quant/core.py`는 Linear 양자화만 함. 이미 `bos`/`non_bos`/`early`로 나눠서 평가
- **개념:**
  ```python
  # 지금: 전체 시퀀스 한 번에
  emit_norm(norm_mode=1, ...)
  # 나중에: BOS만 따로
  emit_norm(norm_mode=1, is_bos=True,  s_in=FP)      # pos 0만
  emit_norm(norm_mode=1, is_bos=False, s_in=0.005)   # pos 1..S-1
  ```
- **미완성:** `compile_to_npu.py` 자체가 아직 완성 안 됨 (Step 1-6만 있음). RTL 수정은 필요 없음. SW 컴파일러에서 `pos==0` 보고 `s_in` 다르게 주는 걸로

## P3 RoPE — 100% (문제 없음)

- `blk.0` Q/K: BOS RoPE diff **정확히 0.0**. `pos=0`이면 `cos=1, sin=0`이라서 회전 없음
- BOS는 Q/K에서 outlier 아님. 오히려 작음: Q BOS 2.29 < 나머지 p999 8.93, K BOS 4.27 < 나머지 p999 9.68
- INT10 NMSE: Q `p999` BOS 0.0002 vs 나머지 0.0074, K BOS 0.0001 vs 나머지 0.0119
- **결론: RoPE는 BOS bypass 필요 없음**

## P4 Integration — 0%

- 아직 안 시작. P2 컴파일러가 완성되면 붙이기

## 재현성

- CSV/JSON/로그는 `diagnostics/ex_norm10_capture_fixed_v2/`에 있음
- 아직 없음: source hash, dataset hash, `verification.json`, 1024-target E2E의 정식 리포트 (지금은 1606개 로그로 대체)

## 전체 진행률

| 항목 | 진행률 |
|---|---:|
| P0-A Capture | 100% |
| P0-B Scale | 100% |
| P1 Propagation | 100% |
| P2 RTL/SW | 30% (개념만) |
| P3 RoPE | 100% |
| P4 Integration | 0% |
| **전체** | **~70%** |

## 남은 일

1. **P2 구현** — `compile_to_npu.py` 완성되면, `pos==0` 분기 + `s_in` 다르게 주는 코드 넣기
2. **P4** — 전체 시스템에 붙이기
3. **MD에 BOS bypass 명시** — "INT10 불가능" → "BOS bypass로 86.5%"로 고치기
4. Long-term: Super 채널 (1002, 114, 1683, 1393) LUT 분리 — 당장은 필요 없음, BOS bypass로 충분

## 핵심 파일

- `diagnostics/ex_norm10_capture_fixed_v2/run_e2e_bos_bypass_3conv.py` → `e2e_bos_bypass_3conv.json` (96개, 88.5%)
- `diagnostics/ex_norm10_capture_fixed_v2/run_e2e_1024_validation.py` → `e2e_1024_validation.json` (1606개, 86.5%)
- `diagnostics/ex_norm10_capture_fixed_v2/run_epsilon_q_check.py` → epsilon_q 전부 0
- `diagnostics/ex_norm10_capture_fixed_v2/run_rope_bos_check.py` → `rope_bos_check.json` (RoPE diff 0.0)
- `/tmp/p9999_absmax.json` — 최종 파라미터 (s_in=absmax BOS 제외, s_out=p9999)
