# Gemma-2 2B INT10 양자화 — 최종 인수인계

## 1. 개요

- 모델: Gemma-2 2B, 26 layers, hidden 2304
- 양자화: INT10 (-511 ~ 511)
- 최종 정책: **BOS FP bypass 금지, BOS도 INT10**
- 검증: 64 conv, prefill 256, tail_k 26, 전 레이어

## 2. 최종 검증 결과 (64 conv)

| 지표 | 값 |
|---|---|
| Strict Top1 | 1429 / 1664 = **85.9%** |
| KL | **0.1393** |
| NMSE | **0.0061** |

참고 — P4 베이스라인 (8 conv): Strict Top1 87.0%, KL 0.1066
- RMSNorm-only: 86.5% / 0.109
- GELU-only: 97.2% / 0.006
- RoPE-only: 100% / 0.0005

## 3. 스케일 정책

### 3.1 BOS 처리

- 일반 토큰: `norm_selected_parameters.json` (일반 스케일)
- **BOS pos 0만**: `calib_bos_absmax_1conv.json` 전용 스케일 사용
- BOS도 INT10으로 양자화, FP 우회 없음

### 3.2 Norm (RMSNorm)

- 대상: 53개 = 26 layers x 2 (attn_norm + ffn_norm) + output_norm 1개
- 스케일 수: **norm당 4개** = general `s_in` / `s_out` + BOS `s_in` / `s_out`
- 총 53 x 4 = 212개, FP32

### 3.3 GELU

- `s_in`: 레이어별 best — `absmax / p9999 / p999` 중 NMSE 최소인 것 선택
- `s_out`: absmax
- LUT: 레이어당 1023개 (input_code -511 ~ 511), INT16
- 총 26 x 1023 = 26598개
- 정본: `npu_lut_tables.json`의 `gelu` 리스트 (`s_in`, `s_out`, `lut` 배열)
- 참고: `gelu_decision.json`의 `per_layer_sin`은 후보 3개만 들어있음, 최종값 아님

### 3.4 RoPE

- Q/K: `p9999` 스케일
- `s_out = s_in`
- BOS도 INT10, bypass 없음
- 정본: `rope_calib.json` `blk.X`의 `s_q` / `s_k`
- 예시: `blk.0 s_q 0.026755137369036674, s_k 0.03146404027938843`

## 4. 전달 파일 — `lut_delivery/csv_final/`

| 파일 | 줄 수 | 내용 |
|---|---|---|
| `norm_scales_final.csv` | 107 (106 + 헤더) | 53 norms x 2 (general / BOS), `s_in` / `s_out` + 주소 |
| `gelu_scales_final.csv` | 27 (26 + 헤더) | 26 layers, `s_in` / `s_out` / `s_in_mode` + LUT 주소 |
| `gelu_lut_full.csv` | 26599 (26598 + 헤더) | 26 x 1023, `input_code` → `input_deq` → `output_code` + `addr` |
| `rope_scales_final.csv` | 27 (26 + 헤더) | 26 layers, `s_q` / `s_k` + 주소 |
| `lut_memory_map_final.csv` | 4 (3 + 헤더) | 전체 메모리 맵 요약 |

검증 완료: 빈칸 없음 (`grep ",,"` 결과 없음), 개수·값 일치

## 5. LUT 메모리 레이아웃

| 섹션 | Base 주소 | 크기 | 엔트리 | 형식 |
|---|---|---|---|---|
| `norm_scales` | 0 | 848 B | 212 | FP32, 53 norms x 4 |
| `gelu_luts` | 2048 | 53248 B | 26598 | INT16, 26 x 1023 + 패딩 |
| `rope_scales` | 55296 | 208 B | 52 | FP32, 26 x `s_q` / `s_k` |
| **합계** | - | **55504 B** | **26862** | - |

### 5.1 Norm 주소 규칙

- norm당 16 B 블록: `[general s_in (4B), general s_out (4B), BOS s_in (4B), BOS s_out (4B)]`
- `block_addr`는 4B align
- 순서: `blk.0.attn_norm, blk.0.ffn_norm, blk.1.attn_norm, ... , output_norm`

### 5.2 GELU 주소 규칙

- 레이어당 2048 B 블록: `lut_base_addr = 2048 + layer * 2048`
- 엔트리당 2 B: `addr = lut_base_addr + index * 2`, `index 0~1022`, `input_code = index - 511`

### 5.3 RoPE 주소 규칙

- 레이어당 8 B 블록: `s_q_addr = 55296 + layer * 8`, `s_k_addr = s_q_addr + 4`

## 6. 원본 파일 위치

```
diagnostics/ex_norm10_p6_bos_int10/lut_delivery/
├── norm/
│   ├── norm_selected_parameters.json   # 일반 스케일 (정본)
│   └── calib_bos_absmax_1conv.json     # BOS 전용 스케일 (1 conv absmax, 53 keys)
├── gelu/
│   └── gelu_decision.json              # 후보만 있음, 최종값 아님
├── rope/
│   └── rope_calib.json                 # blk.X s_q / s_k (정본)
├── npu/
│   └── npu_lut_tables.json             # gelu s_in/s_out/lut, rope s_q/s_k (정본)
└── csv_final/                          # 최종 CSV 5종 (이 폴더를 그대로 전달)
    ├── norm_scales_final.csv
    ├── gelu_scales_final.csv
    ├── gelu_lut_full.csv
    ├── rope_scales_final.csv
    └── lut_memory_map_final.csv
```

## 7. 재생성 방법

```bash
python3 diagnostics/ex_norm10_p6_bos_int10/lut_delivery/generate_lut_csv_final_v3.py
ls -lh diagnostics/ex_norm10_p6_bos_int10/lut_delivery/csv_final/
wc -l diagnostics/ex_norm10_p6_bos_int10/lut_delivery/csv_final/*.csv
# 기대값: norm 107, gelu_scales 27, gelu_lut 26599, rope 27, map 4
```

주의: v2 스크립트에 `base` 변수 섀도잉 버그가 있었음. v3에서 `base_path` / `lut_base` / `r_base`로 분리해 수정됨. 재생성은 반드시 v3 사용.

## 8. 후임자 작업 — 스케일 미세조정

- 이 문서의 CSV가 기준점. 여기서부터 스케일만 미세조정하면 됨
- 우선순위: Norm 일반 스케일에 MSE-optimal clipping 적용 (예상 1~3% 개선, 미수행)
- GELU `s_in`은 이미 레이어별 best로 선택됨, `s_out`은 absmax 유지
- RoPE는 `p9999` 유지, BOS 포함 전 레이어 INT10 유지
- BOS FP bypass를 다시 도입하지 말 것

## 9. 미수행 / 선택 검증 (참고)

- FP vs FP 베이스라인 100% 체크
- PPL 비교
- Greedy 생성 비교
- Top1-Top2 gap 체크 (낮은 conv)
