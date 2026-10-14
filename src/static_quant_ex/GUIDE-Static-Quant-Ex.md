# static_quant_ex 전체 가이드라인 - 16BIN GeLU + p999/511 수정판

## 0. 프로젝트 구조
```
src/static_quant_ex/
  cli.py                          # 원본 CLI
  hardware.py                     # FIXED: select_bin closest
  search.py                       # FIXED: p999/511 + RepresentativeRowSampler + bin-sync
  core.py / gemma.py / export.py
  cal_big_64.jsonl                # 테스트용 64샘플, seq 128, first 64 records
luts/
  gelu_16BIN_combined_16384B_10_8.bin   # 16*1024=16384B, 10->8
  gelu_scale_map_16b.json              # 16개 BIN 고정 스케일 (3.51~40.0)
src/static_quant_ex/ (우리가 추가한 파일)
  capture_all_layers_v2_fixed_64.py    # 전체 레이어 pre/post GeLU 포착 -> summary_A.csv

## 1. 우리가 지금까지 해온 것

### 1.1 문제 발견
- 200MB 원본 report.json (18레이어 gate_proj 전체) 분석
  - L0 pre_p999 23.125 / pre_absmax 78.0인데 BIN0 40.0으로 가서 GELU mse 2.5, 정확도 46%
  - L17 pre_absmax 176 / pre_p999 4.28로 41배 차이, local_nmse 0.191 (80.9%)
  - BIN 분포가 고정 LUT(3.51~40.0)에 오버피팅

### 1.2 수정한 것
**hardware.py**
```python
def select_bin(self, s10_or_pre_p999): # FIXED
  pre = s10*511 if s10<0.2 else s10
  best = min(bins, key=|bin_pre - pre|) # closest
  return bid, in_scale, out_scale

def select_bin_covering(...): # 구 로직 보관
  for bin in sorted_bins:
    if in_scale >= real_in_scale: return bin
```
- covering(첫번째로 커버하는 BIN) -> closest(절대값 차이 최소)로 변경
- L0 23.1 -> 40.0이 아니라 20.44(BIN1)로 가서 mse 0.00848, 정확도 99.82% (300배 개선)

**search.py**
1. `_extract_ndarrays_from_obj()` : Reservoir/RepresentativeRowSampler 대응, ndarray 추출
2. `_get_sample_array()` : amax 0.5~200 필터, size 큰거 우선
3. `_compute_p999(arr, 99.9)` : np.percentile(abs(arr), 99.9)
4. `joint_scale_search()`:
   - s_X = p999/127 (absmax 대신 p999)
   - s10 = p999/511
   - p999>50이면 p99*1.35 fallback (RepresentativeRowSampler 버그 방지)
   - s10>0.08이면 summary_A.csv median으로 cap
   - **bin-sync**: scale_map에서 closest bin 찾아 s10 = bin_in_scale로 덮어씀 (LUT와 일치시키기)
   ```python
   _best = min(scale_map, key=|bin_pre - pre_p999|)
   s10 = _best.in_scale
   ```

**추가 파일**
- `capture_all_layers_v2_fixed.py`: INT10->GeLU->INT10 전체 레이어 한방 포착, summary_A.csv 생성
lut bin 파일, 메모리 설정 등을 추가하기 위해 cli.py, export.py 등도 다같이 수정

### 1.3 결과 (18레이어 gate_proj)
- GELU mse avg 0.0127 (min 0.00848 L0, max 0.0159 L3)
- Local nmse avg 0.0248 -> 정확도 97.5%
- BIN 분포: 2(12.31) 11개, 1(20.44) 5개, 3(8.12) 1개, 9(4.21) 1개
- 병목: L1 89.2%, L17 80.9% (outlier 176)

## 2. 명령어 파라미터 가이드

### 2.1 summary_A.csv 생성 (전체 레이어 pre/post 포착)
```bash
cd ~/lzero_npu_system_sw
PYTHONPATH=src python3 src/static_quant_ex/capture_all_layers_v2_fixed.py \
  --model google/gemma-2b-it \
  --calibration-data ./src/static_quant_ex/cal_big_64.jsonl \
  --output-dir ./luts \
  --layers all \
  --samples 64 --sequence-length 128 \
  --device cpu --dtype bfloat16
# 출력: ./luts/summary_A.csv (layer,pre_absmax,pre_p999,pre_p99,pre_scale_A_p999,post_...,elements)
# elements 4000+ 나와야 64*128 성공
```

### 2.2 LUT 생성
```bash
PYTHONPATH=src python3 src/LUT.py
# luts/gelu_16BIN_combined_16384B_10_8.bin 16384B + gelu_scale_map_16b.json 16BIN 생성
ls -lh luts/gelu_16BIN*
```

### 2.2 캘리브레이션 (gate_proj 전체)

**방법 A: cli.py 직접 (권장)**
```bash
rm -rf ./output_16b_all_p999 && mkdir -p ./output_16b_all_p999

PYTHONPATH=src python3 -m static_quant_ex.cli \
  --calibration-data ./src/static_quant_ex/cal_big_64.jsonl \
  --output-dir ./output_16b_all_p999 \
  --layers all --modules gate_proj \
  --samples 64 --sequence-length 128 \
  --device cpu --dtype bfloat16 \
  --gelu-lut-path ./luts/gelu_16BIN_combined_16384B_10_8.bin \
  --gelu-scale-map-path ./luts/gelu_scale_map_16b.json \
  --allow-unverified-export --allow-inaccurate-parameters
# 출력: report.json (200MB), scale_search.json, manifest.json, qparams.bin
```

**방법 B: cli_16b.py wrapper (기본값 16BIN)**
```bash
PYTHONPATH=src python3 src/static_quant_ex/cli_16b.py \
  --calibration-data ./src/static_quant_ex/cal_big_64.jsonl \
  --output-dir ./output_16b_all_p999 \
  --layers all --modules gate_proj \
  --samples 64 --sequence-length 128 --device cpu --dtype bfloat16 \
  --allow-unverified-export --allow-inaccurate-parameters
```

**방법 C: run_16b.py standalone**
```bash
python3 /mnt/data/run_16b.py \
  --calibration-data ./src/static_quant_ex/cal_big_64.jsonl \
  --output-dir ./output_16b_all_p999 \
  --layers 0,8,17 --modules gate_proj \
  --samples 16 --sequence-length 128
```

**방법 D: 전체 자동 스크립트**
```bash
bash /mnt/data/run_all_p999.sh
# 1. pycache 정리
# 2. 파일 덮어쓰기
# 3. LUT 생성
# 4. output_16b_all_p999 생성
```

### 3.4 파라미터 설명
| 파라미터 | 기본 | 설명 |
|---|---|---|
| --model | google/gemma-2b-it | 모델 |
| --calibration-data | (필수) | cal_big_64.jsonl 경로, first 64 records |
| --output-dir | (필수) | 비어있어야 함 |
| --layers | 0 | all 또는 0,8,17 |
| --modules | q_proj | all 또는 gate_proj,up_proj,down_proj |
| --samples | 8 | 캘리브 샘플 수, 64 권장 |
| --sequence-length | 128 | 토큰 길이, 128 권장 -> elements ~4000 |
| --device | cpu | cpu/cuda |
| --dtype | float32 | bfloat16 권장 (메모리) |
| --gelu-lut-path | ...8192B.bin | 16BIN이면 16384B 파일 |
| --gelu-scale-map-path | ...json | 16BIN map |
| --s10 | None | 고정 s10, None이면 p999/511 자동 |
| --scale-mode | minmax | fixed/minmax/mse |
| --search-rows | 64 | activation search rows |
| --bos-search-rows | 16 | BOS 보장 샘플 |
| --ratio-tolerance | 1e-3 | multiplier/shift 근사 허용 오차 |
| --allow-unverified-export | false | 검증 안된 profile도 export |
| --allow-inaccurate-parameters | false | inaccurate ratio도 허용 |
| --low-mem | false | m/n/k chunk 8/16/64로 줄임 |

### 3.5 결과 집계
```bash
# 200MB report.json -> CSV 5개

## 4. 테스트용 cat_big_64

- 파일: `src/static_quant_ex/cal_big_64.jsonl` / `calibration-data` arg
- 포맷: JSONL text field, first nonempty records in file order
- 요청 64, 실제 64, valid_tokens 1676 (예시 L0)
- tokenizer: GemmaTokenizer, bos 2, eos 1, seq 128
- `capture_all_layers_v2_fixed_64.py`로 64*128 -> elements 4000+ 확인 필요

## 5. 다음 단계

병목 2개

L1 nmse 0.108 (89.2%) - pre 8.6으로 작은데도 local mse 0.36
L17 nmse 0.191 (80.9%) - pre_absmax 176 outlier 때문에 p999 4.28인데도 absmax가 41배 큼, s_X 0.033으로 너무 작게 잡힘

- [ ] L1/L17 병목 해결: p99*1.35 fallback 개선, median cap
- [ ] evaluate_all_bins()로 MSE 최소 BIN 선택으로 변경 (현재는 closest pre)
- [ ] calibration random 256으로 확대 (오버피팅 해결 위해 추가 조정)

---
