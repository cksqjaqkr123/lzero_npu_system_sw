# Gemma 2B static INT32 → INT10 calibration

실행 진입점은 `src/calibrate_gemma.py`이다. 실제 pretrained 모델의 **원래 Linear 입력**을 관측하는 local, **prefill-only** calibration이다. 모델 내부에 INT8/INT10 출력을 주입하지 않는다. Attention QKᵀ/PV activation matmul, decode, 양자화 오차 전파, perplexity, RTL bit-exact 검증은 지원/검증 범위 밖이다.

## 설치와 실행

저장소 root에서 실행한다. 기존 출력은 덮어쓰지 않으며, **새롭거나 빈 output directory**만 허용한다.

```bash
# NumPy만으로 다운로드 없는 필수 테스트 / synthetic 실행
python3 -m pip install 'numpy>=1.26,<3'
python3 -m unittest discover -s tests -v
python3 src/calibrate_gemma.py --synthetic \
  --output-dir calibration_outputs/synthetic --scale-mode mse \
  --allow-unverified-export --save-int8-weights

# 실제 pretrained 모델용 환경 (Python 3.10+ 권장)
python3 -m venv .venv-calibration
source .venv-calibration/bin/activate
python -m pip install -r requirements-calibration.txt
python -m unittest discover -s tests -v
```

모델 접근 권한은 Hugging Face에서 확보하고 기존 HF 로그인 또는 `HF_TOKEN` 환경을 사용한다. CLI는 token 옵션을 받거나 값을 저장/출력하지 않는다. 모델/라이브러리 다운로드에는 네트워크가 필요하다. 로컬 모델은 `--model /path/to/checkpoint --local-files-only`로 사용한다. `--revision`에 commit hash를 지정하면 재현에 유리하다. 원격 config의 resolved commit을 tokenizer와 weight 로딩에도 고정한다. 로컬 checkpoint는 shard/config SHA256을 기록한다.

기본은 **original `google/gemma-2b-it`** 이며, `gemma-2-2b`는 다른 architecture이므로 거부한다. config의 `model_type=gemma`, 18 layers, hidden 2048, intermediate 16384, 8 query heads, 1 KV head, head_dim 256과 실제 선택 weight shape를 검사한다. `from_pretrained`의 missing/unexpected/mismatched weights와 meta weights를 거부한다. 가중치가 없는 경우 random 모델로 대체하지 않는다. [Hugging Face Gemma 문서](https://huggingface.co/docs/transformers/model_doc/gemma)의 pretrained 로딩/모델 구조를 따른다.

### 첫 실행: layer 0 q_proj

```bash
python src/calibrate_gemma.py \
  --model google/gemma-2b-it \
  --calibration-data examples/calibration.jsonl \
  --output-dir calibration_outputs/layer0-q \
  --device cpu --dtype float32 --seed 0 \
  --samples 2 --sequence-length 32 --batch-size 1 \
  --layers 0 --modules q_proj --scale-mode minmax
```

예제 데이터는 실행 확인용이다. 배포 입력 분포를 대표하는 자체 데이터를 사용해야 한다. CPU는 전체 checkpoint를 float32로 보유하므로 모델 로딩에 대략 10GB 이상의 여유 RAM이 필요하며, exact INT64 GEMM은 느리다. 선택 module 앞의 원본 모델만 실행하고 pre-hook에서 멈추므로 layer 0 q_proj 실행은 나머지 layers/큰 logits 연산을 생략한다. checkpoint 자체는 전체 모델로 로드한다.

### 전체 layer의 7개 Linear와 held-out 검증

```bash
python src/calibrate_gemma.py \
  --model google/gemma-2b-it \
  --calibration-data /path/to/calibration.jsonl \
  --validation-data /path/to/heldout.txt \
  --output-dir calibration_outputs/all-linears \
  --device cuda --dtype bfloat16 --seed 42 \
  --samples 128 --validation-samples 32 \
  --sequence-length 256 --batch-size 1 \
  --layers all --modules all --scale-mode mse \
  --search-rows 64 --thresholds 1,.95,.9,.8,.7,.5 \
  --m-chunk 32 --n-chunk 64 --k-chunk 256
```

`--layers 0,3,7 --modules q_proj,o_proj,down_proj`처럼 부분 선택할 수 있다. `--include-lm-head`는 선택한 Linear들에 lm_head를 추가한다. lm_head도 N chunk별 weight를 읽고 입력 hook에서 멈춰 전체 vocabulary logits를 만들지 않는다. 다만 vocab 크기의 각 channel 통계/parameter와 추가 연산 비용은 필요하다. 전체 layer 실행은 module별로 원본 prefix를 재실행하는 메모리 우선 구현이라 매우 느릴 수 있다. GPU는 원본 모델 forward에 사용하고 정수 reference는 CPU NumPy INT64로 실행한다. FP32 GEMM을 exact integer backend로 사용하지 않는다.

고정 LUT 재사용:

```bash
python src/calibrate_gemma.py \
  --calibration-data examples/calibration.jsonl \
  --output-dir calibration_outputs/fixed-lut \
  --layers 0 --modules gate_proj \
  --scale-mode fixed --s10 0.1 --reuse-gelu-lut
```

이 옵션은 선택한 모든 연산에 동일 `s_10=0.1`을 강제한다. `--reuse-gelu-lut`와 탐색 모드 또는 다른 `s_10`의 조합은 오류다. LUT를 거치지 않는 연산은 `--scale-mode fixed --s10 VALUE`만 사용할 수 있다. 검색 결과는 연산마다 scale이 달라질 수 있으므로 별도 scale에 맞는 LUT가 필요하다는 metadata를 만든다. 하나의 기존 LUT와 모두 호환된다고 표시하지 않는다.

**기본 qparams.bin은 생성하지 않는다.** 다음 옵션으로만 실험용 export를 허용한다.

```bash
# 위 실제/synthetic 실행 명령에 추가
--profile generic-rne --allow-unverified-export
# 과거 layout 및 shift 보정 가설을 따로 실험할 때
--profile legacy-guess --allow-unverified-export
```

표현 불가/부정확 ratio가 있으면 위 옵션으로도 binary를 차단한다. `report.json`의 상태를 검토한 후 `--allow-inaccurate-parameters`까지 명시하면 해당 근사 결과를 실험용으로 내보낸다. NPZ/report/manifest는 차단 시에도 생성하며 manifest와 콘솔에 이유를 기록한다. 이 옵션들은 RTL 검증 상태를 바꾸지 않는다.

## 입력 데이터와 재현성

JSONL은 한 줄당 `{"text":"..."}` 객체, `.txt`는 비어 있지 않은 한 줄당 한 sample이다. 첫 N개의 비어 있지 않은 record를 사용한다. 실제 sample 수, 파일 SHA256, 선택 text SHA256, token IDs SHA256을 기록한다. 예산보다 적은 데이터는 실제 수를 기록한다. `.jsonl`의 잘못된 record에는 파일명과 줄 번호를 포함한 오류를 낸다.

기본 `--text-format plain`은 `add_special_tokens=True`, tokenizer의 BOS/EOS 설정을 따른다. `--text-format chat`은 user message 하나에 `apply_chat_template(tokenize=True, add_generation_prompt=False)`를 적용한다. 별도로 BOS를 중복 추가하지 않는다. template, tokenizer class, BOS/EOS/pad ID와 설정, truncation/padding 방향, sequence length를 manifest에 저장한다. 오른쪽 padding의 `attention_mask=0` token은 모든 통계, 탐색 및 오차에서 제외한다. NaN/Inf는 padding에 있어도 명시적 오류다.

`eval()`, `inference_mode()`, 고정 seed, deterministic algorithms, eager attention, CUDA TF32 비활성화를 사용한다. 같은 seed/data/config에서 reservoir와 parameter 선택은 결정적이다. 다른 device/dtype/library 버전의 원본 모델 activation은 다를 수 있다. validation은 별도 replay로 고정 parameter만 평가하며 fitting하지 않는다.

## 수학과 메모리

- `X[M,K]`, `W[N,K]`, `Y=X@W.T[M,N]`. zero point는 0, INT8은 `[-127,127]`.
- `s_W[n]=max(abs(W[n,:]))/127`, `W_q=clip(rint(W/s_W[:,None]),-127,127)`.
- Pass A: 원본 입력의 전체 유효 token absmax로 **static scalar** `s_X`를 fitting한다. `--input-percentile 99.9`는 `--percentile-capacity`개의 균등 priority reservoir 원소로 percentile을 근사한다. sampling 방법/크기/관측 수를 기록한다. 같은 layer Q/K/V, gate/up은 동일 group의 Pass A 결과와 `s_X`를 공유한다.
- Pass B: 같은 데이터에서 고정 INT8 `X_q,W_q`를 사용한다. M/N/K chunk로 exact INT64 accumulator를 계산하고 INT32 범위를 검사한다. K의 모든 partial sum을 합친 다음에만 requantization한다. Gemma의 K는 `K*127² <= INT32_MAX`라 모든 prefix가 안전하다. 더 큰 synthetic K는 절대 곱 합의 보수적 bound로 모든 prefix 안전성을 검사하고, 증명할 수 없으면 거부한다. reference 함수 자체도 각 K chunk 경계에서 overflow를 검사한다.
- 출력 실수 범위는 `A*s_X*s_W[n]`의 absmax를 streaming으로 수집한다. `--search-rows`개의 **입력 row**만 균등 priority reservoir에 보관한다. 무제한 activation/accumulator를 저장하지 않는다.
- Pass C: minmax baseline은 전체 channel의 공통 absmax / 511 (전부 0이면 1). fixed는 사용자 scale, mse는 baseline과 제한된 threshold 후보를 평가한다. 모든 후보에서 profile별 multiplier/shift를 생성하고 실제 정수 reference의 dequantized INT10 MSE를 평가한다. 표본의 accumulator도 chunk별로 다시 계산한다.
- `r[n]=s_X*s_W[n]/s_10`. 한 연산에 공통 `s_10` 하나, N개 channel parameter만 사용한다. M이나 K에 따라 복제하지 않는다. 각 channel의 최대값으로 서로 다른 출력 실수 scale을 만들지 않는다.
- all-zero 입력/weight channel은 scale 1을 sentinel로 사용하고 quantized 값은 0이다. percentile이 0이지만 tensor가 nonzero이면 absmax로 fallback한다. scale 0으로 나누지 않는다.
- 최종 선택 후 전체 calibration 데이터와 optional held-out 데이터를 재실행하여 report를 만든다. 탐색 표본의 오차를 전체 데이터 오차로 표시하지 않는다.

최대 작업 메모리는 모델 외에 현재 token batch 입력, N chunk weight/INT64 copy, M×N chunk accumulator, `search_rows×K` reservoir, channel 통계 및 parameter이다. lm_head 포함해 전체 float weight 복사나 전체 M×N accumulator를 보관하지 않는다. optional weight 파일은 N chunk씩 `.npy` memmap으로 쓴다. model/산출물 SHA256은 streaming으로 계산한다.

## Hardware profile: 둘 다 unverified

현재 checkout에는 `.v`/`.sv` 또는 QuantAct 수치 계약 spec이 없다. `src/LUT.py`도 GeLU 10-bit 입력을 손글씨 설계노트 기반 추정이라고 주석 처리한다. 따라서 다음은 명시적으로 정의한 **소프트웨어 profile**이지 확정 RTL ABI가 아니다.

| 항목 | generic-rne | legacy-guess |
|---|---|---|
| multiplier | UInt16 | UInt16 (가설) |
| shift field | UInt5, effective=shift | UInt5, effective=max(shift−2,0) (가설) |
| 반올림 | nearest, ties to even | arithmetic floor (가설) |
| product | INT32×UInt16, signed 48-bit exact, truncation/wrap 없음 | 동일 가설 |
| zero point | Int8, shift/rounding 뒤 더함; export는 0 | 동일 가설 |
| saturation | zp 적용 후 signed INT10 [-512,511] | 동일 가설 |
| LUT index | 10-bit two's complement `q & 1023` | 동일 가설 |
| UInt32 | zp[7:0], shift[12:8], reserved zero[15:13], multiplier[31:16] | 동일 가설 |
| binary | little endian | little endian |

과거 layout과 shift−2 기록은 사용자 요청에서 제공된 확인되지 않은 설계 이력이다. product 폭, zp 위치, floor rounding 등 나머지 legacy 세부는 이 구현의 **추정**이다. `generic-rne`는 일반적인 `round_even(A*mult/2^shift)` reference이고 legacy와 동일시하지 않는다. 음수 shift는 금지하고 음수 **값**의 arithmetic shift는 floor다. RNE는 정수 quotient/remainder로 구현한다.

ratio마다 모든 32개 shift를 검토해 유효 UInt16 multiplier 후보를 구하고 절대 ratio 오차가 가장 작은 것을 선택한다. 범위 밖 필드를 masking하지 않는다. `status`는 `ok`, `inaccurate`, `underflow_to_zero`, `above_maximum`이며 상대 오차 tolerance 기본값은 0.001이다. 각 channel의 원하는/표현된 ratio, multiplier, shift, 상대/절대 오차를 NPZ에 저장한다. 실제 RTL 입수 후 product, shift, rounding, zp, encoding을 대조하고 test vector를 RTL로 실행하는 작업이 별도로 필요하다. 현재 코드에는 verified로 바꾸는 CLI flag가 없다.

## 기존 코드 조사 및 연결

기준 checkout: `master`, commit `6e6dd26867eb281abdd7f3feb87c83ce9771b80f`; 최초 작업 트리는 clean. root/상위 디렉터리/하위 파일 검색에서 AGENTS.md 없음.

| 기존 파일 | 확인 사항 / 새 구현 관계 |
|---|---|
| `quants.py` | Qwen1.5 GGUF, `t_matrix.so`, 16×16 tile당 float scale/zp/weights의 264B 구조. 기존 memory_map.json을 덮어쓰는 스크립트. 새 INT32→INT10 channel parameter와 별개이며 호출하지 않음. |
| `src/gemma_export.py`, `src/gemma_fusion.py` | `google/gemma-2b-it` config에서 meta model 및 dummy input으로 그래프 생성. 새 calibration은 이 경로를 재사용하지 않고 실제 checkpoint만 로드. |
| `src/compile_to_npu.py` | `build_rs2_struct`의 `quant_param_addr`와 별도 예제 compiler가 있음. 일부 예제 호출은 함수 정의와 다른 keyword를 사용. 전체 compiler 수리 범위 밖. |
| `src/compile_to_bin.py` | import 시 meta export와 출력 파일 생성. `<12Q6I` rs2 packer의 네 번째 UInt64가 quant_param_addr (rs2 내부 byte offset 24). weight 주소는 현재 dummy `0x1000000`. 새 adapter는 이 **Python packer keyword**만 반환하고 실제 ISA 검증을 주장하지 않음. |
| `src/LUT.py`, `luts/gelu_lut_1024B.bin` | signed index×0.1로 x 생성. 기존 binary와 새 임시 생성 파일이 동일함을 테스트. 출력은 `round(GeLU(x)*255)` 후 unsigned [0,255] clipping으로 signed INT8 가정과 불일치. |
| `memory_map.json` | `blk.*`, `output.weight` 등 Qwen GGUF tile offset. Gemma HF 이름 또는 새 parameter physical address로 해석하지 않음. |
| `Makefile` | 기존 `<<<<<<< HEAD` merge conflict marker가 있어 그대로는 make 실행 불가. 이 작업은 별도 venv 명령으로 실행하며 기존 conflict를 임의 해결하지 않음. |
| `Dockerfile` | NumPy/GGUF/Torch/Transformers와 `accelerator` 설치 목록. 새 독립 requirements를 제공하며 Docker 경로를 검증했다고 주장하지 않음. |

새 파일: `hardware.py` (수치/packing), `core.py` (streaming calibration), `gemma.py` (checkpoint/data/hooks), `export.py` (artifact), `compiler_adapter.py` (주소 연결), `cli.py`, `src/calibrate_gemma.py`, 두 test module, requirements 및 examples.

### Compiler adapter 사용

`compiler_mapping.json`과 manifest는 HF module/weight 이름, torch.export parameter symbol, **파일 offset**을 명시한다. underscore symbol은 편의 정보이며 식별은 export graph의 `graph_signature.inputs_to_parameters`로 원래 weight 이름을 얻어 해야 한다. attention matmul에는 이 mapping을 적용하지 않는다.

```python
# src가 Python path에 있거나 기존 compiler 파일에서 호출할 때
from static_quant.compiler_adapter import quant_param_kwargs

# graph_signature.inputs_to_parameters[resolved_weight_placeholder.name]에서 얻은 이름
weight_name = 'model.layers.0.self_attn.q_proj.weight'
# runtime_base_address는 loader/allocator가 실제로 할당한 주소. 파일 offset과 다르다.
kwargs = quant_param_kwargs(
    calibration_directory, weight_name, runtime_base_address,
    output_channel_start=0,  # 다음 QB block은 16,32,...; M/K tile마다 증가시키지 않음
    allow_unverified=True,  # 현재 profile을 실험적으로 쓸 때에만
)
rs2 = build_rs2_struct(**existing_rs2_fields, **kwargs)
```

adapter는 binary 존재, profile opt-in, 정확한 weight 이름, 64B base/offset 정렬, 유효 16-channel block, UInt64 주소 범위를 확인한다. 실제 주소를 임의로 생성하거나 firmware binary를 자동 수정하지 않는다. 기존 compiler의 graph 전체/ISA/fusion correctness 및 weight 로더는 별도 문제다. 테스트는 import side effect를 피하기 위해 `compile_to_bin.py` AST에서 실제 `build_rs2_struct` 함수만 추출하여 adapter 결과가 정확히 네 번째 UInt64에 packing되는지 검사한다. 이 연결은 Python compiler interface 수준 검증이다.

## 산출물과 report

- `manifest.json`: schema 1, model/revision/config, source commit 및 dirty 상태, dataset fingerprints, seed/버전, 각 module의 M/K/N 및 scale key, profile 상태/계약, offset/size/64B 정렬/padding/endian/LUT 조건, 산출물 SHA256.
- `scales.npz`: `op0000.s_X`, `.s_W`, `.s_10` 등. `.s_W`는 N개.
- `qparams.npz`: N개의 multiplier/shift/zero_point/packed word, 원하는/표현 ratio, 오차와 상태. Python `np.load(...,allow_pickle=False)`로 읽음.
- `qparams.bin`: 허용된 경우에만 생성. channel 순서 UInt32, 16개씩 64B. N이 16 배수가 아니면 마지막 block의 남은 word를 0으로 padding. module 시작은 64B 정렬. N개 valid count와 padding/size를 manifest에 기록. offset은 binary 차단 시에도 예정 layout으로 제공하지만 adapter가 binary 없는 연결은 거부.
- `report.json`: calibration 및 optional validation의 MSE/MAE/max absolute error, 각 channel의 동일 오차, 전체/channel별 clipping, input INT8 clipping, worst channel 목록, s_X/s_10/s_W 통계, ratio 오차/상태, 유효 token 및 overflow 여부. sample 수는 manifest의 각 dataset에 기록.
- `compiler_mapping.json`: 원래 weight 이름 → parameter file offset. runtime base는 null.
- `test_vectors.json`: 작은 실제 관측 accumulator 입력, UInt32 parameter, software expected signed INT10 및 LUT index. 아직 RTL 검증 결과가 아님.
- 선택적 `opNNNN.weights.int8.npy`: row-major [N,K] INT8. 기존 NPUTile/GGUF 파일 포맷이 아님.

세 가지 오차는 분리한다:

1. requantization: `Q_hw*s_10` vs `A*s_X*s_W[n]`.
2. local_total: `Q_hw*s_10` vs 원래 `X@W.T` (관측된 입력/로드된 weight로 FP64 float reference).
3. parameter_approximation: `Q_hw` vs `clip(rint(A*r[n]),-512,511)` (INT10 code 단위).

INT10 clipping은 **rounding 후 saturation 직전** 값이 [-512,511] 밖인 비율이다. hardware와 ideal clipping을 각각 보고한다. 최악 channel은 local_total channel MSE 내림차순이다. overflow나 NaN/Inf가 있으면 실행이 실패하므로 실패를 정상 report로 기록하지 않는다.

## 검증 상태

실행 결과는 `docs/static-calibration-validation.md`에 기록한다. 필수 테스트는 외부 다운로드 없는 NumPy/unittest 경로다. optional hook 통합 테스트는 설치된 Torch/Transformers에서 **작은 명시적 synthetic Gemma fixture**만 만들고 pretrained parameter를 생성하지 않는다. 이는 실제 Gemma 2B checkpoint calibration 또는 RTL bit-exact 검증을 대체하지 않는다.
