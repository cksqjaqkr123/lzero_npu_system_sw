# 실행한 검증과 제한

검증일: 2026-09-13. 기반 branch `master`, commit `6e6dd26867eb281abdd7f3feb87c83ce9771b80f`.

## 테스트 결과

| 환경 / 검사 | 결과 |
|---|---|
| macOS arm64, Python 3.9.6, NumPy 2.0.2, Torch/Transformers 없는 기본 환경 | NumPy 기반 22개 통과, optional hook 통합 3개 skip |
| `/tmp/gemma-calibration-test-venv`, Python 3.9.6, NumPy 1.26.4, Torch 2.2.2, Transformers 4.44.2 | **25개 전부 통과**, 마지막 실행 0.152초 (import 시간 제외) |
| synthetic CLI, generic-rne + MSE + explicit unverified binary + optional weights | 성공; 아래 workspace 산출물 생성 |
| synthetic CLI, legacy-guess + fixed s10=0.1 + fixed LUT + explicit unverified binary | 성공; `/tmp/gemma-legacy-fixed-lut-review`에 생성 |
| `compileall` (bytecode cache를 `/tmp`로 지정) | 통과 |
| `git diff --check` | 통과 |
| CLI `--help` | 통과 |
| 기존 GeLU LUT 재생성과 저장소 binary byte 비교 | 동일 |

전체 테스트 재실행:

```bash
HF_HOME=/tmp/gemma-calibration-hf-cache \
  /tmp/gemma-calibration-test-venv/bin/python -m unittest discover -s tests -v
```

테스트 설치는 임시 venv에만 진행했다. 기본 프로젝트 Python 환경에는 Torch/Transformers를 설치하지 않았다. 새로운 환경에서는 `requirements-calibration.txt`로 설치 후 같은 unittest 명령을 사용한다. 검사한 정확한 버전은 위 표와 같으며, requirements가 허용하는 모든 버전을 검사한 것은 아니다.

필수 사례는 channel scale 축, M row 재사용, K chunk별 누산과 독립 Python integer oracle 비교, all-zero, 음수 RNE/floor와 saturation, INT32 overflow/48-bit product, ratio 표현 범위/실패 상태, UInt32 pack/unpack/endian, 64B 정렬 및 N=19 padding, padding 통계 제외/NaN 오류, LUT scale mismatch, 동일 seed 재현, 전체 세 종류 오차와 dense reference 비교, held-out scale 불변성, export→**binary/NPZ reload**→출력 비교, compiler adapter의 기존 packer 필드 연결이다.

optional 3개 테스트는 설치된 Hugging Face Gemma 구현의 **아주 작은 deterministic synthetic fixture**를 사용한다. pretrained checkpoint를 사용하거나 실제 Gemma 2B parameter를 생성한 테스트가 아니다. 원본 Q/K/V hook 입력 일치, padding 제외, q_proj/down_proj/lm_head calibration와 held-out 평가, 예외 시 hook 해제를 검사했다.

## 검토 가능한 synthetic 산출물

`calibration_outputs/synthetic-review/` (gitignore 대상, 기존 산출물과 분리):

```bash
python3 src/calibrate_gemma.py --synthetic \
  --output-dir calibration_outputs/synthetic-review \
  --scale-mode mse --allow-unverified-export --save-int8-weights
```

이미 실행한 디렉터리는 덮어쓰기 방지로 재사용할 수 없다. 재실행 시 다른 새 이름을 지정한다. manifest 모델명은 `SYNTHETIC FIXTURE, NOT GEMMA`이며, 3 samples / 18 valid tokens / K=19 / N=21, N 방향 11개 padding, binary 128 bytes이다. hardware verification은 `unverified`, `rtl_bit_exact=false`이다.

## 실행하지 못한 검증과 이유

- **실제 Gemma 2B calibration 미실행.** 저장소 `models/`에는 README만 있고 기본 HF cache에서 해당 checkpoint/config를 찾지 못했다. 실제 CLI를 `--local-files-only --samples 1 --sequence-length 8`로 실행해 config 단계의 `LocalEntryNotFoundError`/`OSError`를 확인했다. HF 접근 권한을 확인하거나 수 GB의 checkpoint를 다운로드하지 않았으므로 접근이 거부되었다고 단정하지 않는다. 실제 pretrained parameter는 생성하지 않았다.
- **GPU calibration 미실행.** 검사 환경에서 `torch.cuda.is_available()`와 `torch.backends.mps.is_available()` 모두 False였다. CUDA/bfloat16 경로의 실제 실행 성능/정확도는 검증하지 않았다.
- **RTL bit-exact 미검증.** checkout/제공 파일에 QuantAct RTL 또는 확정 수치 spec이 없다. software 기대값만 제공했고 두 profile은 unverified다. 기본 binary export 차단을 테스트했다.
- **전체 양자화 모델 simulation/perplexity/decode 미실행.** 이 구현은 원본 activation을 관측하는 local prefill calibration에 한정한다. attention QKᵀ/PV도 제외한다.
- **기존 compiler 전체 실행/배포 미검증.** adapter와 기존 `build_rs2_struct` 연결만 검사했다. dummy weight 주소, fusion/ISA 등 기존 문제를 고치거나 실제 NPU physical address를 생성하지 않았다.
- **기존 Makefile/Docker 빌드 미실행.** Makefile의 기존 merge conflict marker를 확인했고 기존 파일은 수정하지 않았다.

초기 기본 sandbox의 pip 설치는 package를 찾지 못했고, 임시 venv 대상 승인된 네트워크 설치로 해결했다. 시스템 Python의 기본 bytecode cache는 sandbox 밖이라 compileall이 처음 실패했지만 `/tmp` cache로 재실행해 통과했다. HF cache에도 기본 경로 쓰기 경고가 발생했으며 통합 테스트는 writable `HF_HOME=/tmp/gemma-calibration-hf-cache`를 사용했다. 이것들은 소프트웨어 수치 실패나 RTL 검증 결과가 아니다.
