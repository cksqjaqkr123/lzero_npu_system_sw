# Edge LLM NPU software

Original Gemma 2B용 정적 Linear INT8 × INT8 → INT32 → INT10 calibration은
[실행 가이드](docs/static-calibration.md)와 [검증 결과](docs/static-calibration-validation.md)를 참고하세요.

```bash
python3 -m unittest discover -s tests -v
python3 src/calibrate_gemma.py --synthetic \
  --output-dir calibration_outputs/synthetic --scale-mode mse
```

Synthetic 경로는 NumPy만 필요합니다. 실제 checkpoint용 의존성은 `requirements-calibration.txt`에 있습니다.
현재 hardware profile은 unverified이므로 `qparams.bin`은 기본적으로 생성하지 않습니다.
