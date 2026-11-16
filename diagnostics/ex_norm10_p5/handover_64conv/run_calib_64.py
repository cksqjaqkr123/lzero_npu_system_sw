# 기존 run_sout_sweep / run_gelu_local 같은 calib 스크립트를 32 calib으로 돌리는 용도
# 핵심: calib 32개로 s_in/s_out/p9999 다시 계산 → calib_64_scales.json 저장
# 실제 calib 코드는 기존 파일 재사용, 여기서는 conv_list_64.json의 calib만 읽는다는 것만 명시
print("TODO: 기존 calib 스크립트에 conv_list_64.json calib 32개 연결")
