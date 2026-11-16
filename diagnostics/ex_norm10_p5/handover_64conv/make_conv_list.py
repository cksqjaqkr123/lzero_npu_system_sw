# build_conversations(n_conv=64)로 64개 만들고, 앞 32개는 calib, 뒤 32개는 valid로 나눔
# 프롬프트 겹치면 안 됨 — conv_list_64.json에 저장
import json
from pathlib import Path
import sys
sys.path.insert(0, "diagnostics/ex_norm10_capture_fixed")
from cap_streaming import build_conversations

convs = build_conversations(64, prefill_lengths=[256]*64)
# convs가 dict 리스트라고 가정 — prompt_text만 저장
out = {
    "calib": convs[:32],
    "valid": convs[32:],
}
Path("diagnostics/ex_norm10_p5/handover_64conv/conv_list_64.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"calib 32, valid 32, total 64 saved")
