#!/usr/bin/env python3
"""
Oracle: s_in 후보 3개 x s_out 1개 (gamma 포함 y_p999 기반)를 한 번의 forward로 평가.
- 입력: scale_candidates_v2.csv에서 s_in_p999 / s_in_p9999 / s_in_absmax, s_out_p999
- 출력: oracle_v2.csv (Norm별, 후보별 NMSE, cosine 근사, clipping율)
- dead 채널은 NMSE에서 0으로 나오지만, RMS에는 포함 (포워드와 동일)
"""
import argparse, csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch, random

from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations

EPS = 1e-6

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--input_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    ap.add_argument("--num_conversations", type=int, default=8)
    args = ap.parse_args()

    in_dir = Path(args.input_dir); out_dir = Path(args.output_dir)
    # scale 후보 로드
    cand = {}
    with open(in_dir / "scale_candidates_v2.csv") as f:
        for r in csv.DictReader(f):
            k = (r["layer_name"], r["norm_type"])
            cand[k] = {
                "s_in_p999": float(r["s_in_p999"]),
                "s_in_p9999": float(r["s_in_p9999"]),
                "s_in_absmax": float(r["s_in_absmax"]),
                "s_out": float(r["s_out_p999"]),
            }
    print(f"[*] loaded {len(cand)} Norm scale candidates")

    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    tokenizer = GemmaTokenizer()
    convs = build_conversations(args.num_conversations,
        prefill_lengths=[32, 128, 512, 1024], decode_lengths=[16, 24, 32])

    # gamma 캐시
    def gguf_name(ln, nt):
        if ln == "output_norm": return "output_norm.weight"
        return f"blk.{ln.split('.')[1]}.{nt}.weight"
    gamma_cache = {}
    for (ln, nt) in cand:
        w = model._load_norm(gguf_name(ln, nt)).float().cpu().numpy().astype(np.float32)
        gamma_cache[(ln, nt)] = w  # F32 그대로 = gamma, 1+ 안 함

    # 누적: Norm별, 후보별 sum_err2, sum_y2, clip_count, total
    acc = defaultdict(lambda: {"err2": 0.0, "y2": 0.0, "clip_in": 0, "clip_out": 0, "n": 0})
    rng = random.Random(42)

    for ci, conv in enumerate(convs):
        prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        if not prompt_ids: continue
        prefill_len = len(prompt_ids)
        decode_ids = [rng.randint(0, model.cfg.vocab_size - 1) for _ in range(conv["decode_len"])]
        full_ids = prompt_ids + decode_ids
        captures = model.forward_capture(torch.tensor([full_ids], dtype=torch.long))

        for layer_name, norm_type, arr in captures:
            gamma = gamma_cache[(layer_name, norm_type)]
            c = cand[(layer_name, norm_type)]
            S = arr.shape[0]
            for pos in range(S):
                if pos >= prefill_len: continue  # prefill만, decode 제외
                if pos == 0: continue  # BOS 제외
                x = arr[pos].astype(np.float32)
                rms_fp = np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS)
                y_fp = (x / rms_fp * gamma).astype(np.float64)
                y2 = float(np.sum(y_fp ** 2))

                for cname in ["s_in_p999", "s_in_p9999", "s_in_absmax"]:
                    s_in = c[cname]; s_out = c["s_out"]
                    # 1) 입력 INT8 양자화
                    q = np.clip(np.round(x / s_in), -127, 127)
                    clip_in = int(np.sum(np.abs(x / s_in) > 127.5))
                    x_hat = q * s_in
                    # 2) 양자화된 입력으로 RMS + gamma
                    rms_q = np.sqrt(np.mean(x_hat.astype(np.float64) ** 2) + EPS)
                    y_hat = (x_hat / rms_q * gamma).astype(np.float64)
                    # 3) 출력 INT10 양자화
                    q_out = np.clip(np.round(y_hat / s_out), -512, 511)
                    clip_out = int(np.sum(np.abs(y_hat / s_out) > 511.5))
                    y_out = q_out * s_out
                    err2 = float(np.sum((y_out - y_fp) ** 2))
                    k = (layer_name, norm_type, cname)
                    acc[k]["err2"] += err2; acc[k]["y2"] += y2
                    acc[k]["clip_in"] += clip_in; acc[k]["clip_out"] += clip_out
                    acc[k]["n"] += x.size
        print(f"[*] conv {ci} done")
        del captures
        import gc; gc.collect()

    rows = []
    for (ln, nt, cname), a in sorted(acc.items()):
        nmse = a["err2"] / a["y2"] if a["y2"] > 0 else 0.0
        rows.append({
            "layer_name": ln, "norm_type": nt, "s_in_candidate": cname,
            "s_in": cand[(ln, nt)][cname], "s_out": cand[(ln, nt)]["s_out"],
            "nmse": nmse, "clip_in_rate": a["clip_in"] / a["n"],
            "clip_out_rate": a["clip_out"] / a["n"], "n_elements": a["n"],
        })
        # 요약 로그: 같은 Norm의 3개 후보를 한 줄로 비교하기 위해 나중에 표로
    with open(out_dir / "oracle_v2.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    print(f"[✓] oracle_v2.csv 저장 ({len(rows)} 행)")

    # 보기 좋게: Norm별 best 후보만 출력
    by_norm = defaultdict(list)
    for r in rows: by_norm[(r["layer_name"], r["norm_type"])].append(r)
    print("\n=== Best s_in per Norm (NMSE 최소) ===")
    for k in sorted(by_norm):
        best = min(by_norm[k], key=lambda x: x["nmse"])
        print(f"  {k[0]:12s} {k[1]:9s} best={best['s_in_candidate']:12s} "
              f"nmse={best['nmse']:.6f} clip_in={best['clip_in_rate']:.4f} "
              f"clip_out={best['clip_out_rate']:.4f}")

if __name__ == "__main__":
    main()