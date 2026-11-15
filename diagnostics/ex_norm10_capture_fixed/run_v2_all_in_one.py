#!/usr/bin/env python3
"""
V2 all-in-one: gamma + s_in + channel + s_out 를 한 번의 forward로.
- prefill만 캘리브레이션에 사용, decode는 기록만 하고 제외 (decode는 랜덤 토큰)
- 대화별 첫 토큰(BOS)은 is_bos=1로 표시하고 캘리브레이션 집계에서 제외
- 토큰마다 x와 y = x / RMS(x) * gamma 를 같이 저장
"""
import argparse, csv, gc, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

# 같은 폴더의 cap_streaming.py에서 재사용 (RoPE 패치된 버전이어야 함)
from cap_streaming import (
    StreamingGemma, GemmaTokenizer, build_conversations,
    summarize_1d, cap_scale,
)

EPS = 1e-6
HIDDEN = 2304

def gguf_name(layer_name, norm_type):
    # blk.i + attn_norm -> blk.i.attn_norm.weight
    # output_norm + final_norm -> output_norm.weight
    if layer_name == "output_norm":
        return "output_norm.weight"
    idx = layer_name.split(".")[1]
    return f"blk.{idx}.{norm_type}.weight"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    ap.add_argument("--num_conversations", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    model = StreamingGemma(args.gguf, dtype=dtype, device=args.device)
    tokenizer = GemmaTokenizer()
    convs = build_conversations(
        args.num_conversations,
        prefill_lengths=[32, 128, 512, 1024],
        decode_lengths=[16, 24, 32],
    )
    print(f"[*] {len(convs)} conversations")

    # ── Phase 0: gamma 감사 (forward 없이) ──
    # 모든 Norm의 gamma를 한 번에 미리 로드: 53 * 2304 floats, 약 0.5MB
    gamma_cache = {}
    gamma_rows = []
    print("\n=== Phase 0: gamma audit ===")
    # Norm 키 목록: blk.0~25 attn/ffn + output_norm
    norm_keys = []
    for i in range(model.cfg.num_layers):
        norm_keys.append((f"blk.{i}", "attn_norm"))
        norm_keys.append((f"blk.{i}", "ffn_norm"))
    norm_keys.append(("output_norm", "final_norm"))

    for layer_name, norm_type in norm_keys:
        name = gguf_name(layer_name, norm_type)
        w = model._load_norm(name).detach().float().cpu().numpy().astype(np.float32)
        raw_mean = float(np.mean(w))
        # GGUF에 gamma 그대로 저장됐으면 mean ~1, offset이면 mean ~0
        if abs(raw_mean) < 0.5:
            gamma = 1.0 + w
            convention = "1+weight"
        else:
            gamma = w
            convention = "weight-as-gamma"
        gamma_cache[(layer_name, norm_type)] = gamma
        gamma_rows.append({
            "layer_name": layer_name, "norm_type": norm_type,
            "gguf_name": name, "raw_mean": raw_mean,
            "raw_min": float(np.min(w)), "raw_max": float(np.max(w)),
            "raw_std": float(np.std(w)),
            "gamma_mean": float(np.mean(gamma)),
            "gamma_min": float(np.min(gamma)), "gamma_max": float(np.max(gamma)),
            "gamma_std": float(np.std(gamma)),
            "convention": convention,
            "gamma_top3_idx": ",".join(map(str, np.argsort(gamma)[-3:][::-1].tolist())),
            "gamma_top3_val": ",".join(f"{v:.4f}" for v in np.sort(gamma)[-3:][::-1]),
            "gamma_bottom3_idx": ",".join(map(str, np.argsort(gamma)[:3].tolist())),
        })
        print(f"  {layer_name:12s} {norm_type:9s} raw_mean={raw_mean:+.4f} "
              f"-> {convention}, gamma [{np.min(gamma):.3f}, {np.max(gamma):.3f}]")
        del w
        gc.collect()

    # convention이 섞여 있으면 여기서 멈춤. 전부 동일해야 함.
    convs_set = set(r["convention"] for r in gamma_rows)
    if len(convs_set) != 1:
        print(f"[!] WARNING: gamma convention이 섞여 있음: {convs_set}. 결과를 믿지 마.")
    with open(out_dir / "gamma_audit.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(gamma_rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(gamma_rows)
    print(f"[✓] gamma_audit.csv 저장")

    # ── Phase 1: 통합 캡처 (x + y + 채널 누적) ──
    # 채널 누적: Norm별, prefill + BOS 제외만. decode는 누적하지 않음.
    ch = {}
    for k in norm_keys:
        ch[k] = {
            "count": 0,
            "max_abs_x": np.zeros(HIDDEN, dtype=np.float32),
            "sum_abs_x": np.zeros(HIDDEN, dtype=np.float32),
            "sumsq_x": np.zeros(HIDDEN, dtype=np.float32),
            "max_abs_y": np.zeros(HIDDEN, dtype=np.float32),
            "sum_abs_y": np.zeros(HIDDEN, dtype=np.float32),
        }

    jsonl_path = out_dir / "norm_v2_per_token.jsonl"
    import random
    rng = random.Random(42)

    with open(jsonl_path, "w") as jf:
        for ci, conv in enumerate(convs):
            prompt_ids = tokenizer.encode(conv["prompt_text"], max_length=conv["prefill_len"])
            if not prompt_ids:
                continue
            prefill_len = len(prompt_ids)
            decode_ids = [rng.randint(0, model.cfg.vocab_size - 1) for _ in range(conv["decode_len"])]
            full_ids = prompt_ids + decode_ids
            input_tensor = torch.tensor([full_ids], dtype=torch.long)
            captures = model.forward_capture(input_tensor)

            for layer_name, norm_type, arr in captures:
                gamma = gamma_cache[(layer_name, norm_type)]
                S = arr.shape[0]
                for pos in range(S):
                    group = "prefill" if pos < prefill_len else "decode"
                    is_bos = 1 if pos == 0 else 0
                    x = arr[pos].astype(np.float32)
                    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS)
                    y = (x / rms * gamma).astype(np.float32)
                    sx = summarize_1d(x)
                    sy = summarize_1d(y)
                    jf.write(json.dumps({
                        "conv": ci, "layer_name": layer_name, "norm_type": norm_type,
                        "group": group, "pos": pos, "is_bos": is_bos,
                        "x_absmax": sx["absmax"], "x_p99": sx["p99"],
                        "x_p999": sx["p999"], "x_p9999": sx["p9999"],
                        "x_mean": sx["mean"], "x_std": sx["std"],
                        "y_absmax": sy["absmax"], "y_p99": sy["p99"],
                        "y_p999": sy["p999"], "y_p9999": sy["p9999"],
                        "y_mean": sy["mean"], "y_std": sy["std"],
                    }) + "\n")

                    # 채널 누적은 prefill + BOS 제외만
                    if group == "prefill" and is_bos == 0:
                        c = ch[(layer_name, norm_type)]
                        ax = np.abs(x); ay = np.abs(y)
                        np.maximum(c["max_abs_x"], ax, out=c["max_abs_x"])
                        np.maximum(c["max_abs_y"], ay, out=c["max_abs_y"])
                        c["sum_abs_x"] += ax; c["sumsq_x"] += x * x
                        c["sum_abs_y"] += ay
                        c["count"] += 1
            del captures, arr
            gc.collect()
            print(f"[*] conv {ci} done")
    print(f"[✓] norm_v2_per_token.jsonl 저장")

    # ── Phase 2: 집계 → channel_stats.csv + scale_candidates_v2.csv ──
    # JSONL을 다시 읽어서 prefill + BOS 제외로만 집계
    agg = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r["group"] != "prefill" or r["is_bos"] == 1:
                continue
            agg[(r["layer_name"], r["norm_type"])].append(r)

    scale_rows, channel_rows = [], []
    for (layer_name, norm_type), rs in sorted(agg.items()):
        gamma = gamma_cache[(layer_name, norm_type)]
        # x 기준 scale
        max_x_absmax = max(r["x_absmax"] for r in rs)
        max_x_p999 = max(r["x_p999"] for r in rs)
        max_x_p9999 = max(r["x_p9999"] for r in rs)
        mean_x_std = sum(r["x_std"] for r in rs) / len(rs)
        # y 기준 scale (gamma 포함) — s_out은 여기서만 유도
        max_y_absmax = max(r["y_absmax"] for r in rs)
        max_y_p999 = max(r["y_p999"] for r in rs)
        max_y_p9999 = max(r["y_p9999"] for r in rs)
        mean_y_std = sum(r["y_std"] for r in rs) / len(rs)

        scale_rows.append({
            "layer_name": layer_name, "norm_type": norm_type, "n_tokens": len(rs),
            "x_absmax": max_x_absmax, "x_p999": max_x_p999, "x_p9999": max_x_p9999,
            "x_std_mean": mean_x_std,
            "s_in_p999": max_x_p999 / 127.0,
            "s_in_p9999": max_x_p9999 / 127.0,
            "s_in_absmax": max_x_absmax / 127.0,
            "x_outlier_ratio": max_x_p999 / max_x_absmax if max_x_absmax > 0 else 0.0,
            "y_absmax": max_y_absmax, "y_p999": max_y_p999, "y_p9999": max_y_p9999,
            "y_std_mean": mean_y_std,
            "s_out_p999": cap_scale(max_y_p999 / 511.0),
            "s_out_p9999": cap_scale(max_y_p9999 / 511.0),
            "s_out_absmax": cap_scale(max_y_absmax / 511.0),
            "y_outlier_ratio": max_y_p999 / max_y_absmax if max_y_absmax > 0 else 0.0,
        })

        # 채널: max|x| 기준 top10 + gamma top10 + 겹침
        c = ch[(layer_name, norm_type)]
        cnt = max(c["count"], 1)
        mean_abs_x = c["sum_abs_x"] / cnt
        mean_abs_y = c["sum_abs_y"] / cnt
        top_x_idx = np.argsort(c["max_abs_x"])[-10:][::-1]
        top_gamma_idx = np.argsort(gamma)[-10:][::-1]
        overlap = sorted(set(top_x_idx.tolist()) & set(top_gamma_idx.tolist()))
        for rank, ch_idx in enumerate(top_x_idx):
            ch_idx = int(ch_idx)
            channel_rows.append({
                "layer_name": layer_name, "norm_type": norm_type,
                "rank_by_max_x": rank,
                "channel": ch_idx,
                "max_abs_x": float(c["max_abs_x"][ch_idx]),
                "mean_abs_x": float(mean_abs_x[ch_idx]),
                "rms_x": float(np.sqrt(c["sumsq_x"][ch_idx] / cnt)),
                "max_abs_y": float(c["max_abs_y"][ch_idx]),
                "mean_abs_y": float(mean_abs_y[ch_idx]),
                "gamma": float(gamma[ch_idx]),
                "in_gamma_top10": 1 if ch_idx in top_gamma_idx else 0,
            })
        # 요약 한 줄을 로그로
        print(f"  {layer_name:12s} {norm_type:9s} n={len(rs)} "
              f"x_p999={max_x_p999:.3f} -> s_in={max_x_p999/127:.5f} | "
              f"y_p999={max_y_p999:.3f} -> s_out={cap_scale(max_y_p999/511.0):.5f} | "
              f"ch_overlap_gamma={overlap}")

    with open(out_dir / "scale_candidates_v2.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(scale_rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(scale_rows)
    with open(out_dir / "channel_top10.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(channel_rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(channel_rows)
    print(f"[✓] scale_candidates_v2.csv ({len(scale_rows)} Norm)")
    print(f"[✓] channel_top10.csv ({len(channel_rows)} rows)")
    print(f"[✓] Done. output_dir={out_dir}")

if __name__ == "__main__":
    main()