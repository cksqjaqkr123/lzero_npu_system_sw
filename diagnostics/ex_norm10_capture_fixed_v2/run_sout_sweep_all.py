#!/usr/bin/env python3
"""
53개 Norm 전수 s_out sweep.
- s_in은 oracle_v2.csv에서 best로 고정 (p999 44개, p9999 9개)
- s_out 후보: pooled p999 / p9999 / absmax (히스토그램으로 추정, y 전체 저장 안 함)
- 출력: sout_sweep_all.csv + 콘솔 표
"""
import csv, random
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations

EPS = 1e-6
# 히스토그램: 0~200을 0.1 단위 2000 bins. y max가 80 정도라서 충분.
BINS = np.linspace(0, 200, 2001)
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2

def pct_from_hist(counts, pct):
    total = counts.sum()
    if total == 0: return 0.0
    csum = np.cumsum(counts)
    idx = np.searchsorted(csum, total * pct / 100.0)
    idx = min(idx, len(BIN_CENTERS)-1)
    return float(BIN_CENTERS[idx])

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--input_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    ap.add_argument("--output_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    ap.add_argument("--num_conversations", type=int, default=8)
    args = ap.parse_args()
    in_dir = Path(args.input_dir); out_dir = Path(args.output_dir)

    # 1) oracle에서 Norm별 best s_in 찾기
    best_sin = {}
    rows = list(csv.DictReader(open(in_dir/"oracle_v2.csv")))
    by_norm = defaultdict(list)
    for r in rows: by_norm[(r["layer_name"], r["norm_type"])].append(r)
    for k, lst in by_norm.items():
        best = min(lst, key=lambda x: float(x["nmse"]))
        best_sin[k] = best["s_in_candidate"]  # s_in_p999 / s_in_p9999 / s_in_absmax
    print(f"[*] best s_in 로드: {len(best_sin)} Norm")

    # scale_candidates에서 s_in 값 로드
    cand = {}
    for r in csv.DictReader(open(in_dir/"scale_candidates_v2.csv")):
        cand[(r["layer_name"], r["norm_type"])] = {
            "s_in_p999": float(r["s_in_p999"]),
            "s_in_p9999": float(r["s_in_p9999"]),
            "s_in_absmax": float(r["s_in_absmax"]),
        }

    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    tok = GemmaTokenizer()
    convs = build_conversations(args.num_conversations,
        prefill_lengths=[32,128,512,1024], decode_lengths=[16,24,32])
    def gguf_name(ln,nt):
        return "output_norm.weight" if ln=="output_norm" else f"blk.{ln.split('.')[1]}.{nt}.weight"
    gamma_cache = {k: model._load_norm(gguf_name(k[0],k[1])).float().cpu().numpy().astype(np.float32) for k in best_sin}

    # 2) 1차 패스: |y| 히스토그램 + absmax
    hist = {k: np.zeros(len(BIN_CENTERS), dtype=np.int64) for k in best_sin}
    absmax = {k: 0.0 for k in best_sin}
    rng = random.Random(42)
    for ci, conv in enumerate(convs):
        ids = tok.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        if not ids: continue
        full = ids + [rng.randint(0, model.cfg.vocab_size-1) for _ in range(conv["decode_len"])]
        caps = model.forward_capture(torch.tensor([full], dtype=torch.long))
        for ln,nt,arr in caps:
            k=(ln,nt)
            if k not in best_sin: continue
            g = gamma_cache[k]
            for pos in range(arr.shape[0]):
                if pos>=len(ids) or pos==0: continue
                x = arr[pos].astype(np.float32)
                rms = np.sqrt(np.mean(x.astype(np.float64)**2)+EPS)
                y = np.abs(x/rms*g).astype(np.float64)
                # 히스토그램 업데이트
                idx = np.clip((y/0.1).astype(np.int64), 0, len(BIN_CENTERS)-1)
                np.add.at(hist[k], idx, 1)
                m = float(y.max())
                if m > absmax[k]: absmax[k]=m
        print(f"[*] hist pass conv {ci} done")
        del caps
        import gc; gc.collect()

    sout_cands = {}
    for k in best_sin:
        p999 = pct_from_hist(hist[k], 99.9)
        p9999 = pct_from_hist(hist[k], 99.99)
        sout_cands[k] = {"p999": p999/511, "p9999": p9999/511, "absmax": absmax[k]/511}
        print(f"{k[0]:12s} {k[1]:9s} s_out p999={sout_cands[k]['p999']:.5f} p9999={sout_cands[k]['p9999']:.5f} absmax={sout_cands[k]['absmax']:.5f}")

    # 3) 2차 패스: 각 s_out 후보로 NMSE
    acc = defaultdict(lambda: {"err2":0.0,"y2":0.0})
    for ci, conv in enumerate(convs):
        ids = tok.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        if not ids: continue
        full = ids + [rng.randint(0, model.cfg.vocab_size-1) for _ in range(conv["decode_len"])]
        caps = model.forward_capture(torch.tensor([full], dtype=torch.long))
        for ln,nt,arr in caps:
            k=(ln,nt)
            if k not in best_sin: continue
            g = gamma_cache[k]; s_in = cand[k][best_sin[k]]
            for pos in range(arr.shape[0]):
                if pos>=len(ids) or pos==0: continue
                x = arr[pos].astype(np.float32)
                rms_fp = np.sqrt(np.mean(x.astype(np.float64)**2)+EPS)
                y_fp = (x/rms_fp*g).astype(np.float64)
                q = np.clip(np.round(x/s_in),-127,127); x_hat=q*s_in
                rms_q = np.sqrt(np.mean(x_hat.astype(np.float64)**2)+EPS)
                y_hat = (x_hat/rms_q*g).astype(np.float64)
                for cname, s_out in sout_cands[k].items():
                    q_out = np.clip(np.round(y_hat/s_out),-512,511); y_out=q_out*s_out
                    err = (y_out-y_fp)**2
                    key=(k[0],k[1],cname)
                    acc[key]["err2"]+=err.sum(); acc[key]["y2"]+=(y_fp**2).sum()
        print(f"[*] nmse pass conv {ci} done")
        del caps
        import gc; gc.collect()

    # 4) 저장 + 표
    out_rows = []
    for k in sorted(best_sin):
        best = None
        for cname in ["p999","p9999","absmax"]:
            a=acc[(k[0],k[1],cname)]
            nmse = a["err2"]/a["y2"] if a["y2"]>0 else 0
            out_rows.append({"layer_name":k[0],"norm_type":k[1],"s_out_candidate":cname,
                "s_in_used":best_sin[k],"s_in":cand[k][best_sin[k]],"s_out":sout_cands[k][cname],"nmse":nmse})
            if best is None or nmse < best[1]: best=(cname,nmse)
        print(f"  {k[0]:12s} {k[1]:9s} best_s_out={best[0]:6s} NMSE={best[1]:.5f}  "
              f"(p999={acc[(k[0],k[1],'p999')]['err2']/acc[(k[0],k[1],'p999')]['y2']:.5f} "
              f"p9999={acc[(k[0],k[1],'p9999')]['err2']/acc[(k[0],k[1],'p9999')]['y2']:.5f})")
    with open(out_dir/"sout_sweep_all.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    print(f"[✓] sout_sweep_all.csv 저장 ({len(out_rows)} 행)")

if __name__=="__main__":
    main()