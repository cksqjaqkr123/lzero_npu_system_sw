#!/usr/bin/env python3
"""
Worst Norm들에 대해 s_out 후보를 sweep하고, 슈퍼 채널을 뺐을 때 NMSE도 같이 본다.
- s_out 후보: p999(y)/511, p9999(y)/511, absmax(y)/511, + 2x p999
- 슈퍼 채널: channel_error.csv에서 err_share 상위 채널
"""
import csv, random
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
from cap_streaming import StreamingGemma, GemmaTokenizer, build_conversations

EPS = 1e-6
TARGETS = {
    ("blk.0","attn_norm"): ["1393"],
    ("blk.1","attn_norm"): ["114","1683"],
    ("blk.5","attn_norm"): ["1002"],
    ("blk.5","ffn_norm"): ["1002"],
    ("blk.2","attn_norm"): ["1002"],
}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/parkchanbum/models/gemma-2b-q8_0.gguf")
    ap.add_argument("--input_dir", default="diagnostics/ex_norm10_capture_fixed_v2")
    ap.add_argument("--num_conversations", type=int, default=8)
    args = ap.parse_args()
    in_dir = Path(args.input_dir)

    # scale_candidates에서 s_out 후보 + s_in best 로드
    cand = {}
    with open(in_dir/"scale_candidates_v2.csv") as f:
        for r in csv.DictReader(f):
            cand[(r["layer_name"], r["norm_type"])] = r
    best_sin = {("blk.0","attn_norm"):"s_in_p999", ("blk.1","attn_norm"):"s_in_p9999",
                ("blk.5","attn_norm"):"s_in_p999", ("blk.5","ffn_norm"):"s_in_p999",
                ("blk.2","attn_norm"):"s_in_p9999"}

    model = StreamingGemma(args.gguf, dtype=torch.float16, device="cpu")
    tok = GemmaTokenizer()
    convs = build_conversations(args.num_conversations, prefill_lengths=[32,128,512,1024], decode_lengths=[16,24,32])
    def gguf_name(ln,nt):
        return "output_norm.weight" if ln=="output_norm" else f"blk.{ln.split('.')[1]}.{nt}.weight"
    gamma_cache = {k: model._load_norm(gguf_name(k[0],k[1])).float().cpu().numpy().astype(np.float32) for k in TARGETS}

    # 1차: y 분포에서 s_out 후보 계산용 통계 수집 (p999, p9999, absmax)
    y_all = {k: [] for k in TARGETS}
    rng = random.Random(42)
    for conv in convs:
        ids = tok.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        if not ids: continue
        full = ids + [rng.randint(0, model.cfg.vocab_size-1) for _ in range(conv["decode_len"])]
        caps = model.forward_capture(torch.tensor([full], dtype=torch.long))
        for ln,nt,arr in caps:
            k=(ln,nt)
            if k not in TARGETS: continue
            g = gamma_cache[k]
            for pos in range(arr.shape[0]):
                if pos>=len(ids) or pos==0: continue
                x = arr[pos].astype(np.float32)
                rms = np.sqrt(np.mean(x.astype(np.float64)**2)+EPS)
                y = np.abs(x/rms*g)
                y_all[k].append(y)
        del caps
        import gc; gc.collect()
    sout_cands = {}
    for k, lst in y_all.items():
        y = np.concatenate(lst)
        sout_cands[k] = {
            "p999": np.percentile(y,99.9)/511,
            "p9999": np.percentile(y,99.99)/511,
            "absmax": np.max(y)/511,
            "p999_x2": np.percentile(y,99.9)/511*2,
        }
        print(f"{k} s_out: p999={sout_cands[k]['p999']:.5f} p9999={sout_cands[k]['p9999']:.5f} absmax={sout_cands[k]['absmax']:.5f} p999x2={sout_cands[k]['p999_x2']:.5f}")

    # 2차: 각 s_out 후보로 NMSE + 슈퍼 채널 제외 NMSE 계산
    acc = defaultdict(lambda: {"err2":0.0,"y2":0.0,"err2_no_super":0.0,"y2_no_super":0.0})
    for conv in convs:
        ids = tok.encode(conv["prompt_text"], max_length=conv["prefill_len"])
        if not ids: continue
        full = ids + [rng.randint(0, model.cfg.vocab_size-1) for _ in range(conv["decode_len"])]
        caps = model.forward_capture(torch.tensor([full], dtype=torch.long))
        for ln,nt,arr in caps:
            k=(ln,nt)
            if k not in TARGETS: continue
            g = gamma_cache[k]; s_in = float(cand[k][best_sin[k]])
            super_idx = [int(c) for c in TARGETS[k]]
            mask = np.ones(2304, dtype=bool); mask[super_idx]=False
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
                    acc[key]["err2_no_super"]+=err[mask].sum(); acc[key]["y2_no_super"]+=(y_fp[mask]**2).sum()
        del caps
        import gc; gc.collect()
    print("\n=== s_out sweep (NMSE / NMSE w/o super) ===")
    for k in TARGETS:
        for cname in ["p999","p9999","absmax","p999_x2"]:
            a=acc[(k[0],k[1],cname)]
            print(f"  {k[0]:6s} {k[1]:9s} {cname:7s} NMSE={a['err2']/a['y2']:.4f}  NMSE_no_super={a['err2_no_super']/a['y2_no_super']:.4f}")

if __name__=="__main__":
    main()