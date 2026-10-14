"""INT10 -> GeLU -> INT10 전체 레이어 한방 포착 (저메모리, 1-pass)
   - ImportError fix: relative + absolute both work
   - summary_A.csv: layer,pre_absmax,pre_p999,pre_p99,pre_scale_A_p999,post_absmax,post_p999,post_p99,elements
   - elements 4000+ 나오게 64*128 권장
"""
import argparse, csv
from pathlib import Path
import numpy as np
import torch
try:
    from .gemma import GemmaSource
    from .core import valid_rows
except (ImportError, ModuleNotFoundError, ValueError):
    from gemma import GemmaSource
    from core import valid_rows

def parse_layers(arg, num_layers):
    if arg == 'all':
        return list(range(num_layers))
    return [int(x.strip()) for x in arg.split(',') if x.strip()!='']

def add_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='google/gemma-2b-it')
    p.add_argument('--calibration-data', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--layers', default='all')
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', default='bfloat16', choices=['float32','float16','bfloat16'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--sequence-length', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--text-format', choices=['plain','chat'], default='plain')
    p.add_argument('--revision', default='main')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--s10-before', type=float, default=0.1)
    return p

def main(argv=None):
    args = add_args().parse_args(argv)

    class A: pass
    a = A()
    for k,v in vars(args).items():
        if k in ('output_dir','s10_before'): continue
        setattr(a, k, v)
    a.validation_data = None
    a.validation_samples = 0
    a.modules = 'gate_proj'
    a.include_lm_head = False
    a.layers = 'all'
    a.revision = args.revision
    a.local_files_only = args.local_files_only
    a.calibration_data = args.calibration_data

    print(f"[load] model={args.model} dtype={args.dtype} device={args.device} samples={args.samples} seq={args.sequence_length}")
    source = GemmaSource(a)
    num_layers = len(source.model.model.layers)
    layer_indices = parse_layers(args.layers, num_layers)
    print(f"[info] num_layers={num_layers} target={layer_indices}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_mods = {}
    act_mods = {}
    for li in layer_indices:
        gate_name = f'model.layers.{li}.mlp.gate_proj'
        gate_mods[li] = source.modules.get(gate_name)
        try:
            mlp = source.model.model.layers[li].mlp
            act = getattr(mlp, 'act_fn', None)
            if act is None:
                act = mlp
            act_mods[li] = act
        except Exception as e:
            print(f"[warn] L{li} act not found: {e}")
            act_mods[li] = None

    bufs = {li: {} for li in layer_indices}
    pre_lists = {li: [] for li in layer_indices}
    post_lists = {li: [] for li in layer_indices}

    def make_gate_hook(li):
        def hook(m,i,o):
            bufs[li]['gate'] = o.detach().cpu().float().numpy()
        return hook
    def make_act_hook(li):
        def hook(m,i,o):
            if isinstance(o, tuple):
                o = o[0]
            bufs[li]['gelu'] = o.detach().cpu().float().numpy()
        return hook

    handles = []
    for li in layer_indices:
        if gate_mods[li] is not None:
            handles.append(gate_mods[li].register_forward_hook(make_gate_hook(li)))
        if act_mods[li] is not None and hasattr(act_mods[li], 'register_forward_hook'):
            if act_mods[li] is not gate_mods[li]:
                handles.append(act_mods[li].register_forward_hook(make_act_hook(li)))

    rows = source.datasets['calibration']
    print(f"[capture] samples={len(rows)} batch={source.batch_size} -> expected elements ~ {len(rows)*args.sequence_length//2} (after padding)")

    for start in range(0, len(rows), source.batch_size):
        batch = source.tokenizer.pad({'input_ids': rows[start:start+source.batch_size]},
                                     padding=True, return_tensors='pt', return_attention_mask=True)
        mask = batch['attention_mask'].numpy()
        for li in layer_indices:
            bufs[li].clear()

        with torch.inference_mode():
            source.model(**{k: v.to(source.device) for k,v in batch.items()}, use_cache=False)

        for li in layer_indices:
            if 'gate' in bufs[li] and 'gelu' in bufs[li]:
                gate_f = valid_rows(bufs[li]['gate'], mask)
                gelu_f = valid_rows(bufs[li]['gelu'], mask)
                pre_lists[li].append(gate_f)
                post_lists[li].append(gelu_f)

        if (start//source.batch_size + 1) % 10 == 0:
            print(f"  batch {start//source.batch_size + 1}/{(len(rows)+source.batch_size-1)//source.batch_size}")

    for h in handles:
        h.remove()

    # 저장: 네가 쓰는 포맷으로
    with open(out_dir / 'summary_A.csv','w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer','pre_absmax','pre_p999','pre_p99','pre_scale_A_p999','post_absmax','post_p999','post_p99','elements'])
        for li in layer_indices:
            if not pre_lists[li]:
                print(f"[L{li}] no data")
                continue
            pre = np.concatenate(pre_lists[li], axis=0)
            post = np.concatenate(post_lists[li], axis=0)

            pre_absmax = float(np.abs(pre).max()) if pre.size else 0
            pre_p999 = float(np.percentile(np.abs(pre), 99.9)) if pre.size else 0
            pre_p99 = float(np.percentile(np.abs(pre), 99.0)) if pre.size else 0
            pre_scale = pre_p999 / 511.0 if pre_p999>0 else 0

            post_absmax = float(np.abs(post).max()) if post.size else 0
            post_p999 = float(np.percentile(np.abs(post), 99.9)) if post.size else 0
            post_p99 = float(np.percentile(np.abs(post), 99.0)) if post.size else 0

            elements = int(pre.shape[0])

            print(f"[L{li}] pre_max {pre_absmax:.3f} pre_p999 {pre_p999:.3f} pre_p99 {pre_p99:.3f} post_max {post_absmax:.3f} elements {elements}")
            w.writerow([li, f"{pre_absmax:.4f}", f"{pre_p999:.4f}", f"{pre_p99:.4f}", f"{pre_scale:.6f}",
                        f"{post_absmax:.4f}", f"{post_p999:.4f}", f"{post_p99:.4f}", elements])

    print(f"\n완료: {out_dir} / summary_A.csv elements 확인해라 - 4000 넘어야 64*128 성공")
    print(f"  다음: evaluate_final_8bin.py로 final_stats.csv 만들기")

if __name__ == '__main__':
    main()

