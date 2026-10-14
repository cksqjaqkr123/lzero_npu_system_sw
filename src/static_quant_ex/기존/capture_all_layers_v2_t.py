"""INT10 -> GeLU -> INT10 전체 레이어 스케일 추출 (깔끔 버전)
   - gate_proj만 훅, GeLU는 numpy gelu_pytorch_tanh로 직접 계산 (훅 오류 방지)
   - 인풋 스케일: 0.1 고정 말고 실제 분포에서 0.05, 0.03 등으로 LUT 만들 수 있게 전부 계산
   - 아웃풋 스케일: 255 단위(q=float*scale)로 100, 200, 300 등으로 LUT 만들 수 있게 inv_scale 계산

   실행:
   PYTHONPATH=src python -m static_quant.capture_all_layers_v2 \
     --calibration-data src/static_quant/cal_v3.jsonl \
     --output-dir ./lut_io_clean \
     --layers all --samples 16 --sequence-length 128 --device cpu --dtype bfloat16
"""
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from .gemma import GemmaSource
from .core import valid_rows

def gelu_pytorch_tanh(x):
    """Gemma가 쓰는 gelu 근사: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))"""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * np.power(x,3))))

def parse_layers(arg, num_layers):
    if arg == 'all':
        return list(range(num_layers))
    return [int(x.strip()) for x in arg.split(',') if x.strip()!='']

def add_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='google/gemma-2b-it')
    p.add_argument('--calibration-data', required=True)
    p.add_argument('--calibration-data-2', default=None)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--layers', default='all')
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', default='bfloat16', choices=['float32','float16','bfloat16'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--samples', type=int, default=16)
    p.add_argument('--sequence-length', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--text-format', choices=['plain','chat'], default='plain')
    p.add_argument('--revision', default='main')
    p.add_argument('--local-files-only', action='store_true')
    # 기존 호환용이지만 이제는 실제 분포에서 전부 계산
    p.add_argument('--s10-before', type=float, default=0.1, help='호환용, 결과에는 영향 없음')
    return p

def main(argv=None):
    args = add_args().parse_args(argv)

    class A: pass
    a = A()
    for k,v in vars(args).items():
        if k in ('calibration_data_2','output_dir','s10_before'): continue
        setattr(a, k, v)
    a.calibration_data = args.calibration_data
    a.validation_data = None
    a.validation_samples = 0
    a.modules = 'gate_proj'
    a.include_lm_head = False
    a.layers = 'all'
    a.revision = args.revision
    a.local_files_only = args.local_files_only

    print(f"[load] model={args.model} dtype={args.dtype} device={args.device}")
    source = GemmaSource(a)
    num_layers = len(source.model.model.layers)
    layer_indices = parse_layers(args.layers, num_layers)
    print(f"[info] num_layers={num_layers} target={layer_indices}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_mods = {li: source.modules.get(f'model.layers.{li}.mlp.gate_proj') for li in layer_indices}
    bufs = {li: {} for li in layer_indices}
    pre_lists = {li: [] for li in layer_indices}

    def make_gate_hook(li):
        def hook(m,i,o):
            bufs[li]['gate'] = o.detach().cpu().float().numpy()
        return hook

    handles = []
    for li in layer_indices:
        if gate_mods[li] is not None:
            handles.append(gate_mods[li].register_forward_hook(make_gate_hook(li)))

    rows = source.datasets['calibration']
    print(f"[capture] samples={len(rows)} batch={source.batch_size}")

    for start in range(0, len(rows), source.batch_size):
        batch = source.tokenizer.pad({'input_ids': rows[start:start+source.batch_size]},
                                     padding=True, return_tensors='pt', return_attention_mask=True)
        mask = batch['attention_mask'].numpy()
        for li in layer_indices:
            bufs[li].clear()
        with torch.inference_mode():
            source.model(**{k: v.to(source.device) for k,v in batch.items()}, use_cache=False)
        for li in layer_indices:
            if 'gate' in bufs[li]:
                pre_f = valid_rows(bufs[li]['gate'], mask)
                pre_lists[li].append(pre_f)

    for h in handles:
        h.remove()

    # === 스케일 계산 및 저장 (처음 보는 사람도 이해 쉽게) ===
    # A방식: float = q * scale  (예: scale 0.1이면 q=511 -> float 51.1)
    # B방식: q = float * scale  (예: scale 255면 float 2.0 -> q=510) -> 255 단위
    with open(out_dir / 'summary_A.csv','w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'layer',
            # 1. GeLU 전 (gate) 실제 분포
            'pre_absmax_float','pre_p999_float','pre_p99_float',
            # 1-a. GeLU 전 스케일 A (float = q * scale) - 0.1, 0.05, 0.03 LUT 만들 때 기준
            'pre_scale_A_absmax_float/511','pre_scale_A_p999','pre_scale_A_p99',
            # 1-b. GeLU 전 스케일 B (q = float * scale) - 255 단위 역수
            'pre_scale_B_inv_absmax_511/float','pre_scale_B_inv_p999_255unit','pre_scale_B_inv_p99',
            # 2. GeLU 후 실제 분포
            'post_absmax_float','post_p999_float','post_p99_float',
            # 2-a. GeLU 후 스케일 A
            'post_scale_A_absmax','post_scale_A_p999','post_scale_A_p99',
            # 2-b. GeLU 후 스케일 B - 100, 200, 300 LUT 만들 때 기준 (기존 255 디폴트와 비교)
            'post_scale_B_inv_absmax','post_scale_B_inv_p999_100_300unit','post_scale_B_inv_p99_255unit',
            # 참고: 커버리지 예시
            'input_coverage_example_A_0.1->51.1_0.05->25.5_0.03->15.3_float',
            'output_coverage_example_B_100->5.11_200->2.55_300->1.70_float',
            'elements'
        ])
        # 전체 평균용
        agg = []

        for li in layer_indices:
            if not pre_lists[li]:
                print(f"[L{li}] no data")
                continue
            pre = np.concatenate(pre_lists[li], axis=0)
            post = gelu_pytorch_tanh(pre)  # 진짜 GeLU

            # 통계
            pre_absmax = float(np.abs(pre).max())
            pre_p999 = float(np.percentile(np.abs(pre), 99.9))
            pre_p99 = float(np.percentile(np.abs(pre), 99.0))

            post_absmax = float(np.abs(post).max())
            post_p999 = float(np.percentile(np.abs(post), 99.9))
            post_p99 = float(np.percentile(np.abs(post), 99.0))

            # 스케일 A: float = q * scale => scale = float_max / 511
            pre_sA_abs = pre_absmax / 511
            pre_sA_p999 = pre_p999 / 511
            pre_sA_p99 = pre_p99 / 511

            post_sA_abs = post_absmax / 511
            post_sA_p999 = post_p999 / 511
            post_sA_p99 = post_p99 / 511

            # 스케일 B: q = float * scale => scale = 511 / float_max (255 단위)
            pre_sB_abs = 511 / pre_absmax if pre_absmax else 0
            pre_sB_p999 = 511 / pre_p999 if pre_p999 else 0
            pre_sB_p99 = 511 / pre_p99 if pre_p99 else 0

            post_sB_abs = 511 / post_absmax if post_absmax else 0
            post_sB_p999 = 511 / post_p999 if post_p999 else 0
            post_sB_p99 = 511 / post_p99 if post_p99 else 0

            # 저장 (q10은 이제 0.1 고정 말고, 실제 p999 기준으로도 저장해서 LUT 생성에 바로 쓰게)
            # 기존 호환용 q10_0.1도 저장
            q10_01 = np.clip(np.rint(pre / 0.1), -512, 511).astype(np.int16)
            lut_index_01 = (q10_01 & 1023).astype(np.uint16)

            # 최적 스케일 기준 q10 (p999 기준)
            q10_opt = np.clip(np.rint(pre / pre_sA_p999), -512, 511).astype(np.int16) if pre_sA_p999 else q10_01
            lut_index_opt = (q10_opt & 1023).astype(np.uint16)

            np.savez(out_dir / f'layer{li}_A.npz',
                     pre=pre, post=post,
                     q10_0p1=q10_01, lut_index_0p1=lut_index_01,
                     q10_p999=q10_opt, lut_index_p999=lut_index_opt,
                     pre_absmax=pre_absmax, pre_p999=pre_p999, pre_p99=pre_p99,
                     post_absmax=post_absmax, post_p999=post_p999, post_p99=post_p99)

            # 커버리지 예시 문자열 (처음 보는 사람 이해용)
            in_cov = f"0.1->{0.1*511:.1f} / 0.05->{0.05*511:.1f} / 0.03->{0.03*511:.1f} float"
            out_cov = f"100->{511/100:.2f} / 200->{511/200:.2f} / 300->{511/300:.2f} float"

            print(f"[L{li}] pre {pre_p999:.3f} (scaleA {pre_sA_p999:.5f} scaleB {pre_sB_p999:.1f}) "
                  f"-> post {post_p999:.3f} (scaleA {post_sA_p999:.5f} scaleB {post_sB_p999:.1f}) elems {pre.shape[0]}")

            w.writerow([
                li,
                f"{pre_absmax:.4f}", f"{pre_p999:.4f}", f"{pre_p99:.4f}",
                f"{pre_sA_abs:.6f}", f"{pre_sA_p999:.6f}", f"{pre_sA_p99:.6f}",
                f"{pre_sB_abs:.1f}", f"{pre_sB_p999:.1f}", f"{pre_sB_p99:.1f}",
                f"{post_absmax:.4f}", f"{post_p999:.4f}", f"{post_p99:.4f}",
                f"{post_sA_abs:.6f}", f"{post_sA_p999:.6f}", f"{post_sA_p99:.6f}",
                f"{post_sB_abs:.1f}", f"{post_sB_p999:.1f}", f"{post_sB_p99:.1f}",
                in_cov, out_cov, pre.shape[0]
            ])
            agg.append((pre_absmax, pre_p999, pre_p99, post_absmax, post_p999, post_p99))

    # 전체 평균 요약 출력
    if agg:
        import numpy as _np
        arr = _np.array(agg)
        print("\n=== 전체 평균 (18 레이어) ===")
        print(f"pre  absmax {arr[:,0].mean():.3f} -> scaleA {arr[:,0].mean()/511:.5f} scaleB {511/arr[:,0].mean():.1f}")
        print(f"pre  p999   {arr[:,1].mean():.3f} -> scaleA {arr[:,1].mean()/511:.5f} scaleB {511/arr[:,1].mean():.1f} (0.05 LUT면 {0.05*511:.1f}까지 커버)")
        print(f"pre  p99    {arr[:,2].mean():.3f} -> scaleA {arr[:,2].mean()/511:.5f} scaleB {511/arr[:,2].mean():.1f}")
        print(f"post absmax {arr[:,3].mean():.3f} -> scaleA {arr[:,3].mean()/511:.5f} scaleB {511/arr[:,3].mean():.1f}")
        print(f"post p999   {arr[:,4].mean():.3f} -> scaleA {arr[:,4].mean()/511:.5f} scaleB {511/arr[:,4].mean():.1f} (B 100이면 5.11, 200이면 2.55, 300이면 1.70까지)")
        print(f"post p99    {arr[:,5].mean():.3f} -> scaleA {arr[:,5].mean()/511:.5f} scaleB {511/arr[:,5].mean():.1f} <- 기존 디폴트 255와 비교")

    print(f"\n완료: {out_dir}/summary_A.csv")
    print(f"  - pre_scale_A_* : float = q * scale, 0.1, 0.05, 0.03 LUT 만들 때 사용")
    print(f"  - post_scale_B_* : q = float * scale, 100, 200, 300, 255(기존) LUT 만들 때 사용")

if __name__ == '__main__':
    main()


