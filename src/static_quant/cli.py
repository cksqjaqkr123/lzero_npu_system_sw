import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from .core import InputStats, LinearCalibration, Options, valid_rows
from .export import Exporter
from .gemma import GemmaSource, input_group
from .hardware import get_profile


def parser():
    p = argparse.ArgumentParser(description='Original Gemma 2B static INT8 Linear -> INT10 local prefill calibration')
    p.add_argument('--model', default='google/gemma-2b-it')
    p.add_argument('--revision', default='main')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--calibration-data')
    p.add_argument('--validation-data')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float32')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--validation-samples', type=int, default=8)
    p.add_argument('--sequence-length', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--layers', default='0', help='comma-separated indices or all')
    p.add_argument('--modules', default='q_proj', help='comma-separated projection names or all')
    p.add_argument('--include-lm-head', action='store_true')
    p.add_argument('--text-format', choices=['plain', 'chat'], default='plain')
    p.add_argument('--input-percentile', type=float, help='optional sampled percentile (0,100] instead of exact absmax')
    p.add_argument('--percentile-capacity', type=int, default=65536)
    p.add_argument('--scale-mode', choices=['fixed', 'minmax', 'mse'], default='minmax')
    p.add_argument('--s10', type=float)
    p.add_argument('--reuse-gelu-lut', action='store_true', help='require fixed s10=0.1, matching src/LUT.py index scale')
    p.add_argument('--thresholds', default='1,.95,.9,.8,.7,.5')
    p.add_argument('--search-rows', type=int, default=64)
    p.add_argument('--m-chunk', type=int, default=32)
    p.add_argument('--n-chunk', type=int, default=64)
    p.add_argument('--k-chunk', type=int, default=256)
    p.add_argument('--profile', choices=['generic-rne', 'legacy-guess'], default='generic-rne')
    p.add_argument('--ratio-tolerance', type=float, default=1e-3)
    p.add_argument('--allow-unverified-export', action='store_true')
    p.add_argument('--allow-inaccurate-parameters', action='store_true')
    p.add_argument('--save-int8-weights', action='store_true')
    p.add_argument('--synthetic', action='store_true', help='offline smoke fixture; NEVER real Gemma parameters')
    return p


def provenance():
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return dict(source_commit=commit, source_tree_dirty=dirty, python_version=platform.python_version())


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.samples, args.validation_samples, args.sequence_length, args.batch_size, args.percentile_capacity) < 1:
        raise ValueError('sample/token/batch/capacity values must be positive')
    if args.input_percentile is not None and not 0 < args.input_percentile <= 100:
        raise ValueError('input percentile must be in (0,100]')
    if args.seed < 0:
        raise ValueError('seed must be nonnegative')
    if Path(args.output_dir).exists() and any(Path(args.output_dir).iterdir()):
        raise FileExistsError(f'Output directory must be empty: {args.output_dir}')
    options = Options(m_chunk=args.m_chunk, n_chunk=args.n_chunk, k_chunk=args.k_chunk,
                      search_rows=args.search_rows, seed=args.seed, mode=args.scale_mode, s10=args.s10,
                      fixed_lut_scale=0.1 if args.reuse_gelu_lut else None,
                      thresholds=tuple(float(s) for s in args.thresholds.split(',')), ratio_tolerance=args.ratio_tolerance)
    profile = get_profile(args.profile)
    if args.synthetic:
        if args.calibration_data or args.validation_data:
            raise ValueError('--synthetic uses a labeled fixture; do not pass real data paths')
        rng = np.random.default_rng(args.seed)
        x = rng.normal(size=(3, 7, 19)).astype(np.float32)
        mask = np.ones((3,7), dtype=np.int64)
        mask[1,4:] = 0
        x[1,4:] = 1e6  # excluded padding fixture
        w = rng.normal(size=(21,19))*np.linspace(0.01,3,21)[:,None]
        w[0] = 0
        names = ['synthetic.linear']
        def replay(name, split='calibration'):
            return lambda callback: callback(valid_rows(x, mask))
        weights = {names[0]: w}
        read_weight = lambda name: lambda sl: weights[name][sl]
        shapes = {names[0]: w.shape}
        metadata = dict(model='SYNTHETIC FIXTURE, NOT GEMMA', model_revision_resolved=None,
                        datasets={'calibration': dict(actual_samples=3, valid_tokens=int(mask.sum()),
                            fixture_sha256=hashlib.sha256(x.tobytes()+w.tobytes()+mask.tobytes()).hexdigest())},
                        library_versions={'numpy': np.__version__})
        has_validation = False
    else:
        if not args.calibration_data:
            raise ValueError('--calibration-data is required unless --synthetic is selected')
        print('Loading real pretrained checkpoint and tokenizer...', flush=True)
        source = GemmaSource(args)
        names, replay, read_weight, metadata = source.names, source.replay, source.read_weight, source.metadata
        shapes = {name: tuple(source.modules[name].weight.shape) for name in names}
        has_validation = args.validation_data is not None
    metadata.update(provenance(), seed=args.seed, calibration_options=asdict(options), batch_size=args.batch_size)
    exporter = Exporter(args.output_dir, profile, metadata, args.allow_unverified_export,
                        args.save_int8_weights, args.allow_inaccurate_parameters)
    groups = {}
    for index, name in enumerate(names):
        group = input_group(name)
        print(f'[{index+1}/{len(names)}] {name}: Pass A / B / C / fixed-parameter evaluation', flush=True)
        if group not in groups:
            stats = InputStats(args.seed, args.input_percentile, args.percentile_capacity)
            replay(name)(stats.add)
            groups[group] = (stats, shapes[name][1])
        stats, input_width = groups[group]
        if input_width != shapes[name][1]:
            raise ValueError(f'{group}: shared input width mismatch')
        operation = LinearCalibration(name, shapes[name], read_weight(name), stats.scale(), profile, options)
        report, vectors = operation.run(replay(name), replay(name, 'validation') if has_validation else None)
        exporter.add(operation, report, vectors, stats.metadata(), group)
    manifest = exporter.finish()
    print(f'Artifacts: {Path(args.output_dir).resolve()}\nqparams.bin: {manifest["binary_export"]["status"]}', flush=True)
    return manifest


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, OverflowError) as exc:
        print(f'Calibration failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
