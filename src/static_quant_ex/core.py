"""Bounded-memory, NumPy INT64 reference and streaming local calibration."""
from dataclasses import dataclass
import numpy as np
try:
    from .hardware import finite, positive_scale, integers, ideal, check_lut_scale, GeluLut, load_gelu_lut
except (ImportError, ModuleNotFoundError, ValueError):
    from hardware import finite, positive_scale, integers, ideal, check_lut_scale, GeluLut, load_gelu_lut


def valid_rows(x, mask):
    x = np.asarray(x)
    mask = np.asarray(mask)
    if x.ndim < 2 or x.shape[:-1] != mask.shape:
        raise ValueError(f'activation {x.shape} and token mask {mask.shape} do not match')
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('attention mask must be binary')
    # Check even padded entries: nonfinite inputs must never be silently ignored.
    finite(x, 'Linear input (including padding)')
    return x.reshape(-1, x.shape[-1])[mask.reshape(-1).astype(bool)]


class Reservoir:
    """Uniform priority sampling without replacement, bounded in rows/items."""
    def __init__(self, capacity, seed):
        if capacity < 1:
            raise ValueError('reservoir capacity must be positive')
        self.capacity, self.rng, self.seen = capacity, np.random.default_rng(seed), 0
        self.values = None
        self.keys = np.empty(0)

    def add(self, values):
        values = np.asarray(values)
        # Bound temporary keys/data, including scalar percentile sampling.
        for start in range(0, len(values), self.capacity):
            batch = values[start:start+self.capacity]
            self.seen += len(batch)
            keys = np.concatenate([self.keys, self.rng.random(len(batch))])
            data = batch.copy() if self.values is None else np.concatenate([self.values, batch])
            keep = np.argsort(keys, kind='stable')[:self.capacity]
            self.keys, self.values = keys[keep], data[keep]

    def metadata(self):
        return dict(method='uniform smallest random priorities without replacement',
                    capacity=self.capacity, observed=self.seen,
                    retained=0 if self.values is None else len(self.values))


class InputStats:
    def __init__(self, seed=0, percentile=None, capacity=65536, search_percentiles=()):
        if percentile is not None and not 0 < percentile <= 100:
            raise ValueError('percentile must be in (0,100]')
        self.percentile = percentile
        self.search_percentiles = tuple(search_percentiles)
        self.sample = Reservoir(capacity, seed) if percentile is not None or search_percentiles else None
        self.absmax, self.tokens = 0.0, 0

    def add(self, x, positions=None):
        x = finite(x, 'Pass A input')
        if not len(x):
            return
        self.tokens += len(x)
        self.absmax = max(self.absmax, float(np.abs(x).max()))
        if self.sample is not None:
            self.sample.add(np.abs(x).reshape(-1))

    def scale(self):
        if not self.tokens:
            raise ValueError('No valid calibration tokens')
        threshold = self.absmax if self.percentile is None else float(np.percentile(self.sample.values, self.percentile))
        return threshold / 127 if threshold > 0 else (self.absmax / 127 if self.absmax > 0 else 1.0)

    def metadata(self):
        return dict(method='absmax' if self.percentile is None else 'sampled_percentile',
                    search_percentiles=list(self.search_percentiles),
                    candidate_estimator='bounded uniform scalar reservoir; linear percentile interpolation' if self.search_percentiles else None,
                    percentile=self.percentile, absmax=self.absmax, valid_tokens=self.tokens,
                    s_X=self.scale(), all_zero=self.absmax == 0,
                    zero_scale_policy='all zero -> 1; zero percentile on nonzero tensor -> absmax/127',
                    sampling=None if self.sample is None else self.sample.metadata())

    def candidates(self):
        try:
            from .search import activation_candidates
        except (ImportError, ModuleNotFoundError, ValueError):
            from search import activation_candidates
        return activation_candidates(self.absmax, None if self.sample is None else self.sample.values,
                                     self.search_percentiles)


def quantize_input(x, sx):
    return np.clip(np.rint(finite(x, 'X') / positive_scale(sx, 's_X')), -127, 127).astype(np.int8)


def quantize_weight(w):
    w = finite(np.asarray(w, dtype=np.float64), 'W')
    if w.ndim != 2 or not all(w.shape):
        raise ValueError('W must be nonempty [N,K]')
    peaks = np.abs(w).max(axis=1)
    sw = np.where(peaks > 0, peaks / 127, 1.0)
    return np.clip(np.rint(w / sw[:, None]), -127, 127).astype(np.int8), sw


def exact_accumulator(xq, wq, k_chunk=256):
    """Accumulate all K in INT64, check every partial INT32 sum; never requantize tiles."""
    x = integers(xq, -127, 127, 'X_q')
    w = integers(wq, -127, 127, 'W_q')
    if k_chunk < 1 or x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError('Expected X[M,K], W[N,K] and positive K chunk')
    acc = np.zeros((len(x), len(w)), dtype=np.int64)
    for start in range(0, x.shape[1], k_chunk):
        acc += x[:, start:start+k_chunk] @ w[:, start:start+k_chunk].T
        check_int32(acc)
    return acc


def check_int32(acc):
    if np.any(acc < -(2**31)) or np.any(acc > 2**31-1):
        raise OverflowError('INT32 accumulator overflow (full or K-chunk boundary partial sum)')


@dataclass
class Options:
    m_chunk: int = 32
    n_chunk: int = 64
    k_chunk: int = 256
    search_rows: int = 64
    seed: int = 0
    mode: str = 'minmax'
    s10: float = None
    fixed_lut_scale: float = None
    thresholds: tuple = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)
    ratio_tolerance: float = 1e-3
    bos_search_rows: int = 16
    early_search_rows: int = 16
    early_position_limit: int = 8
    representative_sampling: bool = True
    gelu_lut_path: str = "./luts/gelu_BIN0_7_combined_8192B.bin"
    gelu_scale_map_path: str = "./luts/gelu_scale_map.json"

    def __post_init__(self):
        if not np.isfinite(self.ratio_tolerance) or self.ratio_tolerance < 0:
            raise ValueError('ratio tolerance must be finite and nonnegative')
        if min(self.m_chunk, self.n_chunk, self.k_chunk, self.search_rows) < 1:
            raise ValueError('chunk sizes and search rows must be positive')
        if min(self.bos_search_rows, self.early_search_rows, self.early_position_limit) < 0:
            raise ValueError('sampling quotas and position limit must be nonnegative')
        if self.mode not in ('fixed', 'minmax', 'mse'):
            raise ValueError('mode must be fixed, minmax or mse')
        if self.mode == 'fixed':
            positive_scale(self.s10, 'fixed s10')
        elif self.s10 is not None:
            raise ValueError('s10 is only accepted in fixed mode')
        if self.fixed_lut_scale is not None:
            if self.mode != 'fixed':
                raise ValueError('a reused LUT requires fixed mode (shared scale across operations)')
            check_lut_scale(self.s10, self.fixed_lut_scale)
        if not self.thresholds or any(not np.isfinite(t) or not 0 < t <= 1 for t in self.thresholds):
            raise ValueError('threshold factors must be finite in (0,1]')


class Metrics:
    def __init__(self, n):
        self.count = np.zeros(n, dtype=np.int64)
        self.sum_sq, self.sum_abs, self.maximum = [np.zeros(n) for _ in range(3)]

    def add(self, sl, error):
        finite(error, 'metric error')
        self.count[sl] += len(error)
        self.sum_sq[sl] += (error * error).sum(axis=0)
        self.sum_abs[sl] += np.abs(error).sum(axis=0)
        self.maximum[sl] = np.maximum(self.maximum[sl], np.abs(error).max(axis=0))

    def report(self):
        if np.any(self.count == 0):
            raise ValueError('No observations for metric channel')
        return dict(mse=float(self.sum_sq.sum()/self.count.sum()),
                    mae=float(self.sum_abs.sum()/self.count.sum()),
                    max_absolute_error=float(self.maximum.max()),
                    channel_mse=(self.sum_sq/self.count).tolist(),
                    channel_mae=(self.sum_abs/self.count).tolist(),
                    channel_max_absolute_error=self.maximum.tolist())


class LinearCalibration:
    """replay(callback) supplies valid, ORIGINAL model inputs; callable W reads N slices.

    Only one operation is calibrated at a time. W storage may stay on the model's
    device; no full float or quantized copy is necessary, including for lm_head.
    """
    def __init__(self, name, shape, read_weight, sx, profile, options, activation_thresholds=None, gelu_lut_path=None, gelu_scale_map_path=None):
        self.name, self.n, self.k = name, int(shape[0]), int(shape[1])
        self.read_weight, self.sx, self.profile, self.options = read_weight, positive_scale(sx, 's_X'), profile, options
        self.gelu_lut = None
        # 8192B 10->8 BIN - static_quant이 10->10 대신 이거 사용
        lut_path = gelu_lut_path or getattr(options, 'gelu_lut_path', None)
        scale_path = gelu_scale_map_path or getattr(options, 'gelu_scale_map_path', None)
        if lut_path and scale_path:
            try:
                from pathlib import Path
                if Path(lut_path).exists() and Path(scale_path).exists():
                    self.gelu_lut = load_gelu_lut(lut_path, scale_path)
                    print(f"[{name}] GeLU LUT 8192B loaded: {lut_path} {self.gelu_lut.total_bytes}B mode={self.gelu_lut.mode}")
            except Exception as e:
                print(f"[{name}] GeLU LUT 8192B load fail: {e}")
                self.gelu_lut = None
        self.sw = np.empty(self.n)

        self.zero_channels = []
        if min(self.n, self.k) < 1:
            raise ValueError('empty weight')
        # No possible INT32 partial overflow at ANY K prefix under this bound.
        # Larger K still uses boundary checks, with a conservative data bound below.
        for sl in self.channels():
            w = self.weight(sl)
            _, self.sw[sl] = quantize_weight(w)
            self.zero_channels.extend((np.flatnonzero(np.all(w == 0, axis=1)) + sl.start).tolist())
        try:
            from .sampling import RepresentativeRowSampler
        except (ImportError, ModuleNotFoundError, ValueError):
            from sampling import RepresentativeRowSampler
        self.sample = RepresentativeRowSampler(options.search_rows, options.seed,
            options.bos_search_rows, options.early_search_rows, options.early_position_limit,
            options.representative_sampling)
        self.activation_thresholds = activation_thresholds or [dict(method='supplied', percentile=None,
                                                                     threshold=127*self.sx, s_X=self.sx)]
        self.peaks = np.zeros(self.n)
        self.tokens = 0
        self.params, self.s10 = None, None

    def channels(self):
        for start in range(0, self.n, self.options.n_chunk):
            yield slice(start, min(start+self.options.n_chunk, self.n))

    def weight(self, sl):
        w = finite(np.asarray(self.read_weight(sl), dtype=np.float64), f'{self.name}.weight channels {sl.start}:{sl.stop}')
        if w.shape != (sl.stop-sl.start, self.k):
            raise ValueError(f'{self.name}: unexpected weight shape {w.shape}')
        return w

    def blocks(self, x, original=False):
        x = finite(np.asarray(x), f'{self.name} input')
        if x.ndim != 2 or x.shape[1] != self.k:
            raise ValueError(f'{self.name}: expected input [M,{self.k}], got {x.shape}')
        for sl in self.channels():
            w = self.weight(sl)
            wq = np.clip(np.rint(w/self.sw[sl, None]), -127, 127).astype(np.int8)
            for start in range(0, len(x), self.options.m_chunk):
                xb = x[start:start+self.options.m_chunk].astype(np.float64)
                xq = quantize_input(xb, self.sx)
                if self.k * 127**2 > 2**31-1:
                    # Absolute-product bound protects *all* prefixes, not just chunk ends.
                    bound = np.abs(xq.astype(np.int64)) @ np.abs(wq.astype(np.int64)).T
                    if np.any(bound > 2**31-1):
                        raise OverflowError(f'{self.name}: cannot guarantee INT32 prefix safety (absolute-product bound)')
                try:
                    acc = exact_accumulator(xq, wq, self.options.k_chunk)
                except OverflowError as exc:
                    raise OverflowError(f'{self.name}, channels {sl.start}:{sl.stop}: {exc}') from exc
                target = xb @ w.T if original else None  # FP64 local float reference, NOT integer reference
                yield sl, acc, target

    def observe(self, x, positions=None):
        if not len(x):
            return
        self.tokens += len(x)
        self.sample.add(x, positions)
        for sl, acc, _ in self.blocks(x):
            real = acc * (self.sx*self.sw[sl])
            self.peaks[sl] = np.maximum(self.peaks[sl], np.abs(real).max(axis=0))

    def choose(self):
        if not self.tokens:
            raise ValueError(f'{self.name}: no valid tokens in Pass B')
        o = self.options
        baseline = float(self.peaks.max()/511) if self.peaks.max() else 1.0
        try:
            from .search import joint_scale_search
        except (ImportError, ModuleNotFoundError, ValueError):
            from search import joint_scale_search
        self.sx, self.s10, self.params, self.selection = joint_scale_search(
            self.read_weight, (self.n, self.k), self.sw, self.sample,
            self.activation_thresholds, baseline, self.profile, o)
        return self.params

    def gelu_tanh(self, x):
        import math
        # numpy version for array
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * np.power(x,3))))

    def evaluate(self, replay):
        metrics = {key: Metrics(self.n) for key in ('requantization', 'local_total', 'parameter_approximation')}
        # 10->10 GeLU 단계 메트릭 (BIN 파일 꽂았을 때)
        gelu_metrics = {key: Metrics(self.n) for key in ('gelu_lut', 'gelu_float')} if self.gelu_lut else None
        gelu_group_sums = {name: [0., 0., 0] for name in ('bos', 'non_bos', 'early', 'general', 'all')} if self.gelu_lut else None
        gelu_bin_usage = {} if self.gelu_lut else None
        clipped = np.zeros(self.n, dtype=np.int64)
        ideal_clipped = np.zeros(self.n, dtype=np.int64)
        tokens, input_clipped, input_elements = 0, 0, 0
        vectors = []
        group_sums = {name: [0., 0., 0] for name in ('bos', 'non_bos', 'early', 'general', 'all')}
        input_zero = 0
        def observe(x, positions=None):
            nonlocal tokens, input_clipped, input_elements, input_zero
            if not len(x):
                return
            tokens += len(x)
            input_clipped += int(np.count_nonzero(np.abs(x/self.sx) > 127))
            input_elements += x.size
            input_zero += int((quantize_input(x, self.sx) == 0).sum())
            pos = np.full(len(x), -1) if positions is None else np.asarray(positions)
            if pos.shape != (len(x),) or (positions is not None and (pos.dtype.kind not in 'iu' or np.any(pos < -1))):
                raise ValueError('one integer position per evaluation row required')
            masks = dict(bos=pos == 0, non_bos=pos != 0,
                         early=(pos > 0) & (pos <= self.options.early_position_limit),
                         general=(pos > self.options.early_position_limit) | (pos < 0), all=np.ones(len(x), bool))
            # blocks iterate channels, then M chunks; track the row offset per channel slice.
            previous_channel, row_start = None, 0
            for sl, acc, original in self.blocks(x, original=True):
                if sl.start != previous_channel:
                    previous_channel, row_start = sl.start, 0
                p = self.params
                raw = self.profile.apply(acc, p['multiplier'][sl], p['shift'][sl], saturate=False)
                hw = np.clip(raw, -512, 511).astype(np.int16)
                ref = ideal(acc, p['ratio'][sl])
                metrics['requantization'].add(sl, hw*self.s10 - acc*(self.sx*self.sw[sl]))
                metrics['local_total'].add(sl, hw*self.s10 - original)

                # === 10->10 GeLU 단계: bin 파일 로드해서 LUT lookup 최종통계 ===
                if self.gelu_lut:
                    # pre_gelu_float = hw * s10
                    pre_gelu_float = hw.astype(np.float32) * self.s10
                    post_gelu_true = self.gelu_tanh(pre_gelu_float)
                    # BIN 선택: s10 기준 (또는 pre_p999 기준)
                    bin_id, in_scale, out_scale = self.gelu_lut.select_bin(self.s10)
                    if gelu_bin_usage is not None:
                        gelu_bin_usage[bin_id] = gelu_bin_usage.get(bin_id, 0) + hw.size
                    q_out, lut_idx, _ = self.gelu_lut.lookup(hw, bin_id=bin_id)
                    post_gelu_lut_float = self.gelu_lut.dequant(q_out, out_scale)
                    gelu_metrics['gelu_lut'].add(sl, post_gelu_lut_float - post_gelu_true)
                    gelu_metrics['gelu_float'].add(sl, post_gelu_true - original)  # 참고용
                    # position별 GeLU 오차
                    for group, mask in masks.items():
                        if group not in gelu_group_sums:
                            continue
                        selected = mask[row_start:row_start+len(acc)]
                        # flatten for group
                        for c in range(post_gelu_true.shape[1]):
                            # channel slice handling
                            err = (post_gelu_lut_float[:, c] - post_gelu_true[:, c])[selected]
                            ref_energy = post_gelu_true[:, c][selected]
                            gelu_group_sums[group][0] += float(np.square(err).sum())
                            gelu_group_sums[group][1] += float(np.square(ref_energy).sum())
                            gelu_group_sums[group][2] += err.size
                        # break after first group handling to avoid double count - actually need per channel
                    # 위 loop는 채널 단위 중복이므로 간단히 전체로 계산 (기존 로직과 동일하게)
                    # position_metrics용은 아래에서 all만 정확히 계산
                for group, mask in masks.items():
                    selected = mask[row_start:row_start+len(acc)]
                    error = (hw*self.s10-original)[selected]
                    group_sums[group][0] += float(np.square(error).sum())
                    group_sums[group][1] += float(np.square(original[selected]).sum())
                    group_sums[group][2] += error.size
                row_start += len(acc)
                metrics['parameter_approximation'].add(sl, hw.astype(np.float64)-ref)
                clipped[sl] += ((raw < -512) | (raw > 511)).sum(axis=0)
                rounded_ideal = np.rint(acc*p['ratio'][sl])
                ideal_clipped[sl] += ((rounded_ideal < -512) | (rounded_ideal > 511)).sum(axis=0)
                if not vectors:
                    for row in range(min(4, len(acc))):
                        for col in range(min(16, acc.shape[1])):
                            channel = sl.start+col
                            vectors.append(dict(channel=channel, accumulator=int(acc[row,col]),
                                                word=int(self.profile.pack(p['multiplier'][channel], p['shift'][channel])),
                                                expected_int10=int(hw[row,col]), lut_index=int(hw[row,col]) & 1023))
        replay(observe)
        result = {key: value.report() for key, value in metrics.items()}
        # GeLU LUT 10->10 최종통계 추가
        if self.gelu_lut and gelu_metrics:
            for k, v in gelu_metrics.items():
                try:
                    result[k] = v.report()
                except:
                    result[k] = dict(mse=None, mae=None, max_absolute_error=None)
            result['gelu_lut']['bin_usage'] = gelu_bin_usage
            result['gelu_lut']['lut_metadata'] = self.gelu_lut.metadata()
            result['gelu_lut']['s10'] = self.s10
            # GeLU p99 등 추가 계산을 위해 group_sums도 포함
            if gelu_group_sums:
                result['gelu_position_metrics'] = {g: dict(mse=e/c if c else None,
                      nmse=e/v if v else (0. if c and not e else None), output_elements=c)
                      for g, (e, v, c) in gelu_group_sums.items()}

        error, energy, count = group_sums['all']
        result['local_total']['nmse'] = error/energy if energy else (0. if not error else None)
        result['local_total']['reference_energy'] = energy/count
        result.update(valid_tokens=tokens, int32_overflow=False,
                      overflow_check='INT64 partial sums; for large K conservative absolute-product prefix bound',
                      clipping=dict(definition='rounded result before INT10 saturation outside [-512,511]',
                                    overall=float(clipped.sum()/(tokens*self.n)),
                                    per_channel=(clipped/tokens).tolist(),
                                    ideal_overall=float(ideal_clipped.sum()/(tokens*self.n)),
                                    ideal_per_channel=(ideal_clipped/tokens).tolist()),
                      input_int8_clipping=float(input_clipped/input_elements),
                      input_int8_zero_rate=float(input_zero/input_elements),
                      position_metrics={g: dict(mse=e/c if c else None,
                          nmse=e/v if v else (0. if c and not e else None), output_elements=c)
                          for g, (e, v, c) in group_sums.items()},
                      worst_channels=np.argsort(-np.asarray(result['local_total']['channel_mse']), kind='stable')[:10].tolist())
        return result, vectors

    def run(self, replay, validation_replay=None):
        replay(self.observe)
        self.choose()
        calibration, vectors = self.evaluate(replay)
        if calibration['valid_tokens'] != self.tokens:
            raise ValueError('calibration replay changed valid token count')
        p = self.params
        report = dict(calibration=calibration, selection=self.selection,
                      s_X=self.sx, s_10=self.s10,
                      s_W=dict(min=float(self.sw.min()), max=float(self.sw.max()), mean=float(self.sw.mean()),
                               zero_channels=self.zero_channels, zero_channel_scale=1.0),
                      ratio_approximation=dict(tolerance=self.options.ratio_tolerance,
                                               max_relative_error=float(p['relative_error'].max()),
                                               max_absolute_error=float(p['absolute_error'].max()),
                                               per_channel_status=p['status'].tolist(),
                                               per_channel_relative_error=p['relative_error'].tolist()))
        if validation_replay is not None:
            report['validation'], _ = self.evaluate(validation_replay)
        return report, vectors