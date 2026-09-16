"""Bounded-memory, NumPy INT64 reference and streaming local calibration."""
from dataclasses import dataclass
import numpy as np
from .hardware import finite, positive_scale, integers, ideal, check_lut_scale


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
    def __init__(self, seed=0, percentile=None, capacity=65536):
        if percentile is not None and not 0 < percentile <= 100:
            raise ValueError('percentile must be in (0,100]')
        self.percentile = percentile
        self.sample = Reservoir(capacity, seed) if percentile is not None else None
        self.absmax, self.tokens = 0.0, 0

    def add(self, x):
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
        threshold = self.absmax if self.sample is None else float(np.percentile(self.sample.values, self.percentile))
        return threshold / 127 if threshold > 0 else (self.absmax / 127 if self.absmax > 0 else 1.0)

    def metadata(self):
        return dict(method='absmax' if self.sample is None else 'sampled_percentile',
                    percentile=self.percentile, absmax=self.absmax, valid_tokens=self.tokens,
                    s_X=self.scale(), all_zero=self.absmax == 0,
                    zero_scale_policy='all zero -> 1; zero percentile on nonzero tensor -> absmax/127',
                    sampling=None if self.sample is None else self.sample.metadata())


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

    def __post_init__(self):
        if not np.isfinite(self.ratio_tolerance) or self.ratio_tolerance < 0:
            raise ValueError('ratio tolerance must be finite and nonnegative')
        if min(self.m_chunk, self.n_chunk, self.k_chunk, self.search_rows) < 1:
            raise ValueError('chunk sizes and search rows must be positive')
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
    def __init__(self, name, shape, read_weight, sx, profile, options):
        self.name, self.n, self.k = name, int(shape[0]), int(shape[1])
        self.read_weight, self.sx, self.profile, self.options = read_weight, positive_scale(sx, 's_X'), profile, options
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
        self.sample = Reservoir(options.search_rows, options.seed)
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

    def observe(self, x):
        if not len(x):
            return
        self.tokens += len(x)
        self.sample.add(x)
        for sl, acc, _ in self.blocks(x):
            real = acc * (self.sx*self.sw[sl])
            self.peaks[sl] = np.maximum(self.peaks[sl], np.abs(real).max(axis=0))

    def choose(self):
        if not self.tokens:
            raise ValueError(f'{self.name}: no valid tokens in Pass B')
        o = self.options
        baseline = float(self.peaks.max()/511) if self.peaks.max() else 1.0
        scales = [o.s10] if o.mode == 'fixed' else [baseline]
        if o.mode == 'mse':
            scales = sorted(set([baseline] + [baseline*t for t in o.thresholds]), reverse=True)
        parameters = [self.profile.approximate(self.sx*self.sw/s, o.ratio_tolerance) for s in scales]
        scores = np.zeros(len(scales))
        count = 0
        for sl, acc, _ in self.blocks(self.sample.values):
            target = acc * (self.sx*self.sw[sl])
            count += acc.size
            for i, (s, p) in enumerate(zip(scales, parameters)):
                q = self.profile.apply(acc, p['multiplier'][sl], p['shift'][sl])
                scores[i] += np.square(q*s-target).sum()
        index = int(scores.argmin())
        self.s10, self.params = float(scales[index]), parameters[index]
        check_lut_scale(self.s10, o.fixed_lut_scale)
        self.selection = dict(mode=o.mode, minmax_baseline=baseline, objective='profile output vs dequantized INT32',
                              candidates=[dict(s10=float(s), mse=float(score/count)) for s, score in zip(scales, scores)],
                              sampling=self.sample.metadata())
        return self.params

    def evaluate(self, replay):
        metrics = {key: Metrics(self.n) for key in ('requantization', 'local_total', 'parameter_approximation')}
        clipped = np.zeros(self.n, dtype=np.int64)
        ideal_clipped = np.zeros(self.n, dtype=np.int64)
        tokens, input_clipped, input_elements = 0, 0, 0
        vectors = []
        def observe(x):
            nonlocal tokens, input_clipped, input_elements
            if not len(x):
                return
            tokens += len(x)
            input_clipped += int(np.count_nonzero(np.abs(x/self.sx) > 127))
            input_elements += x.size
            for sl, acc, original in self.blocks(x, original=True):
                p = self.params
                raw = self.profile.apply(acc, p['multiplier'][sl], p['shift'][sl], saturate=False)
                hw = np.clip(raw, -512, 511).astype(np.int16)
                ref = ideal(acc, p['ratio'][sl])
                metrics['requantization'].add(sl, hw*self.s10 - acc*(self.sx*self.sw[sl]))
                metrics['local_total'].add(sl, hw*self.s10 - original)
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
        result.update(valid_tokens=tokens, int32_overflow=False,
                      overflow_check='INT64 partial sums; for large K conservative absolute-product prefix bound',
                      clipping=dict(definition='rounded result before INT10 saturation outside [-512,511]',
                                    overall=float(clipped.sum()/(tokens*self.n)),
                                    per_channel=(clipped/tokens).tolist(),
                                    ideal_overall=float(ideal_clipped.sum()/(tokens*self.n)),
                                    ideal_per_channel=(ideal_clipped/tokens).tolist()),
                      input_int8_clipping=float(input_clipped/input_elements),
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
