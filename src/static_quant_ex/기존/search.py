"""Bounded joint static scale search using population-weighted output MSE."""
import numpy as np
try:
    from .hardware import finite, positive_scale, ideal, check_lut_scale
except (ImportError, ModuleNotFoundError, ValueError):
    from .hardware import finite, positive_scale, ideal, check_lut_scale


DEFAULT_PERCENTILES = (99., 99.5, 99.9, 99.95, 99.99)


def activation_candidates(absmax, absolute_sample=None, percentiles=DEFAULT_PERCENTILES,
                          exact_percentiles=None):
    absmax = float(absmax)
    if not np.isfinite(absmax) or absmax < 0:
        raise ValueError('absmax must be finite and nonnegative')
    result = [dict(method='absmax', percentile=None, threshold=absmax,
                   s_X=absmax/127 if absmax else 1.)]
    for p in percentiles:
        if not np.isfinite(p) or not 0 < p <= 100:
            raise ValueError('percentile must be in (0,100]')
        if exact_percentiles is not None:
            threshold = float(exact_percentiles[p])
        else:
            values = finite(np.asarray(absolute_sample), 'percentile observations')
            if not values.size or np.any(values < 0):
                raise ValueError('nonempty absolute-value observations required')
            threshold = float(np.percentile(values, p))
        if not np.isfinite(threshold) or threshold < 0 or threshold > absmax*(1+1e-12):
            raise ValueError('invalid activation threshold')
        effective = threshold if threshold else absmax
        result.append(dict(method=f'p{p:g}', percentile=float(p), threshold=effective,
                           requested_threshold=threshold, zero_percentile_fallback=threshold == 0,
                           s_X=effective/127 if effective else 1.))
    return result


def joint_scale_search(read_weight, shape, sw, sample, candidates, baseline, profile,
                       options, accumulator=None, matmul=None):
    """All selection data come from the calibration sampler; no validation input.

    Common output grid uses the full-calibration absmax-path minmax baseline.
    Optional arithmetic callables are diagnostic acceleration hooks; the default
    is the existing NumPy INT64 accumulator and FP64 matrix product.
    """
    try:
        from .core import quantize_input, quantize_weight, exact_accumulator
    except (ImportError, ModuleNotFoundError, ValueError):
        from core import quantize_input, quantize_weight, exact_accumulator
    x, positions, weights, ids = sample.selected()
    x = finite(np.asarray(x, np.float64), 'calibration search inputs')
    positive_scale(baseline, 'minmax baseline')
    if accumulator is None:
        accumulator = lambda a, b: exact_accumulator(a, b, options.k_chunk)
    if matmul is None:
        matmul = lambda a, b: a @ b.T
    n, k = shape
    if x.shape[1] != k or sw.shape != (n,):
        raise ValueError('search shape mismatch')
    scales = [options.s10] if options.mode == 'fixed' else [baseline]
    if options.mode == 'mse':
        scales = sorted(set(baseline*t for t in (1.,)+options.thresholds), reverse=True)
    groups = {'bos': positions == 0, 'non_bos': positions != 0,
              'early': (positions > 0) & (positions <= sample.limit),
              'general': (positions > sample.limit) | (positions < 0), 'all': np.ones(len(x), bool)}
    rows = []
    for candidate in candidates:
        sx = positive_scale(candidate['s_X'], 'candidate s_X')
        xq = quantize_input(x, sx)
        activation_clip = float(weights @ (np.abs(x) > 127*sx).mean(axis=1))
        activation_zero = float(weights @ (xq == 0).mean(axis=1))
        params = [profile.approximate(sx*sw/s, options.ratio_tolerance) for s in scales]
        sums = np.zeros((len(scales), len(x)))
        clips = np.zeros_like(sums)
        energy = np.zeros(len(x))
        for start in range(0, n, options.n_chunk):
            sl = slice(start, min(n, start+options.n_chunk))
            w = finite(np.asarray(read_weight(sl), np.float64), 'search weight')
            wq, actual_sw = quantize_weight(w)
            np.testing.assert_array_equal(actual_sw, sw[sl])
            if k*127**2 > 2**31-1 and np.any(np.abs(xq.astype(np.int64)) @ np.abs(wq.astype(np.int64)).T > 2**31-1):
                raise OverflowError('cannot guarantee INT32 prefix safety')
            ref = matmul(x, w)
            acc = accumulator(xq, wq)
            energy += np.square(ref).sum(axis=1)
            for i, (s10, p) in enumerate(zip(scales, params)):
                raw = profile.apply(acc, p['multiplier'][sl], p['shift'][sl], saturate=False)
                y = np.clip(raw, -512, 511)*s10
                sums[i] += np.square(y-ref).sum(axis=1)
                clips[i] += ((raw < -512) | (raw > 511)).sum(axis=1)
        for i, (s10, p) in enumerate(zip(scales, params)):
            row = dict(candidate, s10=float(s10), s_10=float(s10),
                       mse=float(weights @ sums[i]/n), activation_clip_rate=activation_clip,
                       activation_zero_rate=activation_zero, int10_clip_rate=float(weights @ clips[i]/n),
                       bad_ratio_channels=int((p['status'] != 'ok').sum()),
                       max_ratio_relative_error=float(p['relative_error'].max()))
            for name, mask in groups.items():
                mass = float(weights[mask].sum())
                error = float(weights[mask] @ sums[i, mask])
                signal = float(weights[mask] @ energy[mask])
                row[name+'_mse'] = error/(mass*n) if mass else None
                row[name+'_nmse'] = error/signal if signal else (0. if not error and mass else None)
            if not np.isfinite(row['mse']):
                raise ValueError('nonfinite candidate objective')
            rows.append(row)
    eligible = [i for i, row in enumerate(rows) if row['bad_ratio_channels'] == 0]
    if not eligible:
        raise ValueError('no candidate has representable multiplier/shift parameters')
    chosen_index = min(eligible, key=lambda i: rows[i]['mse'])
    chosen = rows[chosen_index]
    params = profile.approximate(chosen['s_X']*sw/chosen['s10'], options.ratio_tolerance)
    check_lut_scale(chosen['s10'], options.fixed_lut_scale)
    selection = dict(mode=options.mode, objective='population-weighted final dequantized HW output MSE vs X_float @ W_float.T',
                     optimization='joint s_X/s_10' if options.mode != 'fixed' else 's_X only; constrained s_10',
                     minmax_baseline=baseline, output_grid_basis='full calibration original absmax INT8-pair absmax / 511',
                     selected_index=chosen_index, selected=chosen, candidates=rows,
                     sampling=sample.metadata(), selection_split='calibration',
                     runtime='one static s_X and one static s_10 per operation')
    return chosen['s_X'], chosen['s10'], params, selection

