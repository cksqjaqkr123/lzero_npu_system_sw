"""Explicit SOFTWARE profiles + 16-bin GeLU LUT support - FIXED select_bin closest."""
from dataclasses import asdict, dataclass
import numpy as np
import json
from pathlib import Path as _Path

def finite(value, name):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        where = np.argwhere(~np.isfinite(value))[0].tolist()
        raise ValueError(f"{name}: NaN/Inf at index {where}")
    return value

def positive_scale(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return float(value)

def integers(value, lo, hi, name):
    value = np.asarray(value)
    if value.dtype.kind not in 'iu' or np.any(value < lo) or np.any(value > hi):
        raise ValueError(f"{name} must contain integers in [{lo}, {hi}]")
    return value.astype(np.int64)

@dataclass(frozen=True)
class Profile:
    name: str = 'generic-rne'
    shift_correction: int = 0
    rounding: str = 'nearest_even'

    def __post_init__(self):
        if (self.name, self.shift_correction, self.rounding) not in {
            ('generic-rne', 0, 'nearest_even'), ('legacy-guess', 2, 'floor')
        }:
            raise ValueError('Only explicitly specified software profiles are supported')

    def metadata(self):
        return dict(asdict(self), verification='unverified', rtl_bit_exact=False,
                    source='hardware.py; 16-bin support FIXED',
                    multiplier='unsigned 16 bit', shift='unsigned 5 bit',
                    effective_shift='max(shift - shift_correction, 0)',
                    lut_index="10-bit two's complement: signed_q10 & 1023",
                    layout={'zp': [0, 7], 'shift': [8, 12], 'reserved_zero': [13, 15],
                            'multiplier': [16, 31]}, endian='little')

    def effective(self, shift):
        return np.maximum(integers(shift, 0, 31, 'shift') - self.shift_correction, 0)

    def pack(self, multiplier, shift, zp=0):
        m = integers(multiplier, 0, 65535, 'multiplier')
        s = integers(shift, 0, 31, 'shift')
        z = integers(zp, -128, 127, 'zero point')
        return ((m << 16) | (s << 8) | (z & 255)).astype(np.uint32)

    def unpack(self, words):
        w = integers(words, 0, 2**32 - 1, 'packed words')
        if np.any(w & 0xE000):
            raise ValueError('reserved bits [15:13] must be zero')
        z = w & 255
        return w >> 16, (w >> 8) & 31, np.where(z >= 128, z - 256, z)

    def apply(self, accumulator, multiplier, shift, zp=0, saturate=True):
        a = integers(accumulator, -(2**31), 2**31-1, 'accumulator')
        m = integers(multiplier, 0, 65535, 'multiplier')
        z = integers(zp, -128, 127, 'zero point')
        s = self.effective(shift)
        product = a * m
        q = np.right_shift(product, s)
        if self.rounding == 'nearest_even':
            remainder = product - np.left_shift(q, s)
            denominator = np.left_shift(np.ones_like(s), s)
            twice = remainder * 2
            q = q + ((twice > denominator) | ((twice == denominator) & ((q & 1) != 0)))
        q = q + z
        return np.clip(q, -512, 511).astype(np.int16) if saturate else q

    def approximate(self, ratios, tolerance=1e-3):
        r = finite(np.asarray(ratios, dtype=np.float64), 'ratio')
        if r.ndim != 1 or np.any(r < 0):
            raise ValueError('ratios must be a nonnegative channel vector')
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('ratio tolerance must be finite and nonnegative')
        shifts = np.arange(32, dtype=np.int64)
        factors = np.exp2(self.effective(shifts))
        candidates = np.rint(np.minimum(r[:, None], 65535 / factors) * factors).astype(np.int64)
        realized = candidates / factors
        errors = np.abs(realized - r[:, None])
        index = errors.argmin(axis=1)
        rows = np.arange(len(r))
        m, s = candidates[rows, index], shifts[index]
        actual = realized[rows, index]
        relative = np.divide(np.abs(actual-r), r, out=np.zeros_like(r), where=r != 0)
        status = np.full(r.shape, 'ok', dtype='<U24')
        status[relative > tolerance] = 'inaccurate'
        status[(r > 0) & (m == 0)] = 'underflow_to_zero'
        status[r > 65535] = 'above_maximum'
        return dict(multiplier=m, shift=s, ratio=r, realized_ratio=actual,
                    relative_error=relative, absolute_error=np.abs(actual-r), status=status)

def get_profile(name):
    if name == 'generic-rne':
        return Profile()
    if name == 'legacy-guess':
        return Profile(name, 2, 'floor')
    raise ValueError(f'Unknown hardware profile: {name}')

def ideal(accumulator, ratios):
    a = integers(accumulator, -(2**31), 2**31-1, 'accumulator')
    return np.clip(np.rint(a * finite(ratios, 'ratio')), -512, 511).astype(np.int16)

def lut_index(q10):
    return (integers(q10, -512, 511, 'INT10') & 1023).astype(np.uint16)

def check_lut_scale(s10, fixed_lut_scale):
    positive_scale(s10, 's10')
    if fixed_lut_scale is not None:
        positive_scale(fixed_lut_scale, 'LUT input scale')
        if not np.isclose(s10, fixed_lut_scale, rtol=1e-12, atol=0):
            raise ValueError(f's10={s10} is incompatible with fixed LUT scale={fixed_lut_scale}')

class GeluLut:
    def __init__(self, lut_path, scale_map_path):
        raw = _Path(lut_path).read_bytes()
        with open(scale_map_path) as f:
            scale_map = json.load(f)
        self.bin_scales = {}
        for m in scale_map:
            self.bin_scales[m['bin']] = (m['in_scale'], m['out_scale'], m['pre_p999'], m['post_p999'])
        self.sorted_bins = sorted([(bid, ins, outs, pre) for bid, (ins, outs, pre, _) in self.bin_scales.items()], key=lambda x: x[3])

        if len(raw) == 8192:
            self.lut = np.frombuffer(raw, dtype=np.uint8).reshape(8, 1024)
            self.mode = '10_8_8bin'
            self.num_bins = 8
        elif len(raw) == 16384 and len(scale_map)==16:
            self.lut = np.frombuffer(raw, dtype=np.uint8).reshape(16, 1024)
            self.mode = '10_8_16bin'
            self.num_bins = 16
        elif len(raw) == 16384 and len(scale_map)==8:
            self.lut = np.frombuffer(raw, dtype=np.int16).reshape(8, 1024)
            self.mode = '10_10_8bin'
            self.num_bins = 8
        elif len(raw) == 32768:
            self.lut = np.frombuffer(raw, dtype=np.int16).reshape(16, 1024)
            self.mode = '10_10_16bin'
            self.num_bins = 16
        else:
            if len(raw) % 1024 == 0:
                bins = len(raw)//1024
                if bins==16:
                    self.lut = np.frombuffer(raw, dtype=np.uint8).reshape(16,1024) if len(raw)==16384 else np.frombuffer(raw, dtype=np.int16).reshape(16,1024)
                    self.mode = f'10_8_{bins}bin' if len(raw)==16384 else f'10_10_{bins}bin'
                    self.num_bins = bins
                else:
                    self.lut = np.frombuffer(raw, dtype=np.uint8).reshape(8,1024)
                    self.mode = '10_8_8bin'
                    self.num_bins = 8
            else:
                raise ValueError(f"GeLU BIN 크기 {len(raw)}B 지원 안함")

        self.lut_path = str(lut_path)
        self.scale_map_path = str(scale_map_path)
        self.total_bytes = len(raw)

    def select_bin(self, s10_or_pre_p999):
        # FIXED: closest pre_p999
        if s10_or_pre_p999 > 0.2:
            pre = float(s10_or_pre_p999)
        else:
            pre = float(s10_or_pre_p999) * 511.0
        best = None
        best_diff = float('inf')
        for bid, ins, outs, bin_pre in self.sorted_bins:
            diff = abs(bin_pre - pre)
            if diff < best_diff:
                best_diff = diff
                best = (bid, ins, outs, bin_pre)
        bid, ins, outs, _ = best
        return bid, ins, outs

    def select_bin_covering(self, s10_or_pre_p999):
        if s10_or_pre_p999 > 0.2:
            real_in_scale = s10_or_pre_p999 / 511.0
        else:
            real_in_scale = s10_or_pre_p999
        for bid, ins, outs, pre in self.sorted_bins:
            if ins >= real_in_scale - 1e-9:
                return bid, ins, outs
        bid, ins, outs, _ = self.sorted_bins[-1]
        return bid, ins, outs

    def select_bin_for_layer(self, layer_id, pre_p999, optimal_map=None):
        if optimal_map and str(layer_id) in optimal_map:
            bid = int(optimal_map[str(layer_id)])
            if bid in self.bin_scales:
                ins, outs, _, _ = self.bin_scales[bid]
                return bid, ins, outs
        return self.select_bin(pre_p999)

    def lookup(self, q10, bin_id=None, s10=None):
        q10 = np.asarray(q10)
        if bin_id is None:
            if s10 is None:
                raise ValueError("bin_id 또는 s10 중 하나는 필요")
            bin_id, _, _ = self.select_bin(s10)
        lut_index = (q10 & 1023).astype(np.uint16)
        q_out = self.lut[bin_id, lut_index]
        return q_out, lut_index, bin_id

    def dequant(self, q_out, out_scale):
        return q_out.astype(np.float32) / out_scale

    def evaluate(self, pre_float, post_true_float, s10=None, bin_id=None):
        if s10 is None:
            pre_p999 = float(np.percentile(np.abs(pre_float), 99.9))
            s10 = pre_p999 / 511.0
        if bin_id is None:
            bin_id, in_scale, out_scale = self.select_bin(s10)
        else:
            in_scale, out_scale = self.bin_scales[bin_id][:2]

        q10 = np.clip(np.rint(pre_float / in_scale), -512, 511).astype(np.int16)
        q_out, lut_index, _ = self.lookup(q10, bin_id=bin_id)
        float_lut_out = self.dequant(q_out, out_scale)

        diff = float_lut_out - post_true_float.astype(np.float32)
        abs_diff = np.abs(diff)
        metrics = dict(
            bin_id=int(bin_id), in_scale=float(in_scale), out_scale=float(out_scale),
            s10=float(s10) if isinstance(s10, (int,float)) else float(in_scale),
            mse=float(np.mean(diff**2)), rmse=float(np.sqrt(np.mean(diff**2))),
            mae=float(np.mean(abs_diff)), max_abs=float(np.max(abs_diff)),
            p99_abs=float(np.percentile(abs_diff, 99)), p999_abs=float(np.percentile(abs_diff, 99.9)),
        )
        return metrics, dict(q10=q10, lut_index=lut_index, q_out=q_out, float_lut_out=float_lut_out, diff=diff)

    def evaluate_all_bins(self, pre_float, post_true_float):
        results = []
        for bid in self.bin_scales.keys():
            metrics, _ = self.evaluate(pre_float, post_true_float, bin_id=bid)
            results.append(metrics)
        results_sorted = sorted(results, key=lambda x: x['mse'])
        return results_sorted

    def metadata(self):
        return dict(lut_path=self.lut_path, scale_map_path=self.scale_map_path,
                    mode=self.mode, total_bytes=self.total_bytes, num_bins=self.num_bins,
                    bins=[dict(bin=bid, in_scale=ins, out_scale=outs, pre_p999=pre, post_p999=post)
                          for bid, (ins, outs, pre, post) in self.bin_scales.items()])

def load_gelu_lut(lut_path, scale_map_path):
    return GeluLut(lut_path, scale_map_path)
