"""Explicit SOFTWARE profiles. Neither profile is a verified QuantAct RTL ABI."""
from dataclasses import asdict, dataclass
import numpy as np


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
                    source='src/static_quant/hardware.py; software definition only; no RTL found',
                    historical_layout_source='user request, unconfirmed design history',
                    multiplier='unsigned 16 bit', shift='unsigned 5 bit',
                    effective_shift='max(shift - shift_correction, 0)',
                    zero_point='signed 8 bit, added AFTER rounded shift; exports use zero',
                    product='signed 48 bit exact; INT32 * UInt16 fits; no truncation or wrap',
                    right_shift='arithmetic (floor for negatives)',
                    saturation='signed INT10 [-512,511], after shift and zero point',
                    lut_index='10-bit two\'s complement: signed_q10 & 1023',
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
        product = a * m  # bounded signed 48 bit, represented exactly in int64
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
        # Clamp candidates explicitly, never mask an out-of-range field.
        candidates = np.rint(np.minimum(r[:, None], 65535 / factors) * factors).astype(np.int64)
        realized = candidates / factors
        errors = np.abs(realized - r[:, None])
        index = errors.argmin(axis=1)  # deterministic lowest field on ties
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
