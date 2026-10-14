"""Deterministic position-stratified rows; positions never affect runtime scales."""
import numpy as np
try:
    from .hardware import finite
except (ImportError, ModuleNotFoundError, ValueError):
    from hardware import finite


def valid_positions(mask):
    mask = __import__('numpy').asarray(mask)
    if mask.ndim != 2 or not __import__('numpy').isin(mask, [0, 1]).all():
        raise ValueError('expected a binary [batch, sequence] mask')
    return (__import__('numpy').cumsum(mask, axis=1)-1)[mask.astype(bool)].astype(__import__('numpy').int64)


class RepresentativeRowSampler:
    def __init__(self, capacity=64, seed=0, bos_rows=16, early_rows=16,
                 early_position_limit=8, representative=True):
        if capacity < 1 or min(seed, bos_rows, early_rows, early_position_limit) < 0:
            raise ValueError('positive capacity and nonnegative sampling options required')
        self.capacity, self.bos_rows, self.early_rows = capacity, bos_rows, early_rows
        self.limit, self.representative = early_position_limit, representative
        self.seed = seed
        self.rng = [np.random.default_rng(seed+i) for i in (1, 2, 0)]
        self.pools = [None]*3
        self.counts = np.zeros(3, dtype=np.int64)
        self.seen = self.unknown = 0
        self.width = None

    def add(self, x, positions=None):
        x = finite(__import__('numpy').asarray(x), 'search rows')
        if x.ndim != 2:
            raise ValueError('search rows must be [M,K]')
        if self.width is not None and self.width != x.shape[1]:
            raise ValueError('search row width changed')
        self.width = x.shape[1]
        if positions is None:
            positions = __import__('numpy').full(len(x), -1, dtype=__import__('numpy').int64)
        positions = __import__('numpy').asarray(positions)
        if positions.shape != (len(x),) or positions.dtype.kind not in 'iu' or __import__('numpy').any(positions < -1):
            raise ValueError('one integer token position per row required')
        ids = __import__('numpy').arange(self.seen, self.seen+len(x), dtype=__import__('numpy').int64)
        self.seen += len(x)
        self.unknown += int((positions < 0).sum())
        categories = __import__('numpy').full(len(x), 2)
        if self.representative:
            if self.bos_rows:
                categories[positions == 0] = 0
            if self.early_rows:
                categories[(positions > 0) & (positions <= self.limit)] = 1
        for group in range(3):
            selected = __import__('numpy').flatnonzero(categories == group)
            self.counts[group] += len(selected)
            for start in range(0, len(selected), self.capacity):
                ii = selected[start:start+self.capacity]
                data = (self.rng[group].random(len(ii)), x[ii], positions[ii], ids[ii])
                if self.pools[group] is not None:
                    data = tuple(__import__('numpy').concatenate([a, b]) for a, b in zip(self.pools[group], data))
                keep = __import__('numpy').argsort(data[0], kind='stable')[:self.capacity]
                self.pools[group] = tuple(a[keep] for a in data)

    def selected(self):
        import numpy as np
        if not self.seen:
            raise ValueError('no search rows')
        available = np.minimum(self.counts, self.capacity)
        allocation = (available > 0).astype(np.int64)
        if allocation.sum() > self.capacity:
            raise ValueError('search_rows too small to represent every nonempty stratum')
        left = min(self.capacity, self.seen)-int(allocation.sum())
        for g, desired in [(0, self.bos_rows), (1, self.early_rows),
                           (2, self.capacity), (1, self.capacity), (0, self.capacity)]:
            add = min(left, max(0, min(desired, int(available[g]))-int(allocation[g])))
            allocation[g] += add
            left -= add
        pieces = [(g, self.pools[g], int(n)) for g, n in enumerate(allocation) if n]
        x = np.concatenate([pool[1][:n] for g, pool, n in pieces])
        pos = np.concatenate([pool[2][:n] for g, pool, n in pieces])
        ids = np.concatenate([pool[3][:n] for g, pool, n in pieces])
        weights = np.concatenate([np.full(n, self.counts[g]/n/self.seen) for g, pool, n in pieces])
        if self.representative and self.bos_rows and self.counts[0] and not np.any(pos == 0):
            raise AssertionError('BOS representation invariant violated')
        return x, pos, weights, ids

    @property
    def values(self):
        return self.selected()[0]

    def metadata(self):
        import numpy as np
        x, pos, weights, ids = self.selected()
        unique, counts = np.unique(pos, return_counts=True)
        return dict(method='position_stratified' if self.representative else 'uniform',
                    seed=self.seed,
                    capacity=self.capacity, observed=self.seen, retained=len(x),
                    bos_rows=int((pos == 0).sum()),
                    early_rows=int(((pos > 0) & (pos <= self.limit)).sum()),
                    general_rows=int((pos > self.limit).sum()), unknown_rows=int((pos < 0).sum()),
                    unknown_observed_rows=self.unknown, early_position_limit=self.limit,
                    requested_bos_rows=self.bos_rows, requested_early_rows=self.early_rows,
                    population_by_sampling_stratum=self.counts.tolist(),
                    position_histogram={str(p): int(n) for p, n in zip(unique, counts)},
                    selected_row_ids=ids.tolist(), row_objective_weights=weights.tolist(),
                    weighting='population stratum fraction / retained stratum rows; sums to one',
                    position_semantics='valid position zero (BOS proxy); no position-dependent runtime quantization')

