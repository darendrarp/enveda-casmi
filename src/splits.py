"""Splitting helpers.

The competition's hidden test set contains molecules never seen in
training, so the honest local validation is a *structure-disjoint* split:
all spectra of a molecule (keyed by InChIKey14) go to exactly one side.
A random spectrum split is also provided as a sanity check (the library
matcher should score high there, proving the harness works).
"""

from __future__ import annotations

import random


def structure_disjoint_split(keys, val_frac=0.1, seed=42, max_val=None):
    """Split unique structure keys into train/val key sets.

    ``keys``: iterable of InChIKey14 values. Returns ``(train_keys, val_keys)``.
    """
    uniq = sorted(set(keys))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * val_frac)
    if max_val is not None:
        n_val = min(n_val, max_val)
    val = set(uniq[:n_val])
    train = set(uniq[n_val:])
    return train, val


def random_split(n, val_frac=0.1, seed=42):
    """Split ``range(n)`` indices randomly. Returns ``(train_idx, val_idx)``."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = int(n * val_frac)
    return idx[n_val:], idx[:n_val]


class ReservoirSampler:
    """Uniform reservoir sample of up to ``capacity`` items from a stream."""

    def __init__(self, capacity, seed=42):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.items: list = []
        self.n_seen = 0

    def add(self, item):
        self.n_seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.n_seen)
            if j < self.capacity:
                self.items[j] = item

    def __len__(self):
        return len(self.items)

