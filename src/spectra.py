"""Spectrum preprocessing: peak cleaning and binned vectorisation.

Conventions (matching the competition data):
- Peak lists arrive as two aligned arrays: ``ms2_mzs`` and
  ``ms2_normalized_intensities`` (largest peak = 1.0 per spectrum).
- Intensities are square-root transformed before use (standard practice:
  it compresses the dynamic range so small diagnostic peaks still matter).
- Binned vectors concatenate a fragment-m/z histogram with a neutral-loss
  histogram (precursor_mz - fragment_mz), both L2-normalised. Neutral-loss
  bins make the representation robust to precursor-mass shifts.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def clean_peaks(mzs, ints, precursor_mz, top_n: int = 128,
                drop_above_precursor: bool = True):
    """Clean one spectrum.

    - drops non-finite / non-positive m/z and intensities
    - drops peaks at/above the precursor (uninformative precursor peak)
    - keeps the ``top_n`` most intense peaks
    - square-root transforms intensities, rescales max to 1.0
    - returns peaks sorted by m/z ascending

    Returns ``(mzs, ints)`` as float32 numpy arrays (possibly empty).
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    if mzs.shape != ints.shape or mzs.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    ok = np.isfinite(mzs) & np.isfinite(ints) & (mzs > 0) & (ints > 0)
    mzs, ints = mzs[ok], ints[ok]
    if drop_above_precursor and precursor_mz is not None and np.isfinite(precursor_mz):
        keep = mzs < precursor_mz - 0.5
        mzs, ints = mzs[keep], ints[keep]
    if mzs.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    if mzs.size > top_n:
        idx = np.argpartition(ints, -top_n)[-top_n:]
        mzs, ints = mzs[idx], ints[idx]
    ints = np.sqrt(ints)
    ints = ints / ints.max()
    order = np.argsort(mzs)
    return mzs[order].astype(np.float32), ints[order].astype(np.float32)


def spectrum_vector(mzs, ints, precursor_mz, bin_width: float = 0.01,
                    mz_max: float = 1000.0, nl_max: float = 500.0,
                    top_n: int = 128):
    """Binned fragment + neutral-loss vector as a 1-row CSR matrix.

    Dimension is ``mz_max/bin_width + nl_max/bin_width`` (150,000 at the
    defaults). The vector is L2-normalised so a dot product is the cosine
    similarity.
    """
    mz, it = clean_peaks(mzs, ints, precursor_mz, top_n=top_n)
    n_mz = int(mz_max / bin_width)
    n_nl = int(nl_max / bin_width)
    dim = n_mz + n_nl
    if mz.size == 0:
        return sparse.csr_matrix((1, dim), dtype=np.float32)
    frag_bins = np.floor(mz / bin_width).astype(np.int64)
    ok_frag = frag_bins < n_mz
    nl = precursor_mz - mz
    nl_bins = np.floor(nl / bin_width).astype(np.int64) + n_mz
    ok_nl = (nl >= 0) & (nl_bins < dim)
    cols = np.concatenate([frag_bins[ok_frag], nl_bins[ok_nl]])
    data = np.concatenate([it[ok_frag], it[ok_nl]]).astype(np.float32)
    vec = sparse.csr_matrix(
        (data, (np.zeros_like(cols), cols)), shape=(1, dim), dtype=np.float32
    )
    vec.sum_duplicates()
    norm = sparse.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def peak_features(mzs, ints, precursor_mz, max_peaks: int = 128):
    """Peak-set features for neural encoders.

    Returns a ``(P, 4)`` float32 array with one row per peak:
    ``[mz/1000, sqrt_intensity, mz/precursor_mz, log10(intensity+eps)]``,
    plus the number of peaks P (<= max_peaks). Peaks are sorted by m/z.
    """
    mz, it = clean_peaks(mzs, ints, precursor_mz, top_n=max_peaks)
    if mz.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    precursor = precursor_mz if precursor_mz and np.isfinite(precursor_mz) else 1000.0
    feats = np.stack(
        [
            mz / 1000.0,
            it,
            mz / precursor,
            np.log10(it + 1e-6),
        ],
        axis=1,
    ).astype(np.float32)
    return feats
