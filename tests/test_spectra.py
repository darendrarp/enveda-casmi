"""Tests for src/spectra.py — peak cleaning and binned vectors."""

import numpy as np

from src.spectra import clean_peaks, peak_features, spectrum_vector


def test_clean_peaks_top_n_and_sort():
    mzs = [300.0, 100.0, 200.0, 150.0]
    ints = [10.0, 100.0, 50.0, 25.0]
    mz, it = clean_peaks(mzs, ints, precursor_mz=400.0, top_n=3)
    assert len(mz) == 3
    assert list(mz) == sorted(mz)  # sorted by m/z ascending
    assert it.max() == 1.0  # normalised
    assert 300.0 not in mz  # least intense dropped


def test_clean_peaks_drops_precursor_and_bad_values():
    mzs = [100.0, 399.9, 500.0, -5.0, float("nan")]
    ints = [10.0, 10.0, 10.0, 10.0, 10.0]
    mz, it = clean_peaks(mzs, ints, precursor_mz=400.0)
    assert list(mz) == [100.0]
    mz2, _ = clean_peaks([], [], precursor_mz=400.0)
    assert len(mz2) == 0


def test_spectrum_vector_normalised_and_deterministic():
    mzs = [100.0, 200.0, 300.0]
    ints = [1.0, 0.5, 0.25]
    v1 = spectrum_vector(mzs, ints, 400.0)
    v2 = spectrum_vector(mzs, ints, 400.0)
    assert v1.shape == (1, 150_000)
    assert abs(float((v1 @ v1.T).toarray()[0, 0]) - 1.0) < 1e-5  # L2 norm 1
    assert (v1 != v2).nnz == 0  # deterministic
    # Identical spectra -> cosine 1; disjoint spectra -> cosine 0.
    assert abs(float((v1 @ v2.T).toarray()[0, 0]) - 1.0) < 1e-5
    v3 = spectrum_vector([600.0, 700.0], [1.0, 1.0], 950.0)
    # No fragment-bin or neutral-loss-bin overlap with v1 -> cosine 0.
    assert float((v1 @ v3.T).toarray()[0, 0]) == 0.0


def test_peak_features_shape():
    feats = peak_features([100.0, 200.0], [1.0, 0.5], 400.0)
    assert feats.shape == (2, 4)
    assert feats.dtype == np.float32
    assert np.all(np.isfinite(feats))
    empty = peak_features([], [], 400.0)
    assert empty.shape == (0, 4)
