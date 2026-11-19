"""Tests for src/pipeline.py — RRF fusion and molecule prediction."""

from src.fingerprint import CandidatePool
from src.pipeline import pool_fallback, predict_molecule, rrf_fuse
from src.retrieval import LibraryIndex

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
ETHANOL = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_rrf_fuse_prefers_consensus():
    # Caffeine appears high in both lists -> beats single-list leaders.
    fused = rrf_fuse([[CAFFEINE, ETHANOL], [ASPIRIN, CAFFEINE]])
    assert fused[0] == CAFFEINE
    assert len(fused) == 3


def test_rrf_fuse_dedupes_by_inchikey14():
    fused = rrf_fuse([["C[C@H](O)C(=O)O"], ["C[C@@H](O)C(=O)O"]])
    assert len(fused) == 1


def test_predict_molecule_library_only():
    rows = [{
        "normalized_smiles": CAFFEINE,
        "inchikey14": "RYYVLZVUVIJVGH",
        "precursor_mz": 195.0877,
        "ms2_mzs": [58.0283, 110.0712, 138.0661],
        "ms2_normalized_intensities": [0.1, 0.5, 1.0],
        "ionization_mode": "positive",
    }]
    lib = LibraryIndex().build(rows)
    spectra = [{
        "ms2_mzs": [58.0283, 110.0712, 138.0661],
        "ms2_normalized_intensities": [0.1, 0.5, 1.0],
        "precursor_mz": 195.0877,
        "adduct": "[M+H]+",
    }]
    out = predict_molecule(spectra, lib_index=lib)
    assert out[0] == CAFFEINE


def test_predict_molecule_no_components():
    assert predict_molecule([{"ms2_mzs": [], "ms2_normalized_intensities": [],
                              "precursor_mz": 100.0}]) == []


def test_pool_fallback_mass_matched():
    pool = CandidatePool([CAFFEINE, ETHANOL, ASPIRIN]).add_exact_masses()
    spectra = [{"precursor_mz": 195.0877, "adduct": "[M+H]+",
                "ms2_mzs": [], "ms2_normalized_intensities": []}]
    out = pool_fallback(pool, spectra, top_n=25)
    assert out[0] == CAFFEINE  # only caffeine matches ~194.08 Da


def test_pool_fallback_no_mass_match_returns_popular():
    pool = CandidatePool([CAFFEINE, ETHANOL, ASPIRIN]).add_exact_masses()
    spectra = [{"precursor_mz": 9999.0, "adduct": "[M+H]+",
                "ms2_mzs": [], "ms2_normalized_intensities": []}]
    out = pool_fallback(pool, spectra, top_n=2)
    assert out == [CAFFEINE, ETHANOL]  # pool order = popularity order


def test_predict_molecule_tops_up_to_25():
    pool = CandidatePool([CAFFEINE, ETHANOL, ASPIRIN]).add_exact_masses()
    rows = [{
        "normalized_smiles": CAFFEINE,
        "inchikey14": "RYYVLZVUVIJVGH",
        "precursor_mz": 195.0877,
        "ms2_mzs": [58.0283, 110.0712, 138.0661],
        "ms2_normalized_intensities": [0.1, 0.5, 1.0],
        "ionization_mode": "positive",
    }]
    lib = LibraryIndex().build(rows)
    spectra = [{
        "ms2_mzs": [58.0283, 110.0712, 138.0661],
        "ms2_normalized_intensities": [0.1, 0.5, 1.0],
        "precursor_mz": 195.0877,
        "adduct": "[M+H]+",
    }]
    out = predict_molecule(spectra, lib_index=lib, pool=pool)
    # Library gives 1 hit; fallback tops up with the other pool structures.
    assert out[0] == CAFFEINE
    assert len(out) == 3
    assert set(out) == {CAFFEINE, ETHANOL, ASPIRIN}


def test_library_gate_excludes_weak_matches():
    # Query shares only one low peak with the library spectrum -> cosine
    # ~0.25, below the default gate (0.5). The library list must not enter
    # the fusion: with a pool that lacks caffeine, caffeine must not appear.
    pool = CandidatePool([ETHANOL, ASPIRIN]).add_exact_masses()
    rows = [{
        "normalized_smiles": CAFFEINE,
        "inchikey14": "RYYVLZVUVIJVGH",
        "precursor_mz": 195.0877,
        "ms2_mzs": [58.0283, 110.0712, 138.0661],
        "ms2_normalized_intensities": [0.1, 0.5, 1.0],
        "ionization_mode": "positive",
    }]
    lib = LibraryIndex().build(rows)
    spectra = [{
        "ms2_mzs": [58.0283],
        "ms2_normalized_intensities": [1.0],
        "precursor_mz": 195.0877,
        "adduct": "[M+H]+",
    }]
    hits = lib.query_molecule(spectra)
    assert hits and hits[0][0] == CAFFEINE
    assert hits[0][1] < 0.5  # weak match, as designed
    gated = predict_molecule(spectra, lib_index=lib, pool=pool)
    assert CAFFEINE not in gated  # gated out; fallback fills from the pool
    assert len(gated) == 2
    ungated = predict_molecule(spectra, lib_index=lib, pool=pool, lib_gate=0.0)
    assert ungated[0] == CAFFEINE  # without the gate the weak hit fuses in
