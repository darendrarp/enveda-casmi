"""Tests for src/retrieval.py — library matching."""

from src.retrieval import LibraryIndex

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
ETHANOL = "CCO"


def _row(smi, ik14, precursor_mz, mzs, ints, mode="positive"):
    return {
        "normalized_smiles": smi,
        "inchikey14": ik14,
        "precursor_mz": precursor_mz,
        "ms2_mzs": mzs,
        "ms2_normalized_intensities": ints,
        "ionization_mode": mode,
    }


CAFF_MZS = [58.0283, 110.0712, 138.0661]
CAFF_INTS = [0.1, 0.5, 1.0]
ETOH_MZS = [29.0, 31.0, 45.0]
ETOH_INTS = [0.3, 1.0, 0.2]


def _build_index():
    rows = [
        _row(CAFFEINE, "RYYVLZVUVIJVGH", 195.0877, CAFF_MZS, CAFF_INTS),
        _row(CAFFEINE, "RYYVLZVUVIJVGH", 195.0876,
             [58.0283, 110.0712, 138.0661, 165.0],
             [0.1, 0.5, 1.0, 0.05]),
        _row(ETHANOL, "LFQSCWFLJHTTHZ", 47.0496, ETOH_MZS, ETOH_INTS),
    ]
    return LibraryIndex(max_per_structure=2).build(rows)


def test_library_build_counts():
    lib = _build_index()
    assert len(lib) == 3
    assert lib.n_structures == 2


def test_query_finds_exact_match_first():
    lib = _build_index()
    hits = lib.query(CAFF_MZS, CAFF_INTS, 195.0877)
    assert hits[0][0] == CAFFEINE
    assert hits[0][1] > 0.99  # near-identical spectrum
    # Ethanol must not appear: precursor mass window excludes it.
    assert all(s != ETHANOL for s, _ in hits)


def test_query_mass_window_blocks_far_precursor():
    lib = _build_index()
    assert lib.query(CAFF_MZS, CAFF_INTS, 500.0) == []


def test_query_molecule_aggregates_spectra():
    lib = _build_index()
    spectra = [
        {"ms2_mzs": CAFF_MZS, "ms2_normalized_intensities": CAFF_INTS,
         "precursor_mz": 195.0877},
        {"ms2_mzs": [58.0283, 110.0712],
         "ms2_normalized_intensities": [0.2, 1.0], "precursor_mz": 195.0878},
    ]
    hits = lib.query_molecule(spectra)
    assert hits[0][0] == CAFFEINE


def test_empty_library():
    lib = LibraryIndex().build([])
    assert len(lib) == 0
    assert lib.query(CAFF_MZS, CAFF_INTS, 195.0877) == []
