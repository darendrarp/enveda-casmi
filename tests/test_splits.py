"""Tests for src/splits.py and streaming library build."""

from src.retrieval import LibraryIndex
from src.splits import ReservoirSampler, structure_disjoint_split

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


def _row(smi, ik14, precursor_mz, n_peaks):
    return {
        "normalized_smiles": smi,
        "inchikey14": ik14,
        "precursor_mz": precursor_mz,
        "ms2_mzs": [100.0 + i for i in range(n_peaks)],
        "ms2_normalized_intensities": [1.0] * n_peaks,
        "ionization_mode": "positive",
    }


def test_structure_disjoint_split_no_overlap():
    keys = [f"KEY{i:03d}" for i in range(100)] * 3  # duplicates on purpose
    train, val = structure_disjoint_split(keys, val_frac=0.1, seed=1)
    assert not (train & val)
    assert len(val) == 10
    assert len(train) == 90
    train2, val2 = structure_disjoint_split(keys, val_frac=0.1, seed=1)
    assert val == val2  # deterministic


def test_reservoir_sampler_capacity():
    rs = ReservoirSampler(capacity=5, seed=0)
    for i in range(100):
        rs.add(i)
    assert len(rs) == 5
    assert rs.n_seen == 100
    assert len(set(rs.items)) == 5
    rs2 = ReservoirSampler(capacity=5, seed=0)
    for i in range(100):
        rs2.add(i)
    assert rs.items == rs2.items  # deterministic


def test_build_streaming_keeps_best_per_structure():
    rows = [
        _row(CAFFEINE, "RYYVLZVUVIJVGH", 195.0877, 5),
        _row(CAFFEINE, "RYYVLZVUVIJVGH", 195.0877, 50),   # most peaks
        _row(CAFFEINE, "RYYVLZVUVIJVGH", 195.0877, 10),
        _row("CCO", "LFQSCWFLJHTTHZ", 47.0496, 3),
    ]
    lib = LibraryIndex(max_per_structure=2).build_streaming(iter(rows))
    assert len(lib) == 3  # 2 caffeine + 1 ethanol
    assert lib.n_structures == 2
    hits = lib.query([100.0 + i for i in range(50)], [1.0] * 50, 195.0877)
    assert hits[0][0] == CAFFEINE
