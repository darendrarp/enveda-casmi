"""Tests for src/io_enveda.py and src/io_kaggle.py."""

import pandas as pd

from src.io_enveda import record_to_row
from src.io_kaggle import (group_test_by_molecule, load_test, load_train,
                           parquet_schema)

ENVEda_RECORD = {
    "title": "enveda_000001_01",
    "pepmass": 195.0877,
    "adduct": "[M+H]+",
    "ionmode": "ESI Positive",
    "instrument": "Bruker timsTOF Pro 2",
    "collision_energies": 40,
    "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
    "formula": "C8H10N4O2",
    "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "mzs": ["58.0283", "110.0712", "138.0661", "195.0877"],
    "intensities": ["100.0", "500.0", "900.0", "1000.0"],
}


def test_record_to_row_happy_path():
    row = record_to_row(ENVEda_RECORD)
    assert row is not None
    assert row["spectrum_id"] == "enveda_000001_01"
    assert row["inchikey14"] == "RYYVLZVUVIJVGH"
    assert row["precursor_mz"] == 195.0877
    assert row["ionization_mode"] == "positive"
    assert row["ingest_lib"] == "enveda-180"
    assert row["ms2_mzs"] == [58.0283, 110.0712, 138.0661, 195.0877]
    # Intensities normalised so max = 1.0.
    assert max(row["ms2_normalized_intensities"]) == 1.0
    assert abs(row["ms2_normalized_intensities"][0] - 0.1) < 1e-9


def test_record_to_row_rejects_bad_records():
    bad = dict(ENVEda_RECORD, smiles="not smiles(((")
    assert record_to_row(bad) is None
    bad = dict(ENVEda_RECORD, mzs=["1.0"], intensities=["1.0", "2.0"])
    assert record_to_row(bad) is None  # misaligned peak lists
    bad = dict(ENVEda_RECORD, pepmass=None)
    assert record_to_row(bad) is None


def _write_parquet(path, df):
    df.to_parquet(path, index=False)


def test_kaggle_loaders_round_trip(tmp_path):
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train_df = pd.DataFrame([{
        "ingest_lib": "enveda-180",
        "normalized_smiles": "CCO",
        "molecular_formula": "C2H6O",
        "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "inchikey14": "LFQSCWFLJHTTHZ",
        "precursor_mz": 47.0496,
        "precursor_error_ppm": 1.5,
        "ms2_mzs": [[29.0, 31.0]],
        "ms2_normalized_intensities": [[0.5, 1.0]],
        "adduct": "[M+H]+",
        "ionization_mode": "positive",
        "instrument": "timsTOF",
        "extra_unknown_col": "ignored",
    }])
    test_df = pd.DataFrame([
        {"molecule_id": "mol_x", "precursor_mz": 47.0496,
         "ms2_mzs": [[29.0, 31.0]], "ms2_normalized_intensities": [[0.5, 1.0]],
         "adduct": "[M+H]+", "ionization_mode": "positive"},
        {"molecule_id": "mol_x", "precursor_mz": 47.0497,
         "ms2_mzs": [[29.0]], "ms2_normalized_intensities": [[1.0]],
         "adduct": "[M+H]+", "ionization_mode": "positive"},
        {"molecule_id": "mol_y", "precursor_mz": 100.0,
         "ms2_mzs": [[50.0]], "ms2_normalized_intensities": [[1.0]],
         "adduct": "[M+H]+", "ionization_mode": "positive"},
    ])
    _write_parquet(train_path, train_df)
    _write_parquet(test_path, test_df)

    assert "normalized_smiles" in parquet_schema(train_path)
    loaded = load_train(train_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["inchikey14"] == "LFQSCWFLJHTTHZ"
    assert "extra_unknown_col" not in loaded.columns  # not requested
    t = load_test(test_path)
    assert len(t) == 3
    groups = group_test_by_molecule(t)
    assert set(groups) == {"mol_x", "mol_y"}
    assert len(groups["mol_x"]) == 2
