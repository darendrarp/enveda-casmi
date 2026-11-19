"""Loaders for the competition files (``train.parquet`` / ``test.parquet``).

Verified schema (competition data page + discussion data card, Nov 2026):
- ``train.parquet``: 2,539,608 rows x 18 cols, one row per spectrum, no
  molecule/spectrum id. Label columns: ``normalized_smiles`` (RDKit
  standardised SMILES), ``inchikey``, ``inchikey14``, ``molecular_formula``.
- ``test.parquet``: 1,213 visible rows x 12 cols (placeholder drawn from
  training data; replaced by the hidden test set on re-run), 400 molecules.
  Has ``molecule_id`` (anonymous) and the shared spectrum columns, no
  structure columns.
- Shared spectrum columns: ``ms2_mzs`` / ``ms2_normalized_intensities``
  (aligned arrays, largest peak = 1.0), ``precursor_mz``, ``adduct``,
  ``ionization_mode``, ``instrument``, ``ingest_lib`` (train only).

Loaders only request columns that exist in the file, so they stay robust
if the shipped schema differs slightly from the documented one.
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

TRAIN_COLUMNS = [
    "ingest_lib",
    "normalized_smiles",
    "molecular_formula",
    "inchikey",
    "inchikey14",
    "precursor_mz",
    "precursor_error_ppm",
    "ms2_mzs",
    "ms2_normalized_intensities",
    "adduct",
    "ionization_mode",
    "instrument",
    "collision_energy",
    "base_peak_intensity",
]

TEST_COLUMNS = [
    "molecule_id",
    "precursor_mz",
    "ms2_mzs",
    "ms2_normalized_intensities",
    "adduct",
    "ionization_mode",
    "instrument",
    "collision_energy",
    "base_peak_intensity",
]


def _available_columns(path, wanted):
    schema = pq.ParquetFile(path).schema_arrow
    have = set(schema.names)
    return [c for c in wanted if c in have]


def parquet_schema(path):
    """Return the list of column names in a parquet file."""
    return pq.ParquetFile(path).schema_arrow.names


def iter_train_batches(path, batch_size=100_000, columns=None):
    """Stream train.parquet as pandas batches (memory-safe for 2.8 GiB)."""
    cols = _available_columns(path, columns or TRAIN_COLUMNS)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        yield batch.to_pandas()


def load_train(path, columns=None, limit=None):
    """Load train.parquet fully (only for small local files)."""
    cols = _available_columns(path, columns or TRAIN_COLUMNS)
    df = pd.read_parquet(path, columns=cols)
    if limit is not None:
        df = df.head(limit)
    return df


def load_test(path, columns=None):
    cols = _available_columns(path, columns or TEST_COLUMNS)
    return pd.read_parquet(path, columns=cols)


def load_sample_submission(path):
    return pd.read_csv(path)


def group_test_by_molecule(test_df):
    """Group test rows by molecule_id -> {molecule_id: [spectrum dicts]}."""
    groups: dict[str, list[dict]] = {}
    for row in test_df.to_dict("records"):
        groups.setdefault(row["molecule_id"], []).append(row)
    return groups


def row_to_spectrum(row):
    """Normalise a parquet row (train or test) to a spectrum dict."""
    return {
        "precursor_mz": float(row["precursor_mz"]),
        "ms2_mzs": list(row["ms2_mzs"]),
        "ms2_normalized_intensities": list(row["ms2_normalized_intensities"]),
        "adduct": row.get("adduct"),
        "ionization_mode": row.get("ionization_mode"),
        "instrument": row.get("instrument"),
        "collision_energy": row.get("collision_energy"),
    }
