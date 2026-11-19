"""Streaming loader for the public Enveda-180 dataset (Zenodo).

Enveda-180 (v3, Zenodo record 21346580) is the open spectral library that
supplies ~1.15M of the competition's ~2.5M training spectra, acquired on the
same Bruker timsTOF as the hidden test set. The ``-filtered.jsonl.gz``
serialisation is used here: one JSON object per spectrum.

Record fields used (verified against the file):
- ``title``: spectrum id, e.g. ``enveda_000001_01``
- ``smiles`` / ``inchikey`` / ``formula``: structure identity
- ``pepmass``: precursor m/z
- ``adduct``: e.g. ``[M+H]+`` (also exotic forms like ``[M-H2+H]+``)
- ``mzs`` / ``intensities``: peak lists (strings -> float)
- ``ionmode``: ``ESI Positive`` / ``ESI Negative``
- ``instrument`` / ``collision_energies``

Rows are normalised to the competition's column naming
(``normalized_smiles``, ``inchikey14``, ``precursor_mz``, ``ms2_mzs``,
``ms2_normalized_intensities``, ``adduct``, ``ionization_mode``) so the
same downstream code runs on local data and on ``train.parquet``.
"""

from __future__ import annotations

import gzip
import json

from .chem import canonical_smiles


def _ion_mode(raw):
    if raw is None:
        return None
    s = str(raw).lower()
    if "neg" in s:
        return "negative"
    if "pos" in s:
        return "positive"
    return raw


def record_to_row(rec):
    """Convert one Enveda-180 JSON record to a competition-style row dict.

    Returns None if the record lacks a usable structure or peak list.
    """
    smiles = canonical_smiles(rec.get("smiles") or "")
    if smiles is None:
        return None
    inchikey = rec.get("inchikey") or ""
    inchikey14 = inchikey[:14] if len(inchikey) >= 14 else None
    if inchikey14 is None:
        return None
    try:
        mzs = [float(x) for x in rec.get("mzs") or []]
        ints = [float(x) for x in rec.get("intensities") or []]
    except (TypeError, ValueError):
        return None
    if not mzs or len(mzs) != len(ints):
        return None
    try:
        precursor_mz = float(rec.get("pepmass"))
    except (TypeError, ValueError):
        return None
    # Normalise intensities so the largest peak is 1.0 (competition format).
    mx = max(ints) if ints else 0.0
    if mx <= 0:
        return None
    ints = [v / mx for v in ints]
    ce = rec.get("collision_energies")
    if isinstance(ce, list):
        ce = ce[0] if ce else None
    return {
        "spectrum_id": rec.get("title"),
        "ingest_lib": "enveda-180",
        "normalized_smiles": smiles,
        "inchikey": inchikey,
        "inchikey14": inchikey14,
        "molecular_formula": rec.get("formula"),
        "precursor_mz": precursor_mz,
        "adduct": rec.get("adduct"),
        "ionization_mode": _ion_mode(rec.get("ionmode")),
        "instrument": rec.get("instrument"),
        "collision_energy": ce,
        "ms2_mzs": mzs,
        "ms2_normalized_intensities": ints,
    }


def iter_enveda_jsonl(path, limit=None):
    """Stream rows from an Enveda-180 ``.jsonl.gz`` file (memory-safe)."""
    n = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = record_to_row(rec)
            if row is None:
                continue
            yield row
            n += 1
            if limit is not None and n >= limit:
                return
