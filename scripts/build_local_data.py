#!/usr/bin/env python3
"""Build a local training/validation set from Enveda-180 (public data).

Streams the Enveda-180 filtered JSONL, keeps a deterministic hash-sample of
compounds (no RDKit work on skipped rows, so the full 1.15M-record stream
takes only a few minutes), and writes a structure-disjoint split:

- ``local_spectra.parquet``: sampled spectra with a ``split`` column
  (``train`` / ``val``), competition-style columns.
- ``local_val_truth.json``: {molecule_id: truth SMILES} for val molecules.

Usage:
    python scripts/build_local_data.py --input data/enveda-180-filtered.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.io_enveda import record_to_row  # noqa: E402
from src.splits import structure_disjoint_split  # noqa: E402


def _hash_frac(key: str) -> float:
    h = hashlib.md5(key.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/enveda-180-filtered.jsonl.gz")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--sample-frac", type=float, default=0.04,
                    help="fraction of compounds to keep (~4%% -> ~7k compounds)")
    ap.add_argument("--val-molecules", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    n_seen = 0
    with gzip.open(args.input, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_seen += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ik = (rec.get("inchikey") or "")[:14]
            if not ik or _hash_frac(ik) >= args.sample_frac:
                continue
            row = record_to_row(rec)
            if row is not None:
                rows.append(row)
    print(f"streamed {n_seen} records, kept {len(rows)} spectra "
          f"from {len({r['inchikey14'] for r in rows})} compounds")

    train_keys, val_keys = structure_disjoint_split(
        [r["inchikey14"] for r in rows],
        val_frac=0.1, seed=args.seed, max_val=args.val_molecules,
    )
    for r in rows:
        r["split"] = "val" if r["inchikey14"] in val_keys else "train"
    n_val = sum(1 for r in rows if r["split"] == "val")
    print(f"train spectra: {len(rows) - n_val}, val spectra: {n_val}, "
          f"val molecules: {len(val_keys)}")

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(args.out_dir, "local_spectra.parquet"), index=False)
    truth = {}
    for r in rows:
        if r["split"] == "val":
            truth[f"val_{r['inchikey14']}"] = r["normalized_smiles"]
    with open(os.path.join(args.out_dir, "local_val_truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    print(f"wrote {os.path.join(args.out_dir, 'local_spectra.parquet')} "
          f"and local_val_truth.json")


if __name__ == "__main__":
    main()
