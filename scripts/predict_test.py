#!/usr/bin/env python3
"""Run the full pipeline on a Kaggle-format test parquet -> submission.csv.

Builds the library index + candidate pool from train.parquet (streamed in
batches), loads trained checkpoints, predicts per molecule and writes a
validated submission file.

Usage:
    python scripts/predict_test.py --train /kaggle/input/.../train.parquet \\
        --test /kaggle/input/.../test.parquet --fp-checkpoint runs/fpnet.pt \\
        --out submission.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fingerprint import CandidatePool, FPNet  # noqa: E402
from src.io_kaggle import (group_test_by_molecule, iter_train_batches,  # noqa: E402
                           load_test, row_to_spectrum)
from src.pipeline import predict_molecule  # noqa: E402
from src.retrieval import LibraryIndex  # noqa: E402
from src.seq2seq import SmilesTokenizer, SpectraTransformer  # noqa: E402
from src.submission import validate_submission, write_submission  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--sample-submission", default="")
    ap.add_argument("--fp-checkpoint", default="")
    ap.add_argument("--seq2seq-checkpoint", default="")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--max-train-spectra", type=int, default=0,
                    help="0 = all (streams in batches)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp-mass-tol-da", type=float, default=0.02)
    ap.add_argument("--lib-tol-da", type=float, default=0.01)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)

    # --- stream train.parquet: library index + structure pool ---
    # The library builder keeps only the best 2 spectra per structure, so
    # memory stays bounded on the 2.5M-row file. Structures are counted so
    # the pool can be built popularity-sorted (used by the fallback).
    from collections import Counter

    struct_counts: Counter = Counter()

    def row_stream():
        n = 0
        for batch in iter_train_batches(args.train, batch_size=100_000):
            for r in batch.to_dict("records"):
                struct_counts[r["normalized_smiles"]] += 1
                n += 1
                yield r
                if args.max_train_spectra and n >= args.max_train_spectra:
                    return
            print(f"  streamed {n} train spectra ({time.time() - t0:.0f}s)")

    lib = LibraryIndex(max_per_structure=2).build_streaming(row_stream())
    print(f"train structures: {len(struct_counts)}")
    print(f"library index: {len(lib)} spectra, {lib.n_structures} structures")
    pool = CandidatePool([s for s, _ in struct_counts.most_common()])
    pool.add_exact_masses()
    print(f"candidate pool: {len(pool)} structures")

    fp_model = None
    if args.fp_checkpoint and os.path.exists(args.fp_checkpoint):
        ckpt = torch.load(args.fp_checkpoint, map_location=device,
                          weights_only=False)
        fp_model = FPNet(d_model=ckpt["config"]["d_model"])
        fp_model.load_state_dict(ckpt["state_dict"])
        fp_model.to(device).eval()
        print(f"loaded FPNet from {args.fp_checkpoint}")

    s2s_model, tokenizer = None, None
    if args.seq2seq_checkpoint and os.path.exists(args.seq2seq_checkpoint):
        ckpt = torch.load(args.seq2seq_checkpoint, map_location=device,
                          weights_only=False)
        tokenizer = SmilesTokenizer(ckpt["vocab"][4:])
        s2s_model = SpectraTransformer(len(tokenizer), **ckpt["config"])
        s2s_model.load_state_dict(ckpt["state_dict"])
        s2s_model.to(device).eval()
        print(f"loaded seq2seq from {args.seq2seq_checkpoint}")

    test_df = load_test(args.test)
    groups = group_test_by_molecule(test_df)
    print(f"test molecules: {len(groups)}")

    rows = []
    for i, (mol_id, test_rows) in enumerate(sorted(groups.items())):
        spectra = [row_to_spectrum(r) for r in test_rows]
        cands = predict_molecule(
            spectra, lib_index=lib, fp_model=fp_model, pool=pool,
            seq2seq_model=s2s_model, tokenizer=tokenizer, device=device,
            lib_tol_da=args.lib_tol_da, fp_mass_tol_da=args.fp_mass_tol_da,
        )
        rows.append((mol_id, cands))
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(groups)} molecules ({time.time() - t0:.0f}s)")

    n = write_submission(rows, args.out)
    print(f"wrote {args.out} ({n} molecules)")
    problems = validate_submission(args.out, expected_molecule_ids=list(groups))
    if problems:
        print("VALIDATION PROBLEMS:")
        for p in problems[:20]:
            print(" -", p)
    else:
        print("submission validation: OK")
    print(f"total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
