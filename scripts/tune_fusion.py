#!/usr/bin/env python3
"""Tune the library gate for the hybrid fusion on the validation split.

Compares MRR@25 for the hybrid at several lib_gate values (0.0 = ungated
RRF) plus FP-only, using the production predict_molecule path. The gate
exists because on structure-disjoint molecules the library list is pure
noise and dilutes the learned ranking.

Usage:
    python scripts/tune_fusion.py --fp-checkpoint runs/fpnet.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fingerprint import CandidatePool, FPNet  # noqa: E402
from src.metrics import mrr_at_k, topk_hit_rate  # noqa: E402
from src.pipeline import predict_molecule  # noqa: E402
from src.retrieval import LibraryIndex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/local_spectra.parquet")
    ap.add_argument("--fp-checkpoint", default="runs/fpnet.pt")
    ap.add_argument("--out", default="runs/fusion_tuning.json")
    ap.add_argument("--max-val-molecules", type=int, default=300)
    ap.add_argument("--gates", default="0.0,0.3,0.5,0.7")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(args.data)
    train_rows = df[df["split"] == "train"].to_dict("records")
    val_rows = df[df["split"] == "val"].to_dict("records")
    device = torch.device(args.device)

    lib = LibraryIndex(max_per_structure=2).build(train_rows)
    from collections import Counter

    train_counts = Counter(r["normalized_smiles"] for r in train_rows)
    train_structs = [s for s, _ in train_counts.most_common()]
    val_structs = sorted({r["normalized_smiles"] for r in val_rows})
    pool_aug = CandidatePool(train_structs + val_structs)
    pool_aug.add_exact_masses()

    ckpt = torch.load(args.fp_checkpoint, map_location=device, weights_only=False)
    fp_model = FPNet(d_model=ckpt["config"]["d_model"])
    fp_model.load_state_dict(ckpt["state_dict"])
    fp_model.to(device).eval()

    groups: dict[str, list[dict]] = {}
    truth: dict[str, str] = {}
    for r in val_rows:
        key = f"val_{r['inchikey14']}"
        groups.setdefault(key, []).append(r)
        truth[key] = r["normalized_smiles"]
    mol_ids = sorted(groups)[: args.max_val_molecules]
    truth_list = [truth[m] for m in mol_ids]

    gates = [float(g) for g in args.gates.split(",")]
    variants = {"fp_only": None}  # None = no library at all
    for g in gates:
        variants[f"hybrid_gate_{g}"] = g
    ranked = {name: [] for name in variants}
    for i, m in enumerate(mol_ids):
        spectra = groups[m]
        for name, gate in variants.items():
            lib_arg = lib if gate is not None else None
            cands = predict_molecule(
                spectra, lib_index=lib_arg, fp_model=fp_model, pool=pool_aug,
                device=device, fp_mass_tol_da=0.02,
                lib_gate=gate if gate is not None else 0.5,
            )
            ranked[name].append(cands)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(mol_ids)} ({time.time() - t0:.0f}s)")

    results = {}
    for name, rl in ranked.items():
        mrr, _ = mrr_at_k(rl, truth_list, k=25)
        hit25 = topk_hit_rate(rl, truth_list, k=25)
        hit1 = topk_hit_rate(rl, truth_list, k=1)
        results[name] = {"mrr@25": round(mrr, 4), "hit@1": round(hit1, 4),
                         "hit@25": round(hit25, 4), "n": len(truth_list)}
        print(f"{name:16s} MRR@25={mrr:.4f}  hit@1={hit1:.4f}  "
              f"hit@25={hit25:.4f}")
    results["meta"] = {"elapsed_s": round(time.time() - t0, 1)}
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
