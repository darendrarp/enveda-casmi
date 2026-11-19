#!/usr/bin/env python3
"""Local evaluation on the structure-disjoint validation split.

Reports MRR@25 for each component and the hybrid:
- library: spectral library matching (mass-windowed cosine)
- fp_trainpool: FPNet ranking over train structures only
- fp_augpool: FPNet ranking over train + val structures (simulates an
  external candidate database that contains the answer, as the competition
  allows — COCONUT/PubChem)
- seq2seq: de novo beam search (subset of molecules, CPU-slow)
- hybrid: RRF fusion of library + fp_augpool (+ seq2seq when available)

Also runs a leave-one-spectrum-out sanity check on train molecules, where
library matching is expected to score high (proves the harness works).

Usage:
    python scripts/evaluate.py --fp-checkpoint runs/fpnet.pt \
        --seq2seq-checkpoint runs/seq2seq.pt --out runs/eval_results.json
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
from src.chem import inchikey14  # noqa: E402
from src.fingerprint import (CandidatePool, FPNet,  # noqa: E402
                             rank_molecule_by_fingerprint)
from src.metrics import mrr_at_k, topk_hit_rate  # noqa: E402
from src.pipeline import predict_molecule  # noqa: E402
from src.retrieval import LibraryIndex  # noqa: E402
from src.seq2seq import SmilesTokenizer, SpectraTransformer  # noqa: E402


def group_by_molecule(rows):
    groups: dict[str, list[dict]] = {}
    truth: dict[str, str] = {}
    for r in rows:
        key = f"val_{r['inchikey14']}"
        groups.setdefault(key, []).append(r)
        truth[key] = r["normalized_smiles"]
    return groups, truth


def report(name, ranked_list, truth_list, results):
    mrr, _ = mrr_at_k(ranked_list, truth_list, k=25)
    hit25 = topk_hit_rate(ranked_list, truth_list, k=25)
    hit1 = topk_hit_rate(ranked_list, truth_list, k=1)
    results[name] = {"mrr@25": round(mrr, 4), "hit@1": round(hit1, 4),
                     "hit@25": round(hit25, 4), "n": len(truth_list)}
    print(f"{name:16s} MRR@25={mrr:.4f}  hit@1={hit1:.4f}  "
          f"hit@25={hit25:.4f}  (n={len(truth_list)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/local_spectra.parquet")
    ap.add_argument("--fp-checkpoint", default="runs/fpnet.pt")
    ap.add_argument("--seq2seq-checkpoint", default="")
    ap.add_argument("--out", default="runs/eval_results.json")
    ap.add_argument("--max-val-molecules", type=int, default=300)
    ap.add_argument("--seq2seq-molecules", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(args.data)
    train_rows = df[df["split"] == "train"].to_dict("records")
    val_rows = df[df["split"] == "val"].to_dict("records")
    print(f"train spectra: {len(train_rows)}, val spectra: {len(val_rows)}")

    device = torch.device(args.device)
    results: dict = {}

    # --- library index (train spectra only) ---
    lib = LibraryIndex(max_per_structure=2).build(train_rows)
    print(f"library index: {len(lib)} spectra, {lib.n_structures} structures")

    # --- candidate pools (popularity-sorted for the fallback) ---
    from collections import Counter

    train_counts = Counter(r["normalized_smiles"] for r in train_rows)
    train_structs = [s for s, _ in train_counts.most_common()]
    pool_train = CandidatePool(train_structs)
    print(f"pool(train): {len(pool_train)} structures")
    val_structs = sorted({r["normalized_smiles"] for r in val_rows})
    pool_aug = CandidatePool(train_structs + val_structs)
    print(f"pool(train+val): {len(pool_aug)} structures")
    pool_aug.add_exact_masses()
    pool_train.add_exact_masses()

    # --- FPNet ---
    fp_model = None
    if args.fp_checkpoint and os.path.exists(args.fp_checkpoint):
        ckpt = torch.load(args.fp_checkpoint, map_location=device,
                          weights_only=False)
        fp_model = FPNet(d_model=ckpt["config"]["d_model"])
        fp_model.load_state_dict(ckpt["state_dict"])
        fp_model.to(device).eval()
        print(f"loaded FPNet from {args.fp_checkpoint}")
    else:
        print("no FPNet checkpoint, skipping fingerprint component")

    # --- seq2seq (optional) ---
    s2s_model, tokenizer = None, None
    if args.seq2seq_checkpoint and os.path.exists(args.seq2seq_checkpoint):
        ckpt = torch.load(args.seq2seq_checkpoint, map_location=device,
                          weights_only=False)
        tokenizer = SmilesTokenizer(ckpt["vocab"][4:])
        s2s_model = SpectraTransformer(len(tokenizer), **ckpt["config"])
        s2s_model.load_state_dict(ckpt["state_dict"])
        s2s_model.to(device).eval()
        print(f"loaded seq2seq from {args.seq2seq_checkpoint}")

    # --- validation molecules ---
    val_groups, val_truth = group_by_molecule(val_rows)
    mol_ids = sorted(val_groups)[: args.max_val_molecules]
    print(f"evaluating {len(mol_ids)} val molecules "
          f"(structure-disjoint split)")

    truth_list = [val_truth[m] for m in mol_ids]
    lib_ranked, fp_train_ranked, fp_aug_ranked = [], [], []
    hybrid_ranked, s2s_ranked, s2s_truth = [], [], []
    for i, m in enumerate(mol_ids):
        spectra = val_groups[m]
        lib_ranked.append([s for s, _ in lib.query_molecule(spectra)])
        if fp_model is not None:
            fp_train_ranked.append([s for s, _ in rank_molecule_by_fingerprint(
                fp_model, spectra, pool_train, device, mass_tol_da=0.02)])
            fp_aug_ranked.append([s for s, _ in rank_molecule_by_fingerprint(
                fp_model, spectra, pool_aug, device, mass_tol_da=0.02)])
        hybrid_ranked.append(predict_molecule(
            spectra, lib_index=lib, fp_model=fp_model, pool=pool_aug,
            device=device, fp_mass_tol_da=0.02))
        if s2s_model is not None and i < args.seq2seq_molecules:
            s2s_ranked.append(predict_molecule(
                spectra, seq2seq_model=s2s_model, tokenizer=tokenizer,
                device=device, seq2seq_beam=10))
            s2s_truth.append(val_truth[m])
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(mol_ids)} molecules "
                  f"({time.time() - t0:.0f}s)")

    print("\n=== structure-disjoint validation (unseen molecules) ===")
    report("library", lib_ranked, truth_list, results)
    if fp_model is not None:
        report("fp_trainpool", fp_train_ranked, truth_list, results)
        report("fp_augpool", fp_aug_ranked, truth_list, results)
    report("hybrid", hybrid_ranked, truth_list, results)
    if s2s_ranked:
        report("seq2seq", s2s_ranked, s2s_truth, results)

    # --- sanity check: leave-one-spectrum-out on train molecules ---
    print("\n=== sanity check: leave-one-spectrum-out (train molecules) ===")
    from collections import defaultdict

    by_struct: dict[str, list[dict]] = defaultdict(list)
    for r in train_rows:
        by_struct[r["inchikey14"]].append(r)
    multi = [v for v in by_struct.values() if len(v) >= 2][:100]
    queries, q_truth, q_rest = [], [], []
    held_out_ids = set()
    for v in multi:
        queries.append(v[0])
        q_truth.append(v[0]["normalized_smiles"])
        held_out_ids.add(id(v[0]))
    rest_rows = [r for r in train_rows if id(r) not in held_out_ids]
    lib2 = LibraryIndex(max_per_structure=2).build(rest_rows)
    san_ranked = []
    for q in queries:
        san_ranked.append([s for s, _ in lib2.query(
            q["ms2_mzs"], q["ms2_normalized_intensities"],
            float(q["precursor_mz"]))])
    report("library_loso", san_ranked, q_truth, results)

    results["meta"] = {
        "train_spectra": len(train_rows),
        "val_spectra": len(val_rows),
        "val_molecules": len(mol_ids),
        "pool_train": len(pool_train),
        "pool_aug": len(pool_aug),
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved {args.out} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
