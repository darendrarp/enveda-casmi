"""Evaluation metrics.

The competition metric is MRR@25 (mean reciprocal rank over the top-25
ranked candidates per molecule). A candidate counts as correct when its
tautomer-canonical InChIKey14 matches the truth molecule's InChIKey14.
"""

from __future__ import annotations

from .chem import inchikey14


def reciprocal_rank(ranked_smiles, truth_smiles, k: int = 25):
    """Reciprocal rank of the first correct candidate (0.0 if none in top-k)."""
    truth_key = inchikey14(truth_smiles)
    if truth_key is None:
        return 0.0
    for rank, cand in enumerate(ranked_smiles[:k], start=1):
        if inchikey14(cand) == truth_key:
            return 1.0 / rank
    return 0.0


def mrr_at_k(ranked_list, truth_list, k: int = 25):
    """Mean reciprocal rank over molecules.

    ``ranked_list``: list (per molecule) of ranked candidate SMILES lists.
    ``truth_list``: list (per molecule) of truth SMILES strings.
    Returns ``(mean_mrr, per_molecule_rr)``.
    """
    assert len(ranked_list) == len(truth_list), "length mismatch"
    if not truth_list:
        return 0.0, []
    # Cache InChIKey14 computations: candidates repeat across molecules.
    cache: dict[str, str | None] = {}

    def key(smi):
        if smi not in cache:
            cache[smi] = inchikey14(smi)
        return cache[smi]

    rrs = []
    for ranked, truth in zip(ranked_list, truth_list):
        tk = key(truth)
        rr = 0.0
        if tk is not None:
            for rank, cand in enumerate(ranked[:k], start=1):
                if key(cand) == tk:
                    rr = 1.0 / rank
                    break
        rrs.append(rr)
    return sum(rrs) / len(rrs), rrs


def topk_hit_rate(ranked_list, truth_list, k: int = 25):
    """Fraction of molecules with the truth anywhere in the top-k."""
    _, rrs = mrr_at_k(ranked_list, truth_list, k=k)
    if not rrs:
        return 0.0
    return sum(1 for r in rrs if r > 0) / len(rrs)
