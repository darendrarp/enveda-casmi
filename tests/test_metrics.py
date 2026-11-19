"""Tests for src/metrics.py — MRR@25 with InChIKey14 matching."""

from src.metrics import mrr_at_k, reciprocal_rank, topk_hit_rate

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
ETHANOL = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
R_LACTIC = "C[C@H](O)C(=O)O"
S_LACTIC = "C[C@@H](O)C(=O)O"


def test_reciprocal_rank_positions():
    assert reciprocal_rank([CAFFEINE, ETHANOL], CAFFEINE) == 1.0
    assert reciprocal_rank([ETHANOL, ASPIRIN, CAFFEINE], CAFFEINE) == 1 / 3
    assert reciprocal_rank([ETHANOL, ASPIRIN], CAFFEINE) == 0.0
    assert reciprocal_rank([], CAFFEINE) == 0.0


def test_reciprocal_rank_stereoisomer_counts():
    # A stereoisomer of the truth is a correct answer (InChIKey14 match).
    assert reciprocal_rank([S_LACTIC], R_LACTIC) == 1.0


def test_reciprocal_rank_beyond_k():
    ranked = [ETHANOL] * 25 + [CAFFEINE]
    assert reciprocal_rank(ranked, CAFFEINE, k=25) == 0.0
    assert reciprocal_rank(ranked, CAFFEINE, k=26) == 1 / 26


def test_mrr_at_k_mean():
    ranked = [[CAFFEINE], [ETHANOL, CAFFEINE], [ASPIRIN]]
    truths = [CAFFEINE, CAFFEINE, CAFFEINE]
    mrr, rrs = mrr_at_k(ranked, truths, k=25)
    assert rrs == [1.0, 0.5, 0.0]
    assert abs(mrr - 0.5) < 1e-9
    assert topk_hit_rate(ranked, truths, k=25) == 2 / 3
    assert topk_hit_rate(ranked, truths, k=1) == 1 / 3
    assert mrr_at_k([], []) == (0.0, [])
