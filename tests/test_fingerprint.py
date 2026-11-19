"""Tests for src/fingerprint.py — FPNet forward pass, training smoke test,
candidate-pool Tanimoto ranking."""

import numpy as np
import torch

from src.chem import morgan_fp
from src.fingerprint import (CandidatePool, FPNet, collate_spectra,
                             predict_fingerprints, train_fpnet)

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
ETHANOL = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def _row(smi, precursor_mz, mzs, ints):
    return {
        "normalized_smiles": smi,
        "precursor_mz": precursor_mz,
        "ms2_mzs": mzs,
        "ms2_normalized_intensities": ints,
        "adduct": "[M+H]+",
        "ionization_mode": "positive",
    }


ROWS = [
    _row(CAFFEINE, 195.0877, [58.0283, 110.0712, 138.0661], [0.1, 0.5, 1.0]),
    _row(ETHANOL, 47.0496, [29.0, 31.0, 45.0], [0.3, 1.0, 0.2]),
    _row(ASPIRIN, 181.0495, [43.0, 121.0, 163.0], [0.4, 1.0, 0.6]),
]


def test_collate_shapes():
    batch = collate_spectra(ROWS)
    assert batch["peaks"].shape == (3, 3, 4)
    assert batch["peak_mask"].shape == (3, 3)
    assert batch["target"].shape == (3, 2048)
    assert batch["target"].sum() > 0


def test_collate_without_labels_inference_rows():
    # Test-format rows have no structure label; inference must still work.
    rows = [{k: v for k, v in r.items() if k != "normalized_smiles"}
            for r in ROWS]
    batch = collate_spectra(rows)
    assert "target" not in batch
    assert batch["peaks"].shape == (3, 3, 4)
    model = FPNet(d_model=64)
    probs = predict_fingerprints(model, rows, torch.device("cpu"))
    assert probs.shape == (3, 2048)


def test_fpnet_forward_shape():
    model = FPNet(d_model=64)
    batch = collate_spectra(ROWS)
    target = batch.pop("target")
    logits = model(**batch)
    assert logits.shape == (3, 2048)
    assert torch.isfinite(logits).all()


def test_train_fpnet_smoke_runs_and_learns():
    torch.manual_seed(0)
    model = FPNet(d_model=64)
    rows = ROWS * 8  # 24 tiny rows
    history = train_fpnet(model, rows, ROWS, torch.device("cpu"),
                          epochs=2, batch_size=8, lr=1e-3)
    assert len(history["train_loss"]) == 2
    assert all(np.isfinite(history["train_loss"]))
    # Loss should not explode on this tiny problem.
    assert history["train_loss"][-1] < 2.0
    probs = predict_fingerprints(model, ROWS, torch.device("cpu"))
    assert probs.shape == (3, 2048)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_candidate_pool_ranking_perfect_prediction():
    pool = CandidatePool([CAFFEINE, ETHANOL, ASPIRIN])
    assert len(pool) == 3
    # A perfect fingerprint prediction ranks the true structure first
    # with Tanimoto 1.0.
    ranked = pool.rank(morgan_fp(CAFFEINE), top_n=3)
    assert ranked[0][0] == CAFFEINE
    assert abs(ranked[0][1] - 1.0) < 1e-6


def test_candidate_pool_dedupes_stereoisomers():
    pool = CandidatePool(["C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O", "CCO"])
    assert len(pool) == 2  # lactic acid stereoisomers collapse to one


def test_candidate_pool_mass_filter():
    pool = CandidatePool([CAFFEINE, ETHANOL, ASPIRIN]).add_exact_masses()
    ranked = pool.rank(morgan_fp(CAFFEINE), top_n=3,
                       neutral_mass_hint=194.0804, mass_tol_da=0.01)
    assert [s for s, _ in ranked] == [CAFFEINE]
