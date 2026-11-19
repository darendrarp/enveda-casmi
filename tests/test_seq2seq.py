"""Tests for src/seq2seq.py — tokenizer, transformer, beam search."""

import numpy as np
import torch

from src.seq2seq import (SmilesTokenizer, SpectraTransformer,
                         collate_seq2seq, tokenize_smiles, train_seq2seq)

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
SMILES = [CAFFEINE, "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
          "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]


def _row(smi, precursor_mz, mzs, ints):
    return {
        "normalized_smiles": smi,
        "precursor_mz": precursor_mz,
        "ms2_mzs": mzs,
        "ms2_normalized_intensities": ints,
        "adduct": "[M+H]+",
        "ionization_mode": "positive",
    }


def test_tokenizer_round_trip():
    tok = SmilesTokenizer.fit(SMILES)
    for smi in SMILES:
        ids = tok.encode(smi)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id
        assert tok.decode(ids) == smi
    assert tokenize_smiles("CC[Br]") == ["C", "C", "[Br]"]
    assert len(tok) > 4


def test_transformer_forward_shape():
    tok = SmilesTokenizer.fit(SMILES)
    model = SpectraTransformer(len(tok), d_model=32, nhead=2,
                               num_encoder_layers=1, num_decoder_layers=1,
                               dim_feedforward=64)
    rows = [_row(s, 200.0, [100.0, 150.0], [0.5, 1.0]) for s in SMILES[:2]]
    batch = collate_seq2seq(rows, tok)
    tgt = batch.pop("tgt")
    logits = model(tgt=tgt[:, :-1], **batch)
    assert logits.shape[:2] == (2, tgt.shape[1] - 1)
    assert logits.shape[2] == len(tok)
    assert torch.isfinite(logits).all()


def test_train_seq2seq_smoke():
    torch.manual_seed(0)
    tok = SmilesTokenizer.fit(SMILES)
    model = SpectraTransformer(len(tok), d_model=32, nhead=2,
                               num_encoder_layers=1, num_decoder_layers=1,
                               dim_feedforward=64)
    rows = [_row(s, 200.0, [100.0, 150.0], [0.5, 1.0]) for s in SMILES] * 4
    history = train_seq2seq(model, tok, rows, rows[:4], torch.device("cpu"),
                            epochs=1, batch_size=8, lr=1e-3)
    assert len(history["train_loss"]) == 1
    assert np.isfinite(history["train_loss"][0])


def test_beam_search_returns_candidates():
    torch.manual_seed(0)
    tok = SmilesTokenizer.fit(SMILES)
    model = SpectraTransformer(len(tok), d_model=32, nhead=2,
                               num_encoder_layers=1, num_decoder_layers=1,
                               dim_feedforward=64)
    model.eval()
    rows = [_row(CAFFEINE, 195.0877, [58.0283, 110.0712], [0.5, 1.0])]
    batch = collate_seq2seq(rows, tok)
    batch.pop("tgt")
    out = model.generate_beam(batch["peaks"], batch["peak_mask"],
                              batch["precursor_mz"], batch["adduct"], tok,
                              beam_width=5, max_len=40)
    assert 1 <= len(out) <= 5
    assert all(isinstance(s, str) and s for s, _ in out)
    scores = [sc for _, sc in out]
    assert scores == sorted(scores, reverse=True)
