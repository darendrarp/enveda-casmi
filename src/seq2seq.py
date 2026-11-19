"""De novo structure generation: transformer spectra -> SMILES.

A sequence-to-sequence transformer (in the spirit of the MassSpecGym
de novo baselines and MS2Mol-style models):
- Encoder: the peak list (m/z, intensity features) plus a precursor token
  (precursor m/z + adduct) is encoded by a Transformer encoder.
- Decoder: an autoregressive Transformer decoder generates SMILES tokens;
  beam search produces up to 25 ranked candidate structures per molecule.

This is the only component that can propose structures absent from every
database — essential for the competition's novel natural products. It is
also the most data-hungry: local CPU smoke-training only verifies the
pipeline end to end; real training happens on Kaggle GPU (see the notebook).
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fingerprint import ADDUCT_INDEX, ADDUCT_VOCAB
from .spectra import peak_features

SMILES_TOKEN_REGEX = (
    r"(\[[^\]]+\]|Br?|Cl?|N|O|P|S|F|I|b|c|n|o|p|s|se|as|\(|\)|\.|=|#|-|\+|"
    r"\\|/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
TOKEN_PATTERN = re.compile(SMILES_TOKEN_REGEX)

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


def tokenize_smiles(smiles):
    return TOKEN_PATTERN.findall(smiles)


class SmilesTokenizer:
    def __init__(self, vocab=None):
        self.specials = [PAD, BOS, EOS, UNK]
        self.vocab = list(self.specials) + sorted(set(vocab or []))
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    @classmethod
    def fit(cls, smiles_list, min_freq=1):
        from collections import Counter

        counts = Counter()
        for smi in smiles_list:
            counts.update(tokenize_smiles(smi))
        vocab = [t for t, c in counts.items() if c >= min_freq]
        return cls(vocab)

    def __len__(self):
        return len(self.vocab)

    def encode(self, smiles, max_len=160):
        ids = [self.bos_id]
        ids += [self.stoi.get(t, self.unk_id) for t in tokenize_smiles(smiles)]
        ids.append(self.eos_id)
        return ids[:max_len]

    def decode(self, ids):
        toks = []
        for i in ids:
            if i in (self.bos_id, self.eos_id, self.pad_id):
                continue
            toks.append(self.itos.get(i, ""))
        return "".join(toks)


class SpectraTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=4,
                 num_decoder_layers=4, dim_feedforward=1024, dropout=0.1,
                 max_peaks=256, max_len=160,
                 n_adducts=len(ADDUCT_VOCAB)):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.peak_proj = nn.Linear(4, d_model)
        self.peak_pos = nn.Embedding(max_peaks + 1, d_model)
        self.prec_proj = nn.Linear(1 + 16, d_model)
        self.adduct_emb = nn.Embedding(n_adducts, 16)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.tok_pos = nn.Embedding(max_len, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def encode(self, peaks, peak_mask, precursor_mz, adduct):
        """peaks: (B, P, 4); peak_mask: (B, P) bool; precursor_mz: (B,);
        adduct: (B,) long. Returns (memory, src_pad_mask)."""
        B, P, _ = peaks.shape
        device = peaks.device
        pos = torch.arange(P + 1, device=device).unsqueeze(0).expand(B, -1)
        peak_emb = self.peak_proj(peaks) + self.peak_pos(pos[:, 1:])
        prec_feat = torch.cat(
            [(precursor_mz / 1000.0).unsqueeze(1), self.adduct_emb(adduct)], dim=1
        )
        prec_emb = self.prec_proj(prec_feat).unsqueeze(1) + self.peak_pos(pos[:, :1])
        src = torch.cat([prec_emb, peak_emb], dim=1)
        src_pad = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=device), ~peak_mask], dim=1
        )
        memory = self.encoder(src, src_key_padding_mask=src_pad)
        return memory, src_pad

    def decode(self, memory, src_pad, tgt):
        T = tgt.shape[1]
        device = tgt.device
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.tok_emb(tgt) + self.tok_pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        h = self.decoder(x, memory, tgt_mask=mask,
                         memory_key_padding_mask=src_pad)
        return self.out(h)

    def forward(self, peaks, peak_mask, precursor_mz, adduct, tgt):
        memory, src_pad = self.encode(peaks, peak_mask, precursor_mz, adduct)
        return self.decode(memory, src_pad, tgt)

    @torch.no_grad()
    def generate_beam(self, peaks, peak_mask, precursor_mz, adduct, tokenizer,
                      beam_width=10, max_len=120, length_penalty=0.6):
        """Beam search for one spectrum. Returns [(smiles, score)] ranked."""
        self.eval()
        device = peaks.device
        # Single-spectrum beam search: memory is (1, S, d). All active beams
        # have the same length at each step, so they are decoded in ONE
        # batched decoder call per step (not one call per beam).
        memory, src_pad = self.encode(peaks, peak_mask, precursor_mz, adduct)
        bos = tokenizer.bos_id
        eos = tokenizer.eos_id
        beams = [(torch.tensor([[bos]], device=device), 0.0)]
        finished = []
        for step in range(max_len):
            active = [(s, sc) for s, sc in beams if s[0, -1].item() != eos]
            finished.extend([(s, sc) for s, sc in beams if s[0, -1].item() == eos])
            if not active:
                break
            tgt = torch.cat([s for s, _ in active], dim=0)  # (A, T)
            mem = memory.repeat(len(active), 1, 1)
            pad = src_pad.repeat(len(active), 1)
            logits = self.decode(mem, pad, tgt)
            logp = F.log_softmax(logits[:, -1, :], dim=-1)  # (A, V)
            topk_logp, topk_ids = torch.topk(logp, beam_width, dim=-1)
            cand = []
            for bi, (seq, score) in enumerate(active):
                for lp, tok in zip(topk_logp[bi].tolist(), topk_ids[bi].tolist()):
                    cand.append((torch.cat([seq, torch.tensor([[tok]], device=device)], dim=1),
                                 score + lp))
            cand.sort(key=lambda t: -t[1])
            beams = cand[:beam_width]
            if all(b[0][0, -1].item() == eos for b in beams):
                finished.extend(beams)
                break
        finished.extend(beams)
        results = []
        for seq, score in finished:
            ids = seq[0].tolist()
            if eos in ids:
                ids = ids[: ids.index(eos)]
            smi = tokenizer.decode(ids)
            lp = ((5.0 + len(ids)) / 6.0) ** length_penalty
            results.append((smi, score / max(lp, 1e-9)))
        results.sort(key=lambda t: -t[1])
        # Deduplicate identical SMILES strings, keep best score.
        seen, out = set(), []
        for smi, sc in results:
            if smi and smi not in seen:
                seen.add(smi)
                out.append((smi, sc))
            if len(out) >= beam_width:
                break
        return out


def collate_seq2seq(rows, tokenizer, max_peaks=128, max_len=160):
    feats, masks, prec, add, targets = [], [], [], [], []
    for r in rows:
        f = peak_features(r["ms2_mzs"], r["ms2_normalized_intensities"],
                          float(r["precursor_mz"]), max_peaks=max_peaks)
        feats.append(f)
        masks.append(np.ones(len(f), dtype=bool))
        prec.append(float(r["precursor_mz"]))
        add.append(ADDUCT_INDEX.get(r.get("adduct"), ADDUCT_INDEX["OTHER"]))
        targets.append(tokenizer.encode(r["normalized_smiles"], max_len=max_len))
    max_p = max((len(f) for f in feats), default=1)
    max_t = max((len(t) for t in targets), default=2)
    B = len(rows)
    peaks = np.zeros((B, max_p, 4), dtype=np.float32)
    mask = np.zeros((B, max_p), dtype=bool)
    tgt = np.full((B, max_t), tokenizer.pad_id, dtype=np.int64)
    for i, (f, m, t) in enumerate(zip(feats, masks, targets)):
        if len(f):
            peaks[i, : len(f)] = f
            mask[i, : len(f)] = m
        tgt[i, : len(t)] = t
    return {
        "peaks": torch.from_numpy(peaks),
        "peak_mask": torch.from_numpy(mask),
        "precursor_mz": torch.tensor(prec, dtype=torch.float32),
        "adduct": torch.tensor(add, dtype=torch.long),
        "tgt": torch.from_numpy(tgt),
    }


def train_seq2seq(model, tokenizer, train_rows, val_rows, device, epochs=3,
                  batch_size=64, lr=3e-4, max_peaks=128, max_len=160,
                  seed=42, log=None):
    """Teacher-forced training. Returns a history dict."""
    import random

    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_steps = epochs * max(1, math.ceil(len(train_rows) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=n_steps, pct_start=0.1
    )
    history = {"train_loss": [], "val_loss": []}
    model.to(device)
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for i in range(0, len(order), batch_size):
            idx = order[i: i + batch_size]
            batch = collate_seq2seq([train_rows[j] for j in idx], tokenizer,
                                    max_peaks=max_peaks, max_len=max_len)
            tgt = batch.pop("tgt").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(tgt=tgt[:, :-1], **batch)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_id, label_smoothing=0.1,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        history["train_loss"].append(tot / max(nb, 1))
        model.eval()
        vtot, vnb = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_rows), batch_size):
                batch = collate_seq2seq(val_rows[i: i + batch_size], tokenizer,
                                        max_peaks=max_peaks, max_len=max_len)
                tgt = batch.pop("tgt").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(tgt=tgt[:, :-1], **batch)
                vtot += F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1),
                    ignore_index=tokenizer.pad_id,
                ).item()
                vnb += 1
        history["val_loss"].append(vtot / max(vnb, 1))
        if log:
            log(f"epoch {epoch + 1}/{epochs} train_loss={history['train_loss'][-1]:.4f} "
                f"val_loss={history['val_loss'][-1]:.4f}")
    return history
