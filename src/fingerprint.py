"""FPNet: predict a Morgan fingerprint from a spectrum, then rank candidates.

Architecture (peak-set encoder, following the spectrum-to-fingerprint
literature — MIST / FPNet style, without formula annotation):
- each peak (m/z, intensity) is embedded by a small MLP;
- peaks are pooled (mean + max) into a fixed spectrum embedding;
- precursor m/z, adduct and ion mode are concatenated;
- an MLP head predicts 2048 Morgan-fingerprint bit logits.

Candidates from a structure pool are ranked by the probabilistic Tanimoto
similarity between the predicted fingerprint and each candidate's Morgan
fingerprint. This generalises to unseen molecules *provided the true
structure is in the candidate pool* (training structures + external
databases such as COCONUT/PubChem) — which is exactly the competition's
intended use of external data.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from .chem import ADDUCT_MASS, morgan_fp_on_bits, neutral_mass
from .spectra import peak_features

ADDUCT_VOCAB = [
    "[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M-H]-",
    "[M+H-H2O]+", "[M+2H]2+", "[M-H2+H]+", "[M+Cl]-", "OTHER",
]
ADDUCT_INDEX = {a: i for i, a in enumerate(ADDUCT_VOCAB)}
N_BITS = 2048


def adduct_idx(adduct):
    return ADDUCT_INDEX.get(adduct, ADDUCT_INDEX["OTHER"])


class FPNet(nn.Module):
    def __init__(self, n_bits: int = N_BITS, d_model: int = 256,
                 n_adducts: int = len(ADDUCT_VOCAB), dropout: float = 0.1):
        super().__init__()
        self.n_bits = n_bits
        self.d_model = d_model
        self.peak_mlp = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
        )
        self.adduct_emb = nn.Embedding(n_adducts, 16)
        head_in = 2 * d_model + 16 + 4  # pooled peaks + adduct + precursor feats
        self.head = nn.Sequential(
            nn.Linear(head_in, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, n_bits),
        )

    def forward(self, peaks, peak_mask, precursor_mz, adduct, ion_mode):
        """peaks: (B, P, 4); peak_mask: (B, P) bool; precursor_mz: (B,);
        adduct: (B,) long; ion_mode: (B,) float in {-1, 0, 1}."""
        h = self.peak_mlp(peaks)  # (B, P, d)
        h = h * peak_mask.unsqueeze(-1)
        counts = peak_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pool = h.sum(dim=1) / counts
        neg_inf = torch.finfo(h.dtype).min
        max_pool = h.masked_fill(~peak_mask.unsqueeze(-1), neg_inf).amax(dim=1)
        max_pool = torch.where(torch.isfinite(max_pool),
                               max_pool, torch.zeros_like(max_pool))
        prec = torch.stack(
            [
                precursor_mz / 1000.0,
                torch.log10(precursor_mz.clamp(min=1.0)),
                counts.squeeze(1) / 128.0,
                ion_mode,
            ],
            dim=1,
        )
        x = torch.cat([mean_pool, max_pool, self.adduct_emb(adduct), prec], dim=1)
        return self.head(x)


def collate_spectra(rows, max_peaks: int = 128):
    """Batch row dicts into FPNet tensors.

    The ``target`` key (Morgan fingerprints) is only included when the rows
    carry ``normalized_smiles`` labels — test rows don't, and inference
    must not require them.
    """
    from .chem import morgan_fp

    has_labels = all("normalized_smiles" in r for r in rows)
    feats, masks, prec, add, ion, targets = [], [], [], [], [], []
    for r in rows:
        f = peak_features(r["ms2_mzs"], r["ms2_normalized_intensities"],
                          float(r["precursor_mz"]), max_peaks=max_peaks)
        feats.append(f)
        masks.append(np.ones(len(f), dtype=bool))
        prec.append(float(r["precursor_mz"]))
        add.append(adduct_idx(r.get("adduct")))
        mode = str(r.get("ionization_mode") or "").lower()
        ion.append(-1.0 if "neg" in mode else 1.0)
        if has_labels:
            targets.append(morgan_fp(r["normalized_smiles"], n_bits=N_BITS))
    max_p = max((len(f) for f in feats), default=0)
    max_p = max(max_p, 1)
    B = len(rows)
    peaks = np.zeros((B, max_p, 4), dtype=np.float32)
    mask = np.zeros((B, max_p), dtype=bool)
    for i, (f, m) in enumerate(zip(feats, masks)):
        if len(f):
            peaks[i, : len(f)] = f
            mask[i, : len(f)] = m
    out = {
        "peaks": torch.from_numpy(peaks),
        "peak_mask": torch.from_numpy(mask),
        "precursor_mz": torch.tensor(prec, dtype=torch.float32),
        "adduct": torch.tensor(add, dtype=torch.long),
        "ion_mode": torch.tensor(ion, dtype=torch.float32),
    }
    if has_labels:
        out["target"] = torch.from_numpy(np.stack(targets).astype(np.float32))
    return out


@torch.no_grad()
def predict_fingerprints(model, rows, device, batch_size=256, max_peaks=128):
    """Predict fingerprint probabilities for a list of spectrum rows."""
    model.eval()
    probs = []
    for i in range(0, len(rows), batch_size):
        batch = collate_spectra(rows[i: i + batch_size], max_peaks=max_peaks)
        batch = {k: v.to(device) for k, v in batch.items() if k != "target"}
        logits = model(**batch)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    if not probs:
        return np.zeros((0, model.n_bits), dtype=np.float32)
    return np.concatenate(probs, axis=0)


def train_fpnet(model, train_rows, val_rows, device, epochs=3, batch_size=256,
                lr=1e-3, max_peaks=128, seed=42, log=None):
    """Train FPNet with BCE loss. Returns a history dict."""
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
            batch = collate_spectra([train_rows[j] for j in idx],
                                    max_peaks=max_peaks)
            target = batch.pop("target").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        history["train_loss"].append(tot / max(nb, 1))
        # validation loss
        model.eval()
        vtot, vnb = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_rows), batch_size):
                batch = collate_spectra(val_rows[i: i + batch_size],
                                        max_peaks=max_peaks)
                target = batch.pop("target").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch)
                vtot += F.binary_cross_entropy_with_logits(logits, target).item()
                vnb += 1
        history["val_loss"].append(vtot / max(vnb, 1))
        if log:
            log(f"epoch {epoch + 1}/{epochs} train_loss={history['train_loss'][-1]:.4f} "
                f"val_loss={history['val_loss'][-1]:.4f}")
    return history


class CandidatePool:
    """Structure pool with Morgan fingerprints for Tanimoto ranking.

    Input order is preserved (after dedup by InChIKey14): pass structures
    popularity-sorted (most frequent first) so the pool-fallback in
    ``pipeline`` degrades gracefully to the most common structures.
    """

    def __init__(self, smiles_list, n_bits: int = N_BITS):
        from .chem import canonical_smiles, inchikey14

        self.smiles: list[str] = []
        self.ik14: list[str] = []
        seen: set[str] = set()
        rows, cols = [], []
        for i, smi in enumerate(smiles_list):
            canon = canonical_smiles(smi)
            if canon is None:
                continue
            key = inchikey14(canon)
            if key is None or key in seen:
                continue
            seen.add(key)
            bits = morgan_fp_on_bits(canon, n_bits=n_bits)
            if bits is None:
                continue
            self.smiles.append(canon)
            self.ik14.append(key)
            j = len(self.smiles) - 1
            rows.extend([j] * len(bits))
            cols.extend(bits)
        data = np.ones(len(rows), dtype=np.float32)
        self.fps = sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(self.smiles), n_bits)
        )
        self.popcounts = np.asarray(self.fps.sum(axis=1)).ravel().astype(np.float32)
        self.n_bits = n_bits
        self.exact_mass: np.ndarray | None = None

    def __len__(self):
        return len(self.smiles)

    def add_exact_masses(self):
        """Precompute neutral exact masses (for precursor-mass filtering)."""
        from rdkit.Chem import Descriptors
        from rdkit import Chem

        masses = np.full(len(self.smiles), np.nan, dtype=np.float64)
        for i, smi in enumerate(self.smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                masses[i] = Descriptors.ExactMolWt(mol)
        self.exact_mass = masses
        return self

    def rank(self, pred_probs, top_n: int = 25, neutral_mass_hint=None,
             mass_tol_da: float | None = None):
        """Rank pool candidates by probabilistic Tanimoto similarity.

        ``pred_probs``: (n_bits,) predicted fingerprint probabilities.
        Returns ``[(smiles, score)]`` sorted by score desc.
        """
        if len(self) == 0:
            return []
        p = np.asarray(pred_probs, dtype=np.float64)
        inter = self.fps @ p  # (n_cand,)
        denom = p.sum() + self.popcounts - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.where(denom > 0, inter / np.maximum(denom, 1e-9), 0.0)
        if (neutral_mass_hint is not None and mass_tol_da is not None
                and self.exact_mass is not None):
            ok = np.abs(self.exact_mass - neutral_mass_hint) <= mass_tol_da
            scores = np.where(ok, scores, -1.0)
        if top_n >= len(scores):
            order = np.argsort(-scores)
        else:
            order = np.argpartition(-scores, top_n)[:top_n]
            order = order[np.argsort(-scores[order])]
        return [(self.smiles[i], float(scores[i])) for i in order
                if scores[i] > 0]


def rank_molecule_by_fingerprint(model, spectra, pool, device, top_n=25,
                                 mass_tol_da=None, batch_size=256):
    """Rank pool candidates for one molecule: average predicted FP over its
    spectra, then Tanimoto-rank the pool (optionally mass-filtered)."""
    probs = predict_fingerprints(model, spectra, device, batch_size=batch_size)
    if probs.shape[0] == 0:
        return []
    mean_prob = probs.mean(axis=0)
    hint = None
    if mass_tol_da is not None:
        masses = [
            neutral_mass(float(s["precursor_mz"]), s.get("adduct"))
            for s in spectra
        ]
        masses = [m for m in masses if m is not None]
        if masses:
            hint = float(np.median(masses))
    return pool.rank(mean_prob, top_n=top_n, neutral_mass_hint=hint,
                     mass_tol_da=mass_tol_da)
