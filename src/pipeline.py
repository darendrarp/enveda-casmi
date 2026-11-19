"""Hybrid prediction: combine retrieval + fingerprint ranking + seq2seq.

Ranked lists from the three components are fused with reciprocal rank
fusion (RRF, k=60), a standard rank-aggregation method that needs no
score calibration across heterogeneous components. Strong library matches
(cosine >= 0.8) are near-certain identifications and are pinned to the
top of the list before fusion. The fused list is canonicalised, deduped
by InChIKey14 and capped at 25 by the submission writer.
"""

from __future__ import annotations

from .chem import inchikey14


def pool_fallback(pool, spectra, top_n: int = 25):
    """Last-resort candidates from the structure pool.

    Mass-matched pool structures (closest |delta mass| first), widening
    the tolerance if needed, then topped up with the most popular pool
    structures (pool order) so the list always reaches ``top_n`` when the
    pool allows. Guarantees a non-empty candidate list whenever the pool
    is non-empty.
    """
    import numpy as np

    from .chem import neutral_mass

    if pool is None or len(pool) == 0:
        return []
    out: list[str] = []
    masses = [
        neutral_mass(float(s["precursor_mz"]), s.get("adduct"))
        for s in spectra
    ]
    masses = [m for m in masses if m is not None]
    if masses and pool.exact_mass is not None:
        hint = float(np.median(masses))
        for tol in (0.02, 0.1, 0.5):
            d = np.abs(pool.exact_mass - hint)
            idx = np.where(d <= tol)[0]
            if len(idx):
                idx = idx[np.argsort(d[idx])]
                out = [pool.smiles[i] for i in idx[:top_n]]
                break
    if len(out) < top_n:
        seen = set(out)
        for smi in pool.smiles:
            if smi not in seen:
                out.append(smi)
                seen.add(smi)
            if len(out) >= top_n:
                break
    return out


def rrf_fuse(ranked_lists, k: int = 60, top_n: int = 25):
    """Fuse ranked SMILES lists by reciprocal rank fusion.

    ``ranked_lists``: list of lists of SMILES (best first). Items are
    keyed by InChIKey14 so the same structure from two components merges.
    Returns a fused ranked list of SMILES.
    """
    scores: dict[str, float] = {}
    repr_smi: dict[str, str] = {}
    for lst in ranked_lists:
        for rank, smi in enumerate(lst, start=1):
            key = inchikey14(smi)
            if key is None:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            repr_smi.setdefault(key, smi)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    return [repr_smi[key] for key, _ in fused[:top_n]]


def predict_molecule(spectra, lib_index=None, fp_model=None, pool=None,
                     seq2seq_model=None, tokenizer=None, device="cpu",
                     top_n: int = 25, lib_tol_da: float | None = 0.01,
                     fp_mass_tol_da: float | None = 0.02,
                     seq2seq_beam: int = 10, strong_lib_threshold: float = 0.8,
                     lib_gate: float = 0.7):
    """Predict a ranked top-25 candidate list for one molecule.

    All components are optional; pass at least one. Returns SMILES strings.

    ``lib_gate``: the library list only enters the fusion when its best
    cosine is at least this high. For novel molecules the library list is
    pure noise and dilutes the learned ranking — verified locally on 300
    structure-disjoint molecules: ungated RRF MRR@25 0.131 vs FP-only
    0.224; gate 0.7 recovers 0.216 while keeping credible library lists
    (strong matches >= 0.8 are pinned first regardless of the gate).
    """
    from .fingerprint import rank_molecule_by_fingerprint

    ranked_lists = []
    strong_first = []

    if lib_index is not None:
        lib_hits = lib_index.query_molecule(spectra, top_n=top_n, tol_da=lib_tol_da)
        strong_first = [s for s, sc in lib_hits if sc >= strong_lib_threshold]
        if lib_hits and lib_hits[0][1] >= lib_gate:
            ranked_lists.append([s for s, _ in lib_hits])

    if fp_model is not None and pool is not None:
        fp_hits = rank_molecule_by_fingerprint(
            fp_model, spectra, pool, device, top_n=top_n,
            mass_tol_da=fp_mass_tol_da,
        )
        ranked_lists.append([s for s, _ in fp_hits])

    if seq2seq_model is not None and tokenizer is not None:
        import torch

        from .fingerprint import ADDUCT_INDEX
        from .spectra import peak_features

        # Generate from the molecule's most informative spectrum (most peaks).
        best = max(spectra, key=lambda s: len(s["ms2_mzs"]))
        feats = peak_features(best["ms2_mzs"], best["ms2_normalized_intensities"],
                              float(best["precursor_mz"]))
        peaks = torch.from_numpy(feats).unsqueeze(0).to(device)
        mask = torch.ones(1, feats.shape[0], dtype=torch.bool, device=device)
        prec = torch.tensor([float(best["precursor_mz"])], device=device)
        add = torch.tensor(
            [ADDUCT_INDEX.get(best.get("adduct"), ADDUCT_INDEX["OTHER"])],
            device=device)
        beams = seq2seq_model.generate_beam(
            peaks, mask, prec, add, tokenizer, beam_width=seq2seq_beam
        )
        ranked_lists.append([s for s, _ in beams])

    if not ranked_lists:
        # Every component came back empty (or the library list was gated
        # out): fall back to mass-matched pool candidates so the molecule
        # still gets a full list.
        if pool is not None:
            return pool_fallback(pool, spectra, top_n=top_n)
        return []

    fused = rrf_fuse(ranked_lists, top_n=top_n)
    # Pin strong library matches first (they are near-certain), keep order.
    if strong_first:
        strong_keys = {inchikey14(s) for s in strong_first}
        rest = [s for s in fused if inchikey14(s) not in strong_keys]
        fused = strong_first + rest
    # Top up with mass-matched pool candidates so the list is never empty
    # and always reaches top_n when the pool allows (extra candidates can
    # only add hit chances — existing ranks are unchanged).
    if len(fused) < top_n and pool is not None:
        have = {inchikey14(s) for s in fused}
        for cand in pool_fallback(pool, spectra, top_n=top_n):
            if inchikey14(cand) not in have:
                fused.append(cand)
                have.add(inchikey14(cand))
            if len(fused) >= top_n:
                break
    return fused[:top_n]
