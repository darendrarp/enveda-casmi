"""Spectral library matching baseline.

A query spectrum is compared against a library of reference spectra with the
cosine similarity of binned fragment + neutral-loss vectors (see
``spectra.spectrum_vector``). Only library spectra whose precursor m/z falls
within a small window of the query precursor are considered — the standard
precursor-mass filter of library search, which also keeps the search fast
on a 2.5M-spectrum library.

Per-molecule predictions aggregate the best score per structure across all
of the molecule's spectra.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .spectra import spectrum_vector


class LibraryIndex:
    """Mass-windowed cosine-similarity search over reference spectra."""

    def __init__(self, max_per_structure: int = 2, bin_width: float = 0.01,
                 mz_max: float = 1000.0, nl_max: float = 500.0):
        self.max_per_structure = max_per_structure
        self.bin_width = bin_width
        self.mz_max = mz_max
        self.nl_max = nl_max
        self.matrix = None          # CSR (n_lib, dim), L2-normalised rows
        self.precursor_mz = None    # (n_lib,) sorted ascending
        self.smiles = None          # (n_lib,) representative SMILES per row
        self.ik14 = None            # (n_lib,) InChIKey14 per row
        self.n_structures = 0

    def __len__(self):
        return 0 if self.matrix is None else self.matrix.shape[0]

    def build(self, rows):
        """Build the index from row dicts (competition / Enveda format).

        Keeps up to ``max_per_structure`` spectra per (inchikey14,
        ionization_mode), preferring spectra with the most peaks.
        """
        by_struct: dict[tuple, list] = {}
        for r in rows:
            key = (r["inchikey14"], r.get("ionization_mode"))
            by_struct.setdefault(key, []).append(r)
        kept = []
        for key, rs in by_struct.items():
            rs.sort(key=lambda r: len(r["ms2_mzs"]), reverse=True)
            kept.extend(rs[: self.max_per_structure])
        kept.sort(key=lambda r: float(r["precursor_mz"]))
        vecs = []
        for r in kept:
            vecs.append(
                spectrum_vector(
                    r["ms2_mzs"], r["ms2_normalized_intensities"],
                    float(r["precursor_mz"]),
                    bin_width=self.bin_width, mz_max=self.mz_max,
                    nl_max=self.nl_max,
                )
            )
        dim = int(self.mz_max / self.bin_width) + int(self.nl_max / self.bin_width)
        if vecs:
            self.matrix = sparse.vstack(vecs, format="csr").astype(np.float32)
        else:
            self.matrix = sparse.csr_matrix((0, dim), dtype=np.float32)
        self.precursor_mz = np.array([float(r["precursor_mz"]) for r in kept],
                                     dtype=np.float64)
        self.smiles = [r["normalized_smiles"] for r in kept]
        self.ik14 = [r["inchikey14"] for r in kept]
        self.n_structures = len({k[0] for k in by_struct})
        return self

    def build_streaming(self, row_iter):
        """Build from a row iterator without holding all rows in memory.

        Keeps up to ``max_per_structure`` spectra per (inchikey14,
        ionization_mode) — the ones with the most peaks — then builds the
        index from the survivors only. Use this for the 2.5M-row
        ``train.parquet``.
        """
        kept: dict[tuple, list] = {}
        for r in row_iter:
            key = (r["inchikey14"], r.get("ionization_mode"))
            n = len(r["ms2_mzs"])
            lst = kept.get(key)
            if lst is None:
                kept[key] = [(n, r)]
            elif len(lst) < self.max_per_structure:
                lst.append((n, r))
            else:
                min_i = min(range(len(lst)), key=lambda i: lst[i][0])
                if n > lst[min_i][0]:
                    lst[min_i] = (n, r)
        rows = [r for lst in kept.values() for _, r in lst]
        return self.build(rows)

    def _window(self, precursor_mz, tol_da):
        if tol_da is None:
            return 0, len(self)
        lo = np.searchsorted(self.precursor_mz, precursor_mz - tol_da, side="left")
        hi = np.searchsorted(self.precursor_mz, precursor_mz + tol_da, side="right")
        return int(lo), int(hi)

    def query(self, mzs, ints, precursor_mz, top_n: int = 25,
              tol_da: float | None = 0.01):
        """Rank library structures for one spectrum.

        Returns ``[(smiles, cosine_score)]`` sorted by score desc, best
        spectrum per structure.
        """
        if len(self) == 0:
            return []
        lo, hi = self._window(float(precursor_mz), tol_da)
        if hi <= lo:
            return []
        q = spectrum_vector(mzs, ints, float(precursor_mz),
                            bin_width=self.bin_width, mz_max=self.mz_max,
                            nl_max=self.nl_max)
        sims = (self.matrix[lo:hi] @ q.T).toarray().ravel()
        best: dict[str, tuple[str, float]] = {}
        for j, s in enumerate(sims):
            if s <= 0:
                continue
            key = self.ik14[lo + j]
            if key not in best or s > best[key][1]:
                best[key] = (self.smiles[lo + j], float(s))
        ranked = sorted(best.values(), key=lambda t: -t[1])
        return ranked[:top_n]

    def query_molecule(self, spectra, top_n: int = 25, tol_da: float | None = 0.01):
        """Rank library structures for one molecule (all its spectra).

        Scores are the max cosine per structure across the molecule's
        spectra (a structure matching any spectrum strongly wins).
        """
        best: dict[str, tuple[str, float]] = {}
        for spec in spectra:
            for smi, score in self.query(
                spec["ms2_mzs"], spec["ms2_normalized_intensities"],
                float(spec["precursor_mz"]), top_n=top_n, tol_da=tol_da,
            ):
                from .chem import inchikey14

                key = inchikey14(smi)
                if key is None:
                    continue
                if key not in best or score > best[key][1]:
                    best[key] = (smi, score)
        ranked = sorted(best.values(), key=lambda t: -t[1])
        return ranked[:top_n]
