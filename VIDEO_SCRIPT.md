# Video / Write-up — Enveda CASMI 2026

## No video is required

Kaggle code competitions do **not** require a video demo. There is no video
deliverable for Enveda CASMI 2026: the submission is the notebook itself,
re-run by Kaggle on the hidden test set, and winners are asked for code
(winning solutions are released under the MIT licence) plus prize paperwork
— not a video.

So instead of a video script, this file contains the **solution write-up
draft** (English), ready to paste as the competition solution description /
discussion post after submitting. Numbers are the measured local results
(see SUBMISSION.md for the full tables).

---

## Solution write-up draft

### Hybrid retrieval + fingerprint ranking (+ optional de novo generation) for molecule ID from MS/MS

**Summary.** For each test molecule we rank up to 25 candidate structures by
fusing three complementary ranked lists with reciprocal rank fusion (RRF):
(1) spectral library matching, (2) learned Morgan-fingerprint ranking over a
structure pool, and (3) optional de novo SMILES generation from a
spectra→SMILES transformer. Candidates are canonicalised with RDKit,
deduplicated by InChIKey14 (the competition's identity rule), and capped at
25 per molecule.

**1. Library matching (baseline + strong-match anchor).** Each spectrum is
encoded as an L2-normalised binned vector (0.01 Da bins) concatenating
fragment m/z and neutral-loss (precursor − fragment) histograms. Queries are
compared by cosine similarity against the training library, restricted to
library spectra within a ±0.01 Da precursor-m/z window. Per molecule we keep
the best score per structure across all its spectra. Matches with cosine
≥ 0.8 are pinned to the top of the final ranking — on the visible test data
these are near-certain identifications.

**2. FPNet fingerprint ranking (the learned component).** A peak-set encoder
(each peak embedded by an MLP over [m/z, intensity] features, mean+max
pooled, concatenated with precursor m/z, adduct and ion-mode features)
predicts a 2048-bit Morgan fingerprint. Candidates from the structure pool
(all unique training structures) are scored by probabilistic Tanimoto
similarity to the predicted fingerprint and optionally filtered by the
precursor-derived neutral mass (±0.02 Da). A library gate keeps the fusion
honest: the library list only enters the RRF when its best cosine is ≥ 0.7
(ungated fusion measurably hurts on unseen molecules — local MRR@25 0.131
vs 0.216 gated). Strong library matches (cosine ≥ 0.8) are pinned first
regardless of the gate. This generalises to unseen
molecules whenever the true structure is in the candidate pool — the intended
use of the competition's external-data allowance. The model is trained on
400k training spectra for 3 epochs on a Kaggle T4.

**3. De novo transformer (optional).** A Transformer encoder over the peak
list plus a precursor token, with an autoregressive SMILES decoder and beam
search (width 10), proposes structures absent from every database — the only
route to truly novel natural products. It is trained separately and fused
into the same RRF ranking.

**Validation (local, structure-disjoint).** We held out 300 molecules from
the public Enveda-180 library (a subset of the competition training data)
with all their spectra, so no validation molecule appears in training.
Local MRR@25: library 0.0000, FPNet vs train-only pool 0.0000, FPNet vs
pool augmented with the held-out structures (simulating an external database
containing the answer) 0.2242, hybrid (gated RRF) 0.2156, seq2seq 0.0000
(50 molecules, ~10% valid SMILES). A leave-one-spectrum-out
sanity check on training molecules gives library MRR@25 0.8250, confirming
the retrieval harness works. Note: local numbers use a small CPU-trained
model on a 4% Enveda-180 sample and are not comparable to leaderboard
scores; the Kaggle run trains on the full 2.5M spectra with GPU.

**Reproducibility.** Code (MIT): https://github.com/darendrarp/enveda-casmi —
`src/` (pipeline), `kaggle_notebook.ipynb` (one-click Run All on Kaggle GPU),
`tests/` (47 unit tests, all passing). Local data: Enveda-180 v3 (Zenodo
21346580, CC BY 4.0). The notebook streams `train.parquet` once, builds the
index + pool + training sample, trains, predicts, and validates the
submission file before writing it.

**Known limitations.** (a) The candidate pool is training structures only;
adding COCONUT/PubChem would extend coverage of natural products — the
pipeline supports any pool, it is a runtime trade-off. (b) The local
seq2seq model is small and CPU-trained; its local MRR is near zero and it
contributes mainly as a diversity source in the fusion. (c) The visible
test file is drawn from training data, so public leaderboard scores are
optimistic relative to the hidden novel-molecule test set.
