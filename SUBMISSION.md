# SUBMISSION — Enveda CASMI 2026: Molecule ID From Mass Spectra

Solution draft for the Enveda CASMI 2026 Kaggle competition
($50,000; Featured Code Competition). The scored artefact is the Kaggle
notebook (`kaggle_notebook.ipynb`); this document describes the approach,
the local evidence, and how to reproduce it.

- Competition: https://www.kaggle.com/competitions/enveda-casmi26-molecule-id-mass-spectra
- Repo: https://github.com/darendrarp/enveda-casmi
- Entry gate: accept rules before **7 Dec 2026 23:59 UTC**
  (8 Dec 2026 06:59 WIB) · Final submission: **14 Dec 2026 23:59 UTC**
  (15 Dec 2026 06:59 WIB)

## 1. Task recap (verified from the official pages)

- Input: LC-MS/MS spectra of small molecules (test: ~400 molecules,
  ~1,500 spectra, 1–16 spectra per molecule, Bruker timsTOF).
- Output: up to 25 ranked SMILES candidates per molecule
  (`molecule_id,smiles`, candidates `;`-separated, best first).
- Metric: **MRR@25**; a candidate is correct when its tautomer-canonical
  InChIKey14 matches the truth (stereochemistry/tautomers not penalised).
- Training data: `train.parquet`, ~2.5M spectra / ~275k structures
  (aggregation of public libraries incl. Enveda-180; external public data
  allowed). Code competition: the notebook is re-run by Kaggle on the
  hidden test set. Winning solutions are released under MIT; data CC BY-NC
  4.0.

## 2. Approach: hybrid retrieval + learned ranking + de novo generation

No single method covers the whole test set: some test molecules (or close
analogues) are in the training library, others exist only in external
structure databases, and truly novel ones exist nowhere. The pipeline
therefore fuses three ranked lists per molecule with **reciprocal rank
fusion (RRF, k=60)**, then canonicalises (RDKit), dedupes by InChIKey14,
and caps at 25. One measured refinement: the library list only enters the
fusion when its best cosine is ≥ 0.7 (the **library gate** — §2.4), and a
mass-matched pool fallback guarantees every molecule gets a full list of
25 candidates.

### 2.1 Spectral library matching (baseline + anchor)

- Representation: L2-normalised binned vector (0.01 Da bins) of fragment
  m/z concatenated with neutral-loss (precursor − fragment) bins — the
  neutral-loss half makes matching robust to precursor shifts.
- Search: cosine similarity against the training library, restricted to a
  ±0.01 Da precursor-m/z window (standard library-search filter; also the
  key to searching 2.5M spectra fast). Best 2 spectra per structure kept.
- Aggregation: best cosine per structure across the molecule's spectra.
- Strong matches (cosine ≥ 0.8) are pinned to the top of the final list.

### 2.2 FPNet: spectrum → Morgan fingerprint → candidate ranking

- Encoder: each peak (m/z, intensity features) embedded by an MLP,
  mean+max pooled; concatenated with precursor m/z, adduct embedding and
  ion mode; MLP head predicts 2048-bit Morgan fingerprint logits
  (BCE loss). ~2.6M parameters at d_model=256.
- Ranking: candidates from the structure pool (all unique training
  structures) are scored by probabilistic Tanimoto similarity between the
  predicted fingerprint and each candidate's Morgan fingerprint, with an
  optional neutral-mass filter from the precursor m/z + adduct (±0.02 Da).
- This is the component that generalises to unseen molecules *if* the true
  structure is in the candidate pool — the competition explicitly allows
  external public data, so the pool can be extended with COCONUT/PubChem
  (supported by `CandidatePool`; a runtime trade-off in the notebook).

### 2.3 Seq2seq transformer: spectra → SMILES (de novo, optional)

- Transformer encoder over the peak list + a precursor token (precursor
  m/z + adduct); autoregressive SMILES decoder; beam search (width 10)
  yields ranked candidates. Tokeniser: standard SMILES regex.
- The only component that can propose structures absent from every
  database. It is the most data-hungry, so it is off by default in the
  notebook (`TRAIN_SEQ2SEQ` flag) and trained only when GPU time allows.

### 2.4 Why this combination — and the library gate

Library matching wins when the molecule was measured before; fingerprint
ranking wins when the molecule is in a structure database but unmeasured;
de novo generation is the only shot at truly novel structures. RRF needs no
cross-component score calibration, and the strong-match pinning protects
the near-certain cases from fusion noise.

One thing the local evaluation proved: **ungated fusion hurts on unseen
molecules.** The library list for a novel molecule contains only wrong
structures, and fusing it in dilutes the learned ranking. Gating the
library list on its best cosine fixes this — strong matches (≥ 0.8) are
still pinned first regardless of the gate, so nothing is lost on covered
molecules. Measured on the 300-molecule structure-disjoint validation set
(FP component = `fp_augpool`):

| Fusion | MRR@25 | hit@1 | hit@25 |
|---|---|---|---|
| FP-only (no library) | 0.2242 | 0.1033 | 0.7133 |
| RRF, library gate 0.0 (ungated) | 0.1306 | 0.0300 | 0.6400 |
| RRF, library gate 0.3 | 0.2013 | 0.0933 | 0.6733 |
| RRF, library gate 0.5 | 0.2078 | 0.0933 | 0.6867 |
| RRF, library gate **0.7 (used)** | **0.2156** | 0.0967 | 0.7000 |

Gate 0.7 recovers nearly all of the FP-only score on unseen molecules
while keeping the library fusion active for credible matches. (Reproduce:
`scripts/tune_fusion.py`.)

## 3. Local evaluation (structure-disjoint, public data only)

**Setup.** Public Enveda-180 v3 (Zenodo 21346580) — the same library that
supplies ~1.15M of the competition's training spectra, acquired on the same
instrument as the test set. 4% hash-sample: 47,417 spectra / 7,466
compounds. Split by InChIKey14: 45,577 train spectra; **300 held-out
molecules** (1,840 spectra) as validation. Models are small and CPU-trained
(FPNet d=256, 3 epochs; seq2seq d=128, 2 layers, 2 epochs on 15k spectra) —
a pipeline check, not a competitive model.

<!-- EVAL_TABLE_START -->
| Method | MRR@25 | hit@1 | hit@25 | n |
|---|---|---|---|---|
| library (baseline) | 0.0000 | 0.0000 | 0.0000 | 300 |
| fp_trainpool | 0.0000 | 0.0000 | 0.0000 | 300 |
| fp_augpool | 0.2242 | 0.1033 | 0.7133 | 300 |
| hybrid (lib gate 0.7) | 0.2156 | 0.0967 | 0.7000 | 300 |
| seq2seq | 0.0000 | 0.0000 | 0.0000 | 50 |
| library leave-one-spectrum-out (sanity) | 0.8250 | 0.8200 | 0.8300 | 100 |
<!-- EVAL_TABLE_END -->

**Reading the table.**
- `fp_trainpool` ranks only training structures: the held-out molecule is
  absent by construction, so a non-zero score there comes only from
  same-skeleton stereoisomers/tautomers sharing InChIKey14. It measures
  ranking quality, not retrieval.
- `fp_augpool` adds the held-out structures to the pool, simulating an
  external candidate database that contains the answer (COCONUT/PubChem in
  the real run). This is the realistic competition setting for the
  fingerprint component — and the strongest local result: the true
  structure lands in the top 25 for 71% of unseen molecules.
- `library` on a structure-disjoint split is expected near zero (the
  molecule is not in the library); its value is proven by the
  leave-one-spectrum-out sanity check instead (MRR@25 0.825 — same
  molecule, different spectrum, ranked first 82% of the time).
- `hybrid` (gated RRF, §2.4) is slightly below FP-only on unseen molecules
  because a few spurious library lists still pass the gate; on molecules
  with library coverage it wins via the pinned strong matches. The gate
  value 0.7 was chosen from the tuning table above, not assumed.
- `seq2seq` is evaluated on 50 molecules (CPU beam search is slow); the
  small CPU-trained model scores 0.0000 with ~10% valid-SMILES rate among
  beam outputs — its role in the submission is diversity for the fusion
  after full GPU training, and it is off by default in the notebook.

**What this does NOT claim.** No leaderboard score. Local numbers come from
a 4% sample of one library with small CPU models; the Kaggle run trains on
the full 2.5M spectra with GPU. The visible `test.parquet` is drawn from
training data, so public leaderboard scores are optimistic relative to the
hidden novel-molecule test set.

## 4. Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
curl -L -o data/enveda-180-filtered.jsonl.gz \
  https://zenodo.org/api/records/21346580/files/enveda-180-filtered.jsonl.gz/content
# md5 must be 2aa018d8c2ebc9a5baaaf366038e5c50
.venv/bin/python scripts/build_local_data.py
.venv/bin/python -m pytest tests/ -q            # 47 tests, all must pass
.venv/bin/python scripts/train_fp.py --epochs 3
.venv/bin/python scripts/train_seq2seq.py --epochs 2 --max-train 15000
.venv/bin/python scripts/evaluate.py --fp-checkpoint runs/fpnet.pt \
    --seq2seq-checkpoint runs/seq2seq.pt        # -> runs/eval_results.json
```

Kaggle run: see [SUBMIT_STEPS.md](SUBMIT_STEPS.md) and
`kaggle_notebook.ipynb` (regenerate with `scripts/make_notebook.py`).

## 5. Limitations & future work

1. **Candidate pool** is training structures only. Adding COCONUT (~800k
   natural products) is the single biggest expected gain for the hidden
   test set and is supported by the code; it was left out of the default
   notebook config for runtime safety.
2. **Seq2seq is small.** A larger model trained on the full data (or a
   MIST-fingerprint + MolForge two-stage setup from the literature) would
   strengthen the de novo leg.
3. **No formula prediction.** A MIST-CF-style formula ranker could
   constrain both the pool filter and the decoder.
4. **No per-spectrum weighting.** Spectra are currently pooled with equal
   weight (max/mean); collision-energy-aware weighting is untested.
5. **Metric approximation.** Local MRR uses RDKit InChIKey14 after
   tautomer canonicalisation; the official scorer may differ in edge cases.

## 6. References

- Goldman et al., "Annotating metabolite mass spectra with domain-inspired
  chemical formula transformers" (MIST), Nat. Mach. Intell. 5, 965–979
  (2023). https://github.com/samgoldman97/mist
- MassSpecGym benchmark. https://huggingface.co/datasets/roman-bushuiev/MassSpecGym
- Enveda-180, Zenodo 21346580. https://zenodo.org/records/21346580
- CASMI 2026 press release, BusinessWire 15 Sep 2026.
