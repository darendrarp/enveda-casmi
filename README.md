# Enveda CASMI 2026 — Molecule ID From Mass Spectra

Hybrid spectra→structure pipeline for the [Enveda CASMI 2026 Kaggle
competition](https://www.kaggle.com/competitions/enveda-casmi26-molecule-id-mass-spectra)
($50,000; predict 2D chemical structures as SMILES from LC-MS/MS spectra,
scored by MRR@25 with InChIKey14 matching).

**Status: submission-ready package.** The Kaggle notebook
(`kaggle_notebook.ipynb`) runs end-to-end with one click (GPU + Internet on)
and writes a validated `submission.csv`. Local pipeline verified with 47
passing unit tests and a structure-disjoint evaluation on public data (see
below). No leaderboard score is claimed — submission needs the user's Kaggle
account (see [SUBMIT_STEPS.md](SUBMIT_STEPS.md)).

## Approach

For each test molecule, three ranked candidate lists are fused with
reciprocal rank fusion (RRF):

1. **Spectral library matching** — binned fragment + neutral-loss cosine
   similarity against the training library, precursor-mass windowed
   (±0.01 Da). Strong matches (cosine ≥ 0.8) are pinned to the top.
2. **FPNet fingerprint ranking** — a peak-set encoder predicts a 2048-bit
   Morgan fingerprint from the spectra; candidates from the structure pool
   are ranked by probabilistic Tanimoto similarity, optionally filtered by
   precursor-derived neutral mass (±0.02 Da).
3. **Seq2seq transformer (optional)** — a spectra→SMILES transformer with
   beam search; the only component that can propose structures absent from
   every database.

The library list only enters the fusion when its best cosine is ≥ 0.7
(the **library gate**): on unseen molecules the library list is noise and
ungated fusion measurably hurts (local MRR@25 0.131 ungated vs 0.216
gated). Candidates are RDKit-canonicalised, deduplicated by InChIKey14,
capped at 25 per molecule, and written as `molecule_id,smiles` (candidates
separated by `;`). Details: [SUBMISSION.md](SUBMISSION.md).

## Repo layout

```
kaggle_notebook.ipynb   One-click Kaggle notebook (GPU): train + predict + submit
src/
  chem.py               SMILES / InChIKey14 / fingerprints (RDKit)
  spectra.py            Peak cleaning, binned spectrum vectors
  metrics.py            MRR@25 (InChIKey14 matching)
  submission.py         submission.csv writer + validator
  io_kaggle.py          Loaders for train.parquet / test.parquet
  io_enveda.py          Streaming loader for public Enveda-180 (Zenodo)
  splits.py             Structure-disjoint splits, reservoir sampling
  retrieval.py          Spectral library matching
  fingerprint.py        FPNet + candidate-pool Tanimoto ranking
  seq2seq.py            Transformer spectra -> SMILES + beam search
  pipeline.py           Hybrid RRF prediction per molecule
scripts/
  build_local_data.py   Sample Enveda-180 -> local train/val parquet
  train_fp.py           Train FPNet locally
  train_seq2seq.py      Train the transformer locally
  evaluate.py           Local MRR@25 evaluation (all components)
  predict_test.py       Full pipeline on a Kaggle-format test parquet
  make_notebook.py      Regenerate kaggle_notebook.ipynb
tests/                  47 unit tests (pytest)
```

## Local evaluation (honest numbers)

Held-out validation uses **300 molecules from public Enveda-180** (a subset
of the competition training data), structure-disjoint: no validation
molecule appears in training. The local models are small and CPU-trained on
a 4% Enveda-180 sample (45,577 train spectra) — the Kaggle run trains on
the full 2.5M spectra with GPU, so local numbers are a floor, not a
prediction of leaderboard score.

<!-- EVAL_TABLE_START -->
| Method | MRR@25 | hit@1 | hit@25 | n |
|---|---|---|---|---|
| library (baseline) | 0.0000 | 0.0000 | 0.0000 | 300 |
| fp_trainpool | 0.0000 | 0.0000 | 0.0000 | 300 |
| fp_augpool | **0.2242** | 0.1033 | 0.7133 | 300 |
| hybrid (lib gate 0.7) | 0.2156 | 0.0967 | 0.7000 | 300 |
| hybrid (ungated RRF) | 0.1306 | 0.0300 | 0.6400 | 300 |
| seq2seq (de novo) | 0.0000 | 0.0000 | 0.0000 | 50 |
| library leave-one-spectrum-out (sanity) | 0.8250 | 0.8200 | 0.8300 | 100 |
<!-- EVAL_TABLE_END -->

Notes on the settings: `fp_trainpool` ranks only training structures (the
held-out molecule is absent by construction, so this measures ranking
quality, not retrieval); `fp_augpool` adds the held-out structures to the
pool, simulating an external candidate database (e.g. COCONUT/PubChem) that
contains the answer — the realistic competition setting. The
leave-one-spectrum-out check proves the retrieval harness works (same
molecule, different spectrum, must rank first).

## Run locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# torch CPU (the pytorch index can be slow behind proxies; if pip times out,
# download the wheel with curl and pip install the file):
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

# 1. Public data (Enveda-180 v3, Zenodo 21346580, ~312 MB, md5 verified)
curl -L -o data/enveda-180-filtered.jsonl.gz \
  https://zenodo.org/api/records/21346580/files/enveda-180-filtered.jsonl.gz/content
.venv/bin/python scripts/build_local_data.py

# 2. Tests (47 tests, must all pass)
.venv/bin/python -m pytest tests/ -q

# 3. Smoke training + evaluation (CPU, ~15-25 min)
.venv/bin/python scripts/train_fp.py --epochs 3
.venv/bin/python scripts/train_seq2seq.py --epochs 2 --max-train 15000
.venv/bin/python scripts/evaluate.py --fp-checkpoint runs/fpnet.pt \
    --seq2seq-checkpoint runs/seq2seq.pt
```

## Run on Kaggle

See [SUBMIT_STEPS.md](SUBMIT_STEPS.md). Short version: accept the rules
**before 7 Dec 2026 23:59 UTC** (entry gate), upload
`kaggle_notebook.ipynb`, set GPU T4 + Internet ON, Run All with
`QUICK_MODE = True` first (~15 min), then `QUICK_MODE = False` (~2–3 h),
submit before **14 Dec 2026 23:59 UTC**.

## Caveats (read before trusting any number here)

- **No leaderboard score.** The official dataset needs a Kaggle account;
  local evaluation uses public Enveda-180 only. The visible `test.parquet`
  is drawn from training data, so public leaderboard scores (other teams')
  are optimistic vs the hidden novel-molecule test set.
- **Local models are small.** CPU smoke-training on 4% of one library is a
  pipeline check, not a competitive model. The Kaggle notebook trains on
  the full data with GPU.
- **Candidate pool = training structures only.** Adding COCONUT/PubChem
  would extend natural-product coverage; the pipeline supports any pool
  (see `CandidatePool`), it is a runtime trade-off for the notebook.
- **Metric approximation.** Local MRR uses RDKit InChIKey14 after tautomer
  canonicalisation, matching the competition's stated rule as closely as
  possible; the official scorer may differ in edge cases.

## Sources

- Competition: https://www.kaggle.com/competitions/enveda-casmi26-molecule-id-mass-spectra
- Press release (BusinessWire, 15 Sep 2026): https://www.businesswire.com/news/home/20260915334355/en/Enveda-Is-Launching-the-Next-AlphaFold-Contest.-This-Time-to-Solve-the-Structures-of-90-of-Biological-Molecules-That-No-One-Can-Identify.
- Enveda-180 dataset (Zenodo): https://zenodo.org/records/21346580
- MIST (Goldman et al., Nat. Mach. Intell. 2023): https://github.com/samgoldman97/mist
- MassSpecGym: https://huggingface.co/datasets/roman-bushuiev/MassSpecGym

## Licence

MIT — see [LICENSE](LICENSE). Competition data is CC BY-NC 4.0
(competition + non-commercial research use only).
