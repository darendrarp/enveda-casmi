#!/usr/bin/env python3
"""Generate the Kaggle-ready notebook (kaggle_notebook.ipynb).

The notebook is deliberately thin: all pipeline code lives in the repo's
``src/`` package, which the notebook clones. Run All executes:
setup -> load data -> build library index + candidate pool -> train FPNet
(-> optionally train seq2seq) -> predict -> write + validate submission.csv.

Usage: python scripts/make_notebook.py
"""

import nbformat as nbf

REPO_URL = "https://github.com/darendrarp/enveda-casmi"

CELLS = []

# ---------------------------------------------------------------- intro
CELLS.append(nbf.v4.new_markdown_cell("""\
# Enveda CASMI 2026 — Molecule ID From Mass Spectra

**Approach: hybrid retrieval + learned fingerprint ranking + (optional) de novo generation.**
For each test molecule we combine three ranked candidate lists with reciprocal
rank fusion (RRF):

1. **Spectral library matching** — binned fragment + neutral-loss cosine
   similarity against the 2.5M training spectra (precursor-mass windowed).
2. **FPNet fingerprint ranking** — a peak-set encoder predicts a 2048-bit
   Morgan fingerprint from the spectra; candidates from the structure pool
   (training structures) are ranked by probabilistic Tanimoto similarity,
   optionally filtered by precursor-derived neutral mass.
3. **Seq2seq transformer (optional, off by default)** — a spectra→SMILES
   transformer with beam search, the only component that can propose
   structures absent from every database.

All pipeline code lives in the cloned repo (`src/`); this notebook only
orchestrates it. Full write-up: `SUBMISSION.md` in the repo.

**Before you run:** set this notebook to **GPU** (Settings → Accelerator →
GPU T4) and turn **Internet ON** (Settings → Internet) — needed to clone
the repo and install RDKit.

**Runtime estimates (T4):** quick mode ~15 min; full mode ~2–3 h
(9 h limit). Start with `QUICK_MODE = True` to smoke-test, then flip it
to `False` for the real run.

- Competition: https://www.kaggle.com/competitions/enveda-casmi26-molecule-id-mass-spectra
- Repo: https://github.com/darendrarp/enveda-casmi
- Entry deadline: 7 Dec 2026 23:59 UTC · Final submission: 14 Dec 2026 23:59 UTC
"""))

# ---------------------------------------------------------------- setup
CELLS.append(nbf.v4.new_code_cell("""\
# --- Setup: GPU check, RDKit, clone the repo ---
import os
import subprocess
import sys

import torch

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU detected — full mode will be very slow. "
          "Set Accelerator to GPU T4 in notebook Settings.")

# RDKit is not preinstalled in Kaggle images.
try:
    import rdkit
    print("rdkit", rdkit.__version__)
except ImportError:
    print("installing rdkit ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rdkit"])
    import rdkit
    print("rdkit", rdkit.__version__)

REPO_DIR = "/kaggle/working/enveda-casmi"
if not os.path.isdir(REPO_DIR):
    print("cloning repo ...")
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/darendrarp/enveda-casmi",
                           REPO_DIR])
sys.path.insert(0, REPO_DIR)

import numpy as np
import pandas as pd

from src.fingerprint import CandidatePool, FPNet, train_fpnet
from src.io_kaggle import (group_test_by_molecule, iter_train_batches,
                           load_test, parquet_schema, row_to_spectrum)
from src.pipeline import predict_molecule
from src.retrieval import LibraryIndex
from src.seq2seq import SmilesTokenizer, SpectraTransformer, train_seq2seq
from src.splits import ReservoirSampler, structure_disjoint_split
from src.submission import validate_submission, write_submission

torch.manual_seed(42)
np.random.seed(42)
print("setup OK")
"""))

# ---------------------------------------------------------------- config
CELLS.append(nbf.v4.new_code_cell("""\
# --- Configuration ---
QUICK_MODE = True        # True: smoke test on a subset (~15 min). False: full run.
TRAIN_SEQ2SEQ = False    # De novo transformer. Off by default: needs ~1-2 h
                         # extra GPU time and helps mainly for structures
                         # absent from the candidate pool. Flip on for the
                         # final run if time allows.
INPUT_DIR = "/kaggle/input/enveda-casmi26-molecule-id-mass-spectra"
WORK_DIR = "/kaggle/working"
TRAIN_PATH = f"{INPUT_DIR}/train.parquet"
TEST_PATH = f"{INPUT_DIR}/test.parquet"
SAMPLE_PATH = f"{INPUT_DIR}/sample_submission.csv"
OUT_PATH = f"{WORK_DIR}/submission.csv"

if QUICK_MODE:
    MAX_TRAIN_SPECTRA = 50_000    # streamed cap for index+pool+sampling
    MAX_TRAIN_ROWS = 20_000       # reservoir sample used to train FPNet
    FP_EPOCHS = 1
    S2S_EPOCHS = 1
else:
    MAX_TRAIN_SPECTRA = 0         # 0 = all 2.5M spectra
    MAX_TRAIN_ROWS = 400_000
    FP_EPOCHS = 3
    S2S_EPOCHS = 2

FP_D_MODEL = 256
FP_BATCH = 256
S2S_D_MODEL = 256
S2S_LAYERS = 4
S2S_BATCH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"QUICK_MODE={QUICK_MODE} TRAIN_SEQ2SEQ={TRAIN_SEQ2SEQ} device={DEVICE}")
"""))

# ---------------------------------------------------------------- data check
CELLS.append(nbf.v4.new_markdown_cell("""\
## 1. Verify the competition files
"""))
CELLS.append(nbf.v4.new_code_cell("""\
import glob

print("input files:", sorted(os.path.basename(p) for p in glob.glob(f"{INPUT_DIR}/*")))
for name, path in [("train", TRAIN_PATH), ("test", TEST_PATH)]:
    print(f"{name} schema: {parquet_schema(path)}")

test_df = load_test(TEST_PATH)
print(f"test rows: {len(test_df)}, molecules: {test_df['molecule_id'].nunique()}")
print(test_df.head(2).to_string())

sample = pd.read_csv(SAMPLE_PATH)
print(f"sample_submission rows: {len(sample)}, columns: {list(sample.columns)}")
print(sample.head(2).to_string())
assert set(sample.columns) == {"molecule_id", "smiles"}, "unexpected sample format"
assert set(sample["molecule_id"]) == set(test_df["molecule_id"]), \\
    "sample_submission molecule ids do not match test.parquet"
print("sample submission format OK: one row per molecule, ';'-separated candidates")
"""))

# ---------------------------------------------------------------- index + pool
CELLS.append(nbf.v4.new_markdown_cell("""\
## 2. Build the library index, candidate pool, and training sample

A single streaming pass over `train.parquet` (2.5M rows) feeds three things:
- the **library index** (best 2 spectra per structure, memory-bounded),
- the **candidate pool** (unique training structures),
- a **reservoir sample** of spectra for model training.
"""))
CELLS.append(nbf.v4.new_code_cell("""\
import time
from collections import Counter

t0 = time.time()
struct_counts = Counter()
reservoir = ReservoirSampler(MAX_TRAIN_ROWS, seed=42)
n_streamed = [0]


def row_stream():
    for batch in iter_train_batches(TRAIN_PATH, batch_size=100_000):
        for r in batch.to_dict("records"):
            struct_counts[r["normalized_smiles"]] += 1
            reservoir.add(r)
            n_streamed[0] += 1
            yield r
            if MAX_TRAIN_SPECTRA and n_streamed[0] >= MAX_TRAIN_SPECTRA:
                return
        print(f"  streamed {n_streamed[0]} spectra ({time.time() - t0:.0f}s)")


lib = LibraryIndex(max_per_structure=2).build_streaming(row_stream())
print(f"streamed {n_streamed[0]} train spectra in {time.time() - t0:.0f}s")
print(f"library index: {len(lib)} spectra, {lib.n_structures} structures")
print(f"candidate pool structures: {len(struct_counts)}")
print(f"training reservoir: {len(reservoir)} spectra")

pool = CandidatePool([s for s, _ in struct_counts.most_common()])
pool.add_exact_masses()
print(f"candidate pool built: {len(pool)} unique structures "
      f"(stereoisomers deduped by InChIKey14)")
"""))

# ---------------------------------------------------------------- train fp
CELLS.append(nbf.v4.new_markdown_cell("""\
## 3. Train FPNet (spectrum → Morgan fingerprint)
"""))
CELLS.append(nbf.v4.new_code_cell("""\
train_rows = reservoir.items
train_keys, val_keys = structure_disjoint_split(
    [r["inchikey14"] for r in train_rows], val_frac=0.05, seed=42)
fp_train = [r for r in train_rows if r["inchikey14"] in train_keys]
fp_val = [r for r in train_rows if r["inchikey14"] in val_keys]
print(f"FPNet train spectra: {len(fp_train)}, val spectra: {len(fp_val)}")

fp_model = FPNet(d_model=FP_D_MODEL)
print(f"FPNet params: {sum(p.numel() for p in fp_model.parameters()):,}")
fp_history = train_fpnet(fp_model, fp_train, fp_val, DEVICE,
                         epochs=FP_EPOCHS, batch_size=FP_BATCH, log=print)
torch.save({"state_dict": fp_model.state_dict(),
            "config": {"d_model": FP_D_MODEL, "n_bits": fp_model.n_bits},
            "history": fp_history},
           f"{WORK_DIR}/fpnet.pt")
print("saved fpnet.pt")
"""))

# ---------------------------------------------------------------- train s2s
CELLS.append(nbf.v4.new_markdown_cell("""\
## 4. (Optional) Train the spectra→SMILES transformer

Skipped unless `TRAIN_SEQ2SEQ = True`. This is the de novo component: it can
propose structures absent from the candidate pool, at the cost of extra GPU
time.
"""))
CELLS.append(nbf.v4.new_code_cell("""\
s2s_model, tokenizer = None, None
if TRAIN_SEQ2SEQ:
    tokenizer = SmilesTokenizer.fit([r["normalized_smiles"] for r in fp_train])
    print(f"tokenizer vocab: {len(tokenizer)}")
    s2s_model = SpectraTransformer(len(tokenizer), d_model=S2S_D_MODEL,
                                   nhead=8, num_encoder_layers=S2S_LAYERS,
                                   num_decoder_layers=S2S_LAYERS)
    print(f"seq2seq params: {sum(p.numel() for p in s2s_model.parameters()):,}")
    s2s_history = train_seq2seq(s2s_model, tokenizer, fp_train, fp_val, DEVICE,
                                epochs=S2S_EPOCHS, batch_size=S2S_BATCH,
                                log=print)
    torch.save({"state_dict": s2s_model.state_dict(),
                "vocab": tokenizer.vocab,
                "config": {"d_model": S2S_D_MODEL, "nhead": 8,
                           "num_encoder_layers": S2S_LAYERS,
                           "num_decoder_layers": S2S_LAYERS},
                "history": s2s_history},
               f"{WORK_DIR}/seq2seq.pt")
    print("saved seq2seq.pt")
else:
    print("TRAIN_SEQ2SEQ is False — skipping de novo transformer.")
"""))

# ---------------------------------------------------------------- predict
CELLS.append(nbf.v4.new_markdown_cell("""\
## 5. Predict all test molecules and write the submission
"""))
CELLS.append(nbf.v4.new_code_cell("""\
groups = group_test_by_molecule(test_df)
print(f"predicting {len(groups)} molecules ...")
t0 = time.time()
rows = []
for i, (mol_id, test_rows) in enumerate(sorted(groups.items())):
    spectra = [row_to_spectrum(r) for r in test_rows]
    cands = predict_molecule(
        spectra, lib_index=lib, fp_model=fp_model, pool=pool,
        seq2seq_model=s2s_model, tokenizer=tokenizer, device=DEVICE,
        lib_tol_da=0.01, fp_mass_tol_da=0.02, seq2seq_beam=10,
    )
    rows.append((mol_id, cands))
    if (i + 1) % 50 == 0:
        print(f"  ... {i + 1}/{len(groups)} ({time.time() - t0:.0f}s)")

n = write_submission(rows, OUT_PATH)
print(f"wrote {OUT_PATH} ({n} molecules) in {time.time() - t0:.0f}s")
"""))

# ---------------------------------------------------------------- validate
CELLS.append(nbf.v4.new_markdown_cell("""\
## 6. Validate the submission
"""))
CELLS.append(nbf.v4.new_code_cell("""\
problems = validate_submission(OUT_PATH,
                               expected_molecule_ids=list(groups))
if problems:
    print("VALIDATION PROBLEMS:")
    for p in problems[:20]:
        print(" -", p)
    raise SystemExit("submission invalid — fix before submitting")
print("submission validation: OK")

sub = pd.read_csv(OUT_PATH)
n_cands = sub["smiles"].str.count(";") + 1
print(f"rows: {len(sub)}")
print(f"candidates per molecule: min={n_cands.min()}, "
      f"median={int(n_cands.median())}, max={n_cands.max()}")
print(sub.head(3).to_string())
print()
print("Done. To submit: in the Kaggle notebook UI, click Submit (top right),")
print("or save this notebook version and use 'Submit to Competition'.")
"""))

# ---------------------------------------------------------------- assemble
nb = nbf.v4.new_notebook()
nb["cells"] = CELLS
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
    "kaggle": {"accelerator": "gpu", "internet": True},
}
out = "kaggle_notebook.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(CELLS)} cells)")
