# SUBMIT_STEPS — Enveda CASMI 2026 (Kaggle)

Step-by-step for the account holder (adt). Everything in this repo is
prepared; the steps below need a human with a Kaggle account because
submission requires login + accepting the competition rules. Nothing here
can be done by the agent (no Kaggle credentials are stored anywhere).

**Deadlines (hard cutoffs, UTC = WIB − 7h)**
- **Entry gate: accept the rules BEFORE 7 Dec 2026 23:59 UTC = 8 Dec 2026 06:59 WIB.**
  After this you cannot join at all. Do this step early — it takes 2 minutes
  and commits you to nothing.
- **Final submission: 14 Dec 2026 23:59 UTC = 15 Dec 2026 06:59 WIB.**
  Your final notebook version must be submitted and selected before this.
- Recommended: run the quick smoke test by 9–10 Dec, the full run by 12 Dec,
  so there is slack for one re-run if something fails.

**Links**
- Competition: https://www.kaggle.com/competitions/enveda-casmi26-molecule-id-mass-spectra
- Rules: https://www.kaggle.com/competitions/enveda-casmi26-molecule-id-mass-spectra/rules
- Data: https://www.kaggle.com/competitions/enveda-casmi26-molecule-id-mass-spectra/data
- Code (notebooks): https://www.kaggle.com/competitions/enveda-casmi26-molecule-id-mass-spectra/code
- This repo: https://github.com/darendrarp/enveda-casmi
- Notebook file: https://github.com/darendrarp/enveda-casmi/raw/main/kaggle_notebook.ipynb

---

## Step 0 — Kaggle account (one time, ~10 min)

1. Go to https://www.kaggle.com/ and sign up / sign in (Google sign-in is fine).
2. **Phone verification:** Kaggle requires a verified phone number to use free
   GPU notebooks. Go to Account settings → verify your phone number if asked.
   (If Kaggle's current flow differs, follow what the site asks for — this
   step needs checking in the browser.)
3. No payment method is needed. Do not enable any paid option.

## Step 1 — Accept the competition rules (BEFORE 8 Dec 06:59 WIB)

1. Open the competition page → **Rules** tab.
2. Read the rules, then click **"I Understand and Accept"** (or the equivalent
   accept button). This is the entry gate.
3. Confirm the competition now shows you as entered (the Data tab becomes
   downloadable / the Code tab allows new notebooks).
4. Note the rules on submissions per day (listed as up to 5/day in secondary
   sources — verify the exact number on the Rules tab) and the winner
   verification requirement (prize-acceptance documents + US tax form W-8BEN
   for non-US winners — already approved by you, only needed if you place
   in the top 5).

## Step 2 — Upload the notebook (~5 min)

1. Download `kaggle_notebook.ipynb` from the repo link above
   (or clone the repo and take the file from it).
2. On the competition page → **Code** tab → **New Notebook** → upload the
   `.ipynb` file. (If the upload flow differs, create a blank notebook and
   paste the cells from the file — the notebook is self-contained apart from
   the repo it clones.)
3. In the notebook's right-hand **Settings** panel, set:
   - **Accelerator: GPU T4** (required — CPU is far too slow for the full run)
   - **Internet: On** (required — the notebook clones this repo and installs
     RDKit)
   - Environment: default is fine.
4. Check the first cell: `QUICK_MODE = True` by default. Leave it for the
   smoke test.

## Step 3 — Smoke test (QUICK_MODE, ~15 min)

1. Click **Run All** (or Save & Run All).
2. Watch the output: it should stream ~50k training spectra, train FPNet for
   1 epoch, predict all test molecules, and end with
   `submission validation: OK` plus a small table.
3. If any cell errors, note the exact error message and stop — do not guess;
   the failure is diagnosable from the log. Common causes: Internet off
   (clone/pip fails), GPU not selected (very slow), or a Kaggle-side outage.

## Step 4 — Full run (QUICK_MODE = False, ~2–3 h)

1. Set `QUICK_MODE = False` in the config cell.
2. Optional: set `TRAIN_SEQ2SEQ = True` to also train the de novo transformer
   (adds ~1–2 h GPU; recommended for the final run if time allows — it is
   the only component that can propose structures absent from the candidate
   pool).
3. **Save & Run All.** Leave the tab open or let it run in the background;
   Kaggle notebooks keep running server-side (9 h limit).
4. Confirm the final cell prints `submission validation: OK` and the
   candidate-count summary (400 rows expected on the visible test file;
   the hidden test set is approximately the same size).

## Step 5 — Submit (before 15 Dec 06:59 WIB)

1. In the finished notebook, click **Submit** (top right) / **Save Version →
   Submit to Competition** and confirm. This creates a competition
   submission from the notebook run.
2. Go to the competition **Leaderboard / My Submissions** and confirm the
   submission appears with a public score. (Note: the visible test file is
   drawn from training data, so the public score is optimistic and NOT
   representative of the hidden test set — do not over-tune to it.)
3. **Select this submission as your final submission** if Kaggle asks you to
   pick (code competitions score the selected submission on the hidden test
   set during re-run). Verify the selection stuck.
4. You may submit up to the daily limit (verify the number on the Rules tab)
   — e.g. once for the quick run and once for the full run.

## Step 6 — If you place in the top 5 (after 15 Dec)

Kaggle/Enveda will contact winners for verification: prize-acceptance
documents + US tax form W-8BEN (non-US winners). This was pre-approved on
19 Nov 2026. No action is needed unless you are contacted.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `git clone` fails / pip install fails | Internet is OFF in notebook settings — turn it on and re-run. |
| Run is extremely slow | Accelerator is CPU — set GPU T4 and re-run. |
| `FileNotFoundError` for `/kaggle/input/...` | Rules not accepted yet, or the notebook is not attached to the competition — do Step 1, then create the notebook from the competition's Code tab. |
| Out of memory | Reduce `MAX_TRAIN_ROWS` in the config cell and re-run. |
| Validation problems listed | Read the exact messages; the submission writer already enforces format, so this usually means a molecule produced zero candidates — check the log for the failing molecule id. |

## What the agent could NOT verify from the VM (check in the browser)

- The exact wording/location of the "accept rules" button and the
  submissions-per-day limit (Rules tab).
- The exact notebook-upload flow in the current Kaggle UI.
- Whether phone verification is still required for free GPU (it was at the
  time of writing).
- The public leaderboard score (needs a logged-in submission).
