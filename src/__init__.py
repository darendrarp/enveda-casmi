"""Enveda CASMI 2026 — spectra-to-structure pipeline.

Modules:
- chem: SMILES validation / canonicalisation / InChIKey14 / fingerprints.
- spectra: peak cleaning and binned spectrum vectors.
- metrics: MRR@25 (InChIKey14 matching).
- submission: submission.csv writer + validator.
- io_kaggle: loaders for the competition parquet files.
- io_enveda: streaming loader for the public Enveda-180 dataset (Zenodo).
- splits: structure-disjoint splitting by InChIKey14.
- retrieval: spectral library matching (binned cosine + neutral loss).
- fingerprint: FPNet — peak-set encoder predicting Morgan fingerprints.
- seq2seq: transformer spectra -> SMILES (de novo generation).
- pipeline: hybrid ranking combining retrieval + fingerprint + seq2seq.
"""

__version__ = "0.1.0"
