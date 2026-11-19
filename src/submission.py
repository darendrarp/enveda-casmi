"""Submission file writer + validator.

Official format (from the competition's sample_submission.csv and data
description): a CSV with header ``molecule_id,smiles`` and one row per
molecule. The ``smiles`` field holds up to 25 candidate structures, ranked
best-guess first, separated by semicolons:

    molecule_id,smiles
    abc123,CCO;CCN;CCC

Candidates are canonicalised with RDKit, de-duplicated by InChIKey14 (first
occurrence wins), and capped at 25.
"""

from __future__ import annotations

import csv

from .chem import canonical_smiles, inchikey14

MAX_CANDIDATES = 25


def normalize_candidates(candidates, k: int = MAX_CANDIDATES):
    """Canonicalise, drop invalid, dedupe by InChIKey14, cap at k."""
    seen: set[str] = set()
    out: list[str] = []
    for cand in candidates:
        smi = canonical_smiles(cand)
        if smi is None:
            continue
        key = inchikey14(smi)
        dedup_key = key if key is not None else smi
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        out.append(smi)
        if len(out) >= k:
            break
    return out


def write_submission(rows, path, k: int = MAX_CANDIDATES):
    """Write ``rows`` (iterable of ``(molecule_id, [candidate SMILES])``)."""
    n_written = 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["molecule_id", "smiles"])
        for molecule_id, candidates in rows:
            cands = normalize_candidates(candidates, k=k)
            writer.writerow([molecule_id, ";".join(cands)])
            n_written += 1
    return n_written


def read_submission(path):
    """Read a submission file -> dict molecule_id -> [candidate SMILES]."""
    out: dict[str, list[str]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_field = row.get("smiles") or ""
            cands = [c for c in smiles_field.split(";") if c]
            out[row["molecule_id"]] = cands
    return out


def validate_submission(path, expected_molecule_ids=None):
    """Validate a submission file. Returns a list of problems (empty = valid).

    Checks: header, one row per expected molecule (if given), no duplicates,
    <= 25 candidates per molecule, >= 1 candidate, all candidates valid
    SMILES, no duplicate InChIKey14 within a molecule's list.
    """
    problems: list[str] = []
    try:
        subs = read_submission(path)
    except Exception as e:  # noqa: BLE001
        return [f"could not read submission: {e}"]
    if expected_molecule_ids is not None:
        expected = set(expected_molecule_ids)
        missing = expected - set(subs)
        extra = set(subs) - expected
        if missing:
            problems.append(f"missing {len(missing)} molecule(s), e.g. {sorted(missing)[:3]}")
        if extra:
            problems.append(f"unexpected {len(extra)} molecule(s), e.g. {sorted(extra)[:3]}")
    for mol_id, cands in subs.items():
        if len(cands) == 0:
            problems.append(f"{mol_id}: no candidates")
        if len(cands) > MAX_CANDIDATES:
            problems.append(f"{mol_id}: {len(cands)} candidates > {MAX_CANDIDATES}")
        seen: set[str] = set()
        for cand in cands:
            if canonical_smiles(cand) is None:
                problems.append(f"{mol_id}: invalid SMILES {cand!r}")
                continue
            key = inchikey14(cand)
            if key is not None:
                if key in seen:
                    problems.append(f"{mol_id}: duplicate InChIKey14 {key}")
                seen.add(key)
    return problems
