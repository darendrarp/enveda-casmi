"""Tests for src/submission.py — CSV format, dedup, validation."""

import csv

from src.submission import (normalize_candidates, read_submission,
                            validate_submission, write_submission)

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
R_LACTIC = "C[C@H](O)C(=O)O"
S_LACTIC = "C[C@@H](O)C(=O)O"


def test_normalize_candidates_dedupes_and_caps():
    cands = [R_LACTIC, S_LACTIC, "invalid(((", CAFFEINE]
    out = normalize_candidates(cands)
    # Stereoisomers collapse to one entry; invalid dropped.
    assert len(out) == 2
    assert CAFFEINE in out
    from src.chem import inchikey14

    assert len({inchikey14(s) for s in out}) == 2
    many = [f"C{'C' * i}O" for i in range(1, 40)]
    assert len(normalize_candidates(many)) == 25


def test_write_read_round_trip(tmp_path):
    path = tmp_path / "sub.csv"
    rows = [("mol_a", [CAFFEINE, "CCO"]), ("mol_b", ["c1ccccc1"])]
    assert write_submission(rows, path) == 2
    with open(path) as f:
        header = f.readline().strip()
    assert header == "molecule_id,smiles"
    subs = read_submission(path)
    assert subs["mol_a"] == [CAFFEINE, "CCO"]
    assert subs["mol_b"] == ["c1ccccc1"]
    assert validate_submission(path, ["mol_a", "mol_b"]) == []


def test_validate_catches_problems(tmp_path):
    path = tmp_path / "bad.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["molecule_id", "smiles"])
        w.writerow(["mol_a", ";".join(["CCO"] * 26)])  # too many + dup keys
        w.writerow(["mol_b", "not_a_smiles"])  # invalid
        w.writerow(["mol_c", ""])  # empty
    problems = validate_submission(path, ["mol_a", "mol_b", "mol_c", "mol_d"])
    joined = "\n".join(problems)
    assert "missing 1 molecule" in joined
    assert "26 candidates" in joined
    assert "invalid SMILES" in joined
    assert "no candidates" in joined
    assert "duplicate InChIKey14" in joined
