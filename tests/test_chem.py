"""Tests for src/chem.py — SMILES handling, InChIKey14, fingerprints."""

from src.chem import (canonical_smiles, inchikey14, is_valid_smiles,
                      morgan_fp, neutral_mass)

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
CAFFEINE_IK14 = "RYYVLZVUVIJVGH"  # verified public value (MassBank EA030309)
ETHANOL = "CCO"
ETHANOL_IK14 = "LFQSCWFLJHTTHZ"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_IK14 = "BSYNRYMUTXBXSQ"
R_LACTIC = "C[C@H](O)C(=O)O"
S_LACTIC = "C[C@@H](O)C(=O)O"


def test_is_valid_smiles():
    assert is_valid_smiles(CAFFEINE)
    assert is_valid_smiles("c1ccccc1")
    assert not is_valid_smiles("not a smiles")
    assert not is_valid_smiles("C(((((")
    assert not is_valid_smiles("")
    assert not is_valid_smiles(None)


def test_canonical_smiles():
    assert canonical_smiles("C1=CC=CC=C1") == "c1ccccc1"
    assert canonical_smiles("CCO") == "CCO"
    assert canonical_smiles("garbage(((") is None


def test_inchikey14_known_values():
    assert inchikey14(CAFFEINE) == CAFFEINE_IK14
    assert inchikey14(ETHANOL) == ETHANOL_IK14
    assert inchikey14(ASPIRIN) == ASPIRIN_IK14
    assert inchikey14("invalid(((") is None


def test_inchikey14_ignores_stereochemistry():
    # Stereoisomers share the connectivity (first) block.
    assert inchikey14(R_LACTIC) == inchikey14(S_LACTIC)
    assert inchikey14(R_LACTIC) == "JVTAAEKCZFNVCJ"


def test_morgan_fp_shape_and_content():
    fp = morgan_fp(CAFFEINE)
    assert fp.shape == (2048,)
    assert fp.dtype.name == "float32"
    assert set(fp.tolist()) <= {0.0, 1.0}
    assert fp.sum() > 10  # caffeine has plenty of on-bits
    assert morgan_fp("invalid(((") is None


def test_neutral_mass_protonated():
    # Caffeine neutral exact mass 194.08038; [M+H]+ = 195.08765
    m = neutral_mass(195.0877, "[M+H]+")
    assert abs(m - 194.0804) < 0.001
    m_na = neutral_mass(217.0696, "[M+Na]+")
    assert abs(m_na - 194.0804) < 0.001
    m_neg = neutral_mass(193.0731, "[M-H]-")
    assert abs(m_neg - 194.0804) < 0.001
    # Unknown adduct falls back to [M+H]+ assumption.
    assert neutral_mass(195.0877, "[M+???]+") is not None
