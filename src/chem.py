"""Chemistry helpers built on RDKit.

All structure identity in this project goes through the InChIKey first block
(InChIKey14): the competition scores a candidate as correct when its
tautomer-canonical InChIKey14 matches the truth, so stereochemistry and
tautomers are not penalised. RDKit's standard InChIKey already ignores
stereochemistry in its first block; we additionally canonicalise tautomers
with MolStandardize when available so the local metric tracks the official
one as closely as possible.
"""

from __future__ import annotations

import numpy as np

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize

    _TAUTOMER_ENUM = rdMolStandardize.TautomerEnumerator()
except Exception:  # pragma: no cover - very old RDKit
    _TAUTOMER_ENUM = None

RDLogger.DisableLog("rdApp.*")

# Proton mass and common adduct mass shifts (Da). Used to convert a measured
# precursor m/z into a neutral-mass estimate for candidate filtering.
PROTON_MASS = 1.00727646688
ADDUCT_MASS = {
    "[M+H]+": PROTON_MASS,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823,
    "[M+H-H2O]+": PROTON_MASS - 18.010565,
    "[M-H2+H]+": -1.008374,  # in-source dehydrogenation, seen in Enveda-180
    "[M+2H]2+": PROTON_MASS,  # charge 2 handled by caller via /2
    "[M-H]-": -PROTON_MASS,
    "[M+Cl]-": 34.969402,
    "[M+FA-H]-": 44.998201,  # formate adduct [M+HCOO]-
    "[M+CH3COO]-": 59.013305,  # acetate adduct
}
ADDUCT_CHARGE = {"[M+2H]2+": 2}


def mol_from_smiles(smiles: str):
    """Parse a SMILES string, returning an RDKit Mol or None if invalid."""
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        return Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    return mol_from_smiles(smiles) is not None


def canonical_smiles(smiles: str) -> str | None:
    """Return the RDKit-canonical SMILES, or None if the input is invalid."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _tautomer_canonical_mol(mol):
    if _TAUTOMER_ENUM is None:
        return mol
    try:
        return _TAUTOMER_ENUM.Canonicalize(mol)
    except Exception:
        return mol


def inchikey14(smiles: str, tautomer_canonical: bool = True) -> str | None:
    """First block (14 chars) of the standard InChIKey, or None if invalid.

    The first block encodes connectivity only, so stereoisomers share it.
    With ``tautomer_canonical=True`` (default) tautomers are canonicalised
    first, matching the competition's tautomer-canonical matching rule.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        if tautomer_canonical:
            mol = _tautomer_canonical_mol(mol)
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    if not key or len(key) < 14:
        return None
    return key[:14]


def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray | None:
    """Morgan (ECFP) fingerprint as a float32 0/1 array, or None if invalid."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


def morgan_fp_on_bits(smiles: str, n_bits: int = 2048, radius: int = 2) -> list[int] | None:
    """On-bit indices of the Morgan fingerprint (memory-efficient form)."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp.GetOnBits())


def neutral_mass(precursor_mz: float, adduct: str | None) -> float | None:
    """Estimate the neutral monoisotopic mass from precursor m/z + adduct."""
    if precursor_mz is None or not np.isfinite(precursor_mz):
        return None
    if adduct is None or adduct not in ADDUCT_MASS:
        # Default assumption for positive mode data.
        adduct = "[M+H]+"
    charge = ADDUCT_CHARGE.get(adduct, 1)
    return precursor_mz * charge - ADDUCT_MASS[adduct] * charge
