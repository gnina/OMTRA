"""Ligand strain energy (UFF), PoseCheck's definition with two defect fixes.

Definition (unchanged from PoseCheck 1.3.1 ``calculate_strain_energy``): strain = E_UFF(pose after a locally constrained relaxation, max displacement 0.1 A)
minus the lowest UFF energy over ``num_confs`` embedded and minimised conformers plus one free relaxation of the pose itself. Hydrogens are added with
``Chem.AddHs(addCoords=True)`` because generated ligands are stored without explicit hydrogens.

Fix 1 (spurious million-kcal/mol values): RDKit assigns an implicit hydrogen to a heavy atom that already has four or more neighbours when its valence is not
saturated. For phosphorus this happens with a phosphate drawn with four single bonds (allowed valence 5). The extra hydrogen cannot be placed on a tetrahedral
centre and ``AddHs(addCoords=True)`` returns a position tens of angstroms away (observed: a 96 A P-H bond), which UFF scores as ~1e6 kcal/mol. PoseBusters does not see this
because it checks the molecule without the added hydrogen. Here, every atom with degree >= 4 and implicit hydrogens gets ``NoImplicit`` so no hydrogen is added to it. Any
remaining added hydrogen that is not bonded within ``MAX_H_BOND`` A is reported as ``bad_hydrogen_placement`` and the value is NaN, never a silent huge number.

Fix 2 (reproducibility): conformer embeddings are seeded (PoseCheck's are not, which moves a single value by ~10 kcal/mol between runs).

This module does not need the protein, so a receptor that PoseCheck's interaction step cannot parse (e.g. Mn, Fe, Pr ions) no longer removes strain.
"""
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers

DEFAULT_SEED = 0xF00D
MAX_H_BOND = 1.8          # A; a bond to an added hydrogen longer than this is a failed placement
STATUS_OK = "ok"
STATUS_BAD_H = "bad_hydrogen_placement"
STATUS_FAILED = "failed"


def prepare_mol(mol: Chem.Mol) -> Chem.Mol:
    """Copy of ``mol`` in which no hydrogen will be added to a heavy atom that already has four or more neighbours."""
    m = Chem.Mol(mol)
    m.UpdatePropertyCache(strict=False)
    for a in m.GetAtoms():
        if a.GetAtomicNum() > 1 and a.GetDegree() >= 4 and a.GetNumImplicitHs() > 0:
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
    m.UpdatePropertyCache(strict=False)
    return m


def _uff_energy(mol_h: Chem.Mol) -> float:
    try:
        return float(rdForceFieldHelpers.UFFGetMoleculeForceField(mol_h, confId=-1).CalcEnergy())
    except Exception:
        return float("nan")


def _bad_hydrogen_placement(mol_h: Chem.Mol) -> bool:
    conf = mol_h.GetConformer()
    for b in mol_h.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            p, q = conf.GetAtomPosition(b.GetBeginAtomIdx()), conf.GetAtomPosition(b.GetEndAtomIdx())
            if p.Distance(q) > MAX_H_BOND:
                return True
    return False


def _relax_constrained(mol_h: Chem.Mol, max_displ: float) -> Optional[Chem.Mol]:
    m = deepcopy(mol_h)
    try:
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(m, confId=-1)
        for i in range(m.GetNumAtoms()):
            ff.UFFAddPositionConstraint(i, maxDispl=max_displ, forceConstant=1.0e5)
        ff.Minimize()
        return m
    except Exception:
        return None


def _relax_global(mol: Chem.Mol, seed: int) -> Optional[Chem.Mol]:
    """Embed (seeded) and UFF-minimise; a failed embedding falls back to minimising the pose, as PoseCheck does."""
    m = Chem.AddHs(deepcopy(mol), addCoords=True)
    AllChem.EmbedMolecule(m, randomSeed=int(seed))
    try:
        AllChem.UFFOptimizeMolecule(m)
    except Exception:
        return None
    return m


def _relax_on_pose(mol: Chem.Mol) -> Optional[Chem.Mol]:
    m = Chem.AddHs(deepcopy(mol), addCoords=True)
    try:
        AllChem.UFFOptimizeMolecule(m)
    except Exception:
        return None
    return m


def strain_energy_checked(mol: Chem.Mol, max_displ: float = 0.1, num_confs: int = 50, seed: int = DEFAULT_SEED) -> Dict[str, object]:
    """-> {'strain': float (NaN on failure), 'status': 'ok' | 'bad_hydrogen_placement' | 'failed'}."""
    if mol is None:
        return dict(strain=float("nan"), status=STATUS_FAILED)
    try:
        m = prepare_mol(mol)
        mh = Chem.AddHs(m, addCoords=True)
        if _bad_hydrogen_placement(mh):
            return dict(strain=float("nan"), status=STATUS_BAD_H)
        local = _relax_constrained(mh, max_displ)
        if local is None:
            return dict(strain=float("nan"), status=STATUS_FAILED)
        local_energy = _uff_energy(local)
        glob = [_relax_global(m, seed + i) for i in range(num_confs)] + [_relax_on_pose(m)]
        energies = [_uff_energy(g) for g in glob if g is not None]
        energies = [e for e in energies if np.isfinite(e)]
        if not energies or not np.isfinite(local_energy):
            return dict(strain=float("nan"), status=STATUS_FAILED)
        return dict(strain=float(local_energy - min(energies)), status=STATUS_OK)
    except Exception:
        return dict(strain=float("nan"), status=STATUS_FAILED)


def strain_energies(mols: List[Chem.Mol], **kw) -> List[float]:
    """Per-molecule strain (NaN where it could not be computed)."""
    return [strain_energy_checked(m, **kw)["strain"] for m in mols]
