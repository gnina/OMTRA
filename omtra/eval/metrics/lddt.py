"""lDDT-PLI metric for docking evaluation.

Computes protein-ligand interaction lDDT: the fraction of ground-truth
protein-ligand contact distances preserved in the predicted pose.
Standard cutoff 6 Å; thresholds 0.5, 1.0, 2.0, 4.0 Å.
"""

from typing import List, Optional
import numpy as np
from rdkit import Chem
import biotite.structure.io.pdb as pdb_io


_CUTOFF = 6.0
_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def _protein_coords(prot_file: str) -> np.ndarray:
    pdb = pdb_io.PDBFile.read(str(prot_file))
    arr = pdb.get_structure(model=1)
    heavy = arr[(~arr.hetero) & (arr.element != "H")]
    return heavy.coord


def _lig_heavy_coords(mol: Chem.Mol) -> np.ndarray:
    positions = mol.GetConformer().GetPositions()
    heavy_idx = [i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() != 1]
    return positions[heavy_idx]


def _lddt_pli(
    gen_prot: np.ndarray,
    gen_lig: np.ndarray,
    true_prot: np.ndarray,
    true_lig: np.ndarray,
) -> float:
    diff = true_prot[:, None, :] - true_lig[None, :, :]
    d_true = np.sqrt((diff ** 2).sum(axis=-1))  # (N_prot, N_lig)

    pi, li = np.where(d_true <= _CUTOFF)
    if len(pi) == 0:
        return float("nan")

    d_ref = d_true[pi, li]
    d_pred = np.sqrt(((gen_prot[pi] - gen_lig[li]) ** 2).sum(axis=-1))
    delta = np.abs(d_pred - d_ref)

    return float(np.mean([np.mean(delta <= t) for t in _THRESHOLDS]))


def lddt_pli_scores(
    gen_ligs: List[Chem.Mol],
    true_lig: Chem.Mol,
    prot_file: str,
    gen_prot_file: Optional[str] = None,
) -> List[float]:
    """Compute lDDT-PLI for each generated ligand.

    When gen_prot_file is None the true protein is used for both reference
    and prediction (rigid docking → lddt_lig).  When gen_prot_file is
    provided the generated protein coordinates are used for the prediction
    (flexible docking → lddt).
    """
    true_prot = _protein_coords(str(prot_file))
    true_lig_coords = _lig_heavy_coords(true_lig)
    gen_prot = _protein_coords(str(gen_prot_file)) if gen_prot_file is not None else true_prot

    scores = []
    for mol in gen_ligs:
        try:
            gen_lig_coords = _lig_heavy_coords(mol)
        except Exception:
            scores.append(float("nan"))
            continue
        scores.append(_lddt_pli(gen_prot, gen_lig_coords, true_prot, true_lig_coords))
    return scores
