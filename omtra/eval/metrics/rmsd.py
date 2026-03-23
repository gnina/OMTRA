from rdkit import Chem
from posebusters.modules.rmsd import check_rmsd


def ligand_rmsd(gen_lig: Chem.Mol, true_lig: Chem.Mol) -> float:
    """Compute RMSD between a generated ligand and a ground truth ligand.

    Returns RMSD in Angstroms, or -1.0 on failure.
    """
    res = check_rmsd(mol_pred=gen_lig, mol_true=true_lig)
    res = res.get("results", {})
    return res.get("rmsd", -1.0)
