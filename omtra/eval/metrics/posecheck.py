from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from posecheck import PoseCheck
from scipy.spatial.distance import cdist

from omtra.data.pharmacophores import (
    smarts_patterns,
    matching_types,
    matching_distance,
    interaction_distance, 
    interaction_angle,
    get_smarts_matches,
    _get_receptor_pharmacophores,
    _get_ligand_pharmacophores,
)


def posecheck_all(
    ligs: List[Chem.Mol],
    prot_file: str,
    true_lig: Optional[Chem.Mol] = None,
    true_prot_file: Optional[str] = None,
    include_strain: bool = True,
    include_interaction_recovery: bool = False,
) -> Dict[str, list]:
    """Run all PoseCheck metrics using a single PoseCheck instance.

    Args:
        ligs: List of generated RDKit molecules.
        prot_file: Path to the protein PDB file for generated poses.
        true_lig: Ground truth ligand (required for interaction recovery).
        true_prot_file: Path to the true protein PDB file (required for interaction recovery).
        include_strain: Whether to compute strain energy.
        include_interaction_recovery: Whether to compute interaction recovery.

    Returns:
        Dict mapping metric names to lists of per-ligand values.
        Keys: "clashes", "strain" (if enabled), interaction types,
        "interaction_recovery" (if enabled).
    """
    interaction_types = ["HBAcceptor", "HBDonor", "Hydrophobic", "PiStacking"]

    pc = PoseCheck()
    pc.load_protein_from_pdb(prot_file)
    pc.load_ligands_from_mols(ligs)

    results: Dict[str, list] = {}

    results["clashes"] = pc.calculate_clashes()

    if include_strain:
        results["strain"] = pc.calculate_strain_energy()

    interactions = pc.calculate_interactions()
    n_lig_atoms = [lig.GetNumAtoms() for lig in ligs]

    for i_type in interaction_types:
        cols = [col for col in interactions.columns if col[2] == i_type]
        i_sum = interactions[cols].sum(axis=1)
        results[i_type] = [
            n_interactions / n_atoms
            for (n_interactions, n_atoms) in zip(i_sum, n_lig_atoms)
        ]

    if include_interaction_recovery:
        results["interaction_recovery"] = _interaction_recovery(
            interactions, ligs, prot_file, true_lig, true_prot_file
        )

    return results


def _interaction_recovery(
    gen_interactions: pd.DataFrame,
    ligs: List[Chem.Mol],
    prot_file: str,
    true_lig: Chem.Mol,
    true_prot_file: str,
) -> List[float]:
    """Compute interaction recovery relative to ground truth."""
    fingerprint_interaction_types = [
        "HBAcceptor", "HBDonor", "PiStacking", "XBDonor",
        "CationPi", "PiCation", "Cationic", "Anionic",
    ]

    pc = PoseCheck()
    pc.load_protein_from_pdb(true_prot_file)
    pc.load_ligands_from_mols([true_lig])
    true_interactions = pc.calculate_interactions()

    true_cols = [col for col in true_interactions.columns if col[2] in fingerprint_interaction_types]
    true_interactions_filtered = true_interactions[true_cols]

    gen_cols = [col for col in gen_interactions.columns if col[2] in fingerprint_interaction_types]
    gen_interactions_filtered = gen_interactions[gen_cols]

    recovery = pd.DataFrame(
        False,
        index=gen_interactions_filtered.index,
        columns=true_interactions_filtered.columns,
    )
    common_cols = gen_interactions_filtered.columns.intersection(true_interactions_filtered.columns)
    recovery[common_cols] = gen_interactions_filtered[common_cols]

    return (recovery.sum(axis=1) / recovery.shape[1]).to_list()

def _interaction_recovery_smarts(
    gen_ligs: List[Chem.Mol],
    true_lig: Chem.Mol,
    true_prot_file: str,
) -> List[float]:
    """Compute interaction recovery using SMARTS-based pharmacophore matching."""
    recovery_feature_types = ["Aromatic", "PositiveIon", "NegativeIon", 
                              "HydrogenAcceptor", "HydrogenDonor", "Halogen"]

    #load receptor and grab pharmacophores
    receptor = Chem.MolFromPDBFile(true_prot_file, removeHs=False)
    receptor = Chem.AddHs(receptor, addCoords=True)
    receptor_pharmacophores = _get_receptor_pharmacophores(receptor, recovery_feature_types)

    #grab ligand pharmacophores (already mol)
    true_pharmacophores = _get_ligand_pharmacophores(true_lig, recovery_feature_types)

    #find interactions between ligand and receptor pharmacophores
    true_interactions = _find_interactions(true_pharmacophores, receptor_pharmacophores)

    if not true_interactions:
        return [float("nan")] * len(gen_ligs)

    #iterate through generated ligands and compute recovery for each
    recovery: List[float] = []
    for lig in gen_ligs:
        lig_pharmacophores = _get_ligand_pharmacophores(lig, recovery_feature_types)
        lig_interactions = _find_interactions(lig_pharmacophores, receptor_pharmacophores)
        recovered = len(true_interactions & lig_interactions) / len(true_interactions)
        recovery.append(recovered)

    return recovery

def _find_interactions(
    ligand_pharmacophores: Dict[str, List[np.ndarray]],
    receptor_pharmacophores: Dict[str, List[Tuple[np.ndarray, tuple]]],
) -> set:
    """Pair ligand pharmacophore features against complementary receptor
    pharmacophore features (same feature-type pairing + cutoffs as
    check_interaction) and record which residue each contact is on.

    Returns: set of (ligand_feature_type, residue_id) contacts.
    """
    found = set()

    for lig_feature, lig_positions in ligand_pharmacophores.items():
        if not lig_positions:
            continue
        lig_positions_arr = np.array(lig_positions)

        paired_features = matching_types[lig_feature]
        cutoffs = matching_distance[lig_feature]

        for rec_feature, cutoff in zip(paired_features, cutoffs):
            rec_matches = receptor_pharmacophores.get(rec_feature, [])
            if not rec_matches:
                continue

            rec_positions_arr = np.array([pos for pos, _ in rec_matches])
            rec_residue_ids = [res_id for _, res_id in rec_matches]

            distances = cdist(lig_positions_arr, rec_positions_arr)
            for i in range(distances.shape[0]):
                hit_idxs = np.where(distances[i] <= cutoff)[0]
                for r_idx in hit_idxs:
                    found.add((lig_feature, rec_residue_ids[r_idx]))

    return found

def posecheck_clashes(ligs: List[Chem.Mol], prot_file: str) -> List[float]:
    """Compute steric clashes for each ligand against the protein."""
    pc = PoseCheck()
    pc.load_protein_from_pdb(prot_file)
    pc.load_ligands_from_mols(ligs)
    return pc.calculate_clashes()


def posecheck_strain(ligs: List[Chem.Mol], prot_file: str) -> List[float]:
    """Compute strain energy for each ligand."""
    pc = PoseCheck()
    pc.load_protein_from_pdb(prot_file)
    pc.load_ligands_from_mols(ligs)
    return pc.calculate_strain_energy()


def posecheck_interactions(
    ligs: List[Chem.Mol],
    prot_file: str,
    interaction_types: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    """Compute per-atom-normalized interaction counts for each ligand.

    Args:
        ligs: List of RDKit molecules.
        prot_file: Path to the protein PDB file.
        interaction_types: Which interaction types to compute.
            Defaults to ["HBAcceptor", "HBDonor", "Hydrophobic", "PiStacking"].

    Returns:
        Dict mapping interaction type names to lists of per-ligand values.
    """
    if interaction_types is None:
        interaction_types = ["HBAcceptor", "HBDonor", "Hydrophobic", "PiStacking"]

    pc = PoseCheck()
    pc.load_protein_from_pdb(prot_file)
    pc.load_ligands_from_mols(ligs)

    interactions = pc.calculate_interactions()
    n_lig_atoms = [lig.GetNumAtoms() for lig in ligs]

    results: Dict[str, List[float]] = {}
    for i_type in interaction_types:
        cols = [col for col in interactions.columns if col[2] == i_type]
        i_sum = interactions[cols].sum(axis=1)
        results[i_type] = [
            n_interactions / n_atoms
            for (n_interactions, n_atoms) in zip(i_sum, n_lig_atoms)
        ]

    return results


def posecheck_interaction_recovery(
    ligs: List[Chem.Mol],
    prot_file: str,
    true_lig: Chem.Mol,
    true_prot_file: str,
) -> List[float]:
    """Compute interaction recovery of generated ligands relative to a ground truth ligand."""
    pc = PoseCheck()
    pc.load_protein_from_pdb(prot_file)
    pc.load_ligands_from_mols(ligs)
    gen_interactions = pc.calculate_interactions()

    return _interaction_recovery(gen_interactions, ligs, prot_file, true_lig, true_prot_file)
