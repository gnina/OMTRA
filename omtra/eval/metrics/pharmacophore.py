from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist

from omtra.data.pharmacophores import get_pharmacophores


def pharmacophore_match(
    gen_ligs: List[Chem.Mol],
    true_pharm_coords: np.ndarray,
    true_pharm_types: np.ndarray,
    threshold: float = 1.0,
) -> Dict[str, list]:
    """Compute pharmacophore matching between generated ligands and ground truth pharmacophores.

    Args:
        gen_ligs: List of generated RDKit molecules.
        true_pharm_coords: (n_pharm, 3) array of pharmacophore positions.
        true_pharm_types: (n_pharm,) array of integer pharmacophore type indices.
        threshold: Distance threshold in Angstroms for matching.

    Returns:
        Dict with keys "perfect_pharm_match" and "frac_true_pharms_matched",
        each a list of length len(gen_ligs).
    """
    results: Dict[str, list] = {
        "perfect_pharm_match": [],
        "frac_true_pharms_matched": [],
    }

    true_coords = np.asarray(true_pharm_coords)
    true_types = np.asarray(true_pharm_types)

    for gen_lig in gen_ligs:
        try:
            gen_coords, gen_types, _, _ = get_pharmacophores(gen_lig)
            gen_coords = np.array(gen_coords)
            gen_types = np.array(gen_types)
        except Exception as e:
            print(f"Failed to get pharmacophores for generated ligand: {e}")
            results["perfect_pharm_match"].append(None)
            results["frac_true_pharms_matched"].append(None)
            continue

        d = cdist(true_coords, gen_coords)
        same_type_mask = true_types[:, None] == gen_types[None, :]
        matching_pharms = (d < threshold) & same_type_mask

        n_true_pharms = true_coords.shape[0]
        all_true_matched = matching_pharms.any(axis=1).all()

        if n_true_pharms == 0:
            n_true_pharms = 1

        results["perfect_pharm_match"].append(all_true_matched)
        results["frac_true_pharms_matched"].append(
            matching_pharms.any(axis=1).sum() / n_true_pharms
        )

    return results


def pharmacophore_match_from_dict(
    gen_ligs: List[Chem.Mol],
    true_pharm: Dict[str, np.ndarray],
    threshold: float = 1.0,
) -> Dict[str, list]:
    """Convenience wrapper accepting the system_pairs pharmacophore dict format.

    Args:
        gen_ligs: List of generated RDKit molecules.
        true_pharm: Dict with keys "coords" and "types_idx".
        threshold: Distance threshold in Angstroms for matching.
    """
    return pharmacophore_match(
        gen_ligs=gen_ligs,
        true_pharm_coords=true_pharm["coords"],
        true_pharm_types=true_pharm["types_idx"],
        threshold=threshold,
    )
