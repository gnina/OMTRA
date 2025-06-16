from typing import List, Dict
from omtra.eval.system import SampledSystem
from collections import defaultdict
import numpy as np
from rdkit import Chem
from omtra.data.pharmacophores import get_pharmacophores
from scipy.spatial.distance import cdist

def compute_pharmacophore_match(sampled_systems: List[SampledSystem], threshold=1.0):

    pharm_counts = defaultdict(int)
    error_counts = defaultdict(int)
    for sample in sampled_systems:

        try:
            rdmol = sample.get_rdkit_ligand()
            gen_coords, gen_types, _, _ = get_pharmacophores(rdmol)
            gen_coords = np.array(gen_coords) # has shape (n_gen_pharms, 3)
            gen_types = np.array(gen_types) # has shape (n_gen_pharms)
        except Chem.rdchem.AtomValenceException:
            error_counts['get_pharmacophores_AtomValenceException'] += 1
            continue
        except Exception as e:
            error_type = f"get_pharmacophores_{type(e).__name__}"
            error_counts[error_type] += 1
            continue

        # get gt pharmacophore
        true_pharm = sample.get_pharmacophore_from_graph(kind='gt')
        true_coords = true_pharm['coords']
        true_types = true_pharm['types_idx']

        # convert to numpy arrays
        true_coords = np.array(true_coords) # has shape (n_true_pharms, 3)
        true_types = np.array(true_types) # has shape (n_true_pharms)

        d = cdist(true_coords, gen_coords)
        same_type_mask = true_types[:, None] == gen_types[None, :]

        matching_pharms = (d < threshold) & same_type_mask

        n_true_pharms = true_coords.shape[0]
        n_gen_pharms = gen_coords.shape[0]
        all_true_matched = int(matching_pharms.any(axis=1).all())
        n_extra = np.logical_not(matching_pharms.any(axis=0)).sum()


        pharm_counts['n_gen_pharms'] += n_gen_pharms
        pharm_counts['n_true_pharms'] += n_true_pharms
        pharm_counts['n_true_matched'] += matching_pharms.any(axis=1).sum()
        pharm_counts['n_gen_unmatched'] += n_extra
        pharm_counts['n_pharms_matched_perfect'] += int(all_true_matched and n_extra == 0)

    print(f', '.join([f"{k}={v}" for k, v in error_counts.items()]))
    n_errors = sum(list(error_counts.values()))
    n_systems_without_error = len(sampled_systems) - n_errors
    if n_systems_without_error == 0:
        n_systems_without_error = 1

    if pharm_counts['n_true_pharms'] == 0:
        pharm_counts['n_true_pharms'] = 1
    if pharm_counts['n_gen_pharms'] == 0:
        pharm_counts['n_gen_pharms'] = 1

    metrics =  {
        "frac_pharm_samples_matching": pharm_counts['n_pharms_matched_perfect']/len(sampled_systems),
        "frac_true_centers_matched": pharm_counts['n_true_matched']/pharm_counts['n_true_pharms'],
        "frac_gen_centers_extra": pharm_counts['n_gen_unmatched']/pharm_counts['n_gen_pharms'],
    }
    return metrics