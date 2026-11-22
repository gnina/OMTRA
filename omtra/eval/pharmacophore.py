from typing import List, Dict
from omtra.eval.system import SampledSystem
from collections import defaultdict
import numpy as np
from rdkit import Chem
from omtra.data.pharmacophores import get_pharmacophores
from scipy.spatial.distance import cdist

def compute_pharmacophore_match(sampled_systems: List[SampledSystem], threshold=1.0, reference="clean"):

    if reference not in {'clean', 'input'}:
        raise ValueError("reference must be 'clean' or 'input'")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('threshold must be positive and finite')
    pharm_counts = defaultdict(int)
    for sample in sampled_systems:
        true_pharm = sample.get_pharmacophore_from_graph(kind='gt' if reference == 'clean' else 'input')
        true_coords = np.asarray(true_pharm['coords']).reshape(-1, 3)
        true_types = np.asarray(true_pharm['types_idx'])
        # Empty conditioning is not a successful pharmacophore-conditioned example.
        if len(true_coords) == 0:
            continue
        pharm_counts['n_true_pharms'] += len(true_coords)
        try:
            gen_coords, gen_types, _, _ = get_pharmacophores(sample.get_rdkit_ligand())
            gen_coords = np.asarray(gen_coords).reshape(-1, 3)
            gen_types = np.asarray(gen_types)
            if len(gen_coords) == 0:
                continue
            d = cdist(true_coords, gen_coords)
            matching = (d < threshold) & (true_types[:, None] == gen_types[None, :])
            matched = matching.any(axis=1)
            pharm_counts['n_true_matched'] += int(matched.sum())
            pharm_counts['n_pharms_matched_perfect'] += int(matched.all())
        except Exception:
            # Invalid/failed molecules remain in both metric denominators.
            continue
    return {
        'frac_pharm_samples_matching': pharm_counts['n_pharms_matched_perfect'] / max(len(sampled_systems), 1),
        'frac_true_centers_matched': pharm_counts['n_true_matched'] / max(pharm_counts['n_true_pharms'], 1),
    }