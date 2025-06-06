from omtra.eval.system import SampledSystem
from omtra.eval.reos import REOS
from omtra.eval.ring_systems import RingSystemCounter, ring_counts_to_df
from omtra.data.pharmacophores import get_pharmacophores
from collections import Counter
from rdkit import Chem
from typing import List, Dict, Any, Optional, Tuple
import peppr
from biotite import structure as struc
import functools
from pathlib import Path
from omtra.utils import omtra_root
import yaml
from collections import defaultdict
import numpy as np

allowed_bonds = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {
        0: [2, 3],
        1: [2, 3, 4],
        -1: 2,
    },  # In QM9, N+ seems to be present in the form NH+ and NH2+
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}


# adapted from flowmol metrics.py
# this function taken from MiDi molecular_metrics.py script
def compute_validity(
    sampled_systems: List[SampledSystem], return_counts: bool = False
) -> Dict[str, Any]:
    n_valid = 0
    n_connected = 0
    num_components = []
    frag_fracs = []
    error_messages = defaultdict(int)
    for sys in sampled_systems:
        if sys.get_n_lig_atoms() == 0:
            error_messages['empty'] += 1
            continue
        rdmol = sys.get_rdkit_ligand()
        if rdmol is not None:
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(
                    rdmol, asMols=True, sanitizeFrags=False
                )
                num_components.append(len(mol_frags))
                if len(mol_frags) > 1:
                    error_messages['disconnected'] += 1
                largest_mol = max(
                    mol_frags, default=rdmol, key=lambda m: m.GetNumAtoms()
                )
                largest_mol_n_atoms = largest_mol.GetNumAtoms()
                largest_frag_frac = largest_mol_n_atoms / sys.get_n_lig_atoms()
                frag_fracs.append(largest_frag_frac)
                n_connected += int(len(mol_frags) == 1)
                # Chem.SanitizeMol(largest_mol)
                Chem.SanitizeMol(rdmol)
                # smiles = Chem.MolToSmiles(largest_mol)
                n_valid += 1
                error_messages['no error'] += 1
            except Chem.rdchem.AtomValenceException:
                error_messages['AtomValence'] += 1
                # print("Valence error in GetmolFrags")
            except Chem.rdchem.KekulizeException:
                error_messages['Kekulize'] += 1
                # print("Can't kekulize molecule")
            except Exception as e:
                error_messages['other'] += 1
    
    error_messages['total'] = len(sampled_systems)
    keys = sorted(error_messages.keys())
    error_str_components = [ f'{k}: {error_messages[k]}' for k in keys ]
    error_str = ', '.join(error_str_components)
    print(error_str)

    frac_valid_mols = n_valid / len(sampled_systems)
    avg_frag_frac = sum(frag_fracs) / len(frag_fracs)
    avg_num_components = sum(num_components) / len(num_components)

    if return_counts:
        metrics = {
            "frac_valid_mols": frac_valid_mols,
            "avg_frac_frac": avg_frag_frac,
            "avg_num_components": avg_num_components,
            "n_valid": n_valid,
            "sum_frag_fracs": sum(frag_fracs),
            "len_frag_fracs": len(frag_fracs),
            "sum_num_components": sum(num_components),
            "len_num_components": len(num_components),
        }
    else:
        metrics = {
            "frac_valid_mols": frac_valid_mols,
            "avg_frag_frac": avg_frag_frac,
            "avg_num_components": avg_num_components,
            "frac_connected": n_connected / len(sampled_systems),
        }
    return metrics


def compute_stability(sampled_systems: List[SampledSystem]):
    n_atoms = 0
    n_stable_atoms = 0
    n_stable_molecules = 0
    n_molecules = len(sampled_systems)
    for sys in sampled_systems:
        n_stable_atoms_this_mol, mol_stable, n_fake_atoms = check_stability(sys)
        n_atoms += sys.get_n_lig_atoms() - n_fake_atoms
        n_stable_atoms += n_stable_atoms_this_mol
        n_stable_molecules += int(mol_stable)

    frac_atoms_stable = (
        n_stable_atoms / n_atoms
    )  # the fraction of generated atoms that have valid valencies
    frac_mols_stable_valence = (
        n_stable_molecules / n_molecules
    )  # the fraction of generated molecules whose atoms all have valid valencies

    metrics = {
        "frac_atoms_stable": frac_atoms_stable,
        "frac_mols_stable_valence": frac_mols_stable_valence,
    }
    return metrics


def group_pharm_types_by_position(positions, types):
    grouped = defaultdict(set)
    for pos, t in zip(positions, types):
        key = tuple(np.round(pos, 3))
        grouped[key].add(t)
    return grouped

def compute_pharmacophore_match(sampled_systems: List[SampledSystem], threshold=1.0):
    total_true = total_gen = total_extra = matched = extra_gen = sample_match = 0
    error_counts = defaultdict(int)
    
    for sample in sampled_systems:

        try:
            rdmol = sample.get_rdkit_ligand()
            gen_coords, gen_types, _, _ = get_pharmacophores(rdmol)
            gen_coords = np.array(gen_coords)
            gen_types = np.array(gen_types)
        except Chem.rdchem.AtomValenceException:
            error_counts['get_pharmacophores_AtomValenceException'] += 1
            continue
        except Exception as e:
            error_type = f"get_pharmacophores_{type(e).__name__}"
            error_counts[error_type] += 1
            continue

        true_coords, true_types, _ = sample.extract_pharm_from_graph()
        true_coords = np.array(true_coords)
        true_types = np.array(true_types)

        # to compare positions that have multiple types
        g_gen = group_pharm_types_by_position(gen_coords, gen_types)
        g_true = group_pharm_types_by_position(true_coords, true_types)
        
        true_items = list(g_true.items())
        gen_items = list(g_gen.items())
        
        matched_gen = set()
        matched_true = 0
        
        for i, (true_pos, true_types_set) in enumerate(true_items):
            best_j, best_dist = None, threshold + 1
            
            for j, (gen_pos, gen_types_set) in enumerate(gen_items):
                if j in matched_gen or true_types_set != gen_types_set:
                    continue
                    
                dist = np.linalg.norm(np.array(true_pos) - np.array(gen_pos))
                if dist <= threshold and dist < best_dist:
                    best_j, best_dist = j, dist
            
            if best_j is not None:
                matched_gen.add(best_j)
                matched_true += len(true_types_set)
        
        extra_gen = sum(len(gen_types_set) for j, (_, gen_types_set) in enumerate(gen_items) 
                          if j not in matched_gen)
        
        matched += matched_true
        total_extra += extra_gen
        total_true += len(true_types)
        total_gen += len(gen_types)
        
        if matched_true == len(true_types) and extra_gen == 0:
            sample_match += 1
    
    for error, count in error_counts.items():
        print(f"{error}: {count}")

    metrics =  {
        "frac_pharm_samples_matching": sample_match / len(sampled_systems),
        "frac_true_centers_matched": matched / total_true if total_true else 0,
        "frac_gen_centers_extra": total_extra / total_gen if total_gen else 0,
        "num_true_centers": total_true,
        "num_gen_centers": total_gen
    }

    return metrics

@functools.lru_cache()
def get_valid_valency_table() -> dict:
    valency_file = Path(omtra_root()) / "omtra_pipelines/pharmit_dataset/train_valency_table.yml"
    valency_table = yaml.safe_load(valency_file.read_text())
    return valency_table


def check_stability(sys: SampledSystem) -> Tuple[int, bool, int]:
    if sys.exclude_charges:
        raise ValueError("Charges excluded, but required for stability check.")

    (
        _,
        atom_types,
        atom_charges,
        _,
        _,
        _,
    ) = sys.extract_ligdata_from_graph()
    atom_types = atom_types
    valencies = sys.compute_valencies()
    charges = atom_charges

    valency_table: dict = get_valid_valency_table()

    n_stable_atoms = 0
    n_fake_atoms = 0
    mol_stable = True
    for i, (atom_type, valency, charge) in enumerate(
        zip(atom_types, valencies, charges)
    ):
        if sys.fake_atoms and atom_type == "Sn":
            n_fake_atoms += 1
            continue

        valency = int(valency)
        charge = int(charge)
        charge_to_valid_valencies = valency_table[atom_type]

        if charge not in charge_to_valid_valencies:
            continue

        valid_valencies = charge_to_valid_valencies[charge]
        if valency in valid_valencies:
            n_stable_atoms += 1

    n_real_atoms = len(atom_types) - n_fake_atoms
    if n_stable_atoms == n_real_atoms:
        mol_stable = True
    else:
        mol_stable = False

    return n_stable_atoms, mol_stable, n_fake_atoms


def reos_and_rings(samples: List[SampledSystem], return_raw=False):
    """samples: list of SampledSystem objects."""
    rd_mols = [sample.get_rdkit_ligand() for sample in samples]
    valid_idxs = []
    sanitized_mols = []
    for i, mol in enumerate(rd_mols):
        try:
            Chem.SanitizeMol(mol)
            sanitized_mols.append(mol)
            valid_idxs.append(i)
        except:
            continue
    reos = REOS(active_rules=["Glaxo", "Dundee"])
    ring_system_counter = RingSystemCounter()

    if len(sanitized_mols) != 0:
        reos_flags = reos.mols_to_flag_arr(sanitized_mols)
        ring_counts = ring_system_counter.count_ring_systems(sanitized_mols)
    else:
        reos_flags = None
        ring_counts = None

    if return_raw:
        result = {
            "reos_flag_arr": reos_flags,
            "reos_flag_header": reos.flag_arr_header,
            "smarts_arr": reos.smarts_arr,
            "ring_counts": ring_counts,
            "valid_idxs": valid_idxs,
        }
        return result

    if reos_flags is not None:
        n_flags = reos_flags.sum()
        n_mols = reos_flags.shape[0]
        flag_rate = n_flags / n_mols

        sample_counts, chembl_counts, n_mols = ring_counts
        df_ring = ring_counts_to_df(sample_counts, chembl_counts, n_mols)
        ood_ring_count = df_ring[df_ring["chembl_count"] == 0]["sample_count"].sum()
        ood_rate = ood_ring_count / n_mols
    else:
        flag_rate = -1
        ood_rate = -1

    return dict(flag_rate=flag_rate, ood_rate=ood_rate)


def compute_peppr_metrics_no_ref(sampled_systems: List[SampledSystem]):
    evaluator = peppr.Evaluator(
        [
            peppr.ClashCount(),
            peppr.BondLengthViolations(),
        ]
    )

    for i, sys in enumerate(sampled_systems):
        pose = sys.get_atom_arr(reference=False)
        evaluator.feed(f"sample_{i}", pose, pose)

    metrics = evaluator.summarize_metrics()
    return metrics


def compute_peppr_metrics_ref(sampled_systems: List[SampledSystem]):
    evaluator = peppr.Evaluator(
        [
            peppr.ClashCount(),
            peppr.BondLengthViolations(),
            peppr.LDDTPLIScore(),
            # peppr.DockQScore(),
            peppr.PocketAlignedLigandRMSD(),
            # peppr.GlobalLDDTScore(),
        ]
    )

    for i, sys in enumerate(sampled_systems):
        ref = sys.get_atom_arr(reference=True)
        pose = sys.get_atom_arr(reference=False)
        evaluator.feed(f"sample_{i}", ref, pose)

    metrics = evaluator.summarize_metrics()
    return metrics


def add_task_prefix(metrics: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    """Add the task name as a prefix to the metric names."""
    new_metrics = {}
    for key, value in metrics.items():
        new_key = f"{task_name}/{key}"
        new_metrics[new_key] = value
    return new_metrics
