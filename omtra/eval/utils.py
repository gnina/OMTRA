from omtra.eval.system import SampledSystem
from omtra.eval.reos import REOS
from omtra.eval.ring_systems import RingSystemCounter, ring_counts_to_df
from collections import Counter
from rdkit import Chem
from typing import List, Dict, Any, Optional, Tuple
import peppr
from biotite import structure as struc
from biotite.interface import rdkit as bt_rdkit
import pandas as pd
import numpy as np
import functools
from pathlib import Path
from omtra.utils import omtra_root
import yaml
from collections import defaultdict
from posebusters import PoseBusters
import numpy as np
from omtra.tasks.register import task_name_to_class
import dgl


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

# redock is for ligands docked into cognate receptor
def bust_redock(sampled_systems: List[SampledSystem]):
    buster = PoseBusters(config="redock")
    metrics_to_log = ["mol_pred_energy", "aromatic_ring_maximum_distance_from_plane", "non-aromatic_ring_maximum_distance_from_plane", "double_bond_maximum_distance_from_plane", "volume_overlap_protein", "rmsd"]
    collected_values = {metric: [] for metric in metrics_to_log}
    for i, sys in enumerate(sampled_systems):
        prot_arr = sys.get_protein_array()
        prot_mol = bt_rdkit.to_mol(prot_arr)
        lig_mol = sys.get_rdkit_ligand()
        ref_mol = sys.get_rdkit_ref_ligand()
        try:
            res = buster.bust([lig_mol], ref_mol, prot_mol, full_report=True)
        except:
            continue
        for metric in metrics_to_log:
            value = res.iloc[0][metric] if metric in res.columns else None
            if not pd.isna(value):
                collected_values[metric].append(value)
    metrics = {
        metric: np.mean(values) if values else -1
        for metric, values in collected_values.items()
    }
    return metrics

# dock is for de novo ligand or docking into non-cognate receptor       
def bust_dock(sampled_systems: List[SampledSystem]):
    buster = PoseBusters(config="dock")
    metrics_to_log = ["mol_pred_energy", "aromatic_ring_maximum_distance_from_plane", "non-aromatic_ring_maximum_distance_from_plane", "double_bond_maximum_distance_from_plane", "volume_overlap_protein"]
    collected_values = {metric: [] for metric in metrics_to_log}
    for i, sys in enumerate(sampled_systems):
        prot_arr = sys.get_protein_array()
        prot_mol = bt_rdkit.to_mol(prot_arr)
        lig_mol = sys.get_rdkit_ligand()
        try:
            res = buster.bust([lig_mol], None, prot_mol, full_report=True)
        except:
            continue
        for metric in metrics_to_log:
            value = res.iloc[0][metric] if metric in res.columns else None
            if not pd.isna(value):
                collected_values[metric].append(value)
    metrics = {
        metric: np.mean(values) if values else -1
        for metric, values in collected_values.items()
    }
    return metrics
        

def bust_mol(sampled_systems: List[SampledSystem]):
    buster = PoseBusters(config="mol")
    lig_mols = [sys.get_rdkit_ligand() for sys in sampled_systems]
    res = buster.bust(lig_mols, None, None, full_report=True)
    metrics_to_log = ["mol_pred_energy", "aromatic_ring_maximum_distance_from_plane", "non-aromatic_ring_maximum_distance_from_plane", "double_bond_maximum_distance_from_plane"]
    
    metrics = {
        metric: res[metric].dropna().mean() if metric in res.columns else -1
        for metric in metrics_to_log
    }

    return metrics


def add_task_prefix(metrics: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    """Add the task name as a prefix to the metric names."""
    new_metrics = {}
    for key, value in metrics.items():
        new_key = f"{task_name}/{key}"
        new_metrics[new_key] = value
    return new_metrics

