from omtra.eval.register import register_eval
from omtra.eval.utils import (
    compute_validity,
    compute_stability,
    reos_and_rings,
    compute_peppr_metrics_no_ref,
    compute_peppr_metrics_ref,
    bust_dock,
    bust_mol,
    bust_redock,
)
import posebusters as pb
from posebusters.modules.distance_geometry import check_geometry
from posebusters.modules.flatness import check_flatness
from posebusters.modules.intermolecular_distance import check_intermolecular_distance
from posebusters.modules.rmsd import check_rmsd
from posebusters.modules.volume_overlap import check_volume_overlap
from omtra.eval.pharmacophore import compute_pharmacophore_match
from omtra.eval.system import SampledSystem
from typing import Dict, Any, Optional, List
from collections import defaultdict
import numpy as np
from omtra.utils import omtra_root
import yaml
from rdkit import Chem

from rdkit import RDLogger

# Disable all standard RDKit logs
RDLogger.DisableLog('rdApp.*')

# Also silence everything below CRITICAL
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

@register_eval("validity")
def validity(
    sampled_systems: List[SampledSystem], params: Dict[str, Any]
) -> Dict[str, Any]:
    return compute_validity(sampled_systems)


@register_eval("stability")
def stability(
    sampled_systems: List[SampledSystem], params: Dict[str, Any]
) -> Dict[str, Any]:
    return compute_stability(sampled_systems)


@register_eval("reos_and_rings")
def check_reos_and_rings(
    sampled_systems: List[SampledSystem], params: Dict[str, Any]
) -> Dict[str, Any]:
    return reos_and_rings(sampled_systems)


@register_eval("geometry")
def geometry(
    sampled_systems: List[SampledSystem], params: Dict[str, Any]
) -> Dict[str, Any]:
    metric_values = defaultdict(list)
    for i, sys in enumerate(sampled_systems):
        mol_pred = sys.get_rdkit_ligand()

        try:
            res = check_geometry(mol_pred=mol_pred, **params)
        except Exception as e:
            print(f"check_geometry failed for molecule {i}: {e}")
            continue

        res = res.get("results", {})
        number_bonds = res.get("number_bonds", -1)
        number_angles = res.get("number_angles", -1)
        number_clashes = res.get("number_clashes", -1)
        number_valid_bonds = res.get("number_valid_bonds", 0)
        number_valid_angles = res.get("number_valid_angles", 0)
        if number_bonds != 0:
            metric_values["geometry_frac_valid_bonds"].append(
                number_valid_bonds / number_bonds
            )
        if number_angles != 0:
            metric_values["geometry_frac_valid_angles"].append(
                number_valid_angles / number_angles
            )
        if number_clashes >= 0:
            metric_values["geometry_number_clashes"].append(number_clashes)

    metrics = {}
    for metric, values in metric_values.items():
        if values:
            metrics[metric] = np.nanmean(values)
        else:
            metrics[metric] = -1.0
    return metrics


@register_eval("flatness")
def flatness(
    sampled_systems: List[SampledSystem], params: Dict[str, Any]
) -> Dict[str, Any]:
    metrics = {}
    metric_values = defaultdict(list)
    for i, sys in enumerate(sampled_systems):
        mol_pred = sys.get_rdkit_ligand()
        res = check_flatness(mol_pred=mol_pred, **params)
        res = res.get("results", {})
        num_systems_checked = res.get("num_systems_checked", -1)
        num_systems_passed = res.get("num_systems_passed", 0)
        max_distance = res.get("max_distance", -1.0)
        if max_distance >= 0:
            metric_values["flatness_max_distance_from_plane"].append(max_distance)
        if num_systems_checked != 0:
            metric_values["flatness_frac_ring_systems_passed"].append(
                num_systems_passed / num_systems_checked
            )

    for metric, values in metric_values.items():
        if values:
            metrics[metric] = np.nanmean(values)
        else:
            metrics[metric] = -1.0
    return metrics


@register_eval("intermolecular_distance")
def intermolecular_distance(
    sampled_systems: List[SampledSystem], params: Dict[str, Any]
) -> Dict[str, Any]:
    metrics = {}
    metric_values = defaultdict(list)
    for i, sys in enumerate(sampled_systems):
        mol_pred = sys.get_rdkit_ligand()
        mol_cond = sys.get_rdkit_protein()
        res = check_intermolecular_distance(
            mol_pred=mol_pred, mol_cond=mol_cond, **params
        )
        res = res.get("results", {})
        smallest_distance = res.get("smallest_distance", -1.0)
        if smallest_distance >= 0:
            metric_values["intermolecular_distance_smallest"].append(smallest_distance)

    for metric, values in metric_values.items():
        if values:
            metrics[metric] = np.nanmean(values)
        else:
            metrics[metric] = -1.0
    return metrics


@register_eval("ligand_rmsd")
def ligand_rmsd(
    sampled_systems: List[SampledSystem], params: Dict[str, Any]
) -> Dict[str, Any]:
    metrics = {}
    metric_values = defaultdict(list)
    for i, sys in enumerate(sampled_systems):
        mol_pred = sys.get_rdkit_ligand()
        mol_true = sys.get_rdkit_ref_ligand()
        res = check_rmsd(mol_pred=mol_pred, mol_true=mol_true, **params)
        res = res.get("results", {})
        rmsd = res.get("rmsd", -1.0)
        if rmsd >= 0:
            metric_values["ligand_rmsd"].append(rmsd)
        kabsch_rmsd = res.get("kabsch_rmsd", -1.0)
        if kabsch_rmsd >= 0:
            metric_values["ligand_kabsch_rmsd"].append(kabsch_rmsd)
        centroid_distance = res.get("centroid_distance", -1.0)
        if centroid_distance >= 0:
            metric_values["ligand_centroid_distance"].append(centroid_distance)

    for metric, values in metric_values.items():
        if values:
            metrics[metric] = np.nanmean(values)
        else:
            metrics[metric] = -1.0
    return metrics


@register_eval("volume_overlap")
def volume_overlap(
    sampled_systems: List[SampledSystem], params: Dict[str, Any]
) -> Dict[str, Any]:
    metrics = {}
    metric_values = defaultdict(list)
    for i, sys in enumerate(sampled_systems):
        mol_pred = sys.get_rdkit_ligand()
        mol_cond = sys.get_rdkit_protein()
        res = check_volume_overlap(mol_pred=mol_pred, mol_cond=mol_cond, **params)
        res = res.get("results", {})
        vol_overlap = res.get("volume_overlap", -1.0)
        if vol_overlap >= 0:
            metric_values["volume_overlap"].append(vol_overlap)
    for metric, values in metric_values.items():
        if values:
            metrics[metric] = np.nanmean(values)
        else:
            metrics[metric] = -1.0
    return metrics


@register_eval("pb_valid_unconditional")
def pb_valid_unconditional(
    sampled_systems: List[SampledSystem], params: Dict[str, Any]
) -> Dict[str, Any]:
    
    metrics = {}
    
    pb_cfg_path = omtra_root()+'/configs/pb_config/default.yaml'

    with open(pb_cfg_path, 'r') as f:
        pb_cfg = yaml.safe_load(f)
    
    valid_rdmols = []
    for i, sys in enumerate(sampled_systems):
        mol_pred = sys.get_rdkit_ligand()
       
        try:
            Chem.SanitizeMol(mol_pred)

            if mol_pred.GetNumAtoms() > 0:
                valid_rdmols.append(mol_pred)
            else:
                print("PoseBusters valid check: Found molecule with no atoms valid check.")
        except Exception as e:
            print("PoseBusters valid check: Molecule failed to sanitize.")
    
    if len(valid_rdmols) == 0:
        metrics['pb_valid'] = 0.0

    else:
        buster = pb.PoseBusters(config=pb_cfg, **params)
        df_pb = buster.bust(valid_rdmols, None, None)
        pb_results = df_pb.mean().to_dict()
        pb_results = { f'pb_{key}': pb_results[key] for key in pb_results }

        n_pb_valid = df_pb[df_pb['sanitization'] == True].values.astype(bool).all(axis=1).sum()
        metrics['pb_valid'] = n_pb_valid / len(sampled_systems) 
        metrics.update(pb_results)
    
    return metrics


@register_eval("pb_valid_pocket")
def pb_valid_pocket(
    sampled_systems: List[SampledSystem], params: Dict[str, Any]
) -> Dict[str, Any]:
    
    metrics = {}
    
    valid_rdmols = []
    valid_receptors = []

    for i, sys in enumerate(sampled_systems):
        mol_pred = sys.get_rdkit_ligand()
        receptor = sys.get_rdkit_protein()
       
        try:
            Chem.SanitizeMol(mol_pred)

            if mol_pred.GetNumAtoms() > 0:
                valid_rdmols.append(mol_pred)
                valid_receptors.append(receptor)
            else:
                print("PoseBusters valid check: Found molecule with no atoms valid check.")
        except Exception as e:
            print("PoseBusters valid check: Molecule failed to sanitize.")
            
    
    if len(valid_rdmols) == 0:
        metrics['pb_valid_pocket'] = 0.0

    else:
        buster = pb.PoseBusters(config="complex", **params)
        df_pb = buster.bust(valid_rdmols, valid_receptors, None)
        n_pb_valid = df_pb[df_pb['sanitization'] == True].values.astype(bool).all(axis=1).sum()
        metrics['pb_valid_pocket'] = n_pb_valid / len(sampled_systems) 
    
    return metrics


@register_eval("pb_valid_conformer")
def pb_valid_conformer(
    sampled_systems: List[SampledSystem], params: Dict[str, Any]
) -> Dict[str, Any]:

    metrics = {}
    
    pb_cfg_path = omtra_root()+'/configs/pb_config/uncond_conf.yaml'

    with open(pb_cfg_path, 'r') as f:
        pb_cfg = yaml.safe_load(f)
    
    valid_rdmols = []
    ref_mols = []
    for i, sys in enumerate(sampled_systems):
        mol_pred = sys.get_rdkit_ligand()
        mol_true = sys.get_rdkit_ref_ligand()
       
        try:
            Chem.SanitizeMol(mol_pred)
            if mol_true is not None:
                Chem.SanitizeMol(mol_true)

            if mol_pred.GetNumAtoms() > 0:
                valid_rdmols.append(mol_pred)
                if mol_true is not None:
                    ref_mols.append(mol_true)
            else:
                print("PoseBusters conformer check: Found molecule with no atoms.")
        except Exception as e:
            print("PoseBusters conformer check: Molecule failed to sanitize.")
    
    if len(valid_rdmols) == 0:
        metrics['pb_valid_conformer'] = 0.0
    else:
        mol_true_list = ref_mols if len(ref_mols) > 0 else None
        
        buster = pb.PoseBusters(config=pb_cfg, **params)
        df_pb = buster.bust(valid_rdmols, mol_true_list, None)
        pb_results = df_pb.mean().to_dict()
        pb_results = { f'pb_conformer_{key}': pb_results[key] for key in pb_results }

        n_pb_valid = df_pb[df_pb['sanitization'] == True].values.astype(bool).all(axis=1).sum()
        metrics['pb_valid_conformer'] = n_pb_valid / len(sampled_systems)
        metrics.update(pb_results)
    
    return metrics


@register_eval("pharm_match")
def check_pharm_match(
    sampled_systems: List[SampledSystem], params: Dict[str, Any]
) -> Dict[str, Any]:
    return compute_pharmacophore_match(sampled_systems, **params)