"""Metrics for partial fragment conditioning tasks.

For each fixed atom/edge, measures whether the model reproduced the same type in the
generated ligand and whether the geometry stayed within 2 Å of ground truth.

Metrics returned per SampledSystem
-----------------------------------
n_atoms_fixed                   : number of fixed atoms
frac_fixed_atoms_redesigned     : fraction of fixed atoms with same type in generated ligand
frac_fixed_atoms_redesigned_≤2Å : fraction of fixed atoms with same type AND within 2 Å
frac_fixed_edges_redesigned     : fraction of fixed bonds with same bond order in generated ligand
frac_fixed_redesigned           : combined (atoms + edges) fraction with same type
frac_fixed_redesigned_≤2Å       : combined (atoms + edges) fraction with same type AND within 2 Å
                                  (edge geometry measured at bond midpoint)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from omtra.eval.system import SampledSystem
from omtra.data.graph.utils import get_upper_edge_mask


def _fixed_frag_metrics(sys: SampledSystem) -> Dict[str, float]:
    """Compute partial-fragment metrics for one SampledSystem.

    Returns an empty dict when no atom_mask_1_true is present (non-partial task).
    """
    g = sys.g
    lig_data = g.nodes["lig"].data

    if "atom_mask_1_true" not in lig_data:
        return {}

    fixed_node_mask = lig_data["atom_mask_1_true"].bool()
    n_fixed = int(fixed_node_mask.sum().item())
    result: Dict[str, float] = {"n_atoms_fixed": float(n_fixed)}

    if n_fixed == 0:
        return result

    # ---- Atom metrics ----
    a_pred = lig_data["a_1"][fixed_node_mask]
    a_true = lig_data["a_1_true"][fixed_node_mask]
    x_pred = lig_data["x_1"][fixed_node_mask].float()
    x_true = lig_data["x_1_true"][fixed_node_mask].float()

    atom_same = a_pred == a_true
    dists = torch.norm(x_pred - x_true, dim=-1)
    n_atom_same = int(atom_same.sum().item())
    n_atom_same_close = int((atom_same & (dists < 2.0)).sum().item())

    result["frac_fixed_atoms_redesigned"] = n_atom_same / n_fixed
    result["frac_fixed_atoms_redesigned_≤2Å"] = n_atom_same_close / n_fixed

    # ---- Edge metrics ----
    n_fixed_edges = 0
    n_edge_same = 0
    n_edge_same_close = 0

    edge_data = g.edges["lig_to_lig"].data
    if all(k in edge_data for k in ("edge_mask_1_true", "e_1", "e_1_true")):
        upper_mask = get_upper_edge_mask(g, etype="lig_to_lig")
        fixed_edge_mask = edge_data["edge_mask_1_true"].bool() & upper_mask
        n_fixed_edges = int(fixed_edge_mask.sum().item())

        if n_fixed_edges > 0:
            e_pred = edge_data["e_1"][fixed_edge_mask]
            e_true = edge_data["e_1_true"][fixed_edge_mask]
            edge_same = e_pred == e_true
            n_edge_same = int(edge_same.sum().item())
            result["frac_fixed_edges_redesigned"] = n_edge_same / n_fixed_edges

            src, dst = g.edges(etype="lig_to_lig")
            src_fixed = src[fixed_edge_mask]
            dst_fixed = dst[fixed_edge_mask]
            x_pred_all = lig_data["x_1"].float()
            x_true_all = lig_data["x_1_true"].float()
            mid_pred = (x_pred_all[src_fixed] + x_pred_all[dst_fixed]) / 2
            mid_true = (x_true_all[src_fixed] + x_true_all[dst_fixed]) / 2
            edge_dists = torch.norm(mid_pred - mid_true, dim=-1)
            n_edge_same_close = int((edge_same & (edge_dists < 2.0)).sum().item())

    # ---- Combined metrics (atoms + edges) ----
    n_total = n_fixed + n_fixed_edges
    result["frac_fixed_redesigned"] = (n_atom_same + n_edge_same) / n_total
    result["frac_fixed_redesigned_≤2Å"] = (n_atom_same_close + n_edge_same_close) / n_total

    # Explicit whole-fragment chemistry/coordinate success, without changing legacy fields.
    for reference, suffix in (('clean','1_true'), ('input','cond')):
        chemical_key = 'cond_a' if 'cond_a_1' in lig_data else 'a'
        target_key = f'{chemical_key}_{suffix}'
        if target_key not in lig_data:
            target_key = f'{chemical_key}_1_true'
        chemistry = lig_data[f'{chemical_key}_1'][fixed_node_mask] == lig_data[target_key][fixed_node_mask]
        if chemical_key == 'a' and 'c_1' in lig_data:
            ckey = f'c_{suffix}' if f'c_{suffix}' in lig_data else 'c_1_true'
            chemistry = chemistry & (lig_data['c_1'][fixed_node_mask] == lig_data[ckey][fixed_node_mask])
        xkey = f'x_{suffix}' if f'x_{suffix}' in lig_data else 'x_1_true'
        displacement = torch.norm(x_pred-lig_data[xkey][fixed_node_mask],dim=-1)
        edges_ok = True
        if n_fixed_edges:
            ekey = f'e_{suffix}' if f'e_{suffix}' in edge_data else 'e_1_true'
            edges_ok = bool((edge_data['e_1'][fixed_edge_mask] == edge_data[ekey][fixed_edge_mask]).all())
        result[f'fixed_fragment_{reference}_chemistry'] = float(bool(chemistry.all()) and edges_ok)
        result[f'fixed_fragment_{reference}_rmsd'] = float(displacement.square().mean().sqrt())
        result[f'fixed_fragment_{reference}_success'] = float(bool(chemistry.all()) and edges_ok and bool((displacement < 2.).all()))

    return result


def fixed_frag_metrics(
    sampled_systems: List[SampledSystem],
    n_systems: int,
    n_replicates: int,
    protein_generated: bool,
) -> pd.DataFrame:
    """Build a DataFrame of fragment metrics with the same multi-index used by compute_metrics.

    Index levels: sys_id, protein_id, gen_ligand_id.

    Parameters
    ----------
    sampled_systems:
        Flat list ordered sys_0_rep_0, sys_0_rep_1, ..., sys_{N-1}_rep_{R-1}.
    n_systems:
        Number of unique systems.
    n_replicates:
        Number of replicates per system.
    protein_generated:
        True when protein_structure is in task.groups_generated (flexible docking).
        Determines how protein_id and gen_ligand_id are assigned.
    """
    rows = []
    for sys_idx in range(n_systems):
        for rep_idx in range(n_replicates):
            flat_idx = sys_idx * n_replicates + rep_idx
            m = _fixed_frag_metrics(sampled_systems[flat_idx])

            sys_id = f"sys_{sys_idx}_gt"
            if protein_generated:
                protein_id = f"gen_prot_{rep_idx}"
                gen_ligand_id = f"gen_ligands_{rep_idx}"
            else:
                protein_id = "protein_0"
                gen_ligand_id = f"gen_ligands_{rep_idx}"

            row = {"sys_id": sys_id, "protein_id": protein_id, "gen_ligand_id": gen_ligand_id}
            row.update(m)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index(["sys_id", "protein_id", "gen_ligand_id"], inplace=True)
    return df
