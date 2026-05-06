"""Metrics for partial fragment conditioning tasks.

Computes per-sample statistics on how much the model redesigned atoms and
bonds that belong to the fixed (conditioned) fragment, and whether redesigned
atoms end up within 2 Å of their ground-truth positions.
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

    # ---- Atom (node) metrics ----
    a_pred = lig_data["a_1"][fixed_node_mask]
    a_true = lig_data["a_1_true"][fixed_node_mask]
    x_pred = lig_data["x_1"][fixed_node_mask].float()
    x_true = lig_data["x_1_true"][fixed_node_mask].float()

    atom_redesigned = a_pred != a_true
    n_redesigned = int(atom_redesigned.sum().item())
    result["frac_fixed_atoms_redesigned"] = n_redesigned / n_fixed

    dists = torch.norm(x_pred - x_true, dim=-1)
    if n_redesigned > 0:
        result["frac_fixed_atoms_redesigned_within_2A"] = (
            float((atom_redesigned & (dists < 2.0)).sum().item()) / n_redesigned
        )
    else:
        result["frac_fixed_atoms_redesigned_within_2A"] = float("nan")

    # ---- Edge (bond) metrics ----
    edge_data = g.edges["lig_to_lig"].data
    if not all(k in edge_data for k in ("edge_mask_1_true", "e_1", "e_1_true")):
        return result

    upper_mask = get_upper_edge_mask(g, etype="lig_to_lig")
    fixed_edge_mask = edge_data["edge_mask_1_true"].bool() & upper_mask
    n_fixed_edges = int(fixed_edge_mask.sum().item())

    if n_fixed_edges == 0:
        return result

    e_pred = edge_data["e_1"][fixed_edge_mask]
    e_true = edge_data["e_1_true"][fixed_edge_mask]

    edge_redesigned = e_pred != e_true
    n_edges_redesigned = int(edge_redesigned.sum().item())
    result["frac_fixed_edges_redesigned"] = n_edges_redesigned / n_fixed_edges

    if n_edges_redesigned > 0:
        src, dst = g.edges(etype="lig_to_lig")
        src_fixed = src[fixed_edge_mask]
        dst_fixed = dst[fixed_edge_mask]

        x_pred_all = lig_data["x_1"].float()
        x_true_all = lig_data["x_1_true"].float()

        mid_pred = (x_pred_all[src_fixed] + x_pred_all[dst_fixed]) / 2
        mid_true = (x_true_all[src_fixed] + x_true_all[dst_fixed]) / 2
        edge_dists = torch.norm(mid_pred - mid_true, dim=-1)

        result["frac_fixed_edges_redesigned_within_2A"] = (
            float((edge_redesigned & (edge_dists < 2.0)).sum().item()) / n_edges_redesigned
        )
    else:
        result["frac_fixed_edges_redesigned_within_2A"] = float("nan")

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
                gen_ligand_id = "gen_ligands_0"
            else:
                protein_id = "protein_0"
                gen_ligand_id = f"gen_ligands_{rep_idx}"

            row = {"sys_id": sys_id, "protein_id": protein_id, "gen_ligand_id": gen_ligand_id}
            row.update(m)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index(["sys_id", "protein_id", "gen_ligand_id"], inplace=True)
    return df
