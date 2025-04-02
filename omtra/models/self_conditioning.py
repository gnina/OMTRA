import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from collections import defaultdict
from typing import Dict

from omtra.models.gvp import _norm_no_nan, _rbf
from omtra.tasks.modalities import MODALITY_ORDER, name_to_modality
from omtra.utils.embedding import rbf_twoscale
from omtra.utils.graph import canonical_node_features
from omtra.load.conf import TaskDatasetCoupling
from omtra.tasks.register import task_name_to_class
from omtra.tasks.tasks import Task
from functools import partial
from einops import rearrange

# TODO: adapt for heterographs

class SelfConditioningResidualLayer(nn.Module):
    def __init__(
        self,
        td_coupling: TaskDatasetCoupling,
        node_embedding_dim,
        edge_embedding_dim,
        rbf_dim,
        rbf_dmax,
        n_pharmvec_channels=4,
    ):
        super().__init__()

        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax

        modalities_present = [m 
            for task_name in td_coupling.task_space
            for m in task_name_to_class(task_name).modalities_present]
        modalities_generated = [m
            for task_name in td_coupling.task_space
            for m in task_name_to_class(task_name).modalities_generated]
        modalities_present = list(set(modalities_present))
        modalities_generated = list(set(modalities_generated))

        self.t1_embedding_fns = nn.ModuleDict()

        # handle node modalities
        for modality in modalities_generated:
            if not modality.is_node:
                continue
            if modality.is_categorical:
                embedding_fn = nn.Linear(
                    modality.n_categories, node_embedding_dim
                )
            elif modality.data_key == 'x': # positions
                embedding_fn = nn.Linear(self.rbf_dim, node_embedding_dim)
                # nn.Sequential(
                #     partial(_rbf, D_max=self.rbf_dmax, D_count=self.rbf_dim),
                #     nn.Linear(self.rbf_dim, node_embedding_dim),
                # )
            elif modality.data_key == 'v': # vectors
                embedding_fn = nn.Linear(n_pharmvec_channels*2, node_embedding_dim)
            else:
                raise ValueError(f"Unexpected modality: {modality.name}")
            self.t1_embedding_fns[modality.name] = embedding_fn

        ntypes_generated = set(m.entity_name for m in modalities_generated if m.is_node)
        self.node_residual_mlps = nn.ModuleDict()
        for ntype in ntypes_generated:
            self.node_residual_mlps[ntype] = nn.Sequential(
                nn.Linear(node_embedding_dim, node_embedding_dim),
                nn.SiLU(),
                # nn.Linear(node_embedding_dim, node_embedding_dim),
                # nn.SiLU(),
            )
        # if we are modeling ligand structure, we want to encode changes in edge length on lig_to_lig edges
        # this may be subject to change in the future, like if we stop maintaining edge features?
        if name_to_modality('lig_x') in modalities_generated:
            input_dim = edge_embedding_dim + name_to_modality('lig_e').n_categories + self.rbf_dim
            self.lig_edge_residual_mlp = nn.Sequential(
                nn.Linear(input_dim, edge_embedding_dim),
                nn.SiLU(),
                # nn.Linear(edge_embedding_dim, edge_embedding_dim),
                # nn.SiLU(),
            )


    def forward(
        self,
        g: dgl.DGLGraph,
        task: Task,
        s_t: Dict[str, torch.Tensor],
        x_t: Dict[str, torch.Tensor],
        v_t: Dict[str, torch.Tensor],
        e_t: Dict[str, torch.Tensor],
        dst_dict: Dict[str, torch.Tensor],
        node_batch_idx: Dict[str, torch.Tensor],
        upper_edge_mask: Dict[str, torch.Tensor],
    ):

        node_residual_inputs = defaultdict(list)
        for m in task.modalities_generated:
            if not m.is_node:
                continue
            ntype = m.entity_name
            if m.data_key == 'x':
                # for positions, we add the distance to the final position and the initial position
                dij = x_t[ntype] - dst_dict[m.name]
                dij = _norm_no_nan(dij, keepdims=True)
                dij = _rbf(dij, D_max=self.rbf_dmax, D_count=self.rbf_dim)
                node_res_input = self.t1_embedding_fns[m.name](dij)
            elif m.data_key == 'v':
                # for vectors, we add pairwise distances between vector features, between t=t and t=1
                # has shape (n_nodes, n_pharmvec_channels, n_pharmvec_channels, 3)
                dij = v_t[ntype].unsqueeze(2) - dst_dict[m.name].unsqueeze(1) 
                dij = _norm_no_nan(dij) # has shape (n_nodes, n_pharmvec_channels, n_pharmvec_channels)
                # flatten to shape (n_nodes, n_pharmvec_channels+n_pharmvec_channels)
                dij = rearrange(dij, 'n c1 c2 -> n (c1 c2)')
                node_res_input = self.t1_embedding_fns[m.name](dij)
            elif m.is_categorical:
                # for categorical features, we just add the final state
                node_res_input = self.t1_embedding_fns[m.name](dst_dict[m.name])
            else:
                raise ValueError(f"Unexpected modality: {m.name}")
            node_residual_inputs[ntype].append(node_res_input)

        # get residuals for nodes
        node_residuals = {}
        for ntype in node_residual_inputs.keys():
            # sum inputs to the residual fn
            res_inputs = sum(node_residual_inputs[ntype])
            # apply the residual fn
            node_residual[ntype] = self.node_residual_mlps[ntype](res_inputs)

        node_residual = self.node_residual_mlp(torch.cat(node_residual_inputs, dim=-1))

        # do edge residual for lig_lig edges (the only edges where we maintain edge features)
        if name_to_modality('lig_x') in task.modalities_generated:
            # for edge features, we add the distance to the final position and the initial position
            etype = ("lig", "lig_to_lig", "lig")
            d_edge_t = self.edge_distances(g, etype, node_positions=g.nodes['lig'].data["x_t"])
            d_edge_1 = self.edge_distances(g, etype, node_positions=dst_dict["lig_x"])
            d_input = (d_edge_1 - d_edge_t)[upper_edge_mask['lig_to_lig']]
            edge_residual_inputs = [
                e_t['lig_to_lig'][upper_edge_mask['lig_to_lig']],  # current state of the edge
                dst_dict["lig_e"],  # final state of the edge
                d_input,  # change in edge length
            ]
            edge_residual = self.lig_edge_residual_mlp(torch.cat(edge_residual_inputs, dim=-1))
            edge_feats_out = torch.zeros_like(e_t['lig_to_lig'])
            edge_feats_out[upper_edge_mask['lig_to_lig']] = edge_residual
            edge_feats_out[~upper_edge_mask['lig_to_lig']] = edge_residual

        # apply residual to get output node features
        node_feats_out = {}
        for ntype in node_residual:
            node_feats_out[ntype] = s_t[ntype] + node_residual[ntype]

        positions_out = x_t
        vectors_out = v_t

        return node_feats_out, positions_out, vectors_out, {'lig_to_lig': edge_feats_out}

    def edge_distances(self, g: dgl.DGLGraph, canonical_etype: tuple, node_positions=None):
        """Precompute the pairwise distances between all nodes in the graph."""

        src_ntype, etype, dst_ntype = canonical_etype

        if src_ntype != dst_ntype and node_positions is not None:
            assert isinstance(node_positions, tuple) and len(node_positions) == 2
        elif src_ntype == dst_ntype and node_positions is not None:
            assert isinstance(node_positions, torch.Tensor)

        with g.local_scope():
            if node_positions is None:
                # g.ndata["x_d"] = g.ndata["x_t"]
                g.nodes[src_ntype].data["x_d"] = g.nodes[src_ntype].data["x_t"]
                g.nodes[dst_ntype].data["x_d"] = g.nodes[dst_ntype].data["x_t"]
            else:
                g.nodes[src_ntype].data["x_d"] = node_positions[0]
                g.nodes[dst_ntype].data["x_d"] = node_positions[1]

            g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"), etype=etype)
            dij = _norm_no_nan(g.edges[etype].data["x_diff"])
            # x_diff = g.edata['x_diff'] / dij
            d = _rbf(dij, D_max=self.rbf_dmax, D_count=self.rbf_dim)

        return d
