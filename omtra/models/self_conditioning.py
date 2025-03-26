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

# TODO: adapt for heterographs

class SelfConditioningResidualLayer(nn.Module):
    def __init__(
        self,
        node_embedding_dim,
        edge_embedding_dim,
        rbf_dim,
        rbf_dmax,
    ):
        super().__init__()

        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.node_types = set()
        self.edge_types = set()

        self.node_residual_mlps = nn.ModuleDict()
        
        extra_dims = defaultdict(int)
        for modality_name in MODALITY_ORDER:
            modality = name_to_modality(modality_name)
            if modality.graph_entity == "node":
                ntype = modality.entity_name
                self.node_types.add(ntype)
                if modality.n_categories is not None:
                    extra_dims[ntype] += modality.n_categories
            else: # edge
                etype = modality.entity_name
                self.edge_types.add(etype)
                if modality.n_categories is not None:
                    extra_dims[etype] += modality.n_categories
        
        for ntype in self.node_types:
            self.node_residual_mlps[ntype] = nn.Sequential(
                nn.Linear(node_embedding_dim + extra_dims[ntype] + rbf_dim, node_embedding_dim),
                nn.SiLU(),
                nn.Linear(node_embedding_dim, node_embedding_dim),
                nn.SiLU(),
            )
        for etype in self.edge_types:
            if etype not in extra_dims:
                continue
            self.edge_residual_mlps[etype] = nn.Sequential(
                nn.Linear(edge_embedding_dim + extra_dims[etype] + rbf_dim, edge_embedding_dim),
                nn.SiLU(),
                nn.Linear(edge_embedding_dim, edge_embedding_dim),
                nn.SiLU(),
            )


    def forward(
        self,
        g: dgl.DGLGraph,
        s_t: Dict[str, torch.Tensor],
        x_t: Dict[str, torch.Tensor],
        v_t: Dict[str, torch.Tensor],
        e_t: Dict[str, torch.Tensor],
        dst_dict: Dict[str, torch.Tensor],
        node_batch_idx: Dict[str, torch.Tensor],
        upper_edge_mask: Dict[str, torch.Tensor],
    ):
        # get distances between each node in current timestep and the same node at t=1
        
        d_node = {}
        for ntype in self.node_types:
            d_node[ntype] = _norm_no_nan(x_t[ntype] - dst_dict[f"{ntype}_x"]) # TODO: fix this to handle when positions are in dst_dict (i.e. when keeping certain nodes fixed)
            d_node[ntype] = _rbf(d_node[ntype], D_max=self.rbf_dmax, D_count=self.rbf_dim)
            
            node_residual_inputs = [ # TODO: figure out how to use task_class/modalities for this
                s_t[ntype],
                dst_dict["a"],
                dst_dict["c"],
                d_node,
            ]
        node_residual = self.node_residual_mlp(torch.cat(node_residual_inputs, dim=-1))

        # get the edge length of every edge in g at time t and also the edge length at t=1
        d_edge_t = self.edge_distances(g, node_positions=x_t)
        d_edge_1 = self.edge_distances(g, node_positions=dst_dict["x"])

        # take only upper-triangle edges, for efficiency
        d_edge_t = d_edge_t[upper_edge_mask]
        d_edge_1 = d_edge_1[upper_edge_mask]

        edge_residual_inputs = [
            e_t[upper_edge_mask],  # current state of the edge
            dst_dict["e"],  # final state of the edge
            d_edge_1 - d_edge_t,  # change in edge length
        ]
        edge_residual = self.edge_residual_mlp(torch.cat(edge_residual_inputs, dim=-1))

        node_feats_out = s_t + node_residual
        positions_out = x_t
        vectors_out = v_t

        edge_feats_out = torch.zeros_like(e_t)
        one_triangle_output = e_t[upper_edge_mask] + edge_residual
        edge_feats_out[upper_edge_mask] = one_triangle_output
        edge_feats_out[~upper_edge_mask] = one_triangle_output

        return node_feats_out, positions_out, vectors_out, edge_feats_out

    def edge_distances(self, g: dgl.DGLGraph, node_positions=None):
        """Precompute the pairwise distances between all nodes in the graph."""

        with g.local_scope():
            if node_positions is None:
                g.ndata["x_d"] = g.ndata["x_t"]
            else:
                g.ndata["x_d"] = node_positions

            g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"))
            dij = _norm_no_nan(g.edata["x_diff"], keepdims=True) + 1e-8
            # x_diff = g.edata['x_diff'] / dij
            d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)

        return d
