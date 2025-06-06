import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from collections import defaultdict
from typing import Dict
import torch.nn.functional as tfn

from omtra.models.gvp import _norm_no_nan, _rbf
from omtra.tasks.modalities import name_to_modality
from omtra.load.conf import TaskDatasetCoupling
from omtra.tasks.register import task_name_to_class
from omtra.tasks.tasks import Task
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
        fake_atoms: bool = False,
    ):
        super().__init__()

        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.n_pharmvec_channels = n_pharmvec_channels
        self.fake_atoms = fake_atoms

        modalities_present = [m 
            for task_name in sorted(td_coupling.task_space)
            for m in task_name_to_class(task_name).modalities_present]
        modalities_generated = [m
            for task_name in sorted(td_coupling.task_space)
            for m in task_name_to_class(task_name).modalities_generated]
        modalities_present = list(set(modalities_present))
        modalities_generated = list(set(modalities_generated))

        self.node_generated_dims = defaultdict(int)
        # handle node modalities
        for modality in modalities_present:
            if not modality.is_node:
                continue
            ntype = modality.entity_name
            if modality.is_categorical:
                self.node_generated_dims[ntype] += modality.n_categories
                if modality.name == 'lig_a' and self.fake_atoms:
                    # if we are using fake atoms, we need to add an extra dimension for the fake atom type
                    self.node_generated_dims[ntype] += 1
            elif modality.data_key == 'x': # positions
                self.node_generated_dims[ntype] += rbf_dim
            elif modality.data_key == 'v': # vectors
                self.node_generated_dims[ntype] += int(self.n_pharmvec_channels**2)
            else:
                raise ValueError(f"Unexpected modality: {modality.name}")
            
        self.residual_generating_fns = nn.ModuleDict()
        for ntype in sorted(list(self.node_generated_dims.keys())):
            self.residual_generating_fns[ntype] = nn.Sequential(
                nn.Linear(node_embedding_dim+self.node_generated_dims[ntype], node_embedding_dim),
                nn.SiLU(),
                nn.Linear(node_embedding_dim, node_embedding_dim),
                nn.SiLU(),
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
        for m in task.modalities_present:
            if not m.is_node:
                continue
            ntype = m.entity_name
            if g.num_nodes(ntype) == 0:
                continue
            m_generated = m.name in dst_dict
            if m.data_key == 'x':
                # for positions, the distance to the final position and the initial position
                # is used to update node scalar features
                if m_generated:
                    x_diff = dst_dict[m.name] - x_t[ntype]
                else:
                    x_diff = torch.zeros_like(x_t[ntype])
                dij = _norm_no_nan(x_diff)
                dij_rbf = _rbf(dij, D_max=self.rbf_dmax, D_count=self.rbf_dim)
                node_res_input = dij_rbf

                # also, the displacement vector is used to update node vector features
                if m_generated:
                    v_t[ntype][:, -1, :] = x_diff / dij.unsqueeze(-1)

            elif m.data_key == 'v':
                # for vectors, pairwise distances between vector features are used
                # to update node scalar features
                # has shape (n_nodes, n_pharmvec_channels, n_pharmvec_channels, 3)
                if m_generated:
                    dij = dst_dict[m.name].unsqueeze(1) - v_t[ntype][:, :self.n_pharmvec_channels].unsqueeze(2)
                    dij = _norm_no_nan(dij) # has shape (n_nodes, n_pharmvec_channels, n_pharmvec_channels)
                    # flatten to shape (n_nodes, n_pharmvec_channels+n_pharmvec_channels)
                    dij = rearrange(dij, 'n c1 c2 -> n (c1 c2)')
                else:
                    dij = torch.zeros(g.num_nodes(ntype), self.n_pharmvec_channels**2, device=g.device, dtype=v_t[ntype].dtype)
                node_res_input = dij

                # also, the last self.n_pharmvec_channels of the vector features are set previously predicted values
                if m_generated:
                    v_t[ntype][:, -self.n_pharmvec_channels:, :] = dst_dict[m.name]
            elif m.is_categorical:
                # for categorical features, we just add the final state
                if m_generated:
                    node_res_input = dst_dict[m.name] # if we generating this feature, just use predicted logits
                else:
                    # if we are not generating this feature (it is fixed), use the current state
                    extra_dim = int(m.name == 'lig_a' and self.fake_atoms)
                    node_res_input = tfn.one_hot(g.nodes[ntype].data[f'{m.data_key}_t'], m.n_categories+extra_dim)
            else:
                raise ValueError(f"Unexpected modality: {m.name}")
            node_residual_inputs[ntype].append(node_res_input)

        node_residuals = {}
        for ntype in node_residual_inputs:
            # add the current state of the node to the residual inputs
            node_residual_inputs[ntype].append(s_t[ntype])
            # compute residuals for this node type
            node_residuals[ntype] = self.residual_generating_fns[ntype](
                torch.cat(node_residual_inputs[ntype], dim=-1)
            )
        # do edge residual for lig_lig edges (the only edges where we maintain edge features)
        if name_to_modality('lig_x') in task.modalities_generated:
            # for edge features, we add the distance to the final position and the initial position
            etype = ("lig", "lig_to_lig", "lig")
            d_edge_t = self.edge_distances(g, etype, node_positions=g.nodes['lig'].data["x_t"])
            d_edge_1 = self.edge_distances(g, etype, node_positions=dst_dict["lig_x"])
            d_input = (d_edge_1 - d_edge_t)[upper_edge_mask['lig_to_lig']]

            if 'lig_e' in dst_dict:
                last_pred_state = dst_dict["lig_e"]
            else:
                last_pred_state = tfn.one_hot(g.edges['lig_to_lig'].data['e_t'][upper_edge_mask['lig_to_lig']], name_to_modality('lig_e').n_categories)

            edge_residual_inputs = [
                e_t['lig_to_lig'][upper_edge_mask['lig_to_lig']],  # current state of the edge
                last_pred_state,  # final state of the edge
                d_input,  # change in edge length
            ]
            edge_residual = self.lig_edge_residual_mlp(torch.cat(edge_residual_inputs, dim=-1))
            edge_feats_w_residual = e_t['lig_to_lig'][upper_edge_mask['lig_to_lig']] + edge_residual
            ll_feats = torch.zeros_like(e_t['lig_to_lig'])
            ll_feats[upper_edge_mask['lig_to_lig']] = edge_feats_w_residual
            ll_feats[~upper_edge_mask['lig_to_lig']] = edge_feats_w_residual
            e_t['lig_to_lig'] = ll_feats

        # apply residual to get output node features
        node_feats_out = {}
        for ntype in node_residuals:
            node_feats_out[ntype] = s_t[ntype] + node_residuals[ntype]

        positions_out = x_t
        vectors_out = v_t
        edge_feats_out = e_t

        return node_feats_out, positions_out, vectors_out, edge_feats_out

    def edge_distances(self, g: dgl.DGLGraph, canonical_etype: tuple, node_positions=None):
        """Precompute the pairwise distances between all nodes in the graph."""

        src_ntype, etype, dst_ntype = canonical_etype

        if src_ntype != dst_ntype and node_positions is not None:
            assert isinstance(node_positions, tuple) and len(node_positions) == 2
            src_positions, dst_positions = node_positions
        elif src_ntype == dst_ntype and node_positions is not None:
            assert isinstance(node_positions, torch.Tensor)
            src_positions = node_positions
            dst_positions = node_positions

        with g.local_scope():
            if node_positions is None:
                # g.ndata["x_d"] = g.ndata["x_t"]
                g.nodes[src_ntype].data["x_d"] = g.nodes[src_ntype].data["x_t"]
                g.nodes[dst_ntype].data["x_d"] = g.nodes[dst_ntype].data["x_t"]
            else:
                g.nodes[src_ntype].data["x_d"] = src_positions
                g.nodes[dst_ntype].data["x_d"] = dst_positions

            g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"), etype=etype)
            dij = _norm_no_nan(g.edges[etype].data["x_diff"])
            # x_diff = g.edata['x_diff'] / dij
            d = _rbf(dij, D_max=self.rbf_dmax, D_count=self.rbf_dim)

        return d
