from omtra.aux_losses.register import register_aux_loss

import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch_cluster as tc

from typing import Dict
from omtra.utils.graph import g_local_scope
from omtra.tasks.tasks import Task
from omtra.tasks.modalities import name_to_modality
from omtra.models.gvp import _norm_no_nan
import torch.nn.functional as F

@register_aux_loss(name='prot_lig_pairdist')
class ProtLigDist(nn.Module):

    # TODO: a better engineer would define a general pairwise distance class that operated across any arbitrary edge type
    # and then perhaps define separate pair distance classes as child class of this general one

    def __init__(self, 
                 weight=1.0, 
                 d_max=4.5
        ):
        super().__init__()
        self.weight = weight
        self.d_max = d_max

    @g_local_scope
    def forward(self, 
                g: dgl.DGLHeteroGraph, 
                dst_dict: Dict[str, torch.Tensor], 
                task: Task, 
                node_batch_idxs: Dict[str, torch.Tensor],
                lig_ue_mask: torch.Tensor,
                time_weights: torch.Tensor
                ):
        # Compute the pairwise distance loss
        

        canonical_etype = ('lig', 'lig_to_prot_atom', 'prot_atom')
        src_ntype, etype, dst_ntype = canonical_etype
        modalities_generated = task.modalities_generated

        # determine the features to use to compute the pairwise distances in the generated structure,
        # set these features as x_d on the relevant node types
        if name_to_modality('lig_x') in modalities_generated:
            g.nodes['lig'].data['x_d'] = dst_dict['lig_x']
        else:
            g.nodes['lig'].data['x_d'] = g.nodes['lig'].data['x_1_true']

        if name_to_modality('prot_atom_x') in modalities_generated:
            g.nodes['prot_atom'].data['x_d'] = dst_dict['prot_atom_x']
        else:
            g.nodes['prot_atom'].data['x_d'] = g.nodes['prot_atom'].data['x_1_true']


        # get edges for pairwise distance. 
        # this requires a 4.5 angstrom radius graph on the ground-truth structure
        edge_idxs = tc.radius(
            x=g.nodes['prot_atom'].data['x_1_true'], 
            y=g.nodes['lig'].data['x_1_true'], 
            batch_x=node_batch_idxs['prot_atom'], 
            batch_y=node_batch_idxs['lig'], 
            r=self.d_max, 
            max_num_neighbors=15)
        src_idxs = edge_idxs[0] # lig atom indicies
        dst_idxs = edge_idxs[1]  # prot atom indicies


        # compute dij for generated structure
        x_diff_gen = g.nodes['lig'].data['x_d'][src_idxs] - g.nodes['prot_atom'].data['x_d'][dst_idxs]
        dij_gen = _norm_no_nan(x_diff_gen)

        # compute dij for ground-truth structure
        x_diff_true = g.nodes['lig'].data['x_1_true'][src_idxs] - g.nodes['prot_atom'].data['x_1_true'][dst_idxs]
        dij_true = _norm_no_nan(x_diff_true)

        # TODO: what about time-dependent weighting? 
        time_scaled_loss = time_weights is not None
        reduction = 'none' if time_scaled_loss else 'mean'
        loss = F.mse_loss(dij_gen, dij_true, reduction=reduction)

        if time_scaled_loss:
            # time_weights is a tensor of shape (batch_size,) containing the weight that needs to be applied to each graph in the batch
            # so we have to expand this out to the pairs on which this loss is being taken
            # i.e., for each pair of atoms (for each edge) - which batch item does it belong to?
            time_weights_expanded = time_weights[ node_batch_idxs['lig'][src_idxs] ]
            loss = loss * time_weights_expanded
            loss = loss.mean()

        return loss * self.weight

    def supports_task(self, task: Task) -> bool:
        """
        Check if this auxiliary loss supports the given task.
        This loss is only applicable to tasks that involve protein-ligand interactions.
        """
        return 'protein_identity' in task.groups_present

@register_aux_loss(name='lig_pairdist')
class LigPairLoss(nn.Module):
    """
    Auxiliary loss for pairwise distances in the ligand structure.
    This loss computes the MSE between the pairwise distances of the generated ligand structure
    and the ground-truth ligand structure.
    """

    def __init__(self, 
                 d_max: float = 4.0, 
                 weight: float = 1.0):
        super().__init__()
        self.d_max = d_max
        self.weight = weight


    def forward(self, 
                g: dgl.DGLHeteroGraph, 
                dst_dict: Dict[str, torch.Tensor], 
                task: Task, 
                node_batch_idxs: Dict[str, torch.Tensor],
                lig_ue_mask: torch.Tensor,
                time_weights: torch.Tensor
                ):


        etype = 'lig_to_lig'
        src_idxs, dst_idxs = g.edges(etype=etype)

        # take only upper-triangle edges
        src_idxs = src_idxs[lig_ue_mask]
        dst_idxs = dst_idxs[lig_ue_mask]

        # pairwise distances on ground-truth
        x_1_true = g.nodes['lig'].data['x_1_true']
        x_diff_true = x_1_true[src_idxs] - x_1_true[dst_idxs]
        dij_true = _norm_no_nan(x_diff_true)

        # get a mask for distances < d_max
        d_mask = dij_true < self.d_max

        # apply that mask on atom pairs
        dij_true = dij_true[d_mask]

        # pairwise distances on predicted structure
        x_1_pred = dst_dict['lig_x']
        x_diff_gen = x_1_pred[src_idxs[d_mask]] - x_1_pred[dst_idxs[d_mask]]
        dij_pred = _norm_no_nan(x_diff_gen)

        time_scaled_loss = time_weights is not None
        reduction = 'none' if time_scaled_loss else 'mean'
        loss = F.mse_loss(dij_pred, dij_true, reduction=reduction)

        if time_scaled_loss:
            # time_weights is a tensor of shape (batch_size,) containing the weight that needs to be applied to each batch item
            # so we have to expand this out to the pairs on which this loss is being taken
            # i.e., for each pair of atoms (for each edge) - which batch item does it belong to?
            time_weights_expanded = time_weights[ node_batch_idxs['lig'][src_idxs[d_mask]] ]
            loss = loss * time_weights_expanded
            loss = loss.mean()

        return loss * self.weight

    def supports_task(self, task: Task) -> bool:
        """
        Check if this auxiliary loss supports the given task.
        """
        return 'ligand_identity' in task.groups_present or 'ligand_identity_condensed' in task.groups_present
