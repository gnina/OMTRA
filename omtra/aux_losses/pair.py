from omtra.aux_losses.register import register_aux_loss

import torch
import torch.nn as nn
import dgl
import dgl.function as fn

from typing import Dict
from omtra.utils.graph import g_local_scope
from omtra.tasks.tasks import Task
from omtra.models.gvp import _norm_no_nan

@register_aux_loss(name='prot_lig_pairdist')
class ProtLigDist(nn.Module):

    # TODO: a better engineer would define a general pairwise distance class that operated across any arbitrary edge type
    # and then perhaps define separate pair distance classes as child class of this general one

    def __init__(self):
        super().__init__()

    @g_local_scope
    def forward(self, 
                g: dgl.DGLHeteroGraph, 
                dst_dict: Dict[str, torch.Tensor], 
                task: Task, 
                lig_ue_mask: torch.Tensor):
        # Compute the pairwise distance loss
        

        canonical_etype = ('lig', 'lig_to_prot_atom', 'prot_atom')
        src_ntype, etype, dst_ntype = canonical_etype
        modalities_generated = task.modalities_generated

        # determine the features to use to compute the pairwise distances in the generated structure,
        # set these features as x_d on the relevant node types
        if 'lig_x' in modalities_generated:
            g.nodes['lig'].data['x_d'] = dst_dict['lig_x']
        else:
            g.nodes['lig'].data['x_d'] = g.nodes['lig'].data['x_1_true']

        if 'prot_atom_x' in modalities_generated:
            g.nodes['prot_atom'].data['x_d'] = dst_dict['prot_atom_x']
        else:
            g.nodes['prot_atom'].data['x_d'] = g.nodes['prot_atom'].data['x_1_true']

        # compute dij for generated structure
        g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff_gen"), etype=etype)
        dij_gen = _norm_no_nan(g.edges[etype].data['x_diff_gen'])

        # compute dij for ground-truth structure
        g.apply_edges(fn.u_sub_v("x_1_true", "x_1_true", "x_diff"), etype=etype)
        dij = _norm_no_nan(g.edges[etype].data['x_diff'])

        # TODO: what about time-dependent weighting? 

        loss = nn.MSELoss()(dij_gen, dij)
        return loss
    
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
    
    def __init__(self, d_max: float = 4.0):
        super().__init__()
        self.d_max = d_max

    def forward(self, 
                g: dgl.DGLHeteroGraph, 
                dst_dict: Dict[str, torch.Tensor], 
                task: Task, 
                lig_ue_mask: torch.Tensor):
        
        # TODO: use task to determine if we need this loss at all


        etype = 'lig_to_lig'

        src_idxs, dst_idxs = g.edges(etype=etype)

        # take only upper-triangle edges
        src_idxs = src_idxs[lig_ue_mask]
        dst_idxs = dst_idxs[lig_ue_mask]

        # pairwise distances on ground-truth
        x_1_true = g.nodes['lig'].data['x_1_true']
        x_diff_true = x_1_true[src_idxs] - x_1_true[dst_idxs]
        dij_true = _norm_no_nan(x_diff_true)
        
        # pairwise distances on predicted structure
        x_1_pred = dst_dict['lig_x']
        x_diff_gen = x_1_pred[src_idxs] - x_1_pred[dst_idxs]
        dij_pred = _norm_no_nan(x_diff_gen)

        # get a mask for distances < d_max
        d_mask = dij_true < self.d_max

        loss = nn.MSELoss()(dij_pred[d_mask], dij_true[d_mask])
        return loss
    
    def supports_task(self, task: Task) -> bool:
        """
        Check if this auxiliary loss supports the given task.
        This loss is only applicable to tasks that involve protein-ligand interactions.
        """
        return 'ligand_identity' in task.groups_present
