"""Margin handedness loss. Default OFF (not in any config).

Per stereo unit the signed volume chi (stereo_geometry.py) of the predicted endpoint must exceed the margin on the side of the crystal
sign z: L_i = [max(0, m - z_i chi_i)/m]^2, m = 0.2. Losses are averaged over the eligible units of each molecule and then over molecules, so molecules with many
centres do not dominate. Nearly planar targets (|chi_true| < 0.05) are excluded from this loss only. Unlike `chirality.py` (logistic, never saturates, mean over
all centres of the batch) the hinge stops pushing once a centre is well formed on the correct side. Tetrahedral centres only; E/Z units are not implemented.
Partial tasks: fixed atoms are not moved by the loss (the predicted endpoint of fixed atoms is the supplied coordinate in the assembled endpoint).
"""
from typing import Dict
import torch
import torch.nn as nn
import dgl

from omtra.aux_losses.register import register_aux_loss
from omtra.aux_losses.stereo_geometry import graph_units, chi_tensor, margin_unit_loss, PLANAR_MIN
from omtra.tasks.tasks import Task
from omtra.tasks.modalities import name_to_modality


@register_aux_loss(name='chirality_margin')
class ChiralityMarginLoss(nn.Module):

    def __init__(self, weight: float = 0.3, margin: float = 0.2, min_true_volume: float = PLANAR_MIN):
        super().__init__()
        self.weight, self.margin, self.min_true_volume = weight, margin, min_true_volume

    def forward(self, g: dgl.DGLHeteroGraph, dst_dict: Dict[str, torch.Tensor], task: Task,
                node_batch_idxs: Dict[str, torch.Tensor], lig_ue_mask: torch.Tensor, time_weights: torch.Tensor):
        # Training-time loss on the network's OUTPUT. pred = predicted ligand coordinates (the endpoint x1 estimate).
        pred = dst_dict['lig_x']
        # All stereocentres of the ligand(s) in the batch and their neighbours (ascending atom index, as everywhere else).
        centres, nbrs = graph_units(g)
        if centres.numel() == 0:
            return pred.sum() * 0.0
        # Hand of each centre in the CRYSTAL pose (the training target). Note: this does not depend on whether the label is hidden from the model.
        true = chi_tensor(g.nodes['lig'].data['x_1_true'].float(), centres, nbrs)
        # Skip centres that are nearly flat in the crystal (|chi| < 0.05): their sign is not reliable.
        usable = true.abs() >= self.min_true_volume
        if not bool(usable.any()):
            return pred.sum() * 0.0
        # Keep only the usable centres.
        centres, nbrs, true = centres[usable], nbrs[usable], true[usable]
        # Hand of each centre in the PREDICTED pose; this is what the loss pushes (gradient flows back through the network weights).
        chi = chi_tensor(pred.float(), centres, nbrs)
        # Penalty per centre: [max(0, 0.2 - z*chi) / 0.2]^2 with z = sign of the crystal's chi. 0 once the centre is clearly right, 1 when flat, large when inverted.
        per_unit = margin_unit_loss(chi, torch.sign(true), self.margin)
        # Average the penalty inside each molecule first, then over molecules, so a molecule with many centres does not dominate the batch.
        # mol = which molecule of the batch each centre belongs to.
        mol = node_batch_idxs['lig'][centres]
        n_mol = int(g.batch_size)
        # Per molecule: sum of the penalties and number of usable centres.
        sums = torch.zeros(n_mol, device=per_unit.device, dtype=per_unit.dtype).index_add(0, mol, per_unit)
        counts = torch.zeros(n_mol, device=per_unit.device, dtype=per_unit.dtype).index_add(0, mol, torch.ones_like(per_unit))
        # Molecules without any usable centre are left out of the mean.
        has = counts > 0
        # Mean over molecules of the per-molecule mean penalty, times the loss weight (0.3).
        return (sums[has] / counts[has]).mean() * self.weight

    def supports_task(self, task: Task) -> bool:
        fixed = list(getattr(task, 'groups_fixed', []))
        return ('ligand_identity_condensed' in fixed or 'ligand_identity' in fixed) \
            and name_to_modality('lig_x') in task.modalities_generated
