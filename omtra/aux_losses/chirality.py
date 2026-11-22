"""Handedness auxiliary loss on the predicted ligand endpoint.

For three-heavy-neighbour centres, use centre-to-neighbour vectors. For four-heavy-neighbour
centres, use the tetrahedron of all four substituents. The loss is a logistic penalty on its
predicted sign against the crystal's sign. Neighbour order is consistent within each graph;
permuting it flips true and predicted signs together.

It is mirror-sensitive (a mirror image flips the sign) while the existing distance-based auxiliary
losses are not. It supplies a direct per-centre gradient for the handedness decision; it cannot add
information the pocket does not carry. It is not time-weighted: the handedness decision is made at
low t, where the 1/(1-t)^2 weights are small.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from typing import Dict
from omtra.aux_losses.register import register_aux_loss
from omtra.tasks.tasks import Task
from omtra.tasks.modalities import name_to_modality


def centre_neighbours(g: dgl.DGLHeteroGraph, include_fourth=False):
    """Return centres and ordered neighbours; optional fourth column is -1 for implicit-H centres."""
    nd = g.nodes['lig'].data
    # Find the stereocentres and their neighbours. Output: 'centres' (K,) and 'neighbours' (K, 3 or 4) with neighbours in ascending atom-index order;
    # a -1 in the fourth column marks a three-neighbour centre (the fourth bond is the implicit hydrogen).
    n = g.num_nodes('lig')
    device = nd['x_1_true'].device
    width = 4 if include_fourth else 3
    empty = (torch.zeros(0, dtype=torch.long, device=device), torch.zeros(0, width, dtype=torch.long, device=device))
    if 'stereo_centre' not in nd or not bool(nd['stereo_centre'].any()):
        return empty
    src, dst = g.edges(etype='lig_to_lig')
    covalent = g.edges['lig_to_lig'].data['e_1_true'] > 0
    # Keep only covalent bonds that START at a stereocentre: (src = centre, dst = neighbour).
    keep = covalent & nd['stereo_centre'].bool()[src]
    s, d = src[keep].long(), dst[keep].long()
    if s.numel() == 0:
        return empty
    # Sort the bonds by (centre, neighbour index) so that each centre's neighbours appear together, in ascending atom-index order.
    order = torch.argsort(s * (n + 1) + d)
    s, d = s[order], d[order]
    # degree = number of covalent neighbours per atom; start = where each centre's block begins; rank = position of a neighbour inside its centre's block.
    degree = torch.bincount(s, minlength=n)
    start = torch.cumsum(degree, 0) - degree
    rank = torch.arange(s.numel(), device=s.device) - start[s]
    # Supported centres have 3 or 4 neighbours; take the first three of each (the fourth, if any, is added below).
    ok = ((degree[s] == 3) | (degree[s] == 4)) & (rank < 3)
    s, d = s[ok], d[ok]
    if s.numel() == 0:
        return empty
    # After the filter every centre contributes exactly 3 consecutive rows: one centre per three rows, neighbours reshaped to (K, 3).
    centres, neighbours = s[::3], d.view(-1, 3)
    if include_fourth:
        # Four-neighbour centres: look up the 4th neighbour (rank 3); -1 stays for three-neighbour centres.
        fourth = torch.full_like(centres, -1)
        four = degree[centres] == 4
        # Recover all covalent neighbours in the same sorted edge order.
        all_dst = dst[keep].long()[order]
        fourth[four] = all_dst[start[centres[four]] + 3]
        neighbours = torch.cat((neighbours, fourth[:, None]), dim=1)
    return centres, neighbours


def signed_volume(x: torch.Tensor, centres: torch.Tensor, neighbours: torch.Tensor) -> torch.Tensor:
    """Normalized oriented volume, including the fourth substituent when supplied."""
    # Older (logistic) handedness measure, same idea as stereo_geometry.signed_volume3/4: three unit bond vectors from the centre, triple product.
    vec = x[neighbours[:, :3]] - x[centres].unsqueeze(1)
    unit = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    volume = (unit[:, 0] * torch.cross(unit[:, 1], unit[:, 2], dim=-1)).sum(-1)
    # Four-neighbour centres: use the volume of the tetrahedron of the four neighbours instead (centre not used), divided by size^3.
    if neighbours.shape[1] == 4:
        points = x[neighbours.clamp_min(0)]
        edges = points[:, :3] - points[:, 3:4]
        raw = (edges[:, 0] * torch.cross(edges[:, 1], edges[:, 2], dim=-1)).sum(-1)
        # Symmetric scale: changing which substituent is the anchor changes only
        # parity, never loss magnitude. Six edge lengths, each counted twice.
        pairwise = points[:, :, None, :] - points[:, None, :, :]
        mean_squared_length = pairwise.square().sum(dim=(-1, -2, -3)) / 12
        scale = mean_squared_length.clamp_min(1e-12).pow(1.5)
        volume = torch.where(neighbours[:, 3] >= 0, raw / scale, volume)
    return volume


@register_aux_loss(name='chirality')
class ChiralityLoss(nn.Module):

    def __init__(self, weight: float = 1.0, temperature: float = 0.25, min_true_volume: float = 0.05):
        super().__init__()
        self.weight = weight
        self.temperature = temperature
        self.min_true_volume = min_true_volume

    def forward(self,
                g: dgl.DGLHeteroGraph,
                dst_dict: Dict[str, torch.Tensor],
                task: Task,
                node_batch_idxs: Dict[str, torch.Tensor],
                lig_ue_mask: torch.Tensor,
                time_weights: torch.Tensor):
        pred = dst_dict['lig_x']
        centres, neighbours = centre_neighbours(g, include_fourth=True)
        if centres.numel() == 0:
            return pred.sum() * 0.0   # keeps the graph so backward yields finite zero gradients
        # Hand of each centre in the crystal (target) and, below, in the predicted pose. This is the OLDER logistic loss (chirality.py); the margin loss
        # (chirality_margin.py) is the one used in the P3 runs.
        true = signed_volume(g.nodes['lig'].data['x_1_true'], centres, neighbours)
        usable = true.abs() >= self.min_true_volume
        if not bool(usable.any()):
            return pred.sum() * 0.0
        guess = signed_volume(pred, centres, neighbours)
        loss = F.softplus(-torch.sign(true[usable]) * guess[usable] / self.temperature)
        return loss.mean() * self.weight

    def supports_task(self, task: Task) -> bool:
        fixed = list(getattr(task, 'groups_fixed', []))
        return ('ligand_identity_condensed' in fixed or 'ligand_identity' in fixed) \
            and name_to_modality('lig_x') in task.modalities_generated
