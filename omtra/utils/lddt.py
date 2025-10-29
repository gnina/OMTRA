"""
LDDT (Local Distance Difference Test) computation for confidence prediction.
"""
import dgl
import torch
import torch.nn.functional as F

from omtra.utils.layout import GraphLayout

"""
Code credit (Cremer et al.) for `_get_cross_pair_mask`, `_get_cross_distance_matrix`, `_compute_lddt`
--> https://github.com/jule-c/flowr_root/blob/be5d660d7dd09864cd56bc1af2d12435de3a3702/flowr/models/losses.py#L693
"""
def _get_cross_pair_mask(mask1, mask2):
    pair_mask = mask1.unsqueeze(2) * mask2.unsqueeze(1)  # [B, N, M]
    # no cleaning of diagonal since we assume mask1 and mask2 are different
    return pair_mask
    
def _get_cross_distance_matrix(coords1, coords2, mask1, mask2):
    coords1 = coords1 * mask1.unsqueeze(-1)
    coords2 = coords2 * mask2.unsqueeze(-1)
    dists = torch.cdist(coords1, coords2, p=2)  # [B, N, M]
    return dists

def _compute_lddt(
    coords_true_ligand,
    coords_pred_ligand,
    mask_ligand,
    coords_true_pocket,
    mask_pocket,
    cutoff: float = 12.0,
    dist_thresholds: list | None = None,
):
    """Compute lDDT loss based on (predicted) coordinates.
    This version uses continuous distances.
    See Section 4.3.1 in https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf

    In practice, assume N_ligand, N_pocket is considered
    """
    cross_pair_mask = _get_cross_pair_mask(
        mask_ligand, mask_pocket
    )  # [B, N_l, N_p]
    dist_true = _get_cross_distance_matrix(
        coords_true_ligand, coords_true_pocket, mask_ligand, mask_pocket
    )  # [B, N_l, N_p]
    dist_pred = _get_cross_distance_matrix(
        coords_pred_ligand, coords_true_pocket, mask_ligand, mask_pocket
    )  # [B, N_l, N_p]

    
    dist_to_consider = (dist_true < cutoff) * cross_pair_mask  # binary target mask <-- a matrix
    
    
    dist_loss = (
        torch.abs(dist_pred - dist_true) * cross_pair_mask
    )  # continuous loss

    
    lddts = []
    if dist_thresholds is None:
        dist_thresholds = [0.5, 1.0, 2.0, 4.0]
    lddts = [
        ((dist_loss < thresh) * cross_pair_mask).float()
        for thresh in dist_thresholds
    ]
    lddts = torch.stack(lddts, dim=-1)  # [B, N_l, N_p, num_bins]
    lddts = lddts.mean(dim=-1)  # Average over bins, # [B, N_l, N_p]
    # lddts \in (0, 1) with shape # [B, N_l, N_p]
    mask_has_no_match = (torch.sum(dist_to_consider, dim=-1) != 0).float()
    # normalize over last dimension (pocket atoms)
    lddts = torch.sum(dist_to_consider * lddts, dim=-1)  # [B, N_l]
    norm = 1.0 / (1e-10 + torch.sum(dist_to_consider, dim=-1))
    lddts = lddts * norm
    return lddts, mask_has_no_match

def compute_lddt(g: dgl.DGLHeteroGraph):
    layout = GraphLayout(g)
    padded_feats, masks = layout.graph_to_padded_sequence(g)

    coords_true_lig    = padded_feats['lig']['x_1_true']
    coords_true_pocket = padded_feats['prot_atom']['x_1_true']
    coords_pred_lig    = padded_feats['lig']['x_1_pred']

    mask_lig    = masks['lig']
    mask_pocket = masks['prot_atom']

    lddt_scores, mask_has_no_match = _compute_lddt(
      coords_true_ligand = coords_true_lig,
      coords_pred_ligand = coords_pred_lig,
      mask_ligand = mask_lig,
      coords_true_pocket = coords_true_pocket,
      mask_pocket = mask_pocket,
      cutoff = 12.0,
      dist_thresholds = [0.5, 1.0, 2.0, 4.0]
    )
    
    return lddt_scores
    