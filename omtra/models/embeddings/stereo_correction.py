"""Geometric stereo-correction vector features. Default OFF (VectorField flag `stereo_correction`).

For every stereo unit whose requested sign z_i is visible, E(x) = sum_i [max(0, m - z_i chi_i(x))/m]^2 (margin m = 0.2, chi from
omtra/aux_losses/stereo_geometry.py) and the correction force f = -dE/dx is averaged over the units an atom takes part in, bounded as
f / max(1, |f|), detached, and injected into the ligand vector channels through a learned linear projection
    v_c <- v_c + w_cur[b, c] * f(x_internal) + w_pred[b, c] * f(x1_pred_detached)
before EVERY message-passing block b. Both weight tables start at zero (exact function preservation); the gradient w.r.t. them is nonzero from the first
step because the downstream network and f are nonzero. Masked (hidden) units contribute nothing. It is a polar-vector feature (reflection-equivariant).
This does NOT move any output coordinate: it is an input feature, distinct from post-hoc repair.

Unit signs are read from node data `stereo_unit_z` (handedness sign of the requested configuration at the unit's centre, 0 elsewhere) multiplied by label
visibility (`stereo_atom` in 2..5). `stereo_unit_z` must be attached by the data path; a missing field on an identity-fixed task raises (no silent off).
"""
import torch
import torch.nn as nn

from omtra.aux_losses.stereo_geometry import graph_units, chi_tensor, margin_unit_loss


class StereoCorrection(nn.Module):

    def __init__(self, n_vec_channels: int, n_blocks: int, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        self.n_blocks = n_blocks
        self.record = False          # set by the trainer's pair audit: keep per-call force statistics (no effect on outputs)
        self.stats, self.coverage = [], {}
        # Learned weights that decide how strongly each vector channel of each message-passing block listens to the correction arrows.
        # w_current: arrows computed from the CURRENT (noisy) coordinates; w_pred: arrows computed from the previous predicted endpoint.
        # Both start at ZERO, so at initialisation the network computes exactly what it computed without the feature (nothing is disturbed).
        self.w_current = nn.Parameter(torch.zeros(n_blocks, n_vec_channels))
        self.w_pred = nn.Parameter(torch.zeros(n_blocks, n_vec_channels))

    @staticmethod
    def units(g):
        """Visible-unit context from the graph: centres, nbrs4, z (K,) with 0 for hidden units. None when there is nothing to correct."""
        # Build the per-molecule context once per forward pass: which centres have a visible target hand, and how many centres each atom belongs to.
        nd = g.nodes['lig'].data
        if 'stereo_unit_z' not in nd:
            raise ValueError('stereo_correction requires node data stereo_unit_z on identity-fixed tasks (attach it in the data path)')
        # All stereocentres of the ligand graph (degree 3 or 4) and their neighbours in ascending atom-index order.
        centres, nbrs = graph_units(g)
        if centres.numel() == 0:
            return None
        # 'shown' = the R/S/r/s label of this centre is visible to the model (a hidden label is stored as 'unspecified', code 1).
        shown = ((nd['stereo_atom'] >= 2) & (nd['stereo_atom'] <= 5)).to(torch.float32)[centres]
        # z = requested hand (+1 / -1) at each centre (from the supplied identity), forced to 0 where the label is hidden => no arrow there.
        z = nd['stereo_unit_z'].to(torch.float32)[centres] * shown
        if not bool((z != 0).any()):
            return None
        # Keep only centres with a visible target hand.
        keep = z != 0
        centres, nbrs, z = centres[keep], nbrs[keep], z[keep]
        # Every atom that takes part in a kept centre, as centre OR neighbour (-1 = missing fourth neighbour is dropped) ...
        atoms = torch.cat([centres, nbrs[nbrs >= 0]])
        # ... and how many centres each atom belongs to (an atom shared by two centres has count 2; its arrow is later averaged).
        count = torch.zeros(g.num_nodes('lig'), device=centres.device).index_add(0, atoms, torch.ones_like(atoms, dtype=torch.float32))
        return {'centres': centres, 'nbrs': nbrs, 'z': z, 'count': count.clamp_min(1.0)}

    def feature(self, x: torch.Tensor, ctx):
        """Bounded, detached correction force at coordinates x (N,3); zeros when there is no visible unit."""
        # No visible centre (or no labels at all): all-zero arrows, i.e. the feature is silent.
        if ctx is None:
            return torch.zeros_like(x, dtype=torch.float32)
        # Differentiate the penalty E(x) with respect to the ATOM COORDINATES (not the network weights). enable_grad so that this also works when
        # the caller runs without gradient tracking (no_grad, sampling) AND inference_mode(False): inside torch.inference_mode, enable_grad alone cannot
        # create a graph (audit finding 7). Inference tensors (the coordinates and the unit context) are cloned into ordinary tensors first, because autograd
        # cannot save inference tensors for backward. Numerics are unchanged; the result is detached below as before.
        with torch.inference_mode(False), torch.enable_grad():
            # Fresh copy of the coordinates that we are allowed to differentiate; detach() so nothing flows back into the network from here.
            xx = x.detach().clone().float().requires_grad_(True)
            c = {k: (v.clone() if torch.is_tensor(v) and v.is_inference() else v) for k, v in ctx.items()}
            # E = sum over visible centres of [max(0, 0.2 - z*chi) / 0.2]^2 : 0 for a centre that is clearly right, large for a flat or inverted one.
            energy = margin_unit_loss(chi_tensor(xx, c['centres'], c['nbrs']), c['z'], self.margin).sum()
            # grad[a] = how E changes if atom a moves; the arrow is the opposite direction (downhill), see the next line.
            (grad,) = torch.autograd.grad(energy, xx)
        # Arrow = -gradient (the direction that lowers the penalty fastest), averaged over the centres the atom belongs to.
        f = -grad / ctx['count'].unsqueeze(-1)
        # Cap the length at 1 (f / max(1, |f|)): longer arrows become unit length, so in practice the network reads the DIRECTION and an on/off signal.
        # detach(): the arrow is a plain input feature; no gradient is propagated through its computation.
        f = (f / f.norm(dim=-1, keepdim=True).clamp_min(1.0)).detach()
        # Optional bookkeeping used by the pair audit (mean / max arrow length, fraction of atoms with an arrow); does not change the output.
        if self.record:
            norms = f.norm(dim=-1)
            self.stats.append((float(norms.mean()), float(norms.max()), float((norms > 1e-6).float().mean())))
        return f

    def inject(self, vec: torch.Tensor, block: int, f_current: torch.Tensor, f_pred=None) -> torch.Tensor:
        """vec (N, C, 3) ligand vector features; returns a new tensor with the two projected feature vectors added."""
        # vec: (atoms, channels, 3) vector features of the ligand. Add the arrow of each atom to EVERY channel, scaled by that channel's learned weight for this block.
        out = vec + self.w_current[block][None, :, None].to(vec.dtype) * f_current[:, None, :].to(vec.dtype)
        # Second set of arrows, computed from the previous predicted endpoint; f_pred is None when there is no previous prediction yet.
        if f_pred is not None:
            out = out + self.w_pred[block][None, :, None].to(vec.dtype) * f_pred[:, None, :].to(vec.dtype)
        return out
