"""Prepare conditioning once, preserving *_1_true as evaluation/loss references.

*_cond is the supplied observation. Inference may provide it explicitly; otherwise
clean observations are copied from the reference. Noise is never resampled by the
vector field during integration.
"""
import math
import torch
from omtra.tasks.modalities import name_to_modality
from omtra.data.graph.utils import get_upper_edge_mask


def check_noise(value, name, probability=False):
    if not math.isfinite(float(value)) or value < 0 or (probability and value > 1):
        raise ValueError(f'{name} must be finite and in ' + ('[0, 1]' if probability else '[0, infinity)'))


def observed(data, key):
    return data[f'{key}_cond'] if f'{key}_cond' in data else data[f'{key}_1_true']


def prepare_coordinate_condition(g, ntype, max_std=0.0, resample=False, uncertainty=None,
                                 noise_mode='atom', generator=None):
    """Add input noise only. max_std samples sigma per point, not displacement."""
    check_noise(max_std, 'max_std')
    if ntype not in g.ntypes or g.num_nodes(ntype) == 0:
        return
    data = g.nodes[ntype].data
    supplied = 'x_cond' in data and not resample
    if resample or 'x_cond' not in data:
        data['x_cond'] = data['x_1_true'].clone()
        data['x_cond_std'] = torch.zeros((g.num_nodes(ntype), 1), device=g.device)
    if noise_mode not in ('atom', 'residue'):
        raise ValueError('noise_mode must be atom or residue')
    if max_std > 0 and not supplied:
        n = g.num_nodes(ntype)
        inverse = None
        if noise_mode == 'residue':
            if not all(k in data for k in ('chain_id', 'res_id')):
                raise ValueError('Residue noise requires chain and residue identifiers')
            batch = torch.repeat_interleave(torch.arange(g.batch_size, device=g.device), g.batch_num_nodes(ntype))
            keys = torch.stack([batch, data['chain_id'], data['res_id']], dim=-1)
            groups, inverse = torch.unique(keys, dim=0, return_inverse=True)
            n = len(groups)
        sigma = torch.rand((n, 1), device=g.device, generator=generator) * max_std
        noise = torch.randn((n, 3), device=g.device, generator=generator) * sigma
        if inverse is not None:
            sigma, noise = sigma[inverse], noise[inverse]
        data['x_cond'] = data['x_cond'] + noise
        data['x_cond_std'] = sigma
    if uncertainty is not None:
        if max_std > 0:
            raise ValueError('Specify simulated coordinate noise OR supplied uncertainty, not both')
        sigma = torch.as_tensor(uncertainty, device=g.device, dtype=data['x_cond'].dtype)
        if sigma.numel() == 1:
            sigma = sigma.expand(g.num_nodes(ntype), 1)
        else:
            sigma = sigma.reshape(g.num_nodes(ntype), 1)
        if not torch.isfinite(sigma).all() or (sigma < 0).any():
            raise ValueError('Coordinate uncertainty must be finite and nonnegative')
        data['x_cond_std'] = sigma
    if 'x_cond_std' not in data:
        data['x_cond_std'] = torch.zeros((g.num_nodes(ntype), 1), device=g.device)
    data['x_t'] = data['x_cond'].clone()
    feature = 'pharm_pos_std' if ntype == 'pharm' else 'prot_pos_std'
    data[feature] = data['x_cond_std']


def prepare_partial_condition(g, task, coord_max_std=0.0, token_max_prob=0.0, resample=False):
    check_noise(coord_max_std, 'fixed_coord_max_std')
    check_noise(token_max_prob, 'fixed_token_max_prob', probability=True)
    for name in task.partial_modalities_fixed:
        m = name_to_modality(name)
        data = (g.nodes[m.entity_name] if m.is_node else g.edges[m.entity_name]).data
        mask = data['atom_mask_1_true' if m.is_node else 'edge_mask_1_true'].bool()
        key = m.data_key
        if resample or f'{key}_cond' not in data:
            # Unrevealed reference information must not enter the conditioning.
            val = data[f'{key}_1_true'].clone()
            val[~mask] = 0
            data[f'{key}_cond'] = val
            data[f'{key}_cond_noise'] = torch.zeros(len(mask), device=g.device)
        val = data[f'{key}_cond'].clone()
        if key == 'x' and coord_max_std > 0:
            sigma = torch.rand(len(mask), device=g.device) * coord_max_std * mask
            val += torch.randn_like(val) * sigma[:, None]
            data[f'{key}_cond_noise'] = sigma
        elif m.is_categorical and token_max_prob > 0:
            prob = torch.rand(len(mask), device=g.device) * token_max_prob * mask
            replace = (torch.rand(len(mask), device=g.device) < prob) & mask
            # Sample actual categories, excluding fake and absorbing-mask tokens.
            replacements = torch.randint(m.n_categories, val.shape, device=g.device)
            val[replace] = replacements[replace]
            if not m.is_node:
                # The graph builder stores reverse ligand edges in matching order.
                upper = get_upper_edge_mask(g, m.entity_name)
                val[~upper] = val[upper]
                prob[~upper] = prob[upper]
            data[f'{key}_cond_noise'] = prob
        if f'{key}_cond_noise' not in data:
            data[f'{key}_cond_noise'] = torch.zeros(len(mask), device=g.device)
        val[~mask] = 0
        data[f'{key}_cond'] = val
