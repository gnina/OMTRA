"""Soft partial conditioning with invariant scalars and equivariant vectors.

Observations are prepared outside forward, so a trajectory sees one stable input.
Clean reference values are never read here. Relative vectors avoid the historical
raw-xyz-to-scalar embedding's dependence on global rotation/translation.
"""
import torch
from torch import nn
from omtra.tasks.modalities import name_to_modality


class FixedConditionEmbedder(nn.Module):
    def __init__(self, modality_names, n_hidden_scalars, n_hidden_edge_feats, n_vec_channels):
        super().__init__()
        self.scalar = nn.ModuleDict()
        self.tokens = nn.ModuleDict()
        self.vector_weights = nn.ParameterDict()
        for name in sorted(set(modality_names)):
            m = name_to_modality(name)
            width = n_hidden_scalars if m.is_node else n_hidden_edge_feats
            if m.is_categorical:
                self.tokens[name] = nn.Embedding(m.n_categories, width)
                self.scalar[name] = nn.Linear(2, width)  # supplied flag, uncertainty
            elif m.data_key == 'x':
                self.scalar[name] = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
                self.vector_weights[name] = nn.Parameter(torch.ones(n_vec_channels) / n_vec_channels)

    def forward(self, g, task):
        nodes, edges, vectors = {}, {}, {}
        for name in task.partial_modalities_fixed:
            m = name_to_modality(name)
            data = (g.nodes[m.entity_name] if m.is_node else g.edges[m.entity_name]).data
            mask = data['atom_mask_1_true' if m.is_node else 'edge_mask_1_true'].bool()
            value = data[f'{m.data_key}_cond']
            noise = data[f'{m.data_key}_cond_noise']
            if m.is_categorical:
                # Unknown/fake rows are masked before looking up real category tokens.
                value = torch.where(mask, value, torch.zeros_like(value))
                feats = self.tokens[name](value) + self.scalar[name](torch.stack([mask.float(), noise], -1))
            else:
                delta = (value - data['x_t']) * mask[:, None]
                distance = torch.linalg.vector_norm(delta, dim=-1)
                feats = self.scalar[name](torch.stack([mask.float(), noise, distance], -1))
                vectors[m.entity_name] = delta[:, None, :] * self.vector_weights[name][None, :, None]
            feats = feats * mask[:, None]
            target = nodes if m.is_node else edges
            target[m.entity_name] = target.get(m.entity_name, 0) + feats
        return nodes, edges, vectors
