import torch
import torch.nn as nn
import numpy as np
from functools import lru_cache
import dgl
from typing import List, Optional
from collections import defaultdict
from omtra.tasks.tasks import Task
from omtra.tasks.modalities import (
    Modality, 
    name_to_modality
)
from omtra.utils.graph import g_local_scope
from omtra.models.gvp import _norm_no_nan
from omtra.data.graph.utils import get_upper_edge_mask

@lru_cache(maxsize=8)
def _load_marginals(marginal_path: str, key: str) -> torch.Tensor:
    """Load marginal probabilities from an .npz file (cached)."""
    data = np.load(marginal_path)
    return torch.from_numpy(data[key]).float()

class FixedConditionEmbedder(nn.Module):
    def __init__(self,
                 partial_modality_fixed_cls: List[Modality],
                 token_dim: int = 64,
                 n_hidden_scalars: int = 64,
                 n_hidden_edge_feats: int = 64,
                 fake_atoms: bool = True,
                 res_id_embed_dim: int = 64,
                 marginal_path: str = None,
                 marginal_keys: dict = None,
                 fixed_coord_max_std: float = 0.0,
                 fixed_coord_std: Optional[float] = None,
                 fixed_token_max_prob: float = 0.0,
                 fixed_token_prob: Optional[float] = None,
                 ):
        super().__init__()

        self.marginal_path = marginal_path
        self.marginal_keys = marginal_keys or {}

        self.fixed_coord_max_std = fixed_coord_max_std
        self.fixed_coord_std = fixed_coord_std

        self.fixed_token_max_prob = fixed_token_max_prob
        self.fixed_token_prob = fixed_token_prob

        # realistically this may be over engineered because we currently only fix ligand atoms, but trying to make this general for the case where we fix other modalities
        ntype_cat_feats = defaultdict(int)
        ntype_coord_feats = defaultdict(int)

        node_types = set(m.entity_name for m in partial_modality_fixed_cls if m.graph_entity == "node")
        node_types = sorted(list(node_types))

        edge_types = set(m.entity_name for m in partial_modality_fixed_cls if m.graph_entity == "edge")
        edge_types.update([f"{m.entity_name}_to_{m.entity_name}" for m in partial_modality_fixed_cls if m.data_key == "x"])
        edge_types = sorted(list(edge_types))

        # create embedders for tokens and raw coordinates
        self.token_embeddings = nn.ModuleDict()
        self.coord_embedding = nn.ModuleDict()

        for m in partial_modality_fixed_cls:
            if m.is_categorical:
                has_fake_atoms = (m.name == 'lig_a' or m.name == 'lig_cond_a') and fake_atoms 
                self.token_embeddings[m.name] = nn.Embedding(
                    m.n_categories + 1 + int(has_fake_atoms), token_dim # partially fixed modalities are always being generated
                )   
                if m.graph_entity == "node":
                    ntype_cat_feats[m.entity_name] += 1

            elif m.data_key == 'x':
                # TODO: should this be token_dim? not n_hidden_scalars/edge_feats
                # embedding for raw coordinates
                self.coord_embedding[m.entity_name] = nn.Sequential(
                    nn.Linear(3, token_dim),
                    nn.SiLU(),
                    nn.LayerNorm(token_dim),
                )
                # embedding for pairwise distances
                self.coord_embedding[f'{m.entity_name}_to_{m.entity_name}'] = nn.Sequential(
                    nn.Linear(1, token_dim),
                    nn.SiLU(),
                    nn.LayerNorm(token_dim),
                )
                ntype_coord_feats[m.entity_name] += 1

        # create scalar embedders
        self.scalar_embedding = nn.ModuleDict()
        for ntype in node_types:
            input_dim = (ntype_cat_feats[ntype] + ntype_coord_feats[ntype]) * token_dim 
            if res_id_embed_dim is not None and ntype == 'prot_atom':
                input_dim += res_id_embed_dim
            if (self.fixed_token_max_prob > 0.0) or (self.fixed_token_prob is not None):
                input_dim += ntype_cat_feats[ntype] 
            if (self.fixed_coord_max_std > 0.0) or (self.fixed_coord_std is not None):    # raw atom coordinates are injected into scalar features
                input_dim += ntype_coord_feats[ntype]

            self.scalar_embedding[ntype] = nn.Sequential(
                nn.Linear(
                    input_dim,
                    n_hidden_scalars,
                ),
                nn.SiLU(),
                nn.LayerNorm(n_hidden_scalars),
            )

        # create edge embedders
        self.edge_embedding = nn.ModuleDict()
        for etype in edge_types:
            input_dim = token_dim
            if (self.fixed_token_max_prob > 0.0) or (self.fixed_token_prob is not None):
                input_dim += 1
            if any((m.data_key == "x" and f"{m.entity_name}_to_{m.entity_name}" == etype) for m in partial_modality_fixed_cls):
                input_dim += token_dim
            self.edge_embedding[etype] = nn.Sequential(
                nn.Linear(input_dim, n_hidden_edge_feats),
                nn.SiLU(),
                nn.LayerNorm(n_hidden_edge_feats),
            )
    
    @g_local_scope
    def forward(self,
                g: dgl.DGLGraph,
                task_class: Task):

        node_modalities = [name_to_modality(m) for m in task_class.partial_modalities_fixed if name_to_modality(m).is_node]
        edge_modalities = [name_to_modality(m) for m in task_class.partial_modalities_fixed if not name_to_modality(m).is_node]

        fixed_node_scalar_features = {}
        fixed_edge_features = {}

        pair_dists = {}
        atom_masks = {}
        edge_masks = {}
        
        # Collect node embeddings
        for modality in node_modalities:
            ntype = modality.entity_name
            if g.num_nodes(ntype) == 0:
                continue
            
            if modality.is_categorical:
                if ntype not in fixed_node_scalar_features:
                    fixed_node_scalar_features[ntype] = []

                gt = g.nodes[ntype].data[f"{modality.data_key}_1_true"].clone()
                mask = g.nodes[ntype].data["atom_mask_1_true"].bool().clone()
                
                if ntype not in atom_masks.keys():
                    atom_masks[ntype] = mask

                if (self.fixed_token_max_prob > 0.0) or (self.fixed_token_prob is not None):
                    gt, prob = self.corrupt_tokens(gt, g.nodes[ntype].data[f"{modality.data_key}_0"], modality.name, mask)
                    fixed_node_scalar_features[ntype].append(prob.unsqueeze(-1))

                fixed_atom_feats = self.token_embeddings[modality.name](gt) * mask.unsqueeze(-1)
                fixed_node_scalar_features[ntype].append(fixed_atom_feats)

            elif modality.data_key == "x":
                if ntype not in fixed_node_scalar_features:
                    fixed_node_scalar_features[ntype] = []

                x = g.nodes[ntype].data["x_1_true"].clone() # ground truth positions
                mask = g.nodes[ntype].data["atom_mask_1_true"].bool().clone()

                if ntype not in atom_masks.keys():
                    atom_masks[ntype] = mask

                # noise coordinates of fixed atoms
                if (self.fixed_coord_max_std > 0.0) or (self.fixed_coord_std is not None):
                    x, sigma = self.corrupt_coords(x)
                    fixed_node_scalar_features[ntype].append(sigma)
                
                fixed_atom_coords = self.coord_embedding[modality.entity_name](x) * mask.unsqueeze(-1)
                fixed_node_scalar_features[ntype].append(fixed_atom_coords)
                
                # TODO: this is probably not how we want to specify that we want lig-to-lig distances for fixed atoms
                etype = f"{ntype}_to_{ntype}" 
                src, dst = g.edges(etype=etype)
                src = src.clone()   # this might be unecessary
                dst = dst.clone()   # this might be unecessary
                diff = x[src] - x[dst] 
                pair_dists[etype] = _norm_no_nan(diff, axis=-1, keepdims=True)

        # Collect edge embeddings
        for modality in edge_modalities:
            etype = modality.entity_name
            if etype not in fixed_edge_features:
                fixed_edge_features[etype] = []

            gt = g.edges[etype].data[f"{modality.data_key}_1_true"].clone()
            mask = g.edges[etype].data["edge_mask_1_true"].bool().clone()
            if etype not in edge_masks.keys():
                edge_masks[etype] = mask

            if (self.fixed_token_max_prob > 0.0) or (self.fixed_token_prob is not None):
                ue_mask = get_upper_edge_mask(g, etype)
                gt, prob = self.corrupt_tokens(gt, g.edges[etype].data[f"{modality.data_key}_0"], modality.name, mask, ue_mask)
                fixed_edge_features[etype].append(prob.unsqueeze(-1))
            
            fixed_edge_feats = self.token_embeddings[modality.name](gt) * mask.unsqueeze(-1)
            fixed_edge_features[etype].append(fixed_edge_feats)

            # pairwise distances
            if etype in pair_dists.keys():
                fixed_pair_dists = pair_dists[etype] * mask.unsqueeze(-1)
                fixed_pair_dists = self.coord_embedding[etype](fixed_pair_dists)
                fixed_edge_features[etype].append(fixed_pair_dists)
                        

        # Construct singular scalar embedding for fixed atoms
        for ntype in fixed_node_scalar_features.keys():
            cat_feats = torch.cat(
                fixed_node_scalar_features[ntype], dim=-1
            )
            fixed_node_scalar_features[ntype] = self.scalar_embedding[ntype](
                cat_feats
            ) * atom_masks[ntype].unsqueeze(-1)     
        
        # Construct singular scalar embedding for fixed atoms
        for etype in fixed_edge_features.keys():
            edge_feats = torch.cat(
                fixed_edge_features[etype], dim=-1
            )
            fixed_edge_features[etype] = self.edge_embedding[etype](
                edge_feats
            ) * edge_masks[etype].unsqueeze(-1)   

        return fixed_node_scalar_features, fixed_edge_features

    def corrupt_tokens(self,
                       gt: torch.Tensor,
                       g0: torch.Tensor,
                       modality_name: str,
                       valid_mask: torch.Tensor,
                       ue_mask: Optional[torch.Tensor] = None):
        """Corrupt discrete tokens for fixed atoms/edges.

        Args:
            gt: Ground truth token indices
            g0: Prior tokens (used to infer n_categories for uniform fallback)
            modality_name: Name of modality for marginal lookup
            valid_mask: Boolean mask indicating which tokens are valid (fixed atoms). Only valid tokens can be corrupted.
            ue_mask: upper edge mask to symmetrize edge corruption

        Returns:
            gt: Corrupted tokens
            prob: Per-token corruption probability used
        """
        if self.fixed_token_prob is not None:     # Use specific user-defined probability
            prob = torch.full((gt.shape[0],), self.fixed_token_prob, device=gt.device)
        else:   # Sample probability from U(0, fixed_token_max_prob)
            prob = torch.rand(gt.shape[0], device=gt.device) * self.fixed_token_max_prob

        # Only corrupt valid (fixed) tokens
        corrupt_mask = (torch.rand(gt.shape[0], device=gt.device) < prob) & valid_mask

        if corrupt_mask.any():
            n_corrupt = corrupt_mask.sum().item()

            marginal_key = self.marginal_keys.get(modality_name)
            if self.marginal_path is not None and marginal_key is not None:
                # Sample from data marginal
                marginal_probs = _load_marginals(self.marginal_path, marginal_key)
                marginal_probs = marginal_probs.to(gt.device)
                corrupt_samples = torch.multinomial(
                    marginal_probs,
                    num_samples=n_corrupt,
                    replacement=True,
                )
            else:
                # Sample uniformly
                mask_index = g0.max().item()
                n_categories = int(mask_index)
                corrupt_samples = torch.randint(
                    low=0,
                    high=n_categories,
                    size=(n_corrupt,),
                    device=gt.device,
                )
            gt[corrupt_mask] = corrupt_samples.to(gt.dtype)
        
        if ue_mask is not None:
            # Symmetrize edge corruption
            gt[~ue_mask] = gt[ue_mask]
            prob[~ue_mask] = prob[ue_mask]

        return gt, prob

    def corrupt_coords(self,
                       x: torch.Tensor):
        if self.fixed_coord_std is not None:    # Use specific user-defined sigma
            sigma = torch.full((x.shape[0], 1), self.fixed_coord_std, device=x.device)
        else:   # Sample sigma from U(0, fixed_coord_max_std)
            sigma = torch.rand((x.shape[0], 1), device=x.device) * self.fixed_coord_max_std    
        eps = torch.randn_like(x) * sigma   # sample episilon from N(0, sigma)
        x = x + eps                         # add noise to the true fixed atom positions
        return x, sigma