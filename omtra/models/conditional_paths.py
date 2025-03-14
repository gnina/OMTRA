import dgl
import torch

def sample_continuous_interpolant(
        x_0: torch.Tensor, 
        x_1: torch.Tensor, 
        alpha_t: torch.Tensor, # for each node/edge
        beta_t: torch.Tensor, # for each node/edge
        ue_mask: torch.Tensor = None
    ):
    x_t = alpha_t*x_0 + beta_t*x_1

    if ue_mask is not None:
        raise NotImplementedError("i didn't think we would model continuous edge features")
    
    return x_t

def sample_masked_ctmc(
        x_1: torch.Tensor,
        p_mask: torch.Tensor, # p_mask for each node/edge
        n_categories: int,
        ue_mask: torch.Tensor = None
    ):
    x_t = x_1.clone()
    x_t[ torch.rand_like(x_t) < p_mask ] = n_categories
    
    if ue_mask is not None:
        x_t[~ue_mask] = x_t[ue_mask]

    return x_t