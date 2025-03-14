from torch.distributions import Exponential
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn.functional import softmax, one_hot
import dgl

from omtra.priors.register import register_train_prior, register_inference_prior
from omtra.priors.align import align_prior

# @register_inference_prior("gaussian")
@register_train_prior("gaussian")
def gaussian(x1: torch.Tensor, std: float = 1.0, ot=False):
    """
    Generate a prior feature by sampling from a Gaussian distribution.
    """

    n, d = x1.shape

    x0 = torch.randn(n, d) * std
    
    if ot:
        # move x0 to the same COM as x1
        x0_mean = x0.mean(dim=0, keepdim=True)
        x1_mean = x1.mean(dim=0, keepdim=True)
        x0 += x1_mean - x0_mean

        # align x0 to x1
        x0 = align_prior(x0, x1, rigid_body=True, permutation=True)

    return x0

@register_train_prior("centered-normal")
def centered_normal_prior(x1: torch.Tensor, std: float = 1.0):
    """
    Generate a prior feature by sampling from a centered normal distribution.
    """
    n, d = x1.shape
    prior_feat = torch.randn(n, d) * std
    prior_feat = prior_feat - prior_feat.mean(dim=0, keepdim=True)
    return prior_feat


@register_inference_prior("centered-normal")
def centered_normal_prior_batched_graph(g: dgl.DGLGraph, node_batch_idx: torch.Tensor, std: float = 1.0):
    raise NotImplementedError
    # TODO: implement this for a heterogeneous graph
    n = g.num_nodes()
    prior_sample = torch.randn(n, 3, device=g.device)
    with g.local_scope():
        g.ndata['prior_sample'] = prior_sample
        prior_sample = prior_sample - dgl.readout_nodes(g, feat='prior_sample', op='mean')[node_batch_idx]

    return prior_sample
    
@register_train_prior("masked")
@register_inference_prior("masked")
def ctmc_masked_prior(x1: torch.Tensor, n_categories: int):
    """
    Sample from a CTMC masked prior. All samples are assigned the mask token at t=0.
    """
    p = torch.ones_like(x1)*n_categories
    return p

@register_train_prior("fixed")
@register_inference_prior("fixed")
def fixed_prior(x1: torch.Tensor):
    """
    Generate a fixed prior feature.
    """
    return x1.clone()
