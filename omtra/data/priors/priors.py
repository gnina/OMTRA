from torch.distributions import Exponential
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn.functional import softmax, one_hot
import dgl

def gaussian(n: int, d: int, std: float = 1.0, simplex_center: bool = False):
    """
    Generate a prior feature by sampling from a Gaussian distribution.
    """
    p = torch.randn(n, d) * std
    
    if simplex_center:
        p = p + 1/d
    return p


def centered_normal_prior(n: int, d: int, std: float = 4.0):
    """
    Generate a prior feature by sampling from a centered normal distribution.
    """
    prior_feat = torch.randn(n, d) * std
    prior_feat = prior_feat - prior_feat.mean(dim=0, keepdim=True)
    return prior_feat

def centered_normal_prior_batched_graph(g: dgl.DGLGraph, node_batch_idx: torch.Tensor, std: float = 4.0):

    n = g.num_nodes()
    prior_sample = torch.randn(n, 3, device=g.device)
    with g.local_scope():
        g.ndata['prior_sample'] = prior_sample
        prior_sample = prior_sample - dgl.readout_nodes(g, feat='prior_sample', op='mean')[node_batch_idx]

    return prior_sample
    

def ctmc_masked_prior(n: int, d: int):
    """
    Sample from a CTMC masked prior. All samples are assigned the mask token at t=0.
    """
    p = torch.full((n,), fill_value=d)
    p = one_hot(p, num_classes=d+1).float()
    return p


