from torch.distributions import Exponential
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn.functional import softmax, one_hot
import dgl

from omtra.priors.register import register_train_prior, register_inference_prior
from omtra.priors.align import align_prior
from typing import Union, Tuple, List

@register_train_prior("gaussian")
def gaussian_train(x1: torch.Tensor, std: float = 1.0, ot=False, permutation=True):
    """
    Generate a prior feature by sampling from a Gaussian distribution.
    """
    x0 = torch.randn_like(x1) * std
    
    if ot:
        # move x0 to the same COM as x1
        x0_mean = x0.mean(dim=0, keepdim=True)
        x1_mean = x1.mean(dim=0, keepdim=True)
        x0 += x1_mean - x0_mean

        # align x0 to x1
        x0 = align_prior(x0, x1, rigid_body=True, permutation=permutation)

    return x0

@register_inference_prior("gaussian")
def gaussian_inference(n: int, d: Union[int, List[int]], std: float = 1.0, **kwargs):
    """
    Generate a prior feature by sampling from a Gaussian distribution.
    """
    if isinstance(d, int):
        x0 = torch.randn(n, d) * std
    else:
        x0 = torch.randn(n, *d) * std
    return x0
    
    
@register_train_prior("masked")
def ctmc_masked_train(x1: torch.Tensor, n_categories: int):
    """
    Sample from a CTMC masked prior. All samples are assigned the mask token at t=0.
    """
    n = x1.shape[0]
    p = ctmc_masked_inference(n, n_categories)
    p.to(x1.device)
    return p

@register_inference_prior("masked")
def ctmc_masked_inference(n: int, d: int):
    """
    Sample from a CTMC masked prior. All samples are assigned the mask token at t=0.
    """
    p = torch.ones(n, dtype=torch.long) * d
    return p

@register_train_prior("target_dependent_gaussian")
@register_inference_prior("target_dependent_gaussian")
def target_dependent_gaussian_prior(x1: torch.Tensor, std: float = 1.0):
    """
    Generate a target-dependent Gaussian prior feature.
    """
    x_0 = x1.clone() + torch.randn_like(x1) * std
    # TODO: adjust COM of x_0??
    return x_0


@register_train_prior("apo_exp")
@register_inference_prior("apo_exp")
def exp_prior(x0: torch.Tensor):
    """
    Generate a prior from unbound experimental structure
    """
    return x0.clone()

@register_train_prior("apo_pred")
@register_inference_prior("apo_pred")
def pred_prior(x0: torch.Tensor):
    """
    Generate a prior from AlphaFold predicted unbound structure
    """
    return x0.clone()
