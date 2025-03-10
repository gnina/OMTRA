from torch.distributions import Exponential
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn.functional import softmax, one_hot
import dgl

def align_prior(prior_feat: torch.Tensor, dst_feat: torch.Tensor, permutation=False, rigid_body=False, n_alignments: int = 1):
    """
    Aligns a prior feature to a destination feature. 
    """
    for _ in range(n_alignments):
        if permutation:
            # solve assignment problem
            cost_mat = torch.cdist(dst_feat, prior_feat, p=2)
            _, prior_idx = linear_sum_assignment(cost_mat)

            # reorder prior to according to optimal assignment
            prior_feat = prior_feat[prior_idx]

        if rigid_body:
            # perform rigid alignment
            prior_feat = rigid_alignment(prior_feat, dst_feat)

    return prior_feat

def rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    # t = t.T

    A_aligned = (R.mm(A.T)).T + t

    return A_aligned


def edge_prior(upper_edge_mask: torch.Tensor, edge_prior_config: dict):

    n_upper_edges = upper_edge_mask.sum().item()
    prior_fn = train_prior_register[edge_prior_config['type']]
    upper_edge_prior = prior_fn(n_upper_edges, 5, **edge_prior_config['kwargs'])

    edge_prior = torch.zeros(upper_edge_mask.shape[0], upper_edge_prior.shape[1])
    edge_prior[upper_edge_mask] = upper_edge_prior
    edge_prior[~upper_edge_mask] = upper_edge_prior
    return edge_prior