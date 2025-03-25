import dgl
import torch
from omtra.models.conditional_paths.path_register import register_conditional_path


@register_conditional_path("continuous_interpolant")
def sample_continuous_interpolant(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    alpha_t: torch.Tensor,  # for each node/edge
    beta_t: torch.Tensor,  # for each node/edge
    ue_mask: torch.Tensor = None,
):
    x_t = alpha_t * x_0 + beta_t * x_1

    if ue_mask is not None:
        raise NotImplementedError(
            "i didn't think we would model continuous edge features"
        )

    return x_t


@register_conditional_path("ctmc_mask")
def sample_masked_ctmc(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    alpha_t: torch.Tensor,  # for each node/edge
    beta_t: torch.Tensor,  # for each node/edge
    n_categories: int,
    ue_mask: torch.Tensor = None,
):
    # raise NotImplementedError("need to alpha_t and beta_t conventions are set correctly, or design a better convention")
    x_t = x_1.clone()
    x_t[torch.rand_like(x_t.float()) < beta_t] = n_categories

    if ue_mask is not None:
        x_t[~ue_mask] = x_t[ue_mask]

    return x_t
