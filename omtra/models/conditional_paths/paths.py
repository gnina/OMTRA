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
    if x_0.ndim == 3: # for pharmacophore vectors
        alpha_t = alpha_t.unsqueeze(-1)
        beta_t = beta_t.unsqueeze(-1)
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
    ue_mask: torch.Tensor = None,
    noise_alpha: float = 0.0,
):
    """
    Masked CTMC conditional path with optional uniform corruption.

    When noise_alpha=0 (default), this is the standard two-way path:
        - With probability (1-t): token is masked
        - With probability t: token stays as target

    When noise_alpha>0, this becomes a three-way path with uniform noise:
        - With probability (1 - t - β_t/2): token is masked
        - With probability (t - β_t/2): token stays as target
        - With probability β_t: token is uniformly corrupted

    where β_t = noise_alpha * t * (1-t), peaking at t=0.5 and zero at boundaries.

    This trains the model to be robust to incorrect unmasked tokens, addressing
    the train-test gap where inference-time states include denoiser errors.

    Args:
        x_0: Source tokens (mask tokens)
        x_1: Target tokens (ground truth)
        alpha_t: Interpolation weight = 1-t
        beta_t: Interpolation weight = t
        ue_mask: Upper edge mask for symmetric edge handling
        noise_alpha: Controls noise magnitude. Default 0 (no noise).
                     Typical experimental value: 0.15.

    Returns:
        x_t: Corrupted tokens at time t
    """
    alpha_t = alpha_t.squeeze(-1)
    beta_t = beta_t.squeeze(-1)

    if noise_alpha == 0.0:
        # Original behavior: simple two-way mask/target
        x_t = x_1.clone()
        mask = torch.rand_like(x_t.float()) < alpha_t
        x_t[mask] = x_0[mask]
    else:
        # Three-way path with uniform corruption
        noise_t = noise_alpha * beta_t * alpha_t

        # Three-way probabilities
        prob_mask = alpha_t - noise_t / 2
        prob_target = beta_t - noise_t / 2

        rand = torch.rand_like(x_1.float())
        x_t = x_1.clone()

        # Tokens that get masked
        mask_tokens = rand < prob_mask
        x_t[mask_tokens] = x_0[mask_tokens]

        # Tokens that get uniform noise
        uniform_tokens = rand >= (prob_mask + prob_target)
        if uniform_tokens.any():
            # x_0 contains the mask index = n_categories
            mask_index = x_0[uniform_tokens].max().item()
            n_categories = int(mask_index)
            uniform_samples = torch.randint(
                low=0,
                high=n_categories,
                size=(uniform_tokens.sum().item(),),
                device=x_1.device,
                dtype=x_1.dtype
            )
            x_t[uniform_tokens] = uniform_samples

    if ue_mask is not None:
        x_t[~ue_mask] = x_t[ue_mask]

    return x_t


