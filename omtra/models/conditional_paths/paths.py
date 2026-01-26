import dgl
import torch
import numpy as np
from pathlib import Path
from functools import lru_cache
from omtra.models.conditional_paths.path_register import register_conditional_path


@lru_cache(maxsize=8)
def _load_marginals(marginal_path: str, key: str) -> torch.Tensor:
    """Load marginal probabilities from an .npz file (cached)."""
    data = np.load(marginal_path)
    return torch.from_numpy(data[key]).float()


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

    # Return tuple for consistency with ctmc_mask (no corruption for continuous)
    return x_t, None


@register_conditional_path("ctmc_mask")
def sample_masked_ctmc(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    alpha_t: torch.Tensor,  # for each node/edge
    beta_t: torch.Tensor,  # for each node/edge
    ue_mask: torch.Tensor = None,
    noise_alpha: float = 0.0,
    marginal_path: str = None,
    marginal_key: str = None,
):
    """
    Masked CTMC conditional path with optional corruption (uniform or data-marginal).

    When noise_alpha=0 (default), this is the standard two-way path:
        - With probability (1-t): token is masked
        - With probability t: token stays as target

    When noise_alpha>0, this becomes a three-way path with noise:
        - With probability (1 - t - β_t/2): token is masked
        - With probability (t - β_t/2): token stays as target
        - With probability β_t: token is corrupted

    where β_t = noise_alpha * t * (1-t), peaking at t=0.5 and zero at boundaries.

    Corruption distribution:
        - Stage 1 (marginal_path=None): Uniform over vocabulary
        - Stage 2 (marginal_path provided): Sample from empirical data marginal

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
        marginal_path: Path to .npz file containing marginal distributions.
                      If provided with marginal_key, use data-marginal corruption.
        marginal_key: Key in the .npz file for the marginal distribution
                     (e.g., 'lig_cond_a_marginal' or 'lig_e_marginal').

    Returns:
        Tuple of:
            x_t: Corrupted tokens at time t
            is_corrupted: Boolean mask indicating which tokens were corrupted (None if noise_alpha=0)
    """
    alpha_t = alpha_t.squeeze(-1)
    beta_t = beta_t.squeeze(-1)

    if noise_alpha == 0.0:
        # Original behavior: simple two-way mask/target
        x_t = x_1.clone()
        mask = torch.rand_like(x_t.float()) < alpha_t
        x_t[mask] = x_0[mask]
        is_corrupted = None
    else:
        # Three-way path with corruption
        noise_t = noise_alpha * beta_t * alpha_t

        # Three-way probabilities
        prob_mask = alpha_t - noise_t / 2
        prob_target = beta_t - noise_t / 2

        rand = torch.rand_like(x_1.float())
        x_t = x_1.clone()

        # Tokens that get masked
        mask_tokens = rand < prob_mask
        x_t[mask_tokens] = x_0[mask_tokens]

        # Tokens that get corrupted (noise)
        corrupt_tokens = rand >= (prob_mask + prob_target)

        # Track which tokens are corrupted for the classification head
        is_corrupted = corrupt_tokens.clone()

        if corrupt_tokens.any():
            n_corrupt = corrupt_tokens.sum().item()

            if marginal_path is not None and marginal_key is not None:
                # Sample from data marginal
                marginal_probs = _load_marginals(marginal_path, marginal_key)
                marginal_probs = marginal_probs.to(x_1.device)
                corrupt_samples = torch.multinomial(
                    marginal_probs,
                    num_samples=n_corrupt,
                    replacement=True,
                )
            else:
                # Sample uniformly
                mask_index = x_0[corrupt_tokens].max().item()
                n_categories = int(mask_index)
                corrupt_samples = torch.randint(
                    low=0,
                    high=n_categories,
                    size=(n_corrupt,),
                    device=x_1.device,
                )

            x_t[corrupt_tokens] = corrupt_samples.to(x_1.dtype)

    if ue_mask is not None:
        x_t[~ue_mask] = x_t[ue_mask]
        # For edges, also symmetrize the corruption mask
        if is_corrupted is not None:
            is_corrupted[~ue_mask] = is_corrupted[ue_mask]

    return x_t, is_corrupted


