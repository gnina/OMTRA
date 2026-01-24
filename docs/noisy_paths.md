# Noisy Paths: Training Robustness for Discrete Flow Matching

This document describes the noisy paths experiment, which addresses a train-test mismatch in discrete flow matching models by introducing uniform corruption during training.

## Problem: Train-Test Mismatch

In discrete flow matching with masked priors, we observe two phenomena at inference time:

1. **Re-masking improves quality**: Allowing tokens to be re-masked during sampling substantially improves sample quality.
2. **Confidence-based unmasking helps**: Unmasking tokens based on model confidence further improves performance.

These observations suggest that during inference, the denoiser operates on **partially incorrect unmasked states**—tokens that are not masked but are wrong. Such states arise from denoiser imperfections and heuristic sampling dynamics.

However, under standard masked corruption, the denoiser is trained only on mixtures of:
- Correct tokens (from the target)
- Mask tokens (from the source)

It never sees incorrect-but-unmasked tokens. This is a form of **exposure bias**: the inference-time state distribution lies outside the training-time corruption distribution.

## Solution: Three-Way Conditional Path

Stage 1 of the noisy paths research plan introduces uniform corruption during training. Instead of the standard two-way path:

```
p_t(x | x_1) = t·δ_{x_1}(x) + (1-t)·δ_mask(x)
```

We use a three-way path:

```
p_t(x | x_1) = (t - β_t/2)·δ_{x_1}(x) + β_t·p_uniform(x) + (1 - t - β_t/2)·δ_mask(x)
```

where:
- `β_t = α · t · (1-t)` with `α = 0.15` (configurable)
- `β_t` peaks at `t=0.5` and is zero at `t=0` and `t=1`
- At `t=0.5` with `α=0.15`: approximately 3.75% of tokens are uniformly corrupted

This forces the denoiser to learn to correct incorrect unmasked tokens, improving robustness to the off-manifold states encountered during inference.

## Implementation

### Core Change: `omtra/models/conditional_paths/paths.py`

The `sample_masked_ctmc` function now accepts a `noise_alpha` parameter:

```python
def sample_masked_ctmc(
    x_0: torch.Tensor,      # Source tokens (mask tokens)
    x_1: torch.Tensor,      # Target tokens (ground truth)
    alpha_t: torch.Tensor,  # = 1-t
    beta_t: torch.Tensor,   # = t
    ue_mask: torch.Tensor = None,
    noise_alpha: float = 0.0,  # NEW: controls uniform corruption
):
```

When `noise_alpha=0` (default), behavior is unchanged. When `noise_alpha>0`:

1. Compute `noise_t = noise_alpha * t * (1-t)`
2. Sample each token from the three-way distribution:
   - Probability `(1-t) - noise_t/2`: masked
   - Probability `t - noise_t/2`: correct target
   - Probability `noise_t`: uniform random from vocabulary

### Configuration: `configs/model/conditional_paths/noisy.yaml`

```yaml
denovo_ligand_condensed:
  lig_cond_a:
    type: ctmc_mask
    params:
      noise_alpha: 0.15
  lig_e_condensed:
    type: ctmc_mask
    params:
      noise_alpha: 0.15

# Similar for other de novo tasks...
```

### Training Objective

**Unchanged.** The model is still trained as a conditional denoiser:

```
L = E_{x_1, t, x_t} [ -Σ_i log p_θ(x_1^i | x_t, t) ]
```

The uniform component acts purely as structured corruption during training. No auxiliary losses or additional heads are required.

### Inference

**Unchanged.** Existing heuristic re-masking and confidence-based unmasking are retained. The goal of Stage 1 is to improve the denoiser's robustness, not to modify sampling.

## Usage

To train with noisy paths:

```bash
python routines/train.py \
    model/conditional_paths=noisy \
    name=noisy_paths_experiment \
    task_group=pharmit5050_cond_a \
    max_steps=200000
```

To compare against baseline (no noise):

```bash
python routines/train.py \
    model/conditional_paths=default \
    name=baseline_experiment \
    task_group=pharmit5050_cond_a \
    max_steps=200000
```

## Testing

Unit tests verify the implementation:

```bash
pytest tests/unit/test_noisy_paths.py -v
```

Tests cover:
- `noise_alpha=0` matches original two-way behavior
- `noise_alpha>0` introduces uniform corruption
- Noise is zero at `t=0` and `t=1` boundaries
- Uniform samples are in valid vocabulary range
- Probability distribution matches theoretical values

## Research Plan Context

This is **Stage 1** of a 5-stage research plan:

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Uniform corruption of unmasked states | **Implemented** |
| 2 | Data-marginal corruption (use empirical token distribution) | Planned |
| 3 | Model-induced corruption (sample from current denoiser) | Planned |
| 4 | Joint denoising + corruption classification head | Planned |
| 5 | Modify sampling to follow three-way marginal path | Planned |

Each stage builds on the previous, with Stage 1 being the simplest intervention that still addresses the core train-test mismatch.

## Expected Outcomes

This experiment tests whether:
1. Training robustness to incorrect unmasked tokens improves denoising quality
2. Uniform corruption is sufficient, or if data-marginal (Stage 2) or model-induced (Stage 3) corruption is needed
3. The improvement reduces the need for heuristic re-masking at inference time

## References

- Research plan: `noisy_paths.zip` (LaTeX document with full mathematical derivation)
- Related work: Discrete flow matching, exposure bias in autoregressive models, scheduled sampling
