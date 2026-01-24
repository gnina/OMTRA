# Noisy Paths: Training Robustness for Discrete Flow Matching

This document describes the noisy paths experiment, which addresses a train-test mismatch in discrete flow matching models by introducing structured corruption during training.

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

The noisy paths research plan introduces a three-way conditional path. Instead of the standard two-way path:

```
p_t(x | x_1) = t·δ_{x_1}(x) + (1-t)·δ_mask(x)
```

We use a three-way path:

```
p_t(x | x_1) = (t - β_t/2)·δ_{x_1}(x) + β_t·p_corrupt(x) + (1 - t - β_t/2)·δ_mask(x)
```

where:
- `β_t = α · t · (1-t)` with `α = 0.15` (configurable)
- `β_t` peaks at `t=0.5` and is zero at `t=0` and `t=1`
- `p_corrupt` is the corruption distribution (varies by stage)

This forces the denoiser to learn to correct incorrect unmasked tokens.

---

## Stage 1: Uniform Corruption

**Corruption distribution**: `p_corrupt = p_uniform` (uniform over vocabulary)

### Implementation

The `sample_masked_ctmc` function in `omtra/models/conditional_paths/paths.py` accepts a `noise_alpha` parameter:

```python
def sample_masked_ctmc(
    x_0, x_1, alpha_t, beta_t,
    noise_alpha: float = 0.0,  # Stage 1: controls uniform corruption
):
```

### Configuration

Use `configs/model/conditional_paths/noisy.yaml`:

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
```

### Usage

```bash
python routines/train.py \
    model/conditional_paths=noisy \
    name=stage1_uniform \
    task_group=pharmit5050_cond_a \
    max_steps=200000
```

---

## Stage 2: Data-Marginal Corruption

**Corruption distribution**: `p_corrupt = p_data` (empirical marginal from training data)

### Rationale

Inference-time denoiser errors are not uniform—they are biased toward frequent, plausible tokens. Corrupting with the data marginal better approximates the structure of realistic denoiser mistakes while remaining model-independent.

### Step 1: Compute Marginals

First, compute the marginal distributions from the training data:

```bash
python omtra_pipelines/compute_marginals/compute_marginals.py \
    --pharmit_path data/pharmit \
    --split train \
    --output_path data/pharmit/train_marginals.npz
```

This creates a `.npz` file containing:
- `lig_cond_a_marginal`: probability distribution over condensed atom types
- `lig_e_marginal`: probability distribution over bond types (accounting for sparse storage)

**Note on sparse edge storage**: The Pharmit zarr store only stores non-zero bond orders. The pipeline infers the number of "no bond" (type 0) edges by computing `total_possible_edges - stored_edges` for each molecule.

### Step 2: Train with Marginal Corruption

The `sample_masked_ctmc` function accepts `marginal_path` and `marginal_key` parameters:

```python
def sample_masked_ctmc(
    x_0, x_1, alpha_t, beta_t,
    noise_alpha: float = 0.0,
    marginal_path: str = None,    # Stage 2: path to marginals .npz
    marginal_key: str = None,     # Stage 2: key in .npz file
):
```

### Configuration

Use `configs/model/conditional_paths/noisy_marginal.yaml`:

```yaml
denovo_ligand_condensed:
  lig_cond_a:
    type: ctmc_mask
    params:
      noise_alpha: 0.15
      marginal_path: ${pharmit_path}/train_marginals.npz
      marginal_key: lig_cond_a_marginal
  lig_e_condensed:
    type: ctmc_mask
    params:
      noise_alpha: 0.15
      marginal_path: ${pharmit_path}/train_marginals.npz
      marginal_key: lig_e_marginal
```

### Usage

```bash
python routines/train.py \
    model/conditional_paths=noisy_marginal \
    name=stage2_marginal \
    task_group=pharmit5050_cond_a \
    max_steps=200000
```

---

## Training Objective

**Unchanged for both stages.** The model is still trained as a conditional denoiser:

```
L = E_{x_1, t, x_t} [ -Σ_i log p_θ(x_1^i | x_t, t) ]
```

The corruption component acts purely as structured noise during training. No auxiliary losses or additional heads are required.

## Inference

**Unchanged for both stages.** Existing heuristic re-masking and confidence-based unmasking are retained. The goal is to improve the denoiser's robustness, not to modify sampling.

---

## Testing

Unit tests verify both stages:

```bash
pytest tests/unit/test_noisy_paths.py -v
```

Tests cover:
- Stage 1: uniform corruption behavior, boundary conditions, probability distributions
- Stage 2: marginal loading, sampling from marginal distribution, difference from uniform

---

## Research Plan Context

| Stage | Description | Corruption Distribution | Status |
|-------|-------------|------------------------|--------|
| 1 | Uniform corruption | `p_uniform` | **Implemented** |
| 2 | Data-marginal corruption | `p_data` (empirical marginal) | **Implemented** |
| 3 | Model-induced corruption | `p_θ` (sample from denoiser) | Planned |
| 4 | Corruption classification head | Add auxiliary head | Planned |
| 5 | Modified sampling | Follow three-way path at inference | Planned |

Each stage builds on the previous, progressively closing the gap between training-time and inference-time state distributions.

## Expected Outcomes

This experiment tests whether:
1. Training robustness to incorrect unmasked tokens improves denoising quality
2. Data-marginal corruption (Stage 2) outperforms uniform corruption (Stage 1)
3. Model-induced corruption (Stage 3) further improves over data-marginal
4. The improvement reduces the need for heuristic re-masking at inference time

## Files

| File | Description |
|------|-------------|
| `omtra/models/conditional_paths/paths.py` | Core three-way path implementation |
| `configs/model/conditional_paths/noisy.yaml` | Stage 1 config (uniform) |
| `configs/model/conditional_paths/noisy_marginal.yaml` | Stage 2 config (data-marginal) |
| `omtra_pipelines/compute_marginals/compute_marginals.py` | Pipeline to compute marginals |
| `tests/unit/test_noisy_paths.py` | Unit tests |
| `noisy_paths.zip` | Research plan with mathematical derivations |

## References

- Research plan: `noisy_paths.zip` (LaTeX document with full mathematical derivation)
- Related work: Discrete flow matching, exposure bias in autoregressive models, scheduled sampling
