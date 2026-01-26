"""
Tests for the noisy paths implementation.
- Stage 1: uniform corruption
- Stage 2: data-marginal corruption
"""
import torch
import pytest
import tempfile
import numpy as np
from pathlib import Path
from omtra.models.conditional_paths.paths import sample_masked_ctmc, _load_marginals


class TestNoisyPathsStage1:
    """Tests for Stage 1: three-way conditional path with uniform corruption."""

    def test_no_noise_matches_original_behavior(self):
        """With noise_alpha=0, behavior should match original two-way path."""
        torch.manual_seed(42)

        n_tokens = 1000
        n_categories = 10
        mask_index = n_categories  # mask token is at index n_categories

        x_0 = torch.full((n_tokens,), mask_index, dtype=torch.long)  # all mask tokens
        x_1 = torch.randint(0, n_categories, (n_tokens,))  # random target tokens

        # At t=0.5: alpha_t = 0.5, beta_t = 0.5
        t = 0.5
        alpha_t = torch.full((n_tokens, 1), 1 - t)
        beta_t = torch.full((n_tokens, 1), t)

        # Run with no noise
        x_t, _ = sample_masked_ctmc(x_0, x_1, alpha_t, beta_t, noise_alpha=0.0)

        # Check that tokens are either mask or target
        is_mask = x_t == mask_index
        is_target = x_t == x_1
        assert (is_mask | is_target).all(), "All tokens should be either mask or target"

        # Approximately 50% should be masked at t=0.5
        mask_frac = is_mask.float().mean().item()
        assert 0.45 < mask_frac < 0.55, f"Expected ~50% masked, got {mask_frac:.2%}"

    def test_noise_introduces_uniform_corruption(self):
        """With noise_alpha>0, some tokens should be uniformly corrupted."""
        torch.manual_seed(42)

        n_tokens = 10000
        n_categories = 10
        mask_index = n_categories

        x_0 = torch.full((n_tokens,), mask_index, dtype=torch.long)
        x_1 = torch.zeros(n_tokens, dtype=torch.long)  # all target tokens are 0

        # At t=0.5: noise_t = 0.15 * 0.5 * 0.5 = 0.0375 (3.75% uniform)
        t = 0.5
        alpha_t = torch.full((n_tokens, 1), 1 - t)
        beta_t = torch.full((n_tokens, 1), t)
        noise_alpha = 0.15

        x_t, _ = sample_masked_ctmc(x_0, x_1, alpha_t, beta_t, noise_alpha=noise_alpha)

        # Check that some tokens are neither mask nor target (i.e., uniform corrupted)
        is_mask = x_t == mask_index
        is_target = x_t == x_1
        is_uniform = ~is_mask & ~is_target

        # Expected: ~3.75% uniform at t=0.5 with noise_alpha=0.15
        uniform_frac = is_uniform.float().mean().item()
        assert 0.02 < uniform_frac < 0.06, f"Expected ~3.75% uniform, got {uniform_frac:.2%}"

    def test_noise_is_zero_at_boundaries(self):
        """At t=0 and t=1, noise_t should be zero."""
        torch.manual_seed(42)

        n_tokens = 1000
        n_categories = 10
        mask_index = n_categories

        x_0 = torch.full((n_tokens,), mask_index, dtype=torch.long)
        x_1 = torch.zeros(n_tokens, dtype=torch.long)

        noise_alpha = 0.15

        # At t=0: all should be masked
        alpha_t = torch.ones((n_tokens, 1))
        beta_t = torch.zeros((n_tokens, 1))
        x_t, _ = sample_masked_ctmc(x_0, x_1, alpha_t, beta_t, noise_alpha=noise_alpha)
        assert (x_t == mask_index).all(), "At t=0, all tokens should be masked"

        # At t=1: all should be target
        alpha_t = torch.zeros((n_tokens, 1))
        beta_t = torch.ones((n_tokens, 1))
        x_t, _ = sample_masked_ctmc(x_0, x_1, alpha_t, beta_t, noise_alpha=noise_alpha)
        assert (x_t == x_1).all(), "At t=1, all tokens should be target"

    def test_uniform_samples_in_valid_range(self):
        """Uniform samples should be in [0, n_categories)."""
        torch.manual_seed(42)

        n_tokens = 10000
        n_categories = 5
        mask_index = n_categories

        x_0 = torch.full((n_tokens,), mask_index, dtype=torch.long)
        x_1 = torch.zeros(n_tokens, dtype=torch.long)  # all zeros

        t = 0.5
        alpha_t = torch.full((n_tokens, 1), 1 - t)
        beta_t = torch.full((n_tokens, 1), t)

        # Use high noise to get more uniform samples
        x_t, _ = sample_masked_ctmc(x_0, x_1, alpha_t, beta_t, noise_alpha=0.5)

        # All values should be in valid range [0, n_categories] (including mask)
        assert (x_t >= 0).all(), "All tokens should be >= 0"
        assert (x_t <= mask_index).all(), f"All tokens should be <= {mask_index}"

        # Uniform samples specifically should be in [0, n_categories)
        is_uniform = (x_t != mask_index) & (x_t != x_1)
        if is_uniform.any():
            uniform_vals = x_t[is_uniform]
            assert (uniform_vals >= 0).all(), "Uniform samples should be >= 0"
            assert (uniform_vals < n_categories).all(), f"Uniform samples should be < {n_categories}"

    def test_probability_distribution_is_correct(self):
        """Test that the three-way probabilities are approximately correct."""
        torch.manual_seed(42)

        n_tokens = 100000  # Large sample for accurate statistics
        n_categories = 10
        mask_index = n_categories

        x_0 = torch.full((n_tokens,), mask_index, dtype=torch.long)
        x_1 = torch.zeros(n_tokens, dtype=torch.long)  # all zeros for easy identification

        t = 0.5
        noise_alpha = 0.2
        alpha_t = torch.full((n_tokens, 1), 1 - t)
        beta_t = torch.full((n_tokens, 1), t)

        x_t, _ = sample_masked_ctmc(x_0, x_1, alpha_t, beta_t, noise_alpha=noise_alpha)

        # Expected probabilities at t=0.5:
        # noise_t = 0.2 * 0.5 * 0.5 = 0.05
        # prob_mask = 0.5 - 0.025 = 0.475
        # prob_target = 0.5 - 0.025 = 0.475
        # prob_uniform = 0.05

        is_mask = x_t == mask_index
        is_target = x_t == x_1
        is_uniform = ~is_mask & ~is_target

        mask_frac = is_mask.float().mean().item()
        target_frac = is_target.float().mean().item()
        uniform_frac = is_uniform.float().mean().item()

        # Allow 1% tolerance
        assert abs(mask_frac - 0.475) < 0.01, f"Expected 47.5% mask, got {mask_frac:.2%}"
        assert abs(target_frac - 0.475) < 0.01, f"Expected 47.5% target, got {target_frac:.2%}"
        assert abs(uniform_frac - 0.05) < 0.01, f"Expected 5% uniform, got {uniform_frac:.2%}"


class TestNoisyPathsStage2:
    """Tests for Stage 2: three-way conditional path with data-marginal corruption."""

    def test_marginal_loading_caches_results(self):
        """Test that marginal loading is cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock marginals file
            marginal_path = Path(tmpdir) / "marginals.npz"
            marginal = np.array([0.5, 0.3, 0.2])
            np.savez(marginal_path, test_marginal=marginal)

            # Load twice - should use cache
            result1 = _load_marginals(str(marginal_path), "test_marginal")
            result2 = _load_marginals(str(marginal_path), "test_marginal")

            assert torch.allclose(result1, result2)
            assert torch.allclose(result1, torch.tensor([0.5, 0.3, 0.2]))

    def test_marginal_corruption_samples_from_distribution(self):
        """With marginal_path provided, samples should follow the marginal distribution."""
        torch.manual_seed(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a heavily skewed marginal (90% type 0, 10% type 1)
            marginal_path = Path(tmpdir) / "marginals.npz"
            marginal = np.array([0.9, 0.1, 0.0, 0.0, 0.0])
            np.savez(marginal_path, test_marginal=marginal)

            n_tokens = 50000
            n_categories = 5
            mask_index = n_categories

            x_0 = torch.full((n_tokens,), mask_index, dtype=torch.long)
            # Use a target that's not in the high-probability region
            x_1 = torch.full((n_tokens,), 4, dtype=torch.long)  # all type 4

            t = 0.5
            noise_alpha = 0.5  # High noise for more samples
            alpha_t = torch.full((n_tokens, 1), 1 - t)
            beta_t = torch.full((n_tokens, 1), t)

            x_t, _ = sample_masked_ctmc(
                x_0, x_1, alpha_t, beta_t,
                noise_alpha=noise_alpha,
                marginal_path=str(marginal_path),
                marginal_key="test_marginal",
            )

            # Get the corrupted tokens (not mask, not target)
            is_corrupt = (x_t != mask_index) & (x_t != x_1)
            corrupt_vals = x_t[is_corrupt]

            if len(corrupt_vals) > 100:  # Need enough samples
                # Count distribution of corrupt samples
                type_0_frac = (corrupt_vals == 0).float().mean().item()
                type_1_frac = (corrupt_vals == 1).float().mean().item()

                # Should be approximately 90% type 0, 10% type 1
                assert 0.85 < type_0_frac < 0.95, f"Expected ~90% type 0, got {type_0_frac:.2%}"
                assert 0.05 < type_1_frac < 0.15, f"Expected ~10% type 1, got {type_1_frac:.2%}"

    def test_marginal_vs_uniform_produces_different_distributions(self):
        """Marginal corruption should differ from uniform corruption."""
        torch.manual_seed(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-uniform marginal
            marginal_path = Path(tmpdir) / "marginals.npz"
            marginal = np.array([0.8, 0.15, 0.05])  # Heavily skewed
            np.savez(marginal_path, test_marginal=marginal)

            n_tokens = 50000
            n_categories = 3
            mask_index = n_categories

            x_0 = torch.full((n_tokens,), mask_index, dtype=torch.long)
            x_1 = torch.full((n_tokens,), 2, dtype=torch.long)  # all type 2 (rare in marginal)

            t = 0.5
            noise_alpha = 0.5
            alpha_t = torch.full((n_tokens, 1), 1 - t)
            beta_t = torch.full((n_tokens, 1), t)

            # Stage 1: Uniform
            torch.manual_seed(42)
            x_t_uniform, _ = sample_masked_ctmc(
                x_0, x_1, alpha_t, beta_t,
                noise_alpha=noise_alpha,
            )

            # Stage 2: Marginal
            torch.manual_seed(42)
            x_t_marginal, _ = sample_masked_ctmc(
                x_0, x_1, alpha_t, beta_t,
                noise_alpha=noise_alpha,
                marginal_path=str(marginal_path),
                marginal_key="test_marginal",
            )

            # Get corrupt samples
            is_corrupt_uniform = (x_t_uniform != mask_index) & (x_t_uniform != x_1)
            is_corrupt_marginal = (x_t_marginal != mask_index) & (x_t_marginal != x_1)

            uniform_type0_frac = (x_t_uniform[is_corrupt_uniform] == 0).float().mean().item()
            marginal_type0_frac = (x_t_marginal[is_corrupt_marginal] == 0).float().mean().item()

            # Uniform should be ~33% type 0, marginal should be ~80% type 0
            assert uniform_type0_frac < 0.5, f"Uniform should have <50% type 0, got {uniform_type0_frac:.2%}"
            assert marginal_type0_frac > 0.7, f"Marginal should have >70% type 0, got {marginal_type0_frac:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
