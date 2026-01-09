"""
Unit tests for InterpolantScheduler and conditional path functions.
These tests don't require real data - they use synthetic tensors.
"""

import pytest
import torch
from omtra.models.interpolant_scheduler import InterpolantScheduler
from omtra.models.conditional_paths.paths import (
    sample_continuous_interpolant,
    sample_masked_ctmc,
)
from omtra.tasks.register import task_name_to_class


class TestInterpolantScheduler:
    """Tests for the InterpolantScheduler class."""
    
    @pytest.fixture
    def scheduler(self):
        """Create an InterpolantScheduler with linear schedule."""
        return InterpolantScheduler(schedule_type="linear")
    
    @pytest.fixture
    def task_class(self):
        """Get a task class for testing."""
        return task_name_to_class("fixed_protein_ligand_denovo_condensed")
    
    def test_weights_at_t0(self, scheduler, task_class):
        """At t=0, alpha should be 1 and beta should be 0."""
        t = torch.tensor([0.0])
        alpha_t, beta_t = scheduler.weights(t, task_class)
        
        for m in task_class.modalities_present:
            assert torch.allclose(alpha_t[m.name], torch.tensor([1.0])), \
                f"alpha_t for {m.name} should be 1.0 at t=0"
            assert torch.allclose(beta_t[m.name], torch.tensor([0.0])), \
                f"beta_t for {m.name} should be 0.0 at t=0"
    
    def test_weights_at_t1(self, scheduler, task_class):
        """At t=1, alpha should be 0 and beta should be 1."""
        t = torch.tensor([1.0])
        alpha_t, beta_t = scheduler.weights(t, task_class)
        
        for m in task_class.modalities_present:
            assert torch.allclose(alpha_t[m.name], torch.tensor([0.0])), \
                f"alpha_t for {m.name} should be 0.0 at t=1"
            assert torch.allclose(beta_t[m.name], torch.tensor([1.0])), \
                f"beta_t for {m.name} should be 1.0 at t=1"
    
    def test_weights_at_t05(self, scheduler, task_class):
        """At t=0.5, alpha and beta should both be 0.5."""
        t = torch.tensor([0.5])
        alpha_t, beta_t = scheduler.weights(t, task_class)
        
        for m in task_class.modalities_present:
            assert torch.allclose(alpha_t[m.name], torch.tensor([0.5])), \
                f"alpha_t for {m.name} should be 0.5 at t=0.5"
            assert torch.allclose(beta_t[m.name], torch.tensor([0.5])), \
                f"beta_t for {m.name} should be 0.5 at t=0.5"
    
    def test_weights_sum_to_one(self, scheduler, task_class):
        """alpha_t + beta_t should always equal 1."""
        t = torch.linspace(0, 1, 11)
        alpha_t, beta_t = scheduler.weights(t, task_class)
        
        for m in task_class.modalities_present:
            sums = alpha_t[m.name] + beta_t[m.name]
            assert torch.allclose(sums, torch.ones_like(sums)), \
                f"alpha_t + beta_t for {m.name} should sum to 1"
    
    def test_weight_derivative_constant(self, scheduler, task_class):
        """For linear schedule, derivatives should be constant (-1 for alpha, 1 for beta)."""
        t = torch.linspace(0, 1, 11)
        alpha_prime, beta_prime = scheduler.weight_derivative(t, task_class)
        
        for m in task_class.modalities_present:
            assert torch.allclose(alpha_prime[m.name], torch.full_like(t, -1.0)), \
                f"alpha' for {m.name} should be -1"
            assert torch.allclose(beta_prime[m.name], torch.full_like(t, 1.0)), \
                f"beta' for {m.name} should be 1"
    
    def test_weights_batch_dimension(self, scheduler, task_class):
        """Weights should work with batch dimension."""
        batch_size = 8
        t = torch.rand(batch_size)
        alpha_t, beta_t = scheduler.weights(t, task_class)
        
        for m in task_class.modalities_present:
            assert alpha_t[m.name].shape == (batch_size,), \
                f"alpha_t for {m.name} should have shape (batch_size,)"
            assert beta_t[m.name].shape == (batch_size,), \
                f"beta_t for {m.name} should have shape (batch_size,)"


class TestContinuousInterpolant:
    """Tests for the continuous interpolant conditional path."""
    
    def test_interpolant_at_t0(self):
        """At t=0 (alpha=1, beta=0), should return x_0."""
        n_nodes = 10
        x_0 = torch.randn(n_nodes, 3)
        x_1 = torch.randn(n_nodes, 3)
        alpha_t = torch.ones(n_nodes, 1)
        beta_t = torch.zeros(n_nodes, 1)
        
        x_t = sample_continuous_interpolant(x_0, x_1, alpha_t, beta_t)
        
        assert torch.allclose(x_t, x_0), "At t=0, x_t should equal x_0"
    
    def test_interpolant_at_t1(self):
        """At t=1 (alpha=0, beta=1), should return x_1."""
        n_nodes = 10
        x_0 = torch.randn(n_nodes, 3)
        x_1 = torch.randn(n_nodes, 3)
        alpha_t = torch.zeros(n_nodes, 1)
        beta_t = torch.ones(n_nodes, 1)
        
        x_t = sample_continuous_interpolant(x_0, x_1, alpha_t, beta_t)
        
        assert torch.allclose(x_t, x_1), "At t=1, x_t should equal x_1"
    
    def test_interpolant_midpoint(self):
        """At t=0.5 (alpha=0.5, beta=0.5), should return midpoint."""
        n_nodes = 10
        x_0 = torch.randn(n_nodes, 3)
        x_1 = torch.randn(n_nodes, 3)
        alpha_t = torch.full((n_nodes, 1), 0.5)
        beta_t = torch.full((n_nodes, 1), 0.5)
        
        x_t = sample_continuous_interpolant(x_0, x_1, alpha_t, beta_t)
        expected = 0.5 * x_0 + 0.5 * x_1
        
        assert torch.allclose(x_t, expected), "At t=0.5, x_t should be midpoint of x_0 and x_1"
    
    def test_interpolant_preserves_shape(self):
        """Interpolant should preserve input shape."""
        n_nodes = 10
        x_0 = torch.randn(n_nodes, 3)
        x_1 = torch.randn(n_nodes, 3)
        alpha_t = torch.full((n_nodes, 1), 0.3)
        beta_t = torch.full((n_nodes, 1), 0.7)
        
        x_t = sample_continuous_interpolant(x_0, x_1, alpha_t, beta_t)
        
        assert x_t.shape == x_0.shape, "Output shape should match input shape"
    
    def test_interpolant_with_vector_features(self):
        """Interpolant should work with 3D tensors (e.g., pharmacophore vectors)."""
        n_nodes = 10
        n_vecs = 4
        x_0 = torch.randn(n_nodes, n_vecs, 3)
        x_1 = torch.randn(n_nodes, n_vecs, 3)
        alpha_t = torch.full((n_nodes, 1), 0.3)
        beta_t = torch.full((n_nodes, 1), 0.7)
        
        x_t = sample_continuous_interpolant(x_0, x_1, alpha_t, beta_t)
        
        assert x_t.shape == x_0.shape, "Output shape should match 3D input shape"
        expected = 0.3 * x_0 + 0.7 * x_1
        assert torch.allclose(x_t, expected), "Interpolation should be correct for 3D tensors"


class TestCTMCMask:
    """Tests for the CTMC mask conditional path (categorical features)."""
    
    def test_ctmc_at_t0_all_masked(self):
        """At t=0 (alpha=1), all features should be masked (equal to x_0)."""
        n_nodes = 100
        mask_token = 10
        x_0 = torch.full((n_nodes,), mask_token)
        x_1 = torch.randint(0, 10, (n_nodes,))
        alpha_t = torch.ones(n_nodes, 1)
        beta_t = torch.zeros(n_nodes, 1)
        
        x_t = sample_masked_ctmc(x_0, x_1, alpha_t, beta_t)
        
        # All should be mask token (x_0)
        assert (x_t == mask_token).all(), "At t=0, all tokens should be masked"
    
    def test_ctmc_at_t1_all_unmasked(self):
        """At t=1 (alpha=0), all features should be unmasked (equal to x_1)."""
        n_nodes = 100
        mask_token = 10
        x_0 = torch.full((n_nodes,), mask_token)
        x_1 = torch.randint(0, 10, (n_nodes,))
        alpha_t = torch.zeros(n_nodes, 1)
        beta_t = torch.ones(n_nodes, 1)
        
        x_t = sample_masked_ctmc(x_0, x_1, alpha_t, beta_t)
        
        # All should be x_1
        assert (x_t == x_1).all(), "At t=1, all tokens should equal x_1"
    
    def test_ctmc_intermediate_partial_masking(self):
        """At intermediate t, some features should be masked and some unmasked."""
        torch.manual_seed(42)  # For reproducibility
        n_nodes = 1000
        mask_token = 10
        x_0 = torch.full((n_nodes,), mask_token)
        x_1 = torch.randint(0, 10, (n_nodes,))
        alpha_t = torch.full((n_nodes, 1), 0.5)
        beta_t = torch.full((n_nodes, 1), 0.5)
        
        x_t = sample_masked_ctmc(x_0, x_1, alpha_t, beta_t)
        
        # Some should be masked, some unmasked
        n_masked = (x_t == mask_token).sum().item()
        # With alpha=0.5, expect ~50% masked, allow some variance
        assert 300 < n_masked < 700, \
            f"Expected ~50% masked at t=0.5, got {n_masked/n_nodes*100:.1f}%"
    
    def test_ctmc_preserves_shape(self):
        """CTMC should preserve input shape."""
        n_nodes = 100
        x_0 = torch.randint(0, 10, (n_nodes,))
        x_1 = torch.randint(0, 10, (n_nodes,))
        alpha_t = torch.full((n_nodes, 1), 0.3)
        beta_t = torch.full((n_nodes, 1), 0.7)
        
        x_t = sample_masked_ctmc(x_0, x_1, alpha_t, beta_t)
        
        assert x_t.shape == x_0.shape, "Output shape should match input shape"
    
    def test_ctmc_monotonic_unmasking(self):
        """As t increases (alpha decreases), more features should be unmasked."""
        torch.manual_seed(42)
        n_nodes = 1000
        mask_token = 10
        x_0 = torch.full((n_nodes,), mask_token)
        x_1 = torch.randint(0, 10, (n_nodes,))
        
        n_masked_at_t = []
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            alpha = 1.0 - t
            beta = t
            alpha_t = torch.full((n_nodes, 1), alpha)
            beta_t = torch.full((n_nodes, 1), beta)
            
            x_t = sample_masked_ctmc(x_0, x_1, alpha_t, beta_t)
            n_masked = (x_t == mask_token).sum().item()
            n_masked_at_t.append(n_masked)
        
        # Number of masked tokens should decrease as t increases
        for i in range(len(n_masked_at_t) - 1):
            assert n_masked_at_t[i] >= n_masked_at_t[i + 1], \
                f"Masked count should decrease as t increases: {n_masked_at_t}"
