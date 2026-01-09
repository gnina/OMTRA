"""
Integration tests for OMTRA model forward pass and training step.
These tests require real data to be available.
"""

import pytest
import torch
import dgl

from omtra.tasks.register import task_name_to_class


@pytest.mark.requires_data
class TestOMTRAInstantiation:
    """Tests for OMTRA model instantiation."""
    
    def test_omtra_instantiates_plinder(self, omtra_model_plinder):
        """OMTRA model should instantiate without errors for plinder config."""
        assert omtra_model_plinder is not None
    
    def test_omtra_instantiates_pharmit(self, omtra_model_pharmit):
        """OMTRA model should instantiate without errors for pharmit config."""
        assert omtra_model_pharmit is not None
    
    def test_omtra_has_vector_field(self, omtra_model_plinder):
        """OMTRA should have a VectorField instance."""
        assert hasattr(omtra_model_plinder, 'vector_field')
        assert omtra_model_plinder.vector_field is not None
    
    def test_omtra_has_loss_functions(self, omtra_model_plinder):
        """OMTRA should have loss function dict."""
        assert hasattr(omtra_model_plinder, 'loss_fn_dict')
        assert len(omtra_model_plinder.loss_fn_dict) > 0
    
    def test_omtra_has_interpolant_scheduler(self, omtra_model_plinder):
        """OMTRA should have an interpolant scheduler."""
        assert hasattr(omtra_model_plinder, 'interpolant_scheduler')
        assert omtra_model_plinder.interpolant_scheduler is not None
    
    def test_omtra_has_td_coupling(self, omtra_model_plinder):
        """OMTRA should have task-dataset coupling."""
        assert hasattr(omtra_model_plinder, 'td_coupling')
        assert omtra_model_plinder.td_coupling is not None
        assert len(omtra_model_plinder.td_coupling.task_space) > 0


@pytest.mark.requires_data
class TestOMTRAForwardFixedProtein:
    """Tests for OMTRA.forward() with fixed_protein_ligand_denovo_condensed task."""
    
    def test_forward_returns_loss_dict(self, omtra_model_plinder, sample_batch_fixed_protein):
        """forward() should return a dict of losses."""
        g, task_name = sample_batch_fixed_protein
        
        omtra_model_plinder.train()
        losses = omtra_model_plinder(g, task_name)
        
        assert isinstance(losses, dict)
        assert len(losses) > 0
    
    def test_forward_loss_keys_match_modalities(self, omtra_model_plinder, sample_batch_fixed_protein):
        """Loss dict keys should correspond to generated modalities."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        omtra_model_plinder.train()
        losses = omtra_model_plinder(g, task_name)
        
        expected_keys = {m.name for m in task_class.modalities_generated}
        actual_keys = set(losses.keys())
        
        # All generated modalities should have a loss
        assert expected_keys <= actual_keys, \
            f"Missing loss keys: {expected_keys - actual_keys}"
    
    def test_forward_loss_values_are_tensors(self, omtra_model_plinder, sample_batch_fixed_protein):
        """All loss values should be tensors."""
        g, task_name = sample_batch_fixed_protein
        
        omtra_model_plinder.train()
        losses = omtra_model_plinder(g, task_name)
        
        for key, loss in losses.items():
            assert isinstance(loss, torch.Tensor), f"Loss {key} should be a tensor"
    
    def test_forward_loss_values_are_finite(self, omtra_model_plinder, sample_batch_fixed_protein):
        """All loss values should be finite (no NaN or Inf)."""
        g, task_name = sample_batch_fixed_protein
        
        omtra_model_plinder.train()
        losses = omtra_model_plinder(g, task_name)
        
        for key, loss in losses.items():
            assert torch.isfinite(loss).all(), f"Loss {key} should be finite, got {loss}"
    
    def test_forward_loss_values_are_non_negative(self, omtra_model_plinder, sample_batch_fixed_protein):
        """All loss values should be non-negative."""
        g, task_name = sample_batch_fixed_protein
        
        omtra_model_plinder.train()
        losses = omtra_model_plinder(g, task_name)
        
        for key, loss in losses.items():
            assert loss >= 0, f"Loss {key} should be non-negative, got {loss}"
    
    def test_forward_with_multi_graph_batch(self, omtra_model_plinder, sample_batch_multi_fixed_protein):
        """forward() should work with batched graphs."""
        g, task_name = sample_batch_multi_fixed_protein
        
        assert g.batch_size > 1, "Batch should have multiple graphs"
        
        omtra_model_plinder.train()
        losses = omtra_model_plinder(g, task_name)
        
        assert isinstance(losses, dict)
        for key, loss in losses.items():
            assert torch.isfinite(loss).all(), f"Loss {key} should be finite"


@pytest.mark.requires_data
class TestOMTRAForwardRigidDocking:
    """Tests for OMTRA.forward() with rigid_docking_condensed task."""
    
    def test_forward_returns_loss_dict(self, omtra_model_plinder, sample_batch_rigid_docking):
        """forward() should return a dict of losses for rigid docking."""
        g, task_name = sample_batch_rigid_docking
        
        omtra_model_plinder.train()
        losses = omtra_model_plinder(g, task_name)
        
        assert isinstance(losses, dict)
        assert len(losses) > 0
    
    def test_forward_loss_values_are_finite(self, omtra_model_plinder, sample_batch_rigid_docking):
        """All loss values should be finite."""
        g, task_name = sample_batch_rigid_docking
        
        omtra_model_plinder.train()
        losses = omtra_model_plinder(g, task_name)
        
        for key, loss in losses.items():
            assert torch.isfinite(loss).all(), f"Loss {key} should be finite, got {loss}"
    
    def test_rigid_docking_generates_only_structure(self, omtra_model_plinder, sample_batch_rigid_docking):
        """Rigid docking should only generate ligand structure, not identity."""
        g, task_name = sample_batch_rigid_docking
        task_class = task_name_to_class(task_name)
        
        # Check task properties
        assert 'ligand_structure' in task_class.groups_generated
        assert 'ligand_identity' in task_class.groups_fixed or 'ligand_identity_condensed' in task_class.groups_fixed
        
        omtra_model_plinder.train()
        losses = omtra_model_plinder(g, task_name)
        
        # Should have position loss but potentially not atom type loss
        assert 'lig_x' in losses


@pytest.mark.requires_data
class TestOMTRAForwardDeNovoLigand:
    """Tests for OMTRA.forward() with denovo_ligand_condensed task (pharmit)."""
    
    def test_forward_returns_loss_dict(self, omtra_model_pharmit, sample_batch_denovo_ligand):
        """forward() should return a dict of losses for denovo ligand."""
        g, task_name = sample_batch_denovo_ligand
        
        omtra_model_pharmit.train()
        losses = omtra_model_pharmit(g, task_name)
        
        assert isinstance(losses, dict)
        assert len(losses) > 0
    
    def test_forward_loss_values_are_finite(self, omtra_model_pharmit, sample_batch_denovo_ligand):
        """All loss values should be finite."""
        g, task_name = sample_batch_denovo_ligand
        
        omtra_model_pharmit.train()
        losses = omtra_model_pharmit(g, task_name)
        
        for key, loss in losses.items():
            assert torch.isfinite(loss).all(), f"Loss {key} should be finite, got {loss}"
    
    def test_denovo_is_unconditional(self, sample_batch_denovo_ligand):
        """denovo_ligand_condensed should be an unconditional task."""
        _, task_name = sample_batch_denovo_ligand
        task_class = task_name_to_class(task_name)
        
        assert task_class.unconditional, "denovo_ligand_condensed should be unconditional"


@pytest.mark.requires_data
class TestOMTRAConditionalPath:
    """Tests for OMTRA.sample_conditional_path()."""
    
    def test_sample_conditional_path_sets_x_t(self, omtra_model_plinder, sample_batch_fixed_protein):
        """sample_conditional_path should populate x_t features on graph."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        # Sample time
        t = torch.rand(g.batch_size)
        
        # Get batch indices
        from omtra.data.graph.utils import get_batch_idxs, get_upper_edge_mask
        node_batch_idxs, edge_batch_idxs = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        
        # Sample conditional path
        g = omtra_model_plinder.sample_conditional_path(
            g, task_class, t, node_batch_idxs, edge_batch_idxs, lig_ue_mask
        )
        
        # Check that x_t features are set for generated modalities
        for m in task_class.modalities_generated:
            if m.is_node:
                if g.num_nodes(m.entity_name) > 0:
                    assert f"{m.data_key}_t" in g.nodes[m.entity_name].data, \
                        f"Missing {m.data_key}_t for node type {m.entity_name}"
            else:
                if g.num_edges(m.entity_name) > 0:
                    assert f"{m.data_key}_t" in g.edges[m.entity_name].data, \
                        f"Missing {m.data_key}_t for edge type {m.entity_name}"
    
    def test_conditional_path_at_t0_near_prior(self, omtra_model_plinder, sample_batch_fixed_protein):
        """At t=0, x_t should be close to prior (x_0)."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        t = torch.zeros(g.batch_size)
        
        from omtra.data.graph.utils import get_batch_idxs, get_upper_edge_mask
        node_batch_idxs, edge_batch_idxs = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        
        g = omtra_model_plinder.sample_conditional_path(
            g, task_class, t, node_batch_idxs, edge_batch_idxs, lig_ue_mask
        )
        
        # For continuous modalities, x_t should equal x_0 at t=0
        for m in task_class.modalities_generated:
            if not m.is_categorical and m.is_node and g.num_nodes(m.entity_name) > 0:
                x_t = g.nodes[m.entity_name].data[f"{m.data_key}_t"]
                x_0 = g.nodes[m.entity_name].data[f"{m.data_key}_0"]
                assert torch.allclose(x_t, x_0, atol=1e-5), \
                    f"At t=0, x_t should equal x_0 for {m.name}"
    
    def test_conditional_path_at_t1_near_target(self, omtra_model_plinder, sample_batch_fixed_protein):
        """At t=1, x_t should be close to target (x_1)."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        t = torch.ones(g.batch_size)
        
        from omtra.data.graph.utils import get_batch_idxs, get_upper_edge_mask
        node_batch_idxs, edge_batch_idxs = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        
        g = omtra_model_plinder.sample_conditional_path(
            g, task_class, t, node_batch_idxs, edge_batch_idxs, lig_ue_mask
        )
        
        # For continuous modalities, x_t should equal x_1_true at t=1
        for m in task_class.modalities_generated:
            if not m.is_categorical and m.is_node and g.num_nodes(m.entity_name) > 0:
                x_t = g.nodes[m.entity_name].data[f"{m.data_key}_t"]
                x_1 = g.nodes[m.entity_name].data[f"{m.data_key}_1_true"]
                assert torch.allclose(x_t, x_1, atol=1e-5), \
                    f"At t=1, x_t should equal x_1_true for {m.name}"


@pytest.mark.requires_data
class TestOMTRATrainingStep:
    """Tests for OMTRA.training_step()."""
    
    def test_training_step_returns_loss(self, omtra_model_plinder, sample_batch_fixed_protein):
        """training_step should return a scalar loss."""
        g, task_name = sample_batch_fixed_protein
        batch_data = (g, task_name, "plinder")  # (graph, task_name, dataset_name)
        
        omtra_model_plinder.train()
        # Mock the manual_checkpoint to avoid file operations
        omtra_model_plinder.manual_checkpoint = lambda x: None
        # Mock all_gather for single-process testing
        omtra_model_plinder.all_gather = lambda x: x.unsqueeze(0)
        
        loss = omtra_model_plinder.training_step(batch_data, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0 or loss.numel() == 1, "Loss should be scalar"
    
    def test_training_step_loss_is_finite(self, omtra_model_plinder, sample_batch_fixed_protein):
        """training_step loss should be finite."""
        g, task_name = sample_batch_fixed_protein
        batch_data = (g, task_name, "plinder")
        
        omtra_model_plinder.train()
        omtra_model_plinder.manual_checkpoint = lambda x: None
        omtra_model_plinder.all_gather = lambda x: x.unsqueeze(0)
        
        loss = omtra_model_plinder.training_step(batch_data, batch_idx=0)
        
        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
    
    def test_training_step_loss_requires_grad(self, omtra_model_plinder, sample_batch_fixed_protein):
        """training_step loss should require gradients."""
        g, task_name = sample_batch_fixed_protein
        batch_data = (g, task_name, "plinder")
        
        omtra_model_plinder.train()
        omtra_model_plinder.manual_checkpoint = lambda x: None
        omtra_model_plinder.all_gather = lambda x: x.unsqueeze(0)
        
        loss = omtra_model_plinder.training_step(batch_data, batch_idx=0)
        
        assert loss.requires_grad, "Loss should require gradients"
    
    def test_training_step_backward_pass(self, omtra_model_plinder, sample_batch_fixed_protein):
        """Should be able to compute gradients through training_step."""
        g, task_name = sample_batch_fixed_protein
        batch_data = (g, task_name, "plinder")
        
        omtra_model_plinder.train()
        omtra_model_plinder.manual_checkpoint = lambda x: None
        omtra_model_plinder.all_gather = lambda x: x.unsqueeze(0)
        
        # Zero gradients
        omtra_model_plinder.zero_grad()
        
        loss = omtra_model_plinder.training_step(batch_data, batch_idx=0)
        loss.backward()
        
        # Check that at least some parameters have gradients
        has_grad = False
        for param in omtra_model_plinder.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "At least some parameters should have gradients after backward"


@pytest.mark.requires_data
class TestOMTRAConfigureOptimizers:
    """Tests for OMTRA.configure_optimizers()."""
    
    def test_configure_optimizers_returns_optimizer(self, omtra_model_plinder):
        """configure_optimizers should return an optimizer."""
        result = omtra_model_plinder.configure_optimizers()
        
        # Can return optimizer directly or dict with optimizer
        if isinstance(result, dict):
            assert 'optimizer' in result
            optimizer = result['optimizer']
        else:
            optimizer = result
        
        assert isinstance(optimizer, torch.optim.Optimizer)
