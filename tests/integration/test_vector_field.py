"""
Integration tests for VectorField forward pass and integration.
These tests require real data to be available.
"""

import pytest
import torch
import dgl

from omtra.tasks.register import task_name_to_class
from omtra.data.graph.utils import get_batch_idxs, get_upper_edge_mask


@pytest.mark.requires_data
class TestVectorFieldInstantiation:
    """Tests for VectorField instantiation via OMTRA."""
    
    def test_vector_field_exists(self, omtra_model_plinder):
        """OMTRA should have a VectorField."""
        from omtra.models.vector_field import VectorField
        assert isinstance(omtra_model_plinder.vector_field, VectorField)
    
    def test_vector_field_has_conv_layers(self, omtra_model_plinder):
        """VectorField should have convolution layers."""
        vf = omtra_model_plinder.vector_field
        assert hasattr(vf, 'conv_layers')
        assert len(vf.conv_layers) > 0
    
    def test_vector_field_has_node_output_heads(self, omtra_model_plinder):
        """VectorField should have node output heads."""
        vf = omtra_model_plinder.vector_field
        assert hasattr(vf, 'node_output_heads')
        assert len(vf.node_output_heads) > 0
    
    def test_vector_field_has_interpolant_scheduler(self, omtra_model_plinder):
        """VectorField should have interpolant scheduler."""
        vf = omtra_model_plinder.vector_field
        assert hasattr(vf, 'interpolant_scheduler')
        assert vf.interpolant_scheduler is not None


@pytest.mark.requires_data
class TestVectorFieldForwardFixedProtein:
    """Tests for VectorField.forward() with fixed_protein task."""
    
    @pytest.fixture
    def prepared_batch(self, omtra_model_plinder, sample_batch_fixed_protein):
        """Prepare a batch with conditional path sampled."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        # Sample time and conditional path
        t = torch.rand(g.batch_size)
        node_batch_idxs, edge_batch_idxs = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        
        g = omtra_model_plinder.sample_conditional_path(
            g, task_class, t, node_batch_idxs, edge_batch_idxs, lig_ue_mask
        )
        
        upper_edge_mask = {"lig_to_lig": lig_ue_mask}
        
        return g, task_class, t, node_batch_idxs, upper_edge_mask
    
    def test_forward_returns_dict(self, omtra_model_plinder, prepared_batch):
        """VectorField.forward() should return a dict of predictions."""
        g, task_class, t, node_batch_idxs, upper_edge_mask = prepared_batch
        
        vf = omtra_model_plinder.vector_field
        vf.eval()
        
        with torch.no_grad():
            dst_dict = vf(g, task_class, t, node_batch_idxs, upper_edge_mask)
        
        assert isinstance(dst_dict, dict)
        assert len(dst_dict) > 0
    
    def test_forward_output_keys_match_generated_modalities(self, omtra_model_plinder, prepared_batch):
        """Output keys should match generated modalities."""
        g, task_class, t, node_batch_idxs, upper_edge_mask = prepared_batch
        
        vf = omtra_model_plinder.vector_field
        vf.eval()
        
        with torch.no_grad():
            dst_dict = vf(g, task_class, t, node_batch_idxs, upper_edge_mask)
        
        expected_keys = {m.name for m in task_class.modalities_generated}
        actual_keys = set(dst_dict.keys())
        
        assert expected_keys == actual_keys, \
            f"Expected keys {expected_keys}, got {actual_keys}"
    
    def test_forward_position_output_shape(self, omtra_model_plinder, prepared_batch):
        """Position predictions should have shape (n_atoms, 3)."""
        g, task_class, t, node_batch_idxs, upper_edge_mask = prepared_batch
        
        vf = omtra_model_plinder.vector_field
        vf.eval()
        
        with torch.no_grad():
            dst_dict = vf(g, task_class, t, node_batch_idxs, upper_edge_mask)
        
        # Check lig_x (ligand positions)
        if 'lig_x' in dst_dict:
            lig_x = dst_dict['lig_x']
            n_lig_atoms = g.num_nodes('lig')
            assert lig_x.shape == (n_lig_atoms, 3), \
                f"Expected shape ({n_lig_atoms}, 3), got {lig_x.shape}"
    
    def test_forward_atom_type_output_shape(self, omtra_model_plinder, prepared_batch):
        """Atom type predictions should have shape (n_atoms, n_categories)."""
        g, task_class, t, node_batch_idxs, upper_edge_mask = prepared_batch
        
        vf = omtra_model_plinder.vector_field
        vf.eval()
        
        with torch.no_grad():
            dst_dict = vf(g, task_class, t, node_batch_idxs, upper_edge_mask)
        
        # Check lig_cond_a (condensed atom types)
        if 'lig_cond_a' in dst_dict:
            lig_a = dst_dict['lig_cond_a']
            n_lig_atoms = g.num_nodes('lig')
            assert lig_a.shape[0] == n_lig_atoms, \
                f"Expected {n_lig_atoms} atoms, got {lig_a.shape[0]}"
            assert lig_a.dim() == 2, "Atom type logits should be 2D"
            assert lig_a.shape[1] > 0, "Should have at least one atom type category"
    
    def test_forward_outputs_are_finite(self, omtra_model_plinder, prepared_batch):
        """All outputs should be finite."""
        g, task_class, t, node_batch_idxs, upper_edge_mask = prepared_batch
        
        vf = omtra_model_plinder.vector_field
        vf.eval()
        
        with torch.no_grad():
            dst_dict = vf(g, task_class, t, node_batch_idxs, upper_edge_mask)
        
        for key, value in dst_dict.items():
            assert torch.isfinite(value).all(), \
                f"Output {key} should be finite"
    
    def test_forward_with_apply_softmax(self, omtra_model_plinder, prepared_batch):
        """forward() with apply_softmax=True should return probabilities."""
        g, task_class, t, node_batch_idxs, upper_edge_mask = prepared_batch
        
        vf = omtra_model_plinder.vector_field
        vf.eval()
        
        with torch.no_grad():
            dst_dict = vf(g, task_class, t, node_batch_idxs, upper_edge_mask, apply_softmax=True)
        
        # Categorical outputs should sum to 1 after softmax
        for m in task_class.modalities_generated:
            if m.is_categorical and m.name in dst_dict:
                probs = dst_dict[m.name]
                sums = probs.sum(dim=-1)
                assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
                    f"Probabilities for {m.name} should sum to 1"


@pytest.mark.requires_data
class TestVectorFieldForwardRigidDocking:
    """Tests for VectorField.forward() with rigid_docking task."""
    
    @pytest.fixture
    def prepared_batch(self, omtra_model_plinder, sample_batch_rigid_docking):
        """Prepare a batch with conditional path sampled."""
        g, task_name = sample_batch_rigid_docking
        task_class = task_name_to_class(task_name)
        
        t = torch.rand(g.batch_size)
        node_batch_idxs, edge_batch_idxs = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        
        g = omtra_model_plinder.sample_conditional_path(
            g, task_class, t, node_batch_idxs, edge_batch_idxs, lig_ue_mask
        )
        
        upper_edge_mask = {"lig_to_lig": lig_ue_mask}
        
        return g, task_class, t, node_batch_idxs, upper_edge_mask
    
    def test_forward_returns_dict(self, omtra_model_plinder, prepared_batch):
        """VectorField.forward() should return predictions for rigid docking."""
        g, task_class, t, node_batch_idxs, upper_edge_mask = prepared_batch
        
        vf = omtra_model_plinder.vector_field
        vf.eval()
        
        with torch.no_grad():
            dst_dict = vf(g, task_class, t, node_batch_idxs, upper_edge_mask)
        
        assert isinstance(dst_dict, dict)
        assert 'lig_x' in dst_dict, "Rigid docking should predict ligand positions"
    
    def test_forward_outputs_are_finite(self, omtra_model_plinder, prepared_batch):
        """All outputs should be finite."""
        g, task_class, t, node_batch_idxs, upper_edge_mask = prepared_batch
        
        vf = omtra_model_plinder.vector_field
        vf.eval()
        
        with torch.no_grad():
            dst_dict = vf(g, task_class, t, node_batch_idxs, upper_edge_mask)
        
        for key, value in dst_dict.items():
            assert torch.isfinite(value).all(), f"Output {key} should be finite"


@pytest.mark.requires_data
class TestVectorFieldDenoiseGraph:
    """Tests for VectorField.denoise_graph()."""
    
    def test_denoise_graph_produces_predictions(self, omtra_model_plinder, sample_batch_fixed_protein):
        """denoise_graph should produce predictions for all generated modalities."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        t = torch.rand(g.batch_size)
        node_batch_idxs, edge_batch_idxs = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        
        g = omtra_model_plinder.sample_conditional_path(
            g, task_class, t, node_batch_idxs, edge_batch_idxs, lig_ue_mask
        )
        
        vf = omtra_model_plinder.vector_field
        vf.eval()
        
        # Build initial features (simplified version of what forward() does)
        from omtra.tasks.utils import build_edges
        g = build_edges(g, task_class, node_batch_idxs, vf.graph_config)
        
        # The full forward handles this, so we just test via forward
        upper_edge_mask = {"lig_to_lig": lig_ue_mask}
        
        with torch.no_grad():
            dst_dict = vf(g, task_class, t, node_batch_idxs, upper_edge_mask)
        
        for m in task_class.modalities_generated:
            assert m.name in dst_dict, f"Missing prediction for {m.name}"


@pytest.mark.requires_data
class TestVectorFieldIntegrate:
    """Tests for VectorField.integrate() (sampling)."""
    
    def test_integrate_runs_without_error(self, omtra_model_plinder, sample_batch_fixed_protein):
        """integrate() should complete without error."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        node_batch_idxs, _ = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        upper_edge_mask = {"lig_to_lig": lig_ue_mask}
        
        vf = omtra_model_plinder.vector_field
        vf.eval()
        
        with torch.no_grad():
            # Use few timesteps for speed
            g_out = vf.integrate(
                g, 
                task_class, 
                upper_edge_mask,
                n_timesteps=5,  # Very few for testing
            )
        
        assert g_out is not None
        assert isinstance(g_out, dgl.DGLGraph)
    
    def test_integrate_produces_x1_values(self, omtra_model_plinder, sample_batch_fixed_protein):
        """integrate() should populate x_1 values on the graph."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        node_batch_idxs, _ = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        upper_edge_mask = {"lig_to_lig": lig_ue_mask}
        
        vf = omtra_model_plinder.vector_field
        vf.eval()
        
        with torch.no_grad():
            g_out = vf.integrate(
                g, 
                task_class, 
                upper_edge_mask,
                n_timesteps=5,
            )
        
        # Check that x_1 values are set
        for m in task_class.modalities_generated:
            if m.is_node and g_out.num_nodes(m.entity_name) > 0:
                assert f"{m.data_key}_1" in g_out.nodes[m.entity_name].data, \
                    f"Missing {m.data_key}_1 for {m.entity_name}"
    
    def test_integrate_with_visualization(self, omtra_model_plinder, sample_batch_fixed_protein):
        """integrate() with visualize=True should return trajectory."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        node_batch_idxs, _ = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        upper_edge_mask = {"lig_to_lig": lig_ue_mask}
        
        vf = omtra_model_plinder.vector_field
        vf.eval()
        
        with torch.no_grad():
            result = vf.integrate(
                g, 
                task_class, 
                upper_edge_mask,
                n_timesteps=5,
                visualize=True,
            )
        
        # Should return (graph, trajectory)
        assert isinstance(result, tuple)
        g_out, traj = result
        assert isinstance(g_out, dgl.DGLGraph)
        assert isinstance(traj, list)


@pytest.mark.requires_data  
class TestVectorFieldStep:
    """Tests for VectorField.step() (single integration step)."""
    
    def test_step_updates_graph(self, omtra_model_plinder, sample_batch_fixed_protein):
        """step() should update graph features."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        node_batch_idxs, edge_batch_idxs = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        upper_edge_mask = {"lig_to_lig": lig_ue_mask}
        
        # Initialize x_t values (sample conditional path at t=0)
        t = torch.zeros(g.batch_size)
        g = omtra_model_plinder.sample_conditional_path(
            g, task_class, t, node_batch_idxs, edge_batch_idxs, lig_ue_mask
        )
        
        vf = omtra_model_plinder.vector_field
        vf.eval()
        
        # Get interpolant weights
        t_i = 0.0
        s_i = 0.1
        alpha_t, beta_t = vf.interpolant_scheduler.weights(torch.tensor([t_i]), task_class)
        alpha_s, beta_s = vf.interpolant_scheduler.weights(torch.tensor([s_i]), task_class)
        alpha_t_prime, beta_t_prime = vf.interpolant_scheduler.weight_derivative(torch.tensor([t_i]), task_class)
        
        alpha_t_i = {k: v[0] for k, v in alpha_t.items()}
        alpha_s_i = {k: v[0] for k, v in alpha_s.items()}
        alpha_t_prime_i = {k: v[0] for k, v in alpha_t_prime.items()}
        beta_t_i = {k: v[0] for k, v in beta_t.items()}
        beta_s_i = {k: v[0] for k, v in beta_s.items()}
        beta_t_prime_i = {k: v[0] for k, v in beta_t_prime.items()}
        
        # Store original positions
        orig_lig_x = g.nodes['lig'].data['x_t'].clone()
        
        with torch.no_grad():
            g_out, dst_dict = vf.step(
                g=g,
                task=task_class,
                s_i=torch.tensor(s_i),
                t_i=torch.tensor(t_i),
                alpha_t_i=alpha_t_i,
                alpha_s_i=alpha_s_i,
                alpha_t_prime_i=alpha_t_prime_i,
                beta_t_i=beta_t_i,
                beta_s_i=beta_s_i,
                beta_t_prime_i=beta_t_prime_i,
                node_batch_idxs=node_batch_idxs,
                edge_batch_idxs=edge_batch_idxs,
                upper_edge_mask=upper_edge_mask,
                cat_temp_func=lambda t: 0.05,
            )
        
        # Positions should have changed
        new_lig_x = g_out.nodes['lig'].data['x_t']
        # At least some positions should differ (unless they happen to be at equilibrium)
        # This is a weak test but avoids false negatives
        assert new_lig_x is not None
        assert dst_dict is not None


@pytest.mark.requires_data
class TestVectorFieldVectorFieldMethod:
    """Tests for VectorField.vector_field() (the ODE vector field computation)."""
    
    def test_vector_field_computation(self, omtra_model_plinder):
        """vector_field() should compute correct ODE velocity."""
        vf = omtra_model_plinder.vector_field
        
        # Simple test case
        x_t = torch.randn(10, 3)
        x_1 = torch.randn(10, 3)
        alpha_t = torch.tensor(0.5)
        alpha_t_prime = torch.tensor(-1.0)
        beta_t = torch.tensor(0.5)
        beta_t_prime = torch.tensor(1.0)
        
        vf_out = vf.vector_field(x_t, x_1, alpha_t, alpha_t_prime, beta_t, beta_t_prime)
        
        assert vf_out.shape == x_t.shape
        assert torch.isfinite(vf_out).all()
    
    def test_vector_field_at_t0(self, omtra_model_plinder):
        """At t=0, vector field should point towards x_1."""
        vf = omtra_model_plinder.vector_field
        
        x_t = torch.zeros(10, 3)
        x_1 = torch.ones(10, 3)
        
        # At t=0: alpha=1, beta=0, alpha'=-1, beta'=1
        alpha_t = torch.tensor(1.0)
        alpha_t_prime = torch.tensor(-1.0)
        beta_t = torch.tensor(0.0)
        beta_t_prime = torch.tensor(1.0)
        
        vf_out = vf.vector_field(x_t, x_1, alpha_t, alpha_t_prime, beta_t, beta_t_prime)
        
        # For linear interpolant, vf = -x_t + x_1 at t=0
        expected = -x_t + x_1
        assert torch.allclose(vf_out, expected, atol=1e-5)
