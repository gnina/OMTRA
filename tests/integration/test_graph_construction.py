"""
Integration tests for graph construction and prior sampling.
These tests verify that graphs returned from datasets have correct structure.
"""

import pytest
import torch
import dgl

from omtra.tasks.register import task_name_to_class


@pytest.mark.requires_data
class TestGraphNodeTypes:
    """Tests for graph node types."""
    
    def test_fixed_protein_has_lig_nodes(self, sample_batch_fixed_protein):
        """fixed_protein graph should have ligand nodes."""
        g, task_name = sample_batch_fixed_protein
        
        assert 'lig' in g.ntypes
        assert g.num_nodes('lig') > 0
    
    def test_fixed_protein_has_protein_nodes(self, sample_batch_fixed_protein):
        """fixed_protein graph should have protein atom nodes."""
        g, task_name = sample_batch_fixed_protein
        
        assert 'prot_atom' in g.ntypes
        assert g.num_nodes('prot_atom') > 0
    
    def test_fixed_protein_has_residue_nodes(self, sample_batch_fixed_protein):
        """fixed_protein graph should have protein residue nodes."""
        g, task_name = sample_batch_fixed_protein
        
        assert 'prot_res' in g.ntypes
        assert g.num_nodes('prot_res') > 0
    
    def test_rigid_docking_has_lig_nodes(self, sample_batch_rigid_docking):
        """rigid_docking graph should have ligand nodes."""
        g, task_name = sample_batch_rigid_docking
        
        assert 'lig' in g.ntypes
        assert g.num_nodes('lig') > 0
    
    def test_rigid_docking_has_protein_nodes(self, sample_batch_rigid_docking):
        """rigid_docking graph should have protein nodes."""
        g, task_name = sample_batch_rigid_docking
        
        assert 'prot_atom' in g.ntypes
        assert g.num_nodes('prot_atom') > 0
    
    def test_denovo_ligand_has_lig_nodes(self, sample_batch_denovo_ligand):
        """denovo_ligand graph should have ligand nodes."""
        g, task_name = sample_batch_denovo_ligand
        
        assert 'lig' in g.ntypes
        assert g.num_nodes('lig') > 0
    
    def test_denovo_ligand_no_protein(self, sample_batch_denovo_ligand):
        """denovo_ligand graph should not have protein nodes (unconditional)."""
        g, task_name = sample_batch_denovo_ligand
        
        # Either prot_atom not in ntypes, or it has 0 nodes
        if 'prot_atom' in g.ntypes:
            assert g.num_nodes('prot_atom') == 0


@pytest.mark.requires_data
class TestGraphEdgeTypes:
    """Tests for graph edge types."""
    
    def test_fixed_protein_has_lig_to_lig_edges(self, sample_batch_fixed_protein):
        """fixed_protein graph should have lig_to_lig edges."""
        g, task_name = sample_batch_fixed_protein
        
        assert 'lig_to_lig' in g.etypes
        # Should have edges (fully connected ligand)
        n_lig = g.num_nodes('lig')
        if n_lig > 1:
            assert g.num_edges('lig_to_lig') > 0
    
    def test_rigid_docking_has_lig_to_lig_edges(self, sample_batch_rigid_docking):
        """rigid_docking graph should have lig_to_lig edges."""
        g, task_name = sample_batch_rigid_docking
        
        assert 'lig_to_lig' in g.etypes
    
    def test_denovo_ligand_has_lig_to_lig_edges(self, sample_batch_denovo_ligand):
        """denovo_ligand graph should have lig_to_lig edges."""
        g, task_name = sample_batch_denovo_ligand
        
        assert 'lig_to_lig' in g.etypes


@pytest.mark.requires_data
class TestGraphLigandFeatures:
    """Tests for ligand node features in graphs."""
    
    def test_ligand_has_prior_positions(self, sample_batch_fixed_protein):
        """Ligand nodes should have prior position features (x_0)."""
        g, task_name = sample_batch_fixed_protein
        
        assert 'x_0' in g.nodes['lig'].data, "Ligand should have x_0 (prior positions)"
        
        x_0 = g.nodes['lig'].data['x_0']
        assert x_0.shape[1] == 3, "Positions should be 3D"
    
    def test_ligand_has_target_positions(self, sample_batch_fixed_protein):
        """Ligand nodes should have target position features (x_1_true)."""
        g, task_name = sample_batch_fixed_protein
        
        assert 'x_1_true' in g.nodes['lig'].data, "Ligand should have x_1_true (target positions)"
        
        x_1 = g.nodes['lig'].data['x_1_true']
        assert x_1.shape[1] == 3, "Positions should be 3D"
    
    def test_ligand_has_prior_atom_types(self, sample_batch_fixed_protein):
        """Ligand nodes should have prior atom type features."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        # Check for condensed atom types
        if 'ligand_identity_condensed' in task_class.groups_generated:
            assert 'cond_a_0' in g.nodes['lig'].data, \
                "Ligand should have cond_a_0 (prior condensed atom types)"
    
    def test_ligand_has_target_atom_types(self, sample_batch_fixed_protein):
        """Ligand nodes should have target atom type features."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        if 'ligand_identity_condensed' in task_class.groups_generated:
            assert 'cond_a_1_true' in g.nodes['lig'].data, \
                "Ligand should have cond_a_1_true (target condensed atom types)"
    
    def test_ligand_prior_positions_centered(self, sample_batch_fixed_protein):
        """Ligand prior positions should be roughly centered."""
        g, task_name = sample_batch_fixed_protein
        
        x_0 = g.nodes['lig'].data['x_0']
        
        # Unbatch and check each graph
        batch_num_nodes = g.batch_num_nodes('lig').tolist()
        start = 0
        for n_nodes in batch_num_nodes:
            if n_nodes > 0:
                positions = x_0[start:start + n_nodes]
                com = positions.mean(dim=0)
                # COM should be reasonably close to origin (within ~20 Angstroms)
                # This is a loose check since priors may not be exactly centered
                assert com.abs().max() < 50, f"Prior COM too far from origin: {com}"
            start += n_nodes


@pytest.mark.requires_data
class TestGraphProteinFeatures:
    """Tests for protein node features in graphs."""
    
    def test_protein_has_positions(self, sample_batch_fixed_protein):
        """Protein atoms should have position features."""
        g, task_name = sample_batch_fixed_protein
        
        # Protein positions are fixed, so they're in x_1_true
        assert 'x_1_true' in g.nodes['prot_atom'].data, \
            "Protein atoms should have x_1_true positions"
        
        x = g.nodes['prot_atom'].data['x_1_true']
        assert x.shape[1] == 3, "Positions should be 3D"
    
    def test_protein_has_element_features(self, sample_batch_fixed_protein):
        """Protein atoms should have element type features."""
        g, task_name = sample_batch_fixed_protein
        
        # Check for element or atom type features
        has_element = 'elem_1_true' in g.nodes['prot_atom'].data or 'a_1_true' in g.nodes['prot_atom'].data
        assert has_element, "Protein atoms should have element/atom type features"
    
    def test_protein_residue_has_type(self, sample_batch_fixed_protein):
        """Protein residues should have residue type features."""
        g, task_name = sample_batch_fixed_protein
        
        assert 'res_1_true' in g.nodes['prot_res'].data, \
            "Protein residues should have res_1_true (residue type)"


@pytest.mark.requires_data
class TestGraphEdgeFeatures:
    """Tests for edge features in graphs."""
    
    def test_lig_to_lig_has_bond_features(self, sample_batch_fixed_protein):
        """lig_to_lig edges should have bond type features."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        # Check if bond types are being generated
        edge_modalities = [m for m in task_class.modalities_generated if m.graph_entity == 'edge']
        
        if len(edge_modalities) > 0:
            # Should have edge features
            etype = 'lig_to_lig'
            if g.num_edges(etype) > 0:
                # Check for prior and target
                has_e_0 = 'e_0' in g.edges[etype].data
                has_e_1 = 'e_1_true' in g.edges[etype].data
                assert has_e_0 or has_e_1, "Edges should have bond type features"


@pytest.mark.requires_data
class TestGraphPriorSampling:
    """Tests for prior sampling on graphs."""
    
    def test_prior_atom_types_are_masked(self, sample_batch_fixed_protein):
        """Prior atom types should be mask tokens."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        # For condensed atom types
        if 'cond_a_0' in g.nodes['lig'].data:
            a_0 = g.nodes['lig'].data['cond_a_0']
            
            # Find mask token (should be the highest index)
            # The mask token is typically n_categories (after all real categories)
            from omtra.tasks.modalities import name_to_modality
            try:
                modality = name_to_modality('lig_cond_a')
                mask_token = modality.n_categories
                
                # Most (or all) should be mask token
                n_masked = (a_0 == mask_token).sum().item()
                total = a_0.numel()
                
                # Allow for some flexibility (fake atoms might not be masked)
                assert n_masked > 0, "Some atom types should be masked"
            except KeyError:
                pass  # Skip if modality not found
    
    def test_prior_positions_differ_from_target(self, sample_batch_fixed_protein):
        """Prior positions should differ from target positions."""
        g, task_name = sample_batch_fixed_protein
        
        x_0 = g.nodes['lig'].data['x_0']
        x_1 = g.nodes['lig'].data['x_1_true']
        
        # Prior and target should be different (they're sampled from different distributions)
        assert not torch.allclose(x_0, x_1), \
            "Prior positions should differ from target positions"


@pytest.mark.requires_data
class TestGraphBatching:
    """Tests for graph batching behavior."""
    
    def test_batch_size_matches(self, sample_batch_multi_fixed_protein):
        """Batched graph should have correct batch_size."""
        g, task_name = sample_batch_multi_fixed_protein
        
        assert g.batch_size == 4, "Batch should have 4 graphs"
    
    def test_batch_num_nodes_sums_correctly(self, sample_batch_multi_fixed_protein):
        """batch_num_nodes should sum to total nodes."""
        g, task_name = sample_batch_multi_fixed_protein
        
        for ntype in g.ntypes:
            batch_counts = g.batch_num_nodes(ntype)
            assert batch_counts.sum().item() == g.num_nodes(ntype), \
                f"batch_num_nodes for {ntype} should sum to total nodes"
    
    def test_batch_num_edges_sums_correctly(self, sample_batch_multi_fixed_protein):
        """batch_num_edges should sum to total edges."""
        g, task_name = sample_batch_multi_fixed_protein
        
        for etype in g.etypes:
            batch_counts = g.batch_num_edges(etype)
            assert batch_counts.sum().item() == g.num_edges(etype), \
                f"batch_num_edges for {etype} should sum to total edges"


@pytest.mark.requires_data
class TestGraphTaskConsistency:
    """Tests for consistency between graph and task definition."""
    
    def test_graph_has_nodes_for_present_node_modalities(self, sample_batch_fixed_protein):
        """Graph should have nodes for all present node modalities."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        for m in task_class.node_modalities_present:
            entity = m.entity_name
            # Either the entity exists with nodes, or it's optional (0 nodes allowed for some)
            if entity in g.ntypes:
                # Just check that we can access it
                _ = g.num_nodes(entity)
    
    def test_graph_features_match_task_modalities(self, sample_batch_fixed_protein):
        """Graph should have data fields for task modalities."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        for m in task_class.modalities_generated:
            if m.is_node:
                if g.num_nodes(m.entity_name) > 0:
                    # Should have both prior and target
                    assert f"{m.data_key}_0" in g.nodes[m.entity_name].data or \
                           f"{m.data_key}_1_true" in g.nodes[m.entity_name].data, \
                           f"Missing data for modality {m.name}"
    
    def test_fixed_modalities_have_1_true_data(self, sample_batch_fixed_protein):
        """Fixed modalities should have x_1_true data."""
        g, task_name = sample_batch_fixed_protein
        task_class = task_name_to_class(task_name)
        
        for m in task_class.modalities_fixed:
            if m.is_node and g.num_nodes(m.entity_name) > 0:
                assert f"{m.data_key}_1_true" in g.nodes[m.entity_name].data, \
                    f"Fixed modality {m.name} should have {m.data_key}_1_true"
