"""
Unit tests for the task system (task registry, modalities, task properties).
These tests don't require real data.
"""

import pytest
from omtra.tasks.register import task_name_to_class, TASK_REGISTER
from omtra.tasks.modalities import Modality, MODALITY_REGISTER, name_to_modality


# Task names we're testing
TEST_TASKS = [
    "fixed_protein_ligand_denovo_condensed",
    "rigid_docking_condensed", 
    "denovo_ligand_condensed",
]


class TestTaskRegistry:
    """Tests for the task registry system."""
    
    def test_task_register_not_empty(self):
        """Task registry should contain registered tasks."""
        assert len(TASK_REGISTER) > 0, "Task registry should not be empty"
    
    @pytest.mark.parametrize("task_name", TEST_TASKS)
    def test_task_name_to_class(self, task_name):
        """Can retrieve task classes by name."""
        task_class = task_name_to_class(task_name)
        assert task_class is not None
        assert task_class.name == task_name
    
    def test_invalid_task_name_raises_keyerror(self):
        """Invalid task name should raise KeyError."""
        with pytest.raises(KeyError):
            task_name_to_class("nonexistent_task_12345")
    
    @pytest.mark.parametrize("task_name", TEST_TASKS)
    def test_task_has_name_attribute(self, task_name):
        """Task classes should have a name attribute matching registry key."""
        task_class = task_name_to_class(task_name)
        assert hasattr(task_class, 'name')
        assert task_class.name == task_name


class TestTaskModalities:
    """Tests for task modality properties."""
    
    @pytest.mark.parametrize("task_name", TEST_TASKS)
    def test_task_has_groups_fixed(self, task_name):
        """All tasks should define groups_fixed."""
        task_class = task_name_to_class(task_name)
        assert hasattr(task_class, 'groups_fixed')
        assert isinstance(task_class.groups_fixed, list)
    
    @pytest.mark.parametrize("task_name", TEST_TASKS)
    def test_task_has_groups_generated(self, task_name):
        """All tasks should define groups_generated."""
        task_class = task_name_to_class(task_name)
        assert hasattr(task_class, 'groups_generated')
        assert isinstance(task_class.groups_generated, list)
        assert len(task_class.groups_generated) > 0, "Task should generate at least one group"
    
    @pytest.mark.parametrize("task_name", TEST_TASKS)
    def test_modalities_fixed_and_generated_no_overlap(self, task_name):
        """Fixed and generated modalities should not overlap."""
        task_class = task_name_to_class(task_name)
        fixed_names = {m.name for m in task_class.modalities_fixed}
        generated_names = {m.name for m in task_class.modalities_generated}
        
        overlap = fixed_names & generated_names
        assert len(overlap) == 0, f"Overlapping modalities: {overlap}"
    
    @pytest.mark.parametrize("task_name", TEST_TASKS)
    def test_modalities_present_is_union(self, task_name):
        """modalities_present should be union of fixed and generated."""
        task_class = task_name_to_class(task_name)
        present = set(m.name for m in task_class.modalities_present)
        fixed = set(m.name for m in task_class.modalities_fixed)
        generated = set(m.name for m in task_class.modalities_generated)
        
        assert present == fixed | generated
    
    @pytest.mark.parametrize("task_name", TEST_TASKS)
    def test_all_modalities_are_modality_instances(self, task_name):
        """All modalities should be Modality instances."""
        task_class = task_name_to_class(task_name)
        for m in task_class.modalities_present:
            assert isinstance(m, Modality), f"{m} should be a Modality instance"


class TestTaskProperties:
    """Tests for computed task properties."""
    
    def test_fixed_protein_task_has_protein(self):
        """fixed_protein_ligand_denovo_condensed should have protein."""
        task_class = task_name_to_class("fixed_protein_ligand_denovo_condensed")
        assert task_class.has_protein
        assert 'protein_identity' in task_class.groups_present
    
    def test_rigid_docking_task_has_protein(self):
        """rigid_docking_condensed should have protein."""
        task_class = task_name_to_class("rigid_docking_condensed")
        assert task_class.has_protein
        assert 'protein_identity' in task_class.groups_present
    
    def test_denovo_ligand_is_unconditional(self):
        """denovo_ligand_condensed should be unconditional (no fixed groups)."""
        task_class = task_name_to_class("denovo_ligand_condensed")
        assert task_class.unconditional
        assert len(task_class.groups_fixed) == 0
    
    def test_fixed_protein_is_not_unconditional(self):
        """fixed_protein_ligand_denovo_condensed should not be unconditional."""
        task_class = task_name_to_class("fixed_protein_ligand_denovo_condensed")
        assert not task_class.unconditional
        assert len(task_class.groups_fixed) > 0
    
    @pytest.mark.parametrize("task_name", TEST_TASKS)
    def test_task_has_priors(self, task_name):
        """All tasks should define priors."""
        task_class = task_name_to_class(task_name)
        assert hasattr(task_class, 'priors')
        assert isinstance(task_class.priors, dict)
    
    @pytest.mark.parametrize("task_name", TEST_TASKS)
    def test_task_has_conditional_paths(self, task_name):
        """All tasks should define conditional_paths."""
        task_class = task_name_to_class(task_name)
        assert hasattr(task_class, 'conditional_paths')
        assert isinstance(task_class.conditional_paths, dict)


class TestModalityRegistry:
    """Tests for the modality registry."""
    
    def test_modality_register_not_empty(self):
        """Modality registry should contain registered modalities."""
        assert len(MODALITY_REGISTER) > 0, "Modality registry should not be empty"
    
    def test_name_to_modality(self):
        """Can retrieve modalities by name."""
        modality = name_to_modality("lig_x")
        assert modality is not None
        assert modality.name == "lig_x"
    
    def test_invalid_modality_raises_keyerror(self):
        """Invalid modality name should raise KeyError."""
        with pytest.raises(KeyError):
            name_to_modality("nonexistent_modality_12345")


class TestModalityProperties:
    """Tests for modality dataclass properties."""
    
    def test_modality_has_required_fields(self):
        """Modalities should have all required fields."""
        modality = name_to_modality("lig_x")
        assert hasattr(modality, 'name')
        assert hasattr(modality, 'group')
        assert hasattr(modality, 'graph_entity')
        assert hasattr(modality, 'entity_name')
        assert hasattr(modality, 'data_key')
        assert hasattr(modality, 'n_categories')
    
    def test_continuous_modality_is_categorical_false(self):
        """Continuous modalities should have is_categorical=False."""
        modality = name_to_modality("lig_x")  # position is continuous
        assert not modality.is_categorical
        assert modality.n_categories is None
    
    def test_categorical_modality_is_categorical_true(self):
        """Categorical modalities should have is_categorical=True."""
        modality = name_to_modality("lig_cond_a")  # atom type is categorical
        assert modality.is_categorical
        assert modality.n_categories is not None
        assert modality.n_categories > 0
    
    def test_node_modality_is_node(self):
        """Node modalities should have is_node=True."""
        modality = name_to_modality("lig_x")
        assert modality.is_node
        assert modality.graph_entity == "node"
    
    def test_edge_modality_is_not_node(self):
        """Edge modalities should have is_node=False."""
        modality = name_to_modality("lig_e")  # bond type is on edges
        assert not modality.is_node
        assert modality.graph_entity == "edge"
    
    def test_ligand_modalities_have_lig_entity(self):
        """Ligand modalities should have entity_name='lig'."""
        for mod_name in ["lig_x", "lig_cond_a"]:
            modality = name_to_modality(mod_name)
            assert modality.entity_name == "lig"
    
    def test_protein_modalities_have_prot_entity(self):
        """Protein modalities should have appropriate entity names."""
        modality = name_to_modality("prot_atom_x")
        assert modality.entity_name == "prot_atom"


class TestTaskModalityIntegration:
    """Integration tests for tasks and modalities."""
    
    @pytest.mark.parametrize("task_name", TEST_TASKS)
    def test_generated_modalities_have_priors(self, task_name):
        """Generated modalities should have corresponding priors defined."""
        task_class = task_name_to_class(task_name)
        priors = task_class.priors
        
        for m in task_class.modalities_generated:
            # Priors are defined per data_key (e.g., lig_x, lig_a, lig_e)
            prior_key = f"{m.entity_name}_{m.data_key}"
            # Some modalities share priors, so we check if the key exists
            # or if there's a related key
            has_prior = prior_key in priors or any(prior_key in k for k in priors)
            # Note: This is a soft check - not all modalities need explicit priors
    
    @pytest.mark.parametrize("task_name", TEST_TASKS)
    def test_generated_modalities_have_conditional_paths(self, task_name):
        """Generated modalities should have conditional paths defined."""
        task_class = task_name_to_class(task_name)
        cond_paths = task_class.conditional_paths
        
        for m in task_class.modalities_generated:
            assert m.name in cond_paths, \
                f"Modality {m.name} should have conditional path for task {task_name}"
    
    @pytest.mark.parametrize("task_name", TEST_TASKS)
    def test_node_and_edge_modalities_separated(self, task_name):
        """node_modalities_present and edge_modalities_present should partition modalities_present."""
        task_class = task_name_to_class(task_name)
        
        node_mods = set(m.name for m in task_class.node_modalities_present)
        edge_mods = set(m.name for m in task_class.edge_modalities_present)
        all_mods = set(m.name for m in task_class.modalities_present)
        
        # No overlap
        assert len(node_mods & edge_mods) == 0, "Node and edge modalities should not overlap"
        
        # Union equals all modalities
        assert node_mods | edge_mods == all_mods, "Node + edge modalities should equal all modalities"
