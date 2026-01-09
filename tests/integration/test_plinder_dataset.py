"""
Integration tests for PlinderDataset.
These tests require real plinder data to be available.
"""

import pytest
import torch
import dgl


@pytest.mark.requires_data
class TestPlinderDatasetInit:
    """Tests for PlinderDataset initialization."""
    
    def test_dataset_initializes(self, plinder_dataset):
        """Dataset should initialize without errors."""
        assert plinder_dataset is not None
        assert len(plinder_dataset) > 0
    
    def test_dynamic_crop_disabled_for_val_split(self, plinder_dataset_factory):
        """Cropping should be disabled for non-train splits."""
        dataset = plinder_dataset_factory(
            split="val",
            crop_min_distance=5.0,
            crop_max_distance=10.0
        )
        assert dataset.dynamic_crop is False
    
    def test_dynamic_crop_enabled_for_train_with_params(self, plinder_dataset_factory):
        """Cropping should be enabled when both params set and split is train."""
        dataset = plinder_dataset_factory(
            split="train",
            crop_min_distance=5.0,
            crop_max_distance=10.0
        )
        assert dataset.dynamic_crop is True
    
    def test_dynamic_crop_disabled_with_missing_min(self, plinder_dataset_factory):
        """Cropping should be disabled when min distance is None."""
        dataset = plinder_dataset_factory(
            split="train",
            crop_min_distance=None,
            crop_max_distance=10.0
        )
        assert dataset.dynamic_crop is False
    
    def test_dynamic_crop_disabled_with_missing_max(self, plinder_dataset_factory):
        """Cropping should be disabled when max distance is None."""
        dataset = plinder_dataset_factory(
            split="train",
            crop_min_distance=5.0,
            crop_max_distance=None
        )
        assert dataset.dynamic_crop is False


@pytest.mark.requires_data
class TestPlinderDatasetGetItem:
    """Tests for PlinderDataset.__getitem__."""
    
    def test_getitem_returns_dgl_graph(self, plinder_dataset, test_task_name):
        """__getitem__ should return a DGL heterograph."""
        idx = 0
        graph = plinder_dataset[(test_task_name, idx)]
        
        assert isinstance(graph, dgl.DGLGraph)
    
    def test_getitem_has_expected_node_types(self, plinder_dataset, test_task_name):
        """Returned graph should have expected node types."""
        idx = 0
        graph = plinder_dataset[(test_task_name, idx)]
        
        ntypes = graph.ntypes
        assert "lig" in ntypes
        assert "prot_atom" in ntypes
        assert "prot_res" in ntypes
    
    def test_getitem_has_ligand_data(self, plinder_dataset, test_task_name):
        """Ligand nodes should have expected data fields."""
        idx = 0
        graph = plinder_dataset[(test_task_name, idx)]
        
        lig_data = graph.nodes["lig"].data
        assert "x_1_true" in lig_data or "x_1" in lig_data
    
    def test_getitem_multiple_samples(self, plinder_dataset, test_task_name):
        """Should be able to get multiple samples without error."""
        for idx in range(min(5, len(plinder_dataset))):
            graph = plinder_dataset[(test_task_name, idx)]
            assert graph is not None
    
    def test_getitem_with_cropping(self, plinder_dataset_with_cropping, test_task_name):
        """__getitem__ should work with cropping enabled."""
        idx = 0
        graph = plinder_dataset_with_cropping[(test_task_name, idx)]
        
        assert isinstance(graph, dgl.DGLGraph)
        assert graph.num_nodes("lig") > 0
        assert graph.num_nodes("prot_atom") > 0


@pytest.mark.requires_data
class TestPlinderDatasetCropping:
    """Tests for dynamic cropping functionality."""
    
    def test_cropping_produces_valid_graphs(self, plinder_dataset_with_cropping, test_task_name):
        """Cropped datasets should still produce valid graphs."""
        for idx in range(min(10, len(plinder_dataset_with_cropping))):
            graph = plinder_dataset_with_cropping[(test_task_name, idx)]
            
            # Graph should have nodes
            assert graph.num_nodes("lig") > 0
            assert graph.num_nodes("prot_atom") >= 0  # Could be 0 if fallback
            
            # Ligand data should be intact
            assert "x_1_true" in graph.nodes["lig"].data or "x_0" in graph.nodes["lig"].data
    
    def test_tight_cropping_reduces_pocket_size(self, plinder_dataset_factory, test_task_name):
        """Tighter crop distance should generally result in fewer protein atoms."""
        ds_no_crop = plinder_dataset_factory(split="train")
        ds_tight_crop = plinder_dataset_factory(
            split="train",
            crop_min_distance=3.0,
            crop_max_distance=4.0,
        )
        
        # Compare first few samples
        tight_smaller_count = 0
        for idx in range(min(10, len(ds_no_crop))):
            g_no_crop = ds_no_crop[(test_task_name, idx)]
            g_cropped = ds_tight_crop[(test_task_name, idx)]
            
            n_no_crop = g_no_crop.num_nodes("prot_atom")
            n_cropped = g_cropped.num_nodes("prot_atom")
            
            if n_cropped <= n_no_crop:
                tight_smaller_count += 1
        
        # Most samples should have smaller or equal pocket after cropping
        assert tight_smaller_count >= 8, "Cropping should generally reduce pocket size"
    
    def test_get_system_with_cropping(self, plinder_dataset_with_cropping):
        """get_system should work correctly with cropping."""
        system = plinder_dataset_with_cropping.get_system(
            index=0,
            include_pharmacophore=False,
            include_protein=True,
            include_extra_feats=False,
            condensed_atom_typing=True,
        )
        
        assert system is not None
        assert system.ligand is not None
        assert system.pocket is not None
        # Pocket should have data
        assert len(system.pocket.coords) > 0
