"""
Integration tests for MultiTaskDataModule.
These tests require real Plinder data to be available.
"""

import pytest
import dgl

from omtra.load.quick import datamodule_from_config
from omtra.dataset.data_module import MultiTaskDataModule


@pytest.mark.requires_data
class TestDataModuleInstantiation:
    """Tests for MultiTaskDataModule instantiation."""
    
    def test_datamodule_instantiates(self, hydra_cfg):
        """DataModule should instantiate without errors."""
        datamodule = datamodule_from_config(hydra_cfg)
        assert datamodule is not None
        assert isinstance(datamodule, MultiTaskDataModule)
    
    def test_datamodule_has_expected_attributes(self, hydra_cfg):
        """DataModule should have expected configuration attributes."""
        datamodule = datamodule_from_config(hydra_cfg)
        
        assert hasattr(datamodule, 'td_coupling')
        assert hasattr(datamodule, 'graph_config')
        assert hasattr(datamodule, 'prior_config')
        assert hasattr(datamodule, 'edges_per_batch')


@pytest.mark.requires_data
class TestDataModuleLoadDataset:
    """Tests for MultiTaskDataModule.load_dataset."""
    
    def test_load_train_dataset(self, hydra_cfg):
        """Should be able to load a train MultitaskDataSet."""
        datamodule = datamodule_from_config(hydra_cfg)
        train_dataset = datamodule.load_dataset('train')
        
        assert train_dataset is not None
        assert hasattr(train_dataset, 'datasets')
        assert 'plinder' in train_dataset.datasets
    
    def test_load_val_dataset(self, hydra_cfg):
        """Should be able to load a val MultitaskDataSet."""
        datamodule = datamodule_from_config(hydra_cfg)
        val_dataset = datamodule.load_dataset('val')
        
        assert val_dataset is not None
        assert hasattr(val_dataset, 'datasets')
    
    def test_dataset_has_task_space(self, hydra_cfg):
        """MultitaskDataSet should have a task space."""
        datamodule = datamodule_from_config(hydra_cfg)
        dataset = datamodule.load_dataset('train')
        
        assert hasattr(dataset, 'task_space')
        assert len(dataset.task_space) > 0
    
    def test_dataset_has_dataset_space(self, hydra_cfg):
        """MultitaskDataSet should have a dataset space."""
        datamodule = datamodule_from_config(hydra_cfg)
        dataset = datamodule.load_dataset('train')
        
        assert hasattr(dataset, 'dataset_space')
        assert len(dataset.dataset_space) > 0


@pytest.mark.requires_data
class TestDataModuleGetItem:
    """Tests for retrieving items from the MultitaskDataSet."""
    
    def test_getitem_returns_graph(self, hydra_cfg, test_task_name):
        """__getitem__ should return a DGL graph."""
        datamodule = datamodule_from_config(hydra_cfg)
        dataset = datamodule.load_dataset('train')
        
        # Get the plinder dataset for this task
        from omtra.tasks.register import task_name_to_class
        task_class = task_name_to_class(test_task_name)
        plinder_link_version = task_class.plinder_link_version
        plinder_dataset = dataset.datasets['plinder'][plinder_link_version]
        
        # Get an item
        graph = plinder_dataset[(test_task_name, 0)]
        
        assert isinstance(graph, dgl.DGLGraph)
    
    def test_getitem_graph_has_ligand_nodes(self, hydra_cfg, test_task_name):
        """Returned graph should have ligand nodes."""
        datamodule = datamodule_from_config(hydra_cfg)
        dataset = datamodule.load_dataset('train')
        
        from omtra.tasks.register import task_name_to_class
        task_class = task_name_to_class(test_task_name)
        plinder_link_version = task_class.plinder_link_version
        plinder_dataset = dataset.datasets['plinder'][plinder_link_version]
        
        graph = plinder_dataset[(test_task_name, 0)]
        
        assert 'lig' in graph.ntypes
        assert graph.num_nodes('lig') > 0
    
    def test_getitem_graph_has_protein_nodes(self, hydra_cfg, test_task_name):
        """Returned graph should have protein nodes for protein-conditioned tasks."""
        datamodule = datamodule_from_config(hydra_cfg)
        dataset = datamodule.load_dataset('train')
        
        from omtra.tasks.register import task_name_to_class
        task_class = task_name_to_class(test_task_name)
        plinder_link_version = task_class.plinder_link_version
        plinder_dataset = dataset.datasets['plinder'][plinder_link_version]
        
        graph = plinder_dataset[(test_task_name, 0)]
        
        # For protein-conditioned tasks, should have protein nodes
        if 'protein' in test_task_name:
            assert 'prot_atom' in graph.ntypes
            assert graph.num_nodes('prot_atom') > 0
    
    def test_getitem_multiple_samples(self, hydra_cfg, test_task_name):
        """Should be able to get multiple samples without error."""
        datamodule = datamodule_from_config(hydra_cfg)
        dataset = datamodule.load_dataset('train')
        
        from omtra.tasks.register import task_name_to_class
        task_class = task_name_to_class(test_task_name)
        plinder_link_version = task_class.plinder_link_version
        plinder_dataset = dataset.datasets['plinder'][plinder_link_version]
        
        for idx in range(min(5, len(plinder_dataset))):
            graph = plinder_dataset[(test_task_name, idx)]
            assert graph is not None
            assert isinstance(graph, dgl.DGLGraph)


@pytest.mark.requires_data
class TestMultitaskDataSetGetItem:
    """Tests for the MultitaskDataSet __getitem__ interface."""
    
    def test_multitask_getitem_format(self, hydra_cfg):
        """MultitaskDataSet __getitem__ should accept (task_idx, dataset_idx, local_idx)."""
        datamodule = datamodule_from_config(hydra_cfg)
        dataset = datamodule.load_dataset('train')
        
        # Get indices for first task and first dataset
        task_idx = 0
        dataset_idx = dataset.dataset_space.index('plinder')
        local_idx = 0
        
        result = dataset[(task_idx, dataset_idx, local_idx)]
        
        # Should return (graph, task_name, dataset_name)
        assert len(result) == 3
        graph, task_name, dataset_name = result
        
        assert isinstance(graph, dgl.DGLGraph)
        assert isinstance(task_name, str)
        assert isinstance(dataset_name, str)
    
    def test_multitask_getitem_task_name_matches(self, hydra_cfg):
        """Returned task_name should match the requested task."""
        datamodule = datamodule_from_config(hydra_cfg)
        dataset = datamodule.load_dataset('train')
        
        task_idx = 0
        expected_task_name = dataset.task_space[task_idx]
        dataset_idx = dataset.dataset_space.index('plinder')
        local_idx = 0
        
        graph, task_name, dataset_name = dataset[(task_idx, dataset_idx, local_idx)]
        
        assert task_name == expected_task_name
    
    def test_multitask_getitem_dataset_name_matches(self, hydra_cfg):
        """Returned dataset_name should match the requested dataset."""
        datamodule = datamodule_from_config(hydra_cfg)
        dataset = datamodule.load_dataset('train')
        
        task_idx = 0
        dataset_idx = dataset.dataset_space.index('plinder')
        local_idx = 0
        
        graph, task_name, dataset_name = dataset[(task_idx, dataset_idx, local_idx)]
        
        assert dataset_name == 'plinder'