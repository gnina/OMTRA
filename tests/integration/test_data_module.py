import os
import zarr
import torch
import pytest
import numpy as np
from pathlib import Path
from collections import defaultdict

from hydra import initialize, compose
from hydra.utils import instantiate

import torch.multiprocessing as mp
import multiprocessing

from omtra.utils.graph import build_lookup_table

def create_dummy_zarr_store(tmp_path: Path, split: str) -> Path:
    """
    Create a minimal dummy Zarr store for specified `split`. 
    This mimics the basic structure expected by the data module
        - Currently this code follows the example in notebooks/zarr_dev.ipynb
    """
    store_path = f"{tmp_path}/{split}.zarr"

    n_molecules = 3000
    unbatched_molecules = defaultdict(list)
    for _ in range(n_molecules):
        n_atoms = np.random.randint(5, 15)
        n_edges = np.random.randint(1, int(n_atoms*0.4))
        n_pharm_nodes = np.random.randint(4, 8)
        x = np.random.randn(n_atoms, 3) # positions
        a = np.random.randint(0, 5, size=n_atoms) # atom types
        c = np.random.randint(0, 5, size=n_atoms) # atom charges
        edge_idxs = np.random.randint(0, n_atoms, size=(n_edges, 2)) # edge indicies for bonds
        e = np.random.randint(1, 4, size=n_edges) # bond orders

        x_pharm = np.random.randn(n_pharm_nodes, 3) # positions
        a_pharm = np.random.randint(0, 5, size=n_pharm_nodes) # atom types

        unbatched_molecules['x'].append(x)
        unbatched_molecules['a'].append(a)
        unbatched_molecules['c'].append(c)
        unbatched_molecules['edge_index'].append(edge_idxs)
        unbatched_molecules['e'].append(e)
        unbatched_molecules['x_pharm'].append(x_pharm)
        unbatched_molecules['a_pharm'].append(a_pharm)


    # now batch the molecules together! there are a few steps here

    # first we need to record the number of nodes and edges in each molecule
    batch_num_nodes = [x.shape[0] for x in unbatched_molecules['x']]
    batch_num_pharm_nodes = [x.shape[0] for x in unbatched_molecules['x_pharm']]
    batch_num_edges = [eidxs.shape[0] for eidxs in unbatched_molecules['edge_index']]

    # convert batch data to numpy arrays
    batch_num_nodes = np.array(batch_num_nodes)
    batch_num_edges = np.array(batch_num_edges)
    batch_num_pharm_nodes = np.array(batch_num_pharm_nodes)

    # concatenate all the data together
    x = np.concatenate(unbatched_molecules['x'], axis=0)
    a = np.concatenate(unbatched_molecules['a'], axis=0)
    c = np.concatenate(unbatched_molecules['c'], axis=0)
    x_pharm = np.concatenate(unbatched_molecules['x_pharm'], axis=0)
    a_pharm = np.concatenate(unbatched_molecules['a_pharm'], axis=0)

    edge_index = np.concatenate(unbatched_molecules['edge_index'], axis=0)
    e = np.concatenate(unbatched_molecules['e'], axis=0)


    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's node features
    node_lookup = build_lookup_table(batch_num_nodes)

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's edge features
    edge_lookup = build_lookup_table(batch_num_edges)

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's pharmacophore node features
    pharm_node_lookup = build_lookup_table(batch_num_pharm_nodes)

    # print("batch_num_nodes:", batch_num_nodes)
    # print("batch_num_edges:", batch_num_edges)
    print("Shape of x:", x.shape)
    print("Shape of a:", a.shape)
    print("Shape of e:", e.shape)
    print("Shape of c:", c.shape)
    print("Shape of x_pharm:", x_pharm.shape)
    print("Shape of a_pharm:", a_pharm.shape)
    print("Shape of edge_index:", edge_index.shape)
    print("Shape of node_lookup:", node_lookup.shape)
    print("Shape of edge_lookup:", edge_lookup.shape)
    print("Shape of pharm_node_lookup:", pharm_node_lookup.shape)

    # store = zarr.storage.MemoryStore()
    graphs_per_chunk = 500 # very important parameter

    store = zarr.storage.LocalStore(store_path)

    # Create a root group
    root = zarr.group(store=store)

    ntypes = ['lig', 'pharm']

    ntype_groups = {}
    for ntype in ntypes:
        ntype_groups[ntype] = root.create_group(ntype)


    lig_node = ntype_groups['lig'].create_group('node')
    lig_edge_data = ntype_groups['lig'].create_group('edge')
    pharm_node_data = ntype_groups['pharm'].create_group('node')

    # Store tensors under different keys with specified chunk sizes

    # some simple heuristics to decide chunk sizes for node and edge data
    mean_lig_nodes_per_graph = int(np.mean(batch_num_nodes))
    mean_ll_edges_per_graph = int(np.mean(batch_num_edges))
    mean_pharm_nodes_per_graph = int(np.mean([x.shape[0] for x in unbatched_molecules['x_pharm']]))
    nodes_per_chunk = graphs_per_chunk * mean_lig_nodes_per_graph
    ll_edges_per_chunk = graphs_per_chunk * mean_ll_edges_per_graph
    pharm_nodes_per_chunk = graphs_per_chunk * mean_pharm_nodes_per_graph

    # create arrays for node data
    lig_node.create_array('x', shape=x.shape, chunks=(nodes_per_chunk, 3), dtype=x.dtype)
    lig_node.create_array('a', shape=a.shape, chunks=(nodes_per_chunk,), dtype=a.dtype)
    lig_node.create_array('c', shape=c.shape, chunks=(nodes_per_chunk,), dtype=c.dtype)

    # create arrays for pharmacophore node data
    pharm_node_data.create_array('x', shape=x_pharm.shape, chunks=(pharm_nodes_per_chunk, 3), dtype=x_pharm.dtype)
    pharm_node_data.create_array('a', shape=a_pharm.shape, chunks=(pharm_nodes_per_chunk,), dtype=a_pharm.dtype)
    pharm_node_data.create_array('graph_lookup', shape=pharm_node_lookup.shape, chunks=pharm_node_lookup.shape, dtype=pharm_node_lookup.dtype)

    # create arrays for edge data
    lig_edge_data.create_array('e', shape=e.shape, chunks=(ll_edges_per_chunk,), dtype=e.dtype)
    lig_edge_data.create_array('edge_index', shape=edge_index.shape, chunks=(ll_edges_per_chunk, 2), dtype=edge_index.dtype)

    # because node_lookup and edge_lookup are relatively small, we may get away with not chunking them
    lig_node.create_array('graph_lookup', shape=node_lookup.shape, chunks=node_lookup.shape, dtype=node_lookup.dtype)
    lig_edge_data.create_array('graph_lookup', shape=edge_lookup.shape, chunks=edge_lookup.shape, dtype=edge_lookup.dtype)

    # write data to the arrays
    lig_node['x'][:] = x
    lig_node['a'][:] = a
    lig_node['c'][:] = c
    lig_edge_data['e'][:] = e
    lig_edge_data['edge_index'][:] = edge_index
    lig_node['graph_lookup'][:] = node_lookup
    lig_edge_data['graph_lookup'][:] = edge_lookup
    pharm_node_data['x'][:] = x_pharm
    pharm_node_data['a'][:] = a_pharm
    pharm_node_data['graph_lookup'][:] = pharm_node_lookup

    root.tree()
    
    return tmp_path

def create_dummy_data_dir(tmp_path: Path) -> Path:
    """
    Create a dummy data directory with both 'train.zarr' and 'val.zarr'
    so that the data module can find both splits.
    """
    # Create a directory to hold the dummy data.
    data_dir = tmp_path / "dummy_data"
    data_dir.mkdir()
    
    # Create both train and val Zarr stores.
    create_dummy_zarr_store(data_dir, "train")
    create_dummy_zarr_store(data_dir, "val")

    return data_dir

@pytest.fixture
def dummy_data_dir(tmp_path: Path) -> Path:
    """
    Pytest fixture to create a dummy data directory containing both train and val zarr stores.
    """
    return create_dummy_data_dir(tmp_path)

@pytest.fixture
def dummy_config(dummy_data_dir: Path):
    """
    Compose a Hydra config that overrides the default processed data path in `config_to_test`
    This makes sure that it points to the newly created temp dummy zarr store.
    """
    config_to_test = "test_config"

    # Initialize Hydra with the test configuration path.
    with initialize(config_path="../../configs", job_name="test_omtra", version_base="1.3"):
        cfg = compose(
            config_name=config_to_test,
            overrides=[f"pharmit_path={dummy_data_dir}"]
        )
    return cfg

def test_data_module_instantiation(dummy_config):
    """
    Integration test for the data module:
      1. Instantiate the data module using the dummy Hydra configuration we just made.
      2. Call setup to build the training and validation datasets.
      3. Retrieve the train+val dataloader and fetch a batch.
      4. TODO: Assert that the batch is in the expected format
    """
    from omtra.dataset.data_module import MultiTaskDataModule
    
    multiprocessing.set_start_method('spawn', force=True)
    mp.set_start_method("spawn", force=True)

    # Instantiate the data module
    datamodule: MultiTaskDataModule = instantiate(
        dummy_config.task_group.data, 
        graph_config=dummy_config.graph
    )

    # Run setup
    datamodule.setup(stage="fit")
    
    # Retrieve the dataloaders
    train_dataloader = datamodule.train_dataloader()
    val_dataloader   = datamodule.val_dataloader()

    train_dataloader_iter = iter(train_dataloader)
    val_dataloader_iter = iter(val_dataloader)

    n_batches = 5
    for _ in range(n_batches):
        g, task_name, dataset_name = next(train_dataloader_iter)

    for _ in range(n_batches):
        g, task_name, dataset_name = next(val_dataloader_iter)