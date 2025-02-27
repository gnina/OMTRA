import argparse
from pathlib import Path
import zarr
import pickle
import numpy as np
import time
import os
from typing import List

from tqdm import tqdm
from multiprocessing import Pool

from omtra.utils.zarr_utils import list_zarr_arrays
from omtra.utils.graph import build_lookup_table


def parse_args():
    p = argparse.ArgumentParser(description='Build Zarr store')
    p.add_argument('--output_dir', type=Path, help='Directory for store.',  default=Path('./zarr_store'))
    p.add_argument('--store_name', type=Path, help='Name of store.',  default=Path('store.zarr'))
    
    p.add_argument('--data_dir', type=Path, help='Data directory.',  default=Path('/net/galaxy/home/koes/icd3/moldiff/OMTRA/omtra_pipelines/pharmit_dataset/outputs/phase1/chunk_data'))
    p.add_argument('--info_dir', type=Path, help='Data info directory.',  default=Path('/net/galaxy/home/koes/icd3/moldiff/OMTRA/omtra_pipelines/pharmit_dataset/outputs/phase1/chunk_info'))
    
    p.add_argument('--n_cpus', type=int, default=1, help='Number of CPUs to use for parallel processing.')
    p.add_argument('--batch_size', type=int, default=3, help='Number of data files to batch togther.')
    p.add_argument('--graphs_per_chunk', type=int, default=500, help='Number of graphs to store per chunk for Zarr store.')

    args = p.parse_args()
    return args


class FileCrawler():
    def __init__(self, info_dir: Path, data_dir: Path, batch_size: int):
        self.info_dir = info_dir
        self.data_dir = data_dir
        self.batch_size = batch_size


    def crawl_files(self):
        for file in self.info_dir.iterdir():
            if file.is_file():
                yield file

    def info_batch_generator(self):
        batch = []

        for file in self.crawl_files():
            batch.append(file)
            if len(batch) == self.batch_size:
                yield batch
                batch = []  # Reset the batch

        if batch:  # Remaining items that didn't fill a complete batch
            yield batch
    


class DataDimensions():
    def __init__(self):
        self.mols = []
        self.atoms = []
        self.edges = []
        self.pharm = []
    
    def update_mols(self, mols):
        self.mols.append(mols)
    
    def update_atoms(self, atoms):
        self.atoms.append(atoms)
    
    def update_edges(self, edges):
        self.edges.append(edges)
    
    def update_pharm(self, pharm):
        self.pharm.append(pharm)

    def mol_counts(self):
        return self.mols
    def atom_counts(self):
        return self.atoms
    def edge_counts(self):
        return self.edges
    def pharm_counts(self):
        return self.pharm


def get_info_batched(info_files: list[Path]):
    batch_info = {}
    for info_file in info_files:
        with open(info_file, 'rb') as file:
            info_data = pickle.load(file)
            filename = os.path.basename(info_data['file'])
            batch_info[filename] = {k: v for k, v in info_data.items() if k != 'file'}
    return batch_info


def get_info(info_file: list[Path], data_dims: DataDimensions):
    info = {}
    with open(info_file, 'rb') as file:
        info_data = pickle.load(file)
        filename = os.path.basename(info_data['file'])
        info[filename] = {k: v for k, v in info_data.items() if k != 'file'}


        data_dims.update_mols(info_data['n_mols'])
        data_dims.update_atoms(info_data['n_atoms'])
        data_dims.update_edges(info_data['n_edges'])
        data_dims.update_pharm(info_data['n_pharm'])
    return info


def get_data(data_dir: Path, data_file: Path):
    return np.load(f"{data_dir}/{data_file}")


class ZarrStore():
    def __init__(self, output_dir: Path, graphs_per_chunk: int, data_dims: DataDimensions):
        self.output_dir = output_dir
        self.graphs_per_chunk = graphs_per_chunk

        # Make output directory
        zarr_dir = self.output_dir 
        zarr_dir.mkdir(parents=True, exist_ok=True)

        id = str(int(time.time() * 1000))[-8:]
        filename = f"store{id}.zarr"
        store = zarr.storage.LocalStore(str(self.output_dir / filename))

        # Create a root group
        self.root = zarr.group(store=store)

        ntypes = ['lig', 'db', 'pharm']

        ntype_groups = {}
        for ntype in ntypes:
            ntype_groups[ntype] = self.root.create_group(ntype)


        self.lig_node = ntype_groups['lig'].create_group('node')
        self.lig_edge_data = ntype_groups['lig'].create_group('edge')

        self.pharm_node_data = ntype_groups['pharm'].create_group('node')
        self.db_node_data = ntype_groups['db'].create_group('node')


        batch_num_nodes = data_dims.atom_counts()
        batch_num_edges = data_dims.edge_counts()
        batch_num_pharm_nodes = data_dims.pharm_counts()
        
        # some simple heuristics to decide chunk sizes for node and edge data
        mean_lig_nodes_per_graph = int(np.mean(batch_num_nodes))
        mean_ll_edges_per_graph = int(np.mean(batch_num_edges))
        mean_pharm_nodes_per_graph = int(np.mean(batch_num_pharm_nodes))

        nodes_per_chunk = self.graphs_per_chunk * mean_lig_nodes_per_graph
        ll_edges_per_chunk = self.graphs_per_chunk * mean_ll_edges_per_graph 
        pharm_nodes_per_chunk = self.graphs_per_chunk * mean_pharm_nodes_per_graph

        # Create node arrays
        self.lig_node.create_array('x', shape=(sum(data_dims.atom_counts()), 3), chunks=(nodes_per_chunk,3), dtype=np.float32)
        self.lig_node.create_array('a', shape=(sum(data_dims.atom_counts()), ), chunks=(nodes_per_chunk, ), dtype=np.uint8)
        self.lig_node.create_array('c', shape=(sum(data_dims.atom_counts()), ), chunks=(nodes_per_chunk, ), dtype=np.int32)
        self.lig_node.create_array('graph_lookup', shape=(sum(data_dims.mol_counts()), 2), chunks=(sum(data_dims.mol_counts()), 2), dtype=np.int64)

        # Create edge arrays
        self.lig_edge_data.create_array('e', shape=(sum(data_dims.edge_counts()), ), chunks=(ll_edges_per_chunk, ), dtype=np.int32)
        self.lig_edge_data.create_array('edge_index', shape=(sum(data_dims.edge_counts()),2), chunks=(ll_edges_per_chunk, 2), dtype=np.int64)
        print((sum(data_dims.mol_counts()),2), sum(data_dims.mol_counts()))
        self.lig_edge_data.create_array('graph_lookup', shape=(sum(data_dims.mol_counts()),2), chunks=(sum(data_dims.mol_counts()), 2), dtype=np.int64)

        # Create pharmacophore arrays
        self.pharm_node_data.create_array('x', shape=(sum(data_dims.pharm_counts()), 3), chunks=(pharm_nodes_per_chunk,3), dtype=np.float32)
        self.pharm_node_data.create_array('a', shape=(sum(data_dims.pharm_counts()), ), chunks=(pharm_nodes_per_chunk, ), dtype=np.uint8)
        self.pharm_node_data.create_array('graph_lookup', shape=(sum(data_dims.mol_counts()), 2), chunks=(sum(data_dims.mol_counts()),2), dtype=np.int64)
        
        # Create database array
        self.db_node_data.create_array('db', shape=(20000, 14), chunks=(20000,14), dtype=bool)


    def add_data(self, tensors: dict):
        """
        batch_num_nodes = [x.shape[0] for x in tensors['positions']]
        batch_num_edges = [eidx.shape[0] for eidx in tensors['bond_idxs']]
        batch_num_pharm_nodes = [x.shape[0] for x in tensors['x_pharm']]

        
        x = np.concatenate(tensors['positions'], axis=0)
        a = np.concatenate(tensors['atom_types'], axis=0)
        c = np.concatenate(tensors['atom_charges'], axis=0)
        e = np.concatenate(tensors['bond_types'], axis=0)
        edge_index = np.concatenate(tensors['bond_idxs'], axis=0)

        x_pharm = np.concatenate(tensors['x_pharm'], axis=0)
        a_pharm = np.concatenate(tensors['a_pharm'], axis=0)

        db = tensors['databases']

        # create an array of indicies to keep track of the start_idx and end_idx of each molecule's node features
        node_lookup = build_lookup_table(batch_num_nodes)

        # create an array of indicies to keep track of the start_idx and end_idx of each molecule's edge features
        edge_lookup = build_lookup_table(batch_num_edges)

        # create an array of indicies to keep track of the start_idx and end_idx of each molecule's pharmacophore node features
        pharm_node_lookup = build_lookup_table(batch_num_pharm_nodes)   
        """

        x = tensors['lig_x']
        a = tensors['lig_a']
        c = tensors['lig_c']
        node_lookup = tensors['node_lookup']
        
        e = tensors['lig_e']
        edge_index = tensors['lig_edge_idx']
        edge_lookup = tensors['edge_lookup']

        x_pharm = tensors['pharm_x']
        a_pharm = tensors['pharm_a']
        pharm_node_lookup = tensors['pharm_lookup']

        db = tensors['database']

        def append_data(group, name, data):
            size = group[name].shape[0]  # Get current length of array
            new_size = size + data.shape[0]  # New length of array
            
            # Append new data to the resized section
            group[name][size:new_size] = data

        # Add node data
        append_data(self.lig_node, 'x', x)
        append_data(self.lig_node, 'a', a)
        append_data(self.lig_node, 'c', c)
        append_data(self.lig_node, 'graph_lookup', node_lookup)

        # Add edge data
        append_data(self.lig_edge_data, 'e', e)
        append_data(self.lig_edge_data, 'edge_index', edge_index)
        append_data(self.lig_edge_data, 'graph_lookup', edge_lookup)

        # Add pharmacophore node data
        append_data(self.pharm_node_data, 'x', x_pharm)
        append_data(self.pharm_node_data, 'a', a_pharm)
        append_data(self.pharm_node_data, 'graph_lookup', pharm_node_lookup)

        # Add database node data
        append_data(self.db_node_data, 'db', db)


        """
        batch_num_nodes = [idx[1] - idx[0] for idx in node_lookup]
        batch_num_edges = [idx[1] - idx[0] for idx in edge_lookup]
        batch_num_pharm_nodes = [idx[1] - idx[0] for idx in pharm_node_lookup]
        
        # some simple heuristics to decide chunk sizes for node and edge data
        mean_lig_nodes_per_graph = int(np.mean(batch_num_nodes))
        mean_ll_edges_per_graph = int(np.mean(batch_num_edges))
        mean_pharm_nodes_per_graph = int(np.mean(batch_num_pharm_nodes))

        nodes_per_chunk = self.graphs_per_chunk * mean_lig_nodes_per_graph
        ll_edges_per_chunk = self.graphs_per_chunk * mean_ll_edges_per_graph 
        pharm_nodes_per_chunk = self.graphs_per_chunk * mean_pharm_nodes_per_graph
        


        def create_or_append(group, name, data, chunk_size):
            #Creates a new Zarr array if it doesn't exist. Otherwise, resizes and appends to exisiting array
            if group.get(name) is None:

                group.create_array(name, shape=data.shape, chunks=chunk_size, dtype=data.dtype)
                group[name][:] = data
               
            else:
                size = group[name].shape[0]  # Get current length of array
                new_size = size + data.shape[0]  # New length of array

                
                # Append new data to the resized section
                group[name][size:new_size] = data

        # Add node data
        create_or_append(self.lig_node, 'x', x, (nodes_per_chunk, 3))
        create_or_append(self.lig_node, 'a', a, (nodes_per_chunk, ))
        create_or_append(self.lig_node, 'c', c, (nodes_per_chunk, ))
        create_or_append(self.lig_node, 'graph_lookup', node_lookup, node_lookup.shape)

        # Add edge data
        create_or_append(self.lig_edge_data, 'e', e, (ll_edges_per_chunk, ))
        create_or_append(self.lig_edge_data, 'edge_index', edge_index, (ll_edges_per_chunk, 2))
        create_or_append(self.lig_edge_data, 'graph_lookup', edge_lookup, edge_lookup.shape)

        # Add pharmacophore node data
        create_or_append(self.pharm_node_data, 'x', x_pharm, (pharm_nodes_per_chunk, 3))
        create_or_append(self.pharm_node_data, 'a', a_pharm, (pharm_nodes_per_chunk, ))
        create_or_append(self.pharm_node_data, 'graph_lookup', pharm_node_lookup, pharm_node_lookup.shape)

        # Add database node data
        create_or_append(self.db_node_data, 'db', db, db.shape)
        """

    def display_structure(self):
        print(self.root.tree())
        

def add_data_worker(output_dir, graphs_per_chunk, tensors):
    store = ZarrStore(output_dir, graphs_per_chunk, data_dims)  # Each worker gets a new instance
    store.add_data(tensors)


def run_parallel(n_cpus: int, file_crawler: FileCrawler, process_args: tuple, max_pending: int = None):
    # Set a default limit if not provided
    if max_pending is None:
        max_pending = n_cpus * 2  # adjust this factor as needed

    total_tasks = len(file_crawler)
    pbar = tqdm(total=total_tasks, desc="Processing", unit="chunks")
    # Use a mutable container to track errors.
    error_counter = [0]

    with Pool(processes=n_cpus) as pool:
        pending = []

        for idx, tensors in enumerate(file_crawler):
            """
            if chunk_saver.chunk_processed(chunk_idx):
                pbar.update(1)
                continue
            """

            # Wait until the number of pending jobs is below the threshold.
            # We remove finished tasks from the list, and if still too many remain,
            # we sleep briefly before re-checking.
            while len(pending) >= max_pending:
                # Filter out jobs that have finished
                pending = [r for r in pending if not r.ready()]
                if len(pending) >= max_pending:
                    time.sleep(0.1)  # brief pause before checking again

            """
            # Wrap the original success callback to also update the progress bar.
            callback_fn = partial(save_and_update, idx=idx, pbar=pbar, chunk_saver=chunk_saver)
            # Wrap the error callback to update the progress bar and error counter.
            error_callback_fn = partial(error_and_update, pbar=pbar, error_counter=error_counter)
            """

            # Submit the job and add its AsyncResult to the pending list
            result = pool.apply_async(
                add_data_worker, 
                args=(tensors, *process_args)
            )
            pending.append(result)

        # After submitting all jobs, wait for any remaining tasks to complete.
        for result in pending:
            result.wait()

        pool.close()
        pool.join()


def run_simple(data_info, store):
    for data_file, info in data_info.items():
        tensors = get_data(args.data_dir, data_file)
        if info['n_mols'] != len(tensors['database']):
            print(data_file, ':', info['n_mols'], len(tensors['database']))
        store.add_data(tensors)

    

if __name__ == '__main__':
    args = parse_args()

    file_crawler = FileCrawler(args.info_dir, args.data_dir, args.batch_size)

    data_dims = DataDimensions()
    data_info = {}

    for file in file_crawler.crawl_files():
        file_info = get_info(file, data_dims)
        data_info.update(file_info)

    store = ZarrStore(args.output_dir, args.graphs_per_chunk, data_dims)

    start_time = time.time()

    if args.n_cpus == 1:
        run_simple(data_info, store)
    else:
        #run_parallel() TODO Implement parallelization
        pass

    store.display_structure()   # Display final Zarr store structrue

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.1f} seconds")
    


    


    
    


        