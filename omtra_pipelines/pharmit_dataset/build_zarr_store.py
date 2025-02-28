import argparse
from pathlib import Path
import zarr
import pickle
import numpy as np
import time
import os
from typing import List, Dict
import pandas as pd
import shutil
import traceback
import functools
import math
from contextlib import contextmanager

from tqdm import tqdm
from multiprocessing import Pool, Lock
from omtra.utils.zarr_utils import list_zarr_arrays
from omtra.utils.graph import build_lookup_table


def parse_args():
    p = argparse.ArgumentParser(description='Build Zarr store')
    p.add_argument('--output_dir', type=Path, help='Directory for store.',  default=Path('./outputs/'))
    p.add_argument('--store_name', type=Path, help='Name of store.',  default=Path('store.zarr'))

    p.add_argument('--n_cpus', type=int, default=1, help='Number of CPUs to use for parallel processing.')
    p.add_argument('--n_chunks_zarr', type=int, default=1000, help='Number of chunks for zarr arrays.')
    p.add_argument('--n_chunks_process', type=int, default=None, help='Number of chunks to process, just for debugging.')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing store.')

    args = p.parse_args()
    return args

global_lock_register = {}
global_zstore = None

def init_worker(locks, store_path):
    """
    Pool initializer to set the global dictionary of locks for each worker.
    """
    global global_lock_register
    global global_zstore
    global_lock_register = locks
    global_zstore: ZarrStore = ZarrStore(store_path, None, None, overwrite=False, build=False)

@contextmanager
def get_all_locks(arr_name: str, zchunk_ids: List[int]):
    sorted_ids = sorted(zchunk_ids)
    acquired_locks = []
    try:
        for lock_id in sorted_ids:
            lock = global_lock_register[arr_name][lock_id]
            lock.acquire()  # Blocking call to acquire the lock.
            acquired_locks.append(lock)
        print("Acquired locks for ids:", sorted_ids)
        yield  # Critical section where the caller can safely do work.
    finally:
        # Release the locks in reverse order.
        for lock in reversed(acquired_locks):
            lock.release()


class FileCrawler():
    def __init__(self, output_dir: Path, n_chunks_process: int = None):
        self.info_dir = output_dir / 'phase1' / 'chunk_info'
        self.data_dir = output_dir / 'phase1' / 'chunk_data'

        # read all chunk info files, convert to dataframe
        cinfo_rows = []
        for chunk_info_file in self.info_dir.iterdir():
            with open(chunk_info_file, 'rb') as f:
                chunk_info = pickle.load(f)
            cinfo_rows.append(chunk_info)


            if n_chunks_process is not None and len(cinfo_rows) >= n_chunks_process:
                break
        self.df = pd.DataFrame(cinfo_rows) # raw info data converted to dataframe

        self.totals = self.df.sum(axis=0, numeric_only=True).to_dict() # sum numerical columns (n_mols, n_atoms, n_edges, n_pharm)

        self.df = self.compute_write_boundaries(self.df)

    def compute_write_boundaries(self, df: pd.DataFrame):

        df['order'] = np.arange(df.shape[0]) # define canonical order for chunks
        df = df.sort_values(by='order')

        # we need boundaries for: ligand nodes, ligand edges, and pharmacophore nodes
        lig_node_boundaries = build_lookup_table(df['n_atoms'].values) # has shape (n_chunks, 2)
        lig_edge_boundaries = build_lookup_table(df['n_edges'].values) # has shape (n_chunks, 2)
        pharm_node_boundaries = build_lookup_table(df['n_pharm'].values) # has shape (n_chunks, 2)

        # we also need boundaries for lig node graph lookup, lig edge graph lookup, and pharm node graph lookup
        mols_per_chunk = df['n_mols'].values
        lig_node_graph_lookup_boundaries = build_lookup_table(mols_per_chunk) # has shape (n_chunks, 2)
        lig_edge_graph_lookup_boundaries = build_lookup_table(mols_per_chunk) # has shape (n_chunks, 2)
        pharm_node_graph_lookup_boundaries = build_lookup_table(mols_per_chunk) # has shape (n_chunks, 2)

        # compute the number of atoms, pharms, and edges preceding each chunk
        df['lig_node_offset'] = np.concatenate([[0], np.cumsum(df['n_atoms'].values)[:-1]])
        df['lig_edge_offset'] = np.concatenate([[0], np.cumsum(df['n_edges'].values)[:-1]])
        df['pharm_node_offset'] = np.concatenate([[0], np.cumsum(df['n_pharm'].values)[:-1]])

        df['lig_node_start'], df['lig_node_end'] = lig_node_boundaries[:, 0], lig_node_boundaries[:, 1]
        df['lig_edge_start'], df['lig_edge_end'] = lig_edge_boundaries[:, 0], lig_edge_boundaries[:, 1]
        df['pharm_node_start'], df['pharm_node_end'] = pharm_node_boundaries[:, 0], pharm_node_boundaries[:, 1]

        df['lig_node_graph_lookup_start'], df['lig_node_graph_lookup_end'] = lig_node_graph_lookup_boundaries[:, 0], lig_node_graph_lookup_boundaries[:, 1]
        df['lig_edge_graph_lookup_start'], df['lig_edge_graph_lookup_end'] = lig_edge_graph_lookup_boundaries[:, 0], lig_edge_graph_lookup_boundaries[:, 1]
        df['pharm_node_graph_lookup_start'], df['pharm_node_graph_lookup_end'] = pharm_node_graph_lookup_boundaries[:, 0], pharm_node_graph_lookup_boundaries[:, 1]

        return df
    
    def zarr_chunks_touched(self, chunk_idx: int) -> Dict[str, list]:

        row = self.df.iloc[chunk_idx]
        # TODO: for each array that needs to be touched
        # find the zarr chunks that would be touched by the write operation
        # return a dictionary with the array name as key and the list of chunks as value
    
    def info_batch_generator(self):
        for _, row in self.df.iterrows():
            yield row.to_dict()

    def __len__(self):
        return self.df.shape[0]

def init_zarr_store(store_path: Path, totals: dict, n_chunks: int):
    store = zarr.storage.LocalStore(store_path)

    # Create a root group
    root = zarr.group(store=store)

    ntypes = ['lig', 'db', 'pharm']

    ntype_groups = {}
    for ntype in ntypes:
        ntype_groups[ntype] = root.create_group(ntype)


    lig_node = ntype_groups['lig'].create_group('node')
    lig_edge_data = ntype_groups['lig'].create_group('edge')

    pharm_node_data = ntype_groups['pharm'].create_group('node')
    db_group = ntype_groups['db']

    n_atoms = totals['n_atoms']
    n_pharms = totals['n_pharm']
    n_edges = totals['n_edges']
    n_graphs = totals['n_mols']
    
    # some simple heuristics to decide chunk sizes for node and edge data
    mean_lig_nodes_per_graph = totals['n_atoms'] // totals['n_mols']
    mean_ll_edges_per_graph = totals['n_edges'] // totals['n_mols']
    mean_pharm_nodes_per_graph = totals['n_pharm'] // totals['n_mols']
    graphs_per_chunk = totals['n_mols'] // n_chunks

    nodes_per_chunk = n_atoms // n_chunks
    ll_edges_per_chunk = n_edges // n_chunks
    pharm_nodes_per_chunk = n_pharms // n_chunks

    # Create node arrays
    lig_node.create_array('x', shape=(n_atoms, 3), chunks=(nodes_per_chunk,3), dtype=np.float32)
    lig_node.create_array('a', shape=(n_atoms,), chunks=(nodes_per_chunk, ), dtype=np.int8)
    lig_node.create_array('c', shape=(n_atoms,), chunks=(nodes_per_chunk, ), dtype=np.int8)
    lig_node.create_array('graph_lookup', shape=(n_graphs, 2), chunks=(graphs_per_chunk, 2), dtype=np.int64)

    # Create edge arrays
    lig_edge_data.create_array('e', shape=(n_edges, ), chunks=(ll_edges_per_chunk, ), dtype=np.int8)
    lig_edge_data.create_array('edge_index', shape=(n_edges, 2), chunks=(ll_edges_per_chunk, 2), dtype=np.int32)
    lig_edge_data.create_array('graph_lookup', shape=(n_graphs,2), chunks=(graphs_per_chunk, 2), dtype=np.int64)

    # Create pharmacophore arrays
    pharm_node_data.create_array('x', shape=(n_pharms, 3), chunks=(pharm_nodes_per_chunk,3), dtype=np.float32)
    pharm_node_data.create_array('a', shape=(n_pharms, ), chunks=(pharm_nodes_per_chunk, ), dtype=np.int8)
    pharm_node_data.create_array('graph_lookup', shape=(n_pharms, 2), chunks=(pharm_nodes_per_chunk,2), dtype=np.int64)
    
    # Create database array
    # TODO: i don't like hard-coding the shape of the database array (and therefore # databases supported)
    db_group.create_array('db', shape=(n_graphs, 14), chunks=(graphs_per_chunk,14), dtype=bool)

    print(root.tree())

    return store, root


def write_data_to_store(data_file: str, data_info: dict, parallel: bool = False):

    tensors = np.load(data_file)
    tensors = tensors

    root = global_zstore.root

    lig_node_group = root['lig/node']
    lig_edge_group = root['lig/edge']
    pharm_node_group = root['pharm/node']

    # get start and end indices for ligand node data
    start_idx = data_info['lig_node_start']
    end_idx = data_info['lig_node_end']

    # find what zarr chunks are touched by this write operation for ligand node data
    if parallel:
        zchunks_touched = global_zstore.get_chunks_touched('lig/node/x', start_idx, end_idx)
    else:
        zchunks_touched = []
    with get_all_locks('lig/node/x', zchunks_touched):
        # write ligand node data
        lig_node_group['x'][start_idx:end_idx] = tensors['lig_x']
        lig_node_group['a'][start_idx:end_idx] = tensors['lig_a']
        lig_node_group['c'][start_idx:end_idx] = tensors['lig_c']

    # write ligand node graph lookup
    # convert lig node graph lookup to global indicies
    start_idx = data_info['lig_node_graph_lookup_start']
    end_idx = data_info['lig_node_graph_lookup_end']

    
    if parallel:
        zchunks_touched = global_zstore.get_chunks_touched('lig/node/graph_lookup', start_idx, end_idx)
    else:
        zchunks_touched = []
    with get_all_locks('lig/node/graph_lookup', zchunks_touched):
        lig_node_graph_lookup = tensors['node_lookup'] + data_info['lig_node_offset']
        lig_node_group['graph_lookup'][start_idx:end_idx] = lig_node_graph_lookup

    # write ligand edge data
    start_idx = data_info['lig_edge_start']
    end_idx = data_info['lig_edge_end']

    if parallel:
        zchunks_touched = global_zstore.get_chunks_touched('lig/edge/e', start_idx, end_idx)
    else:
        zchunks_touched = []
    with get_all_locks('lig/edge/e', zchunks_touched):
        lig_edge_group['e'][start_idx:end_idx] = tensors['lig_e']
        lig_edge_group['edge_index'][start_idx:end_idx] = tensors['lig_edge_idx']

    # write ligand edge graph lookup
    start_idx = data_info['lig_edge_graph_lookup_start']
    end_idx = data_info['lig_edge_graph_lookup_end']

    if parallel:
        zchunks_touched = global_zstore.get_chunks_touched('lig/edge/graph_lookup', start_idx, end_idx)
    else:
        zchunks_touched = []
    with get_all_locks('lig/edge/graph_lookup', zchunks_touched):
        lig_edge_graph_lookup = tensors['edge_lookup'] + data_info['lig_edge_offset']
        lig_edge_group['graph_lookup'][start_idx:end_idx] = lig_edge_graph_lookup


    # write pharmacophore node data
    start_idx = data_info['pharm_node_start']
    end_idx = data_info['pharm_node_end']

    zchunks_touched = global_zstore.get_chunks_touched('pharm/node/x', start_idx, end_idx)
    if not parallel:
        zchunks_touched = []
    with get_all_locks('pharm/node/x', zchunks_touched):
        pharm_node_group['x'][start_idx:end_idx] = tensors['pharm_x']
        pharm_node_group['a'][start_idx:end_idx] = tensors['pharm_a']


    # convert pharm node graph lookup to global indicies
    start_idx = data_info['pharm_node_graph_lookup_start']
    end_idx = data_info['pharm_node_graph_lookup_end']

    zchunks_touched = global_zstore.get_chunks_touched('pharm/node/graph_lookup', start_idx, end_idx)
    if not parallel:
        zchunks_touched = []
    with get_all_locks('pharm/node/graph_lookup', zchunks_touched):
        pharm_node_graph_lookup = tensors['pharm_lookup'] + data_info['pharm_node_offset']
        pharm_node_group['graph_lookup'][start_idx:end_idx] = pharm_node_graph_lookup


    return

def error_and_update(error):
    """Handle errors, update error counter and the progress bar."""
    print(f"Error: {error}")
    traceback.print_exception(type(error), error, error.__traceback__)
    raise Exception("Error encountered during processing.")


def run_parallel(n_cpus: int, file_crawler: FileCrawler, store_path: Path, locks: dict):

    total_tasks = len(file_crawler)
    pbar = tqdm(total=total_tasks, desc="Processing", unit="chunks")

    with Pool(processes=n_cpus, initializer=init_worker, initargs=(locks, store_path)) as pool:
        for pchunk_idx, row_info in enumerate(file_crawler):


            # Submit the job and add its AsyncResult to the pending list
            result = pool.apply_async(
                write_data_to_store, 
                args=(store_path, row_info['file'], row_info, True),
                callback=lambda _: pbar.update(1),
                error_callback=error_and_update
            )

        pool.close()
        pool.join()


def run_simple(file_crawler: FileCrawler, store_path: Path):
    init_worker({}, store_path)
    iterator = tqdm(file_crawler.info_batch_generator(), desc="Processing", unit="chunks")
    for data_info in iterator:
        write_data_to_store(data_info['file'], data_info)

class ZarrStore:
    def __init__(self, store_path: Path, totals: dict, n_chunks: int,
                 overwrite: bool = False, build: bool = True):
        self.totals = totals
        self.n_chunks = n_chunks

        if store_path.exists() and overwrite:
            shutil.rmtree(store_path)

        if not self.store_path.exists() and (build or overwrite):
            init_zarr_store(store_path, totals, n_chunks)

        self.store = zarr.storage.LocalStore(str(store_path), read_only=True)
        self.root = zarr.open(store=self.store, mode='r')

    @functools.cached_property
    def array_keys(self):
        return list_zarr_arrays(self.root)
    
    def n_zarr_chunks(self, array_name: str):
        arr = self.root[array_name]
        chunk_size = arr.chunks[0]
        n_rows = arr.shape[0]
        n_chunks = math.ceil(n_rows / chunk_size)
        return n_chunks

    # @functools.lru_cache()
    def get_chunks_touched(self, array_name: str, start_idx: int, end_idx: int):
        chunk_size = self.root[array_name].chunks[0]
        start_chunk_id, start_chunk_idx = divmod(start_idx, chunk_size)
        end_chunk_id, end_chunk_idx = divmod(end_idx, chunk_size)
        chunk_idxs = list(range(start_chunk_id, end_chunk_id + 1))
        return chunk_idxs

    def display_structure(self):
        print(self.root.tree()) 

def build_lock_register(file_crawler: FileCrawler, zstore: ZarrStore):
    locks = {}
    for array_name in zstore.array_keys:
        n_chunks = zstore.n_zarr_chunks(array_name)
        locks[array_name] = [Lock() for _ in range(n_chunks)]
    return locks

if __name__ == '__main__':
    args = parse_args()

    file_crawler = FileCrawler(args.output_dir, args.n_chunks_process)

    # create zarr store
    zarr_dir = args.output_dir / 'phase2' 
    zarr_dir.mkdir(parents=True, exist_ok=True)
    store_path = zarr_dir / args.store_name

    zstore = ZarrStore(store_path, file_crawler.totals, args.n_chunks_zarr, overwrite=args.overwrite)

    if args.n_cpus > 1:
        locks = build_lock_register(file_crawler, zstore)

    start_time = time.time()

    if args.n_cpus == 1:
        run_simple(file_crawler, store_path)
    else:
        run_parallel(args.n_cpus, file_crawler, store_path, locks)

    # store.display_structure()   # Display final Zarr store structrue

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.1f} seconds")
    


    


    
    


        