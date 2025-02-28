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
import multiprocessing

from omtra_pipelines.pharmit_dataset.guts.phase2 import *


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
    global_zstore = ZarrStore(store_path, None, None, overwrite=False, build=False, mode='a')

@contextmanager
def get_all_locks(arr_name: str, zchunk_ids: List[int]):
    sorted_ids = sorted(zchunk_ids)
    acquired_locks = []
    try:
        for lock_id in sorted_ids:
            lock = global_lock_register[arr_name][lock_id]
            lock.acquire()  # Blocking call to acquire the lock.
            acquired_locks.append(lock)
        # print(f"Acquired locks for {arr_name} chunks ids: {sorted_ids}")
        yield  # Critical section where the caller can safely do work.
    finally:
        # Release the locks in reverse order.
        for lock in reversed(acquired_locks):
            lock.release()

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

    # write database array
    start_idx = data_info['db_start']
    end_idx = data_info['db_end']

    if not parallel:
        zchunks_touched = []
    else:
        zchunks_touched = global_zstore.get_chunks_touched('db/db', start_idx, end_idx)
    with get_all_locks('db/db', zchunks_touched):
        root['db/db'][start_idx:end_idx] = tensors['database']


    return

def error_and_update(error):
    """Handle errors, update error counter and the progress bar."""
    # print(f"Error: {error}")
    traceback.print_exception(type(error), error, error.__traceback__)
    raise error


def run_parallel(n_cpus: int, file_crawler: ChunkInfoManager, store_path: Path, locks: dict):

    total_tasks = len(file_crawler)
    pbar = tqdm(total=total_tasks, desc="Processing", unit="chunks")

    with Pool(processes=n_cpus, initializer=init_worker, initargs=(locks, store_path)) as pool:
        for pchunk_idx, row_info in enumerate(file_crawler):


            # Submit the job and add its AsyncResult to the pending list
            result = pool.apply_async(
                write_data_to_store, 
                args=(row_info['file'], row_info, True),
                callback=lambda _: pbar.update(1),
                error_callback=error_and_update
            )

        pool.close()
        pool.join()


def run_simple(file_crawler: ChunkInfoManager, store_path: Path):
    init_worker({}, store_path)
    iterator = tqdm(file_crawler, desc="Processing", unit="chunks")
    for data_info in iterator:
        write_data_to_store(data_info['file'], data_info)

def build_lock_register(file_crawler: ChunkInfoManager, zstore: ZarrStore):
    locks = {}
    for array_name in zstore.array_keys:
        n_chunks = zstore.n_zarr_chunks(array_name)
        locks[array_name] = [Lock() for _ in range(n_chunks)]
    return locks

if __name__ == '__main__':
    args = parse_args()

    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    file_crawler = ChunkInfoManager(args.output_dir, 
                               args.n_chunks_process,
                               shuffle=args.n_cpus > 1
                               )

    # create zarr store
    zarr_dir = args.output_dir / 'phase2' 
    zarr_dir.mkdir(parents=True, exist_ok=True)
    store_path = zarr_dir / args.store_name

    zstore = ZarrStore(store_path, file_crawler.totals, args.n_chunks_zarr, overwrite=args.overwrite)

    # write p(n_atoms, n_pharms) data to simple npz file
    hist_file = zarr_dir / f'{store_path.stem}_n_nodes_dist.npz'
    np.savez(hist_file, **file_crawler.n_nodes_dist_info)

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
    


    


    
    


        