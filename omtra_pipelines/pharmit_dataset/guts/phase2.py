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

class FileCrawler():
    def __init__(self, output_dir: Path, n_chunks_process: int = None, shuffle: bool = False):
        self.info_dir = output_dir / 'phase1' / 'chunk_info'
        self.data_dir = output_dir / 'phase1' / 'chunk_data'
        self.shuffle = shuffle

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
    
    def __iter__(self):
        iter_data = list(self.df.iterrows())
        if self.shuffle:
            np.random.shuffle(iter_data)
        for _, row in iter_data:
            yield row.to_dict()

    def __len__(self):
        return self.df.shape[0]
    

class ZarrStore:
    def __init__(self, store_path: Path, totals: dict, n_chunks: int,
                 overwrite: bool = False, build: bool = True, mode='r'):
        self.totals = totals
        self.n_chunks = n_chunks
        self.store_path = store_path

        if store_path.exists() and overwrite:
            shutil.rmtree(store_path)

        if not self.store_path.exists() and (build or overwrite):
            init_zarr_store(store_path, totals, n_chunks)

        if mode == 'r':
            read_only = True
        else:
            read_only = False

        self.store = zarr.storage.LocalStore(str(store_path), read_only=read_only)
        self.root = zarr.open(store=self.store, mode=mode)

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
    pharm_node_data.create_array('graph_lookup', shape=(n_graphs, 2), chunks=(graphs_per_chunk,2), dtype=np.int64)
    
    # Create database array
    # TODO: i don't like hard-coding the shape of the database array (and therefore # databases supported)
    db_group.create_array('db', shape=(n_graphs, 14), chunks=(graphs_per_chunk,14), dtype=bool)

    print(root.tree())

    return store, root


