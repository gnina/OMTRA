import argparse
from pathlib import Path
import zarr
import pickle
import numpy as np
from typing import List, Dict
import pandas as pd
import shutil
import functools
import math
from collections import defaultdict

from tqdm import tqdm
from omtra.utils.zarr_utils import list_zarr_arrays
from omtra.utils.graph import build_lookup_table
from omtra.utils.misc import combine_tcv_counts

class ChunkInfoManager():
    def __init__(self, output_dir: Path, atom_map: List[str], n_chunks_process: int = None, shuffle: bool = False):
        self.info_dir = output_dir / 'phase1' / 'chunk_info'
        self.data_dir = output_dir / 'phase1' / 'chunk_data'
        self.shuffle = shuffle

        # read all chunk info files, convert to dataframe
        cinfo_rows = []
        atom_counts = []
        pharm_counts = []
        tcv_counts = []
        for chunk_info_file in self.info_dir.iterdir():
            with open(chunk_info_file, 'rb') as f:
                chunk_info = pickle.load(f)
            atom_counts.append(chunk_info.pop('p_atoms'))
            pharm_counts.append(chunk_info.pop('p_pharms_given_atoms'))
            tcv_counts.append(chunk_info.pop('tcv_counts'))
            cinfo_rows.append(chunk_info)

            if n_chunks_process is not None and len(cinfo_rows) >= n_chunks_process:
                break

        self.n_nodes_dist_info = self.compute_num_node_dists(atom_counts, pharm_counts)
        self.valency_dist_info, self.valency_table = self.compute_valency_dists(tcv_counts, atom_map)

        self.df = pd.DataFrame(cinfo_rows) # raw info data converted to dataframe

        self.totals = self.df.sum(axis=0, numeric_only=True).to_dict() # sum numerical columns (n_mols, n_atoms, n_edges, n_pharm)

        self.df = self.compute_write_boundaries(self.df)

    def compute_valency_dists(self, tcv_counts: List[dict], atom_map) -> dict:

        tcv_counts: defaultdict = combine_tcv_counts(tcv_counts) # default dict mapping from (type_idx, charge, valency) to count (number of times that tcv was observed)

        tcv_support = np.array(list(tcv_counts.keys())) # array containing all tcv values with non-zero counts

        t_space = np.unique(tcv_support[:, 0]).tolist()
        c_space = np.unique(tcv_support[:, 1]).tolist()
        v_space = np.unique(tcv_support[:, 2]).tolist()

        p_tcv_unnormalized = np.zeros((len(t_space), len(c_space), len(v_space)))
        for i, t in enumerate(t_space):
            for j, c in enumerate(c_space):
                for k, v in enumerate(v_space):
                    p_tcv_unnormalized[i, j, k] = tcv_counts[(t, c, v)]
        p_tcv = p_tcv_unnormalized / p_tcv_unnormalized.sum()



        valency_arr = tcv_support
        atom_idx_to_symbol = {i: atom for i, atom in enumerate(atom_map)}

        valency_dict = {}
        for tcv_support in valency_arr:
            atom_idx, charge, valency = tcv_support.tolist()
            atom_symbol = atom_idx_to_symbol[atom_idx]
            if atom_symbol not in valency_dict:
                valency_dict[atom_symbol] = {}
            if charge not in valency_dict[atom_symbol]:
                valency_dict[atom_symbol][charge] = []
            valency_dict[atom_symbol][charge].append(valency)

        tcv_dist_info = {
            'p_tcv_t_space': t_space,
            'p_tcv_c_space': c_space,
            'p_tcv_v_space': v_space,
            'p_tcv': p_tcv
        }

        return tcv_dist_info, valency_dict

    def compute_num_node_dists(self, atom_counts, pharm_counts):
        """Merge the unnormalize p(n_atoms) and p(n_pharms|n_atoms) distributions obtained for each chunk of the dataset

        Args:
            atom_counts (_type_): A list where each element is the output of np.unique(n_atoms_per_mol, return_counts=True)
            pharm_counts (_type_): A dictionary where keys are an integer specifying the number of atoms in a molecule.
                                    The values are the output of np.unique(n_pharms_per_mol, return_counts=True), thus specifying the counts 
                                    that when normalized will yield p(n_pharms|n_atoms)
        """

        merged_atom_counts = defaultdict(int)
        for n_atoms, n_observations in atom_counts:
            for n, c in zip(n_atoms, n_observations):
                merged_atom_counts[n] += c
        
        # get keys of merged_atom_counts and sort them in ascending order
        n_atoms = sorted(merged_atom_counts.keys())
        p_n_atoms = np.array([merged_atom_counts[n] for n in n_atoms])
        p_n_atoms = p_n_atoms / p_n_atoms.sum()

        merged_pharms_given_atoms = defaultdict(lambda: defaultdict(int))
        n_pharms_observed = set()
        for pharm_count_dict in pharm_counts:
            for n_atoms_obs, pharm_counts in pharm_count_dict.items():
                for n_pharms, c in zip(*pharm_counts):
                    merged_pharms_given_atoms[n_atoms_obs][n_pharms] += c
                    n_pharms_observed.add(n_pharms)
        n_pharms_observed = sorted(list(n_pharms_observed))

        p_pharms_given_atoms = np.zeros((len(n_atoms), len(n_pharms_observed)))
        for i, n_atoms_i in enumerate(n_atoms):
            for j, n_pharms_j in enumerate(n_pharms_observed):
                p_pharms_given_atoms[i, j] = merged_pharms_given_atoms[n_atoms_i][n_pharms_j]

        # normalize along rows
        p_pharms_given_atoms = p_pharms_given_atoms / p_pharms_given_atoms.sum(axis=1, keepdims=True)

        # finally, compute the joint distribution p(n_atoms, n_pharms)
        p_atoms_pharms = p_n_atoms[:, None] * p_pharms_given_atoms

        # normalize just to be safe
        p_atoms_pharms = p_atoms_pharms / p_atoms_pharms.sum()

        n_atoms = np.array(n_atoms)
        n_pharms_observed = np.array(n_pharms_observed)

        nodes_dist_info = {
            'p_ap_atoms_space': n_atoms,
            'p_ap_pharms_space': n_pharms_observed,
            'p_ap': p_atoms_pharms
        }

        return nodes_dist_info


    def compute_write_boundaries(self, df: pd.DataFrame):

        # define canonical order for chunks
        if self.shuffle:
            df['order'] = np.random.permutation(df.shape[0])
        else:
            df['order'] = np.arange(df.shape[0]) 
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
        db_array_boundaries = build_lookup_table(mols_per_chunk) # has shape (n_chunks, 2)

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

        # compute boundaries for the database array
        df['db_start'], df['db_end'] = db_array_boundaries[:, 0], db_array_boundaries[:, 1]

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


