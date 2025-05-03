import dgl
import torch
import numpy as np
from omegaconf import DictConfig
import functools
import math
from pathlib import Path

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.data.graph import build_complex_graph
from omtra.data.xace_ligand import sparse_to_dense
from omtra.tasks.register import task_name_to_class
from omtra.tasks.tasks import Task
from omtra.tasks.modalities import name_to_modality
from omtra.utils.misc import classproperty
from omtra.data.graph import edge_builders, approx_n_edges
from omtra.priors.prior_factory import get_prior
from omtra.priors.sample import sample_priors
from omtra.constants import lig_atom_type_map, ph_idx_to_type, charge_map

from line_profiler import LineProfiler

class PharmitDataset(ZarrDataset):
    def __init__(self, 
                 split: str,
                 processed_data_dir: str,
                 graph_config: DictConfig,
                 prior_config: DictConfig,
    ):
        super().__init__(split, processed_data_dir)
        self.graph_config = graph_config
        self.prior_config = prior_config


        # dists_file = Path(processed_data_dir) / f'{split}_dists.npz'
        # dists_dict = np.load(dists_file)

        self.n_categories_dict = {
            'lig_a': len(lig_atom_type_map),
            'lig_c': len(charge_map),
            'lig_e': 4, # hard-coded assumption of 4 bond types (none, single, double, triple)
            'pharm_a': len(ph_idx_to_type),
        }

    @classproperty
    def name(cls):
        return 'pharmit'
    
    @property
    def n_zarr_chunks(self):
        return self.root['lig/node/x'].shape[0] // self.root['lig/node/x'].chunks[0]
    
    @property
    def graphs_per_chunk(self):
        return len(self) // self.n_zarr_chunks


    def __len__(self):
        return self.root['lig/node/graph_lookup'].shape[0]

    @profile
    def __getitem__(self, index) -> dgl.DGLHeteroGraph:
        task_name, idx = index
        task_class: Task = task_name_to_class(task_name)

        # check if this task includes pharmacophore data
        include_pharmacophore = 'pharmacophore' in task_class.groups_present

        # slice lig node data
        xace_ligand = []
        start_idx, end_idx = self.slice_array('lig/node/graph_lookup', idx)
        start_idx, end_idx = int(start_idx), int(end_idx)
        for nfeat in ['x', 'a', 'c']:
            xace_ligand.append(
                self.slice_array(f'lig/node/{nfeat}', start_idx, end_idx)
            )
            
        # get slice indicies for ligand-ligand edges
        edge_slice_idxs = self.slice_array('lig/edge/graph_lookup', idx)

        # slice ligand-ligand edge data
        start_idx, end_idx = edge_slice_idxs
        start_idx, end_idx = int(start_idx), int(end_idx)
        xace_ligand.append(self.slice_array('lig/edge/e', start_idx, end_idx))
        xace_ligand.append(self.slice_array('lig/edge/edge_index', start_idx, end_idx))

        # convert to torch tensors
        # TODO: data typing!! need to design data typing!
        xace_ligand = [torch.from_numpy(arr) for arr in xace_ligand]

        # set data types
        xace_ligand[0] = xace_ligand[0].float()
        xace_ligand[1] = xace_ligand[1].long()
        xace_ligand[2] = xace_ligand[2].long()
        xace_ligand[3] = xace_ligand[3].long()
        xace_ligand[4] = xace_ligand[4].long()

        # convert sparse xae to dense xae
        lig_x, lig_a, lig_c, lig_e, lig_edge_idxs = sparse_to_dense(*xace_ligand)

        # convert charges to token indicies
        charge_map_tensor = torch.tensor(charge_map)
        lig_c = torch.searchsorted(charge_map_tensor, lig_c)

        # construct inputs to graph building function
        g_node_data = {
            'lig': {'x_1_true': lig_x, 'a_1_true': lig_a, 'c_1_true': lig_c},
        }
        g_edge_data = {
            'lig_to_lig': {'e_1_true': lig_e},
        }
        g_edge_idxs = {
            'lig_to_lig': lig_edge_idxs,
        }

        # if this task includes pharmacophore data, then we need to slice and add that data to the graph
        if include_pharmacophore:
            # read pharmacophore data from zarr store
            start_idx, end_idx = self.slice_array('pharm/node/graph_lookup', idx)
            pharm_x = self.slice_array('pharm/node/x', start_idx, end_idx)
            pharm_a = self.slice_array('pharm/node/a', start_idx, end_idx)
            pharm_v = self.slice_array('pharm/node/v', start_idx, end_idx)
            pharm_x = torch.from_numpy(pharm_x).float()
            pharm_a = torch.from_numpy(pharm_a).long()
            pharm_v = torch.from_numpy(pharm_v).float()

            # add target pharmacophore data to graph
            g_node_data['pharm'] =  {
                'x_1_true': pharm_x, 
                'a_1_true': pharm_a, 
                'v_1_true': pharm_v
            }

        g = build_complex_graph(node_data=g_node_data, edge_idxs=g_edge_idxs, edge_data=g_edge_data)

        # sample priors
        priors_fns = get_prior(task_class, self.prior_config, training=True)
        g = sample_priors(g, task_class, priors_fns, training=True)

        return g
    
    def retrieve_graph_chunks(self, frac_start: float, frac_end: float):
        """
        This dataset contains len(self) examples. We divide all samples (or, graphs) into separate chunk. 
        We call these "graph chunks"; this is not the same thing as chunks defined in zarr arrays.
        I know we need better terminology; but they're chunks! they're totally chunks. just a different kind of chunk.
        """
        n_graphs = len(self)
        n_even_chunks, n_graphs_in_last_chunk = divmod(n_graphs, self.graphs_per_chunk)

        n_chunks = n_even_chunks + int(n_graphs_in_last_chunk > 0)

        # construct a tensor containing the index ranges for each chunk
        chunk_index = torch.zeros(n_chunks, 2, dtype=torch.int64)
        chunk_index[:, 0] = self.graphs_per_chunk*torch.arange(n_chunks)
        chunk_index[:-1, 1] = chunk_index[1:, 0]
        chunk_index[-1, 1] = n_graphs

        # if we need to only expose a subset of chunks (due to distributed training), do so here
        if not (frac_start == 0.0 and frac_end == 1.0):
            start_chunk_idx = math.floor(frac_start * n_chunks)
            end_chunk_idx = math.floor(frac_end * n_chunks)
            chunk_index = chunk_index[start_chunk_idx:end_chunk_idx]

        return chunk_index
    
    def get_num_nodes(self, task: Task, start_idx, end_idx, per_ntype=False):
        # here, unlike in other places, start_idx and end_idx are 
        # indexes into the graph_lookup array, not a node/edge data array

        node_types = ['lig']
        if 'pharmacophore' in task.groups_present:
            node_types.append('pharm')

        node_counts = []
        for ntype in node_types:
            graph_lookup = self.slice_array(f'{ntype}/node/graph_lookup', start_idx, end_idx)
            node_counts.append(graph_lookup[:, 1] - graph_lookup[:, 0])

        if per_ntype:
            num_nodes_dict = {ntype: ncount for ntype, ncount in zip(node_types, node_counts)}
            return num_nodes_dict

        node_counts = np.stack(node_counts, axis=0).sum(axis=0)
        node_counts = torch.from_numpy(node_counts)
        return node_counts