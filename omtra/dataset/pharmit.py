import dgl
import torch
import numpy as np
from omegaconf import DictConfig

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.data.graph import build_complex_graph
from omtra.data.xace_ligand import sparse_to_dense
from omtra.tasks.register import task_name_to_class
from omtra.tasks.tasks import Task
from omtra.utils.misc import classproperty

class PharmitDataset(ZarrDataset):
    def __init__(self, 
                 split: str,
                 processed_data_dir: str,
                 graphs_per_chunk: int,
                 graph_config: DictConfig,
    ):
        super().__init__(split, processed_data_dir)
        self.graphs_per_chunk = graphs_per_chunk
        self.graph_config = graph_config

    @classproperty
    def name(cls):
        return 'pharmit'

    def __len__(self):
        return self.root['lig/node/graph_lookup'].shape[0]

    def __getitem__(self, index) -> dgl.DGLHeteroGraph:
        task_name, idx = index
        task_class: Task = task_name_to_class[task_name]

        # slice lig node data
        xace_ligand = []
        start_idx, end_idx = self.slice_array('lig/node/graph_lookup', idx)
        for nfeat in ['x', 'a', 'c']:
            xace_ligand.append(
                self.slice_array(f'lig/node/{nfeat}', start_idx, end_idx)
            )
            
        # get slice indicies for ligand-ligand edges
        edge_slice_idxs = self.slice_array('lig/edge/graph_lookup', idx)

        # slice ligand-ligand edge data
        start_idx, end_idx = edge_slice_idxs
        xace_ligand.append(self.slice_array('lig/edge/e', start_idx, end_idx))
        xace_ligand.append(self.slice_array('lig/edge/edge_index', start_idx, end_idx))

        # convert to torch tensors
        # TODO: data typing!! need to design data typing!
        xace_ligand = [torch.from_numpy(arr) for arr in xace_ligand]

        # convert sparse xae to dense xae
        lig_x, lig_a, lig_c, lig_e, lig_edge_idxs = sparse_to_dense(*xace_ligand)

        # TODO: now that we have task information, we can actually write in necessary flow-matching related functionality:
        # - sampling priors for each modality
        # - doing OT alignment on ligand and pharmacophore nodes
        # - set graph keys to things like x_1_true and x_0 for ground-truth positions and prior samples

        # construct inputs to graph building function
        g_node_data = {
            'lig': {'x': lig_x, 'a': lig_a, 'c': lig_c},
        }
        g_edge_data = {
            'lig_to_lig': {'e': lig_e},
        }
        g_edge_idxs = {
            'lig_to_lig': lig_edge_idxs,
        }

        # if this task includes pharmacophore data, then we need to slice and add that data to the graph
        include_pharmacophore = 'pharmacophore' in task_class.modalities_present
        if include_pharmacophore:
            start_idx, end_idx = self.slice_array('pharm/node/graph_lookup', idx)
            pharm_x = self.slice_array('pharm/node/x', start_idx, end_idx)
            pharm_a = self.slice_array('pharm/node/a', start_idx, end_idx)
            pharm_x = torch.from_numpy(pharm_x)
            pharm_a = torch.from_numpy(pharm_a)
            g_node_data['pharm'] =  {'x': pharm_x, 'a': pharm_a}

        g = build_complex_graph(node_data=g_node_data, edge_idxs=g_edge_idxs, edge_data=g_edge_data)

        return g
    
    def retrieve_graph_chunks(self):
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

        return chunk_index
    
    def get_num_nodes(self, task: Task, start_idx, end_idx):
        # here, unlike in other places, start_idx and end_idx are 
        # indexes into the graph_lookup array, not a node/edge data array

        node_types = ['lig']
        if 'pharmacophore' in task.modalities_present:
            node_types.append('pharm')

        node_counts = []
        for ntype in node_types:
            graph_lookup = self.slice_array(f'{ntype}/node/graph_lookup', start_idx, end_idx)
            node_counts.append(graph_lookup[:, 1] - graph_lookup[:, 0])

        node_counts = np.stack(node_counts, axis=0).sum(axis=0)
        node_counts = torch.from_numpy(node_counts)
        return node_counts
    
    def get_num_edges(self, task: Task, start_idx, end_idx):
        # here, unlike in other places, start_idx and end_idx are 
        # indexes into the graph_lookup array, not a node/edge data array

        raise NotImplementedError('This function is not implemented yet! whats here was written by chatgpt lol')
        # TODO: implement this, requires decisions, or perhaps access to graph config information to infer # edges per graph
        # for lig-lig edges its easy
        # for pharm-pharm and lig-pharm edges, we have not yet decided how to handle them
        edge_types = ['lig_to_lig']
        edge_counts = []
        for etype in edge_types:
            graph_lookup = self.slice_array(f'{etype}/graph_lookup', start_idx, end_idx)
            edge_counts.append(graph_lookup[:, 1] - graph_lookup[:, 0])

        edge_counts = np.stack(edge_counts, axis=0).sum(axis=0)
        edge_counts = torch.from_numpy(edge_counts)
        return edge_counts