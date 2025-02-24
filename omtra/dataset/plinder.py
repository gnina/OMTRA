import dgl
import torch
from omegaconf import DictConfig

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.data.graph import build_complex_graph
from omtra.data.xace_ligand import sparse_to_dense
from omtra.tasks.register import task_name_to_class
from omtra.tasks.tasks import Task
from omtra.utils.misc import classproperty

class PlinderDataset(ZarrDataset):
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
        return 'plinder'

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index) -> dgl.DGLHeteroGraph:
        task_name, idx = index
        task_class: Task = task_name_to_class[task_name]

        # TODO: things!
        g = build_complex_graph()

        return g
    
    def retrieve_graph_chunks(self, apo_systems: bool = False):
        """
        This dataset contains len(self) examples. We divide all samples (or, graphs) into separate chunk. 
        We call these "graph chunks"; this is not the same thing as chunks defined in zarr arrays.
        I know we need better terminology; but they're chunks! they're totally chunks. just a different kind of chunk.
        """
        n_graphs = len(self) # this is wrong! n_graphs depends on apo_systems!!!!
        n_even_chunks, n_graphs_in_last_chunk = divmod(n_graphs, self.graphs_per_chunk)

        n_chunks = n_even_chunks + int(n_graphs_in_last_chunk > 0)

        raise NotImplementedError("need to build capability to modify chunks based on whether or not the task uses the apo state")

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