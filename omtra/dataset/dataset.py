import torch
from abc import ABC, abstractmethod
from omtra.tasks.tasks import Task
from omtra.data.graph import approx_n_edges, get_edges_for_task
import functools

class OMTRADataset(ABC, torch.utils.data.Dataset):
    """Base class for single datasets."""

    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def get_num_nodes(self, task, start_idx, end_idx, per_ntype=False):
        """Get number of nodes in the dataset for a given task."""
        pass


    @functools.lru_cache(1024 * 1024)
    def get_num_edges(self, task: Task, start_idx, end_idx):
        # here, unlike in other places, start_idx and end_idx are
        # indexes into the graph_lookup array, not a node/edge data array

        # get number of nodes in each graph, per node type
        n_nodes_dict = self.get_num_nodes(task, start_idx, end_idx, per_ntype=True)
        node_types, n_nodes_per_type = zip(*n_nodes_dict.items())

        # get edge types modeled under this task
        edge_types = get_edges_for_task(task, self.graph_config)
        
        # evaluate same-ntype edges
        n_edges_total = torch.zeros(end_idx - start_idx, dtype=torch.int64)
        for etype in edge_types:

            # no need to count covalent edges, they're rare and few
            if 'covalent' in etype:
                continue
            n_edges = approx_n_edges(etype, self.graph_config, n_nodes_dict)
            if etype in self.graph_config.symmetric_etypes:
                n_edges *= 2
            n_edges_total += n_edges

        return n_edges_total