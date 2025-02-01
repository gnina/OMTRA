import torch

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.tasks.tasks import Task

class ChunkTracker:
    def __init__(self, 
                 dataset: ZarrDataset,
                 nodes_per_batch: int,
                 *args, **kwargs):
        
        self.dataset = dataset
        self.nodes_per_batch = nodes_per_batch # maximum number of nodes in a batch


        # chunk index is an (n_graph_chunks, 2) array containing the start and end indices of each chunk
        self.chunk_index = self.dataset.retrieve_graph_chunks(*args, **kwargs)
        self.n_chunks = self.chunk_index.shape[0]

        self.chunk_queue = torch.randperm(self.n_chunks)
        self.chunk_queue_idx = 0
        self.reset_queue()

    @property
    def graphs_per_chunk(self):
        return self.dataset.graphs_per_chunk

    def reset_queue(self):
        self.n_samples_served_this_chunk = 0
        self.chunk_queue = torch.randperm(self.n_chunks)
        self.chunk_queue_idx = 0

    def step_queue(self):
        self.chunk_queue_idx += 1
        self.n_samples_served_this_chunk = 0
        if self.chunk_queue_idx >= self.n_chunks:
            self.reset_queue()

    def get_batch_idxs(self, task: Task) -> torch.Tensor:

        # check if we need to move to the next chunk
        if self.n_samples_served_this_chunk >= n_graphs_in_chunk:
            self.step_queue()

        start_idx, end_idx = self.chunk_index[self.chunk_queue[self.chunk_queue_idx]]
        n_graphs_in_chunk = end_idx - start_idx
        node_per_graph = self.dataset.get_num_nodes(task, start_idx, end_idx) # has shape (end_idx - start_idx,)
        batch_idxs = start_idx + adaptive_batch_loader(node_per_graph, self.nodes_per_batch)
        return batch_idxs


def adaptive_batch_loader(nodes_per_graph: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """
    Selects graphs to include in a batch such that the total number of nodes
    does not exceed a user-specified maximum.
    
    Args:
        nodes_per_graph (torch.Tensor): A tensor of shape (N,) containing the number of nodes in each graph.
        max_nodes (int): The maximum number of nodes allowed in a batch.
        
    Returns:
        torch.Tensor: Indices of the graphs selected for the batch.
    """
    num_graphs = nodes_per_graph.size(0)
    indices = torch.randperm(num_graphs)  # Randomly shuffle the indices
    selected_indices = []
    total_nodes = 0
    
    for idx in indices:
        graph_nodes = nodes_per_graph[idx].item()
        if total_nodes + graph_nodes > max_nodes:
            break
        selected_indices.append(idx)
        total_nodes += graph_nodes

    return torch.tensor(selected_indices)
