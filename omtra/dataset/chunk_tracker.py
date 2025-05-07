import torch
import functools

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.tasks.tasks import Task

class GraphChunkTracker:
    def __init__(self, 
                 dataset: ZarrDataset,
                 edges_per_batch: int,
                 frac_start: float,
                 frac_end: float,
                 *args, **kwargs):
        
        self.dataset = dataset
        self.edges_per_batch = edges_per_batch # maximum number of nodes in a batch
        self.construction_args = args
        self.construction_kwargs = kwargs
        self.frac_start = frac_start
        self.frac_end = frac_end

        self.chunk_index = self.dataset.retrieve_graph_chunks(
            frac_start=self.frac_start,
            frac_end=self.frac_end,
            *args, **kwargs)
        
        self.n_chunks = self.chunk_index.shape[0]
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

    def current_chunk_idxs(self):
        idxs = self.chunk_index[self.chunk_queue[self.chunk_queue_idx]]
        idxs = tuple(idx.item() for idx in idxs)
        return idxs

    def get_batch_idxs(self, task: Task) -> torch.Tensor:
        start_idx, end_idx = self.current_chunk_idxs()
        n_graphs_in_chunk = end_idx - start_idx

        # check if we need to move to the next chunk
        if self.n_samples_served_this_chunk >= n_graphs_in_chunk:
            self.step_queue()
            start_idx, end_idx = self.current_chunk_idxs()

        edges_per_graph = self.dataset.get_num_edges(task, start_idx, end_idx) # has shape (end_idx - start_idx,)
        batch_idxs = start_idx + adaptive_batch_loader(edges_per_graph, self.edges_per_batch)
        self.n_samples_served_this_chunk += batch_idxs.size(0)
        return batch_idxs.tolist()


def adaptive_batch_loader(nodes_per_graph: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """
    Selects a *prefix* of a random permutation of graph-indices whose cumulative
    node count stays ≤ max_nodes. Fully vectorized, no Python loops or .item().
    """
    # 1) shuffle once
    perm = torch.randperm(nodes_per_graph.size(0), device=nodes_per_graph.device)
    # 2) grab node counts in that order
    nodes_shuf = nodes_per_graph[perm]
    # 3) cumulative sum
    cumsum = nodes_shuf.cumsum(dim=0)
    # 4) mask: which positions still fit
    mask = cumsum <= max_nodes
    # 5) get the valid prefix positions
    valid_positions = torch.nonzero(mask, as_tuple=False).squeeze(1)
    # 6) ensure at least one graph
    if valid_positions.numel() == 0:
        return perm[:1]
    # 7) return the corresponding original indices
    return perm[valid_positions]
