
from omtra.dataset.zarr_dataset import ZarrDataset

class ChunkTracker:
    def __init__(self, dataset: ZarrDataset, *args, **kwargs):
        self.dataset = dataset
        self.graphs_per_chunk = dataset.graphs_per_chunk

        self.chunk_index = self.dataset.retrieve_graph_chunks(*args, **kwargs)