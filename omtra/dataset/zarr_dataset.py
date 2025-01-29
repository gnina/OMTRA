import torch
import zarr
import numpy as np
import functools
from abc import ABC, abstractmethod

from omtra.utils.zarr_utils import list_zarr_arrays
from omtra.dataset.dataset import OMTRADataset
from omtra.tasks.tasks import Task

class ZarrDataset(OMTRADataset):

    """Base class for single datasets. Specifically a dataset that is stored in a zarr store. Supports caching of chunks to minimize disk access."""

    def __init__(self, zarr_store_path: str, n_chunks_cache: float = 4.25):
        super().__init__()

        self.store_path = zarr_store_path
        self.store = zarr.storage.LocalStore(zarr_store_path, read_only=True)
        self.root = zarr.open(store=self.store, mode='r')
        self.n_chunks_cache = n_chunks_cache
        self.build_cached_chunk_fetchers()
        

    @abstractmethod
    def retrieve_graph_chunks(self) -> torch.Tensor:
        pass

    @functools.cached_property
    def array_keys(self):
        return list_zarr_arrays(self.root)
        
    def build_cached_chunk_fetchers(self):
        self.chunk_fetchers = {}
        # self.chunks_accessed = defaultdict(set) # for debugging

        for array_name in self.array_keys:

            approx_chunk_size = self.root[array_name].nbytes / self.root[array_name].nchunks
            cache_size = int(self.n_chunks_cache*approx_chunk_size)
            
            # TODO: array-dependent cache size
            @functools.lru_cache(cache_size)
            def fetch_chunk(chunk_id, array_name=array_name): # we assume all arrays are chunked only along the first dimension 
                # self.chunks_accessed[array_name].add(chunk_id)
                chunk_size = self.root[array_name].chunks[0]
                chunk_start_idx = chunk_id * chunk_size
                chunk_end_idx = chunk_start_idx + chunk_size
                return self.root[array_name][chunk_start_idx:chunk_end_idx]

            self.chunk_fetchers[array_name] = fetch_chunk

    def slice_array(self, array_name, start_idx, end_idx=None):
        """Slice data from a zarr array but utilize chunk caching to minimize disk access."""

        if array_name not in self.array_keys:
            raise ValueError(f"There is no array with the name {array_name} in the zarr store located at {self.store_path}")

        single_idx = False
        if end_idx is None:
            single_idx = True
            end_idx = start_idx+1

        chunk_size = self.root[array_name].chunks[0] # the number of rows in each chunk (we only chunk and slice along the first dimension)

        # get the chunk id, as well as the index of our slice relative to the chunk, for the start and end of the slice
        start_chunk_id, start_chunk_idx = divmod(start_idx, chunk_size)
        end_chunk_id, end_chunk_idx = divmod(end_idx, chunk_size)

        # retrieve all chunks that are "touched" by the slice, using the cached chunk fetchers
        chunks = [self.chunk_fetchers[array_name](chunk_id) for chunk_id in range(start_chunk_id, end_chunk_id + 1)]

        # slice just the data we want from the chunks
        if len(chunks) == 1:
            chunk_slices = [  chunks[0][start_chunk_idx:end_chunk_idx]  ]
        else:
            chunk_slices = []
            for i in range(len(chunks)):
                if i == 0:
                    chunk_slices = [chunks[i][start_chunk_idx:]]
                elif i == len(chunks) - 1:
                    chunk_slices.append(chunks[i][:end_chunk_idx])
                else:
                    chunk_slices.append(chunks[i])

        data = np.concatenate(chunk_slices)
        if single_idx:
            data = data.squeeze()
        return data
    
    @abstractmethod
    def get_num_nodes(self, task: Task, start_idx: int, end_idx: int):
        pass

    @abstractmethod
    def get_num_edges(self, task: Task, start_idx: int, end_idx: int):
        pass
    
