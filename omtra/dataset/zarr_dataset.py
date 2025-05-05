import torch
import zarr
import numpy as np
import functools
from pathlib import Path
from abc import ABC, abstractmethod

from omtra.utils.zarr_utils import list_zarr_arrays
from omtra.dataset.dataset import OMTRADataset
from omtra.tasks.tasks import Task
from collections import OrderedDict
# from line_profiler import LineProfiler

class ZarrDataset(OMTRADataset):

    """Base class for single datasets. Specifically a dataset that is stored in a zarr store. Supports caching of chunks to minimize disk access."""

    def __init__(self, split: str, processed_data_dir: str, n_chunks_cache: int = 4):
        super().__init__()

        zarr_store_path = Path(processed_data_dir) / f'{split}.zarr'
        if not zarr_store_path.exists():
            raise ValueError(f"There is no zarr store at the path {zarr_store_path}")

        self.store_path = zarr_store_path   
        self.store = zarr.storage.LocalStore(str(zarr_store_path), read_only=True)
        self.root = zarr.open(store=self.store, mode='r')
        self.n_chunks_cache = n_chunks_cache
        self.build_cached_chunk_fetchers()

        self.rows_per_chunk = {}
        for array_name in self.array_keys:
            self.rows_per_chunk[array_name] = self.root[array_name].chunks[0]
        

    @abstractmethod
    def retrieve_graph_chunks(self) -> torch.Tensor:
        pass

    @functools.cached_property
    def array_keys(self):
        return list_zarr_arrays(self.root)
        
    def build_cached_chunk_fetchers(self):
        self.chunk_fetchers = {}
        for array_name in self.array_keys:
            self.chunk_fetchers[array_name] = ChunkFetcher(self.root, array_name, cache_size=self.n_chunks_cache)

    # @profile
    def slice_array(self, array_name, start_idx, end_idx=None):
        """Slice data from a zarr array but utilize chunk caching to minimize disk access."""

        if array_name not in self.array_keys:
            raise ValueError(f"There is no array with the name {array_name} in the zarr store located at {self.store_path}")

        single_idx = False
        if end_idx is None:
            single_idx = True
            end_idx = start_idx+1

        chunk_size = self.rows_per_chunk[array_name] # the number of rows in each chunk (we only chunk and slice along the first dimension)

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
    
class ChunkFetcher:
    def __init__(self, root, array_name, cache_size):
        self.root = root
        self.array_name = array_name
        self.array = self.root[self.array_name]
        self.cache_size = cache_size
        self.cache = OrderedDict()  # Ordered dictionary to maintain LRU order
        self.chunk_size = self.array.chunks[0]

        # self.chunks_touched = 0

    def __call__(self, chunk_id):
        if chunk_id in self.cache:
            # Move the accessed chunk to the end to mark it as recently used
            self.cache.move_to_end(chunk_id)
            # if self.array_name == 'lig/node/x':
            #     print(f"READ from existing chunk: {chunk_id}, {self.chunks_touched} chunks touched")

        else:
            # self.chunks_touched += 1
            # if self.array_name == 'lig/node/x':
            #     print(f"READ from disk: {chunk_id}, {self.chunks_touched} chunks touched")
            if len(self.cache) >= self.cache_size:
                # Remove the least recently used chunk
                self.cache.popitem(last=False)
            # Fetch the chunk and add it to the cache
            chunk_start_idx = chunk_id * self.chunk_size
            chunk_end_idx = chunk_start_idx + self.chunk_size
            self.cache[chunk_id] = self.array[chunk_start_idx:chunk_end_idx]
        return self.cache[chunk_id]