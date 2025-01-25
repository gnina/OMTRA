import torch
import zarr
import numpy as np
import functools
from abc import ABC, abstractmethod

from omtra.utils.zarr_utils import list_zarr_arrays

class ZarrDataset(ABC, torch.utils.data.Dataset):

    """Base class for single datasets. Specifically a dataset that is stored in a zarr store. Supports caching of chunks to minimize disk access."""

    def __init__(self, zarr_store_path: str, n_chunks_cache: float = 4.25):
        super().__init__()

        self.store = zarr.storage.LocalStore(zarr_store_path)
        self.root = zarr.open(store=self.store, mode='r')
        self.n_chunks_cache = n_chunks_cache
        self.build_cached_chunk_fetchers()
        

    @abstractmethod
    def __len__(self):
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

        if end_idx is None:
            end_idx = start_idx+1

        chunk_size = self.root[array_name].chunks[0]
        start_chunk_id = start_idx // chunk_size
        end_chunk_id = end_idx // chunk_size
        chunks = [self.chunk_fetchers[array_name](chunk_id) for chunk_id in range(start_chunk_id, end_chunk_id + 1)]
        chunk_slices = []
        for i in range(len(chunks)):

            # if the slice lays in just one chunk
            if len(chunks) == 1:
                chunk_slices.append(chunks[i][start_idx % chunk_size:end_idx % chunk_size])
                continue

            # for multi-chunk access:
            if i == 0:
                chunk_slices.append(chunks[i][start_idx % chunk_size:])
            elif i == len(chunks) - 1:
                chunk_slices.append(chunks[i][:end_idx % chunk_size])
            else:
                chunk_slices.append(chunks[i])

        data = np.concatenate(chunk_slices)
        if data.shape[0] == 1:
            data = data.squeeze()
        return data