import torch
import numpy as np
import dgl

from omtra.dataset.zarr_dataset import ZarrDataset

class MultiMultiSet(torch.utils.data.Dataset):

    """A dataset capable of serving up samples from multiple zarr datasets."""

    def __init__(self, split: str):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class PharmitDataset(ZarrDataset):

    def __init__(self, zarr_store_path: str):
        super().__init__(zarr_store_path)

    def __len__(self):
        return self.root['node_data']['node_lookup'].shape[0]

    def __getitem__(self, idx) -> dgl.DGLHeteroGraph:
        # lookup start and end indicies for node and edge data to pull just
        # one graph from the full dataset
        node_start_idx, node_end_idx = self.slice_array('node_data/node_lookup', idx)
        edge_start_idx, edge_end_idx = self.slice_array('edge_data/edge_lookup', idx)

        # pull out the data for the graph
        x = self.slice_array('node_data/x', node_start_idx, node_end_idx)
        a = self.slice_array('node_data/a', node_start_idx, node_end_idx)
        e = self.slice_array('edge_data/e', node_start_idx, node_end_idx)
        edge_idxs = self.slice_array('edge_data/edge_index', edge_start_idx, edge_end_idx)

        # TODO: converse to dense representation
        # TODO: convert to DGL graph
        # TODO: build general graph construction code that works across datasets (take from PharmacoFlow)

        return x, a, e, edge_idxs