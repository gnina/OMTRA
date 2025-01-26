import dgl
import torch

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.data.graph import build_complex_graph
from omtra.data.xace_ligand import sparse_to_dense

class PharmitDataset(ZarrDataset):
    def __init__(self, zarr_store_path: str):
        super().__init__(zarr_store_path)

    @classmethod
    def name(cls):
        return 'pharmit'

    def __len__(self):
        return self.root['node_data']['node_lookup'].shape[0]

    def __getitem__(self, idx) -> dgl.DGLHeteroGraph:
       # TODO: nothing here accounts for the presence of pharamcophore data,
       # which actually will be present in the final pharmit dataset!
        node_start_idx, node_end_idx = self.slice_array('node_data/node_lookup', idx)
        edge_start_idx, edge_end_idx = self.slice_array('edge_data/edge_lookup', idx)

        # pull out the data for the graph
        lig_x = self.slice_array('node_data/x', node_start_idx, node_end_idx)
        lig_a = self.slice_array('node_data/a', node_start_idx, node_end_idx)
        lig_c = self.slice_array('node_data/c', node_start_idx, node_end_idx)
        lig_e = self.slice_array('edge_data/e', edge_start_idx, edge_end_idx)
        lig_edge_idxs = self.slice_array('edge_data/edge_index', edge_start_idx, edge_end_idx)

        # convert to torch tensors
        lig_x = torch.from_numpy(lig_x)
        lig_a = torch.from_numpy(lig_a)
        lig_c = torch.from_numpy(lig_c)
        lig_e = torch.from_numpy(lig_e)
        lig_edge_idxs = torch.from_numpy(lig_edge_idxs)

        # convert sparse xae to dense xae
        lig_x, lig_a, lig_c, lig_e, lig_edge_idxs = sparse_to_dense(lig_x, lig_a, lig_c, lig_e, lig_edge_idxs)


        g_node_data = {
            'lig': {'x': lig_x, 'a': lig_a, 'c': lig_c},
        }
        g_edge_data = {
            'lig_to_lig': {'e': lig_e},
        }
        g_edge_idxs = {
            'lig_to_lig': lig_edge_idxs,
        }

        g = build_complex_graph(node_data=g_node_data, edge_idxs=g_edge_idxs, edge_data=g_edge_data)

        return g