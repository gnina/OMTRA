import dgl
import torch
import numpy as np

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.utils.misc import classproperty
from omtra.data.graph import build_complex_graph
from omtra.data.graph.utils import build_lig_edge_idxs

def omtra_collate_fn(batch):
    data_lists = {}
    for k in batch[0].keys():
        data_lists[k] = [d[k] for d in batch]

    batch_data = {}

    batch_data['graph'] = dgl.batch(data_lists['graph'])

    batch_data['system_features'] = {}
    sys_data_dicts = data_lists['system_features']
    for key in sys_data_dicts[0].keys():
        batch_data['system_features'][key] = torch.stack([d[key] for d in sys_data_dicts], dim=0)

    batch_data['task_name'] = data_lists['task_name'][0]
    batch_data['dataset_name'] = data_lists['dataset_name'][0]

    return batch_data

class LatentDataset(ZarrDataset):
    """
    A dataset class to read the pre-computed latent features, coordinates,
    and metrics for training a confidence model.
    """
    def __init__(self, 
                 split: str, 
                 processed_data_dir: str):

        # it would be nice to have a correspondance to pharmit etc () both return dgl graphs
        super().__init__(split, processed_data_dir)

    @classproperty
    def name(cls):
        return 'latent_confidence'
    
    @property
    def n_zarr_chunks(self):
        return self.root['metadata/graph_lookup'].shape[0] // self.root['metadata/graph_lookup'].chunks[0]

    def __len__(self):
        return self.root['metadata/graph_lookup'].shape[0]

    def __getitem__(self, index) -> dict:
        atom_start, atom_end = self.slice_array('metadata/graph_lookup', index)

        rdkit_rmsd = self.slice_array('metrics/rdkit_rmsd', index)
        kabsch_rmsd = self.slice_array('metrics/kabsch_rmsd', index)

        scalar_features = self.slice_array('latents/node_scalar_features', atom_start, atom_end)
        vec_features = self.slice_array('latents/node_vec_features', atom_start, atom_end)
        positions = self.slice_array('latents/node_positions', atom_start, atom_end)

        coords_gt   = self.slice_array('coordinates/coords_gt', atom_start, atom_end)
        coords_pred = self.slice_array('coordinates/coords_pred', atom_start, atom_end)
        
        # construct inputs to graph building function
        g_node_data = {
            'lig': {
                'scalar_latents': torch.from_numpy(scalar_features),
                'vec_latents': torch.from_numpy(vec_features),
                'pos_latents': torch.from_numpy(positions),
                'coords_gt' : torch.from_numpy(coords_gt),
                'coords_pred' : torch.from_numpy(coords_pred),
                },
        }
        
        g_edge_data = {}
        num_atoms = scalar_features.shape[0]

        # build pairs of edge indices for fully connectivity
        g_edge_idxs = {
            'lig_to_lig': build_lig_edge_idxs(num_atoms),
        }

        # build the graph
        g = build_complex_graph(
            node_data=g_node_data, 
            edge_idxs=g_edge_idxs, 
            edge_data=g_edge_data,
        )
        
        # store precomputed metrics (rmsd etc.) as system features
        system_features = {
            'rdkit_rmsd': torch.from_numpy(rdkit_rmsd).float(),
            'kabsch_rmsd': torch.from_numpy(kabsch_rmsd).float()
        }

        return_dict = {
            'graph': g,
            'system_features': system_features,
            'task_name': 'confidence',
            'dataset_name': self.name,
        }

        return return_dict

    # TODO
    def retrieve_graph_chunks(self, *args, **kwargs):
        pass

    def get_num_nodes(self, *args, **kwargs):
        pass