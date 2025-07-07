import dgl
import torch
import numpy as np

import json
from omegaconf import OmegaConf, DictConfig
from omtra.dataset.register import dataset_name_to_class
from pathlib import Path

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.utils.misc import classproperty
from omtra.data.graph import build_complex_graph

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

        super().__init__(split, processed_data_dir)

        # Load the saved config (.json resides alongside the Zarr store)
        saved_config, datamodule_config, graph_config, prior_config = self.obtain_config(processed_data_dir, split)

        # Extract dataset info from the saved datamodule config
        single_dataset_configs = datamodule_config.dataset_config.single_dataset_configs
        
        # TODO: For now, assume pharmit (but we need to generalize this to single vs multiple)
        dataset_name = 'pharmit'
        dataset_class = dataset_name_to_class[dataset_name]
        single_dataset_config = single_dataset_configs[dataset_name]
        
        # Initialize the original dataset with exact same config
        original_split = saved_config['source_split']  # e.g., "val"

        self.original_dataset = dataset_class(
            split=original_split,
            graph_config=graph_config,
            prior_config=prior_config,
            fake_atom_p=saved_config['fake_atom_p'],
            **single_dataset_config
        )

    def obtain_config(self, processed_data_dir, split):
        # Look for config file associated with Zarr store
        zarr_path = Path(processed_data_dir) / split
        config_path = zarr_path.with_suffix('.config.json')
        
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        # Convert resolved configs back to DictConfig
        datamodule_config = OmegaConf.create(saved_config['datamodule_config'])
        graph_config = OmegaConf.create(saved_config['graph_config'])
        prior_config = OmegaConf.create(saved_config['prior_config'])

        return saved_config, datamodule_config, graph_config, prior_config

    @classproperty
    def name(cls):
        return 'latent_confidence'
    
    @property
    def n_zarr_chunks(self):
        return self.root['metadata/graph_lookup'].shape[0] // self.root['metadata/graph_lookup'].chunks[0]

    def __len__(self):
        return self.root['metadata/graph_lookup'].shape[0]

    def __getitem__(self, index) -> dict:
        original_graph = self.original_dataset[('ligand_conformer', index)]
        
        # Get latent data
        atom_start, atom_end = self.slice_array('metadata/graph_lookup', index)
        
        # Load metrics
        rdkit_rmsd = self.slice_array('metrics/rdkit_rmsd', index)
        kabsch_rmsd = self.slice_array('metrics/kabsch_rmsd', index)
        
        # Load latent features
        scalar_features = self.slice_array('latents/node_scalar_features', atom_start, atom_end)
        vec_features = self.slice_array('latents/node_vec_features', atom_start, atom_end)
        positions = self.slice_array('latents/node_positions', atom_start, atom_end)
        
        # Load coordinates
        coords_gt = self.slice_array('coordinates/coords_gt', atom_start, atom_end)
        coords_pred = self.slice_array('coordinates/coords_pred', atom_start, atom_end)
        
        # construct inputs to graph building function
        g_node_data = {
            'lig': {
                # copy keys and values of original graph
                **{k: v for k, v in original_graph.nodes['lig'].data.items()},
                # Add latent features
                'scalar_latents': torch.from_numpy(scalar_features),
                'vec_latents': torch.from_numpy(vec_features),
                'pos_latents': torch.from_numpy(positions),
                'coords_gt' : torch.from_numpy(coords_gt),
                'coords_pred' : torch.from_numpy(coords_pred),
            }
        }
        
        # populate original edge data
        g_edge_idxs = {
            'lig_to_lig': original_graph.edges(etype='lig_to_lig')
        }
        
        g_edge_data = {
            'lig_to_lig': {
                **{k: v for k, v in original_graph.edges['lig_to_lig'].data.items()}
            }
        }
        
        # build the graph (gives us the complete schema that works with batching)
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