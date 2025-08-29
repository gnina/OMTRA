import dgl
import torch
import numpy as np
import warnings

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
        saved_config, datamodule_config, graph_config, prior_config = self.obtain_config()

        # Extract dataset info from the saved datamodule config
        single_dataset_configs = datamodule_config.dataset_config.single_dataset_configs
        
        # Extract which dataset & task_name that was used to create this latent Zarr store
        self.dataset_name = saved_config['dataset_name']
        self.task_name    = saved_config['task_name']
        dataset_class = dataset_name_to_class[self.dataset_name]
        single_dataset_config = single_dataset_configs[self.dataset_name]
        
        # Initialize the original dataset with exact same config
        original_split = saved_config['source_split']  # e.g., "val"

        self.original_dataset = dataset_class(
            split=original_split,
            graph_config=graph_config,
            prior_config=prior_config,
            fake_atom_p=saved_config['fake_atom_p'],
            **single_dataset_config
        )
        
        # Check latent dataset size consistency with its original dataset
        if len(self) > len(self.original_dataset):
            raise AssertionError(
                f"Latent dataset has more examples than the original {self.dataset_name} dataset! "
                f"This indicates a bug in latent generation ({len(self)} > {len(self.original_dataset)} samples)."
            )
        elif len(self) < len(self.original_dataset):
            warnings.warn(
                f"Your latent dataset has less examples than the {self.dataset_name} dataset has. "
                f"Likely you saved only a subset ({len(self)} < {len(self.original_dataset)} samples).",
                UserWarning
            )

    def obtain_config(self):
        # Load config from Zarr store attributes
        saved_config = dict(self.root.attrs['config'])
        
        # Convert resolved configs back to DictConfig
        datamodule_config = OmegaConf.create(saved_config['datamodule_config'])
        graph_config = datamodule_config.graph_config
        prior_config = datamodule_config.prior_config

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
        original_graph = self.original_dataset[(self.task_name, index)]
        
        # Get latent data
        atom_start, atom_end = self.slice_array('metadata/graph_lookup', index)
        
        # Load latent features
        scalar_features = self.slice_array('latents/node_scalar_features', atom_start, atom_end)
        vec_features = self.slice_array('latents/node_vec_features', atom_start, atom_end)
        positions = self.slice_array('latents/node_positions', atom_start, atom_end)
        
        # Load coordinates
        coords_gt = self.slice_array('coordinates/coords_gt', atom_start, atom_end)
        coords_pred = self.slice_array('coordinates/coords_pred', atom_start, atom_end)
        
        # Add latent features directly to the original graph
        original_graph.nodes['lig'].data['scalar_latents'] = torch.from_numpy(scalar_features)
        original_graph.nodes['lig'].data['vec_latents'] = torch.from_numpy(vec_features)
        original_graph.nodes['lig'].data['pos_latents'] = torch.from_numpy(positions)
        original_graph.nodes['lig'].data['coords_gt'] = torch.from_numpy(coords_gt)
        original_graph.nodes['lig'].data['coords_pred'] = torch.from_numpy(coords_pred)
        
        # load available metrics (gt calculations)
        system_features = {}
        for metric_name in self.root['metrics'].keys():
            metric_value = self.slice_array(f'metrics/{metric_name}', index)
            system_features[metric_name] = torch.from_numpy(metric_value).float()

        return_dict = {
            'graph': original_graph,
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