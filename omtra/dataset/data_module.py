import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import List, Dict
import dgl
from omegaconf import DictConfig
from copy import deepcopy
import torch

from omtra.dataset.multitask import MultitaskDataSet
from omtra.dataset.samplers import MultiTaskSampler
from omtra.load.conf import TaskDatasetCoupling, build_td_coupling
import torch.multiprocessing as mp


class MultiTaskDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        dataset_config: DictConfig, 
        task_phases: DictConfig,
        dataset_task_coupling: DictConfig,
        graph_config: DictConfig,
        prior_config: DictConfig, 
        edges_per_batch: int, 
        num_workers: int = 0, 
        distributed: bool = False,
        max_steps: int = None, 
        pin_memory: bool = True,
        fake_atom_p: float = 0.0,
        stereo_features: bool = False,
        stereo_source: str = 'training_geometry',
        identity_catalog_path: str = None,
        matched_mask_path: str = None,
        quarantine_path: str = None,
        stereo_hide_p_all: float = 0.0,
        stereo_hide_p_partial: float = 0.0,
        stereo_hide_partial_rate: float = 0.5,
        seed: int = 42,
        track_consumption: bool = False,
        enable_validation: bool = True,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.distributed = distributed
        self.edges_per_batch = edges_per_batch
        self.num_workers = num_workers
        self.prior_config = prior_config
        self.graph_config = graph_config
        self.max_steps = max_steps
        self.pin_memory = pin_memory
        self.fake_atom_p = fake_atom_p
        self.stereo_features = stereo_features
        self.stereo_source = stereo_source
        self.identity_catalog_path = identity_catalog_path
        self.matched_mask_path = matched_mask_path
        self.quarantine_path = quarantine_path
        self.stereo_hide_p_all = stereo_hide_p_all
        self.stereo_hide_p_partial = stereo_hide_p_partial
        self.stereo_hide_partial_rate = stereo_hide_partial_rate
        self.seed = seed
        self.track_consumption = track_consumption
        self.enable_validation = enable_validation
        self.train_sampler = self.val_sampler = None


        self.td_coupling: TaskDatasetCoupling = build_td_coupling(task_phases, dataset_task_coupling)


        self.save_hyperparameters()

    def setup(self, stage: str):

        if stage == 'fit':
            self.train_dataset = self.load_dataset('train')

            self.train_sampler = MultiTaskSampler(
                self.train_dataset, 
                self.td_coupling,
                self.edges_per_batch, 
                distributed=self.distributed,
                max_steps=self.max_steps,
                seed=self.seed,
                track_consumption=self.track_consumption,
            )
            if not self.enable_validation:
                return
            self.val_dataset = self.load_dataset('val')

            # create a validation t-d coupling that is uniform over the tasks being trained on
            val_td_coupling = deepcopy(self.td_coupling)
            p = val_td_coupling.p_dataset_task.sum(dim=0)
            # Equal mass per task, retaining its conditional dataset mixture.
            # Uniform task/dataset pairs overweights tasks supported by two datasets.
            p_uniform = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            p_uniform = p_uniform / len(val_td_coupling.task_space)
            p_dataset_task = torch.zeros_like(val_td_coupling.p_dataset_task)
            p_dataset_task = p_dataset_task + p_uniform.unsqueeze(0)
            val_td_coupling.p_dataset_task = p_dataset_task


            # TODO: how exactly do we want to sample data for validation? we don't actually
            # need to follow the td_coupling for validation, right?
            self.val_sampler = MultiTaskSampler(
                self.val_dataset, 
                val_td_coupling,
                self.edges_per_batch, 
                distributed=self.distributed,
                max_steps=self.max_steps,
            )

    def load_dataset(self, split: str):
        # TODO: should val dataset toss in fake atoms? need to consider implications for sampling conditonal tasks
        # well, only ligand denovo tasks need fake atoms, so that logic can be done in the dataset
        # which is aware of the task for which it is serving data
        return MultitaskDataSet(split, 
                             td_coupling=self.td_coupling,
                             graph_config=self.graph_config,
                             prior_config=self.prior_config,
                             fake_atom_p=self.fake_atom_p,
                             stereo_features=self.stereo_features,
                             stereo_source=self.stereo_source,
                             identity_catalog_path=self.identity_catalog_path,
                             matched_mask_path=self.matched_mask_path,
                             quarantine_path=self.quarantine_path,
                             stereo_hide_p_all=self.stereo_hide_p_all,
                             stereo_hide_p_partial=self.stereo_hide_p_partial,
                             stereo_hide_partial_rate=self.stereo_hide_partial_rate,
                             **self.dataset_config)
    
    def train_dataloader(self):
        # Allocate most workers to training
        train_workers = max(1, self.num_workers - 1) if self.num_workers > 1 else self.num_workers
        
        dataloader = DataLoader(self.train_dataset, 
                                batch_sampler=self.train_sampler,
                                collate_fn=omtra_collate_fn, 
                                worker_init_fn=None,
                                persistent_workers=train_workers > 0,
                                pin_memory=self.pin_memory,
                                prefetch_factor=5 if train_workers > 0 else None,
                                num_workers=train_workers)

        return dataloader
    

    def val_dataloader(self):
        if not self.enable_validation:
            return None
        # Use at most 1 worker for validation
        val_workers = min(1, self.num_workers) if self.num_workers > 0 else self.num_workers
        
        dataloader = DataLoader(self.val_dataset, 
                                batch_sampler=self.val_sampler,
                                collate_fn=omtra_collate_fn, 
                                worker_init_fn=None,
                                persistent_workers=val_workers > 0,
                                pin_memory=self.pin_memory,
                                num_workers=val_workers)
        return dataloader
    
    def mark_consumed(self):
        if self.train_sampler is not None:
            self.train_sampler.acknowledge()

    def state_dict(self):
        # Save the state of the samplers as part of the datamodule state.
        local = {
            'train_sampler_state': self.train_sampler.state_dict() if self.train_sampler is not None else None,
            'val_sampler_state': self.val_sampler.state_dict() if self.val_sampler is not None else None
        }
        if self.track_consumption and torch.distributed.is_initialized():
            ranks = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ranks, local)
            return dict(local, rank_states=ranks, world_size=len(ranks))
        return local

    def load_state_dict(self, state_dict):
        if 'rank_states' in state_dict:
            if len(state_dict['rank_states']) != self.train_sampler.num_replicas:
                raise ValueError('Exact data-order resume requires the same world size')
            state_dict = state_dict['rank_states'][self.train_sampler.rank]
        if self.train_sampler and state_dict.get('train_sampler_state') is not None:
            self.train_sampler.load_state_dict(state_dict['train_sampler_state'])
        if self.val_sampler and state_dict.get('val_sampler_state') is not None:
            self.val_sampler.load_state_dict(state_dict['val_sampler_state'])

def omtra_collate_fn(batch):
    graphs, task_names, dataset_names = zip(*batch)
    g = dgl.batch(graphs)
    return g, task_names[0], dataset_names[0]

# TODO: overkill because we set start method in train script but just to be safe
def worker_init_fn(worker_id):
    """Ensures a fresh start method for each worker."""
    mp.set_start_method("spawn", force=True)
