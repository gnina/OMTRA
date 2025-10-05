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
        val_reweight_param: float = 0.2,
        val_check_interval: int = 1,
        limit_val_batches: int = 1,
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

        self.val_reweight_param = val_reweight_param
        self.val_check_interval = val_check_interval
        self.limit_val_batches = limit_val_batches


        self.td_coupling: TaskDatasetCoupling = build_td_coupling(task_phases, dataset_task_coupling)


        self.save_hyperparameters()

    def setup(self, stage: str):

        if stage == 'fit':
            self.train_dataset = self.load_dataset('train')
            self.val_dataset = self.load_dataset('val')

            self.train_sampler = MultiTaskSampler(
                self.train_dataset, 
                self.td_coupling,
                self.edges_per_batch, 
                distributed=self.distributed,
                max_steps=self.max_steps,
            )

            # create a validation t-d coupling that is uniform over the tasks being trained on
            val_td_coupling = deepcopy(self.td_coupling)
            p = val_td_coupling.p_dataset_task.sum(dim=0)
            mask = p != 0
            p_uniform = mask.float() / mask.sum()
            p_dataset_task = torch.zeros_like(val_td_coupling.p_dataset_task)
            p_dataset_task = p_dataset_task + p_uniform.unsqueeze(0)

            # interpolate between uniform and training coupling according to val_reweight_param
            p_dataset_task = self.val_reweight_param*p_dataset_task + (1 - self.val_reweight_param)*val_td_coupling.p_dataset_task
            val_td_coupling.p_dataset_task = p_dataset_task

            # adjust phase durations
            val_td_coupling.phase_durations = val_td_coupling.phase_durations * self.limit_val_batches/(self.val_check_interval+int(self.val_check_interval==0))

            


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
                             **self.dataset_config)
    
    def train_dataloader(self):
        # Allocate most workers to training
        train_workers = max(1, self.num_workers - 1) if self.num_workers > 1 else self.num_workers
        
        dataloader = DataLoader(self.train_dataset, 
                                batch_sampler=self.train_sampler,
                                collate_fn=omtra_collate_fn, 
                                worker_init_fn=worker_init_fn,
                                persistent_workers=train_workers > 0,
                                pin_memory=self.pin_memory,
                                prefetch_factor=5 if train_workers > 0 else None,
                                num_workers=train_workers)

        return dataloader
    

    def val_dataloader(self):
        # Use at most 1 worker for validation
        val_workers = min(1, self.num_workers) if self.num_workers > 0 else self.num_workers
        
        dataloader = DataLoader(self.val_dataset, 
                                batch_sampler=self.val_sampler,
                                collate_fn=omtra_collate_fn, 
                                worker_init_fn=worker_init_fn,
                                persistent_workers=val_workers > 0,
                                pin_memory=self.pin_memory,
                                num_workers=val_workers)
        return dataloader
    
    def state_dict(self):
        # Save the state of the samplers as part of the datamodule state.
        return {
            'train_sampler_state': self.train_sampler.state_dict() if self.train_sampler is not None else None,
            'val_sampler_state': self.val_sampler.state_dict() if self.val_sampler is not None else None
        }
    
    def load_state_dict(self, state_dict):
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