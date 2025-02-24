import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import List, Dict
import dgl
from omegaconf import DictConfig

from omtra.dataset.multitask import MultitaskDataSet
from omtra.dataset.samplers import MultiTaskSampler
import torch.multiprocessing as mp


class MultiTaskDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        tasks: List[dict], 
        multitask_dataset_config: DictConfig, 
        graph_config: DictConfig,
        prior_config: dict, 
        edges_per_batch: int, 
        num_workers: int = 0, 
        distributed: bool = False, 
    ):
        super().__init__()
        self.tasks = tasks
        self.multitask_dataset_config = multitask_dataset_config
        self.distributed = distributed
        self.edges_per_batch = edges_per_batch
        self.num_workers = num_workers
        self.prior_config = prior_config
        self.graph_config = graph_config
        self.save_hyperparameters()

    def setup(self, stage: str):

        if stage == 'fit':
            self.train_dataset = self.load_dataset('train')
            self.val_dataset = self.load_dataset('val')

    def load_dataset(self, split: str):
        # TODO: tasks should just be absored into multitask_dataset_config
        return MultitaskDataSet(split, 
                             tasks=self.tasks,
                             graph_config=self.graph_config,
                             **self.multitask_dataset_config)
    
    def train_dataloader(self):
        batch_sampler = MultiTaskSampler(self.train_dataset, self.edges_per_batch, distributed=self.distributed)
        dataloader = DataLoader(self.train_dataset, 
                                batch_sampler=batch_sampler,
                                collate_fn=omtra_collate_fn, 
                                worker_init_fn=worker_init_fn,
                                num_workers=self.num_workers)

        return dataloader
    

    def val_dataloader(self):
        batch_sampler = MultiTaskSampler(self.val_dataset, self.edges_per_batch, distributed=self.distributed)
        dataloader = DataLoader(self.val_dataset, 
                                batch_sampler=batch_sampler,
                                collate_fn=omtra_collate_fn, 
                                worker_init_fn=worker_init_fn,
                                num_workers=self.num_workers)
        return dataloader


def omtra_collate_fn(batch):
    graphs, task_names, dataset_names = zip(*batch)
    g = dgl.batch(graphs)
    return g, task_names[0], dataset_names[0]

# TODO: overkill because we set start method in train script but just to be safe
def worker_init_fn(worker_id):
    """Ensures a fresh start method for each worker."""
    mp.set_start_method("spawn", force=True)