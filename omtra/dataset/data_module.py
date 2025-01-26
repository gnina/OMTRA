import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import List, Dict
import dgl

from omtra.dataset.multitask import MultitaskDataSet


class MultiTaskDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        tasks: List[dict], 
        multitask_dataset_config: dict, 
        prior_config: dict, 
        batch_size: int, 
        num_workers: int = 0, 
        distributed: bool = False, 
        max_num_edges: int = 40000
    ):
        super().__init__()
        self.tasks = tasks
        self.multitask_dataset_config = multitask_dataset_config
        self.distributed = distributed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prior_config = prior_config
        self.max_num_edges = max_num_edges
        self.save_hyperparameters()

    def setup(self, stage: str):

        if stage == 'fit':
            self.train_dataset = self.load_dataset('train')
            self.val_dataset = self.load_dataset('val')

    def load_dataset(self, split: str):
        return MultitaskDataSet(split, 
                             tasks=self.tasks,
                             **self.multitask_dataset_config)
    
    def train_dataloader(self):
        # TODO: we definitely need a custom dataloader here due to multitasks, adaptive batching, etc.
        dataloader = DataLoader(self.train_dataset, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                collate_fn=dgl.batch, 
                                num_workers=self.num_workers)

        return dataloader
    

    def val_dataloader(self):
        # TODO: we definitely need a custom dataloader here due to multitasks, adaptive batching, etc.
        dataloader = DataLoader(self.val_dataset, 
                                batch_size=self.batch_size*2, 
                                shuffle=True, 
                                collate_fn=dgl.batch, 
                                num_workers=self.num_workers)
        return dataloader
