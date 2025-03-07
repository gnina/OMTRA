import torch
import pytorch_lightning as pl
import dgl
from typing import Dict, List
from collections import defaultdict
import wandb

class OMTRA(pl.LightningModule):

    def __init__(self,
      total_loss_weights: Dict[str, float] = {},            
    ):
        super().__init__()
        
        self.total_loss_weights = total_loss_weights
        # TODO: set default loss weights? set canonical order of features?


        # TODO: actually retrieve tasks and datasets for this dataset
        tasks = ['task_a', 'task_b']
        datasets = ['dataset_a', 'dataset_b']
        for task in tasks:
            for dataset in datasets:
                previous_sample_count = wandb.run.summary.get(f'{task}_{dataset}_sample_count', 0)
                self.sample_counts[(task, dataset)] = previous_sample_count

        # TODO: implement periodic inference / eval ... how to do this with multiple tasks?
        # for pocket-conditioned tasks we really should do it on the test set too ... 

    def training_step(self, batch_data, batch_idx):
        g, task_name, dataset_name = batch_data


        # get the total batch size across all devices
        local_batch_size = torch.tensor([g.batch_size], device=g.device)
        all_batch_counts = self.all_gather(local_batch_size)
        total_batch_count = all_batch_counts.sum().item()

        # log the total sample count
        if self.global_rank == 0:
            self.sample_counts[(task_name, dataset_name)] += total_batch_count
            metric_name = f'{task_name}_{dataset_name}_sample_count'
            self.log(
                metric_name, 
                self.sample_counts[(task_name, dataset_name)],
                rank_zero_only=True,
                sync_dist=False,
                commit=False
            )

        # forward pass
        losses = self(g, task_name)

        train_log_dict = {}
        for key in losses:
            train_log_dict[f'{key}_train_loss'] = losses[key]

        total_loss = torch.zeros(1, device=g.device, requires_grad=True)
        for feat in self.canonical_feat_order:
            total_loss = total_loss + self.total_loss_weights[feat]*losses[feat]

        self.log_dict(train_log_dict, sync_dist=True, commit=False)
        self.log('train_total_loss', total_loss, sync_dist=True, commit=True)
        return total_loss

    def forward(self, g: dgl.DGLHeteroGraph, task_name: str):
        pass

    def configure_optimizers(self):
        # implement optimizer
        pass