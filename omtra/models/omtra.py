import torch
import pytorch_lightning as pl
import dgl
from typing import Dict, List
from collections import defaultdict
import wandb
import itertools
import numpy as np

from omtra.load.conf import TaskDatasetCoupling, build_td_coupling
from omtra.data.graph.utils import get_batch_idxs, get_upper_edge_mask
from omtra.tasks.tasks import Task
from omtra.tasks.register import task_name_to_class
from omtra.tasks.modalities import Modality, name_to_modality
from omtra.constants import lig_atom_type_map, ph_idx_to_type
import omtra.models.conditional_paths as cond_paths

class OMTRA(pl.LightningModule):

    def __init__(self,
      task_phases,
      task_dataset_coupling,
      dists_file: str,
      total_loss_weights: Dict[str, float] = {},            
    ):
        super().__init__()

        self.dists_file = dists_file
        
        self.total_loss_weights = total_loss_weights
        # TODO: set default loss weights? set canonical order of features?


        # TODO: actually retrieve tasks and datasets for this dataset
        self.td_coupling: TaskDatasetCoupling = build_td_coupling(task_phases, task_dataset_coupling)
        self.sample_counts = defaultdict(int)
        if self.global_rank == 0:
            if wandb.run is None:
                print('Warning: no wandb run found. Setting previous sample counts to 0.')
            previous_sample_count = 0
            for nonzero_pair in self.td_coupling.support:
                task_idx, dataset_idx = nonzero_pair.tolist()
                task, dataset = self.td_coupling.task_space[task_idx], self.td_coupling.dataset_space[dataset_idx]
                if wandb.run is not None:
                    previous_sample_count = wandb.run.summary.get(f'{task}_{dataset}_sample_count', 0)
                self.sample_counts[(task, dataset)] = previous_sample_count
        self.sample_counts = dict(self.sample_counts)

        # TODO: implement periodic inference / eval ... how to do this with multiple tasks?
        # for pocket-conditioned tasks we really should do it on the test set too ... 

        # number of categories for categoircal features
        # in our generative process
        dists_dict = np.load(self.dists_file)
        lig_c_idx_to_val = dists_dict['p_tcv_c_space'] # a list of unique charges that appear in the dataset
        self.n_categories_dict = {
            'lig': {
                'a': len(lig_atom_type_map),
                'c': len(lig_c_idx_to_val),
                'e': 4, # hard-coded assumption of 4 bond types (none, single, double, triple)
            },
            'pharm': {
                'a': len(ph_idx_to_type),
            }
        }

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
                # commit=False
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
        

        # sample time
        # TODO: what are time sampling methods used in other papers?
        t = torch.rand(g.batch_size, device=g.device).float()

        # maybe not necessary right now, perhaps after we add edges appropriately
        node_batch_idxs, edge_batch_idxs = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, 'lig_to_lig')

        # sample conditional path
        task_class: Task = task_name_to_class(task_name) 
        g = self.sample_conditional_path(g, task_class, t)


    def configure_optimizers(self):
        # implement optimizer
        pass

    def sample_conditional_path(self, 
            g: dgl.DGLHeteroGraph, 
            task_class: Task, 
            t: torch.Tensor,
            node_batch_idxs: Dict[str, torch.Tensor],
            edge_batch_idxs: Dict[str, torch.Tensor],
            lig_ue_mask: torch.Tensor
    ):
        # sample the conditional path
        

        # TODO: support arbitrary alpha and beta functions, independently for each modality
        modalities_generated = task_class.modalities_generated
        alpha_t = {modality: t for modality in modalities_generated}
        beta_t = {modality: 1-t for modality in modalities_generated}

        for modality_name in modalities_generated:
            modality: Modality = name_to_modality(modality_name)

        # interpolate ligand positions (ligand, continuous)
        if 'ligand_structure' in task_class.modgroups_present:
            g.nodes['lig'].data['x_t'] = cond_paths.sample_continuous_interpolant(
                x_0=g.nodes['lig'].data['x_0'], 
                x_1=g.nodes['lig'].data['x_1_true'], 
                alpha_t=alpha_t['ligand_structure_x'][node_batch_idxs['lig']], 
                beta_t=beta_t['ligand_structure_x'][node_batch_idxs['lig']]
            )
    
        # TODO: only supporting masked CTMC modeling
        generating_lig_identity = 'ligand_identity' not in task_class.observed_at_t0 and 'ligand_identity' in task_class.observed_at_t1
        if generating_lig_identity:
            for mod in 'ace':
                data_src = g.edges if mod == 'e' else g.nodes
                batch_idxs = edge_batch_idxs['lig_to_lig'] if mod == 'e' else node_batch_idxs['lig']
                data_src.data[f'{mod}_t'] = cond_paths.sample_masked_ctmc(
                    x_1=data_src.data[f'{mod}_1_true'],
                    p_mask=alpha_t[f'ligand_identity_{mod}'][batch_idxs],
                    n_categories=self.n_categories_dict['lig'][mod],
                    ue_mask=lig_ue_mask if mod == 'e' else None
                )
        elif 'ligand_identity' in task_class.modgroups_present:
            # ligand identity is not being generated but it is in the system; so it is therefore conditiioning information
            for mod in 'ace':
                data_src = g.edges if mod == 'e' else g.nodes
                data_src.data[f'{mod}_t'] = data_src.data[f'{mod}_1_true']

        # TODO: for now, assuming pharmacophores are entirely generated or held fixed
        pharmacophore_present = 'pharmacophore' in task_class.modgroups_present
        generating_pharmacophore = 'pharmacophore' not in task_class.observed_at_t0 and 'pharmacophore' in task_class.observed_at_t1
        if generating_pharmacophore:
            for mod in 'xv':
                # sample paths for continuous pharmacophore modalities
                g.nodes['pharm'].data[f'{mod}_t'] = cond_paths.sample_continuous_interpolant(
                    x_0=g.nodes['pharm'].data[f'{mod}_0'],
                    x_1=g.nodes['pharm'].data[f'{mod}_1_true'],
                    alpha_t=alpha_t[f'pharmacophore_{mod}'][node_batch_idxs['pharm']],
                    beta_t=beta_t[f'pharmacophore_{mod}'][node_batch_idxs['pharm']]
                )
            # sample masked CTMC for pharmacophore types
            g.nodes['pharm'].data['a_t'] = cond_paths.sample_masked_ctmc(
                x_1=g.nodes['pharm'].data['a_1_true'],
                p_mask=alpha_t['pharmacophore_a'][node_batch_idxs['pharm']],
                n_categories=self.n_categories_dict['pharm']['a']
            )
        elif pharmacophore_present:
            # pharmacophores are present but not being generated; they are fixed conditioning information
            for mod in 'xav':
                g.nodes['pharm'].data[f'{mod}_t'] = g.nodes['pharm'].data[f'{mod}_1_true']

        if 'protein' in task_class.modgroups_present:
            raise NotImplementedError("Protein structure is not yet supported in the generative process")