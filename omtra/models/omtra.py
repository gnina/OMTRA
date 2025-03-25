import torch
import torch.nn.functional as fn
import pytorch_lightning as pl
import dgl
from typing import Dict, List, Callable, Tuple
from collections import defaultdict
import wandb
import itertools
import numpy as np
from functools import partial

from omtra.load.conf import TaskDatasetCoupling, build_td_coupling
from omtra.data.graph.utils import get_batch_idxs, get_upper_edge_mask
from omtra.tasks.tasks import Task
from omtra.tasks.register import task_name_to_class
from omtra.tasks.modalities import Modality, name_to_modality
from omtra.constants import lig_atom_type_map, ph_idx_to_type
from omtra.models.conditional_paths.path_factory import get_conditional_path_fns
from omtra.models.vector_field import EndpointVectorField
from omtra.models.interpolant_scheduler import InterpolantScheduler
from omegaconf import DictConfig


class OMTRA(pl.LightningModule):
    def __init__(
        self,
        task_phases,
        task_dataset_coupling,
        dists_file: str,
        graph_config: DictConfig,
        conditional_paths: DictConfig,
        total_loss_weights: Dict[str, float] = {},
    ):
        super().__init__()

        self.dists_file = dists_file
        self.graph_config = graph_config
        self.conditional_path_config = conditional_paths

        self.total_loss_weights = total_loss_weights
        # TODO: set default loss weights? set canonical order of features?

        # TODO: actually retrieve tasks and datasets for this dataset
        self.td_coupling: TaskDatasetCoupling = build_td_coupling(
            task_phases, task_dataset_coupling
        )
        self.sample_counts = defaultdict(int)
        if self.global_rank == 0:
            if wandb.run is None:
                print(
                    "Warning: no wandb run found. Setting previous sample counts to 0."
                )
            previous_sample_count = 0
            for nonzero_pair in self.td_coupling.support:
                task_idx, dataset_idx = nonzero_pair.tolist()
                task, dataset = (
                    self.td_coupling.task_space[task_idx],
                    self.td_coupling.dataset_space[dataset_idx],
                )
                if wandb.run is not None:
                    previous_sample_count = wandb.run.summary.get(
                        f"{task}_{dataset}_sample_count", 0
                    )
                self.sample_counts[(task, dataset)] = previous_sample_count
        self.sample_counts = dict(self.sample_counts)

        # TODO: implement periodic inference / eval ... how to do this with multiple tasks?
        # for pocket-conditioned tasks we really should do it on the test set too ...

        # number of categories for categoircal features
        # in our generative process
        dists_dict = np.load(self.dists_file)
        lig_c_idx_to_val = dists_dict[
            "p_tcv_c_space"
        ]  # a list of unique charges that appear in the dataset
        self.n_categories_dict = {
            "lig_a": len(lig_atom_type_map),
            "lig_c": len(lig_c_idx_to_val),
            "lig_e": 4,  # hard-coded assumption of 4 bond types (none, single, double, triple)
            "pharm_a": len(ph_idx_to_type),
        }
        self.time_scaled_loss = False
        self.interpolant_scheduler = InterpolantScheduler(schedule_type="linear")
        self.vector_field = EndpointVectorField(
            interpolant_scheduler=self.interpolant_scheduler
        )  # TODO: initialize this properly

    def training_step(self, batch_data, batch_idx):
        g, task_name, dataset_name = batch_data

        # get the total batch size across all devices
        local_batch_size = torch.tensor([g.batch_size], device=g.device)
        all_batch_counts = self.all_gather(local_batch_size)
        total_batch_count = all_batch_counts.sum().item()

        # log the total sample count
        if self.global_rank == 0:
            self.sample_counts[(task_name, dataset_name)] += total_batch_count
            metric_name = f"{task_name}_{dataset_name}_sample_count"
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
            train_log_dict[f"{key}_train_loss"] = losses[key]

        total_loss = torch.zeros(1, device=g.device, requires_grad=True)
        for feat in self.canonical_feat_order:
            total_loss = total_loss + self.total_loss_weights[feat] * losses[feat]

        self.log_dict(train_log_dict, sync_dist=True, commit=False)
        self.log("train_total_loss", total_loss, sync_dist=True, commit=True)
        return total_loss

    def forward(self, g: dgl.DGLHeteroGraph, task_name: str):
        # sample time
        # TODO: what are time sampling methods used in other papers?
        t = torch.rand(g.batch_size, device=g.device).float()

        # maybe not necessary right now, perhaps after we add edges appropriately
        node_batch_idxs, edge_batch_idxs = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        upper_edge_mask = {}
        upper_edge_mask["lig_to_lig"] = lig_ue_mask

        # sample conditional path
        task_class: Task = task_name_to_class(task_name)
        g = self.sample_conditional_path(
            g, task_class, t, node_batch_idxs, edge_batch_idxs, lig_ue_mask
        )

        # forward pass for the vector field
        vf_output = self.vector_field.forward(
            g,
            task_class,
            t,
            node_batch_idx=node_batch_idxs,
            upper_edge_mask=upper_edge_mask,
        )

        targets = {}
        for modality in task_class.modalities_generated:
            data_src = g.edges if modality.graph_entity == "edge" else g.nodes
            dk = modality.data_key
            target = data_src[modality.entity_name].data[f"{dk}_1_true"]
            if modality.graph_entity == "edge":
                target = target[upper_edge_mask[modality.entity_name]]
            if dk in ["a", "c", "e"]:
                if modality.graph_entity == "edge":
                    xt_idxs = data_src[modality.entity_name].data[f"{dk}_t"][
                        upper_edge_mask[modality.entity_name]
                    ]
                else:
                    xt_idxs = data_src[modality.entity_name].data[f"{dk}_t"]
                target[
                    xt_idxs != self.n_cat_dict[modality.name]
                ] = -100  # set the target to ignore_index when the feature is already unmasked in xt
            targets[modality.name] = target

        if self.time_scaled_loss:
            time_weights = {}
            for modality in task_class.modalities_generated:
                time_weights[modality.name] = torch.ones_like(t)
                time_weights[modality.name] = (
                    time_weights[modality.name] / time_weights[modality.name].sum()
                )  # TODO: actually implement this

        losses = {}
        for modality in task_class.modalities_generated:
            if self.time_scaled_loss:
                weight = time_weights[modality.name]
                if modality.graph_entity == "edge":
                    weight = weight[edge_batch_idxs[modality.entity_name]][
                        upper_edge_mask[modality.entity_name]
                    ]
                else:
                    weight = weight[node_batch_idxs[modality.entity_name]]
                weight = weight.unsqueeze(-1)
            else:
                weight = 1.0
            target = targets[modality.name]
            losses[modality.name] = (
                self.loss_fn_dict[modality.name](vf_output[modality.name], target)
                * weight
            )
            if self.time_scaled_loss:
                losses[modality.name] = losses[modality.name].mean()
        return losses

    def configure_optimizers(self):
        # implement optimizer
        pass

    def sample_conditional_path(
        self,
        g: dgl.DGLHeteroGraph,
        task_class: Task,
        t: torch.Tensor,
        node_batch_idxs: Dict[str, torch.Tensor],
        edge_batch_idxs: Dict[str, torch.Tensor],
        lig_ue_mask: torch.Tensor,
    ):
        # TODO: support arbitrary alpha and beta functions, independently for each modality
        modalities_generated = task_class.modalities_generated
        alpha_t = {modality.name: t for modality in modalities_generated}
        beta_t = {modality.name: 1 - t for modality in modalities_generated}

        # for all modalities being generated, sample the conditional path
        conditonal_path_fns = get_conditional_path_fns(
            task_class, self.conditional_path_config
        )
        for modality_name in conditonal_path_fns:
            modality: Modality = name_to_modality(modality_name)
            conditional_path_name, conditional_path_fn = conditonal_path_fns[
                modality_name
            ]

            data_src = g.edges if modality.graph_entity == "edge" else g.nodes
            dk = modality.data_key
            source = data_src[modality.entity_name].data[f"{dk}_0"]
            target = data_src[modality.entity_name].data[f"{dk}_1_true"]

            if modality.graph_entity == "edge":
                conditional_path_fn = partial(conditional_path_fn, ue_mask=lig_ue_mask)

            if conditional_path_name == "ctmc_mask":
                n_categories = self.n_categories_dict[modality_name]
                conditional_path_fn = partial(
                    conditional_path_fn, n_categories=n_categories
                )

            # expand alpha_t and beta_t for the nodes/edges
            if modality.graph_entity == "node":
                batch_idxs = node_batch_idxs[modality.entity_name]
            else:
                batch_idxs = edge_batch_idxs[modality.entity_name]
            alpha_t_modality = alpha_t[modality_name][batch_idxs].unsqueeze(-1)
            beta_t_modality = beta_t[modality_name][batch_idxs].unsqueeze(-1)

            data_src[modality.entity_name].data[f"{dk}_t"] = conditional_path_fn(
                source, target, alpha_t_modality, beta_t_modality
            )

        # for all modalities held fixed, convert the true values to the current time
        for modality in task_class.modalities_fixed:
            data_src = g.edges if modality.graph_entity == "edge" else g.nodes
            dk = modality.data_key
            data_src[modality.entity_name].data[f"{dk}_t"] = data_src[
                modality.entity_name
            ].data[f"{dk}_1_true"]

        return g
