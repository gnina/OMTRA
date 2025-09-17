import torch
import torch.nn.functional as fn
import torch.nn as nn
import pytorch_lightning as pl
import dgl
from typing import Dict, List, Callable, Tuple, Optional, Union
from collections import defaultdict
import wandb
import itertools
import numpy as np
from functools import partial
import time
from pathlib import Path
import hydra
import os
import functools

from omtra.load.conf import TaskDatasetCoupling, build_td_coupling
from omtra.data.graph import build_complex_graph
from omtra.data.graph.utils import (
    get_batch_idxs,
    get_upper_edge_mask,
    copy_graph,
    build_lig_edge_idxs,
)
from omtra.eval.system import SampledSystem
from omtra.tasks.tasks import Task
from omtra.tasks.register import task_name_to_class
from omtra.tasks.modalities import Modality, name_to_modality
from omtra.constants import lig_atom_type_map, ph_idx_to_type, charge_map
from omtra.models.conditional_paths.path_factory import get_conditional_path_fns
from omtra.models.vector_field import VectorField
from omtra.models.interpolant_scheduler import InterpolantScheduler
from omtra.data.distributions.plinder_dists import (
    sample_n_lig_atoms_plinder,
    sample_n_pharms_plinder,
)
from omtra.data.distributions.pharmit_dists import (
    sample_n_lig_atoms_pharmit,
    sample_n_pharms_pharmit,
)
from omtra.aux_losses.register import aux_loss_name_to_class

from omegaconf import DictConfig, OmegaConf

from omtra.priors.prior_factory import get_prior
from omtra.priors.sample import sample_priors
from omtra.eval.register import get_eval
from omtra.eval.utils import add_task_prefix
from omtra.data.condensed_atom_typing import CondensedAtomTyper
import traceback

# from line_profiler import profile

class OMTRA(pl.LightningModule):
    def __init__(
        self,
        task_phases,
        task_dataset_coupling,
        dists_file: str,
        graph_config: DictConfig,
        conditional_paths: DictConfig,
        optimizer: DictConfig,
        vector_field: DictConfig,
        aux_losses: DictConfig,
        total_loss_weights: Dict[str, float] = {},
        ligand_encoder: DictConfig = DictConfig({}),
        ligand_encoder_checkpoint: Optional[str] = None,
        prior_config: Optional[DictConfig] = None,
        k_checkpoints: int = 20,
        checkpoint_interval: int = 1000,
        og_run_dir: Optional[str] = None,
        fake_atom_p: float = 0.0,
        distort_p: float = 0.0,
        eval_config: Optional[DictConfig] = None,
        zero_bo_loss_weight: float = 1.0,
        train_t_dist: str = 'uniform',
        t_alpha: float = 1.8,
        cat_loss_weight: float = 1.0,
        time_scaled_loss: bool = False,
        pharm_var: float = 0.0,

    ):
        super().__init__()


        self.k_checkpoints: int = k_checkpoints
        self.checkpoint_interval: int = checkpoint_interval

        self.dists_file = dists_file
        self.graph_config = graph_config
        self.conditional_path_config = conditional_paths
        self.optimizer_cfg = optimizer
        self.prior_config = prior_config
        self.eval_config = eval_config
        self.og_run_dir = og_run_dir
        self.fake_atom_p = fake_atom_p
        self.use_fake_atoms = self.fake_atom_p > 0
        self.distort_p = distort_p
        self.zero_bo_loss_weight = zero_bo_loss_weight
        self.aux_loss_cfg = aux_losses
        self.cat_loss_weight = cat_loss_weight
        self.pharm_var = pharm_var

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
        # dists_dict = np.load(self.dists_file)
        # TODO: update with extra feature sizes?
        self.n_categories_dict = {
            "lig_a": len(lig_atom_type_map),
            "lig_c": len(charge_map),
            "lig_cond_a": 1,
            "lig_e": 4,  # hard-coded assumption of 4 bond types (none, single, double, triple)
            "lig_e_condensed": 4,
            "pharm_a": len(ph_idx_to_type),
        }
        self.time_scaled_loss = time_scaled_loss
        self.interpolant_scheduler = InterpolantScheduler(schedule_type="linear")
        self.vector_field = hydra.utils.instantiate(
            vector_field,
            td_coupling=self.td_coupling,
            interpolant_scheduler=self.interpolant_scheduler,
            graph_config=self.graph_config,
            fake_atoms=self.fake_atom_p>0.0,
        )

        if not ligand_encoder.is_empty():
            self.ligand_encoder = hydra.utils.instantiate(ligand_encoder)
            if ligand_encoder_checkpoint is not None:
                ligand_encoder_pre = type(self.ligand_encoder).load_from_checkpoint(
                    ligand_encoder_checkpoint
                )
                self.ligand_encoder.load_state_dict(ligand_encoder_pre.state_dict())
        else:
            self.ligand_encoder = None

        self.configure_loss_fns()

        self.save_hyperparameters(ignore=["ligand_encoder_checkpoint"])

        # TODO: don't hard-code crossdocked inclusion
        include_crossdocked = 'crossdocked' in self.td_coupling.dataset_space
        self.cond_a_typer = CondensedAtomTyper(fake_atoms=self.fake_atom_p>0.0, include_crossdocked=include_crossdocked)

        # configure train t distribution 
        if train_t_dist == 'uniform':
            self.train_t_sampler = lambda batch_size, device: torch.rand(batch_size, device=device).float()
        elif train_t_dist == 'beta':
            self.train_t_sampler = lambda batch_size, device: torch.distributions.Beta(t_alpha, 1).sample((batch_size,)).to(device).float()
        else:
            raise ValueError(
                f"Unsupported t distribution: Only uniform or beta time distributions are supported, but specified {train_t_dist}."
            )

    # some code for debugging parameter consistency issues across multiple GPUs
    # def setup(self, stage=None):
    #     if stage == "fit" and torch.distributed.is_initialized():
    #         rank = torch.distributed.get_rank()
    #         total_params = sum(p.numel() for p in self.parameters())
    #         print(f"Rank {rank}, total parameters: {total_params}")

    # Examine parameters and their checksums
    #         for name, param in self.named_parameters():
    # Compute a checksum of parameter values
    # This will detect if parameters have same shape but different values
    #             checksum = torch.sum(param).item()
    #             print(f"Rank {rank}, param {name}, shape {param.shape}, checksum {checksum:.4f}")

    def manual_checkpoint(self, batch_idx: int):

        if batch_idx % self.checkpoint_interval == 0 and batch_idx != 0:
            log_dir = self.og_run_dir
            checkpoint_dir = Path(log_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            current_checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if self.global_rank == 0 and len(current_checkpoints) >= self.k_checkpoints:
                current_checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
                # remove the oldest checkpoint
                oldest_checkpoint = current_checkpoints[0]
                oldest_checkpoint.unlink()

            checkpoint_path = checkpoint_dir / f'batch_{batch_idx}.ckpt'
            print('saving checkpoint to ', checkpoint_path, flush=True)
            self.trainer.save_checkpoint(str(checkpoint_path))
            if self.global_rank == 0:
                try:
                    os.chmod(checkpoint_path, 0o644)  # Readable by others
                except Exception as e:
                    print(f"Error changing permissions for {checkpoint_path}: {e}")
            print(f'Saved checkpoint to {checkpoint_path}')
                
    
    def configure_loss_fns(self):
        if self.time_scaled_loss:
            reduction = "none"
        else:
            reduction = "mean"

        # build loss function dict
        self.loss_fn_dict = nn.ModuleDict()

        # get all modalities generated by the model
        modalities_generated = set(
            m
            for task_name in self.td_coupling.task_space
            for m in task_name_to_class(task_name).modalities_generated
        )

        # create loss functions for each modality
        for modality in modalities_generated:
            if modality.is_categorical:

                if ('lig_e' in modality.name or 'lig_e_condensed' in modality.name) and self.zero_bo_loss_weight != 1.0:
                    weights = torch.ones(modality.n_categories, dtype=torch.float)
                    weights[0] = self.zero_bo_loss_weight
                else:
                    weights = None

                self.loss_fn_dict[modality.name] = nn.CrossEntropyLoss(
                    weight=weights,
                    reduction=reduction, 
                    ignore_index=-100
                )
            else:
                self.loss_fn_dict[modality.name] = nn.MSELoss(reduction=reduction)

        # create auxiliary loss functions
        self.aux_loss_fn_dict = nn.ModuleDict()
        for aux_loss_name, aux_loss_cfg in self.aux_loss_cfg.items():
            aux_loss_class = aux_loss_name_to_class(aux_loss_name)
            self.aux_loss_fn_dict[aux_loss_name] = aux_loss_class(
                **aux_loss_cfg.get('params', {})
            )

    # @profile
    def training_step(self, batch_data, batch_idx):
        g, task_name, dataset_name = batch_data

        # print(f"training step {batch_idx} for task {task_name} and dataset {dataset_name}, rank={self.global_rank}", flush=True)
        self.manual_checkpoint(batch_idx)

        # get the total batch size across all devices
        local_batch_size = torch.tensor([g.batch_size], device=g.device)
        all_batch_counts = self.all_gather(local_batch_size)
        total_batch_count = all_batch_counts.sum().item() / 1000.0

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
        for key, loss in losses.items():
            train_log_dict[f"{key}_train_loss"] = loss

        total_loss = torch.zeros(1, device=g.device, requires_grad=True)
        # TODO: loss weighting scheme?
        for loss_name, loss_val in losses.items():
            total_loss = total_loss + 1.0 * loss_val

        # train_log_dict["train_total_loss"] = total_loss
        train_log_dict = add_task_prefix(train_log_dict, task_name)
        self.log_dict(train_log_dict, sync_dist=False, on_step=True)
        self.log(
            f"{task_name}/train_total_loss",
            total_loss,
            prog_bar=False,
            sync_dist=False,
            on_step=True,
        )
        self.log(
            "train_total_loss",
            total_loss,
            prog_bar=True,
            sync_dist=False,
            on_step=True,
        )

        return total_loss

    def validation_step(self, batch_data, batch_idx):
        g, task_name, dataset_name = batch_data

        # print(f"validation step {batch_idx} for task {task_name} and dataset {dataset_name}, rank={self.global_rank}", flush=True)
        device = g.device
        task = task_name_to_class(task_name)
        if task.unconditional:
            # if the task is purely unconditional, g_list is None
            g_list = None
            n_replicates = 2*g.batch_size
        else:
            g_list = dgl.unbatch(g)
            n_replicates = 2

        if (any(group in task.groups_present for group in ["ligand_identity", "ligand_identity_condensed"])) and 'protein_identity' in task.groups_present:
            coms = dgl.readout_nodes(g, feat='x_1_true', op='mean', ntype='lig')
            coms = [ coms[i] for i in range(g.batch_size) ]
        else:
            coms = None

        self.eval()
        # TODO: n_replicates and n_timesteps should not be hard-coded
        samples = self.sample(task_name, g_list=g_list, n_replicates=n_replicates, n_timesteps=200, device=device, coms=coms)
        samples = [s.to("cpu") for s in samples if s is not None]
        
        if not self.eval_config:
            self.train()
            return 0.0
        
        metrics = {}
        for eval in self.eval_config.get(task_name, []):
            for eval_name, config  in eval.items():
                if not config.get("train", False):
                    continue
                eval_fn = get_eval(eval_name)
                try:
                    eval_output = eval_fn(samples, config.get("params", {}))
                except Exception as e:
                    print(f"WARNING: error occurred while evaluating {eval_name} for task {task_name}")
                    traceback.print_exc()
                    print(f"Full error details for {eval_name}: {str(e)}")
                    continue
                metrics.update(eval_output)
        
        if metrics:
            metrics = add_task_prefix(metrics, task_name)
            sample_count_key = f"{task_name}_{dataset_name}_sample_count"
            metrics[sample_count_key] = self.sample_counts[(task_name, dataset_name)]
            self.log_dict(metrics, sync_dist=True, batch_size=1, on_step=True)

        self.train()
        
        return 0.0

    # @profile
    def forward(self, g: dgl.DGLHeteroGraph, task_name: str):
        # sample time
        t = self.train_t_sampler(batch_size=g.batch_size, device=g.device)

        # maybe not necessary right now, perhaps after we add edges appropriately
        node_batch_idxs, edge_batch_idxs = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        upper_edge_mask = {}
        upper_edge_mask["lig_to_lig"] = lig_ue_mask

        # sample conditional path
        # TODO: ctmc conditional path sampling manually sets things to mask token rather 
        # than setting to prior value (which is usually mask token)
        task_class: Task = task_name_to_class(task_name)
        g = self.sample_conditional_path(
            g, task_class, t, node_batch_idxs, edge_batch_idxs, lig_ue_mask
        )

        if self.distort_p > 0.0:
            t_mask = (t > 0.5)[node_batch_idxs["lig"]]
            distort_mask = torch.rand(g.num_nodes("lig"), 1, device=g.device) < self.distort_p
            distort_mask = distort_mask & t_mask.unsqueeze(-1)
            g.nodes["lig"].data['x_t'] = g.nodes["lig"].data['x_t'] + torch.randn_like(g.nodes["lig"].data['x_t'])*distort_mask*0.5
        
        # add noise to pharmacophore coordinates
        if self.pharm_var > 0.0:
            g.nodes["pharm"].data['x_1_true'] = g.nodes["pharm"].data['x_1_true'] + torch.randn_like(g.nodes["pharm"].data['x_1_true']) * self.pharm_var**0.5

        # forward pass for the vector field
        vf_output = self.vector_field.forward(
            g,
            task_class,
            t,
            node_batch_idx=node_batch_idxs,
            upper_edge_mask=upper_edge_mask,
        )

        # compute targets for each of the flow matching losses
        targets = {}
        for modality in task_class.modalities_generated:
            is_categorical = modality.n_categories and modality.n_categories > 0
            data_src = g.edges if modality.graph_entity == "edge" else g.nodes
            dk = modality.data_key
            target = data_src[modality.entity_name].data[f"{dk}_1_true"]
            if modality.graph_entity == "edge":
                target = target[upper_edge_mask[modality.entity_name]]
            if is_categorical:
                xt_idxs = data_src[modality.entity_name].data[f"{dk}_t"]
                if modality.graph_entity == "edge":
                    xt_idxs = xt_idxs[upper_edge_mask[modality.entity_name]]
                # set the target to ignore_index when the feature is already unmasked in xt
                
                # Get masked atom index
                fake_atoms = (self.fake_atom_p > 0.0) and ((modality.data_key == 'a') or ((modality.data_key == 'cond_a'))) and (modality.entity_name == 'lig') # correction for atom type and fake atoms
                n_categories = modality.n_categories + int(fake_atoms) 
                target[xt_idxs != n_categories] = -100
            targets[modality.name] = target

        # determine if we need to disable time-scale our losses
        disable_time_scaled_loss = self.time_scaled_loss and ('protein_identity' not in task_class.groups_present)

        # get weights for time-scaled losses
        if self.time_scaled_loss:
            t_weights = 1.0 / ((1.0 - t).clamp(min=1e-6)) ** 2    # clamp to avoid division by 0 error
            t_weights = torch.clamp(t_weights, max=100.0)
        else:
            t_weights = None

        # compute all flow matching losses
        losses = {}
        for modality in task_class.modalities_generated:
            is_categorical = modality.is_categorical

            if self.time_scaled_loss:
                mod_weight = t_weights

                if is_categorical:
                    mod_weight = mod_weight * self.cat_loss_weight

                if modality.graph_entity == "edge":
                    mod_weight = mod_weight[edge_batch_idxs[modality.entity_name]][
                        upper_edge_mask[modality.entity_name]
                    ]
                else:
                    mod_weight = mod_weight[node_batch_idxs[modality.entity_name]]
                mod_weight = mod_weight.unsqueeze(-1)

            elif is_categorical:
                mod_weight = self.cat_loss_weight
            else:
                mod_weight = 1.0


            inp = vf_output[modality.name]
            target = targets[modality.name]

            # skip loss if there are no nodes of this type
            if modality.is_node and g.num_nodes(modality.entity_name) == 0:
                losses[modality.name] = torch.tensor(0.0, device=g.device)
                continue

            if disable_time_scaled_loss and modality.is_categorical:
                mod_loss = fn.cross_entropy(
                    inp, target, reduction="mean", ignore_index=-100
                )
            elif disable_time_scaled_loss and not modality.is_categorical:
                mod_loss = fn.mse_loss(inp, target, reduction="mean")
            else:
                mod_loss = self.loss_fn_dict[modality.name](inp, target)

            losses[modality.name] = mod_loss
            if self.time_scaled_loss:
                losses[modality.name] = losses[modality.name].mean()

        # compute auxiliary losses
        for aux_loss_name, aux_loss_fn in self.aux_loss_fn_dict.items():
            if not aux_loss_fn.supports_task(task_class):
                continue
            aux_loss = aux_loss_fn(
                g, 
                vf_output, 
                task_class, 
                node_batch_idxs,
                lig_ue_mask,
                time_weights=None if disable_time_scaled_loss else t_weights
                )
            if self.time_scaled_loss:
                aux_loss = aux_loss.mean()
            losses[aux_loss_name] = aux_loss

        return losses

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.optimizer_cfg, params=self.parameters()
        )
        return optimizer

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
        alpha_t, beta_t = self.interpolant_scheduler.weights(t, task_class)

        # for all modalities being generated, sample the conditional path
        conditonal_path_fns = get_conditional_path_fns(
            task_class, self.conditional_path_config
        )
        for modality_name in conditonal_path_fns:
            modality: Modality = name_to_modality(modality_name)

            # skip modalities that are not present in the graph (for example a system with no npndes)
            if modality.is_node and g.num_nodes(modality.entity_name) == 0:
                continue
            elif not modality.is_node and g.num_edges(modality.entity_name) == 0:
                continue

            conditional_path_name, conditional_path_fn = conditonal_path_fns[
                modality_name
            ]

            data_src = g.nodes if modality.is_node else g.edges
            dk = modality.data_key
            source = data_src[modality.entity_name].data[f"{dk}_0"]
            target = data_src[modality.entity_name].data[f"{dk}_1_true"]

            if modality.graph_entity == "edge":
                conditional_path_fn = partial(conditional_path_fn, ue_mask=lig_ue_mask)

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
            data_src = g.nodes if modality.is_node else g.edges
            dk = modality.data_key
            data_src[modality.entity_name].data[f"{dk}_t"] = data_src[
                modality.entity_name
            ].data[f"{dk}_1_true"]

        return g

    @torch.no_grad()
    def sample(
        self,
        task_name: str,
        g_list: Optional[
            List[dgl.DGLHeteroGraph]
        ] = None,  # list of graphs containing conditional information (receptor structure, pharmacphore, ligand identity, etc)
        n_replicates: int = 1,  # number of replicates samples to draw per conditioning input in g_list, or just number of samples if a fully unconditional task
        coms: Optional[
            torch.Tensor
        ] = None,  # center of mass for adding ligands/pharms to systems
        unconditional_n_atoms_dist: str = None,  # distribution to use for sampling number of atoms in unconditional tasks
        n_timesteps: int = None,
        device: Optional[torch.device] = None,
        visualize=False,
        extract_latents_for_confidence=False,
        time_spacing: str = "even",
        stochastic_sampling: bool = False,
        noise_scaler: float = 1.0,
        eps: float = 0.01,
        # use_gt_n_lig_atoms: bool = False,
        n_lig_atom_margin: Union[float, None] = None,

    ) -> List[SampledSystem]:
        task: Task = task_name_to_class(task_name)
        groups_generated = task.groups_generated
        groups_present = task.groups_present
        groups_fixed = task.groups_fixed

        # TODO: user-supplied n_atoms dict?

        # set device
        if device is None and g_list is not None:
            device = g_list[0].device
        elif device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if unconditional_n_atoms_dist is None:
            unconditional_n_atoms_dist = self.infer_n_atoms_dist(task)

        use_gt_n_lig_atoms = n_lig_atom_margin is not None

        # unless this is a completely and totally unconditional task, the user
        # has to provide the conditional information in the graph
        if not task.unconditional and g_list is None:
            raise ValueError(
                f"Task {task_name} requires a user-provided graphs with conditional information, but none was provided."
            )

        # if this is purely unconditional sampling
        # we create initial graphs with no data
        protein_present = "protein_structure" in groups_present
        g_flat: List[dgl.DGLHeteroGraph] = []
        if g_list is None:
            g_flat = []
            for _ in range(n_replicates):
                g_flat.append(
                    build_complex_graph(
                        node_data={},
                        edge_idxs={},
                        edge_data={},
                    )
                )
            coms_flat = [torch.zeros(3, dtype=float)] * len(g_flat)
        else:
            # otherwise, we need to copy the graphs out n_replicates times
            g_flat: List[dgl.DGLHeteroGraph] = []
            coms_flat = []
            for idx, g_i in enumerate(g_list):
                g_flat.extend(copy_graph(g_i, n_replicates))

                if coms is None and protein_present:
                    com_i = g_i.nodes["prot_atom"].data["x_1_true"].mean(dim=0)
                elif coms is None and not protein_present:
                    com_i = torch.zeros(3, dtype=float)
                else:
                    com_i = coms[idx]

                coms_flat.extend([com_i] * n_replicates)

        # TODO: sample number of ligand atoms
        add_ligand = any(group in groups_generated for group in ["ligand_identity", "ligand_identity_condensed"])

        # find number of fake atoms added to each system by the dataset class
        
        if add_ligand and use_gt_n_lig_atoms and self.fake_atom_p > 0.0:
            n_fake_atoms_gt = []
            for g in g_flat:
                if 'cond_a_1_true' not in g.nodes['lig'].data:
                    raise NotImplementedError('expected there to be a ground-truth ligand, and expected condensed atom types to be used')

                n_fake_atoms_gt.append(
                    (g.nodes['lig'].data['cond_a_1_true'] == self.cond_a_typer.fake_atom_idx).sum().item()
                )
            n_fake_atoms_gt = torch.tensor(n_fake_atoms_gt)
        else:
            n_fake_atoms_gt = torch.zeros(len(g_flat), dtype=torch.long)

        
        if protein_present and add_ligand:
            n_prot_atoms = torch.tensor([g.num_nodes("prot_atom") for g in g_flat])
            if "pharmacophore" in groups_fixed:
                n_pharms = torch.tensor([g.num_nodes("pharm") for g in g_flat])
            else:
                n_pharms = None
            
            # use ground truth number of lig atoms
            if use_gt_n_lig_atoms:

                base_n_atoms = torch.tensor([g.num_nodes("lig") for g in g_flat])
                base_n_atoms = base_n_atoms - n_fake_atoms_gt
                max_margins = torch.clamp(base_n_atoms * n_lig_atom_margin, min=0).int()
                margins = torch.stack([
                    torch.randint(0, int(m.item()) + 1, (1,))
                    for m in max_margins
                    ]).squeeze()
                offset_sign = torch.randint(0, 2, (len(g_flat),)) * 2 - 1
                random_offsets = margins * offset_sign
                n_lig_atoms = base_n_atoms + random_offsets
                n_lig_atoms = torch.clamp(n_lig_atoms, min=4)
            
            else:
                n_lig_atoms = sample_n_lig_atoms_plinder(
                    n_prot_atoms=n_prot_atoms, n_pharms=n_pharms
                )

        # if protein is present, sample ligand atoms from p(n_ligand_atoms|n_protein_atoms,n_pharm_atoms)
        # if pharm atoms not present, we marginalize over n_pharm_atoms - this distribution from plinder dataset
        elif not protein_present and add_ligand:
            if "pharmacophore" in groups_fixed:
                n_pharms = torch.tensor([g.num_nodes("pharm") for g in g_flat])
                n_samples = None
            else:
                n_pharms = None
                n_samples = len(g_flat)

            # use ground truth number of lig atoms
            if use_gt_n_lig_atoms:
                base_n_atoms = torch.tensor([g.num_nodes("lig") for g in g_flat])
                base_n_atoms = base_n_atoms - n_fake_atoms_gt
                margins = torch.clamp(base_n_atoms * n_lig_atom_margin, min=1).int()
                random_offsets = torch.randint(-margins.max(), margins.max() + 1, (len(g_flat),))
                # Clamp the offsets to respect per-sample margins
                random_offsets = torch.clamp(random_offsets, -margins, margins)
                n_lig_atoms = base_n_atoms + random_offsets
                n_lig_atoms = torch.clamp(n_lig_atoms, min=4)

            elif unconditional_n_atoms_dist == "plinder":
                n_lig_atoms = sample_n_lig_atoms_plinder(
                    n_pharms=n_pharms, n_samples=n_samples
                )
            elif unconditional_n_atoms_dist == "pharmit":
                n_lig_atoms = sample_n_lig_atoms_pharmit(
                    n_pharms=n_pharms, n_samples=n_samples
                )
            else:
                raise ValueError(f"Unrecognized dist {unconditional_n_atoms_dist}")
            # TODO: if no protein is present, sample p(n_ligand_atoms|n_pharm_atoms), marginalizing if n_pharm_atoms is not present
            # in this case, the distrbution could come from pharmit or plinder dataset..user-chosen option?

        if add_ligand:

            if (self.fake_atom_p > 0.0) and not use_gt_n_lig_atoms: # don't add fake atoms 
                n_real_atoms = n_lig_atoms
                max_num_fake_atoms = torch.ceil(n_real_atoms*self.fake_atom_p).float()
                frac = torch.rand_like(max_num_fake_atoms)
                num_fake_atoms = torch.floor(frac*(max_num_fake_atoms+1)).long()
                n_real_atoms = n_real_atoms + num_fake_atoms

            for g_idx, g_i in enumerate(g_flat):
                # clear ligand nodes (and edges) if they exist
                if g_i.num_nodes("lig") > 0:
                    lig_node_ids = torch.arange(g_i.num_nodes("lig"), device=g_i.device)
                    g_i.remove_nodes(lig_node_ids, ntype="lig")
                    
                # add lig atoms to each graph
                g_i.add_nodes(
                    n_lig_atoms[g_idx].item(),
                    ntype="lig",
                )

                # add lig_to_lig edges to each graph
                edge_idxs = build_lig_edge_idxs(n_lig_atoms[g_idx].item()).to(
                    g_i.device
                )
                assert edge_idxs.shape[0] == 2
                g_i.add_edges(u=edge_idxs[0], v=edge_idxs[1], etype="lig_to_lig")
        
        add_pharm = "pharmacophore" in groups_generated
        if protein_present and add_pharm:
            if any(group in task.groups_present for group in ["ligand_identity", "ligand_identity_condensed"]):
                n_lig_atoms = torch.tensor([g.num_nodes("lig") for g in g_flat])
            else:
                n_lig_atoms = None

            n_prot_atoms = torch.tensor([g.num_nodes("prot_atom") for g in g_flat])

            n_pharm_nodes = sample_n_pharms_plinder(
                n_prot_atoms=n_prot_atoms, n_lig_atoms=n_lig_atoms
            )
        elif not protein_present and add_pharm:
            # TODO sample pharm atoms given n_ligand_atoms, can use plinder or pharmit dataset distributions
            if any(group in task.groups_present for group in ["ligand_identity", "ligand_identity_condensed"]):
                n_lig_atoms = torch.tensor([g.num_nodes("lig") for g in g_flat])
            else:
                raise ValueError(
                    "did not anticipate sampling pharmacophores without ligand or protein present"
                )

            if unconditional_n_atoms_dist == "plinder":
                n_pharm_nodes = sample_n_pharms_plinder(n_lig_atoms=n_lig_atoms)
            elif unconditional_n_atoms_dist == "pharmit":
                n_pharm_nodes = sample_n_pharms_pharmit(n_lig_atoms=n_lig_atoms)
            else:
                raise ValueError(f"Unrecognized dist {unconditional_n_atoms_dist}")

        if add_pharm:
            for g_idx, g_i in enumerate(g_flat):
                # clear pharm nodes (and edges) if they exist
                if g_i.num_nodes("pharm") > 0:
                    pharm_node_ids = torch.arange(g_i.num_nodes("pharm"), device=g_i.device)
                    g_i.remove_nodes(pharm_node_ids, ntype="pharm")
                    
                # add pharm nodes to each graph
                g_i.add_nodes(
                    n_pharm_nodes[g_idx].item(),
                    ntype="pharm",
                )

        # TODO: batch the graphs
        g = dgl.batch(g_flat).to(device)
        com_batch = torch.stack(coms_flat, dim=0).to(device)

        # sample prior distributions for each modality
        prior_fns = get_prior(task, self.prior_config, training=False)
        g = sample_priors(
            g, 
            task, 
            prior_fns, 
            training=False, 
            com=com_batch,
            fake_atoms=self.use_fake_atoms
            )

        # TODO: need to solidify creatioin of upper edge mask (where to do this (not just in sample), etc)
        upper_edge_mask = {}
        # set x_0 to x_t for modalities being generated
        for m in task.modalities_generated:
            if not m.is_node:
                if g.num_edges(m.entity_name) == 0:
                    continue
                upper_edge_mask[m.entity_name] = get_upper_edge_mask(g, m.entity_name).to(device)
            else:
                if g.num_nodes(m.entity_name) == 0:
                    continue
            data_src = g.nodes[m.entity_name] if m.is_node else g.edges[m.entity_name]
            dk = m.data_key
            data_src.data[f"{dk}_t"] = data_src.data[f"{dk}_0"]

        # set x_1_true to x_t for modalities fixed
        for m in task.modalities_fixed:
            if not m.is_node:
                if g.num_edges(m.entity_name) == 0:
                    continue
                upper_edge_mask[m.entity_name] = get_upper_edge_mask(g, m.entity_name)
            else:
                if g.num_nodes(m.entity_name) == 0:
                    continue
            data_src = g.nodes[m.entity_name] if m.is_node else g.edges[m.entity_name]
            dk = m.data_key
            data_src.data[f"{dk}_t"] = data_src.data[f"{dk}_1_true"]


        # optionally set the number of timesteps for the integration
        # TODO: there should be a cleaner way to do this
        # the only reason i'm allowing it to be none by default and manually adding it in
        # is that i don't want to define a default number of timesteps in more than one palce
        # it is already defined as default arg to VectorField.integrate
        itg_kwargs = dict(visualize=visualize, extract_latents_for_confidence=extract_latents_for_confidence, time_spacing=time_spacing, stochastic_sampling=stochastic_sampling, noise_scaler=noise_scaler, eps=eps)
        if n_timesteps is not None:
            itg_kwargs["n_timesteps"] = n_timesteps

        # pass graph to vector field..
        itg_result = self.vector_field.integrate(
            g,
            task,
            upper_edge_mask=upper_edge_mask,
            **itg_kwargs
        )

        if visualize:
            g, per_graph_traj = itg_result
        else:
            g = itg_result

        # integrate returns just a DGL graph for now
        # in the future, when we implement trajectory visualization, it will be a graph + some data structure for trajectory frames
        
        # unbatch DGL graphs and convert to SampledSystem object
        g = g.to('cpu')
        unbatched_graphs = dgl.unbatch(g)
        sampled_systems = []
        for i, g_i in enumerate(unbatched_graphs):
            if visualize:
                ss_kwargs = dict(traj=per_graph_traj[i])
            else:
                ss_kwargs = dict()
            sampled_system = SampledSystem(
                g=g_i,
                task=task,
                fake_atoms=self.use_fake_atoms,
                cond_a_typer=self.cond_a_typer,
                **ss_kwargs,
            )
            sampled_systems.append(sampled_system)
        return sampled_systems
    
    @torch.no_grad()
    def sample_in_batches(
        self,
        task_name: str,
        g_list: Optional[
            List[dgl.DGLHeteroGraph]
        ] = None,  # list of graphs containing conditional information (receptor structure, pharmacphore, ligand identity, etc)
        n_replicates: int = 1,  # number of replicates samples to draw per conditioning input in g_list, or just number of samples if a fully unconditional task
        max_batch_size: int = 500,
        coms: Optional[
            torch.Tensor
        ] = None,  # center of mass for adding ligands/pharms to systems
        unconditional_n_atoms_dist: str = None,  # distribution to use for sampling number of atoms in unconditional tasks
        n_timesteps: int = None,
        device: Optional[torch.device] = None,
        visualize=False,
        extract_latents_for_confidence=False,
        time_spacing: str = "even",
        stochastic_sampling: bool = False,
        noise_scaler: float = 1.0,
        eps: float = 0.01,
        n_lig_atom_margin: Union[float, None] = None,
    ) -> List[SampledSystem]:
        
        n_samples = len(g_list) if g_list is not None else 1
        
        reps_per_batch = min(max_batch_size // n_samples, n_replicates)
        n_full_batches = n_replicates // reps_per_batch
        last_batch_reps = n_replicates % reps_per_batch

        sampled_systems = [[] for _ in range(n_samples)]

        for i in range(n_full_batches):
            batch_results = self.sample(g_list=g_list,
                                        n_replicates=reps_per_batch,
                                        task_name=task_name,
                                        unconditional_n_atoms_dist=unconditional_n_atoms_dist,
                                        device=device,
                                        n_timesteps=n_timesteps,
                                        visualize=visualize,
                                        coms=coms,
                                        extract_latents_for_confidence=extract_latents_for_confidence,
                                        time_spacing=time_spacing,
                                        stochastic_sampling=stochastic_sampling,
                                        noise_scaler=noise_scaler,
                                        eps=eps,
                                        n_lig_atom_margin=n_lig_atom_margin,
                                        )
            # re-order samples
            for i in range(n_samples):
                start_idx = i * reps_per_batch
                end_idx = start_idx + reps_per_batch
                sampled_systems[i].extend(batch_results[start_idx:end_idx])
            
        # last batch
        if last_batch_reps > 0:
            batch_results = self.sample(g_list=g_list,
                                        n_replicates=last_batch_reps,
                                        task_name=task_name,
                                        unconditional_n_atoms_dist=unconditional_n_atoms_dist,
                                        device=device,
                                        n_timesteps=n_timesteps,
                                        visualize=visualize,
                                        coms=coms,
                                        extract_latents_for_confidence=extract_latents_for_confidence,
                                        time_spacing=time_spacing,
                                        stochastic_sampling=stochastic_sampling,
                                        noise_scaler=noise_scaler,
                                        eps=eps,
                                        n_lig_atom_margin=n_lig_atom_margin,
                                        )

            for i in range(n_samples):
                start_idx = i * last_batch_reps
                end_idx = start_idx + last_batch_reps
                sampled_systems[i].extend(batch_results[start_idx:end_idx])

        sampled_systems = [rep for sys in sampled_systems for rep in sys]
        
        return sampled_systems

    @functools.lru_cache()
    def infer_n_atoms_dist(self, task):
        # infer n_atoms_dist if none
        trained_on_pharmit = 'pharmit' in self.td_coupling.dataset_space
        trained_on_plinder = 'plinder' in self.td_coupling.dataset_space
        has_protein = 'protein_identity' in task.groups_present
        if trained_on_pharmit and not trained_on_plinder:
            unconditional_n_atoms_dist = 'pharmit'
        elif trained_on_plinder and not trained_on_pharmit:
            unconditional_n_atoms_dist = 'plinder'
        elif has_protein:
            unconditional_n_atoms_dist = 'plinder'
        elif not has_protein:
            unconditional_n_atoms_dist = 'pharmit'
        return unconditional_n_atoms_dist
