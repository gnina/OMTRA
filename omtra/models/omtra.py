import torch
import torch.nn.functional as fn
import torch.nn as nn
import pytorch_lightning as pl
import dgl
from typing import Dict, List, Callable, Tuple
from collections import defaultdict
import wandb
import itertools
import numpy as np
from functools import partial
import time
import hydra

from omtra.load.conf import TaskDatasetCoupling, build_td_coupling
from omtra.data.graph import build_complex_graph
from omtra.data.graph.utils import get_batch_idxs, get_upper_edge_mask, copy_graph, build_lig_edge_idxs
from omtra.tasks.tasks import Task
from omtra.tasks.register import task_name_to_class
from omtra.tasks.modalities import Modality, name_to_modality
from omtra.constants import lig_atom_type_map, ph_idx_to_type, charge_map
from omtra.models.conditional_paths.path_factory import get_conditional_path_fns
from omtra.models.vector_field import VectorField
from omtra.models.interpolant_scheduler import InterpolantScheduler
from omtra.data.distributions.plinder_dists import sample_n_lig_atoms_plinder, sample_n_pharms_plinder
from omtra.data.distributions.pharmit_dists import sample_n_lig_atoms_pharmit, sample_n_pharms_pharmit
from omegaconf import DictConfig, OmegaConf

from omtra.priors.prior_factory import get_prior
from omtra.priors.sample import sample_priors


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
        total_loss_weights: Dict[str, float] = {},
        ligand_encoder: DictConfig = DictConfig({}),
        ligand_encoder_checkpoint: str = None,
    ):
        super().__init__()

        self.dists_file = dists_file
        self.graph_config = graph_config
        self.conditional_path_config = conditional_paths
        self.optimizer_cfg = optimizer

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
        self.n_categories_dict = {
            "lig_a": len(lig_atom_type_map),
            "lig_c": len(charge_map),
            "lig_e": 4,  # hard-coded assumption of 4 bond types (none, single, double, triple)
            "pharm_a": len(ph_idx_to_type),
        }
        self.time_scaled_loss = False
        self.interpolant_scheduler = InterpolantScheduler(schedule_type="linear")
        self.vector_field =  hydra.utils.instantiate(
            vector_field,
            td_coupling=self.td_coupling,
            interpolant_scheduler=self.interpolant_scheduler,
            graph_config=self.graph_config,
        )

        if not ligand_encoder.is_empty():
            self.ligand_encoder = hydra.utils.instantiate(ligand_encoder)
            if ligand_encoder_checkpoint is not None:
                ligand_encoder_pre = type(self.ligand_encoder).load_from_checkpoint(ligand_encoder_checkpoint)
                self.ligand_encoder.load_state_dict(ligand_encoder_pre.state_dict())
        else:
            self.ligand_encoder = None

        self.configure_loss_fns()

        self.save_hyperparameters(ignore=["ligand_encoder_checkpoint"])
    
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
                           
    def configure_loss_fns(self):

        if self.time_scaled_loss:
            reduction = 'none'
        else:
            reduction = 'mean'

        # build loss function dict
        self.loss_fn_dict = {}

        # get all modalities generated by the model
        modalities_generated = set(m
            for task_name in self.td_coupling.task_space
            for m in task_name_to_class(task_name).modalities_generated)
        
        # create loss functions for each modality
        for modality in modalities_generated:
            if modality.is_categorical:
                self.loss_fn_dict[modality.name] = nn.CrossEntropyLoss(reduction=reduction, ignore_index=-100)
            else:
                self.loss_fn_dict[modality.name] = nn.MSELoss(reduction=reduction)


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
        for key, loss in losses.items():
            train_log_dict[f"{key}_train_loss"] = loss

        total_loss = torch.zeros(1, device=g.device, requires_grad=True)
        # TODO: loss weighting scheme?
        for loss_name, loss_val in losses.items():
            total_loss = total_loss + 1.0 * loss_val

        # train_log_dict["train_total_loss"] = total_loss
        self.log_dict(train_log_dict, sync_dist=True)
        self.log(
            "train_total_loss",
            total_loss,
            prog_bar=True,
            sync_dist=True,
            on_step=True,
        )

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
                target[xt_idxs != modality.n_categories] = -100  
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
            if modality.is_node and g.num_nodes(modality.entity_name) == 0:
                losses[modality.name] = torch.tensor(0.0, device=g.device)
                continue
            losses[modality.name] = (
                self.loss_fn_dict[modality.name](vf_output[modality.name], target)
                * weight
            )
            if self.time_scaled_loss:
                losses[modality.name] = losses[modality.name].mean()
        return losses

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())
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
        alpha_t = {modality.name: t for modality in modalities_generated}
        beta_t = {modality.name: 1 - t for modality in modalities_generated}

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

            if modality.is_categorical:
                conditional_path_fn = partial(
                    conditional_path_fn, n_categories=modality.n_categories
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
            data_src = g.nodes if modality.is_node else g.edges
            dk = modality.data_key
            data_src[modality.entity_name].data[f"{dk}_t"] = data_src[
                modality.entity_name
            ].data[f"{dk}_1_true"]

        return g


    @torch.no_grad()
    def sample(self, 
               task_name: str, 
               g_list: List[dgl.DGLHeteroGraph] = None, # list of graphs containing conditional information (receptor structure, pharmacphore, ligand identity, etc)
               n_replicates: int = 1, # number of replicates samples to draw per conditioning input in g_list, or just number of samples if a fully unconditional task
               coms: torch.Tensor = None, # center of mass for adding ligands/pharms to systems
               unconditional_n_atoms_dist: str = 'plinder', # distribution to use for sampling number of atoms in unconditional tasks
    ):

        task: Task = task_name_to_class(task_name)
        groups_generated = task.groups_generated
        groups_present = task.groups_present
        groups_fixed = task.groups_fixed

        # TODO: user-supplied n_atoms dict?

        # unless this is a completely and totally unconditional task, the user
        # has to provide the conditional information in the graph
        if set(groups_generated) != set(groups_present) and g_list is None:
            raise ValueError(
                f"Task {task_name} requires a user-provided graphs with conditional information, but none was provided."
            )
        
        # if this is purely unconditional sampling
        # we create initial graphs with no data
        g_flat: List[dgl.DGLHeteroGraph] = []
        if g_list is None:
            g_flat = []
            for _ in range(n_replicates):
                g_list.append( build_complex_graph(
                    node_data={},
                    edge_idxs={},
                    edge_data={},
                ))
            coms_flat = [None] * len(g_flat)
        else:
            # otherwise, we need to copy the graphs out n_replicates times
            g_flat: List[dgl.DGLHeteroGraph] = []
            coms_flat = []
            for idx, g_i in enumerate(g_list):
                g_flat.extend(copy_graph(g_i, n_replicates))

                if coms is None:
                    com_i = g_i.nodes['prot_atom'].data['x_1_true'].mean(dim=0)
                else:
                    com_i = coms[idx]

                coms_flat.extend([com_i] * n_replicates)

        # TODO: sample number of ligand atoms
        protein_present = 'protein_structure' in groups_present
        add_ligand = 'ligand_identity' in groups_generated
        if protein_present and add_ligand:

            n_prot_atoms = torch.tensor([ g.num_nodes('prot_atom') for g in g_flat])
            if 'pharmacophore' in groups_fixed:
                n_pharms = torch.tensor([ g.num_nodes('pharm') for g in g_flat])
            else:
                n_pharms = None
            n_lig_atoms = sample_n_lig_atoms_plinder(n_prot_atoms=n_prot_atoms, n_pharms=n_pharms)
            # if protein is present, sample ligand atoms from p(n_ligand_atoms|n_protein_atoms,n_pharm_atoms)
            # if pharm atoms not present, we marginalize over n_pharm_atoms - this distribution from plinder dataset
        elif not protein_present and add_ligand:
            if 'pharmacophore' in groups_fixed:
                n_pharms = torch.tensor([ g.num_nodes('pharm') for g in g_flat])
                n_samples = None
            else:
                n_pharms = None
                n_samples = len(g_flat)

            if unconditional_n_atoms_dist == 'plinder':
                n_lig_atoms = sample_n_lig_atoms_plinder(n_pharms=n_pharms, n_samples=n_samples)
            elif unconditional_n_atoms_dist == 'pharmit':
                n_lig_atoms = sample_n_lig_atoms_pharmit(n_pharms=n_pharms, n_samples=n_samples)
            else:
                raise ValueError(f"Unrecognized dist {unconditional_n_atoms_dist}")
            # TODO: if no protein is present, sample p(n_ligand_atoms|n_pharm_atoms), marginalizing if n_pharm_atoms is not present
            # in this case, the distrbution could come from pharmit or plinder dataset..user-chosen option?

        
        if add_ligand:
            for g_idx, g_i in enumerate(g_flat):
                # add lig atoms to each graph
                g_i.add_nodes(n_lig_atoms[g_idx].item(), ntype="lig",)

                # add lig_to_lig edges to each graph
                edge_idxs = build_lig_edge_idxs(n_lig_atoms[g_idx].item())
                assert edge_idxs.shape[0] == 2
                g_i.add_edges(u=edge_idxs[0], v=edge_idxs[1], etype="lig_to_lig")


        add_pharm = 'pharmacophore' in groups_generated
        if protein_present and add_pharm:

            if 'ligand_identity' in groups_present:
                n_lig_atoms = torch.tensor([ g.num_nodes('lig') for g in g_flat])
            else:
                n_lig_atoms = None

            n_prot_atoms = torch.tensor([ g.num_nodes('prot_atom') for g in g_flat])

            n_pharm_nodes = sample_n_lig_atoms_plinder(n_prot_atoms=n_prot_atoms, n_lig_atoms=n_lig_atoms)
        elif not protein_present and add_pharm:
            # TODO sample pharm atoms given n_ligand_atoms, can use plinder or pharmit dataset distributions
            if 'ligand_identity' in groups_present:
                n_lig_atoms = torch.tensor([ g.num_nodes('lig') for g in g_flat])
            else:
                raise ValueError("did not anticipate sampling pharmacophores without ligand or protein present")
            
            if unconditional_n_atoms_dist == 'plinder':
                n_pharm_nodes = sample_n_pharms_plinder(n_lig_atoms=n_lig_atoms)
            elif unconditional_n_atoms_dist == 'pharmit':
                n_pharm_nodes = sample_n_pharms_pharmit(n_lig_atoms=n_lig_atoms)
            else:
                raise ValueError(f"Unrecognized dist {unconditional_n_atoms_dist}")
    
        if add_pharm:
            for g_idx, g_i in enumerate(g_flat):
                # add pharm nodes to each graph
                g_i.add_nodes(n_pharm_nodes[g_idx].item(), ntype="pharm",)

        # TODO: batch the graphs
        g = dgl.batch(g_flat)
        if coms_flat[0] is None:
            com_batch = None
        else:
            com_batch = torch.stack(coms_flat, dim=0)

        # sample prior distributions for each modality
        prior_fns = get_prior(task, self.prior_config, training=False)
        g = sample_priors(g, task, prior_fns, training=False, com=com_batch)

        # set x_0 to x_t for modalities being generated
        for m in task.modalities_generated:
            data_src = g.nodes if m.is_node else g.edges
            dk = m.data_key
            data_src.data[f'{dk}_t'] = data_src.data[f'{dk}_0']
            

        # set x_1_true to x_t for modalities fixed
        for m in task.modalities_fixed:
            data_src = g.nodes if m.is_node else g.edges
            dk = m.data_key
            data_src.data[f'{dk}_t'] = data_src.data[f'{dk}_1_true']

        # pass graph to vector field..
        itg_result = self.vector_field.integrate(
            g,
            task,
        )

        # vector field returns DGL graph?
        # unbatch DGL graphs and convert to SampledSystem object


        
        