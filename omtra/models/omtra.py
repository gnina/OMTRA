from omtra.utils.graph import g_local_scope
from omtra.utils.training_control import ValidationControlMixin
from omtra.utils import validation_control as vcontrol
from omtra.models.conditioning import prepare_coordinate_condition, prepare_partial_condition, observed, check_noise
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
from omtra.constants import lig_atom_type_map, ph_idx_to_type, charge_map, num_condensed_atom_types
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

class OMTRA(ValidationControlMixin, pl.LightningModule):
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
        pharm_pos_std: float = 0.0,
        pharm_var: float = 0.0,
        fixed_coord_max_std: float = 0.0,
        fixed_token_max_prob: float = 0.0,
        validate_noisy_pharmacophores: bool = False,
        lr_warmup_steps: int = 0,
        lr_pretrained_mult: float = 1.0,
        prot_pos_std: float = 0.0, #standard deviation for adding noise to protein atom positions
        lig_com_offset_std: float = 0.0, #standard deviation for offsetting ligand center of mass
        size_prior_path: Optional[str] = None,
        require_training_size_prior: bool = False,
        isolated_training_rng: bool = False,
        rng_seed: int = 42,
        lig_com_offset_zero_p: float = 0.0, #fraction of samples that get NO ligand COM offset (keeps exact-center precision)
        control_policy: Optional[DictConfig] = None,      # validation-driven LR / stopping policy (omtra.utils.validation_control.Policy as dict); None = legacy LR behaviour
        control_policy_path: Optional[str] = None,        # alternative: path of the frozen calibration/policy JSON (its sha256 is recorded in the checkpoint meta)
        control_run_uuid: Optional[str] = None,
        control_evidence_dir: Optional[str] = None,
        control_poll_every: int = 100,
        control_checkpoint_every: int = 25000,
        control_command_path: Optional[str] = None,
        control_resource_flag: Optional[str] = None,
        control_production: bool = True,
        control_log_path: Optional[str] = None,
        control_meta: Optional[DictConfig] = None,        # recipe/panel hashes and attempt id recorded in the ready manifests

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
        check_noise(pharm_pos_std, 'pharm_pos_std')
        check_noise(prot_pos_std, 'prot_pos_std')
        check_noise(fixed_coord_max_std, 'fixed_coord_max_std')
        check_noise(fixed_token_max_prob, 'fixed_token_max_prob', probability=True)
        self.fixed_coord_max_std = fixed_coord_max_std
        self.fixed_token_max_prob = fixed_token_max_prob
        self.validate_noisy_pharmacophores = validate_noisy_pharmacophores
        self.pharm_pos_std = pharm_pos_std
        self.pharm_var = pharm_var
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_pretrained_mult = float(lr_pretrained_mult)      # learning-rate multiplier of the warm-start (loaded) parameters; new modules keep the full rate
        self.prot_pos_std = prot_pos_std
        self.lig_com_offset_std = lig_com_offset_std
        self.lig_com_offset_zero_p = lig_com_offset_zero_p
        self.size_prior_path = size_prior_path
        self.require_training_size_prior = require_training_size_prior
        self.isolated_training_rng = isolated_training_rng
        self.rng_seed = int(rng_seed)

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
            "lig_cond_a": num_condensed_atom_types,
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

        if (fixed_coord_max_std > 0 or fixed_token_max_prob > 0) and not self.vector_field.soft_partial_conditioning:
            raise ValueError('Partial training noise requires soft_partial_conditioning=True')

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

        if control_policy is None and control_policy_path:
            import json as _json, hashlib as _hashlib
            raw = open(control_policy_path, 'rb').read()
            control_policy = _json.loads(raw)['policy']
            control_meta = dict(control_meta or {}, policy_file=str(control_policy_path), policy_file_sha256=_hashlib.sha256(raw).hexdigest())
        if control_policy is not None:
            policy = vcontrol.Policy.from_dict(OmegaConf.to_container(control_policy, resolve=True) if isinstance(control_policy, DictConfig) else dict(control_policy))
            if int(lr_warmup_steps) != int(policy.warmup):
                raise ValueError(f'lr_warmup_steps {lr_warmup_steps} != control policy warmup {policy.warmup}: one LR owner needs one warmup')
            if lr_pretrained_mult != 1.0:
                raise ValueError('from-scratch validation-control recipe cannot use lr_pretrained_mult')
            self.setup_control(policy, control_run_uuid, control_evidence_dir, poll_every=control_poll_every, command_path=control_command_path,
                               resource_flag_path=control_resource_flag, production=control_production,
                               log_path=control_log_path or (str(Path(og_run_dir) / 'control_events.jsonl') if og_run_dir else None))
            self.control_ckpt_every = int(control_checkpoint_every)
            self.control_meta = OmegaConf.to_container(control_meta, resolve=True) if isinstance(control_meta, DictConfig) else dict(control_meta or {})
            if self.control_ckpt_every % self.control_poll_every != 0:
                raise ValueError('control_checkpoint_every must be a multiple of control_poll_every')

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

            current_checkpoints = [p for p in checkpoint_dir.glob("*.ckpt") if p.stem.startswith("batch_")]     # filter out last.ckpt and only keep batch checkpointss
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
        if not getattr(self, 'control_enabled', False):
            self.manual_checkpoint(batch_idx)       # legacy path; the validation-control path saves in on_train_batch_end by optimizer step

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

        import os
        if os.environ.get('OMTRA_BATCH_HASH_LOG') and self.global_rank == 0:
            # matched-pair audit: identity hash of the batch actually trained on (sample ids), written for later comparison between runs
            import hashlib
            sid = g.nodes['lig'].data['sample_id'] if 'sample_id' in g.nodes['lig'].data else None
            # hash of the UNIQUE (dataset_idx, local_idx) pairs: invariant to per-atom repetition and to augmentation that changes atom counts (fake atoms)
            digest = hashlib.sha256(torch.unique(sid.cpu(), dim=0).numpy().tobytes()).hexdigest()[:16] if sid is not None else 'none'
            with open(os.environ['OMTRA_BATCH_HASH_LOG'], 'a') as stream:
                stream.write(f"{int(self.global_step)},{task_name},{dataset_name},{digest},{g.num_nodes('lig')}\n")

        # forward pass
        losses = self(g, task_name)
        self._pair_audit_collect(g, task_name, losses)

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

    def _pair_audit_collect(self, g, task_name, losses):
        """Matched-pair audit (env OMTRA_PAIR_AUDIT_LOG, first OMTRA_PAIR_AUDIT_STEPS updates): what the arms actually saw and what the correction module did."""
        import os
        path = os.environ.get('OMTRA_PAIR_AUDIT_LOG')
        self._audit_rec = None
        if not path or self.global_rank != 0 or int(self.global_step) >= int(os.environ.get('OMTRA_PAIR_AUDIT_STEPS', '200')):
            return
        import hashlib
        sc = getattr(self.vector_field, 'stereo_correction', None)
        if sc is not None:
            sc.record = True
        t = getattr(self, 'last_flow_times', None)
        nd = g.nodes['lig'].data
        rec = {'step': int(self.global_step), 'task': task_name,
               't_hash': hashlib.sha256(t.numpy().tobytes()).hexdigest()[:12] if t is not None else None,
               'x0_hash': hashlib.sha256(nd['x_0'].detach().cpu().numpy().tobytes()).hexdigest()[:12] if 'x_0' in nd else None,
               'bootstrap': bool(getattr(self.vector_field, 'last_bootstrap', False)),
               'losses': {k: float(v) for k, v in losses.items()}}
        try:
            from omtra.data.requested_stereo import eligible
            from omtra.tasks.register import task_name_to_class
            is_eligible = bool(eligible(task_name_to_class(task_name)))
        except Exception:
            is_eligible = None
        if 'stereo_centre' in nd and 'stereo_atom' in nd:
            labelled = (nd['stereo_atom'] >= 2) & (nd['stereo_atom'] <= 5)
            rec['stereo'] = {'eligible': is_eligible, 'graphs': int(g.batch_size), 'true_centres': int(nd['stereo_centre'].sum()),
                             'visible_labelled_centres': int((labelled & (nd['stereo_centre'] > 0)).sum()),
                             'hidden_centres': int(((~labelled) & (nd['stereo_centre'] > 0)).sum())}
        if sc is not None:
            rec['coverage'] = dict(sc.coverage)
            if sc.stats:
                import numpy as np
                arr = np.array(sc.stats)
                rec['force'] = {'calls': int(len(arr)), 'mean_norm': float(arr[:, 0].mean()), 'max_norm': float(arr[:, 1].max()), 'frac_atoms_nonzero': float(arr[:, 2].mean())}
        self._audit_rec = rec

    def on_after_backward(self):
        rec = getattr(self, '_audit_rec', None)
        if rec is None:
            return
        import json, os
        sc = getattr(self.vector_field, 'stereo_correction', None)
        if sc is not None:
            for name in ('w_current', 'w_pred'):
                p = getattr(sc, name)
                rec[f'{name}_grad_norm'] = float(p.grad.norm()) if p.grad is not None else None
                rec[f'{name}_value_norm'] = float(p.detach().norm())
        with open(os.environ['OMTRA_PAIR_AUDIT_LOG'], 'a') as stream:
            stream.write(json.dumps(rec) + '\n')
        self._audit_rec = None

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
        samples = self.sample(
            task_name, 
            g_list=g_list, 
            n_replicates=n_replicates, 
            n_timesteps=200, 
            n_lig_atom_margin=0.1,
            device=device, 
            coms=coms)
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
        
        if self.validate_noisy_pharmacophores and "pharmacophore" in task.groups_fixed:
            from omtra.eval.pharmacophore import compute_pharmacophore_match
            noisy_samples = self.sample(task_name, g_list=g_list, n_replicates=n_replicates,
                n_timesteps=200, n_lig_atom_margin=0.1, device=device, coms=coms,
                pharm_input_noise_std=self.pharm_pos_std)
            for reference in ('clean', 'input'):
                result = compute_pharmacophore_match(noisy_samples, reference=reference)
                metrics.update({f'noisy_pharm_{reference}/{key}': val for key, val in result.items()})

        if metrics:
            metrics = add_task_prefix(metrics, task_name)
            sample_count_key = f"{task_name}_{dataset_name}_sample_count"
            metrics[sample_count_key] = self.sample_counts[(task_name, dataset_name)]
            self.log_dict(metrics, sync_dist=True, batch_size=1, on_step=True)

        self.train()
        
        return 0.0

    # @profile
    @g_local_scope
    def forward(self, g: dgl.DGLHeteroGraph, task_name: str):
        ctx = getattr(self, '_eval_ctx', None)
        if ctx is not None:
            # explicit evaluation draw (held-out loss): stable seed supplied by the caller, never the checkpoint step
            devices = [g.device.index] if g.device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(int(ctx.seed))
                if devices:
                    torch.cuda.manual_seed(int(ctx.seed))
                return self._forward_conditioned(g, task_name)
        if getattr(self, 'isolated_training_rng', False):
            devices = [g.device.index] if g.device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                seed = self.rng_seed + 1000003*int(self.global_step) + 7919*int(self.global_rank)
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    torch.cuda.manual_seed(seed)
                return self._forward_conditioned(g, task_name)
        return self._forward_conditioned(g, task_name)

    def eval_draw_context(self, ctx):
        """Context manager: `with model.eval_draw_context(EvalDrawContext(...)):` makes forward use the supplied seed / times and record loss statistics."""
        import contextlib

        @contextlib.contextmanager
        def _cm():
            if getattr(self, '_eval_ctx', None) is not None:
                raise RuntimeError('nested eval_draw_context')
            self._eval_ctx = ctx
            try:
                yield ctx
            finally:
                self._eval_ctx = None
        return _cm()

    def _forward_conditioned(self, g, task_name):
        # sample time
        ctx = getattr(self, '_eval_ctx', None)
        if ctx is not None and ctx.t is not None:
            t = ctx.t.to(device=g.device, dtype=torch.float32)
            if t.shape[0] != g.batch_size:
                raise ValueError(f'evaluation times {tuple(t.shape)} do not match batch size {g.batch_size}')
        else:
            t = self.train_t_sampler(batch_size=g.batch_size, device=g.device)
        self.last_flow_times = t.detach().cpu()

        # maybe not necessary right now, perhaps after we add edges appropriately
        node_batch_idxs, edge_batch_idxs = get_batch_idxs(g)
        lig_ue_mask = get_upper_edge_mask(g, "lig_to_lig")
        upper_edge_mask = {}
        upper_edge_mask["lig_to_lig"] = lig_ue_mask

        # sample conditional path
        # TODO: ctmc conditional path sampling manually sets things to mask token rather 
        # than setting to prior value (which is usually mask token)
        task_class: Task = task_name_to_class(task_name)

        # shift ligand prior (x_0) to simulate displaced pocket definition
        # x_1_true is untouched so loss targets are unaffected
        has_ligand = any(
            group in task_class.groups_present
            for group in ["ligand_identity", "ligand_identity_condensed"]
        )
        if has_ligand and "protein_structure" in task_class.groups_present and self.lig_com_offset_std > 0.0 and g.num_nodes("lig") > 0:
            # random direction (uniform on the sphere) + magnitude uniform in [0, lig_com_offset_std].
            # Matches the inference offset (direction + fixed magnitude); here magnitude is sampled.
            direction = torch.randn(g.batch_size, 3, device=g.device)
            direction = direction / direction.norm(dim=1, keepdim=True)                       # (B, 3) unit vectors
            radius = torch.rand(g.batch_size, 1, device=g.device) * self.lig_com_offset_std   # (B, 1) in [0, std)
            if self.lig_com_offset_zero_p > 0.0:
                # a fraction of samples keep the exact (unshifted) center
                keep_center = torch.rand(g.batch_size, 1, device=g.device) < self.lig_com_offset_zero_p
                radius = radius * (~keep_center)
            offset = direction * radius                                                       # (B, 3)
            per_atom_offset = offset[node_batch_idxs["lig"]]                                  # (n_lig_atoms, 3)
            g.nodes["lig"].data['x_0'] = g.nodes["lig"].data['x_0'] + per_atom_offset

        g = self.sample_conditional_path(
            g, task_class, t, node_batch_idxs, edge_batch_idxs, lig_ue_mask
        )

        if self.distort_p > 0.0:
            t_mask = (t > 0.5)[node_batch_idxs["lig"]]
            distort_mask = torch.rand(g.num_nodes("lig"), 1, device=g.device) < self.distort_p
            if "lig_x" in task_class.partial_modalities_fixed:
                distort_mask = distort_mask & t_mask.unsqueeze(-1) & (~g.nodes["lig"].data['atom_mask_1_true'].unsqueeze(-1)) # don't apply geometry distortion to fixed atoms if ligand coordinate modality is partially fixed
            else: 
                distort_mask = distort_mask & t_mask.unsqueeze(-1)
            g.nodes["lig"].data['x_t'] = g.nodes["lig"].data['x_t'] + torch.randn_like(g.nodes["lig"].data['x_t'])*distort_mask*0.5
        
        # Only observations are perturbed. All clean targets remain immutable.
        if "pharmacophore" in task_class.groups_fixed:
            prepare_coordinate_condition(g, 'pharm', self.pharm_pos_std, resample=True)
        # Protein refinement uses its noisy prior; fixed pockets may have noisy observations.
        if "protein_structure" in task_class.groups_fixed:
            if getattr(self, 'isolated_training_rng', False):
                # C's extra draws must not alter common partial corruption or
                # the self-conditioning coin.
                devices = [g.device.index] if g.device.type == 'cuda' else []
                with torch.random.fork_rng(devices=devices):
                    seed = self.rng_seed + 2750159 + 1000003*int(self.global_step) + 7919*int(self.global_rank)
                    torch.random.default_generator.manual_seed(seed)
                    if devices:torch.cuda.manual_seed(seed)
                    prepare_coordinate_condition(g, 'prot_atom', self.prot_pos_std, resample=True)
            else:
                prepare_coordinate_condition(g, 'prot_atom', self.prot_pos_std, resample=True)
        if self.vector_field.soft_partial_conditioning:
            prepare_partial_condition(g, task_class, self.fixed_coord_max_std,
                                      self.fixed_token_max_prob, resample=True)

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
            target = data_src[modality.entity_name].data[f"{dk}_1_true"].clone()
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

            # skip loss if there are no nodes of this type
            if modality.is_node and g.num_nodes(modality.entity_name) == 0:
                losses[modality.name] = torch.tensor(0.0, device=g.device)
                if ctx is not None and ctx.record:     # an absent modality contributes the DEFINED zero: record zero numerator/count per graph so reductions stay exact
                    zeros = torch.zeros(g.batch_size, dtype=torch.float64)
                    ctx.stats[modality.name] = dict(num=zeros, cnt=zeros.clone(), total_num=0.0, total_cnt=0.0, unw_num=zeros.clone(), n_elig=zeros.clone())
                continue

            inp = vf_output[modality.name]
            target = targets[modality.name]
            if is_categorical:
                raw = fn.cross_entropy(inp, target, reduction="none", ignore_index=-100,
                                       weight=self.loss_fn_dict[modality.name].weight)
                valid = target != -100
                raw = raw * self.cat_loss_weight
            else:
                raw = fn.mse_loss(inp, target, reduction="none")
                raw = raw.reshape(raw.shape[0], -1).mean(-1)
                valid = torch.ones_like(raw, dtype=torch.bool)
            raw_unw = raw.detach().clone() if (ctx is not None and ctx.record) else None      # diagnostic evaluation: per-position error BEFORE the inverse-square time weight
            if self.time_scaled_loss and not disable_time_scaled_loss:
                idx = (edge_batch_idxs[modality.entity_name][upper_edge_mask[modality.entity_name]]
                       if modality.graph_entity == "edge" else node_batch_idxs[modality.entity_name])
                raw = raw * t_weights[idx]
            # A completely unmasked categorical modality has no targets at this time.
            losses[modality.name] = raw.sum() / valid.sum().clamp_min(1)
            if ctx is not None and ctx.record:
                gidx = (edge_batch_idxs[modality.entity_name][upper_edge_mask[modality.entity_name]]
                        if modality.graph_entity == "edge" else node_batch_idxs[modality.entity_name])
                num = torch.zeros(g.batch_size, dtype=torch.float64, device=raw.device).index_add_(0, gidx, raw.detach().double())
                cnt = torch.zeros(g.batch_size, dtype=torch.float64, device=raw.device).index_add_(0, gidx, valid.detach().double())
                # fixed-denominator DIAGNOSTIC terms (evaluation only; the training objective is untouched): unweighted error summed over positions per graph, and the number of
                # ELIGIBLE positions per graph, defined before stochastic corruption: categorical = positions that start masked at t=0 (prior), continuous = all positions.
                dsrc = g.edges if modality.graph_entity == "edge" else g.nodes
                if is_categorical:
                    fake = (self.fake_atom_p > 0.0) and ((modality.data_key == 'a') or (modality.data_key == 'cond_a')) and (modality.entity_name == 'lig')
                    prior = dsrc[modality.entity_name].data[f"{modality.data_key}_0"]
                    if modality.graph_entity == "edge":
                        prior = prior[upper_edge_mask[modality.entity_name]]
                    elig = (prior.reshape(-1) == (modality.n_categories + int(fake))).double()
                else:
                    elig = torch.ones_like(raw_unw, dtype=torch.float64)
                unw = torch.zeros(g.batch_size, dtype=torch.float64, device=raw.device).index_add_(0, gidx, (raw_unw * elig.to(raw_unw.dtype)).double())
                nel = torch.zeros(g.batch_size, dtype=torch.float64, device=raw.device).index_add_(0, gidx, elig)
                ctx.stats[modality.name] = dict(num=num.cpu(), cnt=cnt.cpu(), total_num=float(num.sum()), total_cnt=float(cnt.sum()), unw_num=unw.cpu(), n_elig=nel.cpu())

        # compute auxiliary losses
        for aux_loss_name, aux_loss_fn in self.aux_loss_fn_dict.items():
            if ctx is not None and ctx.base_only:
                break                      # held-out driving signal is BASE-only: auxiliary pair / margin losses are never computed here
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
        extra_group = None
        params = self.parameters()
        if self.lr_pretrained_mult != 1.0:
            manifest = getattr(self, 'initialization_manifest', None)
            if manifest is None:
                raise ValueError('lr_pretrained_mult != 1 needs a partial-checkpoint initialization manifest (warm start)')
            loaded = set(manifest['loaded'])
            named = dict(self.named_parameters())
            old_p = [p for n, p in named.items() if n in loaded]
            new_p = [p for n, p in named.items() if n not in loaded]
            if not old_p:
                raise ValueError('No warm-started parameter found for lr_pretrained_mult')
            params = iter(old_p)           # an iterator, like self.parameters(): hydra passes it through unconverted
            extra_group = new_p
        optimizer = hydra.utils.instantiate(
            self.optimizer_cfg, params=params
        )
        if extra_group is not None:      # hydra cannot take a list of tensor groups; build the second group afterwards
            lr = float(self.optimizer_cfg.lr)
            optimizer.param_groups[0]['lr'] = lr * self.lr_pretrained_mult
            if extra_group:
                optimizer.add_param_group({'params': extra_group, 'lr': lr})

        if getattr(self, 'control_enabled', False):
            # single LR owner: base lr x linear warmup x cumulative controller multiplier (ValidationControlMixin.lr_lambda_value)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": self.build_lr_scheduler(optimizer), "interval": "step", "frequency": 1, "name": "lr_validation_control"}}

        # Linear LR warmup
        if self.lr_warmup_steps > 0:
            def lr_lambda(current_step: int):
                if current_step < self.lr_warmup_steps:
                    return float(current_step + 1) / float(self.lr_warmup_steps)
                return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "lr_warmup",
                },
            }

        return optimizer

    def on_train_epoch_start(self):
        if getattr(self, 'control_enabled', False):
            self.control_on_train_epoch_start()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # runs on EVERY rank, after the callbacks (so the EMA already includes this optimizer step)
        if getattr(self, 'control_enabled', False):
            self.control_on_train_batch_end()

    def on_save_checkpoint(self, checkpoint):
        # keep the warm-start key list so a resumed run rebuilds the same parameter groups (lr_pretrained_mult)
        manifest = getattr(self, 'initialization_manifest', None)
        if manifest is not None and 'loaded' in manifest:
            checkpoint['warm_start_loaded_keys'] = list(manifest['loaded'])
        if getattr(self, 'control_enabled', False):
            self.control_on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        if getattr(self, 'control_enabled', False):
            self.control_on_load_checkpoint(checkpoint)
        elif 'validation_control' in checkpoint:
            raise RuntimeError('checkpoint carries validation-control state; configure control_policy to resume it')
        if 'warm_start_loaded_keys' in checkpoint and getattr(self, 'initialization_manifest', None) is None:
            self.initialization_manifest = {'loaded': list(checkpoint['warm_start_loaded_keys'])}

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
            
            #check if the number of nodes of node type is 0
            if modality.is_node and g.num_nodes(modality.entity_name) == 0:
                # print(f"Skipping fixed modality as there are no nodes of type {modality.entity_name}")
                continue

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
        pharm_pos_std: Optional[torch.Tensor] = None, #std deviation for noise to be added to pharm positions
        n_lig_atoms_mean: Union[float, None] = None,
        n_lig_atoms_std: Union[float, None] = None,
        prot_pos_std: Optional[torch.Tensor] = None,
        prot_input_noise_max_std: float = 0.0,
        prot_input_noise_mode: str = "atom",
        size_prior_path: Optional[str] = None,
        allow_oracle_stereo: bool = False,
        pharm_input_noise_std: float = 0.0,
        fixed_coord_max_std: float = 0.0,
        fixed_token_max_prob: float = 0.0,

    ) -> List[SampledSystem]:
        task: Task = task_name_to_class(task_name)
        groups_generated = task.groups_generated
        groups_present = task.groups_present
        groups_fixed = task.groups_fixed
        partial_modality_conditioning = len(task.partial_modalities_fixed) > 0
        if (fixed_coord_max_std > 0 or fixed_token_max_prob > 0) and not self.vector_field.soft_partial_conditioning:
            raise ValueError('Partial input noise requires a model trained with soft_partial_conditioning')
        if pharm_input_noise_std > 0 and pharm_pos_std is not None:
            raise ValueError('Specify simulated pharmacophore noise OR supplied uncertainty, not both')
        if pharm_input_noise_std > 0 and 'pharmacophore' not in task.groups_fixed:
            raise ValueError('Pharmacophore input noise requires a pharmacophore-conditioned task')

        check_noise(prot_input_noise_max_std, 'prot_input_noise_max_std')
        if prot_input_noise_max_std > 0 and 'protein_structure' not in groups_fixed:
            raise ValueError('Protein input noise requires fixed protein structure')
        size_prior_path = size_prior_path or self.size_prior_path
        if g_list is not None and self.vector_field.stereo_embedder is not None and self.vector_field.stereo_enabled:
            from omtra.data.requested_stereo import validate_inference
            for graph in g_list:
                validate_inference(graph, task, allow_oracle=allow_oracle_stereo)

        # TODO: user-supplied n_atoms dict?

        # set device
        if device is None and g_list is not None:
            device = g_list[0].device
        elif device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if unconditional_n_atoms_dist is None:
            unconditional_n_atoms_dist = self.infer_n_atoms_dist(task)

        # TODO: if the task is unconditional, no 
        use_gt_n_lig_atoms = n_lig_atom_margin is not None
        graphs_with_ligs = g_list is not None and all(g.num_nodes("lig") > 0 for g in g_list)
        use_gt_n_lig_atoms = use_gt_n_lig_atoms and graphs_with_ligs

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
                g_i = copy_graph(g_i, 1)[0]
                base_protein_center = observed(g_i.nodes['prot_atom'].data,'x').mean(0) if protein_present else None
                if 'protein_structure' in groups_fixed:
                    prepare_coordinate_condition(g_i, 'prot_atom', prot_input_noise_max_std, noise_mode=prot_input_noise_mode)
                if 'pharmacophore' in groups_fixed:
                    prepare_coordinate_condition(g_i, 'pharm', pharm_input_noise_std)
                if self.vector_field.soft_partial_conditioning:
                    prepare_partial_condition(g_i, task, fixed_coord_max_std, fixed_token_max_prob)
                g_flat.extend(copy_graph(g_i, n_replicates))

                if coms is None and protein_present:
                    com_i = base_protein_center
                elif coms is None and 'pharmacophore' in groups_fixed and g_i.num_nodes('pharm') > 0:
                    com_i = observed(g_i.nodes['pharm'].data, 'x').mean(dim=0)
                elif coms is None and partial_modality_conditioning and g_i.nodes['lig'].data['atom_mask_1_true'].bool().any():
                    mask = g_i.nodes['lig'].data['atom_mask_1_true'].bool()
                    com_i = observed(g_i.nodes['lig'].data, 'x')[mask].mean(dim=0)
                elif coms is None:
                    com_i = torch.zeros(3, dtype=torch.float, device=g_i.device)
                else:
                    com_i = coms[idx]

                coms_flat.extend([com_i] * n_replicates)

        # TODO: sample number of ligand atoms
        add_ligand = any(group in groups_generated for group in ["ligand_identity", "ligand_identity_condensed"])
        size_prior_audit = [{} for _ in g_flat]

        # find number of fake atoms added to each system by the dataset class
        
        if add_ligand and use_gt_n_lig_atoms and self.fake_atom_p > 0.0:
            n_fake_atoms_gt = []
            for g in g_flat:
                if 'cond_a_1_true' in g.nodes['lig'].data:
                    # raise NotImplementedError('expected there to be a ground-truth ligand, and expected condensed atom types to be used')
                    n_fake_atoms_gt.append(
                        (g.nodes['lig'].data['cond_a_1_true'] == self.cond_a_typer.fake_atom_idx).sum().item()
                    )
                elif 'a_1_true' in g.nodes['lig'].data:
                    n_fake_atoms_gt.append(
                        (g.nodes['lig'].data['a_1_true'] == len(lig_atom_type_map)).sum().item()
                    )
                else:
                    raise NotImplementedError('expected there to be a ground-truth ligand')
            n_fake_atoms_gt = torch.tensor(n_fake_atoms_gt)
        else:
            n_fake_atoms_gt = torch.zeros(len(g_flat), dtype=torch.long)

        
        if protein_present and add_ligand:
            n_prot_atoms = torch.tensor([g.num_nodes("prot_atom") for g in g_flat])
            if "pharmacophore" in groups_fixed:
                n_pharms = torch.tensor([g.num_nodes("pharm") for g in g_flat])
            else:
                n_pharms = None
            
            # use normal distribution if mean and std are provided
            if n_lig_atoms_mean is not None and n_lig_atoms_std is not None:
                n_samples = len(g_flat)
                n_lig_atoms = torch.normal(
                    mean=torch.tensor(n_lig_atoms_mean).expand(n_samples),
                    std=torch.tensor(n_lig_atoms_std).expand(n_samples)
                )
                n_lig_atoms = torch.clamp(n_lig_atoms.round().long(), min=4)

                # sampling from truncated normal distribution ensures number of lig atoms >= number of fixed lig atoms
                if partial_modality_conditioning:
                    n_fixed_atoms = torch.tensor([g.nodes['lig'].data['atom_mask_1_true'].bool().sum().item() for g in g_flat])
                    n_lig_atoms = torch.maximum(n_lig_atoms, n_fixed_atoms)

            # use ground truth number of lig atoms
            elif use_gt_n_lig_atoms:
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

                if partial_modality_conditioning:
                    n_fixed_atoms = torch.tensor([g.nodes['lig'].data['atom_mask_1_true'].bool().sum().item() for g in g_flat])
                    n_lig_atoms = torch.maximum(n_lig_atoms, n_fixed_atoms)
            
            else:
                if size_prior_path:
                    from omtra.data.distributions.size_prior import sample_sizes
                    n_lig_atoms, size_prior_audit = sample_sizes(size_prior_path, n_prot_atoms=n_prot_atoms, n_pharms=n_pharms, return_audit=True)
                elif self.require_training_size_prior:
                    raise ValueError('Sampled ligand size requires an explicit training-only size_prior_path')
                else:
                    n_lig_atoms = sample_n_lig_atoms_plinder(n_prot_atoms=n_prot_atoms, n_pharms=n_pharms)

        # if protein is present, sample ligand atoms from p(n_ligand_atoms|n_protein_atoms,n_pharm_atoms)
        # if pharm atoms not present, we marginalize over n_pharm_atoms - this distribution from plinder dataset
        elif not protein_present and add_ligand:
            if "pharmacophore" in groups_fixed:
                n_pharms = torch.tensor([g.num_nodes("pharm") for g in g_flat])
                n_samples = None
            else:
                n_pharms = None
                n_samples = len(g_flat)

            # use normal distribution if mean and std are provided
            if n_lig_atoms_mean is not None and n_lig_atoms_std is not None:
                n_samples_normal = len(g_flat)
                n_lig_atoms = torch.normal(
                    mean=torch.tensor(n_lig_atoms_mean).expand(n_samples_normal),
                    std=torch.tensor(n_lig_atoms_std).expand(n_samples_normal)
                )
                n_lig_atoms = torch.clamp(n_lig_atoms.round().long(), min=4)

                # sampling from truncated normal distribution ensures number of lig atoms >= number of fixed atoms
                if partial_modality_conditioning:
                    n_fixed_atoms = torch.tensor([g.nodes['lig'].data['atom_mask_1_true'].bool().sum().item() for g in g_flat])
                    n_lig_atoms = torch.maximum(n_lig_atoms, n_fixed_atoms)

            # use ground truth number of lig atoms
            elif use_gt_n_lig_atoms:
                base_n_atoms = torch.tensor([g.num_nodes("lig") for g in g_flat])
                base_n_atoms = base_n_atoms - n_fake_atoms_gt
                margins = torch.clamp(base_n_atoms * n_lig_atom_margin, min=0).int()
                random_offsets = torch.stack([torch.randint(-int(m),int(m)+1,()) for m in margins])
                n_lig_atoms = base_n_atoms + random_offsets
                n_lig_atoms = torch.clamp(n_lig_atoms, min=4)

                if partial_modality_conditioning:
                    n_fixed_atoms = torch.tensor([g.nodes['lig'].data['atom_mask_1_true'].bool().sum().item() for g in g_flat])
                    n_lig_atoms = torch.maximum(n_lig_atoms, n_fixed_atoms)

            elif unconditional_n_atoms_dist == "plinder":
                if size_prior_path:
                    from omtra.data.distributions.size_prior import sample_sizes
                    n_lig_atoms, size_prior_audit = sample_sizes(size_prior_path, n_pharms=n_pharms, n_samples=n_samples, return_audit=True)
                elif self.require_training_size_prior:
                    raise ValueError('Sampled ligand size requires size_prior_path')
                else:
                    n_lig_atoms = sample_n_lig_atoms_plinder(n_pharms=n_pharms, n_samples=n_samples)
            elif unconditional_n_atoms_dist == "pharmit":
                n_lig_atoms = sample_n_lig_atoms_pharmit(
                    n_pharms=n_pharms, n_samples=n_samples, dists_file=str(self.dists_file)
                )
            else:
                raise ValueError(f"Unrecognized dist {unconditional_n_atoms_dist}")
            # TODO: if no protein is present, sample p(n_ligand_atoms|n_pharm_atoms), marginalizing if n_pharm_atoms is not present
            # in this case, the distrbution could come from pharmit or plinder dataset..user-chosen option?

        if add_ligand:
            if partial_modality_conditioning:
                minimum = torch.tensor([int(g.nodes['lig'].data['atom_mask_1_true'].sum()) for g in g_flat])
                n_lig_atoms = torch.maximum(n_lig_atoms, minimum)

            requested_real_atoms = n_lig_atoms.clone()
            if (self.fake_atom_p > 0.0) and not use_gt_n_lig_atoms: # don't add fake atoms 
                n_real_atoms = n_lig_atoms
                max_num_fake_atoms = torch.ceil(n_real_atoms*self.fake_atom_p).float()
                frac = torch.rand_like(max_num_fake_atoms)
                num_fake_atoms = torch.floor(frac*(max_num_fake_atoms+1)).long()
                n_lig_atoms = n_real_atoms + num_fake_atoms

            for g_idx, g_i in enumerate(g_flat):
                g_i_copy = g_i.clone()

                # clear ligand nodes (and edges) if they exist
                if g_i.num_nodes("lig") > 0:
                    lig_node_ids = torch.arange(g_i.num_nodes("lig"), device=g_i.device)
                    g_i.remove_nodes(lig_node_ids, ntype="lig")
                    
                # add lig atoms to each graph
                g_i.add_nodes(
                    n_lig_atoms[g_idx].item(),
                    ntype="lig",
                )
                for key,value in dict(requested_ligand_atoms=int(requested_real_atoms[g_idx]),
                    size_prior_protein_fallback=int(size_prior_audit[g_idx].get('protein_fallback',False)),
                    size_prior_marginal_fallback=int(size_prior_audit[g_idx].get('marginal_fallback',False))).items():
                    g_i.nodes['lig'].data[key]=torch.full((g_i.num_nodes('lig'),),value,dtype=torch.long,device=g_i.device)

                # add lig_to_lig edges to each graph
                edge_idxs = build_lig_edge_idxs(n_lig_atoms[g_idx].item()).to(
                    g_i.device
                )
                assert edge_idxs.shape[0] == 2
                g_i.add_edges(u=edge_idxs[0], v=edge_idxs[1], etype="lig_to_lig")

                if partial_modality_conditioning:
                    atom_mask_old = g_i_copy.nodes['lig'].data['atom_mask_1_true'].bool()

                    n_fixed_atoms = atom_mask_old.sum().item()
                    atom_mask_new = torch.zeros(n_lig_atoms[g_idx], dtype=torch.bool, device=g_i.device) 
                    atom_mask_new[:n_fixed_atoms] = True

                    # if n_lig_atoms[g_idx].item() > atom_mask_old.shape[0]:
                    #     atom_pad = torch.zeros(n_lig_atoms[g_idx].item() - atom_mask_old.shape[0], dtype=torch.bool, device=atom_mask_old.device)
                    #     atom_mask_new = torch.cat([atom_mask_old, atom_pad])
                    # else:
                    #     atom_mask_new = atom_mask_old[:n_lig_atoms[g_idx].item()]
                    g_i.nodes['lig'].data['atom_mask_1_true'] = atom_mask_new.long()

                    src, dst = edge_idxs
                    edge_mask_new = atom_mask_new[src] & atom_mask_new[dst]

                    src_old, dst_old = g_i_copy.edges(etype='lig_to_lig')
                    edge_mask_old = atom_mask_old[src_old] & atom_mask_old[dst_old]
                    g_i.edges['lig_to_lig'].data['edge_mask_1_true'] = edge_mask_new.long()

                    for m_name in task.partial_modalities_fixed:
                        m = name_to_modality(m_name)
                        if m.is_node:
                            dks = [dk for dk in g_i_copy.nodes['lig'].data.keys() if dk.startswith(m.data_key + '_')]
                            for dk in dks:
                                g_i.nodes['lig'].data[dk][atom_mask_new] = g_i_copy.nodes['lig'].data[dk][atom_mask_old]
                        else:
                            dks = [dk for dk in g_i_copy.edges['lig_to_lig'].data.keys() if dk.startswith(m.data_key + '_')]
                            for dk in dks:
                                g_i.edges['lig_to_lig'].data[dk][edge_mask_new] = g_i_copy.edges['lig_to_lig'].data[dk][edge_mask_old]

        
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
                n_pharm_nodes = sample_n_pharms_pharmit(n_lig_atoms=n_lig_atoms,dists_file=str(self.dists_file))
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

            if m.name in task.partial_modalities_fixed and not self.vector_field.soft_partial_conditioning:
                if m.is_node:
                    mask = g.nodes[m.entity_name].data['atom_mask_1_true'].bool()
                    gt_labels = g.nodes[m.entity_name].data[f'{m.data_key}_1_true']
                else:
                    # Take upper triangle for edge modalities
                    mask = g.edges[m.entity_name].data['edge_mask_1_true'].bool()
                    gt_labels = g.edges[m.entity_name].data[f'{m.data_key}_1_true']
                data_src.data[f"{dk}_t"][mask] = gt_labels[mask]

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
            data_src.data[f"{dk}_t"] = observed(data_src.data, dk).clone()


        # optionally set the number of timesteps for the integration
        # TODO: there should be a cleaner way to do this
        # the only reason i'm allowing it to be none by default and manually adding it in
        # is that i don't want to define a default number of timesteps in more than one palce
        # it is already defined as default arg to VectorField.integrate
        itg_kwargs = dict(visualize=visualize, extract_latents_for_confidence=extract_latents_for_confidence, time_spacing=time_spacing, stochastic_sampling=stochastic_sampling, noise_scaler=noise_scaler, eps=eps)
        if n_timesteps is not None:
            itg_kwargs["n_timesteps"] = n_timesteps
        
        # Set protein std for sampling
        if 'prot_atom' in g.ntypes and g.num_nodes('prot_atom') > 0:
            if prot_pos_std is None:
                #Default to zeros
                g.nodes['prot_atom'].data['prot_pos_std'] = torch.zeros(
                    g.num_nodes('prot_atom'), 1, device=device
                )
            else:
                #Use the provided standard deviation
                g.nodes['prot_atom'].data['prot_pos_std'] = prot_pos_std

        if "pharmacophore" in task.groups_fixed:
            prepare_coordinate_condition(g, 'pharm', uncertainty=pharm_pos_std)
        if self.vector_field.soft_partial_conditioning:
            prepare_partial_condition(g, task)

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
        prot_input_noise_max_std: float = 0.0,
        prot_input_noise_mode: str = "atom",
        size_prior_path: Optional[str] = None,
        allow_oracle_stereo: bool = False,
        pharm_input_noise_std: float = 0.0,
        fixed_coord_max_std: float = 0.0,
        fixed_token_max_prob: float = 0.0,
    ) -> List[SampledSystem]:
        
        if g_list is not None:
            task = task_name_to_class(task_name)
            prepared = []
            if coms is None and 'protein_structure' in task.groups_present:
                coms = [observed(g.nodes['prot_atom'].data,'x').mean(0) for g in g_list]
            if (fixed_coord_max_std > 0 or fixed_token_max_prob > 0) and not self.vector_field.soft_partial_conditioning:
                raise ValueError('Partial input noise requires a soft-conditioning model')
            if pharm_input_noise_std > 0 and 'pharmacophore' not in task.groups_fixed:
                raise ValueError('Pharmacophore noise requires a pharmacophore-conditioned task')
            check_noise(prot_input_noise_max_std, 'prot_input_noise_max_std')
            if prot_input_noise_max_std > 0 and 'protein_structure' not in task.groups_fixed:
                raise ValueError('Protein input noise requires fixed protein structure')
            for original in g_list:
                graph = copy_graph(original, 1)[0]
                if 'protein_structure' in task.groups_fixed:
                    prepare_coordinate_condition(graph, 'prot_atom', prot_input_noise_max_std, noise_mode=prot_input_noise_mode)
                if 'pharmacophore' in task.groups_fixed:
                    prepare_coordinate_condition(graph, 'pharm', pharm_input_noise_std)
                if self.vector_field.soft_partial_conditioning:
                    prepare_partial_condition(graph, task, fixed_coord_max_std, fixed_token_max_prob)
                prepared.append(graph)
            g_list = prepared
            pharm_input_noise_std = fixed_coord_max_std = fixed_token_max_prob = prot_input_noise_max_std = 0.0

        n_samples = len(g_list) if g_list is not None else 1

        sampled_systems = [[] for _ in range(n_samples)]    # Sampled_systems[i] are the list of replicates for system i

        chunk_size = min(n_samples, max_batch_size) # handle case where max_batch_size < # samples

        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            g_chunk = g_list[start_idx:end_idx] if g_list is not None else None
            coms_chunk = coms[start_idx:end_idx] if coms is not None else None
            n_chunk = len(g_chunk) if g_chunk is not None else 1

            # Determine reps per batch for this chunk
            reps_per_batch = max(1, max_batch_size // n_chunk)
            n_full_batches = n_replicates // reps_per_batch
            last_batch_reps = n_replicates % reps_per_batch

            for i in range(n_full_batches):
                batch_results = self.sample(g_list=g_chunk,
                                            n_replicates=reps_per_batch,
                                            task_name=task_name,
                                            unconditional_n_atoms_dist=unconditional_n_atoms_dist,
                                            device=device,
                                            n_timesteps=n_timesteps,
                                            visualize=visualize,
                                            coms=coms_chunk,
                                            extract_latents_for_confidence=extract_latents_for_confidence,
                                            time_spacing=time_spacing,
                                            stochastic_sampling=stochastic_sampling,
                                            noise_scaler=noise_scaler,
                                            eps=eps,
                                            n_lig_atom_margin=n_lig_atom_margin,
                                            prot_input_noise_max_std=prot_input_noise_max_std,
                                            prot_input_noise_mode=prot_input_noise_mode,
                                            size_prior_path=size_prior_path,
                                            allow_oracle_stereo=allow_oracle_stereo,
                                            pharm_input_noise_std=pharm_input_noise_std,
                                            fixed_coord_max_std=fixed_coord_max_std,
                                            fixed_token_max_prob=fixed_token_max_prob,
                                            )
                # re-order samples
                for i, sys_idx in enumerate(range(start_idx, end_idx)):
                    start = i * reps_per_batch  # starting index in the chunk results
                    end = start + reps_per_batch    # ending index in the chunk results
                    sampled_systems[sys_idx].extend(batch_results[start:end])   # sys_idx is the index relative to all samples
                
            # last batch
            if last_batch_reps > 0:
                batch_results = self.sample(g_list=g_chunk,
                                            n_replicates=last_batch_reps,
                                            task_name=task_name,
                                            unconditional_n_atoms_dist=unconditional_n_atoms_dist,
                                            device=device,
                                            n_timesteps=n_timesteps,
                                            visualize=visualize,
                                            coms=coms_chunk,
                                            extract_latents_for_confidence=extract_latents_for_confidence,
                                            time_spacing=time_spacing,
                                            stochastic_sampling=stochastic_sampling,
                                            noise_scaler=noise_scaler,
                                            eps=eps,
                                            n_lig_atom_margin=n_lig_atom_margin,
                                            prot_input_noise_max_std=prot_input_noise_max_std,
                                            prot_input_noise_mode=prot_input_noise_mode,
                                            size_prior_path=size_prior_path,
                                            allow_oracle_stereo=allow_oracle_stereo,
                                            pharm_input_noise_std=pharm_input_noise_std,
                                            fixed_coord_max_std=fixed_coord_max_std,
                                            fixed_token_max_prob=fixed_token_max_prob,
                                            )

                for i, sys_idx in enumerate(range(start_idx, end_idx)):
                    start = i * last_batch_reps
                    end = start + last_batch_reps
                    sampled_systems[sys_idx].extend(batch_results[start:end])
        
        # Check that each system has expected number of replicates
        for i, sys_reps in enumerate(sampled_systems):
            if len(sys_reps) != n_replicates:
                raise ValueError(f"System {i} has {len(sys_reps)} replicates, expected {n_replicates}")
    
        # flatten list of lists
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
