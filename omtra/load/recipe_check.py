"""Startup contract for the validation-controlled FROM-SCRATCH final recipe. Fails loudly; nothing is silently accepted."""
import hashlib
import json
import os

from omegaconf import DictConfig, OmegaConf


def global_edges(cfg) -> int:
    t = cfg.trainer
    return int(t.get('devices', 1)) * int(t.get('num_nodes', 1)) * int(cfg.edges_per_batch) * int(t.get('accumulate_grad_batches', 1) or 1)


def check_final_recipe(cfg: DictConfig, resuming: bool = False) -> dict:
    """Returns a manifest dict. Raises ValueError listing every violated rule."""
    bad = []
    if cfg.get('partial_ckpt'):
        bad.append(f"partial_ckpt must be null for a from-scratch run, got {cfg.get('partial_ckpt')}")
    if cfg.get('checkpoint') is not None and not resuming:
        bad.append('checkpoint set on a fresh launch (use the resume path)')
    if cfg.model.get('lr_pretrained_mult', 1.0) != 1.0:
        bad.append('lr_pretrained_mult must be 1.0 (no pretrained-parameter LR split)')
    if os.environ.get('OMTRA_RUN_CONTROL'):
        bad.append('legacy OMTRA_RUN_CONTROL is set: a second LR owner is not allowed with validation control')
    cbs = cfg.get('callbacks') or {}
    targets = [str(v.get('_target_', '')) for v in cbs.values() if isinstance(v, DictConfig)]
    if not any(t.endswith('PersistentEMACallback') for t in targets):
        bad.append('PersistentEMACallback missing')
    if any(t.endswith('WeightEMACallback') or 'stop_controller' in t or 'plateau' in t for t in targets):
        bad.append('legacy EMA/controller callback present')
    exp = cfg.get('expected_global_edges')
    ge = global_edges(cfg)
    if exp is None or int(exp) != ge:
        bad.append(f'global edge budget {ge} != expected_global_edges {exp} (devices x nodes x edges_per_batch x accumulation)')
    if int(cfg.trainer.get('limit_val_batches', 0) or 0) != 0:
        bad.append('in-training sampling validation must be off (trainer.limit_val_batches=0)')
    if not cfg.model.get('control_policy_path'):
        bad.append('model.control_policy_path (frozen calibration) is required')
    if cfg.model.get('control_production') is not True:
        bad.append('control_production must be true')
    if cfg.model.get('control_checkpoint_every', 0) % cfg.model.get('control_poll_every', 1) != 0:
        bad.append('control_checkpoint_every must be a multiple of control_poll_every')
    if bad:
        raise ValueError('final recipe check FAILED:\n  - ' + '\n  - '.join(bad))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    return dict(global_edges=ge, devices=int(cfg.trainer.devices), edges_per_rank=int(cfg.edges_per_batch), seed=int(cfg.seed),
                resolved_config_sha256=hashlib.sha256(json.dumps(resolved, sort_keys=True, default=str).encode()).hexdigest(),
                task_shares={p['name']: p['probability'] for p in cfg.task_group.task_phases[0]['tasks']})
