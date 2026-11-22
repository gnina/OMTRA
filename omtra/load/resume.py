"""Fail-closed training resume, before Hydra can write run metadata."""
from pathlib import Path
import os
import sys
import time
import json

from omegaconf import OmegaConf, open_dict
import torch
from pytorch_lightning import Callback


def plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def latest_checkpoint(run_dir):
    # legacy runs write batch_<N>.ckpt; validation-control runs write step_<N>.ckpt (zero padded) after every completed optimizer step block
    paths = [p for pat, pre in (('batch_*.ckpt', 'batch_'), ('step_*.ckpt', 'step_')) for p in (Path(run_dir) / 'checkpoints').glob(pat) if p.stem.removeprefix(pre).isdigit()]
    if not paths:
        raise ValueError(f'No batch/step checkpoints in {run_dir}')
    return max(paths, key=lambda p: int(p.stem.split('_')[-1]))


def validate_checkpoint(cfg, checkpoint, expected_group=None):
    hp = checkpoint.get('hyper_parameters', {})
    if cfg.mode == 'omtra':
        for key, saved in [('task_phases', cfg.task_group.task_phases),
                           ('task_dataset_coupling', cfg.task_group.dataset_task_coupling)]:
            if key not in hp or plain(hp[key]) != plain(saved):
                raise ValueError(f'Resume rejected: checkpoint {key} differs from saved configuration')
    # Validate model recipe fields as well as task weights (especially optimizer and priors).
    for key, value in cfg.get('model', {}).items():
        if key in hp and plain(value) != plain(hp[key]):
            raise ValueError(f'Resume rejected: checkpoint model setting {key} differs')
    for key, cfg_key in [('graph_config', 'graph'), ('prior_config', 'prior')]:
        if key in hp and cfg_key in cfg and plain(hp[key]) != plain(cfg[cfg_key]):
            raise ValueError(f'Resume rejected: checkpoint {key} differs')
    dm_hp = checkpoint.get('datamodule_hyper_parameters', {})
    for key in ('task_phases', 'dataset_task_coupling'):
        if key in dm_hp and plain(dm_hp[key]) != plain(cfg.task_group[key]):
            raise ValueError(f'Resume rejected: checkpoint datamodule {key} differs')
    if expected_group:
        expected = OmegaConf.load(Path(__file__).resolve().parents[2] / 'configs' / 'task_group' / f'{expected_group}.yaml')
        if plain(cfg.task_group.task_phases) != plain(expected.task_phases):
            raise ValueError(f'Resume rejected: saved allocation does not match {expected_group}; do not relabel a contaminated checkpoint')
    for key in ('global_step', 'optimizer_states', 'lr_schedulers', 'loops'):
        if key not in checkpoint:
            raise ValueError(f'Resume requires a full training checkpoint: missing {key}')
    if not checkpoint['optimizer_states']:
        raise ValueError('Resume requires optimizer state')
    if cfg.mode == 'omtra' and not checkpoint.get('MultiTaskDataModule', {}).get('train_sampler_state'):
        raise ValueError('Resume requires the training sampler state')


def load_resume(checkpoint_path, expected_group=None):
    path = Path(checkpoint_path).resolve()
    run_dir = path.parent.parent
    if expected_group is None and run_dir.parent.name == 'final_multitask':
        variant = run_dir.name.split('_')[0]
        if variant in ('current', 'balanced', 'flexible', 'pharmacophore'):
            expected_group = f'final_multitask_{variant}'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    snapshot = run_dir / 'training_config.yaml'
    embedded = checkpoint.get('omtra_training_config')
    if embedded is not None:
        cfg = OmegaConf.create(embedded)
        if snapshot.exists() and plain(OmegaConf.load(snapshot)) != plain(cfg):
            raise ValueError('Immutable training_config.yaml disagrees with checkpoint configuration')
    elif snapshot.exists():
        cfg = OmegaConf.load(snapshot)
    else:
        # Legacy files must be read BEFORE Hydra runs, then checked against the checkpoint.
        cfg = OmegaConf.load(run_dir / '.hydra' / 'config.yaml')
        print('Legacy resume: validating Hydra configuration against checkpoint task allocation.', file=sys.stderr)
    validate_checkpoint(cfg, checkpoint, expected_group)
    step = checkpoint['global_step']
    # Legacy Hydra snapshots precede dataset expansion. Never re-expand new snapshots.
    if embedded is None and not snapshot.exists():
        from omtra.load.conf import merge_task_spec
        cfg = merge_task_spec(cfg)
    dm_cfg = OmegaConf.select(cfg, 'task_group.datamodule', default={})
    for key, value in dm_cfg.items():
        saved = checkpoint.get('datamodule_hyper_parameters', {})
        if key in saved and plain(value) != plain(saved[key]):
            raise ValueError(f'Resume rejected: datamodule setting {key} differs from checkpoint')
    with open_dict(cfg):
        cfg.og_run_dir = str(run_dir)
        cfg.checkpoint = str(path)
    print(f'Validated resume: {path}, global_step={step}', file=sys.stderr)
    return cfg


def preflight_resume(argv):
    from hydra.core.override_parser.overrides_parser import OverridesParser
    overrides = OverridesParser.create().parse_overrides([a for a in argv[1:] if '=' in a])
    values = {o.key_or_group: o.value() for o in overrides}
    checkpoint = values.get('checkpoint')
    if checkpoint is None:
        output = values.get('hydra.run.dir')
        if output and any((Path(output) / name).exists() for name in ('training_config.yaml', '.hydra/config.yaml')) and os.environ.get('LOCAL_RANK', '0') == '0':
            raise ValueError('Existing training run: specify +checkpoint to resume; refusing to overwrite it')
        return None
    if '-m' in argv or '--multirun' in argv:
        raise ValueError('Resume supports one checkpoint per invocation')
    if values.get('hydra.output_subdir', '.hydra') not in ('.hydra', None):
        raise ValueError('Resume requires the standard Hydra metadata subdirectory')
    allowed = {'checkpoint', 'experiment', 'resume_expected_task_group'}
    forbidden = [key for key in values if key not in allowed and not key.startswith('hydra.')]
    if forbidden:
        raise ValueError(f'Resume cannot change training configuration: {forbidden}')
    cfg = load_resume(checkpoint, values.get('resume_expected_task_group'))
    # All workers inherit this argv. A new metadata directory protects the original.
    run_dir = Path(cfg.og_run_dir)
    output = values.get('hydra.run.dir')
    if not output or Path(output).resolve() == run_dir:
        output = str(run_dir / 'resume_attempts' / f'{time.time_ns()}_{os.getpid()}')
    else:
        output = str(Path(output).resolve())
        if (Path(output) / '.hydra').exists() and os.environ.get('LOCAL_RANK', '0') == '0':
            raise ValueError('Resume attempt directory already contains Hydra metadata')
    argv[:] = [arg for arg in argv if not arg.lstrip('+').startswith('hydra.run.dir=')]
    argv.append(f'hydra.run.dir={output}')
    return cfg


def frozen_config(cfg):
    value = OmegaConf.create(plain(cfg))
    for key in ('checkpoint', 'og_run_dir', 'resume_expected_task_group'):
        value.pop(key, None)
    return plain(value)


def save_training_config(cfg, run_dir):
    """Create once; never replace the configuration used to construct the sampler."""
    path = Path(run_dir) / 'training_config.yaml'
    value = frozen_config(cfg)
    if path.exists():
        if plain(OmegaConf.load(path)) != value:
            raise ValueError(f'Refusing to change immutable training configuration: {path}')
    else:
        with path.open('x') as stream:
            OmegaConf.save(OmegaConf.create(value), stream)
    return value


class TrainingConfigCheckpoint(Callback):
    def __init__(self, config):
        self.config = frozen_config(config)
        self.attempt_id=f'{time.time_ns()}_{os.getpid()}'

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['omtra_training_config'] = self.config

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        dm = trainer.datamodule
        if hasattr(dm, 'mark_consumed'):
            dm.mark_consumed()

    def on_fit_start(self, trainer, pl_module):
        dm = trainer.datamodule
        if dm is None or not hasattr(dm, 'td_coupling'):
            return
        from omtra.load.conf import build_td_coupling
        group = self.config['task_group']
        expected = build_td_coupling(group['task_phases'], group['dataset_task_coupling'])
        sampler = getattr(dm, 'train_sampler', getattr(dm, 'sampler', None))
        if sampler is None:
            raise ValueError('Training sampler has not been constructed')
        actual = sampler.td_coupling
        if expected.task_space != actual.task_space or expected.dataset_space != actual.dataset_space:
            raise ValueError('Training sampler task/dataset ordering changed on resume')
        torch.testing.assert_close(actual.p_dataset_task, expected.p_dataset_task, rtol=0, atol=0)
        torch.testing.assert_close(actual.phase_durations, expected.phase_durations, rtol=0, atol=0)
        if trainer.is_global_zero:
            print('Verified training task shares:', dict(zip(actual.task_space, actual.p_dataset_task[0].sum(-1).tolist())))
