"""Resume regressions: Hydra metadata, allocation integrity and Lightning state."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from omtra.load.resume import (preflight_resume, load_resume, save_training_config,
                               TrainingConfigCheckpoint, latest_checkpoint)
from omtra.load.conf import build_td_coupling
from omtra.dataset.samplers import MultiTaskSampler

ROOT = Path(__file__).resolve().parents[1]


def config():
    return OmegaConf.create({'mode': 'omtra', 'task_group': {
        'task_phases': [{'duration_batches': None, 'tasks': [
            {'name': 'rigid_docking_condensed', 'probability': .7},
            {'name': 'ligand_conformer_condensed', 'probability': .3}]}],
        'dataset_task_coupling': {'rigid_docking_condensed': [['plinder', 1.]],
                                  'ligand_conformer_condensed': [['pharmit', 1.]]}},
        'seed': 42, 'max_steps': 4})


class Tiny(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.))

    def training_step(self, batch, batch_idx):
        self.trainer.datamodule.sampler.batch_idx += 1
        return (self.weight - batch[0]).square().mean()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=.1)
        return {'optimizer': opt, 'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.8),
            'interval': 'step'}}


class TinyData(pl.LightningDataModule):
    def setup(self, stage):
        group = config().task_group
        self.td_coupling = build_td_coupling(group.task_phases, group.dataset_task_coupling)
        self.sampler = MultiTaskSampler(SimpleNamespace(datasets=[]), self.td_coupling, 1, max_steps=4)

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(8)), batch_size=1)

    def state_dict(self):
        return {'sampler': self.sampler.state_dict()}

    def load_state_dict(self, state_dict):
        self.sampler.load_state_dict(state_dict['sampler'])


def trainer(steps, callbacks=()):
    return pl.Trainer(accelerator='cpu', devices=1, max_steps=steps, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, callbacks=list(callbacks), num_sanity_val_steps=0)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name) / 'run'
        (self.run / '.hydra').mkdir(parents=True)
        (self.run / 'checkpoints').mkdir()
        self.cfg = config()
        save_training_config(self.cfg, self.run)
        OmegaConf.save(self.cfg, self.run / '.hydra/config.yaml')
        self.path = self.run / 'checkpoints/batch_2.ckpt'
        self.ckpt = {'omtra_training_config': OmegaConf.to_container(self.cfg),
                     'hyper_parameters': {'task_phases': self.cfg.task_group.task_phases,
                         'task_dataset_coupling': self.cfg.task_group.dataset_task_coupling},
                     'global_step': 2, 'optimizer_states': [{}], 'lr_schedulers': [], 'loops': {},
                     'MultiTaskDataModule': {'train_sampler_state': {'batch_idx': 2}}}
        torch.save(self.ckpt, self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_hydra_does_not_overwrite_original_metadata(self):
        before = (self.run / '.hydra/config.yaml').read_bytes()
        script = Path(self.tmp.name) / 'entry.py'
        script.write_text("""import sys
import routines.train as entry
entry._resume_cfg = entry.preflight_resume(sys.argv)
def check_config(cfg):
    assert cfg.task_group.task_phases[0].tasks[0].probability == .7
    assert cfg.checkpoint.endswith('batch_2.ckpt')
entry.train = check_config
entry.main()
""")
        result = subprocess.run([sys.executable, '-B', str(script),
                                 '--config-path', str(ROOT / 'configs'),
                                 '+experiment=final_multitask_m6', f'+checkpoint={self.path}',
                                 f'hydra.run.dir={self.run}'], env={**os.environ, 'PYTHONPATH': str(ROOT)},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, (self.run / '.hydra/config.yaml').read_bytes())
        self.assertEqual(len(list((self.run / 'resume_attempts').glob('*/.hydra/config.yaml'))), 1)

    def test_checkpoint_allocation_mismatch_rejected(self):
        self.ckpt['hyper_parameters']['task_phases'] = [{'duration_batches': None, 'tasks': []}]
        torch.save(self.ckpt, self.path)
        with self.assertRaisesRegex(ValueError, 'task_phases'):
            load_resume(self.path)

    def test_variant_and_group_overrides_rejected(self):
        with self.assertRaisesRegex(ValueError, 'does not match'):
            load_resume(self.path, 'final_multitask_balanced')
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            preflight_resume(['train.py', f'+checkpoint={self.path}', 'task_group=protein'])

    def test_immutable_config_and_numeric_checkpoint_selection(self):
        self.cfg.seed = 99
        with self.assertRaisesRegex(ValueError, 'immutable'):
            save_training_config(self.cfg, self.run)
        (self.run / 'checkpoints/batch_10.ckpt').touch()
        self.assertEqual(latest_checkpoint(self.run).name, 'batch_10.ckpt')

    def test_fresh_worker_can_reenter_but_new_parent_cannot_overwrite_run(self):
        args = ['train.py', f'hydra.run.dir={self.run}']
        with patch.dict(os.environ, {'LOCAL_RANK': '0'}):
            with self.assertRaisesRegex(ValueError, 'Existing training run'):
                preflight_resume(args)
        with patch.dict(os.environ, {'LOCAL_RANK': '1'}):
            self.assertIsNone(preflight_resume(args))

    def test_modified_sampler_is_rejected(self):
        dm = TinyData()
        dm.setup('fit')
        dm.sampler.p_dataset_task[0, 0, 0] += .1
        with self.assertRaises(AssertionError):
            TrainingConfigCheckpoint(config()).on_fit_start(
                SimpleNamespace(datamodule=dm, is_global_zero=True), None)

    def test_weights_only_checkpoint_rejected(self):
        self.ckpt.pop('optimizer_states')
        torch.save(self.ckpt, self.path)
        with self.assertRaisesRegex(ValueError, 'optimizer_states'):
            load_resume(self.path)

    def test_lightning_restores_optimizer_scheduler_step_and_sampler(self):
        full, dm_full = Tiny(), TinyData()
        tr_full = trainer(4, [TrainingConfigCheckpoint(config())])
        tr_full.fit(full, datamodule=dm_full)
        part, dm_part = Tiny(), TinyData()
        tr_part = trainer(2, [TrainingConfigCheckpoint(config())])
        tr_part.fit(part, datamodule=dm_part)
        path = Path(self.tmp.name) / 'tiny.ckpt'
        tr_part.save_checkpoint(path)
        saved = torch.load(path, weights_only=False)
        self.assertEqual(saved['omtra_training_config'], OmegaConf.to_container(config()))
        resumed, dm_resume = Tiny(), TinyData()
        class CheckStart(pl.Callback):
            def on_train_start(self, trainer, model):
                assert trainer.global_step == 2
                assert trainer.datamodule.sampler.batch_idx == 2
                assert trainer.optimizers[0].state[model.weight]['step'] == 2
                assert trainer.lr_scheduler_configs[0].scheduler.last_epoch == 2
        tr_resume = trainer(4, [TrainingConfigCheckpoint(config()), CheckStart()])
        tr_resume.fit(resumed, datamodule=dm_resume, ckpt_path=str(path))
        torch.testing.assert_close(full.weight, resumed.weight, rtol=0, atol=0)
        self.assertEqual(tr_resume.global_step, 4)
        self.assertEqual(dm_resume.sampler.batch_idx, 4)
        torch.testing.assert_close(dm_full.sampler.p_dataset_task, dm_resume.sampler.p_dataset_task)


if __name__ == '__main__':
    unittest.main()
