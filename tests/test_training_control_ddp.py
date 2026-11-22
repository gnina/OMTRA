"""Two real Gloo ranks + the production ValidationControlMixin and PersistentEMACallback (toy model, TEST-ONLY policy numbers).

Exercises: identical actions on both ranks at the same optimizer step, cut -> cut -> floor confirmations -> converged stop, atomic ready
manifests, persistent EMA across interruption, and uninterrupted-vs-resumed equivalence (resume before and after a cut).
Intervals are compressed (tens of steps): this tests MECHANICS, not calibrated production behaviour."""
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

import torch
import pytorch_lightning as pl

os.environ['OMTRA_ALLOW_TEST_ONLY_POLICY'] = '1'
BASE = os.environ.setdefault('TCD_TEST_DIR', tempfile.mkdtemp())          # spawned children re-import this module: share through the environment
from omtra.utils import validation_control as vc                          # noqa: E402
from omtra.utils.persistent_ema import PersistentEMACallback              # noqa: E402
from omtra.utils.training_control import ValidationControlMixin           # noqa: E402

FLAT = {'gain_short': [0.0, -1.0, 1.0], 'gain_long': [0.0, -2.0, 2.0], 'regression': [0.0, -1.0, 1.0]}
PLAN = [60, 100, 150, 190, 230]


def make_policy():
    sig = [vc.SignalSpec('loss/d0', 'driver_loss', 'd0', 3.0, 2.0), vc.SignalSpec('loss/d1', 'driver_loss', 'd1', 3.0, 2.0),
           vc.SignalSpec('loss/g0', 'guard_loss', 'g0', 3.0, 2.0)]
    return vc.Policy(signals=sig, review_every=10, opportunity_spacing=40, cooldown=20, max_cuts=2, floor_confirmations=2, min_confirm_spacing=40,
                     max_evidence_age=100, warmup=10, calibration_status='test_only')


class Toy(ValidationControlMixin, pl.LightningModule):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)                                   # identical random initialization in every run (the comparison needs it)
        self.l = torch.nn.Linear(3, 1)
        self.register_buffer('cnt', torch.zeros(1, dtype=torch.long))
        self.og_run_dir = os.environ['TCD_RUN']
        self.setup_control(make_policy(), 'toy-run', os.environ['TCD_EVID'], poll_every=10, production=False)
        self.control_ckpt_every = 40
        self.control_meta = {'recipe_hash': 'r', 'panel_hash': 'p'}

    def training_step(self, b, i):
        g = torch.Generator().manual_seed(int(self.trainer.global_step) * 7 + self.global_rank)          # data = f(step, rank): resume-invariant
        x, y = torch.randn(4, 3, generator=g), torch.randn(4, 1, generator=g)
        return ((self.l(x) - y) ** 2).mean()

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': self.build_lr_scheduler(opt), 'interval': 'step'}}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.control_on_train_batch_end()

    def on_save_checkpoint(self, checkpoint):
        self.control_on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        self.control_on_load_checkpoint(checkpoint)

    def on_train_end(self):
        w = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in self.state_dict().values() if v.is_floating_point())).hexdigest()
        rec = dict(step=int(self.trainer.global_step), lr=self.trainer.optimizers[0].param_groups[0]['lr'], mult=self.lr_multiplier,
                   state=vc.digest(self.control_state), weights=w, terminal=self.control_terminal)
        with open(os.path.join(os.environ['TCD_RUN'], f'rank{self.global_rank}.json'), 'w') as f:
            json.dump(rec, f)


class Writer(pl.Callback):
    """Rank-0 stand-in for the CPU evaluator: publishes a FLAT bundle for each planned endpoint (as soon as training reaches it)."""
    def on_train_batch_end(self, tr, pm, o, b, i):
        step = int(tr.global_step)
        if tr.global_rank != 0 or step not in PLAN:
            return
        st = pm.control_state
        need = st['stage_start_step'] + (pm.control_policy.cooldown if st['stage'] > 0 else 0)
        pol = pm.control_policy
        sig = {s: {x.id: copy.deepcopy(FLAT) for x in pol.signals} for s in pol.streams}
        b = dict(schema=1, run_uuid=st['run_uuid'], policy_hash=st['policy_hash'], stage=st['stage'], endpoint_step=step, complete=True, signals=sig,
                 horizons={'short': 100000, 'long': 200000}, earliest_step={'short': need, 'long': need})
        b['bundle_hash'] = vc.bundle_hash(b)
        d = os.environ['TCD_EVID']; os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, f'.tmp_{step}'); json.dump(b, open(tmp, 'w')); os.replace(tmp, os.path.join(d, f'bundle_{step:09d}_{b["bundle_hash"][:8]}.json'))


class Interrupt(pl.Callback):
    """Rank-0-only? no: ALL ranks must stop together; trainer.max_steps handles that, so this is unused (kept for clarity)."""


def fit(run, evid, max_steps, ckpt=None):
    os.environ['TCD_RUN'], os.environ['TCD_EVID'] = run, evid
    os.makedirs(run, exist_ok=True)
    ema = PersistentEMACallback(decay=0.9)
    ds = torch.utils.data.TensorDataset(torch.zeros(100000, 1))
    dl = torch.utils.data.DataLoader(ds, batch_size=1)
    tr = pl.Trainer(max_steps=max_steps, accelerator='cpu', devices=2, strategy='ddp_spawn', logger=False, enable_checkpointing=False,
                    enable_progress_bar=False, callbacks=[ema, Writer()])
    tr.fit(Toy(), dl, ckpt_path=ckpt)


def result(run):
    return [json.load(open(os.path.join(run, f'rank{r}.json'))) for r in (0, 1)]


def ema_of(ckpt):
    c = torch.load(ckpt, map_location='cpu', weights_only=False)
    return c['callbacks']['PersistentEMA'], c


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A, cls.Ae = os.path.join(BASE, 'A'), os.path.join(BASE, 'A_evid')
        fit(cls.A, cls.Ae, 400)                                      # uninterrupted; the controller stops it by convergence

    def test_ranks_agree_and_controller_sequence(self):
        r0, r1 = result(self.A)
        self.assertEqual({k: r0[k] for k in ('step', 'lr', 'mult', 'state', 'weights')}, {k: r1[k] for k in ('step', 'lr', 'mult', 'state', 'weights')})
        self.assertEqual(r0['terminal']['stop_status'], 'converged'); self.assertEqual(r1['terminal']['stop_status'], 'converged')
        self.assertEqual(r0['step'], 190); self.assertAlmostEqual(r0['mult'], 0.25)
        self.assertAlmostEqual(r0['lr'], 0.1 * 0.25, places=9)
        term = json.load(open(os.path.join(self.A, 'termination.json'))) if os.path.exists(os.path.join(self.A, 'termination.json')) else None
        self.assertIsNone(term)                                      # control_finalize is called by routines/train.py, not by Trainer.fit

    def test_ready_manifests_and_checkpoints(self):
        names = sorted(os.listdir(os.path.join(self.A, 'ready')))
        self.assertEqual(names, [f'step_{s:09d}.json' for s in (40, 80, 120, 160)])
        m = json.load(open(os.path.join(self.A, 'ready', names[1])))
        self.assertEqual((m['step'], m['stage'], m['lr_multiplier']), (80, 1, 0.5)); self.assertEqual(m['ema']['source_step'], 80); self.assertEqual(m['ema']['n_updates'], 80)
        sd, c = ema_of(m['checkpoint'])
        self.assertEqual(c['global_step'], 80); self.assertEqual(sd['source_step'], 80)
        self.assertEqual(c['validation_control']['state']['stage'], 1)
        h = hashlib.sha256(open(m['checkpoint'], 'rb').read()).hexdigest(); self.assertEqual(h, m['checkpoint_sha256'])
        # EMA and raw weights in one checkpoint are the same step; closed-form check against a 2-tensor identity: EMA != raw
        raw = c['state_dict']['l.weight']; self.assertFalse(torch.equal(raw, sd['ema']['l.weight']))

    def _resume(self, from_step, name):
        run, evid = os.path.join(BASE, name), os.path.join(BASE, name + '_evid')
        part = fit(run, evid, from_step + 10)                     # interrupted by the trainer ceiling shortly after the checkpoint
        ck = os.path.join(run, 'checkpoints', f'step_{from_step:09d}.ckpt')
        self.assertTrue(os.path.exists(ck))
        interrupted = result(run)[0]
        self.assertEqual(interrupted['step'], from_step + 10); self.assertIsNone(interrupted['terminal'])        # no controller verdict: resource ceiling
        shutil.rmtree(os.path.join(run, 'ready'))
        fit(run, evid, 400, ckpt=ck)
        return result(run), run

    def test_resume_before_cut_equals_uninterrupted(self):
        (r0, r1), run = self._resume(40, 'B40')
        a = result(self.A)[0]
        for k in ('step', 'lr', 'mult', 'state', 'weights'):
            self.assertEqual(r0[k], a[k], k)
        self.assertEqual(r0['weights'], r1['weights'])
        ea, _ = ema_of(os.path.join(self.A, 'checkpoints', 'step_000000160.ckpt')); eb, _ = ema_of(os.path.join(run, 'checkpoints', 'step_000000160.ckpt'))
        self.assertEqual(ea['n_updates'], eb['n_updates'])
        for k in ea['ema']:
            self.assertTrue(torch.equal(ea['ema'][k], eb['ema'][k]), k)

    def test_resume_after_cut_equals_uninterrupted(self):
        (r0, r1), run = self._resume(80, 'B80')
        a = result(self.A)[0]
        for k in ('step', 'lr', 'mult', 'state', 'weights'):
            self.assertEqual(r0[k], a[k], k)
        ea, _ = ema_of(os.path.join(self.A, 'checkpoints', 'step_000000160.ckpt')); eb, _ = ema_of(os.path.join(run, 'checkpoints', 'step_000000160.ckpt'))
        for k in ea['ema']:
            self.assertTrue(torch.equal(ea['ema'][k], eb['ema'][k]), k)


if __name__ == '__main__':
    unittest.main()
