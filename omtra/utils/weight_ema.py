"""Passive exponential moving average (EMA) of the model weights, saved to separate files.

Never changes training: the live weights are only read, the EMA copies live beside them, and the saved files are not
Lightning checkpoints (the normal checkpoints are untouched). Any failure is caught, printed once, and switches the
callback off, so it cannot kill a run.

Only rank 0 instantiates callbacks (omtra.load.conf.instantiate_callbacks). The EMA needs no collective, and under DDP
every rank holds identical weights after each optimizer step, so rank 0's EMA is the EMA of the model.

Enable with Hydra overrides:
  +callbacks.weight_ema._target_=omtra.utils.weight_ema.WeightEMACallback
  +callbacks.weight_ema.decays=[0.999,0.9999] +callbacks.weight_ema.save_every=25000

Open-ended runs are ended by scancel (no clean on_train_end), so a save can also be requested while the job runs: create the
empty file <out_dir>/SAVE_NOW; within poll_every steps the current EMA is written as ema_step<N>.pt and the trigger is renamed
to SAVE_NOW.done.

Each save writes <run_dir>/ema/ema_step<global_step>.pt = {'step', 'decays', 'started_at_step', 'ema': {decay: state_dict}}.
The state_dict keys match the LightningModule state_dict (the 'state_dict' entry of a normal checkpoint).
"""
import time
import traceback
from pathlib import Path
from typing import Sequence

import torch
from pytorch_lightning import Callback


class WeightEMACallback(Callback):
    def __init__(self, decays: Sequence[float] = (0.999, 0.9999), save_every: int = 25000, out_dir: str = None, log_every: int = 1000, poll_every: int = 100):
        self.decays = [float(d) for d in decays]
        self.save_every = int(save_every)
        self.out_dir = out_dir
        self.log_every = int(log_every)
        self.poll_every = int(poll_every)
        self.ema = None            # decay -> {name: tensor}
        self.started_at_step = None
        self.enabled = True
        self._last_saved = None

    def _fail(self, where):
        print(f'[weight_ema] DISABLED after an error in {where} (training continues unaffected):', flush=True)
        traceback.print_exc()
        self.enabled = False
        self.ema = None

    def _float_state(self, pl_module):
        return {k: v for k, v in pl_module.state_dict().items() if torch.is_floating_point(v)}

    def on_train_start(self, trainer, pl_module):
        try:
            self.started_at_step = int(trainer.global_step)
            cur = self._float_state(pl_module)
            self.ema = {d: {k: v.detach().clone().float() for k, v in cur.items()} for d in self.decays}
            print(f'[weight_ema] started at step {self.started_at_step}: decays={self.decays}, {len(cur)} tensors, save_every={self.save_every}', flush=True)
        except Exception:
            self._fail('on_train_start')

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.enabled or self.ema is None:
            return
        try:
            cur = self._float_state(pl_module)
            names = list(cur.keys())
            vals = [cur[n].detach().float() for n in names]
            for d, store in self.ema.items():
                dst = [store[n] for n in names]
                torch._foreach_mul_(dst, d)
                torch._foreach_add_(dst, vals, alpha=1.0 - d)
            step = int(trainer.global_step)
            if self.log_every > 0 and step % self.log_every == 0 and step != self._last_saved:
                print(f'[weight_ema] alive at step {step}', flush=True)
            if self.save_every > 0 and step > 0 and step % self.save_every == 0 and step != self._last_saved:
                self._save(trainer, step)
            elif self.poll_every > 0 and step % self.poll_every == 0 and step != self._last_saved and self._trigger() is not None and self._trigger().exists():
                self._save(trainer, step)
                self._trigger().rename(self._trigger().with_name('SAVE_NOW.done'))
        except Exception:
            self._fail('on_train_batch_end')

    def on_train_end(self, trainer, pl_module):
        if not self.enabled or self.ema is None:
            return
        try:
            step = int(trainer.global_step)
            if step != self._last_saved:
                self._save(trainer, step)
        except Exception:
            self._fail('on_train_end')

    def _trigger(self):
        return Path(self.out_dir) / 'SAVE_NOW' if self.out_dir else None

    def _save(self, trainer, step):
        out = Path(self.out_dir) if self.out_dir else Path(trainer.default_root_dir) / 'ema'
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / f'.ema_step{step}.pt.tmp'
        payload = dict(step=step, decays=self.decays, started_at_step=self.started_at_step,
                       ema={d: {k: v.cpu() for k, v in s.items()} for d, s in self.ema.items()})
        t0 = time.time()
        torch.save(payload, tmp)
        tmp.rename(out / f'ema_step{step}.pt')
        self._last_saved = step
        print(f'[weight_ema] saved {out / f"ema_step{step}.pt"} in {time.time() - t0:.1f}s', flush=True)
