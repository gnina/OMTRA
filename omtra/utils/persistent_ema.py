"""Persistent EMA of the model weights, stored INSIDE the resumable checkpoint (Lightning callback state).

Differences from the historical omtra.utils.weight_ema.WeightEMACallback (kept unchanged for old runs):
  * EMA tensors, decay, update count and source step are part of the callback state in every checkpoint and are restored on resume;
    on_train_start never re-initializes a restored EMA from the raw weights.
  * Exactly one update per completed optimizer step (guarded against gradient-accumulation micro-batches and duplicate hook calls).
  * Strict coverage: the EMA covers exactly the floating-point entries of the module state_dict (keys and shapes). A mismatch, a
    resume without EMA history, a wrong decay, or a non-finite EMA is an integrity ERROR (raises). The EMA is never silently disabled.
  * Single decay. Raw weights and EMA are always saved together at the same optimizer step, so raw/EMA pairs cannot be spliced.

Only rank 0 instantiates callbacks (omtra.load.conf.instantiate_callbacks); the EMA needs no collective and every DDP rank holds identical
weights after each optimizer step. Checkpoint SAVING stays a collective on every rank (see training_control.ValidationControlMixin).
"""
from typing import Dict

import torch
from pytorch_lightning import Callback


def float_state(module) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in module.state_dict().items() if torch.is_floating_point(v)}


class PersistentEMACallback(Callback):
    def __init__(self, decay: float = 0.9998):
        if not (0.0 < float(decay) < 1.0):
            raise ValueError('decay must be in (0, 1)')
        self.decay = float(decay)
        self.ema = None                 # name -> float32 tensor
        self.n_updates = 0
        self.source_step = None         # optimizer step of the last update
        self.started_at_step = None
        self._restored = False

    @property
    def state_key(self) -> str:
        return 'PersistentEMA'

    # ---- Lightning callback state ---------------------------------------------------------------------------------------------
    def state_dict(self):
        if self.ema is None:
            return {}
        return dict(decay=self.decay, n_updates=self.n_updates, source_step=self.source_step, started_at_step=self.started_at_step,
                    ema={k: v.detach().cpu().clone() for k, v in self.ema.items()})

    def load_state_dict(self, sd):
        if not sd:
            return
        if abs(float(sd['decay']) - self.decay) > 1e-15:
            raise RuntimeError(f"EMA decay in checkpoint {sd['decay']} != configured {self.decay}; refusing to continue with a different decay")
        self.ema = {k: v.float() for k, v in sd['ema'].items()}
        self.n_updates = int(sd['n_updates'])
        self.source_step = sd['source_step']
        self.started_at_step = sd['started_at_step']
        self._restored = True

    # ---- hooks -----------------------------------------------------------------------------------------------------------------
    def _check_coverage(self, cur):
        if set(cur) != set(self.ema):
            miss, extra = sorted(set(cur) - set(self.ema))[:5], sorted(set(self.ema) - set(cur))[:5]
            raise RuntimeError(f'EMA key coverage mismatch: missing {miss} extra {extra}')
        bad = [k for k in cur if tuple(cur[k].shape) != tuple(self.ema[k].shape)]
        if bad:
            raise RuntimeError(f'EMA shape mismatch for {bad[:5]}')

    def _to_device(self, cur):
        """Keep the EMA on the device of the live parameters (a restored EMA is loaded on CPU; the update is a foreach op that needs one device)."""
        dev = next(iter(cur.values())).device
        if self.ema and next(iter(self.ema.values())).device != dev:
            self.ema = {k: v.to(dev) for k, v in self.ema.items()}

    def on_train_start(self, trainer, pl_module):
        cur = float_state(pl_module)
        step = int(trainer.global_step)
        if self.ema is None:
            if step != 0:
                raise RuntimeError(f'resuming at optimizer step {step} but the checkpoint holds no EMA history; refusing to restart the EMA from raw weights')
            self.ema = {k: v.detach().clone().float() for k, v in cur.items()}
            self.started_at_step, self.source_step = 0, 0
            return
        self._check_coverage(cur)
        self._to_device(cur)
        if self.source_step is not None and int(self.source_step) != step:
            raise RuntimeError(f'EMA source step {self.source_step} != checkpoint optimizer step {step}')

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if self.ema is None:
            raise RuntimeError('EMA used before on_train_start')
        if step == self.source_step:                  # accumulation micro-batch or duplicate call: no new optimizer step
            return
        if step != int(self.source_step) + 1:
            raise RuntimeError(f'EMA expected optimizer step {int(self.source_step) + 1} but saw {step}')
        cur = float_state(pl_module)
        self._to_device(cur)
        names = list(self.ema.keys())
        dst = [self.ema[n] for n in names]
        torch._foreach_mul_(dst, self.decay)
        torch._foreach_add_(dst, [cur[n].detach().float() for n in names], alpha=1.0 - self.decay)
        self.n_updates += 1
        self.source_step = step

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None and int(trainer.global_step) != int(self.source_step):
            raise RuntimeError(f'saving a checkpoint at step {trainer.global_step} but the EMA is at step {self.source_step}')
        if self.ema is not None and not all(bool(torch.isfinite(v).all()) for v in self.ema.values()):
            raise RuntimeError('non-finite EMA tensors: integrity failure (not silently disabled)')
