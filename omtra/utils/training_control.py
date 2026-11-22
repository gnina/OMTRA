"""Training-side wiring of the validation controller (omtra.utils.validation_control).

ValidationControlMixin is mixed into the LightningModule. It runs on EVERY rank (never inside a rank-0-only callback):
  * single LR owner: lr = base * warmup_factor(step) * cumulative multiplier (cuts). Build the scheduler with `build_lr_scheduler`.
  * `control_poll()` is called once per completed optimizer step. Every `poll_every` optimizer steps rank 0 reads completed evidence
    bundles (EvidenceStore), runs the pure `decide`, and broadcasts ONE fixed-size buffer (new state JSON + action). Every rank then
    adopts the same state and applies the same action (multiplier, stop) at the same global step. All ranks enter the collective
    even when rank 0 has nothing to decide.
  * the controller state is stored in every checkpoint (`control_state_dict`/`load_control_state`) so LR state never depends on an
    external file. Missing/malformed/stale manual command files mean "no command" and never clear anything.
"""
import json
import os
import time
from pathlib import Path

import torch

from omtra.utils import validation_control as vc

BUF = 1 << 16                    # fixed broadcast size (bytes) for state + action JSON


def warmup_factor(step: int, warmup: int) -> float:
    return 1.0 if warmup <= 0 or step >= warmup else float(step + 1) / float(warmup)


class EvidenceStore:
    """Directory of atomically published bundle_<endpoint>_<hash>.json files. Unreadable/partial files are ignored (never an action)."""

    def __init__(self, root):
        self.root = Path(root)

    def candidates(self, after_endpoint=-1):
        out = []
        if not self.root.is_dir():
            return out
        for p in sorted(self.root.glob('bundle_*.json')):
            try:
                step = int(p.name.split('_')[1])
            except Exception:
                continue
            if step > after_endpoint:
                out.append((step, p))
        return out

    def latest_valid(self, policy, state, live_step):
        """Newest parseable bundle that passes check_bundle for this state; the rest are left for analysis. -> (bundle|None, notes)."""
        notes = []
        after = max(state['processed']) if state['processed'] else -1
        for step, p in reversed(self.candidates(after)):
            try:
                b = json.load(open(p))
            except Exception as e:                                # partial or malformed: no action
                notes.append(f'{p.name}: unreadable ({type(e).__name__})')
                continue
            ok, reason, detail = vc.check_bundle(b, policy, state, live_step)
            if ok:
                return b, notes
            notes.append(f'{p.name}: {reason} {detail}')
            if reason == 'stale_evidence':                        # older ones are staler still
                break
        return None, notes


def read_manual_command(path, run_uuid, last_seq):
    """Versioned optional manual command {"version":2,"run_uuid":..,"seq":int,"action":"stop"}. -> (command|None, note).
    Missing/null/malformed/unreadable/stale/other-run input is NO COMMAND (never a state change). A legacy 'decay' field is rejected."""
    if not path or not os.path.exists(path):
        return None, 'absent'
    try:
        d = json.load(open(path))
        if not isinstance(d, dict) or d.get('version') != 2:
            return None, 'not a version-2 command'
        if d.get('decay'):
            return None, 'legacy decay control rejected for the validation-control recipe'
        if d.get('run_uuid') != run_uuid:
            return None, 'other run'
        seq = int(d.get('seq', -1))
        if seq <= last_seq:
            return None, 'stale sequence'
        if d.get('action') == 'stop':
            return dict(action='stop', seq=seq), 'ok'
        return None, 'unknown action'
    except Exception as e:
        return None, f'unreadable ({type(e).__name__})'


class ValidationControlMixin:
    """Requires: self.trainer, self.global_rank, self.device. Attributes are created lazily by `setup_control`."""

    control_enabled = False

    def setup_control(self, policy: vc.Policy, run_uuid: str, evidence_dir, poll_every: int = 100, command_path=None,
                      resource_flag_path=None, production: bool = True, log_path=None):
        policy.validate(production=production)
        if policy.test_only and os.environ.get('OMTRA_ALLOW_TEST_ONLY_POLICY') != '1':
            raise RuntimeError('test_only validation-control policy refused (set OMTRA_ALLOW_TEST_ONLY_POLICY=1 only for tests)')
        self.control_policy = policy
        self.control_state = vc.new_state(policy, run_uuid)
        self.control_store = EvidenceStore(evidence_dir)
        self.control_poll_every = int(poll_every)
        self.control_command_path = command_path
        self.control_resource_flag = resource_flag_path
        self.control_log_path = log_path
        self.control_last_polled = -1
        self.control_last_manual_seq = -1
        self.control_enabled = True
        self.lr_multiplier = 1.0
        self.control_warmup = int(policy.warmup)
        self.control_last_ckpt = -1
        self.control_terminal = None

    # ---- LR ownership -------------------------------------------------------------------------------------------------------
    def lr_lambda_value(self, step: int) -> float:
        return warmup_factor(step, getattr(self, 'control_warmup', 0)) * float(getattr(self, 'lr_multiplier', 1.0))

    def build_lr_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: self.lr_lambda_value(s))

    # ---- persistence -------------------------------------------------------------------------------------------------------
    def control_state_dict(self):
        if not self.control_enabled:
            return None
        return dict(state=self.control_state, policy=self.control_policy.to_dict(), lr_multiplier=self.lr_multiplier,
                    last_polled=self.control_last_polled, last_manual_seq=self.control_last_manual_seq)

    def load_control_state(self, blob):
        if blob is None:
            return
        if not self.control_enabled:
            raise RuntimeError('checkpoint carries validation-control state but this run has no control configured')
        pol = vc.Policy.from_dict(blob['policy'])
        if pol.hash() != self.control_policy.hash():
            raise RuntimeError('resume policy hash differs from the checkpoint policy (calibration must not change mid-run)')
        self.control_state = blob['state']
        if self.control_state.get('status') != 'running':                       # a finished run must never silently continue WITHOUT a controller
            if self.control_state.get('stop_status') == 'resource_limited' and os.environ.get('OMTRA_REOPEN_RESOURCE_LIMITED') == '1':
                self.control_state = vc.reopen_after_resource_limit(self.control_state, int(blob.get('last_polled', 0)))
            else:
                raise RuntimeError(f"checkpoint carries a TERMINAL controller state (status={self.control_state.get('status')!r}, stop_status={self.control_state.get('stop_status')!r}); "
                                   'to continue a run that stopped only because of the resource budget either resume the last periodic checkpoint (state still running) '
                                   'with the resource flag cleared, or set OMTRA_REOPEN_RESOURCE_LIMITED=1 to reopen a resource_limited stop; converged/error/user stops cannot be resumed')
        self.lr_multiplier = float(blob['lr_multiplier'])
        if abs(self.lr_multiplier - vc.lr_multiplier(self.control_state)) > 1e-12:
            raise RuntimeError('checkpoint LR multiplier disagrees with controller state')
        self.control_last_polled = int(blob['last_polled'])
        self.control_last_manual_seq = int(blob.get('last_manual_seq', -1))

    # ---- polling -----------------------------------------------------------------------------------------------------------
    def _log_control(self, rec):
        if self.control_log_path and self.global_rank == 0:
            with open(self.control_log_path, 'a') as f:
                f.write(json.dumps(dict(rec, t=time.strftime('%Y-%m-%dT%H:%M:%S'))) + '\n')

    def _rank0_decide(self, step):
        st = self.control_state
        cmd, note = read_manual_command(self.control_command_path, st['run_uuid'], self.control_last_manual_seq)
        resource = bool(self.control_resource_flag and os.path.exists(self.control_resource_flag))
        bundle, notes = (None, [])
        if st['status'] == 'running':
            bundle, notes = self.control_store.latest_valid(self.control_policy, st, step)
        new_state, action = vc.decide(st, self.control_policy, bundle, step, resource_limit=resource, user_stop=cmd is not None)
        if cmd is not None:
            self.control_last_manual_seq = cmd['seq']
        return new_state, action, notes

    def control_poll(self):
        """Call once per completed optimizer step on EVERY rank (e.g. from on_train_batch_end)."""
        if not self.control_enabled:
            return
        step = int(self.trainer.global_step)
        if step == self.control_last_polled or step % self.control_poll_every != 0:
            return
        self.control_last_polled = step
        buf = torch.zeros(BUF, dtype=torch.uint8)
        if self.global_rank == 0:
            new_state, action, notes = self._rank0_decide(step)
            payload = json.dumps(dict(state=new_state, action=action, manual_seq=self.control_last_manual_seq, notes=notes[:5]), sort_keys=True).encode()
            if len(payload) >= BUF - 8:
                raise RuntimeError(f'control payload {len(payload)} bytes exceeds the fixed buffer')
            buf[:8] = torch.tensor(list(len(payload).to_bytes(8, 'little')), dtype=torch.uint8)
            buf[8:8 + len(payload)] = torch.tensor(list(payload), dtype=torch.uint8)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            dev = self.device if torch.distributed.get_backend() == 'nccl' else torch.device('cpu')
            b = buf.to(dev)
            torch.distributed.broadcast(b, src=0)
            buf = b.cpu()
        n = int.from_bytes(bytes(buf[:8].tolist()), 'little')
        msg = json.loads(bytes(buf[8:8 + n].tolist()).decode())
        self._apply_control(step, msg)

    def _apply_control(self, step, msg):
        action, state = msg['action'], msg['state']
        self.control_state = state
        self.control_last_manual_seq = int(msg['manual_seq'])
        if abs(float(state['multiplier']) - float(self.lr_multiplier)) > 1e-15:
            self.lr_multiplier = float(state['multiplier'])
        if action['type'] != 'CONTINUE' or action['reason'] not in ('none', 'no_bundle', 'warmup'):
            self._log_control(dict(step=step, action=action, notes=msg.get('notes'), lr_multiplier=self.lr_multiplier))
        if action['stop'] and not self.trainer.should_stop:
            self.control_terminal = dict(stop_status=action['stop_status'], reason=action['reason'], step=step)
            self.trainer.should_stop = True

    # ---- resumable checkpoint + atomic ready manifest ---------------------------------------------------------------------------
    def control_checkpoint(self):
        """Collective: every rank saves (Lightning barrier); rank 0 then publishes ready/step_<N>.json atomically. The evaluator reads a
        checkpoint only through its manifest, never a file that may still be being written."""
        step = int(self.trainer.global_step)
        if step <= 0 or step % self.control_ckpt_every != 0 or step == self.control_last_ckpt:
            return
        self.control_last_ckpt = step
        root = Path(self.og_run_dir)
        ckdir, ready = root / 'checkpoints', root / 'ready'
        ckdir.mkdir(parents=True, exist_ok=True); ready.mkdir(parents=True, exist_ok=True)
        final, tmp = ckdir / f'step_{step:09d}.ckpt', ckdir / f'.step_{step:09d}.ckpt.tmp'
        self.trainer.save_checkpoint(str(tmp))
        if self.global_rank == 0:
            import hashlib
            os.replace(tmp, final)
            h = hashlib.sha256()
            with open(final, 'rb') as f:
                for blk in iter(lambda: f.read(1 << 24), b''):
                    h.update(blk)
            ema = next((c for c in self.trainer.callbacks if hasattr(c, 'n_updates') and hasattr(c, 'decay')), None)
            man = dict(schema=1, run_uuid=self.control_state['run_uuid'], stage=self.control_state['stage'], stage_start_step=self.control_state['stage_start_step'], step=step, checkpoint=str(final),
                       checkpoint_sha256=h.hexdigest(), lr_multiplier=self.lr_multiplier, policy_hash=self.control_state['policy_hash'],
                       ema=None if ema is None else dict(decay=ema.decay, n_updates=ema.n_updates, source_step=ema.source_step),
                       meta=getattr(self, 'control_meta', {}), created=time.strftime('%Y-%m-%dT%H:%M:%S'))
            mtmp = ready / f'.step_{step:09d}.json.tmp'
            json.dump(man, open(mtmp, 'w'), sort_keys=True)
            os.replace(mtmp, ready / f'step_{step:09d}.json')
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def control_finalize(self):
        """Rank 0, after fit: record WHY training ended. Without a controller verdict the end is a resource/trainer ceiling, never convergence."""
        if not self.control_enabled or self.global_rank != 0:
            return
        term = self.control_terminal or dict(stop_status='resource_limited', reason='trainer ceiling (max_steps/max_time/interruption); NOT convergence',
                                             step=int(self.trainer.global_step))
        term = dict(term, converged=term['stop_status'] == 'converged', controller_state=self.control_state, final_step=int(self.trainer.global_step))
        out = Path(self.og_run_dir) / 'termination.json'
        tmp = out.with_suffix('.tmp')
        json.dump(term, open(tmp, 'w'), sort_keys=True, indent=1); os.replace(tmp, out)

    # ---- single entry points used by the LightningModule hooks (so tests exercise the production code path) -------------------
    def control_on_train_batch_end(self):
        if self.control_enabled:
            self.control_poll()          # collective every control_poll_every optimizer steps; applies cuts / stop identically on all ranks
            self.control_checkpoint()    # collective save + atomic ready manifest (after the poll: a checkpoint at a cut step is post-action)

    @staticmethod
    def _local_rng_state():
        """Main-process RNG states in torch.load(weights_only=True)-safe types (tensors / ints / strings only)."""
        import random
        import numpy as np
        name, keys, pos, has_gauss, cached = np.random.get_state()
        return dict(py=random.getstate(), np=(name, torch.from_numpy(keys.astype(np.int64)), int(pos), int(has_gauss), float(cached)), torch=torch.get_rng_state())

    @staticmethod
    def _set_local_rng_state(st):
        import random
        import numpy as np
        name, keys, pos, has_gauss, cached = st['np']
        random.setstate((st['py'][0], tuple(st['py'][1]), st['py'][2])); np.random.set_state((name, keys.numpy().astype(np.uint32), pos, has_gauss, cached))
        torch.set_rng_state(st['torch'])

    def control_on_save_checkpoint(self, checkpoint):
        if self.control_enabled:
            checkpoint['validation_control'] = self.control_state_dict()
            # per-rank main-process RNG states (python / numpy / torch CPU). Collective: dump_checkpoint runs on every rank. Worker-process streams (num_workers > 0)
            # are NOT captured, so resume is exact for controller/optimizer/EMA/LR state and statistically (not bitwise) equivalent for the data stream.
            local = self._local_rng_state()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                gathered = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(gathered, local)
            else:
                gathered = [local]
            checkpoint['validation_control_rng'] = gathered

    def control_on_load_checkpoint(self, checkpoint):
        if self.control_enabled:
            if 'validation_control' not in checkpoint:
                raise RuntimeError('resuming a validation-control run from a checkpoint without controller state')
            self.load_control_state(checkpoint['validation_control'])
            rng = checkpoint.get('validation_control_rng')
            if rng is not None and len(rng) == (torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1):
                self._pending_rng = rng[self.global_rank if torch.distributed.is_available() and torch.distributed.is_initialized() else 0]
        elif 'validation_control' in checkpoint:
            raise RuntimeError('checkpoint carries validation-control state; configure control_policy to resume it')

    def control_on_train_epoch_start(self):
        """Apply the restored RNG state right before the data iterator of the resumed epoch is created."""
        st = getattr(self, '_pending_rng', None)
        if st is not None:
            self._set_local_rng_state(st)
            self._pending_rng = None
