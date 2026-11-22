"""Validation-driven LR / scientific-stopping controller: a PURE, deterministic decision function (standard library only).

Design:
  * The CPU evaluator publishes small evidence bundles; this module only consumes them. All statistics (paired cluster bootstrap,
    confirmed regression references) are computed OUTSIDE the trainer; a bundle carries, per weight stream and per signal, gain
    intervals (positive = improvement) over a short and a long horizon and a regression interval against a confirmed reference
    (positive = worse), each as [estimate, lower, upper].
  * Streams are 'raw' and 'ema'. They are never merged: every rule below must hold in EVERY required stream.
  * Roles: 'driver_loss' / 'driver_quality' signals of high-share tasks decide LR cuts. 'driver_guard', 'guard_loss' and
    'guard_quality' signals can only block STOPPING; they never trigger or veto a cut. Unknown-label quality is not part of this
    schema at all (it is a final-acceptance check).
  * Missing, stale, incomplete, wrong-stage or uncertain evidence is NEVER a plateau: the action is CONTINUE.
  * A resource ceiling is SAVE_STOP(status='resource_limited'); it is never convergence.

State is plain JSON (returned by decide, stored in every checkpoint). `decide` never mutates its arguments.
"""
import copy
import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

SCHEMA = 1
SCHEMA_V2 = 2                      # evidence bundles of the paired-window rule (one gain window per signal; see RULE_V2)
RULE_V1 = 'two_horizon_v1'         # original rule: short (100k) + long (200k) gains, every driver in EVERY stream must be flat/regressing
RULE_V3 = 'plateau_count_v3'        # Marios 2026-11-10: EMA-only point-estimate ladder: halve when every driver improved < cut_threshold, count halvings that did not help, stop after stop_count in a row
RULE_V2 = 'paired_window_v2'       # paired-window rule: ONE paired window, raw stream drives cuts (progress excluded for every driver, regression never vetoes), EMA stream drives the stop
STREAMS = ('raw', 'ema')
DRIVER_ROLES = ('driver_loss', 'driver_quality')
GUARD_ROLES = ('driver_guard', 'guard_loss', 'guard_quality')
ALL_ROLES = DRIVER_ROLES + GUARD_ROLES

ACTIONS = ('CONTINUE', 'CUT', 'CONVERGED', 'SAVE_STOP')
REASONS = ('none', 'warmup', 'no_bundle', 'bundle_rejected', 'stale_evidence', 'insufficient_history', 'not_opportunity',
           'driver_missing', 'driver_uncertain', 'driver_improving', 'cut', 'floor_driver_not_flat', 'floor_guard_unhealthy',
           'floor_confirmation_recorded', 'converged', 'resource_limited', 'integrity_error', 'user_requested', 'cut_confirmation_recorded')
REASON_CODE = {r: i for i, r in enumerate(REASONS)}
ACTION_CODE = {a: i for i, a in enumerate(ACTIONS)}


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


@dataclass
class SignalSpec:
    id: str                       # e.g. 'loss/rigid_docking_condensed', 'quality/rigid_docking_condensed/primary'
    role: str                     # one of ALL_ROLES
    task: str
    plateau_per_100k: Optional[float] = None   # gain per 100k optimizer updates that still counts as useful improvement (signal units)
    regression: Optional[float] = None         # level worsening vs the confirmed reference that counts as regression (signal units)


@dataclass
class Policy:
    signals: List[SignalSpec]
    review_every: int = 25000
    quality_review_every: int = 50000           # cadence of the (costlier) sampled-quality reviews; must be 2 x review_every so their 2-review windows share the loss windows' centres
    opportunity_spacing: int = 100000
    cooldown: int = 50000
    cut_factor: float = 0.5
    max_cuts: int = 4
    floor_confirmations: int = 2
    min_confirm_spacing: int = 100000
    max_evidence_age: int = 50000
    warmup: int = 1000
    streams: tuple = STREAMS
    calibration_status: str = 'provisional'     # 'frozen' (production) | 'provisional' | 'test_only'
    calibration_manifest_sha256: Optional[str] = None
    # --- paired-window rule (RULE_V2) only; omitted from to_dict()/hash for RULE_V1 so existing policy hashes do not change ---
    rule: str = RULE_V1
    cut_confirmations: int = 2                  # RULE_V2: eligible decisions needed before a cut
    cut_confirm_spacing: int = 50000            # RULE_V2: minimum spacing between those decisions (= the decision interval)
    interval_mode: str = 'bootstrap'            # RULE_V2 bundle intervals: 'bootstrap' = est +/- z*SD(paired cluster bootstrap); 'jitter_inflated' = also adds the pooled checkpoint-level
                                                # jitter the cluster bootstrap cannot see (computed by the external evidence builder)
    interval_z: float = 1.645                   # RULE_V2: interval multiplier (1.645 = pointwise one-sided 95% upper bound)
    # --- plateau-count rule (RULE_V3) only; omitted from to_dict()/hash for RULE_V1 and RULE_V2 so existing policy hashes do not change ---
    cut_threshold: float = 1.5                  # RULE_V3: halve when EVERY driver's EMA gain over the 100k window is below this (percent)
    help_threshold: float = 1.0                 # RULE_V3: a halving 'did not help' when EVERY driver's gain is below this at both confirmations (percent)
    stop_count: int = 3                         # RULE_V3: stop (converged) after this many CONSECUTIVE halvings that did not help (Marios 2026-11-10: 3)

    def validate(self, production: bool):
        if self.rule not in (RULE_V1, RULE_V2, RULE_V3):
            raise ValueError(f'unknown rule {self.rule!r}')
        if self.rule == RULE_V3:
            if not (0.0 < self.help_threshold <= self.cut_threshold) or self.stop_count < 1:
                raise ValueError('invalid RULE_V3 thresholds (need 0 < help_threshold <= cut_threshold, stop_count >= 1)')
        if self.rule in (RULE_V2, RULE_V3):
            if self.cut_confirmations < 1 or self.cut_confirm_spacing < 1:
                raise ValueError('invalid cut confirmation settings')
            if self.interval_mode not in ('bootstrap', 'jitter_inflated') or not (0.5 < self.interval_z < 6.0):
                raise ValueError('invalid interval settings')
            if any(s.id.startswith('quality/') for s in self.signals):
                raise ValueError('RULE_V2/V3 have no sampled-quality drivers (quality is a final-acceptance check, not a controller signal)')
        ids = [s.id for s in self.signals]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate signal ids')
        for s in self.signals:
            if s.role not in ALL_ROLES:
                raise ValueError(f'signal {s.id}: bad role {s.role}')
        if not any(s.role in DRIVER_ROLES for s in self.signals):
            raise ValueError('policy has no driver signal')
        if tuple(self.streams) != STREAMS:
            raise ValueError('both raw and ema streams are required')
        if any(s.id.startswith('quality/') for s in self.signals) and self.quality_review_every != 2 * self.review_every:
            raise ValueError('quality_review_every must be 2 x review_every so that quality windows are centred like the loss windows')
        if not (0.0 < self.cut_factor < 1.0) or self.max_cuts < 0 or self.floor_confirmations < 1:
            raise ValueError('invalid cut/floor settings')
        if production:
            if self.calibration_status != 'frozen' or not self.calibration_manifest_sha256:
                raise ValueError('production control needs a frozen calibration (calibration_status=frozen with manifest hash); '
                                 f'got {self.calibration_status!r}')
            for s in self.signals:
                if s.plateau_per_100k is None or s.regression is None:
                    raise ValueError(f'signal {s.id}: uncalibrated placeholder (None) in production policy')
        return self

    def to_dict(self):
        d = asdict(self)
        d['streams'] = list(self.streams)
        if self.rule == RULE_V1:
            for k in ('rule', 'cut_confirmations', 'cut_confirm_spacing', 'interval_mode', 'interval_z'):
                d.pop(k)
        if self.rule != RULE_V3:
            for k in ('cut_threshold', 'help_threshold', 'stop_count'):
                d.pop(k)
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d['signals'] = [SignalSpec(**s) for s in d['signals']]
        d['streams'] = tuple(d.get('streams', STREAMS))
        return Policy(**d)

    def hash(self):
        return digest(self.to_dict())

    @property
    def test_only(self):
        return self.calibration_status == 'test_only'


def new_state(policy: Policy, run_uuid: str) -> dict:
    st = dict(schema=SCHEMA, run_uuid=run_uuid, policy_hash=policy.hash(), stage=0, multiplier=1.0, stage_start_step=int(policy.warmup),
              action_seq=0, last_opportunity_endpoint=None, floor_endpoints=[], processed=[], status='running', candidate=None,
              stop_status=None, history=[], test_only=policy.test_only)
    if policy.rule in (RULE_V2, RULE_V3):
        st['cut_endpoints'] = []
    if policy.rule == RULE_V3:
        st['cut_flags'] = []                    # per confirmation: did EVERY driver improve < help_threshold?
        st['unhelpful_halvings'] = 0            # consecutive halvings judged 'did not help'
    return st


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _interval(v):
    """[est, lo, hi] or None when missing/non-finite/inconsistent."""
    if not isinstance(v, (list, tuple)) or len(v) != 3 or not all(_finite(x) for x in v):
        return None
    est, lo, hi = v
    if lo > hi + 1e-12:
        return None
    return float(est), float(lo), float(hi)


def classify(spec: SignalSpec, rec, horizons) -> str:
    """-> 'missing' | 'improving' | 'regressing' | 'flat' | 'uncertain' for one signal in one stream.

    rec = {'gain_short':[est,lo,hi], 'gain_long':[..], 'regression':[..]}; horizons = {'short': steps, 'long': steps} (centre separations).
    Margins scale with the horizon: plateau_per_100k * H / 1e5; the regression margin is a level tolerance (not scaled)."""
    if rec is None or spec.plateau_per_100k is None or spec.regression is None:
        return 'missing'
    gs, gl, rg = _interval(rec.get('gain_short')), _interval(rec.get('gain_long')), _interval(rec.get('regression'))
    hs, hl = horizons.get('short'), horizons.get('long')
    if gs is None or gl is None or rg is None or not hs or not hl:
        return 'missing'
    ms, ml, mr = spec.plateau_per_100k * hs / 1e5, spec.plateau_per_100k * hl / 1e5, spec.regression
    if gs[1] > ms or gl[1] > ml:
        return 'improving'
    recent_progress_excluded = gs[2] < ms and gl[2] < ml          # both gains are RESOLVED below their plateau margins
    if rg[1] > mr and recent_progress_excluded:
        return 'regressing'
    if recent_progress_excluded and rg[2] <= mr:
        return 'flat'
    return 'uncertain'                                            # includes: below the historical reference but recovery not excluded (never permits a cut)


def classify_v2(spec: SignalSpec, rec, horizons):
    """RULE_V2 labels for one signal in one stream -> (progress, regression).

    rec = {'gain': [est, lo, hi], 'regression': [est, lo, hi]}; horizons = {'window': steps between the two window centres (100k)}.
    Margin = plateau_per_100k * window / 1e5 (ONE margin: 3% per 100k, no horizon-scaled second margin).
    progress:   'improving' (lower bound > margin: useful progress ESTABLISHED), 'open' (upper bound >= margin but not established: useful progress
                remains POSSIBLE), 'plateau' (upper bound < margin: useful progress EXCLUDED), 'missing'.
    regression: 'regressing' (lower bound > tolerance), 'ok' (upper bound <= tolerance: non-regression established), 'unresolved', 'missing'.
    A cut needs progress == 'plateau' only; regression is never needed to cut. A stop needs progress == 'plateau' AND regression == 'ok'."""
    h = horizons.get('window') if isinstance(horizons, dict) else None
    g = _interval(rec.get('gain')) if rec else None
    prog = 'missing'
    if g is not None and h and spec.plateau_per_100k is not None:
        margin = spec.plateau_per_100k * h / 1e5
        prog = 'improving' if g[1] > margin else ('plateau' if g[2] < margin else 'open')
    rg = _interval(rec.get('regression')) if rec else None
    reg = 'missing'
    if rg is not None and spec.regression is not None:
        reg = 'regressing' if rg[1] > spec.regression else ('ok' if rg[2] <= spec.regression else 'unresolved')
    return prog, reg


def _guard_ok(spec: SignalSpec, rec) -> bool:
    """A guard is healthy when its evidence is complete and non-regression against the confirmed reference is established."""
    if rec is None or spec.regression is None:
        return False
    rg = _interval(rec.get('regression'))
    return rg is not None and rg[2] <= spec.regression


def bundle_hash(bundle: dict) -> str:
    b = {k: v for k, v in bundle.items() if k != 'bundle_hash'}
    return digest(b)


def check_bundle(bundle, policy: Policy, state: dict, live_step: int):
    """-> (ok: bool, reason_code, detail). Only a complete, current-lineage, current-stage, fresh bundle may drive an action."""
    if not isinstance(bundle, dict):
        return False, 'bundle_rejected', 'not a dict'
    for k in ('schema', 'run_uuid', 'policy_hash', 'stage', 'endpoint_step', 'complete', 'signals', 'horizons', 'earliest_step', 'bundle_hash'):
        if k not in bundle:
            return False, 'bundle_rejected', f'missing field {k}'
    try:
        hash_ok = bundle['bundle_hash'] == bundle_hash(bundle)
    except (ValueError, TypeError):                       # non-finite or non-serializable content can never be a valid bundle
        return False, 'bundle_rejected', 'non-finite or non-serializable content'
    if bundle['schema'] != (SCHEMA_V2 if policy.rule in (RULE_V2, RULE_V3) else SCHEMA) or not hash_ok:
        return False, 'bundle_rejected', 'schema or hash mismatch'
    if bundle['run_uuid'] != state['run_uuid'] or bundle['policy_hash'] != state['policy_hash']:
        return False, 'bundle_rejected', 'run or policy lineage mismatch'
    if not bundle['complete']:
        return False, 'bundle_rejected', 'incomplete bundle'
    if bundle['stage'] != state['stage']:
        return False, 'bundle_rejected', f"stage {bundle['stage']} != current {state['stage']} (evidence predates a cut)"
    c = int(bundle['endpoint_step'])
    if c in state['processed']:
        return False, 'bundle_rejected', 'already processed'
    if state['processed'] and c <= max(state['processed']):
        return False, 'bundle_rejected', 'out-of-order endpoint'
    if live_step - c > policy.max_evidence_age:
        return False, 'stale_evidence', f'endpoint {c} is {live_step - c} updates behind live step {live_step}'
    if live_step < c:
        return False, 'bundle_rejected', 'endpoint in the future'
    need = state['stage_start_step'] + (policy.cooldown if state['stage'] > 0 else 0)
    if policy.rule == RULE_V3:                            # the RECENT window (last 3 reviews) must lie after the post-cut cooldown; the earlier window may precede the cut
        if c - 2 * policy.review_every < need:
            return False, 'insufficient_history', f"recent window starts at {c - 2 * policy.review_every} < required {need}"
        return True, 'none', ''
    if policy.rule == RULE_V2:                            # one comparison window; it must lie entirely after the post-cut cooldown
        if int(bundle['earliest_step'].get('window', -1)) < need:
            return False, 'insufficient_history', f"comparison window starts at {bundle['earliest_step'].get('window')} < required {need}"
        return True, 'none', ''
    if int(bundle['earliest_step'].get('long', -1)) < need:
        return False, 'insufficient_history', f"long window starts at {bundle['earliest_step'].get('long')} < required {need}"
    if int(bundle['earliest_step'].get('short', -1)) < need:
        return False, 'insufficient_history', 'short window starts before the post-cut cooldown'
    return True, 'none', ''


def _act(state, policy, typ, reason, detail='', endpoint=None, stop=False, stop_status=None):
    return dict(type=typ, reason=reason, detail=detail, seq=state['action_seq'], expected_stage=state['stage'], multiplier=state['multiplier'],
                endpoint_step=endpoint, stop=bool(stop), stop_status=stop_status, test_only=bool(policy.test_only))


def _record(st, step, action, labels=None):
    rec = dict(step=int(step), type=action['type'], reason=action['reason'], endpoint=action['endpoint_step'], stage=st['stage'])
    if labels is not None:
        rec['labels'] = labels
    st['history'] = (st['history'] + [rec])[-60:]


def decide(state: dict, policy: Policy, bundle: Optional[dict], live_step: int, resource_limit: bool = False,
           integrity_error: Optional[str] = None, user_stop: bool = False):
    """One decision. -> (new_state, action). Deterministic; arguments are not mutated."""
    st = copy.deepcopy(state)
    live_step = int(live_step)
    if st['status'] != 'running':
        return st, _act(st, policy, 'CONTINUE', 'none', 'terminal status already recorded: ' + st['status'])
    if integrity_error:
        st['status'], st['stop_status'] = 'stopped', 'error'
        st['action_seq'] += 1
        a = _act(st, policy, 'SAVE_STOP', 'integrity_error', integrity_error, stop=True, stop_status='error'); _record(st, live_step, a); return st, a
    if user_stop:
        st['status'], st['stop_status'] = 'stopped', 'user_requested'
        st['action_seq'] += 1
        a = _act(st, policy, 'SAVE_STOP', 'user_requested', 'manual stop (not convergence)', stop=True, stop_status='user_requested'); _record(st, live_step, a); return st, a
    if resource_limit:
        st['status'], st['stop_status'] = 'stopped', 'resource_limited'
        st['action_seq'] += 1
        a = _act(st, policy, 'SAVE_STOP', 'resource_limited', 'resource ceiling: NOT convergence', stop=True, stop_status='resource_limited'); _record(st, live_step, a); return st, a
    if live_step < policy.warmup:
        return st, _act(st, policy, 'CONTINUE', 'warmup')
    if bundle is None:
        return st, _act(st, policy, 'CONTINUE', 'no_bundle')

    ok, reason, detail = check_bundle(bundle, policy, st, live_step)
    if not ok:
        return st, _act(st, policy, 'CONTINUE', reason, detail)
    c = int(bundle['endpoint_step'])
    specs = {s.id: s for s in policy.signals}
    if policy.rule == RULE_V3:
        return _decide_v3(st, policy, bundle, specs, c, live_step)
    if policy.rule == RULE_V2:
        return _decide_v2(st, policy, bundle, specs, c, live_step)
    labels = {s: {} for s in policy.streams}
    guard_ok = {s: {} for s in policy.streams}
    for stream in policy.streams:
        recs = bundle['signals'].get(stream) or {}
        for sid, spec in specs.items():
            labels[stream][sid] = classify(spec, recs.get(sid), bundle['horizons'])
            if spec.role in GUARD_ROLES:
                guard_ok[stream][sid] = _guard_ok(spec, recs.get(sid))

    # an eligible, valid bundle: it is an observation. Opportunity spacing decides whether it may act.
    last = st['last_opportunity_endpoint']
    is_opportunity = last is None or c - last >= policy.opportunity_spacing
    st['processed'] = (st['processed'] + [c])[-300:]
    if not is_opportunity:
        a = _act(st, policy, 'CONTINUE', 'not_opportunity', f'endpoint {c} < last opportunity {last} + {policy.opportunity_spacing}', endpoint=c)
        _record(st, live_step, a); return st, a
    st['last_opportunity_endpoint'] = c

    drivers = [sid for sid, s in specs.items() if s.role in DRIVER_ROLES]
    miss = [(stm, sid) for stm in policy.streams for sid in drivers if labels[stm][sid] == 'missing']
    if miss:
        st['floor_endpoints'] = []
        st['last_opportunity_endpoint'] = last          # missing evidence is not an observation: do not consume the opportunity
        a = _act(st, policy, 'CONTINUE', 'driver_missing', ';'.join(f'{m[0]}:{m[1]}' for m in miss[:6]), endpoint=c)
        _record(st, live_step, a, labels); return st, a
    unc = [(stm, sid) for stm in policy.streams for sid in drivers if labels[stm][sid] == 'uncertain']
    imp = [(stm, sid) for stm in policy.streams for sid in drivers if labels[stm][sid] == 'improving']
    if imp:
        st['floor_endpoints'] = []
        a = _act(st, policy, 'CONTINUE', 'driver_improving', ';'.join(f'{m[0]}:{m[1]}' for m in imp[:6]), endpoint=c)
        _record(st, live_step, a, labels); return st, a
    if unc:
        st['floor_endpoints'] = []
        a = _act(st, policy, 'CONTINUE', 'driver_uncertain', ';'.join(f'{m[0]}:{m[1]}' for m in unc[:6]), endpoint=c)
        _record(st, live_step, a, labels); return st, a

    if st['stage'] < policy.max_cuts:                        # every driver in every stream is flat or regressing: cut (guards cannot veto)
        st['stage'] += 1
        st['multiplier'] = float(policy.cut_factor) ** st['stage']
        st['stage_start_step'] = live_step
        st['action_seq'] += 1
        st['last_opportunity_endpoint'] = None
        st['floor_endpoints'] = []
        reg = [f'{m[0]}:{m[1]}' for m in [(stm, sid) for stm in policy.streams for sid in drivers if labels[stm][sid] == 'regressing']]
        a = _act(st, policy, 'CUT', 'cut', ('regressing drivers (monitored): ' + ';'.join(reg[:6])) if reg else 'all drivers flat', endpoint=c)
        _record(st, live_step, a, labels); return st, a

    # at the floor
    not_flat = [(stm, sid) for stm in policy.streams for sid in drivers if labels[stm][sid] != 'flat']
    bad_guard = [(stm, sid) for stm in policy.streams for sid, s in specs.items() if s.role in GUARD_ROLES and not guard_ok[stm][sid]]
    if not_flat:
        st['floor_endpoints'] = []
        a = _act(st, policy, 'CONTINUE', 'floor_driver_not_flat', ';'.join(f'{m[0]}:{m[1]}:{labels[m[0]][m[1]]}' for m in not_flat[:6]), endpoint=c)
        _record(st, live_step, a, labels); return st, a
    if bad_guard:
        st['floor_endpoints'] = []
        a = _act(st, policy, 'CONTINUE', 'floor_guard_unhealthy', ';'.join(f'{m[0]}:{m[1]}' for m in bad_guard[:6]), endpoint=c)
        _record(st, live_step, a, labels); return st, a
    fe = st['floor_endpoints']
    if fe and c - fe[-1] < policy.min_confirm_spacing:
        a = _act(st, policy, 'CONTINUE', 'not_opportunity', 'confirmation spacing', endpoint=c); _record(st, live_step, a, labels); return st, a
    fe = fe + [c]
    st['floor_endpoints'] = fe
    if len(fe) >= policy.floor_confirmations:
        st['status'], st['stop_status'] = 'converged', 'converged'
        st['candidate'] = dict(endpoint_step=c, bundle_hash=bundle['bundle_hash'], confirmations=fe[-policy.floor_confirmations:])
        st['action_seq'] += 1
        a = _act(st, policy, 'CONVERGED', 'converged', f'candidate endpoint {c}; confirmations {fe}', endpoint=c, stop=True, stop_status='converged')
        _record(st, live_step, a, labels); return st, a
    a = _act(st, policy, 'CONTINUE', 'floor_confirmation_recorded', f'{len(fe)}/{policy.floor_confirmations} at endpoint {c}', endpoint=c)
    _record(st, live_step, a, labels); return st, a


def _decide_v2(st, policy, bundle, specs, c, live_step):
    """RULE_V2 decision on a valid bundle (called from decide(); `st` is already a private copy).

    CUT  (stage < max_cuts): RAW stream only. Every driver must have progress == 'plateau' (gain upper bound below the single margin). Regression is
         never needed and never vetoes a cut; guards and the EMA stream cannot veto either. The condition must hold at `cut_confirmations` eligible
         decisions at least `cut_confirm_spacing` apart (any non-eligible decision resets the count).
    STOP (stage == max_cuts, i.e. at the LR floor): EMA stream only. Every driver 'plateau' AND every task (drivers + guards) regression == 'ok'
         (non-regression ESTABLISHED, not merely not-excluded), confirmed `floor_confirmations` times at least `min_confirm_spacing` apart."""
    horizons = bundle['horizons']
    st.setdefault('cut_endpoints', [])
    lab = {stm: {sid: classify_v2(spec, (bundle['signals'].get(stm) or {}).get(sid), horizons) for sid, spec in specs.items()} for stm in policy.streams}
    labels = {stm: {sid: f'{p}/{r}' for sid, (p, r) in d.items()} for stm, d in lab.items()}
    last = st['last_opportunity_endpoint']
    st['processed'] = (st['processed'] + [c])[-300:]
    if not (last is None or c - last >= policy.opportunity_spacing):
        a = _act(st, policy, 'CONTINUE', 'not_opportunity', f'endpoint {c} < last opportunity {last} + {policy.opportunity_spacing}', endpoint=c)
        _record(st, live_step, a); return st, a
    st['last_opportunity_endpoint'] = c
    drivers = [sid for sid, s in specs.items() if s.role in DRIVER_ROLES]
    guards = [sid for sid, s in specs.items() if s.role in GUARD_ROLES]

    def cont(reason, detail, reset=()):
        for k in reset:
            st[k] = []
        a = _act(st, policy, 'CONTINUE', reason, detail, endpoint=c); _record(st, live_step, a, labels); return st, a

    if st['stage'] < policy.max_cuts:
        miss = [sid for sid in drivers if lab['raw'][sid][0] == 'missing']
        if miss:
            st['last_opportunity_endpoint'] = last          # missing evidence is not an observation
            return cont('driver_missing', ';'.join(f'raw:{s}' for s in miss[:6]), ('cut_endpoints', 'floor_endpoints'))
        imp = [sid for sid in drivers if lab['raw'][sid][0] == 'improving']
        if imp:
            return cont('driver_improving', ';'.join(f'raw:{s}' for s in imp[:6]), ('cut_endpoints', 'floor_endpoints'))
        opn = [sid for sid in drivers if lab['raw'][sid][0] == 'open']
        if opn:
            return cont('driver_uncertain', ';'.join(f'raw:{s}' for s in opn[:6]), ('cut_endpoints', 'floor_endpoints'))
        ce = st['cut_endpoints']
        if ce and c - ce[-1] < policy.cut_confirm_spacing:
            return cont('not_opportunity', 'cut confirmation spacing')
        ce = ce + [c]
        st['cut_endpoints'] = ce
        if len(ce) < policy.cut_confirmations:
            return cont('cut_confirmation_recorded', f'{len(ce)}/{policy.cut_confirmations} at endpoint {c}')
        st['stage'] += 1
        st['multiplier'] = float(policy.cut_factor) ** st['stage']
        st['stage_start_step'] = live_step
        st['action_seq'] += 1
        st['last_opportunity_endpoint'] = None
        st['floor_endpoints'] = []
        st['cut_endpoints'] = []
        reg = [sid for sid in drivers if lab['raw'][sid][1] == 'regressing']
        a = _act(st, policy, 'CUT', 'cut', f'progress excluded for all {len(drivers)} raw drivers at endpoints {ce}' + (('; regressing drivers (monitored): ' + ';'.join(reg[:6])) if reg else ''), endpoint=c)
        _record(st, live_step, a, labels); return st, a

    # at the LR floor: the EMA stream decides whether the run is done
    miss = [sid for sid in drivers if lab['ema'][sid][0] == 'missing'] + [sid for sid in drivers + guards if lab['ema'][sid][1] == 'missing']
    if miss:
        st['last_opportunity_endpoint'] = last
        return cont('driver_missing', ';'.join(f'ema:{s}' for s in miss[:6]), ('floor_endpoints',))
    not_flat = [sid for sid in drivers if lab['ema'][sid][0] != 'plateau']
    if not_flat:
        return cont('floor_driver_not_flat', ';'.join(f'ema:{s}:{lab["ema"][s][0]}' for s in not_flat[:6]), ('floor_endpoints',))
    bad = [sid for sid in drivers + guards if lab['ema'][sid][1] != 'ok']
    if bad:
        return cont('floor_guard_unhealthy', ';'.join(f'ema:{s}:{lab["ema"][s][1]}' for s in bad[:6]), ('floor_endpoints',))
    fe = st['floor_endpoints']
    if fe and c - fe[-1] < policy.min_confirm_spacing:
        return cont('not_opportunity', 'confirmation spacing')
    fe = fe + [c]
    st['floor_endpoints'] = fe
    if len(fe) >= policy.floor_confirmations:
        st['status'], st['stop_status'] = 'converged', 'converged'
        st['candidate'] = dict(endpoint_step=c, bundle_hash=bundle['bundle_hash'], confirmations=fe[-policy.floor_confirmations:])
        st['action_seq'] += 1
        a = _act(st, policy, 'CONVERGED', 'converged', f'candidate endpoint {c}; confirmations {fe}', endpoint=c, stop=True, stop_status='converged')
        _record(st, live_step, a, labels); return st, a
    return cont('floor_confirmation_recorded', f'{len(fe)}/{policy.floor_confirmations} at endpoint {c}')


def _decide_v3(st, policy, bundle, specs, c, live_step):
    """RULE_V3 decision on a valid bundle (Marios 2026-11-10). EMA stream and POINT ESTIMATES of the 100k gain only (no error bars, no guards, no raw stream).

    Let g_i be the EMA gain (percent over the 100k window) of driver i.
      all_cut  = every g_i < cut_threshold      (nothing is still improving meaningfully at this step size)
      no_help  = every g_i < help_threshold     (at the confirmation: the last halving did not help)
    `cut_confirmations` eligible decisions at least `cut_confirm_spacing` apart must all have all_cut; any failure resets the count.
    On the confirmed decision:
      - if a halving already happened (stage > 0): that halving 'did not help' iff EVERY confirmation had no_help -> unhelpful_halvings += 1, else it resets to 0;
      - if unhelpful_halvings >= stop_count: STOP (converged) with the candidate endpoint c;
      - otherwise HALVE the learning rate (no cap on the number of halvings; the resource ceiling ends the run as 'resource_limited').
    The first halving (stage == 0) has nothing to judge, so it never changes the counter."""
    st.setdefault('cut_endpoints', []); st.setdefault('cut_flags', []); st.setdefault('unhelpful_halvings', 0)
    last = st['last_opportunity_endpoint']
    st['processed'] = (st['processed'] + [c])[-300:]
    if not (last is None or c - last >= policy.opportunity_spacing):
        a = _act(st, policy, 'CONTINUE', 'not_opportunity', f'endpoint {c} < last opportunity {last} + {policy.opportunity_spacing}', endpoint=c)
        _record(st, live_step, a); return st, a
    st['last_opportunity_endpoint'] = c
    drivers = [sid for sid, s in specs.items() if s.role in DRIVER_ROLES]
    gains = {}
    for sid in drivers:
        iv = _interval(((bundle['signals'].get('ema') or {}).get(sid) or {}).get('gain'))
        gains[sid] = None if iv is None else iv[0]
    labels = {'ema': {sid: ('missing' if g is None else f'{g:+.2f}') for sid, g in gains.items()}}

    def cont(reason, detail, reset=True):
        if reset:
            st['cut_endpoints'] = []; st['cut_flags'] = []
        a = _act(st, policy, 'CONTINUE', reason, detail, endpoint=c); _record(st, live_step, a, labels); return st, a

    miss = [sid for sid, g in gains.items() if g is None]
    if miss:
        st['last_opportunity_endpoint'] = last          # missing evidence is not an observation
        return cont('driver_missing', ';'.join(f'ema:{s}' for s in miss[:6]))
    still = [sid for sid, g in gains.items() if g >= policy.cut_threshold]
    if still:
        return cont('driver_improving', ';'.join(f'ema:{s}:{gains[s]:+.2f}' for s in still[:6]))
    ce, cf = st['cut_endpoints'], st['cut_flags']
    if ce and c - ce[-1] < policy.cut_confirm_spacing:
        return cont('not_opportunity', 'confirmation spacing', reset=False)
    ce = ce + [c]; cf = cf + [all(g < policy.help_threshold for g in gains.values())]
    st['cut_endpoints'], st['cut_flags'] = ce, cf
    if len(ce) < policy.cut_confirmations:
        return cont('cut_confirmation_recorded', f'{len(ce)}/{policy.cut_confirmations} at endpoint {c}', reset=False)
    judged = ce[-policy.cut_confirmations:]; flags = cf[-policy.cut_confirmations:]
    if st['stage'] > 0:
        st['unhelpful_halvings'] = st['unhelpful_halvings'] + 1 if all(flags) else 0
    if st['unhelpful_halvings'] >= policy.stop_count:
        st['status'], st['stop_status'] = 'converged', 'converged'
        st['candidate'] = dict(endpoint_step=c, bundle_hash=bundle['bundle_hash'], confirmations=judged)
        st['action_seq'] += 1
        a = _act(st, policy, 'CONVERGED', 'converged', f'{st["unhelpful_halvings"]} consecutive halvings did not help; candidate endpoint {c}; confirmations {judged}', endpoint=c, stop=True, stop_status='converged')
        _record(st, live_step, a, labels); return st, a
    st['stage'] += 1
    st['multiplier'] = float(policy.cut_factor) ** st['stage']
    st['stage_start_step'] = live_step
    st['action_seq'] += 1
    st['last_opportunity_endpoint'] = None
    st['cut_endpoints'] = []; st['cut_flags'] = []
    a = _act(st, policy, 'CUT', 'cut', f'every driver improved < {policy.cut_threshold}% at endpoints {judged}; unhelpful halvings so far {st["unhelpful_halvings"]}/{policy.stop_count}', endpoint=c)
    _record(st, live_step, a, labels); return st, a


def reopen_after_resource_limit(state: dict, step: int) -> dict:
    """Continue a run that ended ONLY because the resource budget was reached (Marios 2026-11-10: budget 2M with the option to continue).
    Allowed solely for stop_status == 'resource_limited' (never after convergence, an error, an integrity failure or a manual stop). Stage, learning-rate multiplier,
    halving counter, confirmations and history are kept; the reopening is recorded in the history. Pure function: the input is not mutated."""
    if state.get('status') == 'running':
        return copy.deepcopy(state)
    if state.get('stop_status') != 'resource_limited':
        raise ValueError(f"only a resource_limited stop can be reopened (stop_status={state.get('stop_status')!r}; status={state.get('status')!r})")
    st = copy.deepcopy(state)
    st['status'], st['stop_status'] = 'running', None
    st['action_seq'] += 1
    st['last_opportunity_endpoint'] = None
    st['history'] = (st['history'] + [dict(step=int(step), type='REOPEN', reason='reopened_after_resource_limit', endpoint=None, stage=st['stage'])])[-60:]
    return st


def lr_multiplier(state: dict) -> float:
    return float(state['multiplier'])


def dumps(state):
    return canonical(state)


def loads(s):
    return json.loads(s)
