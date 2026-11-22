"""Pure-rule tests for omtra.utils.validation_control (no torch). Test-only policy numbers: NOT calibrated values."""
import copy
import json
import unittest

from omtra.utils import validation_control as vc

H = {'short': 100000, 'long': 200000}
FLAT = {'gain_short': [0.0, -1.0, 1.0], 'gain_long': [0.0, -2.0, 2.0], 'regression': [0.0, -1.0, 1.0]}
IMPROVING = {'gain_short': [8.0, 6.0, 10.0], 'gain_long': [12.0, 9.0, 15.0], 'regression': [-8.0, -9.0, -7.0]}
UNCERTAIN = {'gain_short': [1.0, -4.0, 6.0], 'gain_long': [1.0, -4.0, 6.0], 'regression': [0.0, -1.0, 1.0]}
REGRESSING = {'gain_short': [0.0, -1.0, 1.0], 'gain_long': [0.0, -1.0, 1.0], 'regression': [5.0, 3.0, 7.0]}


def policy(**kw):
    sig = [vc.SignalSpec(f'loss/d{i}', 'driver_loss', f'd{i}', 3.0, 2.0) for i in range(3)]
    sig += [vc.SignalSpec('quality/d0/primary', 'driver_quality', 'd0', 2.0, 3.0),
            vc.SignalSpec('quality/d0/guard', 'driver_guard', 'd0', 2.0, 3.0),
            vc.SignalSpec('loss/g0', 'guard_loss', 'g0', 3.0, 2.0), vc.SignalSpec('quality/g0/primary', 'guard_quality', 'g0', 2.0, 3.0)]
    d = dict(signals=sig, calibration_status='test_only', max_cuts=2, warmup=1000)
    d.update(kw)
    return vc.Policy(**d).validate(production=False)


def bundle(pol, st, endpoint, over=None, stream_over=None, earliest=None, complete=True, **kw):
    """All signals FLAT by default; over = {signal: rec} applied to both streams; stream_over = {(stream, signal): rec}."""
    over, stream_over = over or {}, stream_over or {}
    sig = {}
    for stream in pol.streams:
        sig[stream] = {}
        for s in pol.signals:
            sig[stream][s.id] = copy.deepcopy(stream_over.get((stream, s.id), over.get(s.id, FLAT)))
    need = st['stage_start_step'] + (pol.cooldown if st['stage'] > 0 else 0)
    b = dict(schema=1, run_uuid=st['run_uuid'], policy_hash=st['policy_hash'], stage=st['stage'], endpoint_step=endpoint, complete=complete,
             signals=sig, horizons=dict(H), earliest_step=earliest or {'short': need + 10000, 'long': need + 1000})
    b.update(kw)
    b['bundle_hash'] = vc.bundle_hash(b)
    return b


class T(unittest.TestCase):
    def setUp(self):
        self.pol = policy()
        self.st = vc.new_state(self.pol, 'run-1')

    def act(self, b, live=None, **kw):
        live = live if live is not None else (b['endpoint_step'] + 1000 if b else 5000)
        self.st, a = vc.decide(self.st, self.pol, b, live, **kw)
        return a

    def test_no_bundle_and_warmup_continue(self):
        self.assertEqual(self.act(None)['type'], 'CONTINUE')
        self.assertEqual(vc.decide(self.st, self.pol, None, 10)[1]['reason'], 'warmup')

    def test_all_flat_cuts_and_state(self):
        a = self.act(bundle(self.pol, self.st, 300000))
        self.assertEqual((a['type'], a['reason']), ('CUT', 'cut'))
        self.assertEqual((self.st['stage'], self.st['multiplier'], self.st['action_seq']), (1, 0.5, 1))
        self.assertEqual(self.st['stage_start_step'], 301000)

    def test_one_improving_driver_in_one_stream_blocks_cut(self):
        b = bundle(self.pol, self.st, 300000, stream_over={('ema', 'loss/d1'): IMPROVING})
        a = self.act(b)
        self.assertEqual((a['type'], a['reason']), ('CONTINUE', 'driver_improving')); self.assertEqual(self.st['stage'], 0)

    def test_improving_quality_delays_cut_despite_flat_loss(self):
        a = self.act(bundle(self.pol, self.st, 300000, over={'quality/d0/primary': IMPROVING}))
        self.assertEqual(a['reason'], 'driver_improving')

    def test_uncertain_is_not_flat(self):
        a = self.act(bundle(self.pol, self.st, 300000, over={'loss/d2': UNCERTAIN}))
        self.assertEqual((a['type'], a['reason']), ('CONTINUE', 'driver_uncertain'))

    def test_missing_driver_continues_and_does_not_consume_opportunity(self):
        b = bundle(self.pol, self.st, 300000); del b['signals']['raw']['loss/d1']; b['bundle_hash'] = vc.bundle_hash(b)
        a = self.act(b)
        self.assertEqual(a['reason'], 'driver_missing'); self.assertIsNone(self.st['last_opportunity_endpoint'])
        self.assertEqual(self.act(bundle(self.pol, self.st, 325000))['type'], 'CUT')       # next valid bundle may act immediately

    def test_nan_interval_is_missing(self):
        rec = {'gain_short': [float('nan'), 0, 1], 'gain_long': [0, 0, 1], 'regression': [0, 0, 1]}
        self.assertEqual(vc.classify(self.pol.signals[1], rec, H), 'missing')
        b = bundle(self.pol, self.st, 300000); b['signals']['raw']['loss/d1'] = rec          # a bundle carrying NaN cannot be a valid bundle at all
        self.assertEqual(self.act(b)['reason'], 'bundle_rejected')

    def test_regressing_driver_alone_does_not_forbid_cut(self):
        a = self.act(bundle(self.pol, self.st, 300000, over={'loss/d0': REGRESSING}))
        self.assertEqual(a['type'], 'CUT'); self.assertIn('regressing', a['detail'])

    def test_regression_below_reference_with_unresolved_recent_gain_is_uncertain_and_never_cuts(self):
        """Confirmed worsening against the historical reference must not turn an unresolved recent gain into permission to cut."""
        rec = {'gain_short': [4.0, -2.0, 10.0], 'gain_long': [4.0, -2.0, 10.0], 'regression': [5.0, 3.0, 7.0]}   # recent gain NOT excluded; regression confirmed
        spec = [s for s in self.pol.signals if s.id == 'loss/d0'][0]
        self.assertEqual(vc.classify(spec, rec, H), 'uncertain')
        a = self.act(bundle(self.pol, self.st, 300000, over={'loss/d0': rec}))
        self.assertEqual(a['type'], 'CONTINUE'); self.assertEqual(a['reason'], 'driver_uncertain')

    def test_guards_never_trigger_or_veto_a_cut(self):
        bad = {'gain_short': None, 'gain_long': None, 'regression': None}
        a = self.act(bundle(self.pol, self.st, 300000, over={'loss/g0': REGRESSING, 'quality/g0/primary': bad, 'quality/d0/guard': REGRESSING}))
        self.assertEqual(a['type'], 'CUT')

    def test_unknown_label_signal_is_ignored(self):
        b = bundle(self.pol, self.st, 300000)
        for s in b['signals'].values():
            s['quality/unknown_mode'] = copy.deepcopy(REGRESSING)
        b['bundle_hash'] = vc.bundle_hash(b)
        self.assertEqual(self.act(b)['type'], 'CUT')

    def _to_floor(self):
        e = 300000
        for _ in range(self.pol.max_cuts):
            a = self.act(bundle(self.pol, self.st, e)); self.assertEqual(a['type'], 'CUT')
            e = self.st['stage_start_step'] + self.pol.cooldown + 300000
        return e

    def test_cuts_bounded_then_floor_requires_two_spaced_confirmations(self):
        e = self._to_floor(); self.assertEqual(self.st['stage'], 2); self.assertEqual(self.st['multiplier'], 0.25)
        a = self.act(bundle(self.pol, self.st, e)); self.assertEqual(a['reason'], 'floor_confirmation_recorded'); self.assertEqual(a['type'], 'CONTINUE')
        a = self.act(bundle(self.pol, self.st, e + 25000)); self.assertEqual(a['reason'], 'not_opportunity')
        a = self.act(bundle(self.pol, self.st, e + 100000)); self.assertEqual((a['type'], a['stop_status']), ('CONVERGED', 'converged'))
        self.assertEqual(self.st['candidate']['endpoint_step'], e + 100000)

    def test_floor_guard_unhealthy_blocks_stop_and_resets(self):
        e = self._to_floor()
        self.act(bundle(self.pol, self.st, e))
        a = self.act(bundle(self.pol, self.st, e + 100000, over={'loss/g0': REGRESSING}))
        self.assertEqual(a['reason'], 'floor_guard_unhealthy'); self.assertEqual(self.st['floor_endpoints'], [])

    def test_floor_missing_guard_blocks_stop(self):
        e = self._to_floor()
        b = bundle(self.pol, self.st, e); del b['signals']['ema']['quality/g0/primary']; b['bundle_hash'] = vc.bundle_hash(b)
        self.assertEqual(self.act(b)['reason'], 'floor_guard_unhealthy')

    def test_floor_regressing_driver_never_converges(self):
        e = self._to_floor()
        a = self.act(bundle(self.pol, self.st, e, over={'loss/d1': REGRESSING}))
        self.assertEqual((a['type'], a['reason']), ('CONTINUE', 'floor_driver_not_flat')); self.assertEqual(self.st['status'], 'running')

    def test_either_stream_failing_prevents_stop(self):
        e = self._to_floor()
        a = self.act(bundle(self.pol, self.st, e, stream_over={('ema', 'loss/d2'): UNCERTAIN}))
        self.assertEqual(a['type'], 'CONTINUE'); self.assertEqual(self.st['status'], 'running')

    def test_stale_evidence_continues_not_consumed(self):
        b = bundle(self.pol, self.st, 300000)
        a = self.act(b, live=300000 + self.pol.max_evidence_age + 1)
        self.assertEqual(a['reason'], 'stale_evidence'); self.assertEqual(self.st['processed'], [])

    def test_bad_bundles_rejected(self):
        good = bundle(self.pol, self.st, 300000)
        t = copy.deepcopy(good); t['signals']['raw']['loss/d0'] = copy.deepcopy(IMPROVING)          # tampered, hash stale
        self.assertEqual(self.act(t)['reason'], 'bundle_rejected')
        self.assertEqual(self.act(bundle(self.pol, self.st, 300000, complete=False))['reason'], 'bundle_rejected')
        self.assertEqual(self.act(bundle(self.pol, self.st, 300000, run_uuid='other'))['reason'], 'bundle_rejected')
        self.assertEqual(self.act(bundle(self.pol, self.st, 300000, policy_hash='x'))['reason'], 'bundle_rejected')
        self.assertEqual(self.act(bundle(self.pol, self.st, 300000, stage=7))['reason'], 'bundle_rejected')
        self.assertEqual(self.st['action_seq'], 0)

    def test_pre_cut_evidence_cannot_act_after_cut_and_cooldown_enforced(self):
        old = bundle(self.pol, self.st, 300000)
        self.assertEqual(self.act(old)['type'], 'CUT')
        self.assertEqual(self.act(bundle(self.pol, vc.new_state(self.pol, 'run-1'), 310000, stage=0), live=311000)['reason'], 'bundle_rejected')   # stage 0 evidence
        early = bundle(self.pol, self.st, 330000, earliest={'short': 310000, 'long': 301000})                              # windows start inside the cooldown
        self.assertEqual(self.act(early, live=331000)['reason'], 'insufficient_history')
        self.assertEqual(self.st['stage'], 1)

    def test_duplicate_and_out_of_order_do_not_replay_actions(self):
        b = bundle(self.pol, self.st, 300000)
        self.act(b); seq = self.st['action_seq']
        self.assertEqual(self.act(b)['type'], 'CONTINUE'); self.assertEqual(self.st['action_seq'], seq)

    def test_resource_limit_is_not_convergence(self):
        a = self.act(None, resource_limit=True)
        self.assertEqual((a['type'], a['stop_status']), ('SAVE_STOP', 'resource_limited')); self.assertEqual(self.st['candidate'], None)
        self.assertEqual(self.act(bundle(self.pol, self.st, 300000))['type'], 'CONTINUE')            # terminal: nothing further acts

    def test_integrity_and_user_stop_distinct(self):
        self.assertEqual(self.act(None, integrity_error='nan weights')['stop_status'], 'error')
        self.st = vc.new_state(self.pol, 'run-1')
        self.assertEqual(self.act(None, user_stop=True)['stop_status'], 'user_requested')

    def test_determinism_no_mutation_and_json_resume_identical(self):
        b = bundle(self.pol, self.st, 300000)
        s0 = copy.deepcopy(self.st)
        s1, a1 = vc.decide(self.st, self.pol, b, 301000)
        self.assertEqual(self.st, s0)
        s2, a2 = vc.decide(vc.loads(vc.dumps(self.st)), vc.Policy.from_dict(json.loads(json.dumps(self.pol.to_dict()))), b, 301000)
        self.assertEqual((s1, a1), (s2, a2))

    def test_production_refuses_placeholders(self):
        with self.assertRaises(ValueError):
            policy().validate(production=True)
        p = policy(calibration_status='frozen', calibration_manifest_sha256='ab')
        p.signals[0].plateau_per_100k = None
        with self.assertRaises(ValueError):
            p.validate(production=True)
        policy(calibration_status='frozen', calibration_manifest_sha256='ab').validate(production=True)

    def test_streams_must_be_raw_and_ema(self):
        with self.assertRaises(ValueError):
            policy(streams=('raw',)).validate(production=False)

    def test_full_run_improving_then_plateau_converges_only_after_floor(self):
        pol = policy(max_cuts=2); self.pol = pol; self.st = vc.new_state(pol, 'run-1')
        acts = []; e = 275000
        for k in range(60):
            improving = e < 500000
            b = bundle(pol, self.st, e, over={'loss/d0': IMPROVING} if improving else None)
            a = self.act(b, live=e + 1000); acts.append((e, a['type']))
            e += 25000
            if self.st['status'] != 'running':
                break
        types = [t for _, t in acts]
        self.assertIn('CONVERGED', types); self.assertEqual(types.count('CUT'), 2)
        self.assertLess(types.index('CUT'), types.index('CONVERGED'))
        self.assertGreater(acts[types.index('CUT')][0], 500000)


if __name__ == '__main__':
    unittest.main()
