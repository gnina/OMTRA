"""Pure-rule tests for the paired-window controller rule (omtra.utils.validation_control, rule='paired_window_v2'). Test-only policy numbers: NOT calibrated values.

Rule under test: ONE gain window (two 3-checkpoint means, centres 100k apart); single margin 3% per 100k; the RAW stream alone decides cuts
(progress excluded for every driver = gain upper bound below the margin; regression never vetoes; two eligible decisions 50k apart); at the LR floor the EMA
stream alone decides the stop (every driver plateau AND non-regression ESTABLISHED for all drivers + guards; two confirmations >= 100k apart)."""
import copy
import unittest

from omtra.utils import validation_control as vc

H = {'window': 100000}
PLATEAU = {'gain': [0.0, -0.5, 0.8], 'regression': [0.0, -1.0, 1.0]}                 # progress excluded, non-regression established (upper 1.0 <= 2.0)
PLATEAU_REG_UNRESOLVED = {'gain': [0.0, -0.5, 0.8], 'regression': [1.0, -2.0, 4.0]}   # progress excluded, regression neither established nor excluded
PLATEAU_REGRESSING = {'gain': [0.0, -0.5, 0.8], 'regression': [4.0, 3.0, 5.0]}        # progress excluded, regression established (lower 3.0 > 2.0)
OPEN = {'gain': [1.5, -0.5, 3.5], 'regression': [-1.5, -2.0, -1.0]}                   # upper bound 3.5 >= 3.0: useful progress still possible, not established
IMPROVING = {'gain': [5.0, 3.5, 6.5], 'regression': [-5.0, -6.0, -4.0]}               # lower bound 3.5 > 3.0: progress established
EDGE_UPPER_EQ_MARGIN = {'gain': [1.0, -1.0, 3.0], 'regression': [0.0, -1.0, 1.0]}     # upper bound exactly at the margin is NOT excluded


def policy(**kw):
    sig = [vc.SignalSpec(f'loss/d{i}', 'driver_loss', f'd{i}', 3.0, 2.0) for i in range(3)]
    sig += [vc.SignalSpec('loss/g0', 'guard_loss', 'g0', 3.0, 2.0), vc.SignalSpec('loss/g1', 'guard_loss', 'g1', 3.0, 2.0)]
    d = dict(signals=sig, calibration_status='test_only', max_cuts=2, warmup=1000, rule=vc.RULE_V2, review_every=25000, opportunity_spacing=50000,
             cooldown=50000, cut_confirmations=2, cut_confirm_spacing=50000, floor_confirmations=2, min_confirm_spacing=100000)
    d.update(kw)
    return vc.Policy(**d).validate(production=False)


def bundle(pol, st, endpoint, over=None, stream_over=None, earliest=None, **kw):
    over, stream_over = over or {}, stream_over or {}
    sig = {}
    for stream in pol.streams:
        sig[stream] = {s.id: copy.deepcopy(stream_over.get((stream, s.id), over.get(s.id, PLATEAU))) for s in pol.signals}
    need = st['stage_start_step'] + (pol.cooldown if st['stage'] > 0 else 0)
    b = dict(schema=vc.SCHEMA_V2, run_uuid=st['run_uuid'], policy_hash=st['policy_hash'], stage=st['stage'], endpoint_step=endpoint, complete=True,
             signals=sig, horizons=dict(H), earliest_step=earliest or {'window': need + 10000})
    b.update(kw)
    b['bundle_hash'] = vc.bundle_hash(b)
    return b


class T(unittest.TestCase):
    def setUp(self):
        self.pol = policy()
        self.st = vc.new_state(self.pol, 'run-v2')

    def act(self, endpoint, over=None, stream_over=None, live=None, **kw):
        b = bundle(self.pol, self.st, endpoint, over, stream_over, **kw)
        self.st, a = vc.decide(self.st, self.pol, b, live if live is not None else endpoint + 1000)
        return a

    def to_floor(self):
        """Two plateau decisions per cut, 50k apart, until the LR floor (stage == max_cuts) is reached."""
        e = 300000
        while self.st['stage'] < self.pol.max_cuts:
            self.act(e); self.act(e + 50000); e = self.st['stage_start_step'] + 300000
        return e

    # ---- policy / hash / schema ----
    def test_v1_policy_hash_and_dict_unchanged_by_new_fields(self):
        sig = [vc.SignalSpec('loss/d0', 'driver_loss', 'd0', 3.0, 2.0)]
        p1 = vc.Policy(signals=sig, calibration_status='test_only')
        self.assertEqual(p1.rule, vc.RULE_V1)
        self.assertNotIn('rule', p1.to_dict()); self.assertNotIn('cut_confirmations', p1.to_dict())
        self.assertIn('rule', policy().to_dict())
        self.assertNotEqual(p1.hash(), vc.Policy(signals=sig, calibration_status='test_only', rule=vc.RULE_V2).hash())
        self.assertNotIn('cut_endpoints', vc.new_state(p1, 'r'))
        self.assertEqual(vc.new_state(policy(), 'r')['cut_endpoints'], [])

    def test_roundtrip_and_validation(self):
        p = policy()
        self.assertEqual(vc.Policy.from_dict(p.to_dict()).hash(), p.hash())
        with self.assertRaises(ValueError):
            policy(rule='nope')
        with self.assertRaises(ValueError):
            policy(cut_confirmations=0)
        with self.assertRaises(ValueError):
            policy(signals=[vc.SignalSpec('loss/d0', 'driver_loss', 'd0', 3.0, 2.0), vc.SignalSpec('quality/d0/primary', 'driver_quality', 'd0', 2.0, 3.0)])

    def test_v1_bundle_schema_rejected_by_v2_policy(self):
        b = bundle(self.pol, self.st, 300000, schema=1)
        _, a = vc.decide(self.st, self.pol, b, 301000)
        self.assertEqual((a['type'], a['reason']), ('CONTINUE', 'bundle_rejected'))

    # ---- classification ----
    def test_classify_v2_labels(self):
        sp = vc.SignalSpec('loss/d0', 'driver_loss', 'd0', 3.0, 2.0)
        c = lambda rec: vc.classify_v2(sp, rec, H)
        self.assertEqual(c(PLATEAU), ('plateau', 'ok'))
        self.assertEqual(c(PLATEAU_REG_UNRESOLVED), ('plateau', 'unresolved'))
        self.assertEqual(c(PLATEAU_REGRESSING), ('plateau', 'regressing'))
        self.assertEqual(c(OPEN)[0], 'open')
        self.assertEqual(c(IMPROVING)[0], 'improving')
        self.assertEqual(c(EDGE_UPPER_EQ_MARGIN)[0], 'open')                # upper == margin: progress not excluded
        self.assertEqual(c(None), ('missing', 'missing'))
        self.assertEqual(vc.classify_v2(sp, {'gain': [0, -1, float('nan')], 'regression': None}, H), ('missing', 'missing'))
        self.assertEqual(vc.classify_v2(sp, PLATEAU, {})[0], 'missing')

    def test_single_margin_is_not_horizon_scaled(self):
        sp = vc.SignalSpec('loss/d0', 'driver_loss', 'd0', 3.0, 2.0)
        rec = {'gain': [2.0, 1.0, 5.5], 'regression': [0.0, -1.0, 1.0]}     # upper 5.5: plateau at a 200k window would be < 6, open at 100k
        self.assertEqual(vc.classify_v2(sp, rec, {'window': 100000})[0], 'open')
        self.assertEqual(vc.classify_v2(sp, rec, {'window': 200000})[0], 'plateau')   # the margin scales with the declared window length only

    # ---- cuts: raw stream, confirmation, regression never vetoes ----
    def test_cut_needs_two_eligible_decisions_50k_apart(self):
        a1 = self.act(300000)
        self.assertEqual((a1['type'], a1['reason']), ('CONTINUE', 'cut_confirmation_recorded')); self.assertEqual(self.st['stage'], 0)
        a2 = self.act(350000)
        self.assertEqual((a2['type'], a2['reason']), ('CUT', 'cut'))
        self.assertEqual((self.st['stage'], self.st['multiplier'], self.st['cut_endpoints']), (1, 0.5, []))

    def test_endpoint_inside_decision_interval_is_not_a_decision(self):
        self.act(300000)
        a = self.act(325000)                                                 # 25k later: an evaluation, not a decision
        self.assertEqual(a['reason'], 'not_opportunity'); self.assertEqual(self.st['stage'], 0)
        self.assertEqual(self.act(350000)['type'], 'CUT')

    def test_open_driver_resets_confirmation(self):
        self.act(300000)
        a = self.act(350000, over={'loss/d1': OPEN})
        self.assertEqual((a['type'], a['reason']), ('CONTINUE', 'driver_uncertain')); self.assertEqual(self.st['cut_endpoints'], [])
        self.assertEqual(self.act(400000)['reason'], 'cut_confirmation_recorded')        # count restarted
        self.assertEqual(self.act(450000)['type'], 'CUT')

    def test_improving_driver_blocks_cut(self):
        self.act(300000)
        a = self.act(350000, over={'loss/d2': IMPROVING})
        self.assertEqual((a['type'], a['reason']), ('CONTINUE', 'driver_improving')); self.assertEqual(self.st['stage'], 0)

    def test_driver_with_upper_bound_at_margin_blocks_cut(self):
        self.act(300000)
        self.assertEqual(self.act(350000, over={'loss/d0': EDGE_UPPER_EQ_MARGIN})['reason'], 'driver_uncertain')

    def test_unresolved_or_established_regression_never_vetoes_a_cut(self):
        for rec in (PLATEAU_REG_UNRESOLVED, PLATEAU_REGRESSING):
            self.setUp()
            self.act(300000, over={'loss/d0': rec, 'loss/g0': rec})
            a = self.act(350000, over={'loss/d0': rec, 'loss/g0': rec})
            self.assertEqual((a['type'], a['reason']), ('CUT', 'cut'), rec)
        self.assertIn('regressing drivers (monitored): loss/d0', a['detail'])

    def test_ema_stream_cannot_veto_or_trigger_a_cut(self):
        self.act(300000, stream_over={('ema', 'loss/d0'): IMPROVING})
        self.assertEqual(self.act(350000, stream_over={('ema', 'loss/d0'): IMPROVING, ('ema', 'loss/d1'): OPEN})['type'], 'CUT')   # EMA still improving: no effect
        self.setUp()
        self.assertEqual(self.act(300000, stream_over={('raw', 'loss/d0'): OPEN})['reason'], 'driver_uncertain')               # raw open: no cut however flat EMA is

    def test_missing_raw_driver_does_not_consume_the_opportunity(self):
        a = self.act(300000, over={'loss/d1': None})
        self.assertEqual((a['type'], a['reason']), ('CONTINUE', 'driver_missing')); self.assertIsNone(self.st['last_opportunity_endpoint'])
        self.assertEqual(self.act(325000)['reason'], 'cut_confirmation_recorded')                                        # next endpoint can act

    def test_guards_are_not_cut_drivers(self):
        self.act(300000, over={'loss/g0': IMPROVING})
        self.assertEqual(self.act(350000, over={'loss/g0': IMPROVING})['type'], 'CUT')

    def test_comparison_window_must_lie_after_cooldown(self):
        self.act(300000); self.act(350000)                                    # first cut at live step 351000
        need = self.st['stage_start_step'] + self.pol.cooldown
        a = self.act(400000, earliest={'window': need - 1})
        self.assertEqual((a['type'], a['reason']), ('CONTINUE', 'insufficient_history'))
        self.assertEqual(self.act(410000, earliest={'window': need})['reason'], 'cut_confirmation_recorded')

    def test_stale_stage_bundle_rejected_after_cut(self):
        self.act(300000); b_old = bundle(self.pol, self.st, 350000)
        self.act(350000)                                                      # CUT
        _, a = vc.decide(self.st, self.pol, b_old, 352000)
        self.assertEqual(a['reason'], 'bundle_rejected')

    # ---- stop: EMA stream at the floor ----
    def test_no_stop_before_floor_even_if_everything_is_flat(self):
        self.act(300000); self.act(350000)                                    # cut 1 of 2
        self.assertEqual(self.st['stage'], 1); self.assertEqual(self.st['status'], 'running')

    def test_converges_only_after_two_confirmations_100k_apart(self):
        e = self.to_floor()
        self.assertEqual(self.st['stage'], 2)
        a1 = self.act(e)
        self.assertEqual((a1['type'], a1['reason']), ('CONTINUE', 'floor_confirmation_recorded'))
        a_sp = self.act(e + 50000)                                            # 50k later: too close to count
        self.assertEqual((a_sp['type'], a_sp['reason']), ('CONTINUE', 'not_opportunity'))
        a2 = self.act(e + 100000)
        self.assertEqual((a2['type'], a2['reason'], a2['stop'], a2['stop_status']), ('CONVERGED', 'converged', True, 'converged'))
        self.assertEqual(self.st['status'], 'converged')

    def test_unresolved_regression_blocks_stop_for_driver_and_guard(self):
        e = self.to_floor()
        for sid in ('loss/d0', 'loss/g1'):
            self.setUp(); e = self.to_floor()
            a = self.act(e, stream_over={('ema', sid): PLATEAU_REG_UNRESOLVED})
            self.assertEqual((a['type'], a['reason']), ('CONTINUE', 'floor_guard_unhealthy')); self.assertEqual(self.st['floor_endpoints'], [])
            a = self.act(e + 100000, stream_over={('ema', sid): PLATEAU_REGRESSING})
            self.assertEqual(a['reason'], 'floor_guard_unhealthy')

    def test_open_ema_driver_blocks_stop_and_resets_confirmations(self):
        e = self.to_floor()
        self.assertEqual(self.act(e)['reason'], 'floor_confirmation_recorded')
        a = self.act(e + 100000, stream_over={('ema', 'loss/d2'): OPEN})
        self.assertEqual((a['type'], a['reason']), ('CONTINUE', 'floor_driver_not_flat')); self.assertEqual(self.st['floor_endpoints'], [])
        self.assertEqual(self.act(e + 200000)['reason'], 'floor_confirmation_recorded')                      # count restarted

    def test_raw_stream_irrelevant_at_the_floor(self):
        e = self.to_floor()
        self.act(e, stream_over={('raw', 'loss/d0'): IMPROVING})
        self.assertEqual(self.act(e + 100000, stream_over={('raw', 'loss/d0'): OPEN})['type'], 'CONVERGED')

    def test_missing_ema_task_blocks_stop(self):
        e = self.to_floor()
        a = self.act(e, over={'loss/g0': None})
        self.assertEqual((a['type'], a['reason']), ('CONTINUE', 'driver_missing')); self.assertEqual(self.st['status'], 'running')

    def test_resource_limit_is_not_convergence(self):
        self.st, a = vc.decide(self.st, self.pol, None, 2_000_000, resource_limit=True)
        self.assertEqual((a['type'], a['stop_status']), ('SAVE_STOP', 'resource_limited')); self.assertEqual(self.st['status'], 'stopped')

    def test_state_without_cut_endpoints_key_is_tolerated(self):
        st = copy.deepcopy(self.st); st.pop('cut_endpoints')
        b = bundle(self.pol, st, 300000)
        st2, a = vc.decide(st, self.pol, b, 301000)
        self.assertEqual(a['reason'], 'cut_confirmation_recorded'); self.assertEqual(st2['cut_endpoints'], [300000])

    def test_decide_does_not_mutate_arguments(self):
        b = bundle(self.pol, self.st, 300000); st0 = copy.deepcopy(self.st); b0 = copy.deepcopy(b)
        vc.decide(self.st, self.pol, b, 301000)
        self.assertEqual(self.st, st0); self.assertEqual(b, b0)


if __name__ == '__main__':
    unittest.main()
