"""Pure-rule tests for the plateau-count controller rule (omtra.utils.validation_control, rule='plateau_count_v3'; Marios 2026-11-10). Test-only numbers.

Rule: EMA stream, point estimates of the 100k gain (percent). Halve when EVERY driver improved < cut_threshold (1.5) at 2 eligible decisions 50k apart; the previous halving
'did not help' iff EVERY driver improved < help_threshold (1.0) at both confirmations -> counter +1, otherwise counter = 0; stop (converged) when the counter reaches stop_count (4).
No cap on the number of halvings; first halving never changes the counter; only the RECENT window must lie after the post-cut cooldown."""
import copy
import unittest

from omtra.utils import validation_control as vc

H = {'window': 100000}


def policy(**kw):
    sig = [vc.SignalSpec(f'loss/d{i}', 'driver_loss', f'd{i}', 3.0, 2.0) for i in range(3)]
    sig += [vc.SignalSpec('loss/g0', 'guard_loss', 'g0', 3.0, 2.0)]
    d = dict(signals=sig, calibration_status='test_only', warmup=1000, rule=vc.RULE_V3, review_every=25000, opportunity_spacing=50000, cooldown=50000,
             cut_confirmations=2, cut_confirm_spacing=50000, cut_threshold=1.5, help_threshold=1.0, stop_count=4, max_cuts=0)
    d.update(kw)
    return vc.Policy(**d).validate(production=False)


def bundle(pol, st, endpoint, gains=0.0, raw_gains=None, **kw):
    """gains: float (all drivers/guards) or dict sid -> EMA gain estimate; raw stream gets raw_gains (default: a big improvement that must be ignored)."""
    sig = {'ema': {}, 'raw': {}}
    for s in pol.signals:
        g = gains.get(s.id, 0.0) if isinstance(gains, dict) else gains
        r = (raw_gains if raw_gains is not None else 9.0)
        sig['ema'][s.id] = {'gain': [g, g - 0.3, g + 0.3], 'regression': [0.0, -1.0, 1.0]}
        sig['raw'][s.id] = {'gain': [r, r - 0.3, r + 0.3], 'regression': [0.0, -1.0, 1.0]}
    b = dict(schema=vc.SCHEMA_V2, run_uuid=st['run_uuid'], policy_hash=st['policy_hash'], stage=st['stage'], endpoint_step=endpoint, complete=True, signals=sig,
             horizons=dict(H), earliest_step={'window': endpoint - 150000})
    b.update(kw)
    b['bundle_hash'] = vc.bundle_hash(b)
    return b


class T(unittest.TestCase):
    def setUp(self):
        self.pol = policy()
        self.st = vc.new_state(self.pol, 'run-v3')

    def act(self, endpoint, gains=0.0, live=None, **kw):
        b = bundle(self.pol, self.st, endpoint, gains, **kw)
        self.st, a = vc.decide(self.st, self.pol, b, live if live is not None else endpoint + 1000)
        return a

    def cycle(self, e, gains=0.0):
        """two confirming decisions 50k apart starting at endpoint e; returns the second action"""
        self.act(e, gains); return self.act(e + 50000, gains)

    def next_endpoint(self):
        return self.st['stage_start_step'] + self.pol.cooldown + 2 * self.pol.review_every          # first endpoint whose recent window is post-cooldown

    # ---- policy / hash ----
    def test_other_rules_hashes_and_state_unchanged(self):
        sig = [vc.SignalSpec('loss/d0', 'driver_loss', 'd0', 3.0, 2.0)]
        p1 = vc.Policy(signals=sig, calibration_status='test_only'); p2 = vc.Policy(signals=sig, calibration_status='test_only', rule=vc.RULE_V2)
        for k in ('cut_threshold', 'help_threshold', 'stop_count'):
            self.assertNotIn(k, p1.to_dict()); self.assertNotIn(k, p2.to_dict()); self.assertIn(k, policy().to_dict())
        self.assertNotIn('unhelpful_halvings', vc.new_state(p2, 'r'))
        st = vc.new_state(policy(), 'r'); self.assertEqual((st['unhelpful_halvings'], st['cut_flags'], st['cut_endpoints']), (0, [], []))
        self.assertEqual(vc.Policy.from_dict(policy().to_dict()).hash(), policy().hash())

    def test_validation(self):
        with self.assertRaises(ValueError): policy(help_threshold=2.0)
        with self.assertRaises(ValueError): policy(help_threshold=0.0)
        with self.assertRaises(ValueError): policy(stop_count=0)

    # ---- the cut ----
    def test_first_halving_needs_two_flat_decisions_and_never_touches_the_counter(self):
        a = self.act(250000); self.assertEqual((a['type'], a['reason']), ('CONTINUE', 'cut_confirmation_recorded'))
        a = self.act(300000); self.assertEqual((a['type'], a['reason']), ('CUT', 'cut'))
        self.assertEqual((self.st['stage'], self.st['multiplier'], self.st['unhelpful_halvings']), (1, 0.5, 0))

    def test_raw_stream_is_ignored(self):
        a = self.act(250000, 0.0, raw_gains=50.0); self.assertEqual(a['reason'], 'cut_confirmation_recorded')
        a = self.act(300000, 4.0, raw_gains=-50.0); self.assertEqual(a['reason'], 'driver_improving')

    def test_threshold_is_strict_and_one_improving_driver_blocks(self):
        a = self.act(250000, {'loss/d1': 1.5}); self.assertEqual(a['reason'], 'driver_improving')               # exactly at the threshold is NOT flat
        a = self.act(300000, {'loss/d1': 1.49}); self.assertEqual(a['reason'], 'cut_confirmation_recorded')

    def test_guards_do_not_block_a_cut(self):
        a = self.act(250000, {'loss/g0': 9.0}); self.assertEqual(a['reason'], 'cut_confirmation_recorded')

    def test_improvement_between_confirmations_resets(self):
        self.act(250000); a = self.act(300000, {'loss/d0': 2.0}); self.assertEqual(a['reason'], 'driver_improving')
        self.assertEqual(self.st['cut_endpoints'], [])
        self.act(350000); a = self.act(400000); self.assertEqual(a['type'], 'CUT')

    def test_confirmation_spacing_and_missing_driver(self):
        self.act(250000); a = self.act(275000); self.assertEqual(a['reason'], 'not_opportunity')
        self.assertEqual(self.st['cut_endpoints'], [250000])
        b = bundle(self.pol, self.st, 300000); del b['signals']['ema']['loss/d2']; b['bundle_hash'] = vc.bundle_hash(b)
        self.st, a = vc.decide(self.st, self.pol, b, 301000); self.assertEqual(a['reason'], 'driver_missing')
        self.assertEqual(self.st['last_opportunity_endpoint'], 250000)                                         # missing evidence did not consume the opportunity

    # ---- eligibility after a cut: only the RECENT window must be post-cooldown ----
    def test_recent_window_must_lie_after_the_cooldown(self):
        self.cycle(250000); L = self.st['stage_start_step']                                                    # halved at live 301000
        a = self.act(L + 50000 + 2 * 25000 - 25000, live=L + 99000); self.assertEqual(a['reason'], 'insufficient_history')
        a = self.act(self.next_endpoint(), live=self.next_endpoint() + 1000); self.assertEqual(a['reason'], 'cut_confirmation_recorded')   # older window may precede the cut

    # ---- the stop ----
    def test_four_unhelpful_halvings_in_a_row_stop_the_run(self):
        self.cycle(250000)                                                                                      # halving 1 (stage 1), counter untouched
        for k in range(1, 4):
            e = self.next_endpoint(); a = self.cycle(e)
            self.assertEqual((a['type'], self.st['unhelpful_halvings'], self.st['stage']), ('CUT', k, k + 1))
        e = self.next_endpoint(); a = self.cycle(e)
        self.assertEqual((a['type'], a['reason'], a['stop_status']), ('CONVERGED', 'converged', 'converged'))
        self.assertEqual((self.st['status'], self.st['stage'], self.st['unhelpful_halvings']), ('converged', 4, 4))
        self.assertEqual(self.st['candidate']['endpoint_step'], e + 50000)

    def test_a_helpful_halving_resets_the_counter(self):
        self.cycle(250000)
        self.cycle(self.next_endpoint()); self.assertEqual(self.st['unhelpful_halvings'], 1)
        self.cycle(self.next_endpoint()); self.assertEqual(self.st['unhelpful_halvings'], 2)
        a = self.cycle(self.next_endpoint(), {'loss/d0': 1.2}); self.assertEqual((a['type'], self.st['unhelpful_halvings']), ('CUT', 0))   # 1.0 <= g < 1.5 at both: it helped
        self.assertEqual(self.st['stage'], 4)                                                                   # no cap on halvings

    def test_one_helpful_confirmation_is_enough_to_reset(self):
        self.cycle(250000); self.cycle(self.next_endpoint()); self.assertEqual(self.st['unhelpful_halvings'], 1)
        e = self.next_endpoint(); self.act(e); a = self.act(e + 50000, {'loss/d1': 1.1})
        self.assertEqual((a['type'], self.st['unhelpful_halvings']), ('CUT', 0))

    def test_no_cap_and_halvings_continue_while_helping(self):
        self.cycle(250000)
        for _ in range(7): self.cycle(self.next_endpoint(), {'loss/d2': 1.3})
        self.assertEqual((self.st['stage'], self.st['status'], self.st['multiplier']), (8, 'running', 0.5 ** 8))

    def test_ceiling_is_never_convergence(self):
        self.st, a = vc.decide(self.st, self.pol, None, 2000000, resource_limit=True)
        self.assertEqual((a['stop_status'], self.st['status']), ('resource_limited', 'stopped'))

    def test_stop_count_is_configurable_and_terminal_state_is_final(self):
        self.pol = policy(stop_count=1); self.st = vc.new_state(self.pol, 'r')
        self.cycle(250000); a = self.cycle(self.next_endpoint()); self.assertEqual(a['type'], 'CONVERGED')
        b = bundle(self.pol, self.st, 900000); self.st, a2 = vc.decide(self.st, self.pol, b, 901000); self.assertEqual(a2['type'], 'CONTINUE')

    def test_decide_is_deterministic_and_does_not_mutate_inputs(self):
        b = bundle(self.pol, self.st, 250000); st0 = copy.deepcopy(self.st)
        s1, a1 = vc.decide(self.st, self.pol, b, 251000); s2, a2 = vc.decide(self.st, self.pol, b, 251000)
        self.assertEqual((s1, a1), (s2, a2)); self.assertEqual(self.st, st0)


if __name__ == '__main__':
    unittest.main()


class ContinueAfterBudget(unittest.TestCase):
    """Marios 2026-11-10: 2M resource budget with the option to continue."""
    def setUp(self):
        self.pol = policy(); self.st = vc.new_state(self.pol, 'run-v3')

    def _two_halvings(self):
        for e in (250000,):
            self.st, _ = vc.decide(self.st, self.pol, bundle(self.pol, self.st, e), e + 1000)
            self.st, _ = vc.decide(self.st, self.pol, bundle(self.pol, self.st, e + 50000), e + 51000)

    def test_reopen_keeps_stage_multiplier_and_counters_and_is_recorded(self):
        self._two_halvings(); st0 = copy.deepcopy(self.st)
        stopped, a = vc.decide(self.st, self.pol, None, 2000000, resource_limit=True)
        self.assertEqual((stopped['status'], stopped['stop_status']), ('stopped', 'resource_limited'))
        r = vc.reopen_after_resource_limit(stopped, 2000000)
        self.assertEqual((r['status'], r['stop_status'], r['stage'], r['multiplier'], r['unhelpful_halvings']), ('running', None, st0['stage'], st0['multiplier'], st0['unhelpful_halvings']))
        self.assertEqual(r['history'][-1]['reason'], 'reopened_after_resource_limit'); self.assertGreater(r['action_seq'], stopped['action_seq'])
        self.assertEqual(stopped['status'], 'stopped')                                                          # input not mutated
        e = r['stage_start_step'] + 50000 + 2 * 25000                                                           # and the controller decides again afterwards
        r, a = vc.decide(r, self.pol, bundle(self.pol, r, e), e + 1000)
        self.assertEqual(a['reason'], 'cut_confirmation_recorded')

    def test_only_a_resource_limited_stop_can_be_reopened(self):
        for kw in (dict(user_stop=True), dict(integrity_error='x')):
            st, _ = vc.decide(vc.new_state(self.pol, 'r'), self.pol, None, 5000, **kw)
            with self.assertRaises(ValueError): vc.reopen_after_resource_limit(st, 5000)
        st = vc.new_state(policy(stop_count=1), 'r'); pol = policy(stop_count=1)
        for e in (250000, 300000):
            st, _ = vc.decide(st, pol, bundle(pol, st, e), e + 1000)
        e = st['stage_start_step'] + 100000
        for k in (0, 50000):
            st, a = vc.decide(st, pol, bundle(pol, st, e + k), e + k + 1000)
        self.assertEqual(st['stop_status'], 'converged')
        with self.assertRaises(ValueError): vc.reopen_after_resource_limit(st, 999999)

    def test_resume_guard_in_training_control(self):
        import os
        from omtra.utils.training_control import ValidationControlMixin
        class Dummy(ValidationControlMixin):
            pass
        def make():
            d = Dummy(); d.control_enabled = True; d.control_policy = self.pol; return d
        self._two_halvings()
        blob = lambda st: dict(state=st, policy=self.pol.to_dict(), lr_multiplier=vc.lr_multiplier(st), last_polled=300000, last_manual_seq=-1)
        running = blob(self.st); d = make(); d.load_control_state(running); self.assertEqual(d.control_state['status'], 'running')        # normal resume unchanged
        stopped, _ = vc.decide(self.st, self.pol, None, 2000000, resource_limit=True)
        os.environ.pop('OMTRA_REOPEN_RESOURCE_LIMITED', None)
        with self.assertRaises(RuntimeError): make().load_control_state(blob(stopped))                                                    # terminal state: refused by default
        os.environ['OMTRA_REOPEN_RESOURCE_LIMITED'] = '1'
        try:
            d = make(); d.load_control_state(blob(stopped)); self.assertEqual(d.control_state['status'], 'running'); self.assertEqual(d.lr_multiplier, 0.5)
        finally:
            os.environ.pop('OMTRA_REOPEN_RESOURCE_LIMITED', None)
