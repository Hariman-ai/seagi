"""PATCH 41 (2026-09-12) -- PLACEKEY: a latched shortcut navigates on a paint-blind,
colour-blind key for WHERE he stands; the quotient state keeps saying what a paint does.

Pinned here:
  * gate off: observe/plan with place arguments change nothing
  * observe learns the place automaton (arrows only, never clicks) and what a paint from a
    place puts down, keyed on the click target too
  * _reach_place: a place whose (action, click target) painted the step's cells is the
    target; BFS path; "already there" is (0, None); no known path -> an untried arrow
    (explore, marked as such); no target -> None
  * the place tables and the refutation evidence round-trip through to_dict/from_dict
  * a refuted route length is retried only when paints are known and the places known have
    grown (gate on), at most 3 times, and never when the gate is off
"""
import unittest
from unittest import mock

from seagi.world import relplan as _rp
from seagi.world.relplan import RelPlan
from seagi.brain.tests.test_curious import Toy6, _teach6, _step, _plan, _gates


class _PK(unittest.TestCase):
    def setUp(self):
        self._ps = _gates(_LOOKAHEAD_ON=lambda: True) + [
            mock.patch.object(_rp, '_SHORTCUT_ON', lambda: True),
            mock.patch.object(_rp, '_PLACEKEY_ON', lambda: True),
            mock.patch.object(_rp, '_LAFULL_ON', lambda: False)]
        for p in self._ps:
            p.start()

    def tearDown(self):
        for p in self._ps:
            p.stop()


def _obs(rp, toy, a, aim=None, place=None, place_next=None):
    from seagi.world.relsense import segment
    from seagi.brain.tests.test_relplan import A, B
    Gp = toy.board(); op = segment(Gp, 63)[1]
    toy.step(a, aim)
    Gc = toy.board(); oc = segment(Gc, 63)[1]
    rp.observe(Gp, op, Gc, oc, a, a == 5, aim, A, B, place=place, place_next=place_next)


class TestLearn(_PK):
    def test_gate_off_learns_nothing(self):
        with mock.patch.object(_rp, '_PLACEKEY_ON', lambda: False):
            rp = RelPlan(); toy = Toy6(); rp.begin_life()
            _obs(rp, toy, 2, place='p0', place_next='p1')
            _obs(rp, toy, 4, place='p1', place_next='p1')
        self.assertEqual(rp.place_auto, {}); self.assertEqual(rp.place_paint, {})
        self.assertNotIn('place_auto', rp.to_dict())

    def test_arrows_and_paints_are_learned_per_place(self):
        rp = RelPlan(); _teach6(rp); toy = Toy6(); rp.begin_life()
        _obs(rp, toy, 2, place='p0', place_next='p1')
        _obs(rp, toy, 2, place='p1', place_next='p2')
        _obs(rp, toy, 4, place='p2', place_next='p2')          # a paint from p2
        self.assertEqual(rp.place_auto[('p0', 2)], {'p1': 1})
        self.assertEqual(rp.place_auto[('p1', 2)], {'p2': 1})
        self.assertIn(('p2', 4, None), rp.place_paint)
        (mask, n), = rp.place_paint[('p2', 4, None)].items()
        self.assertGreater(len(mask), 0); self.assertEqual(n, 1)
        self.assertTrue(any(k[0] == 4 for k in rp.place_of))

    def test_clicks_never_enter_the_place_automaton(self):
        rp = RelPlan(); _teach6(rp); toy = Toy6(); rp.begin_life()
        _obs(rp, toy, 5, (2, 47), place='p0', place_next='p0x')
        self.assertNotIn(('p0', 5), rp.place_auto)

    def test_round_trip(self):
        rp = RelPlan(); _teach6(rp); toy = Toy6(); rp.begin_life()
        _obs(rp, toy, 2, place='p0', place_next='p1')
        _obs(rp, toy, 4, place='p1', place_next='p1')
        rp._sc_refuted = 28; rp._sc_refuted_places = 5; rp._sc_retries = 2
        d = rp.to_dict()
        rq = RelPlan(); rq.from_dict(d)
        self.assertEqual(rq.place_auto, rp.place_auto)
        self.assertEqual(rq.place_paint, rp.place_paint)
        self.assertEqual(rq.place_of, rp.place_of)
        self.assertEqual((rq._sc_refuted, rq._sc_refuted_places, rq._sc_retries), (28, 5, 2))


class TestReach(_PK):
    def _rp(self):
        rp = RelPlan(); _teach6(rp); rp.begin_life()
        eff = rp.views()[0]
        op = next(k for k in eff if k[0] == 4)             # some apply effect (ck None)
        return rp, op, frozenset(eff[op][0])

    def test_target_is_a_place_that_paints_the_cells(self):
        rp, op, cells = self._rp()
        rp.place_paint[('far', 4, None)] = {cells: 1}
        rp.place_auto[('here', 2)] = {'mid': 3}; rp.place_auto[('mid', 2)] = {'far': 3}
        self.assertEqual(rp._reach_place('here', op, 6), (2, 2))
        self.assertEqual(rp.place_hits, 1); self.assertFalse(rp._nav_explore)

    def test_a_half_mask_counts_a_stray_or_another_click_target_does_not(self):
        rp, op, cells = self._rp()
        half = frozenset(sorted(cells)[:max(1, len(cells) // 2)])
        rp.place_paint[('far', 4, None)] = {half: 1}
        rp.place_paint[('wrong', 4, None)] = {frozenset(list(cells)[:1] + [999]): 1}
        rp.place_paint[('tile', 4, (14, 'abc', 1, 1))] = {cells: 1}     # another click target
        rp.place_auto[('here', 1)] = {'wrong': 2}; rp.place_auto[('here', 2)] = {'far': 2}
        rp.place_auto[('here', 3)] = {'tile': 2}
        self.assertEqual(rp._reach_place('here', op, 6), (1, 2))

    def test_already_there(self):
        rp, op, cells = self._rp()
        rp.place_paint[('here', 4, None)] = {cells: 1}
        self.assertEqual(rp._reach_place('here', op, 6), (0, None))

    def test_no_known_path_explores_an_untried_arrow(self):
        rp, op, cells = self._rp()
        rp.place_paint[('far', 4, None)] = {cells: 1}
        rp.place_auto[('here', 0)] = {'dead': 2}               # arrow 0 tried, leads nowhere useful
        r = rp._reach_place('here', op, 6)
        self.assertEqual(r[0], 1); self.assertIn(r[1], (1, 2, 3))   # an untried non-paint arrow
        self.assertEqual(rp.place_explores, 1); self.assertTrue(rp._nav_explore)
        self.assertTrue(any(e.startswith('place_explore') for e in rp.events), rp.events)

    def test_no_target_is_none(self):
        rp, op, cells = self._rp()
        rp.place_auto[('here', 2)] = {'mid': 3}
        self.assertIsNone(rp._reach_place('here', op, 6))
        self.assertTrue(any(e.startswith('place_miss why=no_target') for e in rp.events), rp.events)


class TestRetry(_PK):
    def _rp(self):
        rp = RelPlan(); rp._sc_refuted = 28; rp._sc_refuted_places = 2
        rp.place_auto[('a', 0)] = {'b': 1}; rp.place_auto[('b', 0)] = {'a': 1}
        rp.place_paint[('a', 4, None)] = {frozenset([1, 2]): 1}
        return rp

    def test_retry_is_earned_by_new_places(self):
        rp = self._rp()
        self.assertTrue(rp._sc_is_refuted(28))                # 2 places known, as then
        rp.place_auto[('c', 1)] = {'a': 1}                    # a third place learned
        self.assertFalse(rp._sc_is_refuted(28))
        self.assertFalse(rp._sc_is_refuted(30))               # another length: never refuted

    def test_no_paints_known_no_retry(self):
        rp = self._rp(); rp.place_paint = {}
        rp.place_auto[('c', 1)] = {'a': 1}
        self.assertTrue(rp._sc_is_refuted(28))

    def test_a_legacy_refutation_retries_once_paints_exist(self):
        rp = self._rp(); rp._sc_refuted_places = None
        self.assertFalse(rp._sc_is_refuted(28))
        rp.place_paint = {}
        self.assertTrue(rp._sc_is_refuted(28))

    def test_retries_are_capped(self):
        rp = self._rp(); rp.place_auto[('c', 1)] = {'a': 1}
        rp._sc_retries = 3
        self.assertTrue(rp._sc_is_refuted(28))

    def test_gate_off_refuted_stays_refuted(self):
        rp = self._rp(); rp.place_auto[('c', 1)] = {'a': 1}
        with mock.patch.object(_rp, '_PLACEKEY_ON', lambda: False):
            self.assertTrue(rp._sc_is_refuted(28))

    def test_the_verdict_records_the_evidence_and_resets_retries_on_a_new_length(self):
        rp = RelPlan(); rp._sc_life = 28; rp._sc_retries = 2; rp._sc_refuted = 28
        rp.place_auto[('a', 0)] = {'b': 1}; rp.place_auto[('b', 0)] = {'a': 1}
        self.assertTrue(rp.shortcut_outcome(False, None).startswith('shortcut_refuted'))
        self.assertEqual(rp._sc_refuted_places, 2); self.assertEqual(rp._sc_retries, 2)
        rp._sc_life = 20
        rp.shortcut_outcome(True, 25)
        self.assertEqual(rp._sc_refuted, 20); self.assertEqual(rp._sc_retries, 0)


if __name__ == '__main__':
    unittest.main()
