"""PATCH 40 (2026-09-11) -- SHORTCUT: on a level he has CLEARED, his own full search runs
once per attempt at its first plan; a sequence reaching the picture in fewer actions than
the rest of the won route is latched and served ahead of the route.

Pinned here (Toy6: three masks, the whole job is three paints 12 -> 15 -> 9):
  * a shortcut shorter than the route left is latched, tagged, and executed to mm 0
  * one not shorter is not latched and the pick is unchanged
  * gate off: identical to no route_left
  * one try per attempt; begin_life re-arms it
  * the world's verdict: fewer realised actions = won; a clear not shorter, or no clear,
    refutes that route length (persisted, round-trips), and a refuted length is not retried
  * a dropped shortcut frees the lookahead at floor 0; the verdict stays the world's
"""
import unittest
from unittest import mock

from seagi.world import relplan as _rp
from seagi.world.relplan import RelPlan
from seagi.brain.tests.test_curious import Toy6, _teach6, _stalled6, _step, _plan, _gates


class _SC(unittest.TestCase):
    def setUp(self):
        self._ps = _gates(_LOOKAHEAD_ON=lambda: True) + [
            mock.patch.object(_rp, '_SHORTCUT_ON', lambda: True),
            mock.patch.object(_rp, '_LAFULL_ON', lambda: False)]
        for p in self._ps:
            p.start()

    def tearDown(self):
        for p in self._ps:
            p.stop()

    @staticmethod
    def _fresh():
        rp = RelPlan(); _teach6(rp); toy = Toy6(); rp.begin_life()
        return rp, toy


class TestShortcut(_SC):
    def test_a_shortcut_shorter_than_the_route_is_latched_and_clears(self):
        rp, toy = self._fresh()
        r = _plan(rp, toy, floor=0, route_left=(50, 50))
        self.assertIsNotNone(r, rp.why)
        self.assertEqual(r[2].get('shortcut'), 50)
        self.assertEqual(rp.shortcuts, 1); self.assertIsNotNone(rp._seq)
        self.assertEqual(rp._sc_life, 50)
        self.assertTrue(any(e.startswith('shortcut_latch') for e in rp.events), rp.events)
        n = 0
        while r is not None and n < 40:
            _step(rp, toy, r[0], r[1]); n += 1
            if toy.mm() == 0:
                break
            r = _plan(rp, toy, floor=0)
        self.assertEqual(toy.mm(), 0, (n, rp.events))
        self.assertLess(n, 50)
        self.assertEqual(rp.shortcut_done, 1)

    def test_not_shorter_than_the_route_left_is_not_latched(self):
        rp, toy = self._fresh()
        base, btoy = self._fresh()
        r0 = _plan(base, btoy, floor=0)
        r = _plan(rp, toy, floor=0, route_left=(3, 3))
        self.assertEqual(rp.shortcuts, 0); self.assertIsNone(rp._seq); self.assertIsNone(rp._sc_life)
        self.assertTrue(any(e.startswith('shortcut_none') for e in rp.events), rp.events)
        self.assertEqual(None if r is None else (r[0], r[1]), None if r0 is None else (r0[0], r0[1]))
        self.assertIsNone(None if r is None else r[2].get('shortcut'))

    def test_gate_off_is_identical(self):
        with mock.patch.object(_rp, '_SHORTCUT_ON', lambda: False):
            rp, toy = self._fresh(); r = _plan(rp, toy, floor=0, route_left=(50, 50))
            rq, tq = self._fresh(); q = _plan(rq, tq, floor=0)
        self.assertEqual(rp.shortcuts, 0); self.assertIsNone(rp._seq)
        self.assertEqual(None if r is None else (r[0], r[1], r[2]), None if q is None else (q[0], q[1], q[2]))
        self.assertFalse(any('shortcut' in e for e in rp.events), rp.events)
        self.assertEqual(rp.lookaheads, rq.lookaheads)

    def test_one_try_per_attempt_begin_life_rearms(self):
        rp, toy = self._fresh()
        _plan(rp, toy, floor=0, route_left=(3, 3))
        _plan(rp, toy, floor=0, route_left=(50, 50))
        self.assertEqual(rp.shortcuts, 0)
        rp.begin_life()
        _plan(rp, Toy6(), floor=0, route_left=(50, 50))
        self.assertEqual(rp.shortcuts, 1)

    def test_verdict_fewer_actions_wins_anything_else_refutes(self):
        rp, toy = self._fresh(); _plan(rp, toy, floor=0, route_left=(50, 50))
        self.assertTrue(rp.shortcut_outcome(True, 12).startswith('shortcut_won'))
        self.assertIsNone(rp._sc_refuted); self.assertEqual(rp.shortcut_won, 1)
        self.assertIsNone(rp.shortcut_outcome(True, 12))                  # nothing pending
        rp.begin_life(); _plan(rp, Toy6(), floor=0, route_left=(50, 50))
        self.assertTrue(rp.shortcut_outcome(True, 50).startswith('shortcut_refuted'))  # not shorter
        self.assertEqual(rp._sc_refuted, 50)
        rp.begin_life(); _plan(rp, Toy6(), floor=0, route_left=(50, 50))
        self.assertEqual(rp.shortcuts, 2)                                  # refuted length: not tried
        rp.begin_life(); _plan(rp, Toy6(), floor=0, route_left=(40, 40))
        self.assertEqual(rp.shortcuts, 3)                                  # a new route length: tried
        self.assertTrue(rp.shortcut_outcome(False, None).startswith('shortcut_refuted'))  # no clear
        self.assertEqual(rp._sc_refuted, 40); self.assertEqual(rp.shortcut_refuted, 2)

    def test_refutation_persists_and_an_untouched_table_is_unchanged(self):
        rp, toy = self._fresh()
        self.assertNotIn('sc_refuted', rp.to_dict())
        _plan(rp, toy, floor=0, route_left=(50, 50)); rp.shortcut_outcome(False, None)
        d = rp.to_dict(); self.assertEqual(d['sc_refuted'], 50)
        rq = RelPlan(); rq.from_dict(d); self.assertEqual(rq._sc_refuted, 50)
        rq.begin_life(); _plan(rq, Toy6(), floor=0, route_left=(50, 50))
        self.assertEqual(rq.shortcuts, 0)

    def test_a_dropped_shortcut_frees_the_lookahead_at_floor_zero(self):
        rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
        self.assertIsNone(_plan(rp, toy, floor=0)); self.assertEqual(rp.lookaheads, 0)
        rp._seq = {"steps": [(4, None, frozenset(), 12)], "i": 0, "canvases": [None], "shortcut": 50}
        rp._drop_seq("misprediction")
        self.assertTrue(rp._sc_off); self.assertEqual(rp.shortcut_dropped, 1)
        self.assertIsNone(rp._sc_refuted)                                  # the verdict is the world's
        r = _plan(rp, toy, floor=0)
        self.assertIsNotNone(r, rp.why); self.assertEqual(rp.lookaheads, 1)
        with mock.patch.object(_rp, '_SHORTCUT_ON', lambda: False):
            rq = RelPlan(); _teach6(rq); tq = _stalled6(rq); rq._sc_off = True
            self.assertIsNone(_plan(rq, tq, floor=0)); self.assertEqual(rq.lookaheads, 0)


class TestYield(_SC):
    """Review of 40: a greedy plan that completes the picture in fewer steps than
    the latch has left outranks the latch (offline cd82 L4: 16 -> stuck vs 13 clear)."""
    def _latched(self, rp, toy):
        toy.canvas[0:5, :] = 15; toy.canvas[5:8, :] = 12          # rows 8-9 (9) left: one round
        rp._sc_life = 50
        rp._seq = {"steps": [(4, None, frozenset([("x", 0, 0)]), 12)], "i": 0,
                   "canvases": [None], "shortcut": 50}
        return ([(4, None, frozenset([("x", 0, 0)]), 12, 1, 99)], toy.canvas.copy(), 99)

    def test_a_greedy_that_completes_sooner_outranks_the_latch(self):
        rp, toy = self._fresh()
        fake = self._latched(rp, toy)
        with mock.patch.object(rp, '_serve_seq', lambda *a, **k: fake):
            r = _plan(rp, toy, floor=0)
        self.assertIsNotNone(r, rp.why)
        self.assertIsNone(rp._seq); self.assertEqual(rp.shortcut_yields, 1)
        self.assertEqual(r[2]['mm1'], 0)
        self.assertEqual(r[2].get('shortcut'), 50)                    # still outranks the route
        self.assertTrue(any(e.startswith('shortcut_yield') for e in rp.events), rp.events)
        self.assertEqual(rp._sc_life, 50)                             # the verdict stays the world's
        self.assertTrue(rp._sc_off)       # review 2 of 40: a stalled greedy may still search
        r2 = _plan(rp, toy, floor=0)
        self.assertEqual(None if r2 is None else r2[2].get('shortcut'), 50)
        rp.begin_life()
        self.assertFalse(rp._sc_yield); self.assertFalse(rp._sc_off)
        # review 2 of 40: an attempt left mid-way (a game switch) never got a
        # verdict; a later ordinary clear must not be judged against it
        self.assertIsNone(rp._sc_life)

    def test_gate_off_the_latch_is_served_as_before(self):
        with mock.patch.object(_rp, '_SHORTCUT_ON', lambda: False):
            rp, toy = self._fresh()
            fake = self._latched(rp, toy)
            with mock.patch.object(rp, '_serve_seq', lambda *a, **k: fake):
                _plan(rp, toy, floor=0)
        self.assertEqual(rp.shortcut_yields, 0); self.assertIsNotNone(rp._seq)


class TestWorldVerdict(unittest.TestCase):
    def test_the_world_hands_the_outcome_to_that_levels_plan(self):
        from seagi.world.arc_world import ARCWorld
        rp = RelPlan(); rp._sc_life = 28
        k = ('tgame-shortcut', 3)
        old = ARCWorld._relplan.get(k); ARCWorld._relplan[k] = rp
        try:
            ARCWorld._shortcut_verdict(object(), k, True, 14)
            self.assertEqual(rp.shortcut_won, 1); self.assertIsNone(rp._sc_refuted)
            rp._sc_life = 28
            ARCWorld._shortcut_verdict(object(), k, False, None)
            self.assertEqual(rp._sc_refuted, 28)
            ARCWorld._shortcut_verdict(object(), ('no-such-game', 0), True, 1)   # no plan: silent
        finally:
            if old is None:
                ARCWorld._relplan.pop(k, None)
            else:
                ARCWorld._relplan[k] = old


if __name__ == '__main__':
    unittest.main()
