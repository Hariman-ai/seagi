"""PATCH 21 (2026-09-08) -- NAVPAINT: the planner navigates only with
actions that have never painted the canvas.

Pinned here:
  * gate on: a state reachable only through a painting action (apply)
    is NOT reached by _dist; through an arrow it is
  * gate off: the painting transition is used (old behaviour)
  * the paint-action set comes from the effects table and refreshes
    when the table grows
  * end to end on the toy: with a spurious 'apply moves the indicator'
    transition planted in the automaton, the gate keeps the plan from
    issuing an unpredicted apply as its first action
"""
import unittest
from unittest import mock

from seagi.world import relplan as _rp
from seagi.world.relplan import RelPlan


S0 = frozenset([('p', 0, 0)])
S1 = frozenset([('q', 0, 0)])          # not comparable with S0, so no pooled transition applies
S2 = frozenset([('r', 0, 0)])


class TestNavPaint(unittest.TestCase):
    def _rp(self):
        rp = RelPlan()
        # apply (4) is a painter: an effect exists for it
        rp.effects[(4, None, frozenset([('p', 3, 3, 5, 5)]))] = [set([1, 2]), 3]
        return rp

    def test_gate_on_excludes_painting_transitions(self):
        rp = self._rp()
        auto = {(S0, 4): {S1: 5}, (S1, 2): {S2: 5}}
        with mock.patch.object(_rp, '_NAVPAINT_ON', lambda: True):
            d = rp._dist(auto, S0, 6)
        self.assertIn(S0, d); self.assertNotIn(S1, d); self.assertNotIn(S2, d)

    def test_arrows_still_navigate(self):
        rp = self._rp()
        auto = {(S0, 2): {S1: 5}, (S1, 4): {S2: 5}}
        with mock.patch.object(_rp, '_NAVPAINT_ON', lambda: True):
            d = rp._dist(auto, S0, 6)
        self.assertEqual(d[S1][0], 1); self.assertEqual(d[S1][1], 2); self.assertNotIn(S2, d)

    def test_gate_off_uses_the_painting_transition(self):
        rp = self._rp()
        auto = {(S0, 4): {S1: 5}}
        with mock.patch.object(_rp, '_NAVPAINT_ON', lambda: False):
            d = rp._dist(auto, S0, 6)
        self.assertIn(S1, d)

    def test_paint_actions_follow_the_table(self):
        rp = RelPlan()
        self.assertEqual(rp._paint_actions(), frozenset())
        rp.effects[(5, (0, 'tile', 1, 1), S0)] = [set([1]), 1]
        self.assertEqual(rp._paint_actions(), frozenset([5]))
        rp.effects[(4, None, S0)] = [set([2]), 1]
        self.assertEqual(rp._paint_actions(), frozenset([4, 5]))

    def test_toy_plan_never_reaches_by_painting(self):
        from seagi.brain.tests.test_relplan import Toy, teach, A, B
        from seagi.world.relsense import segment
        rp = RelPlan(); teach(rp)
        # plant a spurious 'apply moves the indicator' transition from the
        # fresh-life state to a state that has a gaining effect in shape 1
        toy = Toy(); G = toy.board(); objs = segment(G, 63)[1]
        outs = rp.outside(G, objs, A, B); S = rp.q(rp.state(outs)[0])
        eff, auto = rp.views()
        shape1 = [k[2] for k in eff if k[0] == 4 and k[2] != S]
        self.assertTrue(shape1)
        # full-state key for the automaton: the fresh state's full form
        Sfull = rp.state(outs)[0]
        rp.auto[(Sfull, 4)] = {next(iter(k for (k, a) in rp.auto if a == 2)): 50}
        rp._auto_n += 1
        with mock.patch.object(_rp, '_NAVPAINT_ON', lambda: True):
            r = rp.plan(G, objs, A, B, 6)
        # whatever it picks first, an apply must carry a prediction
        if r is not None and r[0] == 4:
            self.assertTrue(rp._pending['pred'], 'an apply was issued as navigation')


if __name__ == '__main__':
    unittest.main()
