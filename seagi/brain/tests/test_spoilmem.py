"""PATCH 19 (2026-09-07) -- SPOILMEM: a spoil is evidence against the
effect across lives, until the effect learns something new.

Pinned here:
  * an op spoiled more often than it worked is not planned on in the NEXT
    life (gate on); gate off: it is (the old per-life burn only)
  * once it has worked at least as often as it spoiled, it is back
  * a spoil is forgiven when the effect's cells have grown since
  * an executed pick that IMPROVED the picture counts as worked; one
    that WORSENED it counts as spoiled with the cells known then
  * spoiled/worked round-trip through to_dict/from_dict
  * (19b) the memory is keyed on the PAIR (state at pick time, effect):
    the same effect from a DIFFERENT state is untouched
"""
import unittest
from unittest import mock
import numpy as np

from seagi.world import relplan as _rp
from seagi.world.relplan import RelPlan
from seagi.world.relsense import segment
from seagi.brain.tests.test_relplan import Toy, teach, A, B, CLICK


def _apply_ops(rp):
    eff, _ = rp.views()
    return [k for k in eff if k[0] == 4]


def _first(rp, toy):
    G = toy.board(); objs = segment(G, 63)[1]
    return rp.plan(G, objs, A, B, 6, board_key=str(toy.t))


def _S_of(rp, toy):
    """The quotient state the planner reads on this board (the pair key)."""
    G = toy.board(); objs = segment(G, 63)[1]
    return rp.q(rp.state(rp.outside(G, objs, A, B))[0])


class TestSpoilMem(unittest.TestCase):
    def setUp(self):
        self.rp = RelPlan(); teach(self.rp)
        self.ops = _apply_ops(self.rp)
        self.assertTrue(self.ops)

    def _mark_spoiled(self, n=1, cells_delta=0):
        eff, _ = self.rp.views()
        S = _S_of(self.rp, Toy())
        for k in self.ops:
            self.rp.spoiled[(S, k)] = [n, len(eff[k][0]) + cells_delta]

    def test_spoiled_op_not_planned_next_life_gate_on(self):
        self._mark_spoiled()
        self.rp.begin_life()                       # _burned is cleared; spoiled is not
        with mock.patch.object(_rp, '_SPOILMEM_ON', lambda: True):
            r = _first(self.rp, Toy())
        self.assertTrue(r is None or r[0] != 4, r)
        self.assertEqual(self.rp.why, 'no_ops')

    def test_gate_off_keeps_the_old_behaviour(self):
        self._mark_spoiled()
        self.rp.begin_life()
        with mock.patch.object(_rp, '_SPOILMEM_ON', lambda: False):
            r = _first(self.rp, Toy())
        self.assertIsNotNone(r); self.assertEqual(r[0], 4)

    def test_working_as_often_as_it_spoiled_restores_it(self):
        self._mark_spoiled(2)
        S = _S_of(self.rp, Toy())
        for k in self.ops:
            self.rp.worked[(S, k)] = 2
        with mock.patch.object(_rp, '_SPOILMEM_ON', lambda: True):
            r = _first(self.rp, Toy())
        self.assertIsNotNone(r); self.assertEqual(r[0], 4)

    def test_new_cells_forgive_a_spoil(self):
        self._mark_spoiled(1, cells_delta=-1)          # spoiled when less was known
        with mock.patch.object(_rp, '_SPOILMEM_ON', lambda: True):
            r = _first(self.rp, Toy())
        self.assertIsNotNone(r); self.assertEqual(r[0], 4)

    def test_executed_pick_is_scored_worked_or_spoiled(self):
        toy = Toy()
        r = _first(self.rp, toy)                   # a plan pick with a prediction
        self.assertIsNotNone(r); self.assertEqual(r[0], 4)
        op = (self.rp._pending['S'], self.rp._pending['op']); self.assertIsNotNone(op[1])
        Gp = toy.board(); op_ = segment(Gp, 63)[1]
        toy.step(4)                                # the pick runs and improves the picture
        Gc = toy.board(); oc = segment(Gc, 63)[1]
        self.rp.observe(Gp, op_, Gc, oc, 4, False, None, A, B)
        self.assertEqual(self.rp.worked.get(op), 1)
        self.assertNotIn(op, self.rp.spoiled)
        # now a pick whose paint WORSENS the picture: paint the right mask
        # with the wrong colour by lying to the toy about the selection
        r = _first(self.rp, toy)
        if r is not None and r[0] == CLICK:
            toy.step(CLICK, r[1]); r = _first(self.rp, toy)
        if r is not None and r[0] == 2:
            toy.step(2); r = _first(self.rp, toy)
        if r is not None and r[0] == 4:
            op2 = (self.rp._pending['S'], self.rp._pending['op'])
            Gp = toy.board(); op_ = segment(Gp, 63)[1]
            toy.shape = 0; toy.colour = 3          # the world repaints the CORRECT half with a colour nobody wants
            toy.step(4)
            Gc = toy.board(); oc = segment(Gc, 63)[1]
            self.rp.observe(Gp, op_, Gc, oc, 4, False, None, A, B)
            self.assertEqual(self.rp.spoiled.get(op2, [0, 0])[0], 1)
            self.assertIn(op2[1], self.rp._burned)

    def test_a_spoil_from_one_state_leaves_the_effect_free_elsewhere(self):
        eff, _ = self.rp.views()
        other = frozenset([('nowhere', 0, 0)])
        for k in self.ops:
            self.rp.spoiled[(other, k)] = [5, len(eff[k][0])]
        with mock.patch.object(_rp, '_SPOILMEM_ON', lambda: True):
            r = _first(self.rp, Toy())
        self.assertIsNotNone(r); self.assertEqual(r[0], 4)

    def test_round_trip(self):
        self._mark_spoiled(3)
        S = _S_of(self.rp, Toy())
        for k in self.ops:
            self.rp.worked[(S, k)] = 1
        d = self.rp.to_dict()
        rp2 = RelPlan(); rp2.from_dict(d)
        self.assertEqual(rp2.spoiled, self.rp.spoiled)
        self.assertEqual(rp2.worked, self.rp.worked)
        rp2.from_dict(d)                           # idempotent
        self.assertEqual(rp2.spoiled, self.rp.spoiled)


if __name__ == '__main__':
    unittest.main()
