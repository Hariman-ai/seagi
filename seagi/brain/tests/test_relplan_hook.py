"""RELPLAN inside the world: the learning hook, the plan API, persistence.

Pinned here:
  * gate absent: rel_plan_action is None; learned tables still persist
  * the hook learns from a scripted life where a picture is pursued and
    then proposes the first action of a plan, keeping the aim for step()
  * the pick is cleared when the plan has nothing to say
  * the plan rides inside `world_relsense` per level and round-trips
  * errors are counted, never raised
"""
import unittest
import numpy as np

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld
from seagi.world.relsense import RelSense

W = 64; BG = 5
A = (34, 27, 43, 36); B = (3, 3, 12, 12)
SW = {15: (3, 40), 12: (3, 46)}
MASKS = {0: [(r, c) for r in range(5) for c in range(10)],
         1: [(r, c) for r in range(5, 10) for c in range(10)]}
CLICK = 5


class Toy(object):
    def __init__(self):
        self.shape = 0; self.colour = 15; self.canvas = np.zeros((10, 10), int); self.t = 0

    def board(self):
        G = np.full((W, W), BG, int)
        for r in range(10):
            for c in range(10):
                G[B[0] + r, B[1] + c] = 15 if r < 5 else 12
        G[A[0]:A[2] + 1, A[1]:A[3] + 1] = self.canvas
        for col, (r, c) in SW.items():
            G[r:r + 3, c:c + 3] = col
        w = 6 if self.shape == 0 else 9
        G[20:24, 40:40 + w] = self.colour
        G[63, 0:max(1, 60 - self.t)] = 4
        return [[int(x) for x in row] for row in G]

    def step(self, a, aim=None):
        self.t += 1
        if a == 2:
            self.shape = 1 - self.shape
        elif a == 4:
            for r, c in MASKS[self.shape]:
                self.canvas[r, c] = self.colour
        elif a == CLICK and aim is not None:
            for col, (r, c) in SW.items():
                if r <= aim[0] < r + 3 and c <= aim[1] < c + 3:
                    self.colour = col


class _Stub(object):
    def __init__(self, game='vgame'):
        self.game_id = game
        self._levels = 1
        self._grid = None
        self._steps_episode = 0
        self._state = 'NOT_FINISHED'

    _bar_of = lambda self: 63
    _board_hash_nb = lambda self: b'k'
    _rel_objs_for = ARCWorld._rel_objs_for
    _rel_picture = staticmethod(ARCWorld._rel_picture)
    _relplan_step = ARCWorld._relplan_step
    rel_plan_action = ARCWorld.rel_plan_action
    rel_to_dict = ARCWorld.rel_to_dict
    rel_from_dict = ARCWorld.rel_from_dict


def teach(w):
    """A scripted life through the hook: apply, cycle, click 12, apply."""
    toy = Toy()
    w._grid = toy.board()
    # CARRIERHUE (patch 17): the shape-0 indicator must wear two colours
    # before it can be read -- click 12, then 15, then the old lesson
    for k, (a, aim) in enumerate([(CLICK, (4, 47)), (CLICK, (4, 41)),
                                  (4, None), (2, None), (CLICK, (4, 47)), (4, None)]):
        prev = w._grid
        toy.step(a, aim)
        w._grid = toy.board()
        w._steps_episode = k
        w._relplan_step(prev, 1, a, a == CLICK, aim)
    return toy


class _Base(unittest.TestCase):
    ON = True

    def setUp(self):
        self._g1 = arc_world._RELPLAN_ON; self._g2 = arc_world._RELSENSE_ON
        arc_world._RELPLAN_ON = lambda: self.ON
        arc_world._RELSENSE_ON = lambda: True
        ARCWorld._rel.clear(); ARCWorld._relplan.clear()
        ARCWorld._relplan_errors = 0
        rs = RelSense(); rs._pursued = ('eq', A, B)
        ARCWorld._rel[('vgame', 1)] = rs
        self.w = _Stub()

    def tearDown(self):
        arc_world._RELPLAN_ON = self._g1; arc_world._RELSENSE_ON = self._g2
        ARCWorld._rel.clear(); ARCWorld._relplan.clear()


class TestGateOff(_Base):
    ON = False

    def test_nothing(self):
        teach(self.w)
        self.assertIsNone(self.w.rel_plan_action(6))
        self.assertIsNone(self.w._plan_pick)
        # what was learned PERSISTS with the gate off (the review: rm
        # RELPLAN_ON must stop the acting, never erase the model)
        d = self.w.rel_to_dict()
        self.assertIn('plan', (d.get('vgame') or {}).get('1') or {})


class TestHook(_Base):
    def test_learns_and_plans(self):
        teach(self.w)
        rp = ARCWorld._relplan[('vgame', 1)]
        self.assertEqual(len(rp.effects), 2)
        toy = Toy(); self.w._grid = toy.board()
        a = self.w.rel_plan_action(6)
        self.assertEqual(a, 4)
        self.assertEqual(self.w._plan_pick, (4, None))
        toy.step(4); self.w._grid = toy.board()
        a = self.w.rel_plan_action(6)
        self.assertEqual(a, CLICK)
        self.assertEqual(self.w._plan_pick[0], CLICK)
        self.assertIsNotNone(self.w._plan_pick[1])
        self.assertEqual(ARCWorld._relplan_errors, 0)

    def test_plan_works_on_the_picture_whatever_is_pursued(self):
        teach(self.w)
        rs = ARCWorld._rel[('vgame', 1)]
        # the sense chases a marker; the picture is still a candidate
        rs._pursued = ('near', (0, 100))
        rs.v0[('eq', A, B)] = 100.0
        toy = Toy(); self.w._grid = toy.board()
        self.assertEqual(self.w.rel_plan_action(6), 4)
        rs.v0.clear()
        self.assertIsNone(self.w.rel_plan_action(6))

    def test_pick_cleared_when_silent(self):
        teach(self.w)
        toy = Toy(); toy.canvas[:5] = 15; toy.canvas[5:] = 12
        self.w._grid = toy.board()
        self.assertIsNone(self.w.rel_plan_action(6))
        self.assertIsNone(self.w._plan_pick)

    def test_round_trip(self):
        teach(self.w)
        d = self.w.rel_to_dict()
        blob = d['vgame']['1']
        self.assertIn('plan', blob)
        ARCWorld._relplan.clear()
        rs = RelSense(); rs._pursued = ('eq', A, B)
        ARCWorld._rel.clear(); ARCWorld._rel[('vgame', 1)] = rs
        self.assertGreater(self.w.rel_from_dict(d), 0)
        self.assertIn(('vgame', 1), ARCWorld._relplan)
        toy = Toy(); self.w._grid = toy.board()
        self.assertEqual(self.w.rel_plan_action(6), 4)

    def test_errors_counted(self):
        teach(self.w)
        self.w._grid = "not a grid"
        self.assertIsNone(self.w.rel_plan_action(6))
        self.assertGreaterEqual(ARCWorld._relplan_errors, 1)


if __name__ == "__main__":
    unittest.main()
