"""RELSENSE inside the world: the hook, the context, the persistence.

Pinned here:
  * gate absent: no context, nothing persisted (no test reads the file)
  * the first step of a life begins the sense on the RESET board (the
    grid before the action), then observes the new board
  * a level clear confirms on the board before the winning move and
    starts the new level's sense
  * errors are counted, never raised into step()
  * to_dict/from_dict round trip per game
"""
import unittest
import numpy as np

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld
from seagi.world.relsense import STATIC_AFTER

W = 64; BG = 5


def board():
    return [[BG] * W for _ in range(W)]


def paint(G, r0, c0, r1, c1, col):
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            G[r][c] = col


def picture(G, r0, c0):
    for r in range(10):
        for c in range(10):
            G[r0 + r][c0 + c] = 15 if r + c < 9 else 12


class _Stub(object):
    def __init__(self, game='vgame'):
        self.game_id = game
        self._levels = 1
        self._grid = board()
        self._steps_episode = 0
        # a stub must carry what the real class carries: the hook skips
        # a terminal frame, and reads this to know
        self._state = 'NOT_FINISHED'

    _bar_of = lambda self: -1
    _relsense_step = ARCWorld._relsense_step
    rel_context = ARCWorld.rel_context
    rel_to_dict = ARCWorld.rel_to_dict
    rel_from_dict = ARCWorld.rel_from_dict


class _Base(unittest.TestCase):
    ON = True

    def setUp(self):
        self._g = arc_world._RELSENSE_ON
        arc_world._RELSENSE_ON = lambda: self.ON
        ARCWorld._rel.clear()
        ARCWorld._self.pop('vgame', None)
        ARCWorld._rel_confirms = 0
        ARCWorld._rel_errors = 0
        self.w = _Stub()

    def tearDown(self):
        arc_world._RELSENSE_ON = self._g
        ARCWorld._rel.clear()


class TestGateOff(_Base):
    ON = False

    def test_nothing(self):
        self.assertIsNone(self.w.rel_context())
        self.assertEqual(self.w.rel_to_dict(), {})
        self.assertEqual(self.w.rel_from_dict({'vgame': {'1': {'confirmed': ['eq:1,1,2,2:3,3,4,4']}}}), 0)
        self.assertEqual(ARCWorld._rel, {})


class TestHook(_Base):

    def make(self):
        G = board()
        paint(G, 0, 0, 17, 17, 4); paint(G, 2, 2, 15, 15, BG); picture(G, 3, 3)
        paint(G, 34, 27, 43, 36, 0)
        return G

    def test_first_step_begins_the_life_on_the_reset_board(self):
        reset = self.make()
        self.w._grid = [row[:] for row in reset]
        self.w._grid[34][27] = 15               # the first action painted a cell
        self.w._steps_episode = 0
        self.w._relsense_step(reset, 1, False)
        rs = ARCWorld._rel[('vgame', 1)]
        self.assertEqual(rs.lives, 1)
        self.assertIn((34, 27, 43, 36), rs.regs)
        self.assertTrue(rs.regs[(34, 27, 43, 36)][1], 'the canvas changed on step 1')
        self.assertIsNotNone(self.w._rel_last)

    def test_a_clear_confirms_on_the_board_before_the_win(self):
        reset = self.make()
        self.w._grid = [row[:] for row in reset]
        self.w._steps_episode = 0
        self.w._relsense_step(reset, 1, False)
        rs = ARCWorld._rel[('vgame', 1)]
        for k in range(STATIC_AFTER + 1):
            self.w._steps_episode = k + 1
            self.w._relsense_step(reset, 1, False)
        # one cell painted: the pair is discovered with a baseline
        part = [row[:] for row in self.w._grid]
        part[34][27] = 15
        self.w._grid = part
        self.w._steps_episode = STATIC_AFTER + 2
        self.w._relsense_step(reset, 1, False)
        self.assertEqual(len(rs.pairs()), 1)
        pre = [row[:] for row in part]
        picture(pre, 34, 27)                    # the board before the winning move
        self.w._grid = pre
        self.w._steps_episode = STATIC_AFTER + 3
        o = self.w._relsense_step(reset, 1, False)
        self.assertGreater(self.w._rel_last['progress'], 0.0, 'completing the picture is progress')
        # the winning move: level 2 board arrives
        self.w._levels = 2
        self.w._grid = board()
        self.w._relsense_step(pre, 1, True)
        self.assertEqual(ARCWorld._rel_confirms, 1)
        self.assertEqual(len(rs.confirmed), 1)
        self.assertIn(('vgame', 2), ARCWorld._rel)
        d = self.w.rel_to_dict()
        self.assertIn('eq:34,27,43,36:3,3,12,12', d['vgame']['1']['confirmed'])
        ARCWorld._rel.clear()
        self.assertGreater(self.w.rel_from_dict(d), 0)
        self.assertIn(('vgame', 1), ARCWorld._rel)

    def test_context_for_a_picture_and_for_a_marker(self):
        reset = self.make()
        self.w._grid = [row[:] for row in reset]
        self.w._steps_episode = 0
        self.w._relsense_step(reset, 1, False)
        rs = ARCWorld._rel[('vgame', 1)]
        for k in range(STATIC_AFTER + 1):
            self.w._steps_episode = k + 1
            self.w._relsense_step(reset, 1, False)
        self.w._grid[34][27] = 15
        self.w._relsense_step(reset, 1, False)
        self.assertEqual(self.w.rel_context(), ('eq',))
        # a marker game: him + a goal; he is known once he has translated
        ARCWorld._rel.clear()
        G = board(); paint(G, 10, 10, 12, 12, 9); paint(G, 50, 50, 52, 52, 14)
        self.w._grid = [row[:] for row in G]; self.w._steps_episode = 0
        self.w._relsense_step(G, 1, False)
        for k in range(4):
            prev = [row[:] for row in self.w._grid]
            paint(self.w._grid, 10, 10 + k, 12, 12 + k, BG); paint(self.w._grid, 10, 11 + k, 12, 13 + k, 9)
            self.w._steps_episode = k + 1
            self.w._relsense_step(prev, 1, False)
        # a new life derives the markers with him known
        reset = [row[:] for row in G]
        self.w._grid = [row[:] for row in reset]; self.w._steps_episode = 0
        self.w._relsense_step(reset, 1, False)
        self.assertEqual(self.w.rel_context(), ((14, 9), 1, 1))

    def test_errors_are_counted_not_raised(self):
        self.w._grid = None
        self.w._steps_episode = 3
        try:
            self.w._relsense_step(None, 1, False)
        except Exception:
            self.fail('must not raise')


if __name__ == '__main__':
    unittest.main()
