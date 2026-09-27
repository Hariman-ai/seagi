"""I ALREADY PUSHED IT. LET ME PULL IT.

MEASURED 2026-08-27, 46,785 occasions across all 25 games: standing on a
board where an untried action existed, he repeated one he had already
tried HERE 67.2% of the time, against 36.1% for a chooser with no memory
at all -- 31 points WORSE than random, in every game.

These pin the two properties that keep the correction safe:
  * it fires only on a REVISIT (a board where he has tried something)
  * it never empties, so he always has a move
"""
import unittest

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld


class _W(object):
    game_id = 'ug'

    def __init__(self):
        self._grid = [[0] * 64 for _ in range(64)]

    def board(self, tag):
        self._grid[0][0] = tag
        return self._board_hash()

    _board_hash = ARCWorld._board_hash
    _board_hash_nb = ARCWorld._board_hash_nb
    _bar_of = ARCWorld._bar_of
    # BOARDKEY (2026-09-05): the organ reads the board through this;
    # a stub must carry what the real class carries.
    _board_key = ARCWorld._board_key
    untried_here = ARCWorld.untried_here


class TestUntriedHere(unittest.TestCase):

    def setUp(self):
        self._g = arc_world._UNTRIEDHERE_ON
        arc_world._UNTRIEDHERE_ON = lambda: True
        self._saved = dict(ARCWorld._board_act)
        ARCWorld._board_act.clear()
        self.w = _W()

    def tearDown(self):
        arc_world._UNTRIEDHERE_ON = self._g
        ARCWorld._board_act.clear()
        ARCWorld._board_act.update(self._saved)

    def _tried(self, tag, action):
        ARCWorld._board_act[('ug', self.w.board(tag), action)] = [1, 0]

    def test_a_fresh_board_is_not_corrected(self):
        self.w.board(1)
        self.assertEqual(self.w.untried_here(4), [],
                         'a board he has never acted on is not a revisit')

    def test_a_revisit_offers_what_he_has_not_tried(self):
        self._tried(1, 0)
        self.w.board(1)
        self.assertEqual(self.w.untried_here(4), [1, 2, 3])

    def test_an_exhausted_board_offers_nothing(self):
        for a in range(4):
            self._tried(1, a)
        self.w.board(1)
        self.assertEqual(self.w.untried_here(4), [],
                         'he must keep a move when everything is tried')

    def test_boards_are_judged_separately(self):
        self._tried(1, 0)
        self._tried(2, 3)
        self.w.board(1)
        self.assertEqual(self.w.untried_here(4), [1, 2, 3])
        self.w.board(2)
        self.assertEqual(self.w.untried_here(4), [0, 1, 2])

    def test_gate_off_is_silent(self):
        self._tried(1, 0)
        arc_world._UNTRIEDHERE_ON = lambda: False
        self.w.board(1)
        self.assertEqual(self.w.untried_here(4), [])


if __name__ == '__main__':
    unittest.main()
