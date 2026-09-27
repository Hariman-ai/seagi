"""A DECIDED LEVEL MUST NOT FREEZE THE LIFE-BOUNDARY BOOKKEEPING.

Found by adversarial review of BARPERSIST (2026-09-05), but the defect
predates it: `_bar_observe` returned on a decided (game, level) BEFORE
updating `_bar_lv` and popping the per-life tables.  Trace: L0 is being
learned (`_bar_lv[g]` = 0); L1 is decided; every L1 frame returns early
and `_bar_lv[g]` stays 0; he rotates back to L0, `_bar_lv[g] == lv`, so
the L0 per-life `seen` set is NOT reset, the replayed life counts as
`back`, and L0 can never decide.  With persisted lines every cleared
game's L0 would be stuck this way from step 1.

Pinned here:
  * visiting a decided level updates `_bar_lv` and does not learn
  * returning to an undecided level after a decided one resets that
    level's per-life tables (the old order left them in place)
  * a terminal on a decided level still returns without learning
"""
import unittest

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld

NOTF = "NOT_FINISHED"


def grid():
    return [[0] * 64 for _ in range(64)]


class _Stub(object):
    def __init__(self, game='vgame'):
        self.game_id = game
        self._levels = 0
        self._state = NOTF
        self._grid = grid()

    _bar_of = ARCWorld._bar_of
    _bar_observe = ARCWorld._bar_observe


def _clear():
    for d in (ARCWorld._bar_row, ARCWorld._bar_stat, ARCWorld._bar_seen,
              ARCWorld._bar_prev, ARCWorld._bar_lv, ARCWorld._bar_cand):
        d.clear()
    ARCWorld._bar_learn_n = 0
    ARCWorld._bar_decided = 0


class TestDecidedLevelBookkeeping(unittest.TestCase):

    def setUp(self):
        self._b = arc_world._BARMASK_ON
        arc_world._BARMASK_ON = lambda: True
        _clear()
        self.w = _Stub()

    def tearDown(self):
        arc_world._BARMASK_ON = self._b
        _clear()

    def test_a_decided_level_updates_the_level_marker_and_does_not_learn(self):
        ARCWorld._bar_row[('vgame', 1)] = 63
        ARCWorld._bar_lv['vgame'] = 0
        self.w._levels = 1
        n0 = ARCWorld._bar_learn_n
        self.w._bar_observe()
        self.assertEqual(ARCWorld._bar_lv['vgame'], 1)
        self.assertEqual(ARCWorld._bar_learn_n, n0, 'decided: nothing to learn')
        self.assertNotIn(('vgame', 1), ARCWorld._bar_seen)

    def test_returning_to_an_undecided_level_resets_its_per_life_tables(self):
        # L0 mid-learning, with a sentinel per-life table
        sentinel = [set() for _ in range(arc_world._BAR_LINES)]
        ARCWorld._bar_seen[('vgame', 0)] = sentinel
        ARCWorld._bar_prev[('vgame', 0)] = [0] * arc_world._BAR_LINES
        ARCWorld._bar_lv['vgame'] = 0
        # cleared L1, which is decided (loaded or learned)
        ARCWorld._bar_row[('vgame', 1)] = 63
        self.w._levels = 1
        self.w._bar_observe()
        # rotate back to L0: a life boundary
        self.w._levels = 0
        self.w._bar_observe()
        self.assertIsNot(ARCWorld._bar_seen.get(('vgame', 0)), sentinel,
                         'the old order left the L0 tables in place, so the '
                         'replayed life read as "back" and L0 could never decide')

    def test_a_terminal_on_a_decided_level_still_returns_quietly(self):
        ARCWorld._bar_row[('vgame', 1)] = 63
        self.w._levels = 1
        self.w._state = "GAME_OVER"
        n0 = ARCWorld._bar_learn_n
        self.w._bar_observe()
        self.assertEqual(ARCWorld._bar_learn_n, n0)
        self.assertEqual(ARCWorld._bar_lv['vgame'], 1)


if __name__ == '__main__':
    unittest.main()
