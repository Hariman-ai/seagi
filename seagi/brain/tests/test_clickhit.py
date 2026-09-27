"""A CLICK THAT ONLY SPENT BUDGET DID NOT WORK.

`_click_hits` is the ONLY thing choosing his aim in the six click-only
games: `propose` asks the verdict only when there are >= 2 action
INDICES and runs the felt steer only when there are > 1, and in r11l,
vc33, lp85, ft09, s5i5 and tn36 there is exactly one -- the click.  The
real choice there is a 4,096-way aim, and nothing else ranks it.

It records a hit on `self._grid != _prev_grid`, which the budget line
makes true when nothing happened.  Replaying both set constructions over
the corpus: as it ships, membership is ANTI-predictive of a real change
(mean -1.9 points, worse than non-membership in 8 of 17 cells); with the
line masked it is +6.9 points and positive in 16 of 17.

Pinned here:
  * gate absent is BYTE-IDENTICAL, and no test reads the gate file
  * a change confined to the budget line is NOT a change
  * a change anywhere else still is -- including the line's neighbour
  * columns as well as rows
  * an unlearned cell falls through to the raw comparison
"""
import unittest

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld

BAR_ROW = 63
BAR_COL = 64 + 7          # 64-127 are columns; this is column 7


def grid():
    return [[0] * 64 for _ in range(64)]


class _Stub(object):
    def __init__(self, game='vgame'):
        self.game_id = game
        self._levels = 0
        self._grid = grid()

    _bar_of = ARCWorld._bar_of
    _grid_changed_nb = ARCWorld._grid_changed_nb


class _Base(unittest.TestCase):
    ON = True

    def setUp(self):
        self._c = arc_world._CLICKHIT_ON
        self._b = arc_world._BARMASK_ON
        arc_world._CLICKHIT_ON = lambda: self.ON
        arc_world._BARMASK_ON = lambda: True
        ARCWorld._bar_row.clear()
        self.w = _Stub()

    def tearDown(self):
        arc_world._CLICKHIT_ON = self._c
        arc_world._BARMASK_ON = self._b
        ARCWorld._bar_row.clear()


class TestGateOff(_Base):
    ON = False

    def test_a_bar_only_change_still_counts(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        prev = [r[:] for r in self.w._grid]
        self.w._grid[BAR_ROW][3] = 1
        self.assertTrue(self.w._grid_changed_nb(prev),
                        'gate off must be byte-identical to the raw compare')

    def test_no_change_is_still_no_change(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        prev = [r[:] for r in self.w._grid]
        self.assertFalse(self.w._grid_changed_nb(prev))


class TestRowBar(_Base):

    def test_a_change_confined_to_the_budget_row_is_not_a_change(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        prev = [r[:] for r in self.w._grid]
        for c in range(64):
            self.w._grid[BAR_ROW][c] = 5
        self.assertFalse(self.w._grid_changed_nb(prev),
                         'spending budget is not doing something')

    def test_a_change_anywhere_else_still_is(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        prev = [r[:] for r in self.w._grid]
        self.w._grid[BAR_ROW][3] = 5          # budget ticked ...
        self.w._grid[10][10] = 7              # ... and something happened
        self.assertTrue(self.w._grid_changed_nb(prev))

    def test_the_row_next_to_the_budget_is_not_masked(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        prev = [r[:] for r in self.w._grid]
        self.w._grid[BAR_ROW - 1][3] = 5
        self.assertTrue(self.w._grid_changed_nb(prev))


class TestColumnBar(_Base):

    def test_a_change_confined_to_the_budget_column_is_not_a_change(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_COL
        prev = [r[:] for r in self.w._grid]
        for r in range(64):
            self.w._grid[r][7] = 5
        self.assertFalse(self.w._grid_changed_nb(prev),
                         'r11l\'s budget is a COLUMN')

    def test_the_column_next_to_the_budget_is_not_masked(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_COL
        prev = [r[:] for r in self.w._grid]
        self.w._grid[10][8] = 5
        self.assertTrue(self.w._grid_changed_nb(prev))


class TestFallThrough(_Base):

    def test_an_unlearned_cell_uses_the_raw_comparison(self):
        prev = [r[:] for r in self.w._grid]
        self.w._grid[BAR_ROW][3] = 1
        self.assertTrue(self.w._grid_changed_nb(prev),
                        'nothing is masked until he has learned a line')

    def test_the_line_is_per_level(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        self.w._levels = 1
        prev = [r[:] for r in self.w._grid]
        self.w._grid[BAR_ROW][3] = 1
        self.assertTrue(self.w._grid_changed_nb(prev))

    def test_no_previous_grid_is_not_a_crash(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        self.assertTrue(self.w._grid_changed_nb(None))


if __name__ == '__main__':
    unittest.main()
