"""THE BUDGET LINE IS NOT PROGRESS.

`_progress_step` credits a new high-water mark of cells that differ
from the board at level start.  The budget line differs from it on
nearly every step of the first life, so unmasked it (a) fires on the
clock, and (b) sets a mark the real content can never beat later.
Replayed on this process's frames: 977 fires, 72.0% the line, 0.0% in
the four games with no line.

Pinned here:
  * gate absent is BYTE-IDENTICAL (a line-only change still fires),
    and no test reads the gate file
  * gate on: a change confined to the line does NOT fire
  * a content change still fires, and only on a NEW high-water mark
  * columns as well as rows; an unlearned cell falls through to raw
  * the base board is stored unchanged (the mask is in the count only)
"""
import unittest

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld

BAR_ROW = 63
BAR_COL = 64 + 7


def grid():
    return [[0] * 64 for _ in range(64)]


class _Stub(object):
    def __init__(self, game='vgame'):
        self.game_id = game
        self._levels = 0
        self._grid = grid()

    _bar_of = ARCWorld._bar_of
    _progress_step = ARCWorld._progress_step


class _Base(unittest.TestCase):
    ON = True

    def setUp(self):
        self._p = arc_world._PROGRESS_ON
        self._m = arc_world._PROGMASK_ON
        self._b = arc_world._BARMASK_ON
        arc_world._PROGRESS_ON = lambda: True
        arc_world._PROGMASK_ON = lambda: self.ON
        arc_world._BARMASK_ON = lambda: True
        ARCWorld._bar_row.clear()
        ARCWorld._lvl_base.clear()
        ARCWorld._lvl_hwm.clear()
        self._n0 = ARCWorld._progress_n
        self.w = _Stub()
        self.w._prog_here = 0
        # first sight sets the base and never fires
        self.assertEqual(self.w._progress_step(), 0)

    def tearDown(self):
        arc_world._PROGRESS_ON = self._p
        arc_world._PROGMASK_ON = self._m
        arc_world._BARMASK_ON = self._b
        ARCWorld._bar_row.clear()
        ARCWorld._lvl_base.clear()
        ARCWorld._lvl_hwm.clear()

    def tick(self, k):
        for c in range(k):
            self.w._grid[BAR_ROW][c] = 5


class TestGateOff(_Base):
    ON = False

    def test_a_line_only_change_still_fires(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        self.tick(3)
        self.assertEqual(self.w._progress_step(), 3,
                         'gate off must be byte-identical: the line counts')

    def test_no_change_no_fire(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        self.assertEqual(self.w._progress_step(), 0)


class TestRowBar(_Base):

    def test_a_change_confined_to_the_line_is_not_progress(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        self.tick(10)
        self.assertEqual(self.w._progress_step(), 0,
                         'spending budget is not progress')
        self.assertEqual(ARCWorld._progress_n, self._n0)

    def test_content_still_fires_and_only_on_a_new_mark(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        self.tick(10)
        self.w._grid[10][10] = 7
        self.assertEqual(self.w._progress_step(), 1)
        # same content, more budget spent: no new mark
        self.tick(20)
        self.assertEqual(self.w._progress_step(), 0)
        # more content: fires by the increment
        self.w._grid[11][11] = 7
        self.w._grid[12][12] = 7
        self.assertEqual(self.w._progress_step(), 2)
        self.assertEqual(ARCWorld._progress_n, self._n0 + 2)

    def test_the_row_next_to_the_line_is_not_masked(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        self.w._grid[BAR_ROW - 1][3] = 5
        self.assertEqual(self.w._progress_step(), 1)

    def test_the_base_is_stored_unmasked(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        base = ARCWorld._lvl_base[('vgame', 0)]
        self.assertEqual(len(base), 64)
        self.assertEqual(len(base[BAR_ROW]), 64)


class TestColumnBar(_Base):

    def test_a_change_confined_to_the_column_is_not_progress(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_COL
        for r in range(64):
            self.w._grid[r][7] = 5
        self.assertEqual(self.w._progress_step(), 0,
                         "the r11l budget is a COLUMN")

    def test_the_column_next_to_it_still_counts(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_COL
        self.w._grid[10][8] = 5
        self.assertEqual(self.w._progress_step(), 1)


class TestFallThrough(_Base):

    def test_an_unlearned_cell_counts_everything(self):
        self.tick(4)
        self.assertEqual(self.w._progress_step(), 4,
                         'nothing is masked until he has learned a line')

    def test_the_line_is_per_level(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW
        self.w._levels = 1
        self.assertEqual(self.w._progress_step(), 0)   # base for level 1
        self.tick(4)
        self.assertEqual(self.w._progress_step(), 4)


if __name__ == '__main__':
    unittest.main()
