"""HE LEARNS WHICH ROW IS THE BUDGET, AND STOPS SCORING A TICK AS PROGRESS.

These games DRAW the per-level action budget on the board.  It was inside
`_board_hash`, so a step that moved NOTHING BUT THE BAR read as a changed
board -- and FELT scored it +1 competent.  Measured over 489,233 frames:
masking that one row takes his "the last step changed nothing" rate from
26.7% to 70.7% on cd82 level 1 and from 0.0% to 93.9% on vc33 level 1,
while the other 63 rows move it by nothing.

What is pinned here:
  * the gate absent is BYTE-IDENTICAL, and the gate is never read from
    disk by a test -- the sustain work shipped a test that called the
    real gate and silently inverted the moment the gate file appeared
  * the discovery rule names a BUDGET and refuses a SCORE, content that
    goes back, and any tie
  * a "none" is NEVER latched -- at n=400 the rule missed two of 36
    corpus cells, both by saying "none", never by naming a wrong row
  * the learning tables are bounded by ONE life
  * a bar-only tick reads as NO CHANGE, which is the whole point
"""
import unittest

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld

NOTF = 'NOT_FINISHED'
_BAR = 63


def _grid(fill=0):
    return [[fill] * 64 for _ in range(64)]


class _Stub(object):
    """A mirror of ARCWorld, and it must stay one.

    `_judge_step` swallows exceptions, so a stub missing a method fails
    with a wrong VALUE instead of an error.  Every method the code under
    test reaches has to be here.
    """

    def __init__(self, game='vgame'):
        self.game_id = game
        self._levels = 0
        self._state = NOTF
        self._grid = _grid()

    _board_hash = ARCWorld._board_hash
    _board_hash_nb = ARCWorld._board_hash_nb
    _bar_of = ARCWorld._bar_of
    _bar_observe = ARCWorld._bar_observe


def _clear():
    for d in (ARCWorld._bar_row, ARCWorld._bar_stat, ARCWorld._bar_seen,
              ARCWorld._bar_prev, ARCWorld._bar_lv, ARCWorld._bar_cand):
        d.clear()
    ARCWorld._bar_learn_n = 0
    ARCWorld._bar_decided = 0


class _GateOn(unittest.TestCase):
    """Never reads /root/BARMASK_ON.  Ever."""

    def setUp(self):
        self._g = arc_world._BARMASK_ON
        arc_world._BARMASK_ON = lambda: True
        _clear()

    def tearDown(self):
        arc_world._BARMASK_ON = self._g
        _clear()


class TestGateOff(unittest.TestCase):

    def setUp(self):
        self._g = arc_world._BARMASK_ON
        arc_world._BARMASK_ON = lambda: False
        _clear()

    def tearDown(self):
        arc_world._BARMASK_ON = self._g
        _clear()

    def test_masked_hash_is_the_plain_hash_even_with_a_row_learned(self):
        w = _Stub()
        ARCWorld._bar_row[('vgame', 0)] = _BAR      # as if already learned
        w._grid[_BAR][7] = 3
        self.assertEqual(w._board_hash_nb(), w._board_hash(),
                         'gate off must be byte-identical')

    def test_a_bar_only_tick_still_reads_as_a_changed_board(self):
        w = _Stub()
        ARCWorld._bar_row[('vgame', 0)] = _BAR
        pre = w._board_hash_nb()
        w._grid[_BAR][7] = 3
        self.assertNotEqual(w._board_hash_nb(), pre,
                            'gate off must not mask anything')

    def test_observe_records_nothing(self):
        w = _Stub()
        for i in range(10):
            w._grid[_BAR][i] = 1
            w._bar_observe()
        self.assertEqual(ARCWorld._bar_stat, {})
        self.assertEqual(ARCWorld._bar_prev, {})
        self.assertEqual(ARCWorld._bar_learn_n, 0)


class TestMaskedHash(_GateOn):

    def test_the_learned_row_is_ignored(self):
        w = _Stub()
        ARCWorld._bar_row[('vgame', 0)] = _BAR
        a_nb, a_full = w._board_hash_nb(), w._board_hash()
        w._grid[_BAR][11] = 4
        self.assertEqual(w._board_hash_nb(), a_nb,
                         'the budget row must not change his state')
        self.assertNotEqual(w._board_hash(), a_full,
                            'the plain hash must still see it')

    def test_a_column_bar_is_ignored_too(self):
        w = _Stub()
        ARCWorld._bar_row[('vgame', 0)] = 64 + 7        # column 7
        before = w._board_hash_nb()
        for r in range(64):
            w._grid[r][7] = 9
        self.assertEqual(w._board_hash_nb(), before,
                         'the budget column must not change his state')
        self.assertNotEqual(w._board_hash(), before)

    def test_a_column_mask_still_sees_its_neighbour(self):
        w = _Stub()
        ARCWorld._bar_row[('vgame', 0)] = 64 + 7
        before = w._board_hash_nb()
        for r in range(64):
            w._grid[r][8] = 9
        self.assertNotEqual(w._board_hash_nb(), before)

    def test_every_other_row_is_still_seen(self):
        w = _Stub()
        ARCWorld._bar_row[('vgame', 0)] = _BAR
        before = w._board_hash_nb()
        w._grid[_BAR - 1][11] = 4
        self.assertNotEqual(w._board_hash_nb(), before,
                            'masking one row must not blind the other 63')

    def test_an_unlearned_cell_falls_through_to_the_plain_hash(self):
        w = _Stub()
        self.assertEqual(w._bar_of(), -1)
        self.assertEqual(w._board_hash_nb(), w._board_hash())

    def test_the_row_is_per_level(self):
        w = _Stub()
        ARCWorld._bar_row[('vgame', 0)] = _BAR
        w._levels = 1                       # nothing learned here yet
        self.assertEqual(w._bar_of(), -1)
        self.assertEqual(w._board_hash_nb(), w._board_hash())


class TestDecide(unittest.TestCase):
    """The rule itself: the unique row that moves ALONE and never goes back."""

    # 0-63 are rows, 64-127 are columns -- the fixture must be the
    # shape the real caller builds, or it fails on the array bound
    # instead of on the behaviour.
    def _st(self, n, spec):
        L = 128
        tick, solo, back = [0] * L, [0] * L, [0] * L
        for r, (t, s, b) in spec.items():
            tick[r], solo[r], back[r] = t, s, b
        return [n, tick, solo, back]

    def test_names_a_column_budget(self):
        """r11l's bar is col 0 -- masking it takes his no-op rate there
        from 0.0% to 40.3%, and a row-only rule cannot see it."""
        st = self._st(1000, {64 + 0: (1000, 430, 0)})
        self.assertEqual(ARCWorld._bar_decide(st), 64,
                         'a column bar must be findable')

    def test_dominance_settles_a_row_column_overlap(self):
        """A row bar whose marker rests at one column credits that
        column too: vc33 is row 0 at solo 0.97 AND col 63 at 0.18.
        Uniqueness alone would refuse it; dominance takes the row."""
        st = self._st(1000, {0: (1000, 970, 0), 64 + 63: (1000, 180, 0)})
        self.assertEqual(ARCWorld._bar_decide(st), 0)

    def test_refuses_when_neither_candidate_dominates(self):
        st = self._st(1000, {0: (1000, 300, 0), 64 + 63: (1000, 250, 0)})
        self.assertEqual(ARCWorld._bar_decide(st), -1,
                         'two comparable candidates is not an answer')

    def test_names_a_budget(self):
        st = self._st(1000, {_BAR: (1000, 500, 0)})
        self.assertEqual(ARCWorld._bar_decide(st), _BAR)

    def test_refuses_a_score_that_never_moves_alone(self):
        # A score only moves when the board does, so solo is ~0.
        st = self._st(1000, {12: (900, 5, 0)})
        self.assertEqual(ARCWorld._bar_decide(st), -1,
                         'a score is not a budget')

    def test_refuses_content_that_goes_back(self):
        # Moves alone often, but returns to values it has left.
        st = self._st(1000, {12: (900, 400, 300)})
        self.assertEqual(ARCWorld._bar_decide(st), -1)

    def test_refuses_a_tie(self):
        st = self._st(1000, {12: (900, 400, 0), 40: (900, 400, 0)})
        self.assertEqual(ARCWorld._bar_decide(st), -1,
                         'two candidates is not an answer')

    def test_refuses_with_no_evidence(self):
        self.assertEqual(ARCWorld._bar_decide([0, [0] * 64, [0] * 64,
                                               [0] * 64]), -1)

    def test_the_threshold_sits_in_measured_empty_space(self):
        # Corpus: accepted cells run solo 0.23-0.97, rejects top out at
        # 0.13.  Both sides of that gap must land the right way.
        self.assertEqual(
            ARCWorld._bar_decide(self._st(1000, {9: (1000, 130, 0)})), -1)
        self.assertEqual(
            ARCWorld._bar_decide(self._st(1000, {9: (1000, 230, 0)})), 9)


class TestLearning(_GateOn):
    """Drive a synthetic game and watch him find the row himself."""

    def _life(self, w, steps=60, solo_every=2, bar=_BAR, wander=True):
        """Row `bar` advances one new cell per step and never goes back.

        Some steps ALSO move an avatar that revisits its own squares --
        so the avatar row moves often, never alone, and goes back.
        """
        w._state = NOTF
        w._grid = _grid()
        w._bar_observe()
        for i in range(steps):
            w._grid[bar][i % 64] = (i // 64) + 1        # monotone, no return
            if wander and i % solo_every == 0:
                for c in range(64):
                    w._grid[10][c] = 0
                w._grid[10][i % 5] = 7                  # 5 squares, revisited
            w._bar_observe()
        w._state = 'GAME_OVER'
        w._bar_observe()

    def test_he_learns_the_budget_row(self):
        w = _Stub()
        for _ in range(12):
            self._life(w)
        self.assertEqual(ARCWorld._bar_row.get(('vgame', 0)), _BAR,
                         'he did not find the row: %r'
                         % (ARCWorld._bar_row,))
        self.assertEqual(ARCWorld._bar_decided, 1)

    def test_learning_stops_once_decided(self):
        w = _Stub()
        for _ in range(12):
            self._life(w)
        self.assertNotIn(('vgame', 0), ARCWorld._bar_stat,
                         'tables must be freed once decided')
        self.assertNotIn(('vgame', 0), ARCWorld._bar_seen)
        n = ARCWorld._bar_learn_n
        self._life(w)
        self.assertEqual(ARCWorld._bar_learn_n, n,
                         'a decided cell must cost nothing per step')

    def test_it_takes_a_repeat_to_latch(self):
        # One decision point is not enough; the same answer twice is.
        w = _Stub()
        self._life(w, steps=150)
        self.assertNotIn(('vgame', 0), ARCWorld._bar_row,
                         'latched on a single decision point')

    def test_a_none_is_never_latched(self):
        """No bar in this game -- he must keep looking, not conclude."""
        w = _Stub('nobar')
        for _ in range(20):
            w._state = NOTF
            w._grid = _grid()
            w._bar_observe()
            for i in range(60):
                for c in range(64):
                    w._grid[10][c] = 0
                w._grid[10][i % 5] = 7          # revisits; nothing monotone
                w._bar_observe()
            w._state = 'GAME_OVER'
            w._bar_observe()
        self.assertGreater(ARCWorld._bar_learn_n, 400)
        self.assertNotIn(('nobar', 0), ARCWorld._bar_row,
                         'a "none" must never be latched -- at n=400 the '
                         'rule missed 2 of 36 corpus cells, both by '
                         'saying none')

    def test_the_learning_tables_are_bounded_by_one_life(self):
        w = _Stub()
        self._life(w, steps=60)
        self.assertNotIn(('vgame', 0), ARCWorld._bar_seen,
                         'the per-life value sets must be dropped at a '
                         'terminal, or memory grows with the corpus')

    def test_a_paused_bar_is_still_found(self):
        """cd82's bar ticks on only 64% of steps -- 65 values over a ~101
        step budget.  'A new value every step' rejected it; 'never goes
        back' finds it."""
        w = _Stub('paused')
        for _ in range(14):
            w._state = NOTF
            w._grid = _grid()
            w._bar_observe()
            v = 0
            for i in range(60):
                if i % 3:                       # ticks on 2 of every 3
                    v += 1
                    w._grid[_BAR][v % 64] = (v // 64) + 1
                elif i % 6 == 0:
                    w._grid[20][i % 7] = 5      # content, and it revisits
                w._bar_observe()
            w._state = 'GAME_OVER'
            w._bar_observe()
        self.assertEqual(ARCWorld._bar_row.get(('paused', 0)), _BAR)


class TestTheWholePoint(_GateOn):

    def test_a_bar_only_tick_reads_as_no_change(self):
        """The step FELT was scoring +1 competent for.

        `_post == _pre` under the masked hash is exactly FELT's no-op
        branch and the verdict's 'he has been here' test.
        """
        w = _Stub()
        ARCWorld._bar_row[('vgame', 0)] = _BAR
        w._grid[30][30] = 5
        pre_nb, pre_full = w._board_hash_nb(), w._board_hash()
        w._grid[_BAR][12] = 1                   # ONLY the budget ticks
        self.assertEqual(w._board_hash_nb(), pre_nb,
                         'a step that only spends budget must read as '
                         'no change')
        self.assertNotEqual(w._board_hash(), pre_full,
                            'and the unmasked hash is why it did not')

    def test_real_progress_still_reads_as_progress(self):
        w = _Stub()
        ARCWorld._bar_row[('vgame', 0)] = _BAR
        pre = w._board_hash_nb()
        w._grid[_BAR][12] = 1                   # budget ticks ...
        w._grid[30][30] = 5                     # ... and something happened
        self.assertNotEqual(w._board_hash_nb(), pre,
                            'masking must not hide a real change')


if __name__ == '__main__':
    unittest.main()
