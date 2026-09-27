"""Patch 43 CLOCKLINE: A CLOCK NEED NOT MOVE ALONE.

The solo rule (`_bar_decide`) latches a budget line only if it is the only
line on its axis that changed on >= 15% of steps.  On twelve cells the self
changes 2-5 lines on EVERY step, so the marker is never alone (solo 0.00)
and those cells never decide: the clock stays inside the state key, every
step is a new state, and the felt register reads +1 "competent" for a tick.
What those clocks still do is MOVE THEIR HISTOGRAM FORWARD: a fill or drain
changes the colour counts of its line on every tick and never reverses a
colour within a life; a self that moves leaves one cell and enters another,
so its line's counts do not change.  Measured 2026-09-13 on two corpora: the
rule here agrees with the solo rule on every cell it decides (18 + 7) and
names a monotone marker on 21 + 8 it cannot; the sp80 L0 self, in perfect
lockstep with the bar over 4-step lives, reads 0.00 forward.

Pinned here:
  * gate absent: `_bar_decide` decides exactly as before, no histogram or
    lockstep is counted, and a learned line is always one int; no test
    reads a gate file
  * a clock that never moves alone is found; content that goes back, or
    whose histogram does not move, or reverses, is not -- even in perfect
    lockstep with the bar
  * lines in lockstep are ONE structure (adjacent or mirrored); a group of
    more than 2 lines is content, dropped; the old solo floor breaks a tie
  * a structure is masked in FULL by the board key, CLICKHIT and PROGMASK,
    persisted as a list, read back only under the gate, shown by /status
"""
import contextlib
import io
import json
import unittest

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld, _BAR_LINES

NOTF = 'NOT_FINISHED'


def _grid(fill=0):
    return [[fill] * 64 for _ in range(64)]


class _Stub(object):
    """A mirror of ARCWorld -- every method the code under test reaches."""

    def __init__(self, game='vgame'):
        self.game_id = game
        self._levels = 0
        self._state = NOTF
        self._grid = _grid()

    _board_hash = ARCWorld._board_hash
    _board_hash_nb = ARCWorld._board_hash_nb
    _grid_changed_nb = ARCWorld._grid_changed_nb
    _bar_of = ARCWorld._bar_of
    _bar_lines = ARCWorld._bar_lines
    _bar_observe = ARCWorld._bar_observe
    bar_to_dict = ARCWorld.bar_to_dict
    bar_from_dict = ARCWorld.bar_from_dict
    enter = ARCWorld.enter


def _clear():
    for d in (ARCWorld._bar_row, ARCWorld._bar_stat, ARCWorld._bar_seen,
              ARCWorld._bar_prev, ARCWorld._bar_lv, ARCWorld._bar_cand,
              ARCWorld._bar_pflat, ARCWorld._bar_sign):
        d.clear()
    ARCWorld._bar_learn_n = 0
    ARCWorld._bar_decided = 0


class _Gates(unittest.TestCase):
    """Never reads a gate file."""
    CLOCK = True

    def setUp(self):
        self._saved = {}
        for name, val in (('_BARMASK_ON', True), ('_CLOCKLINE_ON', self.CLOCK),
                          ('_CLICKHIT_ON', True), ('_BARPERSIST_ON', True),
                          ('_PROGMASK_ON', True)):
            self._saved[name] = getattr(arc_world, name)
            setattr(arc_world, name, (lambda v=val: v))
        _clear()

    def tearDown(self):
        for name, fn in self._saved.items():
            setattr(arc_world, name, fn)
        _clear()


def _st(n, spec, co=None, meter=True):
    """spec: line -> (tick, solo, back) or (tick, solo, back, fwd);
    co: {(a, b): coticks}.  fwd defaults to tick (a meter)."""
    L = _BAR_LINES
    tick, solo, back, fwd = [0] * L, [0] * L, [0] * L, [0] * L
    for r, v in spec.items():
        tick[r], solo[r], back[r] = v[0], v[1], v[2]
        fwd[r] = v[3] if len(v) > 3 else v[0]
    if not meter:
        return [n, tick, solo, back]
    return [n, tick, solo, back, dict(co or {}), fwd]


def _self_rows(spec, co, rows, tick, bar=None, bar_tick=0):
    """A sprite spanning `rows`: never repeats, always together, histogram
    constant (fwd 0), in lockstep with `bar` on every one of its ticks."""
    for r in rows:
        spec[r] = (tick, 0, 0, 0)
        for r2 in rows:
            if r < r2:
                co[(r, r2)] = tick
        if bar is not None:
            co[(min(bar, r), max(bar, r))] = min(tick, bar_tick)


def _life_dc22(w, steps=60):
    """The dc22 shape: a 2-row self moves on EVERY step (and returns),
    row 63 gains one cell every second step and never goes back."""
    w._state = NOTF
    w._grid = _grid()
    w._bar_observe()
    for i in range(steps):
        for c in range(64):
            w._grid[30][c] = 0
            w._grid[31][c] = 0
        x = 10 + (i % 7)                        # revisits its squares
        w._grid[30][x] = 5
        w._grid[31][x] = 5
        if i % 2 == 0:
            w._grid[63][(i // 2) % 64] = 7        # monotone, never back
        w._bar_observe()
    w._state = 'GAME_OVER'
    w._bar_observe()


# ---------------------------------------------------------------- the rule
class TestClockDecide(unittest.TestCase):

    def test_a_clock_that_never_moves_alone_is_found(self):
        # dc22: row 63 ticks on 50% of steps, one cell, never back, solo 0
        st = _st(1000, {63: (500, 0, 0)})
        self.assertEqual(ARCWorld._bar_decide(st), -1,
                         'the solo rule cannot see it (that is the defect)')
        self.assertEqual(ARCWorld._clock_decide(st), 63)

    def test_agrees_with_the_solo_rule_where_it_decides(self):
        # cd82 L0: tick 0.65, solo 0.32
        st = _st(1000, {63: (650, 320, 0)})
        self.assertEqual(ARCWorld._clock_decide(st), ARCWorld._bar_decide(st))
        # vc33: row 0 at tick 1.0 AND col 63 at 0.18 -> dominance takes row 0
        st = _st(1000, {0: (1000, 970, 0), 64 + 63: (180, 180, 0)})
        self.assertEqual(ARCWorld._clock_decide(st), 0)
        self.assertEqual(ARCWorld._bar_decide(st), 0)

    def test_refuses_content_that_goes_back(self):
        st = _st(1000, {12: (900, 0, 300)})
        self.assertEqual(ARCWorld._clock_decide(st), -1)

    def test_refuses_a_line_whose_histogram_does_not_move(self):
        """The reviewer's case: a self that never repeats a value (4-step
        lives), ticks on 75% of steps, in PERFECT lockstep with the bar.
        No threshold on lockstep can refuse it; its histogram can."""
        spec = {}
        co = {}
        _self_rows(spec, co, (16, 17), 750)
        st = _st(1000, spec, co)
        self.assertEqual(ARCWorld._clock_decide(st), -1)
        # and with the bar present and in lockstep 1.000 with the sprite,
        # the bar alone is the answer (by dominance, no tie-break needed)
        spec = {0: (750, 0, 0)}
        co = {}
        _self_rows(spec, co, (16, 17), 750, bar=0, bar_tick=750)
        self.assertEqual(ARCWorld._clock_decide(_st(1000, spec, co)), 0)

    def test_refuses_a_line_that_reverses(self):
        # r11l L0 rows 21-25: histogram moves on every tick, reverses on 50%
        st = _st(1000, {21: (400, 0, 0, 200)})
        self.assertEqual(ARCWorld._clock_decide(st), -1)
        # bp35 col 18: reverses on 21% -> forward 0.79, below the floor
        st = _st(1000, {64 + 18: (270, 0, 0, 213)})
        self.assertEqual(ARCWorld._clock_decide(st), -1)

    def test_the_meter_floor_sits_in_measured_empty_space(self):
        # budget lines 0.97-1.00 forward, nearest content 0.79
        self.assertEqual(ARCWorld._clock_decide(_st(1000, {63: (1000, 0, 0, 970)})), 63)
        self.assertEqual(ARCWorld._clock_decide(_st(1000, {63: (1000, 0, 0, 790)})), -1)

    def test_refuses_a_line_below_the_tick_floor(self):
        st = _st(1000, {12: (140, 0, 0)})
        self.assertEqual(ARCWorld._clock_decide(st), -1)

    def test_refuses_with_no_evidence(self):
        self.assertEqual(ARCWorld._clock_decide(_st(0, {})), -1)
        self.assertEqual(ARCWorld._clock_decide(_st(1000, {})), -1)

    def test_no_meter_table_is_not_yet(self):
        # a 4-element st (gate flipped on mid-learning) must not raise,
        # and must not decide without meter evidence
        st = _st(1000, {63: (500, 0, 0)}, meter=False)
        self.assertEqual(ARCWorld._clock_decide(st), -1)

    def test_the_old_rule_ignores_the_new_tables(self):
        st = _st(1000, {63: (650, 320, 0)}, {(1, 2): 3})
        self.assertEqual(ARCWorld._bar_decide(st), 63)

    def test_adjacent_lines_in_lockstep_are_one_structure(self):
        # ls20: rows 61 and 62 tick on every step, together, never back
        st = _st(1000, {61: (1000, 0, 0), 62: (1000, 0, 0)}, {(61, 62): 1000})
        self.assertEqual(ARCWorld._clock_decide(st), (61, 62))

    def test_mirrored_rows_in_lockstep_are_one_structure(self):
        # m0r0: row 0 fills from the right, row 63 from the left, in step
        st = _st(1000, {0: (420, 0, 0), 63: (420, 0, 0)}, {(0, 63): 420})
        self.assertEqual(ARCWorld._clock_decide(st), (0, 63))

    def test_a_column_band_is_one_structure(self):
        # sc25: cols 62 and 63, 61% of steps, two cells each, together
        a, b = 64 + 62, 64 + 63
        st = _st(1000, {a: (610, 0, 0), b: (610, 0, 0)}, {(a, b): 610})
        self.assertEqual(ARCWorld._clock_decide(st), (a, b))

    def test_lockstep_below_the_threshold_stays_two_candidates(self):
        # two comparable meters, neither ever alone -> not an answer
        st = _st(1000, {0: (1000, 0, 0), 16: (750, 0, 0)}, {(0, 16): 750})
        self.assertEqual(ARCWorld._clock_decide(st), -1)
        st = _st(1000, {0: (1000, 0, 0), 16: (1000, 0, 0)}, {(0, 16): 1000})
        self.assertEqual(ARCWorld._clock_decide(st), (0, 16))

    def test_a_lockstep_group_larger_than_two_lines_is_content(self):
        """Every measured structure is 2 lines; the 4-55-line lockstep
        groups are content.  Even a 3-row meter-like group that out-ticks
        the clock 2x is dropped, and the clock decides."""
        spec = {63: (500, 0, 0)}
        co = {}
        for r in (20, 21, 22):
            spec[r] = (1000, 0, 0)
            for r2 in (20, 21, 22):
                if r < r2:
                    co[(r, r2)] = 1000
            co[(r, 63)] = 500
        self.assertEqual(ARCWorld._clock_decide(_st(1000, spec, co)), 63)

    def test_only_content_groups_is_not_an_answer(self):
        spec = {}
        co = {}
        for r in (20, 21, 22):
            spec[r] = (1000, 0, 0)
            for r2 in (20, 21, 22):
                if r < r2:
                    co[(r, r2)] = 1000
        self.assertEqual(ARCWorld._clock_decide(_st(1000, spec, co)), -1)

    def test_solo_breaks_a_tie_between_two_meters(self):
        # no dominance; only row 0 passes the old solo floor
        st = _st(1000, {0: (1000, 250, 0), 40: (750, 0, 0)}, {(0, 40): 100})
        self.assertEqual(ARCWorld._clock_decide(st), 0)

    def test_one_stray_solo_event_is_not_a_tie_break(self):
        # measured: 12 of 125 content candidates carry >= 1 solo event,
        # only 3 reach the 15% floor -- one event must not decide
        st = _st(1000, {0: (1000, 1, 0), 40: (750, 0, 0)}, {(0, 40): 100})
        self.assertEqual(ARCWorld._clock_decide(st), -1)

    def test_two_near_ties_that_both_pass_solo_is_not_an_answer(self):
        st = _st(1000, {0: (1000, 300, 0), 40: (800, 300, 0)}, {(0, 40): 100})
        self.assertEqual(ARCWorld._clock_decide(st), -1)

    def test_dominance_needs_no_tie_break(self):
        # bp35: row 63 at 1.00 vs col 12 at 0.31 (a meter too), neither alone
        st = _st(1000, {63: (1000, 0, 0), 64 + 12: (310, 0, 0)}, {(63, 76): 310})
        self.assertEqual(ARCWorld._clock_decide(st), 63)


# ----------------------------------------------------------- gate off
class TestGateOff(_Gates):
    CLOCK = False

    def test_the_solo_rule_still_cannot_see_a_clock_that_never_moves_alone(self):
        w = _Stub()
        for _ in range(14):
            _life_dc22(w)
        self.assertGreater(ARCWorld._bar_learn_n, 400)
        self.assertNotIn(('vgame', 0), ARCWorld._bar_row)

    def test_nothing_new_is_counted_or_kept(self):
        w = _Stub()
        _life_dc22(w, steps=30)
        st = ARCWorld._bar_stat.get(('vgame', 0))
        self.assertIsNotNone(st)
        self.assertEqual(len(st), 4, 'gate off must not grow the table')
        self.assertEqual(ARCWorld._bar_pflat, {})
        self.assertEqual(ARCWorld._bar_sign, {})

    def test_a_persisted_structure_is_not_read_back(self):
        w = _Stub('cd82-x')
        n = w.bar_from_dict({'cd82-x': {'0': [61, 62], '1': 63}})
        self.assertEqual(n, 1)
        self.assertEqual(ARCWorld._bar_row, {('cd82-x', 1): 63})

    def test_bar_lines_and_bar_of_on_an_int(self):
        w = _Stub()
        ARCWorld._bar_row[('vgame', 0)] = 63
        self.assertEqual(w._bar_of(), 63)
        self.assertEqual(w._bar_lines(), (63,))
        del ARCWorld._bar_row[('vgame', 0)]
        self.assertEqual(w._bar_of(), -1)
        self.assertEqual(w._bar_lines(), ())


# ------------------------------------------------------------ learning
class TestLearning(_Gates):

    def test_he_learns_a_clock_that_never_moves_alone(self):
        w = _Stub()
        for _ in range(14):
            _life_dc22(w)
        self.assertEqual(ARCWorld._bar_row.get(('vgame', 0)), 63,
                         'not found: %r' % (ARCWorld._bar_row,))
        self.assertEqual(ARCWorld._bar_decided, 1)
        for d in (ARCWorld._bar_stat, ARCWorld._bar_seen, ARCWorld._bar_prev,
                  ARCWorld._bar_pflat, ARCWorld._bar_sign):
            self.assertNotIn(('vgame', 0), d, 'tables must be freed once decided')

    def test_the_meter_is_measured_from_frames(self):
        w = _Stub()
        _life_dc22(w, steps=59)
        st = ARCWorld._bar_stat[('vgame', 0)]
        self.assertEqual(len(st), 6)
        self.assertEqual(st[5][63], st[1][63], 'every clock tick moves forward')
        # its first appearance in a life is a fill; every move after is not
        self.assertLessEqual(st[5][30], 1, 'the self moves; its histogram does not')
        self.assertGreater(st[1][30], 40)

    def test_he_learns_a_mirrored_pair(self):
        w = _Stub('mirror')
        for _ in range(14):
            w._state = NOTF
            w._grid = _grid()
            w._bar_observe()
            for i in range(60):
                for c in range(64):
                    w._grid[30][c] = 0
                    w._grid[31][c] = 0
                x = 10 + (i % 7)
                w._grid[30][x] = 5
                w._grid[31][x] = 5
                if i % 2 == 0:
                    w._grid[0][63 - (i // 2)] = 7
                    w._grid[63][i // 2] = 7
                w._bar_observe()
            w._state = 'GAME_OVER'
            w._bar_observe()
        self.assertEqual(ARCWorld._bar_row.get(('mirror', 0)), (0, 63))

    def test_a_self_in_lockstep_with_the_bar_over_short_lives_is_not_masked(self):
        """sp80 L0 as the daemon sees it: 4-step lives, the bar ticks on
        every step, a 2-row self moves 4 columns on every step -- never
        repeats a value, lockstep 1.000 with the bar."""
        w = _Stub('sp80')
        for life in range(120):
            w._state = NOTF
            w._grid = _grid(9)
            for c in range(8):
                w._grid[16][c] = 12
                w._grid[17][c] = 12
            w._bar_observe()
            for i in range(4):
                w._grid[0][63 - 2 * i] = 0
                w._grid[0][62 - 2 * i] = 0
                for c in range(64):
                    w._grid[16][c] = 9
                    w._grid[17][c] = 9
                for c in range(4 * (i + 1), 4 * (i + 1) + 8):
                    w._grid[16][c] = 12
                    w._grid[17][c] = 12
                w._bar_observe()
            w._state = 'GAME_OVER'
            w._bar_observe()
        self.assertEqual(ARCWorld._bar_row.get(('sp80', 0)), 0,
                         'got %r' % (ARCWorld._bar_row,))

    def test_a_none_is_never_latched(self):
        w = _Stub('nobar')
        for _ in range(20):
            w._state = NOTF
            w._grid = _grid()
            w._bar_observe()
            for i in range(60):
                for c in range(64):
                    w._grid[10][c] = 0
                w._grid[10][i % 5] = 7
                w._bar_observe()
            w._state = 'GAME_OVER'
            w._bar_observe()
        self.assertGreater(ARCWorld._bar_learn_n, 400)
        self.assertNotIn(('nobar', 0), ARCWorld._bar_row)

    def test_it_takes_a_repeat_to_latch(self):
        w = _Stub()
        _life_dc22(w, steps=210)
        self.assertNotIn(('vgame', 0), ARCWorld._bar_row)

    def test_the_per_life_tables_are_dropped_at_a_terminal(self):
        w = _Stub()
        _life_dc22(w, steps=30)
        self.assertNotIn(('vgame', 0), ARCWorld._bar_sign)
        self.assertNotIn(('vgame', 0), ARCWorld._bar_pflat)


# ----------------------------------------------------------- masking
class TestMaskTheWholeStructure(_Gates):

    def setUp(self):
        super(TestMaskTheWholeStructure, self).setUp()
        self.w = _Stub()
        ARCWorld._bar_row[('vgame', 0)] = (61, 62)

    def test_readers(self):
        self.assertEqual(self.w._bar_lines(), (61, 62))
        self.assertEqual(self.w._bar_of(), 61)

    def test_board_key_ignores_every_line_of_the_structure(self):
        h0 = self.w._board_hash_nb()
        self.w._grid[61][5] = 9
        self.assertEqual(self.w._board_hash_nb(), h0)
        self.w._grid[62][40] = 9
        self.assertEqual(self.w._board_hash_nb(), h0)
        self.assertNotEqual(self.w._board_hash(), h0, 'the plain hash still sees it')
        self.w._grid[20][3] = 9
        self.assertNotEqual(self.w._board_hash_nb(), h0, 'content must still be seen')

    def test_a_column_structure_is_masked_too(self):
        ARCWorld._bar_row[('vgame', 0)] = (64 + 62, 64 + 63)
        h0 = self.w._board_hash_nb()
        self.w._grid[7][62] = 9
        self.w._grid[9][63] = 9
        self.assertEqual(self.w._board_hash_nb(), h0)
        self.w._grid[7][61] = 9
        self.assertNotEqual(self.w._board_hash_nb(), h0)

    def test_clickhit_sees_no_change_when_only_the_structure_ticks(self):
        prev = [row[:] for row in self.w._grid]
        self.w._grid[61][5] = 9
        self.w._grid[62][5] = 9
        self.assertFalse(self.w._grid_changed_nb(prev))
        self.w._grid[30][5] = 9
        self.assertTrue(self.w._grid_changed_nb(prev))

    def test_clickhit_on_a_column_structure(self):
        ARCWorld._bar_row[('vgame', 0)] = (64 + 62, 64 + 63)
        prev = [row[:] for row in self.w._grid]
        self.w._grid[7][62] = 9
        self.w._grid[9][63] = 9
        self.assertFalse(self.w._grid_changed_nb(prev))
        self.w._grid[9][61] = 9
        self.assertTrue(self.w._grid_changed_nb(prev))

    def test_a_single_line_is_masked_exactly_as_before(self):
        ARCWorld._bar_row[('vgame', 0)] = 63
        h0 = self.w._board_hash_nb()
        self.w._grid[63][5] = 9
        self.assertEqual(self.w._board_hash_nb(), h0)
        self.w._grid[62][5] = 9
        self.assertNotEqual(self.w._board_hash_nb(), h0)


# -------------------------------------------------------- persistence
class TestPersist(_Gates):

    def test_a_structure_round_trips_as_a_list(self):
        w = _Stub('ls20-x')
        ARCWorld._bar_row[('ls20-x', 0)] = (61, 62)
        ARCWorld._bar_row[('ls20-x', 1)] = 63
        ARCWorld._bar_row[('ls20-x', 2)] = -1
        d = w.bar_to_dict()
        self.assertEqual(d, {'ls20-x': {'0': [61, 62], '1': 63}})
        d = json.loads(json.dumps(d))
        _clear()
        n = w.bar_from_dict(d)
        self.assertEqual(n, 2)
        self.assertEqual(ARCWorld._bar_row,
                         {('ls20-x', 0): (61, 62), ('ls20-x', 1): 63})
        self.assertEqual(ARCWorld._bar_decided, 2)

    def test_malformed_structures_are_ignored(self):
        w = _Stub('g')
        n = w.bar_from_dict({'g': {'0': [61, 'x'], '1': [200, 3],
                                   '2': [5], '3': 'no'}})
        self.assertEqual(n, 0)
        self.assertEqual(ARCWorld._bar_row, {})

    def test_status_renders_a_structure(self):
        # the /status expression, verbatim shape
        ARCWorld._bar_row[('g', 0)] = (61, 62)
        ARCWorld._bar_row[('g', 1)] = 63
        rows = dict(('%s|%s' % (a, b),
                     list(v) if isinstance(v, (tuple, list)) else int(v))
                    for (a, b), v in ARCWorld._bar_row.items())
        self.assertEqual(rows, {'g|0': [61, 62], 'g|1': 63})


# ------------------------------------------------- a re-entry is a life boundary
def _life_with_a_return(w, steps=60, leave_at=20):
    """The dc22 shape, but he rotates away mid-life and comes back: the
    board RESETS (the bar refills to the values of the life before)."""
    w._state = NOTF
    w._grid = _grid()
    w._bar_observe()
    for i in range(steps):
        if i == leave_at:
            w.enter()                       # rotated away and back
            w._grid = _grid()               # the reset board
            w._bar_observe()
        j = i if i < leave_at else i - leave_at
        for c in range(64):
            w._grid[30][c] = 0
            w._grid[31][c] = 0
        x = 10 + (i % 7)
        w._grid[30][x] = 5
        w._grid[31][x] = 5
        if j % 2 == 0:
            w._grid[63][(j // 2) % 64] = 7
        w._bar_observe()
    w._state = 'GAME_OVER'
    w._bar_observe()


class TestReentryIsALifeBoundary(_Gates):

    def test_he_still_decides_when_he_leaves_and_returns_mid_life(self):
        w = _Stub()
        for _ in range(14):
            _life_with_a_return(w)
        self.assertEqual(ARCWorld._bar_row.get(('vgame', 0)), 63,
                         'the refilled bar read as back: %r' % (ARCWorld._bar_row,))

    def test_enter_drops_the_per_life_tables(self):
        w = _Stub()
        _life_dc22(w, steps=30)
        w._state = NOTF
        w._grid = _grid()
        w._bar_observe()
        for i in range(10):
            w._grid[63][i] = 7
            w._bar_observe()
        self.assertIn(('vgame', 0), ARCWorld._bar_seen)
        w.enter()
        self.assertNotIn('vgame', ARCWorld._bar_lv)
        w._grid = _grid()
        w._bar_observe()                    # the reset frame starts the new life
        self.assertEqual(ARCWorld._bar_lv.get('vgame'), 0)
        seen = ARCWorld._bar_seen.get(('vgame', 0))
        self.assertTrue(all(len(s) <= 1 for s in seen), 'the previous life leaked')


class TestReentryGateOff(_Gates):
    CLOCK = False

    def test_enter_leaves_the_learner_alone(self):
        w = _Stub()
        _life_dc22(w, steps=30)
        w._state = NOTF
        w._grid = _grid()
        w._bar_observe()
        w.enter()
        self.assertEqual(ARCWorld._bar_lv.get('vgame'), 0,
                         'gate off must not touch _bar_lv')


# ------------------------------------------------------- the journal line
class TestJournal(_Gates):

    def test_a_decision_is_written_to_stderr(self):
        w = _Stub()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            for _ in range(14):
                _life_dc22(w)
        self.assertEqual(ARCWorld._bar_row.get(('vgame', 0)), 63)
        self.assertIn('[clockline] DECIDED game=vgame lv=0 lines=63 n=',
                      buf.getvalue())

    def test_gate_off_writes_nothing(self):
        saved = arc_world._CLOCKLINE_ON
        arc_world._CLOCKLINE_ON = lambda: False
        try:
            w = _Stub('quiet')
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                # a solo bar, decided by the old rule
                for _ in range(12):
                    w._state = NOTF
                    w._grid = _grid()
                    w._bar_observe()
                    for i in range(60):
                        w._grid[63][i % 64] = (i // 64) + 1
                        if i % 2 == 0:
                            for c in range(64):
                                w._grid[10][c] = 0
                            w._grid[10][i % 5] = 7
                        w._bar_observe()
                    w._state = 'GAME_OVER'
                    w._bar_observe()
            self.assertEqual(ARCWorld._bar_row.get(('quiet', 0)), 63)
            self.assertEqual(buf.getvalue(), '')
        finally:
            arc_world._CLOCKLINE_ON = saved


if __name__ == '__main__':
    unittest.main()
