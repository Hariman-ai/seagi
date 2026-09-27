"""PATCH 15 (2026-09-07) -- LEVELHOLD: a life is judged on ITS LEVEL.

Pinned here:
  * a fresh level: the first life is a record (depth), then one retry is
    granted, then the drought ejects (dry 2 > max(1, 0))
  * the bound self-scales: after coming back from a 3-life drought he is
    held through 3 dry lives and ejected at the 4th
  * a win, a new depth mark and a better picture are each a record; an
    equal picture is not
  * knowledge gained is logged, never a stay reason
  * gate off: nothing is written and level_hold() is False
  * the curriculum consults level_hold() instead of the win-count term,
    and still ejects when it is False (no cage); with the gate off the
    old term still holds
  * depleted() rule (e) fires past the bound and names itself
"""
import unittest
from unittest import mock

from seagi.world import arc_world
from seagi.world import curriculum_world as _cw_mod
from seagi.world.arc_world import ARCWorld
from seagi.world.curriculum_world import CurriculumWorld
from seagi.world.relsense import RelSense
from seagi.brain.tests.test_curriculum_world import _ArcLike

A = (34, 27, 43, 36); B = (3, 3, 12, 12)


def _grid(mm):
    """A 64x64 board whose canvas differs from the target in `mm` cells."""
    g = [[5] * 64 for _ in range(64)]
    k = 0
    for r in range(10):
        for c in range(10):
            g[B[0] + r][B[1] + c] = 1
            g[A[0] + r][A[1] + c] = 2 if k < mm else 1
            k += 1
    return g


class _W(object):
    game_id = 'vgame'

    def __init__(self, lv=1):
        self._levels = lv
        self._grid = _grid(50)
        self._won_here = 0
        self._life_key = None
        self._life_hwm0 = 0
        self._life_won0 = 0
        self._life_know0 = 0
        self._life_min_mm = None

    _live = True
    _rel_picture = staticmethod(ARCWorld._rel_picture)
    _level_forget_drought = ARCWorld._level_forget_drought
    _level_know = ARCWorld._level_know
    _level_life_begin = ARCWorld._level_life_begin
    _level_records_step = ARCWorld._level_records_step
    _level_life_end = ARCWorld._level_life_end
    level_hold = ARCWorld.level_hold


class _Base(unittest.TestCase):
    ON = True

    def setUp(self):
        self._g = arc_world._LEVELHOLD_ON
        arc_world._LEVELHOLD_ON = lambda: self.ON
        for d in (ARCWorld._lvl_dry, ARCWorld._lvl_gap_max, ARCWorld._lvl_best_mm,
                  ARCWorld._lvl_hwm, ARCWorld._rel, ARCWorld._relplan):
            d.clear()
        ARCWorld._lvl_records = 0; ARCWorld._lvl_dry_ejects = 0; ARCWorld._lvl_errors = 0
        self.w = _W()
        self.key = ('vgame', 1)

    def tearDown(self):
        arc_world._LEVELHOLD_ON = self._g
        for d in (ARCWorld._lvl_dry, ARCWorld._lvl_gap_max, ARCWorld._lvl_best_mm,
                  ARCWorld._lvl_hwm, ARCWorld._rel, ARCWorld._relplan):
            d.clear()

    def life(self, depth=None, mm=None, win=False):
        """One life: a step (opens the book), optional record signals, a terminal."""
        if mm is not None:
            self.w._grid = _grid(mm)
        self.w._level_records_step()
        if depth is not None:
            ARCWorld._lvl_hwm[self.key] = depth
        if win:
            self.w._won_here += 1
        self.w._level_life_end()
        return self.w.level_hold()


class TestRecords(_Base):
    def test_fresh_level_first_life_records_then_one_retry_then_eject(self):
        self.assertTrue(self.life(depth=3))            # record -> dry 0
        self.assertEqual(ARCWorld._lvl_dry[self.key], 0)
        self.assertTrue(self.life())                   # dry 1 <= max(1, 0): one retry
        self.assertFalse(self.life())                  # dry 2 > 1: leave
        self.assertEqual(ARCWorld._lvl_records, 1)
        self.assertEqual(ARCWorld._lvl_errors, 0)

    def test_the_bound_self_scales_from_his_own_droughts(self):
        self.life(depth=1)
        self.life(); self.life(); self.life()          # 3 dry lives
        self.assertTrue(self.life(depth=2))            # came back -> gap_max 3
        self.assertEqual(ARCWorld._lvl_gap_max[self.key], 3)
        for _ in range(3):
            self.assertTrue(self.life())               # held through 3 dry lives
        self.assertFalse(self.life())                  # the 4th ejects

    def test_a_win_is_a_record(self):
        self.life(depth=1); self.life(); self.life()   # dry 2, no hold
        self.assertFalse(self.w.level_hold())
        self.assertTrue(self.life(win=True))
        self.assertEqual(ARCWorld._lvl_dry[self.key], 0)
        self.assertEqual(ARCWorld._lvl_gap_max[self.key], 2)

    def test_a_better_picture_is_a_record_and_an_equal_one_is_not(self):
        rs = RelSense(); rs._pursued = ('eq', A, B)
        ARCWorld._rel[self.key] = rs
        self.assertTrue(self.life(mm=40))              # first picture mark
        self.assertEqual(ARCWorld._lvl_best_mm[self.key], 40)
        self.assertTrue(self.life(mm=40))              # equal: dry 1 (one retry)
        self.assertEqual(ARCWorld._lvl_dry[self.key], 1)
        self.assertTrue(self.life(mm=31))              # better: record, dry 0
        self.assertEqual(ARCWorld._lvl_best_mm[self.key], 31)
        self.assertEqual(ARCWorld._lvl_dry[self.key], 0)

    def test_gate_off_writes_nothing(self):
        self.ON = False
        self.life(depth=5)
        self.assertEqual(ARCWorld._lvl_dry, {})
        self.assertFalse(self.w.level_hold())

    def test_a_midlife_win_closes_one_book_and_opens_the_next(self):
        rs = RelSense(); rs._pursued = ('eq', A, B)
        ARCWorld._rel[('vgame', 1)] = rs
        self.w._level_records_step()                    # life opens on lv1
        self.w._levels = 2                              # he cleared it mid-life
        self.w._grid = _grid(0)
        self.w._level_records_step()                    # lv1 closed as a win, lv2 opened
        self.assertEqual(self.w._life_key, ('vgame', 2))
        self.assertEqual(ARCWorld._lvl_dry[('vgame', 1)], 0)   # win = record on lv1
        # lv1's own mark is the 50 it saw before the win; lv2's perfect
        # picture (0) must NOT have been scored on lv1
        self.assertEqual(ARCWorld._lvl_best_mm[('vgame', 1)], 50)
        self.assertTrue(self.w.level_hold())                    # lv2 fresh -> hold
        ARCWorld._lvl_hwm[('vgame', 2)] = 3             # the arrival stretch changed cells
        self.w._level_life_end()                        # terminal
        self.assertEqual(ARCWorld._lvl_dry[('vgame', 2)], 0)   # scored on ITS level
        self.assertEqual(ARCWorld._lvl_records, 2)

    def test_reentry_forgets_the_drought_and_keeps_the_bound(self):
        self.life(depth=1); self.life(); self.life(); self.life()   # dry 3
        self.assertFalse(self.w.level_hold())
        ARCWorld._lvl_gap_max[self.key] = 2
        self.w._level_forget_drought()                  # enter()
        self.assertNotIn(self.key, ARCWorld._lvl_dry)
        self.assertEqual(ARCWorld._lvl_gap_max[self.key], 2)
        self.assertTrue(self.w.level_hold())            # at least one retry on return

    def test_a_win_into_a_level_that_went_dry_last_visit_does_not_eject(self):
        ARCWorld._lvl_dry[('vgame', 2)] = 5             # went dry there on an earlier visit
        self.w._level_forget_drought()                  # enter()
        self.w._level_records_step()                    # life opens on lv1
        self.w._levels = 2                              # cleared it mid-life
        self.w._level_records_step()                    # arrival booked on lv2
        self.assertTrue(self.w.level_hold(), 'winning into a level ejected him')
        # even with a stale drought left in place the arrival book protects him
        ARCWorld._lvl_dry[('vgame', 2)] = 5
        ARCWorld._lvl_hwm[('vgame', 2)] = 3
        self.w._level_life_end()
        self.assertEqual(ARCWorld._lvl_dry[('vgame', 2)], 0)
        self.assertTrue(self.w.level_hold())

    def test_a_dead_world_is_not_held(self):
        self.w._live = False
        self.assertFalse(self.w.level_hold())


class TestDepletedRuleE(unittest.TestCase):
    def setUp(self):
        self._g = arc_world._LEVELHOLD_ON
        arc_world._LEVELHOLD_ON = lambda: True
        ARCWorld._lvl_dry.clear(); ARCWorld._lvl_gap_max.clear()

    def tearDown(self):
        arc_world._LEVELHOLD_ON = self._g
        ARCWorld._lvl_dry.clear(); ARCWorld._lvl_gap_max.clear()

    def _stub(self):
        class S(object):
            game_id = 'vgame'; _levels = 1
            _won_here = 0; _steps_here = 0; _novel_here = 0
            _max_gap = 0; _since_novel = 0; _gap_n = 0.0; _gap_mean = 0.0
            _prog_here = 0; _dep_why = ''
        return S()

    def test_fires_past_the_bound_and_names_itself(self):
        s = self._stub()
        ARCWorld._lvl_dry[('vgame', 1)] = 1
        self.assertFalse(ARCWorld.depleted.fget(s))
        ARCWorld._lvl_dry[('vgame', 1)] = 2
        self.assertTrue(ARCWorld.depleted.fget(s))
        self.assertTrue(s._dep_why.startswith('e:level_dry'))
        ARCWorld._lvl_gap_max[('vgame', 1)] = 4
        self.assertFalse(ARCWorld.depleted.fget(s))


class TestCurriculumUsesLevelHold(unittest.TestCase):
    def _run(self, hold, gate=True, won_ever=5):
        a, b = _ArcLike('game-A', won_ever=won_ever), _ArcLike('game-B')
        a.depleted = True; b.depleted = True
        if hold is not None:
            a.level_hold = lambda: hold
        cw = CurriculumWorld([a, b], mastery_threshold=5)
        with mock.patch.object(_cw_mod, '_ARC_ROTATE', True), \
                mock.patch.object(_cw_mod, '_ALLOC_ON', lambda: False), \
                mock.patch.object(_cw_mod, '_LEVELHOLD_ON', lambda: gate):
            for _ in range(4):
                cw.step(0)
        return cw

    def test_level_hold_true_holds_him(self):
        cw = self._run(True)
        self.assertEqual(cw.patch_moves, 0)
        self.assertGreater(getattr(cw, 'held_by_level', 0), 0)
        self.assertIn('held_by_level', cw.task_stats())

    def test_level_hold_false_lets_him_leave(self):
        cw = self._run(False)
        self.assertGreater(cw.patch_moves, 0, 'CAGED: level_hold False did not release him')

    def test_gate_off_keeps_the_old_win_term(self):
        cw = self._run(False, gate=False, won_ever=5)   # lives 1 <= won_ever 5 -> old hold
        self.assertEqual(cw.patch_moves, 0)
        self.assertEqual(getattr(cw, 'held_by_level', 0), 0)


if __name__ == '__main__':
    unittest.main()
