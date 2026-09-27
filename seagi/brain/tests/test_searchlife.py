"""Patch 45b SEARCHLIFE: the search hold buys a LIFE, not a step.

MEASURED 2026-09-15 (13 h after patch 45): 6 of 29 `[search] HOLD` terminals
were followed within a second by `DEPLETED -> MOVE` on the same game.  The
hold is asked on the terminal step only; the DEPLETED exit runs every step.

Pinned here (through CurriculumWorld.step, the daemon's order):
  * gate OFF: today's behaviour -- the step after a held terminal moves him
    (the bug, kept reproducible)
  * gate ON: every mid-life step of the held life stays; the counter reads it
  * not a cage: the next terminal asks the search again; a False answer with
    a dry patch moves him exactly once
  * a world that stops answering (`_live` False) is not held by the latch
  * moving to another game spends the latch; it never follows him
  * the latch is a curriculum attribute, absent until the gate first fires
  * task_stats carries the readouts
"""
import unittest
from unittest import mock

from seagi.world import curriculum_world as _cw_mod
from seagi.world.curriculum_world import CurriculumWorld


class _Fake(object):
    """Only the surface the rotation logic reads.  `script` is one bool per
    step: True = this step is a terminal (`done`)."""

    def __init__(self, game_id, script=(), hold_answers=(), successes=()):
        self.game_id = game_id
        self._successes = list(successes)
        self._won_ever = 0
        self._won_here = 0
        self._lives_here = 0
        self._lives_at_last_win = 0
        self._levels = 0
        self.depleted = True          # dry patch: only a hold can keep him
        self._novel_here = 0
        self._live = True
        self._script = list(script)
        self._answers = list(hold_answers)
        self.asked = 0
        self.entered = 0

    def step(self, a):
        term = self._script.pop(0) if self._script else False
        if term:
            self._lives_here += 1
        ok = self._successes.pop(0) if self._successes else False
        return {'success': bool(ok), 'done': bool(term), 'timed_out': False}

    def enter(self):
        self.entered += 1
        self._term_here = 0

    def search_hold(self):
        self.asked += 1
        return self._answers.pop(0) if self._answers else False

    def task_stats(self):
        return {}


def _gates(searchlife):
    return [mock.patch.object(_cw_mod, '_ARC_ROTATE', True),
            mock.patch.object(_cw_mod, '_ALLOC_ON', lambda: False),   # patch 46: the old path is under test here
            mock.patch.object(_cw_mod, '_SEARCH_ON', lambda: True),
            mock.patch.object(_cw_mod, '_SEARCHLIFE_ON', lambda: searchlife),
            mock.patch.object(_cw_mod, '_HYPOTHESIS_ON', lambda: False),
            mock.patch.object(_cw_mod, '_FELTLEAVE_ON', lambda: False),
            mock.patch.object(_cw_mod, '_LEVELHOLD_ON', lambda: False),
            mock.patch.object(_cw_mod, '_YIELDPICK_ON', lambda: False)]


def _run(cw, n, searchlife):
    ps = _gates(searchlife)
    for p in ps:
        p.start()
    try:
        for _ in range(n):
            cw.step(0)
    finally:
        for p in ps:
            p.stop()


class TestSearchLife(unittest.TestCase):

    def _pair(self, script, answers):
        a = _Fake('game-A', script, answers)
        b = _Fake('game-B')
        return a, b, CurriculumWorld([a, b], mastery_threshold=5)

    def test_gate_off_the_step_after_a_held_terminal_moves_him(self):
        # terminal (held by the search), then one mid-life step
        a, b, cw = self._pair([True, False], [True])
        _run(cw, 2, searchlife=False)
        self.assertEqual(a.asked, 1)
        self.assertEqual(cw.held_by_search, 1)
        self.assertEqual(cw.patch_moves, 1, 'the bug: DEPLETED cut the held life')
        self.assertEqual(cw._idx, 1)
        self.assertFalse(hasattr(cw, '_srch_life_game'))

    def test_gate_on_the_held_life_is_not_cut(self):
        a, b, cw = self._pair([True] + [False] * 30, [True])
        _run(cw, 31, searchlife=True)
        self.assertEqual(cw.patch_moves, 0, 'DEPLETED moved him off a search-held life')
        self.assertEqual(cw._idx, 0)
        self.assertEqual(cw.search_latches, 1)
        self.assertEqual(cw.held_by_search_life, 30)
        self.assertEqual(cw._srch_life_game, 'game-A')
        self.assertEqual(a.asked, 1, 'the organ is never asked mid-life')

    def test_not_a_cage_the_next_terminal_decides_again(self):
        # held terminal, 5 mid-life steps, terminal with the search saying no
        a, b, cw = self._pair([True] + [False] * 5 + [True, False], [True, False])
        _run(cw, 7, searchlife=True)
        self.assertEqual(cw.patch_moves, 1, 'a released terminal must move him')
        self.assertEqual(cw._idx, 1)
        self.assertIsNone(cw._srch_life_game)
        self.assertEqual(a.asked, 2)
        _run(cw, 1, searchlife=True)             # a step on B: nothing follows him
        self.assertEqual(cw.patch_moves, 1)
        self.assertIsNone(cw._srch_life_game)

    def test_a_second_held_terminal_relatches(self):
        a, b, cw = self._pair([True, False, False, True, False, False], [True, True])
        _run(cw, 6, searchlife=True)
        self.assertEqual(cw.patch_moves, 0)
        self.assertEqual(cw.search_latches, 2)
        self.assertEqual(cw.held_by_search_life, 4)

    def test_a_dead_world_is_not_held(self):
        a, b, cw = self._pair([True] + [False] * 5, [True])
        _run(cw, 2, searchlife=True)
        self.assertEqual(cw.patch_moves, 0)
        a._live = False                           # the sidecar stopped answering
        _run(cw, 1, searchlife=True)
        self.assertEqual(cw.patch_moves, 1, 'a dead world must still fail open')
        self.assertIsNone(cw._srch_life_game)

    def test_the_latch_does_not_follow_him_to_another_game(self):
        a, b, cw = self._pair([True, False], [True])
        _run(cw, 2, searchlife=True)
        self.assertEqual(cw._srch_life_game, 'game-A')
        cw._idx = 1                               # any other path moved him
        _run(cw, 1, searchlife=True)
        self.assertIsNone(cw._srch_life_game)
        self.assertEqual(cw.held_by_search_life, 1)

    def test_a_world_without_the_method_is_untouched(self):
        a, b, cw = self._pair([True, False], [])
        a.search_hold = None                      # what getattr(..., None) sees without it
        _run(cw, 2, searchlife=True)
        self.assertEqual(cw.patch_moves, 1)
        self.assertIsNone(cw._srch_life_game)
        self.assertEqual(getattr(cw, 'search_latches', 0), 0)

    def test_task_stats_carries_the_readouts(self):
        a, b, cw = self._pair([True, False], [True])
        _run(cw, 2, searchlife=True)
        s = cw.task_stats()
        self.assertEqual(s['held_by_search'], 1)
        self.assertEqual(s['search_latches'], 1)
        self.assertEqual(s['held_by_search_life'], 1)
        self.assertEqual(s['search_life_blocks'], 1)      # DEPLETED was True, the latch stood
        self.assertEqual(s['search_life_game'], 'game-A')
        self.assertIn('held_by_hypothesis', s)

    def test_blocks_count_only_steps_where_depleted_was_true(self):
        a, b, cw = self._pair([True] + [False] * 4, [True])
        _run(cw, 3, searchlife=True)
        a.depleted = False                        # the patch is paying again
        _run(cw, 2, searchlife=True)
        self.assertEqual(cw.held_by_search_life, 4)
        self.assertEqual(cw.search_life_blocks, 2)
        self.assertEqual(cw.patch_moves, 0)

    def test_search_gate_removed_mid_life_spends_the_latch_at_the_terminal(self):
        a, b, cw = self._pair([True, False, True, False], [True, True])
        _run(cw, 2, searchlife=True)
        self.assertEqual(cw._srch_life_game, 'game-A')
        ps = [mock.patch.object(_cw_mod, '_ARC_ROTATE', True),
              mock.patch.object(_cw_mod, '_ALLOC_ON', lambda: False),
              mock.patch.object(_cw_mod, '_SEARCH_ON', lambda: False),
              mock.patch.object(_cw_mod, '_SEARCHLIFE_ON', lambda: True),
              mock.patch.object(_cw_mod, '_HYPOTHESIS_ON', lambda: False),
              mock.patch.object(_cw_mod, '_FELTLEAVE_ON', lambda: False),
              mock.patch.object(_cw_mod, '_LEVELHOLD_ON', lambda: False),
              mock.patch.object(_cw_mod, '_YIELDPICK_ON', lambda: False)]
        for p in ps:
            p.start()
        try:
            cw.step(0)                            # the terminal, search gate off
        finally:
            for p in ps:
                p.stop()
        self.assertIsNone(cw._srch_life_game, 'a stale latch survived the terminal')
        self.assertEqual(a.asked, 1, 'the search gate off: not asked')
        self.assertEqual(cw.patch_moves, 1)       # dry patch, nothing holds: moved

    def test_two_consecutive_terminals_ask_twice_and_move_on_the_second(self):
        a, b, cw = self._pair([True, True, False], [True, False])
        _run(cw, 2, searchlife=True)
        self.assertEqual(a.asked, 2)
        self.assertEqual(cw.patch_moves, 1)
        self.assertIsNone(cw._srch_life_game)

    def test_no_game_id_no_latch(self):
        a, b, cw = self._pair([True, False], [True])
        a.game_id = None
        _run(cw, 2, searchlife=True)
        self.assertEqual(cw.held_by_search, 1)
        self.assertEqual(getattr(cw, 'search_latches', 0), 0)
        self.assertEqual(cw.patch_moves, 1)       # exactly today's behaviour

    def test_a_level_clear_inside_the_held_life_keeps_the_latch(self):
        a = _Fake('game-A', [True, False, False, False], [True], successes=[False, True, False, False])
        b = _Fake('game-B')
        cw = CurriculumWorld([a, b], mastery_threshold=5)
        _run(cw, 4, searchlife=True)
        self.assertEqual(cw._srch_life_game, 'game-A')
        self.assertEqual(cw.patch_moves, 0)
        self.assertEqual(cw.held_by_search_life, 3)


if __name__ == '__main__':
    unittest.main()
