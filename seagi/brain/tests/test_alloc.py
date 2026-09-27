"""Patch 46 ALLOC: one allocation decision.

MEASURED 2026-09-15 (715 terminals in 24 h): nine stacked stay/leave rules
held him on cd82 (official 100) for 322 of 328 terminals and moved him
mid-life 113 times.  Under /root/ALLOC_ON one method decides at the terminal:
commitment > organ testing > official headroom > record (released by a bad
feeling) > official-yield draw.  Mid-life: no exit but a dead world.

Pinned here (through CurriculumWorld.step, the daemon's order):
  * order of the reasons; one move at most per terminal
  * the organs are asked under their own gates, BOTH of them, hypothesis
    first, BEFORE the terminal bookkeeping (_term_here unchanged when asked)
  * mid-life: nothing moves him, whatever depleted/felt/record say
  * mid-life on a dead world: he moves to the next in the ring (fail open)
  * capped game moves even with a record hold; organ testing beats capped;
    the boundary is the score (99.99 has headroom, 100.0 has none)
  * felt bad releases the record hold only; without a hold it is 'dry'
    and felt_leaves does not count it
  * the move target comes from the yield draw when on, else least steps
  * the SEARCHLIFE latch is spent at the terminal
  * the legacy TERM lines the instruments read are still written
  * goal_changed from the world passes through
  * gate off: byte-identical (the old machinery: exactly one move)
  * ladder worlds (no _won_ever) untouched; task_stats carries the readouts
"""
import io
import unittest
from unittest import mock

from seagi.world import curriculum_world as _cw_mod
from seagi.world.curriculum_world import CurriculumWorld


class _Fake(object):
    """Only the surface the allocation reads.  `script`: one bool per step,
    True = this step is a terminal."""

    def __init__(self, game_id, script=(), organ=(), hyp=(), record=False, bad=False):
        self.game_id = game_id
        self._won_ever = 0
        self._won_here = 0
        self._lives_here = 0
        self._lives_at_last_win = 0
        self._levels = 0
        self._steps_last_episode = 7
        self.depleted = True
        self._novel_here = 3
        self._live = True
        self._term_here = 0
        self._script = list(script)
        self._organ = list(organ)
        self._hyp = list(hyp)
        self._record = record
        self._bad = bad
        self.goal_changed_next = False
        self.asked = {'search': 0, 'hyp': 0, 'level': 0, 'felt': 0}
        self.term_here_when_asked = []
        self.entered = 0

    def step(self, a):
        term = self._script.pop(0) if self._script else False
        if term:
            self._lives_here += 1
        r = {'success': False, 'done': bool(term), 'timed_out': False}
        if self.goal_changed_next:
            r['goal_changed'] = True
            self.goal_changed_next = False
        return r

    def enter(self):
        self.entered += 1
        self._term_here = 0

    def search_hold(self):
        self.asked['search'] += 1
        self.term_here_when_asked.append(self._term_here)
        return self._organ.pop(0) if self._organ else False

    def hyp_hold(self):
        self.asked['hyp'] += 1
        return self._hyp.pop(0) if self._hyp else False

    def level_hold(self):
        self.asked['level'] += 1
        return self._record

    def felt_bad_here(self):
        self.asked['felt'] += 1
        return self._bad

    def task_stats(self):
        return {}


def _patches(alloc=True, rec=None, yieldpick=False, search=True, hyp=True):
    return [mock.patch.object(_cw_mod, '_ARC_ROTATE', True),
            mock.patch.object(_cw_mod, '_ALLOC_ON', lambda: alloc),
            mock.patch.object(_cw_mod, '_YIELDPICK_ON', lambda: yieldpick),
            mock.patch.object(_cw_mod, '_SEARCH_ON', lambda: search),
            mock.patch.object(_cw_mod, '_SEARCHLIFE_ON', lambda: False),
            mock.patch.object(_cw_mod, '_HYPOTHESIS_ON', lambda: hyp),
            mock.patch.object(_cw_mod, '_FELTLEAVE_ON', lambda: False),
            mock.patch.object(_cw_mod, '_LEVELHOLD_ON', lambda: False),
            mock.patch.object(CurriculumWorld, '_official_record', lambda self: rec)]


def _run(cw, n, **kw):
    ps = _patches(**kw)
    for p in ps:
        p.start()
    try:
        for _ in range(n):
            cw.step(0)
    finally:
        for p in ps:
            p.stop()


REC = {'game': (1, 6, 4.8), 'capp': (6, 6, 100.0), 'othr': (0, 6, 0.0), 'near': (5, 6, 99.99)}


class TestAllocate(unittest.TestCase):

    def _cw(self, a, *others):
        return CurriculumWorld([a] + list(others), mastery_threshold=5)

    def test_organ_testing_stays_even_when_bad_and_capped(self):
        a = _Fake('capp-1', [True], organ=[True], record=False, bad=True)
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 1, rec=REC)
        self.assertEqual(cw.patch_moves, 0)
        self.assertEqual(cw.alloc_stays, 1)
        self.assertEqual(cw.alloc_why, {'organ:search': 1})
        self.assertEqual(cw.held_by_search, 1)
        self.assertEqual(a.asked['level'], 0, 'the organ decided; nothing below it is asked')

    def test_both_organs_asked_hypothesis_first_and_both_counted(self):
        a = _Fake('game-1', [True], organ=[True], hyp=[True])
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 1, rec=REC)
        self.assertEqual(a.asked, {'search': 1, 'hyp': 1, 'level': 0, 'felt': 0})
        self.assertEqual(cw.alloc_why, {'organ:hyp+search': 1})
        self.assertEqual(cw.held_by_hypothesis, 1)
        self.assertEqual(cw.held_by_search, 1)

    def test_organs_are_asked_under_their_own_gates(self):
        a = _Fake('game-1', [True], organ=[True], hyp=[True])
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 1, rec=REC, search=False, hyp=False)
        self.assertEqual(a.asked['search'], 0)
        self.assertEqual(a.asked['hyp'], 0)
        self.assertEqual(cw.patch_moves, 1)
        self.assertEqual(cw.alloc_why, {'dry': 1})

    def test_organs_are_asked_before_the_terminal_bookkeeping(self):
        a = _Fake('game-1', [True, True], organ=[False, False])
        b = _Fake('othr-1', [True, True], organ=[False, False])
        cw = self._cw(a, b)
        _run(cw, 4, rec=REC)
        # a: terminal (asked with _term_here 0, then 1), moved; b likewise
        self.assertEqual(a.term_here_when_asked, [0, 0])
        self.assertEqual(a._term_here, 0, 'enter() zeroed it on the move away and back')
        self.assertEqual(b.term_here_when_asked, [0, 0])

    def test_capped_game_moves_despite_a_record_hold(self):
        a = _Fake('capp-1', [True], record=True)
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 1, rec=REC)
        self.assertEqual(cw.patch_moves, 1)
        self.assertEqual(cw._idx, 1)
        self.assertEqual(cw.alloc_why, {'capped': 1})

    def test_headroom_boundary_is_the_score(self):
        a = _Fake('near-1', [True], record=True)
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 1, rec=REC)
        self.assertEqual(cw.alloc_why, {'record': 1})
        self.assertEqual(cw.patch_moves, 0)

    def test_record_with_headroom_stays(self):
        a = _Fake('game-1', [True], record=True)
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 1, rec=REC)
        self.assertEqual(cw.patch_moves, 0)
        self.assertEqual(cw.alloc_why, {'record': 1})
        self.assertEqual(cw.held_by_level, 1)

    def test_bad_feeling_releases_the_record_hold(self):
        a = _Fake('game-1', [True], record=True, bad=True)
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 1, rec=REC)
        self.assertEqual(cw.patch_moves, 1)
        self.assertEqual(cw.alloc_why, {'felt_bad': 1})
        self.assertEqual(cw.felt_leaves, 1)

    def test_bad_feeling_without_a_hold_is_dry_not_a_felt_leave(self):
        a = _Fake('game-1', [True], record=False, bad=True)
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 1, rec=REC)
        self.assertEqual(cw.alloc_why, {'dry': 1})
        self.assertEqual(getattr(cw, 'felt_leaves', 0), 0)

    def test_dry_moves(self):
        a = _Fake('game-1', [True])
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 1, rec=REC)
        self.assertEqual(cw.patch_moves, 1)
        self.assertEqual(cw.alloc_why, {'dry': 1})
        self.assertEqual(cw._worlds[1].entered, 1)

    def test_no_record_readable_means_headroom(self):
        a = _Fake('capp-1', [True], record=True)
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 1, rec=None)
        self.assertEqual(cw.patch_moves, 0)
        self.assertEqual(cw.alloc_why, {'record': 1})

    def test_commitment_outranks_everything(self):
        a = _Fake('capp-1', [True], bad=True)
        b = _Fake('othr-1')
        b._won_ever = 3                              # budget > 0
        held = {'v': 'capp-1'}
        cw = CurriculumWorld([a, b], mastery_threshold=5,
                             commitment_provider=lambda: held['v'],
                             commitment_met=lambda t: held.__setitem__('v', None),
                             commitment_lapsed=lambda t: held.__setitem__('v', None))
        _run(cw, 1, rec=REC)
        self.assertEqual(cw.patch_moves, 0)
        self.assertEqual(cw.alloc_why, {'commit': 1})

    def test_mid_life_nothing_moves_him(self):
        a = _Fake('capp-1', [True] + [False] * 40, record=False, bad=True)
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 1, rec=REC)
        self.assertEqual(cw.patch_moves, 1)
        cw._idx = 0                               # back on it by hand
        a._term_here = 5
        _run(cw, 40, rec=REC)
        self.assertEqual(cw.patch_moves, 1)
        self.assertEqual(cw._idx, 0)

    def test_mid_life_dead_world_fails_open_to_the_next_in_ring(self):
        a = _Fake('game-1', [False] * 3)
        b, c = _Fake('othr-1'), _Fake('othr-2')
        cw = self._cw(a, b, c)
        _run(cw, 1, rec=REC)
        self.assertEqual(cw.patch_moves, 0)
        a._live = False
        _run(cw, 1, rec=REC)
        self.assertEqual(cw.patch_moves, 1)
        self.assertEqual(cw._idx, 1)
        self.assertEqual(cw.alloc_dead_moves, 1)
        self.assertEqual(cw.alloc_why, {'dead': 1})

    def test_one_move_per_terminal_and_bookkeeping(self):
        a = _Fake('game-1', [True, True, True])
        b = _Fake('othr-1', [True])
        cw = self._cw(a, b)
        _run(cw, 1, rec=REC)
        self.assertEqual(cw.patch_moves, 1)
        self.assertEqual(a._term_here, 1)
        self.assertEqual(a._novel_at_term, 3)
        self.assertEqual(cw._idx, 1)
        self.assertEqual(b._term_here, 0, 'enter() zeroed the new patch')

    def test_the_searchlife_latch_is_spent_at_the_terminal(self):
        a = _Fake('game-1', [True])
        cw = self._cw(a, _Fake('othr-1'))
        cw._srch_life_game = 'game-1'
        _run(cw, 1, rec=REC)
        self.assertIsNone(cw._srch_life_game)

    def test_target_from_the_yield_draw_when_on(self):
        a = _Fake('game-1', [True])
        b, c = _Fake('othr-1'), _Fake('othr-2')
        cw = self._cw(a, b, c)
        with mock.patch.object(CurriculumWorld, '_yield_pick', lambda self: 2):
            _run(cw, 1, rec=REC, yieldpick=True)
        self.assertEqual(cw._idx, 2)

    def test_target_least_steps_when_no_draw(self):
        a = _Fake('game-1', [True])
        b, c = _Fake('othr-1'), _Fake('othr-2')
        b._steps_ever = 50
        c._steps_ever = 5
        cw = self._cw(a, b, c)
        _run(cw, 1, rec=REC)
        self.assertEqual(cw._idx, 2)

    def test_an_organ_that_raises_is_not_a_hold(self):
        a = _Fake('game-1', [True])
        a.search_hold = lambda: (_ for _ in ()).throw(RuntimeError('x'))
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 1, rec=REC)
        self.assertEqual(cw.patch_moves, 1)
        self.assertEqual(cw.alloc_why, {'dry': 1})

    def test_legacy_lines_are_still_written(self):
        a = _Fake('game-1', [True, True], organ=[True, False])
        cw = self._cw(a, _Fake('othr-1'))
        buf = io.StringIO()
        with mock.patch('sys.stderr', buf):
            _run(cw, 2, rec=REC)
        out = buf.getvalue()
        self.assertIn('[alloc] game=game-1 lv=0 why=organ:search head=0.95', out)
        self.assertIn('[arcrotate] TERM game=game-1 won_ever=0 lives=1 at_win=0 lvl=0 steps=7', out)
        self.assertIn('-> HELD (still paying)', out)
        self.assertIn('[arcrotate] TERM game=game-1 won_here=0 -> MOVE (idx 0->1)', out)
        self.assertIn('[alloc] game=game-1 lv=0 why=dry head=0.95', out)

    def test_goal_changed_passes_through(self):
        a = _Fake('game-1', [False])
        a.goal_changed_next = True
        cw = self._cw(a, _Fake('othr-1'))
        ps = _patches(rec=REC)
        for p in ps:
            p.start()
        try:
            r = cw.step(0)
        finally:
            for p in ps:
                p.stop()
        self.assertTrue(r.get('goal_changed'))

    def test_single_world_never_moves(self):
        a = _Fake('game-1', [True, True])
        cw = CurriculumWorld([a], mastery_threshold=5)
        _run(cw, 2, rec=REC)
        self.assertEqual(cw.patch_moves, 0)
        self.assertEqual(getattr(cw, 'alloc_why', {}), {})

    def test_gate_off_is_the_old_machinery(self):
        a = _Fake('capp-1', [True, False], record=True)
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 2, alloc=False, rec=REC)
        self.assertFalse(hasattr(cw, 'alloc_stays'))
        self.assertFalse(hasattr(cw, 'alloc_why'))
        # old rules: depleted True, nothing holds (LEVELHOLD mocked off) -> exactly one move
        self.assertEqual(cw.patch_moves, 1)

    def test_ladder_worlds_are_untouched(self):
        class _Ladder(object):
            def __init__(self):
                self.n = 0
            def step(self, a):
                self.n += 1
                return {'success': False, 'done': True, 'timed_out': False}
            def task_stats(self):
                return {}
        l1, l2 = _Ladder(), _Ladder()
        cw = CurriculumWorld([l1, l2], mastery_threshold=5)
        _run(cw, 3, rec=REC)
        self.assertEqual(cw.patch_moves, 0)
        self.assertEqual(cw._idx, 0)
        self.assertFalse(hasattr(cw, 'alloc_stays'))

    def test_task_stats_carries_the_readouts(self):
        a = _Fake('game-1', [True, True], organ=[True, False])
        cw = self._cw(a, _Fake('othr-1'))
        _run(cw, 2, rec=REC)
        s = cw.task_stats()
        self.assertEqual(s['alloc_stays'], 1)
        self.assertEqual(s['alloc_moves'], 1)
        self.assertEqual(s['alloc_dead_moves'], 0)
        self.assertEqual(s['alloc_why'], {'organ:search': 1, 'dry': 1})


if __name__ == '__main__':
    unittest.main()
