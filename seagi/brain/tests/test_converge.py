"""Patch 47 CONVERGE: the search organ judges its own progress (gate /root/CONVERGE_ON).

Pinned (in the daemon's order: the hold is asked at the terminal BEFORE the next
life begins, so the life in flight is the one judged):
  * gate off: the patch-45 hold, unchanged (a zero-new life releases)
  * gate on: a zero-new life that executed an untried control HOLDS (a converging
    search), a life with no progress releases, a life with nothing recurring releases
  * a walk (constant new-per-step rate, life after life) releases after WALK_RUN
    consecutive judgements from WALK_MIN_LIVES on, and writes one WALK event
  * a converging visit (rate falling) is never released by the walk test
  * a new visit resets the judgement; a new life within a visit does not
  * the offline series that decided the constants replay to the pre-registered lives
  * the history is capped; a known win / an evicting life still release
"""
import unittest

from seagi.world import goal_search as S


def _on():
    S._CONVERGE_ON = lambda: True


def _off():
    S._CONVERGE_ON = lambda: False


def _life(sr, steps, new, untried=1):
    """One life: `steps` executed steps, `new` of them onto states never seen,
    `untried` of them controls executed for the first time.  Recurring steps
    go to a known state through a control already tried there."""
    sr.begin_life(root='r')
    if 'r' not in sr.n_ctl:
        sr.see('r', [('A', 0), ('A', 1), ('A', 2)])
    k = 0
    for i in range(steps):
        if i < new:
            sr.observe('r', ('A', 0), 'n%d_%d' % (sr.lives, i), False, False)
            sr.see('n%d_%d' % (sr.lives, i), [('A', 0), ('A', 1)])
        elif i < new + untried:
            # an untried control here (a fresh control label per life)
            c = ('C', sr.lives, i)
            sr.observe('r', c, 'r', False, False)
        else:
            sr.observe('r', ('A', 1), 'r', False, False)
            k += 1


class TestConverge(unittest.TestCase):
    def tearDown(self):
        _off()

    def test_gate_off_is_patch_45(self):
        _off()
        sr = S.Search()
        _life(sr, 5, 2); self.assertTrue(sr.testing())          # first life, new, frontier
        _life(sr, 5, 0, untried=2)
        self.assertFalse(sr.testing())                           # patch 45: nothing new releases

    def test_zero_new_with_progress_holds(self):
        _on()
        sr = S.Search()
        _life(sr, 5, 2); self.assertTrue(sr.testing())
        _life(sr, 5, 0, untried=2)
        self.assertEqual(sr.new_this_life, 0)
        self.assertGreater(sr.untried_this_life, 0)
        self.assertTrue(sr.testing())                            # converging: holds
        _life(sr, 5, 0, untried=0)
        self.assertFalse(sr.testing())                           # no progress: releases
        _life(sr, 3, 3, untried=0)
        self.assertGreaterEqual(sr.new_this_life, sr.steps_this_life)
        self.assertFalse(sr.testing())                           # nothing recurred: releases

    def test_walk_releases_after_the_run(self):
        _on()
        sr = S.Search()
        released = None
        for life in range(1, 40):
            _life(sr, 20, 10)                                    # a constant rate, every life
            if not sr.testing():
                released = life
                break
        self.assertEqual(released, S.WALK_MIN_LIVES + S.WALK_RUN - 1)   # 17
        self.assertEqual(sr.walk_run, S.WALK_RUN)
        self.assertEqual([e for e in sr.events if e.startswith('WALK')], ['WALK run=12 hist=16'])
        self.assertFalse(sr.testing())                           # asked twice: one event
        self.assertEqual(len([e for e in sr.events if e.startswith('WALK')]), 1)

    def test_converging_is_never_released_by_the_walk_test(self):
        _on()
        sr = S.Search()
        rates = [10, 6, 4, 2, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for nw in rates:
            _life(sr, 20, nw)
            self.assertTrue(sr.testing(), 'life %d' % sr.lives)
        self.assertLess(sr.walk_run, S.WALK_RUN)

    def test_new_visit_resets_the_judgement(self):
        _on()
        sr = S.Search()
        for _ in range(16):
            _life(sr, 20, 10)
        self.assertTrue(sr.testing())                            # 16 lives: run 11, still held
        sr.begin_life(root='r', visit=True)                      # he came back: a new visit
        self.assertEqual(sr.hist, [])
        self.assertEqual(sr.walk_run, 0)
        for _ in range(10):
            _life(sr, 20, 10)
            self.assertTrue(sr.testing())
        for _ in range(7):
            _life(sr, 20, 10)
        self.assertFalse(sr.testing())                           # 17 lives of the new visit

    def test_hist_is_capped_and_win_and_eviction_release(self):
        _on()
        sr = S.Search()
        for _ in range(S.HIST_MAX + 20):
            sr.begin_life(root='r')
            sr.steps_this_life = 5; sr.new_this_life = 1
        self.assertEqual(len(sr.hist), S.HIST_MAX)
        sr2 = S.Search()
        _life(sr2, 5, 2)
        sr2.observe('r', ('A', 2), 'w', True, False)
        self.assertTrue(sr2.has_win()); self.assertFalse(sr2.testing())
        sr3 = S.Search(); sr3.begin_life(root='r'); sr3.see('r', [('A', 0)])
        for i in range(S.MAX_STATES + 5):
            sr3.observe('s%d' % i, ('A', 0), 's%d' % (i + 1), False, False)
        sr3.begin_life(root='r'); sr3.observe('r', ('A', 0), 's0', False, False)
        for i in range(S.MAX_STATES + 5):
            sr3.observe('t%d' % i, ('A', 0), 't%d' % (i + 1), False, False)
        self.assertFalse(sr3.testing())

    def test_offline_series_replay_to_the_preregistered_lives(self):
        """The decision data (bfs4_out.txt, lv0 lives, one seed): s5i5 never,
        sc25 at 17, sk48 (2 lives) never; the live sc25 series at 17."""
        _on()
        s5i5 = [(50, 15), (50, 1), (50, 4), (50, 1), (50, 4), (27, 0), (30, 1), (50, 13), (50, 2), (17, 0),
                (21, 0), (14, 0), (13, 0), (10, 0), (38, 0), (50, 3), (50, 1), (22, 0), (50, 5), (50, 0),
                (50, 2), (50, 0), (30, 0), (50, 1), (50, 0)]
        sc25 = [(94, 50), (89, 50), (87, 50), (91, 49), (81, 48), (100, 50), (90, 50), (122, 49), (95, 47),
                (93, 47), (88, 43), (88, 45), (99, 46), (97, 47), (95, 47), (93, 47), (88, 43), (88, 45)]

        def first_release(series):
            sr = S.Search()
            for i, (st, nw) in enumerate(series):
                _life(sr, st, nw)
                if not sr.testing():
                    return i + 1
            return None
        self.assertIsNone(first_release(s5i5))
        self.assertEqual(first_release(sc25), 17)

    def test_a_control_dropped_at_the_clock_is_not_progress(self):
        """Review finding 1: the last control of a clock-ended life is dropped
        from the row (patch 45, review 2.4); it must not count as an untried
        control executed, or a frontier one step beyond the clock holds forever."""
        _on()
        sr = S.Search()
        sr.begin_life(root='r'); sr.see('r', [('A', 0), ('A', 1)])
        sr.observe('r', ('A', 0), 's1', False, False); sr.see('s1', [('A', 0), ('A', 1)])
        self.assertTrue(sr.testing())
        sr.max_life = 2
        sr.begin_life(root='r')
        sr.observe('r', ('A', 0), 's1', False, False)              # a known edge
        sr.observe('s1', ('A', 0), None, False, True)              # step 2 == max_life: dropped
        self.assertNotIn(('A', 0), sr.trans['s1'])
        self.assertEqual(sr.untried_this_life, 0)
        self.assertFalse(sr.testing())                             # no progress: releases
        sr.max_life = 9
        sr.begin_life(root='r')
        sr.observe('r', ('A', 0), 's1', False, False)
        sr.observe('s1', ('A', 0), None, False, True)              # before the clock: learned
        self.assertEqual(sr.trans['s1'][('A', 0)], S.OVER)
        self.assertEqual(sr.untried_this_life, 1)
        self.assertTrue(sr.testing())                              # progress, ('A', 1) untried

    def test_a_level_entered_by_a_clear_is_a_new_visit(self):
        """Review finding 2: the table of a level reached by clearing keeps its
        judgement across visits unless the clear hook begins a visit."""
        import os, importlib.util
        spec = importlib.util.spec_from_file_location(
            'test_search_mod', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_search.py'))
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        from seagi.world.arc_world import ARCWorld
        T._reset_tables()
        _on()
        stale = S.Search()
        stale.hist = [(20, 10)] * 16; stale.walk_run = 11
        ARCWorld._srch[('sgame', 1)] = stale
        w = T._Stub()
        clears, holds = T.play(w, lives=30)
        self.assertGreaterEqual(len(clears), 1)
        sr1 = ARCWorld._srch[('sgame', 1)]
        self.assertIs(sr1, stale)
        self.assertEqual(sr1.hist, [])                             # the clear began a visit
        self.assertEqual(sr1.walk_run, 0)
        T._reset_tables()


if __name__ == '__main__':
    unittest.main()
