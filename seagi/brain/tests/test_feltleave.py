"""PATCH 37 (2026-09-11) -- FELTHERE + FELTLEAVE: how it feels to be HERE,
and a place that feels bad cannot hold him.

USER DOCTRINE 2026-08-29: *"he wants to feel good... they will do anything
to change something to get back to feeling good."*  MEASURED 2026-09-11
before a line was written (frames joined to the journal's own terminals,
104,950 steps, 37 patches): the one global felt register hides a spread
from -1.00 (sb26 L0) to +1.00 (cd82 L2); his paying patches read
+0.74..+1.00 and the sinks he sits in -0.93 (lf52 L1, vc33 L1).  The hold
rules never read it: 1,387 HELD vs 624 MOVE in 24 h.

Pinned here:
  * the register is keyed per (game, level) and booked to the level the
    action was taken on
  * a place is unmeasured below _FELT_MIN_N steps; the environment needs
    TWO measured places (one place IS the environment)
  * felt_bad_here is strictly "worse than the environment", False on any
    doubt, False with the gate off
  * the register round-trips through the save and a place measured in
    this process is never overwritten by the save
  * the curriculum releases a non-commitment hold at a terminal when the
    world says it feels bad here, and ONLY then: gate off, mid-life, a
    commitment, or a world without the method are byte-identical
  * task_stats carries the readouts and the counter
"""
import unittest
from unittest import mock

from seagi.world import arc_world
from seagi.world import curriculum_world as _cw_mod
from seagi.world.arc_world import ARCWorld
from seagi.world.curriculum_world import CurriculumWorld
from seagi.brain.tests.test_curriculum_world import _ArcLike


class _W(object):
    """Only the FELTHERE surface, off the real class."""
    _levels = 1

    def __init__(self, game_id='vgame', lv=1):
        self.game_id = game_id
        self._levels = lv

    felt_here = ARCWorld.felt_here
    felt_env = ARCWorld.felt_env
    felt_bad_here = ARCWorld.felt_bad_here
    felt_to_dict = ARCWorld.felt_to_dict
    felt_from_dict = ARCWorld.felt_from_dict


def _settle(key, value, n=200):
    for _ in range(n):
        ARCWorld._felt_here_note(key, value)


class _Base(unittest.TestCase):
    def setUp(self):
        self._saved = dict(ARCWorld._felt_here)
        ARCWorld._felt_here.clear()
        ARCWorld._felt_here_errors = 0
        self._p = [
            mock.patch.object(arc_world, '_FELTHERE_ON', lambda: True),
            mock.patch.object(arc_world, '_FELTLEAVE_ON', lambda: True),
        ]
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in self._p:
            p.stop()
        ARCWorld._felt_here.clear()
        ARCWorld._felt_here.update(self._saved)


class TestRegister(_Base):

    def test_keyed_per_place_and_bounded(self):
        _settle(('g', 0), 1.0)
        _settle(('g', 1), -1.0)
        self.assertGreater(_W('g', 0).felt_here(), 0.5)
        self.assertLess(_W('g', 1).felt_here(), -0.5)
        self.assertLessEqual(abs(_W('g', 0).felt_here()), 1.0)

    def test_unmeasured_below_the_floor(self):
        _settle(('g', 0), 1.0, n=arc_world._FELT_MIN_N - 1)
        self.assertIsNone(_W('g', 0).felt_here())
        ARCWorld._felt_here_note(('g', 0), 1.0)
        self.assertIsNotNone(_W('g', 0).felt_here())

    def test_bad_is_below_good_not_below_his_average(self):
        """Code review 2026-09-11: a steps-weighted environment mean sits
        wherever he wasted the most steps (~-0.5 live), so 'worse than
        the environment' would HOLD him on a -0.30 place.  The rule is
        the felt steer's own: below _FELT_LOW."""
        _settle(('g', 0), -1.0)
        w = _W('g', 0)
        self.assertIsNone(w.felt_env())          # one place: no environment
        self.assertTrue(w.felt_bad_here())       # ...and still bad
        _settle(('h', 0), 1.0)
        self.assertIsNotNone(w.felt_env())
        self.assertFalse(_W('h', 0).felt_bad_here())
        # a mildly bad place is released even when his average is worse
        for i in range(400):
            ARCWorld._felt_here_note(('m', 0), 1.0 if i % 10 < 3 else -1.0)   # ~ -0.4
        _settle(('sink', 0), -1.0, n=20000)                                    # drags env to ~ -1
        self.assertLess(_W('m', 0).felt_env(), -0.9)
        self.assertTrue(_W('m', 0).felt_bad_here())
        # a good place is never released whatever the average
        self.assertFalse(_W('h', 0).felt_bad_here())
        # just below the line counts, just above does not
        ARCWorld._felt_here[('lo', 0)] = [arc_world._FELT_LOW - 0.01, 10000]
        ARCWorld._felt_here[('hi', 0)] = [arc_world._FELT_LOW + 0.01, 10000]
        self.assertTrue(_W('lo', 0).felt_bad_here())
        self.assertFalse(_W('hi', 0).felt_bad_here())

    def test_environment_is_step_weighted(self):
        _settle(('g', 0), -1.0, n=100)
        _settle(('h', 0), 1.0, n=900)
        e = _W('g', 0).felt_env()
        self.assertGreater(e, 0.0)
        self.assertLess(e, _W('h', 0).felt_here())

    def test_gate_off_answers_nothing(self):
        _settle(('g', 0), -1.0)
        _settle(('h', 0), 1.0)
        with mock.patch.object(arc_world, '_FELTLEAVE_ON', lambda: False):
            self.assertFalse(_W('g', 0).felt_bad_here())
        with mock.patch.object(arc_world, '_FELTHERE_ON', lambda: False):
            self.assertIsNone(_W('g', 0).felt_here())
            self.assertIsNone(_W('g', 0).felt_env())
            self.assertFalse(_W('g', 0).felt_bad_here())
            self.assertEqual(_W('g', 0).felt_to_dict(), {})

    def test_doubt_is_false_and_counted_not_raised(self):
        ARCWorld._felt_here[('g', 0)] = ['junk', 'junk']
        ARCWorld._felt_here[('h', 0)] = [1.0, 500]
        w = _W('g', 0)
        self.assertFalse(w.felt_bad_here())
        self.assertIsNone(w.felt_env())
        self.assertGreater(ARCWorld._felt_here_errors, 0)

    def test_note_never_raises(self):
        ARCWorld._felt_here_note(('g', 0), 'junk')
        self.assertGreater(ARCWorld._felt_here_errors, 0)

    def test_a_young_place_reads_its_own_rate_not_the_zero_it_started_from(self):
        """Doctrine review 2026-09-11: raw EWMA at 50 steps carries 22% of
        the signal, so a virgin level would read ~0 and lose to any
        positive environment.  Corrected, 60 competent steps read as such."""
        _settle(('g', 0), 1.0, n=60)
        self.assertGreater(_W('g', 0).felt_here(), 0.95)
        for i in range(60):
            ARCWorld._felt_here_note(('h', 0), 1.0 if i % 2 else -1.0)
        self.assertLess(abs(_W('h', 0).felt_here()), 0.1)
        _settle(('k', 0), -1.0, n=60)
        self.assertLess(_W('k', 0).felt_here(), -0.95)
        # still bounded after thousands of steps
        _settle(('m', 0), 1.0, n=3000)
        self.assertLessEqual(_W('m', 0).felt_here(), 1.0)

    def test_the_veto_reads_the_level_he_is_on(self):
        """The register books a step to the level it was taken on; the
        veto reads the level he is on at the terminal.  A level he has
        not played is unmeasured, so it can never be judged bad."""
        _settle(('g', 0), -1.0)
        w = _W('g', 0)
        self.assertTrue(w.felt_bad_here())
        w._levels = 1                       # advanced to a level with no book
        self.assertIsNone(w.felt_here())
        self.assertFalse(w.felt_bad_here())


class TestPersistence(_Base):

    def test_round_trip_only_this_game(self):
        _settle(('g', 0), 1.0, n=120)
        _settle(('g', 2), -1.0, n=70)
        _settle(('h', 0), 1.0, n=80)
        d = _W('g', 0).felt_to_dict()
        self.assertEqual(set(d), {'g'})
        self.assertEqual(set(d['g']), {'0', '2'})
        self.assertEqual(d['g']['0'][1], 120)
        ARCWorld._felt_here.clear()
        self.assertEqual(_W('g', 0).felt_from_dict(d), 2)
        self.assertEqual(_W('h', 0).felt_from_dict(d), 0)
        self.assertGreater(_W('g', 0).felt_here(), 0.0)
        self.assertLess(_W('g', 2).felt_here(), 0.0)

    def test_a_place_measured_here_beats_the_save(self):
        _settle(('g', 0), 1.0, n=60)
        n = _W('g', 0).felt_from_dict({'g': {'0': [-1.0, 5000]}})
        self.assertEqual(n, 0)
        self.assertGreater(_W('g', 0).felt_here(), 0.0)

    def test_malformed_rows_are_ignored(self):
        n = _W('g', 0).felt_from_dict({'g': {'x': [1.0, 3], '1': ['a', 'b'],
                                             '2': [7.0, -1], '3': [0.25, 60]}})
        self.assertEqual(n, 1)
        self.assertEqual(ARCWorld._felt_here[('g', 3)], [0.25, 60])   # stored raw
        self.assertGreater(_W('g', 3).felt_here(), 0.25)              # read bias-corrected
        self.assertEqual(_W('g', 0).felt_from_dict(None), 0)
        self.assertEqual(_W('g', 0).felt_from_dict({'g': 'junk'}), 0)


class _Arc(_ArcLike):
    """_ArcLike whose enter() resets `_term_here` like the real world's does
    (code review 2026-09-11: without it the per-step DEPLETED branch made a
    SECOND move in the same step, a stub artefact that let a weak
    `patch_moves > 0` assertion pass on its own)."""
    def enter(self):
        super().enter()
        self._term_here = 0


class TestCurriculumReleases(unittest.TestCase):
    """Two held, depleted, unwon-elsewhere worlds; A is held by level_hold.
    With FELTLEAVE and a bad feeling the terminal releases him."""

    def _run(self, bad, gate=True, done=True, method=True, commit=None):
        a, b = _Arc('game-A', won_ever=5), _Arc('game-B')
        a.depleted = True; b.depleted = True
        a.level_hold = lambda: True
        if method:
            a.felt_bad_here = lambda: bad
            a.felt_here = lambda: -0.9
            a.felt_env = lambda: 0.1
        if not done:
            a.step = lambda act: {'success': False, 'done': False,
                                  'timed_out': False}
        held = {'v': commit}
        cw = CurriculumWorld(
            [a, b], mastery_threshold=5,
            commitment_provider=lambda: held['v'],
            commitment_met=lambda t: held.__setitem__('v', None),
            commitment_lapsed=lambda t: held.__setitem__('v', None))
        with mock.patch.object(_cw_mod, '_ARC_ROTATE', True), \
                mock.patch.object(_cw_mod, '_ALLOC_ON', lambda: False), \
                mock.patch.object(_cw_mod, '_LEVELHOLD_ON', lambda: True), \
                mock.patch.object(_cw_mod, '_FELTLEAVE_ON', lambda: gate):
            for _ in range(4):
                cw.step(0)
        return cw

    def test_bad_feeling_releases_the_hold(self):
        cw = self._run(bad=True)
        self.assertGreater(cw.patch_moves, 0, 'CAGED: it felt bad and he stayed')
        self.assertGreater(cw.felt_leaves, 0)
        self.assertIn('felt_leaves', cw.task_stats())

    def test_exactly_one_move_per_released_terminal(self):
        a, b = _Arc('game-A', won_ever=5), _Arc('game-B')
        a.depleted = True; b.depleted = True
        a.level_hold = lambda: True
        a.felt_bad_here = lambda: True
        cw = CurriculumWorld([a, b], mastery_threshold=5)
        with mock.patch.object(_cw_mod, '_ARC_ROTATE', True), \
                mock.patch.object(_cw_mod, '_ALLOC_ON', lambda: False), \
                mock.patch.object(_cw_mod, '_LEVELHOLD_ON', lambda: True), \
                mock.patch.object(_cw_mod, '_FELTLEAVE_ON', lambda: True):
            cw.step(0)
        self.assertEqual(cw.patch_moves, 1, 'a released terminal must move him exactly once')
        self.assertEqual(cw.felt_leaves, 1)
        self.assertEqual(cw._idx, 1)

    def test_good_feeling_changes_nothing(self):
        cw = self._run(bad=False)
        self.assertEqual(cw.patch_moves, 0)
        self.assertEqual(getattr(cw, 'felt_leaves', 0), 0)

    def test_gate_off_is_byte_identical(self):
        cw = self._run(bad=True, gate=False)
        self.assertEqual(cw.patch_moves, 0)
        self.assertEqual(getattr(cw, 'felt_leaves', 0), 0)

    def test_never_mid_life(self):
        cw = self._run(bad=True, done=False)
        self.assertEqual(cw.patch_moves, 0)
        self.assertEqual(getattr(cw, 'felt_leaves', 0), 0)

    def test_a_world_without_the_method_is_untouched(self):
        cw = self._run(bad=True, method=False)
        self.assertEqual(cw.patch_moves, 0)

    def test_a_promise_outranks_the_feeling(self):
        # An unmet commitment (game-A never won; the budget comes from B).
        a, b = _ArcLike('game-A', won_ever=0), _ArcLike('game-B', won_ever=5)
        a.depleted = True; b.depleted = True
        a.level_hold = lambda: True
        a.felt_bad_here = lambda: True
        held = {'v': 'game-A'}
        cw = CurriculumWorld(
            [a, b], mastery_threshold=5,
            commitment_provider=lambda: held['v'],
            commitment_met=lambda t: held.__setitem__('v', None),
            commitment_lapsed=lambda t: held.__setitem__('v', None))
        with mock.patch.object(_cw_mod, '_ARC_ROTATE', True), \
                mock.patch.object(_cw_mod, '_ALLOC_ON', lambda: False), \
                mock.patch.object(_cw_mod, '_LEVELHOLD_ON', lambda: True), \
                mock.patch.object(_cw_mod, '_FELTLEAVE_ON', lambda: True):
            cw.step(0)        # lives 1 <= budget 5: the promise holds
        self.assertEqual(cw.patch_moves, 0, 'a feeling broke a promise')
        self.assertEqual(getattr(cw, 'felt_leaves', 0), 0)
        self.assertGreater(cw.held_by_commitment, 0)


if __name__ == '__main__':
    unittest.main()
