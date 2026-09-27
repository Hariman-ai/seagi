"""CurriculumWorld: graduate on mastery; goal_changed fires ONLY on graduation
(once per rung); a finite ladder is un-farmable (last rung never re-fires)."""
import unittest
from unittest import mock

from seagi.world import curriculum_world as _cw_mod
from seagi.world.curriculum_world import CurriculumWorld
from seagi.world.goal_world import GoalWorld
from seagi.world.maze_world import MazeWorld
from seagi.world.hanoi_world import HanoiWorld


class _Scripted:
    """A world whose successes follow a fixed script (drives mastery exactly)."""
    n_actions = 4

    def __init__(self, results, n_actions=4):
        self._r = list(results)
        self._i = 0
        self.n_actions = n_actions

    def percept(self):
        return {'world_vector': [float(self._i)]}

    def step(self, a):
        ok = self._r[min(self._i, len(self._r) - 1)]
        self._i += 1
        return {'world_vector': [float(self._i)], 'success': ok,
                'timed_out': not ok, 'goal_changed': False, 'steps': 1}

    def task_stats(self):
        return {'i': self._i}


class TestCurriculumWorld(unittest.TestCase):

    def test_graduates_once_per_rung_and_last_is_unfarmable(self):
        cw = CurriculumWorld([_Scripted([True] * 99),
                              _Scripted([True] * 99),
                              _Scripted([True] * 99)], mastery_threshold=5)
        gc = [cw.step(0)['goal_changed'] for _ in range(5)]
        self.assertEqual(gc, [False, False, False, False, True])   # rung 0->1
        self.assertEqual(cw.task_stats()['ladder_index'], 1)
        gc2 = [cw.step(0)['goal_changed'] for _ in range(5)]
        self.assertEqual(gc2, [False, False, False, False, True])  # rung 1->2
        self.assertEqual(cw.task_stats()['ladder_index'], 2)       # last rung
        # last rung: 30 more successes -> NEVER graduates again (no farm door)
        gc3 = [cw.step(0)['goal_changed'] for _ in range(30)]
        self.assertEqual(sum(gc3), 0,
                         'last rung must never re-fire goal_changed (farm)')
        self.assertEqual(cw.graduations, 2)

    def test_failure_resets_consecutive_count(self):
        # successes T T F T T T -> graduation only after 3 CONSECUTIVE (last 3)
        cw = CurriculumWorld([_Scripted([True, True, False, True, True, True]),
                              _Scripted([True] * 9)], mastery_threshold=3)
        gc = [cw.step(0)['goal_changed'] for _ in range(6)]
        self.assertEqual(gc, [False, False, False, False, False, True])

    def test_n_actions_tracks_current_rung(self):
        cw = CurriculumWorld([_Scripted([True], n_actions=4),
                              _Scripted([True], n_actions=6)],
                             mastery_threshold=1)
        self.assertEqual(cw.n_actions, 4)
        cw.step(0)                                   # master rung 0 -> graduate
        self.assertEqual(cw.n_actions, 6)

    def test_top_rung_self_escalates(self):
        # With an escalator, the LAST rung never stalls: mastering it bumps its
        # difficulty (Hanoi n -> n+1) and fires goal_changed (un-farmable).
        class _Hanoi:
            n_actions = 6
            def __init__(self, d): self.disks = d
            def percept(self): return {'world_vector': [float(self.disks)]}
            def step(self, a): return {'world_vector': [0.0], 'success': True,
                                       'timed_out': False, 'goal_changed': False,
                                       'steps': 1}
            def task_stats(self): return {'disks': self.disks}
        cw = CurriculumWorld([_Scripted([True] * 99), _Hanoi(3)],
                             mastery_threshold=3,
                             escalator=lambda w: _Hanoi(w.disks + 1))
        for _ in range(3):                       # master rung 0 -> Hanoi(3)
            cw.step(0)
        self.assertEqual(cw.task_stats()['top_difficulty'], 3)
        gc = [cw.step(0)['goal_changed'] for _ in range(3)]   # master Hanoi(3)
        self.assertEqual(gc, [False, False, True])
        self.assertEqual(cw.task_stats()['top_difficulty'], 4)   # escalated 3->4
        self.assertEqual(cw.escalations, 1)
        for _ in range(3):                       # master Hanoi(4) -> Hanoi(5)
            cw.step(0)
        self.assertEqual(cw.task_stats()['top_difficulty'], 5)
        self.assertEqual(cw.escalations, 2)
        self.assertEqual(cw.graduations, 1)      # only ONE real rung-advance

    def test_real_world_ladder_constructs_and_steps(self):
        cw = CurriculumWorld([GoalWorld(grid=5, seed=1),
                              MazeWorld(size=6, seed=1, k=8),
                              HanoiWorld(disks=3, seed=1, k=8)])
        for _ in range(30):
            r = cw.step(0)
            self.assertIn('world_vector', r)
            self.assertIn('success', r)
            self.assertIn('goal_changed', r)
        st = cw.task_stats()
        self.assertIn('ladder_index', st)
        self.assertIn('ladder_world', st)


class _ArcLike(_Scripted):
    """Minimal stand-in for a rotating world: has the attributes the rotation
    logic reads.  `_won_ever` is what "until you succeed" is measured on."""

    def __init__(self, game_id, won_ever=0):
        super().__init__([False] * 50)
        self.game_id = game_id
        self._won_ever = won_ever
        self._won_here = 0
        self._lives_here = 0
        self._lives_at_last_win = 0
        self._levels = 0
        self.depleted = True          # patch is DRY: only a promise can hold
        self._novel_here = 0          # nothing found here yet
        self.entered = 0

    def step(self, a):
        r = super().step(a)
        r['done'] = True              # every step terminates -> ejection point
        return r

    def enter(self):
        self.entered += 1


class TestTaskCommitment(unittest.TestCase):
    """A task he was GIVEN holds him where appetite would not (2026-08-10,
    user: *"a task given like go play the game until you succeed should also
    make him go and play"*).  Both worlds here are depleted and unwon, so
    engagement and prior-win grace BOTH decline to hold -- only the
    commitment can."""

    def _pair(self, commitment, wins=0):
        a, b = _ArcLike('game-A'), _ArcLike('game-B')
        b._won_ever = wins            # payments elsewhere buy the budget
        held = {'v': commitment}
        cw = CurriculumWorld(
            [a, b], mastery_threshold=5,
            commitment_provider=lambda: held['v'],
            commitment_met=lambda t: held.__setitem__('v', None),
            commitment_lapsed=lambda t: held.__setitem__('v', None),
            commitment_baseline=lambda: 0)
        return cw, a, b, held

    def test_commitment_holds_him_on_a_dry_unwon_game(self):
        cw, a, _b, _held = self._pair('game-A')
        with mock.patch.object(_cw_mod, '_ARC_ROTATE', True), mock.patch.object(_cw_mod, '_ALLOC_ON', lambda: False):
            for _ in range(5):
                cw.step(0)
        self.assertIs(cw.world, a,
                      'he was rotated off a game he had been asked to play '
                      'until he succeeded.')
        self.assertGreater(cw.task_stats()['held_by_commitment'], 0)

    def test_without_a_commitment_he_still_leaves_a_dry_game(self):
        """The mirror: commitment must ADD patience, never BE the only exit.
        Without it a dry unwon patch must still release him, or this has
        quietly become 'never rotate'."""
        cw, _a, _b, _held = self._pair(None)
        with mock.patch.object(_cw_mod, '_ARC_ROTATE', True), mock.patch.object(_cw_mod, '_ALLOC_ON', lambda: False):
            for _ in range(5):
                cw.step(0)
        # Assert he MOVED, not where he landed: with two worlds he
        # ping-pongs A->B->A inside a single step (terminal move, then the
        # depletion move), so the end position is not the signal.
        self.assertGreater(
            cw.patch_moves, 0,
            'a dry, unwon, uncommitted game still held him -- the rotation '
            'exit is gone.')
        self.assertEqual(
            cw.task_stats()['held_by_commitment'], 0,
            'the commitment hold fired with no commitment assigned.')

    def test_succeeding_releases_the_commitment(self):
        """'Until you succeed' has an END.  Without release he would be
        pinned forever on a game he had already beaten."""
        cw, a, _b, held = self._pair('game-A')
        a._won_ever = 1               # he did what he was asked
        cw.step(0)
        self.assertIsNone(held['v'], 'the commitment survived success -- it '
                                     'is a cage, not a promise.')
        self.assertEqual(cw.task_stats()['commitments_met'], 1)


    def test_persistence_is_bounded_and_the_promise_lapses(self):
        """PERSISTENCE, NOT INDEFINITELY (2026-08-10 ruling).  Pointed at a
        game he cannot win, the commitment must run out.  Budget = his total
        wins (here 2, both on game-B), so once this attempt has cost more
        lives than that, trying is released."""
        cw, a, _b, held = self._pair('game-A', wins=2)
        for _ in range(40):
            a._lives_here += 1        # each attempt costs a life
            cw.step(0)
            if held['v'] is None:
                break
        self.assertIsNone(
            held['v'],
            'the commitment never lapsed -- he is pinned forever on a game '
            'he cannot win.')
        self.assertEqual(cw.task_stats()['commitments_lapsed'], 1)

    def test_the_budget_counts_payments_not_attempts(self):
        """The trap this file already documents: a budget denominated in
        ATTEMPTS lets flailing buy patience.  More losses must NOT buy a
        longer promise -- only wins may."""
        cw, a, _b, _h = self._pair('game-A', wins=3)
        base = cw.task_stats()['commitment_budget']
        a._lives_here += 25           # pure flailing, no wins
        cw.step(0)
        self.assertEqual(
            cw.task_stats()['commitment_budget'], base,
            'failing raised his persistence budget -- flailing is buying '
            'patience again.')


    def test_a_game_that_teaches_him_nothing_cannot_buy_patience(self):
        """REGRESSION 2026-08-11.  The engagement hold read `not depleted`,
        which is vacuously true before he has any gap history, so a game he
        learned NOTHING from held him forever: measured on sb26 as
        since_novel 1032 of 1032 steps, 206 deaths, patch_moves 0, and a
        depleted() that could not fire by construction.  Engagement must be
        POSITIVE evidence -- something actually found here."""
        a, b = _ArcLike('game-A'), _ArcLike('game-B')
        a.depleted = False            # depletion CANNOT fire (the trap)
        b.depleted = False
        a._novel_here = 0             # ...and he has learned nothing here
        b._novel_here = 0
        cw = CurriculumWorld([a, b], mastery_threshold=5)
        # The terminal-rotation path is gated on _ARC_ROTATE, which is an
        # ENV FLAG read at import.  Force it on: without this the test passes
        # only when SEAGI_ARCROTATE=1 happens to be set in the shell, and is
        # otherwise vacuously green while exercising nothing.
        with mock.patch.object(_cw_mod, '_ARC_ROTATE', True), mock.patch.object(_cw_mod, '_ALLOC_ON', lambda: False):
            for _ in range(5):
                cw.step(0)
        self.assertGreater(
            cw.patch_moves, 0,
            'a game he has learned NOTHING from still held him, with no '
            'exit able to fire. He is trapped again.')

    def test_finding_something_here_does_buy_patience(self):
        """The other side: once he HAS found something, engagement holds."""
        a, b = _ArcLike('game-A'), _ArcLike('game-B')
        a.depleted = False
        b.depleted = False
        a._novel_here = 3             # he is getting somewhere here
        cw = CurriculumWorld([a, b], mastery_threshold=5)
        with mock.patch.object(_cw_mod, '_ARC_ROTATE', True), mock.patch.object(_cw_mod, '_ALLOC_ON', lambda: False):
            for _ in range(5):
                cw.step(0)
        self.assertIs(cw.world, a,
                      'he was rotated off a game that was still teaching '
                      'him something.')


if __name__ == '__main__':
    unittest.main()
