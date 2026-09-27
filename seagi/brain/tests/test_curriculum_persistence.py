"""His place on the ladder must survive a restart.

Measured 2026-08-18: `curriculum_world.py:62` sets `self._idx = 0` in
`__init__`, `CurriculumWorld` has no `to_dict`/`load_dict`, and
`Brain.to_dict()` carries no ladder key (it persists `world_seed_levels`,
`world_route`, `world_learning`, `world_lives`, `world_svalue`,
`world_click_hits`, `world_paid`, `world_rules` and nothing else about
the world).  `grep -rn "_idx"` outside `curriculum_world.py` finds only
the unrelated `curriculum_prober`.

So every restart returns him to rung 0 and zeroes `graduations` and
`_consec` -- 31 boots in the 7 days to 2026-08-18.  He re-climbs the
ladder from the bottom after every deploy, and a time-gated ladder
baseline is only valid inside one process lifetime.

Partial progress matters as much as the rung: `_consec` is his
run-up to mastery, and losing 4 of 5 consecutive successes to a
restart is the same loss in a smaller unit.

These tests fail until `CurriculumWorld.to_dict()` / `load_dict()`
exist and are wired into `Brain.to_dict()` / `load_personality()`.
"""

from __future__ import annotations

import unittest

from seagi.world.curriculum_world import CurriculumWorld


class _Scripted:
    """A world whose successes follow a fixed script.  `size` is what
    `task_stats()['top_difficulty']` reads, so an escalation is
    observable."""
    n_actions = 4

    def __init__(self, results, n_actions=4, size=3):
        self._r = list(results)
        self._i = 0
        self.n_actions = n_actions
        self.size = size

    def percept(self):
        return {'world_vector': [float(self._i)]}

    def step(self, a):
        ok = self._r[min(self._i, len(self._r) - 1)]
        self._i += 1
        return {'world_vector': [float(self._i)], 'success': ok,
                'timed_out': not ok, 'goal_changed': False, 'steps': 1}

    def task_stats(self):
        return {'i': self._i}


def _always(n=3):
    return [_Scripted([True] * 99, size=n),
            _Scripted([True] * 99, size=n),
            _Scripted([True] * 99, size=n)]


def _escalate(w):
    """A genuinely harder top rung -- Hanoi n -> n+1 in miniature."""
    return _Scripted([True] * 99, size=w.size + 1)


class TestLadderRoundTrip(unittest.TestCase):

    def test_rung_survives_round_trip(self):
        cw = CurriculumWorld(_always(), mastery_threshold=5)
        for _ in range(5):
            cw.step(0)
        self.assertEqual(cw.task_stats()['ladder_index'], 1)

        fresh = CurriculumWorld(_always(), mastery_threshold=5)
        self.assertEqual(fresh.task_stats()['ladder_index'], 0)
        fresh.load_dict(cw.to_dict())

        self.assertEqual(fresh.task_stats()['ladder_index'], 1)
        self.assertEqual(fresh.graduations, 1)

    def test_run_up_to_mastery_survives(self):
        """4 of 5 consecutive successes is progress; a restart must not
        make him start the run-up again."""
        cw = CurriculumWorld(_always(), mastery_threshold=5)
        for _ in range(4):
            cw.step(0)
        self.assertEqual(cw.task_stats()['consec_success'], 4)

        fresh = CurriculumWorld(_always(), mastery_threshold=5)
        fresh.load_dict(cw.to_dict())
        self.assertEqual(fresh.task_stats()['consec_success'], 4)
        # One more success graduates -- it does not take another five.
        self.assertTrue(fresh.step(0)['goal_changed'])
        self.assertEqual(fresh.task_stats()['ladder_index'], 1)

    def test_escalated_top_rung_is_restored(self):
        """Once he is escalating, the rung index alone is not his place
        -- the top rung's DIFFICULTY is.  Restoring idx but not the
        escalation silently hands him an easier task after every
        restart."""
        cw = CurriculumWorld(_always(), mastery_threshold=5,
                             escalator=_escalate)
        for _ in range(20):          # climb 0->1->2, then escalate twice
            cw.step(0)
        self.assertEqual(cw.task_stats()['ladder_index'], 2)
        self.assertEqual(cw.escalations, 2)
        self.assertEqual(cw.task_stats()['top_difficulty'], 5)

        fresh = CurriculumWorld(_always(), mastery_threshold=5,
                                escalator=_escalate)
        fresh.load_dict(cw.to_dict())
        self.assertEqual(fresh.escalations, 2)
        self.assertEqual(fresh.task_stats()['top_difficulty'], 5)
        self.assertEqual(fresh.task_stats()['ladder_index'], 2)

    def test_engagement_shadow_pair_survives_together(self):
        """`held_by_engagement` MINUS `would_hold_recent` is the whole
        measurement (curriculum_world.py:505-506) -- how often the live
        cumulative rule held him on evidence already spent.  Persist the
        minuend without the subtrahend and that difference silently
        overstates itself after the first restart: this project's
        process-local-read-as-all-time bug, third instance."""
        cw = CurriculumWorld(_always(), mastery_threshold=5)
        cw.held_by_engagement = 9
        cw.would_hold_recent = 4

        fresh = CurriculumWorld(_always(), mastery_threshold=5)
        fresh.load_dict(cw.to_dict())
        self.assertEqual(fresh.held_by_engagement, 9)
        self.assertEqual(fresh.would_hold_recent, 4)

    def test_escalations_are_not_claimed_without_an_escalator(self):
        """`escalations` must report what is IN EFFECT.  With no
        escalator the top rung cannot be re-escalated, so a restored
        count would disagree with `top_difficulty` -- a diagnostic that
        lies."""
        cw = CurriculumWorld(_always(), mastery_threshold=5)
        cw.load_dict({'escalations': 3, 'idx': 2})
        self.assertEqual(cw.escalations, 0)
        self.assertEqual(cw.task_stats()['top_difficulty'], 3)

    def test_index_is_clamped_to_the_ladder(self):
        """A save from a longer ladder must not IndexError him out of a
        world -- and `world` is read on every percept."""
        cw = CurriculumWorld(_always(), mastery_threshold=5)
        cw.load_dict({'idx': 99, 'graduations': 99})
        self.assertEqual(cw.task_stats()['ladder_index'], 2)
        self.assertIsNotNone(cw.percept())
        cw.load_dict({'idx': -5})
        self.assertEqual(cw.task_stats()['ladder_index'], 0)
        self.assertIsNotNone(cw.percept())

    def test_load_dict_tolerates_junk(self):
        cw = CurriculumWorld(_always(), mastery_threshold=5)
        for junk in (None, {}, [], 'nope',
                     {'idx': 'x'}, {'idx': None, 'consec': 'y'},
                     {'escalations': 'many'}):
            cw.load_dict(junk)
        self.assertEqual(cw.task_stats()['ladder_index'], 0)
        self.assertIsNotNone(cw.percept())

    def test_serialized_state_is_json_safe(self):
        import json
        cw = CurriculumWorld(_always(), mastery_threshold=5)
        for _ in range(6):
            cw.step(0)
        json.dumps(cw.to_dict(), allow_nan=False)


class TestBrainWiring(unittest.TestCase):

    def test_brain_save_carries_the_rung(self):
        from seagi.body.engine import Engine
        from seagi.brain.runtime import Brain
        brain = Brain(engine=Engine())
        brain.goal_world._idx = 2
        brain.goal_world.graduations = 2
        brain.goal_world._consec = 3

        d = brain.to_dict()
        self.assertIn('curriculum_ladder', d)
        self.assertEqual(d['curriculum_ladder']['idx'], 2)

    def test_brain_restore_puts_him_back_on_his_rung(self):
        from seagi.body.engine import Engine
        from seagi.brain.runtime import Brain
        brain = Brain(engine=Engine())
        brain.goal_world._idx = 2
        brain.goal_world.graduations = 2
        brain.goal_world._consec = 3
        d = brain.to_dict()

        fresh = Brain(engine=Engine())
        self.assertEqual(fresh.goal_world.task_stats()['ladder_index'], 0)
        fresh.load_personality(d)
        self.assertEqual(fresh.goal_world.task_stats()['ladder_index'], 2)
        self.assertEqual(fresh.goal_world.graduations, 2)
        self.assertEqual(
            fresh.goal_world.task_stats()['consec_success'], 3)


if __name__ == '__main__':
    unittest.main()
