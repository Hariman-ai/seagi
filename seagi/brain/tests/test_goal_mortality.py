"""Goal mortality (2026-06-04) — nothing is immortal, not even a goal.

The autonomous loop went dark because one goal ("resolve: doubts")
was IMMORTAL: it could never be achieved (a self-referential focal
yields only hedged reflections, not inference) and never abandoned
(the abandon mechanism keyed on *activity*, and a re-encountered-but-
unsolved focal kept refreshing the activity clock — line 174's dedup-
refresh). It owned the single attention slot forever and starved the
substrate-explore path.

The fix applies the EDGE-MORTALITY law to goals: urgency decays
continuously and is earned back ONLY by real value (a confident, non-
thin thought about the focal — GoalTracker.earn, fired from check_
thought_completion). A goal that produces nothing fades to zero and
dissolves; a productive one keeps earning and lives as long as it
pays its way. No timeout, no activity clock, no dedup-refresh.

See feedback_stop_manufacturing_problems (the dig that found this).
"""

from __future__ import annotations

import unittest

from seagi.brain.capabilities.goal_tracker import (
    GoalTracker, GOAL_RESOLVE_UNCERTAINTY,
    DEFAULT_GOAL_TIMEOUT_CYCLES,
)

WINDOW = DEFAULT_GOAL_TIMEOUT_CYCLES   # the decay timescale (5000)


class TestGoalMortality(unittest.TestCase):

    def _with_goal(self, urgency=1.0, cycle=0):
        t = GoalTracker()
        g = t.spawn(GOAL_RESOLVE_UNCERTAINTY, 'doubts',
                    urgency=urgency, cycle=cycle)
        return t, g

    def test_urgency_decays_continuously(self):
        _, g = self._with_goal(urgency=1.0, cycle=0)
        self.assertAlmostEqual(g.effective_urgency(0), 1.0, places=6)
        self.assertAlmostEqual(
            g.effective_urgency(WINDOW // 2), 0.5, places=2)
        self.assertEqual(g.effective_urgency(WINDOW + 100), 0.0)

    def test_unearned_goal_dissolves(self):
        t, _ = self._with_goal(urgency=1.0, cycle=0)
        self.assertEqual(t.sweep_mortality(WINDOW // 2), 0)   # survives
        self.assertEqual(len(t), 1)
        self.assertEqual(t.sweep_mortality(WINDOW + 100), 1)  # starved
        self.assertEqual(len(t), 0)
        self.assertEqual(t.goals_abandoned, 1)

    def test_earn_renews_life(self):
        t, g = self._with_goal(urgency=1.0, cycle=0)
        t.sweep_mortality(WINDOW - 100)            # nearly starved
        self.assertLess(g.urgency, 0.1)
        self.assertTrue(t.earn('doubts', WINDOW))  # real value renews
        self.assertAlmostEqual(g.effective_urgency(WINDOW), 1.0, places=6)
        self.assertEqual(t.sweep_mortality(WINDOW + 100), 0)  # now lives
        self.assertEqual(len(t), 1)

    def test_respawn_does_not_grant_immortality(self):
        # THE regression test for the "doubts" immortal loop: hammer
        # the spawner with the same unsolved focal forever (as the live
        # uncertainty loop did) WITHOUT ever earning value.  Before the
        # fix this refreshed the goal eternally (goals_abandoned=0);
        # now the dedup does NOT refresh, so it starves and DIES.
        t, _ = self._with_goal(urgency=1.0, cycle=0)
        for c in range(0, WINDOW * 2, 50):
            t.spawn(GOAL_RESOLVE_UNCERTAINTY, 'doubts',
                    urgency=1.0, cycle=c)   # re-encounter = dedup, no refresh
            t.sweep_mortality(c)
        self.assertGreaterEqual(t.goals_abandoned, 1)   # it died — not immortal

    def test_confident_thought_earns_hedged_does_not(self):
        # End-to-end value gate via check_thought_completion (the live
        # call site).  A conf-0.6 reflection (what "doubts" produces)
        # does NOT earn; a confident, non-thin thought does.
        t, g = self._with_goal(urgency=1.0, cycle=0)
        t.sweep_mortality(WINDOW - 100)
        t.check_thought_completion('doubts', confidence=0.6,
                                   thin_substrate=False, cycle=WINDOW)
        self.assertLess(g.effective_urgency(WINDOW), 0.1)     # still starving
        t.check_thought_completion('doubts', confidence=0.85,
                                   thin_substrate=False, cycle=WINDOW)
        self.assertAlmostEqual(g.effective_urgency(WINDOW), 1.0, places=6)

    def test_thin_confident_thought_does_not_earn(self):
        t, g = self._with_goal(urgency=1.0, cycle=0)
        t.sweep_mortality(WINDOW - 100)
        # confident but THIN substrate = not real value
        t.check_thought_completion('doubts', confidence=0.9,
                                   thin_substrate=True, cycle=WINDOW)
        self.assertLess(g.effective_urgency(WINDOW), 0.1)


if __name__ == '__main__':
    unittest.main()
