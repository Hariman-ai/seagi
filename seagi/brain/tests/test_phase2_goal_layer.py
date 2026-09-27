"""Phase 2 goal layer tests (2026-06-01).

Goals self-form from the brain's felt gaps (chiefly the
UncertaintyMonitor window, which aggregates thin-substrate metacog
thoughts AND ACC conflicts), open via the existing GoalTracker, and
steer reverie top-down (Step 3 completion).  Urgency is measured,
not literal; goal focals expand to neighbors when thin.  See
project_seagi_dry_reverie_igniter.
"""

import types
import unittest

from seagi.brain.capabilities.uncertainty_monitor import (
    UncertaintyMonitor, WINDOW_SIZE)
from seagi.brain.capabilities.goal_tracker import (
    GoalTracker, GoalSpawner, THIN_EDGE_THRESHOLD,
    GOAL_RESOLVE_UNCERTAINTY, GOAL_LEARN_ABOUT)
from seagi.body.engine import Engine
from seagi.core.substrate import Concept
from seagi.brain import Brain


class _Bus:
    def publish(self, event):
        pass


class _StubAWM:
    def __init__(self, active):
        self._active = active

    def active_concepts(self):
        return self._active


class _StubConcept:
    def __init__(self, n_edges):
        # edges_out: dict rel -> list; only the COUNT matters here.
        self.edges_out = {f'r{i}': [object()] for i in range(n_edges)}


class _StubLTS:
    def __init__(self, edge_counts):
        self._edge_counts = edge_counts

    def get_concept(self, name):
        n = self._edge_counts.get(name)
        return None if n is None else _StubConcept(n)


def _spawner(tracker, *, awm=None, lts=None, uncertainty=None,
             cycle=1000):
    return GoalSpawner(
        tracker=tracker,
        awm_provider=lambda: awm,
        lts_provider=lambda: lts,
        conversation_provider=lambda: None,
        cycle_provider=lambda: cycle,
        uncertainty_provider=(lambda: uncertainty)
        if uncertainty is not None else None)


class TestUncertaintyAccessor(unittest.TestCase):

    def test_top_unresolved_focals_by_frequency(self):
        um = UncertaintyMonitor(_Bus())
        for f in ('a', 'a', 'a', 'b', 'b', 'c'):
            um._record(1, f)
        top = um.top_unresolved_focals(2)
        self.assertEqual([f for f, _p in top], ['a', 'b'])
        # pressure = occurrences / WINDOW_SIZE (derived, no literal).
        self.assertAlmostEqual(top[0][1], 3.0 / WINDOW_SIZE)
        self.assertAlmostEqual(top[1][1], 2.0 / WINDOW_SIZE)

    def test_empty_window_safe(self):
        um = UncertaintyMonitor(_Bus())
        self.assertEqual(um.top_unresolved_focals(), [])


class TestSpawnerFeltGapTrigger(unittest.TestCase):

    def test_uncertainty_opens_resolve_goals_with_measured_urgency(self):
        tr = GoalTracker()
        sp = _spawner(tr, uncertainty=[('quantum', 0.8),
                                       ('justice', 0.5)])
        spawned = sp.maybe_spawn()
        self.assertEqual(len(spawned), 2)
        by_focal = {g.focal: g for g in spawned}
        self.assertEqual(by_focal['quantum'].kind,
                         GOAL_RESOLVE_UNCERTAINTY)
        # Urgency IS the measured window pressure — not a literal.
        self.assertAlmostEqual(by_focal['quantum'].urgency, 0.8)
        self.assertAlmostEqual(by_focal['justice'].urgency, 0.5)

    def test_skips_self_and_dedups(self):
        tr = GoalTracker()
        sp = _spawner(tr, uncertainty=[('self', 0.9), ('x', 0.4)])
        first = sp.maybe_spawn()
        self.assertEqual([g.focal for g in first], ['x'])  # 'self' skipped
        # Second pass (advance cycle past the spawn gate): no dup.
        sp._cycle_provider = lambda: 2000
        again = sp.maybe_spawn()
        self.assertEqual(again, [])

    def test_thin_awm_urgency_is_derived_from_sparsity(self):
        tr = GoalTracker()
        sp = _spawner(
            tr,
            awm=_StubAWM(['empty', 'oneedge']),
            lts=_StubLTS({'empty': 0, 'oneedge': 1}))
        spawned = {g.focal: g for g in sp.maybe_spawn()}
        self.assertEqual(spawned['empty'].kind, GOAL_LEARN_ABOUT)
        # urgency = 1 - n_edges/THIN_EDGE_THRESHOLD (no literal).
        self.assertAlmostEqual(spawned['empty'].urgency, 1.0)
        self.assertAlmostEqual(
            spawned['oneedge'].urgency,
            1.0 - 1.0 / THIN_EDGE_THRESHOLD)


class TestGoalBiasedFocals(unittest.TestCase):

    def _brain(self):
        engine = Engine()
        return engine, Brain(engine=engine)

    def test_open_goal_focal_is_returned(self):
        engine, brain = self._brain()
        for c in ('a', 'x', 'b', 'c', 'd'):
            engine.substrate.add_concept(Concept(name=c))
        # Dense focal (>= threshold edges) → returned alone.
        for tgt in ('x', 'b', 'c'):
            engine.substrate.add_edge('a', tgt, 'r', strength=0.5,
                                      cycle=0)
        brain.goals.spawn(kind=GOAL_RESOLVE_UNCERTAINTY, focal='a',
                          urgency=0.5, cycle=0)
        focals = brain._goal_focals()
        self.assertIn('a', focals)

    def test_thin_goal_focal_expands_to_neighbors(self):
        engine, brain = self._brain()
        for c in ('g', 'n1', 'n2'):
            engine.substrate.add_concept(Concept(name=c))
        # Thin focal: 2 edges (< THIN_EDGE_THRESHOLD=3).
        engine.substrate.add_edge('g', 'n1', 'r', strength=0.3,
                                  cycle=0)
        engine.substrate.add_edge('g', 'n2', 'r', strength=0.9,
                                  cycle=0)
        brain.goals.spawn(kind=GOAL_LEARN_ABOUT, focal='g',
                          urgency=0.7, cycle=0)
        focals = brain._goal_focals()
        self.assertIn('g', focals)
        # Thin → expanded to its neighbors (bridges into structure).
        self.assertIn('n1', focals)
        self.assertIn('n2', focals)

    def test_no_goals_returns_empty(self):
        engine, brain = self._brain()
        self.assertEqual(brain._goal_focals(), [])


class TestSpawnerTrackerRebind(unittest.TestCase):
    """Regression: load_personality reassigns self.goals; the
    goal_spawner must be re-pointed to the restored tracker, else it
    opens goals into a detached tracker nobody reads (the bug that
    kept fires_from_goal=0 forever)."""

    def test_spawner_tracker_rebinds_after_load(self):
        engine = Engine()
        brain = Brain(engine=engine)
        # Sanity: bound at construction.
        self.assertIs(brain.goal_spawner.tracker, brain.goals)
        # load_personality reassigns self.goals to a NEW tracker.
        brain.load_personality({'goals': brain.goals.to_dict()})
        # The spawner must follow the reassignment.
        self.assertIs(brain.goal_spawner.tracker, brain.goals)
        # A goal opened via the spawner's tracker is visible in the
        # tracker that status / goal_provider read.
        brain.goal_spawner.tracker.spawn(
            kind=GOAL_RESOLVE_UNCERTAINTY, focal='z', urgency=0.5,
            cycle=0)
        self.assertTrue(brain.goals.has_goal_for('z'))


class TestInstrumentation(unittest.TestCase):

    def test_status_exposes_goals(self):
        engine = Engine()
        brain = Brain(engine=engine)
        st = brain.status()
        self.assertIn('goals', st)
        # GoalTracker.stats keys present.
        self.assertIn('goals_spawned', st['goals'])
        self.assertIn('idle_motivation', st)


if __name__ == '__main__':
    unittest.main()
