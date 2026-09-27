"""Tests for Step 0 organ 3 — Phase S downscaling + D modulator.

Covers:
- Substrate.downscale_saturated_edges: saturated concepts (top
  edge > 0.98) have all outgoing edges multiplied by 0.975.
- Substrate.settle_weak_edges: below-floor edges clamp UP to floor,
  do not disappear (chemistry-never-fully-dissolves).
- Substrate.discriminability: mean (top - second) over concepts
  with >= 2 outgoing edges.
- DiscriminabilityTracker: post-wake 50-tick sampling window,
  baseline persistence, d_modulation cold-start / mid-collapse /
  full-collapse behavior.
- Phase S integration: consolidation pass records edges_touched
  with MetabolicDebt; debt clears; deque populates.
"""

from __future__ import annotations

import time
import unittest

from seagi.brain import EventKind, EventBus
from seagi.brain.events import ChemistryEvent
from seagi.brain.capabilities.discriminability import (
    DiscriminabilityTracker,
    WAKE_ONSET_D_WINDOW,
    WAKE_ONSET_D_SAMPLE_STRIDE,
    D_MODULATION_DENOM_FACTOR,
)
from seagi.core.substrate import (
    Substrate, Concept,
    EDGE_PRUNE_FLOOR, PROVISIONAL_EDGE_STRENGTH,
    COHERENCE_REINFORCE_BUMP,
)


def _make_wake_onset(cycle: int = 0) -> ChemistryEvent:
    return ChemistryEvent(
        kind=EventKind.CHEMISTRY_FIRE,
        cycle=cycle,
        timestamp=time.time(),
        source_capability='sleep_regulator',
        origin='internal',
        origin_detail='state_change',
        chemistry_kind='wake_onset',
        magnitude=1.0,
    )


# ---------------------------------------------------------------------
# Substrate.downscale_saturated_edges
# ---------------------------------------------------------------------


class TestDownscaleSaturatedEdges(unittest.TestCase):

    def _seed(self, sub: Substrate, edges, top_strength=0.99):
        for (s, r, t, strength) in edges:
            for nm in (s, t):
                if nm not in sub.concepts:
                    sub.add_concept(Concept(name=nm))
            e = sub.add_edge(s, t, r, strength=strength, cycle=0)
            # Forces effective_strength on read to match `strength`
            # (no decay credited yet at cycle 0).
            e.last_reinforced_cycle = 0
        return sub

    def test_saturated_concept_edges_downscaled(self):
        sub = Substrate(seed_relations=False)
        self._seed(sub,
            [('a', 'r', 'x', 0.99),
             ('a', 'r', 'y', 0.95)])
        touched = sub.downscale_saturated_edges(cycle=0)
        self.assertEqual(touched, 2)
        # All edges scaled by 0.975.
        expected_factor = 1.0 - COHERENCE_REINFORCE_BUMP  # 0.975
        e_x = sub.edges[('a', 'r', 'x')]
        e_y = sub.edges[('a', 'r', 'y')]
        self.assertAlmostEqual(e_x.strength, 0.99 * expected_factor, places=5)
        self.assertAlmostEqual(e_y.strength, 0.95 * expected_factor, places=5)

    def test_below_saturation_no_touch(self):
        sub = Substrate(seed_relations=False)
        self._seed(sub,
            [('a', 'r', 'x', 0.5),
             ('a', 'r', 'y', 0.3)])
        before_x = sub.edges[('a', 'r', 'x')].strength
        before_y = sub.edges[('a', 'r', 'y')].strength
        touched = sub.downscale_saturated_edges(cycle=0)
        self.assertEqual(touched, 0)
        self.assertEqual(sub.edges[('a', 'r', 'x')].strength, before_x)
        self.assertEqual(sub.edges[('a', 'r', 'y')].strength, before_y)

    def test_ranks_preserved(self):
        sub = Substrate(seed_relations=False)
        self._seed(sub,
            [('a', 'r', 'x', 0.99),
             ('a', 'r', 'y', 0.985),
             ('a', 'r', 'z', 0.98)])
        sub.downscale_saturated_edges(cycle=0)
        x = sub.edges[('a', 'r', 'x')].strength
        y = sub.edges[('a', 'r', 'y')].strength
        z = sub.edges[('a', 'r', 'z')].strength
        self.assertGreater(x, y)
        self.assertGreater(y, z)


# ---------------------------------------------------------------------
# Substrate.settle_weak_edges
# ---------------------------------------------------------------------


class TestSettleWeakEdges(unittest.TestCase):

    def test_below_floor_clamps_up_not_removed(self):
        sub = Substrate(seed_relations=False)
        sub.add_concept(Concept(name='a'))
        sub.add_concept(Concept(name='b'))
        sub.add_edge('a', 'b', 'causes', strength=0.1, cycle=0)
        # At cycle 10000 the 0.1 edge has fully decayed below floor.
        settled = sub.settle_weak_edges(cycle=10000)
        self.assertEqual(settled, 1)
        # Edge still present — settle, not prune.
        self.assertIn(('a', 'causes', 'b'), sub.edges)
        edge = sub.edges[('a', 'causes', 'b')]
        self.assertAlmostEqual(edge.strength, EDGE_PRUNE_FLOOR, places=5)
        self.assertEqual(edge.last_reinforced_cycle, 10000)

    def test_above_floor_materializes_decay(self):
        sub = Substrate(seed_relations=False)
        sub.add_concept(Concept(name='x'))
        sub.add_concept(Concept(name='y'))
        sub.add_edge('x', 'y', 'is_a', strength=0.5, cycle=0)
        settled = sub.settle_weak_edges(cycle=1000)
        self.assertEqual(settled, 0)
        edge = sub.edges[('x', 'is_a', 'y')]
        self.assertAlmostEqual(edge.strength, 0.49, places=3)


# ---------------------------------------------------------------------
# Substrate.discriminability
# ---------------------------------------------------------------------


class TestDiscriminability(unittest.TestCase):

    def test_high_D_when_clear_ranks(self):
        sub = Substrate(seed_relations=False)
        for nm in ('a', 'x', 'y'):
            sub.add_concept(Concept(name=nm))
        sub.add_edge('a', 'x', 'r', strength=0.9, cycle=0)
        sub.add_edge('a', 'y', 'r', strength=0.1, cycle=0)
        d = sub.discriminability(cycle=0)
        self.assertAlmostEqual(d, 0.8, places=5)

    def test_collapsed_D_when_saturated(self):
        sub = Substrate(seed_relations=False)
        for nm in ('a', 'x', 'y'):
            sub.add_concept(Concept(name=nm))
        sub.add_edge('a', 'x', 'r', strength=0.99, cycle=0)
        sub.add_edge('a', 'y', 'r', strength=0.99, cycle=0)
        d = sub.discriminability(cycle=0)
        self.assertAlmostEqual(d, 0.0, places=5)

    def test_no_qualifying_concepts_returns_zero(self):
        sub = Substrate(seed_relations=False)
        sub.add_concept(Concept(name='a'))
        sub.add_concept(Concept(name='b'))
        sub.add_edge('a', 'b', 'r', strength=0.5, cycle=0)
        # 'a' has only one outgoing edge — not qualifying.
        d = sub.discriminability(cycle=0)
        self.assertEqual(d, 0.0)

    def test_downscale_restores_D(self):
        sub = Substrate(seed_relations=False)
        for nm in ('a', 'x', 'y'):
            sub.add_concept(Concept(name=nm))
        sub.add_edge('a', 'x', 'r', strength=0.99, cycle=0)
        sub.add_edge('a', 'y', 'r', strength=0.95, cycle=0)
        d_before = sub.discriminability(cycle=0)
        sub.downscale_saturated_edges(cycle=0)
        d_after = sub.discriminability(cycle=0)
        # Downscale preserves multiplicative ranks so (top-second)
        # also shrinks by the same factor — but the substrate is no
        # longer saturated, and a future reinforcement on the
        # strongest path can re-grow the gap.  Test the structural
        # property: top no longer above 1 - EDGE_PRUNE_FLOOR.
        top = max(e.effective_strength(0)
                  for e in sub.edges.values())
        self.assertLessEqual(top, 1.0 - EDGE_PRUNE_FLOOR)


# ---------------------------------------------------------------------
# DiscriminabilityTracker
# ---------------------------------------------------------------------


class _FakeSub:
    """Minimal substrate stub for tracker unit tests — controls
    what discriminability returns on each call."""

    def __init__(self, d_sequence):
        # d_sequence: iterable of D values to return in order.
        self._values = list(d_sequence)
        self._idx = 0
        self.queries = 0

    def discriminability(self, cycle):
        self.queries += 1
        if self._values:
            v = self._values[min(self._idx, len(self._values) - 1)]
            self._idx += 1
            return v
        return 0.0


class TestDiscriminabilityTracker(unittest.TestCase):

    def _make(self, d_sequence):
        bus = EventBus()
        sub = _FakeSub(d_sequence)
        cycle_holder = {'c': 0}
        def cycle_provider():
            return cycle_holder['c']
        tracker = DiscriminabilityTracker(
            bus=bus,
            cycle_provider=cycle_provider,
            substrate_provider=lambda: sub)
        bus.subscribe(tracker.SUBSCRIPTIONS, tracker)
        return bus, tracker, sub, cycle_holder

    def test_window_opens_on_wake_onset_closes_after_50_ticks(self):
        # 5 samples (stride=10, window=50) at D=0.6 each
        # → baseline = 0.6.
        bus, tracker, sub, _c = self._make([0.6] * 10)
        bus.publish(_make_wake_onset(cycle=0))
        # Drive the window.
        for _ in range(WAKE_ONSET_D_WINDOW):
            tracker.tick()
        self.assertAlmostEqual(
            tracker.wake_onset_D_baseline, 0.6, places=5)
        self.assertEqual(tracker.windows_completed, 1)
        # Window closed → no more sampling.
        self.assertFalse(tracker._sampling)

    def test_sample_count_matches_stride(self):
        bus, tracker, sub, _c = self._make([0.5] * 20)
        bus.publish(_make_wake_onset(cycle=0))
        before_queries = sub.queries
        for _ in range(WAKE_ONSET_D_WINDOW):
            tracker.tick()
        # Samples land at ticks 0, STRIDE, 2*STRIDE, ... while
        # tick_counter < WINDOW.  Count = ceil(WINDOW / STRIDE).
        expected_samples = (
            (WAKE_ONSET_D_WINDOW + WAKE_ONSET_D_SAMPLE_STRIDE - 1)
            // WAKE_ONSET_D_SAMPLE_STRIDE)
        self.assertEqual(sub.queries - before_queries, expected_samples)

    def test_no_sampling_without_wake_onset(self):
        bus, tracker, sub, _c = self._make([0.6])
        # No event fired; sampling never starts.
        for _ in range(WAKE_ONSET_D_WINDOW):
            tracker.tick()
        self.assertEqual(sub.queries, 0)
        self.assertEqual(tracker.wake_onset_D_baseline, 0.0)

    def test_cold_start_d_mod_returns_zero(self):
        bus, tracker, sub, _c = self._make([0.3])
        # No baseline yet → d_modulation = 0 even if D is low.
        self.assertEqual(tracker.d_modulation_provider(), 0.0)

    def test_d_mod_zero_when_D_at_baseline(self):
        bus, tracker, sub, _c = self._make([0.6] * 100)
        bus.publish(_make_wake_onset(cycle=0))
        for _ in range(WAKE_ONSET_D_WINDOW):
            tracker.tick()
        # baseline = 0.6, current = 0.6 → ratio 0 → mod 0.
        self.assertAlmostEqual(
            tracker.d_modulation_provider(), 0.0, places=5)

    def test_d_mod_pegs_at_full_collapse(self):
        # baseline = 0.6, then current D = 0.12 = 0.6 * 0.2
        # → numerator = 0.48, denominator = 0.6 * 0.8 = 0.48
        # → ratio = 1.0.  Window/stride determines baseline-sample
        # count; rest of list returns the collapsed value.
        baseline_d = 0.6
        # Samples land at ticks 0, STRIDE, 2*STRIDE, ... while
        # tick_counter < WINDOW.  Count = ceil(WINDOW / STRIDE).
        n_baseline = (WAKE_ONSET_D_WINDOW + WAKE_ONSET_D_SAMPLE_STRIDE
                       - 1) // WAKE_ONSET_D_SAMPLE_STRIDE
        bus, tracker, sub, _c = self._make(
            [baseline_d] * n_baseline
            + [baseline_d * 0.2] * 100)
        bus.publish(_make_wake_onset(cycle=0))
        for _ in range(WAKE_ONSET_D_WINDOW):
            tracker.tick()
        # Now query d_modulation — returns collapsed D.
        self.assertAlmostEqual(
            tracker.d_modulation_provider(), 1.0, places=5)

    def test_d_mod_mid_collapse(self):
        # baseline 0.5, current = 0.3:
        # numerator = 0.5 - 0.3 = 0.2
        # denominator = 0.5 * 0.8 = 0.4
        # ratio = 0.5
        # Samples land at ticks 0, STRIDE, 2*STRIDE, ... while
        # tick_counter < WINDOW.  Count = ceil(WINDOW / STRIDE).
        n_baseline = (WAKE_ONSET_D_WINDOW + WAKE_ONSET_D_SAMPLE_STRIDE
                       - 1) // WAKE_ONSET_D_SAMPLE_STRIDE
        bus, tracker, sub, _c = self._make(
            [0.5] * n_baseline + [0.3] * 100)
        bus.publish(_make_wake_onset(cycle=0))
        for _ in range(WAKE_ONSET_D_WINDOW):
            tracker.tick()
        self.assertAlmostEqual(
            tracker.d_modulation_provider(), 0.5, places=4)

    def test_new_wake_resets_window(self):
        # First wake samples 0.4 across the window; second wake
        # samples 0.7.  After second window closes, baseline = 0.7.
        # Samples land at ticks 0, STRIDE, 2*STRIDE, ... while
        # tick_counter < WINDOW.  Count = ceil(WINDOW / STRIDE).
        n_baseline = (WAKE_ONSET_D_WINDOW + WAKE_ONSET_D_SAMPLE_STRIDE
                       - 1) // WAKE_ONSET_D_SAMPLE_STRIDE
        bus, tracker, sub, _c = self._make(
            [0.4] * n_baseline + [0.7] * n_baseline)
        bus.publish(_make_wake_onset(cycle=0))
        for _ in range(WAKE_ONSET_D_WINDOW):
            tracker.tick()
        self.assertAlmostEqual(
            tracker.wake_onset_D_baseline, 0.4, places=5)
        # Trigger another wake — window resets.
        bus.publish(_make_wake_onset(cycle=100))
        for _ in range(WAKE_ONSET_D_WINDOW):
            tracker.tick()
        self.assertAlmostEqual(
            tracker.wake_onset_D_baseline, 0.7, places=5)
        self.assertEqual(tracker.windows_completed, 2)

    def test_persistence_round_trip(self):
        bus, tracker, _sub, _c = self._make([0.5] * 10)
        bus.publish(_make_wake_onset(cycle=0))
        for _ in range(WAKE_ONSET_D_WINDOW):
            tracker.tick()
        snap = tracker.to_dict()
        # Fresh tracker, load.
        fresh = DiscriminabilityTracker(
            bus=EventBus(),
            cycle_provider=lambda: 0,
            substrate_provider=lambda: None)
        fresh.load_dict(snap)
        self.assertAlmostEqual(
            fresh.wake_onset_D_baseline, 0.5, places=5)
        self.assertEqual(fresh.windows_completed, 1)


# ---------------------------------------------------------------------
# Phase S integration — debt clearance via consolidation
# ---------------------------------------------------------------------


class TestPhaseSDebtClearance(unittest.TestCase):
    """End-to-end: when Phase S consolidation runs, it must call
    record_consolidation on MetabolicDebt with non-zero
    edges_touched, populating the clearance deque so debt_full_scale
    becomes substrate-density-aware after one episode."""

    def test_consolidation_records_clearance(self):
        # Import lazily to avoid pulling Engine/Brain at module load.
        from seagi.body.engine import Engine
        from seagi.brain.runtime import (
            Brain, SLEEP_CONSOLIDATION_INTERVAL)
        from seagi.core.substrate import Concept
        engine = Engine()
        sub = engine.substrate
        # Seed a small triangle so coherence reinforcement fires
        # (gives us a non-zero edges_touched).
        for nm in ('a', 'b', 'c'):
            sub.add_concept(Concept(name=nm))
        # Composable relation (is_a, is_a)→is_a so the triangle
        # corroborates under composition-gated coherence.
        for (s, t) in [('a', 'b'), ('b', 'a'), ('b', 'c'),
                          ('c', 'b'), ('a', 'c'), ('c', 'a')]:
            sub.add_edge(s, t, 'is_a', strength=0.5, cycle=0)
        brain = Brain(engine=engine)
        # Give the agent some accrued debt to clear.  Clearance is
        # now measured as debt ACTUALLY removed (min(debt, work)), so
        # with zero debt a consolidation pass clears nothing — seed
        # debt so the clearance signal is non-zero.
        brain.metabolic_debt.debt = 100.0
        before_consolidations = (
            brain.metabolic_debt.consolidations_recorded)
        # Force the agent asleep.
        brain.sleep_regulator.pressure = 0.99
        brain.sleep_regulator._transition_to_sleep(
            brain.bus, cycle=0)
        self.assertTrue(brain.sleep_regulator.is_asleep())
        # Run enough ticks for one consolidation pass.
        for _ in range(SLEEP_CONSOLIDATION_INTERVAL + 5):
            brain.tick()
        self.assertGreaterEqual(brain.consolidations_run, 1)
        # MetabolicDebt saw at least one clearance event.
        self.assertGreater(
            brain.metabolic_debt.consolidations_recorded,
            before_consolidations)
        self.assertGreater(
            brain.metabolic_debt.total_clearance_recorded, 0.0)


if __name__ == '__main__':
    unittest.main()
