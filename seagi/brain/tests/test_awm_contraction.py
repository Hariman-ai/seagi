"""Tests for Step 0 organ 4a — AWM contraction under debt.

Covers:
- ThoughtProducedEvent(method='inference') with chain_depth N
  appends (N+1, cycle) to AWM._recent_node_counts.
- Non-inference thoughts don't touch the deque.
- awm_capacity_floor = min(valid node-counts), cold-start = 3,
  stale entries (age > 200 ticks) excluded.
- Contraction formula: (debt - baseline)/full_scale, clipped [0,1].
- effective_capacity formula: max(floor, initial × (1 - contraction))
  bounded above by self.capacity.
- Promote-under-contraction evicts instead of expanding.
- Persistence round-trip preserves the deque.
"""

from __future__ import annotations

import time
import unittest

from seagi.brain import EventKind, EventBus
from seagi.brain.events import ThoughtProducedEvent
from seagi.brain.capabilities.awm import (
    ActiveWorkingMemory,
    AWM_CAPACITY_COLD_START_FLOOR,
    AWM_RECENT_NODES_WINDOW,
    AWM_RECENT_NODES_STALENESS_TICKS,
    DEFAULT_AWM_CAPACITY,
)


def _make_inference_thought(chain_depth: int,
                              cycle: int = 0,
                              method: str = 'inference'
                              ) -> ThoughtProducedEvent:
    return ThoughtProducedEvent(
        kind=EventKind.THOUGHT_PRODUCED,
        cycle=cycle,
        timestamp=time.time(),
        source_capability='cortical',
        origin='internal',
        origin_detail='test',
        focal='test',
        relation='rel',
        target='tgt',
        confidence=0.8,
        method=method,
        text='',
        chain_depth=chain_depth,
    )


class _DebtKnobs:
    """Mutable knob carrier for debt-driven contraction tests."""

    def __init__(self, debt=0.0, baseline=0.0, full_scale=50.0):
        self.debt = debt
        self.baseline = baseline
        self.full_scale = full_scale

    def debt_provider(self):
        return self.debt

    def baseline_provider(self):
        return self.baseline

    def debt_full_scale_provider(self):
        return self.full_scale


class TestThoughtProducedRecording(unittest.TestCase):

    def _make_awm(self, cycle_holder):
        bus = EventBus()
        knobs = _DebtKnobs()
        awm = ActiveWorkingMemory(
            bus=bus,
            cycle_provider=lambda: cycle_holder['c'],
            debt_provider=knobs.debt_provider,
            baseline_provider=knobs.baseline_provider,
            debt_full_scale_provider=knobs.debt_full_scale_provider)
        bus.subscribe(awm.SUBSCRIPTIONS, awm)
        return awm, bus, knobs

    def test_inference_thought_appends_node_count(self):
        c = {'c': 100}
        awm, bus, _knobs = self._make_awm(c)
        # chain_depth = 2 hops → node count = 3.
        bus.publish(_make_inference_thought(chain_depth=2, cycle=100))
        self.assertEqual(len(awm._recent_node_counts), 1)
        self.assertEqual(awm._recent_node_counts[0], (3, 100))
        self.assertEqual(awm.inference_node_counts_recorded, 1)

    def test_non_inference_thought_ignored(self):
        c = {'c': 0}
        awm, bus, _knobs = self._make_awm(c)
        bus.publish(_make_inference_thought(
            chain_depth=2, cycle=0, method='causal'))
        bus.publish(_make_inference_thought(
            chain_depth=5, cycle=0, method='schema'))
        self.assertEqual(len(awm._recent_node_counts), 0)

    def test_zero_chain_depth_inference_skipped(self):
        # Defensive: a 0-hop "inference" makes no sense and would
        # poison the floor.  We require chain_depth >= 1.
        c = {'c': 0}
        awm, bus, _knobs = self._make_awm(c)
        bus.publish(_make_inference_thought(chain_depth=0, cycle=0))
        self.assertEqual(len(awm._recent_node_counts), 0)

    def test_deque_caps_at_window(self):
        c = {'c': 0}
        awm, bus, _knobs = self._make_awm(c)
        for i in range(AWM_RECENT_NODES_WINDOW + 25):
            bus.publish(_make_inference_thought(chain_depth=2, cycle=i))
        self.assertEqual(
            len(awm._recent_node_counts), AWM_RECENT_NODES_WINDOW)


class TestCapacityFloor(unittest.TestCase):

    def _make_awm(self, cycle_holder):
        bus = EventBus()
        return ActiveWorkingMemory(
            bus=bus,
            cycle_provider=lambda: cycle_holder['c'])

    def test_cold_start_floor_is_3(self):
        c = {'c': 0}
        awm = self._make_awm(c)
        self.assertEqual(
            awm.awm_capacity_floor(),
            AWM_CAPACITY_COLD_START_FLOOR)

    def test_floor_is_min_of_valid_counts(self):
        c = {'c': 100}
        awm = self._make_awm(c)
        # Recent counts: 5, 4, 7 — min = 4.
        for (count, cycle) in [(5, 50), (4, 80), (7, 100)]:
            awm._recent_node_counts.append((count, cycle))
        self.assertEqual(awm.awm_capacity_floor(cycle=100), 4)

    def test_stale_entries_excluded(self):
        c = {'c': 500}
        awm = self._make_awm(c)
        # Recent (within 200 ticks of cycle 500): cycle >= 300.
        awm._recent_node_counts.append((3, 100))   # stale
        awm._recent_node_counts.append((4, 350))   # valid
        awm._recent_node_counts.append((6, 480))   # valid
        # 3 is stale → min of valid {4, 6} = 4.
        self.assertEqual(awm.awm_capacity_floor(cycle=500), 4)

    def test_all_stale_falls_back_to_cold_start(self):
        c = {'c': 1000}
        awm = self._make_awm(c)
        awm._recent_node_counts.append((4, 100))   # stale
        awm._recent_node_counts.append((5, 200))   # stale
        self.assertEqual(
            awm.awm_capacity_floor(cycle=1000),
            AWM_CAPACITY_COLD_START_FLOOR)

    def test_floor_never_below_cold_start(self):
        # If a (hypothetical) chain_depth=1 inference landed (node
        # count = 2), the floor must still respect the cold-start
        # minimum of 3.
        c = {'c': 0}
        awm = self._make_awm(c)
        awm._recent_node_counts.append((2, 0))
        self.assertEqual(
            awm.awm_capacity_floor(cycle=0),
            AWM_CAPACITY_COLD_START_FLOOR)


class TestContractionFormula(unittest.TestCase):

    def _make_awm(self, knobs):
        return ActiveWorkingMemory(
            bus=EventBus(),
            cycle_provider=lambda: 0,
            debt_provider=knobs.debt_provider,
            baseline_provider=knobs.baseline_provider,
            debt_full_scale_provider=knobs.debt_full_scale_provider)

    def test_no_providers_means_no_contraction(self):
        awm = ActiveWorkingMemory(
            bus=EventBus(), cycle_provider=lambda: 0)
        self.assertEqual(awm._contraction(), 0.0)
        self.assertEqual(
            awm.effective_capacity(), awm.capacity)

    def test_debt_at_baseline_no_contraction(self):
        knobs = _DebtKnobs(debt=10.0, baseline=10.0, full_scale=50.0)
        awm = self._make_awm(knobs)
        self.assertEqual(awm._contraction(), 0.0)

    def test_mid_contraction(self):
        # debt=35, baseline=10, full_scale=50.
        # (35 - 10) / 50 = 0.5
        knobs = _DebtKnobs(debt=35.0, baseline=10.0, full_scale=50.0)
        awm = self._make_awm(knobs)
        self.assertAlmostEqual(awm._contraction(), 0.5, places=6)

    def test_full_contraction_clamps_at_one(self):
        knobs = _DebtKnobs(debt=1000.0, baseline=0.0, full_scale=50.0)
        awm = self._make_awm(knobs)
        self.assertEqual(awm._contraction(), 1.0)

    def test_zero_full_scale_safe(self):
        knobs = _DebtKnobs(debt=100.0, baseline=0.0, full_scale=0.0)
        awm = self._make_awm(knobs)
        self.assertEqual(awm._contraction(), 0.0)

    def test_effective_capacity_at_mid_contraction(self):
        knobs = _DebtKnobs(debt=35.0, baseline=10.0, full_scale=50.0)
        awm = self._make_awm(knobs)
        # initial_capacity = DEFAULT_AWM_CAPACITY = 256.
        # contraction = 0.5 → 256 × 0.5 = 128, well above floor 3.
        self.assertEqual(awm.effective_capacity(), 128)

    def test_effective_capacity_pegs_at_floor(self):
        knobs = _DebtKnobs(debt=10000.0, baseline=0.0, full_scale=50.0)
        awm = self._make_awm(knobs)
        # contraction = 1.0 → 256 × 0 = 0, below floor → returns
        # cold-start floor = 3.
        self.assertEqual(
            awm.effective_capacity(),
            AWM_CAPACITY_COLD_START_FLOOR)

    def test_floor_recorded_chains_lift_effective(self):
        knobs = _DebtKnobs(debt=10000.0, baseline=0.0, full_scale=50.0)
        awm = self._make_awm(knobs)
        # Exercise some 4-node inferences (chain_depth=3).
        for cycle in (0, 10, 20):
            awm._recent_node_counts.append((4, cycle))
        # Floor now = 4; effective = max(4, 0) = 4.
        # Cycle 0: all entries within staleness window.
        self.assertEqual(
            awm.effective_capacity(cycle=0), 4)


class TestPromoteUnderContraction(unittest.TestCase):

    def test_contracted_awm_evicts_instead_of_expanding(self):
        knobs = _DebtKnobs(debt=200.0, baseline=0.0, full_scale=50.0)
        # contraction clamps at 1.0 → effective_capacity = floor (3).
        bus = EventBus()
        awm = ActiveWorkingMemory(
            bus=bus,
            cycle_provider=lambda: 0,
            capacity=10,  # small so we can fill it quickly
            debt_provider=knobs.debt_provider,
            baseline_provider=knobs.baseline_provider,
            debt_full_scale_provider=knobs.debt_full_scale_provider)
        # Fill above the contracted floor (3).
        for i in range(8):
            awm.promote(f'c{i}', salience=0.9, cycle=i)
        # Despite high salience, no expansion happens under
        # contraction — capacity stayed at the constructor value.
        self.assertEqual(awm.capacity, 10)
        self.assertEqual(awm.capacity_expansions, 0)
        # Size capped at the effective_capacity (3) — evictions
        # culled the overflow.
        self.assertLessEqual(awm.size(), 3)


class TestPersistence(unittest.TestCase):

    def test_round_trip_preserves_deque(self):
        bus = EventBus()
        awm = ActiveWorkingMemory(
            bus=bus, cycle_provider=lambda: 100)
        bus.subscribe(awm.SUBSCRIPTIONS, awm)
        for i, depth in enumerate([2, 3, 2, 4]):
            bus.publish(_make_inference_thought(
                chain_depth=depth, cycle=i * 10))
        snap = awm.to_dict()
        # Fresh awm, load.
        fresh = ActiveWorkingMemory(
            bus=EventBus(), cycle_provider=lambda: 100)
        fresh.load_dict(snap)
        self.assertEqual(len(fresh._recent_node_counts), 4)
        # Same node counts, same cycles.
        self.assertEqual(
            list(fresh._recent_node_counts),
            list(awm._recent_node_counts))


if __name__ == '__main__':
    unittest.main()
