"""Tests for Step 0 organ 1 — MetabolicDebt.

Verifies the substrate-write-debt accumulator's contract:
- Substrate writes deposit debt (1 unit per write, regardless of
  strength) per Q5 unit identity.
- Sleep_onset records the baseline; wake_onset closes the episode
  and appends clearance to the rolling deque.
- record_consolidation (Phase S hook) reduces debt and accumulates
  in the current-episode clearance total.
- debt_full_scale auto-calibrates from the rolling deque mean;
  cold-start returns the bootstrap seed (50, derived from
  EDGE_PRUNE_INTERVAL × COHERENCE_REINFORCE_BUMP).
- Persistence round-trip preserves all state.
"""

from __future__ import annotations

import time
import unittest

from seagi.brain import EventKind, EventBus
from seagi.brain.events import (
    SubstrateWriteQueuedEvent,
    ChemistryEvent,
)
from seagi.brain.capabilities.metabolic_debt import (
    MetabolicDebt,
    BOOTSTRAP_SEED_VALUE,
    CLEARANCE_DEQUE_MAXLEN,
)


def _make_write(subject='a', relation='causes', object_='b',
                  strength=0.1, cycle=1):
    return SubstrateWriteQueuedEvent(
        kind=EventKind.SUBSTRATE_WRITE_QUEUED,
        cycle=cycle,
        timestamp=time.time(),
        source_capability='test',
        origin='internal',
        origin_detail='test',
        subject=subject,
        relation=relation,
        object=object_,
        strength=strength,
        write_reason='test',
    )


def _make_chem(kind, cycle=1):
    return ChemistryEvent(
        kind=EventKind.CHEMISTRY_FIRE,
        cycle=cycle,
        timestamp=time.time(),
        source_capability='sleep_regulator',
        origin='internal',
        origin_detail='state_change',
        chemistry_kind=kind,
        magnitude=1.0,
    )


class TestMetabolicDebtInitialState(unittest.TestCase):

    def setUp(self):
        self.bus = EventBus()
        self.debt = MetabolicDebt(self.bus)

    def test_starts_at_zero_debt(self):
        self.assertEqual(self.debt.debt, 0.0)

    def test_starts_with_zero_baseline(self):
        self.assertEqual(self.debt.wake_onset_baseline_debt, 0.0)

    def test_deque_seeded_at_bootstrap_value(self):
        # All 10 slots initialized to BOOTSTRAP_SEED_VALUE=50.
        # debt_full_scale = mean = 50.
        self.assertEqual(self.debt.debt_full_scale,
                         float(BOOTSTRAP_SEED_VALUE))

    def test_deque_length_is_full(self):
        # Deque starts full so debt_full_scale is well-defined
        # from tick 1.
        self.assertEqual(
            len(self.debt._recent_clearance_per_episode),
            CLEARANCE_DEQUE_MAXLEN)

    def test_has_logged_first_episode_starts_false(self):
        self.assertFalse(self.debt.has_logged_first_episode)


class TestSubstrateWriteAccumulation(unittest.TestCase):

    def setUp(self):
        self.bus = EventBus()
        self.debt = MetabolicDebt(self.bus)

    def test_one_write_increments_by_one(self):
        self.debt.handle(_make_write(), self.bus)
        self.assertEqual(self.debt.debt, 1.0)

    def test_many_writes_accumulate(self):
        for i in range(42):
            self.debt.handle(_make_write(cycle=i), self.bus)
        self.assertEqual(self.debt.debt, 42.0)

    def test_high_strength_writes_still_one_unit(self):
        # Q5 unit identity: count is mutation-count, NOT
        # strength-mass.  A high-strength peer assertion deposits
        # the same single unit as a low-strength provisional write.
        self.debt.handle(_make_write(strength=0.1), self.bus)
        self.debt.handle(_make_write(strength=0.7), self.bus)
        self.debt.handle(_make_write(strength=0.95), self.bus)
        self.assertEqual(self.debt.debt, 3.0)

    def test_writes_observed_counter_tracks(self):
        for _ in range(5):
            self.debt.handle(_make_write(), self.bus)
        self.assertEqual(self.debt.writes_observed, 5)


class TestSleepWakeCycle(unittest.TestCase):

    def setUp(self):
        self.bus = EventBus()
        self.debt = MetabolicDebt(self.bus)

    def test_baseline_snapshots_at_wake_not_sleep(self):
        # Accumulate some debt.
        for _ in range(7):
            self.debt.handle(_make_write(), self.bus)
        # sleep_onset does NOT snapshot the baseline anymore — that
        # was the ratchet bug (it captured the pre-sleep peak).
        self.debt.handle(_make_chem('sleep_onset'), self.bus)
        self.assertEqual(self.debt.wake_onset_baseline_debt, 0.0)
        # Phase S clears most of the debt during the sleep.
        self.debt.record_consolidation(5)        # debt 7 -> 2
        # wake_onset snapshots the baseline = debt REMAINING after
        # clearance (the floor sleep cleared us to).
        self.debt.handle(_make_chem('wake_onset'), self.bus)
        self.assertEqual(self.debt.wake_onset_baseline_debt, 2.0)

    def test_debt_above_baseline_resets_after_wake(self):
        # Full cycle: accumulate, sleep, clear all debt, wake.
        for _ in range(7):
            self.debt.handle(_make_write(), self.bus)
        self.debt.handle(_make_chem('sleep_onset'), self.bus)
        self.debt.record_consolidation(7)        # debt 7 -> 0
        self.debt.handle(_make_chem('wake_onset'), self.bus)
        # Baseline is the cleared floor (0); no new writes yet.
        self.assertEqual(self.debt.debt_above_baseline, 0.0)
        # New writes during the next wake accumulate above it.
        self.debt.handle(_make_write(), self.bus)
        self.assertEqual(self.debt.debt_above_baseline, 1.0)

    def test_baseline_does_not_ratchet_across_cycles(self):
        # Regression for the baseline-ratchet bug: clearing all debt
        # each cycle must leave the baseline at the floor, not creep
        # upward toward an ever-higher pre-sleep peak.
        for _ in range(2):
            for _ in range(60):
                self.debt.handle(_make_write(), self.bus)
            self.debt.handle(_make_chem('sleep_onset'), self.bus)
            self.debt.record_consolidation(10 ** 6)   # clear all
            self.debt.handle(_make_chem('wake_onset'), self.bus)
        self.assertEqual(self.debt.wake_onset_baseline_debt, 0.0)

    def test_wake_onset_appends_to_deque(self):
        # Sleep with no consolidation recorded.  wake_onset should
        # use the bootstrap-seed-equivalent fallback so the deque
        # doesn't drop to zero.
        self.debt.handle(_make_chem('sleep_onset'), self.bus)
        self.debt.handle(_make_chem('wake_onset'), self.bus)
        # Deque mean still equals BOOTSTRAP_SEED_VALUE (oldest
        # entry replaced with another BOOTSTRAP_SEED_VALUE
        # because no consolidation was recorded).
        self.assertEqual(self.debt.debt_full_scale,
                         float(BOOTSTRAP_SEED_VALUE))
        self.assertTrue(self.debt.has_logged_first_episode)

    def test_counter_tracking(self):
        self.debt.handle(_make_chem('sleep_onset'), self.bus)
        self.debt.handle(_make_chem('wake_onset'), self.bus)
        self.assertEqual(self.debt.sleep_onsets_observed, 1)
        self.assertEqual(self.debt.wake_onsets_observed, 1)


class TestConsolidationHook(unittest.TestCase):

    def setUp(self):
        self.bus = EventBus()
        self.debt = MetabolicDebt(self.bus)
        # Build up some debt.
        for _ in range(100):
            self.debt.handle(_make_write(), self.bus)

    def test_record_consolidation_reduces_debt(self):
        self.debt.record_consolidation(30)
        self.assertEqual(self.debt.debt, 70.0)

    def test_debt_floors_at_zero(self):
        # Asked to clear more than current debt.
        self.debt.record_consolidation(500)
        self.assertEqual(self.debt.debt, 0.0)

    def test_zero_or_negative_clearance_is_noop(self):
        self.debt.record_consolidation(0)
        self.debt.record_consolidation(-5)
        self.assertEqual(self.debt.debt, 100.0)

    def test_consolidation_accumulates_in_episode_total(self):
        # Phase S would call record_consolidation multiple times
        # per sleep episode (every SLEEP_CONSOLIDATION_INTERVAL
        # = 40 sleep-ticks).  Wake_onset appends the total.
        self.debt.handle(_make_chem('sleep_onset'), self.bus)
        self.debt.record_consolidation(20)
        self.debt.record_consolidation(15)
        self.debt.record_consolidation(10)
        # Pre-wake_onset: the running total is 45.
        self.assertEqual(self.debt._current_episode_clearance, 45.0)
        self.debt.handle(_make_chem('wake_onset'), self.bus)
        # Post-wake_onset: deque should contain 45 as the newest
        # entry.
        self.assertEqual(
            self.debt._recent_clearance_per_episode[-1], 45.0)
        # And the episode accumulator resets.
        self.assertEqual(self.debt._current_episode_clearance, 0.0)


class TestDebtFullScaleEvolution(unittest.TestCase):
    """The deque drifts from bootstrap seed toward real
    clearance values over 10 episodes."""

    def setUp(self):
        self.bus = EventBus()
        self.debt = MetabolicDebt(self.bus)

    def test_initial_full_scale_is_seed(self):
        self.assertEqual(self.debt.debt_full_scale,
                         float(BOOTSTRAP_SEED_VALUE))

    def test_one_episode_with_real_clearance_shifts_mean(self):
        # Clearance == debt actually removed, so the episode must
        # have debt to clear.  Deposit 500 writes, then consolidate
        # them away.
        for _ in range(500):
            self.debt.handle(_make_write(), self.bus)
        self.debt.handle(_make_chem('sleep_onset'), self.bus)
        self.debt.record_consolidation(500)        # cleared = 500
        self.debt.handle(_make_chem('wake_onset'), self.bus)
        # Deque is now [50]*9 + [500].  Mean = (9*50 + 500)/10 = 95.
        self.assertAlmostEqual(self.debt.debt_full_scale, 95.0)

    def test_ten_episodes_fully_overwrites_seed(self):
        for episode in range(CLEARANCE_DEQUE_MAXLEN):
            for _ in range(500):
                self.debt.handle(_make_write(), self.bus)
            self.debt.handle(_make_chem('sleep_onset'), self.bus)
            self.debt.record_consolidation(500)
            self.debt.handle(_make_chem('wake_onset'), self.bus)
        # All slots now = 500.
        self.assertAlmostEqual(self.debt.debt_full_scale, 500.0)

    def test_clearance_records_debt_cleared_not_raw_work(self):
        # Regression for the poisoning bug: a whole-substrate Phase S
        # pass touches far more edges than there is debt.  The deque
        # entry must be the debt ACTUALLY cleared (120), not the raw
        # work (1,000,000) — otherwise debt_full_scale balloons to
        # substrate scale and the agent never sleeps again.
        for _ in range(120):
            self.debt.handle(_make_write(), self.bus)   # debt = 120
        self.debt.handle(_make_chem('sleep_onset'), self.bus)
        self.debt.record_consolidation(1_000_000)       # work >> debt
        self.debt.handle(_make_chem('wake_onset'), self.bus)
        self.assertEqual(
            self.debt._recent_clearance_per_episode[-1], 120.0)
        # Mean stays in the regime of real debt turnover.
        self.assertAlmostEqual(self.debt.debt_full_scale,
                               (9 * 50 + 120) / 10.0)


class TestPersistence(unittest.TestCase):

    def test_round_trip_preserves_state(self):
        bus = EventBus()
        debt = MetabolicDebt(bus)
        # Build up state.
        for _ in range(42):
            debt.handle(_make_write(), bus)
        debt.handle(_make_chem('sleep_onset'), bus)
        debt.record_consolidation(200)
        debt.handle(_make_chem('wake_onset'), bus)
        # Snapshot.
        state = debt.to_dict()

        # Fresh instance, restore.
        debt2 = MetabolicDebt(EventBus())
        debt2.load_dict(state)

        # Critical state preserved.
        self.assertEqual(debt2.debt, debt.debt)
        self.assertEqual(debt2.wake_onset_baseline_debt,
                         debt.wake_onset_baseline_debt)
        self.assertAlmostEqual(debt2.debt_full_scale,
                               debt.debt_full_scale)
        self.assertEqual(debt2.has_logged_first_episode,
                         debt.has_logged_first_episode)


class TestDebtProvider(unittest.TestCase):
    """The debt_provider lambda contract — downstream consumers
    (SleepRegulator, AWM contraction, gate attenuation) all read
    debt via this callable."""

    def test_provider_returns_current_debt(self):
        debt = MetabolicDebt(EventBus())
        self.assertEqual(debt.debt_provider(), 0.0)
        for _ in range(11):
            debt.handle(_make_write(), debt.bus)
        self.assertEqual(debt.debt_provider(), 11.0)


if __name__ == '__main__':
    unittest.main()
