"""Tests for Step 0 organ 2 — AllostaticLoad.

Covers:
- Per-channel tonic_window sampling per tick.
- Sleep_onset drifts each baseline toward its tonic mean by
  DRIFT_RATE = 0.001.
- Channels with empty windows are safely skipped.
- Cold-start: load = 0 even with elevated state (until drift
  starts shifting baselines).
- Sustained elevation: many sleep cycles produce cumulative drift
  → non-zero load.
- Load formula = mean |baseline - initial| over all channels,
  clamped to [0, 1].
- Persistence round-trip preserves baselines + tonic windows.
- Wired into Brain: SleepRegulator's load_provider points at the
  AllostaticLoad organ.
"""

from __future__ import annotations

import time
import unittest

from seagi.brain import EventKind, EventBus
from seagi.brain.events import ChemistryEvent
from seagi.brain.chemistry_types import CHANNELS
from seagi.brain.capabilities.allostatic_load import (
    AllostaticLoad,
    DRIFT_RATE,
    TONIC_WINDOW_DEPTH,
)


def _make_sleep_onset(cycle: int = 0) -> ChemistryEvent:
    return ChemistryEvent(
        kind=EventKind.CHEMISTRY_FIRE,
        cycle=cycle,
        timestamp=time.time(),
        source_capability='sleep_regulator',
        origin='internal',
        origin_detail='state_change',
        chemistry_kind='sleep_onset',
        magnitude=1.0,
    )


class _StateProvider:
    """Mutable container so tests can drive the chemistry state
    over time without rebuilding the organ."""

    def __init__(self, state=None):
        self.state = (
            dict(state) if state is not None
            else {ch: cfg['baseline']
                  for ch, cfg in CHANNELS.items()})

    def __call__(self):
        return self.state


class TestAllostaticLoadConstants(unittest.TestCase):

    def test_drift_rate_is_promille_floor(self):
        self.assertAlmostEqual(DRIFT_RATE, 0.001, places=6)

    def test_tonic_window_matches_raphe_depth(self):
        from seagi.brain.capabilities.neuromodulators import (
            RAPHE_HISTORY_DEPTH)
        self.assertEqual(TONIC_WINDOW_DEPTH, RAPHE_HISTORY_DEPTH)


class TestTickSampling(unittest.TestCase):

    def test_tick_appends_to_each_channel_window(self):
        bus = EventBus()
        provider = _StateProvider()
        organ = AllostaticLoad(
            bus=bus, cycle_provider=lambda: 0,
            chemistry_provider=provider)
        organ.tick()
        organ.tick()
        organ.tick()
        for ch in CHANNELS:
            self.assertEqual(
                len(organ._tonic_windows[ch]), 3)

    def test_tick_no_provider_safe(self):
        bus = EventBus()
        organ = AllostaticLoad(
            bus=bus, cycle_provider=lambda: 0,
            chemistry_provider=None)
        # No crash.
        organ.tick()
        for ch in CHANNELS:
            self.assertEqual(len(organ._tonic_windows[ch]), 0)

    def test_window_caps_at_TONIC_WINDOW_DEPTH(self):
        bus = EventBus()
        provider = _StateProvider()
        organ = AllostaticLoad(
            bus=bus, cycle_provider=lambda: 0,
            chemistry_provider=provider)
        for _ in range(TONIC_WINDOW_DEPTH + 20):
            organ.tick()
        for ch in CHANNELS:
            self.assertEqual(
                len(organ._tonic_windows[ch]),
                TONIC_WINDOW_DEPTH)


class TestSleepOnsetDrift(unittest.TestCase):

    def test_baseline_drifts_toward_tonic_mean(self):
        bus = EventBus()
        # Hold cortisol at 0.50 (well above baseline 0.10) for a
        # full window.
        elevated = {ch: cfg['baseline']
                    for ch, cfg in CHANNELS.items()}
        elevated['cortisol'] = 0.50
        provider = _StateProvider(elevated)
        organ = AllostaticLoad(
            bus=bus, cycle_provider=lambda: 0,
            chemistry_provider=provider)
        bus.subscribe(organ.SUBSCRIPTIONS, organ)
        # Fill all windows.
        for _ in range(TONIC_WINDOW_DEPTH):
            organ.tick()
        initial_cort = organ.baselines['cortisol']
        bus.publish(_make_sleep_onset(cycle=1))
        # Drift step: baseline += DRIFT_RATE × (tonic_mean − baseline).
        # tonic_mean = 0.50; current = 0.10; drift = 0.001 × 0.4 = 0.0004.
        self.assertAlmostEqual(
            organ.baselines['cortisol'] - initial_cort,
            DRIFT_RATE * (0.50 - initial_cort),
            places=6)

    def test_empty_window_channel_safe(self):
        bus = EventBus()
        provider = _StateProvider()
        organ = AllostaticLoad(
            bus=bus, cycle_provider=lambda: 0,
            chemistry_provider=provider)
        # No ticks → all windows empty.  Sleep onset must not crash.
        bus.subscribe(organ.SUBSCRIPTIONS, organ)
        before = dict(organ.baselines)
        bus.publish(_make_sleep_onset(cycle=1))
        # Baselines unchanged (no drift on empty windows).
        for ch in CHANNELS:
            self.assertEqual(organ.baselines[ch], before[ch])

    def test_chronic_elevation_accumulates(self):
        # 100 sleep_onsets with constant elevation: cumulative drift.
        bus = EventBus()
        elevated = {ch: cfg['baseline']
                    for ch, cfg in CHANNELS.items()}
        elevated['cortisol'] = 0.60
        provider = _StateProvider(elevated)
        organ = AllostaticLoad(
            bus=bus, cycle_provider=lambda: 0,
            chemistry_provider=provider)
        bus.subscribe(organ.SUBSCRIPTIONS, organ)
        for ep in range(100):
            for _ in range(TONIC_WINDOW_DEPTH):
                organ.tick()
            bus.publish(_make_sleep_onset(cycle=ep))
        # After 100 drifts toward 0.60 from 0.10 at rate 0.001, the
        # baseline should have closed ~10% of the gap (geometric
        # decay): residual = 0.50 × 0.999^100 ≈ 0.452.
        # Baseline_after ≈ 0.60 - 0.452 ≈ 0.148.  Just verify it
        # has drifted UP from 0.10 (i.e., direction is correct).
        self.assertGreater(organ.baselines['cortisol'], 0.10)


class TestLoadFormula(unittest.TestCase):

    def test_cold_start_load_zero(self):
        bus = EventBus()
        organ = AllostaticLoad(
            bus=bus, cycle_provider=lambda: 0,
            chemistry_provider=_StateProvider())
        self.assertEqual(organ.load(), 0.0)

    def test_load_is_mean_absolute_delta(self):
        bus = EventBus()
        organ = AllostaticLoad(
            bus=bus, cycle_provider=lambda: 0,
            chemistry_provider=_StateProvider())
        # Manually shift baselines.  Use known deltas.
        organ.baselines['cortisol'] = (
            organ.initial_baselines['cortisol'] + 0.10)
        organ.baselines['dopamine'] = (
            organ.initial_baselines['dopamine'] - 0.06)
        # Expected load = (0.10 + 0.06 + 0 + ... + 0) / 8 = 0.02
        expected = 0.16 / float(len(CHANNELS))
        self.assertAlmostEqual(organ.load(), expected, places=6)

    def test_load_clamped_to_one(self):
        bus = EventBus()
        organ = AllostaticLoad(
            bus=bus, cycle_provider=lambda: 0,
            chemistry_provider=_StateProvider())
        # Force all baselines to a huge delta.
        for ch in CHANNELS:
            organ.baselines[ch] = (
                organ.initial_baselines[ch] + 5.0)
        self.assertEqual(organ.load(), 1.0)

    def test_load_provider_callable_matches_load(self):
        bus = EventBus()
        organ = AllostaticLoad(
            bus=bus, cycle_provider=lambda: 0,
            chemistry_provider=_StateProvider())
        organ.baselines['cortisol'] = (
            organ.initial_baselines['cortisol'] + 0.08)
        self.assertEqual(organ.load_provider(), organ.load())


class TestPersistence(unittest.TestCase):

    def test_round_trip_preserves_baselines_and_windows(self):
        bus = EventBus()
        provider = _StateProvider()
        organ = AllostaticLoad(
            bus=bus, cycle_provider=lambda: 0,
            chemistry_provider=provider)
        bus.subscribe(organ.SUBSCRIPTIONS, organ)
        # Drive some state.
        for _ in range(10):
            organ.tick()
        bus.publish(_make_sleep_onset(cycle=1))
        # Force a known baseline shift.
        organ.baselines['cortisol'] = 0.20
        snap = organ.to_dict()
        # Fresh organ, load.
        fresh = AllostaticLoad(
            bus=EventBus(), cycle_provider=lambda: 0,
            chemistry_provider=None)
        fresh.load_dict(snap)
        self.assertAlmostEqual(
            fresh.baselines['cortisol'], 0.20, places=6)
        # initial_baselines preserved.
        self.assertEqual(
            fresh.initial_baselines, organ.initial_baselines)
        # Tonic windows preserved.
        for ch in CHANNELS:
            self.assertEqual(
                len(fresh._tonic_windows[ch]),
                len(organ._tonic_windows[ch]))

    def test_load_dict_tolerates_partial_state(self):
        bus = EventBus()
        organ = AllostaticLoad(
            bus=bus, cycle_provider=lambda: 0,
            chemistry_provider=None)
        # No baselines / windows in snapshot — must not crash.
        organ.load_dict({})
        organ.load_dict({'baselines': {'cortisol': 'not a float'}})
        # Unchanged from default.
        self.assertEqual(
            organ.baselines['cortisol'],
            CHANNELS['cortisol']['baseline'])


class TestBrainWiring(unittest.TestCase):
    """End-to-end: an instantiated Brain has AllostaticLoad wired
    into SleepRegulator's load_provider slot."""

    def test_sleep_regulator_load_provider_calls_allostatic(self):
        from seagi.body.engine import Engine
        from seagi.brain.runtime import Brain
        engine = Engine()
        brain = Brain(engine=engine)
        # Provider slot is populated.
        self.assertIsNotNone(brain.sleep_regulator._load_provider)
        # Provider returns the same value as the organ's load().
        # Force a shift so a non-zero result confirms the wire.
        brain.allostatic_load.baselines['cortisol'] = (
            brain.allostatic_load.initial_baselines['cortisol']
            + 0.10)
        self.assertEqual(
            brain.sleep_regulator._load_provider(),
            brain.allostatic_load.load())

    def test_brain_tick_drives_allostatic_sampling(self):
        from seagi.body.engine import Engine
        from seagi.brain.runtime import Brain
        engine = Engine()
        brain = Brain(engine=engine)
        before = brain.allostatic_load.samples_observed
        for _ in range(5):
            brain.tick()
        self.assertEqual(
            brain.allostatic_load.samples_observed - before, 5)


if __name__ == '__main__':
    unittest.main()
