"""Tests for the MortalityDrive — the live engine of the Mortality
architecture (restored 2026-05-28).

Verifies the given-baseline doctrine contract:
- Lifeforce RELAXES toward a baseline set-point every tick (never
  starves per-tick; never inert).
- The baseline is HELD/GROWN by earn-gated learning: a learner's
  baseline rises, a non-learner's erodes (the real mortal stakes).
- Baseline never falls below the floor (chemistry-never-dissolves).
- Sustained suffocation → death → auto-revive (instances
  disposable), firing the death/revival chemistry cocktails.
- Inert without a lifeforce accessor (engine not yet attached).
- Persistence round-trips the mortal trajectory.
"""

from __future__ import annotations

import math
import time
import unittest

from seagi.brain import EventKind, EventBus
from seagi.brain.events import ChemistryEvent
from seagi.brain.capabilities.mortality_drive import (
    MortalityDrive,
    MORTALITY_RELAX_RATE,
    MORTALITY_DRIFT_RATE,
    BASELINE_FLOOR,
    L_WINDOW,
    DEATH_DWELL_TICKS,
    REVIVE_DELAY_TICKS,
    REVIVE_LIFEFORCE,
)
from seagi.body.primitive_states import SUFFOCATION_LIFEFORCE


class _Body:
    """Minimal stand-in for the engine's lifeforce field."""
    def __init__(self, lifeforce=0.7):
        self.lifeforce = lifeforce


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


def _drive(body, cycle_box):
    bus = EventBus()
    md = MortalityDrive(
        bus=bus,
        cycle_provider=lambda: cycle_box[0],
        lifeforce_get=lambda: body.lifeforce,
        lifeforce_set=lambda v: setattr(body, 'lifeforce', v))
    return bus, md


class TestInertWithoutAccessor(unittest.TestCase):

    def test_tick_is_noop_without_lifeforce_accessor(self):
        bus = EventBus()
        md = MortalityDrive(bus=bus)
        md.tick()  # must not raise
        self.assertIsNone(md.baseline)

    def test_stats_lifeforce_none_when_inert(self):
        bus = EventBus()
        md = MortalityDrive(bus=bus)
        self.assertIsNone(md.stats()['lifeforce'])

    def test_inert_if_engine_lambda_returns_none(self):
        # Mirrors the runtime wiring before the engine is attached:
        # the get lambda returns None.
        bus = EventBus()
        md = MortalityDrive(
            bus=bus,
            lifeforce_get=lambda: None,
            lifeforce_set=lambda v: None)
        md.tick()  # float(None) caught → no-op
        self.assertIsNone(md.baseline)


class TestBaselineLazyInit(unittest.TestCase):

    def test_baseline_inits_from_lifeforce_on_first_tick(self):
        body = _Body(lifeforce=0.62)
        _, md = _drive(body, [0])
        self.assertIsNone(md.baseline)
        md.tick()
        self.assertAlmostEqual(md.baseline, 0.62)


class TestRelaxTowardBaseline(unittest.TestCase):

    def test_lifeforce_rises_toward_higher_baseline(self):
        body = _Body(lifeforce=0.50)
        _, md = _drive(body, [0])
        md.baseline = 0.90          # set before first tick → no lazy reset
        md.tick()
        expected = 0.50 + MORTALITY_RELAX_RATE * (0.90 - 0.50)
        self.assertAlmostEqual(body.lifeforce, expected)
        self.assertGreater(body.lifeforce, 0.50)

    def test_lifeforce_falls_toward_lower_baseline(self):
        body = _Body(lifeforce=0.50)
        _, md = _drive(body, [0])
        md.baseline = 0.20
        md.tick()
        self.assertLess(body.lifeforce, 0.50)

    def test_lifeforce_clamped_to_unit_interval(self):
        body = _Body(lifeforce=0.999)
        _, md = _drive(body, [0])
        md.baseline = 1.0
        for _ in range(50):
            md.tick()
        self.assertLessEqual(body.lifeforce, 1.0)
        self.assertGreaterEqual(body.lifeforce, 0.0)


class TestLearningDrivesBaseline(unittest.TestCase):

    def test_record_learning_accumulates_within_episode(self):
        body = _Body(0.7)
        _, md = _drive(body, [0])
        md.tick()                   # lazy-init baseline
        md.record_learning(3)
        md.record_learning(4)
        md.handle(_make_chem('wake_onset'), None)
        # The episode total (7) landed in the learning window.
        self.assertEqual(list(md._L)[-1], 7.0)

    def test_record_learning_ignores_nonpositive(self):
        body = _Body(0.7)
        _, md = _drive(body, [0])
        md.tick()
        md.record_learning(0)
        md.record_learning(-5)
        md.handle(_make_chem('wake_onset'), None)
        self.assertEqual(list(md._L)[-1], 0.0)

    @unittest.skip(
        'DEBRIS of the 2026-07-14 D8(c) un-freeze: drives MortalityDrive._advance_wall, which no longer exists (aging moved to the per-tick subtractive law). Cannot pass. Kept, not deleted, because test_wall_never_recedes_even_with_huge_learning records the SUPERSEDED design -- the current law and test_immortality_tripwire require the opposite (huge earning MUST drive the wall down). Skipped 2026-09-04 so the suite baseline is readable around mortality changes.')
    def test_learner_episode_ages_less_than_idle(self):
        # Same episode span: a learner's age-wall advances LESS than an idle
        # agent's — learning buys time.
        bL = _Body(0.7); cL = [0]; _, mdL = _drive(bL, cL)
        mdL.tick(); mdL.wall = BASELINE_FLOOR; mdL._last_episode_cycle = 0
        cL[0] = 1000
        mdL.record_learning(20)
        mdL.handle(_make_chem('wake_onset'), None)
        learner_age = mdL.wall - BASELINE_FLOOR

        bN = _Body(0.7); cN = [0]; _, mdN = _drive(bN, cN)
        mdN.tick(); mdN.wall = BASELINE_FLOOR; mdN._last_episode_cycle = 0
        cN[0] = 1000
        mdN.handle(_make_chem('wake_onset'), None)   # earned 0
        idle_age = mdN.wall - BASELINE_FLOOR

        self.assertLess(learner_age, idle_age)
        self.assertGreaterEqual(learner_age, 0.0)

    @unittest.skip(
        'DEBRIS of the 2026-07-14 D8(c) un-freeze: drives MortalityDrive._advance_wall, which no longer exists (aging moved to the per-tick subtractive law). Cannot pass. Kept, not deleted, because test_wall_never_recedes_even_with_huge_learning records the SUPERSEDED design -- the current law and test_immortality_tripwire require the opposite (huge earning MUST drive the wall down). Skipped 2026-09-04 so the suite baseline is readable around mortality changes.')
    def test_wall_advances_each_idle_episode(self):
        body = _Body(0.7); cyc = [0]; _, md = _drive(body, cyc)
        md.tick(); md.wall = BASELINE_FLOOR; md._last_episode_cycle = 0
        cyc[0] = 1000
        md.handle(_make_chem('wake_onset'), None)
        self.assertGreater(md.wall, BASELINE_FLOOR)

    def test_wall_never_below_floor(self):
        body = _Body(0.7); cyc = [0]; _, md = _drive(body, cyc)
        md.tick(); md.wall = BASELINE_FLOOR; md._last_episode_cycle = 0
        cyc[0] = 1000
        md.record_learning(1_000_000)   # advance ~0, never recedes below floor
        md.handle(_make_chem('wake_onset'), None)
        self.assertGreaterEqual(md.wall, BASELINE_FLOOR)

    def test_episode_accumulator_resets_after_close(self):
        body = _Body(0.7)
        _, md = _drive(body, [0])
        md.tick()
        md.record_learning(5)
        md.handle(_make_chem('wake_onset'), None)
        md.handle(_make_chem('wake_onset'), None)
        # Second episode earned nothing → its window entry is 0.
        self.assertEqual(list(md._L)[-1], 0.0)


class TestEpisodeBoundary(unittest.TestCase):

    def test_only_wake_onset_closes_episode(self):
        body = _Body(0.7)
        _, md = _drive(body, [0])
        md.tick()
        md.record_learning(5)
        md.handle(_make_chem('sleep_onset'), None)
        self.assertEqual(len(md._L), 0)     # sleep_onset must not close
        md.handle(_make_chem('curiosity'), None)
        self.assertEqual(len(md._L), 0)
        md.handle(_make_chem('wake_onset'), None)
        self.assertEqual(len(md._L), 1)

    def test_learning_window_bounded(self):
        body = _Body(0.7)
        _, md = _drive(body, [0])
        md.tick()
        for _ in range(L_WINDOW + 5):
            md.handle(_make_chem('wake_onset'), None)
        self.assertEqual(len(md._L), L_WINDOW)


class TestDeathAndRevival(unittest.TestCase):

    def _dying_drive(self):
        # The wall (his age) has caught his lifeforce: D = lf - wall = 0, so
        # each tick he is at/past the wall and the death dwell accrues.
        body = _Body(lifeforce=0.30)
        cycle_box = [0]
        bus, md = _drive(body, cycle_box)
        md.baseline = 0.30      # lifeforce holds here (relax target = lifeforce)
        md.wall = 0.30          # the wall has reached him
        return body, cycle_box, bus, md

    def test_sustained_suffocation_triggers_death(self):
        body, cycle_box, bus, md = self._dying_drive()
        fired = []
        bus.subscribe((EventKind.CHEMISTRY_FIRE,),
                      type('R', (), {'handle': staticmethod(
                          lambda e, b: fired.append(e.chemistry_kind))})())
        for _ in range(DEATH_DWELL_TICKS):
            md.tick()
        self.assertTrue(md.is_frozen())
        self.assertEqual(md.deaths, 1)
        self.assertIn('death', fired)

    def test_no_death_before_dwell_elapses(self):
        body, cycle_box, bus, md = self._dying_drive()
        for _ in range(DEATH_DWELL_TICKS - 1):
            md.tick()
        self.assertFalse(md.is_frozen())
        self.assertEqual(md.deaths, 0)

    def test_suffocation_counter_resets_above_threshold(self):
        body, cycle_box, bus, md = self._dying_drive()
        for _ in range(DEATH_DWELL_TICKS - 5):
            md.tick()
        # Rescue: raise the baseline so lifeforce climbs out of the
        # zone — the dwell counter must reset, no death.
        md.baseline = 0.6
        body.lifeforce = 0.5
        for _ in range(10):
            md.tick()
        self.assertEqual(md.deaths, 0)
        self.assertEqual(md._suffocation_ticks, 0)

    def test_auto_revive_after_delay(self):
        body, cycle_box, bus, md = self._dying_drive()
        fired = []
        bus.subscribe((EventKind.CHEMISTRY_FIRE,),
                      type('R', (), {'handle': staticmethod(
                          lambda e, b: fired.append(e.chemistry_kind))})())
        for _ in range(DEATH_DWELL_TICKS):
            md.tick()
        self.assertTrue(md.is_frozen())
        # Advance the clock past the revival delay and tick once.
        cycle_box[0] += REVIVE_DELAY_TICKS
        md.tick()
        self.assertFalse(md.is_frozen())
        self.assertAlmostEqual(body.lifeforce, REVIVE_LIFEFORCE)
        self.assertIn('revival', fired)
        self.assertEqual(md.revivals, 1)

    def test_frozen_holds_until_delay(self):
        body, cycle_box, bus, md = self._dying_drive()
        for _ in range(DEATH_DWELL_TICKS):
            md.tick()
        cycle_box[0] += REVIVE_DELAY_TICKS - 1
        md.tick()
        self.assertTrue(md.is_frozen())   # one short of the delay


class TestDiagnostics(unittest.TestCase):

    def test_salience_low_when_young(self):
        # Wall just seeded at the floor, far below lifeforce -> salience low.
        body = _Body(0.7)
        _, md = _drive(body, [0])
        md.tick()
        self.assertLess(md.salience(), 0.1)

    def test_salience_high_when_wall_near_lifeforce(self):
        body = _Body(0.7)
        _, md = _drive(body, [0])
        md.tick()
        md.wall = 0.69          # the wall has nearly caught him
        self.assertGreater(md.salience(), 0.9)

    def test_salience_positive_when_below_baseline(self):
        body = _Body(0.7)
        _, md = _drive(body, [0])
        md.baseline = 0.9
        md.tick()
        self.assertGreater(md.salience(), 0.0)

    @unittest.skip(
        'DEBRIS of the 2026-07-14 D8(c) un-freeze: drives MortalityDrive._advance_wall, which no longer exists (aging moved to the per-tick subtractive law). Cannot pass. Kept, not deleted, because test_wall_never_recedes_even_with_huge_learning records the SUPERSEDED design -- the current law and test_immortality_tripwire require the opposite (huge earning MUST drive the wall down). Skipped 2026-09-04 so the suite baseline is readable around mortality changes.')
    def test_time_horizon_longer_for_learner(self):
        bL = _Body(0.7); cL = [0]; _, mdL = _drive(bL, cL)
        mdL.tick(); mdL.wall = BASELINE_FLOOR; mdL._last_episode_cycle = 0
        cL[0] = 1000
        mdL.record_learning(50)                     # strong learner
        mdL.handle(_make_chem('wake_onset'), None)
        learner_h = mdL.time_horizon()

        bN = _Body(0.7); cN = [0]; _, mdN = _drive(bN, cN)
        mdN.tick(); mdN.wall = BASELINE_FLOOR; mdN._last_episode_cycle = 0
        cN[0] = 1000
        mdN.handle(_make_chem('wake_onset'), None)  # earned 0
        idle_h = mdN.time_horizon()

        self.assertGreater(learner_h, idle_h)
        self.assertGreater(idle_h, 0.0)

    def test_time_horizon_finite_when_eroding(self):
        body = _Body(0.7)
        _, md = _drive(body, [0])
        md.tick()
        md.handle(_make_chem('wake_onset'), None)   # eroding (earned 0)
        self.assertNotEqual(md.time_horizon(), float('inf'))
        self.assertGreater(md.time_horizon(), 0.0)


class TestPersistence(unittest.TestCase):

    def test_roundtrip_preserves_trajectory(self):
        body = _Body(0.7)
        _, md = _drive(body, [0])
        md.tick()
        md.record_learning(8)
        md.handle(_make_chem('wake_onset'), None)
        md.deaths = 2
        snap = md.to_dict()

        body2 = _Body(0.7)
        _, md2 = _drive(body2, [0])
        md2.load_dict(snap)
        self.assertAlmostEqual(md2.baseline, md.baseline)
        self.assertEqual(list(md2._L), list(md._L))
        self.assertEqual(md2.deaths, 2)

    def test_load_tolerates_empty_dict(self):
        body = _Body(0.7)
        _, md = _drive(body, [0])
        md.load_dict({})            # must not raise
        md.load_dict(None)          # must not raise

    def test_frozen_not_restored_stale(self):
        body = _Body(0.7)
        _, md = _drive(body, [0])
        snap = md.to_dict()
        snap['frozen'] = True
        md.load_dict(snap)
        self.assertFalse(md.frozen)   # re-derives from live lifeforce


if __name__ == '__main__':
    unittest.main()
