"""Step 0 — cross-organ integration tests.

The four organs (MetabolicDebt, AllostaticLoad, DiscriminabilityTracker,
AWM contraction, gate attenuation) all consume MetabolicDebt
providers and modulate each other through the SleepRegulator gate.
Unit tests cover each organ in isolation.  This file covers their
*composition*: the modulator chain biting together under the
scenarios the wiring plan was designed for.

Per [[seagi-vision-proportion]]: focused high-confidence bite checks,
not an exhaustive sweep of every acceptance item.
"""

from __future__ import annotations

import time
import unittest

from seagi.brain import EventKind, EventBus
from seagi.brain.events import (
    SubstrateWriteQueuedEvent, ChemistryEvent, RawPerceptEvent,
    AttendedPerceptEvent, ThoughtProducedEvent,
)
from seagi.brain.chemistry_types import CHANNELS
from seagi.body.engine import Engine
from seagi.brain.runtime import Brain, SLEEP_CONSOLIDATION_INTERVAL
from seagi.core.substrate import Concept
from seagi.brain.capabilities.metabolic_debt import (
    BOOTSTRAP_SEED_VALUE)


def _make_write(cycle: int = 1,
                  subject: str = 'a',
                  relation: str = 'causes',
                  object_: str = 'b') -> SubstrateWriteQueuedEvent:
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
        strength=0.1,
        write_reason='integration_test',
    )


# ---------------------------------------------------------------------
# (a) Headless-write loop triggers Phase S
# ---------------------------------------------------------------------


class TestWritesTriggerSleep(unittest.TestCase):
    """The whole reason Step 0 exists: a daemon that only writes
    substrate (reverie / Step 2 derivations / SVO ingestion) must
    eventually sleep.  This is the 3-day-no-sleep bug closed.
    """

    def test_substrate_writes_drive_sleep_onset(self):
        engine = Engine()
        brain = Brain(engine=engine)
        self.assertFalse(brain.sleep_regulator.is_asleep())
        # Bootstrap threshold = 50.  Bus order (audit Q1) ensures
        # MetabolicDebt updates BEFORE SleepRegulator reads within
        # a single dispatch, so write #51 sees debt=51 > 50 and
        # transitions exactly.  Fire writes one at a time and
        # assert the transition cycle.
        sleep_onset_cycle = None
        for cyc in range(1, 60):
            brain.bus.publish(_make_write(cycle=cyc))
            if brain.sleep_regulator.is_asleep():
                sleep_onset_cycle = cyc
                break
        self.assertEqual(
            sleep_onset_cycle, BOOTSTRAP_SEED_VALUE + 1,
            f"Sleep should fire exactly at write "
            f"{BOOTSTRAP_SEED_VALUE + 1}, fired at "
            f"{sleep_onset_cycle}; debt at trip = "
            f"{brain.metabolic_debt.debt}")


# ---------------------------------------------------------------------
# (k) Crash-sleep when both modulators max
# ---------------------------------------------------------------------


class TestCrashSleepBothMaxed(unittest.TestCase):
    """Both load_mod = 1.0 AND d_mod = 1.0 → effective_threshold = 0.
    The next substrate write triggers sleep immediately.  Doctrine-
    correct per Rule 2 (modulators must bite all the way; no hand-
    tuned residual floor).  Q9 lock."""

    def test_pegged_modulators_zero_threshold_crash_sleep(self):
        engine = Engine()
        brain = Brain(engine=engine)
        # Force first-episode-done so the bootstrap path is past.
        brain.metabolic_debt._has_logged_first_episode = True
        # Peg both modulators.
        brain.allostatic_load.load = lambda: 1.0
        brain.discriminability_tracker.d_modulation_provider = (
            lambda: 1.0)
        # Re-wire the SleepRegulator providers to these patched
        # methods (the regulator captured the originals at
        # construction).
        brain.sleep_regulator._load_provider = (
            brain.allostatic_load.load)
        brain.sleep_regulator._d_modulation_provider = (
            brain.discriminability_tracker.d_modulation_provider)
        # Single write: post-audit bus ordering (metabolic_debt
        # subscribes before sleep_regulator) means debt=1.0 by the
        # time the regulator handler runs.  debt_above=1.0 >
        # effective_threshold=0.0 → sleep_onset on write 1.
        # Rule-2 ceiling bites immediately, no hand-tuned slack.
        brain.bus.publish(_make_write(cycle=1))
        self.assertTrue(
            brain.sleep_regulator.is_asleep(),
            "Crash-sleep should fire on the first write when "
            "both modulators peg at 1.0")
        self.assertEqual(
            brain.sleep_regulator.last_effective_threshold, 0.0)


# ---------------------------------------------------------------------
# (j) Above-threshold percept survives at max debt
# ---------------------------------------------------------------------


class TestSurvivalBypassAtMaxDebt(unittest.TestCase):
    """At pegged debt the gate attenuates below-threshold percepts
    heavily, but above-threshold percepts (peer / M / I / novelty)
    pass at full magnitude.  Rule 5 (goal-directed channels stay
    reachable) preserved."""

    def test_peer_percept_unattenuated_at_max_debt(self):
        engine = Engine()
        brain = Brain(engine=engine)
        # Force MetabolicDebt to peg the gate's attenuation strength.
        brain.metabolic_debt.debt = 10000.0
        brain.metabolic_debt._wake_onset_baseline_debt = 0.0
        # debt_full_scale stays at BOOTSTRAP_SEED_VALUE = 50 (no
        # episode logged yet); ratio = 200 → clipped to 1.0.
        self.assertAlmostEqual(
            brain.gate._attenuation_strength(), 1.0, places=6)
        # Capture attended events.
        collected = []

        class _Sink:
            SUBSCRIPTIONS = (EventKind.ATTENDED_PERCEPT,)

            def handle(self, ev, bus):
                if isinstance(ev, AttendedPerceptEvent):
                    collected.append(ev)
        brain.bus.subscribe(_Sink.SUBSCRIPTIONS, _Sink())
        # Peer percept — W_PEER × 1.0 = 1.0, well above any
        # chemistry-modulated threshold (capped at 0.5).
        brain.bus.publish(RawPerceptEvent(
            kind=EventKind.RAW_PERCEPT,
            cycle=0,
            timestamp=time.time(),
            source_capability='test',
            origin='peer',
            origin_detail='test',
            modality='text',
            raw_text='danger',
            payload={'danger': 1.0}))
        # Find the gate-emitted attended (not the bus echo from
        # any other capability).
        gate_attendeds = [
            ev for ev in collected
            if ev.source_capability == 'thalamic_gate']
        self.assertGreaterEqual(len(gate_attendeds), 1)
        # Full magnitude preserved.
        self.assertEqual(gate_attendeds[0].salience, 1.0)


# ---------------------------------------------------------------------
# AWM contraction at high debt
# ---------------------------------------------------------------------


class TestAWMContractsUnderDebt(unittest.TestCase):

    def test_high_debt_contracts_awm_to_floor(self):
        engine = Engine()
        brain = Brain(engine=engine)
        # No inference yet → cold-start floor = 3.
        # Peg debt.
        brain.metabolic_debt.debt = 10000.0
        brain.metabolic_debt._wake_onset_baseline_debt = 0.0
        self.assertEqual(
            brain.awm.effective_capacity(), 3)


# ---------------------------------------------------------------------
# (b) Chronic-stress drift recorded by AllostaticLoad
# ---------------------------------------------------------------------


class TestChronicStressDrift(unittest.TestCase):

    def test_sustained_high_cortisol_drifts_baseline_up(self):
        engine = Engine()
        brain = Brain(engine=engine)
        initial = brain.allostatic_load.baselines['cortisol']
        # Pin global cortisol high, run many ticks + several
        # sleep_onsets.
        for episode in range(5):
            brain.chemistry.global_state['cortisol'] = 0.60
            # Fill tonic window for cortisol.
            for _ in range(50):
                brain.allostatic_load.tick()
            brain.bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=episode,
                timestamp=time.time(),
                source_capability='test',
                origin='internal',
                origin_detail='test',
                chemistry_kind='sleep_onset',
                magnitude=1.0))
        self.assertGreater(
            brain.allostatic_load.baselines['cortisol'], initial)
        # Load reflects the drift.
        self.assertGreater(brain.allostatic_load.load(), 0.0)


# ---------------------------------------------------------------------
# (c+e) Phase S during sleep clears debt
# ---------------------------------------------------------------------


class TestPhaseSClearsDebt(unittest.TestCase):

    def test_sleep_cycle_clears_debt_and_logs_clearance(self):
        engine = Engine()
        sub = engine.substrate
        # Seed a coherent triangle so reinforce_coherent_edges has
        # something to do during consolidation.
        for nm in ('a', 'b', 'c'):
            sub.add_concept(Concept(name=nm))
        # Composable relation (is_a, is_a)→is_a so the triangle
        # corroborates under the strict composition-gated coherence.
        for (s, t) in [('a', 'b'), ('b', 'a'), ('b', 'c'),
                          ('c', 'b'), ('a', 'c'), ('c', 'a')]:
            sub.add_edge(s, t, 'is_a', strength=0.5, cycle=0)
        brain = Brain(engine=engine)
        # Push debt up.
        for cyc in range(1, 60):
            brain.bus.publish(_make_write(cycle=cyc))
        peak_debt = brain.metabolic_debt.debt
        self.assertGreater(peak_debt, 0.0)
        self.assertTrue(brain.sleep_regulator.is_asleep())
        # Run enough ticks for at least one consolidation pass.
        for _ in range(SLEEP_CONSOLIDATION_INTERVAL + 5):
            brain.tick()
        # Phase S touched the substrate AND recorded clearance.
        self.assertGreaterEqual(brain.consolidations_run, 1)
        self.assertGreater(
            brain.metabolic_debt.consolidations_recorded, 0)
        self.assertGreater(
            brain.metabolic_debt.total_clearance_recorded, 0.0)
        # Debt fell below the peak.
        self.assertLess(brain.metabolic_debt.debt, peak_debt)


# ---------------------------------------------------------------------
# (h) Save/load round trip across all new persisted state
# ---------------------------------------------------------------------


class TestSaveLoadRoundTrip(unittest.TestCase):

    def test_all_new_step0_state_round_trips(self):
        engine = Engine()
        brain = Brain(engine=engine)
        # Drive non-trivial state across all four organs.
        for cyc in range(1, 60):
            brain.bus.publish(_make_write(cycle=cyc))
        # AllostaticLoad drift via a sleep_onset (the metabolic
        # debt's onset above will produce one too).
        brain.chemistry.global_state['cortisol'] = 0.55
        for _ in range(50):
            brain.allostatic_load.tick()
        brain.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE,
            cycle=999,
            timestamp=time.time(),
            source_capability='test',
            origin='internal',
            origin_detail='test',
            chemistry_kind='sleep_onset',
            magnitude=1.0))
        # AWM inference recording.
        brain.bus.publish(ThoughtProducedEvent(
            kind=EventKind.THOUGHT_PRODUCED,
            cycle=1000,
            timestamp=time.time(),
            source_capability='cortical',
            origin='internal',
            origin_detail='test',
            focal='x', relation='is_a', target='y',
            confidence=0.7,
            method='inference',
            text='',
            chain_depth=3))
        # Discriminability baseline window.
        brain.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE,
            cycle=1001,
            timestamp=time.time(),
            source_capability='test',
            origin='internal',
            origin_detail='test',
            chemistry_kind='wake_onset',
            magnitude=1.0))
        # Snapshot ALL state.
        snap = brain.to_dict()
        # Capture key values for comparison.
        debt_pre = brain.metabolic_debt.debt
        baseline_pre = (
            brain.allostatic_load.baselines['cortisol'])
        deque_pre = list(brain.awm._recent_node_counts)
        wake_d_baseline_pre = (
            brain.discriminability_tracker.wake_onset_D_baseline)
        # Fresh brain, load snapshot.
        fresh_engine = Engine()
        fresh = Brain(engine=fresh_engine)
        fresh.load_personality(snap)
        self.assertAlmostEqual(
            fresh.metabolic_debt.debt, debt_pre, places=6)
        self.assertAlmostEqual(
            fresh.allostatic_load.baselines['cortisol'],
            baseline_pre, places=6)
        self.assertEqual(
            list(fresh.awm._recent_node_counts), deque_pre)
        self.assertAlmostEqual(
            fresh.discriminability_tracker.wake_onset_D_baseline,
            wake_d_baseline_pre, places=6)


# ---------------------------------------------------------------------
# (g) Cold-start safety: no NaN, no div-by-zero across all organs
# ---------------------------------------------------------------------


class TestColdStartSafety(unittest.TestCase):
    """A fresh Brain with no events fired must produce sane,
    non-NaN values from every modulator query."""

    def test_fresh_brain_modulators_all_safe(self):
        import math
        engine = Engine()
        brain = Brain(engine=engine)

        # MetabolicDebt
        self.assertEqual(brain.metabolic_debt.debt, 0.0)
        self.assertEqual(
            brain.metabolic_debt.debt_full_scale,
            float(BOOTSTRAP_SEED_VALUE))

        # AllostaticLoad
        load = brain.allostatic_load.load()
        self.assertEqual(load, 0.0)
        self.assertFalse(math.isnan(load))

        # DiscriminabilityTracker
        d_mod = (
            brain.discriminability_tracker.d_modulation_provider())
        self.assertEqual(d_mod, 0.0)
        self.assertFalse(math.isnan(d_mod))

        # AWM
        eff = brain.awm.effective_capacity()
        self.assertEqual(eff, brain.awm.capacity)
        floor = brain.awm.awm_capacity_floor()
        self.assertEqual(floor, 3)

        # Thalamic gate
        att = brain.gate._attenuation_strength()
        self.assertEqual(att, 0.0)
        threshold = brain.gate._threshold_now()
        self.assertLessEqual(threshold, 0.5)
        self.assertFalse(math.isnan(threshold))

        # Sleep gate stays wake-default before any writes.
        self.assertFalse(brain.sleep_regulator.is_asleep())


if __name__ == '__main__':
    unittest.main()
