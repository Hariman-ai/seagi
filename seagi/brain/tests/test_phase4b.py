"""Phase 4b tests — threat, arbitration, value, prediction.

Capabilities under test:
  Cerebellum    — lateral + vermis prediction errors
  ACC           — conflict / mismatch on PE + peer contradiction
  VTA + LC      — NT modulators on PE + threat
  ValueLandscape— focal value map updated by PE + chemistry
  Amygdala      — threat detection + override capability claim
  BasalGanglia  — 3-loop claim arbitration
  SourceMonitor — wildcard event auditor
  AnteriorPFC   — metacog + prospective intent
  TimePerception— multi-scale tracker
  BodySchema    — operational envelope query API
  CorpusCallosum— two-stream + integration
  Writer        — journaled single-writer substrate
"""

import time
import unittest

from seagi.body.engine import Engine
from seagi.core.substrate import Concept
from seagi.brain import (
    Brain, EventKind, EventBus,
    AttendedPerceptEvent,
    ChemistryEvent,
    InteroceptionEvent,
    ThoughtProducedEvent,
    PredictionErrorEvent,
    ThreatDetectedEvent,
    ConflictDetectedEvent,
    ValueUpdatedEvent,
    ArbitrationDecidedEvent,
    CapabilityClaimEvent,
    SubstrateWriteQueuedEvent,
)
from seagi.brain.capabilities.cerebellum import Cerebellum
from seagi.brain.capabilities.acc import (
    AnteriorCingulateCortex,
)
from seagi.brain.capabilities.neuromodulators import (
    VTA, LocusCoeruleus,
)
from seagi.brain.capabilities.value_landscape import (
    ValueLandscape,
)
from seagi.brain.capabilities.amygdala import Amygdala
from seagi.brain.capabilities.basal_ganglia import (
    BasalGanglia, VALID_LOOPS,
)
from seagi.brain.capabilities.auxiliary import (
    SourceMonitor, AnteriorPFC, TimePerception,
    BodySchema, CorpusCallosum,
)
from seagi.brain.capabilities.writer import (
    JournaledSubstrateWriter,
)


def _seed_substrate(engine, concepts, edges):
    for n in concepts:
        engine.substrate.add_concept(Concept(name=n))
    for (s, r, o, strength) in edges:
        engine.substrate.add_edge(
            source=s, target=o, relation_name=r,
            strength=strength)


def _make_attended(cycle=10, focals=('fire',), origin='peer',
                       origin_detail='harald', m=0.2, i=0.2,
                       novelty=0.5, salience=0.8,
                       raw_text=''):
    return AttendedPerceptEvent(
        kind=EventKind.ATTENDED_PERCEPT,
        cycle=cycle,
        timestamp=time.time(),
        source_capability='test',
        origin=origin,
        origin_detail=origin_detail,
        focals=list(focals),
        payload={f: {} for f in focals},
        raw_text=raw_text,
        modality='text',
        salience=salience,
        novelty=novelty,
        m_content=m,
        i_content=i,
        threshold_used=0.3,
    )


# ---------------------------------------------------------------
# Cerebellum
# ---------------------------------------------------------------


class TestCerebellum(unittest.TestCase):

    def setUp(self):
        self.bus = EventBus()
        self.cer = Cerebellum(
            bus=self.bus, cycle_provider=lambda: 1)
        self.bus.subscribe(self.cer.SUBSCRIPTIONS, self.cer)
        self.pes = []
        self.bus.subscribe(
            (EventKind.PREDICTION_ERROR,),
            lambda ev, b: self.pes.append(ev))

    def test_no_pe_without_prior_observations(self):
        self.bus.publish(_make_attended(focals=('a',), cycle=1))
        self.assertEqual(self.pes, [])

    def test_pe_fires_when_prediction_misses(self):
        # Build a→b transition twice (so it counts as predicted).
        self.bus.publish(_make_attended(focals=('a',), cycle=1))
        self.bus.publish(_make_attended(focals=('b',), cycle=2))
        self.bus.publish(_make_attended(focals=('a',), cycle=3))
        self.bus.publish(_make_attended(focals=('b',), cycle=4))
        self.pes.clear()
        # Now a → c (miss).
        self.bus.publish(_make_attended(focals=('a',), cycle=5))
        self.bus.publish(_make_attended(focals=('c',), cycle=6))
        cog_pes = [e for e in self.pes
                      if e.error_kind == 'cognitive']
        self.assertGreaterEqual(len(cog_pes), 1)

    def test_affective_pe_on_lifeforce_divergence(self):
        # Feed monotone trajectory then a divergence.
        for cycle, lf in [(1, 0.9), (2, 0.85), (3, 0.30)]:
            self.bus.publish(InteroceptionEvent(
                kind=EventKind.INTEROCEPTION,
                cycle=cycle,
                timestamp=time.time(),
                source_capability='test',
                origin='internal',
                origin_detail='body',
                felt_state='waning',
                lifeforce=lf,
                body_integrity=1.0,
                delta_lifeforce=0.0,
                narrative=''))
        aff_pes = [e for e in self.pes
                      if e.error_kind == 'affective']
        self.assertGreaterEqual(len(aff_pes), 1)


# ---------------------------------------------------------------
# Phase I.1a — multi-channel vermis
# ---------------------------------------------------------------


def _make_intero(cycle, lifeforce=0.5):
    return InteroceptionEvent(
        kind=EventKind.INTEROCEPTION,
        cycle=cycle,
        timestamp=time.time(),
        source_capability='test',
        origin='internal',
        origin_detail='body',
        felt_state='settled',
        lifeforce=lifeforce,
        body_integrity=1.0,
        delta_lifeforce=0.0,
        narrative='')


class TestPhaseI1aMultiChannelVermis(unittest.TestCase):
    """Vermis predicts each of the 8 chemistry channels when a
    chemistry_provider is wired.  Per-channel thresholds keep
    total PE volume comparable to lifeforce-only mode."""

    def setUp(self):
        self.bus = EventBus()
        # Mutable dict the test fixture mutates to simulate
        # chemistry trajectories.
        self.state = {
            'norepinephrine': 0.20,
            'acetylcholine':  0.50,
            'gaba':           0.50,
            'dopamine':       0.30,
            'endorphins':     0.10,
            'oxytocin':       0.20,
            'serotonin':      0.50,
            'cortisol':       0.10,
        }
        self.cer = Cerebellum(
            bus=self.bus,
            cycle_provider=lambda: 1,
            chemistry_provider=lambda: dict(self.state))
        self.bus.subscribe(self.cer.SUBSCRIPTIONS, self.cer)
        self.pes = []
        self.bus.subscribe(
            (EventKind.PREDICTION_ERROR,),
            lambda ev, b: self.pes.append(ev))

    def test_no_pe_with_only_one_sample(self):
        # Single interoception — not enough to extrapolate.
        self.bus.publish(_make_intero(cycle=1))
        aff = [e for e in self.pes
                  if e.error_kind == 'affective']
        self.assertEqual(aff, [])

    def test_per_channel_pe_on_cortisol_spike(self):
        # Two stable samples then a big cortisol jump.  Cortisol
        # is the tightest threshold (0.06) and the slowest
        # channel — meaningful jump should fire on cortisol only.
        self.bus.publish(_make_intero(cycle=1))
        self.bus.publish(_make_intero(cycle=2))
        self.state['cortisol'] = 0.40  # big jump
        self.bus.publish(_make_intero(cycle=3))
        aff = [e for e in self.pes
                  if e.error_kind == 'affective']
        # At least one fire, all keyed to cortisol via focal.
        self.assertGreaterEqual(len(aff), 1)
        self.assertTrue(
            all(e.focal == 'chem.cortisol' for e in aff),
            f"expected cortisol-only fires, got "
            f"{[e.focal for e in aff]}")

    def test_channels_fire_independently(self):
        # Distinct channel spikes produce distinct PEs.
        self.bus.publish(_make_intero(cycle=1))
        self.bus.publish(_make_intero(cycle=2))
        self.state['dopamine'] = 0.60     # +0.30 vs predicted
        self.state['cortisol'] = 0.30     # +0.20 vs predicted
        self.bus.publish(_make_intero(cycle=3))
        focals = {e.focal for e in self.pes
                      if e.error_kind == 'affective'}
        self.assertIn('chem.dopamine', focals)
        self.assertIn('chem.cortisol', focals)

    def test_small_drift_below_threshold_does_not_fire(self):
        # Slow gentle drift under every channel's threshold.
        self.bus.publish(_make_intero(cycle=1))
        self.bus.publish(_make_intero(cycle=2))
        # Tiny perturbation only.
        self.state['oxytocin'] = 0.21
        self.bus.publish(_make_intero(cycle=3))
        aff = [e for e in self.pes
                  if e.error_kind == 'affective']
        self.assertEqual(aff, [],
            f"expected silence on sub-threshold drift, got "
            f"{[(e.focal, e.magnitude) for e in aff]}")

    def test_pe_carries_predicted_and_actual(self):
        # Predictor is linear extrapolation from last two
        # samples.  After (0.20, 0.20), predicted=0.20; jump to
        # 0.50 → magnitude 0.30 > NE threshold 0.18.
        self.state['norepinephrine'] = 0.20
        self.bus.publish(_make_intero(cycle=1))
        self.bus.publish(_make_intero(cycle=2))
        self.state['norepinephrine'] = 0.50
        self.bus.publish(_make_intero(cycle=3))
        ne_pes = [e for e in self.pes
                       if e.focal == 'chem.norepinephrine']
        self.assertGreaterEqual(len(ne_pes), 1)
        ev = ne_pes[0]
        self.assertAlmostEqual(ev.predicted, 0.20, places=3)
        self.assertAlmostEqual(ev.actual, 0.50, places=3)
        self.assertEqual(ev.sign, 1.0)   # actual > predicted

    def test_provider_exception_is_swallowed(self):
        # Misbehaving provider should not crash the cerebellum.
        def bad():
            raise RuntimeError("chemistry offline")
        cer = Cerebellum(
            bus=self.bus,
            cycle_provider=lambda: 1,
            chemistry_provider=bad)
        # Don't subscribe the failing one to the bus (we just
        # call the handler directly to verify no raise).
        cer._on_interoception(_make_intero(cycle=1), self.bus)
        # No PE; no exception.
        self.assertTrue(True)

    def test_diagnostics_track_per_channel_counts(self):
        self.bus.publish(_make_intero(cycle=1))
        self.bus.publish(_make_intero(cycle=2))
        self.state['cortisol'] = 0.40
        self.bus.publish(_make_intero(cycle=3))
        stats = self.cer.stats()
        self.assertIn('cortisol', stats['channel_errors_fired'])
        self.assertGreaterEqual(
            stats['channel_errors_fired']['cortisol'], 1)
        self.assertGreaterEqual(stats['channels_tracked'], 8)

    def test_lifeforce_fallback_when_no_provider(self):
        # Without a chemistry_provider, cerebellum keeps Phase 4b
        # lifeforce behaviour and fires on a 'self' focal.
        bus = EventBus()
        cer = Cerebellum(
            bus=bus, cycle_provider=lambda: 1)
        bus.subscribe(cer.SUBSCRIPTIONS, cer)
        pes = []
        bus.subscribe(
            (EventKind.PREDICTION_ERROR,),
            lambda ev, b: pes.append(ev))
        for cycle, lf in [(1, 0.9), (2, 0.85), (3, 0.30)]:
            bus.publish(_make_intero(cycle=cycle, lifeforce=lf))
        aff = [e for e in pes if e.error_kind == 'affective']
        self.assertGreaterEqual(len(aff), 1)
        self.assertTrue(
            all(e.focal == 'self' for e in aff),
            "fallback path should keep 'self' focal")

    def test_chemistry_fire_drives_vermis(self):
        # After diagnostic probe found vermis silent in steady
        # state (insula only fires INTEROCEPTION on body shifts),
        # vermis must also sample on CHEMISTRY_FIRE.
        self.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE, cycle=1,
            timestamp=time.time(), source_capability='test',
            origin='internal', origin_detail='probe',
            chemistry_kind='curiosity', magnitude=0.1))
        self.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE, cycle=2,
            timestamp=time.time(), source_capability='test',
            origin='internal', origin_detail='probe',
            chemistry_kind='curiosity', magnitude=0.1))
        # Spike cortisol and fire another chemistry event.
        self.state['cortisol'] = 0.40
        self.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE, cycle=3,
            timestamp=time.time(), source_capability='test',
            origin='internal', origin_detail='probe',
            chemistry_kind='curiosity', magnitude=0.1))
        cortisol_pes = [
            e for e in self.pes
            if e.focal == 'chem.cortisol'
            and e.error_kind == 'affective']
        self.assertGreaterEqual(len(cortisol_pes), 1)

    def test_chemistry_fire_skipped_without_provider(self):
        # No chemistry_provider → CHEMISTRY_FIRE must be a no-op.
        bus = EventBus()
        cer = Cerebellum(bus=bus, cycle_provider=lambda: 1)
        bus.subscribe(cer.SUBSCRIPTIONS, cer)
        pes = []
        bus.subscribe(
            (EventKind.PREDICTION_ERROR,),
            lambda ev, b: pes.append(ev))
        for c in range(1, 5):
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE, cycle=c,
                timestamp=time.time(), source_capability='test',
                origin='internal', origin_detail='probe',
                chemistry_kind='curiosity', magnitude=0.1))
        affective = [e for e in pes if e.error_kind == 'affective']
        self.assertEqual(affective, [])

    def test_brain_wires_chemistry_provider(self):
        # End-to-end: real Brain initializes cerebellum with the
        # chemistry provider lambda.  A surprising cortisol
        # change in the chemistry state should produce
        # cortisol-keyed affective PE on the next interoception.
        from seagi.body.engine import Engine
        engine = Engine()
        brain = Brain(engine=engine)
        pes = []
        brain.bus.subscribe(
            (EventKind.PREDICTION_ERROR,),
            lambda ev, b: pes.append(ev))
        # Seed two stable interoceptions to build history at
        # baseline chemistry.
        brain.bus.publish(_make_intero(cycle=1))
        brain.bus.publish(_make_intero(cycle=2))
        # Spike cortisol on the actual chemistry engine.
        brain.chemistry.global_state['cortisol'] = 0.50
        brain.bus.publish(_make_intero(cycle=3))
        cortisol_pes = [e for e in pes
                            if e.focal == 'chem.cortisol']
        self.assertGreaterEqual(len(cortisol_pes), 1)


# ---------------------------------------------------------------
# Phase I.1b — error-driven transition reinforcement
# ---------------------------------------------------------------


class TestPhaseI1bTransitionReinforcement(unittest.TestCase):
    """Lateral transitions are float-weighted and self-pruning:
    hit reinforces by observation, miss weakens, unused fade,
    floor prunes."""

    def setUp(self):
        self.bus = EventBus()
        self.cer = Cerebellum(
            bus=self.bus, cycle_provider=lambda: 1)
        self.bus.subscribe(self.cer.SUBSCRIPTIONS, self.cer)
        self.pes = []
        self.bus.subscribe(
            (EventKind.PREDICTION_ERROR,),
            lambda ev, b: self.pes.append(ev))

    def _seq(self, focals, start_cycle=1):
        for i, f in enumerate(focals):
            self.bus.publish(
                _make_attended(focals=(f,),
                                  cycle=start_cycle + i))

    def test_transitions_use_float_weights(self):
        self._seq(['a', 'b', 'a', 'b'])
        # Two observations of a→b should give weight 2.0.
        self.assertAlmostEqual(
            self.cer._transitions[('a', 'b')], 2.0, places=3)

    def test_miss_weakens_failed_prediction(self):
        # Build a→b twice so a→b is the prediction.
        self._seq(['a', 'b', 'a', 'b'])
        before = self.cer._transitions[('a', 'b')]
        # Now a → c (miss).  The wrong prediction (a → b) should
        # be weakened by MISS_WEAKENING (0.5).
        self._seq(['a', 'c'], start_cycle=10)
        after = self.cer._transitions.get(('a', 'b'), 0.0)
        # Weight went down by 0.5 (miss) — observation of (a, c)
        # didn't touch (a, b).
        self.assertAlmostEqual(after, before - 0.5, places=3)

    def test_repeated_miss_drops_below_prediction_threshold(self):
        # Build a→b just past prediction threshold (2.0).  Two
        # misses (each -0.5) take it to 1.0 — below the 1.5
        # prediction threshold — so the predictor stops
        # suggesting it.  At that point the natural flow can't
        # keep weakening it; the stuck-at-1.0 case is the test.
        self._seq(['a', 'b', 'a', 'b'])  # weight 2.0
        self._seq(['a', 'c'], start_cycle=20)   # miss → 1.5
        self._seq(['a', 'c'], start_cycle=22)   # miss → 1.0
        self.assertAlmostEqual(
            self.cer._transitions[('a', 'b')], 1.0, places=3)
        # Predictor must no longer suggest 'b' for 'a' — and in
        # fact prefers 'c' now (weight grew through observation).
        self.assertEqual(self.cer._predict_next('a'), 'c')

    def test_weaken_can_prune_below_floor(self):
        # Direct-call test that the floor really prunes.
        self._seq(['a', 'b', 'a', 'b'])      # weight 2.0
        # Drag the weight under TRANSITION_FLOOR via repeated
        # direct weakenings (each -0.5; floor 0.1).
        for _ in range(5):
            self.cer._weaken_transition('a', 'b')
        self.assertNotIn(('a', 'b'), self.cer._transitions)
        self.assertGreater(self.cer.transitions_pruned, 0)

    def test_hit_reinforces_by_observation(self):
        # Build a→b twice → weight 2.0.  A correct prediction
        # (a → b again) should yield weight 3.0 from observation.
        self._seq(['a', 'b', 'a', 'b'])
        self.assertAlmostEqual(
            self.cer._transitions[('a', 'b')], 2.0, places=3)
        self._seq(['a', 'b'], start_cycle=10)
        self.assertAlmostEqual(
            self.cer._transitions[('a', 'b')], 3.0, places=3)

    def test_decay_runs_in_batches(self):
        # Each attended event increments _attended_count; decay
        # runs every BATCH_SIZE.
        self._seq(['a', 'b', 'a', 'b'])  # 4 events, no decay yet
        self.assertEqual(self.cer.transitions_decayed_batches, 0)
        before = self.cer._transitions[('a', 'b')]
        # Fire enough no-op events to cross batch boundary.
        from seagi.brain.capabilities.cerebellum import (
            TRANSITION_DECAY_BATCH_SIZE,
            TRANSITION_DECAY_PER_BATCH)
        for i in range(TRANSITION_DECAY_BATCH_SIZE):
            self.bus.publish(
                _make_attended(focals=('z',),
                                  cycle=100 + i))
        # At least one decay batch should have fired.
        self.assertGreaterEqual(
            self.cer.transitions_decayed_batches, 1)
        after = self.cer._transitions.get(('a', 'b'), 0.0)
        expected = before * (1.0 - TRANSITION_DECAY_PER_BATCH)
        self.assertAlmostEqual(after, expected, places=3)

    def test_pruned_below_floor_drops_to_zero_entries(self):
        # Seed many weak (z, *) transitions, then force enough
        # decay batches that they fall below floor.
        for i in range(10):
            self._seq(['z', f'q{i}'], start_cycle=200 + 2 * i)
        before = len(self.cer._transitions)
        # Force ~50 decay batches via repeated single-focal events.
        for i in range(2500):
            self.bus.publish(_make_attended(
                focals=('idle',), cycle=10000 + i))
        # Some (z, *) entries should have been pruned to floor.
        self.assertLess(
            len(self.cer._transitions), before + 1,
            "expected pruning under sustained decay")
        self.assertGreater(self.cer.transitions_pruned, 0)

    def test_decay_does_not_break_strong_prediction(self):
        # A strongly-observed transition should survive a few
        # decay batches and still be predictable.
        for i in range(10):
            self._seq(['a', 'b'], start_cycle=1 + 2 * i)
        # Drive decay batch.
        from seagi.brain.capabilities.cerebellum import (
            TRANSITION_DECAY_BATCH_SIZE)
        for i in range(TRANSITION_DECAY_BATCH_SIZE):
            self.bus.publish(_make_attended(
                focals=('idle',), cycle=500 + i))
        # Reset history so prev=='a' is clean.
        self.cer._focal_history.clear()
        # Re-prime with one 'a' so next attended on 'a → ?' uses
        # the table.  Verify prediction still resolves to 'b'.
        predicted = self.cer._predict_next('a')
        self.assertEqual(predicted, 'b')

    def test_global_table_unchanged_when_no_tone_provider(self):
        # Without a tone provider, no entries should appear in
        # the tone overlay — global table is the sole record.
        self._seq(['a', 'b', 'a', 'b'])
        self.assertEqual(len(self.cer._tone_transitions), 0)
        self.assertGreater(len(self.cer._transitions), 0)

    def test_thought_derived_transitions_also_decay(self):
        # Thought-derived edges (cortical inference) should be
        # subject to the same decay loop.
        self.bus.publish(ThoughtProducedEvent(
            kind=EventKind.THOUGHT_PRODUCED,
            cycle=1,
            timestamp=time.time(),
            source_capability='test',
            origin='internal',
            origin_detail='cortical',
            focal='a',
            relation='causes',
            target='b',
            confidence=0.8,
            text=''))
        before = self.cer._transitions.get(('a', 'b'), 0.0)
        self.assertGreater(before, 0.0)
        from seagi.brain.capabilities.cerebellum import (
            TRANSITION_DECAY_BATCH_SIZE,
            TRANSITION_DECAY_PER_BATCH)
        for i in range(TRANSITION_DECAY_BATCH_SIZE):
            self.bus.publish(_make_attended(
                focals=('idle',), cycle=100 + i))
        after = self.cer._transitions.get(('a', 'b'), 0.0)
        self.assertAlmostEqual(
            after,
            before * (1.0 - TRANSITION_DECAY_PER_BATCH),
            places=3)


# ---------------------------------------------------------------
# Phase I.1c — tone-conditional lateral predictor
# ---------------------------------------------------------------


class TestPhaseI1cToneConditional(unittest.TestCase):
    """Same prev focal can predict different successors depending
    on chemistry tone (I/M/neutral)."""

    def setUp(self):
        self.bus = EventBus()
        # Mutable tone dict; tests flip it between buckets.
        self.tone = {'valence': 0.0}
        self.cer = Cerebellum(
            bus=self.bus,
            cycle_provider=lambda: 1,
            tone_provider=lambda: dict(self.tone))
        self.bus.subscribe(self.cer.SUBSCRIPTIONS, self.cer)
        self.pes = []
        self.bus.subscribe(
            (EventKind.PREDICTION_ERROR,),
            lambda ev, b: self.pes.append(ev))

    def _seq(self, focals, start_cycle=1):
        for i, f in enumerate(focals):
            self.bus.publish(
                _make_attended(focals=(f,),
                                  cycle=start_cycle + i))

    def test_bucket_derives_from_valence(self):
        self.tone = {'valence': +0.5}
        self.assertEqual(self.cer._current_bucket(), 'i')
        self.tone = {'valence': -0.5}
        self.assertEqual(self.cer._current_bucket(), 'm')
        self.tone = {'valence': 0.0}
        self.assertEqual(self.cer._current_bucket(), 'n')
        self.tone = {'valence': +0.10}     # below I threshold
        self.assertEqual(self.cer._current_bucket(), 'n')

    def test_observations_populate_tone_overlay(self):
        self.tone = {'valence': +0.5}     # I bucket
        self._seq(['a', 'b'])
        self.assertIn(('a', 'i', 'b'), self.cer._tone_transitions)
        # Global table reinforced too.
        self.assertIn(('a', 'b'), self.cer._transitions)

    def test_same_prev_different_tone_different_successor(self):
        # In I-tone: a→b twice.
        self.tone = {'valence': +0.5}
        self._seq(['a', 'b', 'a', 'b'], start_cycle=1)
        # In M-tone: a→c twice.
        self.tone = {'valence': -0.5}
        self._seq(['a', 'c', 'a', 'c'], start_cycle=10)
        # Now in I-tone, predict from 'a' → 'b'.
        self.tone = {'valence': +0.5}
        self.assertEqual(self.cer._predict_next('a', 'i'), 'b')
        # In M-tone, predict from 'a' → 'c'.
        self.assertEqual(self.cer._predict_next('a', 'm'), 'c')

    def test_falls_back_to_global_when_tone_overlay_empty(self):
        # Learn only in I-tone.
        self.tone = {'valence': +0.5}
        self._seq(['a', 'b', 'a', 'b'])
        # Query in M-tone: nothing in overlay → fallback to global.
        self.assertEqual(self.cer._predict_next('a', 'm'), 'b')

    def test_provider_exception_yields_empty_bucket(self):
        def bad():
            raise RuntimeError("tone offline")
        cer = Cerebellum(
            bus=EventBus(),
            cycle_provider=lambda: 1,
            tone_provider=bad)
        self.assertEqual(cer._current_bucket(), '')

    def test_pe_fires_via_tone_specific_prediction(self):
        # Build a→b in I-tone (so tone overlay learns a→b in 'i').
        self.tone = {'valence': +0.5}
        self._seq(['a', 'b', 'a', 'b'])
        self.pes.clear()
        # Now in same I-tone, a→c (miss).  PE must fire and the
        # tone overlay entry should be weakened.
        before = self.cer._tone_transitions[('a', 'i', 'b')]
        self._seq(['a', 'c'], start_cycle=20)
        cog = [e for e in self.pes if e.error_kind == 'cognitive']
        self.assertGreaterEqual(len(cog), 1)
        after = self.cer._tone_transitions.get(
            ('a', 'i', 'b'), 0.0)
        self.assertLess(after, before)

    def test_tone_overlay_decays_with_global(self):
        self.tone = {'valence': +0.5}
        self._seq(['a', 'b', 'a', 'b'])
        before = self.cer._tone_transitions[('a', 'i', 'b')]
        from seagi.brain.capabilities.cerebellum import (
            TRANSITION_DECAY_BATCH_SIZE,
            TRANSITION_DECAY_PER_BATCH)
        for i in range(TRANSITION_DECAY_BATCH_SIZE):
            self.bus.publish(_make_attended(
                focals=('idle',), cycle=200 + i))
        after = self.cer._tone_transitions.get(
            ('a', 'i', 'b'), 0.0)
        self.assertAlmostEqual(
            after,
            before * (1.0 - TRANSITION_DECAY_PER_BATCH),
            places=3)

    def test_brain_wires_tone_provider(self):
        # End-to-end: real Brain wires tone_provider.  The default
        # chemistry tone summary should yield a valid bucket and
        # the cerebellum should accept it without error.
        from seagi.body.engine import Engine
        engine = Engine()
        brain = Brain(engine=engine)
        bucket = brain.cerebellum._current_bucket()
        self.assertIn(bucket, ('i', 'm', 'n'))

    def test_stats_exposes_tone_overlay_size(self):
        self.tone = {'valence': +0.5}
        self._seq(['a', 'b', 'a', 'b'])
        s = self.cer.stats()
        self.assertIn('tone_transitions_known', s)
        self.assertGreaterEqual(s['tone_transitions_known'], 1)


# ---------------------------------------------------------------
# Phase I.2 — timing prediction (per-transition delay tuning)
# ---------------------------------------------------------------


class TestPhaseI2TimingPrediction(unittest.TestCase):
    """Cerebellum predicts WHEN, not just WHAT.  Per-transition
    running mean of cycle intervals; |actual - predicted| past
    threshold fires a 'timing'-kind PE."""

    def setUp(self):
        self.bus = EventBus()
        self.cer = Cerebellum(
            bus=self.bus, cycle_provider=lambda: 1)
        self.bus.subscribe(self.cer.SUBSCRIPTIONS, self.cer)
        self.pes = []
        self.bus.subscribe(
            (EventKind.PREDICTION_ERROR,),
            lambda ev, b: self.pes.append(ev))

    def _seq_with_cycles(self, focal_cycle_pairs):
        for focal, cycle in focal_cycle_pairs:
            self.bus.publish(
                _make_attended(focals=(focal,), cycle=cycle))

    def test_no_timing_pe_with_too_few_observations(self):
        # Need at least TIMING_MIN_OBSERVATIONS (3) of a
        # transition before any timing PE can fire.
        self._seq_with_cycles([
            ('a', 1), ('b', 5),   # (a→b) interval 4, n=1
            ('a', 10), ('b', 30), # (a→b) interval 20, n=2
        ])
        timing = [e for e in self.pes if e.error_kind == 'timing']
        self.assertEqual(timing, [])

    def test_consistent_intervals_dont_fire(self):
        # Three observations at interval 2 each — mean=2.0,
        # variance ≈ 0.  A 4th at interval 2 should not fire.
        self._seq_with_cycles([
            ('a', 1), ('b', 3),   # interval 2, n=1
            ('a', 5), ('b', 7),   # interval 2, n=2
            ('a', 9), ('b', 11),  # interval 2, n=3
            ('a', 13), ('b', 15), # interval 2, n=4 — predict 2, actual 2, mag 0
        ])
        timing = [e for e in self.pes if e.error_kind == 'timing']
        self.assertEqual(timing, [],
            f"expected no timing PEs on stable intervals, got "
            f"{[(e.predicted, e.actual) for e in timing]}")

    def test_timing_pe_fires_on_late_arrival(self):
        # Three observations at interval 2 establish prediction
        # mean=2.0.  Then a→b with interval 10 (5× the mean).
        self._seq_with_cycles([
            ('a', 1), ('b', 3),
            ('a', 5), ('b', 7),
            ('a', 9), ('b', 11),
            ('a', 100), ('b', 110),   # interval 10, magnitude 4.0 → clamp 1.0
        ])
        timing = [e for e in self.pes if e.error_kind == 'timing']
        self.assertGreaterEqual(len(timing), 1)
        ev = timing[-1]
        self.assertEqual(ev.prev_focal, 'a')
        self.assertEqual(ev.predicted_focal, 'b')
        self.assertEqual(ev.focal, 'b')
        self.assertEqual(ev.sign, -1.0)   # always -1 (both early and late are misses)
        self.assertAlmostEqual(ev.predicted, 2.0, places=3)
        self.assertAlmostEqual(ev.actual, 10.0, places=3)

    def test_timing_pe_fires_on_early_arrival(self):
        # Three observations at interval 10 → mean=10.  Then
        # interval 1 (very early).
        self._seq_with_cycles([
            ('a', 1), ('b', 11),
            ('a', 20), ('b', 30),
            ('a', 40), ('b', 50),
            ('a', 60), ('b', 61),   # interval 1 << predicted 10
        ])
        timing = [e for e in self.pes if e.error_kind == 'timing']
        self.assertGreaterEqual(len(timing), 1)
        ev = timing[-1]
        self.assertLess(ev.actual, ev.predicted)
        self.assertEqual(ev.sign, -1.0)

    def test_magnitude_normalized_to_unit_interval(self):
        # Magnitude is clamped to [0, 1] regardless of how
        # extreme the deviation is.
        self._seq_with_cycles([
            ('a', 1), ('b', 3),
            ('a', 5), ('b', 7),
            ('a', 9), ('b', 11),
            ('a', 100), ('b', 1000),  # interval 900, mean ≈ 2
        ])
        timing = [e for e in self.pes if e.error_kind == 'timing']
        self.assertGreaterEqual(len(timing), 1)
        self.assertLessEqual(timing[-1].magnitude, 1.0)
        self.assertGreaterEqual(timing[-1].magnitude, 0.25)

    def test_small_deviation_below_threshold_does_not_fire(self):
        # Mean=10; arrival at 11 (10% off) — below threshold 0.25.
        self._seq_with_cycles([
            ('a', 1), ('b', 11),
            ('a', 20), ('b', 30),
            ('a', 40), ('b', 50),
            ('a', 60), ('b', 71),   # interval 11, predicted 10, mag 0.10
        ])
        timing = [e for e in self.pes if e.error_kind == 'timing']
        self.assertEqual(timing, [])

    def test_timing_independent_per_transition(self):
        # (a→b) stabilizes at interval 1.  (c→d) stabilizes at 10.
        # Then a→b at interval 10 fires; (c→d) at interval 10 does NOT.
        for k in range(3):
            self._seq_with_cycles([
                ('a', 1 + 2 * k), ('b', 2 + 2 * k),   # a→b interval 1
            ])
        for k in range(3):
            self._seq_with_cycles([
                ('c', 50 + 20 * k), ('d', 60 + 20 * k),  # c→d interval 10
            ])
        self.pes.clear()
        # Now: a→b at interval 10 (should fire); c→d at interval 10 (should NOT fire).
        self._seq_with_cycles([
            ('a', 200), ('b', 210),    # a→b interval 10 vs predicted 1
        ])
        ab = [e for e in self.pes
                if e.error_kind == 'timing' and e.prev_focal == 'a']
        self.assertGreaterEqual(len(ab), 1)
        # Reset to verify (c→d) stays silent at predicted interval.
        self.pes.clear()
        self._seq_with_cycles([
            ('c', 300), ('d', 310),     # c→d interval 10 = predicted
        ])
        cd = [e for e in self.pes
                if e.error_kind == 'timing' and e.prev_focal == 'c']
        self.assertEqual(cd, [])

    def test_first_event_doesnt_compute_interval(self):
        # First attended event ever: no prior cycle, no interval.
        # Shouldn't crash, shouldn't fire timing.
        self.bus.publish(_make_attended(focals=('only',), cycle=42))
        self.assertEqual(
            [e for e in self.pes if e.error_kind == 'timing'], [])

    def test_running_mean_uses_welford(self):
        # Sanity: after 3 observations at intervals 1, 3, 5, the
        # mean is 3.0 exactly.
        self._seq_with_cycles([
            ('a', 0), ('b', 1),   # int 1
            ('a', 5), ('b', 8),   # int 3
            ('a', 20), ('b', 25), # int 5
        ])
        count, mean, _ = self.cer._transition_timing[('a', 'b')]
        self.assertEqual(count, 3)
        self.assertAlmostEqual(mean, 3.0, places=3)

    def test_stats_expose_timing_diagnostics(self):
        self._seq_with_cycles([
            ('a', 1), ('b', 3),
            ('a', 5), ('b', 7),
            ('a', 9), ('b', 11),
            ('a', 100), ('b', 110),
        ])
        s = self.cer.stats()
        self.assertIn('timing_errors_fired', s)
        self.assertIn('timing_transitions_known', s)
        self.assertGreaterEqual(s['timing_transitions_known'], 1)
        self.assertGreaterEqual(s['timing_errors_fired'], 1)

    def test_brain_smoke_test_no_crash(self):
        # End-to-end: real Brain handles streaming attended events
        # with varying cycle deltas without crashing.
        from seagi.body.engine import Engine
        engine = Engine()
        brain = Brain(engine=engine)
        pes = []
        brain.bus.subscribe(
            (EventKind.PREDICTION_ERROR,),
            lambda ev, b: pes.append(ev))
        # Build a→b 3× at interval 2 then surprise with interval 20.
        for k in range(3):
            brain.bus.publish(_make_attended(
                focals=('a',), cycle=1 + 4 * k))
            brain.bus.publish(_make_attended(
                focals=('b',), cycle=3 + 4 * k))
        brain.bus.publish(_make_attended(focals=('a',), cycle=100))
        brain.bus.publish(_make_attended(focals=('b',), cycle=120))
        timing = [e for e in pes if e.error_kind == 'timing']
        self.assertGreaterEqual(len(timing), 1)


# ---------------------------------------------------------------
# ACC
# ---------------------------------------------------------------


class TestACC(unittest.TestCase):

    def setUp(self):
        self.engine = Engine()
        self.bus = EventBus()
        from seagi.brain.capabilities.lts import (
            LongTermSubstrate)
        self.lts = LongTermSubstrate(engine=self.engine)
        self.acc = AnteriorCingulateCortex(
            bus=self.bus,
            lts_provider=lambda: self.lts,
            cycle_provider=lambda: 1)
        self.bus.subscribe(self.acc.SUBSCRIPTIONS, self.acc)
        self.conflicts = []
        self.bus.subscribe(
            (EventKind.CONFLICT_DETECTED,),
            lambda ev, b: self.conflicts.append(ev))

    def test_pe_above_threshold_fires_conflict(self):
        self.bus.publish(PredictionErrorEvent(
            kind=EventKind.PREDICTION_ERROR,
            cycle=10,
            timestamp=time.time(),
            source_capability='cerebellum',
            origin='internal',
            origin_detail='lateral',
            focal='fire',
            predicted=0.5,
            actual=0.0,
            magnitude=0.5,
            sign=-1.0,
            error_kind='cognitive'))
        self.assertEqual(len(self.conflicts), 1)
        self.assertEqual(
            self.conflicts[0].conflict_kind, 'prediction')

    def test_pe_below_threshold_does_not_fire(self):
        self.bus.publish(PredictionErrorEvent(
            kind=EventKind.PREDICTION_ERROR,
            cycle=10, timestamp=time.time(),
            source_capability='cerebellum',
            origin='internal',
            origin_detail='lateral',
            focal='fire',
            predicted=0.5, actual=0.45,
            magnitude=0.05, sign=-1.0,
            error_kind='cognitive'))
        self.assertEqual(len(self.conflicts), 0)

    def test_peer_contradiction_on_opposite_focals(self):
        _seed_substrate(self.engine,
            concepts=['hot', 'cold'],
            edges=[('hot', 'opposite_of', 'cold', 0.9)])
        self.bus.publish(_make_attended(
            cycle=5, focals=('hot', 'cold'),
            origin='peer'))
        peer_conf = [c for c in self.conflicts
                       if c.conflict_kind == 'peer_contradiction']
        self.assertGreaterEqual(len(peer_conf), 1)


# ---------------------------------------------------------------
# VTA + LC
# ---------------------------------------------------------------


class TestNeuromodulators(unittest.TestCase):

    def test_vta_fires_dopamine_on_positive_pe(self):
        bus = EventBus()
        vta = VTA(bus=bus, cycle_provider=lambda: 1)
        bus.subscribe(vta.SUBSCRIPTIONS, vta)
        chem = []
        val = []
        bus.subscribe(
            (EventKind.CHEMISTRY_FIRE,),
            lambda ev, b: chem.append(ev))
        bus.subscribe(
            (EventKind.VALUE_UPDATED,),
            lambda ev, b: val.append(ev))
        bus.publish(PredictionErrorEvent(
            kind=EventKind.PREDICTION_ERROR,
            cycle=5, timestamp=time.time(),
            source_capability='cerebellum',
            origin='internal',
            origin_detail='lateral',
            focal='fire',
            predicted=0.0, actual=0.0,
            magnitude=0.5, sign=+1.0,
            error_kind='cognitive'))
        confirmed = [e for e in chem
                        if e.chemistry_kind == 'confirmed_i']
        self.assertGreaterEqual(len(confirmed), 1)
        self.assertGreaterEqual(len(val), 1)
        self.assertEqual(val[0].focal, 'fire')

    def test_vta_silent_on_negative_pe(self):
        bus = EventBus()
        vta = VTA(bus=bus, cycle_provider=lambda: 1)
        bus.subscribe(vta.SUBSCRIPTIONS, vta)
        chem = []
        bus.subscribe(
            (EventKind.CHEMISTRY_FIRE,),
            lambda ev, b: chem.append(ev))
        bus.publish(PredictionErrorEvent(
            kind=EventKind.PREDICTION_ERROR,
            cycle=5, timestamp=time.time(),
            source_capability='cerebellum',
            origin='internal',
            origin_detail='lateral',
            focal='fire',
            predicted=0.0, actual=0.0,
            magnitude=0.5, sign=-1.0,
            error_kind='cognitive'))
        self.assertEqual(len(chem), 0)

    def test_lc_fires_ne_on_threat(self):
        bus = EventBus()
        lc = LocusCoeruleus(bus=bus, cycle_provider=lambda: 1)
        bus.subscribe(lc.SUBSCRIPTIONS, lc)
        chem = []
        bus.subscribe(
            (EventKind.CHEMISTRY_FIRE,),
            lambda ev, b: chem.append(ev))
        bus.publish(ThreatDetectedEvent(
            kind=EventKind.THREAT_DETECTED,
            cycle=5, timestamp=time.time(),
            source_capability='amygdala',
            origin='internal',
            origin_detail='body',
            threat_kind='body',
            magnitude=0.6,
            focal='', narrative=''))
        threats = [e for e in chem
                      if e.chemistry_kind == 'threat']
        self.assertGreaterEqual(len(threats), 1)


# ---------------------------------------------------------------
# Phase B.1 — Raphe nuclei (serotonin / mood floor)
# ---------------------------------------------------------------


class TestPhaseB1Raphe(unittest.TestCase):
    """Raphe watches rolling cortisol + oxytocin and fires
    'chronic_stress' / 'social_replenish' when either crosses a
    sustained threshold, cooldown-gated."""

    def setUp(self):
        from seagi.brain.capabilities.neuromodulators import (
            RapheNuclei)
        from seagi.brain.chemistry_types import CHANNELS
        self.bus = EventBus()
        self.state = {
            'cortisol': CHANNELS['cortisol']['baseline'],
            'oxytocin': CHANNELS['oxytocin']['baseline'],
        }
        self.raphe = RapheNuclei(
            bus=self.bus,
            cycle_provider=lambda: 1,
            chemistry_provider=lambda: dict(self.state))
        self.bus.subscribe(
            self.raphe.SUBSCRIPTIONS, self.raphe)
        self.fires = []
        self.bus.subscribe(
            (EventKind.CHEMISTRY_FIRE,),
            lambda ev, b: self.fires.append(ev))

    def _fire_chem(self, cycle, kind='probe', magnitude=0.0):
        self.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='test',
            origin='internal',
            origin_detail='probe',
            chemistry_kind=kind,
            magnitude=magnitude))

    def test_does_not_fire_without_history(self):
        # Spike cortisol immediately — raphe should still wait
        # for the warmup window before firing.
        self.state['cortisol'] = 0.50
        for c in range(1, 5):
            self._fire_chem(c)
        chronic = [e for e in self.fires
                       if e.chemistry_kind == 'chronic_stress']
        self.assertEqual(chronic, [])

    def test_chronic_stress_fires_after_sustained_cortisol(self):
        self.state['cortisol'] = 0.30   # well above baseline 0.10
        # Drive enough fires to fill the rolling window past
        # RAPHE_MIN_SAMPLES_FOR_FIRE.
        for c in range(1, 100):
            self._fire_chem(c)
        chronic = [e for e in self.fires
                       if e.chemistry_kind == 'chronic_stress'
                       and e.source_capability == 'raphe']
        self.assertGreaterEqual(len(chronic), 1)
        # Verify origin_detail is set.
        self.assertEqual(chronic[0].origin_detail, 'mood_floor')
        # Magnitude scales with the cortisol excess.
        self.assertGreater(chronic[0].magnitude, 0.0)
        self.assertLessEqual(chronic[0].magnitude, 1.0)

    def test_social_replenish_fires_after_sustained_oxytocin(self):
        self.state['oxytocin'] = 0.40   # above baseline 0.20
        for c in range(1, 100):
            self._fire_chem(c)
        social = [e for e in self.fires
                       if e.chemistry_kind == 'social_replenish'
                       and e.source_capability == 'raphe']
        self.assertGreaterEqual(len(social), 1)

    def test_stress_takes_precedence_over_warmth(self):
        # Both cortisol AND oxytocin elevated — chronic_stress
        # should win (the urgent signal).
        self.state['cortisol'] = 0.30
        self.state['oxytocin'] = 0.40
        for c in range(1, 100):
            self._fire_chem(c)
        chronic = [e for e in self.fires
                       if e.chemistry_kind == 'chronic_stress']
        social = [e for e in self.fires
                       if e.chemistry_kind == 'social_replenish']
        self.assertGreaterEqual(len(chronic), 1)
        self.assertEqual(social, [])

    def test_does_not_fire_below_threshold(self):
        # Cortisol slightly elevated but below the chronic-stress
        # threshold delta (0.05 above baseline 0.10 = 0.15).
        self.state['cortisol'] = 0.13
        for c in range(1, 100):
            self._fire_chem(c)
        chronic = [e for e in self.fires
                       if e.chemistry_kind == 'chronic_stress']
        self.assertEqual(chronic, [])

    def test_cooldown_rate_limits_firing(self):
        # Sustained massive cortisol should still produce only
        # cooldown-spaced fires.
        self.state['cortisol'] = 0.60
        for c in range(1, 500):
            self._fire_chem(c)
        chronic = [e for e in self.fires
                       if e.chemistry_kind == 'chronic_stress'
                       and e.source_capability == 'raphe']
        # 500 cycles, cooldown ~100 → at most ~5 fires.
        self.assertGreaterEqual(len(chronic), 1)
        self.assertLessEqual(len(chronic), 6)

    def test_skipped_without_chemistry_provider(self):
        # Raphe without provider is inert.
        from seagi.brain.capabilities.neuromodulators import (
            RapheNuclei)
        bus = EventBus()
        raphe = RapheNuclei(
            bus=bus, cycle_provider=lambda: 1)
        bus.subscribe(raphe.SUBSCRIPTIONS, raphe)
        fires = []
        bus.subscribe(
            (EventKind.CHEMISTRY_FIRE,),
            lambda ev, b: fires.append(ev))
        for c in range(1, 100):
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE, cycle=c,
                timestamp=time.time(), source_capability='test',
                origin='internal', origin_detail='probe',
                chemistry_kind='probe', magnitude=0.0))
        chronic = [e for e in fires
                       if e.chemistry_kind == 'chronic_stress']
        self.assertEqual(chronic, [])

    def test_chemistry_engine_applies_chronic_stress_deltas(self):
        # End-to-end: a chronic_stress fire moves serotonin down.
        from seagi.body.engine import Engine
        engine = Engine()
        brain = Brain(engine=engine)
        sero_before = brain.chemistry.global_state['serotonin']
        # Manually publish a chronic_stress event with magnitude 1.0.
        brain.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE,
            cycle=1, timestamp=time.time(),
            source_capability='test',
            origin='internal', origin_detail='probe',
            chemistry_kind='chronic_stress',
            magnitude=1.0))
        sero_after = brain.chemistry.global_state['serotonin']
        self.assertLess(sero_after, sero_before,
            f"chronic_stress should lower serotonin "
            f"({sero_before} → {sero_after})")

    def test_chemistry_engine_applies_social_replenish_deltas(self):
        from seagi.body.engine import Engine
        engine = Engine()
        brain = Brain(engine=engine)
        sero_before = brain.chemistry.global_state['serotonin']
        brain.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE,
            cycle=1, timestamp=time.time(),
            source_capability='test',
            origin='internal', origin_detail='probe',
            chemistry_kind='social_replenish',
            magnitude=1.0))
        sero_after = brain.chemistry.global_state['serotonin']
        self.assertGreater(sero_after, sero_before,
            f"social_replenish should raise serotonin "
            f"({sero_before} → {sero_after})")

    def test_brain_wires_raphe(self):
        from seagi.body.engine import Engine
        engine = Engine()
        brain = Brain(engine=engine)
        self.assertTrue(hasattr(brain, 'raphe'))
        self.assertIn('raphe', brain.status())

    def test_stats_expose_diagnostics(self):
        s = self.raphe.stats()
        for key in ('chronic_stress_fires',
                       'social_replenish_fires',
                       'cortisol_window',
                       'oxytocin_window'):
            self.assertIn(key, s)


# ---------------------------------------------------------------
# Phase B.2 — sleep/wake regulator
# ---------------------------------------------------------------


class TestPhaseB2SleepWake(unittest.TestCase):
    """SleepRegulator wake/sleep flip-flop with activity-driven
    pressure + per-tick dissipation."""

    def setUp(self):
        from seagi.brain.capabilities.sleep_wake import (
            SleepRegulator)
        from seagi.brain.capabilities.metabolic_debt import (
            MetabolicDebt)
        self.bus = EventBus()
        # Step 0 refactor: SleepRegulator reads debt via providers
        # from MetabolicDebt.  Sleep onset is now debt-driven, not
        # attended-percept-driven (legacy mechanism failed the
        # homeostatic-cost doctrine on headless reverie).
        self.metabolic_debt = MetabolicDebt(
            bus=self.bus, cycle_provider=lambda: 1)
        self.regulator = SleepRegulator(
            bus=self.bus,
            cycle_provider=lambda: 1,
            debt_provider=self.metabolic_debt.debt_provider,
            baseline_provider=(
                lambda: self.metabolic_debt.wake_onset_baseline_debt),
            debt_full_scale_provider=(
                lambda: self.metabolic_debt.debt_full_scale),
            has_logged_episode_provider=(
                lambda: self.metabolic_debt.has_logged_first_episode))
        self.bus.subscribe(
            self.metabolic_debt.SUBSCRIPTIONS, self.metabolic_debt)
        self.bus.subscribe(
            self.regulator.SUBSCRIPTIONS, self.regulator)
        self.chem_fires = []
        self.bus.subscribe(
            (EventKind.CHEMISTRY_FIRE,),
            lambda ev, b: self.chem_fires.append(ev))

    def _attended(self, cycle, focal='f'):
        self.bus.publish(_make_attended(
            cycle=cycle, focals=(focal,)))

    def _substrate_write(self, cycle):
        """Drive the debt accumulator + sleep-onset trigger by
        publishing a substrate write."""
        self.bus.publish(SubstrateWriteQueuedEvent(
            kind=EventKind.SUBSTRATE_WRITE_QUEUED,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='test',
            origin='internal', origin_detail='',
            subject='a', relation='causes', object='b',
            strength=0.1, write_reason='test'))

    def test_starts_awake_zero_pressure(self):
        self.assertEqual(self.regulator.state, 'wake')
        self.assertEqual(self.regulator.pressure, 0.0)
        self.assertFalse(self.regulator.is_asleep())

    def test_attended_event_raises_pressure(self):
        before = self.regulator.pressure
        self._attended(cycle=1)
        self.assertGreater(self.regulator.pressure, before)

    def test_chemistry_event_raises_pressure_small(self):
        before = self.regulator.pressure
        self.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE, cycle=1,
            timestamp=time.time(), source_capability='test',
            origin='internal', origin_detail='',
            chemistry_kind='curiosity', magnitude=0.1))
        # Pressure rose by less than one attended-event bump.
        from seagi.brain.capabilities.sleep_wake import (
            PRESSURE_PER_ATTENDED, PRESSURE_PER_CHEMISTRY)
        self.assertAlmostEqual(
            self.regulator.pressure - before,
            PRESSURE_PER_CHEMISTRY, places=4)

    def test_falls_asleep_when_debt_crosses_bootstrap_threshold(self):
        # Step 0: sleep is debt-driven, not attended-percept-driven.
        # Bootstrap threshold = 50 (V4 lock: EDGE_PRUNE_INTERVAL ×
        # COHERENCE_REINFORCE_BUMP).  Drive 60 substrate writes;
        # debt crosses 50, sleep fires.
        for c in range(1, 61):
            self._substrate_write(cycle=c)
        self.assertTrue(self.regulator.is_asleep())
        self.assertEqual(self.regulator.transitions_to_sleep, 1)

    def test_sleep_onset_event_fires_with_correct_kind(self):
        for c in range(1, 61):
            self._substrate_write(cycle=c)
        onset = [e for e in self.chem_fires
                       if e.chemistry_kind == 'sleep_onset']
        self.assertGreaterEqual(len(onset), 1)
        self.assertEqual(onset[0].source_capability,
                              'sleep_regulator')

    def test_dissipation_drains_pressure_during_sleep(self):
        # Drive to sleep via debt, then tick until pressure decays
        # below the low threshold and wake-up fires.  Sleep duration
        # is paced by pressure dissipation (legacy mechanism kept
        # as bridge until Phase S debt-clearance lands).
        for c in range(1, 61):
            self._substrate_write(cycle=c)
        self.assertTrue(self.regulator.is_asleep())
        # On sleep_onset, pressure was set to HIGH (0.9).
        peak_pressure = self.regulator.pressure
        for _ in range(300):     # 300 × 0.005 = 1.5 of dissipation
            self.regulator.tick()
        self.assertLess(
            self.regulator.pressure, peak_pressure)
        self.assertFalse(self.regulator.is_asleep())
        self.assertEqual(self.regulator.transitions_to_wake, 1)

    def test_wake_onset_event_fires(self):
        for c in range(1, 61):
            self._substrate_write(cycle=c)
        self.chem_fires.clear()
        for _ in range(300):
            self.regulator.tick()
        wake = [e for e in self.chem_fires
                    if e.chemistry_kind == 'wake_onset']
        self.assertGreaterEqual(len(wake), 1)

    def test_no_spurious_sleep_on_few_events(self):
        # Typical test or short session: a handful of attended
        # events should NOT trigger sleep.  Step 0 also guarantees
        # a handful of substrate writes stays well below the
        # bootstrap threshold (50).
        for c in range(1, 20):
            self._attended(cycle=c)
            self._substrate_write(cycle=c)
        self.assertFalse(self.regulator.is_asleep())

    def test_recursion_guard_on_self_chemistry_event(self):
        # The regulator emits a ChemistryEvent on transition.
        # That event would otherwise recurse into handle() —
        # the guard must suppress it.
        # Drive into sleep via debt.
        for c in range(1, 61):
            self._substrate_write(cycle=c)
        # Only one transition recorded, regardless of how many
        # chemistry events the regulator's own emit produced.
        self.assertEqual(self.regulator.transitions_to_sleep, 1)

    def test_pressure_clamped_to_unit(self):
        # Even after far more events than needed, pressure stays ≤ 1.
        for c in range(1, 2000):
            self._attended(cycle=c)
        self.assertLessEqual(self.regulator.pressure, 1.0)

    def test_chemistry_engine_applies_sleep_onset_deltas(self):
        # End-to-end: sleep_onset shifts gaba up, ACh/NE down.
        from seagi.body.engine import Engine
        engine = Engine()
        brain = Brain(engine=engine)
        gaba_before = brain.chemistry.global_state['gaba']
        ach_before = brain.chemistry.global_state['acetylcholine']
        ne_before = brain.chemistry.global_state['norepinephrine']
        brain.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE, cycle=1,
            timestamp=time.time(), source_capability='test',
            origin='internal', origin_detail='',
            chemistry_kind='sleep_onset', magnitude=1.0))
        self.assertGreater(
            brain.chemistry.global_state['gaba'], gaba_before)
        self.assertLess(
            brain.chemistry.global_state['acetylcholine'],
            ach_before)
        self.assertLess(
            brain.chemistry.global_state['norepinephrine'],
            ne_before)

    def test_chemistry_engine_applies_wake_onset_deltas(self):
        from seagi.body.engine import Engine
        engine = Engine()
        brain = Brain(engine=engine)
        gaba_before = brain.chemistry.global_state['gaba']
        ach_before = brain.chemistry.global_state['acetylcholine']
        brain.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE, cycle=1,
            timestamp=time.time(), source_capability='test',
            origin='internal', origin_detail='',
            chemistry_kind='wake_onset', magnitude=1.0))
        self.assertGreater(
            brain.chemistry.global_state['acetylcholine'],
            ach_before)
        self.assertLess(
            brain.chemistry.global_state['gaba'], gaba_before)

    def test_brain_wires_sleep_regulator(self):
        from seagi.body.engine import Engine
        engine = Engine()
        brain = Brain(engine=engine)
        self.assertTrue(hasattr(brain, 'sleep_regulator'))
        self.assertIn('sleep_regulator', brain.status())
        # And it ticks via brain.tick() — verify state survives.
        brain.tick()
        self.assertEqual(brain.sleep_regulator.state, 'wake')

    def test_stats_expose_diagnostics(self):
        s = self.regulator.stats()
        for key in ('state', 'pressure',
                       'transitions_to_sleep',
                       'transitions_to_wake'):
            self.assertIn(key, s)


# ---------------------------------------------------------------
# Phase C.1.a — NarrativeJournal
# ---------------------------------------------------------------


class TestPhaseC1aNarrativeJournal(unittest.TestCase):
    """NarrativeJournal accumulates compact first-person entries
    on moments worth remembering, persists across sessions."""

    def setUp(self):
        from seagi.brain.capabilities.narrative_journal import (
            NarrativeJournal)
        self.bus = EventBus()
        self.journal = NarrativeJournal(
            bus=self.bus, cycle_provider=lambda: 1)
        self.bus.subscribe(
            self.journal.SUBSCRIPTIONS, self.journal)

    def test_starts_empty(self):
        self.assertEqual(self.journal.recent(10), [])
        self.assertEqual(self.journal.entries_recorded, 0)

    def test_records_chronic_stress_mood_shift(self):
        self.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE, cycle=1,
            timestamp=time.time(), source_capability='raphe',
            origin='internal', origin_detail='mood_floor',
            chemistry_kind='chronic_stress', magnitude=0.5))
        entries = self.journal.recent(5)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['kind'], 'mood_shift')
        self.assertIn('cortisol', entries[0]['summary'])

    def test_records_social_replenish(self):
        self.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE, cycle=1,
            timestamp=time.time(), source_capability='raphe',
            origin='internal', origin_detail='mood_floor',
            chemistry_kind='social_replenish', magnitude=0.5))
        entries = self.journal.recent(5)
        self.assertEqual(len(entries), 1)
        self.assertIn('warmth', entries[0]['summary'])

    def test_records_sleep_transitions(self):
        for kind in ('sleep_onset', 'wake_onset'):
            self.bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE, cycle=1,
                timestamp=time.time(),
                source_capability='sleep_regulator',
                origin='internal', origin_detail='state_change',
                chemistry_kind=kind, magnitude=1.0))
        entries = self.journal.recent(5)
        self.assertEqual(len(entries), 2)
        kinds = [e['kind'] for e in entries]
        self.assertEqual(kinds, ['mood_shift', 'mood_shift'])

    def test_records_band_change_once(self):
        # First interoception in 'settled' band records; second in
        # same band does not; third in 'depleted' band records.
        def intero(cycle, band):
            return InteroceptionEvent(
                kind=EventKind.INTEROCEPTION, cycle=cycle,
                timestamp=time.time(), source_capability='insula',
                origin='internal', origin_detail='body',
                felt_state=band, lifeforce=0.5,
                body_integrity=1.0, delta_lifeforce=0.0,
                narrative='')
        self.bus.publish(intero(1, 'settled'))
        self.bus.publish(intero(2, 'settled'))
        self.bus.publish(intero(3, 'depleted'))
        entries = [e for e in self.journal.recent(10)
                       if e['kind'] == 'felt_state']
        self.assertEqual(len(entries), 2)
        self.assertEqual(
            entries[0]['context']['band'], 'settled')
        self.assertEqual(
            entries[1]['context']['band'], 'depleted')

    def test_records_reflection(self):
        from seagi.brain.events import ReflectionFiredEvent
        self.bus.publish(ReflectionFiredEvent(
            kind=EventKind.REFLECTION_FIRED, cycle=1,
            timestamp=time.time(),
            source_capability='vmdmn',
            origin='internal', origin_detail='',
            trigger='idle_timer',
            reflection_kind='autobiographical'))
        entries = self.journal.recent(5)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['kind'], 'reflection')
        self.assertIn('autobiographical', entries[0]['summary'])

    def test_records_high_confidence_belief(self):
        self.bus.publish(ThoughtProducedEvent(
            kind=EventKind.THOUGHT_PRODUCED, cycle=1,
            timestamp=time.time(), source_capability='cortical',
            origin='internal', origin_detail='causal',
            focal='wisdom', relation='is_a',
            target='virtue', confidence=0.9,
            text=''))
        entries = self.journal.recent(5)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['kind'], 'belief')

    def test_does_not_record_low_confidence_belief(self):
        self.bus.publish(ThoughtProducedEvent(
            kind=EventKind.THOUGHT_PRODUCED, cycle=1,
            timestamp=time.time(), source_capability='cortical',
            origin='internal', origin_detail='guess',
            focal='maybe', relation='is_a',
            target='something', confidence=0.3,
            text=''))
        self.assertEqual(self.journal.recent(5), [])

    def test_belief_threshold_boundary(self):
        # 2026-05-19: threshold lowered to 0.6.  A 0.65-confidence
        # thought records; a 0.55-confidence one does not.
        self.bus.publish(ThoughtProducedEvent(
            kind=EventKind.THOUGHT_PRODUCED, cycle=1,
            timestamp=time.time(), source_capability='cortical',
            origin='internal', origin_detail='causal',
            focal='patience', relation='enables',
            target='wisdom', confidence=0.65, text=''))
        self.bus.publish(ThoughtProducedEvent(
            kind=EventKind.THOUGHT_PRODUCED, cycle=2,
            timestamp=time.time(), source_capability='cortical',
            origin='internal', origin_detail='causal',
            focal='haste', relation='enables',
            target='error', confidence=0.55, text=''))
        beliefs = [e for e in self.journal.recent(10)
                       if e['kind'] == 'belief']
        self.assertEqual(len(beliefs), 1)
        self.assertEqual(beliefs[0]['context']['focal'],
                              'patience')

    def test_ring_capacity_bounds_entries(self):
        from seagi.brain.capabilities.narrative_journal import (
            NarrativeJournal)
        bus = EventBus()
        j = NarrativeJournal(
            bus=bus, cycle_provider=lambda: 1, capacity=3)
        bus.subscribe(j.SUBSCRIPTIONS, j)
        for i in range(10):
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE, cycle=i,
                timestamp=time.time(), source_capability='raphe',
                origin='internal', origin_detail='mood_floor',
                chemistry_kind='sleep_onset', magnitude=1.0))
        # Ring caps at 3 stored entries; counter still goes up.
        self.assertEqual(len(j.all_entries()), 3)
        self.assertEqual(j.entries_recorded, 10)

    def test_persistence_round_trip(self):
        # Fire some events, save, restore into a fresh journal.
        self.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE, cycle=5,
            timestamp=time.time(), source_capability='raphe',
            origin='internal', origin_detail='mood_floor',
            chemistry_kind='chronic_stress', magnitude=0.5))
        from seagi.brain.events import ReflectionFiredEvent
        self.bus.publish(ReflectionFiredEvent(
            kind=EventKind.REFLECTION_FIRED, cycle=6,
            timestamp=time.time(),
            source_capability='vmdmn',
            origin='internal', origin_detail='',
            trigger='idle_timer',
            reflection_kind='autobiographical'))
        snap = self.journal.to_dict()
        from seagi.brain.capabilities.narrative_journal import (
            NarrativeJournal)
        bus2 = EventBus()
        j2 = NarrativeJournal(
            bus=bus2, cycle_provider=lambda: 1)
        j2.load_from_dict(snap)
        self.assertEqual(len(j2.all_entries()), 2)
        self.assertEqual(j2.entries_recorded, 2)

    # test_brain_persists_journal removed (audit #11, 2026-06-04):
    # NarrativeJournal was subtracted from the Brain (write-only organ,
    # no reader).  The isolated NarrativeJournal class tests above still
    # exercise the unwired organ directly.


# ---------------------------------------------------------------
# Phase C.1.b — Autosaver
# ---------------------------------------------------------------


class TestPhaseC1bAutosaver(unittest.TestCase):
    """Autosaver gates save_brain by elapsed time or ticks and
    rotates backups."""

    def setUp(self):
        import tempfile
        from seagi.body.engine import Engine
        from seagi.core.autosave import Autosaver
        self.tmp_dir = tempfile.mkdtemp()
        self.path = self.tmp_dir + '/test_save.json.gz'
        self.engine = Engine()
        self.brain = Brain(engine=self.engine)
        self.autosaver = Autosaver(
            engine=self.engine, brain=self.brain,
            path=self.path,
            min_seconds=999.0,    # disable time gate for tests
            min_ticks=10,
            max_backups=2)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_first_save_writes_file(self):
        wrote = self.autosaver.maybe_save(cycle=1, force=True)
        self.assertTrue(wrote)
        import os
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(self.autosaver.saves, 1)

    def test_debounces_within_interval(self):
        # First save force-bypasses gating.
        self.autosaver.maybe_save(cycle=1, force=True)
        # Second call within the tick gate should NOT write.
        wrote = self.autosaver.maybe_save(cycle=2)
        self.assertFalse(wrote)
        self.assertEqual(self.autosaver.saves, 1)

    def test_saves_after_tick_gate_clears(self):
        self.autosaver.maybe_save(cycle=1, force=True)
        wrote = self.autosaver.maybe_save(cycle=12)
        self.assertTrue(wrote)
        self.assertEqual(self.autosaver.saves, 2)

    def test_force_bypasses_gate(self):
        self.autosaver.maybe_save(cycle=1, force=True)
        wrote = self.autosaver.maybe_save(cycle=2, force=True)
        self.assertTrue(wrote)
        self.assertEqual(self.autosaver.saves, 2)

    def test_rotates_backups(self):
        import os
        # Three saves → expect main + 2 backups (max_backups=2).
        for k, c in enumerate([1, 20, 40]):
            self.autosaver.maybe_save(cycle=c, force=True)
        self.assertTrue(os.path.exists(self.path))
        self.assertTrue(os.path.exists(self.path + '.bak.1'))
        self.assertTrue(os.path.exists(self.path + '.bak.2'))
        # A 4th save should still leave us at main + 2 backups.
        self.autosaver.maybe_save(cycle=60, force=True)
        self.assertTrue(os.path.exists(self.path))
        self.assertTrue(os.path.exists(self.path + '.bak.1'))
        self.assertTrue(os.path.exists(self.path + '.bak.2'))

    def test_stats_diagnostics(self):
        s = self.autosaver.stats()
        for k in ('saves', 'last_save_time', 'last_save_cycle',
                    'last_path', 'path'):
            self.assertIn(k, s)
        self.autosaver.maybe_save(cycle=1, force=True)
        s = self.autosaver.stats()
        self.assertEqual(s['saves'], 1)
        self.assertEqual(s['last_save_cycle'], 1)


# ---------------------------------------------------------------
# Phase C.1.c — IdleMotivation
# ---------------------------------------------------------------


class TestPhaseC1cIdleMotivation(unittest.TestCase):
    """IdleMotivation fires a curiosity event when the bus has
    been quiet for IDLE_TICK_THRESHOLD ticks."""

    def _new(self, awm_concepts):
        from seagi.brain.capabilities.idle_motivation import (
            IdleMotivation)
        bus = EventBus()
        cycles = [0]
        def cycle_provider():
            return cycles[0]
        idle = IdleMotivation(
            bus=bus,
            cycle_provider=cycle_provider,
            awm_provider=lambda: list(awm_concepts))
        bus.subscribe(idle.SUBSCRIPTIONS, idle)
        fires = []
        bus.subscribe(
            (EventKind.CHEMISTRY_FIRE,),
            lambda ev, b: fires.append(ev))
        return bus, idle, fires, cycles

    def test_starts_silent(self):
        _, idle, fires, _ = self._new(['wisdom'])
        self.assertEqual(idle.fires, 0)
        self.assertEqual(fires, [])

    def test_does_not_fire_before_threshold(self):
        from seagi.brain.capabilities.idle_motivation import (
            IDLE_TICK_THRESHOLD)
        _, idle, fires, cycles = self._new(['wisdom'])
        for _ in range(IDLE_TICK_THRESHOLD - 1):
            cycles[0] += 1
            idle.tick()
        self.assertEqual(idle.fires, 0)

    def test_fires_after_quiet_threshold(self):
        from seagi.brain.capabilities.idle_motivation import (
            IDLE_TICK_THRESHOLD)
        _, idle, fires, cycles = self._new(['wisdom'])
        for _ in range(IDLE_TICK_THRESHOLD + 5):
            cycles[0] += 1
            idle.tick()
        self.assertEqual(idle.fires, 1)
        cur = [e for e in fires
                  if e.chemistry_kind == 'curiosity']
        self.assertEqual(len(cur), 1)
        self.assertEqual(cur[0].target_concepts, ['wisdom'])
        self.assertEqual(cur[0].source_capability,
                              'idle_motivation')

    def test_attended_resets_idle_counter(self):
        from seagi.brain.capabilities.idle_motivation import (
            IDLE_TICK_THRESHOLD)
        bus, idle, fires, cycles = self._new(['wisdom'])
        for _ in range(IDLE_TICK_THRESHOLD - 1):
            cycles[0] += 1
            idle.tick()
        # External activity resets the clock.
        bus.publish(_make_attended(focals=('peer_input',),
                                          cycle=cycles[0]))
        for _ in range(50):
            cycles[0] += 1
            idle.tick()
        self.assertEqual(idle.fires, 0)

    def test_cooldown_between_fires(self):
        from seagi.brain.capabilities.idle_motivation import (
            IDLE_TICK_THRESHOLD, IDLE_FIRE_COOLDOWN)
        _, idle, fires, cycles = self._new(['wisdom'])
        # First fire.
        for _ in range(IDLE_TICK_THRESHOLD + 5):
            cycles[0] += 1
            idle.tick()
        first_fires = idle.fires
        self.assertEqual(first_fires, 1)
        # Within cooldown, no second fire even past threshold.
        for _ in range(IDLE_TICK_THRESHOLD + 5):
            cycles[0] += 1
            idle.tick()
        self.assertLessEqual(idle.fires, 2)   # at most one more
        # Past cooldown + threshold, fires again.
        for _ in range(IDLE_FIRE_COOLDOWN):
            cycles[0] += 1
            idle.tick()
        self.assertGreaterEqual(idle.fires, 2)

    def test_silent_with_empty_awm(self):
        from seagi.brain.capabilities.idle_motivation import (
            IDLE_TICK_THRESHOLD)
        _, idle, fires, cycles = self._new([])
        for _ in range(IDLE_TICK_THRESHOLD + 50):
            cycles[0] += 1
            idle.tick()
        self.assertEqual(idle.fires, 0)

    def test_silent_without_awm_provider(self):
        from seagi.brain.capabilities.idle_motivation import (
            IdleMotivation, IDLE_TICK_THRESHOLD)
        bus = EventBus()
        idle = IdleMotivation(
            bus=bus, cycle_provider=lambda: 0)
        for _ in range(IDLE_TICK_THRESHOLD + 50):
            idle.tick()
        self.assertEqual(idle.fires, 0)

    def test_brain_wires_idle_motivation(self):
        from seagi.body.engine import Engine
        engine = Engine()
        brain = Brain(engine=engine)
        self.assertTrue(hasattr(brain, 'idle_motivation'))
        self.assertIn('idle_motivation', brain.status())
        # Driving brain.tick advances the idle counter.
        before = brain.idle_motivation.stats()['idle_ticks']
        for _ in range(5):
            brain.tick()
        after = brain.idle_motivation.stats()['idle_ticks']
        self.assertGreater(after, before)

    # ---- C.1.e patch: substrate fallback ----

    def _new_with_substrate(self, awm_concepts, substrate_list):
        from seagi.brain.capabilities.idle_motivation import (
            IdleMotivation)
        bus = EventBus()
        cycles = [0]
        idle = IdleMotivation(
            bus=bus,
            cycle_provider=lambda: cycles[0],
            awm_provider=lambda: list(awm_concepts),
            substrate_provider=lambda: list(substrate_list))
        bus.subscribe(idle.SUBSCRIPTIONS, idle)
        fires = []
        bus.subscribe(
            (EventKind.CHEMISTRY_FIRE,),
            lambda ev, b: fires.append(ev))
        return bus, idle, fires, cycles

    def test_substrate_fallback_when_awm_empty(self):
        from seagi.brain.capabilities.idle_motivation import (
            IDLE_TICK_THRESHOLD)
        _, idle, fires, cycles = self._new_with_substrate(
            awm_concepts=[],
            substrate_list=['wisdom', 'death', 'love'])
        for _ in range(IDLE_TICK_THRESHOLD + 5):
            cycles[0] += 1
            idle.tick()
        self.assertEqual(idle.fires, 1)
        self.assertEqual(idle.fires_from_substrate, 1)
        self.assertEqual(idle.fires_from_awm, 0)
        self.assertEqual(idle.last_focal, 'wisdom')

    def test_awm_preferred_when_both_present(self):
        from seagi.brain.capabilities.idle_motivation import (
            IDLE_TICK_THRESHOLD)
        _, idle, _, cycles = self._new_with_substrate(
            awm_concepts=['fire'],
            substrate_list=['wisdom'])
        for _ in range(IDLE_TICK_THRESHOLD + 5):
            cycles[0] += 1
            idle.tick()
        self.assertEqual(idle.fires, 1)
        self.assertEqual(idle.fires_from_awm, 1)
        self.assertEqual(idle.fires_from_substrate, 0)
        self.assertEqual(idle.last_focal, 'fire')

    def test_substrate_fallback_rotates_through_pool(self):
        # Drive enough ticks to produce at least 6 fires, then
        # verify the *captured* fires rotate through wisdom →
        # death → love → wisdom → ...
        _, idle, fires, cycles = self._new_with_substrate(
            awm_concepts=[],
            substrate_list=['wisdom', 'death', 'love'])
        for _ in range(3000):
            cycles[0] += 1
            idle.tick()
        # Pull each curiosity event's target.
        cur = [e.target_concepts[0] for e in fires
                  if e.chemistry_kind == 'curiosity'
                  and e.target_concepts]
        self.assertGreaterEqual(len(cur), 6)
        # Expected pattern: round-robin through pool.
        pool = ['wisdom', 'death', 'love']
        expected = [pool[i % 3] for i in range(len(cur))]
        self.assertEqual(cur, expected)

    def test_substrate_fallback_origin_detail_distinguishes(self):
        from seagi.brain.capabilities.idle_motivation import (
            IDLE_TICK_THRESHOLD)
        _, idle, fires, cycles = self._new_with_substrate(
            awm_concepts=[],
            substrate_list=['wisdom'])
        for _ in range(IDLE_TICK_THRESHOLD + 5):
            cycles[0] += 1
            idle.tick()
        ev = [e for e in fires
                  if e.chemistry_kind == 'curiosity'][0]
        self.assertEqual(ev.origin_detail, 'reverie:substrate')

    # ---- C.1.f: reverie publishes synthetic AttendedPerceptEvent ----

    def test_reverie_publishes_attended_percept(self):
        from seagi.brain.capabilities.idle_motivation import (
            IDLE_TICK_THRESHOLD)
        bus, idle, fires, cycles = self._new_with_substrate(
            awm_concepts=[],
            substrate_list=['wisdom'])
        attended = []
        bus.subscribe(
            (EventKind.ATTENDED_PERCEPT,),
            lambda ev, b: attended.append(ev))
        for _ in range(IDLE_TICK_THRESHOLD + 5):
            cycles[0] += 1
            idle.tick()
        self.assertEqual(idle.fires, 1)
        self.assertEqual(len(attended), 1)
        ev = attended[0]
        self.assertEqual(ev.focals, ['wisdom'])
        self.assertEqual(ev.source_capability, 'idle_motivation')
        self.assertEqual(ev.origin, 'internal')
        self.assertEqual(ev.origin_detail, 'reverie:substrate')

    def test_self_attended_does_not_reset_idle(self):
        # The synthetic AttendedPerceptEvent IdleMotivation
        # publishes must NOT reset its own idle counter, or
        # subsequent reveries would be starved.
        from seagi.brain.capabilities.idle_motivation import (
            IDLE_TICK_THRESHOLD, IDLE_FIRE_COOLDOWN)
        _, idle, fires, cycles = self._new_with_substrate(
            awm_concepts=[],
            substrate_list=['wisdom', 'death'])
        # Drive through 4 fire windows.
        for _ in range(4 * (IDLE_TICK_THRESHOLD + IDLE_FIRE_COOLDOWN)):
            cycles[0] += 1
            idle.tick()
        self.assertGreaterEqual(idle.fires, 2,
            f"expected ≥2 fires across multiple windows, got "
            f"{idle.fires}")

    def test_external_attended_still_resets_idle(self):
        from seagi.brain.capabilities.idle_motivation import (
            IDLE_TICK_THRESHOLD)
        bus, idle, fires, cycles = self._new_with_substrate(
            awm_concepts=[],
            substrate_list=['wisdom'])
        for _ in range(IDLE_TICK_THRESHOLD - 10):
            cycles[0] += 1
            idle.tick()
        # External attended percept (peer input) must reset.
        bus.publish(AttendedPerceptEvent(
            kind=EventKind.ATTENDED_PERCEPT,
            cycle=cycles[0], timestamp=time.time(),
            source_capability='thalamic_gate',
            origin='peer', origin_detail='external',
            focals=['question'], payload={},
            raw_text='', modality='text',
            salience=0.5, novelty=0.5,
            m_content=0.1, i_content=0.1,
            threshold_used=0.3))
        for _ in range(50):
            cycles[0] += 1
            idle.tick()
        self.assertEqual(idle.fires, 0,
            "external attended event should have reset idle "
            "counter, no fire expected")

    def test_brain_substrate_top_focals_caches(self):
        from seagi.body.engine import Engine
        from seagi.core.substrate import Concept
        engine = Engine()
        # Seed substrate with a few concepts so the cache has
        # non-empty content.
        for n in ('wisdom', 'death', 'love', 'fear'):
            engine.substrate.add_concept(Concept(name=n))
        brain = Brain(engine=engine)
        focals = brain._substrate_top_focals()
        # All 4 seeded concepts should show up in the cache.
        for n in ('wisdom', 'death', 'love', 'fear'):
            self.assertIn(n, focals)
        # Second call returns cached list (same identity OK in
        # this implementation).
        focals2 = brain._substrate_top_focals()
        self.assertEqual(focals2, focals)

    # ---- Roadmap Step 3: goal-biased reverie ----

    def test_goal_focal_beats_awm_and_substrate(self):
        # Goals have priority over AWM and substrate when set.
        from seagi.brain.capabilities.idle_motivation import (
            IdleMotivation, IDLE_TICK_THRESHOLD)
        bus = EventBus()
        cycles = [0]
        idle = IdleMotivation(
            bus=bus,
            cycle_provider=lambda: cycles[0],
            awm_provider=lambda: ['fire'],
            substrate_provider=lambda: ['wisdom', 'death'],
            goal_provider=lambda: ['curiosity_focal'])
        bus.subscribe(idle.SUBSCRIPTIONS, idle)
        for _ in range(IDLE_TICK_THRESHOLD + 5):
            cycles[0] += 1
            idle.tick()
        self.assertEqual(idle.fires, 1)
        self.assertEqual(idle.last_focal, 'curiosity_focal')
        self.assertEqual(idle.fires_from_goal, 1)
        self.assertEqual(idle.fires_from_awm, 0)
        self.assertEqual(idle.fires_from_substrate, 0)

    def test_falls_back_when_no_open_goals(self):
        # When the goal provider returns empty, IdleMotivation
        # falls back to AWM (or substrate).  No regression.
        from seagi.brain.capabilities.idle_motivation import (
            IdleMotivation, IDLE_TICK_THRESHOLD)
        bus = EventBus()
        cycles = [0]
        idle = IdleMotivation(
            bus=bus,
            cycle_provider=lambda: cycles[0],
            awm_provider=lambda: ['fire'],
            substrate_provider=lambda: ['wisdom'],
            goal_provider=lambda: [])
        bus.subscribe(idle.SUBSCRIPTIONS, idle)
        for _ in range(IDLE_TICK_THRESHOLD + 5):
            cycles[0] += 1
            idle.tick()
        self.assertEqual(idle.last_focal, 'fire')
        self.assertEqual(idle.fires_from_awm, 1)
        self.assertEqual(idle.fires_from_goal, 0)

    def test_origin_detail_marks_goal_source(self):
        from seagi.brain.capabilities.idle_motivation import (
            IdleMotivation, IDLE_TICK_THRESHOLD)
        bus = EventBus()
        cycles = [0]
        idle = IdleMotivation(
            bus=bus,
            cycle_provider=lambda: cycles[0],
            awm_provider=lambda: [],
            goal_provider=lambda: ['the_problem'])
        bus.subscribe(idle.SUBSCRIPTIONS, idle)
        fires = []
        bus.subscribe(
            (EventKind.CHEMISTRY_FIRE,),
            lambda ev, b: fires.append(ev))
        for _ in range(IDLE_TICK_THRESHOLD + 5):
            cycles[0] += 1
            idle.tick()
        cur = [e for e in fires if e.chemistry_kind == 'curiosity']
        self.assertEqual(len(cur), 1)
        self.assertEqual(cur[0].origin_detail, 'reverie:goal')

    def test_brain_wires_goal_provider(self):
        # End-to-end: Brain spawns a goal, IdleMotivation picks
        # the goal focal over AWM/substrate during reverie.
        from seagi.body.engine import Engine
        from seagi.brain.capabilities.idle_motivation import (
            IDLE_TICK_THRESHOLD)
        engine = Engine()
        brain = Brain(engine=engine)
        # Open a goal via the public GoalTracker API.
        brain.goals.spawn(
            focal='puzzle', kind='learn_about',
            urgency=0.8, source='test', cycle=0)
        # Drive the IdleMotivation by ticking past its threshold.
        for _ in range(IDLE_TICK_THRESHOLD + 5):
            brain.tick()
        self.assertGreaterEqual(
            brain.idle_motivation.fires_from_goal, 1,
            "the open goal should have biased autonomous reverie")
        self.assertEqual(
            brain.idle_motivation.last_focal, 'puzzle')


# ---------------------------------------------------------------
# ValueLandscape
# ---------------------------------------------------------------


class TestValueLandscape(unittest.TestCase):

    def setUp(self):
        self.bus = EventBus()
        self.vl = ValueLandscape(
            bus=self.bus, cycle_provider=lambda: 1)
        self.bus.subscribe(self.vl.SUBSCRIPTIONS, self.vl)

    def test_positive_pe_increases_value(self):
        self.bus.publish(PredictionErrorEvent(
            kind=EventKind.PREDICTION_ERROR,
            cycle=5, timestamp=time.time(),
            source_capability='cerebellum',
            origin='internal',
            origin_detail='lateral',
            focal='fire',
            predicted=0.0, actual=0.0,
            magnitude=0.5, sign=+1.0,
            error_kind='cognitive'))
        self.assertGreater(self.vl.value_of('fire'), 0.0)

    def test_negative_pe_decreases_value(self):
        self.bus.publish(PredictionErrorEvent(
            kind=EventKind.PREDICTION_ERROR,
            cycle=5, timestamp=time.time(),
            source_capability='cerebellum',
            origin='internal',
            origin_detail='lateral',
            focal='fire',
            predicted=0.0, actual=0.0,
            magnitude=0.5, sign=-1.0,
            error_kind='cognitive'))
        self.assertLess(self.vl.value_of('fire'), 0.0)

    def test_confirmed_i_chemistry_bumps_value(self):
        self.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE,
            cycle=5, timestamp=time.time(),
            source_capability='vta',
            origin='internal',
            origin_detail='',
            chemistry_kind='confirmed_i',
            magnitude=0.5,
            target_concepts=['water']))
        self.assertGreater(self.vl.value_of('water'), 0.0)


# ---------------------------------------------------------------
# Amygdala
# ---------------------------------------------------------------


class TestAmygdala(unittest.TestCase):

    def setUp(self):
        self.bus = EventBus()
        self.am = Amygdala(
            bus=self.bus, cycle_provider=lambda: 1)
        self.bus.subscribe(self.am.SUBSCRIPTIONS, self.am)
        self.threats = []
        self.claims = []
        self.bus.subscribe(
            (EventKind.THREAT_DETECTED,),
            lambda ev, b: self.threats.append(ev))
        self.bus.subscribe(
            (EventKind.CAPABILITY_CLAIM,),
            lambda ev, b: self.claims.append(ev))

    def test_high_m_percept_triggers_threat(self):
        self.bus.publish(_make_attended(
            focals=('danger',), m=0.8, i=0.0,
            origin='peer'))
        self.assertGreaterEqual(len(self.threats), 1)
        self.assertEqual(
            self.threats[0].threat_kind, 'percept')
        # Override claim emitted with high strength.
        amyg_claims = [c for c in self.claims
                          if c.source_capability == 'amygdala']
        self.assertGreaterEqual(len(amyg_claims), 1)
        self.assertGreaterEqual(amyg_claims[0].claim_strength, 0.7)

    def test_low_m_percept_does_not_trigger(self):
        self.bus.publish(_make_attended(
            focals=('cake',), m=0.1, i=0.6))
        self.assertEqual(len(self.threats), 0)

    def test_body_interoception_threat(self):
        self.bus.publish(InteroceptionEvent(
            kind=EventKind.INTEROCEPTION,
            cycle=5, timestamp=time.time(),
            source_capability='insula',
            origin='internal',
            origin_detail='body',
            felt_state='depleted',
            lifeforce=0.10,
            body_integrity=1.0,
            delta_lifeforce=-0.6,
            narrative=''))
        body_threats = [t for t in self.threats
                           if t.threat_kind == 'body']
        self.assertGreaterEqual(len(body_threats), 1)


# ---------------------------------------------------------------
# Basal Ganglia
# ---------------------------------------------------------------


class TestBasalGanglia(unittest.TestCase):

    def setUp(self):
        self.bus = EventBus()
        self.bg = BasalGanglia(
            bus=self.bus, cycle_provider=lambda: 1)
        self.bus.subscribe(self.bg.SUBSCRIPTIONS, self.bg)
        self.decisions = []
        self.bus.subscribe(
            (EventKind.ARBITRATION_DECIDED,),
            lambda ev, b: self.decisions.append(ev))

    def _claim(self, strength, action, loop='cognitive',
                 source='test'):
        return CapabilityClaimEvent(
            kind=EventKind.CAPABILITY_CLAIM,
            cycle=1, timestamp=time.time(),
            source_capability=source,
            origin='internal',
            origin_detail='',
            claim_strength=strength,
            proposed_action=action,
            loop=loop,
            payload={})

    def test_single_claim_wins(self):
        self.bus.publish(self._claim(0.5, 'think_about:fire'))
        self.bg.arbitrate()
        self.assertEqual(len(self.decisions), 1)
        self.assertEqual(
            self.decisions[0].winning_action, 'think_about:fire')

    def test_higher_strength_beats_lower(self):
        self.bus.publish(self._claim(0.4, 'low',
                                            source='cap_a'))
        self.bus.publish(self._claim(0.9, 'high',
                                            source='cap_b'))
        self.bg.arbitrate()
        self.assertEqual(len(self.decisions), 1)
        self.assertEqual(self.decisions[0].winning_action, 'high')
        self.assertEqual(
            self.decisions[0].winning_capability, 'cap_b')
        self.assertEqual(self.decisions[0].n_competing, 2)

    def test_loops_arbitrate_independently(self):
        self.bus.publish(self._claim(
            0.5, 'cog_a', loop='cognitive', source='ca'))
        self.bus.publish(self._claim(
            0.7, 'speech_a', loop='speech', source='sa'))
        self.bg.arbitrate()
        self.assertEqual(len(self.decisions), 2)
        winners = {d.loop: d.winning_action for d in self.decisions}
        self.assertEqual(winners['cognitive'], 'cog_a')
        self.assertEqual(winners['speech'], 'speech_a')

    def test_value_bonus_breaks_ties(self):
        # Wire a value provider that strongly favors 'fire'.
        class FakeVL:
            def value_of(self, focal):
                return 1.0 if focal == 'fire' else -0.5
        bg = BasalGanglia(
            bus=self.bus,
            value_provider=lambda: FakeVL(),
            cycle_provider=lambda: 1)
        self.bus.subscribe(bg.SUBSCRIPTIONS, bg)
        decisions2 = []
        self.bus.subscribe(
            (EventKind.ARBITRATION_DECIDED,),
            lambda ev, b: decisions2.append(ev))
        # Equal claim strength, different focals.
        self.bus.publish(self._claim(
            0.5, 'do:fire', source='cap_a'))
        self.bus.publish(self._claim(
            0.5, 'do:water', source='cap_b'))
        bg.arbitrate()
        # fire wins via value bonus.
        winning = [d.winning_action for d in decisions2]
        self.assertIn('do:fire', winning)


# ---------------------------------------------------------------
# Phase B.1.b — BG serotonin gating (impulse control)
# ---------------------------------------------------------------


class TestPhaseB1bSerotoninGate(unittest.TestCase):
    """BG action threshold rises with sustained-elevated serotonin.
    Sub-threshold winners are inhibited; baseline serotonin
    preserves current behavior."""

    def _bg(self, serotonin_level):
        bus = EventBus()
        bg = BasalGanglia(
            bus=bus, cycle_provider=lambda: 1,
            serotonin_provider=lambda: serotonin_level)
        bus.subscribe(bg.SUBSCRIPTIONS, bg)
        decisions = []
        bus.subscribe(
            (EventKind.ARBITRATION_DECIDED,),
            lambda ev, b: decisions.append(ev))
        return bus, bg, decisions

    def _claim(self, strength, action, loop='cognitive',
                 source='test'):
        return CapabilityClaimEvent(
            kind=EventKind.CAPABILITY_CLAIM,
            cycle=1, timestamp=time.time(),
            source_capability=source,
            origin='internal', origin_detail='',
            claim_strength=strength,
            proposed_action=action,
            loop=loop, payload={})

    def test_baseline_serotonin_threshold_is_zero(self):
        # At canonical baseline 0.50, threshold = 0 (no inhibition).
        _, bg, _ = self._bg(0.50)
        self.assertEqual(bg._serotonin_threshold(), 0.0)

    def test_below_baseline_serotonin_threshold_is_zero(self):
        # Below baseline → threshold floored at 0 (no inhibition).
        _, bg, _ = self._bg(0.30)
        self.assertEqual(bg._serotonin_threshold(), 0.0)

    def test_above_baseline_raises_threshold(self):
        # +0.20 above baseline → threshold = 0.20 (under cap 0.30).
        _, bg, _ = self._bg(0.70)
        self.assertAlmostEqual(
            bg._serotonin_threshold(), 0.20, places=3)

    def test_threshold_capped(self):
        # Massive serotonin → threshold capped at MAX (0.30).
        _, bg, _ = self._bg(2.0)
        from seagi.brain.capabilities.basal_ganglia import (
            BG_SEROTONIN_GATE_MAX)
        self.assertEqual(
            bg._serotonin_threshold(), BG_SEROTONIN_GATE_MAX)

    def test_no_provider_threshold_zero(self):
        bus = EventBus()
        bg = BasalGanglia(bus=bus, cycle_provider=lambda: 1)
        self.assertEqual(bg._serotonin_threshold(), 0.0)

    def test_weak_claim_inhibited_at_high_serotonin(self):
        # Serotonin 0.70 → threshold 0.20.  Claim 0.15 < threshold →
        # no decision emitted; inhibition counter increments.
        bus, bg, decisions = self._bg(0.70)
        bus.publish(self._claim(0.15, 'idle:flicker'))
        bg.arbitrate()
        self.assertEqual(decisions, [])
        self.assertEqual(bg.serotonin_inhibitions, 1)

    def test_strong_claim_passes_at_high_serotonin(self):
        # Threshold 0.20 but claim 0.50 — wins.
        bus, bg, decisions = self._bg(0.70)
        bus.publish(self._claim(0.50, 'commit:answer'))
        bg.arbitrate()
        self.assertEqual(len(decisions), 1)
        self.assertEqual(
            decisions[0].winning_action, 'commit:answer')

    def test_inhibition_clears_pending(self):
        # After inhibition, the loop's pending should be drained
        # (claims don't roll over).
        bus, bg, _ = self._bg(0.70)
        bus.publish(self._claim(0.10, 'idle:a'))
        bg.arbitrate()
        self.assertEqual(bg.pending_count('cognitive'), 0)

    def test_baseline_preserves_canonical_behavior(self):
        # At baseline serotonin a weak claim still wins.
        bus, bg, decisions = self._bg(0.50)
        bus.publish(self._claim(0.05, 'whisper'))
        bg.arbitrate()
        self.assertEqual(len(decisions), 1)
        self.assertEqual(bg.serotonin_inhibitions, 0)

    def test_brain_wires_serotonin_provider(self):
        # End-to-end: real Brain wires lambda to chemistry.global_state
        from seagi.body.engine import Engine
        engine = Engine()
        brain = Brain(engine=engine)
        # At canonical serotonin baseline the threshold is 0.
        self.assertEqual(
            brain.basal_ganglia._serotonin_threshold(), 0.0)
        # Lift serotonin manually and confirm threshold tracks it.
        brain.chemistry.global_state['serotonin'] = 0.70
        self.assertAlmostEqual(
            brain.basal_ganglia._serotonin_threshold(),
            0.20, places=3)

    def test_stats_expose_diagnostic_fields(self):
        bus, bg, _ = self._bg(0.70)
        bus.publish(self._claim(0.1, 'idle:x'))
        bg.arbitrate()
        s = bg.stats()
        self.assertIn('serotonin_inhibitions', s)
        self.assertIn('serotonin_threshold', s)
        self.assertEqual(s['serotonin_inhibitions'], 1)
        self.assertAlmostEqual(s['serotonin_threshold'], 0.20,
                                       places=3)


# ---------------------------------------------------------------
# Source Monitor
# ---------------------------------------------------------------


class TestSourceMonitor(unittest.TestCase):

    def test_tracks_origin_distribution(self):
        bus = EventBus()
        sm = SourceMonitor(bus=bus)
        bus.subscribe_all(sm)
        bus.publish(_make_attended(origin='peer'))
        bus.publish(_make_attended(origin='forager'))
        bus.publish(_make_attended(origin='peer'))
        stats = sm.stats()
        self.assertEqual(stats['origin_distribution']['peer'], 2)
        self.assertEqual(
            stats['origin_distribution']['forager'], 1)

    def test_flags_untagged_events(self):
        from seagi.brain.events import BrainEvent
        bus = EventBus()
        sm = SourceMonitor(bus=bus)
        bus.subscribe_all(sm)
        # Untagged event.
        bus.publish(BrainEvent(
            kind=EventKind.AWM_ACTIVATION,
            cycle=1, timestamp=time.time(),
            source_capability='', origin='',
            origin_detail=''))
        self.assertEqual(sm.untagged_count, 1)


# ---------------------------------------------------------------
# Anterior PFC
# ---------------------------------------------------------------


class TestAnteriorPFC(unittest.TestCase):

    def test_unproductive_streak_emits_shift_claim(self):
        bus = EventBus()
        pfc = AnteriorPFC(bus=bus, cycle_provider=lambda: 1)
        bus.subscribe(pfc.SUBSCRIPTIONS, pfc)
        claims = []
        bus.subscribe(
            (EventKind.CAPABILITY_CLAIM,),
            lambda ev, b: claims.append(ev))
        # Emit UNPRODUCTIVE_WINDOW thin thoughts.
        for i in range(pfc.UNPRODUCTIVE_WINDOW):
            bus.publish(ThoughtProducedEvent(
                kind=EventKind.THOUGHT_PRODUCED,
                cycle=i, timestamp=time.time(),
                source_capability='cortical',
                origin='internal',
                origin_detail=f'f{i}',
                focal=f'f{i}', relation='',
                target='', confidence=0.1,
                method='metacog', text='',
                thin_substrate=True))
        shifts = [c for c in claims
                    if c.proposed_action == 'shift_focus']
        self.assertGreaterEqual(len(shifts), 1)

    # test_conflict_sets_prospective_intent removed (audit #10,
    # 2026-06-04): the conflict -> prospective-intent sink was
    # subtracted from AnteriorPFC (dead path, no reader; re-attention
    # runs via conflict -> UncertaintyMonitor -> GoalSpawner).


# ---------------------------------------------------------------
# Time Perception
# ---------------------------------------------------------------


class TestTimePerception(unittest.TestCase):

    def test_tracks_since_last(self):
        cycle = [10]
        bus = EventBus()
        tp = TimePerception(
            bus=bus, cycle_provider=lambda: cycle[0])
        bus.subscribe_all(tp)
        bus.publish(_make_attended(cycle=10))
        cycle[0] = 20
        self.assertEqual(
            tp.since_last(str(EventKind.ATTENDED_PERCEPT)), 10)

    def test_tracks_focal_recency(self):
        cycle = [5]
        bus = EventBus()
        tp = TimePerception(
            bus=bus, cycle_provider=lambda: cycle[0])
        bus.subscribe_all(tp)
        bus.publish(_make_attended(cycle=5, focals=('fire',)))
        cycle[0] = 12
        self.assertEqual(tp.since_focal('fire'), 7)


# ---------------------------------------------------------------
# Body Schema
# ---------------------------------------------------------------


class TestBodySchema(unittest.TestCase):

    def test_envelope_pulls_from_insula(self):
        class FakeIns:
            def felt_state(self):
                return {'lifeforce': 0.4, 'body_integrity': 0.9,
                          'band': 'waning',
                          'narrative': 'hm.'}
        bs = BodySchema(insula_provider=lambda: FakeIns(),
                            cycle_provider=lambda: 7)
        env = bs.envelope()
        self.assertEqual(env['lifeforce'], 0.4)
        self.assertEqual(env['band'], 'waning')
        self.assertTrue(env['can_speak'])

    def test_can_speak_false_when_depleted(self):
        class FakeIns:
            def felt_state(self):
                return {'lifeforce': 0.1, 'body_integrity': 1.0,
                          'band': 'depleted',
                          'narrative': ''}
        bs = BodySchema(insula_provider=lambda: FakeIns())
        self.assertFalse(bs.can_speak())


# ---------------------------------------------------------------
# Corpus Callosum
# ---------------------------------------------------------------


class TestCorpusCallosum(unittest.TestCase):

    def test_disagreement_with_value_polarity_fires_conflict(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'hot', 'cold'],
            edges=[('fire', 'is_a', 'hot', 0.9),
                      ('fire', 'opposite_of', 'cold', 0.5)])
        from seagi.brain.capabilities.lts import (
            LongTermSubstrate)
        from seagi.brain.capabilities.cerebellum import (
            Cerebellum)

        class FakeCer:
            def _predict_next(self, focal):
                return 'cold'   # stream A wants cold

        class FakeVL:
            def value_of(self, focal):
                return {'hot': 0.5, 'cold': -0.5}.get(focal, 0.0)

        bus = EventBus()
        cc = CorpusCallosum(
            bus=bus,
            lts_provider=lambda: LongTermSubstrate(engine=engine),
            cerebellum_provider=lambda: FakeCer(),
            value_provider=lambda: FakeVL(),
            cycle_provider=lambda: 1)
        bus.subscribe(cc.SUBSCRIPTIONS, cc)
        conflicts = []
        bus.subscribe(
            (EventKind.CONFLICT_DETECTED,),
            lambda ev, b: conflicts.append(ev))
        bus.publish(_make_attended(focals=('fire',)))
        # Stream A → cold (negative value), stream B → hot
        # (positive value) — polarity differs → conflict.
        self.assertGreaterEqual(len(conflicts), 1)


# ---------------------------------------------------------------
# Journaled writer
# ---------------------------------------------------------------


class TestJournaledWriter(unittest.TestCase):

    def test_first_write_applies(self):
        bus = EventBus()
        w = JournaledSubstrateWriter(
            engine=None, memory_mode=True)
        bus.subscribe(w.SUBSCRIPTIONS, w)
        bus.publish(SubstrateWriteQueuedEvent(
            kind=EventKind.SUBSTRATE_WRITE_QUEUED,
            cycle=1, timestamp=time.time(),
            source_capability='test',
            origin='internal',
            origin_detail='',
            subject='a', relation='r', object='b',
            strength=0.4, write_reason='unit'))
        self.assertEqual(w.writes_applied, 1)
        self.assertEqual(w.memory_edges[('a', 'r', 'b')], 0.4)

    def test_duplicate_skipped(self):
        bus = EventBus()
        w = JournaledSubstrateWriter(
            engine=None, memory_mode=True)
        bus.subscribe(w.SUBSCRIPTIONS, w)
        for _ in range(3):
            bus.publish(SubstrateWriteQueuedEvent(
                kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                cycle=1, timestamp=time.time(),
                source_capability='test',
                origin='internal',
                origin_detail='',
                subject='a', relation='r', object='b',
                strength=0.4, write_reason='unit'))
        self.assertEqual(w.writes_applied, 1)
        self.assertEqual(w.writes_skipped, 2)

    def test_stronger_write_reinforces(self):
        bus = EventBus()
        w = JournaledSubstrateWriter(
            engine=None, memory_mode=True)
        bus.subscribe(w.SUBSCRIPTIONS, w)
        bus.publish(SubstrateWriteQueuedEvent(
            kind=EventKind.SUBSTRATE_WRITE_QUEUED,
            cycle=1, timestamp=time.time(),
            source_capability='test',
            origin='internal',
            origin_detail='',
            subject='a', relation='r', object='b',
            strength=0.3, write_reason='first'))
        bus.publish(SubstrateWriteQueuedEvent(
            kind=EventKind.SUBSTRATE_WRITE_QUEUED,
            cycle=2, timestamp=time.time(),
            source_capability='test2',
            origin='internal',
            origin_detail='',
            subject='a', relation='r', object='b',
            strength=0.9, write_reason='second'))
        self.assertEqual(w.writes_reinforced, 1)
        self.assertGreater(
            w.memory_edges[('a', 'r', 'b')], 0.3)

    def test_journal_records_status(self):
        bus = EventBus()
        w = JournaledSubstrateWriter(
            engine=None, memory_mode=True)
        bus.subscribe(w.SUBSCRIPTIONS, w)
        bus.publish(SubstrateWriteQueuedEvent(
            kind=EventKind.SUBSTRATE_WRITE_QUEUED,
            cycle=1, timestamp=time.time(),
            source_capability='test',
            origin='internal',
            origin_detail='',
            subject='a', relation='r', object='b',
            strength=0.4, write_reason='unit'))
        tail = w.journal_tail(5)
        self.assertEqual(tail[0]['status'], 'applied')


# ---------------------------------------------------------------
# Brain integration — Phase 4b
# ---------------------------------------------------------------


class TestBrainPhase4bEndToEnd(unittest.TestCase):

    def test_status_reports_all_phase4b_capabilities(self):
        brain = Brain(engine=None, memory_writer=True)
        status = brain.status()
        for key in ('cerebellum', 'acc', 'amygdala',
                       'basal_ganglia', 'vta', 'lc',
                       'value_landscape', 'anterior_pfc',
                       'source_monitor', 'time_perception',
                       'body_schema', 'corpus_callosum',
                       'writer'):
            self.assertIn(key, status)

    def test_chat_drives_value_landscape(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat'],
            edges=[('fire', 'produces', 'heat', 0.8)])
        brain = Brain(engine=engine)
        # Synthesize a positive PE for 'fire' directly on the
        # bus so VTA → ValueLandscape sees a confirmed_i +
        # value update.
        brain.bus.publish(PredictionErrorEvent(
            kind=EventKind.PREDICTION_ERROR,
            cycle=1, timestamp=time.time(),
            source_capability='test',
            origin='internal',
            origin_detail='',
            focal='fire',
            predicted=0.0, actual=0.0,
            magnitude=0.4, sign=+1.0,
            error_kind='cognitive'))
        self.assertGreater(
            brain.value_landscape.value_of('fire'), 0.0)

    def test_threat_percept_fires_amygdala_then_lc(self):
        brain = Brain(engine=None, memory_writer=True)
        # Synthesize a high-M attended percept directly.
        brain.bus.publish(_make_attended(
            focals=('danger',), m=0.9, origin='peer'))
        self.assertGreater(brain.amygdala.threats_detected, 0)
        # LC should fire NE on the threat event.
        self.assertGreater(brain.lc.ne_fires, 0)

    def test_arbitration_emits_winner(self):
        brain = Brain(engine=None, memory_writer=True)
        decisions = []
        brain.bus.subscribe(
            (EventKind.ARBITRATION_DECIDED,),
            lambda ev, b: decisions.append(ev))
        # Two competing claims for cognitive loop.
        for s, a in [(0.4, 'low_action'), (0.9, 'high_action')]:
            brain.bus.publish(CapabilityClaimEvent(
                kind=EventKind.CAPABILITY_CLAIM,
                cycle=1, timestamp=time.time(),
                source_capability='test',
                origin='internal',
                origin_detail='',
                claim_strength=s,
                proposed_action=a,
                loop='cognitive',
                payload={}))
        brain.tick()
        self.assertGreaterEqual(len(decisions), 1)
        cog = [d for d in decisions if d.loop == 'cognitive']
        self.assertEqual(cog[0].winning_action, 'high_action')

    def test_source_monitor_tracks_full_chat(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire'],
            edges=[])
        brain = Brain(engine=engine)
        brain.chat('tell me about fire', peer_id='h')
        stats = brain.source_monitor.stats()
        self.assertGreater(stats['events_seen'], 0)
        self.assertIn('peer', stats['origin_distribution'])

    def test_time_perception_tracks_chat_cycles(self):
        brain = Brain(engine=None, memory_writer=True)
        brain.tick()
        # publish an attended percept directly.
        brain.bus.publish(_make_attended(
            cycle=brain._cycle_provider(),
            focals=('fire',)))
        # Now advance a few ticks.
        for _ in range(3):
            brain.tick()
        since = brain.time_perception.since_focal('fire')
        self.assertIsNotNone(since)
        self.assertGreater(since, 0)

    def test_writer_journals_writes(self):
        brain = Brain(engine=None, memory_writer=True)
        brain.bus.publish(SubstrateWriteQueuedEvent(
            kind=EventKind.SUBSTRATE_WRITE_QUEUED,
            cycle=1, timestamp=time.time(),
            source_capability='hippocampus',
            origin='internal',
            origin_detail='episode:1',
            subject='a', relation='co_occurs', object='b',
            strength=0.3, write_reason='consolidation'))
        tail = brain.writer.journal_tail(5)
        self.assertEqual(len(tail), 1)
        self.assertEqual(tail[0]['status'], 'applied')

    def test_chat_is_still_fast_in_phase4b(self):
        # Phase 4b adds many capabilities — make sure chat is
        # still fast.
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat'],
            edges=[('fire', 'produces', 'heat', 0.8)])
        brain = Brain(engine=engine)
        start = time.time()
        for _ in range(5):
            brain.chat('tell me about fire', peer_id='h')
        elapsed = time.time() - start
        # 5 chats in under 2 seconds is plenty of margin.
        self.assertLess(elapsed, 2.0)


# ---------------------------------------------------------------
# BG arbitration → cortical wiring (the live gate)
# ---------------------------------------------------------------


class TestArbitrationDrivesCortical(unittest.TestCase):
    """BG arbitration is no longer shelf-ware: cortical
    subscribes to ARBITRATION_DECIDED and acts on the cognitive
    loop winner.  These tests pin the BEHAVIOR — not the message
    plumbing — so they break loudly if the wiring regresses."""

    def test_attend_threat_arbitration_triggers_cortical_rethink(
            self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['danger', 'pain'],
            edges=[('danger', 'causes', 'pain', 0.7)])
        brain = Brain(engine=engine)
        thoughts = []
        speech_requests = []
        brain.bus.subscribe(
            (EventKind.THOUGHT_PRODUCED,),
            lambda ev, b: thoughts.append(ev))
        brain.bus.subscribe(
            (EventKind.SPEECH_REQUEST,),
            lambda ev, b: speech_requests.append(ev))
        # Amygdala-style threat claim on cognitive loop.
        brain.bus.publish(CapabilityClaimEvent(
            kind=EventKind.CAPABILITY_CLAIM,
            cycle=10, timestamp=time.time(),
            source_capability='amygdala',
            origin='internal',
            origin_detail='percept',
            claim_strength=0.85,
            proposed_action='attend_threat:danger',
            loop='cognitive',
            payload={}))
        brain.basal_ganglia.arbitrate()
        # Cortical re-thought 'danger' and emitted a speech
        # request for it.
        danger_thoughts = [t for t in thoughts
                              if t.focal == 'danger']
        self.assertGreaterEqual(len(danger_thoughts), 1,
            "Cortical did not re-think the threat focal — BG "
            "arbitration is still shelf-ware.")
        danger_speech = [s for s in speech_requests
                            if s.intended_focal == 'danger']
        self.assertGreaterEqual(len(danger_speech), 1,
            "Cortical did not re-emit SPEECH_REQUEST on threat "
            "arbitration — voice override broken.")
        # Speech request carries the arbitration origin tag for
        # source monitoring.
        self.assertTrue(
            any(s.origin_detail.startswith('arbitration_threat:')
                for s in danger_speech))

    def test_attend_threat_arbitration_skips_generic_kind(self):
        # 'attend_threat:body' is not a real focal — Amygdala
        # falls back to threat_kind when no concept is implicated.
        # Cortical must not try to think about a non-focal.
        brain = Brain(engine=None, memory_writer=True)
        thoughts = []
        brain.bus.subscribe(
            (EventKind.THOUGHT_PRODUCED,),
            lambda ev, b: thoughts.append(ev))
        brain.bus.publish(CapabilityClaimEvent(
            kind=EventKind.CAPABILITY_CLAIM,
            cycle=5, timestamp=time.time(),
            source_capability='amygdala',
            origin='internal',
            origin_detail='body',
            claim_strength=0.8,
            proposed_action='attend_threat:body',
            loop='cognitive',
            payload={}))
        brain.basal_ganglia.arbitrate()
        # No thought about the literal word 'body'.
        body_thoughts = [t for t in thoughts if t.focal == 'body']
        self.assertEqual(len(body_thoughts), 0)

    def test_attend_threat_skipped_when_focal_already_top(self):
        # If cortical just thought about the threat focal, the
        # arbitration is redundant — don't re-fire.
        engine = Engine()
        _seed_substrate(engine,
            concepts=['snake'],
            edges=[])
        brain = Brain(engine=engine)
        # Run cortical once on a snake percept so it's recently
        # thought about.
        brain.bus.publish(_make_attended(
            focals=('snake',), origin='peer',
            cycle=brain._cycle_provider()))
        before = brain.cortical.thoughts_produced
        # Now arbitrate attend_threat:snake on a near cycle.
        brain.bus.publish(CapabilityClaimEvent(
            kind=EventKind.CAPABILITY_CLAIM,
            cycle=brain._cycle_provider(),
            timestamp=time.time(),
            source_capability='amygdala',
            origin='internal',
            origin_detail='percept',
            claim_strength=0.85,
            proposed_action='attend_threat:snake',
            loop='cognitive',
            payload={}))
        brain.basal_ganglia.arbitrate()
        # No additional thought (skip-when-recent kicked in).
        self.assertEqual(brain.cortical.thoughts_produced, before)

    def test_shift_focus_arbitration_picks_alternate_focal(self):
        # Seed several concepts; warm AWM with one; arbitration
        # shift_focus should make cortical think about a
        # DIFFERENT focal.
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'water', 'stone'],
            edges=[('fire', 'opposite_of', 'water', 0.5),
                      ('stone', 'is_a', 'matter', 0.4)])
        brain = Brain(engine=engine)
        thoughts = []
        speech_requests = []
        brain.bus.subscribe(
            (EventKind.THOUGHT_PRODUCED,),
            lambda ev, b: thoughts.append(ev))
        brain.bus.subscribe(
            (EventKind.SPEECH_REQUEST,),
            lambda ev, b: speech_requests.append(ev))
        # Warm AWM with fire + water + stone via attended
        # percepts so they're all active concepts.
        for f in ('fire', 'water', 'stone'):
            brain.bus.publish(_make_attended(
                focals=(f,), origin='peer',
                cycle=brain._cycle_provider()))
        thoughts_before_shift = list(thoughts)
        speech_before_shift = list(speech_requests)
        # Arbitration declares shift_focus winner.
        brain.bus.publish(CapabilityClaimEvent(
            kind=EventKind.CAPABILITY_CLAIM,
            cycle=brain._cycle_provider(),
            timestamp=time.time(),
            source_capability='anterior_pfc',
            origin='internal',
            origin_detail='metacog',
            claim_strength=0.6,
            proposed_action='shift_focus',
            loop='cognitive',
            payload={}))
        brain.basal_ganglia.arbitrate()
        new_thoughts = [t for t in thoughts
                          if t not in thoughts_before_shift]
        # Cortical thought about something during arbitration.
        self.assertGreaterEqual(len(new_thoughts), 1,
            "shift_focus arbitration produced no new thought — "
            "the brain didn't actually shift.")
        # NO new SPEECH_REQUEST — shift_focus is internal,
        # not a voice change.
        new_speech = [s for s in speech_requests
                         if s not in speech_before_shift]
        self.assertEqual(len(new_speech), 0,
            "shift_focus emitted a SPEECH_REQUEST — but "
            "this is an internal attention shift, not a "
            "new utterance.")

    def test_non_cognitive_arbitration_ignored_by_cortical(self):
        # Cortical only acts on the cognitive loop.  Winners
        # on motivational / speech loops are NOT its concern.
        brain = Brain(engine=None, memory_writer=True)
        thoughts = []
        brain.bus.subscribe(
            (EventKind.THOUGHT_PRODUCED,),
            lambda ev, b: thoughts.append(ev))
        # Direct synthetic arbitration on the speech loop.
        brain.bus.publish(ArbitrationDecidedEvent(
            kind=EventKind.ARBITRATION_DECIDED,
            cycle=5, timestamp=time.time(),
            source_capability='basal_ganglia',
            origin='internal',
            origin_detail='speech',
            loop='speech',
            winning_action='attend_threat:fire',
            winning_capability='amygdala',
            winning_strength=0.9,
            n_competing=1))
        self.assertEqual(len(thoughts), 0)


# ---------------------------------------------------------------
# Nucleus Accumbens — I-side sentinel
# ---------------------------------------------------------------


class TestNucleusAccumbens(unittest.TestCase):
    """The always-open life-extending monitor.  Fires
    attend_opportunity claims to BG when high-I content arrives
    on perception, chemistry, or body channels.  Symmetric to
    Amygdala."""

    def _setup(self):
        from seagi.brain.capabilities.nucleus_accumbens \
            import NucleusAccumbens
        bus = EventBus()
        nacc = NucleusAccumbens(
            bus=bus, cycle_provider=lambda: 1)
        bus.subscribe(nacc.SUBSCRIPTIONS, nacc)
        claims = []
        bus.subscribe(
            (EventKind.CAPABILITY_CLAIM,),
            lambda ev, b: claims.append(ev))
        return bus, nacc, claims

    def test_high_i_percept_fires_opportunity_claim(self):
        bus, nacc, claims = self._setup()
        bus.publish(_make_attended(
            focals=('insight',), m=0.0, i=0.8,
            origin='peer'))
        opp_claims = [c for c in claims
                          if c.source_capability == 'nucleus_accumbens']
        self.assertGreaterEqual(len(opp_claims), 1,
            "High-i_content percept did not fire opportunity "
            "claim — life-extending monitor is silent.")
        self.assertTrue(opp_claims[0].proposed_action.startswith(
            'attend_opportunity:'))

    def test_low_i_percept_does_not_fire(self):
        bus, nacc, claims = self._setup()
        bus.publish(_make_attended(focals=('cake',),
                                            m=0.1, i=0.2))
        opp_claims = [c for c in claims
                          if c.source_capability == 'nucleus_accumbens']
        self.assertEqual(len(opp_claims), 0)

    def test_confirmed_i_chemistry_fires_opportunity(self):
        bus, nacc, claims = self._setup()
        bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE,
            cycle=5, timestamp=time.time(),
            source_capability='cortical',
            origin='internal', origin_detail='',
            chemistry_kind='confirmed_i',
            magnitude=0.6,
            target_concepts=['growth']))
        opp_claims = [c for c in claims
                          if c.source_capability == 'nucleus_accumbens']
        self.assertGreaterEqual(len(opp_claims), 1)

    def test_skips_own_loop_chemistry(self):
        # NAcc → LC → confirmed_i would loop forever without
        # skip-own-loop guard.
        bus, nacc, claims = self._setup()
        bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE,
            cycle=5, timestamp=time.time(),
            source_capability='vta',  # would-be loop trigger
            origin='internal', origin_detail='',
            chemistry_kind='confirmed_i', magnitude=0.6,
            target_concepts=['x']))
        opp_claims = [c for c in claims
                          if c.source_capability == 'nucleus_accumbens']
        self.assertEqual(len(opp_claims), 0)

    def test_body_replenished_fires_body_opportunity(self):
        bus, nacc, claims = self._setup()
        bus.publish(InteroceptionEvent(
            kind=EventKind.INTEROCEPTION,
            cycle=5, timestamp=time.time(),
            source_capability='insula',
            origin='internal', origin_detail='body',
            felt_state='replenished',
            lifeforce=0.85, body_integrity=1.0,
            delta_lifeforce=0.20,
            narrative=''))
        opp_claims = [c for c in claims
                          if c.source_capability == 'nucleus_accumbens']
        self.assertGreaterEqual(len(opp_claims), 1)
        self.assertTrue(opp_claims[0].proposed_action.startswith(
            'attend_opportunity:body'))

    def test_amygdala_threat_outweighs_nacc_opportunity_in_bg(self):
        """Brain-correct: when both fire on a percept, BG picks
        threat over opportunity.  Mortality has priority — but
        opportunity is on the table, not invisible."""
        brain = Brain(engine=None, memory_writer=True)
        # Percept with BOTH high m_content (amygdala trigger)
        # AND high i_content (NAcc trigger) — both fire claims.
        brain.bus.publish(_make_attended(
            focals=('mixed',), m=0.8, i=0.7,
            origin='peer'))
        decisions = []
        brain.bus.subscribe(
            (EventKind.ARBITRATION_DECIDED,),
            lambda ev, b: decisions.append(ev))
        brain.basal_ganglia.arbitrate()
        cog_decisions = [d for d in decisions
                              if d.loop == 'cognitive']
        # Threat won; the action prefix proves it.
        self.assertGreaterEqual(len(cog_decisions), 1)
        self.assertTrue(
            cog_decisions[0].winning_action.startswith(
                'attend_threat:'),
            "Opportunity beat threat in BG arbitration — but "
            "mortality should have priority.")

    def test_pure_opportunity_arbitration_drives_cortical(self):
        """Synthetic NAcc claim — when arbitration declares
        'attend_opportunity:<focal>' the winner, cortical re-
        thinks and re-emits SPEECH_REQUEST tagged
        arbitration_opportunity.  Symmetric to the threat path."""
        engine = Engine()
        _seed_substrate(engine,
            concepts=['growth', 'flourish'],
            edges=[('growth', 'causes', 'flourish', 0.7)])
        brain = Brain(engine=engine)
        thoughts = []
        speech = []
        brain.bus.subscribe(
            (EventKind.THOUGHT_PRODUCED,),
            lambda ev, b: thoughts.append(ev))
        brain.bus.subscribe(
            (EventKind.SPEECH_REQUEST,),
            lambda ev, b: speech.append(ev))
        # Direct synthetic claim (bypass percept-driven path so
        # we test only the arbitration wiring, not race conditions
        # with the peer-percept-driven cortical pass).
        brain.bus.publish(CapabilityClaimEvent(
            kind=EventKind.CAPABILITY_CLAIM,
            cycle=10, timestamp=time.time(),
            source_capability='nucleus_accumbens',
            origin='internal',
            origin_detail='percept',
            claim_strength=0.7,
            proposed_action='attend_opportunity:growth',
            loop='cognitive',
            payload={}))
        brain.basal_ganglia.arbitrate()
        # Cortical re-thought 'growth' via arbitration handler.
        growth_thoughts = [t for t in thoughts
                                if t.focal == 'growth']
        self.assertGreaterEqual(len(growth_thoughts), 1,
            "I-side arbitration did not drive cortical to "
            "re-think — life-extending pathway is shelf-ware.")
        opp_speech = [s for s in speech
                          if s.origin_detail.startswith(
                              'arbitration_opportunity:')]
        self.assertGreaterEqual(len(opp_speech), 1)


# ---------------------------------------------------------------
# NoveltyMonitor — curiosity-driven sentinel
# ---------------------------------------------------------------


class TestNoveltyMonitor(unittest.TestCase):
    """Curiosity-driven always-open monitor.  Fires
    explore_novel claims when novel content appears in
    perception or substrate chemistry."""

    def _setup(self):
        from seagi.brain.capabilities.novelty_monitor \
            import NoveltyMonitor
        bus = EventBus()
        nov = NoveltyMonitor(bus=bus, cycle_provider=lambda: 1)
        bus.subscribe(nov.SUBSCRIPTIONS, nov)
        claims = []
        bus.subscribe(
            (EventKind.CAPABILITY_CLAIM,),
            lambda ev, b: claims.append(ev))
        return bus, nov, claims

    def test_high_novelty_percept_fires_explore_claim(self):
        bus, nov, claims = self._setup()
        bus.publish(_make_attended(
            focals=('quasar',), novelty=0.8, m=0.0, i=0.0))
        nov_claims = [c for c in claims
                          if c.source_capability == 'novelty_monitor']
        self.assertGreaterEqual(len(nov_claims), 1)
        self.assertTrue(
            nov_claims[0].proposed_action.startswith(
                'explore_novel:'))

    def test_low_novelty_does_not_fire(self):
        bus, nov, claims = self._setup()
        bus.publish(_make_attended(
            focals=('familiar',), novelty=0.2))
        nov_claims = [c for c in claims
                          if c.source_capability == 'novelty_monitor']
        self.assertEqual(len(nov_claims), 0)

    def test_strong_curiosity_chemistry_fires(self):
        bus, nov, claims = self._setup()
        bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE,
            cycle=5, timestamp=time.time(),
            source_capability='chemistry',
            origin='internal', origin_detail='',
            chemistry_kind='curiosity',
            magnitude=0.7,
            target_concepts=['quasar']))
        nov_claims = [c for c in claims
                          if c.source_capability == 'novelty_monitor']
        self.assertGreaterEqual(len(nov_claims), 1)

    def test_own_loop_chemistry_skipped(self):
        bus, nov, claims = self._setup()
        bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE,
            cycle=5, timestamp=time.time(),
            source_capability='vta',  # would-be loop
            origin='internal', origin_detail='',
            chemistry_kind='curiosity',
            magnitude=0.7,
            target_concepts=['x']))
        nov_claims = [c for c in claims
                          if c.source_capability == 'novelty_monitor']
        self.assertEqual(len(nov_claims), 0)


# ---------------------------------------------------------------
# UncertaintyMonitor — epistemic-load sentinel
# ---------------------------------------------------------------


class TestUncertaintyMonitor(unittest.TestCase):
    """Always-open monitor that accumulates unresolved
    epistemic events (conflicts + thin-substrate thoughts)
    and fires resolve_uncertainty when pressure is too high."""

    def _setup(self, cycle_value=1):
        from seagi.brain.capabilities.uncertainty_monitor \
            import UncertaintyMonitor
        bus = EventBus()
        cycle = [cycle_value]
        unc = UncertaintyMonitor(
            bus=bus, cycle_provider=lambda: cycle[0])
        bus.subscribe(unc.SUBSCRIPTIONS, unc)
        claims = []
        bus.subscribe(
            (EventKind.CAPABILITY_CLAIM,),
            lambda ev, b: claims.append(ev))
        return bus, unc, claims, cycle

    def test_few_uncertainties_below_threshold_silent(self):
        bus, unc, claims, _ = self._setup()
        # 4 thin thoughts — under threshold (5).
        for i in range(4):
            bus.publish(ThoughtProducedEvent(
                kind=EventKind.THOUGHT_PRODUCED,
                cycle=i, timestamp=time.time(),
                source_capability='cortical',
                origin='internal',
                origin_detail=f'f{i}',
                focal='shared_focal', relation='',
                target='', confidence=0.1,
                method='metacog', text='',
                thin_substrate=True))
        unc_claims = [c for c in claims
                          if c.source_capability == 'uncertainty_monitor']
        self.assertEqual(len(unc_claims), 0)

    def test_threshold_crossed_fires_resolve_claim(self):
        bus, unc, claims, _ = self._setup()
        # 5 thin thoughts on same focal — crosses threshold.
        for i in range(5):
            bus.publish(ThoughtProducedEvent(
                kind=EventKind.THOUGHT_PRODUCED,
                cycle=i, timestamp=time.time(),
                source_capability='cortical',
                origin='internal',
                origin_detail=f'f{i}',
                focal='shared_focal', relation='',
                target='', confidence=0.1,
                method='metacog', text='',
                thin_substrate=True))
        unc_claims = [c for c in claims
                          if c.source_capability == 'uncertainty_monitor']
        self.assertGreaterEqual(len(unc_claims), 1)
        self.assertEqual(
            unc_claims[0].proposed_action,
            'resolve_uncertainty:shared_focal')

    def test_conflicts_also_count(self):
        bus, unc, claims, _ = self._setup()
        for i in range(5):
            bus.publish(ConflictDetectedEvent(
                kind=EventKind.CONFLICT_DETECTED,
                cycle=i, timestamp=time.time(),
                source_capability='acc',
                origin='internal',
                origin_detail='prediction',
                conflict_kind='prediction',
                magnitude=0.4,
                focals=['contested'], narrative=''))
        unc_claims = [c for c in claims
                          if c.source_capability == 'uncertainty_monitor']
        self.assertGreaterEqual(len(unc_claims), 1)

    def test_refractory_prevents_repeat(self):
        bus, unc, claims, cycle = self._setup(cycle_value=10)
        # Cross threshold twice in a row, same cycle.
        for i in range(7):
            bus.publish(ThoughtProducedEvent(
                kind=EventKind.THOUGHT_PRODUCED,
                cycle=10, timestamp=time.time(),
                source_capability='cortical',
                origin='internal',
                origin_detail=f'f{i}',
                focal='focal', relation='',
                target='', confidence=0.1,
                method='metacog', text='',
                thin_substrate=True))
        unc_claims = [c for c in claims
                          if c.source_capability == 'uncertainty_monitor']
        # Refractory ⇒ only one claim.
        self.assertEqual(len(unc_claims), 1)


# ---------------------------------------------------------------
# Insula — tonic body-state firing (always-open monitor)
# ---------------------------------------------------------------


class TestInsulaTonicPulse(unittest.TestCase):
    """Insula fires a low-magnitude tonic chemistry pulse every
    TONIC_PULSE_INTERVAL cycles reflecting current band.  This
    is what makes the body monitor genuinely always-open, not
    just phasic on shifts."""

    def test_tonic_pulse_fires_on_interval(self):
        from seagi.brain.capabilities.insula import (
            Insula, TONIC_PULSE_INTERVAL)
        cycle = [0]
        bus = EventBus()
        insula = Insula(
            bus=bus, engine=None,
            cycle_provider=lambda: cycle[0])
        fires = []
        bus.subscribe(
            (EventKind.CHEMISTRY_FIRE,),
            lambda ev, b: fires.append(ev))
        # Pin body state replenished (settled is default but
        # tonic fires for any band) — set explicit.
        insula.set_body_state(0.85, 1.0)
        # First sample — body delta makes a phasic fire; also
        # opens the tonic schedule.  Step a tick AHEAD of the
        # tonic interval.
        for c in range(0, TONIC_PULSE_INTERVAL * 3 + 1):
            cycle[0] = c
            insula.sample(c)
        tonic_fires = [f for f in fires
                          if f.origin_detail == 'body_tonic']
        # Over 3× the interval, at least 2 tonic pulses should
        # have fired (the first at cycle 0, then every
        # INTERVAL).
        self.assertGreaterEqual(len(tonic_fires), 2,
            "Insula did not fire tonic pulses on interval — "
            "body monitor is only phasic, not always-open.")

    def test_tonic_pulse_kind_matches_band(self):
        from seagi.brain.capabilities.insula import (
            Insula, TONIC_PULSE_INTERVAL,
            BAND_DEPLETED, BAND_REPLENISHED)
        cycle = [0]
        bus = EventBus()
        insula = Insula(
            bus=bus, engine=None,
            cycle_provider=lambda: cycle[0])
        fires = []
        bus.subscribe(
            (EventKind.CHEMISTRY_FIRE,),
            lambda ev, b: fires.append(ev))
        # Drive body into DEPLETED band.
        insula.set_body_state(0.10, 0.5)
        for c in range(0, TONIC_PULSE_INTERVAL * 2 + 1):
            cycle[0] = c
            insula.sample(c)
        depleted_tonics = [
            f for f in fires
            if f.origin_detail == 'body_tonic'
            and f.chemistry_kind == 'anomaly_spike']
        self.assertGreaterEqual(len(depleted_tonics), 1,
            "Depleted band did not produce M-side tonic fire.")


if __name__ == '__main__':
    unittest.main()
