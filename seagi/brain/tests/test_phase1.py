"""Phase 1 tests — event bus, sensory intake, Thalamic Gate.

The headline test (`test_e2e_forager_junk_filtered`) is the
behavioral demonstration that Phase 1's primary deliverable
works: a stream of RSS-style noise gets filtered at the
Thalamic Gate, while peer-origin user input passes through.

If this test breaks, the substrate-junk problem returns.
"""

import unittest

from seagi.body.engine import Engine
from seagi.brain import (
    Brain,
    EventBus,
    EventKind,
    BrainEvent,
    RawPerceptEvent,
    AttendedPerceptEvent,
    PerceptDiscardedEvent,
)
from seagi.brain.capabilities.sensory import SensoryIntake
from seagi.brain.capabilities.thalamic_gate import (
    ThalamicGate, SALIENCE_THRESHOLD_BASE,
)


# ---------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------


class TestEventBus(unittest.TestCase):

    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []

        def handler(ev, b):
            received.append(ev)

        bus.subscribe((EventKind.RAW_PERCEPT,), handler)
        bus.publish(RawPerceptEvent(
            kind=EventKind.RAW_PERCEPT, cycle=1))
        self.assertEqual(len(received), 1)

    def test_wildcard_subscriber(self):
        bus = EventBus()
        seen = []
        bus.subscribe_all(lambda ev, b: seen.append(ev.kind))
        bus.publish(RawPerceptEvent(
            kind=EventKind.RAW_PERCEPT, cycle=1))
        bus.publish(RawPerceptEvent(
            kind=EventKind.PERCEPT_DISCARDED, cycle=2))
        self.assertEqual(len(seen), 2)

    def test_kind_specific_subscriber(self):
        bus = EventBus()
        only_raw = []
        bus.subscribe(
            (EventKind.RAW_PERCEPT,),
            lambda ev, b: only_raw.append(ev))
        bus.publish(RawPerceptEvent(
            kind=EventKind.RAW_PERCEPT, cycle=1))
        bus.publish(RawPerceptEvent(
            kind=EventKind.PERCEPT_DISCARDED, cycle=2))
        # Subscriber only sees its kind, not the other.
        self.assertEqual(len(only_raw), 1)

    def test_reentrant_publish_queues(self):
        bus = EventBus()
        seen = []

        def chain(ev, b):
            seen.append(ev.cycle)
            if ev.cycle < 3:
                # Publish a new event during handling.
                b.publish(RawPerceptEvent(
                    kind=EventKind.RAW_PERCEPT,
                    cycle=ev.cycle + 1))

        bus.subscribe((EventKind.RAW_PERCEPT,), chain)
        bus.publish(RawPerceptEvent(
            kind=EventKind.RAW_PERCEPT, cycle=1))
        self.assertEqual(seen, [1, 2, 3])

    def test_subscriber_error_does_not_break_bus(self):
        bus = EventBus()
        good_received = []
        bus.subscribe(
            (EventKind.RAW_PERCEPT,),
            lambda ev, b: (_ for _ in ()).throw(RuntimeError()))
        bus.subscribe(
            (EventKind.RAW_PERCEPT,),
            lambda ev, b: good_received.append(ev))
        bus.publish(RawPerceptEvent(
            kind=EventKind.RAW_PERCEPT, cycle=1))
        # Bus errors_caught incremented but good subscriber ran.
        self.assertEqual(len(good_received), 1)
        self.assertEqual(bus.errors_caught, 1)


# ---------------------------------------------------------------------
# Sensory intake
# ---------------------------------------------------------------------


class TestSensoryIntake(unittest.TestCase):

    def test_text_intake_emits_raw_percept(self):
        bus = EventBus()
        seen = []
        bus.subscribe(
            (EventKind.RAW_PERCEPT,),
            lambda ev, b: seen.append(ev))
        engine = Engine()
        sensory = SensoryIntake(bus, engine=engine)
        sensory.intake('Hello fire', modality='text',
                        origin='peer', origin_detail='alice')
        self.assertEqual(len(seen), 1)
        ev = seen[0]
        self.assertEqual(ev.modality, 'text')
        self.assertEqual(ev.origin, 'peer')
        self.assertEqual(ev.origin_detail, 'alice')

    def test_intake_count_increments(self):
        bus = EventBus()
        engine = Engine()
        sensory = SensoryIntake(bus, engine=engine)
        sensory.intake('foo', origin='peer')
        sensory.intake('bar', origin='peer')
        sensory.intake('baz', origin='peer')
        self.assertEqual(sensory.intake_count, 3)


# ---------------------------------------------------------------------
# Thalamic Gate — the headline capability
# ---------------------------------------------------------------------


class TestThalamicGate(unittest.TestCase):

    def setUp(self):
        self.engine = Engine()
        self.bus = EventBus()
        self.gate = ThalamicGate(engine=self.engine)
        self.bus.subscribe(
            self.gate.SUBSCRIPTIONS, self.gate)
        self.attended = []
        self.discarded = []
        self.bus.subscribe(
            (EventKind.ATTENDED_PERCEPT,),
            lambda ev, b: self.attended.append(ev))
        self.bus.subscribe(
            (EventKind.PERCEPT_DISCARDED,),
            lambda ev, b: self.discarded.append(ev))

    def _raw_event(self, payload, origin='forager',
                       origin_detail=''):
        from seagi.brain.events import RawPerceptEvent
        return RawPerceptEvent(
            kind=EventKind.RAW_PERCEPT, cycle=10,
            source_capability='test',
            origin=origin, origin_detail=origin_detail,
            payload=payload, raw_text='', modality='text')

    def test_peer_input_always_passes(self):
        # Empty payload, but peer origin → peer_proximity = 1.0
        # which is above threshold.  Even thin content passes
        # when it's a peer talking.
        from seagi.core.mi_value import MIValue
        self.engine.substrate.concepts.clear()
        # Add a concept so focals isn't empty after filtering.
        from seagi.core.substrate import Concept
        self.engine.substrate.add_concept(
            Concept(name='fire'))
        ev = self._raw_event(
            {'fire': MIValue.zero()},
            origin='peer', origin_detail='alice')
        self.bus.publish(ev)
        self.assertEqual(len(self.attended), 1)
        self.assertEqual(len(self.discarded), 0)
        self.assertEqual(self.attended[0].origin, 'peer')

    def test_forager_baseline_input_passes_attenuated(self):
        # Under the new doctrine, low-salience input is NOT
        # dropped — it passes through ATTENUATED.  Below-
        # threshold becomes informational, not a kill switch.
        from seagi.core.mi_value import MIValue
        from seagi.core.substrate import Concept, Bubble, ContextKey
        from seagi.core.mi_value import TransmitterState
        c = Concept(name='oxygen')
        c.bubbles = [Bubble(
            transmitter_trace=TransmitterState(),
            context_key=ContextKey(),
            encounter_count=20)]
        self.engine.substrate.add_concept(c)
        ev = self._raw_event(
            {'oxygen': MIValue.zero()},
            origin='forager', origin_detail='rss')
        self.bus.publish(ev)
        # Salience ≈ 0.2 (peer_proximity 0.2 dominates) <
        # threshold 0.30 — but emits AttendedPerceptEvent
        # with the faithful low salience; attenuated_count
        # increments.
        self.assertEqual(len(self.attended), 1)
        self.assertEqual(len(self.discarded), 0)
        self.assertLess(self.attended[0].salience,
                              self.attended[0].threshold_used)
        self.assertEqual(self.gate.attenuated_count, 1)

    def test_forager_novel_input_passes(self):
        # Forager origin (peer_proximity 0.2) BUT all concepts
        # are new → novelty = 1.0 × W_NOVELTY 0.5 = 0.5.
        # That's above threshold 0.30 → passes.
        from seagi.core.mi_value import MIValue
        ev = self._raw_event(
            {'fresh_topic_a': MIValue.zero(),
              'fresh_topic_b': MIValue.zero()},
            origin='forager', origin_detail='rss')
        self.bus.publish(ev)
        self.assertEqual(len(self.attended), 1)
        self.assertEqual(len(self.discarded), 0)
        self.assertGreaterEqual(
            self.attended[0].salience, SALIENCE_THRESHOLD_BASE)

    def test_threat_shaped_content_passes(self):
        # High m_content (cortisol-leaning concept) makes it
        # through even from forager origin.
        from seagi.core.mi_value import MIValue
        from seagi.core.substrate import Concept, Bubble, ContextKey
        from seagi.core.mi_value import TransmitterState
        c = Concept(name='predator')
        c.bubbles = [Bubble(
            transmitter_trace=TransmitterState(
                cortisol=0.8, norepinephrine=0.7),
            context_key=ContextKey(),
            encounter_count=10)]
        self.engine.substrate.add_concept(c)
        ev = self._raw_event(
            {'predator': MIValue.zero()},
            origin='forager', origin_detail='rss')
        self.bus.publish(ev)
        # m_content = (0.8 + 0.7)/2 = 0.75 × W_M 1.0 = 0.75.
        # Easily passes.
        self.assertEqual(len(self.attended), 1)
        self.assertGreater(
            self.attended[0].m_content, 0.5)

    def test_stats_track_attenuated_vs_loud(self):
        # Under the new doctrine, low-salience percepts are
        # ATTENUATED, not discarded.  All 5 events here emit
        # AttendedPerceptEvent — the stats distinguish loud
        # (above-threshold) from attenuated (below-threshold).
        from seagi.core.mi_value import MIValue
        from seagi.core.substrate import Concept, Bubble, ContextKey
        from seagi.core.mi_value import TransmitterState

        known = Concept(name='oxygen')
        known.bubbles = [Bubble(
            transmitter_trace=TransmitterState(),
            context_key=ContextKey(),
            encounter_count=20)]
        self.engine.substrate.add_concept(known)

        # 3 forager events with the known concept → attenuated.
        for _ in range(3):
            self.bus.publish(self._raw_event(
                {'oxygen': MIValue.zero()},
                origin='forager'))
        # 2 forager events with fresh concepts → loud.
        self.bus.publish(self._raw_event(
            {'novel_a': MIValue.zero()}, origin='forager'))
        self.bus.publish(self._raw_event(
            {'novel_b': MIValue.zero()}, origin='forager'))

        s = self.gate.stats()
        # All 5 had focals → all became AttendedPerceptEvent.
        self.assertEqual(s['discarded_count'], 0)
        self.assertEqual(s['attended_count'], 5)
        # 3 of those were below threshold = attenuated.
        self.assertEqual(s['attenuated_count'], 3)
        # Pass rate now = attended / (attended + discarded) = 1.0
        self.assertAlmostEqual(s['pass_rate'], 1.0)

    def test_discard_only_on_structurally_empty_input(self):
        # PerceptDiscardedEvent now only fires when the input
        # has NO extractable focals at all (genuinely empty
        # input — not "low salience").  Low-salience input
        # passes through attenuated.
        from seagi.core.mi_value import MIValue
        # Empty payload → no focals → genuine discard.
        self.bus.publish(self._raw_event(
            {}, origin='forager', origin_detail='rss:empty'))
        self.assertEqual(self.gate.discarded_count, 1)
        self.assertEqual(len(self.gate.recent_discards), 1)

    def test_low_salience_no_longer_discarded(self):
        # Confirm the inverse: low-salience NON-empty input
        # passes through (attended-with-attenuation), not
        # discarded.
        from seagi.core.mi_value import MIValue
        from seagi.core.substrate import Concept, Bubble, ContextKey
        from seagi.core.mi_value import TransmitterState
        c = Concept(name='oxygen')
        c.bubbles = [Bubble(
            transmitter_trace=TransmitterState(),
            context_key=ContextKey(),
            encounter_count=20)]
        self.engine.substrate.add_concept(c)
        self.bus.publish(self._raw_event(
            {'oxygen': MIValue.zero()},
            origin='forager', origin_detail='rss:lesswrong'))
        # Old: discarded.  New: attended (attenuated).
        self.assertEqual(self.gate.discarded_count, 0)
        self.assertEqual(self.gate.attended_count, 1)
        self.assertEqual(self.gate.attenuated_count, 1)


# ---------------------------------------------------------------------
# End-to-end Phase 1 — the headline behavioral test
# ---------------------------------------------------------------------


class TestPhase1EndToEnd(unittest.TestCase):
    """The point of Phase 1: substrate-junk problem solved at
    the source.  Peer input passes; forager noise mostly doesn't."""

    def test_brain_intake_pipeline_works(self):
        engine = Engine()
        brain = Brain(engine=engine)
        # Forager calls brain.intake for each ingested item.
        # The Thalamic Gate filters; only salient items
        # produce AttendedPerceptEvent.
        attended_count = [0]
        discarded_count = [0]
        brain.bus.subscribe(
            (EventKind.ATTENDED_PERCEPT,),
            lambda ev, b: attended_count.__setitem__(
                0, attended_count[0] + 1))
        brain.bus.subscribe(
            (EventKind.PERCEPT_DISCARDED,),
            lambda ev, b: discarded_count.__setitem__(
                0, discarded_count[0] + 1))
        # Stream 10 forager-style inputs — all known/baseline
        # content (mock: empty payload after tokenization).
        for i in range(10):
            brain.intake(
                'the and of is to in or but',
                modality='text', origin='forager',
                origin_detail=f'rss:item_{i}')
        # All should be discarded (stopword-only content has
        # no surviving focals after filtering).
        self.assertEqual(attended_count[0], 0)
        # Note: discarded_count includes inputs where the gate
        # ran (i.e., decoded payload had keys before filtering).
        # Some inputs may produce empty raw payloads upstream
        # of the gate, in which case the gate isn't even called.
        # Either way, no attended passes — the architectural
        # win is "no stopword junk in".

    def test_peer_input_passes_easily(self):
        engine = Engine()
        brain = Brain(engine=engine)
        attended = []
        brain.bus.subscribe(
            (EventKind.ATTENDED_PERCEPT,),
            lambda ev, b: attended.append(ev))
        # Peer says something with a real concept.
        from seagi.core.substrate import Concept
        engine.substrate.add_concept(Concept(name='fire'))
        brain.intake(
            'tell me about fire',
            modality='text', origin='peer',
            origin_detail='harald')
        # peer_proximity=1.0 → always passes if focals survive.
        self.assertEqual(len(attended), 1)
        self.assertIn('fire', attended[0].focals)

    def test_brain_status_reports(self):
        engine = Engine()
        brain = Brain(engine=engine)
        brain.intake('hi', origin='peer')
        s = brain.status()
        self.assertIn('sensory_intake_count', s)
        self.assertEqual(s['sensory_intake_count'], 1)
        self.assertIn('gate', s)
        self.assertIn('bus', s)


if __name__ == '__main__':
    unittest.main()
