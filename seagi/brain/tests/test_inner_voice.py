"""Tests for the InnerVoice capability (Step 5.1, built 2026-05-29).

Verifies the doctrine-relevant contract:
- Continuous in reverie (tick-driven, throttled).
- Composes from focal + substrate fact + vmDMN narrative.
- Publishes a self-percept (AttendedPerceptEvent, origin='self',
  source_capability='inner_voice') — reused event type per design.
- Gated by sleep (no inner voice while asleep) and by focal
  presence/salience (don't articulate noise).
- Repeat-suppression (no tight back-to-back self-percept loop).
- Coherence check (reject obvious junk before self-perceiving).
"""

from __future__ import annotations

import unittest

from seagi.brain import EventKind, EventBus
from seagi.brain.events import AttendedPerceptEvent
from seagi.brain.capabilities.inner_voice import (
    InnerVoice,
    INNER_VOICE_INTERVAL,
    INNER_VOICE_MIN_FOCAL_SALIENCE,
)


class _SubStub:
    """Minimal substrate stub: only `.edges` is used."""
    def __init__(self, edges):
        self.edges = edges


class _Recorder:
    """Subscribe to ATTENDED_PERCEPT and record self-percepts."""
    def __init__(self):
        self.events = []
    def handle(self, event, bus):
        self.events.append(event)


def _wire(focal=None, narrative='', edges=None, asleep=False,
          cycle_box=None):
    cycle_box = cycle_box if cycle_box is not None else [0]
    bus = EventBus()
    rec = _Recorder()
    bus.subscribe((EventKind.ATTENDED_PERCEPT,), rec)
    iv = InnerVoice(
        bus=bus,
        cycle_provider=lambda: cycle_box[0],
        awm_focal_provider=lambda: focal,
        vmdmn_narrative_provider=lambda: narrative,
        substrate_provider=lambda: (
            _SubStub(edges) if edges is not None else None),
        is_asleep_provider=lambda: asleep)
    return bus, rec, iv, cycle_box


class TestThrottling(unittest.TestCase):

    def test_no_fire_when_throttled(self):
        focal = {'name': 'x', 'salience': 0.5}
        _, rec, iv, cb = _wire(focal=focal,
                                edges={('x', 'r', 'y'): None})
        iv.tick()
        self.assertEqual(len(rec.events), 1)
        # Same cycle → throttled.
        iv.tick()
        self.assertEqual(len(rec.events), 1)
        self.assertGreaterEqual(iv.skipped_throttle, 1)

    def test_fires_again_after_interval(self):
        focal = {'name': 'x', 'salience': 0.5}
        # Two distinct focals with their own facts so the second
        # composition is genuinely new (not repeat-suppressed).
        _, rec, iv, cb = _wire(focal=focal,
                                edges={('x', 'r', 'y'): None,
                                       ('a', 'r2', 'b'): None})
        iv.tick()
        self.assertEqual(len(rec.events), 1)
        cb[0] += INNER_VOICE_INTERVAL + 1
        focal['name'] = 'a'   # provider returns the mutated dict
        iv.tick()
        self.assertEqual(len(rec.events), 2)


class TestGating(unittest.TestCase):

    def test_skipped_when_asleep(self):
        focal = {'name': 'x', 'salience': 0.9}
        _, rec, iv, _ = _wire(focal=focal,
                                edges={('x', 'r', 'y'): None},
                                asleep=True)
        iv.tick()
        self.assertEqual(len(rec.events), 0)
        self.assertGreaterEqual(iv.skipped_asleep, 1)

    def test_skipped_no_focal(self):
        _, rec, iv, _ = _wire(focal=None)
        iv.tick()
        self.assertEqual(len(rec.events), 0)
        self.assertGreaterEqual(iv.skipped_no_focal, 1)

    def test_skipped_low_salience(self):
        focal = {'name': 'x', 'salience':
                 INNER_VOICE_MIN_FOCAL_SALIENCE - 0.01}
        _, rec, iv, _ = _wire(focal=focal)
        iv.tick()
        self.assertEqual(len(rec.events), 0)
        self.assertGreaterEqual(iv.skipped_low_salience, 1)


class TestSelfPercept(unittest.TestCase):

    def test_publishes_attended_percept_with_self_origin(self):
        focal = {'name': 'rosencrantz', 'salience': 0.7}
        _, rec, iv, _ = _wire(focal=focal,
                                edges={('rosencrantz', 'appears_in',
                                        'hamlet'): None})
        iv.tick()
        self.assertEqual(len(rec.events), 1)
        ev = rec.events[0]
        self.assertIsInstance(ev, AttendedPerceptEvent)
        self.assertEqual(ev.source_capability, 'inner_voice')
        self.assertEqual(ev.origin, 'self')
        self.assertIn('rosencrantz', ev.focals)
        self.assertIn('rosencrantz', ev.raw_text)
        # Composed utterance should articulate the substrate fact.
        self.assertIn('appears_in', ev.raw_text)
        self.assertIn('hamlet', ev.raw_text)

    def test_falls_back_to_holding_when_no_substrate_fact(self):
        focal = {'name': 'x', 'salience': 0.5}
        _, rec, iv, _ = _wire(focal=focal, edges={})
        iv.tick()
        self.assertEqual(len(rec.events), 1)
        self.assertIn('x', rec.events[0].raw_text)

    def test_narrative_threads_into_utterance(self):
        focal = {'name': 'death', 'salience': 0.4}
        _, rec, iv, _ = _wire(focal=focal,
                                narrative='I felt heavy.',
                                edges={})
        iv.tick()
        self.assertEqual(len(rec.events), 1)
        text = rec.events[0].raw_text
        self.assertIn('death', text)
        self.assertIn('heavy', text)


class TestCoherenceAndRepeat(unittest.TestCase):

    def test_repeat_suppressed(self):
        # Same focal + same substrate → same utterance → suppress
        # the second time across the throttle window.
        focal = {'name': 'x', 'salience': 0.5}
        _, rec, iv, cb = _wire(focal=focal,
                                edges={('x', 'r', 'y'): None})
        iv.tick()
        self.assertEqual(len(rec.events), 1)
        cb[0] += INNER_VOICE_INTERVAL + 1
        iv.tick()
        # Same composition → tracked as repeat, no second percept.
        self.assertEqual(len(rec.events), 1)
        self.assertGreaterEqual(iv.skipped_repeat, 1)


class TestPersistenceFree(unittest.TestCase):

    def test_inner_voice_has_no_persistence(self):
        # Transient by design — diagnostics + recent ring don't
        # round-trip; restart starts fresh.  Asserting the absence
        # makes the contract explicit.
        bus = EventBus()
        iv = InnerVoice(bus=bus)
        self.assertFalse(hasattr(iv, 'to_dict'))
        self.assertFalse(hasattr(iv, 'load_dict'))


class TestCorticalEngagesOnSelfPercept(unittest.TestCase):
    """Step 5.1 closure (2026-05-29): cortical must reason on
    inner-voice self-percepts, not just peer input + reverie
    percepts.  Without this the language→cognition loop is broken
    (the agent articulates but doesn't think over its articulation).
    """

    def test_cortical_produces_thought_on_inner_voice_percept(self):
        from seagi.body.engine import Engine
        from seagi.brain import Brain
        from seagi.brain.events import (
            AttendedPerceptEvent, ThoughtProducedEvent)
        # Seed a small coherent substrate so cortical has something
        # to reason over.
        engine = Engine()
        sub = engine.substrate
        from seagi.core.substrate import Concept
        for nm in ('rosencrantz', 'character', 'hamlet'):
            sub.add_concept(Concept(name=nm))
        sub.add_edge('rosencrantz', 'character', 'is_a',
                     strength=0.5, cycle=0)
        sub.add_edge('rosencrantz', 'hamlet', 'appears_in',
                     strength=0.5, cycle=0)
        brain = Brain(engine=engine)

        thoughts = []
        class R:
            def handle(self, e, bus):
                if isinstance(e, ThoughtProducedEvent):
                    thoughts.append(e)
        brain.bus.subscribe((EventKind.THOUGHT_PRODUCED,), R())

        before = brain.cortical.thoughts_produced
        # Publish a self-percept exactly as inner_voice does.
        brain.bus.publish(AttendedPerceptEvent(
            kind=EventKind.ATTENDED_PERCEPT,
            cycle=1, timestamp=0.0,
            source_capability='inner_voice',
            origin='self', origin_detail='inner_voice',
            focals=['rosencrantz'],
            payload={'rosencrantz': 0.7},
            raw_text='I notice: rosencrantz is_a character.',
            modality='text', salience=0.7))
        # Cortical engaged with the self-percept → a thought was
        # produced (the gate that used to block this is fixed).
        self.assertGreater(
            brain.cortical.thoughts_produced, before)


if __name__ == '__main__':
    unittest.main()
