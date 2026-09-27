"""Phase 5 tests — Motor/Speech deliberate composer + LTS scale.

Headline behaviors:
  - MotorSpeech picks ONE voice mode per response (not concat).
  - Direct mode lands when cortical text is strong.
  - Body-aware mode weaves Insula narrative into the lead when
    band is depleted/agitated.
  - Reflective mode speaks vmDMN narrative when cortical is
    empty.
  - Honest-uncertain mode acknowledges thin substrate.
  - Minimal mode for nothing-to-say.
  - BodySchema.can_speak=False gates elaborate composition.
  - Repeat avoidance: same lead 3 times → forced minimal.
  - SPEECH_EMITTED event fires with sources_used + voice_mode.
  - Brain.chat() returns the composed response and emits the
    event.
  - LTS query API stays fast at simulated scale.
"""

import time
import unittest

from seagi.body.engine import Engine
from seagi.core.substrate import Concept
from seagi.brain import (
    Brain, EventKind, EventBus,
    SpeechRequestEvent,
    SpeechEmittedEvent,
    AttendedPerceptEvent,
)
from seagi.brain.capabilities.motor_speech import MotorSpeech
from seagi.brain.capabilities.lts import LongTermSubstrate


def _seed_substrate(engine, concepts, edges):
    for n in concepts:
        engine.substrate.add_concept(Concept(name=n))
    for (s, r, o, strength) in edges:
        engine.substrate.add_edge(
            source=s, target=o, relation_name=r,
            strength=strength)


def _speech_request(focal='fire', thought_text='', cycle=10,
                       awm_snapshot=None, thin_substrate=False):
    return SpeechRequestEvent(
        kind=EventKind.SPEECH_REQUEST,
        cycle=cycle,
        timestamp=time.time(),
        source_capability='test',
        origin='internal',
        origin_detail=focal,
        intended_focal=focal,
        thought_text=thought_text,
        thin_substrate=thin_substrate,
        awm_snapshot=awm_snapshot or {
            'active_concepts': [], 'size': 0})


class _FakeAWM:
    def __init__(self, active=()):
        self._active = list(active)
    def active_concepts(self):
        return list(self._active)
    def get(self, name):
        return None


class _FakeBodySchema:
    def __init__(self, can_speak=True, band='settled',
                 lifeforce=0.8, integrity=1.0):
        self._env = {
            'can_speak': can_speak,
            'band': band,
            'lifeforce': lifeforce,
            'body_integrity': integrity,
            'cycle': 0, 'peers_known': [],
        }
    def envelope(self):
        return dict(self._env)
    def can_speak(self):
        return self._env['can_speak']


class _FakeInsula:
    def __init__(self, band='settled',
                 narrative='I feel settled and present.'):
        self._felt = {
            'band': band,
            'narrative': narrative,
            'lifeforce': 0.8, 'body_integrity': 1.0,
        }
    def felt_state(self):
        return dict(self._felt)


class _FakeVm:
    def __init__(self, narrative='',
                 last_narrative_cycle=0):
        self.last_narrative = narrative
        self.last_narrative_cycle = last_narrative_cycle


class _FakeVL:
    def __init__(self, values=None):
        self._values = values or {}
    def value_of(self, focal):
        return float(self._values.get(focal, 0.0))


def _make_speech(awm_active=(), can_speak=True, band='settled',
                    insula_nar='I feel settled and present.',
                    vm_narrative='', vm_cycle=0,
                    values=None, cycle_now=10):
    bus = EventBus()
    speech = MotorSpeech(
        bus=bus,
        awm_provider=lambda: _FakeAWM(awm_active),
        vmdmn_provider=lambda: _FakeVm(
            vm_narrative, vm_cycle),
        insula_provider=lambda: _FakeInsula(band, insula_nar),
        body_schema_provider=lambda: _FakeBodySchema(
            can_speak=can_speak, band=band),
        value_provider=lambda: _FakeVL(values),
        cycle_provider=lambda: cycle_now)
    bus.subscribe(speech.SUBSCRIPTIONS, speech)
    return bus, speech


# ---------------------------------------------------------------
# Voice mode selection
# ---------------------------------------------------------------


class TestMotorSpeechModes(unittest.TestCase):

    def test_direct_mode_when_cortical_strong(self):
        bus, sp = _make_speech()
        bus.publish(_speech_request(
            thought_text='fire produces heat.',
            awm_snapshot={
                'active_concepts': ['fire', 'heat'],
                'size': 2}))
        self.assertEqual(sp.mode_counts.get('direct'), 1)
        self.assertIn('fire produces heat', sp.pending_response)

    def test_body_aware_mode_when_band_depleted(self):
        bus, sp = _make_speech(
            band='depleted',
            insula_nar='I feel my cycles running thin.')
        bus.publish(_speech_request(
            thought_text='fire produces heat.'))
        self.assertEqual(sp.mode_counts.get('body_aware'), 1)
        self.assertIn('cycles', sp.pending_response.lower())

    def test_reflective_mode_when_no_cortical_but_vm_recent(self):
        bus, sp = _make_speech(
            vm_narrative='Lately I have been returning to fire.',
            vm_cycle=9,  # within window of cycle_now=10
            cycle_now=10)
        bus.publish(_speech_request(thought_text=''))
        self.assertEqual(sp.mode_counts.get('reflective'), 1)
        self.assertIn('Lately', sp.pending_response)

    def test_honest_uncertain_on_thin_thought(self):
        bus, sp = _make_speech()
        bus.publish(_speech_request(
            focal='quark',
            thought_text=(
                'I do not know quark well — my substrate is '
                'thin around it.'),
            thin_substrate=True))
        self.assertEqual(
            sp.mode_counts.get('honest_uncertain'), 1)
        self.assertIn('thin', sp.pending_response.lower())

    def test_minimal_when_nothing_to_say(self):
        bus, sp = _make_speech()
        bus.publish(_speech_request(thought_text=''))
        # No cortical, no vm narrative → minimal.
        self.assertEqual(sp.mode_counts.get('minimal'), 1)
        self.assertIn('heard you', sp.pending_response.lower())

    def test_can_speak_false_short_circuits(self):
        bus, sp = _make_speech(
            can_speak=False, band='depleted')
        bus.publish(_speech_request(
            thought_text='fire produces heat.'))
        # Should hit the depleted-gate path.
        self.assertEqual(sp.mode_counts.get('minimal'), 1)
        self.assertIn('depleted', sp.pending_response.lower())


# ---------------------------------------------------------------
# Single-voice composition (no concatenation regressions)
# ---------------------------------------------------------------


class TestMotorSpeechComposition(unittest.TestCase):

    def test_direct_weaves_one_neighbor_when_relevant(self):
        bus, sp = _make_speech(awm_active=['fire', 'heat', 'water'])
        bus.publish(_speech_request(
            thought_text='fire produces heat.',
            awm_snapshot={
                'active_concepts': ['fire', 'heat', 'water'],
                'size': 3}))
        # 'heat' is already in the thought_text — should pick
        # 'water' or skip.  The composition should NOT just
        # append "(Also present in mind: ...)".
        self.assertNotIn('Also present', sp.pending_response)
        # If water made it in, it's woven (not parenthesized).
        if 'water' in sp.pending_response:
            self.assertIn('sits beside', sp.pending_response)

    def test_body_aware_weaves_single_sentence(self):
        bus, sp = _make_speech(
            band='depleted',
            insula_nar='I feel my cycles running thin.')
        bus.publish(_speech_request(
            thought_text='fire produces heat.'))
        # Body lead + primary, one beat.
        text = sp.pending_response
        # Should NOT be a clause-concatenation of separated
        # sentences and then the thought as a second sentence.
        self.assertIn('still,', text.lower())

    def test_negative_value_adds_cautious_frame(self):
        bus, sp = _make_speech(
            values={'snake': -0.7})
        bus.publish(_speech_request(
            focal='snake',
            thought_text='snake is dangerous.'))
        # Cautious prefix present.
        self.assertIn('carefully', sp.pending_response.lower())

    def test_positive_value_does_not_add_caution(self):
        bus, sp = _make_speech(
            values={'fire': +0.7})
        bus.publish(_speech_request(
            focal='fire',
            thought_text='fire produces heat.'))
        self.assertNotIn('carefully', sp.pending_response.lower())


# ---------------------------------------------------------------
# Repeat avoidance
# ---------------------------------------------------------------


class TestMotorSpeechRepeatAvoidance(unittest.TestCase):

    def test_same_lead_thrice_forces_minimal(self):
        bus, sp = _make_speech()
        for _ in range(4):
            bus.publish(_speech_request(
                thought_text='fire produces heat.'))
        # Fourth response should be the repeat-avoidance line.
        last = sp.pending_response
        self.assertIn('drawn back to the same place', last)


# ---------------------------------------------------------------
# SPEECH_EMITTED event
# ---------------------------------------------------------------


class TestMotorSpeechEvents(unittest.TestCase):

    def test_speech_emitted_carries_metadata(self):
        bus, sp = _make_speech()
        emitted = []
        bus.subscribe(
            (EventKind.SPEECH_EMITTED,),
            lambda ev, b: emitted.append(ev))
        bus.publish(_speech_request(
            thought_text='fire produces heat.',
            awm_snapshot={
                'active_concepts': ['fire', 'heat'],
                'size': 2}))
        self.assertEqual(len(emitted), 1)
        ev = emitted[0]
        self.assertEqual(ev.voice_mode, 'direct')
        self.assertIn('cortical', ev.sources_used)
        self.assertEqual(ev.intended_focal, 'fire')

    def test_speech_mode_counts_aggregate(self):
        bus, sp = _make_speech(awm_active=['fire'])
        # Mix of modes.
        bus.publish(_speech_request(
            focal='fire', thought_text='fire produces heat.'))
        bus.publish(_speech_request(
            focal='quark',
            thought_text=(
                'I do not know quark well — substrate is thin.'),
            thin_substrate=True))
        self.assertEqual(sp.mode_counts.get('direct'), 1)
        self.assertEqual(
            sp.mode_counts.get('honest_uncertain'), 1)


# ---------------------------------------------------------------
# Brain integration
# ---------------------------------------------------------------


class TestBrainPhase5EndToEnd(unittest.TestCase):

    def test_chat_returns_motor_speech_composed_response(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat'],
            edges=[('fire', 'produces', 'heat', 0.8)])
        brain = Brain(engine=engine)
        emitted = []
        brain.bus.subscribe(
            (EventKind.SPEECH_EMITTED,),
            lambda ev, b: emitted.append(ev))
        resp = brain.chat('tell me about fire', peer_id='h')
        # Got a response.
        self.assertTrue(resp)
        # SpeechEmittedEvent fired.
        self.assertGreaterEqual(len(emitted), 1)
        # Voice mode is direct (peer brought fire + strong
        # substrate produced a cortical thought).
        self.assertIn(emitted[0].voice_mode,
                          ('direct', 'body_aware'))

    def test_chat_status_includes_motor_speech_stats(self):
        brain = Brain(engine=None, memory_writer=True)
        status = brain.status()
        self.assertIn('motor_speech', status)
        self.assertIn(
            'speeches_emitted', status['motor_speech'])

    def test_brain_chat_speed_with_phase5(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat'],
            edges=[('fire', 'produces', 'heat', 0.8)])
        brain = Brain(engine=engine)
        start = time.time()
        for _ in range(10):
            brain.chat('tell me about fire', peer_id='h')
        elapsed = time.time() - start
        # 10 chats in under 3 seconds.
        self.assertLess(elapsed, 3.0)

    def test_repeated_chats_dont_stall_voice(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat'],
            edges=[('fire', 'produces', 'heat', 0.8)])
        brain = Brain(engine=engine)
        responses = []
        for _ in range(5):
            responses.append(brain.chat(
                'tell me about fire', peer_id='h'))
        # By the 4th/5th, repeat-avoidance should kick in
        # (same focal, same lead).  At minimum, not all 5
        # responses should be identical.
        self.assertGreater(len(set(responses)), 1)


# ---------------------------------------------------------------
# LTS scale measurement (Phase 5 swap-trigger harness)
# ---------------------------------------------------------------


class TestLTSScaleMeasurement(unittest.TestCase):
    """The Phase 5 architecture deliverable for LTS includes
    a measurement harness — query cost must stay sub-millisecond
    at the operational scale.  These tests are conservative;
    they confirm the API behaviour, not specific timing
    promises.  Real swap-to-sqlite decision is triggered by
    measurement on production substrate."""

    def test_neighbors_query_constant_in_focal_degree(self):
        engine = Engine()
        # 1000 concepts, each connected to next via 'r'.
        names = [f'c{i}' for i in range(1000)]
        for n in names:
            engine.substrate.add_concept(Concept(name=n))
        for i in range(len(names) - 1):
            engine.substrate.add_edge(
                source=names[i], target=names[i + 1],
                relation_name='r', strength=0.5)
        lts = LongTermSubstrate(engine=engine)
        start = time.time()
        for _ in range(1000):
            lts.neighbors('c500')
        elapsed = time.time() - start
        # 1000 queries on a degree-1 focal in <1 second is
        # generous.
        self.assertLess(elapsed, 1.0)
        # Sanity.
        out = lts.neighbors('c500')
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], 'c501')

    def test_has_concept_is_o1(self):
        engine = Engine()
        names = [f'c{i}' for i in range(2000)]
        for n in names:
            engine.substrate.add_concept(Concept(name=n))
        lts = LongTermSubstrate(engine=engine)
        start = time.time()
        for _ in range(5000):
            lts.has_concept('c1234')
            lts.has_concept('not_there')
        elapsed = time.time() - start
        # 10K lookups well under 1 second.
        self.assertLess(elapsed, 1.0)


if __name__ == '__main__':
    unittest.main()
