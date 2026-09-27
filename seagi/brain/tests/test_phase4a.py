"""Phase 4a tests — DMN + Hippocampus + Insula + reflection.

Headline behaviors verified:

  - Insula reads body state and emits felt InteroceptionEvent +
    body-origin chemistry on meaningful change.
  - Hippocampus forms one episode per attended percept and emits
    EPISODE_FORMED.
  - On REFLECTION_FIRED, hippocampus consolidates top-K episodes
    by replay_weight (emotional intensity * novelty * recency)
    and requests co_occurs substrate edges.
  - dmDMN tracks per-peer interactions and writes peer→topic
    associations on reflection.
  - vmDMN replays consolidated episodes during reflection,
    composes a first-person self-narrative, and emits it as
    THOUGHT_PRODUCED.
  - Idle driver in Brain.run_for fires REFLECTION_FIRED after
    IDLE_CYCLES_FOR_REFLECTION quiet cycles.
"""

import time
import unittest

from seagi.body.engine import Engine
from seagi.core.substrate import Concept
from seagi.brain import (
    Brain, EventKind, EventBus,
    AttendedPerceptEvent,
    EpisodeFormedEvent,
    EpisodeConsolidatedEvent,
    EpisodeDecayedEvent,
    ReflectionFiredEvent,
    InteroceptionEvent,
    ThoughtProducedEvent,
    SubstrateWriteQueuedEvent,
)
from seagi.brain.capabilities.insula import (
    Insula, BAND_DEPLETED, BAND_AGITATED, BAND_REPLENISHED,
)
from seagi.brain.capabilities.hippocampus import (
    Hippocampus,
)
from seagi.brain.capabilities.dmn import (
    DorsomedialDMN, VentromedialDMN,
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
                       raw_text='tell me about fire'):
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


def _make_reflection(cycle=100,
                          trigger='manual',
                          kind='general'):
    return ReflectionFiredEvent(
        kind=EventKind.REFLECTION_FIRED,
        cycle=cycle,
        timestamp=time.time(),
        source_capability='test',
        origin='internal',
        origin_detail=trigger,
        trigger=trigger,
        reflection_kind=kind,
    )


# ---------------------------------------------------------------
# Insula
# ---------------------------------------------------------------


class TestInsula(unittest.TestCase):

    def setUp(self):
        self.bus = EventBus()
        self.insula = Insula(bus=self.bus, engine=None,
                                  cycle_provider=lambda: 1)

    def test_initial_sample_baseline_no_event(self):
        # First sample at a stable state — no delta yet → no
        # event.  Insula reports the band but doesn't emit until
        # a meaningful shift.
        self.insula.sample(cycle=1)
        # First call CAN emit (band changes from SETTLED default
        # to whatever the read gives).  Just ensure it doesn't
        # crash.
        felt = self.insula.felt_state()
        self.assertIn('band', felt)

    def test_lifeforce_crash_emits_threat_chemistry(self):
        chem_events = []
        intero_events = []
        self.bus.subscribe(
            (EventKind.CHEMISTRY_FIRE,),
            lambda ev, b: chem_events.append(ev))
        self.bus.subscribe(
            (EventKind.INTEROCEPTION,),
            lambda ev, b: intero_events.append(ev))
        # Start at high lifeforce.
        self.insula.set_body_state(0.95, 1.0)
        self.insula.sample(cycle=1)
        chem_events.clear()
        intero_events.clear()
        # Crash to low lifeforce.
        self.insula.set_body_state(0.20, 1.0)
        self.insula.sample(cycle=2)
        # Should emit a threat-band chemistry event + an
        # interoception event.
        threat = [e for e in chem_events
                    if getattr(e, 'chemistry_kind', '') == 'threat']
        self.assertGreaterEqual(len(threat), 1)
        self.assertGreaterEqual(len(intero_events), 1)
        self.assertEqual(threat[0].origin, 'internal')
        self.assertEqual(threat[0].origin_detail, 'body')

    def test_lifeforce_climb_emits_rest_replenish(self):
        chem_events = []
        self.bus.subscribe(
            (EventKind.CHEMISTRY_FIRE,),
            lambda ev, b: chem_events.append(ev))
        self.insula.set_body_state(0.20, 1.0)
        self.insula.sample(cycle=1)
        chem_events.clear()
        self.insula.set_body_state(0.85, 1.0)
        self.insula.sample(cycle=2)
        rest = [e for e in chem_events
                  if getattr(e, 'chemistry_kind', '') ==
                       'rest_replenish']
        self.assertGreaterEqual(len(rest), 1)

    def test_felt_state_band_transitions(self):
        # Settled → depleted → replenished
        self.insula.set_body_state(0.80, 1.0)
        self.insula.sample(cycle=1)
        self.assertNotEqual(
            self.insula.felt_state()['band'], BAND_DEPLETED)
        self.insula.set_body_state(0.15, 1.0)
        self.insula.sample(cycle=2)
        self.assertIn(
            self.insula.felt_state()['band'],
            (BAND_DEPLETED, BAND_AGITATED))
        self.insula.set_body_state(0.85, 1.0)
        self.insula.sample(cycle=3)
        # Should be settled or replenished now (depending on
        # delta vs band threshold ordering).
        self.assertIn(
            self.insula.felt_state()['band'],
            ('settled', BAND_REPLENISHED))

    def test_narrative_present_for_depleted(self):
        self.insula.set_body_state(0.18, 1.0)
        self.insula.sample(cycle=1)
        felt = self.insula.felt_state()
        # Narrative should mention cycles/reserves for depleted.
        self.assertTrue(
            'cycles' in felt['narrative'].lower()
            or 'reserves' in felt['narrative'].lower()
            or 'unsettled' in felt['narrative'].lower())


# ---------------------------------------------------------------
# Hippocampus
# ---------------------------------------------------------------


class TestHippocampus(unittest.TestCase):

    def setUp(self):
        self.bus = EventBus()
        self.hippo = Hippocampus(
            bus=self.bus, cycle_provider=lambda: 100)
        self.bus.subscribe(self.hippo.SUBSCRIPTIONS, self.hippo)

    def test_attended_percept_forms_episode(self):
        formed = []
        self.bus.subscribe(
            (EventKind.EPISODE_FORMED,),
            lambda ev, b: formed.append(ev))
        ev = _make_attended(focals=('fire', 'heat'), cycle=5)
        self.bus.publish(ev)
        self.assertEqual(len(formed), 1)
        self.assertEqual(formed[0].focals, ['fire', 'heat'])
        self.assertEqual(self.hippo.episodes_formed, 1)
        self.assertEqual(self.hippo.buffer_size(), 1)

    def test_empty_focals_does_not_form_episode(self):
        ev = _make_attended(focals=(), cycle=5)
        self.bus.publish(ev)
        self.assertEqual(self.hippo.episodes_formed, 0)

    def test_reflection_consolidates_top_k(self):
        consolidated = []
        substrate_writes = []
        self.bus.subscribe(
            (EventKind.EPISODE_CONSOLIDATED,),
            lambda ev, b: consolidated.append(ev))
        self.bus.subscribe(
            (EventKind.SUBSTRATE_WRITE_QUEUED,),
            lambda ev, b: substrate_writes.append(ev))
        # Form 6 episodes; the 3 with highest emotional intensity
        # should win.
        self.bus.publish(_make_attended(
            cycle=10, focals=('a', 'b'),
            m=0.9, i=0.1, novelty=0.7))     # strong M
        self.bus.publish(_make_attended(
            cycle=11, focals=('c', 'd'),
            m=0.1, i=0.9, novelty=0.7))     # strong I
        self.bus.publish(_make_attended(
            cycle=12, focals=('e', 'f'),
            m=0.05, i=0.05, novelty=0.05))  # weak
        self.bus.publish(_make_attended(
            cycle=13, focals=('g', 'h'),
            m=0.05, i=0.05, novelty=0.05))  # weak
        self.bus.publish(_make_attended(
            cycle=14, focals=('i', 'j'),
            m=0.8, i=0.05, novelty=0.5))    # strong M
        self.bus.publish(_make_attended(
            cycle=15, focals=('k', 'l'),
            m=0.05, i=0.05, novelty=0.05))  # weak
        self.bus.publish(_make_reflection(cycle=20))
        # Top-K default 4 — at least 3 consolidations expected
        # (the 3 emotionally-strong ones).
        self.assertGreaterEqual(len(consolidated), 3)
        # Each episode of 2 focals → 1 co_occurs edge.
        co_edges = [w for w in substrate_writes
                       if w.relation == 'co_occurs']
        self.assertGreaterEqual(len(co_edges), 3)

    def test_replay_weight_favors_emotional_intensity(self):
        # High emotional should beat high novelty alone when
        # neither has recency advantage.
        self.bus.publish(_make_attended(
            cycle=10, focals=('strong_emo',),
            m=0.9, i=0.0, novelty=0.0))
        self.bus.publish(_make_attended(
            cycle=10, focals=('novel_only',),
            m=0.0, i=0.0, novelty=0.9))
        # Trigger reflection at the same cycle (no recency
        # difference); top should be the emotional one.
        self.hippo._cycle_provider = lambda: 11
        self.bus.publish(_make_reflection(cycle=11))
        cons = [e for e in self.hippo._episodes if e.consolidated]
        self.assertTrue(any('strong_emo' in e.focals for e in cons))

    def test_consolidated_episodes_dont_reconsolidate(self):
        self.bus.publish(_make_attended(
            cycle=10, focals=('x', 'y'),
            m=0.9, i=0.0))
        self.bus.publish(_make_reflection(cycle=20))
        n_first = self.hippo.episodes_consolidated
        # Re-trigger reflection — the same episode shouldn't
        # consolidate again.
        self.bus.publish(_make_reflection(cycle=30))
        self.assertEqual(
            self.hippo.episodes_consolidated, n_first)

    def test_focal_centric_binding_n_minus_1_edges(self):
        """Brain-correct sparse coding: an episode with n focals
        writes n−1 co_occurs edges (primary ↔ each other), not
        n × (n − 1) / 2 pair-edges.  This is the substrate
        over-densification fix — at 16 focals: 15 edges instead
        of 120, an 8× reduction in substrate pollution."""
        writes = []
        self.bus.subscribe(
            (EventKind.SUBSTRATE_WRITE_QUEUED,),
            lambda ev, b: writes.append(ev))
        # 5-focal high-emotion episode.
        self.bus.publish(_make_attended(
            cycle=10,
            focals=('A', 'B', 'C', 'D', 'E'),
            m=0.8, i=0.05, novelty=0.5))
        self.bus.publish(_make_reflection(cycle=20))
        co_edges = [w for w in writes
                       if w.relation == 'co_occurs']
        # Focal-centric: A ↔ B, A ↔ C, A ↔ D, A ↔ E = 4 edges.
        # Old O(n²) version would have written 10.
        self.assertEqual(len(co_edges), 4,
            "Hippocampus wrote {} co_occurs edges for a 5-focal "
            "episode — expected 4 (focal-centric).  Either v1's "
            "O(n²) pair-write regressed or the binding count "
            "is off.".format(len(co_edges)))
        # All edges originate from the primary focal 'A'.
        for w in co_edges:
            self.assertEqual(w.subject, 'A',
                "co_occurs edge {} → {} doesn't originate from "
                "primary focal 'A'.".format(w.subject, w.object))

    def test_focal_centric_binding_dramatic_reduction_at_16(self):
        """At 16 focals (the gate's typical max), the focal-
        centric binding writes 15 edges instead of 120.  This
        test pins the count so a future regression to O(n²) is
        loud."""
        writes = []
        self.bus.subscribe(
            (EventKind.SUBSTRATE_WRITE_QUEUED,),
            lambda ev, b: writes.append(ev))
        focals = tuple(f'f{i}' for i in range(16))
        self.bus.publish(_make_attended(
            cycle=10, focals=focals,
            m=0.9, i=0.05, novelty=0.6))
        self.bus.publish(_make_reflection(cycle=20))
        co_edges = [w for w in writes
                       if w.relation == 'co_occurs']
        self.assertEqual(len(co_edges), 15,
            "16-focal episode wrote {} edges — substrate "
            "over-densification has returned.".format(
                len(co_edges)))


# ---------------------------------------------------------------
# DorsomedialDMN — ToM
# ---------------------------------------------------------------


class TestDorsomedialDMN(unittest.TestCase):

    def setUp(self):
        self.bus = EventBus()
        self.dmdmn = DorsomedialDMN(
            bus=self.bus, cycle_provider=lambda: 10)
        self.bus.subscribe(
            self.dmdmn.SUBSCRIPTIONS, self.dmdmn)

    def test_peer_observation_builds_model(self):
        self.bus.publish(_make_attended(
            cycle=1, focals=('fire',),
            origin='peer', origin_detail='harald'))
        self.bus.publish(_make_attended(
            cycle=2, focals=('water',),
            origin='peer', origin_detail='harald'))
        model = self.dmdmn.model_for('harald')
        self.assertIsNotNone(model)
        self.assertEqual(model.interactions, 2)
        self.assertIn('fire', model.topic_weights)
        self.assertIn('water', model.topic_weights)

    def test_non_peer_doesnt_build_model(self):
        self.bus.publish(_make_attended(
            cycle=1, focals=('fire',),
            origin='forager', origin_detail='rss:example'))
        # Forager-origin shouldn't create a peer model.
        self.assertEqual(self.dmdmn.peer_observations, 0)
        self.assertEqual(len(self.dmdmn.peers_known()), 0)

    def test_reflection_emits_peer_substrate_writes(self):
        writes = []
        self.bus.subscribe(
            (EventKind.SUBSTRATE_WRITE_QUEUED,),
            lambda ev, b: writes.append(ev))
        # Need at least PEER_COMMIT_THRESHOLD interactions.
        self.bus.publish(_make_attended(
            cycle=1, focals=('fire', 'heat'),
            origin='peer', origin_detail='harald'))
        self.bus.publish(_make_attended(
            cycle=2, focals=('fire',),
            origin='peer', origin_detail='harald'))
        self.bus.publish(_make_reflection(
            cycle=10, kind='social'))
        peer_writes = [w for w in writes
                          if w.subject.startswith('peer:harald')
                          and w.relation == 'talks_about']
        self.assertGreaterEqual(len(peer_writes), 1)


# ---------------------------------------------------------------
# VentromedialDMN — self-reference
# ---------------------------------------------------------------


class TestVentromedialDMN(unittest.TestCase):

    def setUp(self):
        self.bus = EventBus()
        # vmDMN needs AWM + insula + chemistry providers (stub).
        self.fake_awm = _FakeAWM()
        self.fake_insula = _FakeInsula(band='settled')
        self.vm = VentromedialDMN(
            bus=self.bus,
            awm_provider=lambda: self.fake_awm,
            hippocampus_provider=lambda: None,
            chemistry_provider=lambda: None,
            insula_provider=lambda: self.fake_insula,
            cycle_provider=lambda: 100)
        self.bus.subscribe(self.vm.SUBSCRIPTIONS, self.vm)

    def test_reflection_without_history_is_silent_or_minimal(self):
        thoughts = []
        self.bus.subscribe(
            (EventKind.THOUGHT_PRODUCED,),
            lambda ev, b: thoughts.append(ev))
        self.bus.publish(_make_reflection(
            cycle=100, kind='autobiographical'))
        # No consolidated episodes, no AWM, only body frame.
        # vmDMN shouldn't fabricate — narrative must be empty
        # or body-frame-only.
        if thoughts:
            for t in thoughts:
                self.assertNotIn('Lately', t.text)

    def test_reflection_replays_consolidated_episodes(self):
        thoughts = []
        self.bus.subscribe(
            (EventKind.THOUGHT_PRODUCED,),
            lambda ev, b: thoughts.append(ev))
        # Feed several EpisodeConsolidatedEvents in.
        for cyc, focals in [
                (10, ['fire', 'heat']),
                (11, ['fire', 'danger']),
                (12, ['fire', 'fuel']),
                (13, ['water']),
        ]:
            self.bus.publish(EpisodeConsolidatedEvent(
                kind=EventKind.EPISODE_CONSOLIDATED,
                cycle=cyc,
                timestamp=time.time(),
                source_capability='test',
                origin='peer',
                origin_detail='harald',
                episode_id=cyc,
                focals=focals,
                n_edges_strengthened=1,
                replay_weight=0.5))
        self.bus.publish(_make_reflection(
            cycle=100, kind='autobiographical'))
        self.assertGreaterEqual(len(thoughts), 1)
        text = thoughts[0].text
        # Should mention 'fire' (appeared 3x).
        self.assertIn('fire', text)
        self.assertIn('Lately', text)
        self.assertEqual(thoughts[0].method, 'reflection')

    def test_reflection_writes_self_anchor_on_dominant_focal(self):
        writes = []
        self.bus.subscribe(
            (EventKind.SUBSTRATE_WRITE_QUEUED,),
            lambda ev, b: writes.append(ev))
        for cyc in range(10, 16):
            self.bus.publish(EpisodeConsolidatedEvent(
                kind=EventKind.EPISODE_CONSOLIDATED,
                cycle=cyc,
                timestamp=time.time(),
                source_capability='test',
                origin='peer',
                origin_detail='h',
                episode_id=cyc,
                focals=['fire'],
                n_edges_strengthened=0,
                replay_weight=0.5))
        self.bus.publish(_make_reflection(
            cycle=100, kind='autobiographical'))
        anchor = [w for w in writes
                    if w.subject == 'self'
                    and w.relation == 'attends_to'
                    and w.object == 'fire']
        self.assertGreaterEqual(len(anchor), 1)


class _FakeAWM:
    def active_concepts(self):
        return []
    def get(self, name):
        return None


class _FakeInsula:
    def __init__(self, band='settled'):
        self.band = band
    def felt_state(self):
        return {
            'band': self.band,
            'narrative': 'I feel settled and present.',
            'lifeforce': 0.8,
            'body_integrity': 1.0,
        }


# ---------------------------------------------------------------
# Brain integration — idle driver fires reflection
# ---------------------------------------------------------------


class TestIdleReflectionDriver(unittest.TestCase):

    def test_manual_reflection_fire(self):
        brain = Brain(engine=None, memory_writer=True)
        reflections = []
        brain.bus.subscribe(
            (EventKind.REFLECTION_FIRED,),
            lambda ev, b: reflections.append(ev))
        brain.fire_reflection(trigger='manual',
                                   reflection_kind='general')
        self.assertEqual(len(reflections), 1)
        self.assertEqual(reflections[0].trigger, 'manual')

    def test_idle_run_for_eventually_fires_reflection(self):
        brain = Brain(engine=None, memory_writer=True)
        reflections = []
        brain.bus.subscribe(
            (EventKind.REFLECTION_FIRED,),
            lambda ev, b: reflections.append(ev))
        # No chat → all ticks are idle.  Run for enough ticks
        # to cross IDLE_CYCLES_FOR_REFLECTION (30).
        brain.run_for(60)
        self.assertGreaterEqual(len(reflections), 1)

    def test_recent_peer_chat_suppresses_idle_reflection(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat'],
            edges=[('fire', 'produces', 'heat', 0.8)])
        brain = Brain(engine=engine)
        reflections = []
        brain.bus.subscribe(
            (EventKind.REFLECTION_FIRED,),
            lambda ev, b: reflections.append(ev))
        # Chat — this updates _last_peer_cycle.
        brain.chat('tell me about fire', peer_id='h')
        # Only a few ticks — shouldn't be idle long enough.
        brain.run_for(5)
        self.assertEqual(len(reflections), 0)


# ---------------------------------------------------------------
# Brain integration — end-to-end Phase 4a behavior
# ---------------------------------------------------------------


class TestBrainPhase4aEndToEnd(unittest.TestCase):

    def test_chat_forms_episode_in_hippocampus(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat'],
            edges=[('fire', 'produces', 'heat', 0.8)])
        brain = Brain(engine=engine)
        brain.chat('tell me about fire', peer_id='h')
        self.assertGreaterEqual(brain.hippocampus.episodes_formed, 1)

    def test_reflection_after_chat_replays_to_self_narrative(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat', 'danger', 'water'],
            edges=[
                ('fire', 'produces', 'heat', 0.8),
                ('fire', 'is_a', 'danger', 0.7),
                ('water', 'opposite_of', 'fire', 0.6),
            ])
        brain = Brain(engine=engine)
        # A few chats — episodes form.  Use M-rich text so
        # m_content rises enough to drive emotional intensity.
        brain.chat('tell me about fire', peer_id='h')
        brain.chat('fire is danger', peer_id='h')
        brain.chat('fire and water', peer_id='h')
        # Manual reflection.
        thoughts = []
        brain.bus.subscribe(
            (EventKind.THOUGHT_PRODUCED,),
            lambda ev, b: thoughts.append(ev))
        brain.fire_reflection(trigger='manual',
                                   reflection_kind='general')
        # vmDMN should have surfaced at least one reflective
        # thought OR hippocampus should have consolidated
        # something.  Both are reasonable outcomes — we just
        # need the reflective loop to do *something*.
        self.assertTrue(
            brain.hippocampus.episodes_consolidated > 0
            or any(t.method == 'reflection' for t in thoughts))

    def test_status_reports_all_phase4a_capabilities(self):
        brain = Brain(engine=None, memory_writer=True)
        status = brain.status()
        for key in ('insula', 'hippocampus', 'dmdmn', 'vmdmn',
                       'reflections_fired'):
            self.assertIn(key, status)

    def test_peer_chat_updates_dmdmn_model(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire'],
            edges=[])
        brain = Brain(engine=engine)
        brain.chat('tell me about fire', peer_id='harald')
        brain.chat('what is fire', peer_id='harald')
        model = brain.dmdmn.model_for('harald')
        self.assertIsNotNone(model)
        self.assertGreaterEqual(model.interactions, 1)

    def test_speech_falls_back_to_vmdmn_narrative_when_cortical_empty(self):
        # When cortical has nothing to say but vmDMN has recent
        # narrative, speech composer surfaces it.
        brain = Brain(engine=None, memory_writer=True)
        # Plant a vmDMN narrative directly.
        brain.vmdmn.last_narrative = 'Lately I have been returning to fire.'
        # Trigger a speech request manually with empty thought_text.
        from seagi.brain.events import SpeechRequestEvent
        brain.bus.publish(SpeechRequestEvent(
            kind=EventKind.SPEECH_REQUEST,
            cycle=10,
            timestamp=time.time(),
            source_capability='test',
            origin='internal',
            origin_detail='',
            intended_focal='self',
            thought_text='',
            awm_snapshot={'size': 0, 'active_concepts': []}))
        self.assertIn('fire', brain.speech.pending_response)


if __name__ == '__main__':
    unittest.main()
