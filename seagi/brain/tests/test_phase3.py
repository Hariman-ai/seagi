"""Phase 3 tests — event processor + Cortical Reasoning + chat().

Headline: `brain.chat("tell me about fire")` returns a real
response composed from substrate state, in milliseconds, with
no engine_lock contention.

Also covers cortical sub-functions:
  - metacognitive thin-substrate detection
  - causal chain reasoning (1-2 hops)
  - identity walk (is_a)
  - schema matching on AWM
  - counterfactual hint
  - substrate writes from schema inferences
"""

import time
import unittest

from seagi.body.engine import Engine
from seagi.core.substrate import Concept, Bubble, ContextKey
from seagi.core.mi_value import TransmitterState
from seagi.brain import (
    Brain, EventKind, EventBus,
    AttendedPerceptEvent, ThoughtProducedEvent,
    SpeechRequestEvent, SubstrateWriteQueuedEvent,
)
from seagi.brain.capabilities.cortical import (
    CorticalReasoner, STANDARD_SCHEMAS, Schema,
)
from seagi.brain.capabilities.awm import ActiveWorkingMemory
from seagi.brain.capabilities.lts import LongTermSubstrate


def _seed_substrate(engine, concepts, edges):
    """Helper: add a small substrate."""
    for n in concepts:
        engine.substrate.add_concept(Concept(name=n))
    for (s, r, o, strength) in edges:
        engine.substrate.add_edge(
            source=s, target=o, relation_name=r,
            strength=strength)


# ---------------------------------------------------------------------
# Cortical sub-functions
# ---------------------------------------------------------------------


class TestCorticalThinking(unittest.TestCase):

    def setUp(self):
        self.engine = Engine()
        self.brain = Brain(engine=self.engine)

    def _think(self, focal):
        # Use the internal _think_about directly for unit tests.
        return self.brain.cortical._think_about(
            focal, cycle=10)

    def test_thin_substrate_metacog_response(self):
        # Concept exists but no edges — thin substrate.
        self.engine.substrate.add_concept(Concept(name='quark'))
        thought = self._think('quark')
        self.assertIsNotNone(thought)
        self.assertTrue(thought.thin_substrate)
        self.assertEqual(thought.method, 'metacog')
        self.assertIn('thin', thought.text.lower())

    def test_unknown_concept_returns_thin(self):
        thought = self._think('never_heard_of_this')
        self.assertIsNotNone(thought)
        self.assertTrue(thought.thin_substrate)

    def test_causal_chain_one_hop(self):
        _seed_substrate(self.engine,
            concepts=['fire', 'heat', 'danger', 'fuel'],
            edges=[
                ('fire', 'produces', 'heat', 0.8),
                ('fire', 'is_a', 'danger', 0.6),
                ('fire', 'requires', 'fuel', 0.7),
            ])
        thought = self._think('fire')
        self.assertIsNotNone(thought)
        self.assertEqual(thought.method, 'causal')
        # 'produces' is in CAUSAL_RELATIONS — should be picked.
        self.assertIn(thought.target, ('heat', 'danger', 'fuel'))

    def test_causal_chain_two_hops(self):
        _seed_substrate(self.engine,
            concepts=['fire', 'heat', 'pain', 'sense'],
            edges=[
                ('fire', 'causes', 'heat', 0.9),
                ('heat', 'causes', 'pain', 0.9),
                # No edges from fire directly to pain.
            ])
        thought = self._think('fire')
        self.assertIsNotNone(thought)
        # Phase R.1: cortical now COMPOSES the chain — fire causes
        # heat, heat causes pain → derives 'fire leads_to pain', a
        # conclusion the substrate doesn't directly contain.
        self.assertEqual(thought.target, 'pain')
        self.assertEqual(thought.method, 'inference')
        self.assertEqual(thought.relation, 'leads_to')

    def test_identity_walk_when_no_causal(self):
        _seed_substrate(self.engine,
            concepts=['hammer', 'tool'],
            edges=[('hammer', 'is_a', 'tool', 0.9)])
        thought = self._think('hammer')
        self.assertIsNotNone(thought)
        # 'is_a' is in IDENTITY_RELATIONS — picked as fallback.
        self.assertEqual(thought.target, 'tool')

    def test_counterfactual_when_only_downstream(self):
        # focal causes things but no other relations.
        _seed_substrate(self.engine,
            concepts=['rain', 'wetness', 'puddle'],
            edges=[
                ('rain', 'causes', 'wetness', 0.8),
                ('rain', 'causes', 'puddle', 0.6),
            ])
        thought = self._think('rain')
        self.assertIsNotNone(thought)
        # 'causes' picked up by causal_chain first.
        self.assertEqual(thought.method, 'causal')


class TestPhaseR1Inference(unittest.TestCase):
    """Phase R.1 — compositional multi-hop inference.  Cortical
    composes relations across hops to derive conclusions the
    substrate does not directly contain."""

    def setUp(self):
        self.engine = Engine()
        self.brain = Brain(engine=self.engine)

    def _think(self, focal):
        return self.brain.cortical._think_about(focal, cycle=10)

    def test_syllogism_is_a_transitive(self):
        # Socrates is_a man, man is_a mortal ⊢ Socrates is_a mortal.
        # The substrate has NO direct socrates→mortal edge.
        _seed_substrate(self.engine,
            concepts=['socrates', 'man', 'mortal'],
            edges=[
                ('socrates', 'is_a', 'man', 0.9),
                ('man', 'is_a', 'mortal', 0.9),
            ])
        t = self._think('socrates')
        self.assertIsNotNone(t)
        self.assertEqual(t.method, 'inference')
        self.assertEqual(t.relation, 'is_a')
        self.assertEqual(t.target, 'mortal')
        # The conclusion is derived, not stored.
        self.assertNotIn(
            ('socrates', 'is_a', 'mortal'),
            self.engine.substrate.edges)

    def test_property_inheritance(self):
        # dog is_a mammal, mammal has_property warm_blooded
        #   ⊢ dog has_property warm_blooded.
        _seed_substrate(self.engine,
            concepts=['dog', 'mammal', 'warm_blooded'],
            edges=[
                ('dog', 'is_a', 'mammal', 0.9),
                ('mammal', 'has_property', 'warm_blooded', 0.9),
            ])
        t = self._think('dog')
        self.assertIsNotNone(t)
        self.assertEqual(t.method, 'inference')
        self.assertEqual(t.relation, 'has_property')
        self.assertEqual(t.target, 'warm_blooded')

    def test_deep_causal_chain_composes(self):
        # 4-hop causes chain → leads_to terminal.
        _seed_substrate(self.engine,
            concepts=['a', 'b', 'c', 'd', 'e'],
            edges=[
                ('a', 'causes', 'b', 0.9),
                ('b', 'causes', 'c', 0.9),
                ('c', 'causes', 'd', 0.9),
                ('d', 'causes', 'e', 0.9),
            ])
        t = self._think('a')
        self.assertIsNotNone(t)
        self.assertEqual(t.method, 'inference')
        self.assertEqual(t.relation, 'leads_to')
        # Walk bounded at INFERENCE_MAX_HOPS (4) → terminal 'e'.
        self.assertEqual(t.target, 'e')

    def test_chain_breaks_on_noncomposable_relation(self):
        # is_a then opposite_of — that pair is NOT in
        # RELATION_COMPOSITION, so the chain stops after hop 1.
        # One hop is not an inference → falls through to the
        # plain identity walk.
        _seed_substrate(self.engine,
            concepts=['x', 'y', 'z'],
            edges=[
                ('x', 'is_a', 'y', 0.9),
                ('y', 'opposite_of', 'z', 0.9),
            ])
        t = self._think('x')
        self.assertIsNotNone(t)
        # Not an inference — only a 1-hop is_a is available.
        self.assertNotEqual(t.method, 'inference')

    def test_inference_text_explains_the_chain(self):
        _seed_substrate(self.engine,
            concepts=['fire', 'heat', 'pain'],
            edges=[
                ('fire', 'causes', 'heat', 0.9),
                ('heat', 'causes', 'pain', 0.9),
            ])
        t = self._think('fire')
        self.assertEqual(t.method, 'inference')
        # The rendered thought names the conclusion AND the chain.
        self.assertIn('infer', t.text.lower())
        self.assertIn('because', t.text.lower())
        self.assertIn('fire causes heat', t.text.lower())
        self.assertIn('heat causes pain', t.text.lower())

    def test_confidence_decays_with_chain_length(self):
        # A 2-hop inference is more confident than a 4-hop one.
        _seed_substrate(self.engine,
            concepts=['p', 'q', 'r'],
            edges=[
                ('p', 'causes', 'q', 0.9),
                ('q', 'causes', 'r', 0.9),
            ])
        short = self._think('p')
        engine2 = Engine()
        brain2 = Brain(engine=engine2)
        _seed_substrate(engine2,
            concepts=['a', 'b', 'c', 'd', 'e'],
            edges=[
                ('a', 'causes', 'b', 0.9),
                ('b', 'causes', 'c', 0.9),
                ('c', 'causes', 'd', 0.9),
                ('d', 'causes', 'e', 0.9),
            ])
        deep = brain2.cortical._think_about('a', cycle=10)
        self.assertGreater(short.confidence, deep.confidence)

    def test_single_hop_is_not_inference(self):
        # Just one edge — a stored fact, not a derivation.
        _seed_substrate(self.engine,
            concepts=['cat', 'animal'],
            edges=[('cat', 'is_a', 'animal', 0.9)])
        t = self._think('cat')
        self.assertIsNotNone(t)
        self.assertNotEqual(t.method, 'inference')

    def test_inheritance_of_causal_power(self):
        # X is_a Y, Y causes Z ⊢ X causes Z.
        _seed_substrate(self.engine,
            concepts=['ember', 'fire', 'burn'],
            edges=[
                ('ember', 'is_a', 'fire', 0.9),
                ('fire', 'causes', 'burn', 0.9),
            ])
        t = self._think('ember')
        self.assertEqual(t.method, 'inference')
        self.assertEqual(t.relation, 'causes')
        self.assertEqual(t.target, 'burn')


class TestRoadmapStep2ReasoningCompounds(unittest.TestCase):
    """Roadmap Step 2: a successful inference consolidates as
    new substrate knowledge — reasoning is no longer read-only.
    Knowledge accumulates."""

    def _think_and_publish(self, brain, focal):
        """Helper: run cortical reasoning AND publish the thought
        on the bus, so the ReasoningConsolidator subscriber sees
        it (mirrors the production _on_attended → publish path)."""
        thought = brain.cortical._think_about(
            focal, cycle=brain._cycle_provider())
        if thought is not None:
            brain.bus.publish(thought)
        return thought

    def test_derived_edge_lands_in_substrate(self):
        # Socrates is_a man, man is_a mortal ⊢ Socrates is_a mortal.
        # After the inference, the derived edge is in substrate.
        engine = Engine()
        _seed_substrate(engine,
            concepts=['socrates', 'man', 'mortal'],
            edges=[
                ('socrates', 'is_a', 'man', 0.9),
                ('man', 'is_a', 'mortal', 0.9),
            ])
        brain = Brain(engine=engine)
        self.assertNotIn(
            ('socrates', 'is_a', 'mortal'),
            engine.substrate.edges)
        thought = self._think_and_publish(brain, 'socrates')
        self.assertEqual(thought.method, 'inference')
        self.assertIn(
            ('socrates', 'is_a', 'mortal'),
            engine.substrate.edges,
            "derived edge should have been consolidated")

    def test_consolidated_edge_is_provisional(self):
        from seagi.core.substrate import PROVISIONAL_EDGE_STRENGTH
        engine = Engine()
        _seed_substrate(engine,
            concepts=['a', 'b', 'c'],
            edges=[('a', 'is_a', 'b', 0.9),
                       ('b', 'is_a', 'c', 0.9)])
        brain = Brain(engine=engine)
        self._think_and_publish(brain, 'a')
        derived = engine.substrate.edges.get(('a', 'is_a', 'c'))
        self.assertIsNotNone(derived)
        self.assertAlmostEqual(
            derived.strength, PROVISIONAL_EDGE_STRENGTH, places=2)

    def test_non_inference_thoughts_do_not_consolidate(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat'],
            edges=[('fire', 'causes', 'heat', 0.9)])
        brain = Brain(engine=engine)
        before = (
            brain.reasoning_consolidator.consolidations_emitted)
        thought = self._think_and_publish(brain, 'fire')
        self.assertNotEqual(thought.method, 'inference')
        self.assertEqual(
            brain.reasoning_consolidator.consolidations_emitted,
            before)

    def test_compounding_inference_builds_on_derived_edge(self):
        # An inference's derived edge becomes usable for future
        # reasoning — knowledge accumulates.
        engine = Engine()
        _seed_substrate(engine,
            concepts=['a', 'b', 'c', 'd'],
            edges=[('a', 'is_a', 'b', 0.9),
                       ('b', 'is_a', 'c', 0.9),
                       ('c', 'is_a', 'd', 0.9)])
        brain = Brain(engine=engine)
        t1 = self._think_and_publish(brain, 'a')
        self.assertEqual(t1.method, 'inference')
        self.assertIn(('a', 'is_a', 'd'),
                          engine.substrate.edges)
        self.assertGreaterEqual(
            brain.reasoning_consolidator.consolidations_emitted, 1)

    def test_consolidator_stats_in_brain_status(self):
        engine = Engine()
        brain = Brain(engine=engine)
        self.assertIn('reasoning_consolidator', brain.status())


class TestCorticalSchemaMatching(unittest.TestCase):

    def test_transitivity_match_emits_substrate_write(self):
        engine = Engine()
        # Seed: dog is_a mammal; mammal is_a animal.
        # Schema should infer dog is_a animal.
        _seed_substrate(engine,
            concepts=['dog', 'mammal', 'animal'],
            edges=[
                ('dog', 'is_a', 'mammal', 0.9),
                ('mammal', 'is_a', 'animal', 0.9),
            ])
        brain = Brain(engine=engine)
        # Promote both 'dog' and 'mammal' to AWM — schema can
        # only see AWM-bound starting points.
        brain.awm.promote('dog', salience=0.5)
        brain.awm.promote('mammal', salience=0.5)
        # Run schema match.
        brain.cortical._match_schemas(brain.bus, trigger='test')
        # dog → animal edge should now exist (via SubstrateWrite).
        edge = engine.substrate.edges.get(('dog', 'is_a', 'animal'))
        self.assertIsNotNone(edge)
        self.assertGreater(brain.cortical.inferences_emitted, 0)


# ---------------------------------------------------------------------
# Cortical event handling — peer input fires reasoning
# ---------------------------------------------------------------------


class TestCorticalEventFlow(unittest.TestCase):

    def test_attended_percept_triggers_thought(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat'],
            edges=[('fire', 'produces', 'heat', 0.8)])
        brain = Brain(engine=engine)
        # Collect THOUGHT_PRODUCED events.
        thoughts = []
        brain.bus.subscribe(
            (EventKind.THOUGHT_PRODUCED,),
            lambda ev, b: thoughts.append(ev))
        # Peer says "tell me about fire" — should pass gate
        # and produce a cortical thought.
        brain.intake('tell me about fire',
                       modality='text', origin='peer',
                       origin_detail='harald')
        self.assertGreaterEqual(len(thoughts), 1)
        thought = thoughts[0]
        self.assertEqual(thought.focal, 'fire')

    def test_attended_percept_triggers_speech_request(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat'],
            edges=[('fire', 'produces', 'heat', 0.8)])
        brain = Brain(engine=engine)
        requests = []
        brain.bus.subscribe(
            (EventKind.SPEECH_REQUEST,),
            lambda ev, b: requests.append(ev))
        brain.intake('tell me about fire',
                       modality='text', origin='peer',
                       origin_detail='harald')
        self.assertGreaterEqual(len(requests), 1)

    def test_forager_attended_does_not_fire_cortical_thought(self):
        # Forager-origin attended percepts should NOT produce a
        # full cortical reasoning response (only opportunistic
        # schema check).  Otherwise every RSS item would
        # trigger a thought.
        engine = Engine()
        brain = Brain(engine=engine)
        thoughts = []
        brain.bus.subscribe(
            (EventKind.THOUGHT_PRODUCED,),
            lambda ev, b: thoughts.append(ev))
        # Synthesize an AttendedPerceptEvent directly with
        # forager origin to bypass the gate (which is what
        # would happen with a salient forager arrival).
        ev = AttendedPerceptEvent(
            kind=EventKind.ATTENDED_PERCEPT,
            cycle=10, source_capability='gate',
            focals=['some_concept'],
            salience=0.5, novelty=0.5,
            origin='forager', origin_detail='rss')
        brain.bus.publish(ev)
        # Forager-origin → no cortical respond → no thought event.
        self.assertEqual(len(thoughts), 0)

    def test_reverie_attended_triggers_thought_no_speech(self):
        # Phase C.1.g: a reverie-origin attended percept (internal
        # origin, origin_detail='reverie:*') should engage
        # deliberate cortical reasoning — but NOT emit a speech
        # request (the agent is thinking to itself).
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat'],
            edges=[('fire', 'produces', 'heat', 0.8)])
        brain = Brain(engine=engine)
        thoughts = []
        requests = []
        brain.bus.subscribe(
            (EventKind.THOUGHT_PRODUCED,),
            lambda ev, b: thoughts.append(ev))
        brain.bus.subscribe(
            (EventKind.SPEECH_REQUEST,),
            lambda ev, b: requests.append(ev))
        brain.bus.publish(AttendedPerceptEvent(
            kind=EventKind.ATTENDED_PERCEPT,
            cycle=10, source_capability='idle_motivation',
            focals=['fire'],
            salience=0.5, novelty=0.2,
            origin='internal',
            origin_detail='reverie:substrate'))
        # Reverie engaged cortical → a thought about 'fire'.
        fire_thoughts = [t for t in thoughts
                              if t.focal == 'fire']
        self.assertGreaterEqual(len(fire_thoughts), 1)
        # But no speech request — reverie is inner dialogue.
        self.assertEqual(len(requests), 0)


# ---------------------------------------------------------------------
# End-to-end — brain.chat() returns a response
# ---------------------------------------------------------------------


class TestPhase3EndToEnd(unittest.TestCase):
    """The headline: brain.chat() works end-to-end."""

    def test_chat_returns_substrate_grounded_response(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat', 'danger'],
            edges=[
                ('fire', 'produces', 'heat', 0.8),
                ('fire', 'is_a', 'danger', 0.5),
            ])
        brain = Brain(engine=engine)
        response = brain.chat('tell me about fire',
                                 peer_id='harald')
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        # Response should mention fire (the focal we
        # discussed).
        self.assertIn('fire', response.lower())

    def test_chat_on_thin_substrate_is_honest(self):
        engine = Engine()
        # Concept exists but no edges.
        engine.substrate.add_concept(Concept(name='quark'))
        brain = Brain(engine=engine)
        response = brain.chat('tell me about quark')
        self.assertIn('quark', response.lower())
        # Honest response about thin substrate.
        self.assertTrue(
            'thin' in response.lower()
            or 'do not know' in response.lower()
            or 'no strong thought' in response.lower())

    def test_chat_is_fast(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire', 'heat', 'danger', 'fuel', 'water'],
            edges=[
                ('fire', 'produces', 'heat', 0.8),
                ('fire', 'is_a', 'danger', 0.5),
                ('fire', 'requires', 'fuel', 0.6),
                ('water', 'cools', 'fire', 0.4),
            ])
        brain = Brain(engine=engine)
        t = time.time()
        response = brain.chat('tell me about fire')
        elapsed = time.time() - t
        # Phase 3 goal: chat in milliseconds even with
        # substrate present.  Generous bound for CI.
        self.assertLess(elapsed, 0.5,
                          msg=f'chat took {elapsed*1000:.1f}ms')
        self.assertGreater(len(response), 0)

    def test_chat_builds_awm_state(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['fire'],
            edges=[])
        brain = Brain(engine=engine)
        brain.chat('what about fire')
        # AWM should have promoted fire.
        self.assertTrue(brain.awm.is_active('fire'))

    def test_run_for_advances_maintenance(self):
        engine = Engine()
        brain = Brain(engine=engine)
        # Spike chemistry.
        brain.chemistry.global_state['cortisol'] = 0.8
        # Run maintenance ticks.
        brain.run_for(20)
        # Cortisol should have decayed (slow channel — still
        # elevated but lower than 0.8).
        self.assertLess(brain.chemistry.global_state['cortisol'], 0.8)


# ---------------------------------------------------------------------
# Substrate writes from schema inferences
# ---------------------------------------------------------------------


class TestSubstrateWrites(unittest.TestCase):

    def test_inferred_edges_get_written(self):
        engine = Engine()
        _seed_substrate(engine,
            concepts=['dog', 'mammal', 'animal'],
            edges=[
                ('dog', 'is_a', 'mammal', 0.9),
                ('mammal', 'is_a', 'animal', 0.9),
            ])
        brain = Brain(engine=engine)
        # Force AWM activation for the chain.
        brain.awm.promote('dog', salience=0.5)
        brain.awm.promote('mammal', salience=0.5)
        brain.cortical._match_schemas(brain.bus, trigger='test')
        # dog → animal edge written.
        edge = engine.substrate.edges.get(
            ('dog', 'is_a', 'animal'))
        self.assertIsNotNone(edge)
        self.assertGreater(brain.writer.writes_applied, 0)

    def test_writer_idempotent_on_existing_edge(self):
        engine = Engine()
        # Pre-existing edge — should not be re-written.
        _seed_substrate(engine,
            concepts=['dog', 'mammal', 'animal'],
            edges=[
                ('dog', 'is_a', 'mammal', 0.9),
                ('mammal', 'is_a', 'animal', 0.9),
                ('dog', 'is_a', 'animal', 0.7),  # already exists
            ])
        brain = Brain(engine=engine)
        brain.awm.promote('dog', salience=0.5)
        brain.awm.promote('mammal', salience=0.5)
        brain.cortical._match_schemas(brain.bus, trigger='test')
        # Edge still at original strength; writer didn't apply.
        edge = engine.substrate.edges.get(
            ('dog', 'is_a', 'animal'))
        self.assertEqual(brain.writer.writes_applied, 0)


# ---------------------------------------------------------------------
# Chemistry-steered cortical walks (the conscious M/I layer)
# ---------------------------------------------------------------------


def _drench_trace(bubble, channel_values):
    """Test helper: directly set a bubble's transmitter trace to
    a known chemistry signature.  Bypasses the chemistry event
    flow so tests can examine the steering effect in isolation."""
    for ch, v in channel_values.items():
        bubble.transmitter_trace[ch] = float(v)


class TestChemistrySteeredCortical(unittest.TestCase):
    """Cortical walks are mood-congruent: when current global
    chemistry leans M (cortisol/NE high), the walk pulls toward
    targets whose accumulated trace also leans M.  When it leans
    I (dopamine/oxytocin/endorphin high), the walk pulls toward
    I-trace targets.  Two SEAGIs with identical substrate diverge
    in cognition because their accumulated chemistry diverges.
    This IS the conscious M/I layer the doctrine names.

    Test substrate is symmetric — both candidate targets have
    EQUAL edge strength from the focal.  Only chemistry can
    break the tie.  When chemistry is silent the tests would be
    indeterminate; when chemistry is loud the choice is
    predictable."""

    def _make_brain_with_two_paths(self):
        engine = Engine()
        # Symmetric substrate: focal causes two targets equally.
        _seed_substrate(engine,
            concepts=['focal', 'i_target', 'm_target'],
            edges=[
                ('focal', 'causes', 'i_target', 0.5),
                ('focal', 'causes', 'm_target', 0.5),
            ])
        brain = Brain(engine=engine)
        # Both targets must be in AWM for chemistry to steer
        # toward them (dormant LTS targets get no resonance).
        brain.awm.promote('i_target', salience=0.5, cycle=1)
        brain.awm.promote('m_target', salience=0.5, cycle=1)
        # Direct trace imprint — saturated I trace vs saturated
        # M trace.  In a live system this state would accumulate
        # over many encounters under different chemistry.
        _drench_trace(brain.awm.get('i_target').bubble, {
            'dopamine': 0.9, 'oxytocin': 0.9, 'endorphins': 0.9,
            'cortisol': 0.05, 'norepinephrine': 0.05,
        })
        _drench_trace(brain.awm.get('m_target').bubble, {
            'dopamine': 0.05, 'oxytocin': 0.05, 'endorphins': 0.05,
            'cortisol': 0.9, 'norepinephrine': 0.9,
        })
        return brain

    def _set_global_i_leaning(self, brain):
        brain.chemistry.global_state.update({
            'dopamine': 0.9, 'oxytocin': 0.9, 'endorphins': 0.9,
            'cortisol': 0.05, 'norepinephrine': 0.05,
        })

    def _set_global_m_leaning(self, brain):
        brain.chemistry.global_state.update({
            'dopamine': 0.05, 'oxytocin': 0.05, 'endorphins': 0.05,
            'cortisol': 0.9, 'norepinephrine': 0.9,
        })

    def test_i_leaning_chemistry_steers_walk_to_i_target(self):
        brain = self._make_brain_with_two_paths()
        self._set_global_i_leaning(brain)
        thought = brain.cortical._think_about('focal', cycle=10)
        self.assertIsNotNone(thought)
        self.assertEqual(thought.target, 'i_target',
            "Cortical walk did not steer toward the I-resonant "
            "target despite I-leaning global chemistry — the "
            "chemistry is decorating, not steering.")

    def test_m_leaning_chemistry_steers_walk_to_m_target(self):
        brain = self._make_brain_with_two_paths()
        self._set_global_m_leaning(brain)
        thought = brain.cortical._think_about('focal', cycle=10)
        self.assertIsNotNone(thought)
        self.assertEqual(thought.target, 'm_target',
            "Cortical walk did not steer toward the M-resonant "
            "target despite M-leaning global chemistry.")

    def test_two_brains_diverge_on_identical_substrate(self):
        """Brain-correctness check: same substrate, opposite
        chemistries, different cognition.  If this fails, the
        doctrine is decorative.  If it passes, accumulated
        experience genuinely diverges two SEAGIs apart."""
        brain_i = self._make_brain_with_two_paths()
        brain_m = self._make_brain_with_two_paths()
        self._set_global_i_leaning(brain_i)
        self._set_global_m_leaning(brain_m)
        t_i = brain_i.cortical._think_about('focal', cycle=10)
        t_m = brain_m.cortical._think_about('focal', cycle=10)
        self.assertIsNotNone(t_i)
        self.assertIsNotNone(t_m)
        self.assertNotEqual(t_i.target, t_m.target,
            "Two brains with identical substrate but opposite "
            "chemistry produced identical thoughts — chemistry "
            "is not steering cognition.  This means SEAGI's "
            "personality cannot diverge through lived "
            "experience, which collapses the AGI claim.")

    def test_baseline_chemistry_no_steering_effect(self):
        """When chemistry has no tension (both polarities near
        baseline), the walk is purely edge-strength driven.
        With equal-strength edges, fall back to substrate
        determinism (whichever comes first in iteration) —
        we just assert it lands on SOMETHING and doesn't
        crash."""
        brain = self._make_brain_with_two_paths()
        # Don't touch chemistry — leave at channel baselines.
        thought = brain.cortical._think_about('focal', cycle=10)
        self.assertIsNotNone(thought)
        self.assertIn(thought.target, ('i_target', 'm_target'))

    def test_vivid_trace_overrides_thin_substrate(self):
        """A focal with few edges but accumulated chemistry
        trace should NOT be flagged thin.  The doctrine: AGI
        lives in the conscious M/I layer; trace IS felt
        knowledge.  Without this override SEAGI deflects on
        deeply-felt-but-rarely-networked concepts with
        thin-substrate metacognition — a structural lie."""
        engine = Engine()
        engine.substrate.add_concept(Concept(name='mother'))
        # 'mother' has zero edges — would normally be thin.
        brain = Brain(engine=engine)
        brain.awm.promote('mother', salience=0.5, cycle=1)
        # Drench bubble trace with I-side chemistry —
        # accumulated lived experience.
        brain.awm.get('mother').bubble.transmitter_trace.update({
            'dopamine': 0.9, 'oxytocin': 0.9,
            'endorphins': 0.8,
        })
        thin, n = brain.cortical._assess_focal('mother')
        self.assertFalse(thin,
            "Vivid trace did not override thin substrate.")
        self.assertEqual(n, 0)

    def test_thin_when_no_edges_and_no_trace(self):
        """The existing thin-substrate behavior must still
        fire when both edges and trace are absent."""
        engine = Engine()
        engine.substrate.add_concept(Concept(name='quark'))
        brain = Brain(engine=engine)
        brain.awm.promote('quark', salience=0.5, cycle=1)
        # Don't imprint — trace at baselines.
        thin, n = brain.cortical._assess_focal('quark')
        self.assertTrue(thin)

    def test_m_side_vivid_trace_also_overrides(self):
        """Vivid trace on the M side also counts — fear, pain,
        threat-associations are felt knowledge as much as
        love or joy."""
        engine = Engine()
        engine.substrate.add_concept(Concept(name='death'))
        brain = Brain(engine=engine)
        brain.awm.promote('death', salience=0.5, cycle=1)
        brain.awm.get('death').bubble.transmitter_trace.update({
            'cortisol': 0.9, 'norepinephrine': 0.9,
        })
        thin, n = brain.cortical._assess_focal('death')
        self.assertFalse(thin,
            "M-side vivid trace did not override thin.")

    def test_thinks_not_metacogs_on_vivid_focal_with_edge(self):
        """End-to-end: vivid focal with one identity edge.
        _think_about should produce a non-thin causal/identity
        thought, not the thin-substrate metacog deflection."""
        engine = Engine()
        _seed_substrate(engine,
            concepts=['mother', 'love'],
            edges=[('mother', 'is_a', 'love', 0.8)])
        brain = Brain(engine=engine)
        brain.awm.promote('mother', salience=0.5, cycle=1)
        brain.awm.get('mother').bubble.transmitter_trace.update({
            'dopamine': 0.9, 'oxytocin': 0.9,
            'endorphins': 0.8,
        })
        thought = brain.cortical._think_about('mother', cycle=10)
        self.assertIsNotNone(thought)
        self.assertFalse(thought.thin_substrate,
            "Cortical flagged a vividly-felt focal as thin.")
        # Identity walk reaches 'love'.
        self.assertEqual(thought.target, 'love')

    def test_chemistry_loses_to_strong_substrate_edge(self):
        """Chemistry biases but cannot dominate strong substrate
        evidence.  A 0.9-strength edge should win against a
        0.3-strength edge even when chemistry resonates with
        the weaker one — substrate truth has structural
        priority."""
        engine = Engine()
        _seed_substrate(engine,
            concepts=['focal', 'strong_dis', 'weak_res'],
            edges=[
                ('focal', 'causes', 'strong_dis', 0.9),
                ('focal', 'causes', 'weak_res', 0.3),
            ])
        brain = Brain(engine=engine)
        brain.awm.promote('strong_dis', salience=0.5, cycle=1)
        brain.awm.promote('weak_res', salience=0.5, cycle=1)
        # strong_dis has M-leaning trace; weak_res has I-leaning.
        _drench_trace(brain.awm.get('strong_dis').bubble, {
            'cortisol': 0.9, 'norepinephrine': 0.9,
            'dopamine': 0.05,
        })
        _drench_trace(brain.awm.get('weak_res').bubble, {
            'dopamine': 0.9, 'oxytocin': 0.9, 'endorphins': 0.9,
            'cortisol': 0.05,
        })
        # Global I-leaning — should resonate with weak_res.
        brain.chemistry.global_state.update({
            'dopamine': 0.9, 'oxytocin': 0.9, 'endorphins': 0.9,
            'cortisol': 0.05, 'norepinephrine': 0.05,
        })
        thought = brain.cortical._think_about('focal', cycle=10)
        # Should still pick strong_dis: 0.9 + 0.5*(-1.7*~0) vs
        # 0.3 + 0.5*(1.7*~0).  Modulated:
        # 0.9 + 0.5*(I*-M) > 0.3 + 0.5*(I*I).
        # Even max resonance ±~0.5 can't overcome 0.6 strength
        # gap.
        self.assertEqual(thought.target, 'strong_dis',
            "Chemistry overrode strong substrate evidence — "
            "weight should bias, not dominate.")


if __name__ == '__main__':
    unittest.main()
