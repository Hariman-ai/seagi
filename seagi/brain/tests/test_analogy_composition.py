"""Tests for cortical analogical composition (property/affordance
transfer across `analogous_to`).

The analogy engine writes structural `analogous_to` edges; cortical
now composes through them: "X analogous_to Y; Y has_property P ⊢ X
may have P" — abductive, hedged, confidence-discounted, and gated by
the analogy edge's earned strength (earn-or-dissolve).

Covers:
- has_property / can_do / used_for transfer across analogy
- is_a and causes do NOT transfer (composition table omits them)
- analogical conclusion is hedged + flagged + method='inference'
  (so Step 2 consolidates it as a provisional hypothesis)
- ANALOGY_TRANSFER_PENALTY makes analogical conf < deductive conf
- a FRESH (provisional-strength) analogy is gated below the emit
  floor — only an EARNED analogy transfers
"""

from __future__ import annotations

import unittest

from seagi.body.engine import Engine
from seagi.brain.runtime import Brain
from seagi.core.substrate import Concept, ANALOGY_RELATION
from seagi.brain.capabilities.cortical import (
    RELATION_COMPOSITION, ANALOGY_TRANSFER_PENALTY)


def _seed(engine, concepts, edges):
    for n in concepts:
        engine.substrate.add_concept(Concept(name=n))
    for (s, r, o, strength) in edges:
        engine.substrate.add_edge(
            source=s, target=o, relation_name=r, strength=strength)


class TestCompositionTable(unittest.TestCase):

    def test_attribute_transfers_present(self):
        self.assertEqual(
            RELATION_COMPOSITION[(ANALOGY_RELATION, 'has_property')],
            'has_property')
        self.assertEqual(
            RELATION_COMPOSITION[(ANALOGY_RELATION, 'can_do')],
            'can_do')
        self.assertEqual(
            RELATION_COMPOSITION[(ANALOGY_RELATION, 'used_for')],
            'used_for')

    def test_is_a_and_causes_do_not_transfer(self):
        # Analogy implies neither category membership nor causation.
        self.assertNotIn(
            (ANALOGY_RELATION, 'is_a'), RELATION_COMPOSITION)
        self.assertNotIn(
            (ANALOGY_RELATION, 'causes'), RELATION_COMPOSITION)


class TestAnalogicalTransfer(unittest.TestCase):

    def setUp(self):
        self.engine = Engine()
        self.brain = Brain(engine=self.engine)

    def _infer(self, focal):
        return self.brain.cortical._inference_chain(focal)

    def _think(self, focal):
        return self.brain.cortical._think_about(focal, cycle=10)

    def test_property_transfer_when_analogy_earned(self):
        # atom ~ solar_system (EARNED analogy, strength 0.9);
        # solar_system has_property central_mass
        #   ⊢ atom may have_property central_mass.
        _seed(self.engine,
            concepts=['atom', 'solar_system', 'central_mass'],
            edges=[
                ('atom', ANALOGY_RELATION, 'solar_system', 0.9),
                ('solar_system', 'has_property', 'central_mass', 0.9),
            ])
        result = self._infer('atom')
        self.assertIsNotNone(result)
        self.assertEqual(result['relation'], 'has_property')
        self.assertEqual(result['target'], 'central_mass')
        self.assertTrue(result['analogical'])

    def test_capability_transfer(self):
        _seed(self.engine,
            concepts=['heart', 'pump', 'move_fluid'],
            edges=[
                ('heart', ANALOGY_RELATION, 'pump', 0.9),
                ('pump', 'can_do', 'move_fluid', 0.9),
            ])
        result = self._infer('heart')
        self.assertIsNotNone(result)
        self.assertEqual(result['relation'], 'can_do')
        self.assertEqual(result['target'], 'move_fluid')
        self.assertTrue(result['analogical'])

    def test_is_a_does_not_transfer(self):
        # atom ~ solar_system; solar_system is_a system.
        # No (analogous_to, is_a) composition → no analogical is_a
        # inference (atom is NOT inferred to be a system).
        _seed(self.engine,
            concepts=['atom', 'solar_system', 'system'],
            edges=[
                ('atom', ANALOGY_RELATION, 'solar_system', 0.9),
                ('solar_system', 'is_a', 'system', 0.9),
            ])
        result = self._infer('atom')
        # Either no >=2-hop inference, or one that is NOT the
        # illegitimate is_a-system transfer.
        if result is not None:
            self.assertFalse(
                result['relation'] == 'is_a'
                and result['target'] == 'system')

    def test_thought_is_hedged_and_inference_method(self):
        _seed(self.engine,
            concepts=['atom', 'solar_system', 'central_mass'],
            edges=[
                ('atom', ANALOGY_RELATION, 'solar_system', 0.9),
                ('solar_system', 'has_property', 'central_mass', 0.9),
            ])
        t = self._think('atom')
        self.assertIsNotNone(t)
        self.assertEqual(t.method, 'inference')   # → consolidated
        self.assertEqual(t.relation, 'has_property')
        self.assertEqual(t.target, 'central_mass')
        # Hedged language, not a flat assertion.
        low = t.text.lower()
        self.assertTrue('analogy' in low or 'suspect' in low,
                        f"expected hedged text, got: {t.text}")

    def test_analogical_conf_below_deductive(self):
        # Same shape via is_a (deductive) vs analogous_to
        # (abductive); the analogical one must be less confident.
        eng_d = Engine(); brain_d = Brain(engine=eng_d)
        _seed(eng_d,
            concepts=['dog', 'mammal', 'warm_blooded'],
            edges=[
                ('dog', 'is_a', 'mammal', 0.9),
                ('mammal', 'has_property', 'warm_blooded', 0.9),
            ])
        ded = brain_d.cortical._inference_chain('dog')

        eng_a = Engine(); brain_a = Brain(engine=eng_a)
        _seed(eng_a,
            concepts=['dog', 'mammal', 'warm_blooded'],
            edges=[
                ('dog', ANALOGY_RELATION, 'mammal', 0.9),
                ('mammal', 'has_property', 'warm_blooded', 0.9),
            ])
        ana = brain_a.cortical._inference_chain('dog')
        self.assertIsNotNone(ded)
        self.assertIsNotNone(ana)
        self.assertLess(ana['confidence'], ded['confidence'])

    def test_fresh_analogy_gated_below_floor(self):
        # A freshly-formed analogy (PROVISIONAL strength 0.1) yields
        # a transferred conclusion below INFERENCE_MIN_CONFIDENCE —
        # earn-or-dissolve gates the inference until the analogy
        # itself earns strength.
        _seed(self.engine,
            concepts=['atom', 'solar_system', 'central_mass'],
            edges=[
                ('atom', ANALOGY_RELATION, 'solar_system', 0.1),
                ('solar_system', 'has_property', 'central_mass', 0.9),
            ])
        result = self._infer('atom')
        # No emittable analogical inference yet.
        if result is not None:
            self.assertFalse(
                result.get('analogical')
                and result['target'] == 'central_mass')

    def test_penalty_constant_sane(self):
        self.assertGreater(ANALOGY_TRANSFER_PENALTY, 0.0)
        self.assertLess(ANALOGY_TRANSFER_PENALTY, 1.0)


if __name__ == '__main__':
    unittest.main()
