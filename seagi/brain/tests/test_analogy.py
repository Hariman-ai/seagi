"""Tests for the analogy engine (Substrate.form_analogies).

Structural analogy: two concepts sharing an outgoing relation-
skeleton (>= ANALOGY_MIN_SKELETON relation types) with DISJOINT
targets get a provisional `analogous_to` edge, earn-or-dissolved
by Phase S.

Covers:
- disjoint-target same-skeleton pair → analogous_to edge
- shared-target pair (overlapping content) → NO analogy
- skeleton smaller than min → skipped
- analogy edge enters at provisional strength
- re-attest reinforces an existing analogy
- abstraction concepts / analogous_to edges excluded from skeletons
- sleep consolidation forms analogies end-to-end
- earn-or-dissolve: idle analogy settles toward floor
"""

from __future__ import annotations

import unittest

from seagi.core.substrate import (
    Substrate, Concept,
    ANALOGY_RELATION, ANALOGY_MIN_SKELETON,
    PROVISIONAL_EDGE_STRENGTH, ABSTRACTION_NAME_PREFIX,
)


def _seed(sub, edges):
    """edges: list of (source, relation, target)."""
    for (s, r, t) in edges:
        for nm in (s, t):
            if nm not in sub.concepts:
                sub.add_concept(Concept(name=nm))
        sub.add_edge(s, t, r, strength=0.5, cycle=0)


class TestFormAnalogies(unittest.TestCase):

    def test_disjoint_same_skeleton_forms_analogy(self):
        sub = Substrate(seed_relations=False)
        # atom and solar_system share skeleton {has_part, attracts}
        # with disjoint targets.
        _seed(sub, [
            ('atom', 'has_part', 'electron'),
            ('atom', 'attracts', 'proton'),
            ('solar_system', 'has_part', 'planet'),
            ('solar_system', 'attracts', 'comet'),
        ])
        n = sub.form_analogies(cycle=0)
        self.assertEqual(n, 1)
        # Canonical sorted order: atom < solar_system.
        self.assertIn(('atom', ANALOGY_RELATION, 'solar_system'),
                      sub.edges)

    def test_analogy_enters_provisional(self):
        sub = Substrate(seed_relations=False)
        _seed(sub, [
            ('a', 'r1', 'x'), ('a', 'r2', 'y'),
            ('b', 'r1', 'p'), ('b', 'r2', 'q'),
        ])
        sub.form_analogies(cycle=0)
        edge = sub.edges[('a', ANALOGY_RELATION, 'b')]
        self.assertAlmostEqual(
            edge.strength, PROVISIONAL_EDGE_STRENGTH, places=5)

    def test_shared_target_no_analogy(self):
        sub = Substrate(seed_relations=False)
        # Same skeleton {r1, r2} but they SHARE target 'x' on r1 →
        # overlapping content, not a cross-domain analogy.
        _seed(sub, [
            ('a', 'r1', 'x'), ('a', 'r2', 'y'),
            ('b', 'r1', 'x'), ('b', 'r2', 'q'),
        ])
        n = sub.form_analogies(cycle=0)
        self.assertEqual(n, 0)
        self.assertNotIn(('a', ANALOGY_RELATION, 'b'), sub.edges)

    def test_skeleton_below_min_skipped(self):
        sub = Substrate(seed_relations=False)
        # Only ONE shared relation type (< ANALOGY_MIN_SKELETON=2).
        self.assertEqual(ANALOGY_MIN_SKELETON, 2)
        _seed(sub, [
            ('a', 'r1', 'x'),
            ('b', 'r1', 'p'),
        ])
        n = sub.form_analogies(cycle=0)
        self.assertEqual(n, 0)

    def test_different_skeletons_no_analogy(self):
        sub = Substrate(seed_relations=False)
        # a has {r1,r2}; b has {r1,r3} — skeletons differ → not
        # exact-skeleton analogs.
        _seed(sub, [
            ('a', 'r1', 'x'), ('a', 'r2', 'y'),
            ('b', 'r1', 'p'), ('b', 'r3', 'q'),
        ])
        n = sub.form_analogies(cycle=0)
        self.assertEqual(n, 0)

    def test_reattest_credits_only_on_engagement(self):
        # Self-cleaning F2 (2026-07-24): a static repeat pass must NOT
        # stamp strength onto the engine's own analogy edge; credit
        # fires only after a fresh cognition-external engagement.
        sub = Substrate(seed_relations=False)
        _seed(sub, [
            ('a', 'r1', 'x'), ('a', 'r2', 'y'),
            ('b', 'r1', 'p'), ('b', 'r2', 'q'),
        ])
        sub.form_analogies(cycle=0)
        edge = sub.edges[('a', ANALOGY_RELATION, 'b')]
        before = edge.strength
        n2 = sub.form_analogies(cycle=10)
        self.assertEqual(n2, 0)
        self.assertEqual(
            sub.edges[('a', ANALOGY_RELATION, 'b')].strength, before)
        # An external touch engages it -> the next pass credits.
        edge.last_engaged_cycle = 20
        sub.form_analogies(cycle=30)
        self.assertGreater(
            sub.edges[('a', ANALOGY_RELATION, 'b')].strength, before)

    def test_abstraction_concepts_excluded(self):
        sub = Substrate(seed_relations=False)
        abs = f'{ABSTRACTION_NAME_PREFIX}causes_heat'
        _seed(sub, [
            (abs, 'r1', 'x'), (abs, 'r2', 'y'),
            ('b', 'r1', 'p'), ('b', 'r2', 'q'),
        ])
        n = sub.form_analogies(cycle=0)
        # The abstraction concept is excluded from skeleton building
        # → no analogy involving it.
        self.assertEqual(n, 0)

    def test_analogy_edges_not_reused_as_skeleton(self):
        sub = Substrate(seed_relations=False)
        _seed(sub, [
            ('a', 'r1', 'x'), ('a', 'r2', 'y'),
            ('b', 'r1', 'p'), ('b', 'r2', 'q'),
        ])
        sub.form_analogies(cycle=0)
        # Now a and b each have an extra analogous_to edge.  A second
        # pass must NOT treat analogous_to as part of the skeleton
        # (else it'd distort structure).  Re-run is stable: still
        # just the one analogy, reinforced.
        n2 = sub.form_analogies(cycle=10)
        self.assertEqual(n2, 0)
        analogy_edges = [k for k in sub.edges
                         if k[1] == ANALOGY_RELATION]
        self.assertEqual(len(analogy_edges), 1)


class TestSleepIntegration(unittest.TestCase):

    def test_sleep_consolidation_forms_analogies(self):
        from seagi.body.engine import Engine
        from seagi.brain.runtime import Brain, SLEEP_CONSOLIDATION_INTERVAL
        engine = Engine()
        sub = engine.substrate
        _seed(sub, [
            ('atom', 'has_part', 'electron'),
            ('atom', 'attracts', 'proton'),
            ('solar_system', 'has_part', 'planet'),
            ('solar_system', 'attracts', 'comet'),
        ])
        brain = Brain(engine=engine)
        brain.sleep_regulator.pressure = 0.99
        brain.sleep_regulator._transition_to_sleep(brain.bus, cycle=0)
        for _ in range(SLEEP_CONSOLIDATION_INTERVAL + 5):
            brain.tick()
        self.assertGreaterEqual(brain.analogies_formed, 1)
        self.assertIn(('atom', ANALOGY_RELATION, 'solar_system'),
                      sub.edges)

    def test_idle_analogy_settles_toward_floor(self):
        # An analogy never re-attested decays under settle_weak_edges
        # toward the floor (earn-or-dissolve), not staying strong.
        from seagi.core.substrate import EDGE_PRUNE_FLOOR
        sub = Substrate(seed_relations=False)
        _seed(sub, [
            ('a', 'r1', 'x'), ('a', 'r2', 'y'),
            ('b', 'r1', 'p'), ('b', 'r2', 'q'),
        ])
        sub.form_analogies(cycle=0)
        key = ('a', ANALOGY_RELATION, 'b')
        self.assertIn(key, sub.edges)
        # Many maintenance passes WITHOUT re-forming analogies (the
        # structural pair is removed so it can't re-attest): the
        # provisional analogy decays and settles to floor.
        del sub.edges[('b', 'r1', 'p')]  # break b's skeleton
        for cyc in range(0, 40000, 2000):
            sub.settle_weak_edges(cyc)
        self.assertIn(key, sub.edges)  # settled, not deleted
        self.assertAlmostEqual(
            sub.edges[key].strength, EDGE_PRUNE_FLOOR, places=5)


if __name__ == '__main__':
    unittest.main()
