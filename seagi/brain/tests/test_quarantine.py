"""Tests for the substrate quarantine tier (2026-05-30).

Proves: noise is the OUTPUT of cognition, not its precondition.
Edges that none of the four cognitive bid-paths engaged across
L_QUARANTINE cycles get moved to the quiescent quarantine pool.
Auto-restore on any bid-path touch.  Doctrine: chemistry-never-
fully-dissolves (quarantine ≠ delete).

Also covers the coherence-by-composition fix that ships in the
same batch — coherence now requires the corroborating A→X→B chain's
relations to compose to the candidate edge's own relation.
"""

from __future__ import annotations

import unittest

from seagi.core.substrate import (
    Substrate, Concept, Edge, EDGE_PRUNE_FLOOR,
)


class TestCompositionGatedCoherence(unittest.TestCase):
    """Fix 1: coherence requires composition, not just topology."""

    def test_composable_triangle_corroborates(self):
        # is_a ∘ is_a → is_a — a real composable chain.
        sub = Substrate()
        for n in ('a', 'b', 'c'):
            sub.add_concept(Concept(name=n))
        sub.add_edge('a', 'b', 'is_a', strength=0.3, cycle=0)
        sub.add_edge('b', 'c', 'is_a', strength=0.3, cycle=0)
        sub.add_edge('a', 'c', 'is_a', strength=0.3, cycle=0)
        before = sub.edges[('a', 'is_a', 'c')].strength
        reinforced, newly = sub.reinforce_coherent_edges(cycle=1)
        after = sub.edges[('a', 'is_a', 'c')].strength
        self.assertGreater(after, before)
        self.assertGreaterEqual(reinforced, 1)
        self.assertGreaterEqual(newly, 1)

    def test_noncomposable_triangle_does_not_corroborate(self):
        # 'co_occurs' is NOT in RELATION_COMPOSITION — no chain
        # composes to it.  Triangle exists topologically; under the
        # new strict test, the candidate edge earns nothing.  This
        # is exactly the noise the fix is designed to evict.
        sub = Substrate()
        for n in ('a', 'b', 'c'):
            sub.add_concept(Concept(name=n))
        for (s, t) in [('a', 'b'), ('b', 'c'), ('a', 'c')]:
            sub.add_edge(s, t, 'co_occurs', strength=0.3, cycle=0)
        before = sub.edges[('a', 'co_occurs', 'c')].strength
        reinforced, newly = sub.reinforce_coherent_edges(cycle=1)
        after = sub.edges[('a', 'co_occurs', 'c')].strength
        self.assertEqual(after, before)   # no reinforcement
        self.assertEqual(reinforced, 0)
        self.assertEqual(newly, 0)

    def test_wrong_composition_does_not_corroborate(self):
        # (produces, causes) composes to 'leads_to', NOT to 'causes'.
        # So (a, causes, c) is NOT corroborated by (a produces b,
        # b causes c) — that chain composes to the wrong target
        # relation.
        sub = Substrate()
        for n in ('a', 'b', 'c'):
            sub.add_concept(Concept(name=n))
        sub.add_edge('a', 'b', 'produces', strength=0.3, cycle=0)
        sub.add_edge('b', 'c', 'causes', strength=0.3, cycle=0)
        sub.add_edge('a', 'c', 'causes', strength=0.3, cycle=0)
        before = sub.edges[('a', 'causes', 'c')].strength
        sub.reinforce_coherent_edges(cycle=1)
        after = sub.edges[('a', 'causes', 'c')].strength
        self.assertEqual(after, before)
        # But (a, leads_to, c) WOULD be corroborated.
        sub.add_edge('a', 'c', 'leads_to', strength=0.3, cycle=0)
        before_lt = sub.edges[('a', 'leads_to', 'c')].strength
        sub.reinforce_coherent_edges(cycle=2)
        after_lt = sub.edges[('a', 'leads_to', 'c')].strength
        self.assertGreater(after_lt, before_lt)


class TestEngagementStamping(unittest.TestCase):
    """Coherence corroboration stamps engagement on the candidate
    AND the two middle-hop edges (corroborators).  Engagement is
    the substrate-side signal that protects against quarantine."""

    def test_coherence_stamps_engagement_on_all_three_edges(self):
        sub = Substrate()
        for n in ('a', 'b', 'c'):
            sub.add_concept(Concept(name=n))
        sub.add_edge('a', 'b', 'is_a', strength=0.3, cycle=0)
        sub.add_edge('b', 'c', 'is_a', strength=0.3, cycle=0)
        sub.add_edge('a', 'c', 'is_a', strength=0.3, cycle=0)
        sub.reinforce_coherent_edges(cycle=100)
        for (s, r, t) in (('a', 'is_a', 'b'),
                          ('b', 'is_a', 'c'),
                          ('a', 'is_a', 'c')):
            self.assertEqual(sub.edges[(s, r, t)].last_engaged_cycle,
                             100)


class TestQuarantineSweep(unittest.TestCase):
    """quarantine_inert_edges moves floor-strength + stale-engaged
    edges to the quarantine pool.  Gated on migration-complete."""

    def _build_with_one_inert(self):
        sub = Substrate()
        for n in ('a', 'b'):
            sub.add_concept(Concept(name=n))
        edge = sub.add_edge('a', 'b', 'co_occurs',
                            strength=EDGE_PRUNE_FLOOR, cycle=0)
        # Force stale engagement.
        edge.last_engaged_cycle = 0
        # Edge-mortality split (2026-06-02): quarantine now PRESERVES
        # only edges that ONCE cohered (a faded past-self); never-
        # cohered stillborns are DELETED by reap_stillborn_edges, not
        # quarantined.  Mark this edge as having cohered so it is a
        # quarantine candidate — these tests exercise the preservation
        # arm; the reaping arm is covered in test_edge_mortality.
        edge.first_coherent_cycle = 1
        # Pretend migration already done so the sweep can run.
        sub._quarantine_migrated = True
        return sub

    def test_inert_edge_quarantined(self):
        sub = self._build_with_one_inert()
        n = sub.quarantine_inert_edges(
            cycle=sub.L_QUARANTINE_CYCLES + 10,
            max_per_pass=10)
        self.assertEqual(n, 1)
        self.assertNotIn(('a', 'co_occurs', 'b'), sub.edges)
        self.assertIn(('a', 'co_occurs', 'b'), sub.quarantine_edges)

    def test_engaged_edge_not_quarantined(self):
        sub = self._build_with_one_inert()
        # Stamp engagement at a fresh cycle.
        edge = sub.edges[('a', 'co_occurs', 'b')]
        edge.last_engaged_cycle = sub.L_QUARANTINE_CYCLES + 5
        n = sub.quarantine_inert_edges(
            cycle=sub.L_QUARANTINE_CYCLES + 10,
            max_per_pass=10)
        self.assertEqual(n, 0)
        self.assertIn(('a', 'co_occurs', 'b'), sub.edges)

    def test_above_floor_edge_not_quarantined(self):
        sub = self._build_with_one_inert()
        # Push strength above floor.
        edge = sub.edges[('a', 'co_occurs', 'b')]
        edge.strength = 0.5
        # Set its decay anchor to the current cycle so effective
        # strength stays high even after L_QUARANTINE_CYCLES of
        # elapsed lazy-decay.
        edge.last_reinforced_cycle = sub.L_QUARANTINE_CYCLES + 10
        n = sub.quarantine_inert_edges(
            cycle=sub.L_QUARANTINE_CYCLES + 10,
            max_per_pass=10)
        self.assertEqual(n, 0)

    def test_migration_gate_blocks_sweep(self):
        sub = self._build_with_one_inert()
        sub._quarantine_migrated = False
        n = sub.quarantine_inert_edges(
            cycle=sub.L_QUARANTINE_CYCLES + 10,
            max_per_pass=10)
        self.assertEqual(n, 0)
        self.assertIn(('a', 'co_occurs', 'b'), sub.edges)


class TestQuarantineRestore(unittest.TestCase):
    """Auto-restore on any bid-path touch."""

    def _build_with_quarantined(self):
        sub = Substrate()
        for n in ('a', 'b'):
            sub.add_concept(Concept(name=n))
        sub.add_edge('a', 'b', 'is_a',
                     strength=EDGE_PRUNE_FLOOR, cycle=0)
        sub.edges[('a', 'is_a', 'b')].last_engaged_cycle = 0
        # Once-cohered past-self → quarantine-eligible (never-cohered
        # edges are reaped, not quarantined; see edge-mortality split).
        sub.edges[('a', 'is_a', 'b')].first_coherent_cycle = 1
        sub._quarantine_migrated = True
        n = sub.quarantine_inert_edges(
            cycle=sub.L_QUARANTINE_CYCLES + 10,
            max_per_pass=10)
        assert n == 1
        return sub

    def test_re_attestation_restores(self):
        sub = self._build_with_quarantined()
        cycle = sub.L_QUARANTINE_CYCLES + 100
        # Forager-style re-attestation via add_edge.
        restored = sub.add_edge('a', 'b', 'is_a',
                                strength=0.3, cycle=cycle)
        self.assertIn(('a', 'is_a', 'b'), sub.edges)
        self.assertNotIn(('a', 'is_a', 'b'), sub.quarantine_edges)
        self.assertEqual(restored.last_engaged_cycle, cycle)

    def test_awm_promotion_style_restore(self):
        # restore_concept_quarantine is the AWM bid-path; it
        # restores ALL quarantined edges touching the concept.
        sub = self._build_with_quarantined()
        cycle = sub.L_QUARANTINE_CYCLES + 200
        restored_count = sub.restore_concept_quarantine('a',
                                                         cycle=cycle)
        self.assertEqual(restored_count, 1)
        self.assertIn(('a', 'is_a', 'b'), sub.edges)
        self.assertEqual(sub.edges[('a', 'is_a', 'b')]
                         .last_engaged_cycle, cycle)

    def test_target_endpoint_restore(self):
        # Quarantined edge restorable via its target endpoint too.
        sub = self._build_with_quarantined()
        cycle = 99999
        n = sub.restore_concept_quarantine('b', cycle=cycle)
        self.assertEqual(n, 1)
        self.assertIn(('a', 'is_a', 'b'), sub.edges)


class TestMigrationPass(unittest.TestCase):
    """One-shot migration of legacy first_coherent_cycle stamps."""

    def test_legacy_stamp_cleared_when_not_composable(self):
        sub = Substrate()
        for n in ('a', 'b', 'c'):
            sub.add_concept(Concept(name=n))
        # Triangle with non-composable relation — yet the edge has
        # a legacy stamp (simulating pre-fix topological coherence).
        for (s, t) in [('a', 'b'), ('b', 'c'), ('a', 'c')]:
            e = sub.add_edge(s, t, 'co_occurs',
                             strength=0.3, cycle=0)
            e.first_coherent_cycle = 1   # legacy false stamp
        cleared = sub.migrate_legacy_coherence_stamps(
            cycle=1000, max_per_pass=1000)
        self.assertGreaterEqual(cleared, 3)
        for (s, t) in [('a', 'b'), ('b', 'c'), ('a', 'c')]:
            self.assertEqual(
                sub.edges[(s, 'co_occurs', t)].first_coherent_cycle, 0)
        self.assertTrue(sub._quarantine_migrated)

    def test_legacy_stamp_preserved_when_composable(self):
        sub = Substrate()
        for n in ('a', 'b', 'c'):
            sub.add_concept(Concept(name=n))
        sub.add_edge('a', 'b', 'is_a', strength=0.3, cycle=0)
        sub.add_edge('b', 'c', 'is_a', strength=0.3, cycle=0)
        e = sub.add_edge('a', 'c', 'is_a', strength=0.3, cycle=0)
        e.first_coherent_cycle = 42       # legitimate stamp
        cleared = sub.migrate_legacy_coherence_stamps(
            cycle=1000, max_per_pass=1000)
        # The (a,is_a,c) edge passes the new composable test — stamp
        # preserved.
        self.assertEqual(
            sub.edges[('a', 'is_a', 'c')].first_coherent_cycle, 42)


class TestPersistence(unittest.TestCase):
    """Quarantine state round-trips through to_dict / from_dict."""

    def test_quarantine_round_trip(self):
        sub = Substrate()
        for n in ('a', 'b'):
            sub.add_concept(Concept(name=n))
        sub.add_edge('a', 'b', 'is_a',
                     strength=EDGE_PRUNE_FLOOR, cycle=0)
        sub.edges[('a', 'is_a', 'b')].last_engaged_cycle = 0
        # Once-cohered → quarantine-eligible under the edge-mortality
        # split (never-cohered edges are reaped, not quarantined).
        sub.edges[('a', 'is_a', 'b')].first_coherent_cycle = 1
        sub._quarantine_migrated = True
        sub.quarantine_inert_edges(
            cycle=sub.L_QUARANTINE_CYCLES + 10, max_per_pass=10)
        d = sub.to_dict()
        sub2 = Substrate.from_dict(d)
        self.assertIn(('a', 'is_a', 'b'), sub2.quarantine_edges)
        self.assertNotIn(('a', 'is_a', 'b'), sub2.edges)
        self.assertIn('a', sub2._quarantine_index)
        self.assertIn('b', sub2._quarantine_index)
        self.assertTrue(sub2._quarantine_migrated)


if __name__ == '__main__':
    unittest.main()


class TestFadeNotDeleteStillborn(unittest.TestCase):
    """World-value model (2026-06-08): the reaper is retired.  A
    never-cohered (stillborn) relation between real words is still an
    imprint of the world — it FADES to the quiescent quarantine pool,
    is NEVER deleted, and stays cue-able back via any bid-path.  This
    locks in 'nothing in the substrate is deleted' against a future
    re-introduction of deletion."""

    def _stillborn_at_floor(self):
        sub = Substrate()
        for n in ('alphaword', 'betaword'):
            sub.add_concept(Concept(name=n))
        sub.add_edge('alphaword', 'betaword', 'relatesto',
                     strength=EDGE_PRUNE_FLOOR, cycle=1)
        key = next(iter(sub.edges))            # (source, relation, target)
        e = sub.edges[key]
        e.first_coherent_cycle = 0             # NEVER cohered = stillborn
        e.last_reinforced_cycle = 1
        e.last_engaged_cycle = 1
        sub._quarantine_migrated = True
        return sub, key, e

    def test_stillborn_fades_to_quarantine_not_deleted(self):
        sub, key, e = self._stillborn_at_floor()
        cyc = 1 + sub.L_QUARANTINE_CYCLES + 10
        n = sub.quarantine_inert_edges(cyc, max_per_pass=50)
        self.assertEqual(n, 1)
        self.assertNotIn(key, sub.edges)              # left active
        self.assertIn(key, sub.quarantine_edges)      # faded, NOT deleted
        self.assertIs(sub.quarantine_edges[key], e)   # same object preserved

    def test_quarantined_stillborn_is_restorable(self):
        sub, key, e = self._stillborn_at_floor()
        cyc = 1 + sub.L_QUARANTINE_CYCLES + 10
        sub.quarantine_inert_edges(cyc, max_per_pass=50)
        restored = sub.restore_concept_quarantine(key[0], cycle=cyc + 1)
        self.assertEqual(restored, 1)
        self.assertIn(key, sub.edges)                 # cue-able back to active


if __name__ == '__main__':
    unittest.main()
