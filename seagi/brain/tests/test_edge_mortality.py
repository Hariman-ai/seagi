"""Edge-mortality + scoped consolidation — the foundation (2026-06-02).

Two organs, one build:

  (1) EDGE-MORTALITY (reap_stillborn_edges).  Never-cohered
      (first_coherent_cycle == 0), at-floor, stale edges are DELETED.
      The mortality vision applied to the substrate itself: an edge
      earns existence by cohering; one the parser/reverie threw up that
      the graph never corroborated across a full engagement window, and
      that has decayed to the floor, never earned existence — it dies
      rather than lingering.  This is what makes the substrate self-
      clean without a janitor.  Honors chemistry-never-fully-dissolves
      at the correct boundary: STILLBORN != PAST-SELF.  Once-cohered
      edges (fcc>0) are PRESERVED via quarantine, not reaped.

  (2) SCOPED CONSOLIDATION.  reinforce_coherent_edges takes an optional
      `candidates` set (from consume_dirty) and re-evaluates coherence
      ONLY in the changed neighborhood — O(new), not the
      O(N×branching) whole-substrate rescan that hard-stalled the
      daemon at ~697K edges.  Coherence is structural, so unchanged
      regions keep their persisted stamps.

See project_seagi_edge_mortality.
"""

from __future__ import annotations

import unittest

from seagi.core.substrate import (
    Substrate, Concept, EDGE_PRUNE_FLOOR,
)


class TestEdgeMortalityReaping(unittest.TestCase):

    def _migrated(self):
        sub = Substrate()
        sub._quarantine_migrated = True   # reaping is gated on migration
        return sub

    def test_stillborn_reaped(self):
        # Never cohered (fcc==0), at floor, stale -> DELETED outright.
        sub = self._migrated()
        e = sub.add_edge('a', 'b', 'co_occurs',
                         strength=EDGE_PRUNE_FLOOR, cycle=0)
        e.last_engaged_cycle = 0
        self.assertEqual(e.first_coherent_cycle, 0)
        n = sub.reap_stillborn_edges(
            cycle=sub.L_QUARANTINE_CYCLES + 10, max_per_pass=10)
        self.assertEqual(n, 1)
        # Gone — and NOT moved to quarantine (true deletion).
        self.assertNotIn(('a', 'co_occurs', 'b'), sub.edges)
        self.assertNotIn(('a', 'co_occurs', 'b'), sub.quarantine_edges)

    def test_cohered_edge_survives_reaping_then_quarantines(self):
        # fcc>0 at-floor stale -> reaper skips it; quarantine preserves.
        sub = self._migrated()
        e = sub.add_edge('a', 'b', 'is_a',
                         strength=EDGE_PRUNE_FLOOR, cycle=0)
        e.last_engaged_cycle = 0
        e.first_coherent_cycle = 5        # a past-self that once cohered
        cyc = sub.L_QUARANTINE_CYCLES + 10
        reaped = sub.reap_stillborn_edges(cycle=cyc, max_per_pass=10)
        self.assertEqual(reaped, 0)
        self.assertIn(('a', 'is_a', 'b'), sub.edges)
        q = sub.quarantine_inert_edges(cycle=cyc, max_per_pass=10)
        self.assertEqual(q, 1)
        self.assertIn(('a', 'is_a', 'b'), sub.quarantine_edges)

    def test_fresh_stillborn_protected_by_grace(self):
        # Never cohered, at floor, but engaged within the grace window.
        sub = self._migrated()
        cyc = sub.L_QUARANTINE_CYCLES + 10
        e = sub.add_edge('a', 'b', 'co_occurs',
                         strength=EDGE_PRUNE_FLOOR, cycle=0)
        e.last_engaged_cycle = cyc - 5    # touched recently
        n = sub.reap_stillborn_edges(cycle=cyc, max_per_pass=10)
        self.assertEqual(n, 0)
        self.assertIn(('a', 'co_occurs', 'b'), sub.edges)

    def test_above_floor_stillborn_protected(self):
        # Never cohered, stale, but still ABOVE the floor -> not reaped.
        sub = self._migrated()
        cyc = sub.L_QUARANTINE_CYCLES + 10
        e = sub.add_edge('a', 'b', 'co_occurs', strength=0.5, cycle=0)
        e.last_engaged_cycle = 0
        e.last_reinforced_cycle = cyc     # keep effective strength high
        n = sub.reap_stillborn_edges(cycle=cyc, max_per_pass=10)
        self.assertEqual(n, 0)
        self.assertIn(('a', 'co_occurs', 'b'), sub.edges)

    def test_reap_gated_on_migration(self):
        sub = Substrate()                 # NOT migrated
        e = sub.add_edge('a', 'b', 'co_occurs',
                         strength=EDGE_PRUNE_FLOOR, cycle=0)
        e.last_engaged_cycle = 0
        n = sub.reap_stillborn_edges(
            cycle=sub.L_QUARANTINE_CYCLES + 10, max_per_pass=10)
        self.assertEqual(n, 0)
        self.assertIn(('a', 'co_occurs', 'b'), sub.edges)

    def test_reap_bounded_by_max_per_pass(self):
        sub = self._migrated()
        cyc = sub.L_QUARANTINE_CYCLES + 10
        for i in range(10):
            e = sub.add_edge('s%d' % i, 't%d' % i, 'co_occurs',
                             strength=EDGE_PRUNE_FLOOR, cycle=0)
            e.last_engaged_cycle = 0
        n = sub.reap_stillborn_edges(cycle=cyc, max_per_pass=3)
        self.assertEqual(n, 3)            # capped
        self.assertEqual(len(sub.edges), 7)

    def test_reaped_edge_removed_from_edges_out(self):
        sub = self._migrated()
        cyc = sub.L_QUARANTINE_CYCLES + 10
        e = sub.add_edge('a', 'b', 'co_occurs',
                         strength=EDGE_PRUNE_FLOOR, cycle=0)
        e.last_engaged_cycle = 0
        sub.reap_stillborn_edges(cycle=cyc, max_per_pass=10)
        a = sub.concepts.get('a')
        self.assertNotIn('co_occurs', a.edges_out)


class TestStillbornBacklogSweep(unittest.TestCase):

    def test_sweep_preserves_cohered_deletes_stillborn(self):
        sub = Substrate()
        sub._quarantine_migrated = True   # isolate sweep from migration
        for i in range(5):                # noise: never cohered, at floor
            e = sub.add_edge('n%d' % i, 'm%d' % i, 'co_occurs',
                             strength=EDGE_PRUNE_FLOOR, cycle=0)
            e.last_engaged_cycle = 0
        keep = sub.add_edge('real_a', 'real_b', 'is_a',
                            strength=EDGE_PRUNE_FLOOR, cycle=0)
        keep.last_engaged_cycle = 0
        keep.first_coherent_cycle = 7     # a real past-self
        before = len(sub.edges)
        reaped = sub.sweep_stillborn_backlog(cycle=10_000, grace_window=0)
        self.assertEqual(reaped, 5)
        self.assertIn(('real_a', 'is_a', 'real_b'), sub.edges)
        for i in range(5):
            self.assertNotIn(('n%d' % i, 'co_occurs', 'm%d' % i),
                             sub.edges)
        self.assertEqual(len(sub.edges), before - 5)

    def test_sweep_forces_migration(self):
        # Un-migrated substrate with only fcc==0 noise: the sweep must
        # drive migration to completion (so fcc==0 is trustworthy) then
        # reap.  No fcc>0 edges here, so migration is a trivial pass.
        sub = Substrate()
        self.assertFalse(sub._quarantine_migrated)
        for i in range(3):
            e = sub.add_edge('n%d' % i, 'm%d' % i, 'co_occurs',
                             strength=EDGE_PRUNE_FLOOR, cycle=0)
            e.last_engaged_cycle = 0
        reaped = sub.sweep_stillborn_backlog(cycle=10_000, grace_window=0)
        self.assertTrue(sub._quarantine_migrated)
        self.assertEqual(reaped, 3)
        self.assertEqual(len(sub.edges), 0)


class TestScopedConsolidation(unittest.TestCase):

    def _coherent_triangle(self):
        # a is_a x, x is_a b  =>  (a, is_a, b) coheres via is_a∘is_a→is_a
        sub = Substrate()
        for n in ('a', 'x', 'b'):
            sub.add_concept(Concept(name=n))
        sub.add_edge('a', 'x', 'is_a', strength=0.3, cycle=0)
        sub.add_edge('x', 'b', 'is_a', strength=0.3, cycle=0)
        sub.add_edge('a', 'b', 'is_a', strength=0.3, cycle=0)
        return sub

    def test_full_scan_backcompat(self):
        # candidates=None -> the whole-substrate scan (unchanged).
        sub = self._coherent_triangle()
        reinforced, newly = sub.reinforce_coherent_edges(cycle=1)
        self.assertGreaterEqual(newly, 1)
        self.assertGreater(
            sub.edges[('a', 'is_a', 'b')].first_coherent_cycle, 0)

    def test_scoped_to_source_finds_coherence(self):
        sub = self._coherent_triangle()
        reinforced, newly = sub.reinforce_coherent_edges(
            cycle=1, candidates={'a'})
        self.assertGreaterEqual(newly, 1)
        self.assertGreater(
            sub.edges[('a', 'is_a', 'b')].first_coherent_cycle, 0)

    def test_scoped_excludes_unscoped_sources(self):
        # The cohering candidate's source is 'a'; scoping to {'x'} must
        # NOT discover it — proves the scope actually bounds the work.
        sub = self._coherent_triangle()
        reinforced, newly = sub.reinforce_coherent_edges(
            cycle=1, candidates={'x'})
        self.assertEqual(newly, 0)
        self.assertEqual(
            sub.edges[('a', 'is_a', 'b')].first_coherent_cycle, 0)

    def test_empty_candidates_is_noop(self):
        sub = self._coherent_triangle()
        reinforced, newly = sub.reinforce_coherent_edges(
            cycle=1, candidates=set())
        self.assertEqual(reinforced, 0)
        self.assertEqual(newly, 0)

    def test_add_edge_marks_dirty_and_consume_clears(self):
        sub = Substrate()
        sub.add_edge('a', 'b', 'is_a', strength=0.3, cycle=0)
        self.assertIn('a', sub._dirty_concepts)
        self.assertIn('b', sub._dirty_concepts)
        d = sub.consume_dirty()
        self.assertEqual(d, {'a', 'b'})
        self.assertEqual(sub._dirty_concepts, set())   # cleared

    def test_dirty_drives_scoped_reinforce_end_to_end(self):
        sub = self._coherent_triangle()
        cand = sub.consume_dirty()
        self.assertIn('a', cand)
        reinforced, newly = sub.reinforce_coherent_edges(
            cycle=1, candidates=cand)
        self.assertGreaterEqual(newly, 1)

    def test_dirty_not_serialized_empty_on_load(self):
        # Runtime-only: empty on load means the first post-load pass
        # re-consolidates nothing (safe), and the existing coherent
        # core keeps its persisted stamps regardless.
        sub = self._coherent_triangle()
        self.assertTrue(sub._dirty_concepts)
        d = sub.to_dict()
        self.assertNotIn('dirty_concepts', d)
        sub2 = Substrate.from_dict(d)
        self.assertEqual(sub2._dirty_concepts, set())


if __name__ == '__main__':
    unittest.main()
