"""Grounding world-model loop (BUILD 1) — the unfakeable test battery.

Proves the loop is GROUNDING, not a lookup table, and that the
audit's three must-fixes hold:

  - learns a deterministic world (predict-confirm-earn);
  - GENERALISES to held-out states a lookup table cannot (the
    unfakeable bar — lookup baseline must FAIL where the loop wins);
  - a Myhill-Nerode state-distinguishing probe (distinct futures ->
    distinct predictions; same future -> merged equivalence class);
  - DECOUPLED from lifeforce: transitions_to is absent from
    RELATION_COMPOSITION, so transition edges never cohere and never
    reach mortality's record_learning (must-fix 1);
  - MISS-writes do NOT engage; only a correct prediction grants
    survival (must-fix 2);
  - earn-or-dissolve via prediction: an engaged transition at the
    floor survives a reap pass; a stale un-predicted one is reaped.

See project_seagi_grounding_step1.
"""

from __future__ import annotations

import types
import unittest

from seagi.core.substrate import (
    Substrate, EDGE_PRUNE_FLOOR,
)
from seagi.brain.capabilities.grounding import GroundingLoop, WORLD_RELATION
from seagi.brain.capabilities.writer import JournaledSubstrateWriter


class _DeliverBus:
    """Minimal bus that DELIVERS published writes to the writer (so the
    single-writer contract is exercised end-to-end in the test)."""

    def __init__(self, writer):
        self.writer = writer
        self.published = []

    def publish(self, event):
        self.published.append(event)
        self.writer.handle(event, self)


def _rig():
    sub = Substrate()
    sub._quarantine_migrated = True          # enable the reaper
    engine = types.SimpleNamespace(substrate=sub)
    writer = JournaledSubstrateWriter(engine=engine)
    bus = _DeliverBus(writer)
    loop = GroundingLoop(engine=engine, bus=bus)
    return sub, writer, bus, loop


class TestLearnsWorld(unittest.TestCase):

    def test_learns_deterministic_world(self):
        sub, _, _, loop = _rig()
        seq = ['a', 'b', 'c'] * 6        # a->b->c->a... deterministic
        for i in range(1, len(seq)):
            loop.observe(seq[i - 1], seq[i], cycle=i)
        st = loop.stats()
        # after the first pass every step confirms
        self.assertGreater(st['predictions_made'], 5)
        self.assertGreater(st['confirm_rate'], 0.6)
        # learned transitions exist and the confirmed ones are ENGAGED
        e = sub.edges[('a', WORLD_RELATION, 'b')]
        self.assertGreater(e.last_engaged_cycle, 0)


class TestGeneralizationVsLookup(unittest.TestCase):

    def test_generalizes_to_held_out_where_lookup_fails(self):
        sub, _, _, loop = _rig()
        # category facts (known): a1..a4 are typeA.  Hidden rule:
        # everything of typeA transitions to X.
        for m in ('a1', 'a2', 'a3', 'a4'):
            sub.add_edge(m, 'typeA', 'is_a', strength=0.5, cycle=0)
        observed = set()
        c = 10
        for m in ('a1', 'a2', 'a3'):        # train (a4 held out)
            loop.observe(m, 'X', cycle=c); c += 1
            loop.observe(m, 'X', cycle=c); c += 1
            observed.add((m, 'X'))

        # a4 was NEVER observed transitioning.  The loop predicts it by
        # what its KIND does — generalisation.
        pred, via = loop._predict('a4', cycle=c)
        self.assertEqual(pred, 'X')
        self.assertEqual(via, 'inherit')

        # The lookup-table baseline CANNOT generalise — the unfakeable
        # bar.  It scores zero on the held-out state.
        def lookup(s):
            for (p, t) in observed:
                if p == s:
                    return t
            return None
        self.assertIsNone(lookup('a4'))

        # And the held-out transition CONFIRMS on first encounter, via
        # inheritance (a generalised confirm).
        g0 = loop.generalized
        loop.observe('a4', 'X', cycle=c)
        self.assertEqual(loop.generalized, g0 + 1)


class TestStateDistinguishingProbe(unittest.TestCase):

    def test_myhill_nerode_probe(self):
        sub, _, _, loop = _rig()
        for m in ('a1', 'a2'):
            sub.add_edge(m, 'typeA', 'is_a', strength=0.5, cycle=0)
        for m in ('b1', 'b2'):
            sub.add_edge(m, 'typeB', 'is_a', strength=0.5, cycle=0)
        c = 10
        for m in ('a1', 'a2'):
            loop.observe(m, 'X', cycle=c); c += 1
            loop.observe(m, 'X', cycle=c); c += 1
        for m in ('b1', 'b2'):
            loop.observe(m, 'Y', cycle=c); c += 1
            loop.observe(m, 'Y', cycle=c); c += 1
        # distinct futures -> distinct predictions
        pa, _ = loop._predict('a1', cycle=c)
        pb, _ = loop._predict('b1', cycle=c)
        self.assertEqual(pa, 'X')
        self.assertEqual(pb, 'Y')
        self.assertNotEqual(pa, pb)
        # a held-out typeA member merges into the same equivalence class
        sub.add_edge('a3', 'typeA', 'is_a', strength=0.5, cycle=0)
        pa3, via = loop._predict('a3', cycle=c)
        self.assertEqual(pa3, 'X')
        self.assertEqual(via, 'inherit')


class TestDecoupledFromLifeforce(unittest.TestCase):

    def test_transitions_relation_not_a_composition_target(self):
        # If any composition produced 'transitions_to', a transition
        # edge could cohere -> newly_coherent -> record_learning ->
        # lifeforce.  Must-fix 1: it must NOT be a composition output.
        from seagi.brain.capabilities.cortical import RELATION_COMPOSITION
        self.assertNotIn(WORLD_RELATION,
                         set(RELATION_COMPOSITION.values()))

    def test_world_edges_never_cohere(self):
        sub, _, _, loop = _rig()
        for i, (p, a) in enumerate(
                [('a', 'b'), ('b', 'c'), ('c', 'a')] * 4):
            loop.observe(p, a, cycle=i + 1)
        # a full coherence pass must NOT stamp any transition edge
        sub.reinforce_coherent_edges(cycle=100)
        for (s, r, t), e in sub.edges.items():
            if r == WORLD_RELATION:
                self.assertEqual(
                    e.first_coherent_cycle, 0,
                    "world transition cohered -> would reach lifeforce")


class TestMustFix2EngagementWire(unittest.TestCase):

    def test_miss_does_not_engage_confirm_does(self):
        sub, _, _, loop = _rig()
        # a confidently-WRONG standing prediction: x -> wrong (strong)
        sub.add_edge('x', 'wrong', WORLD_RELATION, strength=0.8, cycle=0)
        # observe x -> right (a MISS: predicted 'wrong')
        loop.observe('x', 'right', cycle=10)
        self.assertEqual(loop.misses, 1)
        e_right = sub.edges[('x', WORLD_RELATION, 'right')]
        # must-fix 2: an observed-but-not-predicted transition is NOT
        # engaged — being seen is not earning.
        self.assertEqual(e_right.last_engaged_cycle, 0)

        # keep observing x->right; it climbs, overtakes x->wrong, then
        # is PREDICTED correctly -> confirm -> engaged.
        c = 11
        for _ in range(220):
            loop.observe('x', 'right', cycle=c); c += 1
        e_right = sub.edges[('x', WORLD_RELATION, 'right')]
        self.assertGreater(loop.confirms, 0)
        self.assertGreater(e_right.last_engaged_cycle, 0)


class TestEarnOrDissolve(unittest.TestCase):

    def test_engaged_at_floor_survives_stale_orphan_reaped(self):
        sub, _, _, loop = _rig()
        # an engaged transition (predicted correctly) ...
        for c in range(1, 8):
            loop.observe('p', 'q', cycle=c)
        # ... and an orphan observed once, never predicted correctly
        loop.observe('m', 'n', cycle=1)
        self.assertIn(('p', WORLD_RELATION, 'q'), sub.edges)
        self.assertIn(('m', WORLD_RELATION, 'n'), sub.edges)

        # Both pushed to the floor.  The engaged one was touched
        # recently (survival signal); the orphan is stale.
        cyc = sub.L_QUARANTINE_CYCLES + 50
        e_pq = sub.edges[('p', WORLD_RELATION, 'q')]
        e_pq.strength = EDGE_PRUNE_FLOOR
        e_pq.last_reinforced_cycle = 0
        e_pq.last_engaged_cycle = sub.L_QUARANTINE_CYCLES + 40  # fresh
        e_mn = sub.edges[('m', WORLD_RELATION, 'n')]
        e_mn.strength = EDGE_PRUNE_FLOOR
        e_mn.last_reinforced_cycle = 0
        self.assertEqual(e_mn.last_engaged_cycle, 0)            # never

        sub.reap_stillborn_edges(cycle=cyc, max_per_pass=50)
        # engaged-recently -> survives; stale orphan -> reaped (dissolve)
        self.assertIn(('p', WORLD_RELATION, 'q'), sub.edges)
        self.assertNotIn(('m', WORLD_RELATION, 'n'), sub.edges)


if __name__ == '__main__':
    unittest.main()
