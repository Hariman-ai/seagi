"""Reinforce-on-recall igniter tests (2026-05-31).

The dry-reverie deadlock: reverie WALKS edges (1-hop 'causal'
recalls) but those thoughts were discarded, so nothing wrote back,
no debt accrued, the agent never slept, and edges decayed to the
quarantine floor.  The igniter routes recall through the metered
consolidator -> writer chain so writes accrue MetabolicDebt AND
stamp last_engaged_cycle, while earn-or-dissolve still governs
survival.

These tests prove (a) it works and (b) it is NOT a scheduler in
disguise — recall can never farm permanence or mortality credit;
only a later coherence pass can.  See
project_seagi_dry_reverie_igniter.
"""

import os
import types
import unittest

from seagi.core.substrate import (
    Substrate,
    EDGE_STRENGTH_BUMP_PER_USE,
    COHERENCE_REINFORCE_BUMP,
    EDGE_PRUNE_FLOOR,
    PROVISIONAL_EDGE_STRENGTH,
)
from seagi.brain.events import (
    EventKind, ThoughtProducedEvent, SubstrateWriteQueuedEvent,
)
from seagi.brain.capabilities.reasoning_consolidator import (
    ReasoningConsolidator,
)
from seagi.brain.capabilities.writer import JournaledSubstrateWriter
from seagi.brain.capabilities.metabolic_debt import MetabolicDebt
from seagi.body.engine import Engine
from seagi.core.substrate import Concept
from seagi.brain import Brain


class _CaptureBus:
    """Minimal bus: records published events; no dispatch."""

    def __init__(self):
        self.published = []

    def publish(self, event):
        self.published.append(event)


def _thought(cycle, focal, relation, target, method, edges_walked=()):
    return ThoughtProducedEvent(
        kind=EventKind.THOUGHT_PRODUCED,
        cycle=cycle,
        source_capability='cortical',
        origin='internal',
        focal=focal, relation=relation, target=target,
        method=method, confidence=0.5,
        edges_walked=tuple(edges_walked))


def _write(cycle, subject, relation, obj, reason,
           strength=PROVISIONAL_EDGE_STRENGTH):
    return SubstrateWriteQueuedEvent(
        kind=EventKind.SUBSTRATE_WRITE_QUEUED,
        cycle=cycle,
        source_capability='reasoning_consolidator',
        origin='internal',
        subject=subject, relation=relation, object=obj,
        strength=strength, write_reason=reason)


class TestConsolidatorDispatch(unittest.TestCase):

    def setUp(self):
        self.bus = _CaptureBus()
        self.con = ReasoningConsolidator(self.bus)

    def _writes(self):
        return [e for e in self.bus.published
                if isinstance(e, SubstrateWriteQueuedEvent)]

    def test_causal_recall_emits_one_recall_reattest_write(self):
        self.con.handle(
            _thought(10, 'socrates', 'is_a', 'man', 'causal'),
            self.bus)
        ws = self._writes()
        self.assertEqual(len(ws), 1)
        self.assertEqual(ws[0].write_reason, 'recall_reattest')
        self.assertEqual(
            (ws[0].subject, ws[0].relation, ws[0].object),
            ('socrates', 'is_a', 'man'))
        self.assertEqual(self.con.recalls_reattested, 1)
        self.assertEqual(self.con.recall_writes_emitted, 1)
        self.assertEqual(self.con.recent_recall_keys(),
                         [('socrates', 'is_a', 'man')])

    def test_inference_still_consolidates_unchanged(self):
        self.con.handle(
            _thought(10, 'socrates', 'is_a', 'mortal', 'inference'),
            self.bus)
        ws = self._writes()
        self.assertEqual(len(ws), 1)
        self.assertEqual(ws[0].write_reason, 'inference_consolidate')
        self.assertEqual(self.con.consolidations_emitted, 1)
        # Inference is NOT counted as a recall.
        self.assertEqual(self.con.recalls_reattested, 0)

    def test_other_methods_drop(self):
        for m in ('metacog', 'counterfactual', 'schema', 'recall'):
            self.con.handle(
                _thought(10, 'a', 'is_a', 'b', m), self.bus)
        self.assertEqual(len(self._writes()), 0)
        self.assertEqual(self.con.recalls_reattested, 0)
        self.assertEqual(self.con.consolidations_emitted, 0)

    def test_guards_drop_self_loop_and_incomplete(self):
        # focal == target
        self.con.handle(
            _thought(10, 'x', 'is_a', 'x', 'causal'), self.bus)
        # missing target
        self.con.handle(
            _thought(10, 'x', 'is_a', '', 'causal'), self.bus)
        self.assertEqual(len(self._writes()), 0)
        self.assertEqual(self.con.recalls_reattested, 0)


class TestWriterRecallStamp(unittest.TestCase):

    def setUp(self):
        self.sub = Substrate()
        self.engine = types.SimpleNamespace(substrate=self.sub)
        self.writer = JournaledSubstrateWriter(engine=self.engine)

    def test_recall_reinforces_and_stamps_last_engaged(self):
        e = self.sub.add_edge('socrates', 'man', 'is_a',
                              strength=0.10, cycle=100)
        e.last_engaged_cycle = 50           # stale prior engagement
        before = e.strength
        # Write at the creation cycle so lazy decay is zero and the
        # bump is clean (decay itself is tested elsewhere / is
        # correct behavior).
        self.writer._apply(
            _write(100, 'socrates', 'is_a', 'man', 'recall_reattest'))
        # Reinforced by the promille per-use bump (default delta).
        self.assertAlmostEqual(
            e.strength, before + EDGE_STRENGTH_BUMP_PER_USE, places=6)
        # AND the quarantine engagement signal is stamped fresh.
        self.assertEqual(e.last_engaged_cycle, 100)

    def test_recall_does_not_fake_coherence(self):
        e = self.sub.add_edge('a', 'b', 'is_a',
                              strength=0.10, cycle=100)
        self.assertEqual(e.first_coherent_cycle, 0)
        self.writer._apply(
            _write(200, 'a', 'is_a', 'b', 'recall_reattest'))
        # Recall reinforces strength but can NEVER set the coherence
        # stamp — only Phase S corroboration does.  This is the
        # anti-gaming wall: recall can't manufacture mortality credit.
        self.assertEqual(e.first_coherent_cycle, 0)

    def test_stamp_scoped_to_cognitive_reasons(self):
        e = self.sub.add_edge('a', 'b', 'is_a',
                              strength=0.10, cycle=100)
        e.last_engaged_cycle = 100
        # A non-cognitive write_reason still reinforces but must NOT
        # stamp engagement (smallest-version scoping; corpus re-
        # attestation is a deliberate follow-up).
        self.writer._apply(_write(200, 'a', 'is_a', 'b', 'corpus'))
        self.assertEqual(e.last_engaged_cycle, 100)


class TestMetabolicDebtAccrues(unittest.TestCase):

    def test_recall_write_accrues_debt(self):
        bus = _CaptureBus()
        debt = MetabolicDebt(bus)
        self.assertEqual(debt.writes_observed, 0)
        debt.handle(
            _write(10, 'a', 'is_a', 'b', 'recall_reattest'), bus)
        # Debt is reason- and strength-agnostic: one write = one unit.
        # This is BOTH what enables sleep AND the accepted thrashing-
        # debt cost (a noise-only substrate feels fatigue).
        self.assertEqual(debt.writes_observed, 1)
        self.assertEqual(debt.debt, 1.0)


class TestQuarantineImmunity(unittest.TestCase):

    def setUp(self):
        self.sub = Substrate()
        self.sub._quarantine_migrated = True
        self.engine = types.SimpleNamespace(substrate=self.sub)
        self.writer = JournaledSubstrateWriter(engine=self.engine)

    def test_recalled_edge_is_immune_to_cleanup(self):
        # An edge at floor strength but STALE -> eligible for cleanup.
        e = self.sub.add_edge('a', 'b', 'rel', strength=0.005,
                              cycle=0)
        e.strength = 0.005                 # <= EDGE_PRUNE_FLOOR
        e.last_engaged_cycle = 0           # very stale
        self.assertLessEqual(
            e.effective_strength(1000), EDGE_PRUNE_FLOOR)
        # Recall it at cycle 1000 -> writer stamps last_engaged=1000.
        self.writer._apply(
            _write(1000, 'a', 'rel', 'b', 'recall_reattest'))
        # At 1005 the recall stamp makes staleness 5 < grace 400 -> the
        # engagement grants immunity from BOTH cleanup arms.  'rel'
        # never coheres (fcc==0), so the relevant arm is the reaper.
        n = self.sub.reap_stillborn_edges(1005, max_per_pass=50)
        self.assertEqual(n, 0)
        self.assertIn(('a', 'rel', 'b'), self.sub.edges)

    def test_unengaged_floor_edge_is_reaped(self):
        # Control: the SAME edge, never recalled, IS cleaned up.
        # Edge-mortality split (2026-06-02): 'rel' never composes, so
        # this edge never coheres (first_coherent_cycle == 0) — it is a
        # STILLBORN.  The unengaged floor stillborn is REAPED (deleted),
        # not quarantined (quarantine now preserves only once-cohered
        # past-selves).  Engagement via recall still grants immunity
        # (the paired test above): a recall stamps last_engaged, which
        # protects against BOTH cleanup arms within the grace window.
        e = self.sub.add_edge('c', 'd', 'rel', strength=0.005,
                              cycle=0)
        e.strength = 0.005
        e.last_engaged_cycle = 0
        n = self.sub.reap_stillborn_edges(1005, max_per_pass=50)
        self.assertGreaterEqual(n, 1)
        self.assertNotIn(('c', 'rel', 'd'), self.sub.edges)
        # Truly deleted — NOT relocated to quarantine.
        self.assertNotIn(('c', 'rel', 'd'), self.sub.quarantine_edges)


class TestInvariantAndFlywheel(unittest.TestCase):

    def test_one_to_five_quantum_invariant(self):
        # Guards anyone "fixing" the igniter by raising the recall
        # quantum: 5 recalls must equal exactly one coherence bump.
        self.assertAlmostEqual(
            EDGE_STRENGTH_BUMP_PER_USE * 5,
            COHERENCE_REINFORCE_BUMP, places=9)

    def test_five_recalls_equal_one_coherence_bump(self):
        sub = Substrate()
        e = sub.add_edge('a', 'b', 'rel', strength=0.0, cycle=100)
        for _ in range(5):
            e.reinforce(100)                       # default delta
        recall_total = e.strength
        e2 = sub.add_edge('c', 'd', 'rel', strength=0.0, cycle=100)
        e2.reinforce(100, COHERENCE_REINFORCE_BUMP)
        self.assertAlmostEqual(recall_total, e2.strength, places=6)

    def test_flywheel_recall_lifts_edge_off_the_floor(self):
        sub = Substrate()
        e = sub.add_edge('a', 'b', 'rel',
                         strength=PROVISIONAL_EDGE_STRENGTH, cycle=100)
        seq = [e.strength]
        for _ in range(10):
            e.reinforce(100)
            seq.append(e.strength)
        # Monotonic climb, and lifted well clear of the prune floor.
        self.assertTrue(all(seq[i] < seq[i + 1]
                            for i in range(len(seq) - 1)))
        self.assertGreater(e.strength, PROVISIONAL_EDGE_STRENGTH)
        self.assertGreater(e.effective_strength(100), EDGE_PRUNE_FLOOR)


class TestCoherentFractionInstrumentation(unittest.TestCase):

    def test_coherent_fraction_counts_active_and_quarantined(self):
        sub = Substrate()
        sub._quarantine_migrated = True
        e1 = sub.add_edge('a', 'b', 'rel', strength=0.5, cycle=10)
        e1.first_coherent_cycle = 50                # cohered
        e2 = sub.add_edge('c', 'd', 'rel', strength=0.5, cycle=10)
        # e2 never cohered (first_coherent_cycle = 0)
        e3 = sub.add_edge('e', 'f', 'rel', strength=0.005, cycle=10)
        e3.first_coherent_cycle = 70
        e3.last_engaged_cycle = 0
        # Drift e3 into quarantine; it must still be counted.
        sub.quarantine_inert_edges(1000, max_per_pass=50)
        self.assertIn(('e', 'rel', 'f'), sub.quarantine_edges)
        keys = [('a', 'rel', 'b'), ('c', 'rel', 'd'),
                ('e', 'rel', 'f'), ('missing', 'rel', 'key')]
        out = sub.coherent_fraction_of(keys)
        self.assertEqual(out['n'], 3)              # missing skipped
        self.assertEqual(out['n_coherent'], 2)     # e1, e3
        self.assertAlmostEqual(out['fraction'], 2.0 / 3.0, places=6)

    def test_empty_keys_safe(self):
        sub = Substrate()
        out = sub.coherent_fraction_of([])
        self.assertEqual(out, {'n': 0, 'n_coherent': 0,
                               'fraction': 0.0})


class TestMakeThoughtDemotion(unittest.TestCase):

    def test_make_thought_no_longer_directly_reinforces(self):
        # Regression guard for the demotion: _make_thought must be
        # SALIENCE-ONLY now — recall reinforcement flows through the
        # metered consolidator->writer path, not a debt-free direct
        # edge.reinforce inside cortical.
        import seagi.brain.capabilities.cortical as cmod
        with open(cmod.__file__, 'r', encoding='utf-8') as f:
            src = f.read()
        start = src.index('def _make_thought')
        nxt = src.index('\n    def ', start + 1)
        body = src[start:nxt]
        # Strip comment lines — the demotion comment legitimately
        # mentions the old call; we only care about live code.
        code = '\n'.join(ln for ln in body.splitlines()
                         if not ln.lstrip().startswith('#'))
        self.assertNotIn('edge.reinforce', code)
        self.assertIn('bump_salience', code)        # salience kept


class TestPerHopConsolidator(unittest.TestCase):
    """Phase 2: a recall thought reinforces the REAL walked hops
    (evidence), not the fabricated collapsed end-to-end triple."""

    def setUp(self):
        self.bus = _CaptureBus()
        self.con = ReasoningConsolidator(self.bus)

    def _recall_writes(self):
        return [e for e in self.bus.published
                if isinstance(e, SubstrateWriteQueuedEvent)
                and e.write_reason == 'recall_reattest']

    def test_2hop_reinforces_real_hops_not_collapsed(self):
        # A 2-hop causal walk A->X->B collapses to thought
        # (A, leads_to, B) but the real evidence is the two hops.
        self.con.handle(
            _thought(10, 'a', 'leads_to', 'b', 'causal',
                     edges_walked=[('a', 'causes', 'x'),
                                   ('x', 'causes', 'b')]),
            self.bus)
        ws = self._recall_writes()
        triples = {(w.subject, w.relation, w.object) for w in ws}
        self.assertEqual(triples,
                         {('a', 'causes', 'x'), ('x', 'causes', 'b')})
        # The fabricated collapsed (a, leads_to, b) is NOT written.
        self.assertNotIn(('a', 'leads_to', 'b'), triples)
        # Counts: one recall THOUGHT, two hop WRITES.
        self.assertEqual(self.con.recalls_reattested, 1)
        self.assertEqual(self.con.recall_writes_emitted, 2)
        self.assertEqual(len(self.con.recent_recall_keys()), 2)

    def test_empty_edges_walked_falls_back_to_triple(self):
        # 1-hop / identity / process walks set no edges_walked →
        # fall back to the single (focal, relation, target).
        self.con.handle(
            _thought(10, 'socrates', 'is_a', 'man', 'causal'),
            self.bus)
        ws = self._recall_writes()
        self.assertEqual(len(ws), 1)
        self.assertEqual((ws[0].subject, ws[0].relation, ws[0].object),
                         ('socrates', 'is_a', 'man'))

    def test_inference_reinforces_its_premises(self):
        # Premise-reinforcement (2026-06-05): an inference writes the
        # CONCLUSION (inference_consolidate) AND reattests the PREMISES
        # that proved it — a proven premise earns strength, moving away
        # from death.  Conclusion and premises are credited distinctly
        # (no fabricated edges).
        self.con.handle(
            _thought(10, 'socrates', 'is_a', 'mortal', 'inference',
                     edges_walked=[('socrates', 'is_a', 'man'),
                                   ('man', 'is_a', 'mortal')]),
            self.bus)
        # Conclusion written once via inference_consolidate.
        cons = [e for e in self.bus.published
                if isinstance(e, SubstrateWriteQueuedEvent)
                and e.write_reason == 'inference_consolidate']
        self.assertEqual(len(cons), 1)
        self.assertEqual(
            (cons[0].subject, cons[0].relation, cons[0].object),
            ('socrates', 'is_a', 'mortal'))
        self.assertEqual(self.con.consolidations_emitted, 1)
        # Both PREMISES reattested (reinforced) via recall writes.
        ws = self._recall_writes()
        self.assertEqual(len(ws), 2)
        self.assertEqual(
            {(w.subject, w.relation, w.object) for w in ws},
            {('socrates', 'is_a', 'man'), ('man', 'is_a', 'mortal')})
        self.assertEqual(self.con.recalls_reattested, 1)
        self.assertEqual(self.con.recall_writes_emitted, 2)

    def test_degenerate_hop_skipped(self):
        # A self-loop hop is skipped; a valid sibling still writes.
        self.con.handle(
            _thought(10, 'a', 'leads_to', 'b', 'causal',
                     edges_walked=[('x', 'causes', 'x'),
                                   ('x', 'causes', 'b')]),
            self.bus)
        triples = {(w.subject, w.relation, w.object)
                   for w in self._recall_writes()}
        self.assertEqual(triples, {('x', 'causes', 'b')})


class TestPerHopEndToEndAndIgnition(unittest.TestCase):
    """End-to-end through a real Brain: per-hop recall reinforces the
    genuine hops, and enough reinforcement IGNITES inference."""

    def _brain_with_chain(self, s1, s2):
        engine = Engine()
        for c in ('a', 'x', 'b'):
            engine.substrate.add_concept(Concept(name=c))
        engine.substrate.add_edge('a', 'x', 'causes',
                                  strength=s1, cycle=0)
        engine.substrate.add_edge('x', 'b', 'causes',
                                  strength=s2, cycle=0)
        return engine, Brain(engine=engine)

    def test_per_hop_reinforces_real_edges_via_bus(self):
        engine, brain = self._brain_with_chain(0.20, 0.20)
        e1 = engine.substrate.edges[('a', 'causes', 'x')]
        e2 = engine.substrate.edges[('x', 'causes', 'b')]
        b1, b2 = e1.strength, e2.strength
        brain.bus.publish(_thought(
            0, 'a', 'leads_to', 'b', 'causal',
            edges_walked=[('a', 'causes', 'x'), ('x', 'causes', 'b')]))
        # Both genuine hops reinforced + engagement-stamped.
        self.assertAlmostEqual(e1.strength,
                               b1 + EDGE_STRENGTH_BUMP_PER_USE, places=6)
        self.assertAlmostEqual(e2.strength,
                               b2 + EDGE_STRENGTH_BUMP_PER_USE, places=6)
        self.assertEqual(e1.last_engaged_cycle, 0)
        self.assertEqual(e2.last_engaged_cycle, 0)
        # No fabricated collapsed (a, leads_to, b) edge was created.
        self.assertNotIn(('a', 'leads_to', 'b'), engine.substrate.edges)

    def test_per_hop_recall_ignites_inference(self):
        # Seed both hops just below the 2-hop inference floor
        # (0.40*0.7*0.40 = 0.112 < INFERENCE_MIN_CONFIDENCE 0.12).
        engine, brain = self._brain_with_chain(0.40, 0.40)
        self.assertIsNone(
            brain.cortical._inference_chain('a'),
            "precondition: chain too weak to infer yet")
        # Drive recall on the real hops (same cycle 0 → no decay).
        for _ in range(12):
            brain.bus.publish(_thought(
                0, 'a', 'leads_to', 'b', 'causal',
                edges_walked=[('a', 'causes', 'x'),
                              ('x', 'causes', 'b')]))
        # Hops now ~0.46 → 0.46*0.7*0.46 = 0.148 >= 0.12 → IGNITION.
        result = brain.cortical._inference_chain('a')
        self.assertIsNotNone(
            result, "per-hop recall should lift the hops past the "
                    "inference floor and ignite a derivation")
        self.assertEqual(result['target'], 'b')
        self.assertEqual(result['relation'], 'leads_to')


class TestCausalChainWalkedHops(unittest.TestCase):
    """_causal_chain returns the REAL traversed hops, not just the
    collapsed end-to-end triple."""

    def test_1hop_returns_the_real_edge(self):
        engine = Engine()
        for c in ('a', 'b'):
            engine.substrate.add_concept(Concept(name=c))
        engine.substrate.add_edge('a', 'b', 'causes',
                                  strength=0.6, cycle=0)
        brain = Brain(engine=engine)
        out = brain.cortical._causal_chain('a', max_hops=2)
        self.assertIsNotNone(out)
        # 1-hop walk → the single genuine edge.
        self.assertEqual(out[-1], [('a', 'causes', 'b')])

    def test_2hop_captures_two_genuine_hops_with_real_mid(self):
        # NOTE: under neutral chemistry a 2-hop never beats its own
        # 1-hop prefix (conf2 = conf1*0.7*str2 < conf1*0.9), so 2-hop
        # causal walks are chemistry-gated.  Force the 2-hop to win by
        # boosting the terminal's modulated score, to exercise the
        # walked-hops capture logic (the core of the per-hop fix).
        engine = Engine()
        for c in ('a', 'x', 'b'):
            engine.substrate.add_concept(Concept(name=c))
        engine.substrate.add_edge('a', 'x', 'causes',
                                  strength=0.6, cycle=0)
        engine.substrate.add_edge('x', 'b', 'causes',
                                  strength=0.6, cycle=0)
        brain = Brain(engine=engine)
        brain.cortical._modulated_score = (
            lambda conf, tgt: 10.0 if tgt == 'b' else conf)
        out = brain.cortical._causal_chain('a', max_hops=2)
        self.assertIsNotNone(out)
        self.assertEqual(out[4], 2, "should be a 2-hop result")
        # The captured hops are the two GENUINE edges with the real
        # middle node 'x' — NOT the collapsed (a, causes, b).
        self.assertEqual(
            out[-1], [('a', 'causes', 'x'), ('x', 'causes', 'b')])


if __name__ == '__main__':
    unittest.main()
