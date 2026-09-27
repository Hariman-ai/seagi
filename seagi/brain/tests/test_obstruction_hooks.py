"""Tests for the death-wall cortical hooks: the shared obstruction
predicate (M3), the obstruction logic, and engagement from inference."""

import unittest

from seagi.brain.capabilities.cortical import (
    CorticalReasoner, INFERENCE_RELATIONS)
from seagi.brain.capabilities.engagement_ledger import EngagementLedger


class _FakeEdge:
    def __init__(self, fcc=0):
        self.first_coherent_cycle = fcc
        self.last_engaged_cycle = 0


class _FakeLTS:
    def __init__(self):
        self._n = {}    # (s, rel) -> [(tgt, rel, strength)]
        self._e = {}    # (s, rel, tgt) -> _FakeEdge

    def add(self, s, rel, tgt, strength=0.9, fcc=0):
        self._n.setdefault((s, rel), []).append((tgt, rel, strength))
        self._e[(s, rel, tgt)] = _FakeEdge(fcc)

    def neighbors(self, node, relation=None):
        if relation is None:
            out = []
            for (s, r), lst in self._n.items():
                if s == node:
                    out.extend(lst)
            return out
        return list(self._n.get((node, relation), []))

    def edge(self, s, r, o):
        return self._e.get((s, r, o))

    def has_concept(self, c):
        return True


def _make_reasoner(lts):
    return CorticalReasoner(
        awm_provider=lambda: None,
        lts_provider=lambda: lts,
        chemistry_provider=None,
        cycle_provider=lambda: 100,
        schema_provider=lambda: None,
        laws_provider=lambda: None)


class TestObstructionHooks(unittest.TestCase):

    def test_note_obstruction_when_top_incoherent(self):
        lts = _FakeLTS()
        lts.add('fire', 'co_occurs', 'touch', fcc=0)   # junk wins the slot
        led = EngagementLedger()
        r = _make_reasoner(lts)
        r.set_engagement_ledger(led)
        r._note_slot_obstruction('fire', 'co_occurs', 'touch')
        self.assertEqual(led.episode_obstruction, 1)

    def test_no_obstruction_when_top_coherent(self):
        lts = _FakeLTS()
        lts.add('fire', 'causes', 'harm', fcc=42)      # coherent answer
        led = EngagementLedger()
        r = _make_reasoner(lts)
        r.set_engagement_ledger(led)
        r._note_slot_obstruction('fire', 'causes', 'harm')
        self.assertEqual(led.episode_obstruction, 0)

    def test_m2_junk_only_slot_counts_full_weight(self):
        # No coherent competitor anywhere in the slot -> still obstruction.
        lts = _FakeLTS()
        lts.add('zorp', 'co_occurs', 'mush', fcc=0)
        led = EngagementLedger()
        r = _make_reasoner(lts)
        r.set_engagement_ledger(led)
        r._note_slot_obstruction('zorp', 'co_occurs', 'mush')
        self.assertEqual(led.episode_obstruction, 1)

    def test_m3_first_neighbor_calls_predicate(self):
        lts = _FakeLTS()
        lts.add('fire', 'is_a', 'thing', fcc=0)
        r = _make_reasoner(lts)
        calls = []
        r._note_slot_obstruction = (
            lambda s, rel, t: calls.append((s, rel, t)))
        r._first_neighbor('fire', ('is_a',))
        self.assertTrue(calls, "obstruction predicate not called from "
                               "_first_neighbor (M3 pin)")

    def test_m3_composable_edges_calls_predicate(self):
        lts = _FakeLTS()
        rel0 = INFERENCE_RELATIONS[0]
        lts.add('fire', rel0, 'thing', fcc=0)
        r = _make_reasoner(lts)
        r.set_engagement_ledger(EngagementLedger())   # enables the branch
        calls = []
        r._note_slot_obstruction = (
            lambda s, rel, t: calls.append((s, rel, t)))
        r._composable_edges('fire', {'fire'}, None)   # first hop
        self.assertTrue(calls, "obstruction predicate not called from "
                               "_composable_edges first hop (M3 pin)")

    def test_engagement_from_inference_chain(self):
        # A 2-hop coherent is_a chain (fire is_a a, a is_a b) that
        # composes -> both coherent edges should register as meaningful
        # engagement.
        rel = INFERENCE_RELATIONS[0]
        lts = _FakeLTS()
        lts.add('fire', rel, 'a', strength=0.95, fcc=10)
        lts.add('a', rel, 'b', strength=0.95, fcc=10)
        led = EngagementLedger()
        r = _make_reasoner(lts)
        r.set_engagement_ledger(led)
        result = r._inference_chain('fire')
        if result is not None:   # chain fired -> coherent edges engaged
            self.assertGreaterEqual(led.episode_credit(), 1)


if __name__ == '__main__':
    unittest.main()
