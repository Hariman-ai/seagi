"""Unit tests for EngagementLedger (death-wall organs b/c accumulator)."""

import unittest

from seagi.brain.capabilities.engagement_ledger import EngagementLedger


class _FakeEdge:
    def __init__(self, source, relation, target, first_coherent_cycle):
        self.source = source
        self.relation = relation
        self.target = target
        self.first_coherent_cycle = first_coherent_cycle
        self.key = (source, relation, target)


class TestEngagementLedger(unittest.TestCase):

    def test_engaged_counts_distinct_coherent_edges(self):
        L = EngagementLedger()
        e1 = _FakeEdge('a', 'is_a', 'b', 100)
        e2 = _FakeEdge('a', 'is_a', 'c', 200)
        L.note_engaged(e1)
        L.note_engaged(e1)   # same edge twice -> dedupe to 1
        L.note_engaged(e2)
        self.assertEqual(L.episode_credit(), 2)

    def test_incoherent_edges_never_credit(self):
        # first_coherent_cycle == 0 -> junk, cannot buy life.
        L = EngagementLedger()
        L.note_engaged(_FakeEdge('x', 'co', 'y', 0))
        self.assertEqual(L.episode_credit(), 0)

    def test_obstruction_dedupes_per_slot(self):
        L = EngagementLedger()
        L.note_obstruction('fire', 'co_occurs')
        L.note_obstruction('fire', 'co_occurs')   # same slot -> 1
        L.note_obstruction('fire', 'causes')      # different slot -> +1
        self.assertEqual(L.episode_obstruction, 2)

    def test_roll_episode_appends_and_resets(self):
        L = EngagementLedger()
        L.note_obstruction('a', 'r')
        L.note_obstruction('b', 'r')
        L.note_engaged(_FakeEdge('a', 'r', 'b', 5))
        self.assertEqual(L.episode_obstruction, 2)
        self.assertEqual(L.episode_credit(), 1)
        L.roll_episode()
        self.assertEqual(L.episode_obstruction, 0)
        self.assertEqual(L.episode_credit(), 0)
        self.assertEqual(L.A_scale, 2.0)   # mean of [2.0]

    def test_A_scale_empty_is_zero(self):
        self.assertEqual(EngagementLedger().A_scale, 0.0)

    def test_slot_dedupe_clears_each_episode(self):
        L = EngagementLedger()
        L.note_obstruction('a', 'r')
        L.roll_episode()
        L.note_obstruction('a', 'r')   # same slot, NEW episode -> counts
        self.assertEqual(L.episode_obstruction, 1)

    def test_persistence_roundtrips_deque(self):
        L = EngagementLedger()
        L.note_obstruction('a', 'r'); L.roll_episode()
        L.note_obstruction('a', 'r'); L.note_obstruction('b', 'r')
        L.roll_episode()
        L2 = EngagementLedger()
        L2.load_dict(L.to_dict())
        self.assertEqual(list(L2._obstruction_deque), [1.0, 2.0])
        self.assertEqual(L2.A_scale, L.A_scale)


if __name__ == '__main__':
    unittest.main()
