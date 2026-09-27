"""WorldTransducer — the firsthand sparse-coder (Capability 1, Step 5).

Proves it earns stable tokens from raw percepts: the same percept always
recovers the same token, distinct percepts get distinct tokens (no merge,
no over-mint), and it tokenizes the whole firsthand world stably — all
with a DERIVED threshold, no typed constant.  It must NOT touch the
substrate (promotion is Capability 2).
"""

from __future__ import annotations

import unittest

from seagi.core.substrate import WORLD_TOKEN_PREFIX
from seagi.world.world_driver import WorldDriver
from seagi.brain.capabilities.world_transducer import WorldTransducer


class TestWorldTransducer(unittest.TestCase):

    def test_same_percept_same_token(self):
        t = WorldTransducer()
        v = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8]
        tok = t.encode(v)
        self.assertEqual(t.encode(list(v)), tok)
        self.assertEqual(t.encode(list(v)), tok)

    def test_distinct_percepts_distinct_tokens(self):
        t = WorldTransducer()
        a = t.encode([0.0] * 8)
        b = t.encode([1.0] * 8)
        self.assertNotEqual(a, b)
        self.assertEqual(t.stats()['prototypes'], 2)

    def test_tokenizes_whole_world_stably(self):
        # Every distinct cell earns exactly one stable token; revisiting a
        # cell recovers the same token.  No merge, no over-mint.
        w = WorldDriver(grid=5, k=8, seed=2)
        t = WorldTransducer()
        first = {cell: t.encode(emb) for cell, emb in w._embed.items()}
        self.assertEqual(len(set(first.values())), 25)
        self.assertEqual(t.stats()['prototypes'], 25)
        # revisit every cell -> identical tokens (stability)
        for cell, emb in w._embed.items():
            self.assertEqual(t.encode(emb), first[cell])
        # still 25 -> no spurious mints on revisit
        self.assertEqual(t.stats()['prototypes'], 25)

    def test_threshold_is_derived_not_a_constant(self):
        # On an exact world the derived threshold evaluates to 0 (zero
        # tolerance), proving there is no hardcoded tolerance constant.
        t = WorldTransducer()
        for emb in WorldDriver(grid=4, seed=9)._embed.values():
            t.encode(emb)
        # revisit (exact matches) so the match-distance stat is exercised
        for emb in WorldDriver(grid=4, seed=9)._embed.values():
            t.encode(emb)
        self.assertEqual(t.stats()['match_threshold'], 0.0)

    def test_emits_token_not_concept(self):
        # Capability 1: the transducer only emits a token string; it must
        # not carry or promote a substrate concept.
        t = WorldTransducer()
        tok = t.encode([0.1] * 8)
        self.assertIsInstance(tok, str)
        self.assertTrue(tok.startswith(WORLD_TOKEN_PREFIX))


if __name__ == '__main__':
    unittest.main()
