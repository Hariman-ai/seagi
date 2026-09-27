"""THE PROBLEM YIELDING FEELS GOOD — step 1 of the M/I chain.

`insight` is his largest positive chemistry event (dopamine +0.010,
"the moment that must be encoded") and until 2026-08-19 it had ONE call
site in the whole codebase: `forager.py:255`, on finishing a READ.  So
solving produced no pleasure -- only winning did -- and on the 18 of 25
ARC games he has never won there was nothing to feel and nothing to want.
64% of his executions go to those games.

These tests pin the felt pulse for a regularity that was FAILING and now
HOLDS, and -- more importantly -- pin that it does NOT degenerate into
`confirmed_i`, the routine correctness tag that fires several times a
minute and is satisfiable by confirming that walls do not move.
"""

from __future__ import annotations

import unittest

from seagi.body.engine import Engine
from seagi.brain.runtime import Brain
from seagi.brain.capabilities import world_actor as wa_mod


class _Gate:
    """Swap the module-level file gate for a deterministic one."""

    def __init__(self, on: bool):
        self.on = on
        self._prev = None

    def __enter__(self):
        self._prev = wa_mod._INSIGHT_ON
        wa_mod._INSIGHT_ON = (lambda: self.on)
        return self

    def __exit__(self, *exc):
        wa_mod._INSIGHT_ON = self._prev
        return False


def _play(ticks: int = 4000):
    """A fresh brain on the maze rung -- deterministic, and a world where
    transitions genuinely start mispredicted and then stabilise."""
    b = Brain(engine=Engine())
    b.goal_world._idx = 1
    b.goal_world.mastery_threshold = 10 ** 9
    for _ in range(ticks):
        b.tick()
    return b


class TestInsightFiresOnYield(unittest.TestCase):

    def test_gate_off_never_fires(self):
        """Inert until deliberately switched on -- the deploy default."""
        with _Gate(False):
            b = _play()
        self.assertEqual(b.world_actor.insight_fires, 0)

    def test_gate_on_fires_on_a_yielding_regularity(self):
        with _Gate(True):
            b = _play()
        self.assertGreater(
            b.world_actor.insight_fires, 0,
            'a regularity that was failing and then held must be felt')

    def test_is_rarer_than_routine_correctness(self):
        """THE GUARD THAT MATTERS.  If this ever approaches
        `efficacy_confirms` it has become `confirmed_i` -- being right about
        one pixel -- and the whole point is lost."""
        with _Gate(True):
            b = _play()
        w = b.world_actor
        self.assertGreater(w.efficacy_confirms, 0)
        self.assertLess(
            w.insight_fires, w.efficacy_confirms,
            'insight must be rarer than routine correctness')

    def test_rate_decays_as_the_world_settles(self):
        """THE ANTI-FARM PROPERTY, and the one review caught me missing.

        A regularity that has settled pays nothing more, so a world he has
        learned must go QUIET.  The first version of this fired on the same
        predicate as `confirmed_i`; in ARC, where prediction flaps at 41%,
        that would have paid him ~44 times per 1k executions forever, most
        of it for the transitions he understood LEAST.  If this rate does
        not fall, that failure has returned."""
        with _Gate(True):
            b = Brain(engine=Engine())
            b.goal_world._idx = 1
            b.goal_world.mastery_threshold = 10 ** 9
            w = b.world_actor
            for _ in range(3000):
                b.tick()
            f1, e1 = w.insight_fires, w.executions
            for _ in range(9000):
                b.tick()
            f2, e2 = w.insight_fires - f1, w.executions - e1
        self.assertGreater(f1, 0)
        self.assertLess(
            f2 / max(1, e2), f1 / max(1, e1),
            'a settled world must pay less; repetition must earn nothing')

    def test_settled_keys_are_marked_not_reburned(self):
        """Each settled regularity is banked once.  Fires may exceed the
        banked count only when a regularity BROKE and settled again."""
        with _Gate(True):
            b = _play(ticks=3000)
        w = b.world_actor
        banked = sum(1 for v in w._pred_ok.values() if v is True)
        self.assertGreater(banked, 0)
        self.assertGreaterEqual(w.insight_fires, banked)

    def test_counter_is_surfaced(self):
        with _Gate(True):
            b = _play(ticks=800)
        s = b.status()['world_actor']
        self.assertIn('insight_fires', s)
        self.assertIn('insight_on', s)


if __name__ == '__main__':
    unittest.main()
