"""Unit tests for the LIVE death-wall in MortalityDrive.

The wall is the agent's AGE: each sleep episode it creeps toward his lifeforce
at the substrate's own physical decay rate, SLOWED by what he learned that
episode (self._episode_earned).  advance = aging * (1 - tanh(earned)).  The wall
only ever advances (he is mortal) — learning buys time, it never stops the clock.
"""

import unittest

from seagi.brain.capabilities.mortality_drive import (
    MortalityDrive, BASELINE_FLOOR, EDGE_STRENGTH_DECAY_PER_CYCLE)


class _FakeLedger:
    def __init__(self):
        self.rolled = 0

    def roll_episode(self):
        self.rolled += 1


def _make(lf=0.9, ledger=None):
    cyc = [0]
    state = {'lf': lf}
    d = MortalityDrive(
        bus=None,
        cycle_provider=lambda: cyc[0],
        lifeforce_get=lambda: state['lf'],
        lifeforce_set=lambda v: state.__setitem__('lf', v),
        engagement_ledger=ledger)
    return d, cyc, state


@unittest.skip(
    'DEBRIS of the 2026-07-14 D8(c) un-freeze: drives MortalityDrive._advance_wall, which no longer exists (aging moved to the per-tick subtractive law). Cannot pass. Kept, not deleted, because test_wall_never_recedes_even_with_huge_learning records the SUPERSEDED design -- the current law and test_immortality_tripwire require the opposite (huge earning MUST drive the wall down). Skipped 2026-09-04 so the suite baseline is readable around mortality changes.')
class TestDeathWall(unittest.TestCase):

    def test_wall_seeds_to_floor_then_ages(self):
        d, cyc, _ = _make()
        self.assertIsNone(d.wall)
        cyc[0] = 1000
        d._advance_wall()                # idle (earned 0) -> full aging
        self.assertGreater(d.wall, BASELINE_FLOOR)

    def test_idle_ages_monotonically(self):
        d, cyc, _ = _make()
        prev = None
        for i in range(1, 6):
            cyc[0] = i * 1000
            d._advance_wall()            # earned 0 every episode
            if prev is not None:
                self.assertGreater(d.wall, prev)
            prev = d.wall

    def test_learning_slows_aging(self):
        # Same episode span: a learner ages LESS than an idle agent.
        d1, c1, _ = _make()
        c1[0] = 1000
        d1._episode_earned = 0.0
        d1._advance_wall()
        idle_advance = d1._last_wall_advance

        d2, c2, _ = _make()
        c2[0] = 1000
        d2._episode_earned = 5.0
        d2._advance_wall()
        learner_advance = d2._last_wall_advance

        self.assertLess(learner_advance, idle_advance)
        self.assertGreaterEqual(learner_advance, 0.0)   # never recedes

    def test_aging_capped_at_one_interval(self):
        # A pathologically long episode cannot age more than one prune-floor's
        # worth (the derived cap) — kills the unbounded-span blow-up.
        d, cyc, _ = _make()
        cyc[0] = 10 ** 9
        d._advance_wall()
        self.assertLessEqual(d._last_f_decay, BASELINE_FLOOR + 1e-12)

    def test_wall_never_recedes_even_with_huge_learning(self):
        d, cyc, _ = _make()
        d._episode_earned = 1_000_000.0
        cyc[0] = 1000
        d._advance_wall()
        self.assertGreaterEqual(d._last_wall_advance, 0.0)
        self.assertGreaterEqual(d.wall, BASELINE_FLOOR)

    def test_advance_wall_does_not_touch_lifeforce(self):
        d, cyc, state = _make(lf=0.9)
        cyc[0] = 1000
        d._advance_wall()
        self.assertEqual(state['lf'], 0.9)   # the wall moves, lifeforce doesn't

    def test_ledger_rolled_each_episode(self):
        L = _FakeLedger()
        d, cyc, _ = _make(ledger=L)
        for i in range(3):
            cyc[0] = i * 100
            d._advance_wall()
        self.assertEqual(L.rolled, 3)

    def test_distance_to_wall(self):
        d, cyc, _ = _make(lf=0.9)
        cyc[0] = 100
        d._advance_wall()
        self.assertAlmostEqual(d.distance_to_wall(), 0.9 - d.wall, places=9)

    def test_persistence_roundtrips_wall(self):
        d, cyc, _ = _make()
        for i in range(3):
            cyc[0] = i * 100
            d._advance_wall()
        snap = d.to_dict()
        d2, _, _ = _make()
        d2.load_dict(snap)
        self.assertAlmostEqual(d2.wall, d.wall, places=9)


if __name__ == '__main__':
    unittest.main()
