"""MortalWorld — the survival grid-world's world-law / anti-gaming invariants
(Step 5, game test).  These pin the properties the doctrine audit required:
death is world-caused and frequent, sitting-still/camping is lethal by world
law, sustenance moves so no safe oscillation can refeed, and the dynamics stay
reproducible.  No agent / lifeforce is involved here — this is purely the
world's physics."""

from __future__ import annotations

import random
import unittest

from seagi.world.mortal_world import MortalWorld


class TestMortalWorld(unittest.TestCase):

    def test_random_policy_dies_frequently(self):
        """A mortal world must actually KILL — often — under a random policy."""
        w = MortalWorld(grid=5, seed=1)
        rng = random.Random(42)
        deaths = sum(1 for _ in range(2000)
                     if w.step(rng.randrange(w.n_actions))['dead'])
        self.assertGreater(deaths, 20, "world should kill frequently")

    def test_camper_starves(self):
        """A non-navigating camper (2-cell oscillation) must die — the energy
        clock + relocating food make sitting-still lethal by world law."""
        w = MortalWorld(grid=6, seed=3)
        died = False
        for i in range(500):
            if w.step(3 if i % 2 == 0 else 2)['dead']:
                died = True
                self.assertEqual(w.last_cause, 'starve')
                break
        self.assertTrue(died, "a camper must starve")

    def test_energy_clock_starves_at_budget(self):
        """With no hazards and food out of reach, death is starvation at
        exactly the energy budget — the clock is real, not decorative."""
        w = MortalWorld(grid=5, seed=2, stochastic=False)
        w._hazards = set()
        w._sustenance = (0, 2)            # off the column we will walk
        steps = 0
        for _ in range(w.energy_budget + 5):
            r = w.step(0)                 # action 0 = Up, stays in column 0
            steps += 1
            if r['dead']:
                break
        self.assertEqual(steps, w.energy_budget)
        self.assertEqual(r['cause'], 'starve')

    def test_hazard_kills(self):
        """Entering a hazard cell ends the trajectory immediately."""
        w = MortalWorld(grid=5, seed=2, stochastic=False)
        haz = next(iter(w._hazards))
        w._pos = ((haz[0] - 1) % 5, haz[1])    # one above
        r = w.step(1)                          # Down -> onto hazard
        self.assertTrue(r['dead'])
        self.assertEqual(r['cause'], 'hazard')

    def test_food_relocates_far_on_consumption(self):
        """Consuming sustenance refuels AND relocates it >= grid//2 away, so
        no fixed safe oscillation can keep refeeding (the safe-pocket fix)."""
        g = 7
        w = MortalWorld(grid=g, seed=5, stochastic=False)
        w._hazards = set()                     # clean approach path
        food = w.sustenance_pos
        w._pos = ((food[0] - 1) % g, food[1])
        r = w.step(1)                          # Down -> onto food
        self.assertTrue(r['at_sustenance'])
        self.assertEqual(w.energy, w.energy_budget)       # refueled
        self.assertNotEqual(w.sustenance_pos, food)       # moved
        self.assertGreaterEqual(
            w._toroidal_manhattan(w.sustenance_pos, food), g // 2)

    def test_old_food_pocket_cannot_refeed(self):
        """After consuming, camping on the OLD food cell still starves — the
        agent must travel to the moved target, not sit where food used to be."""
        g = 7
        w = MortalWorld(grid=g, seed=5, stochastic=False)
        w._hazards = set()
        food = w.sustenance_pos
        w._pos = ((food[0] - 1) % g, food[1])
        w.step(1)                              # consume; food moves away
        # now oscillate on the old food cell region
        died = False
        for i in range(w.energy_budget + 3):
            if w.step(0 if i % 2 == 0 else 1)['dead']:
                died = True
                break
        self.assertTrue(died, "old food pocket must not sustain life")

    def test_reproducible(self):
        """Deterministic given (seed, action sequence) — including death,
        relocation, and the stochastic branch (all seeded)."""
        acts = [random.Random(9).randrange(4) for _ in range(300)]

        def run():
            w = MortalWorld(grid=5, seed=7)
            return [(r['dead'], r['cause'], r['energy'])
                    for r in (w.step(a) for a in acts)]

        self.assertEqual(run(), run())

    def test_percept_shape_preserved(self):
        """The mortal world still hands out a raw R^k world_vector — so the
        existing WorldTransducer keeps working unchanged."""
        w = MortalWorld(grid=5, k=8, seed=1)
        r = w.step(0)
        self.assertIn('world_vector', r)
        self.assertEqual(len(r['world_vector']), 8)

    def test_death_resets_to_start_for_next_life(self):
        """After death the body resets (position to start, energy restored) so
        a new life begins — deaths accumulate across lives."""
        w = MortalWorld(grid=5, seed=2, stochastic=False)
        haz = next(iter(w._hazards))
        w._pos = ((haz[0] - 1) % 5, haz[1])
        r = w.step(1)
        self.assertTrue(r['dead'])
        self.assertEqual(w._pos, w._start)
        self.assertEqual(w.energy, w.energy_budget)
        self.assertEqual(w.steps_this_life, 0)
        self.assertEqual(w.deaths, 1)


if __name__ == '__main__':
    unittest.main()
