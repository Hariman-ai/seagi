"""WorldDriver — the firsthand micro-world (Capability 1, Step 5).

Proves the world is firsthand (raw vectors), action->consequence,
deterministic-and-reproducible, and that it carries HIDDEN structure the
percept does not expose (the phase flip) — the non-triviality the
grounding proof rests on.
"""

from __future__ import annotations

import unittest

from seagi.world.world_driver import WorldDriver


class TestWorldDriver(unittest.TestCase):

    def test_percept_is_raw_vector_of_k(self):
        w = WorldDriver(grid=5, k=8, seed=1)
        p = w.percept()
        self.assertIn('world_vector', p)
        self.assertEqual(len(p['world_vector']), 8)
        # firsthand: no token / symbol handed to the agent
        self.assertNotIn('token', p)
        self.assertNotIn('concept', p)

    def test_deterministic_given_seed_and_actions(self):
        a = WorldDriver(seed=7)
        b = WorldDriver(seed=7)
        acts = [0, 1, 2, 3, 1, 0, 3, 2, 1, 1]
        sa = [a.step(x)['world_vector'] for x in acts]
        sb = [b.step(x)['world_vector'] for x in acts]
        self.assertEqual(sa, sb)

    def test_action_has_consequence(self):
        # From the same start, different actions lead to different
        # percepts (the world responds to what the agent does).
        diff = 0
        for act in range(4):
            w = WorldDriver(seed=3)
            stay = WorldDriver(seed=3)
            moved = w.step(act)['world_vector']
            start = stay.percept()['world_vector']
            if moved != start:
                diff += 1
        self.assertGreaterEqual(diff, 3)   # at least 3/4 actions move us

    def test_distinct_cells_distinct_percepts(self):
        # The grid is faithfully encoded: every cell has its own vector,
        # so the transducer can recover a stable token per cell.
        w = WorldDriver(grid=5, k=8, seed=2)
        vecs = [tuple(v) for v in w._embed.values()]
        self.assertEqual(len(vecs), 25)
        self.assertEqual(len(set(vecs)), 25)        # all distinct

    def test_sensitive_cells_stochastic_others_deterministic(self):
        # The Cap-1.5 non-triviality: even-parity (phase-sensitive) cells
        # branch STOCHASTICALLY — the same cell+action yields more than
        # one outcome over repeats — while odd-parity cells are fully
        # deterministic.  That heterogeneity is the precision spread.
        w = WorldDriver(grid=5, k=8, seed=5)
        even = set()
        for _ in range(40):
            w._pos = (2, 2)                  # even parity -> sensitive
            even.add(tuple(w.step(1)['world_vector']))
        self.assertGreater(len(even), 1)     # genuinely uncertain
        odd = set()
        for _ in range(40):
            w._pos = (2, 1)                  # odd parity -> deterministic
            odd.add(tuple(w.step(1)['world_vector']))
        self.assertEqual(len(odd), 1)        # always the same

    def test_deterministic_mode_is_fully_predictable(self):
        # stochastic=False -> even cells are deterministic too (the
        # fully-learnable world the recovery test uses).
        w = WorldDriver(grid=5, k=8, seed=5, stochastic=False)
        outs = set()
        for _ in range(20):
            w._pos = (2, 2)
            outs.add(tuple(w.step(1)['world_vector']))
        self.assertEqual(len(outs), 1)

    def test_reset_law_changes_dynamics_not_percepts(self):
        w = WorldDriver(grid=5, k=8, seed=4)
        before_embed = {kc: tuple(v) for kc, v in w._embed.items()}
        # same start, same action sequence, before vs after a law change
        w._pos = (0, 0)
        traj_a = [tuple(w.step(x)['world_vector']) for x in [3, 3, 1, 1]]
        w.reset_law(seed=99)
        w._pos = (0, 0)
        traj_b = [tuple(w.step(x)['world_vector']) for x in [3, 3, 1, 1]]
        # dynamics changed -> a different trajectory ...
        self.assertNotEqual(traj_a, traj_b)
        # ... but the percepts (cell embeddings) are unchanged
        after_embed = {kc: tuple(v) for kc, v in w._embed.items()}
        self.assertEqual(before_embed, after_embed)


if __name__ == '__main__':
    unittest.main()
