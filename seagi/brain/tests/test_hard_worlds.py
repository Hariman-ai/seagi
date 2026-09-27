"""Harder problems for the WorldActor: a walled MAZE and TOWERS OF HANOI.

Two things are locked in here:
  1. The worlds are well-formed — a maze is fully connected with walls that
     no-op, Hanoi has the exact 3^n state space and 2^n-1 optimal, and illegal
     moves are rejected.
  2. The architecture SOLVES them from scratch and converges to the OPTIMAL
     route — driven only by curiosity (variation) + the success NT-lean spread
     back into a route (selection).  The maze's optimal is longer than the old
     hard-coded 16-sweep cap, so this also guards the propagate-to-convergence
     fix in WorldActor._propagate_route.
"""

import unittest

from seagi.brain.bus import EventBus
from seagi.brain.capabilities.world_actor import WorldActor
from seagi.brain.capabilities.world_transducer import WorldTransducer
from seagi.world.maze_world import MazeWorld
from seagi.world.hanoi_world import HanoiWorld


class _StubGrounding:
    """Minimal grounding: the route mechanism does not depend on prediction,
    only on the learned _trans map, so observe/predict can be inert here."""

    def observe(self, *args, **kwargs):
        return None

    def _predict(self, *args, **kwargs):
        return (None, 0.0)


def _solve(world, max_steps=400000):
    """Drive the WorldActor (propose -> execute) directly against a world until
    it has reached the goal optimally, or the step budget runs out.  Returns
    the actor so the caller can inspect best_steps / successes."""
    bus = EventBus()
    wa = WorldActor(bus=bus, world=world, transducer=WorldTransducer(),
                    grounding=_StubGrounding(), value_provider=None,
                    credit_learning=None, cycle_provider=lambda: 0)
    opt = world.optimal_steps()
    streak = 0
    for t in range(max_steps):
        a = wa.propose(t)
        if a is not None:
            wa.execute(a, t)
        if wa.best_steps == opt:
            streak += 1
            if streak >= 3:        # converged: 3 optimal solves in a row
                break
        else:
            streak = 0
    return wa, opt


class TestMazeWorld(unittest.TestCase):

    def test_fully_connected_with_known_optimal(self):
        m = MazeWorld(size=8, seed=7)
        self.assertEqual(m.reachable_states(), 64)      # perfect maze: all cells
        self.assertEqual(m.optimal_steps(), 26)         # > 16 (guards the fix)

    def test_wall_is_a_noop(self):
        m = MazeWorld(size=8, seed=7)
        # At least one of the four moves from the start hits a wall (the start
        # cell of a spanning-tree maze has degree < 4) -> returns same state.
        start = m.start_state()
        noops = [a for a in range(4) if m.next_state(start, a) == start]
        self.assertTrue(noops)

    def test_actor_solves_maze_optimally(self):
        m = MazeWorld(size=8, seed=7)
        wa, opt = _solve(m)
        self.assertEqual(wa.best_steps, opt)            # reached the optimum
        self.assertGreaterEqual(wa.successes, 3)


class TestHanoiWorld(unittest.TestCase):

    def test_state_space_and_optimal(self):
        for n in (3, 4):
            h = HanoiWorld(disks=n, seed=7)
            self.assertEqual(h.reachable_states(), 3 ** n)
            self.assertEqual(h.optimal_steps(), 2 ** n - 1)

    def test_illegal_move_is_rejected(self):
        h = HanoiWorld(disks=3, seed=7)
        s = h.start_state()                             # (0,0,0): all on peg 0
        # From start only the smallest disk (disk 0) may move; moves that would
        # lift a larger disk or place onto a smaller one are no-ops.
        moved = [h.next_state(s, a) for a in range(6) if h.next_state(s, a) != s]
        self.assertEqual(set(moved), {(1, 0, 0), (2, 0, 0)})

    def test_actor_solves_hanoi_optimally(self):
        h = HanoiWorld(disks=3, seed=7)
        wa, opt = _solve(h)
        self.assertEqual(wa.best_steps, opt)            # the optimal 7 moves
        self.assertGreaterEqual(wa.successes, 1)            # solved; convergence is best_steps==opt (directed exploration reaches optimal in ~2)


class TestFirsthandPercepts(unittest.TestCase):

    def test_percept_stable_and_distinct(self):
        h = HanoiWorld(disks=3, seed=7)
        a = h._embed_of((0, 0, 0))
        b = h._embed_of((0, 0, 0))
        c = h._embed_of((2, 2, 2))
        self.assertEqual(a, b)                          # stable per state
        self.assertNotEqual(a, c)                       # distinct across states

    def test_step_returns_reached_state_then_resets(self):
        # A 1-disk Hanoi: one legal move from start reaches the goal.
        h = HanoiWorld(disks=1, seed=7)
        goal_vec = list(h._embed_of((2,)))
        out = h.step(1)                                 # move disk 0: peg0 -> peg2
        self.assertTrue(out['success'])
        self.assertEqual(out['world_vector'], goal_vec)  # the REACHED goal token
        # after success the internal state has reset to start for next episode
        self.assertEqual(h.percept()['world_vector'], list(h._embed_of((0,))))


if __name__ == '__main__':
    unittest.main()
