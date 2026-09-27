"""MazeWorld — a much harder navigation problem: a real maze with WALLS.

The open grid the agent first learned to solve (GoalWorld) has no structure:
every cell connects to its neighbours, so any wander reaches the goal and the
optimal route is a straight diagonal.  A MAZE removes that: most moves hit a
wall (a no-op), there is exactly ONE long winding corridor from start to goal,
and the optimal path is many times longer than the grid's diagonal.  Solving it
demands the agent (a) EXPLORE far enough through dead-ends to first stumble on
the goal, and (b) establish a route that winds the whole corridor back to the
start — a deep value-spread the open grid never exercised.

The maze is a perfect maze (a random spanning tree, so the solution is unique
and every cell is reachable).  The agent sees only opaque per-cell percepts and
success/timeout — it is told nothing of walls, corridors, or the goal.
"""

from __future__ import annotations

import random
from typing import FrozenSet, Set, Tuple

from .planning_world import PlanningWorld


Cell = Tuple[int, int]
_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))     # U / D / L / R


class MazeWorld(PlanningWorld):
    """A perfect maze on a size x size grid; reach the far corner."""

    def __init__(self, size: int = 12, seed: int = 7, k: int = 8):
        self.size = int(size)
        self._goal: Cell = (self.size - 1, self.size - 1)
        self._open: Set[FrozenSet[Cell]] = self._carve(seed)
        super().__init__(seed=seed, k=k)
        self.n_actions = 4

    def _carve(self, seed: int) -> Set[FrozenSet[Cell]]:
        """Randomised-DFS perfect maze: a spanning tree of open passages."""
        rng = random.Random(seed * 131 + 7)
        size = self.size
        open_edges: Set[FrozenSet[Cell]] = set()
        visited: Set[Cell] = {(0, 0)}
        stack = [(0, 0)]
        while stack:
            r, c = stack[-1]
            nbrs = [(r + dr, c + dc) for dr, dc in _MOVES
                    if 0 <= r + dr < size and 0 <= c + dc < size
                    and (r + dr, c + dc) not in visited]
            if not nbrs:
                stack.pop()
                continue
            nxt = rng.choice(nbrs)
            open_edges.add(frozenset({(r, c), nxt}))
            visited.add(nxt)
            stack.append(nxt)
        return open_edges

    # ---- the PROBLEM ----
    def start_state(self) -> Cell:
        return (0, 0)

    def goal_state(self) -> Cell:
        return self._goal

    def next_state(self, state: Cell, action: int) -> Cell:
        dr, dc = _MOVES[action % 4]
        r, c = state
        nr, nc = r + dr, c + dc
        if (0 <= nr < self.size and 0 <= nc < self.size
                and frozenset({(r, c), (nr, nc)}) in self._open):
            return (nr, nc)
        return state                              # wall -> no-op

    def _default_budget(self) -> int:
        # Generous: enough to wind the whole corridor (and then some) in one
        # attempt once the route exists.  Derived from the world's scale.
        return 4 * self.size * self.size
