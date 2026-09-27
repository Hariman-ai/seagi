"""QuestWorld — an UNBOUNDED compounding ladder where every learned step is the
literal basis for the next.

Design (from the empirically-proven transfer law: transfer rides shared surface
tokens):
  - ONE stable courtyard (fixed walls, fixed seed) across ALL levels -> the same
    (cell, phase) state always mints the same token, forever.
  - Level N: collect N keys IN ORDER (each key is a navigation leg on the same
    map), then enter an (N+1)-digit combo at the door.
  - Key positions and combo digits are PREFIXES of fixed streams: level N+1 =
    level N's ENTIRE solution + one new key leg + one new combo digit.
    So everything he learned at level N is directly reused at N+1 — knowledge
    compounds by construction, difficulty grows without bound.
  - State space grows LINEARLY per level (~grid^2 new tokens per level), the
    deliberate fix for why the old Hanoi-depth escalator (3^n states) was capped.

The escalator (quest_escalator) makes the top rung endless: mastering level N
replaces it with level N+1. Each mastery credits record_learning -> the age wall
recedes -> he out-learns death BY learning, exactly the mortality design.

Observation targets (measure from HIM, no parallel actor):
  - escalations count over time = the learning curve (does compounding keep
    ticks-to-master flat as levels grow? that IS knowledge becoming exponential
    relative to difficulty);
  - horizon (does he out-earn the wall);
  - creativity signals: self-set goals / novel recombinations over quest tokens.
"""
from __future__ import annotations

import random
from typing import Tuple

from .planning_world import PlanningWorld

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

State = Tuple[Tuple[int, int], int, int]   # (cell, keys_collected, combo_pos)


class QuestWorld(PlanningWorld):
    """Level-N quest on a fixed walled courtyard: N ordered keys, then a combo."""

    def __init__(self, level: int = 1, grid: int = 7, seed: int = 101, k: int = 8):
        self.level = int(level)
        self.grid = int(grid)
        self.world_seed = int(seed)
        rng = random.Random(f'quest|{seed}')
        self.start_cell = (0, 0)
        self.door_cell = (grid - 1, grid - 1)
        cells = [(r, c) for r in range(grid) for c in range(grid)
                 if (r, c) not in (self.start_cell, self.door_cell)]
        rng.shuffle(cells)
        n_walls = (grid * grid) // 8
        self.walls = frozenset(cells[:n_walls])
        # Stable key stream: the SAME cells at every level; level N uses the
        # first N -> each new level appends ONE new leg after the old ones.
        open_cells = cells[n_walls:]
        self.key_cells = list(open_cells[:min(self.level, len(open_cells))])
        # Stable combo stream: level N uses the first N+1 digits -> the learned
        # combo is always a PREFIX of the next level's combo.
        digits = [rng.randrange(4) for _ in range(4096)]
        self.combo = tuple(digits[: 1 + self.level])
        super().__init__(seed=seed, k=k)
        self.n_actions = 4
        # Expose the level as the curriculum's top_difficulty telemetry
        # (CurriculumWorld.task_stats reads .disks then .size).
        self.size = self.level

    # ---- the PROBLEM ----
    def start_state(self) -> State:
        return (self.start_cell, 0, 0)

    def goal_state(self) -> State:
        return (self.door_cell, len(self.key_cells), len(self.combo))

    def next_state(self, s: State, action: int) -> State:
        cell, kg, cp = s
        n = len(self.key_cells)
        a = action % 4
        if cell == self.door_cell and kg == n:
            # At the door with all keys: actions are combo digits.  The right
            # digit advances; a wrong one is a no-op (as in the lock he knows).
            if cp < len(self.combo) and a == self.combo[cp]:
                return (cell, kg, cp + 1)
            return s
        dr, dc = _DIRS[a]
        nr, nc = cell[0] + dr, cell[1] + dc
        if not (0 <= nr < self.grid and 0 <= nc < self.grid):
            return s
        if (nr, nc) in self.walls:
            return s
        ncell = (nr, nc)
        if kg < n and ncell == self.key_cells[kg]:
            return (ncell, kg + 1, 0)
        return (ncell, kg, cp)

    def _default_budget(self) -> int:
        # Legs grow linearly with level; generous linear budget.
        return max(256, 96 * (self.level + 2))


def quest_escalator(prev: QuestWorld) -> QuestWorld:
    """Top rung mastered -> the next level of the SAME quest (same map, same
    streams): everything learned so far is the literal prefix of the new
    problem.  Endless -> he can keep learning; each mastery earns life."""
    w = QuestWorld(level=prev.level + 1, grid=prev.grid,
                   seed=prev.world_seed, k=prev.k)
    w.goal_concept = getattr(prev, 'goal_concept', None)
    return w
