"""Sophisticated problem definitions — harder worlds that STRESS the frontier
faculties, prepared to be added to the ONE Seagi's CurriculumWorld ladder.

ONE Seagi. These are NOT run in any parallel harness or against any fresh actor —
that would be a second path, and a throwaway actor cannot evolve anyway. They are
lived by the one live daemon's real world-loop (real transducer, real grounding,
real substrate). Evolution is measured ONLY from HIM: his success/mastery on these
rungs over time, and what his substrate does with them.

Each world subclasses PlanningWorld and targets one frontier faculty:
  - KeyDoorWorld        -> DEEP COMPOSITION (chain two coupled sub-skills, ordered)
  - StructuralLockWorld -> STRUCTURAL ABSTRACTION (same structure, varied surface)
  - AliasWorld          -> BELIEF / PARTIAL OBSERVABILITY (aliased percepts)

Integration (a deliberate, single change to the one daemon, with the user's go):
add the chosen worlds to the CurriculumWorld ladder in runtime.py so his world_actor
faces them in sequence. Nothing here runs on its own.
"""
from __future__ import annotations
import random
from .planning_world import PlanningWorld


class KeyDoorWorld(PlanningWorld):
    """A grid whose GOAL (the door) only counts once the KEY is collected.
    Reaching the door without the key is not the goal. Forces chaining two
    sub-skills in order: reach-key, THEN reach-door. Stresses COMPOSITION."""
    def __init__(self, size=5, seed=7, k=8):
        self.size = int(size)
        self.start_cell = (0, 0)
        self.key_cell = (size - 1, 0)
        self.door_cell = (size - 1, size - 1)
        super().__init__(seed=seed, k=k)
        self.n_actions = 4

    def start_state(self): return (self.start_cell, False)
    def goal_state(self): return (self.door_cell, True)

    def next_state(self, state, action):
        (r, c), has_key = state
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[action % 4]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return state                        # wall = no-op
        ncell = (nr, nc)
        return (ncell, has_key or (ncell == self.key_cell))


class StructuralLockWorld(PlanningWorld):
    """An N-stage lock: at each stage exactly one action advances; any other is a
    no-op. STRUCTURE (combo + topology) is set by `combo`; SURFACE (percept tokens)
    is set by `surface`. Two instances with the SAME combo but DIFFERENT surface are
    the SAME structure under DIFFERENT percepts -> whether he transfers across them
    is the STRUCTURAL-ABSTRACTION readout (currently a missing faculty; the point is
    to watch whether it ever emerges)."""
    def __init__(self, combo=(2, 0, 3, 1), surface=0, seed=7, k=8):
        self.combo = tuple(int(x) for x in combo)
        self.n_stages = len(self.combo)
        self.surface = int(surface)
        super().__init__(seed=seed, k=k)
        self.n_actions = 4

    def start_state(self): return 0
    def goal_state(self): return self.n_stages

    def next_state(self, stage, action):
        if stage >= self.n_stages:
            return stage
        if (action % self.n_actions) == self.combo[stage]:
            return stage + 1
        return stage                            # wrong action = no-op (stay)

    def _embed_of(self, state):
        v = self._embed_cache.get(state)
        if v is None:
            rng = random.Random(f'{self.surface}|{self._seed}|{state!r}')
            v = [rng.uniform(-1.0, 1.0) for _ in range(self.k)]
            self._embed_cache[state] = v
        return v


class AliasWorld(PlanningWorld):
    """A corridor of `length` cells (goal at the end) where distinct cells share a
    percept: cell c maps to token (c % aliases). If aliases < length the agent
    cannot tell some cells apart from the percept alone -> it must track hidden
    state / belief to act correctly. Stresses BELIEF / PARTIAL OBSERVABILITY."""
    def __init__(self, length=8, aliases=3, seed=7, k=8):
        self.length = int(length)
        self.aliases = int(aliases)
        super().__init__(seed=seed, k=k)
        self.n_actions = 2                      # 0=left, 1=right

    def start_state(self): return 0
    def goal_state(self): return self.length - 1

    def next_state(self, cell, action):
        if action % 2 == 1:
            return min(self.length - 1, cell + 1)
        return max(0, cell - 1)

    def _embed_of(self, cell):
        alias = cell % self.aliases
        v = self._embed_cache.get(alias)
        if v is None:
            rng = random.Random(f'{self._seed}|alias{alias}')
            v = [rng.uniform(-1.0, 1.0) for _ in range(self.k)]
            self._embed_cache[alias] = v
        return v
