"""Human puzzles — classic problems from human problem-solving research, as
rungs in the ONE Seagi's ladder (he LIVES them; ability is measured from him).

  - WolfGoatCabbageWorld: the river-crossing puzzle (constraint planning).
    Humans: usually solved in a few attempts; optimal = 7 crossings.
  - JugPourWorld: Luchins water jars, 3 & 5 -> exactly 4 (subgoal arithmetic).
    Optimal = 7 pours to (0,4).
  - SlidePuzzleWorld: 2x3 five-puzzle (spatial planning; 8-puzzle's sibling).
    Start is K backward shuffles from the goal (always solvable).

Honest framing for any human comparison: humans are TOLD the rules; he must
DISCOVER them by acting (his trials include rule-learning). The comparison is a
yardstick of problem class, not a species-fair race.
"""
from __future__ import annotations

import random
from typing import Tuple

from .planning_world import PlanningWorld


class WolfGoatCabbageWorld(PlanningWorld):
    """State (farmer, wolf, goat, cabbage), each 0/1 = bank. Actions: cross
    alone / with wolf / with goat / with cabbage. A move that leaves the wolf
    with the goat, or the goat with the cabbage, unattended — or takes an item
    not on the farmer's bank — is a no-op. Goal: everyone on bank 1."""

    def __init__(self, seed: int = 7, k: int = 8):
        super().__init__(seed=seed, k=k)
        self.n_actions = 4

    def start_state(self): return (0, 0, 0, 0)
    def goal_state(self): return (1, 1, 1, 1)

    def next_state(self, s, action):
        f, w, g, c = s
        a = action % 4
        nf = 1 - f
        nw, ng, nc = w, g, c
        if a == 1:                                   # take wolf
            if w != f: return s
            nw = nf
        elif a == 2:                                 # take goat
            if g != f: return s
            ng = nf
        elif a == 3:                                 # take cabbage
            if c != f: return s
            nc = nf
        # unattended conflicts on the bank the farmer LEAVES
        if nw == ng and nw != nf: return s           # wolf eats goat
        if ng == nc and ng != nf: return s           # goat eats cabbage
        return (nf, nw, ng, nc)


class JugPourWorld(PlanningWorld):
    """Two jugs (3, 5); measure exactly 4 in the big jug, small one empty.
    Actions: fill A, fill B, empty A, empty B, pour A->B, pour B->A.
    Classic optimal = 7 to reach (0, 4)."""

    CAP_A, CAP_B = 3, 5

    def __init__(self, seed: int = 7, k: int = 8):
        super().__init__(seed=seed, k=k)
        self.n_actions = 6

    def start_state(self): return (0, 0)
    def goal_state(self): return (0, 4)

    def next_state(self, s, action):
        a_, b_ = s
        act = action % 6
        if act == 0: ns = (self.CAP_A, b_)
        elif act == 1: ns = (a_, self.CAP_B)
        elif act == 2: ns = (0, b_)
        elif act == 3: ns = (a_, 0)
        elif act == 4:
            t = min(a_, self.CAP_B - b_); ns = (a_ - t, b_ + t)
        else:
            t = min(b_, self.CAP_A - a_); ns = (a_ + t, b_ - t)
        return ns if ns != s else s


class SlidePuzzleWorld(PlanningWorld):
    """2x3 sliding five-puzzle. State = 6-tuple, 0 = blank, goal (1,2,3,4,5,0).
    Actions move the BLANK up/down/left/right (the neighboring tile slides into
    it); impossible directions are no-ops. Start = `shuffles` random legal
    backward moves from the goal (fixed seed -> deterministic, always solvable)."""

    ROWS, COLS = 2, 3

    def __init__(self, shuffles: int = 25, seed: int = 7, k: int = 8):
        self.shuffles = int(shuffles)
        rng = random.Random(f'slide|{seed}|{shuffles}')
        s = (1, 2, 3, 4, 5, 0)
        prev = None
        for _ in range(self.shuffles):
            opts = [ns for a in range(4)
                    if (ns := self._slide(s, a)) != s and ns != prev]
            if opts:
                prev, s = s, rng.choice(opts)
        self._start = s
        super().__init__(seed=seed, k=k)
        self.n_actions = 4

    def _slide(self, s, action):
        i = s.index(0)
        r, c = divmod(i, self.COLS)
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[action % 4]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.ROWS and 0 <= nc < self.COLS):
            return s
        j = nr * self.COLS + nc
        lst = list(s)
        lst[i], lst[j] = lst[j], lst[i]
        return tuple(lst)

    def start_state(self): return self._start
    def goal_state(self): return (1, 2, 3, 4, 5, 0)

    def next_state(self, s, action):
        return self._slide(s, action)
