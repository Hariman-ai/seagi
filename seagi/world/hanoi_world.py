"""HanoiWorld — Towers of Hanoi: a genuine multi-step PLANNING puzzle.

Navigation (grid, maze) only ever asks "find a path through space."  Hanoi is a
different, harder kind of problem: a planning puzzle whose state is a
CONFIGURATION (which peg each disk sits on), whose legal moves depend on that
configuration (you may only place a smaller disk on a larger one), and whose
shortest solution is EXPONENTIAL in the number of disks (2^n - 1 moves).  Most
move sequences lead away from the goal; there is no spatial intuition to exploit
— only the structure of the problem, which the agent must discover by acting.

The agent is told nothing.  It sees an opaque percept per configuration and,
through its body, whether an attempt reached the goal (all disks stacked on the
target peg).  Curiosity maps the configuration graph; the success NT-lean spreads
back across the whole 2^n-1-move solution to establish a followable route.  No
RL, no reward shaping, no knowledge of the rules — solving Hanoi from scratch
with nothing but the architecture.
"""

from __future__ import annotations

from typing import Tuple

from .planning_world import PlanningWorld


# The 6 directed peg-pair moves (from, to).  At any state most are illegal.
_PEG_MOVES = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))

State = Tuple[int, ...]     # state[d] = peg (0,1,2) that disk d sits on; disk 0
#                              is the SMALLEST.  Top of a peg = smallest disk on it.


class HanoiWorld(PlanningWorld):
    """Towers of Hanoi with `disks` disks; move all from peg 0 to peg 2."""

    def __init__(self, disks: int = 3, seed: int = 7, k: int = 8):
        self.disks = int(disks)
        super().__init__(seed=seed, k=k)
        self.n_actions = 6

    # ---- the PROBLEM ----
    def start_state(self) -> State:
        return (0,) * self.disks                  # all disks on peg 0

    def goal_state(self) -> State:
        return (2,) * self.disks                  # all disks on peg 2

    def next_state(self, state: State, action: int) -> State:
        frm, to = _PEG_MOVES[action % 6]
        # Top of a peg = the SMALLEST disk index resting on it.
        top_frm = min((d for d in range(self.disks) if state[d] == frm),
                      default=None)
        if top_frm is None:                       # illegal: source peg empty
            return state
        top_to = min((d for d in range(self.disks) if state[d] == to),
                     default=self.disks)          # sentinel: larger than any disk
        if top_frm > top_to:                      # illegal: bigger onto smaller
            return state
        ns = list(state)
        ns[top_frm] = to
        return tuple(ns)

    def _default_budget(self) -> int:
        # Enough to follow the full 2^n-1-move solution with margin once the
        # route exists; derived from the problem's own exponential depth.
        return max(64, 16 * (2 ** self.disks))
