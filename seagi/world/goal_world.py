"""GoalWorld — an ESCALATING task: keep reaching goals in an unknown world.

Reaching the goal = the task solved.  But a single fixed task is a dead end:
once mastered there is nothing left to learn, the agent stagnates, and (by the
architecture's own mortality logic) it dies.  So the world ESCALATES — when the
agent has mastered the current goal (reached it reliably), the goal MOVES to a
new cell.  There is always a new challenge, so life is prolonged only by
CONTINUAL learning.

The agent is told nothing — it sees only the opaque per-cell percept and,
through its body, whether an attempt SUCCEEDED and whether a new challenge has
begun.  Goal placement, the mastery bar, and the step budget are WORLD-SPEC (the
curriculum's shape), not behavioural knobs of the agent.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .world_driver import WorldDriver


class GoalWorld(WorldDriver):
    """A WorldDriver with a goal that moves once the agent masters it."""

    def __init__(self, grid: int = 6, k: int = 8, n_actions: int = 4,
                 seed: int = 1, stochastic: bool = True,
                 goal: Optional[Tuple[int, int]] = None,
                 mastery_threshold: int = 5):
        super().__init__(grid=grid, k=k, n_actions=n_actions, seed=seed,
                         stochastic=stochastic)
        self._start: Tuple[int, int] = (0, 0)
        self._goal_rng = random.Random(seed * 31 + 5)
        self._task_seed: int = int(seed)
        self.goal: Tuple[int, int] = goal or self._pick_goal((0, 0))
        # Step budget for one attempt — derived from the world's spatial scale.
        self.step_budget: int = 4 * self.n_cells()
        # Mastery = this many consecutive successes on the CURRENT goal; then
        # the challenge escalates (the goal moves).  World-spec curriculum bar.
        self.mastery_threshold: int = int(mastery_threshold)
        self._pos = self._start
        # --- task bookkeeping ---
        self.episodes: int = 0
        self.successes: int = 0
        self.goal_changes: int = 0          # challenges mastered + escalated
        self._consec_success: int = 0
        self.steps_this_episode: int = 0
        self.last_steps: int = 0
        self.last_success: bool = False

    def percept(self) -> Dict[str, object]:
        return {'world_vector': list(self._embed[self._pos])}

    def _pick_goal(self, away_from: Tuple[int, int]) -> Tuple[int, int]:
        """A non-start, deterministic (odd-parity) cell, different from the
        current goal — so reaching it is reliable, but the route still crosses
        the world's stochastic cells."""
        cur = getattr(self, 'goal', None)
        cells: List[Tuple[int, int]] = [
            (r, c) for r in range(self.grid) for c in range(self.grid)
            if (r, c) != self._start and (r, c) != cur and (r + c) % 2 == 1]
        if not cells:
            cells = [(r, c) for r in range(self.grid) for c in range(self.grid)
                     if (r, c) != self._start and (r, c) != cur]
        return self._goal_rng.choice(cells) if cells else self._start

    def _new_task(self) -> None:
        """A genuinely new task to learn FROM SCRATCH: re-randomise the hidden
        DYNAMICS (`reset_law` permutes the action->move map + re-seeds the
        stochastic branch, so the agent's learned action->consequence model
        breaks and must be re-earned by prediction) plus a new goal — WITHOUT
        minting new percepts, so the substrate stays BOUNDED (no per-task token
        bloat) while the task is genuinely new to solve."""
        self._task_seed += 1
        self.reset_law(self._task_seed)      # WorldDriver: permute moves + reseed
        self._pos = self._start
        self.goal = self._pick_goal(self._start)
        self.goal_changes += 1

    def step(self, action: int) -> Dict[str, object]:
        percept = super().step(action)          # advances self._pos
        self.steps_this_episode += 1
        success = (self._pos == self.goal)
        timed_out = (not success
                     and self.steps_this_episode >= self.step_budget)
        goal_changed = False
        if success or timed_out:
            self.episodes += 1
            self.last_steps = self.steps_this_episode
            self.last_success = success
            if success:
                self.successes += 1
                self._consec_success += 1
                if self._consec_success >= self.mastery_threshold:
                    # MASTERED — present a GENUINELY NEW task (new percepts,
                    # new dynamics, new goal) to learn FROM SCRATCH.  Continual
                    # from-scratch problem-solving, never running dry — unlike
                    # just moving the goal in an already-learned world.
                    self._new_task()
                    self._consec_success = 0
                    goal_changed = True
            else:
                self._consec_success = 0
            self._pos = self._start
            self.steps_this_episode = 0
        percept['success'] = success
        percept['timed_out'] = timed_out
        percept['goal_changed'] = goal_changed
        percept['steps'] = self.last_steps if (success or timed_out) else \
            self.steps_this_episode
        return percept

    def task_stats(self) -> Dict[str, object]:
        return {
            'episodes': self.episodes,
            'successes': self.successes,
            'success_rate': (self.successes / self.episodes
                             if self.episodes else 0.0),
            'goal_changes': self.goal_changes,       # challenges mastered
            'goal': self.goal,
            'last_success': self.last_success,
            'step_budget': self.step_budget,
        }
