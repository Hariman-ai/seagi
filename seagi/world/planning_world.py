"""PlanningWorld — a firsthand world over an ABSTRACT state graph.

Generalises WorldDriver beyond a toroidal grid: a state can be ANY hashable
configuration — a maze cell, a Towers-of-Hanoi peg layout, a puzzle position.
Each distinct state earns a fixed, distinct R^k embedding (its firsthand
percept), so the WorldTransducer mints one stable token per state and the
WorldActor's route mechanism runs over the TRUE state graph.  The contract is
identical to GoalWorld: opaque percept in, {success, timed_out, steps} out.
The agent is told NOTHING about the task's structure — no symbol, no rules, no
goal description; it must learn the whole map by acting (the harder the world,
the more there is to learn, and the longer the route that success must
establish).

A subclass defines ONLY the problem — start_state, goal_state, the legal
action->next-state map (an illegal move is a no-op), and n_actions.  Everything
else (firsthand percepts, episode/reset bookkeeping, the step budget) is shared
and problem-blind, so a new hard problem is just a new subclass, never new
agent machinery.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, Hashable, List, Optional


class PlanningWorld:
    """Abstract firsthand world over a hashable state graph."""

    def __init__(self, seed: int = 1, k: int = 8):
        self.k = int(k)
        self._seed = seed
        self._embed_cache: Dict[Hashable, List[float]] = {}
        self.n_actions: int = 4                      # subclass overrides
        self._state: Hashable = self.start_state()
        self.step_budget: int = self._default_budget()
        # --- task bookkeeping (mirrors GoalWorld) ---
        self.episodes: int = 0
        self.successes: int = 0
        self.goal_changes: int = 0
        self.steps_this_episode: int = 0
        self.last_steps: int = 0
        self.last_success: bool = False

    # ---- subclass hooks: the PROBLEM (the only thing that changes) ----
    def start_state(self) -> Hashable:
        raise NotImplementedError

    def goal_state(self) -> Hashable:
        raise NotImplementedError

    def next_state(self, state: Hashable, action: int) -> Hashable:
        """The state `action` leads to from `state`.  An ILLEGAL move is a
        no-op (returns `state` unchanged) — the agent learns walls/illegal
        moves as self-transitions, exactly as it learns real moves."""
        raise NotImplementedError

    def _default_budget(self) -> int:
        return 256

    # ---- firsthand per-state embedding (stable, distinct, opaque) ----
    def _embed_of(self, state: Hashable) -> List[float]:
        v = self._embed_cache.get(state)
        if v is None:
            # Deterministic across runs (Random seeds a str via sha512) and
            # distinct per state -> the transducer mints one stable token each.
            rng = random.Random(f'{self._seed}|{state!r}')
            v = [rng.uniform(-1.0, 1.0) for _ in range(self.k)]
            self._embed_cache[state] = v
        return v

    def percept(self) -> Dict[str, object]:
        return {'world_vector': list(self._embed_of(self._state))}

    def step(self, action: int) -> Dict[str, object]:
        reached = self.next_state(self._state, int(action))
        self._state = reached
        self.steps_this_episode += 1
        success = (reached == self.goal_state())
        timed_out = (not success
                     and self.steps_this_episode >= self.step_budget)
        # The percept returned is the state ACTUALLY REACHED (so the route
        # anchors on the true goal token); the reset only affects next percept.
        reached_vec = list(self._embed_of(reached))
        if success or timed_out:
            self.episodes += 1
            self.last_steps = self.steps_this_episode
            self.last_success = success
            if success:
                self.successes += 1
            self._state = self.start_state()
            self.steps_this_episode = 0
        return {
            'world_vector': reached_vec,
            'success': success,
            'timed_out': timed_out,
            'goal_changed': False,
            'steps': self.last_steps if (success or timed_out)
            else self.steps_this_episode,
        }

    # ---- introspection (world-spec, not agent state) ----
    def optimal_steps(self) -> Optional[int]:
        """Shortest solution from start to goal over the TRUE graph (BFS).
        For reporting only — the agent never sees it."""
        start, goal = self.start_state(), self.goal_state()
        seen = {start}
        q: deque = deque([(start, 0)])
        while q:
            s, d = q.popleft()
            if s == goal:
                return d
            for a in range(self.n_actions):
                ns = self.next_state(s, a)
                if ns not in seen:
                    seen.add(ns)
                    q.append((ns, d + 1))
        return None

    def reachable_states(self) -> int:
        start = self.start_state()
        seen = {start}
        q: deque = deque([start])
        while q:
            s = q.popleft()
            for a in range(self.n_actions):
                ns = self.next_state(s, a)
                if ns not in seen:
                    seen.add(ns)
                    q.append(ns)
        return len(seen)

    def task_stats(self) -> Dict[str, object]:
        return {
            'episodes': self.episodes,
            'successes': self.successes,
            'success_rate': (self.successes / self.episodes
                             if self.episodes else 0.0),
            'goal_changes': self.goal_changes,
            'last_success': self.last_success,
            'step_budget': self.step_budget,
        }
