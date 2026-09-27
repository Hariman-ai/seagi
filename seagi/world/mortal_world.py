"""Survival grid-world: the MORTAL extension of WorldDriver (Step 5, game test).

The toroidal `WorldDriver` is an *immortal* world — an infinite torus where
nothing can ever die.  A test of the thesis "intrinsic mortality + NT dynamics
produce intelligence that extends life" needs a world that can KILL, frequently
and for real.  `MortalWorld` adds exactly the three things a mortal world needs,
and nothing the agent could mistake for a reward:

  * DEATH (hazard cells).  A seed-fixed subset of cells are absorbing HAZARDs:
    entering one ends the trajectory.  The layout is world-law — fixed by seed,
    NEVER a function of the agent's action — so survival can only be EARNED by
    predicting and avoiding it, never farmed by steering the policy.  (That
    action-correlated death is the exact backdoor that retired the bare loop.)

  * STARVATION (an energy clock).  Energy decrements every step; at zero the
    trajectory ends.  Sitting still is therefore lethal BY WORLD LAW, not by an
    anti-camping penalty constant.

  * SUSTENANCE that MOVES (the anti-gaming world-law).  Reaching the sustenance
    cell resets energy — but on consumption the sustenance RELOCATES to a new,
    distant cell (seed-determined).  Food does not stay put, so no fixed 2-cell
    oscillation can refeed: surviving REQUIRES navigating to a moving target
    while avoiding hazards, which necessarily crosses the world's stochastic
    (even-parity) cells.  This closes the "safe-pocket" exploit the doctrine
    audit flagged — by physics, not by a knob.

`grid`, `hazard_density`, and the energy budget are WORLD-SPEC dimensions — you
are defining a world's physics (its size, how deadly it is, how far metabolism
reaches), NOT behavioural knobs of the agent.  The agent never sees them; it
only ever receives the per-cell R^k percept and, through its own body, the
consequences.  The energy budget is DERIVED from the world's own spatial scale
(its toroidal diameter), so it is not a free constant: metabolic reach scales
with the size of the world it must cross.

Reproducible: deterministic given (seed, action sequence), including death,
relocation, and the stochastic branch (all seeded) — so tests are repeatable
while the dynamics stay unpredictable to a phase-blind predictor.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .world_driver import WorldDriver


class MortalWorld(WorldDriver):
    """A WorldDriver that can kill.  Same firsthand R^k percepts and the same
    even/odd-parity stochastic dynamics, plus hazards, an energy clock, and
    relocating sustenance.  `step()` augments the percept dict with the body
    consequences (`dead`, `cause`, `energy`) that the brain feels — never a
    score."""

    def __init__(self, grid: int = 5, k: int = 8, n_actions: int = 4,
                 seed: int = 1, stochastic: bool = True,
                 hazard_density: float = 0.16):
        super().__init__(grid=grid, k=k, n_actions=n_actions, seed=seed,
                         stochastic=stochastic)
        # --- world-spec, derived where possible ---
        # Energy budget = the world's toroidal diameter, doubled: the max
        # wrap distance per axis is grid//2, two axes -> grid; doubling gives
        # enough reach to navigate to relocated food (placed >= grid//2 away)
        # with margin for hazard-avoidance detours, but never an infinite
        # larder.  Scales with world size -> not a free constant.
        self.energy_budget: int = max(4, 2 * self.grid)
        # Layout RNG kept separate from the dynamics RNG so hazard/food
        # placement is reproducible but independent of the move branch.
        self._layout_rng = random.Random(seed * 104729 + 17)
        self._start: Tuple[int, int] = (0, 0)

        # --- place hazards: a seed-fixed subset, excluding the start cell ---
        n_haz = int(round(hazard_density * self.n_cells()))
        n_haz = max(1, min(n_haz, self.n_cells() - 4))  # leave room to live
        cells = [(r, c) for r in range(self.grid) for c in range(self.grid)
                 if (r, c) != self._start]
        self._layout_rng.shuffle(cells)
        self._hazards: set = set(cells[:n_haz])

        # --- place initial sustenance: a deterministic (odd-parity), non-
        # hazard cell, far from start so the first journey is a real one ---
        self._sustenance: Tuple[int, int] = self._pick_sustenance(self._start)

        # --- mortal body state ---
        self.energy: int = self.energy_budget
        self.alive: bool = True
        self.deaths: int = 0
        self.steps_this_life: int = 0
        self.last_cause: Optional[str] = None
        self._pos = self._start

    # ---- world-law helpers (world-spec introspection, not agent state) ----
    def is_hazard(self, cell: Tuple[int, int]) -> bool:
        return cell in self._hazards

    def n_hazards(self) -> int:
        return len(self._hazards)

    @property
    def sustenance_pos(self) -> Tuple[int, int]:
        return self._sustenance

    def _toroidal_manhattan(self, a: Tuple[int, int],
                            b: Tuple[int, int]) -> int:
        dr = abs(a[0] - b[0]); dr = min(dr, self.grid - dr)
        dc = abs(a[1] - b[1]); dc = min(dc, self.grid - dc)
        return dr + dc

    def _pick_sustenance(self, away_from: Tuple[int, int]) -> Tuple[int, int]:
        """A non-hazard, odd-parity cell at least grid//2 away (toroidal) from
        `away_from`.  Far placement forces a genuine crossing of the world
        (incl. its stochastic cells) to refeed — the anti-gaming property.
        Falls back to the farthest legal cell if the distance gate is too
        strict for a small/dense world."""
        min_dist = max(1, self.grid // 2)
        candidates: List[Tuple[int, int]] = []
        for r in range(self.grid):
            for c in range(self.grid):
                cell = (r, c)
                if cell in self._hazards or cell == away_from:
                    continue
                if (r + c) % 2 == 0:   # prefer odd-parity (deterministic) food
                    continue
                if self._toroidal_manhattan(cell, away_from) >= min_dist:
                    candidates.append(cell)
        if not candidates:
            # relax the parity preference, then the distance gate, in order
            for r in range(self.grid):
                for c in range(self.grid):
                    cell = (r, c)
                    if cell in self._hazards or cell == away_from:
                        continue
                    if self._toroidal_manhattan(cell, away_from) >= min_dist:
                        candidates.append(cell)
        if not candidates:
            # last resort: the farthest non-hazard cell available
            legal = [(r, c) for r in range(self.grid)
                     for c in range(self.grid)
                     if (r, c) not in self._hazards and (r, c) != away_from]
            legal.sort(key=lambda cell: self._toroidal_manhattan(cell,
                                                                 away_from))
            return legal[-1] if legal else away_from
        return self._layout_rng.choice(candidates)

    # ---- action -> consequence (mortal) ----
    def step(self, action: int) -> Dict[str, object]:
        """Move (reusing the toroidal + stochastic dynamics), then apply the
        world's mortality.  Returns the percept of the cell entered, augmented
        with the felt body consequences.  On death the percept is the LETHAL
        cell (so the pre-death state can be tagged), and position silently
        resets to the start for the next life."""
        percept = super().step(action)        # advances self._pos
        self.energy -= 1
        self.steps_this_life += 1
        entered = self._pos
        dead = False
        ate = False
        cause: Optional[str] = None

        if entered in self._hazards:
            dead, cause = True, 'hazard'
        elif self.energy <= 0:
            dead, cause = True, 'starve'
        elif entered == self._sustenance:
            # consumed: refuel and relocate the food far away.  Capture the
            # `ate` flag BEFORE relocating, else the at_sustenance signal
            # compares against the already-moved food and never fires.
            ate = True
            self.energy = self.energy_budget
            self._sustenance = self._pick_sustenance(entered)

        if dead:
            self.deaths += 1
            self.last_cause = cause
            self.alive = False           # transient: dead for THIS percept
            # reset for the next life; the lethal percept is still returned
            self._pos = self._start
            self.energy = self.energy_budget
            self.steps_this_life = 0
            # food is reset to a fresh far cell so each life is a real journey
            self._sustenance = self._pick_sustenance(self._start)
        else:
            self.alive = True

        percept['dead'] = dead
        percept['cause'] = cause
        percept['energy'] = self.energy
        percept['at_sustenance'] = ate
        return percept

    # ---- introspection ----
    def life_stats(self) -> Dict[str, object]:
        return {
            'energy': self.energy,
            'energy_budget': self.energy_budget,
            'alive': self.alive,
            'deaths': self.deaths,
            'steps_this_life': self.steps_this_life,
            'last_cause': self.last_cause,
            'n_hazards': self.n_hazards(),
            'sustenance_pos': self._sustenance,
        }
