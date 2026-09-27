"""StructuredGridWorld — Seagi's first VISUAL SENSE (Stage 1, 2026-07-09).

An input-side EYE, nothing more. A colored 2D grid navigation world whose
per-state percept (`_embed_of`) is the RAW egocentric COLOR-WINDOW around the
agent's cell: the discrete color codes of the (2r+1)x(2r+1) neighborhood, fixed
orientation, out-of-bounds = a sentinel code. Recurring local color patterns
therefore hash to the SAME token through the exact-match WorldTransducer, so the
transition-model + tags learned on one local pattern are already waiting wherever
that pattern recurs (cross-world near-transfer).

This is a SENSOR: it lives ENTIRELY in the `_embed_of` subclass hook
(planning_world.py:63) — the transducer, world_actor, and substrate are
UNTOUCHED. Adding senses, not tuning. Window radius is DERIVED (smallest r making
the window injective over reachable cells of BOTH shipped grids), not a knob. A
BLIND twin (same grid + dynamics, base random percept) is the A/B control.

Colors are the world's given alphabet (input data); the CATEGORY "this local
config helps" is left for confirmed_i tagging + form_abstractions to EARN.
"""
from __future__ import annotations

from collections import deque
from typing import Hashable, List, Tuple

from .planning_world import PlanningWorld

WALL = 0
GOAL = 9
SENTINEL = -1.0

# Shared sub-region S — a richly-varied 7x7 colored block, IDENTICAL in both
# grids, so its interior cells produce identical local windows -> shared tokens
# -> transfer. Colors 1-6. (Size chosen so several cells are window-interior at
# the injectivity radius; SN is used to derive the transfer set, not tuned.)
_SN = 7
_S = [
    [1, 2, 3, 4, 5, 6, 1],
    [2, 4, 6, 1, 3, 5, 2],
    [3, 6, 2, 5, 1, 4, 3],
    [4, 1, 5, 2, 6, 3, 4],
    [5, 3, 1, 6, 2, 1, 5],
    [6, 5, 4, 3, 1, 2, 6],
    [1, 2, 3, 4, 5, 6, 2],
]


def _blank(n: int) -> List[List[int]]:
    g = [[1] * n for _ in range(n)]
    for i in range(n):
        g[0][i] = g[n - 1][i] = g[i][0] = g[i][n - 1] = WALL
    return g


def _stamp(g, s, r0, c0):
    for i in range(len(s)):
        for j in range(len(s[0])):
            g[r0 + i][c0 + j] = s[i][j]


def _build(tag: int, s_at, start, goal, n: int = 11):
    """An n x n colored grid: border walls, interior floor given a per-grid
    varied color (deterministic, DIFFERENT between grids so non-shared cells
    don't collide), the shared block S stamped over its region, one goal cell."""
    g = _blank(n)
    for r in range(1, n - 1):
        for c in range(1, n - 1):
            g[r][c] = 1 + ((r * 5 + c * 3 + tag * 11) % 6)   # varied fill
    _stamp(g, _S, *s_at)                                     # shared region
    g[goal[0]][goal[1]] = GOAL
    return g, start, goal


# A: S top-left; B: S shifted — start/goal placed so the route crosses S.
_GRID_A, _START_A, _GOAL_A = _build(0, (1, 1), (9, 1), (1, 9))
_GRID_B, _START_B, _GOAL_B = _build(1, (3, 3), (9, 9), (1, 1))
_S_AT_A, _S_AT_B = (1, 1), (3, 3)


def _reachable(grid, start) -> List[Tuple[int, int]]:
    rows, cols = len(grid), len(grid[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL and (nr, nc) not in seen:
                seen.add((nr, nc)); q.append((nr, nc))
    return sorted(seen)


def _window(grid, r, c, rad) -> Tuple[float, ...]:
    rows, cols = len(grid), len(grid[0])
    out = []
    for dr in range(-rad, rad + 1):
        for dc in range(-rad, rad + 1):
            rr, cc = r + dr, c + dc
            out.append(float(grid[rr][cc]) if 0 <= rr < rows and 0 <= cc < cols else SENTINEL)
    return tuple(out)


def _derive_radius() -> int:
    """Smallest r making the color-window injective over reachable cells of BOTH
    shipped grids — a MEASUREMENT of the worlds, not a passed knob."""
    cellsA, cellsB = _reachable(_GRID_A, _START_A), _reachable(_GRID_B, _START_B)
    for rad in range(1, 9):
        wA = [_window(_GRID_A, r, c, rad) for (r, c) in cellsA]
        wB = [_window(_GRID_B, r, c, rad) for (r, c) in cellsB]
        if len(set(wA)) == len(cellsA) and len(set(wB)) == len(cellsB):
            return rad
    return 8


_RADIUS = _derive_radius()          # derived once, shared by both grids


class StructuredGridWorld(PlanningWorld):
    """Colored-grid navigation seen through the egocentric color-window EYE."""

    def __init__(self, grid=None, start=None, goal=None, seed: int = 7, k: int = 8):
        self._grid = grid if grid is not None else _GRID_A
        self._start = start if start is not None else _START_A
        self._goal_cell = goal if goal is not None else _GOAL_A
        self._rows, self._cols = len(self._grid), len(self._grid[0])
        self._radius = _RADIUS
        super().__init__(seed=seed, k=k)
        self.n_actions = 4

    def start_state(self): return self._start
    def goal_state(self): return self._goal_cell

    def next_state(self, s, a):
        r, c = s
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[a % 4]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self._rows and 0 <= nc < self._cols):
            return s
        if self._grid[nr][nc] == WALL:
            return s
        return (nr, nc)

    def _embed_of(self, state: Hashable) -> List[float]:
        # THE EYE: raw egocentric color-window (discrete), replacing the base
        # random per-state vector. Lives here only; brain untouched.
        v = self._embed_cache.get(state)
        if v is None:
            r, c = state
            v = list(_window(self._grid, r, c, self._radius))
            self._embed_cache[state] = v
        return v


class StructuredGridWorldBlind(StructuredGridWorld):
    """A/B CONTROL: identical grid + dynamics, but the OLD blind random percept
    (no eye). Isolates the eye's contribution (auditor must-fix #2)."""

    def _embed_of(self, state: Hashable) -> List[float]:
        return PlanningWorld._embed_of(self, state)


# convenience constructors for the ladder (A learned first, B shares S).
# Distinct seeds per grid so the BLIND control shares nothing by coordinate
# coincidence (base _embed_of seeds by (seed, state)); the EYE ignores seed
# (window-based) so A_eye/B_eye still share tokens on the identical S region.
def world_A_eye(k=8):   return StructuredGridWorld(_GRID_A, _START_A, _GOAL_A, seed=7, k=k)
def world_B_eye(k=8):   return StructuredGridWorld(_GRID_B, _START_B, _GOAL_B, seed=8, k=k)
def world_A_blind(k=8): return StructuredGridWorldBlind(_GRID_A, _START_A, _GOAL_A, seed=7, k=k)
def world_B_blind(k=8): return StructuredGridWorldBlind(_GRID_B, _START_B, _GOAL_B, seed=8, k=k)
