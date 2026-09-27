"""StructuredTerritoryWorld — an ESCALATING structured-grid world: the
transferable analog of QuestWorld's escalator, seen through the EXISTING
egocentric color-window EYE.

Why it exists (2026-07-20): QuestWorld's escalator dead-ended — mastering
level N replaced it with level N+1, a puzzle that COMPOUNDS the prior one and
requires transfer/abstraction the agent does not yet have, so he stalled (level
4, 0/32). This world is the OTHER kind of endless top: each level appends ONE
room to a colored corridor; rooms 0..N-1 are byte-identical to every higher
level (same walls, same gates, same per-coordinate colors, same absolute
positions), so their egocentric color-windows RECUR -> the WorldTransducer mints
the SAME tokens -> the learned transitions/tags/route over those rooms are
already waiting at the next level (transfer by shared surface, the proven law).
Only room N is fresh territory (the frontier room where the agent STARTS); the
goal stays PINNED at room 0 so the learned route home transfers intact. So
difficulty grows by DERIVED STRUCTURE (one room per mastery, grid width follows)
— never a tuned knob — and every prior level's learning is the literal prefix of
the next: compounding by TRANSFER, not by an abstraction he lacks. He climbs it.

This is a SENSOR + a WORLD, nothing more. It lives in the world classes only:
the percept is the RAW (2r+1)^2 egocentric color-window (`_embed_of`), radius
DERIVED for window-injectivity over a fixed design set of levels (a MEASUREMENT,
not a passed knob) and held CONSTANT across levels so tokens stay comparable.
The transducer / world_actor / substrate / chemistry / clock are UNTOUCHED —
adding territory, not tuning the brain. A BLIND twin (same grid + dynamics +
moving goal + growth, base random percept, seed bumped every level so nothing
recurs by coordinate coincidence) is the A/B control that isolates the eye's
contribution — the same discipline as structured_grid_world's four auditor
must-fixes (raw window / no rotation-reflection canonicalization; real
injectivity; a blind control twin; abstraction FORMATION credits zero lifeforce,
an existing substrate guard this file never touches).

Colors are the world's given alphabet (input data); the CATEGORY "this local
config is a passage / helps" is left for confirmed_i tagging + form_abstractions
to EARN. No goal color is painted: the goal is identified only by the success
signal on its (naturally unique) window token; the goal never moves, so the
shared prefix transfers cleanly and the new room is added at the start end.
"""
from __future__ import annotations

import hashlib
from typing import Hashable, List, Tuple

from .planning_world import PlanningWorld
from .structured_grid_world import _window, _reachable, WALL

# --- structural unit sizes: the repeating BUILDING BLOCK (like _SN=7 in
# structured_grid_world or grid=7 in quest_world) — NOT difficulty knobs.
# Difficulty = number of rooms = the level, which the escalator DERIVES (one
# room appended per mastery); the width of the world follows from it. ---
_ROOM_H = 5          # interior height of every room (rows)
_ROOM_W = 5          # interior width of every room (cols)
_NCOLORS = 6         # the given color alphabet (1.._NCOLORS); 0 = WALL

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _cell_color(r: int, c: int) -> int:
    """Deterministic floor color at ABSOLUTE (r, c).  Depends ONLY on the
    coordinate, so a cell keeps its color at EVERY level (shared region ->
    identical windows -> recurring tokens -> transfer), while distinct
    coordinates get de-correlated colors (drives window injectivity)."""
    h = hashlib.sha1(f'ter|{r}|{c}'.encode()).digest()
    return 1 + (h[0] % _NCOLORS)


def _gate_row(i: int) -> int:
    """Interior row (1.._ROOM_H) of the single opening in the wall between room
    i and room i+1 — a per-room, LEVEL-INDEPENDENT constant, so the gate is in
    the same place at every level (the navigation skill for room i transfers).

    NOTE the fixed-goal / moving-start geometry (see _build_territory): the GOAL
    is pinned at room 0 for ALL levels and the fresh room is appended at the far
    (start) end, so the learned route-to-goal over the shared prefix stays
    exactly valid at the next level.  An earlier design moved the goal INTO the
    new room; the warm actor's persistent route then pointed at the now-stale
    old-goal cell and it looped there instead of exploring — transfer HURT
    (warm ~80x slower than cold).  Pinning the goal removes that wrong-attractor
    while keeping the shared-prefix windows recurring."""
    h = hashlib.sha1(f'gate|{i}'.encode()).digest()
    return 1 + (h[0] % _ROOM_H)


def _col0(i: int) -> int:
    """Left interior column of room i (level-independent absolute position)."""
    return 1 + i * (_ROOM_W + 1)


def _build_territory(level: int):
    """An (_ROOM_H+2) x (level*(_ROOM_W+1)+1) colored corridor of `level` rooms.
    Border walls; each room = open colored floor; a wall column with ONE gate
    between adjacent rooms.  Rooms 0..level-2 are IDENTICAL to any higher level
    (shared prefix); room level-1 is the fresh FRONTIER room where the agent
    STARTS; the goal is PINNED at room 0's far cell for every level (natural
    token, not painted).  Fixing the goal keeps the learned route-to-goal over
    the shared prefix valid at the next level; the new room is added at the
    start end, so only it needs exploring while the known route carries the
    agent home (see the wrong-attractor note in _gate_row)."""
    H = _ROOM_H
    rows = H + 2
    cols = _col0(level - 1) + _ROOM_W + 1          # last room interior + border
    mid = 1 + H // 2

    grid = [[WALL] * cols for _ in range(rows)]
    # colored floor over the whole interior span
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            grid[r][c] = _cell_color(r, c)
    # wall column with a single gate between room i and room i+1
    for i in range(level - 1):
        wc = _col0(i) + _ROOM_W                     # the dividing column
        for r in range(1, rows - 1):
            grid[r][wc] = WALL
        gr = _gate_row(i)
        grid[gr][wc] = _cell_color(gr, wc)          # carve the gate (floor)

    goal = (mid, _col0(0))                           # room 0 far-left: PINNED
    start = (mid, _col0(level - 1) + _ROOM_W - 1)    # newest room: the frontier
    return grid, start, goal


def _derive_radius() -> int:
    """Smallest window radius making the color-window INJECTIVE over reachable
    cells of every level in a fixed DESIGN set, taken as the max across the set
    (injectivity is monotonic in radius, so that max is injective for all of
    them).  A MEASUREMENT of the worlds — like structured_grid_world deriving
    from its two shipped grids — held constant across ALL levels so shared-room
    windows keep hashing to the same token.  Verified to hold well beyond the
    design set in the smoke harness."""
    best = 1
    for L in (1, 2, 3, 6):
        grid, start, _goal = _build_territory(L)
        cells = _reachable(grid, start)
        for rad in range(1, 9):
            ws = [_window(grid, r, c, rad) for (r, c) in cells]
            if len(set(ws)) == len(cells):
                best = max(best, rad)
                break
        else:
            best = max(best, 8)
    return best


_RADIUS = _derive_radius()          # derived once, shared by every level


class StructuredTerritoryWorld(PlanningWorld):
    """Colored-corridor navigation seen through the egocentric color-window eye;
    escalates by appending one shared+one-fresh room per mastery."""

    def __init__(self, level: int = 1, seed: int = 202, k: int = 8):
        self.level = max(1, int(level))
        grid, start, goal = _build_territory(self.level)
        self._grid = grid
        self._start = start
        self._goal_cell = goal
        self._rows, self._cols = len(grid), len(grid[0])
        self._radius = _RADIUS
        super().__init__(seed=seed, k=k)
        self.n_actions = 4
        # Expose the level as the curriculum's top_difficulty telemetry
        # (CurriculumWorld.task_stats reads .disks then .size).
        self.size = self.level
        self.goal_concept = None

    def start_state(self) -> Hashable:
        return self._start

    def goal_state(self) -> Hashable:
        return self._goal_cell

    def next_state(self, s, a):
        r, c = s
        dr, dc = _DIRS[a % 4]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self._rows and 0 <= nc < self._cols):
            return s
        if self._grid[nr][nc] == WALL:
            return s
        return (nr, nc)

    def _default_budget(self) -> int:
        # DERIVED from the level's OWN graph (a measurement, not a difficulty
        # knob): a bounded exploration allowance proportional to the reachable
        # territory, so it scales automatically as the world grows.  Each
        # reachable cell may be visited a small bounded number of times before
        # the goal is found; the learned map persists across episodes, so this
        # is an efficiency allowance, never a gate on eventual solvability.
        return 6 * self.reachable_states()

    def _embed_of(self, state: Hashable) -> List[float]:
        # THE EYE: raw egocentric color-window (discrete), replacing the base
        # random per-state vector.  Lives here only; brain untouched.
        v = self._embed_cache.get(state)
        if v is None:
            r, c = state
            v = list(_window(self._grid, r, c, self._radius))
            self._embed_cache[state] = v
        return v


class StructuredTerritoryWorldBlind(StructuredTerritoryWorld):
    """A/B CONTROL: identical grid + dynamics + moving goal + growth, but the
    OLD blind random percept (no eye).  Its escalator BUMPS the seed each level
    so no token recurs across levels by coordinate coincidence -> zero transfer,
    isolating the eye's contribution (auditor must-fix #2 / the sibling
    world_A_blind/B_blind pattern)."""

    def _embed_of(self, state: Hashable) -> List[float]:
        return PlanningWorld._embed_of(self, state)


def territory_escalator(prev: StructuredTerritoryWorld) -> StructuredTerritoryWorld:
    """Top rung mastered -> one MORE room appended (level+1).  Rooms 0..level-1
    are the literal shared prefix (identical windows -> shared tokens ->
    transfer, route-to-goal still valid); room `level` is the fresh frontier
    where the agent now starts, the goal staying pinned at room 0.  Endless,
    with bounded token growth (~one room's cells per level), each mastery
    crediting record_learning -> the age wall recedes."""
    w = StructuredTerritoryWorld(level=prev.level + 1, seed=prev._seed, k=prev.k)
    w.goal_concept = getattr(prev, 'goal_concept', None)
    return w


def territory_blind_escalator(prev: StructuredTerritoryWorldBlind) -> StructuredTerritoryWorldBlind:
    """Control escalator: SAME growth, but BUMP the seed so the base random
    percepts do NOT recur across levels (no coordinate-coincidence transfer) ->
    the honest no-eye baseline for the A/B."""
    w = StructuredTerritoryWorldBlind(level=prev.level + 1,
                                      seed=prev._seed + 1, k=prev.k)
    w.goal_concept = getattr(prev, 'goal_concept', None)
    return w


# convenience constructors for the ladder.
def territory_eye(level: int = 1, k: int = 8) -> StructuredTerritoryWorld:
    return StructuredTerritoryWorld(level=level, seed=202, k=k)


def territory_blind(level: int = 1, k: int = 8) -> StructuredTerritoryWorldBlind:
    return StructuredTerritoryWorldBlind(level=level, seed=303, k=k)
