"""In-process micro-world for grounding (Capabilities 1 & 1.5, Step 5).

A grid automaton — the firsthand world the GroundingLoop predicts against,
replacing the degenerate secondhand text-attention stream.  Commitments:

  * FIRSTHAND: each percept is a raw vector in R^k (a fixed per-cell
    embedding), NOT a concept token.  The agent is handed no symbol.
  * ACTION -> CONSEQUENCE: the next state is determined by the agent's
    own action.
  * NON-DERIVABLE FROM TEXT: the grid dynamics and the per-cell
    embeddings are arbitrary numeric structure present in no corpus.
  * REPRODUCIBLE: deterministic given (seed, action sequence) — including
    the stochastic branch, which is driven by a seeded RNG so tests are
    repeatable while remaining unpredictable to a phase-blind predictor.

Cap-1.5 NON-TRIVIALITY (the precision-spread signal): a DETERMINISTIC
world is fully predictable — a perfect predictor has uniform reliability
and NO spread (empirically confirmed: a periodic hidden phase merely
phase-locks with the agent's cycle and stays ~98% predictable).  A real
precision spread needs IRREDUCIBLE (aleatoric) uncertainty.  So PHASE-
SENSITIVE cells — a world-intrinsic even-parity partition, ~half the
cells — have a STOCHASTIC transition (the move inverts on a fair branch).
Transitions FROM sensitive cells earn LOW reliability (genuinely
uncertain); transitions from the deterministic (odd-parity) cells earn
HIGH reliability.  That heterogeneity IS the sustained precision spread.
`stochastic=False` gives a fully deterministic world (the recovery test).

`grid`, `k`, `n_actions` are WORLD-SPEC dimensions (you are defining a
world), NOT behavioural knobs of the agent — the agent never sees them.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple


class WorldDriver:
    """A toroidal g x g grid automaton.  The agent occupies a cell; an
    action moves it; the percept is the cell's fixed R^k embedding.
    Even-parity cells branch stochastically (the irreducible uncertainty
    that produces a precision spread); odd-parity cells are deterministic."""

    def __init__(self, grid: int = 5, k: int = 8, n_actions: int = 4,
                 seed: int = 1, stochastic: bool = True):
        self.grid = int(grid)
        self.k = int(k)
        self.n_actions = int(n_actions)
        self.stochastic = bool(stochastic)
        rng = random.Random(seed)
        # Fixed, distinct per-cell embeddings = the firsthand percepts.
        self._embed: Dict[Tuple[int, int], List[float]] = {}
        for r in range(self.grid):
            for c in range(self.grid):
                self._embed[(r, c)] = [rng.uniform(-1.0, 1.0)
                                       for _ in range(self.k)]
        # action -> grid move (toroidal); first four are U/D/L/R.
        base = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self._moves: List[Tuple[int, int]] = base[:self.n_actions]
        self._pos: Tuple[int, int] = (0, 0)
        # Separate seeded RNG for the stochastic branch: reproducible
        # across runs, but uncorrelated with the observable state, so it
        # is unpredictable to a phase-blind predictor.
        self._dyn_rng = random.Random(seed * 7919 + 1)

    # ---- the hidden law can be changed (recovery test) ----
    def reset_law(self, seed: int) -> None:
        """Change the hidden dynamics: permute the action->move map and
        re-seed the stochastic branch.  Learned transitions break and
        must be re-earned by prediction — what the recovery test triggers.
        Percepts (cell embeddings) are unchanged; only the dynamics move."""
        rng = random.Random(seed)
        moves = list(self._moves)
        rng.shuffle(moves)
        self._moves = moves
        self._dyn_rng = random.Random(seed * 7919 + 3)

    # ---- observation ----
    def percept(self) -> Dict[str, object]:
        """The current raw percept (no action taken)."""
        return {'world_vector': list(self._embed[self._pos])}

    # ---- action -> consequence ----
    def step(self, action: int) -> Dict[str, object]:
        a = int(action) % self.n_actions
        dr, dc = self._moves[a]
        r0, c0 = self._pos
        # Phase-sensitive (even-parity) cells branch stochastically: the
        # move inverts on a fair branch -> irreducible uncertainty a
        # deterministic world cannot provide.  Odd cells are deterministic.
        if (self.stochastic and (r0 + c0) % 2 == 0
                and self._dyn_rng.random() < 0.5):
            dr, dc = -dr, -dc
        r = (r0 + dr) % self.grid
        c = (c0 + dc) % self.grid
        self._pos = (r, c)
        return {'world_vector': list(self._embed[self._pos])}

    # ---- introspection (world-spec, not agent state) ----
    def n_cells(self) -> int:
        return self.grid * self.grid
