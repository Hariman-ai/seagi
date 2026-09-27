"""World transducer (Capability 1, Step 5) — online sparse-coder.

Turns a raw world percept vector into a STABLE discrete token, below the
concept-identity layer.  A recurring vector earns a prototype; the token
IS that prototype's id.  This is the tagging principle applied to raw
percepts — a stable, recurring pattern earns a code with no human naming
it first.  It is the one genuinely-new organ of the grounding build.

The mint threshold is DERIVED, not a typed constant: a percept whose
nearest prototype is farther than the running (mean + sigma) of OBSERVED
match distances is novel and mints a new prototype.  On the deterministic
precursor world every recurrence is an exact match (distance 0), so the
threshold is 0 and distinct cells get distinct, stable tokens.  Adaptive
tolerance for a NOISY world (seeding the match-distance statistic from a
cold start) is the Phase-2 stochastic build, not hand-tuned here.

Capability 1 ONLY emits the token (consumed by GroundingLoop as a focal);
it does NOT promote the token to a substrate concept — promotion, and its
abstraction-credit guard, is Capability 2.
"""

from __future__ import annotations

import math
import hashlib
from typing import Dict, List, Tuple

from seagi.core.substrate import WORLD_TOKEN_PREFIX, FACET_NODE_PREFIX


class WorldTransducer:
    """Online sparse-coder: raw percept vector -> stable token."""

    def __init__(self, sharpen: float = 0.0):
        self._protos: List[List[float]] = []
        self._counts: List[int] = []
        # Welford running stats over OBSERVED match distances — drives
        # the derived mint threshold.  Updated on a MATCH only, so the
        # exact-world threshold stays 0 and distinct cells never merge.
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0
        self.sharpen = float(sharpen)
        self.mints: int = 0
        # EXACT-MATCH INDEX: vector tuple -> prototype id.  The mint
        # threshold is 0 on a deterministic world, and d <= 0 means
        # IDENTICAL, so recognition is a dict lookup and the O(n)
        # distance scan is pure waste.  Rebuilt lazily after load.
        self._exact = {}

    @staticmethod
    def _dist(a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))

    def _nearest(self, v: List[float]) -> Tuple[int, float]:
        best_i, best_d = -1, float('inf')
        for i, p in enumerate(self._protos):
            d = self._dist(v, p)
            if d < best_d:
                best_d, best_i = d, i
        return best_i, best_d

    def _threshold(self) -> float:
        sigma = math.sqrt(self._m2 / self._n) if self._n > 1 else 0.0
        return self._mean + sigma

    def _observe_dist(self, d: float) -> None:
        self._n += 1
        delta = d - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (d - self._mean)

    def encode(self, vector) -> str:
        v = [float(x) for x in vector]
        if self._protos:
            # O(1) EXACT PATH.  Only when the derived threshold is 0
            # (identity) and sharpen is off (sharpen mutates protos and
            # would stale the index).  Same token, same mint decision,
            # same Welford update -- just without scanning every
            # prototype, which is quadratic over a life.
            if self.sharpen <= 0.0 and self._threshold() <= 0.0:
                if len(self._exact) != len(self._protos):
                    self._exact = {tuple(p): k
                                   for k, p in enumerate(self._protos)}
                _hit = self._exact.get(tuple(v))
                if _hit is not None:
                    self._counts[_hit] += 1
                    self._observe_dist(0.0)
                    return self._token(_hit)
                self._protos.append(v)
                self._counts.append(1)
                self._exact[tuple(v)] = len(self._protos) - 1
                self.mints += 1
                return self._token(len(self._protos) - 1)
            i, d = self._nearest(v)
            if d <= self._threshold():
                self._counts[i] += 1
                self._observe_dist(d)
                if self.sharpen > 0.0:
                    self._protos[i] = [
                        p + self.sharpen * (x - p)
                        for p, x in zip(self._protos[i], v)]
                return self._token(i)
        # novel -> mint a new prototype
        self._protos.append(v)
        self._counts.append(1)
        self.mints += 1
        return self._token(len(self._protos) - 1)

    @staticmethod
    def facets(vector) -> frozenset:
        """PARALLEL channel to encode() (facet-docking, 2026-07-21).

        Returns the percept's OWN (slot_index, value) componentry as a
        frozenset of facet keys — the many docking points that give a raw
        percept the same many-edged shape a substrate concept already has
        (the user's lock-and-key: byte-identity is not required, a MULTITUDE
        of overlapping keys is).

        PURE: no state, no threshold, no mint, no RNG.  `encode()` is left
        untouched (byte-exact sha1, threshold 0) so identity / _trans /
        _route / the live map are unaffected and the ~60%-injectivity false-
        merge collapse cannot recur by construction.  A raw (slot, value)
        pair is PRE-teleological — it reports the sensory surface without
        naming what matters — so it is a container, not a smuggled category.

        Values are canonicalised with the same %.6g precision the token hash
        uses, so a recurring discrete percept yields recurring facet keys
        while a continuous (blind-twin) percept yields all-distinct keys that
        never share a carrier — the twin manufactures no similarity.
        """
        return frozenset(
            (i, '%.6g' % float(x)) for i, x in enumerate(vector))

    @staticmethod
    def facet_node(slot: int, value: str) -> str:
        """The synthetic `_facet_{slot}_{value}` node name for a facet key."""
        return f'{FACET_NODE_PREFIX}{int(slot)}_{value}'

    def _token(self, i: int) -> str:
        # STABLE IDENTITY ("everything only once"): name a world token by a
        # hash of its prototype VECTOR, not the mint-order index.  The index
        # resets on restart, so a state would be re-named and COLLIDE with a
        # prior life's token -- inheriting its stale, word-less bubble, which
        # blocked grounding (reuse re-engages the old bubble so it never fades).
        # A content hash makes a state always itself, stable across restarts,
        # never inheriting a stranger's bubble.  (_world_s namespace preserved
        # for the abstraction/analogy guard.)
        proto = self._protos[i]
        key = ','.join('%.6g' % float(x) for x in proto).encode('utf-8')
        return f'{WORLD_TOKEN_PREFIX}{hashlib.sha1(key).hexdigest()[:12]}'

    def stats(self) -> Dict[str, object]:
        return {
            'prototypes': len(self._protos),
            'mints': self.mints,
            'match_threshold': self._threshold(),
        }
