"""Long-Term Substrate — query API for all concepts ever encountered.

Brain analog: temporal-lobe declarative memory.  All concepts /
edges / bubbles / episodes the agent has ever encountered.

Critical contract: callers QUERY this; they NEVER iterate it.
That's how v1 became O(N²) at 14K concepts — every cognitive
primitive iterated `substrate.concepts.items()`.  In v2 this
capability owns substrate access; everyone else uses these
named queries.

Phase 2: thin wrapper over v1's Substrate.  Storage backend
(Python dict vs sqlite vs duckdb vs external graph) is decided
in Phase 4 based on actual scale.  Until then we keep the v1
Substrate; only the access pattern changes.

Query surface
-------------
    get_concept(name)              fetch a single concept
    has_concept(name)              cheap existence check
    neighbors(name, relation=None) list neighbor names
    edge(s, r, o)                  fetch a specific edge or None
    similar(name, top_k)           top-k by embedding similarity
                                    (later phase — Phase 4)
    episodes_mentioning(name)      episodes whose concept-set
                                    contains name
    bubbles_for(name)              all v1 bubbles for a concept
    best_bubble(name)              highest-encounter v1 bubble

What this is NOT
----------------
- NOT a scanner.  No method here returns "all concepts."
  If you need to find concepts matching some criterion,
  query by name / by neighbors / by episodes — never by
  full iteration.  v2's correctness depends on this.
- NOT a writer.  Phase 2 writes still go through v1 substrate
  directly.  Phase 4 wraps writes in a journal/single-writer
  model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class LongTermSubstrate:
    """Query API over the agent's substrate of concepts/edges/
    bubbles/episodes.  Phase 2: thin wrapper around v1.

    Constructor takes a v1 `engine` (or any object with a
    `substrate` attribute).  All queries route through it."""

    def __init__(self, engine: Any = None):
        self.engine = engine

    @property
    def substrate(self):
        return getattr(self.engine, 'substrate', None) if self.engine else None

    # ---- named queries ----

    def has_concept(self, name: str) -> bool:
        sub = self.substrate
        if sub is None:
            return False
        return name in getattr(sub, 'concepts', {})

    def get_concept(self, name: str) -> Optional[Any]:
        sub = self.substrate
        if sub is None:
            return None
        return getattr(sub, 'concepts', {}).get(name)

    def neighbors(self,
                    name: str,
                    relation: Optional[str] = None
                    ) -> List[Tuple[str, str, float]]:
        """Return (neighbor_name, relation, strength) for every
        outgoing edge.  If `relation` is given, filter.

        Returns a flat list — does NOT walk the whole edge
        index.  Uses concept.edges_out, the per-concept
        adjacency.
        """
        concept = self.get_concept(name)
        if concept is None:
            return []
        edges_out = getattr(concept, 'edges_out', None) or {}
        out: List[Tuple[str, str, float]] = []
        for rel, edge_list in edges_out.items():
            if relation is not None and rel != relation:
                continue
            for edge in edge_list:
                tgt = str(getattr(edge, 'target', '') or '')
                strength = float(getattr(edge, 'strength', 0.0) or 0.0)
                if tgt:
                    out.append((tgt, rel, strength))
        return out

    def edge(self,
              source: str,
              relation: str,
              target: str) -> Optional[Any]:
        sub = self.substrate
        if sub is None:
            return None
        return getattr(sub, 'edges', {}).get(
            (source, relation, target))

    def episodes_mentioning(self,
                                  name: str,
                                  limit: int = 20
                                  ) -> List[Any]:
        """Episodes whose `.concepts` set contains `name`.
        Recent-first, capped at `limit`.

        NOTE: in Phase 2 this scans episode list — v1 doesn't
        have an inverse index from concept → episodes.  Phase 4
        will add it.  For now the episode list is bounded
        (substrate has an episodes cap), so the scan is
        bounded.
        """
        sub = self.substrate
        if sub is None:
            return []
        out: List[Any] = []
        episodes = getattr(sub, 'episodes', None) or []
        for ep in reversed(episodes):
            if name in (getattr(ep, 'concepts', None) or set()):
                out.append(ep)
                if len(out) >= limit:
                    break
        return out

    def bubbles_for(self, name: str) -> List[Any]:
        concept = self.get_concept(name)
        if concept is None:
            return []
        return list(getattr(concept, 'bubbles', None) or [])

    def best_bubble(self, name: str) -> Optional[Any]:
        """The bubble with the highest encounter_count — used
        when AWM bootstraps an EnrichedBubble for a concept
        coming back into active set."""
        bubbles = self.bubbles_for(name)
        if not bubbles:
            return None
        return max(bubbles, key=lambda b: int(
            getattr(b, 'encounter_count', 0) or 0))

    # NOTE: persist_v2_bubble + write-back mechanism was the
    # translation layer between v1 Bubble and v2 EnrichedBubble.
    # Removed 2026-05-14 when those two types unified into the
    # single seagi.core.bubble.Bubble.  The AWM Bubble IS the
    # substrate Bubble now; no translation needed; persistence
    # by construction.

    # ---- placeholder for embedding similarity ----

    def similar(self,
                 name: str,
                 top_k: int = 5,
                 threshold: float = 0.3
                 ) -> List[Tuple[str, float]]:
        """Top-k semantically-similar concepts.  Phase 2 stub
        — delegates to v1's semantic_lookup_by_text when
        available; Phase 4 will own the embedding index
        directly."""
        sub = self.substrate
        if sub is None:
            return []
        method = getattr(sub, 'semantic_lookup_by_text', None)
        if callable(method):
            try:
                return method(name, top_k=top_k,
                                threshold=threshold)
            except Exception:
                pass
        return []

    # ---- diagnostic ----

    def stats(self) -> Dict[str, int]:
        sub = self.substrate
        if sub is None:
            return {'concepts': 0, 'edges': 0, 'episodes': 0}
        return {
            'concepts': len(getattr(sub, 'concepts', {})),
            'edges': len(getattr(sub, 'edges', {})),
            'episodes': len(getattr(sub, 'episodes', []) or []),
        }
