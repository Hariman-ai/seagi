"""Identity — self-model curator.

Phase F.10 (2026-05-16).  v1 had identity.py; v2 lost it.  The
audit flagged the gap: vmDMN narrative does some self-reference
implicitly through autobiographical episodes, but there's no
coherent self-model curator that tracks "I am X / I value Y /
I fear Z" as first-class data.

What this module is
-------------------
A bounded registry of self-edges — tuples (relation, object,
strength, encounter_count, crystallization) where the implicit
subject is always 'self'.  The self-model is the answer to:
  "Who is Seagi?"
  "What does Seagi value?"
  "What does Seagi fear?"

Self-edges are NOT in the substrate.  They're brain-state, like
goals.  They're derived FROM substrate + dialog history + vmDMN
narrative, but they live in a small dedicated registry the
introspective handler queries directly.

How self-edges grow
-------------------
1. User assertions about Seagi: "You are wise", "You value
   patience" → routed by F.4 statement integration to
   SelfModel.record (subject 'you'/'i'/'seagi' → self).
2. Self-talk in vmDMN narrative: future hook where vmDMN
   identifies first-person assertions in its narrative and
   feeds them here.
3. Cortical introspection: when the introspective handler
   composes "Most recently I held X about myself", the held
   thought can be promoted to a self-edge.

How self-edges crystallize
--------------------------
Encounter count + age weighted, same as Bubble.crystallization
(Phase B).  An identity claim made many times across a long
session becomes a stable anchor; a one-time claim stays
malleable.

How contradictions get detected
-------------------------------
When a new self-edge arrives, SelfModel.coherence_check looks
for an existing edge with the same relation but an OPPOSITE
object (via the substrate's `opposite` edges, F.6 machinery).
If found and the existing edge is highly crystallized, the
self-model RESISTS the new assertion — pushes back rather than
overwriting.  This is identity stability.

Doctrine alignment
------------------
- Self-model lives in brain working memory.  Bounded.  Persisted
  via to_dict for cross-session continuity.
- Promille reinforcement scale per assertion.  Crystallization
  through accumulation, not single claims.
- "I am" claims by user are real signals but not absolute —
  high-crystallization self-edges (built from many
  reinforcements) can resist a single contradicting peer claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Maximum self-edges retained.  Identity should be a small,
# stable set of anchors, not an unbounded log.
DEFAULT_IDENTITY_CAPACITY = 32

# Promille reinforcement per assertion.  Same as substrate
# salience bumps (Phase C, SALIENCE_BUMP_DEFAULT = 0.005).
# Corrected 2026-05-16 (post-Phase-F audit): was 0.05, which
# was 10× percent-scale and inconsistent with the rest of the
# architecture.  Identity dynamics now match substrate
# salience dynamics.
SELF_EDGE_BUMP = 0.005

# Crystallization growth rate.  Matches Phase B's
# CRYSTALLIZATION_GROWTH_K = 0.005 in chemistry.py.  An edge
# reinforced ~200 times anchors firmly; matches the chemistry-
# side bubble crystallization timeline.
# Corrected 2026-05-16 from 0.05.
SELF_CRYSTALLIZATION_GROWTH = 0.005

# Crystallization age scale — older self-edges anchor stronger
# under repetition.
SELF_CRYSTALLIZATION_AGE_SCALE = 20.0

# Self-subject aliases.  When a statement's subject is in this
# set (case-insensitive), the statement gets routed to the
# SelfModel rather than the substrate.
SELF_SUBJECT_ALIASES = frozenset({
    'self', 'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself',
    'seagi',
})


@dataclass
class SelfEdge:
    """One first-class self-claim."""
    relation: str
    object: str
    strength: float = 0.5
    encounter_count: int = 1
    crystallization: float = 0.0
    created_cycle: int = 0
    last_reinforced_cycle: int = 0
    source: str = ''      # 'peer' / 'cortical' / 'narrative'

    def effective_strength(self, cycle: int) -> float:
        # Self-edges decay slowly toward 0; reinforcement keeps
        # them alive.  Same lazy-decay pattern as substrate Edge
        # (Phase C).
        elapsed = cycle - self.last_reinforced_cycle
        if elapsed <= 0:
            return self.strength
        decayed = self.strength - 0.00001 * elapsed
        return max(0.0, decayed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'relation': self.relation,
            'object': self.object,
            'strength': self.strength,
            'encounter_count': self.encounter_count,
            'crystallization': self.crystallization,
            'created_cycle': self.created_cycle,
            'last_reinforced_cycle': self.last_reinforced_cycle,
            'source': self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SelfEdge':
        return cls(
            relation=d.get('relation', ''),
            object=d.get('object', ''),
            strength=float(d.get('strength', 0.5)),
            encounter_count=int(d.get('encounter_count', 1)),
            crystallization=float(d.get('crystallization', 0.0)),
            created_cycle=int(d.get('created_cycle', 0)),
            last_reinforced_cycle=int(d.get(
                'last_reinforced_cycle', 0)),
            source=d.get('source', ''),
        )


class SelfModel:
    """Bounded registry of self-edges."""

    def __init__(self,
                 capacity: int = DEFAULT_IDENTITY_CAPACITY):
        self.capacity = int(capacity)
        # Keyed by (relation, object) for fast dedup and lookup.
        self._edges: Dict[Tuple[str, str], SelfEdge] = {}
        # Diagnostics.
        self.edges_recorded: int = 0
        self.edges_reinforced: int = 0
        self.conflicts_detected: int = 0

    # ---- queries ----

    def __len__(self) -> int:
        return len(self._edges)

    def all_edges(self) -> List[SelfEdge]:
        return list(self._edges.values())

    def edges_by_relation(self, relation: str) -> List[SelfEdge]:
        """All self-edges with a given relation, e.g. all the
        'value' or 'fear' statements."""
        return [e for (r, _o), e in self._edges.items()
                  if r == relation]

    def top_edges(self, n: int = 5) -> List[SelfEdge]:
        """Top N self-edges ranked by crystallization (most-
        firm-identity first), with strength as tiebreaker."""
        edges = list(self._edges.values())
        edges.sort(key=lambda e: (
            -e.crystallization, -e.strength))
        return edges[:max(1, n)]

    def get(self,
                relation: str,
                obj: str) -> Optional[SelfEdge]:
        return self._edges.get((relation, obj))

    # ---- recording / reinforcement ----

    def record(self,
                  relation: str,
                  obj: str,
                  cycle: int,
                  *,
                  strength: float = 0.5,
                  source: str = '') -> SelfEdge:
        """Add or reinforce a self-edge.  Crystallization grows
        with each encounter, age-weighted (older encounters
        anchor stronger).
        """
        if not relation or not obj:
            # Skip empty assertions silently.
            return None  # type: ignore[return-value]
        key = (relation, obj)
        existing = self._edges.get(key)
        if existing is None:
            # New edge.  Capacity-evict if at limit.
            if len(self._edges) >= self.capacity:
                self._evict_weakest()
            e = SelfEdge(
                relation=relation, object=obj,
                strength=float(strength),
                encounter_count=1,
                crystallization=0.0,
                created_cycle=int(cycle),
                last_reinforced_cycle=int(cycle),
                source=source)
            self._edges[key] = e
            self.edges_recorded += 1
            return e
        # Reinforce existing.
        existing.strength = min(1.0,
            existing.strength + SELF_EDGE_BUMP)
        existing.encounter_count += 1
        existing.last_reinforced_cycle = int(cycle)
        # Crystallization grows; early imprints anchor stronger
        # (same age weighting as substrate Bubble — Phase B).
        age_weight = 1.0 / (
            1.0 + existing.encounter_count
            / SELF_CRYSTALLIZATION_AGE_SCALE)
        existing.crystallization = min(1.0,
            existing.crystallization
            + SELF_CRYSTALLIZATION_GROWTH * age_weight)
        if source and not existing.source:
            existing.source = source
        self.edges_reinforced += 1
        return existing

    def _evict_weakest(self) -> None:
        if not self._edges:
            return
        weakest_key = min(
            self._edges,
            key=lambda k: (
                self._edges[k].crystallization,
                self._edges[k].strength))
        del self._edges[weakest_key]

    # ---- coherence ----

    def coherence_check(self,
                              relation: str,
                              obj: str,
                              substrate: Optional[Any] = None
                              ) -> Optional[SelfEdge]:
        """Look for an existing self-edge that conflicts with the
        proposed (relation, obj) claim.

        Two patterns (mirrors F.6 substrate contradiction):
          (a) Inverse relation: existing edge has a relation
              that's in INVERSE_RELATIONS[relation].
          (b) Opposite object: existing edge has the same relation
              but an opposite object (via substrate's `opposite`
              edges, if substrate available).

        Returns the conflicting SelfEdge, or None.  The caller
        decides what to do (push back, flag, accept).
        """
        from seagi.core.substrate import INVERSE_RELATIONS

        # (a) Inverse relation.
        for r_inv in INVERSE_RELATIONS.get(relation, ()):
            e = self._edges.get((r_inv, obj))
            if e is not None and e.strength > 0:
                self.conflicts_detected += 1
                return e

        # (b) Opposite object (same relation, opposite object).
        if substrate is None:
            return None
        edges = getattr(substrate, 'edges', {}) or {}
        for (r, o), self_edge in self._edges.items():
            if r != relation or o == obj:
                continue
            # Does o have an `opposite` edge to obj?
            if (edges.get((o, 'opposite', obj)) is not None
                    or edges.get((obj, 'opposite', o)) is not None):
                self.conflicts_detected += 1
                return self_edge
        return None

    # ---- introspection rendering ----

    def render_introspection(self, top_n: int = 3) -> List[str]:
        """Render top self-edges as first-person clauses for the
        introspective handler.  Returns up to top_n strings like
        'I value patience', 'I fear loss', 'I am wise'.
        """
        out: List[str] = []
        for e in self.top_edges(top_n):
            # Check the raw relation name (with underscores)
            # before replacing — 'has_property' and 'is_a'
            # both render as "I am X".
            if e.relation in ('is_a', 'has_property', 'is'):
                out.append(f'I am {e.object}')
            else:
                rel_phrase = e.relation.replace('_', ' ')
                out.append(f'I {rel_phrase} {e.object}')
        return out

    # ---- persistence ----

    # M/I-weighted persistence threshold (Phase H.1).  An edge is
    # "felt enough to matter for personality" when crystallization
    # crosses the salience floor OR the edge has been reinforced
    # at least 3 times.  Below this floor, the edge is dialog
    # noise — let it dissolve with the session.
    PERSIST_CRYSTALLIZATION_FLOOR = 0.05
    PERSIST_MIN_ENCOUNTER = 3

    def _persist_ok(self, e: SelfEdge) -> bool:
        return (e.crystallization >= self.PERSIST_CRYSTALLIZATION_FLOOR
                or e.encounter_count >= self.PERSIST_MIN_ENCOUNTER)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'edges': [e.to_dict() for e in self._edges.values()
                      if self._persist_ok(e)],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SelfModel':
        m = cls()
        for ed in d.get('edges', []):
            e = SelfEdge.from_dict(ed)
            m._edges[(e.relation, e.object)] = e
        return m

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'edges': len(self._edges),
            'edges_recorded': self.edges_recorded,
            'edges_reinforced': self.edges_reinforced,
            'conflicts_detected': self.conflicts_detected,
        }
