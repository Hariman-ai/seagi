"""Schema discovery — finding recurring multi-slot patterns.

Phase F.15 (2026-05-16), final piece of Phase F.

v1 had `schemas.py` + `schema_library.py`; v2 lost them.
Doctrine ([[project_seagi_relational_schemas_2026_05_10]]):
multi-slot edge patterns + inference.  When substrate contains
enough "X R1 Y AND Y R2 Z" chains, the schema (R1, R2 →
inferable) is itself a discovery.

What this module is
-------------------
A periodic discoverer that scans Substrate edges for recurring
multi-slot patterns:

  TRANSITIVE_CAUSES   X causes Y, Y causes Z → X likely causes Z
  TRANSITIVE_ENABLES  X enables Y, Y enables Z → X likely enables Z
  TRANSITIVE_IS_A     X is_a Y, Y is_a Z → X is_a Z (taxonomic)
  SHARED_CAUSE        X causes Y, X causes Z → Y, Z share cause
  DOUBLE_OPPOSITE     X opposite Y, A opposite Y → A similar X

When a pattern is observed N≥3 times in substrate, it's
promoted to a `Schema` record.  Each schema carries example
triples so cortical can later recall WHY the pattern emerged.

Schemas are NOT substrate writes by default — they're brain
working memory.  A future hook could let cortical "test" a
schema-derived inference by querying for the implied edge; if
it lands consistently from corpus, the substrate-side edge
gets reinforced naturally via the existing F.4/F.5 paths.

Doctrine alignment
------------------
- Promille-scale promotion threshold (N≥3 instances).  One-off
  chains don't make a schema.
- Bounded registry — evicts least-supported when full.
- No automatic substrate writes — schemas are observations
  about substrate, not new facts to inject.  Cortical can
  consult them; user-confirmed inferences become real edges
  via the normal F.4 statement-integration path.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Set,
                       Tuple)


# Pattern kinds.
SCHEMA_TRANSITIVE_CAUSES = 'transitive_causes'
SCHEMA_TRANSITIVE_ENABLES = 'transitive_enables'
SCHEMA_TRANSITIVE_IS_A = 'transitive_is_a'
SCHEMA_SHARED_CAUSE = 'shared_cause'
SCHEMA_DOUBLE_OPPOSITE = 'double_opposite'


# Promotion threshold — pattern must occur this many times in
# substrate to graduate to a Schema.
PROMOTION_THRESHOLD = 3

# Maximum schemas retained.
DEFAULT_SCHEMA_CAPACITY = 64

# Minimum cycles between discoverer passes.  Substrate scans
# are O(E²) for transitive patterns; doing this on every
# reflection would be wasteful.
MIN_CYCLES_BETWEEN_PASSES = 500

# Maximum edges to scan per pass.  Caps cost on large
# substrates.  Schemas accumulate across many passes, so
# bounded scans are fine.
MAX_EDGES_PER_PASS = 5000


@dataclass
class Schema:
    """One discovered multi-slot pattern."""
    id: str
    kind: str
    support: int = 0           # number of substrate instances
    examples: List[Tuple[str, str, str, str, str]] = field(
        default_factory=list)  # tuples of (a, rel1, b, rel2, c)
    discovered_cycle: int = 0
    last_confirmed_cycle: int = 0

    @property
    def confidence(self) -> float:
        """Confidence rises with support, saturating gently."""
        return min(1.0, self.support / 10.0)

    def describe(self) -> str:
        """Human-readable description."""
        if self.kind == SCHEMA_TRANSITIVE_CAUSES:
            return (f'Transitive causes: if A causes B and B '
                      f'causes C, A likely causes C '
                      f'(support {self.support})')
        if self.kind == SCHEMA_TRANSITIVE_ENABLES:
            return (f'Transitive enables: A→B→C chain '
                      f'(support {self.support})')
        if self.kind == SCHEMA_TRANSITIVE_IS_A:
            return (f'Taxonomic chain: A is_a B is_a C → A is_a C '
                      f'(support {self.support})')
        if self.kind == SCHEMA_SHARED_CAUSE:
            return (f'Shared cause: B and C both arise from a '
                      f'common A (support {self.support})')
        if self.kind == SCHEMA_DOUBLE_OPPOSITE:
            return (f'Double opposite: A and X both oppose Y → '
                      f'A similar X (support {self.support})')
        return f'Schema {self.kind} (support {self.support})'


class SchemaLibrary:
    """Bounded registry of discovered schemas."""

    def __init__(self,
                 capacity: int = DEFAULT_SCHEMA_CAPACITY):
        self.capacity = int(capacity)
        self._schemas: Dict[str, Schema] = {}
        self._next_id: int = 1
        # Diagnostics.
        self.schemas_promoted: int = 0
        self.schemas_evicted: int = 0

    def __len__(self) -> int:
        return len(self._schemas)

    def all_schemas(self) -> List[Schema]:
        return list(self._schemas.values())

    def by_kind(self, kind: str) -> List[Schema]:
        return [s for s in self._schemas.values()
                  if s.kind == kind]

    def top_schemas(self, n: int = 5) -> List[Schema]:
        s = list(self._schemas.values())
        s.sort(key=lambda x: -x.support)
        return s[:max(1, n)]

    def get_or_create(self, kind: str, cycle: int) -> Schema:
        """One Schema per kind; either return existing or create."""
        for s in self._schemas.values():
            if s.kind == kind:
                return s
        if len(self._schemas) >= self.capacity:
            # Evict lowest-support.
            weakest = min(self._schemas.values(),
                              key=lambda s: s.support)
            del self._schemas[weakest.id]
            self.schemas_evicted += 1
        sid = f'sc{self._next_id:04d}'
        self._next_id += 1
        s = Schema(id=sid, kind=kind,
                      discovered_cycle=cycle,
                      last_confirmed_cycle=cycle)
        self._schemas[sid] = s
        return s

    def render_summary(self, top_n: int = 3) -> List[str]:
        """Top schemas as human-readable strings."""
        return [s.describe() for s in self.top_schemas(top_n)]

    # ---- Phase H.1 (2026-05-17): M/I-weighted persistence ----

    # A schema is "felt enough to matter for personality" once it
    # has met the promotion threshold.  Since SchemaLibrary only
    # creates a record AFTER PROMOTION_THRESHOLD instances exist,
    # every record in the library by definition crossed that
    # floor — persist all of them.
    def to_dict(self) -> Dict[str, Any]:
        out: List[Dict[str, Any]] = []
        for s in self._schemas.values():
            out.append({
                'id': s.id, 'kind': s.kind, 'support': s.support,
                'examples': [list(t) for t in s.examples],
                'discovered_cycle': s.discovered_cycle,
                'last_confirmed_cycle': s.last_confirmed_cycle,
            })
        return {
            'next_id': self._next_id,
            'schemas_promoted': self.schemas_promoted,
            'schemas': out,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SchemaLibrary':
        lib = cls()
        lib._next_id = int(d.get('next_id', 1))
        lib.schemas_promoted = int(d.get('schemas_promoted', 0))
        for sd in d.get('schemas', []):
            sid = str(sd.get('id', ''))
            if not sid:
                continue
            examples = [tuple(e) for e in sd.get('examples', [])]
            s = Schema(
                id=sid, kind=str(sd.get('kind', '')),
                support=int(sd.get('support', 0)),
                examples=examples,
                discovered_cycle=int(sd.get('discovered_cycle', 0)),
                last_confirmed_cycle=int(sd.get(
                    'last_confirmed_cycle', 0)))
            lib._schemas[sid] = s
        return lib


class SchemaDiscoverer:
    """Periodic scanner that finds schema patterns in substrate."""

    def __init__(self,
                 lts_provider: Callable,
                 library: Optional[SchemaLibrary] = None,
                 cycle_provider: Optional[Callable] = None):
        self._lts_provider = lts_provider
        self._cycle_provider = cycle_provider or (lambda: 0)
        self.library = library or SchemaLibrary()
        self._last_pass_cycle: int = -10**6
        self.passes: int = 0
        self.edges_scanned_total: int = 0

    # ---- public ----

    def maybe_discover(self) -> List[Schema]:
        """Run a discovery pass if interval has elapsed.  Returns
        newly-promoted or newly-updated schemas."""
        cycle = self._cycle_provider()
        if cycle - self._last_pass_cycle < MIN_CYCLES_BETWEEN_PASSES:
            return []
        self._last_pass_cycle = cycle
        self.passes += 1
        return self.discover_now(cycle)

    def discover_now(self, cycle: int) -> List[Schema]:
        """Force a discovery pass.  Returns newly-promoted or
        re-confirmed schemas."""
        try:
            lts = self._lts_provider()
        except Exception:
            return []
        if lts is None:
            return []
        substrate = getattr(lts, 'substrate', None)
        if substrate is None:
            return []
        edges = getattr(substrate, 'edges', {}) or {}
        if not edges:
            return []

        # Index edges by (source, relation) for fast lookup.
        # Cap at MAX_EDGES_PER_PASS to bound cost.
        edges_list = list(edges.items())[:MAX_EDGES_PER_PASS]
        self.edges_scanned_total += len(edges_list)
        from seagi.core.substrate import EDGE_PRUNE_FLOOR
        by_source: Dict[Tuple[str, str], List[str]] = (
            defaultdict(list))
        for (s, r, t), _e in edges_list:
            # Earn-gate (2026-06-04 audit #16): schema support counts
            # only edges that EARNED strength — skip noise-floor edges —
            # so a promoted schema reflects real structural regularity,
            # not the provisional/stillborn noise that a ~81%-never-
            # cohered substrate is full of.  A schema built on noise then
            # nudges noise walks (+15% confidence).  Reuses the existing
            # prune floor; no new constant.
            try:
                if _e.effective_strength(cycle) <= EDGE_PRUNE_FLOOR:
                    continue
            except Exception:
                pass
            by_source[(s, r)].append(t)

        promoted_or_updated: List[Schema] = []

        # ---- TRANSITIVE patterns (causes / enables / is_a) ----
        for transitive_rel, schema_kind in (
                ('causes', SCHEMA_TRANSITIVE_CAUSES),
                ('enables', SCHEMA_TRANSITIVE_ENABLES),
                ('is_a', SCHEMA_TRANSITIVE_IS_A)):
            count, examples = self._count_transitive(
                by_source, transitive_rel)
            if count >= PROMOTION_THRESHOLD:
                s = self.library.get_or_create(schema_kind, cycle)
                if s.support < count:
                    if s.support == 0:
                        self.library.schemas_promoted += 1
                    s.support = count
                    # Keep up to 3 illustrative examples.
                    s.examples = examples[:3]
                    s.last_confirmed_cycle = cycle
                    promoted_or_updated.append(s)

        # ---- SHARED_CAUSE: X causes Y AND X causes Z ----
        shared_count, shared_examples = self._count_shared_cause(
            by_source)
        if shared_count >= PROMOTION_THRESHOLD:
            s = self.library.get_or_create(
                SCHEMA_SHARED_CAUSE, cycle)
            if s.support < shared_count:
                if s.support == 0:
                    self.library.schemas_promoted += 1
                s.support = shared_count
                s.examples = shared_examples[:3]
                s.last_confirmed_cycle = cycle
                promoted_or_updated.append(s)

        # ---- DOUBLE_OPPOSITE: X opp Y AND A opp Y ----
        double_count, double_examples = self._count_double_opposite(
            by_source, edges)
        if double_count >= PROMOTION_THRESHOLD:
            s = self.library.get_or_create(
                SCHEMA_DOUBLE_OPPOSITE, cycle)
            if s.support < double_count:
                if s.support == 0:
                    self.library.schemas_promoted += 1
                s.support = double_count
                s.examples = double_examples[:3]
                s.last_confirmed_cycle = cycle
                promoted_or_updated.append(s)

        return promoted_or_updated

    # ---- pattern counters ----

    def _count_transitive(self,
                                by_source: Dict[Tuple[str, str], List[str]],
                                rel: str
                                ) -> Tuple[int, List[Tuple]]:
        """Count A R B AND B R C instances."""
        count = 0
        examples: List[Tuple] = []
        for (a, r), targets in by_source.items():
            if r != rel:
                continue
            for b in targets:
                # Does b have any outgoing rel-edges?
                next_targets = by_source.get((b, rel), [])
                for c in next_targets:
                    if c == a or c == b:
                        continue
                    count += 1
                    if len(examples) < 10:
                        examples.append((a, rel, b, rel, c))
        return (count, examples)

    def _count_shared_cause(self,
                                  by_source: Dict[Tuple[str, str], List[str]]
                                  ) -> Tuple[int, List[Tuple]]:
        """Count cases where a single source A has ≥2 outgoing
        'causes' edges (so A causes B AND A causes C).  Each
        such case counts toward shared-cause support."""
        count = 0
        examples: List[Tuple] = []
        for (a, r), targets in by_source.items():
            if r != 'causes':
                continue
            if len(targets) >= 2:
                count += len(targets) - 1
                if len(examples) < 10 and len(targets) >= 2:
                    examples.append(
                        (a, 'causes', targets[0],
                          'causes', targets[1]))
        return (count, examples)

    def _count_double_opposite(self,
                                       by_source: Dict[Tuple[str, str], List[str]],
                                       edges: Dict[Any, Any]
                                       ) -> Tuple[int, List[Tuple]]:
        """Count cases where two distinct concepts X and A both
        have an 'opposite' edge to the same Y."""
        # Group opposite-targets by target.
        by_target: Dict[str, List[str]] = defaultdict(list)
        for (s, r), targets in by_source.items():
            if r != 'opposite':
                continue
            for t in targets:
                by_target[t].append(s)
        count = 0
        examples: List[Tuple] = []
        for y, sources in by_target.items():
            if len(sources) >= 2:
                count += len(sources) - 1
                if len(examples) < 10:
                    examples.append(
                        (sources[0], 'opposite', y,
                          'opposite', sources[1]))
        return (count, examples)

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'passes': self.passes,
            'schemas': len(self.library),
            'schemas_promoted': self.library.schemas_promoted,
            'edges_scanned_total': self.edges_scanned_total,
        }
