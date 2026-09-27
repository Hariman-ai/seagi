"""Creative hypothesis generation.

Phase F.13 (2026-05-16).  v1 had `creativity.py`; v2 lost it.
Doctrine ([[project_seagi_imagination_wire_2026_05_10]]):
surfacing of creativity_pass hypotheses — "I wonder: what if
X could Y."

What this module is
-------------------
A `CreativityDaemon` that, on idle reflection, picks two
currently-AWM-active concepts with no strong existing edge
between them and proposes a SPECULATIVE relation as a
`CreativeHypothesis` record.

Hypotheses are NOT substrate writes.  They're working-memory
speculation — the agent wondering, not asserting.  The
`CreativeHypothesis` record carries:
  - concept_a, concept_b
  - proposed_relation (defaults to 'might_relate_to')
  - speculation_strength (low — clearly tentative)
  - origin chemistry tone snapshot
  - cycle

A bounded ring of recent hypotheses is queryable.  The
introspective handler surfaces top hypotheses as "I wonder
if X might relate to Y" wondering-voice clauses.

Doctrine alignment
------------------
- Hypotheses are promille-scale speculations.  Many can be
  generated; only those that withstand testing (e.g. the
  user confirms, or substrate reinforcement happens via
  encounter) ever become real edges.
- Daemon runs on idle reflection — the brain wonders when
  it has bandwidth, not under task pressure.
- Pair selection prefers concepts with chemistry-context
  overlap (both currently AWM-active with similar coactive
  fingerprints) — speculations between random unrelated
  pairs are not interesting; speculations between concepts
  that just co-occurred and have no edge yet ARE.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


# Capacity of the recent-hypotheses ring buffer.
DEFAULT_HYPOTHESES_CAPACITY = 20

# Minimum cycles between creativity passes — like goal_spawner,
# don't fire on every reflection.
MIN_CYCLES_BETWEEN_PASSES = 100

# Speculation strength on a new hypothesis.  Low — these are
# wonderings, not assertions.  Subsequent corroboration (a
# real substrate edge forming) is what would lift the actual
# substrate edge's strength.
DEFAULT_SPECULATION_STRENGTH = 0.1

# When picking a pair, require AT LEAST this many AWM concepts
# active.  Below this, the working memory is too sparse to
# meaningfully creative-combine.
MIN_AWM_FOR_CREATIVITY = 3

# Default proposed relation when no better candidate is known.
DEFAULT_RELATION = 'might_relate_to'


@dataclass
class CreativeHypothesis:
    """One wondering — a proposed link between concepts."""
    id: str
    concept_a: str
    concept_b: str
    relation: str = DEFAULT_RELATION
    speculation_strength: float = DEFAULT_SPECULATION_STRENGTH
    tone_label: str = ''       # chemistry tone at moment of wondering
    created_cycle: int = 0
    confirmations: int = 0     # if it later got tested + confirmed
    contradictions: int = 0    # if testing contradicted

    def describe(self) -> str:
        """First-person rendering for introspective use."""
        rel_phrase = self.relation.replace('_', ' ')
        # The default relation 'might_relate_to' already encodes
        # speculation; use 'I wonder if' framing so "could" +
        # "might" doesn't double up.
        return (f'I wonder if {self.concept_a} '
                  f'{rel_phrase} {self.concept_b}.')


class CreativityDaemon:
    """Periodic daemon that proposes hypotheses about AWM contents.

    Called from `Brain._maybe_fire_reflection` (same hook as
    goal_spawner).  Cheap; respects MIN_CYCLES_BETWEEN_PASSES.
    """

    def __init__(self,
                 awm_provider: Callable,
                 lts_provider: Callable,
                 cycle_provider: Callable,
                 chemistry_provider: Optional[Callable] = None,
                 capacity: int = DEFAULT_HYPOTHESES_CAPACITY):
        self._awm_provider = awm_provider
        self._lts_provider = lts_provider
        self._cycle_provider = cycle_provider
        self._chemistry_provider = chemistry_provider
        self._hypotheses: Deque[CreativeHypothesis] = deque(
            maxlen=int(capacity))
        # Dedup index — (a, b) tuple of sorted concepts so we
        # don't propose the same pair twice in a short window.
        self._proposed_pairs: set = set()
        self._next_id: int = 1
        self._last_pass_cycle: int = -10**6
        # RNG with seed for determinism in tests.  Caller can
        # reseed via `seed()`.
        self._rng = random.Random()
        # Diagnostics.
        self.passes: int = 0
        self.hypotheses_generated: int = 0
        self.passes_skipped_too_soon: int = 0
        self.passes_skipped_awm_too_sparse: int = 0

    def seed(self, n: int) -> None:
        """Set the RNG seed for deterministic pair selection."""
        self._rng.seed(int(n))

    # ---- public queries ----

    def __len__(self) -> int:
        return len(self._hypotheses)

    def recent(self, n: int = 5) -> List[CreativeHypothesis]:
        """Last n hypotheses, most-recent first."""
        return list(self._hypotheses)[-n:][::-1]

    def render_wonderings(self, top_n: int = 2) -> List[str]:
        """Render top recent hypotheses as first-person clauses
        for introspection."""
        return [h.describe() for h in self.recent(top_n)]

    # ---- generation ----

    def maybe_propose(self) -> Optional[CreativeHypothesis]:
        """One creativity pass.  Returns a new hypothesis if
        conditions are right, None otherwise."""
        cycle = self._cycle_provider()
        if cycle - self._last_pass_cycle < MIN_CYCLES_BETWEEN_PASSES:
            self.passes_skipped_too_soon += 1
            return None
        self._last_pass_cycle = cycle
        self.passes += 1

        # Pull AWM-active concepts.
        try:
            awm = self._awm_provider()
        except Exception:
            awm = None
        if awm is None:
            return None
        active = list(awm.active_concepts())
        # Filter out trivial / stopword-ish names (output_filter
        # is downstream).
        active = [c for c in active
                   if c and len(c) >= 3 and c.isalpha()]
        if len(active) < MIN_AWM_FOR_CREATIVITY:
            self.passes_skipped_awm_too_sparse += 1
            return None

        # Pick a pair.  Prefer pairs without an existing strong
        # edge between them.
        try:
            lts = self._lts_provider()
        except Exception:
            lts = None
        chosen = self._pick_pair(active, lts, cycle)
        if chosen is None:
            return None
        a, b = chosen

        # Tone snapshot at the moment of wondering.
        tone_label = ''
        if self._chemistry_provider is not None:
            try:
                chem = self._chemistry_provider()
                if chem is not None:
                    tone = chem.tone_summary()
                    tone_label = tone.get('label', '')
            except Exception:
                pass

        hid = f'h{self._next_id:04d}'
        self._next_id += 1
        h = CreativeHypothesis(
            id=hid,
            concept_a=a,
            concept_b=b,
            relation=DEFAULT_RELATION,
            speculation_strength=DEFAULT_SPECULATION_STRENGTH,
            tone_label=tone_label,
            created_cycle=cycle,
        )
        self._hypotheses.append(h)
        self._proposed_pairs.add(_pair_key(a, b))
        # Bound the dedup set to recent proposals only.
        if len(self._proposed_pairs) > self._hypotheses.maxlen * 2:
            self._proposed_pairs.clear()
        self.hypotheses_generated += 1
        return h

    def _pick_pair(self,
                       active: List[str],
                       lts: Optional[Any],
                       cycle: int
                       ) -> Optional[Tuple[str, str]]:
        """Pick two AWM-active concepts with no strong edge between
        them.  Tries up to MAX_TRIES random pairs before giving
        up (every pair currently has a strong edge → working
        memory is well-connected, no creative gap to fill)."""
        MAX_TRIES = 8
        candidates = list(active)
        for _ in range(MAX_TRIES):
            if len(candidates) < 2:
                return None
            a, b = self._rng.sample(candidates, 2)
            key = _pair_key(a, b)
            if key in self._proposed_pairs:
                continue
            # Check for existing edge.  An edge with strength <0.3
            # is "weak enough" for the pair to still count as
            # under-explored.
            if lts is not None:
                try:
                    found_strong = False
                    for nbr_name, _rel, strength in lts.neighbors(a):
                        if nbr_name == b and strength >= 0.3:
                            found_strong = True
                            break
                    if not found_strong:
                        for nbr_name, _rel, strength in lts.neighbors(b):
                            if nbr_name == a and strength >= 0.3:
                                found_strong = True
                                break
                    if found_strong:
                        continue
                except Exception:
                    pass
            return (a, b)
        return None

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'hypotheses': len(self._hypotheses),
            'passes': self.passes,
            'hypotheses_generated': self.hypotheses_generated,
            'passes_skipped_too_soon': self.passes_skipped_too_soon,
            'passes_skipped_awm_too_sparse': (
                self.passes_skipped_awm_too_sparse),
        }

    # ---- Phase H.1 (2026-05-17): M/I-weighted persistence ----

    # A hypothesis is "felt enough to matter for personality"
    # ONLY when it has been confirmed via dialog (G.7 testing).
    # Untested wonderings are session-local speculation —
    # personality is shaped by what the agent DECIDED, not what
    # it briefly entertained.  Confirmed hypotheses already
    # become substrate edges via G.7, but the CreativeHypothesis
    # record persists too as a "discovery memory."
    def _persist_ok(self, h: CreativeHypothesis) -> bool:
        return h.confirmations >= 1

    def to_dict(self) -> Dict[str, Any]:
        out: List[Dict[str, Any]] = []
        for h in self._hypotheses:
            if not self._persist_ok(h):
                continue
            out.append({
                'id': h.id,
                'concept_a': h.concept_a,
                'concept_b': h.concept_b,
                'relation': h.relation,
                'speculation_strength': h.speculation_strength,
                'tone_label': h.tone_label,
                'created_cycle': h.created_cycle,
                'confirmations': h.confirmations,
                'contradictions': h.contradictions,
            })
        return {'next_id': self._next_id, 'hypotheses': out}

    def load_from_dict(self, d: Dict[str, Any]) -> None:
        """Restore confirmed hypotheses from a saved snapshot.
        The pair-dedup set is also reset."""
        self._hypotheses.clear()
        self._proposed_pairs.clear()
        self._next_id = int(d.get('next_id', 1))
        for hd in d.get('hypotheses', []):
            h = CreativeHypothesis(
                id=str(hd.get('id', '')),
                concept_a=str(hd.get('concept_a', '')),
                concept_b=str(hd.get('concept_b', '')),
                relation=str(hd.get('relation', DEFAULT_RELATION)),
                speculation_strength=float(hd.get(
                    'speculation_strength',
                    DEFAULT_SPECULATION_STRENGTH)),
                tone_label=str(hd.get('tone_label', '')),
                created_cycle=int(hd.get('created_cycle', 0)),
                confirmations=int(hd.get('confirmations', 0)),
                contradictions=int(hd.get('contradictions', 0)))
            if h.id and h.concept_a and h.concept_b:
                self._hypotheses.append(h)
                self._proposed_pairs.add(_pair_key(
                    h.concept_a, h.concept_b))


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    """Order-independent key for a concept pair."""
    if a < b:
        return (a, b)
    return (b, a)
