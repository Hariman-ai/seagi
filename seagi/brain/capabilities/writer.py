"""Journaled single-writer substrate.

Brain analog: hippocampal-cortical consolidation pipeline.
Multiple sources propose changes; one writer applies them.
Provides a journal that can be replayed for diagnostics.

Replaces Phase 3's `_StubSubstrateWriter` / `_MemoryWriter` in
runtime.  Same contract: subscribes SUBSTRATE_WRITE_QUEUED,
applies to v1 substrate (or in-memory map), is idempotent.

What this adds vs the stubs
---------------------------
- An append-only journal (capacity-bounded) of every applied
  AND every skipped write, with reason
- Per-source-capability stats (who is writing what)
- Strength reinforcement on duplicate writes (when same edge
  comes through twice, bumps strength toward the higher of
  the two; doesn't double-write)
- Single-writer guarantee — substrate is mutated ONLY here,
  not anywhere else in v2.  (v1 still writes directly; the
  v2 contract is "no other Phase-4+ capability touches
  substrate.add_edge.")
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional, Tuple

from ..events import (
    EventKind, BrainEvent,
    SubstrateWriteQueuedEvent,
)
from ..bus import EventBus
from seagi.core.substrate import EDGE_STRENGTH_BUMP_PER_USE


JOURNAL_CAPACITY = 2000
REINFORCEMENT_FACTOR = 0.5    # blend toward stronger value

# Earn-by-predicting, NEVER earn-by-being-seen (facet-docking C1, RESCOPED
# 2026-07-21 per the ignition re-audit).  A write whose reason is listed here
# merely OBSERVED a fact — it is applied UN-engaged on creation, and a
# DUPLICATE of it does NOT reinforce (no strength ratchet on perception).
# Strength for such a fact moves ONLY through the contrastive dock-earning
# channel (write_reason='facet_confirm' / 'facet_disconfirm').
#
# C1 RESCOPE (Flag-C, 2026-07-21): 'facet_observe' ONLY.  The live
# world_observe duplicate-reinforce path (measured ~478:1) is ALSO a
# perception ratchet, but closing it removes recovery-by-re-observation on
# the attention-stream grounding (a confidently-wrong standing prediction
# would never heal — measured: 0 confirms after 5,001 re-observes).  That
# recovery question needs its own design pass, so world_observe semantics
# are UNCHANGED here: un-engaged on creation (the pre-existing special case
# below), duplicates still reinforce.  The facet channel is born under the
# strict law; the world channel keeps its incumbent behavior.
OBSERVE_REASONS = frozenset({'facet_observe'})


class JournaledSubstrateWriter:
    """Single source of v2 substrate writes."""

    SUBSCRIPTIONS = (EventKind.SUBSTRATE_WRITE_QUEUED,)

    def __init__(self,
                 engine: Any = None,
                 memory_mode: bool = False):
        self.engine = engine
        self.memory_mode = (engine is None) or memory_mode
        self.journal: Deque[Dict[str, Any]] = deque(
            maxlen=JOURNAL_CAPACITY)
        self.memory_edges: Dict[Tuple[str, str, str], float] = {}
        self.writes_applied: int = 0
        self.writes_reinforced: int = 0
        self.writes_skipped: int = 0
        self.writes_weakened: int = 0   # facet_disconfirm (contrastive earning)
        self.by_source: Dict[str, int] = defaultdict(int)

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if not isinstance(event, SubstrateWriteQueuedEvent):
            return
        self._apply(event)

    def _apply(self,
                  event: SubstrateWriteQueuedEvent) -> None:
        key = (event.subject, event.relation, event.object)
        if not all(key):
            self._record(event, 'skipped', 'incomplete_key')
            self.writes_skipped += 1
            return
        if self.memory_mode:
            existing = self.memory_edges.get(key)
            if existing is None:
                self.memory_edges[key] = event.strength
                self._record(event, 'applied', '')
                self.writes_applied += 1
                self.by_source[event.source_capability] += 1
            elif event.strength > existing:
                blended = existing + REINFORCEMENT_FACTOR * (
                    event.strength - existing)
                self.memory_edges[key] = blended
                self._record(event, 'reinforced',
                                  f'{existing:.3f}->{blended:.3f}')
                self.writes_reinforced += 1
            else:
                self._record(event, 'skipped', 'duplicate')
                self.writes_skipped += 1
            return
        # v1 substrate-backed path.
        substrate = getattr(self.engine, 'substrate', None)
        if substrate is None:
            self._record(event, 'skipped', 'no_substrate')
            self.writes_skipped += 1
            return
        try:
            existing = substrate.edges.get(key)
            if existing is None:
                if event.write_reason == 'facet_disconfirm':
                    # Contrastive dock-earning (2026-07-21): a disconfirm
                    # WEAKENS existing evidence; it must never CREATE an
                    # edge (there is nothing to weaken).
                    self.writes_skipped += 1
                    self._record(event, 'skipped', 'disconfirm_missing')
                    return
                # Grounding world-loop (2026-06-03): a 'world_observe'
                # write records a transition that was merely OBSERVED
                # (a miss — not predicted correctly).  It must NOT be
                # engaged, so the only survival signal for a world-
                # transition edge is a correct prediction.  Facet-docking
                # C1 (2026-07-21): 'facet_observe' (the percept's facet
                # componentry, merely SEEN) is applied un-engaged for the
                # same reason.  All other new writes engage (valid imports
                # get a grace window).
                substrate.add_edge(
                    source=event.subject,
                    target=event.object,
                    relation_name=event.relation,
                    strength=event.strength,
                    cycle=event.cycle,
                    engage=(event.write_reason != 'world_observe'
                            and event.write_reason not in OBSERVE_REASONS))
                self.writes_applied += 1
                self.by_source[event.source_capability] += 1
                self._record(event, 'applied', '')
            elif event.write_reason in OBSERVE_REASONS:
                # OBSERVE-DUPLICATE (facet-docking C1, 2026-07-21): the same
                # observed fact (facet_observe) arrived again merely by being
                # RE-SEEN.  Earn-by-predicting doctrine: being seen twice is
                # not earning — NO reinforce, NO engagement stamp, NO strength
                # move.  This keeps uninformative-but-ubiquitous facets from
                # ratcheting on sight (the inverse-IDF failure the audit
                # flagged).  The fact still lives in the substrate; it gains
                # strength only through a dock CONFIRM.
                # (world_observe duplicates deliberately NOT included — the
                # rider stays out pending the recovery design pass; see
                # OBSERVE_REASONS above.)
                self.writes_skipped += 1
                self._record(event, 'skipped', 'observe_duplicate')
            elif event.write_reason == 'facet_disconfirm':
                # Contrastive dock-earning, the SUBTRACTIVE half (2026-07-21):
                # this elector's remembered signature DISAGREED with the
                # observed outcome in a split event.  Symmetric promille:
                # weaken by the SAME existing bump constant the confirm side
                # uses (EDGE_STRENGTH_BUMP_PER_USE = 0.005), after applying
                # lazy decay, floored at 0.0 (below the prune floor → the
                # reaper takes it; no engagement stamp, no new constant).
                pre = existing.effective_strength(int(event.cycle))
                existing.strength = pre
                existing.weaken(EDGE_STRENGTH_BUMP_PER_USE)
                existing.last_reinforced_cycle = int(event.cycle)
                self.writes_weakened += 1
                self._record(event, 'weakened',
                             f'{pre:.3f}->{existing.strength:.3f}')
            else:
                # Phase S.1: a re-attested edge is being USED —
                # the same proposition arrived from another
                # sentence.  Every re-attestation reinforces at
                # promille (edge.reinforce applies lazy decay then
                # bumps).  This is the "earn it" half of earn-or-
                # dissolve: a triple seen many times across the
                # corpus accumulates strength and survives pruning;
                # a one-off stays provisional and decays away.
                cur_strength = float(
                    getattr(existing, 'strength', 0.0))
                try:
                    existing.reinforce(int(event.cycle),
                                       origin='cognition')
                    # Stage 2 grounding valence: land M/I direction on the
                    # edge ONLY on a predicted (world_confirm) re-encounter.
                    if getattr(event, 'outcome', ''):
                        try:
                            existing.evidence.record(
                                event.outcome,
                                float(getattr(event, 'mi_m', 0.0)),
                                float(getattr(event, 'mi_i', 0.0)),
                                int(event.cycle))
                        except Exception:
                            pass
                    # Reinforce-on-recall igniter (2026-05-31): a
                    # recall/inference re-attestation is a cognitive
                    # bid-path touching this edge.  reinforce() bumps
                    # strength and anchors decay (last_reinforced_
                    # cycle) but does NOT update last_engaged_cycle —
                    # the quarantine engagement signal.  Stamp it
                    # here so the engaged coherent core stays OUT of
                    # the quiescent quarantine pool.  Scoped to the
                    # metered cognitive write_reasons; corpus re-
                    # attestation is the same bid-path class and a
                    # deliberate follow-up, not the smallest version.
                    if event.write_reason in (
                            'recall_reattest',
                            'inference_consolidate',
                            'world_confirm',
                            'facet_confirm',
                            'user_assertion',
                            'hypothesis_confirmed'):
                        # audit #15 (2026-06-04): user/peer assertion and
                        # a confirmed hypothesis are genuine deliberate
                        # bid-path touches — the same engagement class as
                        # recall — so an actively re-asserted edge no
                        # longer drifts to quarantine while in use.
                        existing.last_engaged_cycle = int(event.cycle)
                    self.writes_reinforced += 1
                    self._record(
                        event, 'reinforced',
                        f'{cur_strength:.3f}->'
                        f'{existing.strength:.3f}')
                except Exception:
                    self.writes_skipped += 1
                    self._record(event, 'skipped',
                                      'reinforce_failed')
        except Exception as e:
            self.writes_skipped += 1
            self._record(event, 'skipped',
                              f'error:{type(e).__name__}')

    def _record(self,
                  event: SubstrateWriteQueuedEvent,
                  status: str,
                  detail: str) -> None:
        self.journal.append({
            'cycle': event.cycle,
            'subject': event.subject,
            'relation': event.relation,
            'object': event.object,
            'strength': event.strength,
            'source_capability': event.source_capability,
            'reason': event.write_reason,
            'status': status,
            'detail': detail,
        })

    # ---- queries ----

    def journal_tail(self, n: int = 20) -> list:
        return list(self.journal)[-n:]

    def stats(self) -> Dict[str, Any]:
        return {
            'memory_mode': self.memory_mode,
            'writes_applied': self.writes_applied,
            'writes_reinforced': self.writes_reinforced,
            'writes_skipped': self.writes_skipped,
            'writes_weakened': self.writes_weakened,
            'journal_size': len(self.journal),
            'by_source': dict(self.by_source),
        }
