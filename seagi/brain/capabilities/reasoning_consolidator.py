"""ReasoningConsolidator — derived inferences become substrate.

Phase Step.2 of the roadmap (2026-05-22).  Before this, cortical's
R.1 inference engine derived conclusions ("Socrates is_a mortal"
from "Socrates is_a man, man is_a mortal") and emitted them as
thoughts — then threw them away.  Each time the agent reasoned
about Socrates it re-walked the chain.  Reasoning was read-only.

Real cognition COMPOUNDS — a derived fact, once reasoned, is
available going forward.  This capability listens for inference
thoughts and writes the derived edge back to substrate.

Doctrine compliance — the architecture composes perfectly here:

  - **Promille / small trace**: the derived edge enters at
    PROVISIONAL_EDGE_STRENGTH — one inference is one encounter,
    a small trace.
  - **Earn-or-dissolve (Phase S)**: the edge is provisional.  On
    the next coherence pass it will find the supporting path A→X→B
    (by construction — that's how it was derived) and get
    reinforced.  An inference whose supports later decay (because
    they didn't cohere with anything else) loses corroboration
    and itself dissolves.  Self-consistent.
  - **Wire what's there**: standard SubstrateWriteQueuedEvent →
    JournaledSubstrateWriter → the same earn-or-dissolve pipeline
    every other edge goes through.
  - **No new event types**.

Subscribes
----------
THOUGHT_PRODUCED  — filters on method='inference'

Emits
-----
SUBSTRATE_WRITE_QUEUED  — provisional derived edge
"""

from __future__ import annotations

import time
from collections import deque
from typing import (
    Any, Callable, Deque, Dict, List, Optional, Tuple)

from ..events import (
    EventKind, BrainEvent,
    ThoughtProducedEvent,
    SubstrateWriteQueuedEvent,
)
from ..bus import EventBus


class ReasoningConsolidator:
    """Writes inference-derived edges to substrate as provisional
    traces.  The earn-or-dissolve pipeline filters them."""

    SUBSCRIPTIONS = (EventKind.THOUGHT_PRODUCED,)

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        # Diagnostics.
        self.consolidations_emitted: int = 0
        # Reinforce-on-recall igniter (2026-05-31): recall ('causal')
        # thoughts route through the metered write chain too.
        self.recalls_reattested: int = 0
        self.recall_writes_emitted: int = 0
        # Bounded ring of recently recall-reattested edge keys so the
        # runtime can read back the coherent fraction (the make-or-
        # break number) WITHOUT a feedback path into cognition.
        self._recall_keys: Deque[Tuple[str, str, str]] = deque(
            maxlen=2000)

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if not isinstance(event, ThoughtProducedEvent):
            return
        # Two-tier dispatch (reinforce-on-recall igniter, 2026-05-31;
        # per-hop reinforcement added 2026-06-01):
        #   'inference' → DERIVATION: a NEW conclusion the agent
        #       reasoned out (R.1).  Write the composed CONCLUSION as
        #       a provisional edge (unchanged).
        #   'causal'    → RECALL: the agent WALKED existing edge(s).
        #       That traversal IS a re-attestation / the quarantine
        #       4th bid-path — route it through the SAME metered chain
        #       so (a) MetabolicDebt accrues (writes → sleep), (b) the
        #       writer stamps last_engaged_cycle (lifts the edge out
        #       of quarantine), (c) it reinforces by the promille
        #       per-use bump.  PER-HOP: reinforce the REAL walked hops
        #       (event.edges_walked) — the genuine A->X->B EVIDENCE —
        #       NOT the fabricated collapsed end-to-end triple (which
        #       has no composing path and never coheres).  Recall
        #       reinforces evidence; inference writes conclusions —
        #       kept distinct (no double-credit, no fabricated edges).
        #       Survival is STILL governed by earn-or-dissolve: recall
        #       buys candidacy and transient engagement, NEVER
        #       permanence or mortality credit — only a later
        #       coherence pass (newly_coherent) does.  See
        #       project_seagi_dry_reverie_igniter.  All other methods
        #       (metacog, counterfactual, schema) still drop.
        if event.method == 'inference':
            self._consolidate_inference(event, bus)
            # Premise-reinforcement (2026-06-05): the chain that PROVED
            # this conclusion — its premises earned strength by being
            # USEFUL for reasoning (a proven premise moves away from
            # death).  Reattest the walked hops through the SAME metered
            # writer path as recall (reinforce + engagement-stamp), so
            # reasoning strengthens its own fuel and a reason-able core
            # becomes self-sustaining.  Gated by the same confidence
            # floor that let the inference fire — only genuinely-
            # composable premises earn.
            if event.edges_walked:
                self._reattest_recall(event, bus)
        elif event.method == 'causal':
            self._reattest_recall(event, bus)
        # else: drop.

    def _consolidate_inference(self,
                                  event: ThoughtProducedEvent,
                                  bus: EventBus) -> None:
        """Write the composed CONCLUSION of an R.1 inference as a
        provisional derived edge (unchanged from the original
        ReasoningConsolidator behavior)."""
        if not (event.focal and event.relation and event.target):
            return
        if event.focal == event.target:
            return  # degenerate self-loop, ignore
        from seagi.core.substrate import PROVISIONAL_EDGE_STRENGTH
        try:
            bus.publish(SubstrateWriteQueuedEvent(
                kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                cycle=event.cycle,
                timestamp=time.time(),
                source_capability='reasoning_consolidator',
                origin='internal',
                origin_detail=f'inference:{event.focal}',
                subject=event.focal,
                relation=event.relation,
                object=event.target,
                strength=PROVISIONAL_EDGE_STRENGTH,
                write_reason='inference_consolidate'))
            self.consolidations_emitted += 1
        except Exception:
            pass

    def _reattest_recall(self,
                            event: ThoughtProducedEvent,
                            bus: EventBus) -> None:
        """Reinforce the REAL hops a recall thought walked.  Per-hop
        (Phase 2): a 2-hop walk reinforces its two genuine edges, not
        the collapsed (focal, relation, target).  Falls back to the
        single triple for 1-hop / identity / process walks that set
        no edges_walked."""
        hops = (list(event.edges_walked) if event.edges_walked
                else [(event.focal, event.relation, event.target)])
        from seagi.core.substrate import PROVISIONAL_EDGE_STRENGTH
        emitted = 0
        for hop in hops:
            try:
                s, r, o = hop
            except (TypeError, ValueError):
                continue
            if not (s and r and o) or s == o:
                continue  # incomplete / degenerate hop
            try:
                bus.publish(SubstrateWriteQueuedEvent(
                    kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                    cycle=event.cycle,
                    timestamp=time.time(),
                    source_capability='reasoning_consolidator',
                    origin='internal',
                    origin_detail=f'recall:{s}',
                    subject=s,
                    relation=r,
                    object=o,
                    strength=PROVISIONAL_EDGE_STRENGTH,
                    write_reason='recall_reattest'))
                self.recall_writes_emitted += 1
                self._recall_keys.append((s, r, o))
                emitted += 1
            except Exception:
                pass
        # Count a recall THOUGHT only when at least one valid hop was
        # written (degenerate/incomplete walks don't inflate the
        # counter).  recall_writes_emitted counts the hop WRITES.
        if emitted:
            self.recalls_reattested += 1

    def recent_recall_keys(self) -> List[Tuple[str, str, str]]:
        """Read-only snapshot of the recall-reattest ring."""
        return list(self._recall_keys)

    def stats(self) -> Dict[str, Any]:
        return {
            'consolidations_emitted': self.consolidations_emitted,
            'recalls_reattested': self.recalls_reattested,
            'recall_writes_emitted': self.recall_writes_emitted,
        }
