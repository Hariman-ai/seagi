"""ACC — Anterior Cingulate Cortex.  Conflict / mismatch detector.

Brain analog: ACC fires when reality diverges from prediction,
when claims conflict, when a planned action runs into an
unexpected obstacle.  It is the brain's "something doesn't add
up" signal.

What ACC fires on
-----------------
1. PREDICTION_ERROR (cerebellum) — wrap as CONFLICT_DETECTED
   when magnitude crosses threshold.
2. Peer ATTENDED_PERCEPT whose focal contradicts a recent
   substrate edge (e.g. peer says "fire is wet" — we hold
   fire→hot→opposite_of→wet).
3. THOUGHT_PRODUCED whose claim flips a very recent cortical
   thought (cognitive self-contradiction).

What it emits
-------------
CONFLICT_DETECTED with payload + narrative.
CHEMISTRY_FIRE 'anomaly_spike' so global state shifts toward
alert.

Phase 4b: FULL.  Substrate-contradiction check uses the v1
`opposite_of` relation as proxy; full opposite-resolution logic
is intentionally minimal — the brain just FLAGS it, then
cortical / vmDMN reason about how to resolve.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from ..events import (
    EventKind, BrainEvent,
    AttendedPerceptEvent,
    ThoughtProducedEvent,
    PredictionErrorEvent,
    ConflictDetectedEvent,
    ChemistryEvent,
)
from ..bus import EventBus


PE_CONFLICT_THRESHOLD = 0.25
RECENT_THOUGHTS = 6


class AnteriorCingulateCortex:
    """The mismatch detector."""

    SUBSCRIPTIONS = (
        EventKind.PREDICTION_ERROR,
        EventKind.ATTENDED_PERCEPT,
        EventKind.THOUGHT_PRODUCED,
    )

    def __init__(self,
                 bus: EventBus,
                 lts_provider: Callable,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._lts_provider = lts_provider
        self._cycle_provider = cycle_provider or (lambda: 0)
        # Bounded recent-thoughts cache.
        self._recent_thoughts: Deque[ThoughtProducedEvent] = (
            deque(maxlen=RECENT_THOUGHTS))
        # Diagnostics.
        self.conflicts_fired: int = 0
        self.pe_conflicts: int = 0
        self.peer_contradictions: int = 0
        self.self_contradictions: int = 0

    # ---- event handler ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, PredictionErrorEvent):
            self._on_prediction_error(event, bus)
        elif isinstance(event, AttendedPerceptEvent):
            if event.origin == 'peer':
                self._check_peer_contradiction(event, bus)
        elif isinstance(event, ThoughtProducedEvent):
            self._check_self_contradiction(event, bus)
            self._recent_thoughts.append(event)

    # ---- prediction-error → conflict ----

    def _on_prediction_error(self,
                                    ev: PredictionErrorEvent,
                                    bus: EventBus) -> None:
        if ev.magnitude < PE_CONFLICT_THRESHOLD:
            return
        self._fire_conflict(
            bus, conflict_kind='prediction',
            magnitude=ev.magnitude,
            focals=[ev.focal] if ev.focal else [],
            narrative=(
                f"My prediction about {ev.focal or 'self'} "
                f"missed by {ev.magnitude:.2f}."),
            cycle=ev.cycle,
            also_anomaly=True)
        self.pe_conflicts += 1

    # ---- peer-contradiction check ----

    def _check_peer_contradiction(self,
                                            ev: AttendedPerceptEvent,
                                            bus: EventBus) -> None:
        """If peer asserts a focal that holds an `opposite_of`
        edge with something currently in substrate, flag.  Cheap
        scan: per focal, fetch its opposite_of neighbors, see
        if any are also in the same percept."""
        lts = self._lts_provider() if self._lts_provider else None
        if lts is None:
            return
        focals = list(ev.focals or [])
        if len(focals) < 2:
            return
        focal_set = set(focals)
        for f in focals:
            opposites = lts.neighbors(f, relation='opposite_of')
            for tgt, _r, _s in opposites:
                if tgt in focal_set and tgt != f:
                    self._fire_conflict(
                        bus, conflict_kind='peer_contradiction',
                        magnitude=0.6,
                        focals=[f, tgt],
                        narrative=(
                            f"Peer mentioned {f} and {tgt} "
                            "together but substrate holds "
                            "them as opposites."),
                        cycle=ev.cycle,
                        also_anomaly=True)
                    self.peer_contradictions += 1
                    return  # one is enough

    # ---- self-contradiction check ----

    def _check_self_contradiction(self,
                                            ev: ThoughtProducedEvent,
                                            bus: EventBus) -> None:
        """A new cortical thought whose target opposes a recent
        thought's target for the same focal."""
        if not ev.focal or not ev.target:
            return
        lts = self._lts_provider() if self._lts_provider else None
        for prior in self._recent_thoughts:
            if prior.focal != ev.focal or not prior.target:
                continue
            if prior.target == ev.target:
                continue
            if lts is not None:
                opps = lts.neighbors(
                    prior.target, relation='opposite_of')
                if any(t == ev.target for t, _r, _s in opps):
                    self._fire_conflict(
                        bus, conflict_kind='thought_thought',
                        magnitude=0.5,
                        focals=[ev.focal,
                                  prior.target, ev.target],
                        narrative=(
                            f"My thought about {ev.focal} "
                            f"reached {ev.target}, but earlier "
                            f"I reached {prior.target} which "
                            "is the opposite."),
                        cycle=ev.cycle,
                        also_anomaly=False)
                    self.self_contradictions += 1
                    return

    # ---- helpers ----

    def _fire_conflict(self,
                            bus: EventBus,
                            conflict_kind: str,
                            magnitude: float,
                            focals: List[str],
                            narrative: str,
                            cycle: int,
                            also_anomaly: bool = False) -> None:
        try:
            bus.publish(ConflictDetectedEvent(
                kind=EventKind.CONFLICT_DETECTED,
                cycle=cycle,
                timestamp=time.time(),
                source_capability='acc',
                origin='internal',
                origin_detail=conflict_kind,
                conflict_kind=conflict_kind,
                magnitude=magnitude,
                focals=list(focals),
                narrative=narrative,
            ))
            self.conflicts_fired += 1
        except Exception:
            pass
        if also_anomaly:
            try:
                bus.publish(ChemistryEvent(
                    kind=EventKind.CHEMISTRY_FIRE,
                    cycle=cycle,
                    timestamp=time.time(),
                    source_capability='acc',
                    origin='internal',
                    origin_detail=conflict_kind,
                    chemistry_kind='anomaly_spike',
                    magnitude=max(0.3, magnitude),
                    target_concepts=list(focals),
                ))
            except Exception:
                pass

    def stats(self) -> Dict[str, int]:
        return {
            'conflicts_fired': self.conflicts_fired,
            'pe_conflicts': self.pe_conflicts,
            'peer_contradictions': self.peer_contradictions,
            'self_contradictions': self.self_contradictions,
        }
