"""Auxiliary capabilities — Source Monitor, AnteriorPFC,
TimePerception, BodySchema, CorpusCallosum.

These are smaller capabilities that share the brain's "passive
auditor / utility" character — they don't drive behavior by
themselves, they shape or annotate what other capabilities do.
Keeping them in one file makes the relationships visible.

Phase 4b: FULL implementations.

  SourceMonitor   — wildcard auditor; tags integrity of every
                     emitted event.  Critical for self/other
                     distinction.
  AnteriorPFC     — metacognition + prospective intent.  Watches
                     thoughts and conflicts; sets prospective
                     'return to X' intents.
  TimePerception  — multi-scale time tracker.  Queried by other
                     capabilities for 'since-X' answers.
  BodySchema      — operational envelope.  Wraps integrity /
                     lifeforce / available channels into a
                     queryable map.
  CorpusCallosum  — two parallel processing streams + integration
                     when they disagree.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from ..events import (
    EventKind, BrainEvent,
    AttendedPerceptEvent,
    ThoughtProducedEvent,
    ConflictDetectedEvent,
    SubstrateWriteQueuedEvent,
    EpisodeFormedEvent,
)
from ..bus import EventBus


# ---------------------------------------------------------------
# Source Monitor — self/other distinction auditor
# ---------------------------------------------------------------


class SourceMonitor:
    """Wildcard auditor.  Inspects every event for source-
    monitoring integrity: origin set, origin_detail consistent
    with source_capability.  Flags anomalies (events with
    inconsistent or missing origin tags) for diagnostic use.

    Brain analog: TPJ + mPFC.  Continuously asks 'where did this
    come from' for every cognitive product.
    """

    SUBSCRIPTIONS = ()   # wildcard via subscribe_all

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.events_seen: int = 0
        self.untagged_count: int = 0
        self.recent_untagged: List[Dict[str, Any]] = []
        self.origin_counts: Dict[str, int] = defaultdict(int)
        self.capability_counts: Dict[str, int] = defaultdict(int)

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        self.events_seen += 1
        self.origin_counts[event.origin or '(empty)'] += 1
        self.capability_counts[
            event.source_capability or '(empty)'] += 1
        if not event.origin or not event.source_capability:
            self.untagged_count += 1
            if len(self.recent_untagged) >= 20:
                self.recent_untagged.pop(0)
            self.recent_untagged.append({
                'kind': str(event.kind),
                'cycle': event.cycle,
                'source_capability': event.source_capability,
                'origin': event.origin,
                'origin_detail': event.origin_detail,
            })

    def tag_for(self, event: BrainEvent) -> Dict[str, str]:
        """Return the source-monitoring tag a downstream
        consumer can attach to derived state.  Phase 4b: this
        is the contract; SourceMonitor is the canonical
        answerer."""
        return {
            'origin': event.origin or '',
            'origin_detail': event.origin_detail or '',
            'source_capability': event.source_capability or '',
        }

    def stats(self) -> Dict[str, Any]:
        return {
            'events_seen': self.events_seen,
            'untagged_count': self.untagged_count,
            'untagged_rate': (
                self.untagged_count / max(1, self.events_seen)),
            'origin_distribution': dict(self.origin_counts),
            'capability_distribution': dict(
                self.capability_counts),
        }


# ---------------------------------------------------------------
# Anterior PFC — metacognition + prospective intent
# ---------------------------------------------------------------


class AnteriorPFC:
    """Watches the agent's own processes.

    Two specific functions:
      1. After a CONFLICT_DETECTED, set a prospective intent
         to return to the conflicting focals later.
      2. Score recent thoughts for 'productive vs unproductive'
         (cheap heuristic: confidence + non-thin substrate).
         Surface as a metacog claim if recent thoughts are all
         unproductive — request that arbitration shift.
    """

    SUBSCRIPTIONS = (
        EventKind.THOUGHT_PRODUCED,
    )

    UNPRODUCTIVE_WINDOW = 5
    UNPRODUCTIVE_CONFIDENCE = 0.25

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._recent_thoughts: Deque[ThoughtProducedEvent] = (
            deque(maxlen=self.UNPRODUCTIVE_WINDOW))
        self.claims_emitted: int = 0

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, ThoughtProducedEvent):
            self._on_thought(event, bus)

    def _on_thought(self,
                          ev: ThoughtProducedEvent,
                          bus: EventBus) -> None:
        self._recent_thoughts.append(ev)
        if len(self._recent_thoughts) < self.UNPRODUCTIVE_WINDOW:
            return
        all_unproductive = all(
            (t.thin_substrate
              or t.confidence < self.UNPRODUCTIVE_CONFIDENCE)
            for t in self._recent_thoughts)
        if all_unproductive:
            # Emit a metacog claim asking arbitration to shift
            # focus.  Brain-correct: PFC asking BG to switch
            # task.
            from ..events import CapabilityClaimEvent
            bus.publish(CapabilityClaimEvent(
                kind=EventKind.CAPABILITY_CLAIM,
                cycle=ev.cycle,
                timestamp=time.time(),
                source_capability='anterior_pfc',
                origin='internal',
                origin_detail='metacog',
                claim_strength=0.5,
                proposed_action='shift_focus',
                loop='cognitive',
                payload={'reason': 'unproductive_streak'},
            ))
            self.claims_emitted += 1

    # audit #10 (2026-06-04): the conflict -> prospective-intent sink
    # (_set_intent / _prospective_intents / pending_intents) was
    # SUBTRACTED — it stored {focals} on every ConflictDetected but had
    # NO live reader (the deque just overwrote at cap 16), so the
    # designed "return to the conflicting focal later" action never
    # reached BG.  Re-attention to conflicts is already driven by the
    # live conflict -> UncertaintyMonitor -> GoalSpawner path; a second,
    # dead path is liability.  Kept: the unproductive-streak shift_focus
    # claim (_on_thought), which IS wired.

    def stats(self) -> Dict[str, Any]:
        return {
            'claims_emitted': self.claims_emitted,
            'recent_thought_count': len(self._recent_thoughts),
        }


# ---------------------------------------------------------------
# Time Perception — multi-scale tracker
# ---------------------------------------------------------------


class TimePerception:
    """Multi-scale time tracker.  Records the cycle of last
    occurrence per event kind + per concept, supports 'since X'
    queries.

    Four scales running over the same data:
      rhythmic    — last 8 inter-event intervals per kind
      interval    — cycles since-last for each kind
      long_range  — cycles since-first for each kind
      episodic    — cycles since each named episode (focals)
    """

    SUBSCRIPTIONS = ()   # wildcard via subscribe_all

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._last_cycle_by_kind: Dict[str, int] = {}
        self._first_cycle_by_kind: Dict[str, int] = {}
        self._intervals: Dict[str, Deque[int]] = defaultdict(
            lambda: deque(maxlen=8))
        self._focal_last_cycle: Dict[str, int] = {}

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        kind = str(event.kind)
        cycle = event.cycle
        if kind in self._last_cycle_by_kind:
            interval = cycle - self._last_cycle_by_kind[kind]
            if interval > 0:
                self._intervals[kind].append(interval)
        else:
            self._first_cycle_by_kind[kind] = cycle
        self._last_cycle_by_kind[kind] = cycle
        # Focal tracking from attended percepts.
        if isinstance(event, AttendedPerceptEvent):
            for f in event.focals or []:
                self._focal_last_cycle[f] = cycle

    # ---- queries ----

    def since_last(self, kind: str) -> Optional[int]:
        last = self._last_cycle_by_kind.get(kind)
        if last is None:
            return None
        return self._cycle_provider() - last

    def since_first(self, kind: str) -> Optional[int]:
        first = self._first_cycle_by_kind.get(kind)
        if first is None:
            return None
        return self._cycle_provider() - first

    def mean_interval(self, kind: str) -> Optional[float]:
        ivs = self._intervals.get(kind)
        if not ivs:
            return None
        return sum(ivs) / len(ivs)

    def since_focal(self, focal: str) -> Optional[int]:
        last = self._focal_last_cycle.get(focal)
        if last is None:
            return None
        return self._cycle_provider() - last

    def stats(self) -> Dict[str, Any]:
        return {
            'kinds_tracked': len(self._last_cycle_by_kind),
            'focals_tracked': len(self._focal_last_cycle),
        }


# ---------------------------------------------------------------
# Body Schema — operational envelope
# ---------------------------------------------------------------


class BodySchema:
    """The agent's running model of its operational envelope.

    Reads:
      - Insula (via provider) for body band + lifeforce + integrity
      - Cycle for time available
      - List of known peers (from dmDMN if provider given)

    Queries:
      envelope() -> dict of current operational state
      can_speak() -> bool (false if band is depleted critically)

    Doesn't subscribe to events directly — it's a query API.
    """

    SUBSCRIPTIONS = ()

    def __init__(self,
                 insula_provider: Optional[Callable] = None,
                 dmdmn_provider: Optional[Callable] = None,
                 cycle_provider: Optional[Callable] = None):
        self._insula_provider = insula_provider
        self._dmdmn_provider = dmdmn_provider
        self._cycle_provider = cycle_provider or (lambda: 0)

    def envelope(self) -> Dict[str, Any]:
        lifeforce = 1.0
        integrity = 1.0
        band = 'settled'
        if self._insula_provider is not None:
            try:
                ins = self._insula_provider()
                if ins is not None:
                    felt = ins.felt_state()
                    lifeforce = felt.get('lifeforce', 1.0)
                    integrity = felt.get('body_integrity', 1.0)
                    band = felt.get('band', 'settled')
            except Exception:
                pass
        peers: List[str] = []
        if self._dmdmn_provider is not None:
            try:
                dm = self._dmdmn_provider()
                if dm is not None:
                    peers = dm.peers_known()
            except Exception:
                pass
        return {
            'lifeforce': lifeforce,
            'body_integrity': integrity,
            'band': band,
            'cycle': self._cycle_provider(),
            'peers_known': peers,
            'can_speak': band not in ('depleted',),
        }

    def can_speak(self) -> bool:
        return self.envelope()['can_speak']

    def stats(self) -> Dict[str, Any]:
        return self.envelope()


# ---------------------------------------------------------------
# Corpus Callosum — hemispheric split + integration
# ---------------------------------------------------------------


class CorpusCallosum:
    """Two parallel processing streams over each attended percept:

      STREAM A (local / sequential / familiar-pattern):
          Tries known transitions first.  Fast, conservative.
          Result: the most-frequent successor focal.

      STREAM B (broad / holistic / novel):
          Looks for unusual LTS neighbors — concepts strongly
          related but rarely co-active.  Slower, creative.

    Integration: when both streams produce something, BodySchema-
    aware integration emits a single integrated thought claim.
    When they DISAGREE (different focals) and value differs in
    sign, emit a CONFLICT_DETECTED so ACC handles.

    Phase 4b implementation is intentionally small — the brain
    capability is real but lightweight here.  Future phases
    can deepen each stream.
    """

    SUBSCRIPTIONS = (EventKind.ATTENDED_PERCEPT,)

    def __init__(self,
                 bus: EventBus,
                 lts_provider: Optional[Callable] = None,
                 cerebellum_provider: Optional[Callable] = None,
                 value_provider: Optional[Callable] = None,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._lts_provider = lts_provider
        self._cerebellum_provider = cerebellum_provider
        self._value_provider = value_provider
        self._cycle_provider = cycle_provider or (lambda: 0)
        self.integrations: int = 0
        self.disagreements: int = 0

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if not isinstance(event, AttendedPerceptEvent):
            return
        if not event.focals:
            return
        focal = event.focals[0]
        a = self._stream_a(focal)
        b = self._stream_b(focal)
        if not a and not b:
            return
        if a and b and a != b:
            # Disagreement — only flag if value polarity differs.
            if self._value_provider is not None:
                vp = self._value_provider()
                va = vp.value_of(a) if vp else 0.0
                vb = vp.value_of(b) if vp else 0.0
                if (va > 0 and vb < 0) or (va < 0 and vb > 0):
                    from ..events import ConflictDetectedEvent
                    bus.publish(ConflictDetectedEvent(
                        kind=EventKind.CONFLICT_DETECTED,
                        cycle=event.cycle,
                        timestamp=time.time(),
                        source_capability='corpus_callosum',
                        origin='internal',
                        origin_detail='stream_disagree',
                        conflict_kind='thought_thought',
                        magnitude=0.4,
                        focals=[focal, a, b],
                        narrative=(
                            f'Stream A reaches {a}, '
                            f'stream B reaches {b} '
                            f'for {focal}.'),
                    ))
                    self.disagreements += 1
        self.integrations += 1

    def _stream_a(self, focal: str) -> str:
        """Local / familiar — read cerebellum's most-frequent
        successor."""
        if self._cerebellum_provider is None:
            return ''
        try:
            cer = self._cerebellum_provider()
            if cer is None:
                return ''
            return cer._predict_next(focal) or ''
        except Exception:
            return ''

    def _stream_b(self, focal: str) -> str:
        """Holistic — strongest LTS neighbor of any relation."""
        if self._lts_provider is None:
            return ''
        try:
            lts = self._lts_provider()
            if lts is None:
                return ''
            best = ('', 0.0)
            for n, _r, s in lts.neighbors(focal):
                if s > best[1]:
                    best = (n, s)
            return best[0]
        except Exception:
            return ''

    def stats(self) -> Dict[str, int]:
        return {
            'integrations': self.integrations,
            'disagreements': self.disagreements,
        }
