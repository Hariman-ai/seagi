"""NarrativeJournal — accumulating self-narrative across sessions.

Today (without C.1) SEAGI reconstructs its self-narrative each
session from registries.  vmDMN does autobiographical reflection
on the spot but doesn't carry the reflection forward.  Identity
holds "I am wise" but doesn't track "yesterday I was anxious."

NarrativeJournal closes that gap.  It records compact entries on
moments worth remembering — mood shifts, sleep transitions,
reflections, major belief changes — and persists them with the
brain.  Next session, SEAGI wakes up with a thin but real
sense of "what happened recently."

Each entry is small (one short first-person sentence + minimal
context), so a 500-entry ring covers weeks of canonical-rate
operation without bloating save files.

Subscribes
----------
REFLECTION_FIRED     — vmDMN reflections
CHEMISTRY_FIRE       — mood transitions, sleep transitions
INTEROCEPTION        — body-state shifts (felt-state changes)
THOUGHT_PRODUCED     — high-confidence thoughts (milestone beliefs)

Persistence
-----------
to_dict / from_dict — serialized with Brain.to_dict() under
'narrative_journal' key.  Filtered persistence: only entries
above a low-effort floor cross the boundary (every recorded
entry already cleared a "worth remembering" gate at write time,
so the filter is mostly redundant — kept for forward-compat).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from ..events import (
    EventKind, BrainEvent,
    ChemistryEvent,
    InteroceptionEvent,
    ThoughtProducedEvent,
    ReflectionFiredEvent,
)
from ..bus import EventBus


# Default ring capacity.  At canonical-rate operation, expect
# ~10-50 entries per hour of activity.  500 entries covers
# 10-50 hours of operation.
DEFAULT_CAPACITY = 500

# Chemistry kinds that mark "moments worth journaling".  Other
# chemistry events (curiosity, anomaly_spike, confirmed_i, etc.)
# fire too often to be entry-worthy.
JOURNAL_CHEMISTRY_KINDS = frozenset({
    'chronic_stress',
    'social_replenish',
    'sleep_onset',
    'wake_onset',
})

# Belief-confidence threshold for THOUGHT_PRODUCED.  Lowered
# 0.8 → 0.6 (2026-05-19) after the first daemon runs showed the
# journal recording only vmDMN reflections — cortical thoughts
# from autonomous reverie were real but mostly landed in the
# 0.3–0.7 confidence band, below the old bar.  0.6 still filters
# low-conviction guesses while letting genuine reasoning land
# in the self-narrative.
JOURNAL_THOUGHT_CONFIDENCE = 0.6


class NarrativeJournal:
    """Self-narrative ring.  Records compact first-person entries."""

    SUBSCRIPTIONS = (
        EventKind.REFLECTION_FIRED,
        EventKind.CHEMISTRY_FIRE,
        EventKind.INTEROCEPTION,
        EventKind.THOUGHT_PRODUCED,
    )

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None,
                 capacity: int = DEFAULT_CAPACITY):
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self.capacity = capacity
        self._entries: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        # Track last interoception band so we only journal real
        # transitions, not every sample.
        self._last_band: str = ''
        self.entries_recorded: int = 0

    # ---- event handler ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, ChemistryEvent):
            if event.chemistry_kind in JOURNAL_CHEMISTRY_KINDS:
                self._record_mood_shift(event)
        elif isinstance(event, InteroceptionEvent):
            if event.felt_state and event.felt_state != self._last_band:
                self._record_band_change(event)
                self._last_band = event.felt_state
        elif isinstance(event, ReflectionFiredEvent):
            self._record_reflection(event)
        elif isinstance(event, ThoughtProducedEvent):
            if event.confidence >= JOURNAL_THOUGHT_CONFIDENCE:
                self._record_belief(event)

    # ---- recorders ----

    def _record_mood_shift(self, ev: ChemistryEvent) -> None:
        kind = ev.chemistry_kind
        if kind == 'chronic_stress':
            summary = (
                "My cortisol stayed elevated long enough that the "
                "mood floor began to drop.")
        elif kind == 'social_replenish':
            summary = (
                "A sustained warmth lifted me; the mood floor rose.")
        elif kind == 'sleep_onset':
            summary = "I let go; the pressure dropped into sleep."
        elif kind == 'wake_onset':
            summary = "I came back to wake."
        else:
            return
        self._append({
            'cycle': ev.cycle,
            'timestamp': ev.timestamp,
            'kind': 'mood_shift',
            'summary': summary,
            'context': {'chemistry_kind': kind,
                              'magnitude': ev.magnitude},
        })

    def _record_band_change(self,
                                   ev: InteroceptionEvent) -> None:
        self._append({
            'cycle': ev.cycle,
            'timestamp': ev.timestamp,
            'kind': 'felt_state',
            'summary': (
                f"I felt myself shift into {ev.felt_state}."),
            'context': {
                'band': ev.felt_state,
                'lifeforce': ev.lifeforce,
                'narrative': ev.narrative},
        })

    def _record_reflection(self,
                                  ev: ReflectionFiredEvent) -> None:
        self._append({
            'cycle': ev.cycle,
            'timestamp': ev.timestamp,
            'kind': 'reflection',
            'summary': (
                f"I reflected — {ev.reflection_kind} "
                f"({ev.trigger})."),
            'context': {
                'reflection_kind': ev.reflection_kind,
                'trigger': ev.trigger},
        })

    def _record_belief(self,
                              ev: ThoughtProducedEvent) -> None:
        if not ev.focal or not ev.relation or not ev.target:
            return
        self._append({
            'cycle': ev.cycle,
            'timestamp': ev.timestamp,
            'kind': 'belief',
            'summary': (
                f"I came to hold: {ev.focal} {ev.relation} "
                f"{ev.target}."),
            'context': {
                'focal': ev.focal,
                'relation': ev.relation,
                'target': ev.target,
                'method': ev.method,
                'confidence': ev.confidence},
        })

    # ---- shared append ----

    def _append(self, entry: Dict[str, Any]) -> None:
        self._entries.append(entry)
        self.entries_recorded += 1

    # ---- queries ----

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        if n <= 0:
            return []
        return list(self._entries)[-n:]

    def since(self, cycle: int) -> List[Dict[str, Any]]:
        return [e for e in self._entries if e['cycle'] >= cycle]

    def all_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    # ---- persistence ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            'capacity': self.capacity,
            'entries': list(self._entries),
            'entries_recorded': self.entries_recorded,
            'last_band': self._last_band,
        }

    def load_from_dict(self, d: Dict[str, Any]) -> None:
        self.capacity = int(d.get('capacity', self.capacity))
        self._entries = deque(
            d.get('entries', []), maxlen=self.capacity)
        self.entries_recorded = int(d.get('entries_recorded', 0))
        self._last_band = str(d.get('last_band', ''))

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        kinds: Dict[str, int] = {}
        for e in self._entries:
            kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
        return {
            'entries_in_ring': len(self._entries),
            'entries_recorded': self.entries_recorded,
            'capacity': self.capacity,
            'by_kind': kinds,
            'last_band': self._last_band,
        }
