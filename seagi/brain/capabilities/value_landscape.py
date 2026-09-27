"""Value Landscape — learned outcome value map.

Brain analog: mesolimbic / orbitofrontal cortex.  An indexed
queryable surface of "what's worth pursuing" emergent from M/I
tagging history.  Edges accumulate value-weight based on
confirmed_i vs falsified_i firing patterns.

Critical contract: this is a STRUCTURED, QUERYABLE map.  Other
capabilities (Basal Ganglia, Cortical) consume it via
`value_of(focal)` — they don't iterate.

Phase 4b: FULL.  Lives as a per-focal accumulator that listens
on PREDICTION_ERROR (signed) + CHEMISTRY_FIRE (confirmed_i /
falsified_i).  Emits VALUE_UPDATED with the new value.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from ..events import (
    EventKind, BrainEvent,
    PredictionErrorEvent,
    ChemistryEvent,
    ValueUpdatedEvent,
)
from ..bus import EventBus


VALUE_BOUND = 1.0
VALUE_DECAY = 0.001    # tiny per-update drift toward neutral
POSITIVE_PE_GAIN = 0.10
NEGATIVE_PE_GAIN = 0.08
CONFIRMED_I_GAIN = 0.05
FALSIFIED_I_GAIN = 0.06


class ValueLandscape:
    """Accumulates a value score in [-1, 1] for each focal."""

    SUBSCRIPTIONS = (
        EventKind.PREDICTION_ERROR,
        EventKind.CHEMISTRY_FIRE,
        EventKind.VALUE_UPDATED,
    )

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._values: Dict[str, float] = {}
        self.updates_total: int = 0

    # ---- queries ----

    def value_of(self, focal: str) -> float:
        return self._values.get(focal, 0.0)

    def top_k_positive(self, k: int = 5):
        return sorted(
            self._values.items(),
            key=lambda kv: -kv[1])[:k]

    def top_k_negative(self, k: int = 5):
        return sorted(
            self._values.items(),
            key=lambda kv: kv[1])[:k]

    # ---- event handler ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, PredictionErrorEvent):
            self._on_prediction_error(event, bus)
        elif isinstance(event, ChemistryEvent):
            self._on_chemistry(event, bus)
        elif isinstance(event, ValueUpdatedEvent):
            # VTA emits a VALUE_UPDATED with delta but value=0;
            # we apply the delta and re-emit (or skip re-emit
            # — origin tracking avoids loops).
            if event.source_capability == 'value_landscape':
                return
            if event.focal and event.delta:
                self._apply(event.focal, event.delta,
                                bus, kind=event.update_kind,
                                cycle=event.cycle)

    def _on_prediction_error(self,
                                    ev: PredictionErrorEvent,
                                    bus: EventBus) -> None:
        if not ev.focal:
            return
        delta = (POSITIVE_PE_GAIN if ev.sign > 0
                   else -NEGATIVE_PE_GAIN) * ev.magnitude
        kind = ('positive_pe' if ev.sign > 0
                  else 'negative_pe')
        self._apply(ev.focal, delta, bus,
                       kind=kind, cycle=ev.cycle)

    def _on_chemistry(self,
                            ev: ChemistryEvent,
                            bus: EventBus) -> None:
        if not ev.target_concepts:
            return
        if ev.chemistry_kind == 'confirmed_i':
            delta = +CONFIRMED_I_GAIN * ev.magnitude
        elif ev.chemistry_kind == 'falsified_i':
            delta = -FALSIFIED_I_GAIN * ev.magnitude
        else:
            return
        for focal in ev.target_concepts[:4]:
            self._apply(focal, delta, bus,
                          kind='chemistry', cycle=ev.cycle)

    def _apply(self,
                 focal: str,
                 delta: float,
                 bus: EventBus,
                 kind: str,
                 cycle: int) -> None:
        cur = self._values.get(focal, 0.0)
        # Drift toward neutral by tiny amount each update — old
        # values fade if not refreshed.
        cur *= (1.0 - VALUE_DECAY)
        new = max(-VALUE_BOUND, min(VALUE_BOUND, cur + delta))
        self._values[focal] = new
        self.updates_total += 1
        try:
            bus.publish(ValueUpdatedEvent(
                kind=EventKind.VALUE_UPDATED,
                cycle=cycle,
                timestamp=time.time(),
                source_capability='value_landscape',
                origin='internal',
                origin_detail=focal,
                focal=focal,
                delta=delta,
                value=new,
                update_kind=kind,
            ))
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        return {
            'focals_known': len(self._values),
            'updates_total': self.updates_total,
            'top_positive': self.top_k_positive(3),
            'top_negative': self.top_k_negative(3),
        }
