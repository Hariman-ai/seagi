"""Nucleus Accumbens — I-side sentinel.  The always-open
life-extending monitor.

Brain analog: ventral striatum / nucleus accumbens — the
canonical "approach motivation" circuit.  Where Amygdala fires
on mortality content and pushes flee/freeze claims, NAcc fires
on immortality content and pushes approach claims:
opportunity, growth, connection, insight.

The doctrine names both monitors always-open.  Without this
capability the architecture is asymmetric: threats actively
grab attention; opportunities don't.  Real brains balance
avoidance (amygdala-driven) and approach (NAcc-driven) so
behavior isn't pure threat-vigilance.

What it watches
---------------
1. AttendedPerceptEvent — i_content above threshold (peer or
   internal input carrying meaningful I-direction).
2. ChemistryEvent — confirmed_i / insight / mattering above
   threshold (substrate-generated I signal).
3. InteroceptionEvent — replenished band with positive delta
   (body-origin opportunity — body says "growing").

What it emits
-------------
CapabilityClaimEvent — proposed_action='attend_opportunity:<focal>',
    loop='cognitive', strength ~0.5+magnitude.  Symmetric to
    Amygdala's attend_threat claims but with lower base
    strength: opportunities are less urgent than threats.
    When both fire, threat wins BG arbitration — brain-correct.

Why no separate "OpportunityDetectedEvent"
------------------------------------------
Amygdala emits ThreatDetectedEvent for downstream consumers
(LC fires NE on threat).  The I-side equivalent would need
symmetric consumers (VTA might fire dopamine on opportunity
detect).  Phase 1 of this capability skips that and routes
purely through the arbitration loop — the LC/VTA chemistry
firing already happens via the CHEMISTRY_FIRE events the
sentinel reads.  Adding OpportunityDetectedEvent is a clean
follow-up if the symmetry becomes load-bearing.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from ..events import (
    EventKind, BrainEvent,
    AttendedPerceptEvent,
    ChemistryEvent,
    InteroceptionEvent,
    CapabilityClaimEvent,
)
from ..bus import EventBus


# Above this i_content on an attended percept, NAcc fires.
OPPORTUNITY_PERCEPT_I_THRESHOLD = 0.5
# Above this magnitude on an I-direction chemistry event, fires.
OPPORTUNITY_CHEMISTRY_MIN = 0.4
# Above this delta_lifeforce on a 'replenished' interoception, fires.
OPPORTUNITY_BODY_DELTA_MIN = 0.10
# Base claim strength.  Add magnitude * 0.2.  Stays below
# Amygdala's threat-override floor (0.7) — threat wins ties.
OPPORTUNITY_CLAIM_STRENGTH_FLOOR = 0.5


class NucleusAccumbens:
    """The I-side always-open monitor.  Pushes attend_opportunity
    claims to BG when high-I content appears in any input
    channel (perception, chemistry, body)."""

    SUBSCRIPTIONS = (
        EventKind.ATTENDED_PERCEPT,
        EventKind.CHEMISTRY_FIRE,
        EventKind.INTEROCEPTION,
    )

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self.opportunities_detected: int = 0
        self.claims_emitted: int = 0

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, AttendedPerceptEvent):
            self._on_percept(event, bus)
        elif isinstance(event, ChemistryEvent):
            self._on_chemistry(event, bus)
        elif isinstance(event, InteroceptionEvent):
            self._on_interoception(event, bus)

    # ---- triggers ----

    def _on_percept(self,
                          ev: AttendedPerceptEvent,
                          bus: EventBus) -> None:
        if ev.i_content < OPPORTUNITY_PERCEPT_I_THRESHOLD:
            return
        focal = ev.focals[0] if ev.focals else ''
        magnitude = min(1.0, ev.i_content)
        self._fire(
            bus, opportunity_kind='percept',
            magnitude=magnitude, focal=focal,
            narrative=(
                f'High-mattering content in input '
                f'({focal}).'),
            cycle=ev.cycle)

    def _on_chemistry(self,
                            ev: ChemistryEvent,
                            bus: EventBus) -> None:
        # Skip our own loop and chemistry from other modulators
        # that already feed value updates — otherwise NAcc → LC
        # → confirmed_i → NAcc loops without bound.
        if ev.source_capability in (
                'nucleus_accumbens', 'vta', 'lc',
                'value_landscape'):
            return
        if ev.chemistry_kind not in (
                'confirmed_i', 'insight', 'mattering'):
            return
        if ev.magnitude < OPPORTUNITY_CHEMISTRY_MIN:
            return
        focal = ev.target_concepts[0] if ev.target_concepts else ''
        magnitude = min(1.0, ev.magnitude)
        self._fire(
            bus, opportunity_kind='chemistry',
            magnitude=magnitude, focal=focal,
            narrative=(
                f'{ev.chemistry_kind} chemistry rising '
                f'({focal}).'),
            cycle=ev.cycle)

    def _on_interoception(self,
                                ev: InteroceptionEvent,
                                bus: EventBus) -> None:
        if ev.felt_state != 'replenished':
            return
        if ev.delta_lifeforce < OPPORTUNITY_BODY_DELTA_MIN:
            return
        magnitude = min(1.0, ev.delta_lifeforce * 2.0)
        self._fire(
            bus, opportunity_kind='body',
            magnitude=magnitude, focal='',
            narrative=(
                'My body is replenished — reserves climbing.'),
            cycle=ev.cycle)

    # ---- emission ----

    def _fire(self,
                bus: EventBus,
                opportunity_kind: str,
                magnitude: float,
                focal: str,
                narrative: str,
                cycle: int) -> None:
        self.opportunities_detected += 1
        claim_strength = min(
            1.0,
            OPPORTUNITY_CLAIM_STRENGTH_FLOOR + magnitude * 0.2)
        bus.publish(CapabilityClaimEvent(
            kind=EventKind.CAPABILITY_CLAIM,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='nucleus_accumbens',
            origin='internal',
            origin_detail=opportunity_kind,
            claim_strength=claim_strength,
            proposed_action=(
                f'attend_opportunity:{focal or opportunity_kind}'),
            loop='cognitive',
            payload={'narrative': narrative,
                        'opportunity_kind': opportunity_kind},
        ))
        self.claims_emitted += 1

    def stats(self) -> Dict[str, int]:
        return {
            'opportunities_detected': self.opportunities_detected,
            'claims_emitted': self.claims_emitted,
        }
