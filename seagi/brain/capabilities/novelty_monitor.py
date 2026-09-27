"""Novelty Monitor — curiosity-driven sentinel.

Brain analog: hippocampal CA1 novelty detector + VTA
curiosity-reward loop.  Distinct from Amygdala (M-side threat)
and Nucleus Accumbens (I-side approach) — this is the
*exploratory* always-open monitor: notices when something is
NEW, regardless of M/I polarity, and pulls attention toward
investigating.

The doctrine: the always-open monitor landscape is plural, not
a single M ↔ I axis.  Curiosity is its own sentinel — a child
encountering snow doesn't care that snow is neutral; he cares
that it's novel.

What it watches
---------------
1. AttendedPerceptEvent — novelty above threshold (the gate's
   novelty score has already been computed; we pick high-
   novelty cases for claim-firing).
2. ChemistryEvent — strong 'curiosity' chemistry fires (these
   are the substrate-side curiosity signal; if it's loud
   enough to be a sentinel-worth event, claim attention).

What it emits
-------------
CapabilityClaim — proposed_action='explore_novel:<focal>',
    loop='cognitive', strength roughly 0.4 + novelty × 0.2.
    LOWER than threat (0.7) and opportunity (0.5) base — novel
    things deserve attention but yield to urgent ones.  Drives
    cortical to think about the novel focal even when peer
    input is on something else.

Why no speech override
----------------------
Curiosity-driven exploration is INTERNAL.  When SEAGI is mid-
conversation and a novel concept appears in the peer's input,
the peer-driven speech still wins; novelty fires internal
thinking on the side.  Symmetric to shift_focus arbitration
handling.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from ..events import (
    EventKind, BrainEvent,
    AttendedPerceptEvent,
    ChemistryEvent,
    CapabilityClaimEvent,
)
from ..bus import EventBus


# Above this percept novelty, fire.
NOVELTY_PERCEPT_THRESHOLD = 0.5
# (audit #17, 2026-06-04) CURIOSITY_CHEMISTRY_MIN removed — the 0.4
# floor silently ate every 0.3 reverie curiosity event; deleted with
# its use so the emitter's own gate + earn-or-dissolve govern.
# Base claim strength.  Add magnitude * 0.2.
NOVELTY_CLAIM_STRENGTH_FLOOR = 0.4


class NoveltyMonitor:
    """Curiosity-driven sentinel.  Always-open.  Pushes
    explore_novel claims when something genuinely new appears
    in perception or substrate chemistry."""

    SUBSCRIPTIONS = (
        EventKind.ATTENDED_PERCEPT,
        EventKind.CHEMISTRY_FIRE,
    )

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self.novel_detected: int = 0
        self.claims_emitted: int = 0

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, AttendedPerceptEvent):
            self._on_percept(event, bus)
        elif isinstance(event, ChemistryEvent):
            self._on_chemistry(event, bus)

    def _on_percept(self,
                          ev: AttendedPerceptEvent,
                          bus: EventBus) -> None:
        if ev.novelty < NOVELTY_PERCEPT_THRESHOLD:
            return
        if not ev.focals:
            return
        focal = ev.focals[0]
        magnitude = min(1.0, ev.novelty)
        self._fire(
            bus, source_kind='percept',
            focal=focal, magnitude=magnitude,
            cycle=ev.cycle)

    def _on_chemistry(self,
                            ev: ChemistryEvent,
                            bus: EventBus) -> None:
        # Skip our own loop and chemistry from modulators.
        if ev.source_capability in (
                'novelty_monitor', 'vta', 'lc'):
            return
        if ev.chemistry_kind != 'curiosity':
            return
        # Audit #17 (2026-06-04): removed the CURIOSITY_CHEMISTRY_MIN
        # floor (0.4).  It silently ate every reverie curiosity event
        # (IdleMotivation fires at magnitude 0.3 < 0.4), so internal
        # exploration never produced an explore_novel claim.  The
        # emitter already gates when it fires, and any junk focal faces
        # BG arbitration + earn-or-dissolve downstream — the consumer
        # need not re-floor it.  Two hand-tuned constants that had to
        # agree but never did; deleting the floor removes the mismatch.
        focal = ev.target_concepts[0] if ev.target_concepts else ''
        if not focal:
            return
        self._fire(
            bus, source_kind='chemistry',
            focal=focal, magnitude=ev.magnitude,
            cycle=ev.cycle)

    def _fire(self,
                bus: EventBus,
                source_kind: str,
                focal: str,
                magnitude: float,
                cycle: int) -> None:
        self.novel_detected += 1
        claim_strength = min(
            1.0,
            NOVELTY_CLAIM_STRENGTH_FLOOR + magnitude * 0.2)
        bus.publish(CapabilityClaimEvent(
            kind=EventKind.CAPABILITY_CLAIM,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='novelty_monitor',
            origin='internal',
            origin_detail=source_kind,
            claim_strength=claim_strength,
            proposed_action=f'explore_novel:{focal}',
            loop='cognitive',
            payload={'source_kind': source_kind,
                        'novelty': magnitude},
        ))
        self.claims_emitted += 1

    def stats(self) -> Dict[str, int]:
        return {
            'novel_detected': self.novel_detected,
            'claims_emitted': self.claims_emitted,
        }
