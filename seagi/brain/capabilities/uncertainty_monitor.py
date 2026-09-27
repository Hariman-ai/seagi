"""Uncertainty Monitor — epistemic-state sentinel.

Brain analog: ACC + anterior PFC working together to track
epistemic discomfort.  Always-open monitor that accumulates
unresolved conflicts and thin-substrate thoughts; when the
load is high enough, it claims attention to resolve.

Why this exists
---------------
Without it, SEAGI can chug through individual conflicts and
thin-thought metacog moments without ever stepping back to
ACTUALLY resolve any of them.  Each one fires its event,
nobody integrates the load.  This sentinel is the integrator:
it watches the rate of unresolved epistemic events and forces
a "stop and resolve" claim when the pressure crosses a
sensible bar.

Distinct from Amygdala / NAcc / NoveltyMonitor: those react to
INPUT (perception, chemistry).  Uncertainty Monitor reacts to
SEAGI's own cognitive STATE — the accumulation of his own
unresolved thinking.

What it watches
---------------
1. ConflictDetectedEvent — ACC's mismatch events.
2. ThoughtProducedEvent — when cortical flags thin_substrate.

What it emits
-------------
CapabilityClaim — proposed_action='resolve_uncertainty:<focal>',
    cognitive loop, strength scaled by accumulated pressure.
    Picks the focal most frequently appearing in the recent
    unresolved set (the one most worth attacking).

Refractory: doesn't fire the same claim repeatedly within a
window.  We want SEAGI to step back when load is real, not
spam the same resolution claim forever.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple

from ..events import (
    EventKind, BrainEvent,
    ConflictDetectedEvent,
    ThoughtProducedEvent,
    CapabilityClaimEvent,
)
from ..bus import EventBus
from .goal_tracker import EARN_CONFIDENCE_MIN


# Sliding window of recent unresolved events.
WINDOW_SIZE = 10
# Pressure threshold: when window count crosses this, fire.
PRESSURE_THRESHOLD = 5
# Base claim strength.  Add (pressure / WINDOW_SIZE) * 0.3.
UNCERTAINTY_CLAIM_STRENGTH_FLOOR = 0.45
# Refractory window — don't re-fire within this many cycles.
REFRACTORY_CYCLES = 50


class UncertaintyMonitor:
    """Epistemic-load tracker.  Always-open.  Fires when
    accumulated unresolved conflicts/thin-thoughts exceed
    threshold."""

    SUBSCRIPTIONS = (
        EventKind.CONFLICT_DETECTED,
        EventKind.THOUGHT_PRODUCED,
    )

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._unresolved: Deque[Tuple[int, str]] = deque(
            maxlen=WINDOW_SIZE)
        # Habituation = MORTALITY on the uncertainty (2026-06-04).
        # Per-focal count of claims fired for it WITHOUT resolution.
        # An uncertainty earns continued grip only by being RESOLVED;
        # one you keep failing to resolve loses pressure (count minus
        # habituation), and once its net pressure hits zero it stops
        # driving — the agent gives up on its unsolvables and attention
        # falls through to exploration.  Without this, a self-
        # referential focal ("doubts") that re-records itself every
        # reflection holds full pressure FOREVER (the immortal-
        # uncertainty disease).  Habituation is ITSELF mortal: it
        # resets when a focal leaves the window.  No new tuned constant.
        self._habituation: Dict[str, int] = {}
        self._last_claim_cycle: int = -10**6
        self.uncertainties_recorded: int = 0
        self.claims_emitted: int = 0
        self.habituated_suppressions: int = 0

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, ConflictDetectedEvent):
            focal = event.focals[0] if event.focals else ''
            self._record(event.cycle, focal)
            self._maybe_fire(bus)
        elif isinstance(event, ThoughtProducedEvent):
            if event.thin_substrate:
                self._record(event.cycle, event.focal)
                self._maybe_fire(bus)
            elif event.focal and getattr(
                    event, 'confidence', 0.0) >= EARN_CONFIDENCE_MIN:
                # A CONFIDENT, non-thin thought (>= the shared earn bar)
                # = the agent thought WELL about this focal = the
                # uncertainty is RESOLVED.  Clear it from the window and
                # reset its habituation.  audit #4c/#7 (2026-06-04): a
                # weak hedge (conf 0.2-0.3, non-thin) no longer discharges
                # a real uncertainty — only a genuine resolution does, the
                # SAME bar check_thought_completion uses to EARN the goal,
                # so 'uncertainty resolved' and 'goal earned its keep' are
                # ONE productive signal sharing ONE threshold.  An
                # unresolvable uncertainty stays pressing and is cleared
                # only by habituation (the designed give-up escape).
                self._resolve(event.focal)

    def _record(self, cycle: int, focal: str) -> None:
        self._unresolved.append((cycle, focal))
        self.uncertainties_recorded += 1

    def _maybe_fire(self, bus: EventBus) -> None:
        if len(self._unresolved) < PRESSURE_THRESHOLD:
            return
        now = self._cycle_provider()
        if now - self._last_claim_cycle < REFRACTORY_CYCLES:
            return
        counts = Counter(f for (_c, f) in self._unresolved if f)
        if not counts:
            return
        # Pick the focal with the highest NET pressure (occurrences
        # MINUS habituation).  A focal you keep failing to resolve is
        # habituated down; once EVERY focal's net pressure is <= 0 the
        # agent has given up on its unsolvables — fire NOTHING, so the
        # goal layer opens no goal and attention falls through to
        # exploration.  This is the escape the immortal loop never had.
        best_focal, best_net = None, 0
        for f, c in counts.items():
            net = c - self._habituation.get(f, 0)
            if net > best_net:
                best_net, best_focal = net, f
        self._prune_habituation(counts)
        if best_focal is None:
            self.habituated_suppressions += 1
            return
        # Attended again, still unresolved → habituate one notch (it
        # must EARN relief by being resolved, not by being re-noticed).
        self._habituation[best_focal] = (
            self._habituation.get(best_focal, 0) + 1)
        pressure = best_net / float(WINDOW_SIZE)
        claim_strength = min(
            1.0,
            UNCERTAINTY_CLAIM_STRENGTH_FLOOR + pressure * 0.3)
        bus.publish(CapabilityClaimEvent(
            kind=EventKind.CAPABILITY_CLAIM,
            cycle=now,
            timestamp=time.time(),
            source_capability='uncertainty_monitor',
            origin='internal',
            origin_detail='accumulated',
            claim_strength=claim_strength,
            proposed_action=f'resolve_uncertainty:{best_focal}',
            loop='cognitive',
            payload={'window_count': len(self._unresolved),
                        'pressure': pressure},
        ))
        self._last_claim_cycle = now
        self.claims_emitted += 1

    def _resolve(self, focal: str) -> None:
        """A non-thin thought resolved `focal` — clear it from the
        unresolved window and reset its habituation (the mortality
        earn: resolving wipes the slate clean)."""
        if not focal:
            return
        kept = [(c, f) for (c, f) in self._unresolved if f != focal]
        if len(kept) != len(self._unresolved):
            self._unresolved = deque(kept, maxlen=WINDOW_SIZE)
        self._habituation.pop(focal, None)

    def _prune_habituation(self, counts) -> None:
        """Habituation is itself MORTAL: a focal that has left the
        window (no longer pressing) loses its habituation, so it can
        be reconsidered fresh if it ever genuinely returns."""
        for f in list(self._habituation):
            if f not in counts:
                self._habituation.pop(f, None)

    def top_unresolved_focals(self, k: int = 3):
        """Read-only (Phase 2 goal layer, 2026-06-01): the currently-
        pressing felt-gap focals.  Both thin-substrate metacog
        thoughts AND conflicts (ACC) land in `_unresolved` (see
        handle()), so this window is the brain's AGGREGATE epistemic-
        gap surface — the GoalSpawner reads it to open goals on what
        the agent actually feels it cannot resolve, steering reverie
        there.  Returns [(focal, pressure)] for the k most-frequent
        focals, pressure = occurrences / WINDOW_SIZE in [0, 1] —
        derived from the existing window, no new constant."""
        counts = Counter(f for (_c, f) in self._unresolved if f)
        if not counts:
            return []
        # NET pressure (occurrences minus habituation): a focal the
        # agent has habituated to (net <= 0) drops out, so the goal
        # spawner stops opening goals for it — the same mortality the
        # claim path applies, applied to the goal-layer read.
        scored = []
        for f, c in counts.items():
            net = c - self._habituation.get(f, 0)
            if net > 0:
                scored.append((f, net / float(WINDOW_SIZE)))
        scored.sort(key=lambda p: -p[1])
        return scored[:k]

    def stats(self) -> Dict[str, Any]:
        return {
            'uncertainties_recorded': self.uncertainties_recorded,
            'claims_emitted': self.claims_emitted,
            'window_size': len(self._unresolved),
            'habituated_suppressions': self.habituated_suppressions,
            'habituated_focals': len(self._habituation),
        }
