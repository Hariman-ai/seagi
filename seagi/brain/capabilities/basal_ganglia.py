"""Basal Ganglia — claim arbitration.

Brain analog: striatum + globus pallidus.  Three parallel
sub-loops mediate three different competition spaces:

    COGNITIVE     — which thought to pursue next
    MOTIVATIONAL — which goal / intent gets active claim
    SPEECH       — when to commit to saying something

Capabilities publish CapabilityClaimEvent with `loop=X`.  BG
arbitrates per loop on each tick (or on each new claim, if
configured for low-latency).  Winners emit
ArbitrationDecidedEvent which downstream consumers (motor /
cortical / speech) listen for.

Strategy (per loop)
-------------------
- Collect open claims arriving inside an arbitration window
  (default 1 tick from first claim).
- Pick the claim with highest claim_strength * value_bonus
  where value_bonus reads from ValueLandscape for the focal
  named in proposed_action.
- Emit ArbitrationDecidedEvent + clear that loop's pending.

Phase 4b: FULL but tight.  Per-tick arbitration is simpler than
the actual brain's continuous version; gives clear semantics
without sacrificing brain-correctness in the small.

Phase B.1.b (2026-05-18): serotonin gating.  When a
serotonin_provider is wired, the BG's per-loop arbitration gets
an action threshold that rises with sustained-elevated serotonin
(post social_replenish).  Sub-threshold winners are silently
inhibited — no decision emitted, claims dropped.  At baseline
serotonin, the threshold is 0 (current behavior preserved).
This is the deliberate-vs-impulsive axis: high serotonin =
more deliberation.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional

from ..events import (
    EventKind, BrainEvent,
    CapabilityClaimEvent,
    ArbitrationDecidedEvent,
    ValueUpdatedEvent,
)
from ..bus import EventBus
from ..chemistry_types import CHANNELS


VALID_LOOPS = ('cognitive', 'motivational', 'speech', 'motor')

# Phase B.1.b: serotonin gating constants.  Threshold is
# max(0, SCALE * (serotonin - baseline)) capped at MAX.  At
# baseline this is 0 → current behavior preserved; above
# baseline it rises → deliberation increases.
BG_SEROTONIN_GATE_SCALE = 1.0
BG_SEROTONIN_GATE_MAX = 0.30


class BasalGanglia:
    """Three-loop claim arbitrator."""

    SUBSCRIPTIONS = (
        EventKind.CAPABILITY_CLAIM,
        EventKind.VALUE_UPDATED,
    )

    def __init__(self,
                 bus: EventBus,
                 value_provider: Optional[Callable] = None,
                 cycle_provider: Optional[Callable] = None,
                 reward_provider: Optional[Callable] = None,
                 serotonin_provider: Optional[Callable] = None):
        self.bus = bus
        self._value_provider = value_provider
        self._cycle_provider = cycle_provider or (lambda: 0)
        # Phase F.8 (2026-05-16): reward_provider lets arbitration
        # consult the eligibility-trace credit ledger.  Successful
        # action_kinds bias future winners.  Optional — without
        # it, scoring runs as before.
        self._reward_provider = reward_provider
        # Phase B.1.b (2026-05-18): serotonin_provider returns the
        # current serotonin level.  When above baseline, the BG's
        # action threshold rises and sub-threshold winners are
        # silently inhibited (the deliberate-vs-impulsive axis).
        # At baseline serotonin (0.50) the threshold is 0 — current
        # behavior preserved.  Optional.
        self._serotonin_provider = serotonin_provider
        self._pending: Dict[str, List[CapabilityClaimEvent]] = (
            defaultdict(list))
        self.arbitrations: int = 0
        self.last_winners: Dict[str, str] = {}
        # B.1.b diagnostics.
        self.serotonin_inhibitions: int = 0

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, CapabilityClaimEvent):
            loop = event.loop if event.loop in VALID_LOOPS \
                else 'cognitive'
            self._pending[loop].append(event)
        # VALUE_UPDATED is read at decision time via the value
        # provider; we don't need to cache it.

    # ---- arbitration driver ----

    def arbitrate(self,
                     loop: Optional[str] = None) -> List[ArbitrationDecidedEvent]:
        """Resolve pending claims for the given loop, or all
        loops if None.  Called per tick by the brain runtime.
        Returns the list of decisions emitted (for tests +
        diagnostics)."""
        if loop is None:
            loops = list(VALID_LOOPS)
        else:
            loops = [loop]
        decisions: List[ArbitrationDecidedEvent] = []
        for L in loops:
            d = self._arbitrate_loop(L)
            if d is not None:
                decisions.append(d)
        return decisions

    def _arbitrate_loop(self,
                              loop: str
                              ) -> Optional[ArbitrationDecidedEvent]:
        claims = self._pending.get(loop, [])
        if not claims:
            return None
        scored = []
        for c in claims:
            score = self._score(c)
            scored.append((score, c))
        scored.sort(key=lambda t: -t[0])
        winner_score, winner = scored[0]
        # Phase B.1.b: serotonin gate.  Computed once per loop;
        # sub-threshold winners are inhibited (no decision, claims
        # cleared).  Threshold is 0 at baseline serotonin so
        # canonical behavior is preserved when raphe is quiet.
        threshold = self._serotonin_threshold()
        if winner_score < threshold:
            self._pending[loop] = []
            self.serotonin_inhibitions += 1
            return None
        cycle = self._cycle_provider()
        decision = ArbitrationDecidedEvent(
            kind=EventKind.ARBITRATION_DECIDED,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='basal_ganglia',
            origin='internal',
            origin_detail=loop,
            loop=loop,
            winning_action=winner.proposed_action,
            winning_capability=winner.source_capability,
            winning_strength=winner_score,
            n_competing=len(claims),
        )
        self.bus.publish(decision)
        self._pending[loop] = []
        self.arbitrations += 1
        self.last_winners[loop] = winner.proposed_action
        return decision

    def _serotonin_threshold(self) -> float:
        """Phase B.1.b: action threshold modulated by serotonin.
        Returns 0 when no provider is wired or when serotonin is
        at/below baseline.  Capped at BG_SEROTONIN_GATE_MAX."""
        if self._serotonin_provider is None:
            return 0.0
        try:
            sero = float(self._serotonin_provider())
        except Exception:
            return 0.0
        excess = sero - CHANNELS['serotonin']['baseline']
        if excess <= 0:
            return 0.0
        return min(
            BG_SEROTONIN_GATE_MAX,
            BG_SEROTONIN_GATE_SCALE * excess)

    def _score(self, claim: CapabilityClaimEvent) -> float:
        score = claim.claim_strength
        # Value bonus: read ValueLandscape for the focal in
        # proposed_action if shape is 'verb:focal'.
        if self._value_provider is not None:
            try:
                vp = self._value_provider()
                if vp is not None:
                    focal = ''
                    if ':' in claim.proposed_action:
                        focal = claim.proposed_action.split(
                            ':', 1)[1]
                    if focal:
                        v = float(vp.value_of(focal))
                        score += 0.2 * v
            except Exception:
                pass
        # Phase F.8 (2026-05-16): reward-credit bias.  Look up
        # what experience says about this action_kind — both in
        # the current chemistry context AND aggregated across
        # contexts.  Total bias capped so it never dominates
        # claim_strength; it's a learning signal that lets
        # repeated success nudge future arbitration.
        # The 'motor' loop (WorldActor, the mortal-game test) is deliberately
        # cut off from the reward-credit bias: letting eligibility-trace credit
        # (which integrates lifeforce deltas) steer the survival policy would be
        # reinforcement learning on the score — the exact thing the no-external-
        # reward test forbids.  Motor action is chosen ONLY by the substrate's
        # predicted-next-state lean inside WorldActor.
        if self._reward_provider is not None and claim.loop != 'motor':
            try:
                rl = self._reward_provider()
                if rl is not None:
                    c_ctx = float(rl.credit_for(
                        claim.proposed_action))
                    c_total = float(rl.total_credit_for(
                        claim.proposed_action))
                    # Context-specific bias dominates (more
                    # relevant), but generalized credit
                    # contributes too.
                    score += 0.2 * c_ctx + 0.1 * c_total
            except Exception:
                pass
        return score

    # ---- diagnostics ----

    def pending_count(self, loop: str) -> int:
        return len(self._pending.get(loop, []))

    def stats(self) -> Dict[str, Any]:
        return {
            'arbitrations': self.arbitrations,
            'pending_per_loop': {
                L: len(self._pending.get(L, []))
                for L in VALID_LOOPS},
            'last_winners': dict(self.last_winners),
            'serotonin_inhibitions': self.serotonin_inhibitions,
            'serotonin_threshold': self._serotonin_threshold(),
        }
