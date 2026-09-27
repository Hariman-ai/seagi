"""IdleMotivation — the reason to think when nothing's happening.

In a function-call model (Brain.chat → tick → exit), idle is
death.  The agent only exists when something asks it to.  C.1
wants SEAGI to *exist on its own*.  That requires a reason to
keep cycling when the bus has gone quiet.

IdleMotivation tracks how many ticks have passed without an
attended percept or produced thought.  When the quiet stretch
crosses IDLE_TICK_THRESHOLD, it picks something SEAGI was
recently attending to (AWM-resident, highest salience) and
fires a `curiosity` ChemistryEvent for it — a small inner nudge
that's indistinguishable, downstream, from an externally-triggered
wonder.  The chemistry chain handles it from there: cortical
walks may produce a thought, vmDMN may reflect on it, schemas
may match.

After firing, IdleMotivation enters a cooldown so it doesn't
chatter.  If the bus stays quiet, it can fire again after the
cooldown — slow internal rhythm.

Doctrine fit
------------
- promille curiosity already in EVENT_DELTAS
- fire-when-needed gated by long quiet stretch + cooldown
- no new event types
- wires existing AWM + chemistry channels

Subscribes
----------
ATTENDED_PERCEPT     — resets the idle counter (external input)

Note: we deliberately do NOT reset on THOUGHT_PRODUCED.  vmDMN's
own idle reflections publish ThoughtProducedEvent every ~50
cycles when the bus is quiet — if those reset our counter, we'd
never reach the 200-tick threshold and substrate-reverie would
never fire alongside vmDMN's autobiographical reflection.  Both
forms of spontaneous activity (inward reflection + outward
curiosity about substrate concepts) should coexist.

Emits
-----
CHEMISTRY_FIRE with chemistry_kind='curiosity'.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from ..events import (
    EventKind, BrainEvent,
    AttendedPerceptEvent,
    ChemistryEvent,
    ThoughtProducedEvent,
)
from ..bus import EventBus
from seagi.core.substrate import WORLD_TOKEN_PREFIX


# Quiet stretch (ticks) before the daemon fires its first
# idle-curiosity event.  At standard tick rate, 200 ticks ≈ a
# meaningful idle pause without spamming.
IDLE_TICK_THRESHOLD = 200
# After firing, wait this many ticks before next fire.  Slow
# internal rhythm — closer to "spontaneous reverie" than "constant
# chatter".
IDLE_FIRE_COOLDOWN = 300
# Magnitude of the curiosity event.  Promille — same scale as
# external curiosity.
IDLE_MAGNITUDE = 0.3


class IdleMotivation:
    """Generates spontaneous curiosity events during prolonged quiet."""

    SUBSCRIPTIONS = (
        EventKind.ATTENDED_PERCEPT,
    )

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None,
                 awm_provider: Optional[Callable] = None,
                 substrate_provider: Optional[Callable] = None,
                 goal_provider: Optional[Callable] = None,
                 value_provider: Optional[Callable] = None):
        """`awm_provider`, when wired, returns the current AWM
        active concepts (list of focal names, ordered by salience).

        `substrate_provider` (C.1.e fallback), when wired, returns
        a list of substrate concept names to draw from when AWM
        is empty.  IdleMotivation rotates through this list across
        successive fires, so the agent doesn't fixate on one
        concept during deep idle.

        `goal_provider` (Roadmap Step 3, 2026-05-22), when wired,
        returns a list of open-goal focals ordered by urgency.
        This is the top-down hook: when the agent has open goals,
        autonomous reverie pulls toward them — the agent ruminates
        on what it is trying to figure out.  Goals win priority
        over both AWM and substrate fallback.  Without it, the
        daemon falls back to AWM-then-substrate as before.

        Without any provider, the daemon is inert.
        """
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._awm_provider = awm_provider
        self._substrate_provider = substrate_provider
        self._goal_provider = goal_provider
        # Value->attention (2026-06-05): the value system's per-concept
        # verdict, so reverie attention can follow VALUE, not just
        # salience — the missing value->attention link.
        self._value_provider = value_provider
        self._idle_ticks: int = 0
        self._last_fire_cycle: int = -IDLE_FIRE_COOLDOWN
        # Round-robin index into the substrate-fallback list so
        # the agent's reverie shifts across concepts over time.
        self._substrate_index: int = 0
        # Diagnostics.
        self.fires: int = 0
        self.last_focal: str = ''
        # Track fire-origin separately: GOAL (top-down, the agent
        # ruminates on an open problem); AWM (recent attention);
        # SUBSTRATE (spontaneous reverie when working memory is
        # empty and no goal is open).
        self.fires_from_goal: int = 0
        self.fires_from_awm: int = 0
        self.fires_from_substrate: int = 0

    # ---- event handler ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        # Only EXTERNAL stimulus resets the idle clock — peer
        # input (ATTENDED_PERCEPT).  The agent's own internal
        # thoughts (THOUGHT_PRODUCED), reflections, etc. do NOT
        # reset, so substrate-reverie can fire alongside
        # autobiographical reflection during silent stretches.
        # Phase C.1.f (2026-05-19): we now publish our own
        # AttendedPerceptEvent on each reverie fire to seed AWM
        # and engage cortical.  Filter those self-published ones
        # so they don't reset our own counter.
        if isinstance(event, AttendedPerceptEvent):
            if event.source_capability == 'idle_motivation':
                return
            # The autonomous mortal-game loop publishes its own world percepts
            # (modality='world') every awake tick.  That is the agent's OWN
            # action stream, not external stimulus, so — exactly like the
            # self-published reverie percepts filtered above — it must not reset
            # the reverie clock; otherwise the game would suppress all
            # spontaneous thought.  Reverie and acting coexist.
            if getattr(event, 'modality', '') == 'world':
                return
            self._idle_ticks = 0

    # ---- per-tick driver ----

    def tick(self) -> None:
        """Called by Brain.tick().  Increments the idle counter
        and fires when the quiet stretch crosses threshold."""
        self._idle_ticks += 1
        if self._idle_ticks < IDLE_TICK_THRESHOLD:
            return
        cycle = self._cycle_provider()
        if cycle - self._last_fire_cycle < IDLE_FIRE_COOLDOWN:
            return
        focal, source = self._pick_focal()
        if not focal:
            return
        self._fire_curiosity(cycle, focal, source)

    def _pick_focal(self) -> tuple:
        """Pick something to reverie about.  Returns (focal,
        source) where source is 'goal', 'awm', or 'substrate'.
        Empty focal means nothing to fire on.

        Priority order (top-down to bottom-up):
          1. GOALS — the agent's open problems (top-urgency
             first).  This is the top-down hook: cognition
             follows what the agent is trying to figure out.
          2. AWM active concepts — recent attention.
          3. Substrate fallback — round-robin through top
             concepts when working memory is empty.
        """
        def _val(f: str) -> float:
            if self._value_provider is None:
                return 0.0
            try:
                vp = self._value_provider()
                return float(vp.value_of(f)) if vp is not None else 0.0
            except Exception:
                return 0.0

        def _ok(f) -> bool:
            # Wall down (2026-06-30): a lived, M/I-tagged world-percept token IS
            # a first-class focal — recruited into reverie/goals like any
            # concept.  The value-bias below + earn-or-dissolve decide which
            # focals win; no wall holds world experience out of cognition.
            return bool(f)

        # 1. Top-down: an open goal — VALUE-BIASED (2026-06-05).  A
        # net-negative-value focal (one the value system has judged
        # unproductive — a ruminated dead-end like the contaminating
        # hub) is SUPPRESSED: it loses the attention slot (the striatal
        # NoGo / indirect pathway), so reverie stops fixating on what it
        # has condemned.  Among survivors, prefer the highest value
        # (incentive salience).  Value weight mirrors BG's
        # (basal_ganglia._score) — no new constant.  A not-yet-valued
        # focal (0.0) is neutral, kept, and ordered by the provider.
        if self._goal_provider is not None:
            try:
                goal_focals = [str(f)
                               for f in (self._goal_provider() or [])
                               if _ok(f)]
            except Exception:
                goal_focals = []
            viable = [f for f in goal_focals if _val(f) >= 0.0]
            if viable:
                viable.sort(key=lambda f: -_val(f))
                return (viable[0], 'goal')
        # 2. Try AWM — same value suppression / preference.
        if self._awm_provider is not None:
            try:
                actives = [str(a)
                           for a in (self._awm_provider() or []) if _ok(a)]
            except Exception:
                actives = []
            viable = [a for a in actives if _val(a) >= 0.0]
            if viable:
                viable.sort(key=lambda a: -_val(a))
                return (viable[0], 'awm')
        # 3. Substrate fallback.
        if self._substrate_provider is not None:
            try:
                pool = [p for p in (self._substrate_provider() or [])
                        if _ok(p)]
            except Exception:
                pool = []
            if pool:
                # Round-robin so reverie drifts across concepts.
                idx = self._substrate_index % len(pool)
                self._substrate_index += 1
                try:
                    pick = pool[idx]
                except (IndexError, TypeError):
                    pick = None
                if pick:
                    return (str(pick), 'substrate')
        return ('', '')

    def _fire_curiosity(self,
                                  cycle: int,
                                  focal: str,
                                  source: str = '') -> None:
        origin_detail = (
            f'reverie:{source}' if source else 'reverie')
        # 1. Chemistry nudge — the felt accompaniment of reaching
        #    for the concept.
        try:
            self.bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=cycle,
                timestamp=time.time(),
                source_capability='idle_motivation',
                origin='internal',
                origin_detail=origin_detail,
                chemistry_kind='curiosity',
                magnitude=IDLE_MAGNITUDE,
                target_concepts=[focal],
            ))
        except Exception:
            return
        # 2. Phase C.1.f: synthetic attended percept so the focal
        #    actually enters AWM and cortical engages.  Without
        #    this, reverie is just a chemistry heartbeat; with it,
        #    the reverie is a thought the agent attends to
        #    internally.  Origin 'internal' + source 'idle_motivation'
        #    keeps it distinguishable from external peer input.
        try:
            self.bus.publish(AttendedPerceptEvent(
                kind=EventKind.ATTENDED_PERCEPT,
                cycle=cycle,
                timestamp=time.time(),
                source_capability='idle_motivation',
                origin='internal',
                origin_detail=origin_detail,
                focals=[focal],
                payload={focal: {}},
                raw_text='',
                modality='text',
                salience=0.5,
                novelty=0.2,
                m_content=0.0,
                i_content=0.0,
                threshold_used=0.3,
            ))
        except Exception:
            pass
        self._last_fire_cycle = cycle
        self.fires += 1
        self.last_focal = focal
        if source == 'goal':
            self.fires_from_goal += 1
        elif source == 'awm':
            self.fires_from_awm += 1
        elif source == 'substrate':
            self.fires_from_substrate += 1
        # Reset idle counter so we don't fire again on the next
        # tick if the curiosity fire didn't itself produce
        # attended activity.
        self._idle_ticks = 0

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'idle_ticks': self._idle_ticks,
            'fires': self.fires,
            'fires_from_goal': self.fires_from_goal,
            'fires_from_awm': self.fires_from_awm,
            'fires_from_substrate': self.fires_from_substrate,
            'last_focal': self.last_focal,
            'last_fire_cycle': self._last_fire_cycle,
            'substrate_index': self._substrate_index,
        }
