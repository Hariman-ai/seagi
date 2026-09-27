"""Goal tracker + spawner — autonomous agenda-setting.

Phase F.7 (2026-05-16).  v1 had `goal.py` / `goal_spawner.py` /
`daemon_goals.py`; v2 consolidation dropped them.  The audit
flagged the loss: BasalGanglia arbitrates capability claims but
no claims come from a goal-setter, so the agent has no internal
drives between peer turns.

What this module is
-------------------
- `Goal` dataclass: an open intention with a focal, kind,
   urgency, status, and source.
- `GoalTracker`: bounded registry of active + recently-completed
   goals.  Queryable from anywhere in the brain.
- `GoalSpawner`: examines current state on each reflection
   cycle and proposes new goals when triggers fire:
     - Thin substrate on a recently-active focal → 'learn_about'
     - High-uncertainty focal → 'resolve_uncertainty'
     - Prediction failure on a focal → 'reconcile_failure'
     - Unanswered question from peer → 'answer_for_peer'
- Goals contribute promille chemistry nudges (curiosity at
   spawn, mattering at completion, falsified_i at abandonment).

Doctrine alignment
------------------
Goals are not stored on substrate — they're brain-state, transient,
re-derived each session from accumulated experience.  Their EFFECT
is on substrate (which concepts get cortical attention, which
edges get reinforced) but they themselves live in working memory.

Promille magnitudes throughout — spawn nudges, completion fires,
abandonment fires all at 0.001-0.005 scale.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional


# Goal kinds — small fixed taxonomy.
GOAL_LEARN_ABOUT = 'learn_about'
GOAL_RESOLVE_UNCERTAINTY = 'resolve_uncertainty'
GOAL_RECONCILE_FAILURE = 'reconcile_failure'
GOAL_ANSWER_FOR_PEER = 'answer_for_peer'
GOAL_PURSUE_CURIOSITY = 'pursue_curiosity'


# Statuses.
STATUS_OPEN = 'open'
STATUS_ACHIEVED = 'achieved'
STATUS_ABANDONED = 'abandoned'


# Default registry size.  Goals are working-memory state, not
# persistent; bounded so the registry doesn't grow unbounded.
DEFAULT_GOAL_REGISTRY_SIZE = 64

# How many cycles a goal can stay open before automatic
# abandonment (no progress in this many ticks → give up).  Retained
# as the DECAY TIMESCALE for goal-mortality (below), not as a hard
# timeout.
DEFAULT_GOAL_TIMEOUT_CYCLES = 5000

# Goal mortality (2026-06-04): goals are MORTAL like edges — nothing
# is immortal, not even a goal.  Urgency DECAYS continuously and is
# earned back ONLY by real value (a confident, non-thin thought about
# the focal — the same bar check_thought_completion uses).  A goal
# that produces nothing fades to zero and dissolves; a productive one
# keeps earning and lives as long as it pays its way.  This replaces
# the old activity-timeout (which a re-encountered-but-unsolved focal
# could keep alive forever — the immortal "doubts" loop).  The decay
# rate REUSES the existing patience window as its timescale (an
# unearning goal at full urgency fades over one window); no new tuned
# constant — the opposing pressure is the earn-refresh on value.
URGENCY_DECAY_PER_CYCLE = 1.0 / float(DEFAULT_GOAL_TIMEOUT_CYCLES)

# Minimum cycles between spawn passes (don't fire goal-spawner
# on every reflection).
MIN_CYCLES_BETWEEN_SPAWNS = 50

# Maximum active goals at any time — keeps the agent focused.
MAX_ACTIVE_GOALS = 5

# A concept with fewer than this many outgoing edges is "thin"
# substrate — a learning gap.  ONE definition, shared by the
# thin-AWM spawn trigger here and the goal-biased neighbor
# expansion in runtime (so the threshold isn't duplicated as a
# bare literal in two places).  Phase 2 (2026-06-01).
THIN_EDGE_THRESHOLD = 3

# The "confident, non-hedge" bar shared by check_thought_completion
# (goal EARN) and UncertaintyMonitor._resolve (uncertainty discharge) —
# audit #4c/#7 (2026-06-04).  Naming the previously-duplicated 0.7
# literal so the two signals CANNOT diverge: a thought clears this bar
# iff it both RESOLVES the uncertainty AND EARNS the goal's keep, so a
# weak hedge can do neither.  Not a new tuned constant — it replaces an
# existing magic literal.
EARN_CONFIDENCE_MIN = 0.7


@dataclass
class Goal:
    """One open intention."""
    id: str
    kind: str
    focal: str
    target: str = ''         # secondary concept if applicable
    urgency: float = 0.5     # [0, 1], shapes BG arbitration priority
    status: str = STATUS_OPEN
    source: str = ''         # 'spawner' / 'peer' / 'manual'
    created_cycle: int = 0
    last_activity_cycle: int = 0   # legacy/diag; no longer drives life
    last_urgency_cycle: int = 0    # mortality decay anchor
    achievement_cycle: Optional[int] = None
    abandoned_cycle: Optional[int] = None
    notes: str = ''          # human-readable description

    def is_open(self) -> bool:
        return self.status == STATUS_OPEN

    def is_resolved(self) -> bool:
        return self.status in (STATUS_ACHIEVED, STATUS_ABANDONED)

    def effective_urgency(self, cycle: int) -> float:
        """Urgency with continuous mortality decay applied since the
        last time it was earned (last_urgency_cycle).  Earned back
        only by value — GoalTracker.earn — exactly as Edge strength
        is earned by reinforcement.  A goal nobody feeds fades to 0."""
        elapsed = int(cycle) - int(self.last_urgency_cycle)
        if elapsed <= 0:
            return self.urgency
        return max(0.0, self.urgency
                   - URGENCY_DECAY_PER_CYCLE * elapsed)


class GoalTracker:
    """Bounded registry of goals.  Active set ordered by urgency."""

    def __init__(self,
                 max_active: int = MAX_ACTIVE_GOALS,
                 history_size: int = DEFAULT_GOAL_REGISTRY_SIZE):
        self.max_active = int(max_active)
        self._active: Dict[str, Goal] = {}
        self._history: Deque[Goal] = deque(maxlen=int(history_size))
        self._next_id: int = 1
        # Diagnostics.
        self.goals_spawned: int = 0
        self.goals_achieved: int = 0
        self.goals_abandoned: int = 0

    # ---- queries ----

    def __len__(self) -> int:
        return len(self._active)

    def active(self, cycle: Optional[int] = None) -> List[Goal]:
        """Active goals ordered by EFFECTIVE (decay-applied) urgency
        when a cycle is given, else by stored urgency — audit #6
        (2026-06-04): arbitration / eviction / reverie consumers must
        see the decayed truth, not the value last materialized by
        spawn / earn / sweep_mortality (which lags up to a sweep
        interval, letting a decayed goal win attention it hasn't
        earned)."""
        if cycle is None:
            return sorted(
                self._active.values(),
                key=lambda g: (-g.urgency, g.created_cycle))
        return sorted(
            self._active.values(),
            key=lambda g: (-g.effective_urgency(cycle), g.created_cycle))

    def primary(self, cycle: Optional[int] = None) -> Optional[Goal]:
        """Highest-urgency open goal, or None."""
        a = self.active(cycle)
        return a[0] if a else None

    def has_goal_for(self, focal: str) -> bool:
        """True if there's an active goal whose focal matches."""
        for g in self._active.values():
            if g.focal == focal:
                return True
        return False

    def goal_for(self, focal: str) -> Optional[Goal]:
        """First active goal whose focal matches, or None."""
        for g in self._active.values():
            if g.focal == focal:
                return g
        return None

    def history(self, n: int = 10) -> List[Goal]:
        """Last N resolved goals (achieved or abandoned), most-
        recent first."""
        return list(self._history)[-n:][::-1]

    # ---- mutation ----

    def spawn(self,
                kind: str,
                focal: str,
                *,
                target: str = '',
                urgency: float = 0.5,
                source: str = 'spawner',
                cycle: int = 0,
                notes: str = '') -> Optional[Goal]:
        """Add a new goal.  Returns it, or None if a duplicate
        already exists OR the active set is full of higher-urgency
        goals."""
        if not focal:
            return None
        # Wall down (2026-06-30): world tokens CAN become goals.  The prior
        # reverie-starvation was the HALF-open deadlock — world goals that were
        # un-pickable because idle_motivation._ok filtered them; that filter is
        # now also down, so a world token is fully first-class.  The M/I value-
        # bias picks the goals that matter; the rest dedup and fade.  No wall.
        # Dedup — don't spawn a second goal for the same focal +
        # kind combination.  Re-encountering an UNRESOLVED focal is
        # NOT value (it's the opposite — zero progress), so it must
        # NOT refresh the goal's life.  Removing the old activity-
        # refresh here is what killed the immortal-"doubts" loop:
        # only earn() (a real confident thought) renews a goal.
        for g in self._active.values():
            if g.focal == focal and g.kind == kind:
                return g
        # Capacity check — if at max, only spawn if higher urgency
        # than the weakest active.
        if len(self._active) >= self.max_active:
            sorted_active = self.active(cycle)
            weakest = sorted_active[-1]
            if urgency <= weakest.effective_urgency(cycle):
                return None
            # Evict the weakest to make room.
            self.abandon(weakest.id, cycle=cycle,
                            reason='evicted_by_higher_urgency')
        gid = f'g{self._next_id:04d}'
        self._next_id += 1
        g = Goal(
            id=gid, kind=kind, focal=focal,
            target=target, urgency=max(0.0, min(1.0, urgency)),
            status=STATUS_OPEN, source=source,
            created_cycle=int(cycle),
            last_activity_cycle=int(cycle),
            last_urgency_cycle=int(cycle),
            notes=notes)
        self._active[gid] = g
        self.goals_spawned += 1
        return g

    def complete(self,
                    gid: str,
                    cycle: int,
                    notes: str = '') -> Optional[Goal]:
        """Mark a goal achieved.  Moves it to history."""
        g = self._active.pop(gid, None)
        if g is None:
            return None
        g.status = STATUS_ACHIEVED
        g.achievement_cycle = int(cycle)
        if notes:
            g.notes = (g.notes + ' | ' + notes
                          if g.notes else notes)
        self._history.append(g)
        self.goals_achieved += 1
        return g

    def abandon(self,
                   gid: str,
                   cycle: int,
                   reason: str = '') -> Optional[Goal]:
        """Mark a goal abandoned.  Moves it to history."""
        g = self._active.pop(gid, None)
        if g is None:
            return None
        g.status = STATUS_ABANDONED
        g.abandoned_cycle = int(cycle)
        if reason:
            g.notes = (g.notes + ' | abandoned: ' + reason
                          if g.notes else f'abandoned: {reason}')
        self._history.append(g)
        self.goals_abandoned += 1
        return g

    def sweep_timeouts(self, cycle: int,
                            timeout: int = DEFAULT_GOAL_TIMEOUT_CYCLES
                            ) -> int:
        """Abandon goals that have stalled past `timeout` cycles
        without activity.  Returns count abandoned."""
        to_abandon = [
            gid for gid, g in self._active.items()
            if cycle - g.last_activity_cycle >= timeout]
        for gid in to_abandon:
            self.abandon(gid, cycle=cycle, reason='timeout')
        return len(to_abandon)

    def earn(self, focal: str, cycle: int) -> bool:
        """Real value fired for `focal` (a confident, non-thin
        thought — see check_thought_completion) — RENEW the urgency
        of any open goal for it, resetting its mortality decay.  The
        goal analog of Edge.reinforce: value buys life.  A full
        refresh (to the urgency ceiling) means one real thought
        re-earns a goal's whole patience window.  Returns True if a
        goal was refreshed."""
        refreshed = False
        for g in self._active.values():
            if g.focal == focal:
                g.urgency = min(1.0, g.effective_urgency(cycle) + 1.0)
                g.last_urgency_cycle = int(cycle)
                refreshed = True
        return refreshed

    def sweep_mortality(self, cycle: int) -> int:
        """Goal mortality (replaces the activity-timeout).  Materialize
        each active goal's decayed urgency; DISSOLVE (abandon) any
        whose urgency has decayed to zero — a goal that produced no
        value over its patience window starves and dies, exactly as a
        never-cohered edge does.  No timeout, no activity clock; the
        only thing that keeps a goal alive is earning value.  Returns
        the count dissolved."""
        dissolved = 0
        for gid, g in list(self._active.items()):
            eff = g.effective_urgency(cycle)
            if eff <= 0.0:
                self.abandon(gid, cycle=cycle, reason='starved')
                dissolved += 1
            else:
                # Materialize the decay so it persists, re-anchor.
                g.urgency = eff
                g.last_urgency_cycle = int(cycle)
        return dissolved

    def touch(self, focal: str, cycle: int) -> None:
        """Refresh activity on any open goal for this focal — keeps
        the goal alive across the timeout window."""
        for g in self._active.values():
            if g.focal == focal:
                g.last_activity_cycle = int(cycle)

    def stats(self) -> Dict[str, Any]:
        return {
            'active': len(self._active),
            'goals_spawned': self.goals_spawned,
            'goals_achieved': self.goals_achieved,
            'completions_via_grain': getattr(
                self, 'completions_via_grain', 0),
            'goals_abandoned': self.goals_abandoned,
        }

    # ---- Phase H.1 (2026-05-17): M/I-weighted persistence ----

    # A goal is "felt enough to matter for personality" when its
    # urgency is non-trivial.  Stalled / abandoned / low-urgency
    # goals are ephemeral.
    PERSIST_URGENCY_FLOOR = 0.5

    def _persist_ok(self, g: Goal) -> bool:
        return g.is_open() and g.urgency >= self.PERSIST_URGENCY_FLOOR

    def _goal_to_dict(self, g: Goal) -> Dict[str, Any]:
        return {
            'id': g.id, 'kind': g.kind, 'focal': g.focal,
            'target': g.target, 'urgency': g.urgency,
            'status': g.status, 'source': g.source,
            'created_cycle': g.created_cycle,
            'last_activity_cycle': g.last_activity_cycle,
            'last_urgency_cycle': g.last_urgency_cycle,
            'notes': g.notes,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'next_id': self._next_id,
            'goals': [self._goal_to_dict(g)
                       for g in self._active.values()
                       if self._persist_ok(g)],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GoalTracker':
        t = cls()
        t._next_id = int(d.get('next_id', 1))
        for gd in d.get('goals', []):
            g = Goal(
                id=str(gd.get('id', '')),
                kind=str(gd.get('kind', '')),
                focal=str(gd.get('focal', '')),
                target=str(gd.get('target', '')),
                urgency=float(gd.get('urgency', 0.5)),
                status=str(gd.get('status', STATUS_OPEN)),
                source=str(gd.get('source', '')),
                created_cycle=int(gd.get('created_cycle', 0)),
                last_activity_cycle=int(gd.get(
                    'last_activity_cycle', 0)),
                last_urgency_cycle=int(gd.get(
                    'last_urgency_cycle',
                    gd.get('created_cycle', 0))),
                notes=str(gd.get('notes', '')))
            if g.id and g.is_open():
                t._active[g.id] = g
        return t

    # ---- Phase G.6 (2026-05-16): goal auto-completion ----

    def check_thought_completion(self,
                                          focal: str,
                                          confidence: float,
                                          thin_substrate: bool,
                                          cycle: int,
                                          bus: Any = None) -> Optional[Goal]:
        """Phase G.6: when Cortical produces a confident,
        non-thin thought about a focal, check if any open
        learn_about goal exists for that focal.  If yes,
        complete the goal and fire confirmed_i to feed the
        reward ledger.

        Args:
          focal: the focal of the cortical thought
          confidence: thought's confidence in [0,1]
          thin_substrate: cortical's metacog verdict
          cycle: current cycle
          bus: optional event bus for confirmed_i emission

        Returns the completed Goal, or None.

        Completion conditions:
          - confidence ≥ 0.7 (genuinely confident, not hedge)
          - NOT thin_substrate (had enough to think well)
          - Open goal exists for focal with kind=learn_about
        """
        if (not focal or thin_substrate
                or confidence < EARN_CONFIDENCE_MIN):
            return None
        # D3 Wire A (2026-07-13): capture the goal and its PRE-earn
        # effective (decay-applied) urgency BEFORE the earn() renewal —
        # the confirmed_i reward magnitude must reflect how urgent the
        # goal ACTUALLY was, not the post-earn ceiling (~1.0) that earn()
        # lifts it to.  goal_for() is unaffected by earn (earn only
        # renews urgency, never adds/removes goals), so capturing g here
        # selects the same goal as before.
        g = self.goal_for(focal)
        pre_urgency = (
            g.effective_urgency(cycle) if g is not None else 0.0)
        # Goal-mortality EARN: a confident, non-thin thought about
        # this focal is real value — renew ANY open goal for it
        # (any kind), resetting its decay.  This is the signal that
        # keeps productive goals alive.  A rumination focal like
        # "doubts" only ever yields hedged reflections (conf < 0.7),
        # never clears this bar, so it never earns and dissolves
        # under sweep_mortality.
        self.earn(focal, cycle)
        # learn_about (own desire to know), answer_for_peer (peer asked)
        # AND resolve_uncertainty (D3 Wire A) complete when a confident,
        # non-thin thought arrives: the agent thought WELL about the
        # focal, so the felt gap / question is resolved.  (reconcile_
        # failure, pursue_curiosity keep their own completion semantics.)
        if g is None or g.kind not in (
                GOAL_LEARN_ABOUT, GOAL_ANSWER_FOR_PEER,
                GOAL_RESOLVE_UNCERTAINTY):
            return None
        completed = self.complete(g.id, cycle=cycle,
                                          notes='resolved by cortical')
        if completed is None or bus is None:
            return completed
        # Fire confirmed_i to register the goal achievement as
        # positive reward.  The reward ledger picks it up and
        # propagates credit to recent action sequences.  Magnitude =
        # PRE-earn urgency (D3 Wire A), never record_learning.
        try:
            from ..events import ChemistryEvent, EventKind
            import time as _time
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=int(cycle),
                timestamp=_time.time(),
                source_capability='goal_tracker',
                origin='internal',
                origin_detail=f'goal_complete:{g.id}',
                chemistry_kind='confirmed_i',
                magnitude=float(pre_urgency),
                target_concepts=[focal]))
        except Exception:
            pass
        return completed

    def complete_on_coherence(self,
                                focal: str,
                                cycle: int,
                                bus: Any = None,
                                uncertainty_monitor: Any = None
                                ) -> Optional[Goal]:
        """D3 Wire B (2026-07-13): a focal reached FIRST-EVER coherence in
        the substrate — the success condition for an open
        resolve_uncertainty goal on it (the felt gap got structurally
        corroborated, not merely thought-about).  Complete that goal
        (STATUS_ACHIEVED — distinct from give-up's STATUS_ABANDONED) and
        fire confirmed_i ONLY (never record_learning — completion is a
        reward signal, not earn-gated growth).

        Coherence is NOT a THOUGHT_PRODUCED event, so the
        UncertaintyMonitor never sees it and won't auto-clear the felt
        gap; this method explicitly resolves it via the passed monitor.
        confirmed_i magnitude = PRE-completion effective urgency (how
        urgent the gap was).  Returns the completed Goal, or None if no
        open resolve_uncertainty goal exists for `focal`."""
        g = self.goal_for(focal)
        if g is None and '|a' in focal:
            # GRAIN BRIDGE (2026-08-09).  Goals are asked about a STATE
            # (the uncertainty monitor's focals are bare), but world
            # coherence arrives at STATE|aN, because a transitions_to
            # edge is `STATE|aN -> next`.  Exact equality could never
            # match, so goals_achieved sat at 0 across 197 goals while
            # both completion paths were correctly wired.
            # Knowing what action N does in state S genuinely resolves
            # uncertainty about S -- that is what was asked.
            _bare = focal[:focal.rfind('|a')]
            if _bare:
                g = self.goal_for(_bare)
                if g is not None:
                    self.completions_via_grain = getattr(
                        self, 'completions_via_grain', 0) + 1
        if g is None or g.kind != GOAL_RESOLVE_UNCERTAINTY:
            return None
        pre_urgency = g.effective_urgency(cycle)
        completed = self.complete(
            g.id, cycle=cycle, notes='resolved by coherence')
        if completed is None:
            return None
        # A coherence event is not a THOUGHT_PRODUCED, so the monitor
        # won't clear the uncertainty on its own — do it explicitly.
        if uncertainty_monitor is not None:
            try:
                uncertainty_monitor._resolve(focal)
            except Exception:
                pass
        if bus is None:
            return completed
        try:
            from ..events import ChemistryEvent, EventKind
            import time as _time
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=int(cycle),
                timestamp=_time.time(),
                source_capability='goal_tracker',
                origin='internal',
                origin_detail=f'goal_complete_coherence:{g.id}',
                chemistry_kind='confirmed_i',
                magnitude=float(pre_urgency),
                target_concepts=[focal]))
        except Exception:
            pass
        return completed

    # ---- Phase G.1 (2026-05-16): emit BG claims ----

    def emit_claims(self,
                       bus: Any,
                       cycle: int,
                       awm_active: Optional[Any] = None,
                       cap_per_pass: int = 3) -> int:
        """Publish CapabilityClaim events for open goals — one
        per goal, urgency-scaled, on the motivational loop.

        Doctrine (Phase G.1): goals are not just trackable
        intentions; they should COMPETE for the next moment's
        attention.  BG arbitration then picks among:
          - cortical's reflection claims
          - amygdala/NAcc sentinel claims
          - goal-driven motivational claims
          - speech claims
        The action format is `pursue:{focal}` — a goal's
        intent is to "pursue the focal."  Cortical doesn't
        directly act on this yet; future hooks can route the
        winning pursue claim into a forced AWM-promote or a
        targeted cortical reasoning pass.

        If `awm_active` is provided (collection of names
        currently in AWM), only goals whose focal is in AWM are
        emitted — they're the ones plausibly actionable right
        now.  If not provided, all open goals emit.

        `cap_per_pass` limits how many claims fire per call
        (don't flood BG with stale goal claims).  Returns the
        number of claims published.

        Returns 0 silently if bus is None or emit fails.
        """
        if bus is None:
            return 0
        # Import lazily to avoid circular import at module load.
        try:
            from ..events import (
                CapabilityClaimEvent, EventKind)
        except Exception:
            return 0
        import time as _time
        awm_set = (set(awm_active) if awm_active is not None
                       else None)
        count = 0
        # Sort by urgency so highest-urgency claims fire first
        # (within the cap).
        for g in sorted(self._active.values(),
                            key=lambda x: -x.effective_urgency(cycle)):
            if count >= cap_per_pass:
                break
            if awm_set is not None and g.focal not in awm_set:
                continue
            try:
                bus.publish(CapabilityClaimEvent(
                    kind=EventKind.CAPABILITY_CLAIM,
                    cycle=int(cycle),
                    timestamp=_time.time(),
                    source_capability='goal_tracker',
                    origin='internal',
                    origin_detail=f'goal:{g.id}',
                    claim_strength=float(g.effective_urgency(cycle)),
                    proposed_action=f'pursue:{g.focal}',
                    loop='motivational',
                    payload={'goal_id': g.id,
                                'goal_kind': g.kind}))
                count += 1
            except Exception:
                pass
        return count


class GoalSpawner:
    """Opens goals from the brain's felt gaps so reverie is steered
    top-down (Step 3 spine; Phase 2 goal layer, 2026-06-01).

    DOCTRINE FRAMING (audit must-fix #4): this is directional
    PLUMBING, not a homeostatic regulator.  Nothing degrades if it
    never opens a goal — there is no felt survival cost to its
    silence (that cost lives in MetabolicDebt + the mortality drive).
    It is a steering wheel on a car that already drives straight:
    useful for pointing recall at structure worth building, not
    load-bearing for survival.

    SENSED-GAP SURFACE (audit must-fix #5 — be honest about the
    closed set): goals self-form from felt-gap SIGNALS, with topics
    (focals) emerging from live signals rather than any hardcoded
    list.  The signal channels are:
      - Thin-substrate concept in recent AWM → 'learn_about'
      - The UncertaintyMonitor window → 'resolve_uncertainty'.  That
        window aggregates BOTH thin-substrate metacog thoughts AND
        ACC conflicts, so the metacog-gap and conflict-gap channels
        both surface here (one read, broad coverage).
      - Unanswered peer question → 'answer_for_peer' (dormant
        without peers).
    KNOWN LIMITATION: a felt gap that produces NONE of these signals
    (e.g. boredom, a goal that should CLOSE, a contradiction between
    two already-coherent edges) opens no goal.  The gap-detector set
    is itself a closed enumeration one level up; making it growable
    (detectors that earn-or-dissolve) is a documented follow-up, not
    this layer.
    """

    def __init__(self,
                 tracker: GoalTracker,
                 awm_provider: Callable,
                 lts_provider: Callable,
                 conversation_provider: Callable,
                 cycle_provider: Callable,
                 chemistry_provider: Optional[Callable] = None,
                 uncertainty_provider: Optional[Callable] = None):
        self.tracker = tracker
        self._awm_provider = awm_provider
        self._lts_provider = lts_provider
        self._conversation_provider = conversation_provider
        self._cycle_provider = cycle_provider
        self._chemistry_provider = chemistry_provider
        # Phase 2: () -> [(focal, pressure)] from UncertaintyMonitor.
        self._uncertainty_provider = uncertainty_provider
        self._last_spawn_cycle: int = -10**6
        self.spawn_passes: int = 0
        # Per-pass diagnostic: {spawned, errors} from the last
        # maybe_spawn (Phase 2, 2026-06-01).
        self._last_debug: dict = {}

    def maybe_spawn(self) -> List[Goal]:
        """One spawn pass.  Returns goals newly spawned."""
        cycle = self._cycle_provider()
        if cycle - self._last_spawn_cycle < MIN_CYCLES_BETWEEN_SPAWNS:
            return []
        self._last_spawn_cycle = cycle
        self.spawn_passes += 1
        # Goal mortality first — decay urgency, dissolve any goal
        # that has starved (produced no value over its patience
        # window).  Replaces the old activity-timeout.
        self.tracker.sweep_mortality(cycle)

        spawned: List[Goal] = []

        errors: List[str] = []

        # Trigger 1: thin substrate in recent AWM.  FULLY guarded
        # (Phase 2 fix 2026-06-01): the loop — not just the provider
        # calls — is inside the try, because a throw here previously
        # propagated out of maybe_spawn and was swallowed by the
        # caller, silently aborting every LATER trigger (incl. the
        # felt-gap channel) so goals never opened.
        try:
            awm = self._awm_provider()
            lts = self._lts_provider()
            if awm is not None and lts is not None:
                for name in list(awm.active_concepts())[:10]:
                    if self.tracker.has_goal_for(name):
                        continue
                    concept = lts.get_concept(name)
                    if concept is None:
                        continue
                    n_edges = len(
                        getattr(concept, 'edges_out', {}) or {})
                    if n_edges < THIN_EDGE_THRESHOLD:
                        # Urgency MEASURED (#3): sparser = more
                        # urgent, derived from the shared thin
                        # threshold — no literal.
                        g = self.tracker.spawn(
                            kind=GOAL_LEARN_ABOUT,
                            focal=name,
                            urgency=1.0 - (
                                n_edges
                                / float(THIN_EDGE_THRESHOLD)),
                            source='spawner_thin_substrate',
                            cycle=cycle,
                            notes=(f'substrate has {n_edges} edges '
                                     f'for {name} — learning gap'))
                        if g is not None:
                            spawned.append(g)
        except Exception as e:
            errors.append(f'thin_awm:{type(e).__name__}:{e}')

        # Trigger 1b (Phase 2 goal layer): the brain's felt epistemic
        # gaps via the UncertaintyMonitor window — the AGGREGATE of
        # thin-substrate metacog thoughts AND ACC conflicts.  The
        # channel that actually fires live (thin-AWM rarely does on a
        # dense substrate).  Urgency = MEASURED window pressure (#3).
        uncertainty_seen: list = []
        try:
            if self._uncertainty_provider is not None:
                uncertainty_seen = list(
                    self._uncertainty_provider() or [])
                for focal, pressure in uncertainty_seen:
                    if not focal or focal == 'self':
                        continue
                    if self.tracker.has_goal_for(focal):
                        continue
                    g = self.tracker.spawn(
                        kind=GOAL_RESOLVE_UNCERTAINTY,
                        focal=focal,
                        urgency=max(0.0, min(1.0, float(pressure))),
                        source='spawner_uncertainty',
                        cycle=cycle,
                        notes=(f'{focal} recurs unresolved '
                                 f'(pressure {float(pressure):.2f})'))
                    if g is not None:
                        spawned.append(g)
        except Exception as e:
            errors.append(f'uncertainty:{type(e).__name__}:{e}')

        # Trigger 2: unanswered factual question from peer (dormant
        # without peers).
        try:
            conv = self._conversation_provider()
            if conv is not None:
                for turn in conv.recent(n=6):
                    if turn.role != 'peer':
                        continue
                    if 'question' not in turn.intent_kind:
                        continue
                    if not turn.focal or turn.focal == 'self':
                        continue
                    if self.tracker.has_goal_for(turn.focal):
                        continue
                    g = self.tracker.spawn(
                        kind=GOAL_ANSWER_FOR_PEER,
                        focal=turn.focal,
                        # DECLARED DEBT (#3): literal urgency, dormant
                        # peer path; derive when the peer loop lives.
                        urgency=0.6,
                        source='spawner_unanswered_peer',
                        cycle=cycle,
                        notes=(f'{turn.peer_id} asked about '
                                 f'{turn.focal}'))
                    if g is not None:
                        spawned.append(g)
        except Exception as e:
            errors.append(f'peer:{type(e).__name__}:{e}')

        self._last_debug = {
            'spawned': len(spawned),
            'errors': errors,
            'provider_set': self._uncertainty_provider is not None,
            'top_seen': uncertainty_seen,
            'tracker_spawned': self.tracker.goals_spawned,
        }
        return spawned
