"""Active Working Memory — sparse hot set with chemistry time-series.

Brain analog: prefrontal cortex working memory + anterior
cingulate.  Holds the ~100-500 concepts currently in mind, with
each concept's enriched bubble AND a ring buffer of recent
chemistry samples (the chemistry time-series).

This is the layer every cognitive primitive operates on — not
LTS.  Cognitive operations on AWM are bounded-fast (proportional
to AWM size, not substrate size).

Phase 2: FULL implementation.

Subscribes : ATTENDED_PERCEPT (promotion source — gate-passed
             percepts become AWM entries)
Emits      : AWM_ACTIVATION (concept newly active)
             AWM_EVICTION   (concept fell out of active set)
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Deque, Dict, List, Optional, Tuple,
)

from ..events import (
    EventKind, BrainEvent, AttendedPerceptEvent, ThoughtProducedEvent,
)
from ..bus import EventBus
from ..chemistry_types import (
    CHANNELS, ChemistrySample, ChemistryRingBuffer, EnrichedBubble,
)


# Soft starting capacity.  AWM is sparse — far below substrate
# size — but the doctrine: "brain-correct as starting point,
# not ceiling."  256 is a reasonable v1 starting capacity (not
# the human ~7±2 limit).  When sustained chemistry pressure
# (high salience promotion attempts on a full AWM) signals
# that current capacity is too tight, the cap GROWS.  This
# avoids the don't-inherit-human-limits failure mode where
# SEAGI ends up artificially constrained by a number that
# "felt right" to me.
DEFAULT_AWM_CAPACITY = 256

# Maximum expansion step.  When capacity grows under pressure,
# it grows by a fraction (additive) — not unbounded.  Keeps
# growth predictable.
AWM_GROWTH_FACTOR = 1.25

# Salience threshold for "capacity pressure."  When a high-
# salience promotion would have to evict an entry that ALSO
# has high salience or vivid trace, that's the signal that
# current capacity is too tight.
CAPACITY_PRESSURE_SALIENCE_FLOOR = 0.5

# Phase F.14 (2026-05-16): decay-back constants.
# After the burst-expansion grew the soft cap under pressure,
# we need a path back DOWN when the load subsides.  Without
# this, capacity would only ratchet up across a long session.
#
# DECAY_UTILIZATION_THRESHOLD — utilization fraction below
#   which AWM is considered "quiet."  At 0.5, AWM has to be
#   less than half-full to count as quiet.
# DECAY_QUIET_TICKS_REQUIRED — quiet ticks before each shrink
#   step.  At 200 ticks, decay-back is gentle — burst
#   expansions don't immediately undo themselves.
# DECAY_FACTOR — multiplicative shrink per pass.  0.9 = drops
#   capacity by 10% each decay step.
# Doctrine: brain-correct is starting point not ceiling, but
# also not a permanent floor — capacity reflects current load.
DECAY_UTILIZATION_THRESHOLD = 0.5
DECAY_QUIET_TICKS_REQUIRED = 200
DECAY_FACTOR = 0.9

# Hard ceiling — eventually we DO need an upper bound for
# memory safety.  Far above human limits and well above any
# expected operational scale.
AWM_HARD_MAX = 8192

# Per-concept chemistry ring buffer size.  Small enough to be
# cheap; large enough to support trajectory queries
# ("rising / falling / steady over the last N samples").
DEFAULT_HISTORY_PER_CONCEPT = 64

# Eviction weighting.  When AWM is full and a new concept
# arrives, the entry with the LOWEST `eviction_priority` gets
# dropped.  Score = recency * salience-at-promotion *
# crystallization-floor.  Mortality-relevant concepts (high M
# in bubble) get a small protective bonus — we don't want fear
# to evict itself.
RECENCY_TAU = 200  # cycles

# Step 0 organ 4a (2026-05-27): AWM contraction.
# Recent-node-count deque tracks the size of recent R.1 inference
# walks the agent actually exercised — (chain_depth + 1) per
# inference thought.  The MIN over the rolling window becomes the
# safety floor below which sleep-deprivation-driven contraction
# cannot drop AWM.  Doctrine traces: AWM_RECENT_NODES_WINDOW = 50
# (= RAPHE_HISTORY_DEPTH), AWM_RECENT_NODES_STALENESS = 200 ticks
# (= DECAY_QUIET_TICKS_REQUIRED), AWM_CAPACITY_COLD_START_FLOOR = 3
# (minimum legal node count: a 2-hop inference chain visits 3
# concepts).
AWM_RECENT_NODES_WINDOW = 50
AWM_RECENT_NODES_STALENESS_TICKS = DECAY_QUIET_TICKS_REQUIRED
AWM_CAPACITY_COLD_START_FLOOR = 3


@dataclass
class AWMEntry:
    """One concept active in working memory."""
    concept_name: str
    bubble: EnrichedBubble
    chemistry_history: ChemistryRingBuffer
    promoted_at_cycle: int
    promoted_from_origin: str
    salience_at_promotion: float
    last_active_cycle: int
    activation: float = 1.0  # current activation strength
    # Quarantine-tier signal (2026-05-30): True iff this concept
    # had no live outgoing edges at promotion (after auto-restoring
    # any quarantined edges).  Downstream organs (idle_motivation,
    # cortical_focal_selection) can deprioritize naked focals
    # without blocking restoration — the chemistry-imprint bid-path
    # still runs because we promote anyway.
    naked: bool = False

    def update_activation(self, cycle: int) -> None:
        cycles_since = max(0, cycle - self.last_active_cycle)
        self.activation = (self.activation
                              * math.exp(-cycles_since / RECENCY_TAU))


class ActiveWorkingMemory:
    """Sparse, bounded set of currently-active concepts.

    Public API
    ----------
    promote(name, ...)        — add or refresh a concept
    evict(name)              — remove a concept (returns dropped entry)
    get(name) -> Optional[AWMEntry]
    active_concepts()        — list of currently-active names
    append_chemistry_sample(name, sample)
    trajectory(name, window) — chemistry time-series for a concept
    decay_tick(cycle)        — refresh activations
    stats()
    """

    SUBSCRIPTIONS = (
        EventKind.ATTENDED_PERCEPT,
        # Step 0 organ 4a: track R.1 inference chain depths so the
        # AWM capacity floor reflects what cognition has actually
        # exercised, not a hardcoded number.
        EventKind.THOUGHT_PRODUCED,
    )

    def __init__(self,
                 bus: EventBus,
                 lts_provider: Optional[Callable] = None,
                 capacity: int = DEFAULT_AWM_CAPACITY,
                 history_per_concept: int = (
                     DEFAULT_HISTORY_PER_CONCEPT),
                 cycle_provider: Optional[Callable] = None,
                 chemistry_provider: Optional[Callable] = None,
                 debt_provider: Optional[Callable] = None,
                 baseline_provider: Optional[Callable] = None,
                 debt_full_scale_provider: Optional[Callable] = None,
                 information_request_fn: Optional[Callable] = None):
        """
        bus: event bus for emitting AWM_ACTIVATION / _EVICTION
        lts_provider: callable returning the LTS instance — used
            to look up existing bubbles when promoting a concept
            that's already in long-term substrate
        capacity: max entries
        history_per_concept: ring buffer size per concept
        cycle_provider: callable returning current cycle
        chemistry_provider: callable returning the chemistry
            engine.  Used at bootstrap time to capture the
            current global chemistry into a ContextKey so the
            new bubble's context fingerprint reflects "what
            felt-state was firing when this concept was first
            promoted into AWM."  Optional — if absent, bootstrap
            falls back to ContextKey() (empty) and the bubble
            pool collapses to one-per-concept.
        """
        self.bus = bus
        self._lts_provider = lts_provider
        self._chemistry_provider = chemistry_provider
        # Quarantine-tier hook (2026-05-30): when a promoted concept
        # is "naked" (no live outgoing edges, even after auto-
        # restoring its quarantined edges), call this fn so the
        # forager's curiosity channel receives a fresh request.
        # The agent's attentional focus routes around the empty
        # neighborhood via existing machinery — no parallel system.
        # Optional; absent → no naked-concept request is emitted
        # (the bubble + naked flag still get set on the entry).
        self._information_request_fn = information_request_fn
        self.capacity = int(capacity)
        self.history_per_concept = int(history_per_concept)
        self._cycle_provider = cycle_provider
        self._entries: Dict[str, AWMEntry] = {}
        # Diagnostics.
        self.promotions: int = 0
        self.evictions: int = 0
        self.capacity_expansions: int = 0
        self.bubbles_spawned: int = 0
        self.bubbles_reused: int = 0
        # Phase F.14 — decay-back state.  Initial capacity is
        # the floor we never shrink below.
        self._initial_capacity: int = int(self.capacity)
        self._quiet_ticks: int = 0
        self.capacity_decays: int = 0
        # Step 0 organ 4a (2026-05-27): AWM contraction state.
        # MetabolicDebt providers drive contraction:
        #   contraction = clip((debt - baseline)/full_scale, 0, 1).
        # AWM capacity floor comes from recent inference-chain
        # depths the agent has actually exercised — captured via
        # ThoughtProducedEvent on every method='inference' fire.
        # (node_count, tick) pairs; stale entries (age > staleness)
        # filtered on read; empty/all-stale window → cold-start
        # floor (3).  Persisted per R2.
        self._debt_provider = debt_provider
        self._baseline_provider = baseline_provider
        self._debt_full_scale_provider = debt_full_scale_provider
        self._recent_node_counts: Deque[Tuple[int, int]] = deque(
            maxlen=AWM_RECENT_NODES_WINDOW)
        self.inference_node_counts_recorded: int = 0

    # ---- public access ----

    def get(self, name: str) -> Optional[AWMEntry]:
        return self._entries.get(name)

    def active_concepts(self) -> List[str]:
        return list(self._entries.keys())

    def size(self) -> int:
        return len(self._entries)

    def is_active(self, name: str) -> bool:
        return name in self._entries

    # ---- promotion / eviction ----

    def promote(self,
                 name: str,
                 *,
                 salience: float = 0.0,
                 origin: str = '',
                 cycle: Optional[int] = None) -> AWMEntry:
        """Add `name` to AWM (or refresh if already present).
        Looks up bubble from LTS if available; creates default
        bubble otherwise.  Emits AWM_ACTIVATION event."""
        if cycle is None:
            cycle = self._cycle()

        # Salience credit is EARNED, not farmed (2026-06-04 audit #4a):
        # the concept-salience bump moved to the NEW-promotion path
        # below (after the refresh early-return), so re-promoting an
        # already-active concept no longer pays salience.  Re-activation
        # is not a productive outcome — frequency-farming here ranked
        # ruminated noise into the reverie focal pool.  The bubble /
        # quarantine bid-path still fires on every promotion.
        substrate = None
        concept = None
        if self._lts_provider is not None:
            try:
                lts = self._lts_provider()
                if lts is not None:
                    concept = lts.get_concept(name)
                    substrate = getattr(lts, 'substrate', None) or lts
            except Exception:
                pass

        # Quarantine bid-path #4 (AWM promotion is the strongest
        # restore signal — it encodes goal-relevance / attended
        # focus).  Restore any quarantined edges touching this
        # concept BEFORE checking for naked-concept emptiness;
        # otherwise auto-restore would never fire on the very
        # promotion that should trigger it.
        if substrate is not None and hasattr(
                substrate, 'restore_concept_quarantine'):
            try:
                substrate.restore_concept_quarantine(
                    name, cycle=cycle)
            except Exception:
                pass

        existing = self._entries.get(name)
        if existing is not None:
            # Refresh — re-activate.
            existing.last_active_cycle = cycle
            existing.activation = min(1.0, existing.activation + 0.3)
            existing.salience_at_promotion = max(
                existing.salience_at_promotion, salience)
            return existing

        # Genuinely-new promotion EARNS the salience bump (audit #4a):
        # first attention to a concept this residence credits it; mere
        # re-promotion (handled above) does not.
        if concept is not None and hasattr(concept, 'bump_salience'):
            try:
                from seagi.core.substrate import SALIENCE_BUMP_DEFAULT
                concept.bump_salience(
                    SALIENCE_BUMP_DEFAULT * max(0.1, float(salience)),
                    cycle)
            except Exception:
                pass

        # New entry.  Try to bootstrap bubble from LTS, using
        # context-aware selection (find-or-spawn against the
        # concept's existing bubble pool).
        bubble = self._bootstrap_bubble(name, salience, cycle)
        # Naked-concept detection: a concept whose outgoing edges
        # are entirely quarantined (post-restoration above) OR who
        # has no outgoing edges at all is "naked" — reverie reaching
        # for it produces no derivations.  We PROMOTE ANYWAY (the
        # round-2-audit fix: skip-promotion would sever the
        # chemistry-imprint bid-path that's the only mechanism to
        # restore the concept later).  We mark `naked=True` and
        # signal the forager so its curiosity channel seeks input
        # — the rumination plateau routes around empty
        # neighborhoods via existing machinery.
        is_naked = False
        if substrate is not None:
            concept_obj = substrate.concepts.get(name)
            if concept_obj is not None:
                # Count live outgoing edges (post-restore).
                live_out = sum(
                    len(b)
                    for b in concept_obj.edges_out.values())
                is_naked = (live_out == 0)
            else:
                # Concept not in substrate at all — naked by
                # construction.
                is_naked = True
        if is_naked and self._information_request_fn is not None:
            try:
                self._information_request_fn(name)
            except Exception:
                pass
        entry = AWMEntry(
            concept_name=name,
            bubble=bubble,
            chemistry_history=ChemistryRingBuffer(
                capacity=self.history_per_concept),
            promoted_at_cycle=cycle,
            promoted_from_origin=origin,
            salience_at_promotion=salience,
            last_active_cycle=cycle,
            activation=1.0,
            naked=is_naked,
        )
        # Capacity check.  Step 0 organ 4a (2026-05-27): the
        # operating cap is `effective_capacity`, which equals the
        # soft `self.capacity` under no debt but contracts toward
        # `awm_capacity_floor` as debt rises.  Under contraction
        # we do NOT grow even if the incoming entry is high-
        # salience — burst expansion is for normal-state pressure,
        # not sleep-deprivation pressure.  Eviction is the right
        # response when contracted.
        op_capacity = self.effective_capacity(cycle=cycle)
        if len(self._entries) >= op_capacity:
            contracted = (op_capacity < self.capacity)
            if (not contracted
                    and salience >= CAPACITY_PRESSURE_SALIENCE_FLOOR
                    and self._victim_is_valuable(cycle)
                    and self.capacity < AWM_HARD_MAX):
                self._expand_capacity()
            else:
                self._evict_lowest_priority(cycle)
        self._entries[name] = entry
        self.promotions += 1
        # Emit activation event.
        try:
            self.bus.publish(BrainEvent(
                kind=EventKind.AWM_ACTIVATION,
                cycle=cycle,
                timestamp=time.time(),
                source_capability='awm',
                origin=origin,
                origin_detail=name,
            ))
        except Exception:
            pass
        return entry

    def evict(self, name: str) -> Optional[AWMEntry]:
        entry = self._entries.pop(name, None)
        if entry is None:
            return None
        # Post-merge (2026-05-14): the Bubble in this entry IS
        # the substrate's Bubble for this concept.  Eviction
        # just removes the AWMEntry wrapper; the bubble stays
        # in concept.bubbles[] with whatever chemistry trace
        # accumulated during AWM residency.  No write-back
        # needed; persistence is by construction.
        self.evictions += 1
        try:
            self.bus.publish(BrainEvent(
                kind=EventKind.AWM_EVICTION,
                cycle=self._cycle(),
                timestamp=time.time(),
                source_capability='awm',
                origin='internal',
                origin_detail=name,
            ))
        except Exception:
            pass
        return entry

    def flush_all_to_substrate(self) -> int:
        """No-op after the v1/v2 bubble unification.  Bubbles in
        AWM ARE substrate bubbles; there's nothing to flush.
        Method kept for backward-compat with any caller that
        still calls it (run_foundational_ingestion.py, the
        Brain.flush_to_substrate convenience).  Returns the
        size of AWM for diagnostic continuity."""
        return len(self._entries)

    # ---- Step 0 organ 4a: contraction queries ----

    def _contraction(self) -> float:
        """Returns contraction ∈ [0, 1] driven by MetabolicDebt:
            contraction = clip((debt − baseline) / full_scale, 0, 1).
        Absent providers → 0 (cold-start safety, no contraction)."""
        if (self._debt_provider is None
                or self._debt_full_scale_provider is None):
            return 0.0
        try:
            debt = float(self._debt_provider())
            baseline = (float(self._baseline_provider())
                          if self._baseline_provider else 0.0)
            scale = float(self._debt_full_scale_provider())
        except Exception:
            return 0.0
        if scale <= 0.0:
            return 0.0
        ratio = (debt - baseline) / scale
        if ratio <= 0.0:
            return 0.0
        if ratio >= 1.0:
            return 1.0
        return ratio

    def awm_capacity_floor(self, cycle: Optional[int] = None) -> int:
        """The minimum cap below which sleep-deprivation-driven
        contraction cannot drop AWM.

        Floor = min over the rolling deque of valid (non-stale)
        node-counts the agent has actually exercised in R.1
        inference.  Stale entries (age > AWM_RECENT_NODES_STALENESS_
        TICKS = 200) excluded.  Empty / all-stale window → cold-
        start floor = AWM_CAPACITY_COLD_START_FLOOR = 3 (the
        simplest legal inference chain visits 3 concepts)."""
        if cycle is None:
            cycle = self._cycle()
        valid: List[int] = []
        for (count, tick) in self._recent_node_counts:
            if (cycle - tick) <= AWM_RECENT_NODES_STALENESS_TICKS:
                valid.append(count)
        if not valid:
            return AWM_CAPACITY_COLD_START_FLOOR
        return max(AWM_CAPACITY_COLD_START_FLOOR, min(valid))

    def effective_capacity(self, cycle: Optional[int] = None) -> int:
        """The operating capacity right now, after debt-driven
        contraction.

        effective_capacity = max(awm_capacity_floor,
                                 initial_capacity × (1 − contraction))

        Bounded above by the burst-expanded self.capacity (this
        method NEVER widens the cap beyond the soft ratchet).  At
        cold-start (no debt providers) returns self.capacity
        unchanged.
        """
        contraction = self._contraction()
        if contraction <= 0.0:
            return int(self.capacity)
        floor = self.awm_capacity_floor(cycle=cycle)
        contracted = max(floor,
                          int(self._initial_capacity
                              * (1.0 - contraction)))
        # Never enlarge beyond the soft cap (only the burst-
        # expansion path widens it).
        if contracted > self.capacity:
            return int(self.capacity)
        return int(contracted)

    def _victim_is_valuable(self, cycle: int) -> bool:
        """True when the entry that WOULD be evicted is itself
        high-value — meaning the current capacity is too tight
        for what's competing for it.  This is the signal to
        GROW rather than evict.

        Signal: would-be victim's eviction-priority score is
        above the pressure floor."""
        if not self._entries:
            return False
        victim = min(self._entries.values(),
                         key=lambda e: self._eviction_priority(e, cycle))
        return (self._eviction_priority(victim, cycle)
                  >= CAPACITY_PRESSURE_SALIENCE_FLOOR)

    def _expand_capacity(self) -> None:
        """Grow the soft cap by AWM_GROWTH_FACTOR — sustained
        pressure justifies more working memory.  Capped at
        AWM_HARD_MAX for memory safety."""
        new_capacity = min(
            AWM_HARD_MAX,
            max(self.capacity + 1,
                int(self.capacity * AWM_GROWTH_FACTOR)))
        if new_capacity > self.capacity:
            self.capacity = new_capacity
            self.capacity_expansions += 1
        # Expansion resets the quiet counter — the system is
        # under load again.
        self._quiet_ticks = 0

    def maybe_decay_capacity(self) -> None:
        """Phase F.14 (2026-05-16): per-tick capacity decay-back.

        AWM under-utilization (load < DECAY_UTILIZATION_THRESHOLD
        × capacity) increments a quiet-ticks counter.  Once the
        counter reaches DECAY_QUIET_TICKS_REQUIRED, shrink the
        soft cap by DECAY_FACTOR (never below the initial
        capacity floor) and reset the counter.

        Called once per Brain.tick().  Cheap — no scan, just
        threshold checks and an occasional integer divide.

        Doctrine: AWM is a soft cap, not a ratchet.  Burst-
        expansion under sustained pressure → eventual decay-back
        when pressure subsides.
        """
        threshold = self.capacity * DECAY_UTILIZATION_THRESHOLD
        if len(self._entries) < threshold:
            self._quiet_ticks += 1
        else:
            # Load is back up; reset.
            self._quiet_ticks = 0
            return
        if self._quiet_ticks < DECAY_QUIET_TICKS_REQUIRED:
            return
        if self.capacity <= self._initial_capacity:
            # Already at floor; nothing to decay.  Reset so the
            # counter doesn't grow indefinitely.
            self._quiet_ticks = 0
            return
        new_capacity = max(
            self._initial_capacity,
            int(self.capacity * DECAY_FACTOR))
        if new_capacity < self.capacity:
            self.capacity = new_capacity
            self.capacity_decays += 1
        self._quiet_ticks = 0

    def _eviction_priority(self,
                                  e: AWMEntry,
                                  cycle: int) -> float:
        cycles_since = max(0, cycle - e.last_active_cycle)
        recency = math.exp(-cycles_since / RECENCY_TAU)
        crystal_floor = max(0.0, e.bubble.crystallization)
        mortality_bonus = max(0.0, e.bubble.m_polarity() - 0.1)
        return (
            0.4 * recency
            + 0.3 * e.salience_at_promotion
            + 0.2 * crystal_floor
            + 0.1 * mortality_bonus)

    def _evict_lowest_priority(self, cycle: int) -> None:
        """Drop the entry with the lowest eviction_priority.
        Score components:
          recency: exp(-cycles_since_active / RECENCY_TAU)
          salience: salience_at_promotion
          crystallization_floor: bubble.crystallization
          mortality bonus: small protective for M-leaning concepts
        Lowest combined → evicted."""
        if not self._entries:
            return
        victim = min(
            self._entries.values(),
            key=lambda e: self._eviction_priority(e, cycle))
        self.evict(victim.concept_name)

    # ---- chemistry time-series ----

    def append_chemistry_sample(self,
                                    name: str,
                                    sample: ChemistrySample
                                    ) -> None:
        entry = self._entries.get(name)
        if entry is not None:
            entry.chemistry_history.append(sample)

    def trajectory(self,
                     name: str,
                     window: int = 10) -> List[ChemistrySample]:
        entry = self._entries.get(name)
        if entry is None:
            return []
        return entry.chemistry_history.trajectory(window)

    # ---- decay (called per tick / event-processor step) ----

    def decay_tick(self, cycle: Optional[int] = None) -> None:
        if cycle is None:
            cycle = self._cycle()
        # Update activations.  Below floor → evict.
        EVICT_FLOOR = 0.05
        to_evict: List[str] = []
        for name, entry in self._entries.items():
            entry.update_activation(cycle)
            if (entry.activation < EVICT_FLOOR
                    and cycle - entry.last_active_cycle > RECENCY_TAU):
                to_evict.append(name)
        for name in to_evict:
            self.evict(name)

    # ---- event handling ----

    def handle(self,
                 event: BrainEvent,
                 bus: EventBus) -> None:
        if isinstance(event, AttendedPerceptEvent):
            self._on_attended(event)
        elif isinstance(event, ThoughtProducedEvent):
            # Step 0 organ 4a: record inference chain depths for
            # the capacity-floor calculation.  Only inference paths
            # contribute (chain_depth > 0 on those, == 0 on others).
            if event.method == 'inference':
                node_count = int(event.chain_depth) + 1
                if node_count >= 2:
                    self._recent_node_counts.append(
                        (node_count, int(event.cycle)))
                    self.inference_node_counts_recorded += 1

    def _on_attended(self, ev: AttendedPerceptEvent) -> None:
        """Gate-passed percept becomes AWM activations."""
        for focal in ev.focals:
            self.promote(
                focal,
                salience=ev.salience,
                origin=ev.origin,
                cycle=ev.cycle)

    # ---- bootstrap helpers ----

    def _bootstrap_bubble(self,
                             name: str,
                             salience: float = 0.0,
                             cycle: int = 0) -> EnrichedBubble:
        """Find or spawn the Bubble for this concept under the
        CURRENT context.

        Doctrine: same concept, different context → different
        bubble.  This function uses `find_or_spawn_bubble` from
        seagi.core.bubble to:
          1. Build a ContextKey from current global chemistry +
             currently-coactive AWM concept names
          2. Compare against each existing bubble's context_key
          3. Reuse closest match if similarity is high enough,
             OR spawn a new bubble if context drift + intensity
             both justify the cost

        Returns the Bubble that subsequent chemistry imprints
        should land on.  The bubble is guaranteed to be in
        `concept.bubbles` (appended if newly spawned, persisted
        by construction).

        The `salience` arg is used as the "intensity" signal for
        the spawn decision.  Low-salience encounters reuse the
        closest existing bubble; high-salience ones can spawn.
        """
        from seagi.core.bubble import (
            compute_context_key, find_or_spawn_bubble)

        # Build the current ContextKey.
        ck = None
        if self._chemistry_provider is not None:
            try:
                chem = self._chemistry_provider()
            except Exception:
                chem = None
            if chem is not None:
                global_state = getattr(chem, 'global_state', {})
                # Coactive concepts = those CURRENTLY in AWM
                # (this one isn't yet — it's being promoted).
                coactive = list(self._entries.keys())[
                    :  # caller-provided salience order would be
                       # nicer but list iteration order is good
                       # enough for context fingerprint purposes
                ]
                try:
                    ck = compute_context_key(global_state, coactive)
                except Exception:
                    ck = None

        if self._lts_provider is None:
            b = EnrichedBubble(concept_name=name,
                                  context_key=ck,
                                  created_cycle=cycle,
                                  last_active_cycle=cycle)
            self.bubbles_spawned += 1
            return b
        try:
            lts = self._lts_provider()
        except Exception:
            lts = None
        if lts is None:
            b = EnrichedBubble(concept_name=name,
                                  context_key=ck,
                                  created_cycle=cycle,
                                  last_active_cycle=cycle)
            self.bubbles_spawned += 1
            return b
        try:
            concept = lts.get_concept(name)
        except Exception:
            concept = None
        if concept is None:
            b = EnrichedBubble(concept_name=name,
                                  context_key=ck,
                                  created_cycle=cycle,
                                  last_active_cycle=cycle)
            self.bubbles_spawned += 1
            return b
        bubbles = getattr(concept, 'bubbles', None)
        if bubbles is None:
            try:
                concept.bubbles = []
            except Exception:
                b = EnrichedBubble(concept_name=name,
                                      context_key=ck,
                                      created_cycle=cycle,
                                      last_active_cycle=cycle)
                self.bubbles_spawned += 1
                return b
            bubbles = concept.bubbles

        # Use context-aware selection if we have a real
        # ContextKey; otherwise (no chemistry provider) fall
        # back to highest-encounter-count single-bubble logic.
        if ck is not None:
            n_before = len(bubbles)
            bubble = find_or_spawn_bubble(
                concept=concept,
                current_context=ck,
                magnitude=float(salience),
                cycle=cycle)
            # Stamp concept_name (v1 path may leave it blank).
            if not bubble.concept_name:
                bubble.concept_name = name
            if len(bubbles) > n_before:
                self.bubbles_spawned += 1
            else:
                self.bubbles_reused += 1
            return bubble

        # Legacy fallback (no chemistry provider): pick highest-
        # encounter or create fresh.  Used in test setups that
        # construct AWM directly without a chemistry provider.
        if bubbles:
            best = max(
                bubbles,
                key=lambda b: int(
                    getattr(b, 'encounter_count', 0) or 0))
            if hasattr(best, 'concept_name') and not best.concept_name:
                try:
                    best.concept_name = name
                except Exception:
                    pass
            return best
        # Fresh bubble — create and append to substrate NOW so
        # it's persisted from promotion.  The reference AWM
        # holds is the same one substrate holds.
        new_bubble = EnrichedBubble(concept_name=name)
        try:
            bubbles.append(new_bubble)
        except Exception:
            pass
        return new_bubble

    def _cycle(self) -> int:
        try:
            return int(self._cycle_provider()) if self._cycle_provider else 0
        except Exception:
            return 0

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'size': len(self._entries),
            'capacity': self.capacity,
            # Step 0 organ 4a (2026-05-27): contraction diagnostics.
            'effective_capacity': self.effective_capacity(),
            'capacity_floor': self.awm_capacity_floor(),
            'contraction': float(self._contraction()),
            'recent_node_counts_size': len(self._recent_node_counts),
            'inference_node_counts_recorded':
                int(self.inference_node_counts_recorded),
            'promotions_total': self.promotions,
            'evictions_total': self.evictions,
            'capacity_expansions': self.capacity_expansions,
            'concepts_sample': list(self._entries.keys())[:10],
        }

    # ---- persistence (Step 0 organ 4a — recent_node_counts) ----

    def to_dict(self) -> Dict[str, Any]:
        """Persist only the contraction-relevant rolling state.
        AWM entries / bubbles are session-local and live on the
        substrate; they are NOT serialized here."""
        return {
            'recent_node_counts': [
                [int(c), int(t)]
                for (c, t) in self._recent_node_counts
            ],
        }

    def load_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        values = state.get('recent_node_counts')
        if isinstance(values, (list, tuple)):
            self._recent_node_counts.clear()
            for item in values:
                try:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        self._recent_node_counts.append(
                            (int(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
