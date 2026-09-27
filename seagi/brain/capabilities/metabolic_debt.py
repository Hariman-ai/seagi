"""MetabolicDebt — substrate write-debt accumulator (Step 0 organ 1).

Brain analog: adenosine accumulation in basal forebrain glutamatergic
neurons.  Every neural activation deposits a tiny metabolic cost;
clearance is gated to sleep (glymphatic clearance / synaptic
homeostasis, SHY).  An organism that doesn't sleep keeps writing
without consolidating, and the cost compounds until cognition
degrades.

In SEAGI today, the legacy SleepRegulator runs on attended-percept +
chemistry-fire pressure (PRESSURE_PER_ATTENDED=0.005 etc.) that
does not register during pure headless reverie.  The 3-day daemon
run on alpha (2026-05-23 → 2026-05-26) confirmed: pressure stayed
0.00 the entire time, no sleep, no consolidation, and SEAGI was
unharmed.  Sleep didn't earn its existence.

MetabolicDebt replaces that mechanism with a substrate-write-debt
scalar.  Every persistent substrate mutation (per Q5: Edge.reinforce,
Substrate.add_edge, Concept.bump_salience, ReasoningConsolidator
publish, S.2 reinforce_coherent_edges per-edge) deposits one debt
unit.  Headless reverie produces debt because Step 2 derivations and
salience bumps are substrate writes — that's the whole point.

The debt scalar drives the new sleep gate (see SleepRegulator
refactor).  Clearance is gated to Phase S consolidation events
(coming with organ 3).  debt_full_scale auto-calibrates as the
rolling mean of recent episodes' clearance.

Doctrine cross-refs
-------------------
* [[feedback_homeostatic_cost_doctrine]] — every regulator must
  earn its existence; failure must carry a survival cost
* [[seagi-agi-as-self-preservation]] — architecture handles its
  own noise; trust Phase S earn-or-dissolve
* [[seagi-chemistry-never-fully-dissolves]] — presence floor;
  nothing fully retires
* [[project_seagi_step0_wiring_plan]] — full spec
* V4 (resolved): bootstrap_threshold = EDGE_PRUNE_INTERVAL ×
  COHERENCE_REINFORCE_BUMP = 2000 × 0.025 = 50.  Two existing
  constants composed, zero free parameters.  Substrate-independent
  at bootstrap because it fires once; debt_full_scale becomes
  substrate-derived (= mean(clearance_deque)) after the first
  real episode is recorded.

Subscribes
----------
SUBSTRATE_WRITE_QUEUED  — debt += 1 (one mutation = one debt unit)
CHEMISTRY_FIRE          — filters on chemistry_kind='sleep_onset'
                            (records wake_onset_baseline_debt) and
                            'wake_onset' (closes the episode and
                            appends clearance to the deque).

Public surface
--------------
record_consolidation(edges_touched) — called by Phase S (Organ 3)
    each S.3 pass to register actual clearance.  Until Organ 3
    lands, this is unused and clearance fall-back uses the
    bootstrap seed.
debt_provider()        — callable that returns current debt
debt_full_scale        — property returning mean(clearance_deque)
wake_onset_baseline_debt — property returning the snapshot at
                            last sleep_onset (or 0 if no sleep
                            has happened yet)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple

from ..events import (
    EventKind, BrainEvent,
    SubstrateWriteQueuedEvent,
    ChemistryEvent,
)
from ..bus import EventBus


# V4-locked bootstrap: EDGE_PRUNE_INTERVAL × COHERENCE_REINFORCE_BUMP
# = 2000 × 0.025 = 50.  Two existing Phase S constants composed.
# Used to seed _recent_clearance_per_episode before the first real
# sleep episode logs actual clearance, AND as the initial sleep
# threshold absolute the legacy gate compares against.
BOOTSTRAP_SEED_VALUE = 50

# V5-locked deque length: RAPHE_HISTORY_DEPTH / 5 = 50 / 5 = 10.
# One chemistry window's worth of sleep episodes.  Long enough to
# smooth across substrate-density changes; short enough to track
# real shifts.
CLEARANCE_DEQUE_MAXLEN = 10


class MetabolicDebt:
    """Substrate-write-debt accumulator.  Source of truth for the
    sleep-pressure organ in the post-Step-0 architecture."""

    SUBSCRIPTIONS = (
        EventKind.SUBSTRATE_WRITE_QUEUED,
        EventKind.CHEMISTRY_FIRE,
    )

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)

        # Core state (persisted across save/load per Q1).
        self.debt: float = 0.0
        self._wake_onset_baseline_debt: float = 0.0

        # Clearance deque (persisted per R2).  Cold-start seeded
        # with BOOTSTRAP_SEED_VALUE × CLEARANCE_DEQUE_MAXLEN; first
        # real sleep episode replaces the oldest entry; deque mean
        # drifts toward true clearance over CLEARANCE_DEQUE_MAXLEN
        # cycles.
        self._recent_clearance_per_episode: Deque[float] = deque(
            [float(BOOTSTRAP_SEED_VALUE)] * CLEARANCE_DEQUE_MAXLEN,
            maxlen=CLEARANCE_DEQUE_MAXLEN)

        # Per-episode accumulator.  Phase S calls
        # record_consolidation(edges_touched) on each S.3 pass; the
        # totals add into _current_episode_clearance.  On wake_onset
        # this becomes the new deque entry, and the accumulator
        # resets.  Not persisted (re-derives across restarts).
        self._current_episode_clearance: float = 0.0

        # Whether a sleep_onset has been observed.  Until the first
        # sleep happens, debt_full_scale stays at the bootstrap
        # seed (BOOTSTRAP_SEED_VALUE) by construction.  After the
        # first real consolidation, the deque shifts toward
        # measured values.  Diagnostic flag (not persisted).
        self._has_logged_first_episode: bool = False

        # Diagnostics (not persisted).
        self.writes_observed: int = 0
        self.sleep_onsets_observed: int = 0
        self.wake_onsets_observed: int = 0
        self.consolidations_recorded: int = 0
        self.total_clearance_recorded: float = 0.0

    # ---- event handler ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, SubstrateWriteQueuedEvent):
            # Per Q5: every persistent substrate mutation = one
            # debt unit.  Strength of the write does NOT matter
            # (a re-attestation at strength 0.1 counts the same
            # as a high-strength write).  The unit identity is
            # mutation-count, not strength-mass.
            self.debt += 1.0
            self.writes_observed += 1
            return

        if isinstance(event, ChemistryEvent):
            kind = event.chemistry_kind
            if kind == 'sleep_onset':
                # Open the episode: reset the per-episode clearance
                # accumulator.  The debt baseline is NOT snapshotted
                # here — it is captured at wake_onset as the
                # post-clearance residual (see below).  Snapshotting
                # the pre-sleep PEAK here was the baseline-ratchet
                # bug: it forced each cycle to re-accumulate debt a
                # prior sleep had already cleared, so the peak (and
                # the sleep interval) grew without bound.
                self._current_episode_clearance = 0.0
                self.sleep_onsets_observed += 1
                return
            if kind == 'wake_onset':
                # Close out the episode.  If Phase S (organ 3)
                # registered any clearance during the sleep, that
                # value becomes the new deque entry.  If not
                # (e.g., the sleep was too short for any S.3 pass),
                # use a bootstrap-seed-equivalent entry so the deque
                # doesn't get a zero (which would crash
                # debt_full_scale toward zero and trigger
                # continuous sleep).
                episode_clearance = self._current_episode_clearance
                if episode_clearance <= 0.0:
                    episode_clearance = float(BOOTSTRAP_SEED_VALUE)
                self._recent_clearance_per_episode.append(
                    episode_clearance)
                self._has_logged_first_episode = True
                self._current_episode_clearance = 0.0
                # Snapshot the baseline = debt REMAINING after Phase S
                # cleared it during this sleep.  debt_above_baseline
                # then measures NEW accrual since waking:
                #     should_sleep := (debt - wake_onset_baseline_debt)
                #                     > effective_debt_threshold
                # Anchoring to the cleared floor (not the pre-sleep
                # peak) is what makes the wake→sleep→wake loop stable
                # and periodic instead of ratcheting.
                self._wake_onset_baseline_debt = self.debt
                self.wake_onsets_observed += 1
                return

    # ---- Phase S hook (organ 3 will call this) ----

    def record_consolidation(self,
                              edges_touched: float) -> None:
        """Called by Phase S on each S.3 consolidation pass.
        `edges_touched` is the count of substrate mutations Phase S
        APPLIED during the pass (reinforce + abstractions + analogies
        + downscale + settle).

        Critical units distinction (the calibration anchor):
        `edges_touched` is whole-substrate WORK — settle/downscale
        iterate every edge and re-touch the same edges across passes,
        so on a dense substrate one sleep's edges_touched runs into
        the millions.  Debt, by contrast, is deposited at 1 unit per
        WAKE write (the day's new derivations + salience bumps).  The
        clearance that calibrates debt_full_scale must be measured in
        the SAME currency as debt — debt actually REMOVED, not work
        done — otherwise the threshold balloons to whole-substrate
        scale and the agent never sleeps.  So we accumulate `cleared`
        (bounded by available debt), not `edges_touched`.  Debt floors
        at 0.
        """
        if edges_touched <= 0:
            return
        # Clear debt (floored at 0).  `cleared` = debt actually
        # removed this pass, the quantity that calibrates the gate.
        cleared = min(self.debt, float(edges_touched))
        self.debt -= cleared
        if self.debt < 0.0:
            self.debt = 0.0
        self._current_episode_clearance += cleared
        self.consolidations_recorded += 1
        self.total_clearance_recorded += cleared

    # ---- queries ----

    def debt_provider(self) -> float:
        """Callable contract for downstream consumers (Sleep
        Regulator, AWM contraction, Gate attenuation)."""
        return self.debt

    @property
    def debt_full_scale(self) -> float:
        """The dimensional anchor.  Mean of recent sleep-episode
        clearance values.  Substrate-density-aware by construction
        because clearance scales with how much consolidation work
        Phase S actually did.

        Cold-start: returns BOOTSTRAP_SEED_VALUE because the deque
        is seeded with that value × CLEARANCE_DEQUE_MAXLEN.
        After the first real sleep episode logs clearance, this
        drifts toward measured values over ~10 cycles.
        """
        if not self._recent_clearance_per_episode:
            return float(BOOTSTRAP_SEED_VALUE)
        return (sum(self._recent_clearance_per_episode) /
                float(len(self._recent_clearance_per_episode)))

    @property
    def wake_onset_baseline_debt(self) -> float:
        """Debt level at the most recent sleep_onset.  Used by
        effective_debt_threshold formula to measure "debt
        accumulated above last wake-onset baseline."""
        return self._wake_onset_baseline_debt

    @property
    def debt_above_baseline(self) -> float:
        """`debt - wake_onset_baseline_debt`, floored at 0.  The
        canonical input to the new sleep gate formula."""
        delta = self.debt - self._wake_onset_baseline_debt
        return delta if delta > 0.0 else 0.0

    @property
    def has_logged_first_episode(self) -> bool:
        """True once any sleep episode has recorded clearance.
        Used by SleepRegulator to know whether to fall back to
        the bootstrap-absolute gate or use the modulator
        formula."""
        return self._has_logged_first_episode

    # ---- per-tick (no slow dynamics; tick is a no-op) ----

    def tick(self) -> None:
        """No per-tick activity.  Debt updates only on events.
        Method present for Brain.tick() symmetry with other
        capabilities; safely no-ops."""
        return

    # ---- persistence ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            'debt': float(self.debt),
            'wake_onset_baseline_debt':
                float(self._wake_onset_baseline_debt),
            'recent_clearance_per_episode':
                list(self._recent_clearance_per_episode),
            'has_logged_first_episode':
                bool(self._has_logged_first_episode),
        }

    def load_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        try:
            self.debt = float(state.get('debt', 0.0))
        except (TypeError, ValueError):
            self.debt = 0.0
        try:
            self._wake_onset_baseline_debt = float(
                state.get('wake_onset_baseline_debt', 0.0))
        except (TypeError, ValueError):
            self._wake_onset_baseline_debt = 0.0
        deque_values = state.get('recent_clearance_per_episode')
        if isinstance(deque_values, (list, tuple)) and deque_values:
            try:
                self._recent_clearance_per_episode = deque(
                    (float(v) for v in deque_values),
                    maxlen=CLEARANCE_DEQUE_MAXLEN)
            except (TypeError, ValueError):
                pass
        self._has_logged_first_episode = bool(
            state.get('has_logged_first_episode', False))
        # Re-derived; not persisted.
        self._current_episode_clearance = 0.0

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'debt': float(self.debt),
            'debt_above_baseline': float(self.debt_above_baseline),
            'wake_onset_baseline_debt':
                float(self._wake_onset_baseline_debt),
            'debt_full_scale': float(self.debt_full_scale),
            'clearance_deque':
                list(self._recent_clearance_per_episode),
            'has_logged_first_episode':
                bool(self._has_logged_first_episode),
            'writes_observed': int(self.writes_observed),
            'sleep_onsets_observed':
                int(self.sleep_onsets_observed),
            'wake_onsets_observed':
                int(self.wake_onsets_observed),
            'consolidations_recorded':
                int(self.consolidations_recorded),
            'total_clearance_recorded':
                float(self.total_clearance_recorded),
        }
