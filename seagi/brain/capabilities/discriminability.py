"""DiscriminabilityTracker — substrate-D sampler & d_modulation
provider for the Step 0 sleep gate.

Brain analog: cortical microcircuits depend on rank separation
between top-firing and second-firing assemblies to discriminate.
When edges saturate to ceiling, the gradient collapses and the
substrate behaves like undifferentiated mass — a clear cost.
That cost should pull the sleep gate DOWN (cheaper sleep when
discriminability has collapsed), and Phase S downscaling can
restore the gradient.

D = Substrate.discriminability() = mean of (top - second) edge
strengths over concepts with >= 2 outgoing edges.

This tracker captures `wake_onset_D_baseline` as the average D
over the first WAKE_ONSET_D_WINDOW ticks after each wake_onset.
The baseline is persisted (R2 lock).  Current-D vs baseline gives
the d_modulation term in the SleepRegulator gate:

    d_modulation =
        clip((baseline - current_D) /
             (baseline * (1 - EDGE_PRUNE_FLOOR/PROVISIONAL_EDGE_STRENGTH)),
             0, 1)

When D = baseline → d_modulation = 0 (no modulation).
When D = baseline × (EDGE_PRUNE_FLOOR/PROVISIONAL_EDGE_STRENGTH)
= baseline × 0.2 → d_modulation = 1 (full modulation, gate halves).

Cold-start (no baseline captured yet): d_modulation = 0 gates the
term out automatically.

Doctrine traces
---------------
WAKE_ONSET_D_WINDOW = RAPHE_HISTORY_DEPTH = 50.
D_COLLAPSE_RATIO    = EDGE_PRUNE_FLOOR/PROVISIONAL_EDGE_STRENGTH
                    = 0.02 / 0.1 = 0.2.
Both numbers trace; no free parameters.

Cost note: D is sampled sparsely during the post-wake window
(WAKE_ONSET_D_SAMPLE_STRIDE = 10 ticks → 5 samples per window) and
on demand for d_modulation queries.  Full-pass cost is O(E), which
on the canonical substrate is ~500K ops — fast Python but not
free.  Per-tick during the window: 1 sample every STRIDE ticks.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..events import EventKind, BrainEvent, ChemistryEvent
from ..bus import EventBus
from .awm import AWM_CAPACITY_COLD_START_FLOOR


# Width of the post-wake D-baseline collection window, in ticks.
# Equals RAPHE_HISTORY_DEPTH for doctrine trace.
WAKE_ONSET_D_WINDOW = 50

# Sample stride within the window.  Derived from existing doctrine
# constants — one sample per minimum-legal-chain-depth's worth of
# ticks (audit Q5).  WAKE_ONSET_D_WINDOW (= RAPHE_HISTORY_DEPTH =
# 50) divided by AWM_CAPACITY_COLD_START_FLOOR (= 3 = the smallest
# inference chain a substrate can support) = 16 ticks/sample.
# Reading: "the rolling D window samples at the granularity of the
# smallest cognition unit it serves."  Zero free parameters; the
# count of samples per window (3) IS the cold-start floor.
WAKE_ONSET_D_SAMPLE_STRIDE = (
    WAKE_ONSET_D_WINDOW // AWM_CAPACITY_COLD_START_FLOOR)

# Denominator factor for d_modulation = 1 - EDGE_PRUNE_FLOOR /
# PROVISIONAL_EDGE_STRENGTH = 1 - 0.02/0.1 = 0.8.  When current_D
# falls to baseline * (1 - this) = baseline * 0.2, the modulator
# pegs at 1.0.  Locked at module load to avoid an import-time
# circular into seagi.core.substrate.
D_MODULATION_DENOM_FACTOR = 0.8


class DiscriminabilityTracker:
    """Captures wake_onset_D_baseline + exposes the d_modulation
    provider for the SleepRegulator gate."""

    SUBSCRIPTIONS = (EventKind.CHEMISTRY_FIRE,)

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None,
                 substrate_provider: Optional[Callable] = None):
        """
        Args:
            substrate_provider: () -> Substrate or None.  Returns
                the live engine substrate.  None when no engine
                attached (brain-standalone mode) — tracker stays
                silent and d_modulation returns 0.
        """
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._substrate_provider = substrate_provider

        # Persisted: the most recent finalized baseline.
        self.wake_onset_D_baseline: float = 0.0

        # Active-window state (re-initialized per wake_onset).
        # Not persisted: a wake_onset event re-opens the window
        # on next session anyway.
        self._post_wake_samples: List[float] = []
        self._post_wake_tick_counter: int = 0
        self._sampling: bool = False

        # Diagnostics (not persisted).
        self.windows_completed: int = 0
        self.last_sampled_D: float = 0.0

        # d_modulation throttle cache (2026-06-04).  The sleep gate
        # queries d_modulation PER PERCEPT, and each query recomputed
        # the O(E) discriminability sweep (~500K ops on the canonical).
        # When intake frequency rose (a fed reading corpus) that sweep
        # pegged the daemon at 100% CPU and held the brain-lock,
        # collapsing the tick rate and timing out HTTP.  D drifts far
        # slower than the per-percept query rate, so serve a cached D
        # for up to WAKE_ONSET_D_SAMPLE_STRIDE cycles (the established
        # D-sampling granularity — no new constant).  _compute_D /
        # current_D / the post-wake tick sampling stay EXACT.
        self._dmod_cache_D: Optional[float] = None
        self._dmod_cache_cycle: int = -10 ** 9

    # ---- event handler ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, ChemistryEvent):
            if event.chemistry_kind == 'wake_onset':
                # Open a fresh sampling window.  Any in-progress
                # window is dropped (the new wake supersedes).
                self._post_wake_samples = []
                self._post_wake_tick_counter = 0
                self._sampling = True

    # ---- per-tick ----

    def tick(self) -> None:
        """Drive the post-wake sampling window.  No-op outside
        the window."""
        if not self._sampling:
            return
        # Tick the window counter.  Sample on stride-aligned
        # ticks for bounded cost.
        if (self._post_wake_tick_counter
                % WAKE_ONSET_D_SAMPLE_STRIDE) == 0:
            d = self._compute_D()
            if d is not None:
                self._post_wake_samples.append(d)
                self.last_sampled_D = d
        self._post_wake_tick_counter += 1
        if self._post_wake_tick_counter >= WAKE_ONSET_D_WINDOW:
            # Close the window — finalize baseline.
            if self._post_wake_samples:
                self.wake_onset_D_baseline = (
                    sum(self._post_wake_samples)
                    / float(len(self._post_wake_samples)))
                self.windows_completed += 1
            self._post_wake_samples = []
            self._sampling = False

    # ---- queries ----

    def _compute_D(self) -> Optional[float]:
        if self._substrate_provider is None:
            return None
        try:
            sub = self._substrate_provider()
        except Exception:
            return None
        if sub is None:
            return None
        try:
            return float(sub.discriminability(self._cycle_provider()))
        except Exception:
            return None

    def current_D(self) -> float:
        """Returns the live D value.  Used by external diagnostics
        and by d_modulation_provider on demand."""
        d = self._compute_D()
        return 0.0 if d is None else d

    def d_modulation_provider(self) -> float:
        """Returns d_modulation in [0, 1] for the SleepRegulator
        gate.  Cold-start (no baseline captured yet): returns 0
        — the term gates itself out, debt threshold stays
        unmodulated by D until a baseline is established."""
        baseline = float(self.wake_onset_D_baseline)
        if baseline <= 0.0:
            return 0.0
        denom = baseline * D_MODULATION_DENOM_FACTOR
        if denom <= 0.0:
            return 0.0
        # Throttle the O(E) recompute on this per-percept hot path:
        # serve a cached D for up to WAKE_ONSET_D_SAMPLE_STRIDE cycles.
        cyc = int(self._cycle_provider())
        if (self._dmod_cache_D is not None
                and 0 <= cyc - self._dmod_cache_cycle
                < WAKE_ONSET_D_SAMPLE_STRIDE):
            d = self._dmod_cache_D
        else:
            d = self._compute_D()
            if d is not None:
                self._dmod_cache_D = d
                self._dmod_cache_cycle = cyc
        if d is None:
            return 0.0
        ratio = (baseline - d) / denom
        if ratio <= 0.0:
            return 0.0
        if ratio >= 1.0:
            return 1.0
        return ratio

    # ---- persistence ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            'wake_onset_D_baseline':
                float(self.wake_onset_D_baseline),
            'windows_completed': int(self.windows_completed),
        }

    def load_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        try:
            self.wake_onset_D_baseline = float(
                state.get('wake_onset_D_baseline', 0.0))
        except (TypeError, ValueError):
            self.wake_onset_D_baseline = 0.0
        try:
            self.windows_completed = int(
                state.get('windows_completed', 0))
        except (TypeError, ValueError):
            self.windows_completed = 0

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'wake_onset_D_baseline':
                float(self.wake_onset_D_baseline),
            'last_sampled_D': float(self.last_sampled_D),
            'sampling_window_open': bool(self._sampling),
            'samples_in_window': int(
                len(self._post_wake_samples)),
            'windows_completed': int(self.windows_completed),
        }
