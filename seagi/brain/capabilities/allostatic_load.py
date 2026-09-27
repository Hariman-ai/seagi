"""AllostaticLoad — chemistry baseline drift (Step 0 organ 2).

Brain analog: allostatic load is the cumulative cost of repeated
or sustained adaptation.  When a channel sits above (or below) its
healthy baseline for long stretches, the homeostatic set-point
slowly shifts.  Chronic stress shifts cortisol's baseline up;
chronic safety shifts it down.  The shifts accumulate quietly and
are released through sleep — the period when the body re-evaluates
what "normal" should mean.

This organ generalizes RapheNuclei's rolling-mean pattern to ALL
chemistry channels.  Each channel keeps a tonic_window of recent
concentration samples.  On every SleepOnsetEvent, each channel's
baseline drifts toward its tonic mean by DRIFT_RATE = 0.001
(EVENT_DELTAS promille floor) — the slowest timescale in the
architecture, as allostasis should be.

The summed |baseline − initial_baseline| across channels is the
**allostatic load** scalar, surfaced as `load_provider` for the
SleepRegulator's `(1 - load_modulation)` factor.  Higher load
pulls the effective sleep threshold down → tired agent sleeps
sooner.  Q3 gate (locked).

NO retirement
-------------
Per [[seagi-chemistry-never-fully-dissolves]]: chemistry tags do
NOT retire.  The dormant-K=3-sleeps logic from the original Step 0
brief is deliberately omitted.  Tags fade with disuse to a
miniature molecular presence but stay re-engageable.

Doctrine traces
---------------
DRIFT_RATE = 0.001  (EVENT_DELTAS promille floor)
TONIC_WINDOW_DEPTH = RAPHE_HISTORY_DEPTH = 50

Persistence
-----------
All per-channel `tonic_windows`, `baselines`, and
`initial_baselines` persist via R2 lock.  Restoring after a
session means drift continues from where the last shift left off,
not a cold-start that erases the agent's accumulated state.

Subscribes
----------
CHEMISTRY_FIRE — filters on chemistry_kind='sleep_onset' to drive
                  the baseline-drift step.

Per-tick
--------
tick() samples the current concentration of every channel into
its tonic_window.  Tick-based (not event-based) sampling is the
right semantic for "tonic" — it captures sustained presence
including quiescent stretches, not just event-driven spikes.

Cross-refs
----------
[[project_seagi_step0_wiring_plan]] — Task 3 spec
[[project_seagi_phase_b1_raphe]] — RapheNuclei pattern template
[[seagi-chemistry-never-fully-dissolves]] — no retirement
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

from ..events import EventKind, BrainEvent, ChemistryEvent
from ..bus import EventBus
from ..chemistry_types import CHANNELS
from .neuromodulators import RAPHE_HISTORY_DEPTH


# Slowest timescale in the architecture: allostatic baselines drift
# at the EVENT_DELTAS promille floor.  Doctrine: allostasis should
# be the slowest dynamic.  V2-locked.
DRIFT_RATE = 0.001

# Tonic window length, doctrine-traced to RAPHE_HISTORY_DEPTH (50).
TONIC_WINDOW_DEPTH = RAPHE_HISTORY_DEPTH


class AllostaticLoad:
    """Per-channel baseline drift + load_provider for the
    SleepRegulator gate."""

    SUBSCRIPTIONS = (EventKind.CHEMISTRY_FIRE,)

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None,
                 chemistry_provider: Optional[Callable] = None):
        """
        Args:
            chemistry_provider: () -> Dict[str, float].  Returns
                the live ChemistryEngine.global_state.  Absent →
                the organ is inert (load stays 0, baselines stay
                fixed).
        """
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._chemistry_provider = chemistry_provider

        # Per-channel tonic-concentration history (rolling window).
        # Persisted per R2.
        self._tonic_windows: Dict[str, Deque[float]] = {
            ch: deque(maxlen=TONIC_WINDOW_DEPTH)
            for ch in CHANNELS
        }
        # Drifting baselines (mutable).  Initialized to the static
        # channel baseline.  Persisted per Q1.
        self.baselines: Dict[str, float] = {
            ch: float(cfg['baseline']) for ch, cfg in CHANNELS.items()
        }
        # Frozen reference baselines (immutable post-construction).
        # Used to compute load = Σ|baseline − initial|/N.  Persisted
        # per Q1: a restart must NOT reset the agent's reference.
        self.initial_baselines: Dict[str, float] = {
            ch: float(cfg['baseline']) for ch, cfg in CHANNELS.items()
        }

        # Diagnostics (not persisted).
        self.sleep_onsets_observed: int = 0
        self.samples_observed: int = 0

    # ---- event handler ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, ChemistryEvent):
            if event.chemistry_kind == 'sleep_onset':
                self._on_sleep_onset()

    # ---- per-tick sampling ----

    def tick(self) -> None:
        """Sample the current concentration of every channel into
        its tonic window.  Tick-based (not event-based): captures
        sustained presence + quiescent stretches alike.  Cheap:
        one dict read + one deque append per channel = ~8 ops."""
        if self._chemistry_provider is None:
            return
        try:
            state = self._chemistry_provider() or {}
        except Exception:
            return
        for ch in CHANNELS:
            value = state.get(ch, CHANNELS[ch]['baseline'])
            self._tonic_windows[ch].append(float(value))
        self.samples_observed += 1

    # ---- drift step on sleep_onset ----

    def _on_sleep_onset(self) -> None:
        """Drift each channel's baseline toward its tonic mean by
        DRIFT_RATE.  Empty-window channels skipped (cold-start
        safety)."""
        for ch in CHANNELS:
            window = self._tonic_windows[ch]
            if not window:
                continue
            tonic_mean = sum(window) / float(len(window))
            current = self.baselines[ch]
            self.baselines[ch] = current + DRIFT_RATE * (
                tonic_mean - current)
        self.sleep_onsets_observed += 1

    # ---- queries ----

    def load(self) -> float:
        """Allostatic load scalar in [0, 1].

        load = (1/N) Σ_c |baseline_c − initial_baseline_c|

        Cold-start (no drift yet): returns 0 — the load_modulation
        term gates itself out automatically.

        Clamped to 1.0 so a runaway accumulation cannot drive the
        SleepRegulator's effective threshold negative.
        """
        n = len(CHANNELS)
        if n == 0:
            return 0.0
        total = 0.0
        for ch in CHANNELS:
            total += abs(
                self.baselines[ch] - self.initial_baselines[ch])
        load_val = total / float(n)
        if load_val < 0.0:
            return 0.0
        if load_val > 1.0:
            return 1.0
        return load_val

    def load_provider(self) -> float:
        """Callable contract for SleepRegulator's load_provider
        slot.  Returns the current load scalar."""
        return self.load()

    # ---- persistence ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            'baselines': dict(self.baselines),
            'initial_baselines': dict(self.initial_baselines),
            'tonic_windows': {
                ch: list(w) for ch, w in self._tonic_windows.items()
            },
            'sleep_onsets_observed':
                int(self.sleep_onsets_observed),
        }

    def load_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        bl = state.get('baselines', {})
        if isinstance(bl, dict):
            for ch in CHANNELS:
                if ch in bl:
                    try:
                        self.baselines[ch] = float(bl[ch])
                    except (TypeError, ValueError):
                        pass
        init_bl = state.get('initial_baselines', {})
        if isinstance(init_bl, dict):
            for ch in CHANNELS:
                if ch in init_bl:
                    try:
                        self.initial_baselines[ch] = float(
                            init_bl[ch])
                    except (TypeError, ValueError):
                        pass
        windows = state.get('tonic_windows', {})
        if isinstance(windows, dict):
            for ch in CHANNELS:
                values = windows.get(ch)
                if isinstance(values, (list, tuple)):
                    try:
                        self._tonic_windows[ch] = deque(
                            (float(v) for v in values),
                            maxlen=TONIC_WINDOW_DEPTH)
                    except (TypeError, ValueError):
                        pass
        try:
            self.sleep_onsets_observed = int(
                state.get('sleep_onsets_observed', 0))
        except (TypeError, ValueError):
            self.sleep_onsets_observed = 0

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'load': float(self.load()),
            'baselines': dict(self.baselines),
            'baseline_deltas': {
                ch: float(
                    self.baselines[ch]
                    - self.initial_baselines[ch])
                for ch in CHANNELS
            },
            'tonic_window_sizes': {
                ch: int(len(w))
                for ch, w in self._tonic_windows.items()
            },
            'sleep_onsets_observed':
                int(self.sleep_onsets_observed),
            'samples_observed': int(self.samples_observed),
        }
