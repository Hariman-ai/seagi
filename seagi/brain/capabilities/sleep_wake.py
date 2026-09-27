"""SleepRegulator — wake/sleep state machine.

Brain analog: brainstem reticular activating system (RAS) +
ventrolateral preoptic nucleus (VLPO) flip-flop.  Wake-promoting
and sleep-promoting cells inhibit each other → bistable state
governed by an accumulating "sleep pressure" (adenosine analog).

SEAGI doesn't have a body, so "sleep" is metaphorical — but the
state distinction is architecturally real:

    WAKE  — active processing: percept handling, chemistry imprints,
            prediction, decisions.  Pressure rises with activity.

    SLEEP — consolidation: hippocampus replay, AWM decay, substrate
            pruning happen at their normal rates.  Pressure
            dissipates per tick.

Transitions emit a ChemistryEvent (`sleep_onset` / `wake_onset`)
which the chemistry engine applies as the EVENT_DELTAS prescribe.
Downstream behavior shifts (thalamic gate, cerebellum, arousal
modulator) follow automatically through the existing
chemistry-modulated paths — no explicit `is_asleep` plumbing
needed for most consumers.

Step 0 refactor (2026-05-26)
----------------------------
The legacy attended-percept-pressure mechanism failed the
homeostatic-cost doctrine: a 3-day headless daemon run showed
pressure idling at 0.00 because reverie doesn't generate
AttendedPerceptEvent at the rate the calibration assumed.  Phase S
never engaged.

The post-Step-0 mechanism: sleep onset is driven by **substrate-
write-debt** (MetabolicDebt organ 1).  Every persistent substrate
mutation deposits 1 debt unit; debt is the source-of-truth for
sleep pressure.  The legacy `pressure` field is preserved as a
display/diagnostic quantity and as the wake-timer (until Phase S
organ 3 lands and debt-clearance drives wake too).

Sleep-onset rule (post-Step-0):
    Until first sleep episode: `debt > 50` (bootstrap absolute
        = EDGE_PRUNE_INTERVAL × COHERENCE_REINFORCE_BUMP, V4 lock).
    After first episode:
        effective_debt_threshold =
            debt_full_scale × (1 − load_mod) × (1 − d_mod)
        should_sleep := debt_above_baseline > effective_threshold
    Load + D modulators are pulled via optional providers; absent
    providers contribute 0 (no modulation) — design tolerates
    organs 2/3 not yet wired.

Wake transition remains pressure-dissipation-driven (legacy)
until organ 3 wires Phase S debt-clearance.  On sleep_onset we
set pressure := SLEEP_PRESSURE_HIGH so the existing dissipation
gives us a proper sleep duration (~160 ticks).

Subscribes
----------
ATTENDED_PERCEPT       — legacy pressure update (diagnostic only;
                          no longer triggers sleep_onset)
CHEMISTRY_FIRE         — legacy pressure update (diagnostic only)
SUBSTRATE_WRITE_QUEUED — NEW: poll debt and trigger sleep_onset
                          when threshold exceeded

Emits
-----
CHEMISTRY_FIRE with chemistry_kind='sleep_onset' / 'wake_onset'.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

from ..events import (
    EventKind, BrainEvent,
    AttendedPerceptEvent,
    ChemistryEvent,
    SubstrateWriteQueuedEvent,
)
from ..bus import EventBus


# Legacy pressure constants — preserved for diagnostic continuity
# and for the wake-timer (sleep duration before pressure decays
# below SLEEP_PRESSURE_LOW).  See module docstring for the Step 0
# refactor.
PRESSURE_PER_ATTENDED = 0.005
PRESSURE_PER_CHEMISTRY = 0.001
# Slow per-tick decay during wake.
WAKE_NATURAL_DECAY = 0.0005
# Per-tick dissipation during sleep — gives sleep its ~160-tick
# duration even before Phase S debt-clearance is wired.
SLEEP_DISSIPATION_RATE = 0.005
# Pressure thresholds.  HIGH is no longer a sleep-onset gate
# (that moved to debt); it's just the value we set pressure to
# on sleep_onset so dissipation produces the right sleep duration.
# LOW is the wake-onset trigger (preserved).
SLEEP_PRESSURE_HIGH = 0.9
SLEEP_PRESSURE_LOW = 0.1

# Step 0 bootstrap absolute (V4 lock).  Before any sleep episode
# has recorded clearance, this is the debt level that triggers
# the first sleep.  Derived from EDGE_PRUNE_INTERVAL ×
# COHERENCE_REINFORCE_BUMP = 2000 × 0.025 = 50.  Two existing
# Phase S constants; zero free parameters.
BOOTSTRAP_SLEEP_THRESHOLD = 50.0

# Adenosine channel baseline (part-b v2 fallback when no provider wired).
# Sourced from the canonical channel config — not a free literal.
from ..chemistry_types import CHANNELS as _CHANNELS  # noqa: E402
CHANNEL_ADENOSINE_BASELINE = _CHANNELS['adenosine']['baseline']


def _DHALF_ON():
    """The discriminability modulator halves the sleep gate instead of
    zeroing it.  /root/DHALF_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/DHALF_ON")
    except Exception:
        return False


class SleepRegulator:
    """Wake/sleep flip-flop driven by substrate-write-debt
    (post-Step-0).  Legacy pressure tracking preserved for
    diagnostics + wake-timer until Phase S debt-clearance lands."""

    SUBSCRIPTIONS = (
        EventKind.ATTENDED_PERCEPT,
        EventKind.CHEMISTRY_FIRE,
        EventKind.SUBSTRATE_WRITE_QUEUED,
    )

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None,
                 debt_provider: Optional[Callable] = None,
                 baseline_provider: Optional[Callable] = None,
                 debt_full_scale_provider: Optional[Callable] = None,
                 has_logged_episode_provider: Optional[Callable] = None,
                 load_provider: Optional[Callable] = None,
                 d_modulation_provider: Optional[Callable] = None,
                 adenosine_provider: Optional[Callable] = None,
                 adenosine_baseline_provider: Optional[Callable] = None,
                 idle_provider: Optional[Callable] = None):
        """
        Providers (all optional — absent provider = no modulation):
            debt_provider: () -> float
                Current accumulated debt.
            baseline_provider: () -> float
                wake_onset_baseline_debt at last sleep_onset.
            debt_full_scale_provider: () -> float
                The dimensional anchor (mean clearance per episode).
            has_logged_episode_provider: () -> bool
                Whether any sleep episode has logged real clearance.
                Until True, we use the bootstrap absolute threshold.
            load_provider: () -> float  (organ 2)
                Allostatic load in [0, 1].  Absent → no load mod.
            d_modulation_provider: () -> float  (organ 3)
                d_modulation in [0, 1] (1 = full D-collapse).
                Absent → no D mod.
        """
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._debt_provider = debt_provider
        self._baseline_provider = baseline_provider
        self._debt_full_scale_provider = debt_full_scale_provider
        self._has_logged_episode_provider = (
            has_logged_episode_provider)
        self._load_provider = load_provider
        self._d_modulation_provider = d_modulation_provider
        # Adenosine onset providers (part-b v2, 2026-07-13).  Onset is
        # driven by SLEEP-PRESSURE (adenosine) — the two-process Process
        # S.  It accrues from cognitive EFFORT (thinking / conflict /
        # world-solving) and only a NAP discharges it.  debt_* providers
        # are retained for telemetry only (de-wired from onset).
        # idle_provider now feeds a SOFT idle-readiness term, NOT a hard
        # quiescence gate (dirty_empty is dropped).
        self._adenosine_provider = adenosine_provider
        self._adenosine_baseline_provider = adenosine_baseline_provider
        self._idle_provider = idle_provider
        # Clearance calibration: full_scale = mean of adenosine DISCHARGED
        # per nap episode, SEEDED from the FIRST observed discharge (no
        # transplanted literal).  Until then, the bootstrap threshold
        # THETA_BOOT = (baseline + 1.0)/2 governs onset.
        from collections import deque as _deque
        try:
            from .metabolic_debt import CLEARANCE_DEQUE_MAXLEN as _CDM
        except Exception:
            _CDM = 10
        self._adenosine_clearance = _deque(maxlen=_CDM)
        self._episode_discharge: float = 0.0
        self._has_clearance_episode: bool = False

        self.pressure: float = 0.0
        self.state: str = 'wake'    # 'wake' or 'sleep'
        # Diagnostics.
        self.transitions_to_sleep: int = 0
        self.transitions_to_wake: int = 0
        # Part b: count of watchdog wakes (drain-bounded exit failed to
        # fire within the dissipation window).  Should stay 0 in healthy
        # operation; every increment is logged at WARNING.
        self._watchdog_wakes: int = 0
        # Last-computed need (P_sat + fade_backlog + dormant) at the most
        # recent onset check — telemetry only.
        self.last_need: int = 0
        # Track the effective threshold at the most recent
        # transition for inspection.
        self.last_effective_threshold: float = (
            BOOTSTRAP_SLEEP_THRESHOLD)
        # Guard re-entrancy: handle() can fire ChemistryEvent on
        # transition, which would otherwise recurse into handle().
        self._in_transition: bool = False

    # ---- event handler ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if self._in_transition:
            return
        if isinstance(event, AttendedPerceptEvent):
            # Legacy diagnostic pressure update.  No longer
            # triggers sleep_onset.
            if self.state == 'wake':
                self.pressure = min(
                    1.0, self.pressure + PRESSURE_PER_ATTENDED)
            return
        if isinstance(event, SubstrateWriteQueuedEvent):
            # Part b (2026-07-13): onset is now NEED-driven, not write-
            # driven — a substrate write means the agent is ACTIVE (not
            # idle), the opposite of an onset moment.  The need-driven
            # gate is polled at end-of-tick (Brain.tick → maybe_onset)
            # when it can see the settled, quiescent state.  Nothing to
            # do here.
            return
        if isinstance(event, ChemistryEvent):
            # Suppress recursion on our own emitted events.
            if event.source_capability == 'sleep_regulator':
                return
            # Legacy diagnostic pressure update.  No longer
            # triggers sleep_onset.
            if self.state == 'wake':
                self.pressure = min(
                    1.0, self.pressure + PRESSURE_PER_CHEMISTRY)

    # ---- post-Step-0 sleep gate ----

    def maybe_onset(self, cycle: int) -> None:
        """End-of-tick onset poll (part-b v2).  Brain.tick calls this
        AFTER idle_motivation has updated the idle clock, so idle-
        readiness reflects the settled state."""
        if self.state == 'wake':
            self._maybe_sleep(cycle)

    def _adenosine_baseline(self) -> float:
        if self._adenosine_baseline_provider is not None:
            try:
                return float(self._adenosine_baseline_provider())
            except Exception:
                pass
        return CHANNEL_ADENOSINE_BASELINE

    def _idle_readiness(self) -> float:
        """SOFT gate: idle_provider returns clip(idle_ticks /
        IDLE_TICK_THRESHOLD, 0, 1) — a readiness in [0,1] that lowers the
        onset threshold as the agent quiesces (it does NOT hard-block)."""
        if self._idle_provider is None:
            return 0.0
        try:
            return max(0.0, min(1.0, float(self._idle_provider())))
        except Exception:
            return 0.0

    def _mod(self, provider) -> float:
        if provider is None:
            return 0.0
        try:
            return max(0.0, min(1.0, float(provider())))
        except Exception:
            return 0.0

    def _full_scale(self) -> float:
        """Onset scale: mean of adenosine DISCHARGED per nap episode,
        seeded from the FIRST observed discharge.  Before any episode:
        the bootstrap midpoint THETA_BOOT − baseline = (1 − baseline)/2
        (no transplanted literal)."""
        _boot = max(0.0, (1.0 - self._adenosine_baseline()) / 2.0)
        if self._has_clearance_episode and self._adenosine_clearance:
            _learned = (sum(self._adenosine_clearance)
                        / len(self._adenosine_clearance))
            # FLOORED at his own bootstrap prior (2026-08-28).
            # MEASURED: with debt at 0 a nap discharges ~nothing, and
            # the mean then TEACHES him that ~nothing is normal --
            # 0.45 -> 0.00145, a 300x collapse -- so he napped
            # constantly: 1243 sleeps, 1242 one-tick wakes, ~95%
            # asleep, throughput ~1,200 steps/h vs ~6,500 awake.
            # The loop is backwards: naps that clear nothing should
            # make him nap LESS.  Learning may still RAISE the scale;
            # it can no longer collapse it toward zero.
            return max(_learned, _boot)
        return _boot

    def _maybe_sleep(self, cycle: int) -> None:
        """Adenosine (Process-S) sleep onset (part-b v2, 2026-07-13;
        idle-decoupled 2026-07-14).  Sleep pressure accrues from cognitive
        EFFORT and only a nap discharges it.  Onset is FATIGUE-GATED ONLY:

            excess = adenosine - adenosine_baseline
            theta  = full_scale * (1-d_mod)*(1-load_mod)
            sleep  when  excess > theta

        The `(1-idle_readiness)` term is REMOVED (idle-decoupling): a nap
        fires on genuine sleep pressure regardless of idle state, and idle
        alone never triggers one.  This collapses the idle-driven over-
        napping (v2 slept ~56% under idle reverie) — a RESTED idle agent
        does NOT nap.  full_scale is the bootstrap midpoint until the
        first nap records a discharge, then the clearance-deque mean.
        Absent adenosine provider ⇒ no onset (safe default for tests)."""
        if self._adenosine_provider is None:
            return
        try:
            aden = float(self._adenosine_provider())
        except Exception:
            return
        excess = max(0.0, aden - self._adenosine_baseline())
        self.last_need = excess
        # RECORD THE TERMS, not just the product.  `theta` read 0.0
        # while full_scale and load could not explain it, and two
        # inferences about which term was responsible were wrong.
        _fs = self._full_scale()
        _dm = self._mod(self._d_modulation_provider)
        _lm = self._mod(self._load_provider)
        # DHALF (2026-09-07): the D term HALVES the gate at full
        # modulation, per discriminability.py's own contract ('gate
        # halves').  (1 - d_mod) zeroed it: with d_mod = 1.0 in 13 of
        # 15 hourly samples theta was exactly 0, one ARC step re-armed
        # an 80-tick nap after a 1-tick wake, and he slept 71% of his
        # ticks (08-29, before D collapsed: 18%).  Gate absent => old.
        if _DHALF_ON():
            theta = _fs * (1.0 - 0.5 * _dm) * (1.0 - _lm)
        else:
            theta = _fs * (1.0 - _dm) * (1.0 - _lm)
        self.last_full_scale = _fs
        self.last_d_mod = _dm
        self.last_load_mod = _lm
        self.last_excess_at_eval = excess
        self.gate_evals = getattr(self, "gate_evals", 0) + 1
        if excess > theta:
            self.gate_slept = getattr(self, "gate_slept", 0) + 1
        self.last_effective_threshold = theta
        if excess > theta:
            self._transition_to_sleep(self.bus, cycle)

    # ---- per-tick dynamics ----

    def tick(self) -> None:
        """Called by Brain.tick().  Drives the slow pressure
        dynamics that don't tie to specific events: wake-natural-
        decay during wake, dissipation during sleep, plus the
        wake-up transition check.

        Sleep duration is paced by pressure dissipation until
        Phase S debt-clearance lands.  On sleep_onset we set
        pressure := SLEEP_PRESSURE_HIGH so the existing dissipation
        produces a proper ~160-tick sleep."""
        cycle = self._cycle_provider()
        if self.state == 'wake':
            self.pressure = max(
                0.0, self.pressure - WAKE_NATURAL_DECAY)
        else:
            # Wake exit is DRAIN-BOUNDED (note_nap_drain).  Pressure
            # dissipation is kept ONLY as a watchdog: if the drain-bounded
            # exit hasn't woken the agent by the time pressure decays
            # below LOW (~160 ticks), a nap made no progress — force wake
            # and log a WARNING (this should never fire in healthy op).
            self.pressure = max(
                0.0, self.pressure - SLEEP_DISSIPATION_RATE)
            if self.pressure <= SLEEP_PRESSURE_LOW:
                self._watchdog_wakes += 1
                logger.warning(
                    "SleepRegulator watchdog fired at cycle %s: drain-"
                    "bounded exit did not wake within the dissipation "
                    "window (nap made no progress?).  Forcing wake.",
                    cycle)
                self._transition_to_wake(self.bus, cycle)

    def note_nap_drain(self, edges_drained: int,
                         adenosine_discharged: float,
                         cycle: int) -> None:
        """Drain-bounded wake (part-b v2, 2026-07-13).  The runtime's NAP-
        GLOBAL branch calls this each pass with (i) the edges MOVED
        (telemetry) and (ii) the adenosine DISCHARGED this pass.

        The exit is ADENOSINE-PRIMARY: a nap exists to discharge Process-S
        (sleep pressure), so it ENDS when adenosine is back at baseline
        (a pass discharges nothing).  Global edge-maintenance (downscale /
        quarantine / replay) runs opportunistically DURING the nap but
        does NOT gate its length — on a dense substrate that maintenance
        never fully drains, and gating on it would pin every nap to the
        160-tick watchdog (measured).  Two-process-correct: sleep ends
        when the pressure is discharged, not when all cleanup is done.
        The discharge accumulates into the episode total (→ clearance
        deque on wake) so full_scale self-calibrates."""
        if self.state != 'sleep':
            return
        d = max(0.0, float(adenosine_discharged))
        self._episode_discharge += d
        if d <= 1e-9:
            self._transition_to_wake(self.bus, cycle)

    # ---- transitions ----

    def _transition_to_sleep(self,
                                    bus: EventBus,
                                    cycle: int) -> None:
        self._in_transition = True
        try:
            self.state = 'sleep'
            self.transitions_to_sleep += 1
            # Start a fresh adenosine-clearance episode.
            self._episode_discharge = 0.0
            # Set pressure to HIGH so the dissipation WATCHDOG fires
            # wake after ~160 ticks if the drain-bounded exit somehow
            # doesn't (part-b v2: exit is drain-bounded; this is a
            # safety net only).
            self.pressure = SLEEP_PRESSURE_HIGH
            try:
                bus.publish(ChemistryEvent(
                    kind=EventKind.CHEMISTRY_FIRE,
                    cycle=cycle,
                    timestamp=time.time(),
                    source_capability='sleep_regulator',
                    origin='internal',
                    origin_detail='state_change',
                    chemistry_kind='sleep_onset',
                    magnitude=1.0,
                ))
            except Exception:
                pass
        finally:
            self._in_transition = False

    def _transition_to_wake(self,
                                   bus: EventBus,
                                   cycle: int) -> None:
        self._in_transition = True
        try:
            self.state = 'wake'
            self.transitions_to_wake += 1
            # Record this nap's adenosine clearance so full_scale self-
            # calibrates.  SEEDED from the first observed discharge (no
            # transplanted literal); zero-discharge naps don't poison it.
            if self._episode_discharge > 0.0:
                self._adenosine_clearance.append(self._episode_discharge)
                self._has_clearance_episode = True
            try:
                bus.publish(ChemistryEvent(
                    kind=EventKind.CHEMISTRY_FIRE,
                    cycle=cycle,
                    timestamp=time.time(),
                    source_capability='sleep_regulator',
                    origin='internal',
                    origin_detail='state_change',
                    chemistry_kind='wake_onset',
                    magnitude=1.0,
                ))
            except Exception:
                pass
        finally:
            self._in_transition = False

    # ---- queries / diagnostics ----

    def is_asleep(self) -> bool:
        return self.state == 'sleep'

    def stats(self) -> Dict[str, Any]:
        return {
            'state': self.state,
            'pressure': self.pressure,
            'last_effective_threshold':
                float(self.last_effective_threshold),
            'transitions_to_sleep': self.transitions_to_sleep,
            'transitions_to_wake': self.transitions_to_wake,
            'watchdog_wakes': self._watchdog_wakes,
            'last_full_scale': float(getattr(self, 'last_full_scale', -1.0)),
            'last_d_mod': float(getattr(self, 'last_d_mod', -1.0)),
            'last_load_mod': float(getattr(self, 'last_load_mod', -1.0)),
            'last_excess_at_eval': float(getattr(self, 'last_excess_at_eval', -1.0)),
            'gate_evals': int(getattr(self, 'gate_evals', 0)),
            'gate_slept': int(getattr(self, 'gate_slept', 0)),
            'adenosine_excess': float(self.last_need),
            'adenosine_full_scale': float(self._full_scale()),
            'clearance_episodes': len(self._adenosine_clearance),
        }
