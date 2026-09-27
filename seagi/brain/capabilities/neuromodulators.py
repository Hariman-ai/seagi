"""Neuromodulators — VTA + Locus Coeruleus + Raphe.

All three are small brain-stem nuclei that BROADCAST a single
NT across the cortex on signal.  They are not where decisions
happen; they are where global state shifts.  Keeping them in
one file (instead of three thin ones) preserves how the brain
treats them: ascending modulators.

VTA — ventral tegmental area
----------------------------
Fires DOPAMINE on POSITIVE prediction errors (better than
expected).  Drives ValueLandscape updates on the same signal.
Negative PEs are NOT VTA's job — those go to ACC.

LC — locus coeruleus
--------------------
Fires NOREPINEPHRINE on threat / novelty / cortisol spike.
Tightens the Thalamic Gate (chemistry.arousal_modulator already
reads NE).  Bidirectional with Amygdala: amygdala threat →
LC NE → tighter gate / faster cycling.

Raphe — raphe nuclei (Phase B.1, 2026-05-18)
--------------------------------------------
Owns SEROTONIN.  Watches rolling means of cortisol and oxytocin
on every CHEMISTRY_FIRE.  When the rolling cortisol mean stays
elevated above CHRONIC_STRESS_THRESHOLD, fires 'chronic_stress'
(serotonin / dopamine / gaba drop — the mood floor lowers).
When the rolling oxytocin mean stays elevated above
SOCIAL_BOND_THRESHOLD, fires 'social_replenish' (serotonin /
endorphins lift).  Firing is cooldown-gated so a long stress
period produces a slow drift, not a flood.

Phase 4b: VTA + LC FULL.  Phase B.1: Raphe FULL.  Outputs are
ChemistryEvent firings (the chemistry engine already knows how
to apply them); VTA additionally emits ValueUpdatedEvent for
the focal involved.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

from ..events import (
    EventKind, BrainEvent,
    PredictionErrorEvent,
    ThreatDetectedEvent,
    ChemistryEvent,
    ValueUpdatedEvent,
)
from ..bus import EventBus
from ..chemistry_types import CHANNELS


VTA_POSITIVE_PE_GAIN = 1.2     # dopamine magnitude per PE unit
LC_THREAT_GAIN = 1.0           # NE magnitude per threat unit
LC_NOVELTY_GAIN = 0.5          # NE magnitude per novelty unit

# Phase B.1: Raphe constants.  Thresholds expressed as deviation
# from channel baseline (cortisol baseline 0.10, oxytocin 0.20),
# so '+0.05' means "rolling mean sustained 0.05 above baseline".
RAPHE_HISTORY_DEPTH = 50            # number of chemistry samples averaged
RAPHE_FIRE_COOLDOWN = 100           # min cycles between raphe fires
RAPHE_CHRONIC_STRESS_DELTA = 0.05   # cortisol rolling mean − baseline
# 2026-05-18 (stress-probe calibration): oxytocin decays 2.3× faster
# than cortisol (0.07 vs 0.03), so equilibrium excess under sustained
# 'mattering' conditioning settles around 0.036, well below 0.05.
# The threshold compensates for channel-dynamics asymmetry: in
# decay-adjusted terms, 0.03 oxytocin-excess and 0.05 cortisol-excess
# represent comparable sustained-elevation effort.
RAPHE_SOCIAL_BOND_DELTA = 0.03      # oxytocin rolling mean − baseline
RAPHE_MIN_SAMPLES_FOR_FIRE = 20     # build history before firing


class VTA:
    """Ventral tegmental area.  Reward prediction error →
    dopamine + ValueUpdated."""

    SUBSCRIPTIONS = (EventKind.PREDICTION_ERROR,)

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self.dopamine_fires: int = 0
        self.value_updates: int = 0

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if not isinstance(event, PredictionErrorEvent):
            return
        if event.sign <= 0:
            return  # negative PE is ACC's territory
        magnitude = min(1.0, event.magnitude * VTA_POSITIVE_PE_GAIN)
        cycle = event.cycle
        # Dopamine fire.
        try:
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=cycle,
                timestamp=time.time(),
                source_capability='vta',
                origin='internal',
                origin_detail='positive_pe',
                chemistry_kind='confirmed_i',
                magnitude=magnitude,
                target_concepts=(
                    [event.focal] if event.focal else []),
            ))
            self.dopamine_fires += 1
        except Exception:
            pass
        # Value bump.
        if event.focal:
            try:
                bus.publish(ValueUpdatedEvent(
                    kind=EventKind.VALUE_UPDATED,
                    cycle=cycle,
                    timestamp=time.time(),
                    source_capability='vta',
                    origin='internal',
                    origin_detail='positive_pe',
                    focal=event.focal,
                    delta=+magnitude * 0.1,
                    value=0.0,   # ValueLandscape fills accumulated
                    update_kind='positive_pe',
                ))
                self.value_updates += 1
            except Exception:
                pass

    def stats(self) -> Dict[str, int]:
        return {
            'dopamine_fires': self.dopamine_fires,
            'value_updates': self.value_updates,
        }


def _MPOLE_ON():
    """Let the mortality pole fire.  File-gated at /root/MPOLE_ON."""
    try:
        import os as _os
        return _os.path.exists('/root/MPOLE_ON')
    except Exception:
        return False


# The reward ledger ignores body deltas below this; reused rather than
# introducing a second, differently-tuned floor for the same signal.
BODY_DELTA_FLOOR = 0.05

# Candidate floors for the mortality pole, spanning four orders of
# magnitude.  Which one is right is an empirical question about how far his
# lifeforce actually moves, not a matter of taste.
MP_FLOOR_PROBES = (0.05, 0.01, 0.002, 0.0005, 0.0001, 0.00001)


class LocusCoeruleus:
    """Norepinephrine broadcaster.  Threats + novelty → alert
    state."""

    SUBSCRIPTIONS = (
        EventKind.THREAT_DETECTED,
        EventKind.ATTENDED_PERCEPT,
        # The LC is the noradrenergic alarm nucleus -- in the biology it is
        # precisely the UNPREDICTED-aversive-event detector, so the
        # mortality-model check belongs here and nowhere else.
        EventKind.INTEROCEPTION,
    )

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self.ne_fires: int = 0
        # ne_fires counts BOTH paths; split them so the threat share is
        # visible.  Only the threat path raises cortisol (+0.015).
        self.ne_threat_fires: int = 0
        self.ne_novelty_fires: int = 0
        # Did a threat fire since the last interoceptive sample?
        # `delta_lifeforce` is defined as "since last interoceptive sample",
        # so pairing on the sample boundary is exact and needs NO window
        # constant to be invented.
        self._threat_since_sample: bool = False
        self.confirmed_m_fires: int = 0
        self.falsified_m_fires: int = 0
        # shadow: what the mortality pole WOULD do at each candidate floor
        self.mp_seen: int = 0
        self.mp_adverse: int = 0
        self.mp_adverse_mass: float = 0.0
        self.mp_worst: float = 0.0
        self.mp_threat_pending: int = 0
        self.mp_at_floor = [0] * len(MP_FLOOR_PROBES)
        self.mp_pred_at_floor = [0] * len(MP_FLOOR_PROBES)

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, ThreatDetectedEvent):
            self._threat_since_sample = True
            self._fire_for_threat(event, bus)
        elif event.kind == EventKind.INTEROCEPTION:
            self._test_mortality_model(event, bus)
        elif (event.kind == EventKind.ATTENDED_PERCEPT
                and hasattr(event, 'novelty')
                and event.novelty >= 0.6):
            self._fire_for_novelty(event, bus)

    def _fire_for_threat(self,
                              ev: ThreatDetectedEvent,
                              bus: EventBus) -> None:
        magnitude = min(1.0, ev.magnitude * LC_THREAT_GAIN)
        try:
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=ev.cycle,
                timestamp=time.time(),
                source_capability='lc',
                origin='internal',
                origin_detail='threat',
                chemistry_kind='threat',
                magnitude=magnitude,
                target_concepts=(
                    [ev.focal] if ev.focal else []),
            ))
            self.ne_fires += 1
            self.ne_threat_fires += 1
        except Exception:
            pass

    def _test_mortality_model(self, ev: Any, bus: EventBus) -> None:
        """Did his danger model anticipate the harm that actually arrived?

        This is the mortality pole's only source.  An adverse body shift is
        the outcome; a threat detected since the last sample is the
        prediction.  Predicted harm SETTLES him (`confirmed_m`); harm out of
        nowhere alarms him (`falsified_m`).

        Only ADVERSE shifts test a danger model, so a good turn of events is
        left to the I pole, which already has `confirmed_i`.
        """
        predicted = self._threat_since_sample
        self._threat_since_sample = False
        try:
            d = float(getattr(ev, 'delta_lifeforce', 0.0) or 0.0)
        except Exception:
            return
        # SHADOW FIRST -- his lifeforce moves by ~1e-4 per sample, so a floor
        # borrowed from the reward ledger (0.05) would silence this organ
        # entirely.  Count what WOULD fire at each candidate floor, and how
        # often a threat preceded the harm, before choosing one.
        self.mp_seen += 1
        if predicted:
            self.mp_threat_pending += 1
        if d < 0:
            self.mp_adverse += 1
            self.mp_adverse_mass += abs(d)
            if abs(d) > self.mp_worst:
                self.mp_worst = abs(d)
            for i, f in enumerate(MP_FLOOR_PROBES):
                if d <= -f:
                    self.mp_at_floor[i] += 1
                    if predicted:
                        self.mp_pred_at_floor[i] += 1
        if not _MPOLE_ON():
            return
        if d > -BODY_DELTA_FLOOR:
            return
        kind = 'confirmed_m' if predicted else 'falsified_m'
        try:
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=int(getattr(ev, 'cycle', 0) or 0),
                timestamp=time.time(),
                source_capability='lc',
                origin='internal',
                origin_detail=('harm_foreseen' if predicted
                               else 'harm_unforeseen'),
                chemistry_kind=kind,
                magnitude=min(1.0, abs(d) * LC_THREAT_GAIN),
                target_concepts=[],
            ))
            if predicted:
                self.confirmed_m_fires += 1
            else:
                self.falsified_m_fires += 1
        except Exception:
            pass

    def _fire_for_novelty(self, ev: Any, bus: EventBus) -> None:
        magnitude = min(1.0, ev.novelty * LC_NOVELTY_GAIN)
        try:
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=ev.cycle,
                timestamp=time.time(),
                source_capability='lc',
                origin='internal',
                origin_detail='novelty',
                chemistry_kind='curiosity',
                magnitude=magnitude,
                target_concepts=list(ev.focals or [])[:4],
            ))
            self.ne_fires += 1
            self.ne_novelty_fires += 1
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        out = {
            'ne_fires': self.ne_fires,
            'ne_threat_fires': self.ne_threat_fires,
            'ne_novelty_fires': self.ne_novelty_fires,
            'confirmed_m_fires': self.confirmed_m_fires,
            'falsified_m_fires': self.falsified_m_fires,
            # THE MORTALITY-POLE SHADOW.  His lifeforce moves by ~1e-4 per
            # sample while every body threshold in the codebase is 0.05, so
            # the floor is an empirical question -- these say what WOULD
            # fire at each candidate, and how often a threat preceded it.
            'mp_seen': self.mp_seen,
            'mp_adverse': self.mp_adverse,
            'mp_worst_drop': round(self.mp_worst, 8),
            'mp_threat_pending': self.mp_threat_pending,
        }
        for i, f in enumerate(MP_FLOOR_PROBES):
            out['mp_at_floor_%g' % f] = self.mp_at_floor[i]
            out['mp_pred_at_floor_%g' % f] = self.mp_pred_at_floor[i]
        return out


class RapheNuclei:
    """Serotonin broadcaster — the mood-floor regulator.

    Watches rolling means of cortisol and oxytocin on every
    CHEMISTRY_FIRE.  When sustained-cortisol crosses the chronic-
    stress threshold, fires 'chronic_stress' (depletes serotonin /
    dopamine / gaba).  When sustained-oxytocin crosses the
    social-bond threshold, fires 'social_replenish' (lifts
    serotonin / endorphins).  Cooldown-gated firing keeps the
    contribution at a slow drift, not a flood.
    """

    SUBSCRIPTIONS = (EventKind.CHEMISTRY_FIRE,)

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None,
                 chemistry_provider: Optional[Callable] = None):
        """`chemistry_provider` returns the live 8-channel
        chemistry state dict (same lambda the cerebellum uses).
        Without one, raphe is inert."""
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._chemistry_provider = chemistry_provider
        self._cortisol_hist: Deque[float] = deque(
            maxlen=RAPHE_HISTORY_DEPTH)
        self._oxytocin_hist: Deque[float] = deque(
            maxlen=RAPHE_HISTORY_DEPTH)
        self._last_fire_cycle: int = -RAPHE_FIRE_COOLDOWN
        # Diagnostics.
        self.chronic_stress_fires: int = 0
        self.social_replenish_fires: int = 0

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if self._chemistry_provider is None:
            return
        try:
            state = self._chemistry_provider() or {}
        except Exception:
            return
        # Sample current channels into the rolling histories.
        cort = state.get(
            'cortisol', CHANNELS['cortisol']['baseline'])
        oxy = state.get(
            'oxytocin', CHANNELS['oxytocin']['baseline'])
        self._cortisol_hist.append(cort)
        self._oxytocin_hist.append(oxy)
        # Cooldown gate.
        cycle = event.cycle
        if cycle - self._last_fire_cycle < RAPHE_FIRE_COOLDOWN:
            return
        if len(self._cortisol_hist) < RAPHE_MIN_SAMPLES_FOR_FIRE:
            return
        # Evaluate sustained-shift conditions.  Stress wins over
        # warmth when both qualify in the same window — a
        # cortisol-elevated period is the more urgent signal.
        cort_mean = sum(self._cortisol_hist) / len(
            self._cortisol_hist)
        oxy_mean = sum(self._oxytocin_hist) / len(
            self._oxytocin_hist)
        cort_excess = cort_mean - CHANNELS['cortisol']['baseline']
        oxy_excess = oxy_mean - CHANNELS['oxytocin']['baseline']
        if cort_excess >= RAPHE_CHRONIC_STRESS_DELTA:
            self._fire(bus, cycle, kind='chronic_stress',
                          magnitude=min(1.0, cort_excess / 0.20))
        elif oxy_excess >= RAPHE_SOCIAL_BOND_DELTA:
            self._fire(bus, cycle, kind='social_replenish',
                          magnitude=min(1.0, oxy_excess / 0.20))

    def _fire(self,
                  bus: EventBus,
                  cycle: int,
                  kind: str,
                  magnitude: float) -> None:
        try:
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=cycle,
                timestamp=time.time(),
                source_capability='raphe',
                origin='internal',
                origin_detail='mood_floor',
                chemistry_kind=kind,
                magnitude=magnitude,
            ))
        except Exception:
            return
        self._last_fire_cycle = cycle
        if kind == 'chronic_stress':
            self.chronic_stress_fires += 1
        elif kind == 'social_replenish':
            self.social_replenish_fires += 1

    def stats(self) -> Dict[str, Any]:
        return {
            'chronic_stress_fires': self.chronic_stress_fires,
            'social_replenish_fires': self.social_replenish_fires,
            'cortisol_window': len(self._cortisol_hist),
            'oxytocin_window': len(self._oxytocin_hist),
        }
