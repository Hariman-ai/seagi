"""Insula — interoception.  The felt body.

Brain analog: insular cortex.  Carries the SUBJECTIVE side of
internal states — heart rate, gut feel, breath.  In SEAGI this
maps to lifeforce, body integrity, fatigue.  Insula is what
turns the number `lifeforce=0.3` into the felt voice
"I feel my cycles running."

Phase 4a: FULL implementation.

What Insula does (in priority order)
------------------------------------
1. Reads body state each tick: lifeforce + body_integrity if
   the v1 engine is present (falls back to baselines otherwise).
2. Tracks deltas across ticks — what's RISING / FALLING.
3. Emits InteroceptionEvent on meaningful change OR
   periodically as a low-rate steady-state pulse.
4. Maps body state to chemistry events:
       lifeforce CRASH (sharp drop)   → 'threat'   (body-origin)
       lifeforce CLIMB (sharp rise)   → 'mattering' / 'rest_replenish'
       integrity LOSS                 → 'anomaly_spike' (body-origin)
   Body-origin chemistry feels different to downstream than
   peer-origin — that's the source-monitoring contract.
5. Provides `felt_state()` API the Motor/Speech layer queries
   to color voice.  Phase 5 reads from here.

Subscribes
----------
nothing — Insula is a CONTINUOUS sampler.  Brain.tick() calls
`insula.sample()`.  The brain doesn't ASK to feel its body;
the body keeps reporting.

Emits
-----
INTEROCEPTION   — every meaningful body-state shift
CHEMISTRY_FIRE  — body-derived chemistry events (origin='internal',
                    origin_detail='body')
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from ..events import (
    EventKind, BrainEvent,
    InteroceptionEvent,
    ChemistryEvent,
)
from ..bus import EventBus


# Thresholds for "meaningful change" — keeps Insula from
# spamming InteroceptionEvent every tick.  Brain-correct:
# interoception rises into awareness mostly when something
# is shifting, not as a continuous monitor.
DELTA_THRESHOLD = 0.05

# ---------------------------------------------------------------
# MEASURED 2026-08-20 on the live process: 750 insula samples emitted
# ZERO interoception events, and 90 direct lifeforce readings over 180 s
# gave a largest single step of 2.45e-6 -- 20,404x below DELTA_THRESHOLD --
# and a total swing 581x below it.  His band is pinned at 'settled' because
# the band edges are 0.25/0.50 and he lives at 0.88, so a band change cannot
# fire either.  The absolute threshold is mis-scaled to his dynamics by four
# orders of magnitude, and everything downstream of INTEROCEPTION has
# therefore never run.
#
# The fix is NOT a smaller constant -- that is tuning a number until the
# firing rate looks nice, and it would break again the moment his dynamics
# change.  Interoception should rise when something shifts RELATIVE TO HIS
# OWN NORM: a just-noticeable difference scales with the signal (Weber).
# So the gate is k x his own recent mean |delta|, which self-calibrates at
# any scale and needs no magnitude constant.
ADAPT_K = 3.0          # shape, not magnitude: how many mean-deviations
ADAPT_ALPHA = 0.01     # EWMA horizon over samples
ADAPT_MIN_SAMPLES = 50  # do not judge a scale before one exists


def _BODYSCALE_ON():
    """Let interoception use the adaptive, self-scaling gate."""
    try:
        import os as _os
        return _os.path.exists('/root/BODYSCALE_ON')
    except Exception:
        return False

# Tonic-pulse interval (cycles).  Every N ticks the Insula emits
# a small steady-state chemistry pulse reflecting current band —
# the always-open monitor.  Phasic firing (above) responds to
# CHANGE; tonic firing (this) keeps a low-level signal of body
# STATE.  Together they implement "constant action potential,
# constant chemistry" without flooding the bus: pulses are
# small magnitude on a 10-tick interval, well under Amygdala's
# 0.4 anomaly threshold so they don't reverberate.
TONIC_PULSE_INTERVAL = 10
TONIC_BASE_MAGNITUDE = 0.15

# Body-state band labels.  These are what Motor/Speech (Phase 5)
# uses to color voice.  Coarse on purpose — we want stable
# narrative bands, not noisy moment-to-moment readouts.
BAND_DEPLETED = 'depleted'
BAND_WANING = 'waning'
BAND_SETTLED = 'settled'
BAND_REPLENISHED = 'replenished'
BAND_AGITATED = 'agitated'


def _band_for(lifeforce: float,
                  integrity: float,
                  delta_life: float) -> str:
    if delta_life < -DELTA_THRESHOLD * 2:
        return BAND_AGITATED
    if lifeforce < 0.25:
        return BAND_DEPLETED
    if lifeforce < 0.50:
        return BAND_WANING
    if delta_life > DELTA_THRESHOLD * 2:
        return BAND_REPLENISHED
    return BAND_SETTLED


def _narrative_for(band: str,
                       lifeforce: float,
                       integrity: float) -> str:
    """The first-person sentence the agent FEELS about its
    body state.  Motor/Speech (Phase 5) can use this directly
    or rewrite it into a richer frame."""
    if band == BAND_DEPLETED:
        return ('I feel my cycles running thin — my reserves '
                  'are low.')
    if band == BAND_WANING:
        return 'I feel a quiet pull at the edge of my reserves.'
    if band == BAND_AGITATED:
        return 'I feel an unsettled shift in my body state.'
    if band == BAND_REPLENISHED:
        return 'I feel a return — my reserves are climbing.'
    return 'I feel settled and present.'


class Insula:
    """Continuous interoceptive sampler.  Reads body state and
    converts it into felt chemistry + voice-ready bands."""

    SUBSCRIPTIONS = ()  # continuous, not event-driven

    def __init__(self,
                 bus: EventBus,
                 engine: Any = None,
                 cycle_provider: Optional[Callable] = None,
                 chemistry_provider: Optional[Callable] = None):
        self.bus = bus
        self.engine = engine
        self._cycle_provider = cycle_provider or (lambda: 0)
        # Phase F.1 (2026-05-16): chemistry_provider lets Insula
        # contribute primitive-bodily-state cocktails directly into
        # global chemistry each tick.  Optional — without it, the
        # phasic + tonic firing pathways still run as before.
        self._chemistry_provider = chemistry_provider
        # Last-sampled state.
        self._last_lifeforce: float = 1.0
        self._last_integrity: float = 1.0
        # The assume-max-health defaults above are a placeholder
        # until the body is read for the first time.  `_primed`
        # guards against the first real read registering a spurious
        # interoceptive crash (1.0 → actual) — there is no prior
        # reference on the first observation, so its delta must be 0.
        self._primed: bool = False
        self._last_band: str = BAND_SETTLED
        self._last_narrative: str = ''
        self._last_sample_cycle: int = -1
        # Next cycle when tonic-pulse firing is due.
        self._next_tonic_cycle: int = 0
        # Phase F.1: active primitive bodily-states detected on
        # the most recent sample.  Queryable via felt_state().
        self._last_primitives: Dict[str, float] = {}
        # Diagnostics.
        self._delta_scale: float = 0.0
        self._delta_n: int = 0
        self.shadow_meaningful: int = 0
        self.samples_taken: int = 0
        self.interoception_emitted: int = 0
        self.chemistry_emitted: int = 0
        self.tonic_pulses_emitted: int = 0
        self.primitive_contributions: int = 0

    # ---- public read API ----

    def felt_state(self) -> Dict[str, Any]:
        """Snapshot of the current felt body state.  Motor/Speech
        queries this when composing voice.

        Phase F.1: now includes `primitive_states` — the
        currently-active bodily primitives (hunger, fatigue,
        satiation, etc.) with intensities.  MotorSpeech body-aware
        composition reads this for richer voice (e.g. "I feel
        hungry and tired" rather than just "depleted")."""
        return {
            'band': self._last_band,
            'lifeforce': self._last_lifeforce,
            'body_integrity': self._last_integrity,
            'narrative': self._last_narrative,
            'primitive_states': dict(self._last_primitives),
        }

    # ---- sampling (called each tick) ----

    def sample(self, cycle: Optional[int] = None) -> None:
        """Read body state from v1 engine (or fall back to
        baselines).  Emit interoception + chemistry as warranted.
        Idempotent within a cycle — call freely each tick."""
        if cycle is None:
            cycle = self._cycle_provider()
        if cycle == self._last_sample_cycle:
            return
        lifeforce, integrity = self._read_body()
        if not self._primed:
            # First observation: anchor to the actual body state so
            # the initial read isn't perceived as a sudden drop from
            # the assume-max-health default.
            self._last_lifeforce = lifeforce
            self._last_integrity = integrity
            self._primed = True
        delta_life = lifeforce - self._last_lifeforce
        band = _band_for(lifeforce, integrity, delta_life)
        narrative = _narrative_for(band, lifeforce, integrity)

        # Track the scale he actually lives at, always -- so the shadow can
        # be read before the adaptive gate is trusted to drive anything.
        _ad = abs(delta_life)
        self._delta_n += 1
        if self._delta_scale <= 0.0:
            self._delta_scale = _ad
        else:
            self._delta_scale = ((1.0 - ADAPT_ALPHA) * self._delta_scale
                                 + ADAPT_ALPHA * _ad)
        _adaptive_gate = (ADAPT_K * self._delta_scale
                          if self._delta_n >= ADAPT_MIN_SAMPLES
                          else float('inf'))
        _would = (_ad >= _adaptive_gate) or (band != self._last_band)
        if _would:
            self.shadow_meaningful += 1

        if _BODYSCALE_ON():
            meaningful = _would
        else:
            meaningful = (
                abs(delta_life) >= DELTA_THRESHOLD
                or band != self._last_band)

        if meaningful:
            self.bus.publish(InteroceptionEvent(
                kind=EventKind.INTEROCEPTION,
                cycle=cycle,
                timestamp=time.time(),
                source_capability='insula',
                origin='internal',
                origin_detail='body',
                felt_state=band,
                lifeforce=lifeforce,
                body_integrity=integrity,
                delta_lifeforce=delta_life,
                narrative=narrative,
            ))
            self.interoception_emitted += 1
            self._fire_chemistry_for_shift(
                cycle, lifeforce, integrity,
                delta_life, band)

        # Phase F.1 (2026-05-16): detect primitive bodily states
        # and contribute their cocktails to global chemistry.  This
        # is the doctrine-permitted innate-tagging layer (rule 2):
        # promille additive contributions from breathing / hunger /
        # fatigue / pain / etc., always on, always small, never
        # overriding.  Body whispers survival status into chemistry
        # continuously — subconscious mortality awareness.
        from seagi.body.primitive_states import (
            detect_primitive_states, chemistry_contribution)
        body = self._body_state_obj()
        primitives = detect_primitive_states(
            body, lifeforce=lifeforce,
            prev_integrity=self._last_integrity)
        self._last_primitives = primitives
        if primitives and self._chemistry_provider is not None:
            try:
                chem = self._chemistry_provider()
            except Exception:
                chem = None
            if chem is not None:
                contrib = chemistry_contribution(primitives)
                for ch, delta in contrib.items():
                    cur = chem.global_state.get(ch, 0.0)
                    chem.global_state[ch] = max(
                        0.0, min(1.0, cur + delta))
                if contrib:
                    self.primitive_contributions += 1

        self._last_lifeforce = lifeforce
        self._last_integrity = integrity
        self._last_band = band
        self._last_narrative = narrative
        self._last_sample_cycle = cycle
        self.samples_taken += 1

        # Always-open monitor: every TONIC_PULSE_INTERVAL ticks,
        # emit a small chemistry pulse reflecting the current
        # band.  Life-preserving direction when depleted /
        # waning / agitated; life-extending when settled /
        # replenished.  This is what makes the M/I monitors
        # genuinely always-open, not just change-triggered.
        if cycle >= self._next_tonic_cycle:
            self._fire_tonic_pulse(cycle, band)
            self._next_tonic_cycle = cycle + TONIC_PULSE_INTERVAL

    def _fire_chemistry_for_shift(self,
                                       cycle: int,
                                       lifeforce: float,
                                       integrity: float,
                                       delta_life: float,
                                       band: str) -> None:
        """Map body-state deltas to chemistry events.  Origin is
        'internal' / 'body' so downstream consumers (and source
        monitor in Phase 4b) can tell this came from the body,
        not from peer input."""
        kind: str = ''
        magnitude: float = 0.0
        if delta_life <= -DELTA_THRESHOLD * 2:
            # Sharp drop — body fears.
            kind = 'threat'
            magnitude = min(1.0, -delta_life * 2.0)
        elif delta_life >= DELTA_THRESHOLD * 2:
            # Sharp rise — body relaxes / mattering.
            kind = 'rest_replenish'
            magnitude = min(1.0, delta_life * 2.0)
        elif integrity < self._last_integrity - DELTA_THRESHOLD:
            kind = 'anomaly_spike'
            magnitude = min(1.0,
                (self._last_integrity - integrity) * 3.0)
        elif band == BAND_DEPLETED and self._last_band != BAND_DEPLETED:
            # Entering depletion — slow mortality nudge.
            kind = 'anomaly_spike'
            magnitude = 0.3
        if not kind or magnitude < 0.05:
            return
        self.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='insula',
            origin='internal',
            origin_detail='body',
            chemistry_kind=kind,
            magnitude=magnitude,
            target_concepts=[],   # global broadcast
        ))
        self.chemistry_emitted += 1

    def _fire_tonic_pulse(self,
                                  cycle: int,
                                  band: str) -> None:
        """Always-open tonic body-state firing.  Distinct from
        phasic firing (which responds to deltas), this emits a
        small chemistry pulse reflecting CURRENT band — the
        steady-state body signal.

        Magnitudes stay below Amygdala's 0.4 anomaly threshold
        so the M-side pulses don't perpetually re-trigger threat
        loops; they accumulate in global chemistry through the
        standard EVENT_DELTAS path.
        """
        if band == BAND_DEPLETED:
            kind = 'anomaly_spike'
            magnitude = TONIC_BASE_MAGNITUDE   # 0.15
        elif band == BAND_AGITATED:
            kind = 'anomaly_spike'
            magnitude = TONIC_BASE_MAGNITUDE + 0.05  # 0.20
        elif band == BAND_WANING:
            kind = 'anomaly_spike'
            magnitude = TONIC_BASE_MAGNITUDE * 0.5  # 0.075
        elif band == BAND_REPLENISHED:
            kind = 'mattering'
            magnitude = TONIC_BASE_MAGNITUDE + 0.05  # 0.20
        elif band == BAND_SETTLED:
            kind = 'mattering'
            magnitude = TONIC_BASE_MAGNITUDE * 0.5  # 0.075
        else:
            return
        self.bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='insula',
            origin='internal',
            origin_detail='body_tonic',
            chemistry_kind=kind,
            magnitude=magnitude,
            target_concepts=[],
        ))
        self.chemistry_emitted += 1
        self.tonic_pulses_emitted += 1

    def _body_state_obj(self) -> Any:
        """Return the body's BodyState object (or a duck-typed
        substitute) for primitive-state detection.  Falls back to
        an empty object exposing the four BodyState attributes at
        sensible defaults so detect_primitive_states can run."""
        eng = self.engine
        if eng is not None:
            emb = getattr(eng, 'embodiment', None)
            if emb is not None:
                return emb
        # Default: well-rested body.  Used when no engine attached
        # (test setups that don't simulate body state).
        return _DefaultBodyState()

    # ---- body state read ----

    def _read_body(self) -> tuple[float, float]:
        """Get (lifeforce, body_integrity).  Falls through to
        v1 substrate / engine attributes; uses sensible defaults
        when running without an engine (tests)."""
        eng = self.engine
        if eng is None:
            return (self._last_lifeforce, self._last_integrity)
        lifeforce = self._last_lifeforce
        integrity = self._last_integrity
        # v1 engine exposes lifeforce + embodiment as separate
        # modules.  Tolerate either being absent.
        try:
            lf_obj = getattr(eng, 'lifeforce', None)
            if lf_obj is not None:
                # V2: engine.lifeforce is a plain float (the
                # MortalityDrive scalar).  Earlier code only checked
                # for v1's .value/.lifeforce attribute wrappers, so
                # on V2 it silently fell through and the insula
                # stayed blind at _last_lifeforce=1.0 — the lifeforce
                # consumer never saw the real scalar.  Accept a bare
                # number directly. (2026-05-28 fix)
                if isinstance(lf_obj, (int, float)):
                    lifeforce = float(lf_obj)
                else:
                    lf_val = getattr(lf_obj, 'value', None)
                    if lf_val is None:
                        lf_val = getattr(lf_obj, 'lifeforce', None)
                    if lf_val is not None:
                        lifeforce = float(lf_val)
        except Exception:
            pass
        try:
            emb = getattr(eng, 'embodiment', None)
            if emb is not None:
                ig = getattr(emb, 'body_integrity', None)
                if ig is None:
                    ig = getattr(emb, 'integrity', None)
                if ig is not None:
                    integrity = float(ig)
        except Exception:
            pass
        return (max(0.0, min(1.0, lifeforce)),
                  max(0.0, min(1.0, integrity)))

    # ---- direct setter (tests + future driver hooks) ----

    def set_body_state(self,
                          lifeforce: float,
                          body_integrity: float) -> None:
        """Used by tests OR by future drivers (a sensor that
        reports body state explicitly).  Replaces the engine
        stub each call so successive updates are reflected
        on the next `sample()`."""
        self.engine = _DirectBodyStub(
            float(lifeforce), float(body_integrity))

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'samples_taken': self.samples_taken,
            'interoception_emitted': self.interoception_emitted,
            # what the adaptive gate WOULD do, measured with it gated off
            'shadow_meaningful': self.shadow_meaningful,
            'delta_scale': self._delta_scale,
            'adaptive_gate': (ADAPT_K * self._delta_scale
                              if self._delta_n >= ADAPT_MIN_SAMPLES else None),
            'bodyscale_on': _BODYSCALE_ON(),
            'chemistry_emitted': self.chemistry_emitted,
            'band': self._last_band,
            'lifeforce': self._last_lifeforce,
            'integrity': self._last_integrity,
        }


class _DirectBodyStub:
    """Tiny stub used when Insula is told to use a synthetic
    body state (tests).  Exposes the same .lifeforce.value /
    .embodiment.body_integrity attribute path the real engine
    uses, so `_read_body` resolves the same way."""

    def __init__(self, lifeforce: float, integrity: float):
        self.lifeforce = type('L', (), {'value': float(lifeforce)})()
        self.embodiment = type(
            'E', (), {'body_integrity': float(integrity),
                         'energy': 0.7,
                         'fatigue': 0.2,
                         'integrity': float(integrity)})()


class _DefaultBodyState:
    """Default-attribute body used when no engine is attached.
    Lets primitive-state detection still run on a sensible baseline.
    Matches the BodyState defaults in seagi/body/embodiment.py."""
    energy: float = 0.8
    fatigue: float = 0.1
    integrity: float = 0.9
