"""Thalamic Gate — salience filter, the brain's noise filter.

Brain analog: thalamus (sensory relay nuclei) + reticular
activating system (arousal gating).  In real brains, the
thalamus relays only a small fraction of raw sensory data to
cortex.  What gets through is determined by arousal state
(reticular formation) and cortical feedback (predicted vs.
unpredicted).

In v1, every RSS token became a substrate concept.  The
forager-junk problem.  This capability is the fix at the
SOURCE: most input never reaches awareness in the first place.

Phase 1: FULL implementation.  This is the headline deliverable
of Phase 1 — gets junk OUT of the architecture before the rest
of v2 is built on top of it.

Reads        : RawPerceptEvent
Writes       : nothing directly
Emits        : AttendedPerceptEvent (when salience clears bar)
                OR PerceptDiscardedEvent (when below)
Modulated by : chemistry state (Phase 2+ — currently stub)
Brain analog : thalamus + reticular activating system

Salience computation
--------------------
salience = max(
    W_NOVELTY * novelty,
    W_M_CONTENT * m_content,
    W_I_CONTENT * i_content,
    W_PEER * peer_proximity,
)

We use MAX (not sum) so a strongly-novel-but-emotionally-neutral
input can still pass, and so can an emotionally-strong-but-known
input.  Brain analog: any single high-salience channel is enough
to grab attention.

Threshold modulation
--------------------
Base threshold = SALIENCE_THRESHOLD.
Arousal modulator (Phase 2): when overall NE/cortisol is high,
threshold drops (alert state, more passes through).  When
arousal is low, threshold rises (more filtering).

Phase 1: threshold is fixed.  Phase 2 adds chemistry modulation.

Peer proximity
--------------
User dialog input has peer_proximity = 1.0 (always passes —
peer is co-present, must attend).  Forager input has lower
peer_proximity (RSS articles aren't a peer in conversation).
This is the brain's "social attention" reflex translated to the
agent's modality mix.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..events import (
    EventKind, BrainEvent,
    RawPerceptEvent, AttendedPerceptEvent, PerceptDiscardedEvent,
)
from ..bus import EventBus


# Salience component weights.  MAX-of-weighted, not sum.
W_NOVELTY = 0.5
W_M_CONTENT = 1.0
W_I_CONTENT = 0.7
W_PEER = 1.0      # peer-proximity dominant when present

# Default threshold for "attended."  Phase 2 makes this
# chemistry-modulated.
SALIENCE_THRESHOLD_BASE = 0.30

# Step 0 organ 4b (2026-05-27): structural safety ceiling on the
# effective threshold.  Even under high chemistry modulation, the
# threshold cannot rise above the weakest survival-class salience
# weight × 1.0.  This guarantees a high-magnitude peer/M/I/novelty
# percept can always cross the bar — survival channels stay
# reachable regardless of deprivation state.
#
# Derived from existing weights: min(W_PEER=1.0, W_M_CONTENT=1.0,
# W_I_CONTENT=0.7, W_NOVELTY=0.5) = 0.5.  No free parameter.
THRESHOLD_SAFETY_CEILING = min(
    W_PEER, W_M_CONTENT, W_I_CONTENT, W_NOVELTY) * 1.0

# Peer proximity by origin.  Higher = stronger pull on attention.
PEER_PROXIMITY = {
    'peer': 1.0,        # direct chat input — always passes
    'sensor': 0.6,      # future direct sensory input
    'ingestion': 0.5,   # deliberate reading — Layer 3 corpus
                          # ingestion.  Moderate proximity: a
                          # text Seagi is reading is closer than
                          # background scraping (0.2) but less
                          # immediate than a peer in conversation
                          # (1.0).  The pace of ingestion is slow
                          # enough (sentence-by-sentence with full
                          # chemistry tick between) that
                          # individual concepts still imprint
                          # richly even at moderate proximity.
    'internal': 0.5,    # internally-generated thought re-perception
    'forager': 0.2,     # background RSS / file ingestion
    '': 0.3,            # unspecified / default
}


def _novelty_score(payload: Dict[str, Any],
                      substrate: Any) -> float:
    """Fraction of perceived focals that are NEW or barely
    encountered in substrate.  Same shape as v1's
    sensory_input.novelty_score for continuity."""
    if not payload:
        return 0.0
    concepts = (getattr(substrate, 'concepts', None) or {}
                  if substrate is not None else {})
    new_count = 0
    total = 0
    for name in payload:
        s = str(name)
        # Defer to the central output-appropriateness filter so
        # stopwords don't even count as candidates.
        try:
            from seagi.core.output_filter import (
                _is_output_appropriate)
            if not _is_output_appropriate(s):
                continue
        except Exception:
            pass
        total += 1
        c = concepts.get(s)
        if c is None:
            new_count += 1
            continue
        # Barely-rooted → still novel-ish.
        encounter_total = 0
        bubbles = getattr(c, 'bubbles', None) or []
        for b in bubbles:
            encounter_total += int(
                getattr(b, 'encounter_count', 0) or 0)
        if encounter_total < 3:
            new_count += 1
    if total == 0:
        return 0.0
    return new_count / total


def _mi_content(payload: Dict[str, Any],
                  substrate: Any) -> tuple:
    """Aggregate (M, I) across focals' best bubbles in substrate.
    Same shape as v1's sensory_input.mi_content."""
    if not payload:
        return (0.0, 0.0)
    concepts = (getattr(substrate, 'concepts', None) or {}
                  if substrate is not None else {})
    if not concepts:
        return (0.0, 0.0)
    m_acc = 0.0
    i_acc = 0.0
    weight = 0.0
    try:
        from seagi.core.output_filter import (
            _is_output_appropriate)
    except Exception:
        _is_output_appropriate = None
    for name in payload:
        s = str(name)
        if (_is_output_appropriate is not None
                and not _is_output_appropriate(s)):
            continue
        c = concepts.get(s)
        if c is None:
            continue
        bubbles = getattr(c, 'bubbles', None) or []
        if not bubbles:
            continue
        best_b = max(bubbles, key=lambda b: int(
            getattr(b, 'encounter_count', 0) or 0))
        tx = getattr(best_b, 'transmitter_trace', None)
        if tx is None:
            continue
        m = (float(getattr(tx, 'cortisol', 0.0))
              + float(getattr(tx, 'norepinephrine', 0.0))) / 2.0
        i = (float(getattr(tx, 'dopamine', 0.0))
              + float(getattr(tx, 'oxytocin', 0.0))
              + float(getattr(tx, 'endorphins', 0.0))) / 3.0
        m_acc += m
        i_acc += i
        weight += 1.0
    if weight == 0.0:
        return (0.0, 0.0)
    return (m_acc / weight, i_acc / weight)


class ThalamicGate:
    """The salience filter.  Subscribes to RawPerceptEvent;
    emits AttendedPerceptEvent for passes and
    PerceptDiscardedEvent for drops.

    Stats
    -----
    Tracks pass-count, discard-count, and a sample of recent
    discards so we can audit what the gate is suppressing.
    """

    SUBSCRIPTIONS = (EventKind.RAW_PERCEPT,)

    def __init__(self,
                 engine: Any = None,
                 threshold: float = SALIENCE_THRESHOLD_BASE,
                 chemistry_provider: Optional[Any] = None,
                 debt_provider: Optional[Any] = None,
                 baseline_provider: Optional[Any] = None,
                 debt_full_scale_provider: Optional[Any] = None):
        """
        engine: optional v1 engine for substrate access (until LTS
                capability comes online in Phase 2).
        threshold: base threshold — used ONLY for the
                attended/attenuated distinction in diagnostics
                AND for some downstream consumers that want a
                "conscious-tier" signal.  Below-threshold
                percepts are NO LONGER DROPPED — they pass
                through attenuated.  The brain doesn't erase
                subliminal input; it processes everything at
                magnitude-faithful response.
        chemistry_provider: callable returning current arousal
                modulator in [0, 1] for threshold adjustment.
        debt_provider, baseline_provider, debt_full_scale_provider:
                Step 0 organ 4b (2026-05-27).  MetabolicDebt
                wires that drive below-threshold attenuation:
                  attenuation_strength = clip(
                      (debt - baseline)/full_scale, 0, 1).
                Below-threshold percepts have their salience
                multiplied by
                  (1 − attenuation_strength × (1 − s/threshold)).
                Above-threshold percepts pass at full magnitude:
                survival bypass.  Absent providers → no debt
                attenuation (no Step 0 behavior).
        """
        self.engine = engine
        self.threshold_base = float(threshold)
        self._chemistry_provider = chemistry_provider
        self._debt_provider = debt_provider
        self._baseline_provider = baseline_provider
        self._debt_full_scale_provider = debt_full_scale_provider
        # Diagnostics
        self.attended_count: int = 0
        self.attenuated_count: int = 0
        self.discarded_count: int = 0   # structurally-empty only
        self.recent_discards: list = []  # bounded ring (cap 20)
        # Step 0 organ 4b diagnostics.
        self.debt_attenuated_count: int = 0
        self.last_attenuation_strength: float = 0.0

    def _arousal_modulator(self) -> float:
        """Returns a 0..1 modulator.  Lower → threshold drops
        (alert state, more passes).  Higher → threshold rises
        (filtering harder).  Phase 1 stub: returns 0.5 (neutral
        modulation).  Phase 2 wires actual chemistry state."""
        if self._chemistry_provider is not None:
            try:
                v = float(self._chemistry_provider())
                return max(0.0, min(1.0, v))
            except Exception:
                pass
        return 0.5

    def _threshold_now(self) -> float:
        """Chemistry-modulated threshold, capped from above by
        THRESHOLD_SAFETY_CEILING (Step 0 organ 4b structural lock).

        mod=0 → threshold × 0.6 (alert, more passes)
        mod=1 → threshold × 1.4 (calm, filtering harder)

        Even with extreme chemistry pushing the threshold high,
        the ceiling guarantees survival-class percepts at full
        magnitude (peer=1.0, M=1.0, I=0.7, novelty=0.5) can
        still cross.
        """
        mod = self._arousal_modulator()
        chemistry_threshold = self.threshold_base * (0.6 + 0.8 * mod)
        return min(chemistry_threshold, THRESHOLD_SAFETY_CEILING)

    def _attenuation_strength(self) -> float:
        """Step 0 organ 4b — clip((debt − baseline)/full_scale, 0, 1).
        Absent providers → 0 (no debt attenuation)."""
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

    def _apply_debt_attenuation(self,
                                       salience: float,
                                       threshold: float) -> float:
        """Step 0 organ 4b — steepen the attenuation curve below
        threshold.  Above-threshold percepts pass at full magnitude
        (survival bypass).  Below-threshold percepts multiplied by
            (1 − attenuation_strength × (1 − s/threshold)).
        At s=threshold: factor = 1 (no attenuation).
        At s≈0: factor = (1 − attenuation_strength) (heavy at max
                debt; identity at no debt).
        """
        if salience >= threshold:
            return salience
        strength = self._attenuation_strength()
        self.last_attenuation_strength = float(strength)
        if strength <= 0.0:
            return salience
        if threshold <= 0.0:
            return salience
        depth = 1.0 - (salience / threshold)
        if depth < 0.0:
            depth = 0.0
        elif depth > 1.0:
            depth = 1.0
        factor = 1.0 - strength * depth
        if factor < 0.0:
            factor = 0.0
        self.debt_attenuated_count += 1
        return salience * factor

    def handle(self,
                 event: BrainEvent,
                 bus: EventBus) -> None:
        """Main entry — invoked by the bus for every
        RawPerceptEvent."""
        if not isinstance(event, RawPerceptEvent):
            return
        self._gate(event, bus)

    def _gate(self,
                event: RawPerceptEvent,
                bus: EventBus) -> None:
        substrate = (getattr(self.engine, 'substrate', None)
                       if self.engine is not None else None)

        # Compute salience components.
        novelty = _novelty_score(event.payload, substrate)
        m_content, i_content = _mi_content(event.payload, substrate)
        peer_prox = PEER_PROXIMITY.get(event.origin,
                                          PEER_PROXIMITY[''])

        # Filter focal list to output-appropriate names (so
        # downstream consumers don't get stopwords).
        focals = []
        try:
            from seagi.core.output_filter import (
                _is_output_appropriate)
            for name in (event.payload or {}):
                s = str(name)
                if _is_output_appropriate(s):
                    focals.append(s)
        except Exception:
            focals = [str(n) for n in (event.payload or {})]

        # MAX-of-weighted-components.
        salience = max(
            W_NOVELTY * novelty,
            W_M_CONTENT * m_content,
            W_I_CONTENT * i_content,
            W_PEER * peer_prox,
        )

        threshold = self._threshold_now()

        if not focals:
            # Structurally empty input — no concepts to attend
            # to at all.  This is the genuine drop case (not a
            # threshold-based one): nothing to perceive.
            self._emit_discarded(
                event, bus, salience,
                novelty, m_content, i_content,
                threshold, 0)
            return

        # Step 0 organ 4b (2026-05-27): below-threshold percepts
        # take an additional debt-driven attenuation step.
        # Above-threshold survival bypass: pass at full magnitude
        # regardless of debt.  Below-threshold: factor
        #   (1 − attenuation_strength × (1 − s/threshold))
        # steepens noise rejection as debt rises.
        effective_salience = self._apply_debt_attenuation(
            salience, threshold)
        # ALWAYS emit AttendedPerceptEvent — doctrine: nothing
        # is fully discarded.  Below-threshold percepts pass
        # through ATTENUATED, with magnitude-faithful salience.
        # Downstream consumers naturally process based on
        # magnitude (chemistry fires faint, AWM promotes weakly,
        # cortical may skip expensive reasoning when salience
        # is low).  The threshold becomes informational, not a
        # kill switch.
        self._emit_attended(
            event, bus, focals, effective_salience,
            novelty, m_content, i_content, threshold)
        if effective_salience < threshold:
            self.attenuated_count += 1

    def _emit_attended(self,
                          ev: RawPerceptEvent,
                          bus: EventBus,
                          focals: list,
                          salience: float,
                          novelty: float,
                          m_content: float,
                          i_content: float,
                          threshold: float) -> None:
        self.attended_count += 1
        out = AttendedPerceptEvent(
            kind=EventKind.ATTENDED_PERCEPT,
            cycle=ev.cycle,
            timestamp=time.time(),
            source_capability='thalamic_gate',
            origin=ev.origin,
            origin_detail=ev.origin_detail,
            focals=focals[:16],
            payload=dict(ev.payload),
            raw_text=ev.raw_text,
            modality=ev.modality,
            salience=salience,
            novelty=novelty,
            m_content=m_content,
            i_content=i_content,
            threshold_used=threshold,
        )
        bus.publish(out)

    def _emit_discarded(self,
                            ev: RawPerceptEvent,
                            bus: EventBus,
                            salience: float,
                            novelty: float,
                            m_content: float,
                            i_content: float,
                            threshold: float,
                            n_focals: int) -> None:
        self.discarded_count += 1
        # Keep small audit trail of recent discards (capped).
        if len(self.recent_discards) >= 20:
            self.recent_discards.pop(0)
        self.recent_discards.append({
            'cycle': ev.cycle,
            'origin': ev.origin,
            'origin_detail': ev.origin_detail,
            'salience': salience,
            'novelty': novelty,
            'm_content': m_content,
            'i_content': i_content,
            'threshold_used': threshold,
            'excerpt': ev.raw_text[:120],
        })
        out = PerceptDiscardedEvent(
            kind=EventKind.PERCEPT_DISCARDED,
            cycle=ev.cycle,
            timestamp=time.time(),
            source_capability='thalamic_gate',
            origin=ev.origin,
            origin_detail=ev.origin_detail,
            modality=ev.modality,
            salience=salience,
            novelty=novelty,
            m_content=m_content,
            i_content=i_content,
            threshold_used=threshold,
            n_focals=n_focals,
            raw_text_excerpt=ev.raw_text[:120],
        )
        bus.publish(out)

    def stats(self) -> Dict[str, Any]:
        total = self.attended_count + self.discarded_count
        pass_rate = (self.attended_count / total
                       if total else 0.0)
        return {
            'attended_count': self.attended_count,
            'attenuated_count': self.attenuated_count,
            'debt_attenuated_count': self.debt_attenuated_count,
            'discarded_count': self.discarded_count,
            'pass_rate': pass_rate,
            'threshold_now': self._threshold_now(),
            'last_attenuation_strength':
                float(self.last_attenuation_strength),
        }
