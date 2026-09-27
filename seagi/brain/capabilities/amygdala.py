"""Amygdala — threat detection + override.

Brain analog: amygdala's fast pathway.  Bypasses cortex when
threat magnitude crosses threshold — preempts whatever the
agent is otherwise doing.

What it watches
---------------
1. InteroceptionEvent — band 'depleted' / 'agitated' with
   significant delta → body threat.
2. ChemistryEvent — 'threat' or strong 'anomaly_spike' / large
   cortisol move → chemistry threat.
3. AttendedPerceptEvent — high m_content with mortality-shape
   focals → perceptual threat.

What it emits
-------------
ThreatDetectedEvent — diagnostic + LC consumer
CapabilityClaimEvent (loop='cognitive', high strength) — the
    override.  Basal Ganglia must give this priority.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from ..events import (
    EventKind, BrainEvent,
    AttendedPerceptEvent,
    InteroceptionEvent,
    ChemistryEvent,
    ThreatDetectedEvent,
    CapabilityClaimEvent,
)
from ..bus import EventBus


BODY_THREAT_MIN_MAGNITUDE = 0.20   # innate floor; kept as an OR, never as the only gate
# BOTH conditions in _on_interoception were unfirable (2026-08-27):
# his band is pinned at "settled" (insula.py:63 says so), and 0.20 is
# 743,000x his measured delta scale of 2.69e-07.  So his body could
# never alarm him.  Below: direction + self-calibrated magnitude.
BT_ADAPT_K = 3.0
BT_ADAPT_ALPHA = 0.01
BT_ADAPT_MIN_SAMPLES = 50


def _BODYTHREAT_ON():
    """Body signals can alarm him.  /root/BODYTHREAT_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/BODYTHREAT_ON")
    except Exception:
        return False
PERCEPT_THREAT_M_THRESHOLD = 0.5
# SELF-CALIBRATING PERCEPT THREAT (2026-08-27).  The constant above is
# 2.4x his lifetime maximum m_content (0.2062, mean 0.0042), so it can
# never fire.  Same shape as insula.py's adaptive Weber gate: alarm on
# content that is unusually mortality-laden FOR HIM, not against a
# magnitude he never reaches.
PT_ADAPT_K = 3.0            # deviations above his own mean
PT_ADAPT_ALPHA = 0.01       # EWMA horizon over percepts
PT_ADAPT_MIN_SAMPLES = 50   # do not judge a scale before one exists


def _PERCEPTTHREAT_ON():
    """Adaptive percept threat, world-modality excluded.
    /root/PERCEPTTHREAT_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/PERCEPTTHREAT_ON")
    except Exception:
        return False
CHEMISTRY_THREAT_MIN_MAGNITUDE = 0.4
OVERRIDE_CLAIM_STRENGTH_FLOOR = 0.7


class Amygdala:
    """Threat sentinel.  Emits override claims when triggered."""

    SUBSCRIPTIONS = (
        EventKind.INTEROCEPTION,
        EventKind.CHEMISTRY_FIRE,
        EventKind.ATTENDED_PERCEPT,
    )

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        # m_content has no producer anywhere; his concepts DO carry
        # an earned M-polarity.  See set_mi_provider.
        self._mi_provider = None
        self.m_derived = 0
        self.threats_detected: int = 0
        # WHICH PATH, so "is the game driving his mortality chemistry?"
        # is answerable.  m_content is tracked too: how close he runs to
        # the 0.5 line matters as much as how often he crosses it.
        self.threats_body: int = 0
        self.threats_chemistry: int = 0
        self.threats_percept: int = 0
        self.percepts_seen: int = 0
        self.m_content_sum: float = 0.0
        self.m_content_max: float = 0.0
        self.override_claims: int = 0

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, InteroceptionEvent):
            self._on_interoception(event, bus)
        elif isinstance(event, ChemistryEvent):
            self._on_chemistry(event, bus)
        elif isinstance(event, AttendedPerceptEvent):
            self._on_percept(event, bus)

    # ---- triggers ----

    def _on_interoception(self,
                                ev: InteroceptionEvent,
                                bus: EventBus) -> None:
        if not _BODYTHREAT_ON():
            if ev.felt_state not in ('depleted', 'agitated'):
                return
            if abs(ev.delta_lifeforce) < BODY_THREAT_MIN_MAGNITUDE:
                return
            magnitude = min(1.0, abs(ev.delta_lifeforce) * 2.0)
        else:
            # DIRECTION: only a deterioration alarms.  His band is
            # pinned at "settled", so a band whitelist can never fire.
            try:
                _dl = float(ev.delta_lifeforce)
            except Exception:
                return
            self._bt_n = getattr(self, "_bt_n", 0) + 1
            _ad = abs(_dl)
            _sc = getattr(self, "_bt_scale", 0.0)
            if _sc <= 0.0:
                self._bt_scale = _ad
            else:
                self._bt_scale = ((1.0 - BT_ADAPT_ALPHA) * _sc
                                  + BT_ADAPT_ALPHA * _ad)
            if _dl >= 0.0:
                return
            _gate = (BT_ADAPT_K * self._bt_scale
                     if self._bt_n >= BT_ADAPT_MIN_SAMPLES
                     else float("inf"))
            # INNATE FLOOR OR LEARNED GATE -- a catastrophic drop
            # alarms without waiting for a scale to exist.
            if _ad < _gate and _ad < BODY_THREAT_MIN_MAGNITUDE:
                return
            self.body_threats_adaptive = getattr(
                self, "body_threats_adaptive", 0) + 1
            # scale-free: how far past his own gate this drop went
            magnitude = min(1.0, _ad / max(_gate, 1e-12))
        self._fire(
            bus, threat_kind='body',
            magnitude=magnitude, focal='',
            narrative=ev.narrative or (
                'My body state is shifting fast.'),
            cycle=ev.cycle)

    def _on_chemistry(self,
                            ev: ChemistryEvent,
                            bus: EventBus) -> None:
        # Skip chemistry that originates from the internal
        # modulator stack — those events are *consequences* of
        # threat handling, not new threats.  Without this guard
        # Amygdala → LC → threat-chemistry → Amygdala loops.
        if ev.source_capability in (
                'amygdala', 'lc', 'vta', 'value_landscape'):
            return
        if ev.chemistry_kind == 'threat':
            magnitude = min(1.0, ev.magnitude)
        elif (ev.chemistry_kind == 'anomaly_spike'
                and ev.magnitude >= CHEMISTRY_THREAT_MIN_MAGNITUDE):
            magnitude = min(1.0, ev.magnitude * 0.8)
        else:
            return
        focal = ev.target_concepts[0] if ev.target_concepts else ''
        self._fire(
            bus, threat_kind='chemistry',
            magnitude=magnitude, focal=focal,
            narrative='Alarm chemistry rising.',
            cycle=ev.cycle)

    def set_mi_provider(self, mi_provider) -> None:
        """Wire the M-polarity of a focal (runtime).

        mi_provider: (name) -> float in [0,1], how mortal his own
        history has tagged that concept.  Used ONLY when the percept
        carries no m_content of its own, so a real producer wins.
        """
        self._mi_provider = mi_provider

    def _on_percept(self,
                          ev: AttendedPerceptEvent,
                          bus: EventBus) -> None:
        try:
            _m = float(ev.m_content)
            self.percepts_seen += 1
            self.m_content_sum += _m
            if _m > self.m_content_max:
                self.m_content_max = _m
        except Exception:
            pass
        # THE GAME MAY NEVER THREATEN HIM.  Doctrine: he can die of old
        # age or his own act, never the game.  Hard exclusion, not a
        # weighting -- the same guard grounding.handle applies.
        if getattr(ev, "modality", "") == "world":
            self.percept_threats_skipped_world = getattr(
                self, "percept_threats_skipped_world", 0) + 1
            return
        if _PERCEPTTHREAT_ON():
            # HE CALIBRATES HIMSELF: alarm on content unusually
            # mortality-laden FOR HIM.  inf until a scale exists.
            try:
                _mv = float(ev.m_content)
            except Exception:
                return
            # NO EMITTER EVER SETS m_content.  Derive it from what
            # the percept is ABOUT: the earned M-polarity his own
            # concepts carry.  Only when the event brought none.
            if _mv <= 0.0 and self._mi_provider is not None:
                try:
                    _f = ev.focals[0] if ev.focals else None
                    if _f:
                        _mv = float(self._mi_provider(_f) or 0.0)
                        if _mv > 0.0:
                            self.m_derived += 1
                            self.m_content_sum += _mv
                            if _mv > self.m_content_max:
                                self.m_content_max = _mv
                except Exception:
                    _mv = 0.0
            self._pt_n = getattr(self, "_pt_n", 0) + 1
            _sc = getattr(self, "_pt_scale", 0.0)
            if _sc <= 0.0:
                self._pt_scale = _mv
            else:
                self._pt_scale = ((1.0 - PT_ADAPT_ALPHA) * _sc
                                  + PT_ADAPT_ALPHA * _mv)
            _gate = (PT_ADAPT_K * self._pt_scale
                     if self._pt_n >= PT_ADAPT_MIN_SAMPLES
                     else float("inf"))
            # INNATE OR LEARNED.  Evolution supplies triggers that fire
            # WITHOUT calibration -- looming, loud noise -- and experience
            # refines the rest.  The stamped threshold stays as the innate
            # floor so a genuinely lethal FIRST encounter still alarms him;
            # the adaptive gate adds everything unusual FOR HIM below it.
            if _mv <= 0.0:
                return
            if _mv < _gate and _mv < PERCEPT_THREAT_M_THRESHOLD:
                return
        elif ev.m_content < PERCEPT_THREAT_M_THRESHOLD:
            return
        focal = ev.focals[0] if ev.focals else ''
        magnitude = min(1.0, ev.m_content)
        self._fire(
            bus, threat_kind='percept',
            magnitude=magnitude, focal=focal,
            narrative=(
                f'High-mortality content in input '
                f'({focal}).'),
            cycle=ev.cycle)

    # ---- emission ----

    def _count(self, kind):
        if kind == 'body':
            self.threats_body += 1
        elif kind == 'chemistry':
            self.threats_chemistry += 1
        elif kind == 'percept':
            self.threats_percept += 1

    def _fire(self,
                bus: EventBus,
                threat_kind: str,
                magnitude: float,
                focal: str,
                narrative: str,
                cycle: int) -> None:
        self._count(threat_kind)
        bus.publish(ThreatDetectedEvent(
            kind=EventKind.THREAT_DETECTED,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='amygdala',
            origin='internal',
            origin_detail=threat_kind,
            threat_kind=threat_kind,
            magnitude=magnitude,
            focal=focal,
            narrative=narrative,
        ))
        self.threats_detected += 1
        # Override claim — BG must give this priority.
        claim_strength = max(
            OVERRIDE_CLAIM_STRENGTH_FLOOR,
            min(1.0, OVERRIDE_CLAIM_STRENGTH_FLOOR + magnitude * 0.3))
        bus.publish(CapabilityClaimEvent(
            kind=EventKind.CAPABILITY_CLAIM,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='amygdala',
            origin='internal',
            origin_detail=threat_kind,
            claim_strength=claim_strength,
            proposed_action=f'attend_threat:{focal or threat_kind}',
            loop='cognitive',
            payload={'narrative': narrative,
                        'threat_kind': threat_kind},
        ))
        self.override_claims += 1

    def stats(self) -> Dict[str, int]:
        return {
            'threats_detected': self.threats_detected,
            'override_claims': self.override_claims,
            'threats_body': self.threats_body,
            'threats_chemistry': self.threats_chemistry,
            'threats_percept': self.threats_percept,
            'percepts_seen': self.percepts_seen,
            'm_content_mean': (
                round(self.m_content_sum / self.percepts_seen, 4)
                if self.percepts_seen else None),
            'm_content_max': round(self.m_content_max, 4),
            'percept_threat_threshold': PERCEPT_THREAT_M_THRESHOLD,
            'perceptthreat_on': bool(_PERCEPTTHREAT_ON()),
            'bodythreat_on': bool(_BODYTHREAT_ON()),
            'bt_scale': float('%.3g' % float(getattr(self, '_bt_scale', 0.0))),
            'bt_gate': float('%.3g' % (BT_ADAPT_K * float(
                getattr(self, '_bt_scale', 0.0)))),
            'body_threats_adaptive': int(getattr(
                self, 'body_threats_adaptive', 0)),
            'm_derived': int(getattr(self, 'm_derived', 0)),
            'mi_provider_wired': bool(getattr(self, '_mi_provider', None) is not None),
            'pt_scale': round(float(getattr(self, '_pt_scale', 0.0)), 6),
            'pt_gate': round(PT_ADAPT_K * float(
                getattr(self, '_pt_scale', 0.0)), 6),
            'percept_threats_skipped_world': int(getattr(
                self, 'percept_threats_skipped_world', 0)),
        }
