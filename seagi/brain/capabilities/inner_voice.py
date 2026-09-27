"""InnerVoice — continuous internal articulation during reverie.

Step 5.1 (architect + auditor APPROVED 2026-05-26, smallest version).

Closes the language→cognition loop: the agent thinks IN language
during reverie, that articulation is re-perceived as a self-percept,
and the percept feeds cortical inference + chemistry imprint.  When
cortical derives a new claim from a self-perceived utterance,
ReasoningConsolidator writes it back as a provisional edge — which
Phase S then earn-or-dissolves.  So articulation IS substrate-
building, with the substrate's own coherence gate filtering junk
("fear enables child" never survives).

The five wires (from the locked design):
1. Continuous in reverie — tick during idle/wake, throttled.
2. Self-perception via REUSED AttendedPerceptEvent — no new event
   type; the standard percept handlers (cortical, chemistry,
   hippocampus) pick it up.  `origin='self'`, `source_capability=
   'inner_voice'` distinguish it from peer/external percepts so
   downstream consumers can tell where the percept came from.
3. MotorSpeech remains the outward gate — outward speech only on
   SPEECH_REQUEST.  Inner voice is private.
4. vmDMN narrative feeds composition — recent self-reflection
   colors what gets articulated.
5. Coherence check — reject obviously-incoherent utterances before
   self-perceiving them; we don't want to feed reverie its own junk.

Substrate-write properties (doctrine + design):
- Self-utterances are CHEMISTRY-IMPRINT ONLY at the percept level
  (the existing AttendedPerceptEvent → bubble-imprint path handles
  this; no direct substrate write).
- Real substrate growth from inner voice arrives via cortical
  inference triggered by the self-percept → ReasoningConsolidator
  writes a provisional edge → Phase S coherence-tests it (the
  earn-gate).  Junk dies; corroborated claims survive.  This is
  the architectural anti-gaming guarantee.

Cortical re-entry is INHERITED-BLOCKED: cortical only reasons on
its own gating conditions (AWM focal, confidence, etc.).  Inner
voice doesn't directly re-call cortical; the loop is substrate-
mediated (utterance → percept → AWM update → next reverie tick
maybe articulates again, gated by all the standard mechanisms).

Known reservation (from the auditor pass, R8): runtime's
`_substrate_top_focals_cache` (runtime.py) is permanent —
inner voice may iterate on the same focals if reverie's focal
selection doesn't refresh.  Tracked separately; inner voice does
its part (picks the current AWM focal each tick) and inherits
whatever variability the focal-selection upstream provides.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from ..events import (
    EventKind,
    AttendedPerceptEvent,
)
from ..bus import EventBus


# How many ticks between inner-voice firings.  Reverie cadence —
# slower than the per-tick chemistry decay, faster than sleep
# consolidation.  At 5 Hz this is ~10 s per inner utterance, leaving
# room for the substrate-mediated loop to settle (percept → cortical
# → consolidator → next-tick AWM/chemistry update) without tight
# self-feedback.
INNER_VOICE_INTERVAL = 50
# Focal must clear this salience for the agent to bother articulating
# it.  The Thalamic Gate's existing salience scale (0..1).  At low
# values reverie is just background; articulation is reserved for
# what AWM is actually holding.
INNER_VOICE_MIN_FOCAL_SALIENCE = 0.10
# Cap on inner-utterance length.  Short clauses keep the SVO
# composition cheap and the re-perception meaningful.
INNER_VOICE_MAX_LEN = 80
# Recent-utterance ring — repeats are silently dropped (the agent
# doesn't re-perceive the same inner phrase back-to-back; that would
# be a tight rumination loop, not thought).
RECENT_UTTERANCE_CAP = 10


class InnerVoice:
    """Continuous internal articulation during reverie.

    Tick-driven (no event subscriptions).  Composes a short inner
    utterance about the current AWM focal, threaded with recent
    vmDMN narrative, and publishes it as a self-perceived
    AttendedPerceptEvent so the standard percept path engages.
    """

    SUBSCRIPTIONS = ()      # tick-driven only

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None,
                 awm_focal_provider: Optional[Callable] = None,
                 vmdmn_narrative_provider: Optional[Callable] = None,
                 substrate_provider: Optional[Callable] = None,
                 is_asleep_provider: Optional[Callable] = None):
        """
        Providers (all optional — absent provider = inner voice
        skips that source; capability is robust to partial wiring):

        - awm_focal_provider: () -> {'name': str, 'salience': float}
            or None.  The concept AWM currently holds at top
            salience.  Absent or None → inner voice has nothing to
            articulate; skips.
        - vmdmn_narrative_provider: () -> str or None.  Recent
            self-reflective narrative.  Threaded into the utterance
            when present.
        - substrate_provider: () -> Substrate or None.  Used to find
            one or two facts about the focal to articulate.
        - is_asleep_provider: () -> bool.  Inner voice does not run
            during sleep (Phase S owns the substrate; cognition
            gated).
        """
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._awm_focal = awm_focal_provider
        self._vmdmn = vmdmn_narrative_provider
        self._substrate = substrate_provider
        self._is_asleep = is_asleep_provider or (lambda: False)

        self._last_fire_cycle: int = -10_000
        self._last_utterance: str = ''
        self._recent: Deque[str] = deque(maxlen=RECENT_UTTERANCE_CAP)

        # Diagnostics.
        self.utterances_emitted: int = 0
        self.skipped_asleep: int = 0
        self.skipped_throttle: int = 0
        self.skipped_no_focal: int = 0
        self.skipped_low_salience: int = 0
        self.skipped_incoherent: int = 0
        self.skipped_repeat: int = 0

    # ---- per-tick ----

    def tick(self) -> None:
        """Try to articulate one inner utterance.  Cheap fast-paths
        first; composition only when all gates pass."""
        if self._is_asleep():
            self.skipped_asleep += 1
            return
        cycle = int(self._cycle_provider())
        if cycle - self._last_fire_cycle < INNER_VOICE_INTERVAL:
            self.skipped_throttle += 1
            return

        focal_info = self._awm_focal() if self._awm_focal else None
        if not focal_info or not focal_info.get('name'):
            self.skipped_no_focal += 1
            return
        focal = str(focal_info['name'])
        salience = float(focal_info.get('salience', 0.0))
        if salience < INNER_VOICE_MIN_FOCAL_SALIENCE:
            self.skipped_low_salience += 1
            return

        text = self._compose(focal)
        if not self._is_coherent(text, focal):
            self.skipped_incoherent += 1
            return
        if text in self._recent:
            self.skipped_repeat += 1
            return

        self._recent.append(text)
        self._last_utterance = text
        self._last_fire_cycle = cycle
        self.utterances_emitted += 1
        self._publish_self_percept(focal, salience, text, cycle)

    # ---- composition ----

    def _compose(self, focal: str) -> str:
        """Compose a short inner utterance from focal + substrate
        facts + recent vmDMN narrative.  First-person, internal."""
        fact_clause = self._first_substrate_fact(focal)
        narrative = self._recent_narrative_clause()
        if fact_clause and narrative:
            text = f"I notice: {fact_clause}. {narrative}"
        elif fact_clause:
            text = f"I notice: {fact_clause}."
        elif narrative:
            # Narrative-only — still anchor the focal in the text so
            # the self-percept attends to it (and so the coherence
            # check passes).
            text = f"I think about {focal}: {narrative}"
        else:
            text = f"I am holding: {focal}."
        return text[:INNER_VOICE_MAX_LEN]

    def _first_substrate_fact(self, focal: str) -> str:
        """Find one substrate outgoing edge of `focal` and return
        it as a clause.  Skip trivial/structural relations so the
        articulation is contentful."""
        sub = self._substrate() if self._substrate else None
        if sub is None:
            return ''
        SKIP_RELATIONS = {'is_a', 'a', 'the', 'of', '_abstract'}
        # Fast path: walk only this focal's own out-edges via the
        # concept index, instead of scanning the whole (600k+) edge
        # dict every utterance.  Same result, bounded by out-degree.
        concepts = getattr(sub, 'concepts', None)
        concept = concepts.get(focal) if concepts is not None else None
        if concept is not None:
            edges_out = getattr(concept, 'edges_out', None) or {}
            for r, elist in edges_out.items():
                if r in SKIP_RELATIONS:
                    continue
                for e in elist:
                    try:
                        _s, er, t = e.key
                    except Exception:
                        continue
                    return f"{focal} {er} {t}"
            # No contentful relation — is_a fallback (still
            # informative even if structural).
            for e in edges_out.get('is_a', []):
                try:
                    _s, _er, t = e.key
                except Exception:
                    continue
                return f"{focal} is_a {t}"
            return ''
        # Fallback for substrates without a concepts index
        # (e.g. LongTermSubstrate): original full-edge scan.
        edges = getattr(sub, 'edges', None)
        if not edges:
            return ''
        for key, _edge in edges.items():
            try:
                s, r, t = key
            except Exception:
                continue
            if s != focal:
                continue
            if r in SKIP_RELATIONS:
                continue
            return f"{focal} {r} {t}"
        for key, _edge in edges.items():
            try:
                s, r, t = key
            except Exception:
                continue
            if s == focal and r == 'is_a':
                return f"{focal} is_a {t}"
        return ''

    def _recent_narrative_clause(self) -> str:
        """Pull a short clause from vmDMN's last narrative if
        available.  Truncated; the inner utterance must stay short."""
        if self._vmdmn is None:
            return ''
        try:
            n = self._vmdmn()
        except Exception:
            return ''
        if not isinstance(n, str) or not n.strip():
            return ''
        # First sentence (or first 60 chars), whichever is shorter.
        clause = n.strip()
        for sep in ('. ', '? ', '! '):
            i = clause.find(sep)
            if i > 0:
                clause = clause[:i + 1]
                break
        return clause[:60]

    # ---- coherence check (the doctrine-aligned junk filter) ----

    def _is_coherent(self, text: str, focal: str) -> bool:
        """Reject obviously-incoherent utterances before they are
        self-perceived.  Cheap structural checks; not a semantic
        judgment (Phase S earn-or-dissolve is the real filter on
        whatever substrate growth follows)."""
        if not text or len(text.split()) < 3:
            return False
        if focal not in text:
            return False
        return True

    # ---- self-percept emission ----

    def _publish_self_percept(self,
                              focal: str,
                              salience: float,
                              text: str,
                              cycle: int) -> None:
        """Publish the inner utterance as an AttendedPerceptEvent
        with origin='self'.  Reuses the standard percept type so
        cortical / chemistry / hippocampus engage automatically;
        the origin tag lets downstream consumers source-monitor it
        as self-generated (vs peer / forager / sensor)."""
        try:
            self.bus.publish(AttendedPerceptEvent(
                kind=EventKind.ATTENDED_PERCEPT,
                cycle=cycle,
                timestamp=time.time(),
                source_capability='inner_voice',
                origin='self',
                origin_detail='inner_voice',
                focals=[focal],
                payload={focal: salience},
                raw_text=text,
                modality='text',
                salience=salience,
                novelty=0.0,
                m_content=0.0,
                i_content=0.0,
                threshold_used=0.0,
            ))
        except Exception:
            # The capability is best-effort; never raise into the
            # tick loop.
            pass

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'utterances_emitted': self.utterances_emitted,
            'last_utterance': self._last_utterance[:INNER_VOICE_MAX_LEN],
            'last_fire_cycle': self._last_fire_cycle,
            'skipped_asleep': self.skipped_asleep,
            'skipped_throttle': self.skipped_throttle,
            'skipped_no_focal': self.skipped_no_focal,
            'skipped_low_salience': self.skipped_low_salience,
            'skipped_incoherent': self.skipped_incoherent,
            'skipped_repeat': self.skipped_repeat,
        }
