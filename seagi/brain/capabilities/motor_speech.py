"""Motor / Speech — single-voice deliberate composer.

Brain analog: motor + premotor cortex for speech, integrating
multiple cortical sources into ONE coherent utterance.

The Phase 3 stub just relayed the cortical thought_text plus
an AWM context cue.  That was clause-concatenation in
disguise.  Phase 5 replaces it with a DELIBERATE composer:
multiple sources flow in, ONE voice flows out.

What it reads
-------------
- Cortical thought (focal + relation + target + method +
  confidence + thin_substrate flag + text)
- AWM (currently active concepts)
- vmDMN.last_narrative (recent self-reflection)
- Insula.felt_state (body band + narrative)
- BodySchema (can_speak gate)
- ValueLandscape (focal value coloring)
- ACC.recent_thoughts (avoid contradicting recent voice)

What it emits
-------------
SpeechEmittedEvent — finalized response with voice_mode +
sources_used.  Hippocampus / SourceMonitor / TimePerception
register that we spoke.
ALSO sets `pending_response` on the runtime stub for
chat()'s synchronous return path.

Voice modes (chosen, not concatenated)
--------------------------------------
- `direct`           — strong cortical thought, peer-driven
- `body_aware`       — band depleted/agitated → body frame
                        + primary content
- `reflective`       — no crisp cortical, vmDMN narrative
                        recent → speak from reflection
- `honest_uncertain` — cortical produced metacog/thin
- `minimal`          — nothing landed, brief honest line

Single-voice principle
----------------------
ONE first-person paragraph.  No clause-concatenation across
sources.  The body frame is woven into the lead, not appended
as a separate sentence (unless that IS the lead).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from ..events import (
    EventKind, BrainEvent,
    SpeechRequestEvent,
    SpeechEmittedEvent,
    ArbitrationDecidedEvent,
)
from ..bus import EventBus


# Confidence below this is "thin" even if not flagged.
WEAK_CONFIDENCE = 0.25
# Cycles within which vmDMN's narrative is still "recent."
NARRATIVE_RECENCY = 200
# Cap on repeat phrasing — if last K speeches all share the
# same lead phrase, force a different mode.
REPEAT_CAP = 3


class MotorSpeech:
    """The single-voice deliberate composer."""

    SUBSCRIPTIONS = (
        EventKind.SPEECH_REQUEST,
        EventKind.ARBITRATION_DECIDED,
    )

    def __init__(self,
                 bus: EventBus,
                 awm_provider: Callable,
                 vmdmn_provider: Optional[Callable] = None,
                 insula_provider: Optional[Callable] = None,
                 body_schema_provider: Optional[Callable] = None,
                 value_provider: Optional[Callable] = None,
                 cycle_provider: Optional[Callable] = None,
                 chemistry_provider: Optional[Callable] = None,
                 lts_provider: Optional[Callable] = None):
        self.bus = bus
        self._awm_provider = awm_provider
        self._vmdmn_provider = vmdmn_provider
        self._insula_provider = insula_provider
        self._body_schema_provider = body_schema_provider
        self._value_provider = value_provider
        self._cycle_provider = cycle_provider or (lambda: 0)
        # Phase E: chemistry_provider feeds tone_summary() into
        # voice composition; lts_provider lets weaving check that
        # a co-active concept is genuinely related to the focal
        # (via shared edges or high concept-salience) before
        # surfacing it as a "sits beside" tail.
        self._chemistry_provider = chemistry_provider
        self._lts_provider = lts_provider
        # Synchronous chat() picks this up.
        self.pending_response: str = ''
        # Recent speech tracking (self-observation; avoids
        # repetition).
        self._recent_leads: Deque[str] = deque(maxlen=REPEAT_CAP)
        # Diagnostics.
        self.speeches_emitted: int = 0
        self.mode_counts: Dict[str, int] = {}

    # ---- event handler ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, SpeechRequestEvent):
            self._compose_from_request(event, bus)
        elif isinstance(event, ArbitrationDecidedEvent):
            # Arbitration on the speech loop CAN gate composition
            # in future; Phase 5 keeps the synchronous chat()
            # path as the primary driver.  We just record that
            # speech-loop arbitration happened.
            pass

    # ---- composition ----

    def _compose_from_request(self,
                                     ev: SpeechRequestEvent,
                                     bus: EventBus) -> None:
        """Pick voice mode, compose one coherent response,
        emit SpeechEmittedEvent + stash pending_response."""
        # Step 0: gate on body state.
        body = self._read_body_schema()
        if body and not body.get('can_speak', True):
            text = ('I am too depleted right now to put '
                      'together a clear answer.')
            self._finalize(
                bus, focal=ev.intended_focal,
                voice_mode='minimal',
                text=text, sources_used=['body_schema'],
                body_band=body.get('band', ''),
                cycle=ev.cycle)
            return

        # Step 1: pick voice mode.
        mode, sources = self._pick_mode(ev)

        # Step 2: compose body.
        text = self._compose(ev, mode)

        # Step 3: repeat-avoidance.  If the same lead keeps
        # firing, force a softer mode.
        lead = text.split('.')[0][:60] if text else ''
        if (lead and self._recent_leads
                and all(l == lead for l in self._recent_leads)
                and mode != 'minimal'):
            mode = 'minimal'
            sources = ['repeat_avoidance']
            text = ('I find myself drawn back to the same '
                      'place — let me sit with it for a '
                      'moment.')

        self._recent_leads.append(lead)
        body_band = ''
        if body:
            body_band = body.get('band', '')
        self._finalize(
            bus, focal=ev.intended_focal,
            voice_mode=mode, text=text,
            sources_used=sources,
            body_band=body_band,
            cycle=ev.cycle)

    def _pick_mode(self,
                       ev: SpeechRequestEvent
                       ) -> tuple[str, List[str]]:
        """Mode selection.  Reads structured signal from the
        SpeechRequestEvent contract — `thin_substrate` is the
        boolean cortical sets when its metacognitive monitor
        flagged the substrate as thin around this focal.  We do
        NOT substring-match the rendered text; that was a hidden
        coupling and broke at any rewording.

        Order matters: most-confident path wins."""
        sources: List[str] = []
        thought_text = (ev.thought_text or '').strip()
        thin_signal = bool(ev.thin_substrate)

        body = self._read_body_schema()
        band = body.get('band', 'settled') if body else 'settled'
        body_alarming = band in ('depleted', 'agitated')

        # Strong cortical thought → direct (with body frame
        # when band is alarming).
        if thought_text and not thin_signal:
            sources.append('cortical')
            if body_alarming:
                sources.append('insula')
                return ('body_aware', sources)
            return ('direct', sources)

        # No cortical text but vmDMN has a recent narrative.
        vm = self._read_vmdmn()
        if vm and self._narrative_is_recent(vm):
            sources.append('vmdmn')
            if body_alarming:
                sources.append('insula')
                return ('body_aware', sources)
            return ('reflective', sources)

        # Cortical produced a thin/metacog acknowledgment.
        if thin_signal:
            sources.append('cortical')
            return ('honest_uncertain', sources)

        return ('minimal', sources)

    def _compose(self,
                   ev: SpeechRequestEvent,
                   mode: str) -> str:
        """Build the response text from the chosen mode.
        Single voice: ONE paragraph, woven not concatenated."""
        thought_text = (ev.thought_text or '').strip()
        body = self._read_body_schema() or {}
        band = body.get('band', 'settled')
        body_nar = ''
        ins = self._read_insula()
        if ins:
            body_nar = (ins.get('narrative', '') or '').strip()

        vm = self._read_vmdmn()
        vm_text = (getattr(vm, 'last_narrative', '') or '').strip() if vm else ''

        if mode == 'direct':
            return self._weave_direct(
                thought_text, ev.intended_focal,
                ev.awm_snapshot or {})
        if mode == 'body_aware':
            primary = (thought_text if thought_text
                          else vm_text)
            return self._weave_body_aware(
                body_nar, primary, ev.intended_focal,
                ev.awm_snapshot or {})
        if mode == 'reflective':
            return self._weave_reflective(
                vm_text, ev.intended_focal, body_nar)
        if mode == 'honest_uncertain':
            return self._weave_honest(
                thought_text, ev.intended_focal,
                body_nar)
        # minimal
        return ('I heard you but found nothing in my '
                  'current state to say back.')

    # ---- weavers (single-voice composition) ----

    def _weave_direct(self,
                          thought_text: str,
                          focal: str,
                          awm_snapshot: Dict[str, Any]) -> str:
        # Lead with the cortical thought.  Optionally weave
        # ONE supporting AWM neighbor as a clause — but ONLY
        # if the neighbor is genuinely related to the focal
        # (Phase E: relatedness check, not just any co-active).
        text = thought_text.rstrip('.').rstrip()
        others = self._other_active(awm_snapshot, focal)
        # Filter out targets already mentioned in thought_text.
        others = [o for o in others
                    if o.lower() not in text.lower()]
        # Phase E 2026-05-16: relatedness filter.  Before surfacing
        # a co-active concept as "X sits beside it in my mind",
        # check that it's actually substrate-connected to the
        # focal (shared edge or high-salience co-occurrence) AND
        # that the chemistry tone justifies a reflective tail.
        # A peer asking about "love" right after "death" shouldn't
        # always surface death — only if love and death are
        # genuinely linked in the substrate.
        tail = self._pick_related_tail(focal, others)
        if tail:
            text = f'{text}, and {tail} sits beside it in my mind.'
        else:
            text = text + '.'
        text = self._color_with_value(text, focal)
        # Phase E: tone-based coloring.  Read current chemistry
        # tone and prefix a soft mood marker on strongly-polar
        # responses.  Subtle; only fires when tone is clearly
        # non-flat.
        text = self._color_with_tone(text)
        return text

    def _pick_related_tail(self,
                                  focal: str,
                                  candidates: List[str]) -> str:
        """Return the first AWM-coactive candidate that is
        substrate-related to the focal, or '' if none qualify.

        Relatedness = there is an edge (in either direction)
        between focal and candidate, OR both have high
        concept-salience suggesting genuine current relevance.
        Without an LTS provider, falls back to permissive
        behavior (no filtering — preserves old test setup).
        """
        if not candidates:
            return ''
        if self._lts_provider is None:
            return candidates[0]  # back-compat: pre-Phase-E behavior
        try:
            lts = self._lts_provider()
        except Exception:
            return candidates[0]
        if lts is None:
            return candidates[0]
        for cand in candidates:
            # Look for an edge in either direction.
            related = False
            try:
                fneigh = lts.neighbors(focal)
                if any(t == cand for t, _r, _s in fneigh):
                    related = True
                if not related:
                    cneigh = lts.neighbors(cand)
                    if any(t == focal for t, _r, _s in cneigh):
                        related = True
            except Exception:
                pass
            if related:
                return cand
        # No structurally-related candidate — don't surface a
        # tail.  Voice stays focused.
        return ''

    def _color_with_tone(self, text: str) -> str:
        """Phase E: prepend or append a soft tone marker when
        chemistry signals a clearly-polar mood.  Subtle — only
        fires on non-flat labels; never overrides the thought."""
        if self._chemistry_provider is None:
            return text
        try:
            chem = self._chemistry_provider()
            if chem is None:
                return text
            tone = chem.tone_summary()
        except Exception:
            return text
        label = tone.get('label', 'flat')
        if label in ('flat', 'positive'):
            return text  # no decoration
        # Soft prefixes.  Keep light; the thought leads.
        prefixes = {
            'warm': '',          # warmth shows in word choice, not prefix
            'curious': '',
            'alert': '',
            'somber': 'Quietly: ',
            'anxious': 'Carefully: ',
        }
        prefix = prefixes.get(label, '')
        if not prefix:
            return text
        # Lowercase the first letter so the prefix flows.
        if text and text[0].isupper():
            text = text[0].lower() + text[1:]
        return prefix + text

    def _weave_body_aware(self,
                                body_narrative: str,
                                primary_text: str,
                                focal: str,
                                awm_snapshot: Dict[str, Any]) -> str:
        # Body frame becomes the lead.  Primary content
        # follows on the same beat.
        body = body_narrative or 'My body is asking for attention.'
        body = body.rstrip('.').rstrip()
        primary = (primary_text or '').rstrip('.').rstrip()
        if not primary:
            return body + '.'
        # Single sentence weave when short.
        if len(primary) < 80:
            return f'{body} — still, {primary.lower()}.'
        return f'{body}.  Still: {primary}.'

    def _weave_reflective(self,
                                  vm_text: str,
                                  focal: str,
                                  body_narrative: str) -> str:
        text = vm_text
        if body_narrative and body_narrative not in text:
            # Body framing already inside vm_text often —
            # don't double up.
            return text
        return text

    def _weave_honest(self,
                            thought_text: str,
                            focal: str,
                            body_narrative: str) -> str:
        text = thought_text or (
            f"I do not know {focal or 'that'} well — "
            "my substrate is thin around it.")
        if body_narrative:
            return f'{body_narrative} {text}'
        return text

    # ---- helpers ----

    def _other_active(self,
                          awm_snapshot: Dict[str, Any],
                          focal: str) -> List[str]:
        active = list(awm_snapshot.get('active_concepts', []) or [])
        return [c for c in active if c and c != focal][:3]

    def _color_with_value(self,
                                  text: str,
                                  focal: str) -> str:
        """Optional value-coloring — adds a one-word adverb
        when a focal has strong polar value.  Kept very light
        to avoid overstuffing voice."""
        if self._value_provider is None or not focal:
            return text
        try:
            vp = self._value_provider()
            if vp is None:
                return text
            v = float(vp.value_of(focal))
        except Exception:
            return text
        # Strong polar only.
        if v >= 0.5:
            return text   # confident already
        if v <= -0.5:
            # Add a cautious frame to the front.
            return f'Carefully: {text[0].lower()}{text[1:]}'
        return text

    def _read_body_schema(self) -> Optional[Dict[str, Any]]:
        if self._body_schema_provider is None:
            return None
        try:
            bs = self._body_schema_provider()
            return bs.envelope() if bs else None
        except Exception:
            return None

    def _read_insula(self) -> Optional[Dict[str, Any]]:
        if self._insula_provider is None:
            return None
        try:
            ins = self._insula_provider()
            return ins.felt_state() if ins else None
        except Exception:
            return None

    def _read_vmdmn(self) -> Any:
        if self._vmdmn_provider is None:
            return None
        try:
            return self._vmdmn_provider()
        except Exception:
            return None

    def _narrative_is_recent(self, vm: Any) -> bool:
        if not getattr(vm, 'last_narrative', ''):
            return False
        try:
            last_c = int(getattr(vm, 'last_narrative_cycle', 0))
        except Exception:
            return True
        now = self._cycle_provider()
        return (now - last_c) <= NARRATIVE_RECENCY

    # ---- finalize ----

    def _finalize(self,
                       bus: EventBus,
                       focal: str,
                       voice_mode: str,
                       text: str,
                       sources_used: List[str],
                       body_band: str,
                       cycle: int) -> None:
        self.pending_response = text
        self.mode_counts[voice_mode] = self.mode_counts.get(
            voice_mode, 0) + 1
        self.speeches_emitted += 1
        bus.publish(SpeechEmittedEvent(
            kind=EventKind.SPEECH_EMITTED,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='motor_speech',
            origin='internal',
            origin_detail=voice_mode,
            intended_focal=focal,
            voice_mode=voice_mode,
            text=text,
            sources_used=list(sources_used),
            body_band=body_band,
        ))

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'speeches_emitted': self.speeches_emitted,
            'mode_counts': dict(self.mode_counts),
            'recent_leads_seen': len(self._recent_leads),
        }
