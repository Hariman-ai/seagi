"""Sensory intake — raw input decoding.

Brain analog: primary sensory cortices.  First contact with the
world.  Decodes modality-specific input (text, future image,
audio, proprioception) into the modality-agnostic activation
representation downstream capabilities consume.

Phase 1: FULL for text.  Image/audio/proprioception decoders
plug in here as additional branches without changing anything
downstream.

Reads        : nothing on the bus (it's the source)
Writes       : nothing directly to substrate (Thalamic Gate
                gatekeeps writes)
Emits        : RawPerceptEvent — every input becomes one
Brain analog : V1 (visual), A1 (auditory), S1 (somatosensory),
                Wernicke's-style decoding for text

Design note
-----------
The sensory layer DOES NOT decide what's important.  It just
decodes and emits.  The Thalamic Gate decides what reaches
awareness.  This separation is brain-correct: V1 fires for
every visual pattern; thalamus decides which patterns reach
cortex.  Mixing the two in v1 was a failure mode.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..events import EventKind, RawPerceptEvent
from ..bus import EventBus


class SensoryIntake:
    """Source of all percept events.  One instance per running
    brain.  Public interface is `intake(payload, modality, ...)`
    which any input source (forager, chat handler, future
    sensors) calls."""

    def __init__(self,
                 bus: EventBus,
                 engine: Any = None,
                 cycle_provider: Optional[Any] = None):
        """
        bus: the brain's event bus
        engine: optional v1 engine reference for substrate access
                during text decoding (text_to_observation needs it)
        cycle_provider: callable returning current cycle int.
                If None, uses engine.hierarchy.cycle_counter.
        """
        self.bus = bus
        self.engine = engine
        self._cycle_provider = cycle_provider
        self.intake_count: int = 0

    def _cycle(self) -> int:
        if self._cycle_provider is not None:
            try:
                return int(self._cycle_provider())
            except Exception:
                pass
        if self.engine is not None:
            try:
                return int(self.engine.hierarchy.cycle_counter)
            except Exception:
                pass
        return 0

    def intake(self,
                 payload: Any,
                 modality: str = 'text',
                 origin: str = 'internal',
                 origin_detail: str = '',
                 ) -> None:
        """Decode `payload` and emit a RawPerceptEvent.

        payload: modality-dependent input.  For text: str.
            For future image: bytes or PIL image.  For audio:
            bytes or np.array.
        modality: 'text' / 'image' / 'audio' / 'proprioception'
        origin: source-monitoring tag — 'peer' / 'forager'
            / 'sensor' / 'internal'
        origin_detail: peer_id / rss_url / etc.

        Does NOT block.  Just decodes and publishes.  Downstream
        capabilities (Thalamic Gate, etc.) react asynchronously.
        """
        cycle = self._cycle()
        decoded: Dict[str, Any] = {}
        raw_text = ''

        if modality == 'text':
            raw_text = str(payload or '')
            if raw_text and self.engine is not None:
                try:
                    from seagi.core.text_io import text_to_observation
                    decoded = text_to_observation(
                        raw_text, self.engine.substrate,
                        cycle=cycle) or {}
                except Exception:
                    decoded = {}
        elif modality == 'world':
            # Grounding world-loop (2026-06-03): a non-text modality
            # that carries a world-state token (or a structured dict).
            # No symbol decoder — the GroundingLoop reads the state
            # directly from the percept and predicts the next one.
            # This exercises the formerly-stubbed non-text branch and
            # keeps grounding modality-agnostic at the percept level.
            if isinstance(payload, dict):
                decoded = dict(payload)
            else:
                decoded = {'world_state': str(payload)}
        else:
            # Image/audio/etc. will dispatch here in later phases.
            # For now, accept the raw payload and let downstream
            # handle the empty `decoded` dict.
            pass

        ev = RawPerceptEvent(
            kind=EventKind.RAW_PERCEPT,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='sensory',
            origin=origin,
            origin_detail=origin_detail,
            payload=decoded,
            raw_text=raw_text[:1000],  # cap stored excerpt
            modality=modality,
        )
        self.intake_count += 1
        self.bus.publish(ev)
