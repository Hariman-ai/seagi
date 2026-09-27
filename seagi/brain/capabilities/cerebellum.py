"""Cerebellum — prediction + timing + error correction.

Brain analog: cerebellum.  Two sub-functions on shared
machinery:

    LATERAL (cognitive prediction):
        Predicts the next attended focal / next thought target
        from recent AWM trajectory.  When the prediction misses,
        fires a cognitive PredictionErrorEvent.

    VERMIS (affective prediction):
        Predicts the agent's own chemistry trajectory.  When the
        actual chemistry diverges from the prediction (an
        unexpected emotional shift), fires an affective
        PredictionErrorEvent.

Why two error kinds in one capability: real cerebellar circuits
share the same machinery but route to different downstream
populations.  Same here — one predictor with two output
populations.

Phase 4b: FULL implementation (tight, not extravagant).
Phase I.1a (2026-05-18): vermis becomes multi-channel.  When a
`chemistry_provider` is wired, predicts each of the 8 chemistry
channels separately and fires per-channel affective PE
(focal='chem.<channel>') with channel-specific thresholds.
Without a provider, falls back to single-channel lifeforce
prediction (the Phase 4b behavior).

Phase I.2 (2026-05-18): lateral now also predicts WHEN (not
just WHAT).  Each (prev, next) transition accumulates a
running mean + variance of cycle intervals via Welford's
online algorithm.  After TIMING_MIN_OBSERVATIONS, a normalized
interval deviation above TIMING_PE_THRESHOLD fires a timing PE
(error_kind='timing', sign=-1).  This is the cerebellar
delay-tuning signature (real Purkinje cells fire at specific
interval lengths).

Subscribes
----------
ATTENDED_PERCEPT    — log focal trajectory; check prediction
THOUGHT_PRODUCED    — refine cognitive predictor
CHEMISTRY_FIRE      — log chemistry; check vermis prediction
INTEROCEPTION       — affective prediction grounding

Emits
-----
PREDICTION_ERROR    — when |predicted - actual| crosses threshold
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from ..events import (
    EventKind, BrainEvent,
    AttendedPerceptEvent,
    ThoughtProducedEvent,
    ChemistryEvent,
    InteroceptionEvent,
    PredictionErrorEvent,
)
from ..bus import EventBus


COGNITIVE_PE_THRESHOLD = 0.4
# Lifeforce-only fallback threshold (used when no chemistry
# provider is wired).
AFFECTIVE_PE_THRESHOLD = 0.12
TRAJECTORY_DEPTH = 8

# Phase I.1b: lateral transitions are float-weighted and decay
# at promille scale.  Hit (observed transition) adds +1.0; miss
# (predicted-but-didn't-arrive) subtracts MISS_WEAKENING from
# the wrong prediction.  Slow background decay applied in batches
# so unused transitions fade — doctrine "use it or lose it".
TRANSITION_DECAY_PER_BATCH = 0.05
TRANSITION_DECAY_BATCH_SIZE = 50    # apply decay every N attended events
TRANSITION_MISS_WEAKENING = 0.5
TRANSITION_FLOOR = 0.1              # entries below this are pruned
TRANSITION_PREDICTION_THRESHOLD = 1.5  # weight required to count as a prediction

# Phase I.1c: tone-conditional buckets.  Doctrine: same word +
# different context = different cocktail.  When a tone_provider
# is wired, observations and predictions are refined by current
# chemistry tone.  Bucket boundaries are based on the valence
# scalar from chemistry.tone_summary() (in [-1, 1]).
TONE_BUCKET_I_THRESHOLD = 0.15
TONE_BUCKET_M_THRESHOLD = -0.15

# Phase I.2: per-transition timing prediction — the cerebellar
# delay-tuning signature (real Purkinje cells fire at specific
# interval lengths).  For each (prev, next) transition we track
# a running mean + variance of observed cycle intervals via
# Welford's online algorithm.  After TIMING_MIN_OBSERVATIONS,
# a normalized |interval_error| above TIMING_PE_THRESHOLD fires
# a 'timing'-kind PE.  Sign is always -1 (mirrors cognitive PE
# convention — both early and late arrivals are prediction
# failures; ACC fires conflict, VTA stays silent).
TIMING_MIN_OBSERVATIONS = 3
TIMING_PE_THRESHOLD = 0.25

# Phase I.1a: per-channel affective PE thresholds.  Higher than
# the lifeforce threshold because fanning out across 8 channels
# would otherwise multiply event volume — doctrine: low energy /
# fire when needed.  Slow channels (cortisol, serotonin) get
# tighter thresholds because their natural drift is small; fast
# channels (NE) get looser thresholds because larger swings are
# normal.
CHANNEL_PE_THRESHOLDS: Dict[str, float] = {
    'norepinephrine': 0.18,
    'acetylcholine':  0.15,
    'gaba':           0.15,
    'dopamine':       0.12,
    'endorphins':     0.10,
    'oxytocin':       0.10,
    'serotonin':      0.08,
    'cortisol':       0.06,
}
# Per-channel trajectory depth.  Same shape across channels.
CHANNEL_HISTORY_DEPTH = 8


class Cerebellum:
    """Lateral (cognitive) + vermis (affective) predictor."""

    SUBSCRIPTIONS = (
        EventKind.ATTENDED_PERCEPT,
        EventKind.THOUGHT_PRODUCED,
        EventKind.CHEMISTRY_FIRE,
        EventKind.INTEROCEPTION,
    )

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None,
                 chemistry_provider: Optional[Callable] = None,
                 tone_provider: Optional[Callable] = None):
        """`chemistry_provider`, when wired, returns a dict of
        channel→level (the live 8-channel chemistry state).  When
        provided, vermis runs per-channel predictions.  When
        absent, vermis falls back to lifeforce-only prediction
        (the Phase 4b behavior).

        `tone_provider` (I.1c), when wired, returns the current
        chemistry tone summary dict (output of
        chemistry.tone_summary()).  When provided, lateral
        predictions are refined by an overlay table keyed by
        tone bucket — same prev focal can predict different
        successors depending on whether the agent is in an
        I-leaning, M-leaning, or neutral mood."""
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._chemistry_provider = chemistry_provider
        self._tone_provider = tone_provider
        # Lateral: rolling focal-transition weights (float; Phase
        # I.1b — was int counts).  Hit reinforces via observation
        # (+1); miss weakens (predicted, prev) by MISS_WEAKENING;
        # slow batched decay applied every BATCH_SIZE attended
        # events.  Entries below TRANSITION_FLOOR are pruned.
        self._transitions: Dict[tuple, float] = {}
        # I.1c: tone-conditional overlay.  Key (prev, bucket, next).
        # Populated only when a tone_provider is wired.  Predictor
        # uses bucket-specific entries when present, else falls back
        # to the global _transitions table.
        self._tone_transitions: Dict[tuple, float] = {}
        self._focal_history: Deque[str] = deque(
            maxlen=TRAJECTORY_DEPTH)
        # I.2: cycle of the last attended event (drives interval
        # arithmetic for timing).  None before first event.
        self._last_attended_cycle: Optional[int] = None
        # I.2: per-transition timing stats via Welford's online
        # mean/variance.  Value tuple: (count, mean, m2) where m2
        # is the sum of squared deviations from mean (variance =
        # m2 / count when count > 1).
        self._transition_timing: Dict[tuple, list] = {}
        self._attended_count: int = 0
        # Vermis fallback: rolling lifeforce samples.
        self._affective_history: Deque[float] = deque(maxlen=8)
        # Vermis multi-channel: rolling per-channel samples.
        # Populated lazily as channels are observed.
        self._channel_histories: Dict[str, Deque[float]] = {}
        # Diagnostics.
        self.cognitive_errors_fired: int = 0
        self.affective_errors_fired: int = 0
        # Per-channel affective error counts (multi-channel mode).
        self.channel_errors_fired: Dict[str, int] = {}
        # I.1b diagnostics.
        self.transitions_decayed_batches: int = 0
        self.transitions_pruned: int = 0
        # I.2 diagnostics.
        self.timing_errors_fired: int = 0

    # ---- event handler ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, AttendedPerceptEvent):
            self._on_attended(event, bus)
        elif isinstance(event, ThoughtProducedEvent):
            self._on_thought(event)
        elif isinstance(event, InteroceptionEvent):
            self._on_interoception(event, bus)
        elif isinstance(event, ChemistryEvent):
            # Phase I.1a follow-up (2026-05-18): vermis samples
            # chemistry on every chemistry-fire too, not only on
            # interoception.  Diagnostic probe showed insula only
            # fires interoception on body-state shifts, leaving
            # multi-channel vermis silent in steady-state.
            # CHEMISTRY_FIRE is the right trigger because it
            # marks any chemistry shift the rest of the system
            # already cares about.
            self._on_chemistry_fire(event, bus)

    # ---- lateral (cognitive) ----

    def _on_attended(self,
                          ev: AttendedPerceptEvent,
                          bus: EventBus) -> None:
        if not ev.focals:
            return
        actual = ev.focals[0]
        self._attended_count += 1
        # Phase I.1b: slow background decay so unused transitions
        # fade ("use it or lose it").  Applied in batches to amortize.
        if self._attended_count % TRANSITION_DECAY_BATCH_SIZE == 0:
            self._apply_decay()
        # Phase I.1c: read tone bucket once per event so the same
        # context drives both the prediction and the observation.
        bucket = self._current_bucket()
        # If we have a prediction for what should follow the
        # last seen focal, check error.
        if self._focal_history:
            prev = self._focal_history[-1]
            predicted = self._predict_next(prev, bucket)
            if predicted and predicted != actual:
                # Phase I.1b/c: weaken the failed prediction in
                # whichever table produced it (tone-specific if
                # active, else global).  Floor at 0.
                self._weaken_transition(prev, predicted, bucket)
                self._fire_cognitive_pe(
                    bus, prev=prev, focal=actual,
                    predicted=predicted)
        # Record transition for future predictions.  This is the
        # "hit reinforces by observation" half of I.1b — the
        # observed (prev → actual) transition gets +1.0 each
        # time it occurs, including when a different prediction
        # was made and missed.
        if self._focal_history:
            prev = self._focal_history[-1]
            key = (prev, actual)
            self._transitions[key] = self._transitions.get(
                key, 0.0) + 1.0
            # I.1c: also reinforce the tone-specific overlay so
            # the agent learns "in M-leaning tone, a tends to b".
            if bucket:
                tone_key = (prev, bucket, actual)
                self._tone_transitions[tone_key] = \
                    self._tone_transitions.get(tone_key, 0.0) + 1.0
            # I.2: timing — check against the per-transition
            # interval estimate (if mature), then update the
            # estimate with this observation.
            if self._last_attended_cycle is not None:
                interval = ev.cycle - self._last_attended_cycle
                if interval >= 0:
                    self._maybe_fire_timing_pe(
                        bus, prev=prev, focal=actual,
                        interval=interval, ev=ev)
                    self._update_timing(prev, actual, interval)
        self._focal_history.append(actual)
        self._last_attended_cycle = ev.cycle

    def _current_bucket(self) -> str:
        """I.1c: derive tone bucket from chemistry.tone_summary.
        Empty string when no tone provider is wired or when the
        provider fails."""
        if self._tone_provider is None:
            return ''
        try:
            tone = self._tone_provider() or {}
            valence = float(tone.get('valence', 0.0))
        except Exception:
            return ''
        if valence >= TONE_BUCKET_I_THRESHOLD:
            return 'i'
        if valence <= TONE_BUCKET_M_THRESHOLD:
            return 'm'
        return 'n'

    def _predict_next(self, prev: str, bucket: str = '') -> str:
        """Predict the most likely successor of `prev` from
        accumulated transitions.  Empty string if unknown.

        Phase I.1c: when `bucket` is non-empty AND the
        tone-specific overlay has entries for (prev, bucket, *),
        use those.  Otherwise fall back to the global table.
        Soft gating — context refines but does not erase priors."""
        if bucket and self._tone_transitions:
            best = ('', 0.0)
            for (p, b, n), c in self._tone_transitions.items():
                if p == prev and b == bucket and c > best[1]:
                    best = (n, c)
            if best[1] >= TRANSITION_PREDICTION_THRESHOLD:
                return best[0]
        # Global fallback (or sole table when no tone wired).
        best = ('', 0.0)
        for (p, n), c in self._transitions.items():
            if p == prev and c > best[1]:
                best = (n, c)
        return best[0] if best[1] >= TRANSITION_PREDICTION_THRESHOLD \
            else ''

    def _apply_decay(self) -> None:
        """Phase I.1b: slow exponential decay across all
        transitions.  Entries that fall below TRANSITION_FLOOR
        are removed.  I.1c: decays the tone overlay table too."""
        factor = 1.0 - TRANSITION_DECAY_PER_BATCH
        decayed_anything = False
        if self._transitions:
            new: Dict[tuple, float] = {}
            pruned = 0
            for key, w in self._transitions.items():
                w2 = w * factor
                if w2 >= TRANSITION_FLOOR:
                    new[key] = w2
                else:
                    pruned += 1
            self._transitions = new
            self.transitions_pruned += pruned
            decayed_anything = True
        if self._tone_transitions:
            new_tone: Dict[tuple, float] = {}
            pruned_tone = 0
            for key, w in self._tone_transitions.items():
                w2 = w * factor
                if w2 >= TRANSITION_FLOOR:
                    new_tone[key] = w2
                else:
                    pruned_tone += 1
            self._tone_transitions = new_tone
            self.transitions_pruned += pruned_tone
            decayed_anything = True
        if decayed_anything:
            self.transitions_decayed_batches += 1

    def _weaken_transition(self,
                                 prev: str,
                                 predicted: str,
                                 bucket: str = '') -> None:
        """Phase I.1b/c: weaken the failed prediction.  If the
        prediction came from the tone overlay (bucket non-empty
        and the overlay has the entry), weaken there; otherwise
        weaken in the global table."""
        if bucket:
            tkey = (prev, bucket, predicted)
            if tkey in self._tone_transitions:
                w = self._tone_transitions[tkey] - \
                    TRANSITION_MISS_WEAKENING
                if w < TRANSITION_FLOOR:
                    del self._tone_transitions[tkey]
                    self.transitions_pruned += 1
                else:
                    self._tone_transitions[tkey] = w
                return
        key = (prev, predicted)
        if key not in self._transitions:
            return
        w = self._transitions[key] - TRANSITION_MISS_WEAKENING
        if w < TRANSITION_FLOOR:
            del self._transitions[key]
            self.transitions_pruned += 1
        else:
            self._transitions[key] = w

    def _fire_cognitive_pe(self,
                                  bus: EventBus,
                                  prev: str,
                                  focal: str,
                                  predicted: str) -> None:
        """Fire a cognitive PE.  Magnitude reflects how CONFIDENT
        the prediction was — a strongly-expected successor that
        misses is high-magnitude surprise; a barely-favored
        successor that misses is low-magnitude.  Sign stays -1
        (the miss is by definition unexpected for ValueLandscape /
        ACC consumers)."""
        cycle = self._cycle_provider()
        # Confidence in the prediction = P(predicted | prev) from
        # transition counts.  This is the "how confidently we
        # would have bet on predicted" number.
        total_prev = sum(
            c for (p, _), c in self._transitions.items()
            if p == prev)
        count_pred = self._transitions.get((prev, predicted), 0)
        magnitude = count_pred / total_prev if total_prev > 0 \
            else 0.0
        bus.publish(PredictionErrorEvent(
            kind=EventKind.PREDICTION_ERROR,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='cerebellum',
            origin='internal',
            origin_detail='lateral',
            focal=focal,
            predicted=0.0,    # categorical PE — kept zero
            actual=0.0,
            magnitude=magnitude,
            sign=-1.0,
            error_kind='cognitive',
            # Phase F.5: carry the prediction's context so the
            # substrate-side handler can fire falsified_i on the
            # right bubble + weaken the (prev → predicted) edge.
            prev_focal=prev,
            predicted_focal=predicted,
        ))
        self.cognitive_errors_fired += 1

    # ---- I.2: timing (per-transition delay tuning) ----

    def _maybe_fire_timing_pe(self,
                                       bus: EventBus,
                                       prev: str,
                                       focal: str,
                                       interval: int,
                                       ev: AttendedPerceptEvent
                                       ) -> None:
        """If the (prev, focal) transition has matured timing
        stats (>= TIMING_MIN_OBSERVATIONS) and the actual
        interval diverges past TIMING_PE_THRESHOLD (normalized
        by predicted mean), fire a timing PE."""
        stats = self._transition_timing.get((prev, focal))
        if stats is None or stats[0] < TIMING_MIN_OBSERVATIONS:
            return
        count, mean, _m2 = stats
        # Normalized magnitude: deviation as a fraction of
        # predicted interval.  max(mean, 1) keeps the divisor
        # sane for zero-interval predictions (parallel arrivals).
        magnitude = abs(interval - mean) / max(mean, 1.0)
        if magnitude < TIMING_PE_THRESHOLD:
            return
        bus.publish(PredictionErrorEvent(
            kind=EventKind.PREDICTION_ERROR,
            cycle=ev.cycle,
            timestamp=time.time(),
            source_capability='cerebellum',
            origin='internal',
            origin_detail='lateral',
            focal=focal,
            predicted=mean,
            actual=float(interval),
            magnitude=min(1.0, magnitude),
            sign=-1.0,    # both early and late are misses
            error_kind='timing',
            prev_focal=prev,
            predicted_focal=focal,
        ))
        self.timing_errors_fired += 1

    def _update_timing(self,
                              prev: str,
                              focal: str,
                              interval: int) -> None:
        """Welford's online update for (prev, focal) interval
        mean + sum-of-squared-deviations."""
        key = (prev, focal)
        stats = self._transition_timing.get(key)
        if stats is None:
            self._transition_timing[key] = [1, float(interval), 0.0]
            return
        stats[0] += 1
        n = stats[0]
        delta = interval - stats[1]
        stats[1] += delta / n
        stats[2] += delta * (interval - stats[1])

    def _on_thought(self, ev: ThoughtProducedEvent) -> None:
        """Cortical produced a thought.  Treat the (focal,
        target) as a soft transition prediction so future
        attended percepts on that target reinforce."""
        if ev.focal and ev.target:
            key = (ev.focal, ev.target)
            # Half-weight count — cortical inference is weaker
            # signal than observed transition.
            self._transitions[key] = self._transitions.get(
                key, 0.0) + 1.0

    # ---- vermis (affective) ----

    def _on_interoception(self,
                                ev: InteroceptionEvent,
                                bus: EventBus) -> None:
        """Predict where the agent's affective trajectory is
        heading; fire affective PE on divergence.

        Multi-channel path (Phase I.1a): when a chemistry
        provider is wired, predict each of the 8 channels
        separately.  Single-channel fallback path: predict
        lifeforce (the Phase 4b behavior)."""
        if self._chemistry_provider is not None:
            self._on_interoception_multichannel(ev, bus)
        else:
            self._on_interoception_lifeforce(ev, bus)

    def _on_interoception_multichannel(
            self,
            ev: InteroceptionEvent,
            bus: EventBus) -> None:
        """Phase I.1a path on InteroceptionEvent — defers to
        _sample_chemistry_for_vermis."""
        self._sample_chemistry_for_vermis(ev.cycle, bus)

    def _on_chemistry_fire(self,
                                    ev: ChemistryEvent,
                                    bus: EventBus) -> None:
        """Phase I.1a follow-up: chemistry events drive vermis
        sampling.  Skipped when no chemistry_provider is wired
        (Phase 4b lifeforce-only mode)."""
        if self._chemistry_provider is None:
            return
        self._sample_chemistry_for_vermis(ev.cycle, bus)

    def _sample_chemistry_for_vermis(
            self,
            cycle: int,
            bus: EventBus) -> None:
        """Phase I.1a: per-channel vermis.  For each channel in
        the current chemistry state, run a 2-sample linear
        predictor; fire affective PE for any channel whose
        actual diverges past its channel-specific threshold."""
        try:
            state = self._chemistry_provider() or {}
        except Exception:
            # Provider misbehaviour shouldn't crash the cerebellum.
            state = {}
        for channel, actual in state.items():
            history = self._channel_histories.setdefault(
                channel, deque(maxlen=CHANNEL_HISTORY_DEPTH))
            predicted = self._predict_channel(history)
            history.append(actual)
            if predicted is None:
                continue
            magnitude = actual - predicted
            threshold = CHANNEL_PE_THRESHOLDS.get(
                channel, AFFECTIVE_PE_THRESHOLD)
            if abs(magnitude) >= threshold:
                bus.publish(PredictionErrorEvent(
                    kind=EventKind.PREDICTION_ERROR,
                    cycle=cycle,
                    timestamp=time.time(),
                    source_capability='cerebellum',
                    origin='internal',
                    origin_detail='vermis',
                    focal='chem.' + channel,
                    predicted=predicted,
                    actual=actual,
                    magnitude=abs(magnitude),
                    sign=1.0 if magnitude > 0 else -1.0,
                    error_kind='affective',
                ))
                self.affective_errors_fired += 1
                self.channel_errors_fired[channel] = \
                    self.channel_errors_fired.get(channel, 0) + 1

    def _on_interoception_lifeforce(
            self,
            ev: InteroceptionEvent,
            bus: EventBus) -> None:
        """Single-channel fallback (Phase 4b behaviour)."""
        actual = ev.lifeforce
        predicted = self._predict_affective()
        self._affective_history.append(actual)
        if predicted is None:
            return
        magnitude = actual - predicted
        if abs(magnitude) >= AFFECTIVE_PE_THRESHOLD:
            bus.publish(PredictionErrorEvent(
                kind=EventKind.PREDICTION_ERROR,
                cycle=ev.cycle,
                timestamp=time.time(),
                source_capability='cerebellum',
                origin='internal',
                origin_detail='vermis',
                focal='self',
                predicted=predicted,
                actual=actual,
                magnitude=abs(magnitude),
                sign=1.0 if magnitude > 0 else -1.0,
                error_kind='affective',
            ))
            self.affective_errors_fired += 1

    @staticmethod
    def _predict_channel(
            history: Deque[float]) -> Optional[float]:
        """Linear extrapolation from last two samples on one
        channel's history.  None when not enough samples."""
        if len(history) < 2:
            return None
        last = history[-1]
        prev = history[-2]
        return last + (last - prev)

    def _predict_affective(self) -> Optional[float]:
        """Linear extrapolation from last two samples — simplest
        useful predictor.  None if too few samples."""
        if len(self._affective_history) < 2:
            return None
        last = self._affective_history[-1]
        prev = self._affective_history[-2]
        return last + (last - prev)

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'transitions_known': len(self._transitions),
            'tone_transitions_known': len(self._tone_transitions),
            'focal_history_size': len(self._focal_history),
            'cognitive_errors_fired': self.cognitive_errors_fired,
            'affective_errors_fired': self.affective_errors_fired,
            'channels_tracked': len(self._channel_histories),
            'channel_errors_fired': dict(self.channel_errors_fired),
            'transitions_decayed_batches':
                self.transitions_decayed_batches,
            'transitions_pruned': self.transitions_pruned,
            'timing_transitions_known':
                len(self._transition_timing),
            'timing_errors_fired': self.timing_errors_fired,
        }
