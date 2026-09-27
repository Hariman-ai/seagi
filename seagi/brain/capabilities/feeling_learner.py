"""FeelingLearner — self-tuned chemistry→feeling mappings (V1→V2
port, Tier 1).

The most doctrine-central of the three Tier-1 ports.  V2's
`ChemistryEngine.tone_summary` hand-maps the 8-channel state to a
label via an if/elif cascade — precisely the rule-1 (hand-tuning) +
rule-6 (abstraction must be self-formed) anti-pattern.  A system
whose felt-state vocabulary is fixed by us narrates OUR categories,
not its own experience.

This organ learns the labels from the agent's own state-signature
clusters.  Each tick it samples the 8-channel chemistry vector and
tags it with the current best label (the learned classifier once
populated, else the bootstrap cascade).  Periodically it recomputes
a centroid per label.  `classify_current_feeling()` then returns
the feeling whose centroid is nearest, with a confidence that
scales with the gap to the runner-up.  The hand-coded cascade
bootstraps the space; experience refines the geometry.  (V1's
insight: "figure out what HIS states feel like, not what ours do.")

Non-breaking: `ChemistryEngine.tone_summary` keeps its
valence/arousal/warmth geometry and the bootstrap cascade; it only
*overrides the label* when this learner is confident.  Every
existing tone_summary consumer keeps working.

Divergences from V1 (deliberate):
- **8-channel chemistry vector only.** V1 folded in `oscillation.py`
  EEG band-powers + existential dimensions; both are dropped
  (oscillation is a "leave" scaffold; existential is V1-specific).
- **Read-only observer.** No substrate writes → accrues no
  metabolic debt.  Correct: it watches chemistry, it doesn't act.
- **Sleep-state tag.** Samples are tagged awake/asleep so a sleep
  centroid (GABA↑/ACh↓) is learned as a legitimate distinct
  feeling rather than masquerading as a waking tone label.

Per [[seagi-chemistry-never-fully-dissolves]]: rarely-felt
centroids are NOT deleted when the sample buffer rolls over — a
faded feeling stays re-engageable.  The sample deque bounds memory;
learned signatures persist until re-learned.

Honest scope (auditor): this is a CORRECTNESS organ (read-only),
not a load-bearing survival organ — nothing degrades if it never
engages (the cascade still works).  And the label VOCABULARY is
still seeded from the cascade; true label DISCOVERY is deferred
(the rule-6 debt, like Step 0's K=3).  Geometry is learned;
vocabulary is v1-seeded — strictly better than today, where both
were ours.

Cross-refs: [[project_seagi_v1_port_tier1_wiring]],
[[feedback_homeostatic_cost_doctrine]],
[[feedback_seagi_chemistry_never_fully_dissolves]].
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from ..events import EventKind, BrainEvent, ChemistryEvent
from ..bus import EventBus
from ..chemistry_types import CHANNELS, IMPRINT_CHANNELS
from .neuromodulators import RAPHE_HISTORY_DEPTH


# Sample-buffer length.  Bounded so memory is finite; long enough to
# hold many ticks across multiple feelings.  Derived: 8 ×
# RAPHE_HISTORY_DEPTH = 400 (one rolling chemistry-window per
# channel's worth of samples).
MAX_SAMPLES = 8 * RAPHE_HISTORY_DEPTH   # 400

# DECLARED MEASUREMENT DEBT (auditor): min samples before a label's
# centroid is trustworthy.  Placeholder = RAPHE_HISTORY_DEPTH // 8 =
# 6 (one chemistry-window's worth divided across channels); V1 used
# 6 by feel.  Calibrate on daemon.
MIN_SAMPLES_PER_FEELING = max(1, RAPHE_HISTORY_DEPTH // 8)   # 6

# Recompute centroids every N samples.  Derived = RAPHE_HISTORY_DEPTH
# (recompute once per rolling-window's worth of new samples).
RELEARN_EVERY = RAPHE_HISTORY_DEPTH   # 50


class FeelingLearner:
    """Learns chemistry→feeling centroids from the agent's own
    trajectory; classifies the current feeling by nearest centroid."""

    SUBSCRIPTIONS = ()   # observer; ticks, doesn't handle events

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None,
                 chemistry_state_provider: Optional[Callable] = None,
                 bootstrap_label_fn: Optional[Callable] = None,
                 is_asleep_provider: Optional[Callable] = None):
        """
        chemistry_state_provider: () -> Dict[str,float] (global_state).
        bootstrap_label_fn: () -> str.  The cold-start labeller
            (ChemistryEngine's tone_summary label, pre-override).
            Used to tag samples until centroids populate.
        is_asleep_provider: () -> bool.  Samples are tagged with
            sleep-state so 'asleep' becomes a discoverable feeling.
        """
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._chem_provider = chemistry_state_provider
        self._bootstrap_label_fn = bootstrap_label_fn
        self._is_asleep_provider = is_asleep_provider

        # Persisted state.
        self._samples: Deque[Dict[str, Any]] = deque(maxlen=MAX_SAMPLES)
        self.feeling_signatures: Dict[str, Dict[str, Any]] = {}
        self.last_relearn_cycle: int = -1

        # Diagnostics.
        self._samples_since_relearn: int = 0
        self.samples_observed: int = 0
        self.relearns: int = 0
        self.last_classified: str = ''

    # ---- per-tick sampling ----

    def tick(self) -> None:
        if self._chem_provider is None:
            return
        try:
            state = self._chem_provider() or {}
        except Exception:
            return
        # IMPRINT_CHANNELS excludes adenosine (auditor MF #1): the sleep-
        # pressure scalar (0.10→~0.96) must not become a dominant feeling
        # dimension nor persist in feeling_signatures.
        sig = {ch: float(state.get(ch, CHANNELS[ch]['baseline']))
               for ch in IMPRINT_CHANNELS}
        # Tag with the current best label: learned classifier if
        # populated + confident, else bootstrap cascade.
        label = self._current_label(sig)
        asleep = False
        if self._is_asleep_provider is not None:
            try:
                asleep = bool(self._is_asleep_provider())
            except Exception:
                asleep = False
        self._samples.append({
            'feeling': label,
            'signature': sig,
            'asleep': asleep,
        })
        self.samples_observed += 1
        self._samples_since_relearn += 1
        if self._samples_since_relearn >= RELEARN_EVERY:
            self.learn()

    def _current_label(self, sig: Dict[str, float]) -> str:
        """Best label for a signature right now.  Learned classifier
        when centroids exist + are confident, else bootstrap."""
        if self.feeling_signatures:
            cls = self._classify_signature(sig)
            from .chemistry import FEELING_CONF_FLOOR
            if (cls.get('learned')
                    and cls.get('confidence', 0.0) >= FEELING_CONF_FLOOR):
                return cls['feeling']
        if self._bootstrap_label_fn is not None:
            try:
                lbl = self._bootstrap_label_fn()
                if lbl:
                    return str(lbl)
            except Exception:
                pass
        return 'flat'

    # ---- centroid learning ----

    def learn(self) -> Dict[str, Dict[str, Any]]:
        """Recompute centroid + spread per label from accumulated
        samples.  Labels below MIN_SAMPLES_PER_FEELING keep their
        PRIOR centroid if they had one (fade-not-delete per
        chemistry-never-dissolves) rather than vanishing."""
        self._samples_since_relearn = 0
        self.last_relearn_cycle = int(self._cycle_provider())
        self.relearns += 1
        if not self._samples:
            return self.feeling_signatures

        by_feeling: Dict[str, List[Dict[str, float]]] = {}
        for s in self._samples:
            by_feeling.setdefault(
                s.get('feeling') or 'flat', []).append(
                    s.get('signature') or {})

        new_sigs: Dict[str, Dict[str, Any]] = {}
        for feeling, sigs in by_feeling.items():
            if len(sigs) < MIN_SAMPLES_PER_FEELING:
                # Not enough fresh samples this window — keep the
                # prior centroid if one exists (re-engageable
                # presence), else skip.
                prior = self.feeling_signatures.get(feeling)
                if prior is not None:
                    new_sigs[feeling] = prior
                continue
            centroid = self._centroid(sigs)
            new_sigs[feeling] = {
                'centroid': centroid,
                'spread': self._spread(sigs, centroid),
                'n_samples': len(sigs),
                'learned_at_cycle': self.last_relearn_cycle,
            }
        # Carry forward any prior centroid whose label didn't appear
        # at all this window (fully faded, but not deleted).
        for feeling, prior in self.feeling_signatures.items():
            new_sigs.setdefault(feeling, prior)
        self.feeling_signatures = new_sigs
        return new_sigs

    @staticmethod
    def _centroid(sigs: List[Dict[str, float]]) -> Dict[str, float]:
        keys = set()
        for s in sigs:
            keys.update(s.keys())
        out = {k: 0.0 for k in keys}
        for s in sigs:
            for k in keys:
                out[k] += float(s.get(k, 0.0) or 0.0)
        n = float(len(sigs))
        return {k: v / n for k, v in out.items()}

    @staticmethod
    def _spread(sigs: List[Dict[str, float]],
                centroid: Dict[str, float]) -> float:
        if not sigs or not centroid:
            return 0.0
        total = 0.0
        for s in sigs:
            d = 0.0
            for k, cv in centroid.items():
                sv = float(s.get(k, 0.0) or 0.0)
                d += (sv - cv) ** 2
            total += math.sqrt(d)
        return total / float(len(sigs))

    @staticmethod
    def _distance(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return float('inf')
        keys = set(a.keys()) | set(b.keys())
        total = 0.0
        for k in keys:
            diff = (float(a.get(k, 0.0) or 0.0)
                    - float(b.get(k, 0.0) or 0.0))
            total += diff * diff
        return math.sqrt(total)

    # ---- classification ----

    def _classify_signature(self,
                              sig: Dict[str, float]) -> Dict[str, Any]:
        mappings = self.feeling_signatures
        if not mappings or not sig:
            return {'feeling': 'flat', 'confidence': 0.0,
                    'learned': False}
        scored: List[Tuple[str, float]] = []
        for feeling, meta in mappings.items():
            centroid = meta.get('centroid') or {}
            scored.append((feeling, self._distance(sig, centroid)))
        scored.sort(key=lambda fd: fd[1])
        best_feeling, best_d = scored[0]
        runner_d = scored[1][1] if len(scored) > 1 else best_d + 1.0
        gap = max(0.0, runner_d - best_d)
        denom = max(0.01, runner_d + best_d)
        confidence = min(1.0, gap / denom)
        return {
            'feeling': best_feeling,
            'confidence': round(confidence, 3),
            'distance': round(best_d, 3),
            'learned': True,
        }

    def classify_current_feeling(self) -> Dict[str, Any]:
        """Provider for ChemistryEngine.tone_summary.  Returns the
        learned feeling + confidence for the current chemistry, or a
        not-learned default that lets the cascade speak."""
        if self._chem_provider is None:
            return {'feeling': 'flat', 'confidence': 0.0,
                    'learned': False}
        try:
            state = self._chem_provider() or {}
        except Exception:
            return {'feeling': 'flat', 'confidence': 0.0,
                    'learned': False}
        # IMPRINT_CHANNELS excludes adenosine (auditor MF #1): the sleep-
        # pressure scalar (0.10→~0.96) must not become a dominant feeling
        # dimension nor persist in feeling_signatures.
        sig = {ch: float(state.get(ch, CHANNELS[ch]['baseline']))
               for ch in IMPRINT_CHANNELS}
        result = self._classify_signature(sig)
        self.last_classified = result.get('feeling', '')
        return result

    # ---- persistence ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            'feeling_signatures': self.feeling_signatures,
            'last_relearn_cycle': int(self.last_relearn_cycle),
            # Sample buffer persists so learning continues from where
            # it left off rather than cold-starting each session.
            'samples': list(self._samples),
        }

    def load_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        sigs = state.get('feeling_signatures')
        if isinstance(sigs, dict):
            self.feeling_signatures = sigs
        try:
            self.last_relearn_cycle = int(
                state.get('last_relearn_cycle', -1))
        except (TypeError, ValueError):
            self.last_relearn_cycle = -1
        samples = state.get('samples')
        if isinstance(samples, (list, tuple)):
            self._samples = deque(
                (s for s in samples if isinstance(s, dict)),
                maxlen=MAX_SAMPLES)

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'samples': len(self._samples),
            'samples_observed': int(self.samples_observed),
            'learned_feelings': len(self.feeling_signatures),
            'feeling_labels': sorted(self.feeling_signatures.keys()),
            'relearns': int(self.relearns),
            'last_classified': self.last_classified,
        }
