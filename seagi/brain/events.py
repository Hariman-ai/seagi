"""Event types — the universal communication between capabilities.

In v2, capabilities don't call each other directly.  They publish
events to a shared bus; interested capabilities subscribe.  This
is what makes the architecture genuinely parallel — no
capability blocks on another's completion.

Brain analog: action potentials propagating through interconnected
populations.  The cerebellum doesn't "call" the cortex; it fires,
and any neuron tuned to that signal responds.

Event-type design principles
----------------------------
- Every event is immutable after construction (dataclass(frozen=...))
- Every event carries a timestamp + cycle (when it fired)
- Every event carries source-monitoring info: which capability
  emitted it
- Events flow ONE-WAY: capability A → bus → capability B.  If B
  needs to respond, B emits a NEW event.  No request/response
  pairs; no synchronous return paths.

In Phase 1 we only need a few event types.  Later phases add
more.  All events inherit from BrainEvent so the bus can
dispatch uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EventKind(str, Enum):
    """Closed enum of event types.  New types added as new
    capabilities come online.  String values for cheap logging."""

    # Phase 1 — sensory + gating
    RAW_PERCEPT = 'raw_percept'
    ATTENDED_PERCEPT = 'attended_percept'
    PERCEPT_DISCARDED = 'percept_discarded'

    # Phase 2 — AWM + chemistry
    AWM_ACTIVATION = 'awm_activation'
    AWM_EVICTION = 'awm_eviction'
    CHEMISTRY_FIRE = 'chemistry_fire'
    BUBBLE_TAGGED = 'bubble_tagged'

    # Phase 3 — cortical reasoning
    THOUGHT_PRODUCED = 'thought_produced'
    SUBSTRATE_WRITE_QUEUED = 'substrate_write_queued'

    # Phase 4 — DMN, hippocampus, amygdala, etc.
    EPISODE_FORMED = 'episode_formed'
    EPISODE_CONSOLIDATED = 'episode_consolidated'
    EPISODE_DECAYED = 'episode_decayed'
    REFLECTION_FIRED = 'reflection_fired'
    INTEROCEPTION = 'interoception'
    THREAT_DETECTED = 'threat_detected'
    PREDICTION_ERROR = 'prediction_error'
    CONFLICT_DETECTED = 'conflict_detected'
    VALUE_UPDATED = 'value_updated'

    # Phase 4b — arbitration
    CAPABILITY_CLAIM = 'capability_claim'
    ARBITRATION_DECIDED = 'arbitration_decided'

    # Phase 5 — motor/speech
    SPEECH_REQUEST = 'speech_request'
    SPEECH_EMITTED = 'speech_emitted'


@dataclass(frozen=True)
class BrainEvent:
    """Base for all events.  Carries timing + source-monitoring
    metadata.  Subclasses add payload."""
    kind: EventKind
    cycle: int                  # engine cycle when emitted
    timestamp: float = 0.0      # wall-clock time (optional)
    source_capability: str = '' # which capability emitted this
    # Source-monitoring tags travel WITH every event.  This is
    # the architectural mechanism for self/other distinction —
    # downstream consumers can always trace where info came from.
    origin: str = ''            # 'internal' / 'peer' / 'forager' / 'sensor'
    origin_detail: str = ''     # peer_id / rss_url / etc

    def short(self) -> str:
        """One-line summary for logs."""
        return (f'{self.kind.value} @{self.cycle} '
                  f'from={self.source_capability} '
                  f'orig={self.origin}/{self.origin_detail}')


@dataclass(frozen=True)
class RawPerceptEvent(BrainEvent):
    """Raw input decoded by Sensory intake, BEFORE thalamic
    gating.  Most will be discarded; the salient few become
    AttendedPerceptEvent.

    payload: the decoded representation.  Today: dict of
        concept_name → MIValue.  Future: image embeddings,
        audio features, proprioceptive signals — all reduced
        to the same concept-activation shape after decoding.
    raw_text: original input text (when modality is 'text').
        Optional; used for episode content + source tracing.
    modality: 'text' / 'image' / 'audio' / 'proprioception'
    """
    payload: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ''
    modality: str = 'text'


@dataclass(frozen=True)
class AttendedPerceptEvent(BrainEvent):
    """Percept that the Thalamic Gate let through.  Carries
    salience score + the components that produced it (for
    downstream diagnostic + chemistry firing).

    focals: list of concept names that survived filtering
    salience: 0..1 combined score
    novelty / m_content / i_content: components
    threshold_used: the gate's threshold at decision time
                     (chemistry-modulated)
    """
    focals: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ''
    modality: str = 'text'
    salience: float = 0.0
    novelty: float = 0.0
    m_content: float = 0.0
    i_content: float = 0.0
    threshold_used: float = 0.0


@dataclass(frozen=True)
class PerceptDiscardedEvent(BrainEvent):
    """Percept the Thalamic Gate filtered out.  Emitted so
    diagnostic tools can audit what's being suppressed, AND
    so the gate's behavior can be inspected.  Carries the
    same component scores as AttendedPerceptEvent for
    symmetry."""
    modality: str = 'text'
    salience: float = 0.0
    novelty: float = 0.0
    m_content: float = 0.0
    i_content: float = 0.0
    threshold_used: float = 0.0
    n_focals: int = 0
    # Truncated to keep the event small; full payload was
    # discarded anyway.
    raw_text_excerpt: str = ''


@dataclass(frozen=True)
class ChemistryEvent(BrainEvent):
    """Chemistry capability emitting a transmitter event.
    Used by every other capability to signal felt-state shifts.
    Phase 1 stub — full semantics arrive in Phase 2."""
    chemistry_kind: str = ''    # 'curiosity' / 'anomaly_spike' / ...
    magnitude: float = 0.0
    target_concepts: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ThoughtProducedEvent(BrainEvent):
    """Cortical reasoning produced a thought about a focal.

    Phase 3 deliverable.  Carries the (focal, relation, target)
    triple the reasoner committed to, plus the method that
    produced it (causal / schema / recall / counterfactual /
    metacognitive) and a confidence in [0, 1].

    `text` is a rendered first-person prose summary the
    Motor/Speech composer (Phase 5) can use directly.  Until
    then, a Phase 3 stub composer reads `text` as the response.
    """
    focal: str = ''
    relation: str = ''
    target: str = ''
    confidence: float = 0.0
    method: str = ''         # 'causal' / 'schema' / 'recall' /
                              # 'counterfactual' / 'metacog' /
                              # 'inference'
    text: str = ''
    thin_substrate: bool = False  # metacognitive: did we have
                                    # enough substrate to think well?
    # Step 0 organ 4a (2026-05-27): R.1 inference chain depth in
    # hops.  Non-inference paths leave this at 0.  AWM contraction
    # tracks (chain_depth + 1) = the node count of the strongest
    # inference walks recently exercised — sets the floor below
    # which sleep-deprivation cannot collapse working memory.
    chain_depth: int = 0
    # Phase 2 per-hop reinforcement (2026-06-01): the REAL (s, r, o)
    # edges the reasoner actually traversed to produce this thought.
    # A multi-hop causal/inference walk collapses to a single
    # (focal, relation, target) above, but the corroborating
    # evidence is the individual hops.  The ReasoningConsolidator
    # reinforces these walked hops (not the fabricated collapsed
    # triple) so recall builds the A->X->B structure that
    # reinforce_coherent_edges and _inference_chain require.  Empty
    # for thoughts with no edge walk; defaults preserve back-compat.
    # See project_seagi_dry_reverie_igniter.
    edges_walked: Tuple[Tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class SpeechRequestEvent(BrainEvent):
    """Cortical asks Motor/Speech to compose a response.  Phase 5
    Motor/Speech consumes this; Phase 3 stub composer responds
    directly inline.

    `thin_substrate` carries the cortical metacognitive verdict
    directly — Motor/Speech reads the boolean to pick the
    `honest_uncertain` voice mode.  This is the architectural
    contract; we do NOT substring-match the rendered text.
    """
    intended_focal: str = ''
    thought_text: str = ''   # the cortical output to ground speech
    thin_substrate: bool = False
    awm_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpeechEmittedEvent(BrainEvent):
    """Motor/Speech produced a finalized response.  Phase 5.

    Other capabilities (hippocampus / source monitor / time
    perception) react to this so the brain knows what IT just
    said — important for self-monitoring and avoiding
    repetitive output.
    """
    intended_focal: str = ''
    voice_mode: str = ''     # 'direct' / 'reflective' /
                              # 'honest_uncertain' /
                              # 'body_aware' / 'minimal'
    text: str = ''
    sources_used: List[str] = field(default_factory=list)
                              # ['cortical', 'vmdmn', 'insula', ...]
    body_band: str = ''


@dataclass(frozen=True)
class SubstrateWriteQueuedEvent(BrainEvent):
    """A capability requests a substrate write.  Phase 4 will
    have a single writer thread apply these from a journal;
    Phase 3 applies them inline via LTS.write_edge() helper."""
    subject: str = ''
    relation: str = ''
    object: str = ''
    strength: float = 0.4
    write_reason: str = ''   # 'cortical_inference' / 'schema_match' / ...
    mi_m: float = 0.0        # grounding valence M-lean (Stage 2)
    mi_i: float = 0.0        # grounding valence I-lean (Stage 2)
    outcome: str = ''        # '' | 'hit' — non-empty -> evidence.record in the writer


# ---------------------------------------------------------------
# Phase 4a — episodes + reflection
# ---------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeFormedEvent(BrainEvent):
    """Hippocampus bound a new episode from an attended percept.

    An episode is a multi-feature snapshot: which concepts were
    in focus, what was said, what the agent felt at the time.
    Recent episodes live in hippocampal buffer; reflection
    decides which graduate to LTS vs decay.
    """
    episode_id: int = 0
    focals: List[str] = field(default_factory=list)
    raw_text: str = ''
    salience: float = 0.0
    m_polarity: float = 0.0
    i_polarity: float = 0.0
    novelty: float = 0.0


@dataclass(frozen=True)
class EpisodeConsolidatedEvent(BrainEvent):
    """Hippocampus picked this episode during a reflection pass
    and wrote its content into long-term substrate (edges +
    crystallization bumps)."""
    episode_id: int = 0
    focals: List[str] = field(default_factory=list)
    n_edges_strengthened: int = 0
    replay_weight: float = 0.0


@dataclass(frozen=True)
class EpisodeDecayedEvent(BrainEvent):
    """An episode failed reflection's bar and is being dropped
    from the hippocampal buffer.  The substrate keeps anything
    already consolidated — only the episodic snapshot decays."""
    episode_id: int = 0
    age_cycles: int = 0


@dataclass(frozen=True)
class ReflectionFiredEvent(BrainEvent):
    """Idle driver (or chemistry spike) requests a reflection
    pass.  DMN sub-networks + Hippocampus consume this.

    trigger:
        'idle_timer'      — N cycles elapsed without peer input
        'chemistry_spike' — strong global NE/dopamine deflection
        'manual'          — test hook
    reflection_kind:
        'autobiographical' — vmDMN replays self
        'social'           — dmDMN updates peer models
        'general'          — both
    """
    trigger: str = 'idle_timer'
    reflection_kind: str = 'general'


@dataclass(frozen=True)
class InteroceptionEvent(BrainEvent):
    """Insula reports a body-state shift.  This is the felt
    layer over lifeforce / body integrity / fatigue — the
    mechanism that turns 'lifeforce=0.3' into 'I feel my
    cycles running.'  Consumed by Cortical (frames thoughts)
    and Motor/Speech (frames voice)."""
    felt_state: str = ''       # 'depleted' / 'settled' / 'agitated' / 'replenished'
    lifeforce: float = 0.5
    body_integrity: float = 1.0
    delta_lifeforce: float = 0.0   # since last interoceptive sample
    narrative: str = ''            # 'I feel my cycles running'
    # Set True when the delta is caused by the world / a game-death, so the
    # reward ledger refuses to turn it into action credit (no RL on survival).
    world_caused: bool = False


# ---------------------------------------------------------------
# Phase 4b — threat / arbitration / value / prediction
# ---------------------------------------------------------------


@dataclass(frozen=True)
class PredictionErrorEvent(BrainEvent):
    """Cerebellum (or any predictor) fires when a prediction
    diverged from outcome.  Sign is signed magnitude:
      positive = better than predicted (drives VTA dopamine)
      negative = worse than predicted (drives ACC/cortisol)
    `kind` distinguishes cognitive (lateral cerebellum) from
    affective (vermis) prediction errors.

    Phase F.5 (2026-05-16): `prev_focal` and `predicted_focal`
    carry the predictor's context — the focal whose prediction
    failed (prev_focal) and the concept it incorrectly anticipated
    (predicted_focal).  These let downstream consumers update
    substrate (weaken the bad edge, fire falsified_i on the
    failing bubble).  Empty strings for affective errors where
    the surprise is on a scalar (lifeforce), not a concept."""
    focal: str = ''
    predicted: float = 0.0
    actual: float = 0.0
    magnitude: float = 0.0   # |actual - predicted|, signed in `sign`
    sign: float = 0.0        # +1 / -1 / 0
    error_kind: str = ''     # 'cognitive' / 'affective'
    prev_focal: str = ''     # the source of the failed prediction
    predicted_focal: str = ''  # what was anticipated (and didn't arrive)


@dataclass(frozen=True)
class ThreatDetectedEvent(BrainEvent):
    """Amygdala raised a threat alarm.  Always carries enough
    info for downstream (LC, BG, cortical) to react without
    rescanning:
      threat_kind: 'body' (interoception) / 'percept' (high-M
                    peer/sensor input) / 'chemistry' (cortisol
                    spike)
      magnitude:   0..1 — drives override priority
      focal:       focal concept implicated (may be '' for
                    body-origin)
    """
    threat_kind: str = ''
    magnitude: float = 0.0
    focal: str = ''
    narrative: str = ''


@dataclass(frozen=True)
class ConflictDetectedEvent(BrainEvent):
    """ACC detected a mismatch.  Either a prediction-error
    that exceeded threshold, OR a peer claim contradicting
    substrate, OR a cortical thought conflicting with another
    recent cortical thought."""
    conflict_kind: str = ''      # 'prediction' / 'peer_contradiction'
                                  # / 'thought_thought'
    magnitude: float = 0.0
    focals: List[str] = field(default_factory=list)
    narrative: str = ''


@dataclass(frozen=True)
class ValueUpdatedEvent(BrainEvent):
    """Value landscape updated for a focal.  Drives BG
    arbitration: claims tied to high-value focals win ties.
    `delta` is the change applied this update; `value` is the
    new accumulated value in [-1, 1]."""
    focal: str = ''
    delta: float = 0.0
    value: float = 0.0
    update_kind: str = ''   # 'positive_pe' / 'negative_pe' / 'chemistry'


@dataclass(frozen=True)
class ArbitrationDecidedEvent(BrainEvent):
    """Basal Ganglia chose a winning claim.  Carries the
    proposed_action of the winner + the loop that decided it
    (cognitive / motivational / speech).  Other capabilities
    (motor, cortical) listen for this to know what to act on."""
    loop: str = ''                  # 'cognitive' / 'motivational' / 'speech'
    winning_action: str = ''
    winning_capability: str = ''
    winning_strength: float = 0.0
    n_competing: int = 0


@dataclass(frozen=True)
class CapabilityClaimEvent(BrainEvent):
    """A capability is asserting it should drive the next moment.
    Picked up by basal-ganglia arbitration.  Phase 4b semantics.

    `loop` selects which BG sub-loop the claim competes in:
        'cognitive'     — next thought to pursue
        'motivational'  — next goal/intent
        'speech'        — whether to commit to saying something
    `proposed_action` is a short stable identifier
    (capability-specific) describing what would happen if this
    claim wins.  Optional `payload` carries richer detail."""
    claim_strength: float = 0.0
    proposed_action: str = ''
    loop: str = 'cognitive'
    payload: Dict[str, Any] = field(default_factory=dict)
