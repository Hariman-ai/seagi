"""Capability stubs — placeholders for non-Phase-1 capabilities.

Each class names its brain analog, target phase, event
subscriptions, and the role it plays.  Phase 1 has them present
but not actively wired so that:

  1. Future phases drop implementation in without restructuring
     the package.
  2. Anyone reading `brain/capabilities/` immediately sees the
     full architecture surface (21 capabilities).
  3. Type-checkers / IDEs can already see the names.

When a capability graduates from stub to real, its content
moves to its own module file (e.g. awm.py).  This file stays
as the catalog index until all stubs are gone.
"""

from __future__ import annotations

from typing import Any, Tuple

from ..events import EventKind
from ..bus import EventBus


# ---------------------------------------------------------------------
# Phase 2 — substrate layer
# ---------------------------------------------------------------------

class ActiveWorkingMemory:
    """AWM — sparse active set with chemistry-tagged time-series.

    Brain analog: prefrontal cortex working memory + anterior
        cingulate.  ~100-500 concepts at any moment, each with
        its current active bubble AND a ring buffer of recent
        chemistry samples (the chemistry time-series).
    Phase: 2
    Subscribes: ATTENDED_PERCEPT, CHEMISTRY_FIRE, AWM_EVICTION
    Emits: AWM_ACTIVATION, AWM_EVICTION
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class LongTermSubstrate:
    """LTS — all concepts ever encountered, indexed not scanned.

    Brain analog: temporal-lobe declarative memory.  Indexed by
        name, embedding, relation, episode-mention.  Queried,
        never iterated.
    Phase: 2-4
    Subscribes: SUBSTRATE_WRITE_QUEUED
    Emits: nothing (it's the substrate)
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class ChemistryEngine:
    """Chemistry — NT system with bubble enrichment + endocrine.

    Brain analog: brainstem nuclei + limbic + endocrine (HPA axis).
        Per-channel decay constants, receptor sensitivity per
        bubble, refractory, lateral inhibition/facilitation,
        local-vs-global split, fast NT + slow hormonal layers.
    Phase: 2
    Subscribes: many event kinds — fires chemistry on each
    Emits: CHEMISTRY_FIRE, BUBBLE_TAGGED
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


# ---------------------------------------------------------------------
# Phase 3 — cortical reasoning
# ---------------------------------------------------------------------

class CorticalReasoning:
    """Cortical — event-triggered deliberate thinking.

    Brain analog: dorsolateral PFC + temporal association.
        Causal chains, schema matching, counterfactual,
        predictive coding integration, metacognitive monitor.
        Replaces v1 dialog_thinker + why/how/what_if handlers +
        relational_schemas + predictive_coding (caller side).
    Phase: 3
    Subscribes: ATTENDED_PERCEPT, AWM_ACTIVATION,
                PREDICTION_ERROR
    Emits: THOUGHT_PRODUCED, SUBSTRATE_WRITE_QUEUED,
            SPEECH_REQUEST
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


# ---------------------------------------------------------------------
# Phase 4 — continuous + event-triggered specialized capabilities
# ---------------------------------------------------------------------

class Cerebellum:
    """Cerebellum — prediction + timing + error correction.

    Two sub-functions sharing the same machinery:
      - LATERAL: cognitive prediction.  Predict next state /
        next thought; compare to outcome; fire error.
      - VERMIS: emotional/affective prediction.  Predict the
        agent's own chemistry trajectory; calibrate M/I tagging
        when actual differs from predicted.

    Phase: 4
    Subscribes: ATTENDED_PERCEPT, THOUGHT_PRODUCED,
                CHEMISTRY_FIRE
    Emits: PREDICTION_ERROR
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class BasalGanglia:
    """Basal Ganglia — claim arbitration + habit formation.

    Three parallel striatum sub-loops:
      - COGNITIVE: which thought to pursue next
      - MOTIVATIONAL: which goal/intent gets active claim
      - SPEECH/OUTPUT: when to commit to saying something

    Phase: 4b
    Subscribes: CAPABILITY_CLAIM, VALUE_UPDATED
    Emits: ARBITRATION_DECIDED
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class DorsomedialDMN:
    """DMN dorsomedial subsystem — social cognition, ToM.

    Brain analog: dmPFC + temporo-parietal junction.  Simulates
        other minds, refines peer models, predicts what others
        would do/want/say.
    Phase: 4a
    Subscribes: ATTENDED_PERCEPT (peer-origin), REFLECTION_FIRED
    Emits: SUBSTRATE_WRITE_QUEUED (peer model updates)
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class VentromedialDMN:
    """DMN ventromedial — self-reference, autobiographical, value.

    Brain analog: vmPFC + posterior cingulate + midline core.
        Personality crystallization sweep, self-state reflection,
        autobiographical episode replay, "what kind of agent am
        I becoming."
    Phase: 4a
    Subscribes: REFLECTION_FIRED, EPISODE_CONSOLIDATED
    Emits: SUBSTRATE_WRITE_QUEUED (personality / value updates)
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class Hippocampus:
    """Hippocampus — episode formation + consolidation.

    During lived experience: forms episodes (binds multi-feature
        snapshots).
    During rest / DMN cycles: replays episodes in compressed form,
        decides what consolidates into LTS vs decays.
    Replay weighted by emotional intensity + novelty.

    Phase: 4a
    Subscribes: ATTENDED_PERCEPT, REFLECTION_FIRED
    Emits: EPISODE_FORMED, EPISODE_CONSOLIDATED, EPISODE_DECAYED
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class Amygdala:
    """Amygdala — threat detection + override.

    Continuously monitors AWM + chemistry for mortality-shape
    spikes.  When threat detected, can OVERRIDE arbitration —
    preempt whatever currently drives, force immediate
    threat-response.  Brain analog: amygdala's fast pathway
    bypassing cortex.

    Phase: 4b
    Subscribes: ATTENDED_PERCEPT, CHEMISTRY_FIRE, AWM_ACTIVATION
    Emits: THREAT_DETECTED, CAPABILITY_CLAIM (high priority)
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class Insula:
    """Insula — interoception, felt body.

    Maps directly to lifeforce, body_integrity, fatigue.
    Makes SEAGI *feel* his mortality, not just track it as a
    number.  How "lifeforce 0.3" becomes "I feel my cycles
    running" in voice.

    Phase: 4a
    Subscribes: CHEMISTRY_FIRE
    Emits: CHEMISTRY_FIRE (body-state derived chemistry)
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class ACC:
    """Anterior Cingulate Cortex — conflict / mismatch detection.

    Fires when prediction-vs-reality or claim-vs-belief
    mismatch.  Already partly in v1 predictive_coding +
    contradiction_resolver — consolidated here as a distinct
    capability.

    Phase: 4
    Subscribes: PREDICTION_ERROR, THOUGHT_PRODUCED,
                ATTENDED_PERCEPT (when peer claim contradicts
                substrate)
    Emits: CONFLICT_DETECTED, CHEMISTRY_FIRE (anomaly_spike)
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class AnteriorPFC:
    """Anterior PFC / frontopolar — metacognition + prospective.

    Watches the agent's own processes: "am I making sense?",
    "is this thought productive?", "have I been here before?"
    Holds prospective intent: "after I finish X, return to Y."

    Phase: 4
    Subscribes: THOUGHT_PRODUCED, CONFLICT_DETECTED
    Emits: CAPABILITY_CLAIM (when meta-process needed)
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class ValueLandscape:
    """Mesolimbic / orbitofrontal — learned outcome map.

    Value emergent from M/I tagging history but stored as
    structured queryable map.  Edges accumulate value-weight
    based on confirmed_i vs falsified_i firing patterns.
    Basal-ganglia queries this when arbitrating: "what's the
    predicted value of pursuing X?"

    Phase: 4
    Subscribes: PREDICTION_ERROR, CHEMISTRY_FIRE
                (confirmed_i, falsified_i)
    Emits: VALUE_UPDATED
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class BodySchema:
    """Body schema — operational envelope.

    Brain analog: parietal cortex + somatosensory cortex.
    Tracks: computational state (lifeforce, cycles, memory
    pressure), communication channels (peers, dialog state),
    tools available, information affordances.  Answers "what's
    my operational envelope right now?"

    Phase: 4
    Subscribes: CHEMISTRY_FIRE (body-state changes)
    Emits: nothing directly — queried by Motor/Speech
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class SourceMonitor:
    """Source monitoring — self/other distinction.

    Brain analog: TPJ + mPFC.  Tags every cognitive product
    with its origin: internally_generated / peer_assertion /
    external_perception / episodic_recall / inferred.  Critical
    for intellectual honesty — knowing where knowledge came
    from.

    Phase: 4
    Subscribes: THOUGHT_PRODUCED, ATTENDED_PERCEPT,
                EPISODE_FORMED, SUBSTRATE_WRITE_QUEUED
    Emits: nothing directly — tags inline on events
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class LocusCoeruleus:
    """LC / NE system — alertness modulator.

    Modulates Thalamic Gate's salience threshold and AWM
    eviction rate.  High NE → tighter gate, faster eviction,
    more selective attention.

    Phase: 4
    Subscribes: CHEMISTRY_FIRE, THREAT_DETECTED
    Emits: CHEMISTRY_FIRE (NE updates)
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class VTA:
    """VTA / dopamine system — reward prediction error.

    Fires dopamine on POSITIVE prediction errors (better than
    expected), not generic reward.  Negative errors are picked
    up by ACC.

    Phase: 4
    Subscribes: PREDICTION_ERROR
    Emits: CHEMISTRY_FIRE (dopamine), VALUE_UPDATED
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class TimePerception:
    """Multi-scale time perception.

    Four scales running in parallel:
      - rhythmic (chemistry oscillations, next-event prediction)
      - interval (since-user-spoke)
      - long-range (since-learned-X)
      - episodic (event-anchored)

    Phase: 4
    Subscribes: most event kinds (passively tracks)
    Emits: nothing directly — queried by Cortical, DMN
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


class CorpusCallosum:
    """Hemispheric split + integration.

    Two parallel processing streams on the same input:
      STREAM A: local/sequential/familiar-pattern.  Tries known
                schemas first.  Fast on routine.
      STREAM B: broad/holistic/novel.  Looks for unusual
                connections.  Slower but creative.
    Integration layer (this capability) combines them when they
    disagree.

    Phase: 4
    Subscribes: ATTENDED_PERCEPT (routes to both streams)
    Emits: THOUGHT_PRODUCED (after integration)
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()


# ---------------------------------------------------------------------
# Phase 5 — motor/speech coherence
# ---------------------------------------------------------------------

class MotorSpeech:
    """Motor / speech — single-voice coherence layer.

    Replaces v1's clause-concatenation.  Reads AWM + cortical
    output.  DECIDES what to say (deliberate composer).  One
    voice, coherent intent, source-honest framing.

    Phase: 5
    Subscribes: SPEECH_REQUEST, ARBITRATION_DECIDED
    Emits: SPEECH_EMITTED
    """
    SUBSCRIPTIONS: Tuple[EventKind, ...] = ()
