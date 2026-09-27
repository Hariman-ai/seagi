"""SEAGI v2 — brain-correct architecture.

Interconnected functional capabilities running in parallel, not
clean modules called in sequence.  M/I NT tagging under
mortality is the load-bearing vision.  Each capability has a
brain analog (sensory cortex, thalamus, hippocampus, etc.) but
the architecture is holistic — every cognitive act recruits
many capabilities simultaneously.

See `project_seagi_v2_architecture.md` for the foundational
design.  Read that before working in this package.

Phase status (current: Phase 5 — v2 feature-complete)
------------------------------------------------------
All 22 capabilities live and wired:

    Phase 1:  Sensory, ThalamicGate, EventBus
    Phase 2:  AWM (sparse hot set), ChemistryEngine, LTS query API
    Phase 3:  CorticalReasoner (causal / schema / counterfactual /
               metacognitive — sparse over AWM, no LTS scans)
    Phase 4a: Insula, Hippocampus, dmDMN, vmDMN, idle reflection driver
    Phase 4b: Cerebellum (lateral + vermis), ACC, VTA, LC,
               ValueLandscape, Amygdala, BasalGanglia, SourceMonitor,
               AnteriorPFC, TimePerception, BodySchema,
               CorpusCallosum, JournaledSubstrateWriter
    Phase 5:  MotorSpeech (deliberate single-voice composer),
               LTS scale-measurement harness

Bridge-to-live: replace serve.py chat handler with brain.chat().
v1 substrate stays as the LTS backing store until a measured
SQLite/DuckDB swap is justified.
"""

from .events import (
    EventKind,
    BrainEvent,
    RawPerceptEvent,
    AttendedPerceptEvent,
    PerceptDiscardedEvent,
    ChemistryEvent,
    CapabilityClaimEvent,
    ThoughtProducedEvent,
    SpeechRequestEvent,
    SubstrateWriteQueuedEvent,
    EpisodeFormedEvent,
    EpisodeConsolidatedEvent,
    EpisodeDecayedEvent,
    ReflectionFiredEvent,
    InteroceptionEvent,
    PredictionErrorEvent,
    ThreatDetectedEvent,
    ConflictDetectedEvent,
    ValueUpdatedEvent,
    ArbitrationDecidedEvent,
    SpeechEmittedEvent,
)
from .bus import EventBus
from .runtime import Brain

__all__ = [
    'EventKind',
    'BrainEvent',
    'RawPerceptEvent',
    'AttendedPerceptEvent',
    'PerceptDiscardedEvent',
    'ChemistryEvent',
    'CapabilityClaimEvent',
    'ThoughtProducedEvent',
    'SpeechRequestEvent',
    'SubstrateWriteQueuedEvent',
    'EpisodeFormedEvent',
    'EpisodeConsolidatedEvent',
    'EpisodeDecayedEvent',
    'ReflectionFiredEvent',
    'InteroceptionEvent',
    'PredictionErrorEvent',
    'ThreatDetectedEvent',
    'ConflictDetectedEvent',
    'ValueUpdatedEvent',
    'ArbitrationDecidedEvent',
    'SpeechEmittedEvent',
    'EventBus',
    'Brain',
]
