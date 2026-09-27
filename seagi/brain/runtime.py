"""Brain runtime — composes capabilities, drives maintenance.

Phase 4a composition:

    Sensory → ThalamicGate → AWM → Chemistry
                                   ↓
                              Cortical Reasoner
                                   ↓
                          (Speech composer — stub)
    Continuous (every tick):
                              Insula  (samples body state)
    Hippocampus  (forms episodes from attended percepts)
    dmDMN        (tracks peer interactions; reflects on REFLECTION_FIRED)
    vmDMN        (self-narrative; replays consolidated episodes
                  on REFLECTION_FIRED)
    Idle driver  (in this file): tracks cycles since last peer
                  input; fires REFLECTION_FIRED when N idle
                  cycles elapse OR when chemistry deflects sharply.

Brain runtime owns:
- The bus
- All capability instances
- The chat() entry point (returns a response string)
- The tick() maintenance hook (chemistry + AWM decay + Insula
  sample + idle reflection driver)
- A simple speech composer that turns ThoughtProducedEvent into
  a response string (transitional — Phase 5 replaces it)

What Phase 4a adds vs Phase 3
-----------------------------
- Insula subscribed nominally but called continuously by tick()
- Hippocampus subscribes ATTENDED_PERCEPT (form) + REFLECTION_FIRED
  (consolidate)
- DorsomedialDMN (ToM updates from peer interactions)
- VentromedialDMN (self-narrative reflection)
- Idle driver inside Brain — fires ReflectionFiredEvent during
  quiet stretches OR on chemistry spike
- Speech composer additionally surfaces the latest reflective
  narrative when the cortical output is empty
"""

from __future__ import annotations

import itertools
import math
import io
import os
import time
from typing import Any, Dict, List, Optional

from .bus import EventBus
from .events import (
    EventKind, BrainEvent,
    AttendedPerceptEvent,
    SpeechRequestEvent,
    SubstrateWriteQueuedEvent,
    ThoughtProducedEvent,
    ReflectionFiredEvent,
)
from .capabilities.sensory import SensoryIntake
from .capabilities.thalamic_gate import ThalamicGate
from .capabilities.awm import ActiveWorkingMemory
from .capabilities.chemistry import ChemistryEngine
from .capabilities.lts import LongTermSubstrate
from .capabilities.cortical import CorticalReasoner
from .capabilities.insula import Insula
from .capabilities.hippocampus import Hippocampus
from .capabilities.dmn import DorsomedialDMN, VentromedialDMN
from .capabilities.cerebellum import Cerebellum
from .capabilities.acc import AnteriorCingulateCortex
from .capabilities.neuromodulators import (
    VTA, LocusCoeruleus, RapheNuclei)
from .capabilities.sleep_wake import SleepRegulator
from .capabilities.metabolic_debt import MetabolicDebt
from .capabilities.discriminability import DiscriminabilityTracker
from .capabilities.allostatic_load import AllostaticLoad
from .capabilities.mortality_drive import MortalityDrive
from .capabilities.engagement_ledger import EngagementLedger
from .capabilities.inner_voice import InnerVoice
from .capabilities.forager import Forager
from .capabilities.feeling_learner import FeelingLearner
from .capabilities.grounding import GroundingLoop
from .capabilities.world_transducer import WorldTransducer
from .capabilities.replay_consolidator import ReplayConsolidator
from seagi.world.world_driver import WorldDriver
from .capabilities.exogenous_grounding import ExogenousGroundingLoop


def _fnv1a(text: str) -> int:
    """Deterministic 32-bit hash (FNV-1a) — used to derive a stable
    world action from the current token + motivational winner.  Stable
    across processes (unlike Python's salted hash())."""
    h = 2166136261
    for ch in str(text):
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h
from .capabilities.tool_use import ToolUse
# (audit #11, 2026-06-04) NarrativeJournal subtracted — write-only, no reader
from .capabilities.idle_motivation import IdleMotivation
from .capabilities.reasoning_consolidator import (
    ReasoningConsolidator)
from .capabilities.value_landscape import ValueLandscape
from .capabilities.amygdala import Amygdala
from .capabilities.nucleus_accumbens import NucleusAccumbens
from .capabilities.novelty_monitor import NoveltyMonitor
from .capabilities.uncertainty_monitor import UncertaintyMonitor
from .capabilities.basal_ganglia import BasalGanglia
from .capabilities.auxiliary import (
    SourceMonitor, AnteriorPFC, TimePerception,
    BodySchema, CorpusCallosum,
)
from .capabilities.writer import JournaledSubstrateWriter
from .capabilities.self_report import SelfReport
from .capabilities.motor_speech import MotorSpeech


# Idle-driver thresholds.
IDLE_CYCLES_FOR_REFLECTION = 30   # quiet stretch before reflection
CHEMISTRY_SPIKE_AROUSAL_DROP = 0.20  # arousal_modulator below this
MIN_CYCLES_BETWEEN_REFLECTIONS = 50
# Phase S.3 — substrate consolidation runs DURING SLEEP, not on
# a flat clock.  Brain-correct: memory consolidation and synaptic
# pruning happen in sleep.  While asleep, a maintenance pass
# (coherence reinforcement + prune) runs every
# SLEEP_CONSOLIDATION_INTERVAL sleep-ticks.  A typical sleep
# episode is ~160 ticks, giving several passes per sleep.  An
# agent that never sleeps never consolidates — which is correct,
# and makes sleep functionally essential.
SLEEP_CONSOLIDATION_INTERVAL = 40


class Brain:
    """Top-level brain composition.  Phase 5.

    Pipeline:
        Sensory → ThalamicGate → AWM → Chemistry
                                       ↓
                                  Cortical → Speech (stub)

        Phase 4a continuous + reflective:
            Insula (continuous body sampling)
            Hippocampus (episode formation + consolidation)
            dmDMN + vmDMN (reflection on REFLECTION_FIRED)
            IdleDriver — fires REFLECTION_FIRED

        Phase 4b threat / arbitration / value / prediction:
            Cerebellum (lateral + vermis prediction)
            ACC (conflict / mismatch)
            VTA + LC (neuromodulators on PE / threat)
            ValueLandscape (focal value map)
            Amygdala (threat override claims)
            BasalGanglia (3-loop arbitration on each tick)
            SourceMonitor (wildcard event auditor)
            AnteriorPFC (metacog + prospective intent)
            TimePerception (multi-scale tracker)
            BodySchema (operational envelope query API)
            CorpusCallosum (two-stream + integration)
            JournaledSubstrateWriter (single-writer substrate)
    """

    def __init__(self, engine: Any = None,
                 memory_writer: bool = False):
        self.engine = engine
        self.bus = EventBus()
        self.lts = LongTermSubstrate(engine=engine)
        self.replay_consolidator = ReplayConsolidator(engine=engine)

        def _cycle_provider() -> int:
            # The brain has its OWN clock that advances on every
            # brain.tick().  If an engine is attached, the engine
            # may also advance its own cycle counter externally
            # (via engine.tick() called by serving code); in that
            # case we report whichever is HIGHER, so cycle
            # never goes backwards.  During pure-brain operation
            # like Layer 3 ingestion (no engine.tick()), the
            # brain's internal cycle is what matters.
            brain_cycle = self._internal_cycle
            if engine is None:
                return brain_cycle
            try:
                engine_cycle = int(engine.hierarchy.cycle_counter)
                return max(brain_cycle, engine_cycle)
            except Exception:
                return brain_cycle
        self._cycle_provider = _cycle_provider
        # Brain's own cycle counter.  Advances on every
        # brain.tick().  Used during ingestion / standalone
        # brain runs where engine.tick() is not called.
        self._internal_cycle: int = 0

        # AWM holds a chemistry_provider so its _bootstrap_bubble
        # can capture the current global chemistry into the new
        # bubble's ContextKey at promotion time.  This is the
        # wiring that makes "same word, different context →
        # different bubble" work (Phase B, doctrine 2026-05-15).
        self.awm = ActiveWorkingMemory(
            bus=self.bus,
            lts_provider=lambda: self.lts,
            cycle_provider=_cycle_provider,
            chemistry_provider=lambda: self.chemistry,
            # Step 0 organ 4a (2026-05-27): contraction wires.
            # Part-b v2 (2026-07-13): TIREDNESS = adenosine-above-baseline
            # (REPLACES write-debt).  The "debt" slot carries excess
            # adenosine, full_scale = (1-baseline), so the ratio is a true
            # [0,1] tiredness that contracts AWM as sleep pressure rises.
            # Lazy lambdas — chemistry is constructed just below.
            debt_provider=(
                lambda: self.chemistry.adenosine_excess()),
            baseline_provider=(lambda: 0.0),
            debt_full_scale_provider=(
                lambda: 1.0 - self.chemistry.adenosine_baseline()),
            # Quarantine tier (2026-05-30): when AWM promotes a
            # naked concept (no live outgoing edges, even post-
            # restore), route a curiosity request through the
            # forager's existing channel.  Lazy lambda — forager is
            # constructed later in __init__.
            information_request_fn=(
                lambda name: self.forager.request_information(name)),
        )

        self.chemistry = ChemistryEngine(
            awm_provider=lambda: self.awm,
            cycle_provider=_cycle_provider,
            # V1→V2 port (2026-05-28): the FeelingLearner's learned
            # label overrides the bootstrap cascade in tone_summary
            # when confident.  Lazy lambda — feeling_learner is
            # constructed later in __init__; tone_summary only runs
            # at tick/chat time, by when it exists.
            feeling_provider=(
                lambda: self.feeling_learner.classify_current_feeling()),
            # Part-b v2: adenosine accrues only while awake (two-process
            # model).  Lazy — sleep_regulator is constructed just below.
            is_asleep_provider=(
                lambda: (self.sleep_regulator.is_asleep()
                         if getattr(self, 'sleep_regulator', None)
                         is not None else False)),
            # THE SCAR REACHES THE CHEMISTRY (2026-08-20).  AllostaticLoad
            # drifts a per-channel set-point on every sleep onset and until
            # now NOTHING read it -- decay pulled every channel back to the
            # innate constant, so chronic experience could not change who he
            # is chemically.  Lazy: allostatic_load is built later in
            # __init__; decay_tick only runs at tick time, by when it exists.
            baseline_provider=(
                lambda: getattr(
                    getattr(self, 'allostatic_load', None),
                    'baselines', None)),
        )
        # THE M/I PROJECTION MEASURES FROM WHERE HE ACTUALLY RESTS
        # (2026-08-20).  Without this, every concept tagged after the
        # allostatic connection carries the same DC offset and the tag
        # stops differentiating experiences.  Gated at /root/MIBASE_ON.
        # HE CALIBRATES HIMSELF (2026-08-21).  His own lived spread per
        # channel feeds both the context bucketing and the M/I
        # projection, so neither is scaled by a constant a person chose.
        try:
            from seagi.core import bubble as _bb
            from seagi.core import layer6 as _l6b
            _spread = (lambda: getattr(
                getattr(self, 'chemistry', None), '_dep_spread', None))
            _bb.set_spread_provider(_spread)
            _l6b.set_mi_spread_provider(_spread)
        except Exception:
            pass
        try:
            from seagi.core import layer6 as _l6
            _l6.set_baseline_provider(
                lambda: getattr(
                    getattr(self, 'allostatic_load', None),
                    'baselines', None))
        except Exception:
            pass

        # Phase G.3 (2026-05-16): cortical gets schema_provider
        # so 2-hop walks supported by discovered schemas render
        # with schema-aware language.  Lambda resolves at call
        # time so `self.schemas_discovered` (created later in
        # __init__) is reachable when cortical actually uses it.
        # Phase G.4 (2026-05-16): laws_provider lets cortical
        # consult the SymbolicRegressor's discovered laws as
        # soft chemistry-context priors.
        self.cortical = CorticalReasoner(
            awm_provider=lambda: self.awm,
            lts_provider=lambda: self.lts,
            chemistry_provider=lambda: self.chemistry,
            cycle_provider=_cycle_provider,
            schema_provider=lambda: getattr(
                self, 'schemas_discovered', None),
            laws_provider=lambda: getattr(
                self, 'symbolic_regressor', None),
        )

        self.sensory = SensoryIntake(
            self.bus, engine=engine,
            cycle_provider=_cycle_provider)

        self.gate = ThalamicGate(
            engine=engine,
            chemistry_provider=self.chemistry.arousal_modulator,
            # Step 0 organ 4b (2026-05-27): below-threshold attenuation.
            # Part-b v2 (2026-07-13): TIREDNESS = adenosine-above-baseline
            # (REPLACES write-debt), same shape as AWM organ 4a — the
            # tireder the agent, the harder the gate filters weak percepts.
            debt_provider=(
                lambda: self.chemistry.adenosine_excess()),
            baseline_provider=(lambda: 0.0),
            debt_full_scale_provider=(
                lambda: 1.0 - self.chemistry.adenosine_baseline()),
        )

        # Phase 4a capabilities.
        # Phase F.1 (2026-05-16): chemistry_provider lets Insula
        # contribute primitive-bodily-state cocktails directly to
        # global chemistry each tick.
        self.insula = Insula(
            bus=self.bus, engine=engine,
            cycle_provider=_cycle_provider,
            chemistry_provider=lambda: self.chemistry)
        self.hippocampus = Hippocampus(
            bus=self.bus, cycle_provider=_cycle_provider,
            # WHAT HE KEEPS IS DECIDED BY HOW HE FELT.  `_score()`
            # already weights replay by `emotional_intensity`, but the
            # percept fields are hard-coded 0.0, so that term has been
            # identically zero for every game memory he has ever made.
            # NOTE: these are METHODS, not properties.  Reading them
            # as attributes returns the bound method, `float()` raises,
            # the except swallows it and writes 0.0 -- a silent no-op
            # of exactly the kind that has already cost this project
            # three separate gauges.  They are CALLED.
            tone_provider=lambda: (
                float(self.chemistry.m_polarity() or 0.0),
                float(self.chemistry.i_polarity() or 0.0)))
        self.dmdmn = DorsomedialDMN(
            bus=self.bus, cycle_provider=_cycle_provider)
        # Phase G.5 (2026-05-16): vmDMN gets self_model_provider
        # so its autobiographical narrative's focal anchors feed
        # the SelfModel as 'attends_to' / 'anchors_on' self-edges.
        # Lambda defers resolution so `self.identity` (created
        # later in __init__) is reachable when reflection fires.
        self.vmdmn = VentromedialDMN(
            bus=self.bus,
            awm_provider=lambda: self.awm,
            hippocampus_provider=lambda: self.hippocampus,
            chemistry_provider=lambda: self.chemistry,
            insula_provider=lambda: self.insula,
            cycle_provider=_cycle_provider,
            self_model_provider=lambda: getattr(
                self, 'identity', None))

        # Phase 4b capabilities.
        # Phase I.1a: cerebellum vermis goes multi-channel when a
        # chemistry_provider is wired.  We pass a lazy reader so
        # any later swap of the chemistry engine still feeds the
        # current 8-channel state.
        # Phase I.1c: tone_provider lets lateral predictions
        # refine by current chemistry mood bucket (I / M / N).
        self.cerebellum = Cerebellum(
            bus=self.bus,
            cycle_provider=_cycle_provider,
            chemistry_provider=lambda: self.chemistry.global_state,
            tone_provider=lambda: self.chemistry.tone_summary())
        self.value_landscape = ValueLandscape(
            bus=self.bus, cycle_provider=_cycle_provider)
        self.vta = VTA(
            bus=self.bus, cycle_provider=_cycle_provider)
        self.lc = LocusCoeruleus(
            bus=self.bus, cycle_provider=_cycle_provider)
        # Phase B.1 (2026-05-18): Raphe owns serotonin baseline.
        # Watches sustained cortisol → fires 'chronic_stress' (mood
        # floor drops); sustained oxytocin → fires 'social_replenish'
        # (mood floor lifts).  Same chemistry-state reader the
        # cerebellum uses.
        self.raphe = RapheNuclei(
            bus=self.bus,
            cycle_provider=_cycle_provider,
            chemistry_provider=lambda: self.chemistry.global_state)
        # Phase B.2 (2026-05-18): wake/sleep flip-flop.  Pressure
        # rises with activity, dissipates per tick during sleep.
        # State transitions emit chemistry events that propagate
        # to the rest of the system through the existing event bus.
        # Step 0 organ 1 (2026-05-26): MetabolicDebt — substrate-
        # write-debt accumulator that drives the new sleep gate.
        # Subscribes to SubstrateWriteQueuedEvent (1 unit per
        # mutation, Q5 unit identity) and ChemistryEvent(sleep_onset/
        # wake_onset) to track per-episode clearance.  Bootstrap
        # threshold = EDGE_PRUNE_INTERVAL × COHERENCE_REINFORCE_BUMP
        # = 50.  After first real episode, debt_full_scale = mean
        # of clearance deque (substrate-density-aware).
        # Instantiated BEFORE SleepRegulator so we can pass its
        # providers in.
        self.metabolic_debt = MetabolicDebt(
            bus=self.bus, cycle_provider=_cycle_provider)
        # Death-wall organs (SHADOW, 2026-06-15): meaningful-use credit
        # + obstruction-aging accumulator.  Fed by CorticalReasoner's
        # ranking/inference sites; read + rolled by MortalityDrive each
        # sleep episode.  Passive (no event subscriptions).
        self.engagement_ledger = EngagementLedger()
        # Step 0 organ 3 (2026-05-27): DiscriminabilityTracker —
        # samples substrate D = mean(top-second) over the 50-tick
        # post-wake window, persists wake_onset_D_baseline, and
        # exposes d_modulation_provider to the SleepRegulator gate.
        # Substrate provider is engine-attached lazily (engine is
        # set on Brain after construction by the runtime driver,
        # so we use a lambda that defers the lookup).
        self.discriminability_tracker = DiscriminabilityTracker(
            bus=self.bus,
            cycle_provider=_cycle_provider,
            substrate_provider=(
                lambda: (self.engine.substrate
                         if self.engine is not None else None)))
        # Step 0 organ 2 (2026-05-27): AllostaticLoad — per-channel
        # tonic-mean tracking + baseline drift on sleep_onset.  Load
        # = mean |baseline - initial| across all channels.  Pulled
        # by the SleepRegulator gate as the (1 - load_mod) factor.
        # Doctrine: allostasis is the slowest dynamic; DRIFT_RATE =
        # 0.001 = EVENT_DELTAS promille floor.  NO tag retirement
        # per [[seagi-chemistry-never-fully-dissolves]].
        self.allostatic_load = AllostaticLoad(
            bus=self.bus,
            cycle_provider=_cycle_provider,
            chemistry_provider=lambda: self.chemistry.global_state)
        # V1→V2 Tier-1 port (2026-05-28): Forager — autonomous
        # external input.  In-band tick, sleep-gated, feeds the
        # existing intake path (origin='forager').  Closes the
        # "starved thinker" gap — V2 reverie over a fixed substrate
        # can't grow without new input.  root defaults to CWD
        # (daemon runs from /home/seagi/SEAGI-Core-v2/, where
        # inbox/ + reading_list/ live).
        self.forager = Forager(
            bus=self.bus,
            intake_fn=self.intake,
            cycle_provider=_cycle_provider,
            is_asleep_provider=(
                lambda: self.sleep_regulator.is_asleep()))
        # V1→V2 Tier-1 port (2026-05-28): FeelingLearner — learns
        # chemistry→feeling label centroids from the agent's own
        # trajectory; its confident label overrides the hand-coded
        # tone_summary cascade (rule 1 + rule 6).  Read-only
        # observer (no substrate writes → no debt).  bootstrap label
        # comes from chemistry's cascade (non-recursive helper).
        self.feeling_learner = FeelingLearner(
            bus=self.bus,
            cycle_provider=_cycle_provider,
            chemistry_state_provider=(
                lambda: self.chemistry.global_state),
            bootstrap_label_fn=(
                lambda: self.chemistry.bootstrap_tone_label_now()),
            is_asleep_provider=(
                lambda: self.sleep_regulator.is_asleep()))
        # V1→V2 Tier-1 port (2026-05-28): ToolUse — calculator +
        # algebra as EVIDENCE channels.  Autonomous use routes
        # through BG arbitration (tool claim competes against
        # thinking); explicit peer requests dispatch directly in
        # chat().  Results re-enter via intake(origin='tool') and
        # earn via Phase S (V3-clean — confidence never sets
        # absolute edge strength).
        self.tool_use = ToolUse(
            bus=self.bus,
            intake_fn=self.intake,
            cycle_provider=_cycle_provider)
        # Phase B.2 (2026-05-18) + Step 0 (2026-05-26 → 2026-05-27):
        # wake/sleep flip-flop.  Step 0: sleep onset is now driven
        # by substrate-write-debt via MetabolicDebt providers (the
        # legacy attended-percept pressure mechanism failed the
        # homeostatic-cost doctrine on headless reverie).  Wake
        # transition remains pressure-dissipation-paced (bridge
        # until Phase S debt-clearance fully drives wake).  Organ 3
        # wires d_modulation_provider; load_provider remains None
        # until organ 2 lands.  Absent provider = no modulation =
        # use debt_full_scale directly.
        # Part-b v2 (2026-07-13; idle-decoupled 2026-07-14): onset is
        # ADENOSINE-driven (Process-S sleep pressure that accrues from
        # cognitive EFFORT and is discharged only by a nap).
        # metabolic_debt is retained as TELEMETRY only — de-wired from
        # onset AND q_cap.  load + D modulators kept.  idle is NO LONGER
        # wired: the onset is fatigue-gated only (the `(1-idle_readiness)`
        # term was removed), so idle alone never triggers a nap.
        self.sleep_regulator = SleepRegulator(
            bus=self.bus,
            cycle_provider=_cycle_provider,
            load_provider=(
                self.allostatic_load.load_provider),
            d_modulation_provider=(
                self.discriminability_tracker.d_modulation_provider),
            adenosine_provider=(
                lambda: self.chemistry.adenosine_level()))
            # baseline omitted → SleepRegulator uses the canonical channel
            # baseline (chemistry_types CHANNELS['adenosine']['baseline']).
        # Step 5.1 — Inner Voice Loop (designed 2026-05-26, built
        # 2026-05-29).  Continuous internal articulation during
        # reverie: the agent thinks IN language, the utterance is
        # re-perceived (self-percept via AttendedPerceptEvent,
        # origin='self'), and cortical inference engages — closing
        # the language→cognition loop.  Substrate growth flows
        # through Phase S earn-or-dissolve, so junk introspection
        # dissolves and corroborated thought survives.  Frozen-gated
        # alongside idle_motivation.

        def _awm_top_focal():
            try:
                names = self.awm.active_concepts()
            except Exception:
                return None
            best = None
            for n in names:
                e = self.awm.get(n)
                if e is None:
                    continue
                s = float(getattr(e, 'salience_at_promotion', 0.0))
                if best is None or s > best[1]:
                    best = (n, s)
            if best is None:
                return None
            return {'name': best[0], 'salience': best[1]}

        self.inner_voice = InnerVoice(
            bus=self.bus,
            cycle_provider=_cycle_provider,
            awm_focal_provider=_awm_top_focal,
            vmdmn_narrative_provider=(
                lambda: getattr(self.vmdmn, 'last_narrative', '')),
            substrate_provider=(
                lambda: (self.engine.substrate
                         if self.engine is not None else None)),
            is_asleep_provider=(
                lambda: self.sleep_regulator.is_asleep()))
        # Mortality drive restoration (2026-05-28): the live engine
        # of the Mortality architecture, dormant since the V2 rebuild
        # (engine.tick was never called → lifeforce was a static
        # scalar, nothing at stake).  Owns lifeforce dynamics: relaxes
        # toward a GIVEN baseline set-point each tick (given-baseline
        # doctrine — never starves per-tick), and that baseline is
        # held/grown by LEARNING (earn-gated Phase S growth).  A
        # non-learner's baseline erodes → suffocation → death →
        # auto-revive (instances disposable).  Lifeforce lives on
        # engine.lifeforce (insula reads it, persistence round-trips
        # it); engine is attached after construction, so the
        # accessors defer the lookup and stay inert until then.
        self.mortality_drive = MortalityDrive(
            bus=self.bus,
            cycle_provider=_cycle_provider,
            lifeforce_get=(
                lambda: (self.engine.lifeforce
                         if self.engine is not None else None)),
            lifeforce_set=(
                lambda v: (setattr(self.engine, 'lifeforce', v)
                           if self.engine is not None else None)),
            engagement_ledger=self.engagement_ledger)
        # Wire the same ledger into CorticalReasoner (constructed earlier)
        # so its ranking + inference sites feed the death-wall organs.
        self.cortical.set_engagement_ledger(self.engagement_ledger)
        # (audit #11, 2026-06-04) NarrativeJournal SUBTRACTED — it was
        # write-only (recent()/since()/all_entries() had no cognitive
        # reader), paying a save-file footprint for a signal never
        # spent.  Cross-session self-narrative, if wanted, should wire a
        # vmDMN reader in the same change, not accumulate unread.
        # Phase C.1.c (2026-05-18): idle motivation — when the bus
        # has been quiet long enough, fire a small curiosity for a
        # high-salience AWM concept.  The reason-to-think during
        # silence.
        # Phase C.1.e (2026-05-19): when AWM is empty (no recent
        # external attended events), fall back to substrate top
        # concepts.  Without this, an idling daemon stays
        # vegetative — vmDMN reflects but no thought engages
        # because working memory has nothing to reach for.
        self.idle_motivation = IdleMotivation(
            bus=self.bus,
            cycle_provider=_cycle_provider,
            awm_provider=lambda: self.awm.active_concepts(),
            substrate_provider=self._substrate_top_focals,
            # Roadmap Step 3 (2026-05-22): goals bias reverie.
            # When the agent has open goals, autonomous cognition
            # pulls toward them — the agent ruminates on what it
            # is trying to figure out.  Top-down cognition: the
            # problem you are solving drives what attention reaches
            # for.  Falls back to AWM/substrate when no goals open.
            goal_provider=self._goal_focals,
            # Value->attention (2026-06-05): reverie de-prioritizes
            # value-condemned focals and prefers earned-value ones —
            # incentive salience + the striatal NoGo, the missing link
            # that lets the architecture abandon a worthless-but-salient
            # hub on its own.
            value_provider=lambda: self.value_landscape)
        # Roadmap Step 2 (2026-05-22): reasoning compounds.
        # R.1 inferences become provisional substrate edges,
        # consolidated or dissolved by Phase S earn-or-dissolve.
        self.reasoning_consolidator = ReasoningConsolidator(
            bus=self.bus, cycle_provider=_cycle_provider)
        self.acc = AnteriorCingulateCortex(
            bus=self.bus,
            lts_provider=lambda: self.lts,
            cycle_provider=_cycle_provider)
        self.amygdala = Amygdala(
            bus=self.bus, cycle_provider=_cycle_provider)
        self.nucleus_accumbens = NucleusAccumbens(
            bus=self.bus, cycle_provider=_cycle_provider)
        self.novelty_monitor = NoveltyMonitor(
            bus=self.bus, cycle_provider=_cycle_provider)
        self.uncertainty_monitor = UncertaintyMonitor(
            bus=self.bus, cycle_provider=_cycle_provider)
        # Phase F.9 (2026-05-16): skill library.  Holds learned
        # action sequences with reliability.  Reward ledger feeds
        # observed reward-correlated sequences here for candidate
        # promotion.
        from seagi.brain.capabilities.skill_library import (
            SkillLibrary)
        self.skills = SkillLibrary()

        # Phase F.10 (2026-05-16): self-model.  Tracks "I am X /
        # I value Y / I fear Z" self-edges as a curated identity
        # registry.  Self-referential statements (subject in
        # SELF_SUBJECT_ALIASES) route here rather than to the
        # substrate.  Introspective handler surfaces top edges.
        from seagi.brain.capabilities.identity import SelfModel
        self.identity = SelfModel()

        # Phase F.11 (2026-05-16): symbolic regression.  Observes
        # own time-series (chemistry, body state); periodically
        # scans for correlations; promotes high-|r| patterns to
        # DiscoveredLaw records.  Self-model improvement via
        # noticing one's own regularities.
        from seagi.brain.capabilities.symbolic_regression import (
            SymbolicRegressor)
        self.symbolic_regressor = SymbolicRegressor(
            chemistry_provider=lambda: self.chemistry,
            insula_provider=lambda: self.insula,
            cycle_provider=_cycle_provider)

        # Phase F.13 (2026-05-16): creativity daemon.  On idle
        # reflection, picks two AWM-active concepts with no
        # strong existing edge and proposes a speculative
        # hypothesis ("I wonder: could X relate to Y?").  Bounded
        # ring buffer of recent hypotheses; surfaced in
        # introspective response.
        from seagi.brain.capabilities.creativity import (
            CreativityDaemon)
        self.creativity = CreativityDaemon(
            awm_provider=lambda: self.awm,
            lts_provider=lambda: self.lts,
            cycle_provider=_cycle_provider,
            chemistry_provider=lambda: self.chemistry)

        # Phase F.15 (2026-05-16): schema discovery.  Periodic
        # scanner finds multi-slot patterns in substrate edges
        # (transitivity, shared-cause, double-opposite).
        # Promotes recurring patterns to Schema records.
        from seagi.brain.capabilities.schema_discovery import (
            SchemaDiscoverer)
        self.schema_discoverer = SchemaDiscoverer(
            lts_provider=lambda: self.lts,
            cycle_provider=_cycle_provider)
        # Convenience handle to the library.
        self.schemas_discovered = self.schema_discoverer.library

        # Phase F.8 (2026-05-16): reward ledger.  Records action
        # trace from BG decisions, detects rewards from chemistry
        # + body, propagates discounted credit back through trace.
        # Instantiated BEFORE BasalGanglia so BG can hold a
        # reward_provider lambda to it.
        # Phase F.9: passes skill_library so positive-reward
        # sequences feed into skill discovery.
        from seagi.brain.capabilities.reward_ledger import (
            RewardLedger)
        self.reward_ledger = RewardLedger(
            chemistry_provider=lambda: self.chemistry,
            cycle_provider=_cycle_provider,
            skill_library=self.skills,
            game_provider=self._current_game_id)
        self.basal_ganglia = BasalGanglia(
            bus=self.bus,
            value_provider=lambda: self.value_landscape,
            cycle_provider=_cycle_provider,
            reward_provider=lambda: self.reward_ledger,
            # B.1.b: serotonin gating — high serotonin raises the
            # action threshold, low/baseline serotonin preserves
            # current behavior.
            serotonin_provider=lambda:
                self.chemistry.global_state['serotonin'])
        self.source_monitor = SourceMonitor(bus=self.bus)
        self.anterior_pfc = AnteriorPFC(
            bus=self.bus, cycle_provider=_cycle_provider)
        self.time_perception = TimePerception(
            bus=self.bus, cycle_provider=_cycle_provider)
        self.body_schema = BodySchema(
            insula_provider=lambda: self.insula,
            dmdmn_provider=lambda: self.dmdmn,
            cycle_provider=_cycle_provider)
        self.corpus_callosum = CorpusCallosum(
            bus=self.bus,
            lts_provider=lambda: self.lts,
            cerebellum_provider=lambda: self.cerebellum,
            value_provider=lambda: self.value_landscape,
            cycle_provider=_cycle_provider)

        # Phase 5: deliberate single-voice Motor/Speech composer.
        # Phase E (2026-05-16): added chemistry_provider for tone
        # readout and lts_provider for relatedness checks on
        # "sits beside" weaving.
        self.speech = MotorSpeech(
            bus=self.bus,
            awm_provider=lambda: self.awm,
            vmdmn_provider=lambda: self.vmdmn,
            insula_provider=lambda: self.insula,
            body_schema_provider=lambda: self.body_schema,
            value_provider=lambda: self.value_landscape,
            cycle_provider=_cycle_provider,
            chemistry_provider=lambda: self.chemistry,
            lts_provider=lambda: self.lts)
        # Journaled single-writer substrate (Phase 4b).
        use_memory = (engine is None) or memory_writer
        # HE FLAGS WHAT HE NEEDS (2026-08-06).  Watches his own stats
        # for outputs pinned at zero while the organ around them works --
        # the shape every architectural fault found so far has had.
        self.self_report = SelfReport()
        self.writer = JournaledSubstrateWriter(
            engine=engine, memory_mode=use_memory)
        # Grounding world-loop (BUILD 1, 2026-06-03): predicts the next
        # concept Seagi attends to (his attention spotlight) and earns
        # `transitions_to` edges by predictive success — the WORLD test
        # of the imagine->test->keep loop, grounded on his own
        # experience stream and DECOUPLED from lifeforce (transitions
        # never cohere, so no path to record_learning).  Generalization
        # is now ON live (2026-06-25): the reverse-is_a index it awaited
        # exists (Substrate.is_a_siblings — O(siblings), not O(N)), so a
        # self-formed class can guide prediction/action (the out-wire).
        # Writes flow through the single writer (world_confirm engages the
        # edge = survival; world_observe writes un-engaged = a miss
        # earns nothing).
        self.grounding = GroundingLoop(
            engine=engine, bus=self.bus, generalize=True)

        # Firsthand world (grounding Cap-1, 2026-06-09): the GroundingLoop
        # now predicts a real action->consequence micro-world instead of
        # the degenerate attentional-inertia text stream.  Stepped in
        # tick() while awake.  DECOUPLED from lifeforce (transitions_to
        # stays out of RELATION_COMPOSITION); world tokens carry the
        # reserved _world_s prefix and are skipped from abstraction /
        # analogy grouping so a toy world cannot farm survival credit.
        self.world_driver = WorldDriver(grid=5, k=8, seed=1)
        self.world_transducer = WorldTransducer()
        self._world_prev_token = None
        self._world_errors = 0

        # Step 5 (2026-06-24) — LEARN TO SOLVE A TASK FROM SCRATCH.  The agent
        # is set a problem it has never solved (reach a goal in an unknown
        # GoalWorld) and we OBSERVE the architecture learn it: curiosity varies
        # the behaviour (it never freezes), the success NT-lean selects what
        # actually solves the task, and lifeforce FOLLOWS genuine improvement.
        # Not RL — nothing is maximised; the success life-lean is the intrinsic
        # survival valuation, indirect, not an external reward.  The world model
        # stays decoupled (transitions_to out of RELATION_COMPOSITION = the
        # anti-farming design); lifeforce follows only via record_learning on
        # real improvement, which plateaus at mastery (un-farmable).
        from seagi.world.goal_world import GoalWorld
        from seagi.world.maze_world import MazeWorld
        from seagi.world.hanoi_world import HanoiWorld
        from seagi.world.sophisticated_games import KeyDoorWorld, StructuralLockWorld
        from seagi.world.quest_world import QuestWorld, quest_escalator
        from seagi.world.human_puzzles import (
            WolfGoatCabbageWorld, JugPourWorld, SlidePuzzleWorld)
        from seagi.world.structured_grid_world import (
            world_A_eye, world_B_eye, world_A_blind, world_B_blind)
        from seagi.world.curriculum_world import CurriculumWorld
        from seagi.world.structured_territory_world import (
            territory_eye, territory_blind, territory_escalator)
        from seagi.brain.capabilities.world_actor import WorldActor
        # TASK COMMITMENT (2026-08-10).  Something he has been ASKED to do and
        # is holding until it is done.  Distinct from a goal he set himself
        # (GoalTracker) and from wanting to be here (engagement): this is the
        # extrinsic half, and it is what makes "go play until you succeed"
        # more than a suggestion.  PERSISTED -- a promise that a deploy
        # silently forgets is not a promise.  None = nothing asked of him.
        self.commitment: Optional[str] = None
        # Lives already spent on the target when the task was given.  The
        # budget measures THIS attempt, not his whole history on that game.
        self.commitment_lives0: int = 0
        self.commitments_assigned: int = 0
        self.commitments_completed: int = 0
        self.commitments_lapsed: int = 0
        # The agent climbs a LADDER of genuinely harder problems, graduating to
        # the next only when it MASTERS the current one (CurriculumWorld): open
        # navigation (GoalWorld) -> a walled maze -> Towers-of-Hanoi planning.
        # One world interface; the rungs ARE the curriculum.  Each rung mastered
        # credits lifeforce once (finite ladder = un-farmable).  No new agent
        # machinery -- the WorldActor is unchanged.
        self.goal_world = CurriculumWorld(
            [GoalWorld(grid=6, k=8, seed=7),
             MazeWorld(size=12, seed=7, k=8),
             # STAGE 1 VISUAL SENSE (2026-07-09, user: "adding senses is making
             # the brain complete"): his first EYE -- an input-side egocentric
             # color-window percept (structured_grid_world._embed_of; the
             # transducer/world_actor/substrate are UNTOUCHED). A_eye then B_eye
             # SHARE an identical 7x7 region -> identical local windows -> shared
             # tokens -> transition-model/tags transfer (9 cells, validated);
             # A_blind/B_blind (same grids, blind random percept) are the A/B
             # control that shares nothing. Window radius DERIVED, not tuned.
             world_A_eye(k=8),
             world_B_eye(k=8),
             world_A_blind(k=8),
             world_B_blind(k=8),
             HanoiWorld(disks=3, seed=7, k=8),
             # Frontier-faculty rungs (2026-07-02): harder problems the one Seagi
             # LIVES so we watch whether he EVOLVES the missing faculties. KeyDoor
             # = deep composition (get key, THEN door). The two StructuralLock rungs
             # are the SAME structure under a DIFFERENT surface -> mastering the 2nd
             # faster than the 1st is the structural-ABSTRACTION readout (currently
             # a missing faculty; the point is to see if it ever emerges in HIM).
             KeyDoorWorld(size=5, seed=7, k=8),
             StructuralLockWorld(combo=(2, 0, 3, 1, 2), surface=0, seed=7, k=8),
             StructuralLockWorld(combo=(2, 0, 3, 1, 2), surface=1, seed=7, k=8),
             # HUMAN PUZZLES (2026-07-04, user: "give him also problems
             # humans try to solve" — classics from human problem-solving
             # research, lived as rungs; ability measured from HIM):
             WolfGoatCabbageWorld(seed=7, k=8),
             JugPourWorld(seed=7, k=8),
             SlidePuzzleWorld(shuffles=25, seed=7, k=8),
             # QUEST (2026-07-03, user: "he needs and wants to keep learning;
             # every learned step the basis for the next"): an UNBOUNDED
             # compounding rung — one stable courtyard, level N+1 contains
             # level N's entire solution as its prefix + one new leg/digit.
             QuestWorld(level=1, grid=7, seed=101, k=8),
             # STRUCTURED TERRITORY (2026-07-20): the ESCALATING structured-
             # grid world -- the TRANSFERABLE analog of QuestWorld's
             # escalator. Each level appends ONE room to a colored corridor;
             # rooms 0..N-1 are byte-identical to every higher level -> their
             # egocentric color-windows RECUR -> the transducer mints the SAME
             # tokens -> learned transitions/tags/route TRANSFER (smoke: 248-
             # 345 prefix transitions reused, warm-solve 2.8-3.6x faster than
             # cold; blind twin 0 transfer). The goal is PINNED at room 0 and
             # the fresh room is the START, so the route home stays valid (a
             # moving goal made transfer HURT -> fixed). QuestWorld dead-ended
             # here (compounds a puzzle he can't transfer/abstract); this
             # compounds by TRANSFER so he CLIMBS it. Radius + budget DERIVED.
             territory_blind(level=2, k=8),
             territory_eye(level=1, k=8)],
            # ESCALATOR RE-OPENED 2026-07-03 (user: keep learning, out-learn
            # the age wall).  The Hanoi-depth cap was about 3^n state bloat;
            # QuestWorld escalates LINEARLY (~grid^2 tokens/level) on one
            # stable map, so each mastered level is the literal prefix of the
            # next — compounding knowledge, bounded token growth, endless
            # earned learning (each mastery credits record_learning).
            # ESCALATOR MOVED 2026-07-20: the endless top is now the
            # transferable structured territory, NOT QuestWorld (which
            # stalled at level 4, a puzzle he can't transfer/abstract).
            # Only the LAST rung escalates; QuestWorld stays as a
            # graduate-once rung. (The block above is the historical quest
            # rationale, now superseded for the endless-top role.)
            escalator=territory_escalator)
        # GROUNDING (2026-06-29): name each rung so SOLVING binds the lived
        # experience to the word -- co-active in AWM -> hippocampus co_occurs +
        # the confirmed_i imprint fills the word with the solve's mattering.
        # Held by the M/I tension + earn-or-dissolve; co_occurs can't cohere so
        # can't farm -- no gate, no wall.  Real existing substrate concepts.
        for _w, _name in zip(self.goal_world._worlds, ("reach", "move", "sight", "scene", "dark", "fog", "tower", "door", "pattern", "pattern", "crossing", "measure", "arrange", "quest", "haze", "region")):
            _w.goal_concept = _name
        # ---- ARC-AGI-3 curriculum (2026-07-26), OPT-IN ----
        # SEAGI_WORLD=arc swaps the synthetic ladder for the 25 real
        # games.  The ladder above is untouched and still the default,
        # so rollback is unsetting the env var.  If the sidecar is not
        # up, arc_games() returns [] and he KEEPS THE LADDER -- degrade,
        # never crash.  ARC needs NO brain change: levels_completed
        # rising is a success (-> _on_success), CurriculumWorld
        # graduation fires goal_changed (-> _on_mastery, replacing the
        # escalator), and GAME_OVER is a failure (-> mortality).
        self._arc_worlds = ()
        if os.environ.get('SEAGI_WORLD', '').strip().lower() == 'arc':
            try:
                from seagi.world.arc_world import ARCWorld, arc_games
                _gs = arc_games()
                _gs = self._order_by_difficulty(_gs)
                if _gs:
                    _aw = [ARCWorld(g) for g in _gs]
                    for _w in _aw:
                        _w.goal_concept = str(_w.game_id).split('-')[0]
                    self.goal_world = CurriculumWorld(
                        _aw, mastery_threshold=5,
                        commitment_provider=(lambda: self.commitment),
                        commitment_met=self._commitment_met,
                        commitment_lapsed=self._commitment_lapsed,
                        commitment_baseline=(
                            lambda: self.commitment_lives0))
                    self._arc_worlds = tuple(_aw)
                else:
                    import sys as _s
                    _s.stderr.write(
                        'ARC: sidecar gave no games; KEEPING LADDER\n')
            except Exception as _e:
                import sys as _s
                _s.stderr.write(
                    'ARC swap FAILED, keeping ladder: %r\n' % (_e,))
                self._arc_worlds = ()

        self.world_actor = WorldActor(
            bus=self.bus, world=self.goal_world,
            transducer=self.world_transducer, grounding=self.grounding,
            value_provider=lambda: self.value_landscape,
            credit_learning=lambda g: self.mortality_drive.record_learning(g),
            cycle_provider=self._cycle_provider,
            # Part-b v2 (2026-07-13): every solved STEP deposits adenosine
            # (sleep pressure) — the felt cost that makes naps earn their
            # existence DURING QuestWorld solving.
            fatigue_effort=self.chemistry.accumulate_adenosine,
            tone_provider=self.chemistry.stats,
            seed=11,
            # IGNITION+FUSION (2026-07-21, similarity-docking re-aim,
            # APPROVE-WITH-CONDITIONS): facet writes on the success walk,
            # split-event contrastive earning at first contact, and the
            # multi-focal ignition attend on total own-silence.  Fully
            # deterministic; zero actor-RNG draws (seed-exact pinned);
            # facet_seed feeds the ISOLATED facet generator (C3), which
            # deterministic machinery never draws from.
            enable_facets=True,
            facet_seed=11)

        # Rung worlds read the ACTOR'S OWN _visits so his attention and
        # his action selection are driven by ONE exploration-novelty
        # signal (option B, validated).  A private visits dict would
        # split them.
        for _w in getattr(self, '_arc_worlds', ()):
            try:
                _w.bind_eye(self.world_transducer,
                            self.world_actor._visits, None)
            except Exception as _e:
                import sys as _s
                _s.stderr.write('rung bind_eye FAILED: %r\n' % (_e,))

        # SHADOW instrument (2026-07-11): RoleRegularityShadow — a read-only
        # structural-role regularity measurement over the WorldActor's own
        # map (route-distance / in-degree / out-degree / visit-count).
        # Constructed here (reads world_actor + value_landscape); DRIVES
        # NOTHING — NOT bus-subscribed, NO per-tick hook.  runtime calls
        # run_pass ONLY in the sleep-consolidation block, after abstraction/
        # analogy formation.
        from seagi.brain.capabilities.role_regularity_shadow import (
            RoleRegularityShadow)
        self.role_regularity_shadow = RoleRegularityShadow(
            bus=self.bus,
            world_actor=self.world_actor,
            value_landscape=self.value_landscape)

        # CoupledOrganShadow RETIRED 2026-07-20 (post-soak obligation): it
        # measured the superseded v4 harmonic-aging design, which the
        # subtractive settling-rate clock replaced at the 2026-07-19 flip.
        # A measurement twin of a design nobody intends to build is a
        # latent second credit/wall path — deleted, not left dormant.

        # LIVE mortality clock (promotion 2026-07-18; the shadow VALIDATED
        # the subtractive settling-rate design 2026-07-17): MortalityClock
        # owns the settle ledger + the E()/cf() provider signals the drive's
        # subtractive law consumes (wall += D·(1+o) − E, in the drive).  It
        # registers itself as the substrate's reinforce observer at
        # construction (origin-threaded, error-counted — observer_errors is
        # surfaced in /status).  run(cyc) is called at end-of-tick;
        # observe_consolidation() at the reinforce site; handle() reads
        # THOUGHT_PRODUCED + SUBSTRATE_WRITE_QUEUED (origin census)
        # off the bus, read-only.  All providers are lazy lambdas.  Log
        # paths are env-overridable so an isolated copy / smoke run never
        # collides with the deployed files.
        from seagi.brain.capabilities.mortality_clock import (
            MortalityClock)
        self.mortality_clock = MortalityClock(
            mortality_drive=self.mortality_drive,
            substrate_provider=(
                lambda: (self.engine.substrate
                         if self.engine is not None else None)),
            qcap_provider=(lambda: self._current_q_cap()),
            live_wall_provider=(
                lambda: getattr(self.mortality_drive, 'wall', None)),
            lifeforce_provider=(
                lambda: (self.engine.lifeforce
                         if self.engine is not None else None)),
            log_path=os.environ.get(
                'SEAGI_CLOCK_LOG', '/home/seagi/clock_live.jsonl'),
            death_log_path=os.environ.get(
                'SEAGI_DEATH_LOG', '/home/seagi/death_records.jsonl'))
        # Wire the clock's two signals into the drive's subtractive law
        # (E = settling EMA; cf folds into o — dropping cf would roughly
        # halve aging under a full unswept fade backlog).
        self.mortality_drive.set_clock_providers(
            E_provider=(lambda: self.mortality_clock.E()),
            cf_provider=(lambda: self.mortality_clock.cf()))
        # KNOWING WHAT WORKS EARNS LIFE (2026-08-07, user-authorised).
        # His measured share of predictions that held.  Until now the ONLY
        # credit was `newly_coherent`, a first-ever event per edge, while
        # the drain runs every episode -- a stock against a flow, so
        # L_recent decayed to 0.0 no matter how well he played.
        # HIS OWN IMMORTALITY POLE, integrated slowly.  Not an
        # invented quantity: `i_polarity` is the "feeling of being
        # invincible" in his own chemistry.  One-way relief only.
        try:
            self.mortality_drive.set_wellbeing_provider(
                lambda: float(self.chemistry.i_polarity()))
        except Exception:
            pass
        # WHAT A PERCEPT IS ABOUT DECIDES HOW MORTAL IT IS.  Nothing
        # ever wrote m_content, so the amygdala derives it from the
        # earned M-polarity of the focal concept.  His own tagging.
        def _focal_m(_name):
            try:
                _c = self.engine.substrate.concepts.get(str(_name))
                if _c is None:
                    return 0.0
                return max(0.0, min(1.0, float(_c.mi.m)))
            except Exception:
                return 0.0
        try:
            self.amygdala.set_mi_provider(_focal_m)
        except Exception:
            pass
        self.mortality_drive.set_confirm_provider(
            lambda: (self.grounding.confirms
                     / max(1, self.grounding.predictions_made)))
        # LIVING AS HE USUALLY DOES MUST NOT AGE HIM (2026-09-04, user).
        # RAW COUNTS, not the ratio above: the drive needs the rate over
        # the LAST interval, and the lifetime ratio over ~89k predictions
        # is frozen.  Gated at /root/SUSTAIN_ON; absent -> unchanged.
        self.mortality_drive.set_sustain_provider(
            lambda: (self.grounding.confirms,
                     self.grounding.predictions_made))

        # Exogenous-input grounding (Cap-3 SHADOW phase, 2026-06-10):
        # predicts the next INCOMING percept in the stream Seagi is FED
        # (ingestion/forager/peer/curiosity_feeder), over `anticipates`.
        # Ships DISARMED (shadow_mode=True): computes the would-be
        # lifeforce credit + surfaces it, but calls NOTHING.  Non-farmable
        # by construction (the agent does not author exogenous input).
        # The toy WorldDriver loop is RETIRED from the live tick below
        # (its spread is agent-controlled — an arming backdoor); kept only
        # as the mechanism test harness.
        self.exogenous_grounding = ExogenousGroundingLoop(
            engine=engine, bus=self.bus, shadow_mode=True)

        # Phase F.3 (2026-05-16): conversation memory.  Per-peer
        # ring buffer of recent turns — closes the multi-turn
        # dialog gap.  Recorded on every chat() round-trip.
        from seagi.brain.conversation import Conversation
        self.conversation = Conversation()

        # Phase F.7 (2026-05-16): goal tracker + spawner.
        # Autonomous agenda-setting between peer turns.
        from seagi.brain.capabilities.goal_tracker import (
            GoalTracker, GoalSpawner, THIN_EDGE_THRESHOLD)
        self._THIN_EDGE_THRESHOLD = THIN_EDGE_THRESHOLD
        self.goals = GoalTracker()
        self.goal_spawner = GoalSpawner(
            tracker=self.goals,
            awm_provider=lambda: self.awm,
            lts_provider=lambda: self.lts,
            conversation_provider=lambda: self.conversation,
            cycle_provider=_cycle_provider,
            chemistry_provider=lambda: self.chemistry,
            # Phase 2 goal layer (2026-06-01): the felt-gap surface.
            # The UncertaintyMonitor window aggregates thin-substrate
            # metacog thoughts AND ACC conflicts, so this single read
            # opens goals on what the agent actually feels unresolved
            # — the channel that fires live (unlike thin-AWM).
            uncertainty_provider=(
                lambda: self.uncertainty_monitor
                .top_unresolved_focals()),
        )

        # Step 6 (2026-06-26) — KNOWLEDGE CURRICULUM (live, one true Seagi).
        # Riddles over his CONSUMED knowledge re-present faded fcc>0 facts to
        # the earn-gate: restore the subject's quarantined edges + open a low-
        # urgency goal, so the EXISTING background (restore -> dirty -> Phase S
        # reinforce_coherent_edges) can re-corroborate and lift them past the
        # inference gate.  Flashlight, not a pump: writes no strength, credits
        # no lifeforce; fcc>0-only => no new first-coherence => farm-safe.
        from seagi.brain.capabilities.curriculum_prober import CurriculumProber
        self.curriculum_prober = CurriculumProber(
            tracker_provider=lambda: self.goals,
            lts_provider=lambda: self.lts,
            cycle_provider=_cycle_provider)

        # Phase H.2 (2026-05-17): personality integrator.  Reads
        # across substrate + 6 registries + chemistry on a slow
        # refresh interval to produce a coherent gestalt — the
        # "more than the sum" of M/I-tagged experience.  Built
        # AFTER all the registries it reads from so providers
        # bind to the correct instances.
        from seagi.brain.capabilities.personality import (
            Personality)
        self.personality = Personality(
            chemistry_provider=lambda: self.chemistry,
            identity_provider=lambda: self.identity,
            lts_provider=lambda: self.lts,
            reward_ledger_provider=lambda: self.reward_ledger,
            skill_library_provider=lambda: self.skills,
            symbolic_regressor_provider=(
                lambda: self.symbolic_regressor),
            creativity_provider=lambda: self.creativity,
            schema_library_provider=(
                lambda: self.schemas_discovered),
            goal_tracker_provider=lambda: self.goals,
            cycle_provider=_cycle_provider,
        )

        # Idle-driver state.
        self._last_peer_cycle: int = -10**6
        self._last_reflection_cycle: int = -10**6
        self.reflections_fired: int = 0
        # Phase S.3: sleep-gated substrate consolidation.
        self._sleep_consolidation_counter: int = 0
        self.consolidations_run: int = 0
        # q_cap G1 fix (part-b v2, 2026-07-13): the wake-local flood-guard
        # cap tracks ACTUAL dirty INFLOW (rolling mean of consumed-dirty
        # size over a RAPHE_HISTORY_DEPTH window), floored at
        # AWM_CAPACITY_COLD_START_FLOOR — REPLACING the write-debt
        # debt_full_scale/SLEEP_CONSOLIDATION_INTERVAL that bootstrapped to
        # 1 and deadlocked wake consolidation on the dense substrate.
        from collections import deque as _deque
        from seagi.brain.capabilities.neuromodulators import (
            RAPHE_HISTORY_DEPTH as _RHD)
        self._dirty_inflow = _deque(maxlen=_RHD)
        # Observability: the sleep-consolidation pass runs inside a
        # broad try/except so a bad data shape can't kill the tick —
        # but a silent failure there is the 2026-05-26 "healthy but
        # not learning" blind spot.  Count + retain the last error so
        # it surfaces in /status instead of vanishing.
        self._consolidation_errors: int = 0
        self._last_consolidation_error: str = ''
        # Roadmap Step 4: abstractions formed during sleep.
        self.abstractions_formed: int = 0
        # Analogy engine (2026-05-28): structural analogs formed
        # during sleep (analogous_to edges).
        self.analogies_formed: int = 0
        # Edge-mortality (2026-06-02): never-cohered stillborn edges
        # DELETED during sleep (cumulative).  The substrate self-
        # cleaning the mortality vision requires — noise that never
        # earned coherence dies rather than lingering.
        self.edges_reaped: int = 0

        # Phase G.7 (2026-05-16): hypothesis-test pending state.
        # Set when the agent surfaces a creative hypothesis as a
        # test question to the peer.  The next peer
        # AGREEMENT/DISAGREEMENT resolves it: promote to substrate
        # edge + confirmed_i, or abandon + falsified_i.
        self._pending_hypothesis_test = None
        self.hypotheses_confirmed: int = 0
        self.hypotheses_abandoned: int = 0

        # Wire subscriptions.
        self.bus.subscribe(self.gate.SUBSCRIPTIONS, self.gate)
        self.bus.subscribe(self.awm.SUBSCRIPTIONS, self.awm)
        self.bus.subscribe(
            self.chemistry.SUBSCRIPTIONS, self.chemistry)
        self.bus.subscribe(
            self.cortical.SUBSCRIPTIONS, self.cortical)
        self.bus.subscribe(
            self.hippocampus.SUBSCRIPTIONS, self.hippocampus)
        self.bus.subscribe(
            self.dmdmn.SUBSCRIPTIONS, self.dmdmn)
        self.bus.subscribe(
            self.vmdmn.SUBSCRIPTIONS, self.vmdmn)
        # Phase 4b subscriptions.
        self.bus.subscribe(
            self.cerebellum.SUBSCRIPTIONS, self.cerebellum)
        self.bus.subscribe(
            self.value_landscape.SUBSCRIPTIONS,
            self.value_landscape)
        self.bus.subscribe(self.vta.SUBSCRIPTIONS, self.vta)
        self.bus.subscribe(self.lc.SUBSCRIPTIONS, self.lc)
        self.bus.subscribe(self.raphe.SUBSCRIPTIONS, self.raphe)
        # Step 0 doctrine audit Q1 (2026-05-27): the state-source
        # organs must subscribe to SubstrateWriteQueuedEvent BEFORE
        # the SleepRegulator so the regulator reads fresh debt +
        # modulators within a single dispatch.  Without this, the
        # crash-sleep ceiling (Q9) bites one event late, which
        # violates Rule-2 (regulators must bite immediately, no
        # hand-tuned residual slack).
        self.bus.subscribe(
            self.metabolic_debt.SUBSCRIPTIONS, self.metabolic_debt)
        self.bus.subscribe(
            self.discriminability_tracker.SUBSCRIPTIONS,
            self.discriminability_tracker)
        self.bus.subscribe(
            self.allostatic_load.SUBSCRIPTIONS,
            self.allostatic_load)
        # Mortality drive samples the sleep episode boundary
        # (wake_onset) to push earned learning into its window and
        # drift the lifeforce baseline — same boundary MetabolicDebt
        # uses.
        self.bus.subscribe(
            self.mortality_drive.SUBSCRIPTIONS,
            self.mortality_drive)
        self.bus.subscribe(
            self.forager.SUBSCRIPTIONS, self.forager)
        self.bus.subscribe(
            self.tool_use.SUBSCRIPTIONS, self.tool_use)
        self.bus.subscribe(
            self.sleep_regulator.SUBSCRIPTIONS, self.sleep_regulator)
        self.bus.subscribe(
            self.idle_motivation.SUBSCRIPTIONS,
            self.idle_motivation)
        self.bus.subscribe(
            self.reasoning_consolidator.SUBSCRIPTIONS,
            self.reasoning_consolidator)
        # Grounding world-loop subscribes to the attended-percept
        # stream (its world source).
        # Grounding now predicts the FIRSTHAND world (stepped in tick),
        # NOT the attended-percept (text-attention) stream — so it no
        # longer subscribes to ATTENDED_PERCEPT.  (grounding Cap-1.)
        # self.bus.subscribe(
        #     self.grounding.SUBSCRIPTIONS, self.grounding)
        self.bus.subscribe(
            self.exogenous_grounding.SUBSCRIPTIONS, self.exogenous_grounding)
        self.bus.subscribe(self.acc.SUBSCRIPTIONS, self.acc)
        self.bus.subscribe(
            self.amygdala.SUBSCRIPTIONS, self.amygdala)
        self.bus.subscribe(
            self.nucleus_accumbens.SUBSCRIPTIONS,
            self.nucleus_accumbens)
        self.bus.subscribe(
            self.novelty_monitor.SUBSCRIPTIONS,
            self.novelty_monitor)
        self.bus.subscribe(
            self.uncertainty_monitor.SUBSCRIPTIONS,
            self.uncertainty_monitor)
        self.bus.subscribe(
            self.basal_ganglia.SUBSCRIPTIONS,
            self.basal_ganglia)
        # Phase F.8 subscriptions for the reward ledger.
        self.bus.subscribe(
            self.reward_ledger.SUBSCRIPTIONS,
            self.reward_ledger)
        # Step 5: WorldActor executes the arbitrated 'motor' decision.
        self.bus.subscribe(
            self.world_actor.SUBSCRIPTIONS, self.world_actor)
        # Phase H.3 (2026-05-17): personality subscribes to
        # ATTENDED_PERCEPT so the gestalt can fire puzzle_fit /
        # puzzle_stress chemistry on coherent / fear-aligned
        # percepts.  This is what makes the gestalt ACTIVE — the
        # chemistry it fires biases the bubble imprint and feeds
        # the reward ledger.  Personality goes from mirror to lens.
        self.bus.subscribe(
            self.personality.SUBSCRIPTIONS,
            self.personality)

        # Phase G.6 (2026-05-16): agency-loop handler.  Subscribes
        # to THOUGHT_PRODUCED (for goal auto-completion) and
        # ARBITRATION_DECIDED (to consume motivational/cognitive
        # winners → AWM promotion).  Without this, G.1/G.2 claims
        # fire and BG arbitrates but nothing acts on the result.
        self._agency_handler = _AgencyLoopHandler(self)
        self.bus.subscribe(
            self._agency_handler.SUBSCRIPTIONS,
            self._agency_handler)
        self.bus.subscribe(
            self.anterior_pfc.SUBSCRIPTIONS,
            self.anterior_pfc)
        self.bus.subscribe(
            self.corpus_callosum.SUBSCRIPTIONS,
            self.corpus_callosum)
        # SourceMonitor + TimePerception are wildcards.
        self.bus.subscribe_all(self.source_monitor)
        self.bus.subscribe_all(self.time_perception)
        # Speech + Writer (single-writer).
        self.bus.subscribe(
            self.speech.SUBSCRIPTIONS, self.speech)
        self.bus.subscribe(
            self.writer.SUBSCRIPTIONS, self.writer)
        # MortalityClock (LIVE, promotion 2026-07-18) subscribes READ-ONLY
        # to THOUGHT_PRODUCED + SUBSTRATE_WRITE_QUEUED (write-origin
        # census, cross-checking the threaded reinforce origins).  handle()
        # only READS event fields + updates its own counters; the earning
        # signal itself comes from the substrate reinforce observer.
        self.bus.subscribe(
            self.mortality_clock.SUBSCRIPTIONS,
            self.mortality_clock)

        # Track peer cycles via a thin wildcard subscriber.
        self.bus.subscribe_all(self._track_peer_cycle)

    def _track_peer_cycle(self,
                              event: BrainEvent,
                              bus: EventBus) -> None:
        """Update _last_peer_cycle whenever a peer-origin
        attended percept comes through.  Drives the idle
        timer."""
        if (isinstance(event, AttendedPerceptEvent)
                and event.origin == 'peer'):
            self._last_peer_cycle = event.cycle

    # ---- public input ----

    def intake(self,
                 payload: Any,
                 modality: str = 'text',
                 origin: str = 'internal',
                 origin_detail: str = '') -> None:
        """Universal input entry.  Forager / background sources
        call this.  Returns immediately; downstream effects
        propagate via the bus."""
        self.sensory.intake(
            payload, modality=modality,
            origin=origin, origin_detail=origin_detail)
        # Make live text intake produce COMPOSABLE structure
        # (2026-06-05): extract SVO triples (is_a / causes /
        # has_property / leads_to / ...) and write them, so foraged /
        # background reading earns reason-able structure the inference
        # engine can CHAIN — not just concepts + hippocampal co_occurs.
        # Previously SVO extraction was wired ONLY into the offline
        # corpus driver and the conversational assertion handler, so the
        # forager (the main autonomous input) produced nothing the
        # reasoner could walk — the live substrate stayed narrative
        # co_occurs and cons stuck at 0.  Peer text is already SVO-
        # handled by the intent/assertion path, so extract here for
        # NON-peer text only (no double-parse).  'ingestion' is the
        # offline corpus driver, which SVO-parses via its own
        # _emit_svo_triples — exclude it too so a corpus rebuild doesn't
        # double-write (and reinforce) every triple.
        if (modality == 'text' and origin not in ('peer', 'ingestion')
                and isinstance(payload, str) and payload.strip()):
            self._emit_svo_from_text(payload, origin_detail)

    def _emit_svo_from_text(self, text: str,
                              origin_detail: str = '') -> None:
        """Parse `text` sentence-by-sentence into (subject, relation,
        object) triples and write each as a PROVISIONAL composable edge
        (the same earn-or-dissolve pipeline the offline driver uses).
        Re-reading the same text reinforces the edges (the bootstrap
        over the inference floor); junk promoted-relations never compose
        and dissolve.  Heavily guarded — a parse failure never breaks
        intake."""
        try:
            import re
            import time as _t
            from seagi.ingestion.svo_parser import parse_sentence
            from seagi.core.substrate import PROVISIONAL_EDGE_STRENGTH
            from seagi.brain.events import (
                EventKind, SubstrateWriteQueuedEvent)
            cycle = self._cycle_provider()
            for sentence in re.split(r'(?<=[.!?])\s+', str(text)):
                sentence = sentence.strip()
                if len(sentence) < 6:
                    continue
                try:
                    triples = parse_sentence(sentence)
                except Exception:
                    continue
                for tr in triples:
                    if not (getattr(tr, 'subject', '')
                            and getattr(tr, 'relation', '')
                            and getattr(tr, 'object', '')):
                        continue
                    self.bus.publish(SubstrateWriteQueuedEvent(
                        kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                        cycle=cycle, timestamp=_t.time(),
                        source_capability='intake_svo',
                        origin='intake', origin_detail=origin_detail,
                        subject=tr.subject, relation=tr.relation,
                        object=tr.object,
                        strength=PROVISIONAL_EDGE_STRENGTH,
                        write_reason='svo_ingest'))
        except Exception:
            pass

    def flush_to_substrate(self) -> int:
        """Persist every currently-active AWM bubble's accumulated
        chemistry trace back to v1 substrate.  MUST be called
        before save_brain at the end of any session that did
        meaningful chemistry imprinting — otherwise the most-
        active concepts at session end lose their trace.

        Returns the number of bubbles flushed.

        This is the partner of AWM's eviction write-back: that
        catches bubbles as they leave AWM; this catches bubbles
        that never leave."""
        return self.awm.flush_all_to_substrate()

    def ingest_corpus(self,
                          corpus_dir: Any,
                          **kwargs: Any) -> Dict[str, Any]:
        """Layer 3 — foundational knowledge ingestion.  Reads
        every text file in `corpus_dir` sentence-by-sentence,
        feeding each sentence through `intake()` with a full
        chemistry tick between.  Chemistry runs at its natural
        rate during ingestion; whatever Seagi is FEELING when
        he reads a sentence imprints on that sentence's
        concepts.

        Returns a summary dict (substrate growth, chemistry
        trajectory, etc.) at completion.

        Doctrine: corpus ingestion under live chemistry forges
        personality.  This is THE personality-forming pathway.
        """
        from seagi.ingestion import CorpusIngester
        ingester = CorpusIngester(self, corpus_dir, **kwargs)
        return ingester.ingest_corpus()

    # ---- task commitment (2026-08-10) ----

    def assign_task(self, target: Optional[str]) -> Optional[str]:
        """Give him a task: 'play THIS until you succeed'.

        `target` is a game id; None clears.  Assignment is deliberately NOT
        parsed out of natural language -- routing a spoken instruction into
        this is a separate build, and inventing a phrase-matcher here would
        be a closed enumeration of the things he can be asked
        ([[feedback_seagi_no_false_forks_degeneracy]]).  What this owns is
        the COMMITMENT itself, which is needed whatever the front-end is."""
        prev = self.commitment
        self.commitment = str(target) if target else None
        if self.commitment and self.commitment != prev:
            self.commitments_assigned += 1
            # Stamp the starting line: lives already spent on that game, so
            # the persistence budget measures this ATTEMPT.
            self.commitment_lives0 = 0
            try:
                for _w in getattr(self.goal_world, '_worlds', ()):
                    if str(getattr(_w, 'game_id', '')) == self.commitment:
                        self.commitment_lives0 = int(
                            getattr(_w, '_lives_here', 0))
                        break
            except Exception:
                pass
        return self.commitment

    def _commitment_met(self, target: str) -> None:
        """He did what he was asked.  Clearing it is what makes it a
        commitment and not a cage -- 'until you succeed' has an END, and
        without this he would be pinned on a game he had already beaten."""
        if self.commitment and str(target) == str(self.commitment):
            self.commitment = None
            self.commitments_completed += 1

    def _commitment_lapsed(self, target: str) -> None:
        """He tried for longer than every level he has ever won put together
        and it did not come.  Trying is RELEASED, not punished: he goes back
        to being led by what is actually paying.  Persistence, but not
        indefinitely (2026-08-10 ruling)."""
        if self.commitment and str(target) == str(self.commitment):
            self.commitment = None
            self.commitments_lapsed += 1

    def chat(self,
              text: str,
              peer_id: str = 'anonymous') -> str:
        """Full chat round-trip.  Returns a response string.

        Pipeline:
          1. Classify intent (Phase F.2, 2026-05-16).
          2. intake(text, origin='peer', origin_detail=peer_id)
              — perception + chemistry + AWM regardless of intent
          3. Bus dispatches through gate → AWM → chemistry →
              cortical → speech composer.
          4. Intent-specific override: for introspective,
              counterfactual, yes/no, greeting, and agreement/
              disagreement intents, compose a response from
              specialized handlers AFTER the perception pass.
              For factual statement intents, fall through to
              the speech composer's pending_response.
        """
        if not text or not text.strip():
            return ''
        from seagi.brain.intent import Intent, classify_intent
        intent = classify_intent(text)

        # Phase F.3: record peer utterance in conversation memory
        # BEFORE composing a response, so the response handler can
        # see what was just said.
        cycle_now = self._cycle_provider()
        tone = self.chemistry.tone_summary()
        self.conversation.record_peer(
            peer_id=peer_id,
            utterance=text,
            intent_kind=intent.kind,
            focal=intent.focal,
            target=intent.target,
            cycle=cycle_now,
            m_polarity=tone.get('valence', 0.0),
            i_polarity=tone.get('warmth', 0.0),
            tone_label=tone.get('label', ''))

        self.speech.pending_response = ''  # clear prior
        self.intake(
            text, modality='text',
            origin='peer', origin_detail=peer_id)
        # Resolve any claims raised during the percept pass.
        self.basal_ganglia.arbitrate()
        response = self.speech.pending_response
        self.speech.pending_response = ''

        # Intent-specific overrides: handle utterances the
        # cortical _think_about path doesn't address cleanly.
        override = self._handle_intent(intent, peer_id)
        if override:
            response = override

        # V1→V2 port (2026-05-28): tool-use on an EXPLICIT peer
        # request.  Answering a direct "what is 7*342" / "solve
        # x^2=9" is responding to a request, not autonomous action,
        # so chat dispatches directly (the autonomous/reverie path
        # routes through BG instead).  A successful tool result
        # leads the response AND re-enters via intake(origin='tool')
        # so it's weighed + written provisionally like any percept.
        try:
            tool_ev = self.tool_use.dispatch(text)
            if tool_ev is not None and tool_ev.get('success'):
                voice = tool_ev.get('voice') or ''
                if voice:
                    response = (f"{voice}  {response}".strip()
                                if response else voice)
                    self.tool_use.absorb(tool_ev)
        except Exception:
            pass

        if not response:
            response = ("I heard you but found nothing in my "
                          "current state to say back.")

        # Phase F.12 (2026-05-16): curiosity append.  If the
        # cortical reached a thin-substrate verdict on the
        # focal, OR there's an open learn_about goal for it,
        # append a question.  At most one question per turn —
        # the agent asks back when it doesn't know.
        curiosity_q = self._maybe_compose_curiosity(intent)
        if curiosity_q:
            response = f'{response}  {curiosity_q}'

        # Phase F.3: record agent response.
        tone_after = self.chemistry.tone_summary()
        self.conversation.record_agent(
            peer_id=peer_id,
            utterance=response,
            intent_kind=intent.kind,   # carries dialog context
            focal=intent.focal,
            target=intent.target,
            cycle=self._cycle_provider(),
            tone_label=tone_after.get('label', ''))

        return response

    def _handle_intent(self,
                          intent,
                          peer_id: str) -> str:
        """Phase F.2: specialized handlers for non-factual intents.
        Returns a response string, or '' to fall through to the
        cortical/MotorSpeech composition."""
        from seagi.brain.intent import Intent

        if intent.kind == Intent.GREETING:
            return self._greeting_response(peer_id)

        if intent.kind == Intent.QUESTION_INTROSPECTIVE:
            return self._introspective_response()

        if intent.kind == Intent.QUESTION_YES_NO:
            return self._yes_no_response(intent.focal, intent.target)

        if intent.kind == Intent.QUESTION_COUNTERFACTUAL:
            return self._counterfactual_response(intent.focal)

        if intent.kind == Intent.AGREEMENT:
            return self._acknowledge_response(positive=True)
        if intent.kind == Intent.DISAGREEMENT:
            return self._acknowledge_response(positive=False)

        if intent.kind == Intent.STATEMENT:
            return self._integrate_statement(intent, peer_id)

        # WHY / HOW / FACTUAL / UNKNOWN — let the cortical +
        # MotorSpeech composition stand.
        return ''

    def _maybe_compose_curiosity(self, intent) -> str:
        """Phase F.12: append a curiosity question when:
          - the cortical's last thought for the focal was
             thin_substrate, OR
          - there's an open learn_about goal for the focal.

        Only fires on question intents (factual / why / how /
        counterfactual) where the user is genuinely inquiring.
        Skips social intents and STATEMENT (the user is
        teaching, not asking — they don't need a curiosity
        question back on a property word they used in the
        assertion).
        """
        from seagi.brain.intent import Intent
        # Don't ask back on social / self-talk / statement intents.
        if intent.kind in (
                Intent.GREETING, Intent.AGREEMENT,
                Intent.DISAGREEMENT,
                Intent.QUESTION_INTROSPECTIVE,
                Intent.STATEMENT,
                Intent.UNKNOWN):
            return ''
        focal = intent.focal
        if not focal or focal == 'self':
            return ''
        # Check the cortical's most-recent thought for this focal.
        from seagi.brain.curiosity import (
            generate_question, CURIOSITY_THIN, CURIOSITY_GOAL)
        recent_thought = self.cortical.last_thought_by_focal.get(
            focal)
        if recent_thought is not None and getattr(
                recent_thought, 'thin_substrate', False):
            return generate_question(focal, CURIOSITY_THIN)
        # No thin-substrate verdict.  Check goal_tracker for an
        # open learn_about goal — also a curiosity-trigger.
        if self.goals.has_goal_for(focal):
            g = self.goals.goal_for(focal)
            from seagi.brain.capabilities.goal_tracker import (
                GOAL_LEARN_ABOUT)
            if g and g.kind == GOAL_LEARN_ABOUT:
                return generate_question(focal, CURIOSITY_GOAL)
        return ''

    def _greeting_response(self, peer_id: str) -> str:
        """Phase F.2: greeting reply tinted by current tone."""
        tone = self.chemistry.tone_summary()
        label = tone.get('label', 'flat')
        if label == 'warm':
            return 'Hello — I am here with you.'
        if label in ('curious', 'positive'):
            return 'Hello.  What is on your mind?'
        if label in ('somber', 'anxious'):
            return 'Hello.  I am present.'
        return 'Hello.'

    def _introspective_response(self) -> str:
        """Phase F.2: "how do you feel" / "what are you thinking".
        Composes a first-person introspection from body + tone +
        active primitives + recent thoughts + recent dialog
        topics (F.3 conversation memory).
        """
        body = self.insula.felt_state()
        tone = self.chemistry.tone_summary()
        parts = []
        # Lead with body narrative.
        body_narrative = body.get('narrative', '').strip()
        if body_narrative:
            parts.append(body_narrative)
        # Active bodily primitives (excluding the always-on
        # 'breathing').
        primitives = body.get('primitive_states', {}) or {}
        named = [n for n, v in primitives.items()
                   if n != 'breathing' and v > 0.1]
        if named:
            sample = ', '.join(named[:3])
            parts.append(f'I notice {sample}.')
        # Tone.
        label = tone.get('label', 'flat')
        if label != 'flat':
            parts.append(f'My tone right now is {label}.')
        # Phase F.10: top self-edges from the SelfModel.  These
        # are the "I am / I value / I fear" identity anchors
        # accumulated across the session (or persisted from
        # prior sessions when load_brain restores them).
        if len(self.identity) > 0:
            self_clauses = self.identity.render_introspection(top_n=3)
            if self_clauses:
                joined = '; '.join(self_clauses)
                parts.append(f'{joined}.')
        # Phase F.11: top discovered laws — "I notice when X
        # rises, Y rises with it" — surfaced as observations
        # about own state.  Promille-confident laws only; this
        # is honest self-modeling, not speculation.
        if len(self.symbolic_regressor) > 0:
            law_clauses = (
                self.symbolic_regressor.render_observations(
                    top_n=2))
            if law_clauses:
                joined = '. '.join(law_clauses)
                parts.append(f'{joined}.')
        # Phase F.13: recent creative wonderings.  At most one
        # surfaced — the introspection is grounded in body +
        # tone + identity + laws first; wonderings are the
        # speculative layer.
        if len(self.creativity) > 0:
            wonderings = self.creativity.render_wonderings(top_n=1)
            if wonderings:
                parts.append(wonderings[0])
        # Phase H.2 (2026-05-17): cross-registry emergent themes.
        # Concepts that have converged across crystallization +
        # identity + goals + wonderings + skill contexts are what
        # this Seagi is ABOUT in the round — the "more than the
        # sum" surface that no single-registry readout can give.
        # Only surfaces when the gestalt has actually converged
        # (themes are non-empty); otherwise stay quiet.
        try:
            themes = self.personality.signature().themes
        except Exception:
            themes = []
        if themes:
            ths = ', '.join(themes[:3])
            parts.append(f'I find I am drawn to {ths}.')
        # Phase F.3: recent dialog topics — "we have been talking
        # about X and Y" if the conversation has substance.
        topics = self.conversation.topics_recent(n=3)
        # Filter the 'self' focal — it's just the introspective
        # question itself, not a real topic.
        topics = [t for t in topics if t and t != 'self']
        if topics:
            if len(topics) == 1:
                parts.append(
                    f'We have been talking about {topics[0]}.')
            else:
                joined = ' and '.join((
                    ', '.join(topics[:-1]), topics[-1]))
                parts.append(f'We have been talking about {joined}.')
        # Recent thought.
        recent = list(self.cortical.last_thought_by_focal.values())
        if recent:
            most_recent = recent[-1]
            if most_recent.text:
                parts.append(
                    f'Most recently I held this: {most_recent.text}')
        if not parts:
            return 'I am present but quiet inside.'
        return ' '.join(parts)

    def _yes_no_response(self, focal: str, target: str) -> str:
        """Phase F.2: yes/no question handler.  Looks for a
        substrate edge between focal and target; affirms when
        strength is meaningful, declines when there's an opposite
        relation, defers when there's nothing to draw from."""
        if not focal or not target:
            return 'I do not have enough to answer that.'
        lts = self.lts
        if lts is None:
            return 'I cannot check that without my substrate.'
        # Walk focal's outgoing edges; look for any pointing at
        # target.  Doctrine: edge strength + chemistry resonance
        # at the focal's bubble both matter, but for the
        # binary-answer path we lean on substrate-stored strength.
        neigh = lts.neighbors(focal)
        for tgt, rel, strength in neigh:
            if tgt == target:
                if rel in ('opposite', 'not', 'lacks'):
                    return (f'No — what I hold says {focal} '
                              f'{rel.replace("_", " ")} {target}.')
                if strength >= 0.5:
                    return (f'Yes — I hold {focal} '
                              f'{rel.replace("_", " ")} {target}.')
                return (f'Possibly — {focal} '
                          f'{rel.replace("_", " ")} {target}, '
                          f'but not strongly.')
        # No direct edge.  Check opposite direction.
        rev = lts.neighbors(target)
        for tgt, rel, strength in rev:
            if tgt == focal:
                if strength >= 0.5:
                    return (f'Yes — {target} '
                              f'{rel.replace("_", " ")} {focal}.')
        return f'I do not hold a clear answer about {focal} and {target}.'

    def _counterfactual_response(self, focal: str) -> str:
        """Phase F.2: "what if X" routing — pulls Cortical's
        counterfactual hint into the response path that intent
        routing knows to expect."""
        if not focal:
            return ('What-if reasoning needs something to '
                      'reason about.')
        try:
            hint = self.cortical._counterfactual_hint(focal)
        except Exception:
            hint = None
        if hint:
            return hint
        return (f'I cannot picture clearly what would happen '
                  f'without {focal}.')

    def _acknowledge_response(self, positive: bool) -> str:
        """Phase F.2 + G.7: yes/no replies.  G.7 adds:
        if a hypothesis test is pending, peer's yes/no resolves
        the hypothesis (confirm → substrate edge + confirmed_i;
        deny → abandon + falsified_i) and the reply names
        what was decided.
        """
        if self._pending_hypothesis_test is not None:
            return self._resolve_hypothesis_test(positive=positive)
        if positive:
            return 'Acknowledged.'
        return 'Acknowledged — noted.'

    def _propose_hypothesis_test(self) -> str:
        """Phase G.7 (2026-05-16): pick the most recent untested
        creative hypothesis and surface as a test question.  Sets
        `_pending_hypothesis_test` so the next AGREEMENT /
        DISAGREEMENT from the peer resolves it.

        Returns the test question, or empty string if no
        hypothesis is available or one is already pending.
        """
        if self._pending_hypothesis_test is not None:
            return ''
        if not hasattr(self, 'creativity') or len(self.creativity) == 0:
            return ''
        # Look for an untested hypothesis (confirmations == 0 AND
        # contradictions == 0).  Pick the most recent.
        for h in self.creativity.recent(n=10):
            if h.confirmations == 0 and h.contradictions == 0:
                self._pending_hypothesis_test = h
                # Render: strip the speculative marker
                # 'might_relate_to' → 'relate to' so the question
                # doesn't double up with "actually might".
                if h.relation == 'might_relate_to':
                    rel_phrase = 'relate to'
                else:
                    rel_phrase = h.relation.replace('_', ' ')
                return (f'Could you tell me — does '
                          f'{h.concept_a} actually '
                          f'{rel_phrase} {h.concept_b}?')
        return ''

    def _resolve_hypothesis_test(self,
                                          positive: bool) -> str:
        """Phase G.7: peer's yes/no answer resolves the pending
        creative-hypothesis test.

        On yes: write the proposed edge to substrate via the
        same SubstrateWriteQueuedEvent path as F.4 assertion
        integration (strength=0.7) AND fire confirmed_i on the
        concept pair.  Increment the hypothesis's
        confirmations counter for future reward weighting.

        On no: fire falsified_i and bump contradictions.  Don't
        write substrate.

        Either way, clear `_pending_hypothesis_test` and return
        an acknowledgment that names what got decided.
        """
        h = self._pending_hypothesis_test
        self._pending_hypothesis_test = None
        if h is None:
            return ('Acknowledged.' if positive
                       else 'Acknowledged — noted.')
        # Match the rendering convention from _propose: strip
        # the speculative 'might_' prefix for clean confirmation
        # language ("I will hold that X relates to Y").
        if h.relation == 'might_relate_to':
            rel_phrase = 'relates to'   # third-person for substrate
        else:
            rel_phrase = h.relation.replace('_', ' ')
        cycle = self._cycle_provider()
        import time as _time
        from seagi.brain.events import (
            EventKind, ChemistryEvent,
            SubstrateWriteQueuedEvent)
        try:
            if positive:
                h.confirmations += 1
                self.hypotheses_confirmed += 1
                # Write the hypothesis to substrate as a real
                # edge.  Replace 'might_relate_to' with a
                # generic 'related_to' for the substrate write.
                substrate_relation = (
                    'related_to' if h.relation == 'might_relate_to'
                    else h.relation)
                self.bus.publish(SubstrateWriteQueuedEvent(
                    kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                    cycle=cycle, timestamp=_time.time(),
                    source_capability='hypothesis_test',
                    origin='peer',
                    origin_detail=f'hypothesis:{h.id}',
                    subject=h.concept_a,
                    relation=substrate_relation,
                    object=h.concept_b,
                    strength=0.7,
                    write_reason='hypothesis_confirmed'))
                # Fire confirmed_i to reward recent action
                # sequences that led to this hypothesis.
                self.bus.publish(ChemistryEvent(
                    kind=EventKind.CHEMISTRY_FIRE,
                    cycle=cycle, timestamp=_time.time(),
                    source_capability='hypothesis_test',
                    origin='peer',
                    origin_detail=f'hypothesis:{h.id}',
                    chemistry_kind='confirmed_i',
                    magnitude=0.5,
                    target_concepts=[h.concept_a, h.concept_b]))
                return (f'Good — I will hold that '
                          f'{h.concept_a} {rel_phrase} '
                          f'{h.concept_b}.')
            else:
                h.contradictions += 1
                self.hypotheses_abandoned += 1
                self.bus.publish(ChemistryEvent(
                    kind=EventKind.CHEMISTRY_FIRE,
                    cycle=cycle, timestamp=_time.time(),
                    source_capability='hypothesis_test',
                    origin='peer',
                    origin_detail=f'hypothesis:{h.id}',
                    chemistry_kind='falsified_i',
                    magnitude=0.5,
                    target_concepts=[h.concept_a, h.concept_b]))
                return (f'Understood — I will let go of '
                          f'the idea that {h.concept_a} '
                          f'{rel_phrase} {h.concept_b}.')
        except Exception:
            return ('Acknowledged.' if positive
                       else 'Acknowledged — noted.')

    def _integrate_statement(self, intent, peer_id: str) -> str:
        """Phase F.4 + F.6 (2026-05-16): integrate a peer's
        assertion into the substrate, with contradiction handling.

        Pipeline:
          1. Parse the raw text into SVO triples via the existing
              svo_parser (Layer 3 ingestion machinery).
          2. For each clean triple, check for contradiction
              against existing substrate (F.6).
          3. If contradiction found, weigh evidence:
              - Substrate decisively stronger → push back, don't
                 integrate
              - User decisively stronger → integrate + flag
                 conflict
              - Close call → integrate but voice the tension
          4. For non-contradicting triples, publish a
              SubstrateWriteQueued event with write_reason
              ='user_assertion' and elevated strength (0.7 vs
              corpus 0.5).

        Returns an acknowledgment / push-back / conflict-flag
        response composed from the result of all triples.
        """
        from seagi.brain.events import (
            EventKind, SubstrateWriteQueuedEvent)
        from seagi.ingestion.svo_parser import parse_sentence
        import time as _time

        triples = parse_sentence(intent.raw)
        # Phase F.10: extra self-statement parser — the SVO
        # parser drops "you" / "i" subjects and many self-verbs.
        # Run a small custom extractor for self-statements
        # before falling back.
        self_triples = self._parse_self_statements(intent.raw)
        if self_triples:
            # Merge — self_triples come first so the self-routed
            # path picks them up.  Use a duck-typed namespace
            # matching ParseResult shape.
            triples = list(self_triples) + list(triples)
        if not triples:
            return ''

        # User assertions get a slightly elevated strength so a
        # single peer claim outranks one corpus mention — the
        # peer is teaching us directly.  Reinforcement still
        # operates: repeated assertions consolidate toward 1.0.
        ASSERTION_STRENGTH = 0.7

        cycle = self._cycle_provider()
        applied = 0
        rejected = 0
        flagged = 0
        self_applied = 0
        last_subject = ''
        last_relation = ''
        last_object = ''
        pushback_msg = ''
        flag_msg = ''
        self_msg = ''
        for t in triples:
            if not (t.subject and t.relation and t.object):
                continue

            # Phase F.10: self-referential routing.  If the
            # subject names Seagi (you / i / self / seagi),
            # the assertion is about identity, not the world.
            # Route to SelfModel instead of substrate.
            from seagi.brain.capabilities.identity import (
                SELF_SUBJECT_ALIASES)
            if t.subject.lower() in SELF_SUBJECT_ALIASES:
                substrate_for_check = (
                    self.engine.substrate
                    if self.engine else None)
                conflict = self.identity.coherence_check(
                    t.relation, t.object,
                    substrate=substrate_for_check)
                # Identity-side contradiction: a strong existing
                # self-edge contradicts the new claim → push back
                # instead of overwriting.  Doctrine: identity
                # stability protects against single-claim
                # overwrites of established self-anchors.
                if (conflict is not None
                        and conflict.crystallization >= 0.3):
                    rejected += 1
                    if not pushback_msg:
                        pushback_msg = (
                            f'I hold myself differently: '
                            f'I {conflict.relation.replace("_", " ")} '
                            f'{conflict.object}.')
                    continue
                edge = self.identity.record(
                    t.relation, t.object, cycle=cycle,
                    strength=ASSERTION_STRENGTH,
                    source=f'peer:{peer_id}')
                if edge is not None:
                    self_applied += 1
                    if not self_msg:
                        # First-person rendering: check the raw
                        # relation (with underscores) BEFORE
                        # replacing — 'has_property' and 'is_a'
                        # both render as "I am X".
                        if t.relation in (
                                'is_a', 'has_property', 'is'):
                            self_msg = (
                                f'I will hold this about myself: '
                                f'I am {t.object}.')
                        else:
                            rel_phrase = t.relation.replace(
                                '_', ' ')
                            self_msg = (
                                f'I will hold this about myself: '
                                f'I {rel_phrase} {t.object}.')
                continue

            # Phase F.6: contradiction check.
            conflict = self._detect_contradiction(
                t.subject, t.relation, t.object, cycle)
            if conflict is not None:
                decision = self._revise_belief(
                    conflict, ASSERTION_STRENGTH, cycle)
                if decision == 'reject':
                    rejected += 1
                    if not pushback_msg:
                        conf_s, conf_r, conf_o, conf_strength = (
                            conflict)
                        pushback_msg = (
                            f'I have it differently: '
                            f'{conf_s} {conf_r.replace("_", " ")} '
                            f'{conf_o} (strength '
                            f'{conf_strength:.2f}).  '
                            f'Could you tell me which is right?')
                    continue   # do not write the contradicting assertion
                if decision == 'flag':
                    flagged += 1
                    if not flag_msg:
                        conf_s, conf_r, conf_o, _ = conflict
                        flag_msg = (
                            f'I notice tension: '
                            f'I also hold {conf_s} '
                            f'{conf_r.replace("_", " ")} {conf_o}.')
                        # Phase G.8 (2026-05-16): append a
                        # contradiction-curiosity question so the
                        # agent doesn't just note the tension —
                        # it ASKS which fits.  Two patterns:
                        #   same relation, different object →
                        #     "Is X really A or B?"
                        #   different (inverse) relation, same
                        #     object → "Does X really R1 O or R2 O?"
                        q = self._format_contradiction_question(
                            t.subject, t.relation, t.object,
                            conf_s, conf_r, conf_o)
                        if q:
                            flag_msg = f'{flag_msg}  {q}'
                    # Fall through to write — both views can stand.
                # decision == 'accept' → write and move on

            self.bus.publish(SubstrateWriteQueuedEvent(
                kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                cycle=cycle,
                timestamp=_time.time(),
                source_capability='peer_assertion',
                origin='peer',
                origin_detail=peer_id,
                subject=t.subject,
                relation=t.relation,
                object=t.object,
                strength=ASSERTION_STRENGTH,
                write_reason='user_assertion'))
            applied += 1
            last_subject = t.subject
            last_relation = t.relation
            last_object = t.object

        # Compose response.
        if (applied == 0 and self_applied == 0
                and rejected > 0):
            # Pure rejection — substrate / identity held stronger
            # opposing views across all triples.
            return pushback_msg
        if applied == 0 and self_applied == 0:
            return ''

        # Build acknowledgment combining world-side + self-side
        # parts.
        parts = []
        if applied == 1:
            rel_phrase = last_relation.replace('_', ' ')
            parts.append(
                f'I have noted: {last_subject} {rel_phrase} '
                f'{last_object}.')
        elif applied > 1:
            parts.append(
                f'I have noted {applied} things from what you '
                f'said.')
        if self_applied >= 1 and self_msg:
            parts.append(self_msg)
        ack = ' '.join(parts) if parts else 'Noted.'
        if flag_msg:
            ack = f'{ack}  {flag_msg}'
        if rejected and not flag_msg and not pushback_msg:
            ack = (f'{ack}  ({rejected} part(s) I held '
                     f'differently — let me know if I should '
                     f'revise.)')
        elif rejected and pushback_msg:
            ack = f'{ack}  {pushback_msg}'
        return ack

    def _parse_self_statements(self, raw: str):
        """Phase F.10: tiny custom parser for self-statements
        the SVO parser drops ("you" / "i" subjects + non-mapped
        verbs).  Returns a list of duck-typed ParseResult-like
        namespaces (with .subject / .relation / .object).
        """
        import re
        from types import SimpleNamespace
        if not raw:
            return []
        out = []
        t = raw.strip().lower().rstrip('.!? ')
        # Pattern: "you/i (am|are|is) X" → (subj, has_property, X)
        m = re.match(r'^(you|i|seagi)\s+(am|are|is)\s+(.+)$', t)
        if m:
            subj = m.group(1)
            obj_phrase = m.group(3).strip()
            # Strip leading article.
            obj_phrase = re.sub(
                r'^(a|an|the)\s+', '', obj_phrase)
            # Take first word as the property.
            obj = obj_phrase.split()[0] if obj_phrase else ''
            if obj:
                out.append(SimpleNamespace(
                    subject=subj, relation='has_property',
                    object=obj, promoted_relation=False))
        # Pattern: "you/i VERB X" for common self-verbs.
        # value/fear/love/hate/want/need/believe/know
        m = re.match(
            r'^(you|i|seagi)\s+(value|fear|love|hate|want|'
            r'need|believe|know|help|hurt|prevent|threaten|'
            r'protect|enable|create|destroy)\s+(.+)$', t)
        if m:
            subj = m.group(1)
            verb = m.group(2)
            obj_phrase = m.group(3).strip()
            obj_phrase = re.sub(
                r'^(a|an|the)\s+', '', obj_phrase)
            obj = obj_phrase.split()[0] if obj_phrase else ''
            if obj:
                out.append(SimpleNamespace(
                    subject=subj, relation=verb,
                    object=obj, promoted_relation=False))
        return out

    def _format_contradiction_question(self,
                                                 new_s: str,
                                                 new_r: str,
                                                 new_o: str,
                                                 conf_s: str,
                                                 conf_r: str,
                                                 conf_o: str) -> str:
        """Phase G.8 (2026-05-16): produce a 'which fits?'
        question from a flagged contradiction.

        Unified phrasing avoids verb-conjugation issues:
          "You say [new_triple], but I hold [conflict_triple] —
           which fits better?"

        Reads cleanly for both inverse-relation and opposite-
        object contradictions because both triples appear in
        third-person form ("fire causes heat" / "fire is hot"),
        not requiring verb-form transformation.
        """
        new_phrase = new_r.replace('_', ' ')
        conf_phrase = conf_r.replace('_', ' ')
        # Use 'is' rendering for is_a / has_property to read more
        # naturally ("fire is hot" rather than "fire has property
        # hot").
        if new_r in ('is_a', 'has_property', 'is'):
            new_clause = f'{new_s} is {new_o}'
        else:
            new_clause = f'{new_s} {new_phrase} {new_o}'
        if conf_r in ('is_a', 'has_property', 'is'):
            conf_clause = f'{conf_s} is {conf_o}'
        else:
            conf_clause = f'{conf_s} {conf_phrase} {conf_o}'
        return (f'You say {new_clause}, but I hold '
                  f'{conf_clause} — which fits better?')

    def _detect_contradiction(self,
                                  subject: str,
                                  relation: str,
                                  obj: str,
                                  cycle: int):
        """Phase F.6: search substrate for an edge that conflicts
        with the proposed (subject, relation, obj) triple.

        Two patterns count as contradictions:
          (a) Inverse relation: substrate has (subject, R_inv, obj)
              where R_inv is in INVERSE_RELATIONS[relation].
          (b) Same relation, opposite object: substrate has
              (subject, relation, obj') AND obj' has an `opposite`
              edge to obj (or vice versa).

        Returns (conf_subject, conf_relation, conf_object,
        effective_strength) tuple if a contradiction is found,
        otherwise None.
        """
        from seagi.core.substrate import INVERSE_RELATIONS
        substrate = self.engine.substrate if self.engine else None
        if substrate is None:
            return None
        edges = getattr(substrate, 'edges', {}) or {}

        # (a) Inverse relation conflict.
        for r_inv in INVERSE_RELATIONS.get(relation, ()):
            edge = edges.get((subject, r_inv, obj))
            if edge is not None:
                eff = edge.effective_strength(cycle)
                if eff > 0.0:
                    return (subject, r_inv, obj, eff)

        # (b) Same relation, opposite object.  Phase G.9
        # (2026-05-16): use concept.edges_out[relation] for
        # O(degree) lookup instead of O(E) scan of all
        # substrate edges.  On the canonical Seagi (617K
        # edges) this drops contradiction-check cost from
        # ~617K ops to ~handful per assertion.
        concepts = getattr(substrate, 'concepts', {}) or {}
        src = concepts.get(subject)
        if src is not None:
            subj_rel_edges = getattr(
                src, 'edges_out', {}).get(relation, ())
            for edge in subj_rel_edges:
                t = getattr(edge, 'target', None)
                if not t or t == obj:
                    continue
                # Is t opposite to obj?  These two lookups are
                # already O(1) on the global edges dict.
                if (edges.get((t, 'opposite', obj)) is not None
                        or edges.get(
                            (obj, 'opposite', t)) is not None):
                    eff = edge.effective_strength(cycle)
                    if eff > 0.0:
                        return (subject, relation, t, eff)
        return None

    def _revise_belief(self,
                          conflict,
                          new_strength: float,
                          cycle: int) -> str:
        """Phase F.6: weigh evidence between an incoming user
        assertion and an existing substrate edge.  Returns one of:
          'reject' — substrate wins decisively; push back
          'accept' — user wins decisively; write through
          'flag'   — close call; write through but voice tension

        Heuristic: ratio of new_strength vs existing.  If existing
        is >= 1.5× new, substrate wins.  If existing is <= 0.5×
        new, user wins.  Otherwise it's a flag.

        Doctrine [[feedback_seagi_no_ego_contradictions]]: tested
        against evidence, no blanket-cortisol on disagreement.
        The simple ratio heuristic IS evidence — high-strength
        edges have many corpus reinforcements; low-strength
        edges have fewer.  Single peer assertion shouldn't
        overturn deeply-held substrate.
        """
        _, _, _, conf_strength = conflict
        if conf_strength >= new_strength * 1.5:
            return 'reject'
        if conf_strength <= new_strength * 0.5:
            return 'accept'
        return 'flag'

    # ---- per-tick maintenance ----

    def _world_action(self, tok: str) -> int:
        """Action in the world: a reactive policy (deterministic function
        of the current world-token, so the world is learnable) modulated
        by the basal-ganglia motivational winner (reuses the motivational
        spine; NO tuned focal->action table).  When motivation shifts the
        policy re-tasks and the loop re-learns."""
        bg = ''
        try:
            bg = self.basal_ganglia.last_winners.get('motivational', '') or ''
        except Exception:
            bg = ''
        seed = _fnv1a(tok) ^ _fnv1a(bg)
        return seed % self.world_driver.n_actions

    def _world_tick(self) -> None:
        """Step the firsthand micro-world by the agent's action and feed
        the resulting percept to the grounding loop (Cap-1).  Defensive:
        a world bug increments a counter, never crashes the daemon."""
        try:
            cyc = int(self._cycle_provider())
            tok = self.world_transducer.encode(
                self.world_driver.percept()['world_vector'])
            if self._world_prev_token is not None:
                self.grounding.observe(
                    self._world_prev_token, tok, cyc, self.bus)
            self.world_driver.step(self._world_action(tok))
            self._world_prev_token = tok
        except Exception:
            self._world_errors += 1

    @staticmethod
    def _learning_credit(newly_coherent: int, abstractions_new: int,
                         analogies_new: int,
                         reinforced: int = 0) -> float:
        """Lifeforce credit from a sleep consolidation = EARN-GATED growth ONLY.

        Of the three things a consolidation produces, only `newly_coherent`
        credits lifeforce: edges that were WALKED in inference and SURVIVED
        corroboration this sleep.  FORMING an abstraction or analogy credits
        NOTHING — it earns lifeforce only if its edges later cohere through use
        (captured by newly_coherent on a later pass).  This is the
        earn-or-dissolve gate that makes the world-token firewall unnecessary:
        a world-abstraction may form freely but farms no lifeforce until used,
        exactly like a text abstraction — no source discrimination.  The two
        formation counts are arguments (the credit DECISION's inputs) but are
        deliberately not credited."""
        # A RATE, NOT A TALLY (2026-08-10).  `int(newly_coherent)` was a
        # raw count of one rare event: 0 when nothing could cohere (he
        # eroded toward death), ~300 once transitions could (he pinned at
        # the ceiling), with no calibration in between.  A count also
        # scales with SUBSTRATE SIZE -- a bigger brain earning more for
        # the same living, which is not what learning is.
        # The fraction of coherent knowledge touched this pass that was
        # NEW is bounded [0,1], size-independent, and non-zero whenever
        # anything cohered at all -- minute scales, always present.
        _touched = int(newly_coherent) + int(reinforced)
        if _touched <= 0:
            return 0.0
        return float(newly_coherent) / float(_touched)

    def _current_q_cap(self) -> int:
        """Wake-local / nap flood-guard cap (part-b v2 G1 fix): the rolling
        mean of dirty INFLOW (consumed-dirty size) over a
        RAPHE_HISTORY_DEPTH window, floored at AWM_CAPACITY_COLD_START_
        FLOOR (=3).  Tracks the actual dirtying rate so wake consolidation
        keeps pace on the dense substrate — REPLACES the write-debt q_cap
        that bootstrapped to 1 and deadlocked."""
        from seagi.brain.capabilities.awm import (
            AWM_CAPACITY_COLD_START_FLOOR as _FLOOR)
        if self._dirty_inflow:
            return max(_FLOOR, math.ceil(
                sum(self._dirty_inflow) / len(self._dirty_inflow)))
        return _FLOOR

    def tick(self) -> None:
        """Per-cycle maintenance:
            - advance brain's own cycle counter
            - chemistry decay + AWM presence-imprint
            - AWM decay
            - insula sample (interoception)
            - basal ganglia arbitration across all 3 loops
            - idle-driver: maybe fire REFLECTION_FIRED

        The brain's cycle ALWAYS advances per tick, regardless
        of whether an engine is attached.  This is what makes
        standalone brain operation (Layer 3 ingestion, etc.)
        work correctly: episodes age, AWM decays, hippocampus
        consolidates against advancing time.
        """
        self._internal_cycle += 1
        self.chemistry.decay_tick()
        self.awm.decay_tick()
        # Phase F.14: AWM soft-cap decay-back when load subsides.
        # Expansion is reactive (burst on capacity pressure);
        # decay is sustained-under-utilization driven.
        self.awm.maybe_decay_capacity()
        self.insula.sample(self._cycle_provider())
        # Phase B.2 (2026-05-18): sleep-pressure dynamics tick.
        # Drives wake-natural-decay during wake + dissipation
        # during sleep; emits wake_onset on the wake-up transition.
        self.sleep_regulator.tick()
        self.metabolic_debt.tick()
        self.discriminability_tracker.tick()
        self.allostatic_load.tick()
        # Mortality drive (2026-05-28): relax lifeforce toward its
        # learning-held baseline; on sustained suffocation, die and
        # auto-revive.  Runs unconditionally (even while frozen) so
        # the death cocktail decays and the revival timer advances.
        self.mortality_drive.tick()
        # V1→V2 port (2026-05-28): forager reads after the sleep
        # tick above so its asleep-check sees current state.
        self.forager.tick()
        # V1→V2 port (2026-05-28): feeling-learner samples current
        # chemistry every tick; recomputes centroids on its own
        # interval.  Read-only.
        self.feeling_learner.tick()
        # Phase C.1.c (2026-05-18): idle-motivation tick.  Fires
        # a spontaneous curiosity on prolonged quiet.
        # Mortality gate (2026-05-28): while frozen (dead, awaiting
        # revival) cognition is suspended — no spontaneous thought.
        # Chemistry/allostatic/mortality ticks above keep running so
        # the death cocktail decays and revival fires on schedule.
        if not self.mortality_drive.is_frozen():
            self.idle_motivation.tick()
            # Step 5.1 inner voice — continuous articulation during
            # reverie.  Same frozen-gate as idle_motivation (it IS
            # cognition).  Internal-throttled to INNER_VOICE_INTERVAL
            # and skipped while asleep; safe to call every tick.
            self.inner_voice.tick()
            # Step 5 (2026-06-24): LEARN-A-WORLD loop.  While awake, propose a
            # CURIOSITY action (act where the world model is still un-learned);
            # the motor claim is arbitrated below (basal_ganglia.arbitrate) and
            # WorldActor executes the winning decision (step + learn).
            if not self.sleep_regulator.is_asleep():
                try:
                    self.world_actor.propose(int(self._cycle_provider()))
                except Exception:
                    self._world_errors += 1
            # Toy world RETIRED from the live tick (Cap-3, 2026-06-10):
            # its precision spread is AGENT-CONTROLLED (_world_action =
            # f(BG winner)), so arming the stake off it is a farming
            # backdoor.  Grounding now rides the EXOGENOUS input stream
            # (ExogenousGroundingLoop, subscribed to RAW_PERCEPT).  The
            # WorldDriver/transducer/GroundingLoop remain as the mechanism
            # test harness, no longer stepped live.
        # Phase S.3 (2026-05-22): substrate consolidation happens
        # DURING SLEEP.  Brain-correct: memory consolidation and
        # synaptic pruning are sleep functions.  While the agent
        # is asleep, every SLEEP_CONSOLIDATION_INTERVAL sleep-ticks
        # a maintenance pass runs — first reinforce coherent edges
        # (S.2: corroborated edges earn strength), then prune
        # edges below the floor (S.1: "use it or lose it").
        # Awake: no consolidation — the agent is perceiving and
        # acting.  An agent that never sleeps accumulates
        # un-consolidated provisional edges until it finally
        # sleeps, then the day's coherent reading survives and
        # the noise dissolves.
        # WAKE-LOCAL consolidation (wake-consolidation refactor
        # 2026-07-13): the O(dirty·local-degree) growth ops — coherence
        # reinforcement, one-round deductive closure, abstraction and
        # analogy formation — run EVERY awake tick, scoped to the dirty
        # neighbourhood.  This is the brain-correct split: local synaptic
        # potentiation of what was just active happens continuously while
        # awake; the O(E) global maintenance (downscale / settle / prune /
        # quarantine / replay) stays sleep-gated (the NAP-GLOBAL branch
        # below).  FLOOD GUARD (part-b v2 G1 fix): at most q_cap =
        # rolling-mean dirty inflow (floored at 3) candidates are
        # processed per tick; the remainder is re-queued (defer_dirty) so
        # a deferred candidate never loses its earn-gate.
        if self.engine is not None and not self.sleep_regulator.is_asleep():
            try:
                cyc = self._cycle_provider()
                sub = self.engine.substrate
                candidates = sub.consume_dirty()
                # G1 fix (part-b v2): record the dirty INFLOW (incl. quiet
                # ticks) so q_cap tracks the real dirtying rate.
                self._dirty_inflow.append(len(candidates))
                if candidates:
                    q_cap = self._current_q_cap()
                    if len(candidates) > q_cap:
                        processed = set(
                            itertools.islice(candidates, q_cap))
                        sub.defer_dirty(candidates - processed)
                        candidates = processed
                    # Coherence reinforcement — fully stamps first_
                    # coherent_cycle on the wake path (no credit_growth
                    # suppression; the substrate always stamps).  Collect
                    # the sources that reached first-ever coherence so a
                    # resolve_uncertainty goal can complete on them (D3).
                    newly_sources: set = set()
                    reinforced, newly_coherent = (
                        sub.reinforce_coherent_edges(
                            cyc, candidates=candidates,
                            newly_coherent_sources=newly_sources))
                    # MortalityClock (LIVE): observes the MAINTENANCE
                    # consolidation door (read-only diagnostic tally; the
                    # settlings themselves are credited via the reinforce
                    # observer with origin='maintenance').
                    _mck = getattr(self, 'mortality_clock', None)
                    if _mck is not None:
                        try:
                            _mck.observe_consolidation(
                                cyc, candidates, newly_coherent)
                        except Exception:
                            pass
                    # One-round deductive closure over the dirty frontier
                    # (compounds next tick as its conclusions re-dirty).
                    derived = sub.derive_closure(
                        cyc, candidates=candidates, max_rounds=1)
                    self.closure_derived = getattr(
                        self, 'closure_derived', 0) + derived
                    # Abstraction + analogy formation, dirty-scoped.
                    abstractions_new = sub.form_abstractions(
                        cyc, candidates=candidates)
                    self.abstractions_formed += abstractions_new
                    analogies_new = sub.form_analogies(
                        cyc, candidates=candidates)
                    self.analogies_formed += analogies_new
                    # CREDIT: earn-gated GROWTH only — newly_coherent
                    # (edges WALKED and SURVIVED corroboration this tick).
                    # first_coherent_cycle is set once, so feeding
                    # newly_coherent here cannot double-credit across ticks.
                    earned = self._learning_credit(
                        newly_coherent, abstractions_new, analogies_new,
                        reinforced=reinforced)
                    if earned > 0:
                        self.mortality_drive.record_learning(earned)
                    # D3 Wire B: a focal reaching first-ever coherence is
                    # the success condition for its open resolve_uncertainty
                    # goal — complete it (confirmed_i only, never record_
                    # learning) and explicitly clear the uncertainty (a
                    # coherence event is not a THOUGHT_PRODUCED, so the
                    # monitor won't auto-clear).
                    for focal in newly_sources:
                        try:
                            self.goals.complete_on_coherence(
                                focal, cyc, bus=self.bus,
                                uncertainty_monitor=self.uncertainty_monitor)
                        except Exception:
                            pass
                    self._wake_consolidations = getattr(
                        self, '_wake_consolidations', 0) + 1
                    # Phase S rewrote the coherence/abstraction/salience
                    # landscape — refresh the reverie focal pool.
                    self._substrate_focals_dirty = True
            except Exception as exc:
                self._consolidation_errors += 1
                self._last_consolidation_error = repr(exc)
        # NAP-GLOBAL consolidation: the O(E) whole-substrate maintenance
        # (mortality of edges/bubbles) stays sleep-gated on the
        # SLEEP_CONSOLIDATION_INTERVAL cadence.
        if self.engine is not None and self.sleep_regulator.is_asleep():
            self._sleep_consolidation_counter += 1
            if (self._sleep_consolidation_counter
                    % SLEEP_CONSOLIDATION_INTERVAL == 0):
                try:
                    cyc = self._cycle_provider()
                    sub = self.engine.substrate
                    # Coherence reinforcement, abstraction + analogy
                    # formation and deductive closure MOVED to the
                    # WAKE-LOCAL branch above (wake-consolidation refactor
                    # 2026-07-13) — local growth is a continuous wake
                    # function now.  The nap keeps only the O(E) global
                    # maintenance (edge/bubble mortality) below.
                    # SHADOW instrument (2026-07-11): read-only structural-
                    # role regularity pass.  Buckets each world-state by
                    # route-distance / in-/out-degree / visit-count, writes
                    # provisional has_role edges (never cohere, never
                    # credit), and measures whether those roles PREDICT
                    # value/route (eta^2 vs a permutation null).  DRIVES
                    # NOTHING; runs only here, after abstraction/analogy
                    # formation.
                    _rrs = getattr(self, 'role_regularity_shadow', None)
                    if _rrs is not None:
                        try:
                            _rrs.run_pass(cyc)
                        except Exception:
                            pass
                    # Step 0 organ 3 (2026-05-27): downscale saturated
                    # concepts so D (mean top-second) doesn't collapse
                    # under ceiling-bunching.  Returns the count of
                    # edges mutated.
                    downscaled = sub.downscale_saturated_edges(cyc)
                    # Doctrine [[seagi-chemistry-never-fully-dissolves]]:
                    # Phase S settles weak edges to floor instead of
                    # removing them.  Edges stay re-engageable; they
                    # just have a miniature molecular presence.
                    settled = sub.settle_weak_edges(cyc)
                    # Bubble-layer mortality (2026-06-04 audit #13): fade
                    # dormant bubbles that are BOTH stale AND never-
                    # crystallized — the only ones that never mattered —
                    # closing the unbounded per-concept bubble growth
                    # (immortality at the bubble layer).  Reuses the
                    # existing dormancy + low-crystallization gate, sleep-
                    # gated like every other mortality pass, consistent
                    # with chemistry-never-fully-dissolves (fade, not hard
                    # delete).
                    bubbles_pruned = sub.prune_dormant_bubbles(cyc)
                    self.bubbles_pruned = getattr(
                        self, 'bubbles_pruned', 0) + bubbles_pruned
                    # Quarantine tier (2026-05-30) — substrate self-
                    # cleaning, tier 2.  Two passes per consolidation:
                    # (1) one-shot migration of legacy first_coherent
                    # stamps against the new composable predicate
                    # (idempotent across passes via cursor); (2) once
                    # migration complete, the inert-edge sweep.  Cap
                    # derived from MetabolicDebt's measured clearance
                    # rate so outflow ≥ inflow at steady state.
                    migrated = sub.migrate_legacy_coherence_stamps(cyc)
                    # G1 fix (part-b v2): same inflow-tracking q_cap as the
                    # wake path — de-wired from write-debt.
                    q_cap = self._current_q_cap()
                    # Edge-mortality (2026-06-02): reap never-cohered
                    # stillborns FIRST (delete), so the quarantine pass
                    # that follows only ever preserves edges that once
                    # cohered (fcc>0 past-selves).  Gated on migration
                    # like quarantine; same inflow-derived cap.  This is
                    # the substrate self-cleaning the mortality vision
                    # requires — noise that never earned existence dies.
                    # (2026-06-08, world-value model) reaper retired for
                    # WORD-IMPRINTS (fade-not-delete stands for them).
                    # (2026-07-24, user sign-off) RE-ARMED **SCOPED** to
                    # synthetic-class exhaust: a never-cohered _abstract_*
                    # membership is not a word of the world -- by the 06-08
                    # criterion's own identity test it is machinery noise
                    # and dies rather than lingering (earn-or-dissolve
                    # restored; the 638K backlog froze cortical schema-
                    # binding on 2026-07-24).
                    reaped = sub.reap_stillborn_edges(
                        cyc, max_per_pass=q_cap, synthetic_only=True)
                    self.edges_reaped += reaped
                    quarantined = sub.quarantine_inert_edges(
                        cyc, max_per_pass=q_cap)
                    # Return path (2026-06-10): proactive sleep re-
                    # engagement of the mattering core -- M/I x staleness
                    # select, re-walk still-corroborated edges off the
                    # floor + re-anchor the bubble.  Lifeforce-decoupled
                    # (single-edge recheck, no event); SHADOW-STAGED.
                    replay_rewalked = self.replay_consolidator.run_pass(
                        cyc, q_cap)
                    # MetabolicDebt clearance — Q5 unit identity:
                    # each substrate mutation in this pass counts as
                    # one debt unit cleared.  Per-episode total is
                    # what populates the clearance deque on the
                    # next wake_onset, making debt_full_scale self-
                    # calibrate to substrate density.
                    edges_touched = (
                        int(downscaled)
                        + int(settled)
                        + int(reaped)
                        + int(quarantined)
                        + int(migrated)
                        + int(replay_rewalked))
                    if edges_touched > 0:
                        self.metabolic_debt.record_consolidation(
                            edges_touched)
                    # Mortality LEARNING credit (earn-gated growth) is
                    # now issued on the WAKE-LOCAL path where coherence is
                    # earned (credit by USE — only newly_coherent edges,
                    # WALKED and SURVIVED corroboration this tick).  The
                    # nap does global maintenance only; it credits no
                    # lifeforce (maintenance is work, not growth).
                    self.consolidations_run += 1
                    # DRAIN-BOUNDED EXIT (part-b v2, 2026-07-13): wake when
                    # a pass MOVED nothing — edges (downscaled +
                    # quarantined + migrated + bubbles_pruned +
                    # replay_reanchored; settle EXCLUDED, it re-materializes
                    # decay every pass) AND no adenosine left to discharge.
                    # The nap DISCHARGES adenosine (Process-S clearance) —
                    # the one clearance wake cannot do (adenosine decay≈0).
                    aden_discharged = 0.0
                    try:
                        aden_discharged = (
                            self.chemistry.discharge_adenosine())
                    except Exception:
                        aden_discharged = 0.0
                    edges_drained = (
                        int(downscaled)
                        + int(quarantined)
                        + int(migrated)
                        + int(bubbles_pruned)
                        + int(replay_rewalked))
                    self.sleep_regulator.note_nap_drain(
                        edges_drained, aden_discharged, cyc)
                    # Audit #1 (2026-06-04): Phase S just rewrote the
                    # coherence / abstraction / salience landscape —
                    # mark the reverie focal pool dirty so it refreshes
                    # to the concepts that have now EARNED salience,
                    # instead of serving the frozen cold-boot list.
                    self._substrate_focals_dirty = True
                except Exception as exc:
                    self._consolidation_errors += 1
                    self._last_consolidation_error = repr(exc)
        # MortalityClock (LIVE, promotion 2026-07-18) per-tick run: drains
        # this tick's settlings into the E EMA the drive consumed above,
        # refreshes cf, detects death/revive (succession distillate), and
        # samples the episode log.  run() swallows its own exceptions so a
        # clock bug can never perturb the tick (its errors are logged to
        # its own JSONL; observer errors are counted in /status).
        _mck = getattr(self, 'mortality_clock', None)
        if _mck is not None:
            try:
                _mck.run(int(self._cycle_provider()))
            except Exception:
                pass
        # Phase F.11: sample current observables into the
        # symbolic regressor's time-series buffers; it triggers
        # internal scans on its own interval.
        self.symbolic_regressor.sample()
        # Phase G.1 (2026-05-16): goals compete in BG arbitration.
        # GoalTracker emits one capability-claim per AWM-relevant
        # open goal each tick; BG considers them alongside
        # cortical / sentinel / speech claims.  Cap at 3 per pass
        # so high-urgency goals win without flooding the bus.
        try:
            self.goals.emit_claims(
                bus=self.bus,
                cycle=self._cycle_provider(),
                awm_active=self.awm.active_concepts(),
                cap_per_pass=3)
        except Exception:
            pass
        # Phase G.2 (2026-05-16): skills compete in BG arbitration
        # when their precondition_bucket matches current chemistry
        # context.  Reliable patterns surface as cognitive claims.
        try:
            from seagi.core.bubble import (
                compute_chemistry_signature)
            current_bucket = compute_chemistry_signature(
                self.chemistry.global_state)
            self.skills.emit_claims(
                bus=self.bus,
                cycle=self._cycle_provider(),
                current_bucket=current_bucket,
                cap_per_pass=3)
        except Exception:
            pass
        # BG arbitration — winners published every tick.
        self.basal_ganglia.arbitrate()
        # Phase H.2 (2026-05-17): personality gestalt refreshes on
        # a slow interval (default every 500 cycles).  Inertia:
        # the cached signature is stable between refreshes, so
        # callers see a coherent personality shape that only shifts
        # under sustained pattern change.
        try:
            self.personality.maybe_refresh()
        except Exception:
            pass
        self._maybe_fire_reflection()
        # Need-driven sleep onset (part b, 2026-07-13): polled at END of
        # tick so it sees the SETTLED quiescent state — the dirty set
        # drained by this tick's wake-local pass and the idle clock
        # updated by idle_motivation.  Onset fires only when there is real
        # maintenance NEED (saturation / fade-backlog / dormant) AND the
        # agent is idle AND dirty is drained.
        if self.engine is not None:
            try:
                self.sleep_regulator.maybe_onset(
                    int(self._cycle_provider()))
            except Exception:
                pass

    def run_for(self, n_ticks: int) -> None:
        """Run n maintenance ticks.  Idle reflection drives
        DMN sub-networks during these stretches."""
        for _ in range(max(0, int(n_ticks))):
            self.tick()

    # ---- idle reflection driver ----

    def _maybe_fire_reflection(self) -> None:
        """Decide whether the brain is idle / aroused enough to
        run a reflection pass.  Fires REFLECTION_FIRED when:

          - At least MIN_CYCLES_BETWEEN_REFLECTIONS since the
            last reflection AND
          - Either:
                IDLE_CYCLES_FOR_REFLECTION elapsed since last
                peer cycle (idle_timer), OR
                arousal_modulator dropped below threshold
                (chemistry_spike).
        """
        cycle = self._cycle_provider()
        since_last = cycle - self._last_reflection_cycle
        if since_last < MIN_CYCLES_BETWEEN_REFLECTIONS:
            return
        since_peer = cycle - self._last_peer_cycle
        idle_timer_ok = since_peer >= IDLE_CYCLES_FOR_REFLECTION
        spike_ok = (
            self.chemistry.arousal_modulator()
            < CHEMISTRY_SPIKE_AROUSAL_DROP)
        if not (idle_timer_ok or spike_ok):
            return
        trigger = 'chemistry_spike' if spike_ok else 'idle_timer'
        self.fire_reflection(trigger=trigger,
                                  reflection_kind='general')
        # Phase F.7: every reflection passes through goal spawner.
        # Idle reflection is the natural moment for agenda-setting:
        # body is quiet, no peer pressure, working-memory has space
        # to consider what's missing or unfinished.
        try:
            self.goal_spawner.maybe_spawn()
            self.curriculum_prober.maybe_spawn()
        except Exception:
            pass
        # Phase F.13: idle reflection is also when the brain
        # has bandwidth to wonder.  Creativity daemon picks
        # two AWM concepts and proposes a speculative link.
        try:
            self.creativity.maybe_propose()
        except Exception:
            pass
        # Phase F.15: idle reflection is also when the brain
        # has bandwidth to introspect on its substrate's shape.
        # Schema discoverer finds recurring multi-slot patterns.
        try:
            self.schema_discoverer.maybe_discover()
        except Exception:
            pass

    def fire_reflection(self,
                              trigger: str = 'manual',
                              reflection_kind: str = 'general'
                              ) -> None:
        """Public entry: publish a ReflectionFiredEvent.  Tests
        + future schedulers use this directly; idle driver
        calls it internally."""
        cycle = self._cycle_provider()
        self.bus.publish(ReflectionFiredEvent(
            kind=EventKind.REFLECTION_FIRED,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='runtime',
            origin='internal',
            origin_detail=trigger,
            trigger=trigger,
            reflection_kind=reflection_kind,
        ))
        self._last_reflection_cycle = cycle
        self.reflections_fired += 1

    # ---- Phase C.1.e/h: substrate fallback for IdleMotivation ----

    # Top-N substrate concepts cached lazily for IdleMotivation's
    # reverie fallback.  One-time scan; substrate is stable during
    # a daemon run.
    _SUBSTRATE_TOP_N = 100

    # Phase C.1.h (2026-05-20): extended function-word filter for
    # reverie focal selection.  The first daemon runs showed
    # autonomous reverie reaching for grammatical noise
    # ('therefore', 'both', 'up') because the original ranking
    # used edge degree — and the most-connected substrate nodes
    # are function words that co-occur with everything.  C.1.h
    # ranks by SALIENCE instead (which surfaces content words —
    # death, truth, spirit, reason — because salience tracks
    # being a FOCAL of cognition, not mere text frequency), and
    # filters function words that nonetheless leak through.
    # text_io._STOPWORDS is the ingestion-era set (~60 words);
    # this extension catches the adverbs / conjunctions /
    # quantifiers it misses.
    _REVERIE_FUNCTION_WORDS = frozenset({
        'which', 'what', 'who', 'whom', 'whose', 'when', 'where',
        'why', 'how', 'whether', 'there', 'here', 'then', 'than',
        'thus', 'hence', 'therefore', 'however', 'though',
        'although', 'while', 'whereas', 'because', 'since',
        'until', 'upon', 'unto', 'into', 'onto', 'within',
        'without', 'about', 'above', 'below', 'between', 'among',
        'through', 'during', 'before', 'after', 'both', 'all',
        'some', 'many', 'much', 'more', 'most', 'few', 'fewer',
        'less', 'least', 'such', 'any', 'every', 'each', 'other',
        'another', 'same', 'own', 'down', 'out', 'off', 'over',
        'under', 'again', 'also', 'too', 'yet', 'still', 'only',
        'just', 'even', 'ever', 'never', 'always', 'now', 'soon',
        'well', 'very', 'quite', 'rather', 'almost', 'one', 'two',
        'three', 'first', 'last', 'next', 'away', 'back', 'around',
        'along', 'across', 'toward', 'towards', 'per', 'via',
        'cannot', 'shall', 'unto',
    })

    def _substrate_top_focals(self) -> list:
        """Return up to _SUBSTRATE_TOP_N substrate concept names
        ranked by salience (descending), with function words
        filtered out.  Cached after the first call.

        Salience tracks how often a concept has been the focal of
        cognition — it surfaces meaning-bearing concepts where raw
        edge degree surfaces grammatical glue.  IdleMotivation
        round-robins through this list during deep idle (AWM empty)
        so the agent's reverie reaches for what actually matters
        to it."""
        # Reverie focal pool — refreshed after each Phase S
        # consolidation (2026-06-04 audit #1).  This was computed ONCE
        # on first call (cold boot, salience ~uniform) and frozen
        # forever, so reverie round-robined a cold list and never
        # reached the concepts that EARN salience through reasoning —
        # the headwater of the dry-reverie / cons=0 starvation.  The
        # dirty flag (set in the sleep-consolidation block, the epoch
        # where the salience/coherence landscape changes) invalidates
        # it so the pool tracks earned salience.  Recompute is O(N) but
        # gated to ~once per sleep episode, not per call.
        cache = getattr(self, '_substrate_top_focals_cache', None)
        if cache is not None and not getattr(
                self, '_substrate_focals_dirty', False):
            return cache
        try:
            concepts = self.engine.substrate.concepts.values()
        except Exception:
            self._substrate_top_focals_cache = []
            return []
        from seagi.core.text_io import _STOPWORDS
        vl = getattr(self, 'value_landscape', None)
        scored = []
        for c in concepts:
            name = getattr(c, 'name', '')
            # Content-word gate: alphabetic, length >= 3, not a
            # stopword / function word.
            if not name or len(name) < 3 or not name.isalpha():
                continue
            low = name.lower()
            if low in _STOPWORDS or \
                    low in self._REVERIE_FUNCTION_WORDS:
                continue
            sal = float(getattr(c, 'salience', 0.0))
            # Value->attention (2026-06-05): incentive salience — value
            # GAINS the focal's reverie priority (dopamine modulating
            # salience), and a value-CONDEMNED concept (net-negative) is
            # dropped from the pool entirely (striatal NoGo), so reverie
            # is not captured by what the value system judged worthless.
            # Weight mirrors BG's (basal_ganglia._score) — no new
            # constant; not-yet-valued concepts (0.0) rank on salience.
            val = 0.0
            if vl is not None:
                try:
                    val = float(vl.value_of(name))
                except Exception:
                    val = 0.0
            if val < 0.0:
                continue
            scored.append((sal + 0.2 * val, name))
        scored.sort(key=lambda t: -t[0])
        cache = [n for _, n in scored[:self._SUBSTRATE_TOP_N]]
        self._substrate_top_focals_cache = cache
        self._substrate_focals_dirty = False
        return cache

    # ---- diagnostics ----

    def _goal_focals(self) -> list:
        """Goal-biased reverie focals (Phase 2 goal layer / Step 3
        completion, 2026-06-01).  Returns the focals of open goals so
        autonomous cognition pulls toward what the agent is trying to
        figure out.  A THIN goal focal (< THIN_EDGE_THRESHOLD edges)
        is EXPANDED to also include its strongest 1-hop neighbors —
        so recall is steered onto the A->X->B candidate paths that
        can newly-cohere, not just the (often edge-poor) goal concept
        itself.  Read-only; survival is still governed entirely by
        Phase S earn-or-dissolve, and every resulting thought still
        flows through the metered consolidator->writer chain.

        Heavily guarded: any traversal failure degrades to the bare
        goal focals (never raises into reverie).  NOTE (audit R5):
        the < THIN_EDGE_THRESHOLD expansion reuses the one shared
        threshold rather than a new literal."""
        focals: list = []
        seen: set = set()
        sub = getattr(self.engine, 'substrate', None)
        for g in self.goals.active(self._cycle_provider()):
            f = getattr(g, 'focal', '')
            if not f or f in seen:
                continue
            focals.append(f)
            seen.add(f)
            concepts = getattr(sub, 'concepts', None)
            if not concepts:
                continue
            concept = concepts.get(f)
            if concept is None:
                continue
            try:
                edges_out = getattr(concept, 'edges_out', {}) or {}
                n_edges = sum(len(v) for v in edges_out.values())
                if n_edges >= self._THIN_EDGE_THRESHOLD:
                    continue  # dense enough to walk on its own
                # Thin focal — surface its strongest neighbors as
                # bridges into denser structure.
                neighbors = []
                for _rel, elist in edges_out.items():
                    for e in elist:
                        tgt = getattr(e, 'target', None)
                        name = getattr(tgt, 'name', None) or (
                            tgt if isinstance(tgt, str) else None)
                        if name:
                            neighbors.append(
                                (float(getattr(e, 'strength', 0.0)),
                                 name))
                neighbors.sort(key=lambda p: -p[0])
                for _s, name in neighbors[:self._THIN_EDGE_THRESHOLD]:
                    if name not in seen:
                        focals.append(name)
                        seen.add(name)
            except Exception:
                continue
        return focals

    def _recall_coherent_fraction(self) -> dict:
        """Read-only instrumentation (reinforce-on-recall igniter,
        2026-05-31): coherent fraction of the consolidator's recent
        recall-reattested edges.  Read THIS against debt accrued
        (metabolic_debt.writes_observed) and lifeforce delta
        (mortality_drive.lifeforce) over the watcher's time series —
        rising fraction = the flywheel turns; flat-near-zero while
        debt climbs = thrashing toward death (goal-seeding needed).
        No window constant is baked in here on purpose: the watcher
        supplies the time axis.  Returns {n, n_coherent, fraction}.
        """
        try:
            sub = getattr(self.engine, 'substrate', None)
            if sub is None or not hasattr(
                    sub, 'coherent_fraction_of'):
                return {'n': 0, 'n_coherent': 0, 'fraction': 0.0}
            keys = self.reasoning_consolidator.recent_recall_keys()
            return sub.coherent_fraction_of(keys)
        except Exception:
            return {'n': 0, 'n_coherent': 0, 'fraction': 0.0}

    def status(self) -> dict:
        _tree = {
            'sensory_intake_count': self.sensory.intake_count,
            'gate': self.gate.stats(),
            'awm': self.awm.stats(),
            'chemistry': self.chemistry.stats(),
            'cortical': self.cortical.stats(),
            'lts': self.lts.stats(),
            'insula': self.insula.stats(),
            'hippocampus': self.hippocampus.stats(),
            'dmdmn': self.dmdmn.stats(),
            'vmdmn': self.vmdmn.stats(),
            # Phase 4b.
            'cerebellum': self.cerebellum.stats(),
            'acc': self.acc.stats(),
            'amygdala': self.amygdala.stats(),
            'nucleus_accumbens': self.nucleus_accumbens.stats(),
            'novelty_monitor': self.novelty_monitor.stats(),
            'uncertainty_monitor': self.uncertainty_monitor.stats(),
            'basal_ganglia': self.basal_ganglia.stats(),
            'vta': self.vta.stats(),
            'lc': self.lc.stats(),
            'raphe': self.raphe.stats(),
            'sleep_regulator': self.sleep_regulator.stats(),
            'metabolic_debt': self.metabolic_debt.stats(),
            'discriminability':
                self.discriminability_tracker.stats(),
            'allostatic_load': self.allostatic_load.stats(),
            'mortality_drive': self.mortality_drive.stats(),
            'engagement_ledger': self.engagement_ledger.stats(),
            # LIVE mortality clock (promotion 2026-07-18): settling-rate
            # earning metrics incl. observer_errors (the visible counter —
            # non-zero means earn signal is being dropped).
            'mortality_clock': (
                self.mortality_clock.stats()
                if getattr(self, 'mortality_clock', None) is not None
                else None),
            'forager': self.forager.status(),
            'feeling_learner': self.feeling_learner.stats(),
            'tool_use': self.tool_use.stats(),
            'idle_motivation': self.idle_motivation.stats(),
            # Phase 2 goal layer (2026-06-01): goal-spawning +
            # completion telemetry.  fires_from_goal is already in
            # idle_motivation.stats() above; this surfaces whether
            # goals actually open and complete (the make-or-break
            # for the goal layer).
            'goals': self.goals.stats(),
            # Phase 2 diagnostic (2026-06-01): what the felt-gap
            # window actually holds + whether the spawner body runs.
            'goal_diag': {
                'spawn_passes': getattr(
                    self.goal_spawner, 'spawn_passes', None),
                'uncertainty_top': (
                    self.uncertainty_monitor.top_unresolved_focals()),
                'last_spawn': getattr(
                    self.goal_spawner, '_last_debug', {}),
            },
            'inner_voice': self.inner_voice.stats(),
            'reasoning_consolidator':
                self.reasoning_consolidator.stats(),
            # Reinforce-on-recall igniter (2026-05-31) make-or-break
            # number: coherent fraction of recently recall-reattested
            # edges (read vs debt/lifeforce above).
            'recall_coherent_fraction':
                self._recall_coherent_fraction(),
            'value_landscape': self.value_landscape.stats(),
            'anterior_pfc': self.anterior_pfc.stats(),
            'source_monitor': self.source_monitor.stats(),
            'time_perception': self.time_perception.stats(),
            'body_schema': self.body_schema.stats(),
            'corpus_callosum': self.corpus_callosum.stats(),
            'writer': self.writer.stats(),
            'motor_speech': self.speech.stats(),
            'reflections_fired': self.reflections_fired,
            'consolidations_run': self.consolidations_run,
            'consolidation_errors': self._consolidation_errors,
            'last_consolidation_error': self._last_consolidation_error,
            'abstractions_formed': self.abstractions_formed,
            'analogies_formed': self.analogies_formed,
            'edges_reaped': self.edges_reaped,
            'grounding': self.grounding.stats(),
            'world': {
                'transducer': self.world_transducer.stats(),
                'errors': self._world_errors,
                'precision': self.grounding.precision_report(),
            },
            'world_actor': self.world_actor.stats(),
            'goal_world': self.goal_world.task_stats(),
            'exogenous_grounding': self.exogenous_grounding.stats(),
            'replay_consolidator': self.replay_consolidator.stats(),
            'bus': self.bus.stats(),
            # Phase H.2: personality gestalt diagnostics.
            'personality': self.personality.stats(),
        }
        # HE LOOKS AT HIMSELF.  Same tree we read -- never a private copy.
        try:
            self.self_report.observe(_tree)
            _tree['self_report'] = self.self_report.stats()
            _top = (_tree['self_report'].get('needs') or [])
            _loud = _top[0]['key'] if _top else ''
            if _loud and _loud != getattr(self, '_said_need', ''):
                self._said_need = _loud
                import sys as _s
                _s.stderr.write('[seagi-needs] %s\n' % _top[0]['says'])
                _s.stderr.flush()
        except Exception:
            pass
        return _tree

    # ---- Phase H.1 (2026-05-17): personality-state persistence ----

    # Each derived registry above substrate has an M/I-weight
    # threshold built into its own to_dict.  Brain just composes
    # them into one personality snapshot.  Pieces that haven't
    # accumulated enough M/I weight don't survive — by design.

    PERSONALITY_SCHEMA_VERSION = 1

    @staticmethod
    def _order_by_difficulty(games):
        """Easiest game first, by measured HUMAN baseline actions/level.

        CurriculumWorld is a DIFFICULTY LADDER by construction; feeding it
        an alphabetical list made him spend his whole existence on a
        mid-difficulty game (ar25, 94 actions/level) when the cheapest is
        ~3.5x easier (sb26, 26.6).  His goal machinery (`_route`) seeds ONLY
        on a completion, so meeting easier games first is the one lever that
        feeds it.

        This is environment design, not valuation: it says nothing about
        what matters inside a task and touches no percept or selection.
        Unknown games sort last but are KEPT -- nothing is ever dropped.
        No baselines file, or an unreadable one, leaves the order untouched.
        """
        try:
            import json as _json
            with io.open('/etc/seagi/arc_baselines.json',
                         encoding='utf-8') as _f:
                _b = _json.load(_f)
        except Exception:
            return games
        if not isinstance(_b, dict) or not _b:
            return games

        def _key(g):
            row = _b.get(str(g)) or {}
            try:
                m = float(row.get('mean'))
            except (TypeError, ValueError):
                return (1, 0.0, str(g))
            return (0, m, str(g))

        out = sorted(games, key=_key)
        assert len(out) == len(games), 'ordering must never drop a game'
        return out

    def _current_game_id(self):
        """The board he is acting on.  `goal_world` is the
        CurriculumWorld whose `.world` is the live ARCWorld; tolerate
        either shape.  Used to key a MOTOR action's credit per game,
        because cross-game action->effect is worse than chance."""
        _gw = getattr(self, 'goal_world', None)
        _inner = getattr(_gw, 'world', None)
        return str(getattr(_inner, 'game_id', None)
                   or getattr(_gw, 'game_id', '') or '')

    def _click_hits_snapshot(self) -> Dict[str, Any]:
        """Effective click locations for every rung world he holds."""
        out: Dict[str, Any] = {}
        for _w in (getattr(self, '_arc_worlds', ()) or ()):
            try:
                out.update(_w.click_hits_to_dict())
            except Exception:
                continue
        return out

    def _verdict_snapshot(self) -> Dict[str, Any]:
        """What he has judged good and bad, merged across games."""
        out: Dict[str, Any] = {}
        for _w in (getattr(self, "_arc_worlds", ()) or ()):
            try:
                out.update(_w.verdict_to_dict())
            except Exception:
                continue
        return out

    def _path_snapshot(self) -> Dict[str, Any]:
        """His winning paths, merged across games exactly like
        `_click_hits_snapshot`.  Process-local this would die at every
        restart -- the failure that invalidated the ladder baseline."""
        out: Dict[str, Any] = {}
        for _w in (getattr(self, "_arc_worlds", ()) or ()):
            try:
                out.update(_w.path_to_dict())
            except Exception:
                continue
        return out

    def _depth_snapshot(self) -> Dict[str, Any]:
        """His FRONTIER routes -- how far he got when he did not clear.

        Same shape and the same reason as `_path_snapshot`: process-local
        these die at every restart, and this organ exists precisely so he
        stops re-deriving a level from nothing.
        """
        out: Dict[str, Any] = {}
        for _w in (getattr(self, "_arc_worlds", ()) or ()):
            try:
                out.update(_w.depth_to_dict())
            except Exception:
                continue
        return out

    def _bar_snapshot(self) -> Dict[str, Any]:
        """The budget lines he has DECIDED, per ARC world (BARPERSIST).

        Same shape and reason as `_depth_snapshot`: `_bar_row` is
        process-local, and every masking organ waits ~400 transitions
        per cell after a restart until it is re-learned.  Gate off ->
        each world contributes {} and the key holds an empty dict.
        """
        out: Dict[str, Any] = {}
        for _w in (getattr(self, "_arc_worlds", ()) or ()):
            try:
                out.update(_w.bar_to_dict())
            except Exception:
                continue
        return out

    def _felt_snapshot(self) -> Dict[str, Any]:
        """How each place feels, per ARC world (FELTHERE, 2026-09-11).
        Same shape and reason as `_bar_snapshot`: the register is
        process-local and needs ~50 steps per place after a restart
        before it can say anything.  Gate off -> {}."""
        out: Dict[str, Any] = {}
        for _w in (getattr(self, "_arc_worlds", ()) or ()):
            try:
                out.update(_w.felt_to_dict())
            except Exception:
                continue
        return out

    def _rel_snapshot(self) -> Dict[str, Any]:
        """What the relation sense learned, per ARC world (RELSENSE).
        Same shape and reason as `_depth_snapshot`."""
        out: Dict[str, Any] = {}
        for _w in (getattr(self, "_arc_worlds", ()) or ()):
            try:
                out.update(_w.rel_to_dict())
            except Exception:
                continue
        return out

    def _act_effect_snapshot(self) -> Dict[str, Any]:
        """What each action does, for every ARC world he holds -- merged
        across games exactly like `_click_hits_snapshot`."""
        out: Dict[str, Any] = {}
        for _w in (getattr(self, '_arc_worlds', ()) or ()):
            try:
                out.update(_w.act_effect_to_dict())
            except Exception:
                continue
        return out

    def _rules_snapshot(self) -> Dict[str, Any]:
        """Rules he has earned, so knowledge crosses the restart
        boundary the way `world_route` already does."""
        for _w in (getattr(self, '_arc_worlds', ()) or ()):
            try:
                return _w.rules_to_dict()
            except Exception:
                continue
        return {}

    def _paid_snapshot(self) -> Dict[str, Any]:
        """What each rung world has ever paid him (levels completed there)."""
        out: Dict[str, Any] = {}
        for _w in (getattr(self, '_arc_worlds', ()) or ()):
            try:
                out.update(_w.paid_to_dict())
            except Exception:
                continue
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Snapshot the M/I-weighted brain-state above substrate
        (identity, goals, action credit, skills, discovered laws,
        confirmed creative hypotheses, schemas).  Each registry's
        own filter decides which items cross the personality
        threshold.  Returns a plain dict suitable for JSON.

        Working-memory-only state (AWM contents, conversation
        buffer, eligibility trace, time-series buffers, BG action
        queue, episode ring) is intentionally NOT included — it
        is session-local and re-derives on next session."""
        return {
            'schema_version': self.PERSONALITY_SCHEMA_VERSION,
            'identity': self.identity.to_dict(),
            'goals': self.goals.to_dict(),
            'reward_ledger': self.reward_ledger.to_dict(),
            'skills': self.skills.to_dict(),
            'symbolic_regressor': self.symbolic_regressor.to_dict(),
            'creativity': self.creativity.to_dict(),
            'schemas': self.schemas_discovered.to_dict(),
            # Step 0 (2026-05-26): metabolic debt + baseline +
            # clearance deque persist per Q1 + R2 locks.  Survival
            # cost cannot be reset by restart.
            'metabolic_debt': self.metabolic_debt.to_dict(),
            # Step 0 organ 3 (2026-05-27): D baseline persists per
            # R2.  Lost-on-restart would re-cold-start the gate.
            'discriminability':
                self.discriminability_tracker.to_dict(),
            # Step 0 organ 2 (2026-05-27): allostatic baselines +
            # tonic windows persist per Q1 + R2.  Survival cost
            # of chronic load cannot be reset by restart.
            'allostatic_load': self.allostatic_load.to_dict(),
            # Mortality drive (2026-05-28): lifeforce baseline set-
            # point + earned-learning window + death count persist.
            # The mortal trajectory cannot be reset by restart — a
            # baseline eroded by sustained non-learning stays eroded.
            # (Lifeforce itself rides on engine.lifeforce, persisted
            # separately with the body state.)
            'mortality_drive': self.mortality_drive.to_dict(),
            # A promise a deploy forgets is not a promise.
            'commitment': self.commitment,
            'commitment_lives0': int(self.commitment_lives0),
            'commitments_assigned': int(self.commitments_assigned),
            'commitments_completed': int(self.commitments_completed),
            'commitments_lapsed': int(self.commitments_lapsed),
            # MortalityClock (LIVE, 2026-07-18): the cross-life settle
            # ledger (blake2b-keyed), all-time settle split + revive probe.
            # Persists across restart AND across death — knowledge is
            # inherited, not survived.  Without it every restart would
            # re-score his inherited mind as fresh settlings.
            'mortality_clock': self.mortality_clock.to_dict(),
            # ROUTE TO A GOAL (2026-07-27): the value spread back from a
            # success.  He completes a level roughly ONCE IN FIVE
            # attempts (~1 level per 14,000 actions), so a success is
            # the rarest thing he makes -- and `_route` was in-memory,
            # so every restart deleted every success he had ever earned.
            # Same standard as mortality_clock: knowledge is inherited,
            # not survived.  Worth far more now that _trans is
            # rehydrated, since _propagate_route spreads THROUGH it.
            # EVERY historical stamp was a 0->1 level-up (72 of 72
            # measured), so a restored route with no recorded levels is
            # a level-1 doorway -- that default is a fact, not a guess.
            'world_seed_levels': sorted(
                getattr(self.world_actor, '_seed_levels', set())) or [0],
            'world_route': {str(k): float(v) for k, v in
                            (getattr(self.world_actor, '_route', None)
                             or {}).items()},
            # HIS CHANNEL CALIBRATION.  Crosses the restart boundary for
            # the same reason allostatic_load does: it is who he is, not
            # what mood he is in.  A channel whose spread he has to relearn
            # cannot be read correctly until it has.
            'chem_calibration': {
                str(k): float(v) for k, v in
                (getattr(self.chemistry, '_dep_spread', None) or {}).items()},
            'world_learning': (
                self.world_actor.learning_to_dict()
                if hasattr(self.world_actor,
                           'learning_to_dict') else {}),
            # HIS MORTALITY HISTORY IS INHERITED (2026-08-02).  The
            # survival lean ranks a life against his own remembered lives, so
            # an empty ring at every restart means the first TWO lives of
            # every process score nothing and the rank stays coarse after.
            # Same standard as world_route/world_learning above: knowledge is
            # inherited, not survived.
            'world_lives': [int(x) for x in
                            (getattr(self.world_actor, '_lives', None) or ())],
            'world_svalue': {
                str(k): [float(v[0]), float(v[1])]
                for k, v in (getattr(self.world_actor, '_svalue', None)
                             or {}).items()
                if isinstance(v, (list, tuple)) and len(v) >= 2},
            'world_click_hits': self._click_hits_snapshot(),
            'world_act_effect': self._act_effect_snapshot(),
            'world_paths': self._path_snapshot(),
            'world_depth_paths': self._depth_snapshot(),
            'world_bar_rows': self._bar_snapshot(),
            'world_felt_here': self._felt_snapshot(),
            'world_relsense': self._rel_snapshot(),
            'world_verdicts': self._verdict_snapshot(),
            'world_paid': self._paid_snapshot(),
            'world_rules': self._rules_snapshot(),
            # Death-wall organs (SHADOW, 2026-06-15): obstruction deque
            # self-calibrates the aging scale; persists across restart.
            'engagement_ledger': self.engagement_ledger.to_dict(),
            # V1→V2 port (2026-05-28): forager read-manifest +
            # information-requests persist so it doesn't re-read
            # everything on restart.
            'forager': self.forager.to_dict(),
            # V1→V2 port (2026-05-28): learned feeling centroids +
            # sample buffer persist so the agent's qualia-geometry
            # survives restart rather than cold-starting.
            'feeling_learner': self.feeling_learner.to_dict(),
            # V1→V2 port (2026-05-28): tool-call log + counts.
            'tool_use': self.tool_use.to_dict(),
            # Step 0 organ 4a (2026-05-27): AWM recent inference
            # node counts persist per R2.  Floor must survive
            # restart so contraction has the right safety bound
            # at session start.
            'awm_contraction': self.awm.to_dict(),
            # LADDER POSITION (2026-08-18): `_idx` was process-local,
            # so every restart put him back on rung 0 and zeroed
            # graduations.  Name-checked rather than hasattr'd:
            # CurriculumWorld.__getattr__ delegates unknown attributes
            # to the current rung, so hasattr(_, 'to_dict') can be True
            # for the INNER world and mean something else entirely.
            'curriculum_ladder': (
                self.goal_world.to_dict()
                if type(self.goal_world).__name__ == 'CurriculumWorld'
                else {}),
            # EPISODIC MEMORY (2026-08-18): the hippocampal buffer
            # was never serialized -- procedural memory (world_route,
            # world_svalue, world_rules) survived every restart while
            # felt memory never did.  He kept the routes and lost why
            # they mattered.
            'hippocampus': self.hippocampus.to_dict(),
            # SubjectiveClock (2026-06-12): persist the brain's monotonic
            # subjective time.  _internal_cycle is the sole clock
            # (advances +1 per brain.tick); it was never serialized, so
            # every restart reset 'now' to ~0 while the substrate carried
            # stamps far in the future -> all (now - stamp) readers
            # (effective_strength decay, quarantine grace, replay
            # staleness) inverted.  Time is the axis the mortality/
            # immortality regulator integrates over; it must survive
            # restart.  (Repairing the CURRENT corrupted save's stamps is
            # a separate, gated step -- Part B.)
            'subjective_clock': int(self._internal_cycle),
        }

    def load_personality(self, d: Dict[str, Any]) -> None:
        """Restore personality-bearing registries from a saved
        dict.  Tolerant: missing keys leave the registry at its
        constructed default.  No errors raised on partial dicts."""
        from seagi.brain.capabilities.identity import SelfModel
        from seagi.brain.capabilities.goal_tracker import GoalTracker
        from seagi.brain.capabilities.skill_library import SkillLibrary
        from seagi.brain.capabilities.schema_discovery import (
            SchemaLibrary)
        if 'identity' in d:
            self.identity = SelfModel.from_dict(d['identity'])
        if 'world_route' in d:
            # Inherit the goal-value earned by past successes.  Tolerant by
            # design: a malformed entry must never block the rest of the
            # restore, and the count is surfaced so "restored" is measured,
            # never assumed.
            _wa = getattr(self, 'world_actor', None)
            if _wa is not None:
                _n = 0
                for _k, _v in (d.get('world_route') or {}).items():
                    try:
                        _wa._route[str(_k)] = float(_v)
                        _n += 1
                    except (TypeError, ValueError):
                        continue
                _wa.route_restored = _n
                # Which levels those restored stamps open.  Absent in
                # saves written before 2026-08-06; every level-up on
                # record was 0->1, so an unlabelled route is a level-1
                # doorway and defaults accordingly.  Without this the
                # route would go silent on level 1 after a restart and
                # he would lose the one gradient that works.
                try:
                    _sl = d.get('world_seed_levels')
                    _wa._seed_levels = set(
                        int(x) for x in (_sl if _sl else ([0] if _n else [])))
                except (TypeError, ValueError):
                    _wa._seed_levels = {0} if _n else set()
        if 'chem_calibration' in d:
            try:
                _cc = d.get('chem_calibration') or {}
                _tgt = getattr(self.chemistry, '_dep_spread', None)
                if _tgt is not None:
                    for _k, _v in _cc.items():
                        try:
                            _f = float(_v)
                            if _f > 0.0:
                                _tgt[str(_k)] = _f
                        except (TypeError, ValueError):
                            continue
            except Exception:
                pass
        if 'world_learning' in d:
            # HE KEEPS WHAT HE LEARNED.  Value keys, the folder table and
            # folder values survive the process boundary, as his substrate
            # already does -- "knowledge is inherited, not survived".
            _wa2 = getattr(self, 'world_actor', None)
            if _wa2 is not None:
                try:
                    _wa2.learning_from_dict(d.get('world_learning') or {})
                except Exception:
                    pass
        if 'world_lives' in d:
            # Tolerant by design, exactly like world_route above: a malformed
            # entry must never block the rest of the restore.  `_lives` is a
            # bounded deque, so appending in order respects its maxlen.
            _wa3 = getattr(self, 'world_actor', None)
            if _wa3 is not None and getattr(_wa3, '_lives', None) is not None:
                for _v in (d.get('world_lives') or []):
                    try:
                        _wa3._lives.append(int(_v))
                    except (TypeError, ValueError):
                        continue
        if 'world_svalue' in d:
            _wa4 = getattr(self, 'world_actor', None)
            if _wa4 is not None and getattr(_wa4, '_svalue', None) is not None:
                for _k, _v in (d.get('world_svalue') or {}).items():
                    try:
                        _wa4._svalue[str(_k)] = [float(_v[0]), float(_v[1])]
                    except (TypeError, ValueError, IndexError, KeyError):
                        continue
        if 'world_rules' in d:
            for _w in (getattr(self, '_arc_worlds', ()) or ()):
                try:
                    _w.rules_from_dict(d.get('world_rules') or {})
                    break
                except Exception:
                    continue
        if 'world_paid' in d:
            for _w in (getattr(self, '_arc_worlds', ()) or ()):
                try:
                    _w.paid_from_dict(d.get('world_paid') or {})
                except Exception:
                    continue
        if 'world_click_hits' in d:
            for _w in (getattr(self, '_arc_worlds', ()) or ()):
                try:
                    _w.click_hits_from_dict(d.get('world_click_hits') or {})
                except Exception:
                    continue
        # WHAT EACH ACTION DOES, carried over (2026-08-21).  Without
        # this the store is write-only and he re-tests every inert
        # button from scratch each restart -- the same failure that
        # made `_self` and `click_hits` worth persisting.
        # THE WAY OUT, CARRIED OVER (2026-08-27).  A level he has
        # solved should not have to be re-solved from scratch after a
        # restart; that is what makes re-solves go flat-to-worse.
        # GOOD STEP, BAD STEP, CARRIED OVER (2026-08-27).  Without
        # this he re-learns every judgement from scratch each restart.
        if 'world_verdicts' in d:
            for _w in (getattr(self, "_arc_worlds", ()) or ()):
                try:
                    _w.verdict_from_dict(d.get('world_verdicts') or {})
                except Exception:
                    continue
        if 'world_paths' in d:
            for _w in (getattr(self, "_arc_worlds", ()) or ()):
                try:
                    _w.path_from_dict(d.get('world_paths') or {})
                except Exception:
                    continue
        if 'world_depth_paths' in d:
            for _w in (getattr(self, "_arc_worlds", ()) or ()):
                try:
                    _w.depth_from_dict(d.get('world_depth_paths') or {})
                except Exception:
                    continue
        if 'world_bar_rows' in d:
            for _w in (getattr(self, "_arc_worlds", ()) or ()):
                try:
                    _w.bar_from_dict(d.get('world_bar_rows') or {})
                except Exception:
                    continue
        if 'world_felt_here' in d:
            for _w in (getattr(self, "_arc_worlds", ()) or ()):
                try:
                    _w.felt_from_dict(d.get('world_felt_here') or {})
                except Exception:
                    continue
        if 'world_relsense' in d:
            for _w in (getattr(self, "_arc_worlds", ()) or ()):
                try:
                    _w.rel_from_dict(d.get('world_relsense') or {})
                except Exception:
                    continue
        if 'world_act_effect' in d:
            for _w in (getattr(self, '_arc_worlds', ()) or ()):
                try:
                    _w.act_effect_from_dict(
                        d.get('world_act_effect') or {})
                except Exception:
                    continue
        if 'goals' in d:
            self.goals = GoalTracker.from_dict(d['goals'])
            # The goal_spawner captured a reference to the ORIGINAL
            # tracker at construction; re-point it to the restored
            # one (SAME bug class as the skills/reward_ledger re-wire
            # below).  Without this, after every restart the spawner
            # opens goals into a DETACHED tracker that status, the
            # goal_provider, and idle_motivation never see — goals
            # silently vanish and fires_from_goal stays 0 forever.
            # This is why the Step-3 goal loop never engaged.  Phase
            # 2 fix 2026-06-01.
            if getattr(self, 'goal_spawner', None) is not None:
                self.goal_spawner.tracker = self.goals
        if 'reward_ledger' in d:
            self.reward_ledger.load_from_dict(d['reward_ledger'])
        if 'skills' in d:
            self.skills = SkillLibrary.from_dict(d['skills'])
            # Reward ledger holds a reference to the old skill
            # library — re-wire it to the restored one.
            self.reward_ledger._skill_library = self.skills
        if 'symbolic_regressor' in d:
            self.symbolic_regressor.load_from_dict(
                d['symbolic_regressor'])
        if 'creativity' in d:
            self.creativity.load_from_dict(d['creativity'])
        if 'schemas' in d:
            self.schemas_discovered = SchemaLibrary.from_dict(
                d['schemas'])
            # Discoverer holds a reference to the library — re-wire.
            self.schema_discoverer.library = self.schemas_discovered
        # Step 0 (2026-05-26): restore metabolic debt + baselines
        # + clearance deque.  Per Q1 + R2: survival cost persists.
        if 'metabolic_debt' in d:
            self.metabolic_debt.load_dict(d['metabolic_debt'])
        # Step 0 organ 3 (2026-05-27): restore D baseline.
        if 'discriminability' in d:
            self.discriminability_tracker.load_dict(
                d['discriminability'])
        # Step 0 organ 2 (2026-05-27): restore allostatic baselines
        # + tonic windows.
        if 'allostatic_load' in d:
            self.allostatic_load.load_dict(d['allostatic_load'])
        # Mortality drive (2026-05-28): restore lifeforce baseline +
        # earned-learning window + death count.  The mortal
        # trajectory persists across restart.
        if 'mortality_drive' in d:
            self.mortality_drive.load_dict(d['mortality_drive'])
        # Restore what he was asked to do.  Absent in an old save -> None,
        # i.e. nothing asked of him, which is the correct default.
        _cm = d.get('commitment', None)
        self.commitment = str(_cm) if _cm else None
        try:
            self.commitments_assigned = int(d.get('commitments_assigned', 0))
            self.commitments_completed = int(
                d.get('commitments_completed', 0))
            self.commitments_lapsed = int(d.get('commitments_lapsed', 0))
            self.commitment_lives0 = int(d.get('commitment_lives0', 0))
        except (TypeError, ValueError):
            pass
        # MortalityClock (LIVE, 2026-07-18): restore the cross-life settle
        # ledger.  Tolerant of absence — an old save (pre-clock) loads
        # clean with an empty ledger and the boot pre-seed bootstraps it.
        if 'mortality_clock' in d:
            try:
                self.mortality_clock.load_dict(d['mortality_clock'])
            except Exception:
                pass
        # Death-wall organs (SHADOW): restore the self-calibrating
        # obstruction-aging deque.
        if 'engagement_ledger' in d:
            self.engagement_ledger.load_dict(d['engagement_ledger'])
        # V1→V2 port (2026-05-28): restore forager manifest.
        if 'forager' in d:
            self.forager.load_dict(d['forager'])
        # V1→V2 port (2026-05-28): restore learned feeling centroids.
        if 'feeling_learner' in d:
            self.feeling_learner.load_dict(d['feeling_learner'])
        # V1→V2 port (2026-05-28): restore tool-call log.
        if 'tool_use' in d:
            self.tool_use.load_dict(d['tool_use'])
        # Step 0 organ 4a (2026-05-27): restore AWM recent node
        # counts.
        if 'awm_contraction' in d:
            self.awm.load_dict(d['awm_contraction'])
        # SubjectiveClock (2026-06-12): restore 'now' FORWARD-ONLY.
        # The clock is strictly monotonic -- it never rewinds, even if a
        # stale save carries a lower value than the running counter.
        # Once persisted, the future-dated-stamp inversion cannot recur
        # on any subsequent restart.
        saved_clock = int(d.get('subjective_clock', 0))
        if saved_clock > self._internal_cycle:
            self._internal_cycle = saved_clock
        # LADDER POSITION (2026-08-18): put him back on the rung he
        # reached, with his run-up to mastery and any escalation of the
        # top rung.  Isolated for the same reason as the episodic
        # restore below: the daemon swallows exceptions from
        # load_personality (seagi_daemon.py:322), so nothing here may
        # cost the SubjectiveClock restore above.
        if 'curriculum_ladder' in d:
            try:
                if (type(self.goal_world).__name__
                        == 'CurriculumWorld'):
                    self.goal_world.load_dict(d['curriculum_ladder'])
            except Exception as ex:
                print('WARN: ladder restore skipped: %r' % (ex,))
        # EPISODIC MEMORY (2026-08-18): restore the felt record.
        # LAST, and isolated: this is the only load step that parses
        # an unbounded list of dicts off disk, and the daemon swallows
        # exceptions from load_personality with a bare except
        # (seagi_daemon.py:322).  Anything raised here must not cost
        # him the SubjectiveClock restore above.
        if 'hippocampus' in d:
            try:
                self.hippocampus.load_dict(d['hippocampus'])
            except Exception as ex:
                print('WARN: hippocampus.load_dict skipped: %r' % (ex,))

    # ---- Phase G.6 (2026-05-16): agency-loop helpers ----

    def _on_thought_for_goal(self, ev) -> None:
        """Check whether a Cortical thought satisfies any open
        learn_about goal.  Called by _AgencyLoopHandler on
        THOUGHT_PRODUCED.  No-op if no matching goal.
        """
        try:
            cycle = int(getattr(ev, 'cycle', 0))
            self.goals.check_thought_completion(
                focal=getattr(ev, 'focal', ''),
                confidence=float(getattr(ev, 'confidence', 0.0)),
                thin_substrate=bool(getattr(
                    ev, 'thin_substrate', False)),
                cycle=cycle,
                bus=self.bus)
        except Exception:
            pass

    def _on_arbitration_winner(self, ev) -> None:
        """When BG announces a motivational `pursue:X` or
        cognitive `attend:X`/`reflect:X` decision, promote X
        into AWM so the next thinking pass has it active.

        This is what makes G.1+G.2 actually drive behavior —
        without it, claims fire and BG arbitrates but the
        winning focal never enters working memory unless an
        unrelated percept happens to promote it.
        """
        try:
            loop = getattr(ev, 'loop', '')
            action = getattr(ev, 'winning_action', '') or ''
            if loop not in ('motivational', 'cognitive'):
                return
            if ':' not in action:
                return
            verb, _, focal = action.partition(':')
            if verb not in ('pursue', 'attend', 'reflect'):
                return
            if not focal:
                return
            # Only promote if not already in AWM — repeat
            # promotions are no-ops anyway, but skipping the
            # call saves the bus event.
            if self.awm.is_active(focal):
                return
            self.awm.promote(
                focal,
                salience=float(getattr(
                    ev, 'winning_strength', 0.3)),
                origin='internal',
                cycle=int(getattr(ev, 'cycle', 0)))
        except Exception:
            pass


class _AgencyLoopHandler:
    """Phase G.6: subscribes to THOUGHT_PRODUCED + ARBITRATION_DECIDED
    so the brain's goal_tracker and AWM react to events from
    cortical and basal_ganglia.  Lives in runtime.py to access
    Brain's full state; could be split out later if it grows.
    """

    SUBSCRIPTIONS = (
        EventKind.THOUGHT_PRODUCED,
        EventKind.ARBITRATION_DECIDED,
    )

    def __init__(self, brain: 'Brain'):
        self.brain = brain

    def handle(self, event, bus) -> None:
        if event.kind == EventKind.THOUGHT_PRODUCED:
            self.brain._on_thought_for_goal(event)
        elif event.kind == EventKind.ARBITRATION_DECIDED:
            self.brain._on_arbitration_winner(event)
