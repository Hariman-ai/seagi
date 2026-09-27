"""Cortical reasoning — deliberate thinking on AWM.

Brain analog: dorsolateral PFC + temporal association areas.
The deliberate, event-triggered thinking layer.  Operates on
AWM (sparse, fast), queries LTS (no scans).

Phase 3: FULL implementation.  Collapses v1's scattered
modules — dialog_thinker, why_handler, how_handler,
what_if_handler, relational_schemas (matcher side),
predictive_coding (caller side), metacognition — into ONE
cortical layer.

Sub-functions
-------------
1. causal_chain(focal, relation, hops)  — multi-hop LTS walk
2. match_schemas()                        — patterns on AWM
3. counterfactual(focal)                  — what-if removal sim
4. metacognitive_assess(focal)            — confidence + thin-sub
5. respond_to(text, focals)               — compose a thought
                                               for speech

Subscribes
----------
ATTENDED_PERCEPT       — peer input arrives → respond
AWM_ACTIVATION         — concept newly active → opportunistic
                          schema check
PREDICTION_ERROR       — re-examine a failed prediction
                          (Phase 4 when cerebellum fires these)

Emits
-----
THOUGHT_PRODUCED       — every committed thought
SPEECH_REQUEST         — when responding to peer input
SUBSTRATE_WRITE_QUEUED — when committing a new edge
CHEMISTRY_FIRE         — confirmed_i on successful reasoning;
                          falsified_i on no useful path

What makes this FAST at scale
-----------------------------
- Causal chain: queries LTS.neighbors(focal, relation) → O(degree
  of focal), not O(substrate)
- Schema matching: walks AWM only — bounded to AWM capacity
  (256), not substrate (millions possible)
- Counterfactual: simulates over AWM, not LTS
- Metacognitive: per-concept LTS query, not cross-substrate scan

Even at 1M substrate concepts, cortical reasoning stays
proportional to AWM size + per-focal LTS query cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..events import (
    EventKind, BrainEvent,
    AttendedPerceptEvent,
    ThoughtProducedEvent,
    SpeechRequestEvent,
    SubstrateWriteQueuedEvent,
    ChemistryEvent,
    ArbitrationDecidedEvent,
)
from ..bus import EventBus
from ..chemistry_types import (
    M_CHANNELS, I_CHANNELS, CHANNELS,
)


# Baseline M/I polarity values — the average of the channel
# baselines on each axis.  Used by the vivid-trace check to
# measure how far a bubble's trace has DEVIATED from neutral.
_BASELINE_M_POLARITY = (
    sum(CHANNELS[ch]['baseline'] for ch in M_CHANNELS)
    / max(1, len(M_CHANNELS)))
_BASELINE_I_POLARITY = (
    sum(CHANNELS[ch]['baseline'] for ch in I_CHANNELS)
    / max(1, len(I_CHANNELS)))


# Relations cortical reasoning probes for causal / process /
# identity chains.  Mirrors v1's dialog-thinker default set.
CAUSAL_RELATIONS = ('causes', 'leads_to', 'produces',
                       'results_in', 'enables', 'requires')
PROCESS_RELATIONS = ('how', 'mechanism', 'enables', 'requires',
                        'composed_of', 'works_by')
IDENTITY_RELATIONS = ('is_a', 'has_property', 'instance_of',
                          'subtype_of')


# Built-in relational schemas for AWM-bound matching.  Same
# shape as v1 relational_schemas.STANDARD_SCHEMAS but the
# matcher runs over AWM only (sparse, fast).
@dataclass(frozen=True)
class Schema:
    name: str
    premises: Tuple[Tuple[str, str, str], ...]   # (s_slot, rel, o_slot)
    inferences: Tuple[Tuple[str, str, str], ...]
    description: str = ''


STANDARD_SCHEMAS: Tuple[Schema, ...] = (
    Schema(
        name='transitivity_is_a',
        premises=(('X', 'is_a', 'Y'), ('Y', 'is_a', 'Z')),
        inferences=(('X', 'is_a', 'Z'),),
        description='If X is_a Y and Y is_a Z, then X is_a Z.'),
    Schema(
        name='causal_chain',
        premises=(('X', 'causes', 'Y'), ('Y', 'causes', 'Z')),
        inferences=(('X', 'eventually_causes', 'Z'),),
        description='X causes Y causes Z → X eventually causes Z.'),
    Schema(
        name='double_opposition',
        premises=(('X', 'opposite_of', 'Y'),
                    ('Y', 'opposite_of', 'Z')),
        inferences=(('X', 'aligns_with', 'Z'),),
        description=(
            'X opposite_of Y opposite_of Z → X aligns_with Z.')),
)


# ---------------------------------------------------------------
# Phase R.1 (2026-05-22) — compositional inference.
#
# The old _causal_chain walked 2 hops but dragged the FIRST
# edge's relation across — that's association, not inference.
# RELATION_COMPOSITION makes the walk DEDUCTIVE: at each hop the
# accumulated relation and the next edge's relation compose into
# a new relation, OR the chain breaks.  A walk that composes
# cleanly across N hops derives a conclusion the substrate does
# not directly contain — the classic syllogism:
#   Socrates is_a man, man is_a mortal  ⊢  Socrates is_a mortal.
#
# Key:  (accumulated_relation, next_edge_relation) -> composed.
# Absence of a key means the two relations do NOT compose — the
# inference chain stops there (you cannot validly chain past it).
RELATION_COMPOSITION: Dict[Tuple[str, str], str] = {
    # --- transitivity (same relation chains) ---
    ('is_a', 'is_a'): 'is_a',
    ('instance_of', 'is_a'): 'instance_of',
    ('subtype_of', 'subtype_of'): 'subtype_of',
    ('subtype_of', 'is_a'): 'is_a',
    ('causes', 'causes'): 'leads_to',
    ('leads_to', 'leads_to'): 'leads_to',
    ('leads_to', 'causes'): 'leads_to',
    ('causes', 'leads_to'): 'leads_to',
    ('produces', 'produces'): 'leads_to',
    ('requires', 'requires'): 'requires',
    ('enables', 'enables'): 'enables',
    ('composed_of', 'composed_of'): 'composed_of',
    ('opposite_of', 'opposite_of'): 'aligns_with',
    # --- mixed causal — anything causal-ish chains to leads_to ---
    ('causes', 'produces'): 'leads_to',
    ('produces', 'causes'): 'leads_to',
    ('causes', 'enables'): 'leads_to',
    ('enables', 'causes'): 'leads_to',
    ('causes', 'results_in'): 'leads_to',
    ('results_in', 'causes'): 'leads_to',
    ('leads_to', 'produces'): 'leads_to',
    ('produces', 'leads_to'): 'leads_to',
    ('enables', 'produces'): 'leads_to',
    # --- inheritance — properties / powers flow down is_a ---
    ('is_a', 'has_property'): 'has_property',
    ('is_a', 'causes'): 'causes',
    ('is_a', 'can_do'): 'can_do',
    ('is_a', 'requires'): 'requires',
    ('is_a', 'has_part'): 'has_part',
    ('is_a', 'produces'): 'produces',
    ('instance_of', 'has_property'): 'has_property',
    ('instance_of', 'causes'): 'causes',
    ('subtype_of', 'has_property'): 'has_property',
    # --- analogical transfer (2026-05-28) — attributes / affordances
    # flow across `analogous_to` (the analogy engine's structural
    # edges).  "X is like Y; Y has property P  ⊢  X may have P."
    # This is ABDUCTIVE, not deductive: a conjecture, not a
    # certainty.  Two guards keep it honest: (1) ANALOGY_TRANSFER_
    # PENALTY discounts confidence for any chain through an analogy
    # hop; (2) analogous_to edges start at PROVISIONAL_EDGE_STRENGTH,
    # so a fresh analogy's transferred conclusions fall below the
    # emit floor until the analogy itself EARNS strength via Phase S
    # — earn-or-dissolve gates the inference.  Only attribute-like
    # relations transfer (has_property / can_do / used_for); is_a
    # and causes deliberately do NOT (analogy implies neither
    # category membership nor causation).  'analogous_to' mirrors
    # substrate.ANALOGY_RELATION.
    ('analogous_to', 'has_property'): 'has_property',
    ('analogous_to', 'can_do'): 'can_do',
    ('analogous_to', 'used_for'): 'used_for',
}

# Relations the inference walk will follow at each node.  audit #21b
# (2026-06-04): KEYS union VALUES — a relation that appears ONLY as a
# composition RESULT (e.g. aligns_with, leads_to) must be walkable as a
# premise too, otherwise it never earns the inference-walk engagement
# stamp and gets quarantined as inert.
INFERENCE_RELATIONS: Tuple[str, ...] = tuple(sorted(
    {r for pair in RELATION_COMPOSITION for r in pair}
    | set(RELATION_COMPOSITION.values())))

# Inference walk bounds.  4 hops is deep enough to be real
# multi-step reasoning; the per-hop penalty keeps deep chains
# appropriately uncertain.
INFERENCE_MAX_HOPS = 4
# Minimum confidence for a composed inference to be worth
# emitting as a thought.
INFERENCE_MIN_CONFIDENCE = 0.12
# Branching cap for the bounded inference search — top-K
# composable edges explored per node.  Bounds cost at
# BRANCH_CAP ** MAX_HOPS paths.
INFERENCE_BRANCH_CAP = 5


# Mirrors substrate.ABSTRACTION_RELATION — the relation a member holds to
# a synthetic class.  Literal here for the same reason _ANALOGY_RELATION
# is: the composition machinery stays import-free of substrate.
_MEMBERSHIP_RELATION = 'is_a'


def compose(accumulated: Optional[str], rel: str) -> Optional[str]:
    """THE composition step — one definition site, used by every walker.

    `accumulated` is None at the head of a chain (the first edge's own
    relation begins the accumulation); otherwise the two relations
    compose per RELATION_COMPOSITION, or do NOT compose (None), which
    ends the chain.  _inference_chain / _composable_edges and the
    mortality clock's family-grain keyer all route through this, so the
    deductive rule can never fork into two implementations.
    """
    if accumulated is None:
        return rel
    return RELATION_COMPOSITION.get((accumulated, rel))


class _WalkHit(Exception):
    """Internal early-exit signal for family_grain_relations."""


def family_grain_relations(neighbors_fn: Callable,
                               out_degree_fn: Callable,
                               class_node: str,
                               target: str,
                               *,
                               hub_degree: int,
                               max_hops: int = INFERENCE_MAX_HOPS,
                               want_relation: Optional[str] = None
                               ) -> frozenset:
    """READ-ONLY compositional reachability from a synthetic CLASS node.

    The caller has ALREADY taken the fixed first hop
    `member --is_a--> class_node` (the membership hop), so the walk
    starts AT `class_node` with the accumulated relation already equal
    to 'is_a' and hop 1 already spent.  Returns the SET of composed
    relations by which a clean composition chain reaches `target`.

    WHY IT LIVES HERE.  The mortality clock keys a member's derived edge
    (member, r, t) to the FAMILY fact (class_node, r, t) exactly when r
    is in this set — the class-level fact is what was learned, so the
    family settles once instead of every member settling separately.
    That decision must be made by the DERIVER'S OWN machinery: this
    function walks the same RELATION_COMPOSITION table, through the same
    compose() step, that produced the derived edge in the first place.
    The clock imports it; it never re-implements it.

    Callbacks (the caller owns substrate access, so this module stays
    import-free of substrate):
        neighbors_fn(node)   -> iterable of (relation, target)
        out_degree_fn(node)  -> int

    Bounds — all REUSED constants, none invented:
      * `max_hops` = INFERENCE_MAX_HOPS, counted as TOTAL hops INCLUDING
        the membership hop, so at most max_hops-1 onward edges.
      * `hub_degree` = COHERENCE_HUB_DEGREE.  A node whose out-degree
        exceeds it is never expanded — a path through a hub is weak
        corroboration, the identical rule substrate coherence applies.

    DETERMINISM (the settle ledger depends on it).  The answer is a
    function of TOPOLOGY ALONE — never of edge strength, iteration
    order, or chemistry.  `visited` is path-local, exactly as
    _inference_chain does it, so no globally-visited node can mask a
    second path and make the result order-dependent.  A fact key must
    hash identically across reinforces, restarts and processes, or the
    ledger silently double-pays.

    SIDE EFFECTS: NONE.  No engagement stamps, no reinforcement, no
    chemistry modulation — unlike _inference_chain, which mutates the
    edges of the chain it emits.  Safe to call from a reinforce
    observer.

    `want_relation` is a pure early-exit optimisation: reachability of
    that one relation is decided identically, just sooner.
    """
    found: Set[str] = set()
    target = str(target)
    limit = max(0, int(max_hops) - 1)      # onward edges after membership

    def _expand(node: str, accumulated: Optional[str], onward: int,
                    visited: frozenset) -> None:
        if onward >= limit:
            return
        try:
            if int(out_degree_fn(node)) > int(hub_degree):
                return                      # hub-skip: weak corroboration
        except Exception:
            return
        for rel, tgt in neighbors_fn(node):
            composed = compose(accumulated, rel)
            if composed is None:
                continue                    # the relations do not compose
            tgt = str(tgt)
            if tgt == target:
                found.add(composed)
                if want_relation is not None and composed == want_relation:
                    raise _WalkHit
            if tgt in visited:
                continue                    # cycle within THIS chain only
            _expand(tgt, composed, onward + 1, visited | {tgt})

    node0 = str(class_node)
    try:
        _expand(node0, _MEMBERSHIP_RELATION, 0, frozenset((node0,)))
    except _WalkHit:
        pass
    return frozenset(found)


# Confidence dynamics for a single cortical thought.
# `evidence_strength` is the edge strength of the relations
# walked.  `multi_hop_penalty` scales confidence down with depth.
MULTI_HOP_PENALTY = 0.7   # per extra hop

# Analogical transfer is ABDUCTIVE (a conjecture), so a chain that
# crosses an `analogous_to` hop takes an extra confidence discount
# on top of MULTI_HOP_PENALTY — analogical conclusions stay hedged
# relative to deductive ones even after the analogy edge has earned
# strength.  Tunable, in the same family as MULTI_HOP_PENALTY /
# INFERENCE_MIN_CONFIDENCE (reasoning-confidence calibration, not
# architecture).  0.5 ≈ "a priori as likely false as true, pending
# corroboration."
ANALOGY_TRANSFER_PENALTY = 0.5
# Mirrors substrate.ANALOGY_RELATION (literal here to keep the
# composition table import-free).
_ANALOGY_RELATION = 'analogous_to'
MIN_CONFIDENCE = 0.10


# Minimum LTS connectivity for "the substrate isn't thin around
# this focal."  Below this, metacognitive monitor consults the
# chemistry trace before flagging thin — a concept rarely
# networked but deeply FELT (mother, death) is not thin.
THIN_SUBSTRATE_THRESHOLD = 3   # edges (in or out)

# Combined absolute deviation (|m_pol − baseline_m| +
# |i_pol − baseline_i|) above which a bubble's trace is
# considered VIVID — meaningful lived experience has shifted
# it off baseline.  At chemistry baselines the sum is ~0.35;
# a deviation of 0.10 represents a meaningful chemistry imprint.
VIVID_TRACE_DEVIATION = 0.10


# How strongly chemistry resonance biases the walk.  Resonance
# is the alignment between current global M/I tension and a
# target's accumulated NT-trace tension — the brain-correct
# expression of mood-congruent cognition.
#
# Modulation is MULTIPLICATIVE: modulated = strength * (1 + W * resonance).
# Chemistry scales the substrate signal; it does not add to it.
# Biologically, neuromodulators change neural EXCITABILITY,
# they don't replace synaptic weight.  This preserves the
# relative ordering of strong vs weak substrate edges while
# letting mood meaningfully bias close calls.
#
# At W=0.5, resonance ∈ [-1,1], chemistry can scale a target's
# effective pull within [0.5x, 1.5x].  A 3x substrate-strength
# ratio (e.g. 0.9 vs 0.3) survives even max-opposite chemistry;
# equal-strength substrate is fully decided by chemistry.
CHEMISTRY_RESONANCE_WEIGHT = 0.5


class CorticalReasoner:
    """Event-triggered deliberate thinking.  Subscribes to
    AttendedPerceptEvent (mainly peer-origin); emits
    ThoughtProducedEvent + SpeechRequestEvent."""

    SUBSCRIPTIONS = (
        EventKind.ATTENDED_PERCEPT,
        EventKind.AWM_ACTIVATION,
        EventKind.PREDICTION_ERROR,
        EventKind.ARBITRATION_DECIDED,
    )

    def __init__(self,
                 awm_provider: Callable,
                 lts_provider: Callable,
                 chemistry_provider: Optional[Callable] = None,
                 cycle_provider: Optional[Callable] = None,
                 schema_provider: Optional[Callable] = None,
                 laws_provider: Optional[Callable] = None):
        """
        awm_provider: callable returning AWM instance
        lts_provider: callable returning LTS instance
        chemistry_provider: callable returning ChemistryEngine.
            When wired, cortical biases its causal/identity walks
            toward neighbors whose accumulated NT trace resonates
            with current global chemistry — the conscious M/I
            tagging layer the doctrine names as where AGI lives.
            When None, the walk is purely edge-strength driven
            (legacy behavior).
        cycle_provider: callable returning current cycle
        schema_provider: callable returning SchemaLibrary (Phase G.3).
            When a 2-hop causal-chain thought fires AND a matching
            schema (e.g. SCHEMA_TRANSITIVE_CAUSES) is established
            with reasonable support, cortical renders the thought
            with schema-aware language ("I infer via a transitive
            pattern I've noticed") and modestly boosts confidence.
        """
        self._awm_provider = awm_provider
        self._lts_provider = lts_provider
        self._chemistry_provider = chemistry_provider
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._schema_provider = schema_provider
        # Phase G.4 (2026-05-16): laws_provider lets cortical
        # consult SymbolicRegressor's DiscoveredLaws as soft
        # context when reasoning.  A high-confidence law whose
        # variables are both currently elevated above baseline
        # gets appended as a chemistry-aware footnote on thought
        # text — "while noticing cortisol rises with NE."
        self._laws_provider = laws_provider
        # Diagnostics
        self.thoughts_produced: int = 0
        self.schemas_matched: int = 0
        self.inferences_emitted: int = 0
        self.responses_composed: int = 0
        # Death-wall organs (SHADOW): the EngagementLedger receives
        # MEANINGFUL-use credit (coherent edges engaged in an emitted
        # inference) and OBSTRUCTION events (an incoherent edge out-ranks
        # a slot).  Wired by runtime; None → no-op (legacy behaviour).
        self._engagement_ledger = None
        # Most-recent thought per focal (small cache for
        # downstream consumers like Speech).
        self.last_thought_by_focal: Dict[str, ThoughtProducedEvent] = {}

    # ---- event handler ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, AttendedPerceptEvent):
            self._on_attended(event, bus)
        elif isinstance(event, ArbitrationDecidedEvent):
            self._on_arbitration(event, bus)
        elif event.kind == EventKind.AWM_ACTIVATION:
            # Opportunistic: when something new enters AWM, run
            # a schema-match pass.  Cheap because AWM is small.
            self._match_schemas(bus, trigger='awm_activation')
        elif event.kind == EventKind.PREDICTION_ERROR:
            # Phase 4 will refine this; for Phase 3 we just
            # log via thoughts_produced bump.
            self.thoughts_produced += 0  # placeholder

    def _on_attended(self,
                        ev: AttendedPerceptEvent,
                        bus: EventBus) -> None:
        """Main triggering path.  Peer input → run reasoning →
        emit thought + speech request.  Reverie (C.1.f, internal
        origin) → run reasoning + emit thought, but NO speech
        request (the agent is thinking to itself, not responding
        to anyone).  Other non-peer attended percepts (e.g.
        salient forager arrivals) get a schema pass only."""
        cycle = self._cycle_provider()
        is_peer = (ev.origin == 'peer')
        # Phase C.1.g (2026-05-19): autonomous reverie should
        # actually engage deliberate reasoning.  A reverie percept
        # is the agent surfacing a substrate concept to think
        # about during silence — genuine internal cognition.
        is_reverie = (
            ev.origin == 'internal'
            and str(ev.origin_detail).startswith('reverie'))
        # Step 5.1 closure (2026-05-29): inner-voice self-percepts
        # must trigger inference too — otherwise the agent
        # articulates without thinking, and the language→cognition
        # loop is broken (empirically observed on alpha:
        # inferences_emitted=0 over 42 thoughts despite 58 inner
        # utterances).  Inner voice publishes with origin='self'
        # and source_capability='inner_voice'; that pattern is
        # distinguishable from reverie (substrate-walked focal)
        # and peer input.  Cortical re-entry stays inherited-blocked
        # because inner_voice is tick-driven (no event subscriptions
        # → no direct re-call), and its 50-tick throttle + repeat-
        # suppression ring prevent tight loops.
        is_self = (
            ev.origin == 'self'
            and ev.source_capability == 'inner_voice')
        if (is_peer or is_reverie or is_self) and ev.focals:
            # Pick the top focal — first one passed the gate.
            focal = ev.focals[0]
            thought = self._think_about(
                focal, raw_text=ev.raw_text, cycle=cycle)
            if thought is not None:
                bus.publish(thought)
                # Speech is only for peer-driven thoughts —
                # reverie is inner dialogue, not a response.
                if is_peer:
                    bus.publish(SpeechRequestEvent(
                        kind=EventKind.SPEECH_REQUEST,
                        cycle=cycle,
                        timestamp=time.time(),
                        source_capability='cortical',
                        origin='internal',
                        origin_detail=focal,
                        intended_focal=focal,
                        thought_text=thought.text,
                        thin_substrate=thought.thin_substrate,
                        awm_snapshot=self._awm_snapshot(),
                    ))
                    self.responses_composed += 1
        # Always opportunistically check schemas on new input.
        self._match_schemas(bus, trigger='attended')

    # ---- arbitration-driven attention ----

    def _on_arbitration(self,
                              ev: ArbitrationDecidedEvent,
                              bus: EventBus) -> None:
        """BG declared a cognitive-loop winner.  Two recognized
        actions today:

          attend_threat:<focal>  (from Amygdala) — re-think the
              named focal and re-emit SPEECH_REQUEST.  The
              re-emit overrides any pending peer-driven voice
              because Motor/Speech's pending_response is a
              last-writer-wins buffer.  Skipped when the focal
              is already top-of-mind (we don't re-think the
              same thing within 2 cycles), and skipped when the
              "focal" is actually a generic threat_kind word
              ('body' / 'chemistry' / 'percept' — no concrete
              focal to attend to).

          shift_focus  (from AnteriorPFC after an unproductive
              streak) — internal attention shift.  Pick a
              different focal from AWM and think about it.
              Does NOT emit SPEECH_REQUEST: this is the brain's
              attention wandering, not a new utterance.

        Other winning actions (cognitive loop or other loops)
        are ignored by cortical — they belong to other
        consumers.
        """
        if ev.loop != 'cognitive':
            return
        action = (ev.winning_action or '').strip()
        if not action:
            return
        if action.startswith('attend_threat:'):
            focal = action.split(':', 1)[1].strip()
            self._handle_attend_threat(focal, ev.cycle, bus)
        elif action.startswith('attend_opportunity:'):
            focal = action.split(':', 1)[1].strip()
            self._handle_attend_opportunity(focal, ev.cycle, bus)
        elif action.startswith('explore_novel:'):
            focal = action.split(':', 1)[1].strip()
            self._handle_explore_novel(focal, ev.cycle, bus)
        elif action.startswith('resolve_uncertainty:'):
            focal = action.split(':', 1)[1].strip()
            self._handle_resolve_uncertainty(
                focal, ev.cycle, bus)
        elif action == 'shift_focus':
            self._handle_shift_focus(ev.cycle, bus)

    _GENERIC_THREAT_KINDS = ('body', 'chemistry', 'percept', '')

    def _handle_attend_threat(self,
                                       focal: str,
                                       cycle: int,
                                       bus: EventBus) -> None:
        # Skip when the arbitration target isn't a real focal
        # (Amygdala falls back to threat_kind when no focal
        # is implicated — body/chemistry threats with no
        # specific concept).
        if focal in self._GENERIC_THREAT_KINDS:
            return
        # Skip if we just thought about this focal — no point
        # re-firing the same thought.
        last = self.last_thought_by_focal.get(focal)
        if last is not None and (cycle - last.cycle) < 2:
            return
        thought = self._think_about(focal, cycle=cycle)
        if thought is None:
            return
        bus.publish(thought)
        # Re-emit SPEECH_REQUEST.  Motor/Speech's pending_response
        # is overwritten — the threat-aware voice wins over the
        # prior peer-driven response.
        bus.publish(SpeechRequestEvent(
            kind=EventKind.SPEECH_REQUEST,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='cortical',
            origin='internal',
            origin_detail=f'arbitration_threat:{focal}',
            intended_focal=focal,
            thought_text=thought.text,
            thin_substrate=thought.thin_substrate,
            awm_snapshot=self._awm_snapshot(),
        ))
        self.responses_composed += 1

    _GENERIC_OPPORTUNITY_KINDS = ('percept', 'chemistry', 'body', '')

    def _handle_attend_opportunity(self,
                                              focal: str,
                                              cycle: int,
                                              bus: EventBus) -> None:
        """BG declared an I-side winner (Nucleus Accumbens).
        Symmetric to attend_threat but with lower urgency:
        we still re-think the opportunity focal and re-emit
        SPEECH_REQUEST when it's not already top-of-mind, but
        Amygdala threats compete higher in BG so opportunity
        won't override imminent danger.

        Skipped when:
          - "focal" is actually an opportunity_kind word
            ('percept' / 'chemistry' / 'body' — no concrete
            focal); body-origin opportunities surface via the
            chemistry side instead.
          - the focal was just thought about (< 2 cycles ago).
        """
        if focal in self._GENERIC_OPPORTUNITY_KINDS:
            return
        last = self.last_thought_by_focal.get(focal)
        if last is not None and (cycle - last.cycle) < 2:
            return
        thought = self._think_about(focal, cycle=cycle)
        if thought is None:
            return
        bus.publish(thought)
        bus.publish(SpeechRequestEvent(
            kind=EventKind.SPEECH_REQUEST,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='cortical',
            origin='internal',
            origin_detail=f'arbitration_opportunity:{focal}',
            intended_focal=focal,
            thought_text=thought.text,
            thin_substrate=thought.thin_substrate,
            awm_snapshot=self._awm_snapshot(),
        ))
        self.responses_composed += 1

    def _handle_explore_novel(self,
                                       focal: str,
                                       cycle: int,
                                       bus: EventBus) -> None:
        """NoveltyMonitor arbitration win — think about a novel
        focal that came through perception or chemistry.  This
        is internal exploration (curiosity-led thinking); no
        SPEECH_REQUEST — peer-driven voice keeps priority.
        Skip when focal is already top-of-mind."""
        if not focal:
            return
        last = self.last_thought_by_focal.get(focal)
        if last is not None and (cycle - last.cycle) < 2:
            return
        thought = self._think_about(focal, cycle=cycle)
        if thought is None:
            return
        bus.publish(thought)

    def _handle_resolve_uncertainty(self,
                                                  focal: str,
                                                  cycle: int,
                                                  bus: EventBus) -> None:
        """UncertaintyMonitor arbitration win — accumulated
        epistemic load has crossed threshold; step back and
        think about the most-pressed focal.  Internal — no
        speech — the resolution is for SEAGI's own coherence."""
        if not focal:
            return
        last = self.last_thought_by_focal.get(focal)
        if last is not None and (cycle - last.cycle) < 2:
            return
        thought = self._think_about(focal, cycle=cycle)
        if thought is None:
            return
        bus.publish(thought)

    def _handle_shift_focus(self,
                                     cycle: int,
                                     bus: EventBus) -> None:
        awm = self._awm()
        if awm is None:
            return
        active = awm.active_concepts()
        if not active:
            return
        # Prefer a focal we haven't recently thought about.
        novel = [c for c in active
                    if c not in self.last_thought_by_focal]
        if novel:
            focal = novel[0]
        else:
            # All AWM focals have been thought about — pick the
            # one we thought about LEAST recently.
            focal = min(
                active,
                key=lambda c: self.last_thought_by_focal[c].cycle
                if c in self.last_thought_by_focal else 0)
        if not focal:
            return
        thought = self._think_about(focal, cycle=cycle)
        if thought is None:
            return
        bus.publish(thought)
        # NOTE: deliberate — no SPEECH_REQUEST here.  This is
        # the mind shifting, not the voice changing.

    # ---- think_about: top-level reasoning entry ----

    def _think_about(self,
                        focal: str,
                        raw_text: str = '',
                        cycle: int = 0
                        ) -> Optional[ThoughtProducedEvent]:
        """Compose a thought about `focal` from AWM + LTS query.

        Strategy (in priority order):
          1. If LTS has thin substrate for focal → emit
             metacognitive thin-substrate thought
          2. Try causal_chain (1-2 hops)
          3. Try identity walk (is_a)
          4. Try counterfactual hint (what-if removal)

        Returns the strongest thought or None.
        """
        if not focal:
            return None

        # Metacognitive baseline.  We use this to flag thin-
        # substrate AT THE END if reasoning yields nothing.
        # We do NOT short-circuit on it here — the brain should
        # try to think with what little it has, and only
        # acknowledge thinness when reasoning truly fails.
        thin, n_edges = self._assess_focal(focal)

        # If the concept isn't even in LTS, no point reasoning.
        if n_edges == 0:
            lts = self._lts()
            if lts is None or not lts.has_concept(focal):
                return self._make_thought(
                    focal=focal, relation='', target='',
                    confidence=0.3, method='metacog',
                    text=(f"I do not know {focal} — I have "
                            "not encountered it before."),
                    thin_substrate=True,
                    cycle=cycle)

        # Phase R.1: compositional inference FIRST.  A multi-hop
        # chain whose relations compose cleanly derives a
        # conclusion the substrate doesn't directly state — the
        # strongest form of reasoning cortical can do.  Preferred
        # over the plain causal walk when it lands.
        inference = self._inference_chain(focal)
        if inference is not None:
            conf = inference['confidence']
            text = self._render_inference(inference)
            # Phase G.3 carried into R.1: when the composed
            # relation matches a discovered transitive schema,
            # the inference has empirical backing — surface the
            # support count and nudge confidence.
            schema = self._matching_schema_for_chain(
                inference['relation'])
            if schema is not None:
                self.schemas_matched += 1
                conf = min(1.0, conf * 1.15)
                text = (text[:-1] if text.endswith('.') else text)
                text += (f" (a transitive pattern I have noticed, "
                             f"supported by {schema.support} similar "
                             f"chains).")
            footnote = self._law_aware_context()
            if footnote:
                text = (text[:-1] + footnote + '.'
                            if text.endswith('.')
                            else text + footnote)
            return self._make_thought(
                focal=focal,
                relation=inference['relation'],
                target=inference['target'],
                confidence=conf,
                method='inference',
                text=text, cycle=cycle,
                chain_depth=int(inference.get('hops', 0)),
                # Premise-reinforcement (2026-06-05): carry the walked
                # chain so the consolidator can reinforce the premises
                # that PROVED this conclusion (they earned strength by
                # being useful for reasoning — moving away from death).
                edges_walked=tuple(inference.get('chain', ())))

        # Causal chain: try 1-hop, then 2-hop.
        causal = self._causal_chain(focal, max_hops=2)
        if causal is not None:
            f, r, t, conf, hops, evidence, walked = causal
            # Phase G.3 (2026-05-16): when the result came from a
            # 2-hop walk AND a matching schema is established with
            # good support, render with schema-aware language and
            # nudge confidence.
            method = 'causal'
            text = self._render_causal(f, r, t, hops, evidence)
            if hops >= 2:
                schema = self._matching_schema_for_chain(r)
                if schema is not None:
                    self.schemas_matched += 1
                    method = 'schema'
                    # Confidence boost: cap at 1.0; gentle so it
                    # doesn't override evidence weakness.
                    conf = min(1.0, conf * 1.15)
                    text = (
                        f'I infer via a transitive pattern I have '
                        f'noticed: {f} likely {r.replace("_", " ")} '
                        f'{t}.  (Supported by {schema.support} '
                        f'similar chains in my substrate.)')
            # Phase G.4 (2026-05-16): law-aware footnote.  If a
            # discovered law currently applies (its variables are
            # in the pattern it predicts), append a brief
            # contextual phrase.  Subtle — only fires when the
            # chemistry is meaningfully active relative to the law.
            footnote = self._law_aware_context()
            if footnote:
                # Append before the period, if any.
                if text.endswith('.'):
                    text = text[:-1] + footnote + '.'
                else:
                    text = text + footnote
            return self._make_thought(
                focal=focal, relation=r, target=t,
                confidence=conf, method=method,
                text=text, cycle=cycle,
                # Phase 2: reinforce the real walked hops (the 2-hop
                # case carries two genuine edges; 1-hop carries one).
                # Only method='causal' is consolidated as recall; a
                # 'schema'-upgraded 2-hop drops (rare; left as a minor
                # follow-up, not in the smallest version).
                edges_walked=walked)

        # Identity walk: is_a / has_property.
        ident = self._first_neighbor(focal, IDENTITY_RELATIONS)
        if ident is not None:
            r, t, strength = ident
            return self._make_thought(
                focal=focal, relation=r, target=t,
                confidence=min(1.0, strength),
                method='causal',
                text=(f"I hold {focal} as {r.replace('_', ' ')} "
                        f"{t}."),
                cycle=cycle)

        # Process walk: how / enables.
        process = self._first_neighbor(focal, PROCESS_RELATIONS)
        if process is not None:
            r, t, strength = process
            return self._make_thought(
                focal=focal, relation=r, target=t,
                confidence=min(1.0, strength),
                method='causal',
                text=(f"{focal} {r.replace('_', ' ')} {t}."),
                cycle=cycle)

        # Counterfactual hint as last resort.
        cf = self._counterfactual_hint(focal)
        if cf is not None:
            return self._make_thought(
                focal=focal, relation='', target='',
                confidence=0.3,
                method='counterfactual',
                text=cf, cycle=cycle)

        # Nothing landed.  Use metacognitive assessment to be
        # honest about why.
        if thin:
            return self._make_thought(
                focal=focal, relation='', target='',
                confidence=0.3, method='metacog',
                text=(f"I do not know {focal} well — "
                        "my substrate is thin around it."),
                thin_substrate=True,
                cycle=cycle)
        # Not flagged thin by _assess_focal — vivid trace
        # accumulated, or many edges that just didn't produce
        # a useful walk this time.  Acknowledge presence
        # without claiming thinness; thin_substrate stays
        # False so Motor/Speech won't pick honest_uncertain
        # mode.
        return self._make_thought(
            focal=focal, relation='', target='',
            confidence=0.2, method='metacog',
            text=(f"{focal} sits in my substrate but I have "
                    "no strong thought to offer about it."),
            thin_substrate=False,
            cycle=cycle)

    # ---- helpers ----

    def _awm(self):
        return self._awm_provider()

    def _lts(self):
        return self._lts_provider()

    def _cycle(self) -> int:
        try:
            return int(self._cycle_provider())
        except Exception:
            return 0

    def _make_thought(self,
                          *,
                          focal: str,
                          relation: str,
                          target: str,
                          confidence: float,
                          method: str,
                          text: str,
                          cycle: int,
                          thin_substrate: bool = False,
                          chain_depth: int = 0,
                          edges_walked: tuple = ()
                          ) -> ThoughtProducedEvent:
        ev = ThoughtProducedEvent(
            kind=EventKind.THOUGHT_PRODUCED,
            cycle=cycle,
            timestamp=time.time(),
            source_capability='cortical',
            origin='internal',
            origin_detail=focal,
            focal=focal,
            relation=relation,
            target=target,
            confidence=max(MIN_CONFIDENCE, min(1.0, confidence)),
            method=method,
            text=text,
            thin_substrate=thin_substrate,
            chain_depth=int(chain_depth),
            # Phase 2 per-hop reinforcement: the real walked hops
            # (see ThoughtProducedEvent.edges_walked).  Normalize to a
            # tuple of (s, r, o) tuples; empty if no edge walk.
            edges_walked=tuple(
                (str(s), str(r), str(o))
                for (s, r, o) in (edges_walked or ())),
        )
        self.thoughts_produced += 1
        self.last_thought_by_focal[focal] = ev
        # Phase C 2026-05-15: a thought IS a use → bump the salience
        # of the focal + target concepts.  Promille-scale — frequent
        # thinking-about strengthens; quiet concepts fade via lazy
        # decay.
        #
        # Reinforce-on-recall demotion (2026-05-31): edge
        # reinforcement USED to happen here as a direct, debt-free,
        # quarantine-invisible edge.reinforce(cycle).  That double-
        # counted against — and now is fully superseded by — the
        # metered consolidator→writer path: a 'causal' recall thought
        # is published as a SubstrateWriteQueuedEvent
        # (write_reason='recall_reattest'), which accrues MetabolicDebt
        # AND stamps last_engaged_cycle.  Keeping a direct reinforce
        # here would re-introduce the unmetered write the igniter
        # exists to eliminate, so this is now SALIENCE-ONLY.  See
        # project_seagi_dry_reverie_igniter.
        # Salience credit is EARNED by productive cognition, not farmed
        # by the act of thinking (2026-06-04 audit #4b).  Only a thought
        # that actually USED substrate structure — a derivation
        # ('inference'), a real edge walk ('causal'), or a schema match
        # ('schema') — credits salience.  Contentless metacognitive /
        # counterfactual hedges no longer pay, so rumination on a
        # structureless focal can't rank itself into the reverie pool.
        if method in ('inference', 'causal', 'schema'):
            try:
                lts = self._lts()
                if lts is not None:
                    from seagi.core.substrate import (
                        SALIENCE_BUMP_DEFAULT)
                    focal_concept = lts.get_concept(focal)
                    if focal_concept is not None and hasattr(
                            focal_concept, 'bump_salience'):
                        focal_concept.bump_salience(
                            SALIENCE_BUMP_DEFAULT, cycle)
                    if target and relation:
                        target_concept = lts.get_concept(target)
                        if target_concept is not None and hasattr(
                                target_concept, 'bump_salience'):
                            target_concept.bump_salience(
                                SALIENCE_BUMP_DEFAULT, cycle)
            except Exception:
                pass
        return ev

    # ---- chemistry resonance (mood-congruent walk bias) ----

    def _resonance(self, target: str) -> float:
        """How well a target's accumulated NT trace aligns with
        current global chemistry, computed over the FULL
        8-channel fingerprint (not the 2D M/I projection).
        Returns a signed score in [-1, 1]:

          positive — target's trace pattern matches current
                      chemistry's pattern (mood-congruent
                      retrieval; the brain RESONATES with this
                      concept right now)
          near 0  — target not in AWM, either side at baseline
                      (no chemistry to compare), or orthogonal
                      patterns
          negative — pattern is OPPOSITE to current chemistry
                      (mood-incongruent; feels off)

        Doctrine: M and I are the SPECTRUM endpoints; the
        thousands of in-between channel balances are what
        cognition actually traverses.  The 2D projection
        collapses those — two bubbles with identical M/I
        tension can have radically different felt character
        if their underlying 8-channel mix diverges.

        Implementation: centered cosine similarity.  Each
        channel is centered on its baseline so trace-at-
        baseline contributes zero (no felt experience yet =
        no resonance).  Cosine over deviations, not raw
        levels, so two bubbles whose chemistry just happens
        to sit at baselines don't spuriously resonate.

        Only AWM-resident targets get steering — dormant LTS
        concepts walk by edge strength alone (brain-correct:
        you can only be steered toward what's lit up in your
        active mind).
        """
        if self._chemistry_provider is None:
            return 0.0
        chem = self._chemistry_provider()
        if chem is None:
            return 0.0
        awm = self._awm()
        if awm is None:
            return 0.0
        entry = awm.get(target)
        if entry is None:
            return 0.0
        bubble = entry.bubble
        # Centered cosine similarity over all NT channels.
        dot = 0.0
        norm_a_sq = 0.0
        norm_b_sq = 0.0
        for ch, cfg in CHANNELS.items():
            baseline = cfg['baseline']
            a = chem.global_state.get(ch, baseline) - baseline
            b = bubble.transmitter_trace.get(
                ch, baseline) - baseline
            dot += a * b
            norm_a_sq += a * a
            norm_b_sq += b * b
        # Either side at baseline (no chemistry deviation
        # anywhere) → no resonance signal.
        if norm_a_sq < 1e-9 or norm_b_sq < 1e-9:
            return 0.0
        return dot / ((norm_a_sq * norm_b_sq) ** 0.5)

    def _modulated_score(self,
                              raw_strength: float,
                              target: str) -> float:
        """Combine raw edge strength with chemistry resonance,
        multiplicatively.  At chemistry baseline (resonance=0),
        returns raw_strength unchanged.  Chemistry can scale the
        effective pull within roughly [0.5x, 1.5x] but cannot
        override large substrate-strength gaps."""
        return raw_strength * (
            1.0 + CHEMISTRY_RESONANCE_WEIGHT * self._resonance(target))

    # ---- metacognitive assessment ----

    def _assess_focal(self, focal: str) -> Tuple[bool, int]:
        """Returns (thin_substrate, n_outgoing_edges).

        A focal is thin only when BOTH conditions hold:
          - edge count below THIN_SUBSTRATE_THRESHOLD, AND
          - chemistry trace is near baseline (no vivid felt
            experience accumulated)

        A concept rarely connected but FELT strongly (the
        prototypical example: 'death', 'mother' — few edges in
        a young SEAGI's substrate, but heavy trace from
        encounter chemistry) is NOT thin.  It's "deeply felt
        but not yet networked."  The doctrine: AGI lives in the
        conscious M/I layer.  Trace IS the felt knowledge.

        Without this override, cortical would deflect with
        thin-substrate metacog ("I do not know X well") on
        concepts SEAGI feels intensely.  That's a structural
        lie in the engine.
        """
        lts = self._lts()
        if lts is None:
            return (True, 0)
        if not lts.has_concept(focal):
            return (True, 0)
        n = len(lts.neighbors(focal))
        if n >= THIN_SUBSTRATE_THRESHOLD:
            return (False, n)
        # Few edges — consult chemistry trace before flagging.
        if self._has_vivid_trace(focal):
            return (False, n)
        return (True, n)

    def expected_companions(self,
                                  focal: str,
                                  top_k: int = 5) -> List[str]:
        """Phase E (2026-05-16): generative context prediction.

        Returns concepts the agent EXPECTS to co-occur with `focal`
        based on the focal's active bubble's accumulated context.
        Each bubble's ContextKey carries `coactive_concepts` — the
        names of concepts that were in AWM when this bubble was
        last imprinted.  Reading them back is a generative
        prediction: "when I hold fire, I anticipate fuel alongside."

        Doctrine: substrate read AS a generative model.  Bubbles
        carry not just felt-state but contextual companions.  The
        same concept's different bubbles have different coactive
        sets — a war-context "death" bubble predicts soldiers,
        blood, fear; a stoic-context "death" bubble predicts
        acceptance, rest, virtue.  Which expectation fires depends
        on which bubble is currently active.

        Returns up to `top_k` names, ordered by frequency across
        all of the concept's bubbles (concepts that appear in
        coactive sets of more bubbles rank higher).  Falls back
        to empty list if no bubbles or no context information.

        Used by: MotorSpeech (anticipation voice), Cerebellum
        (context-based prediction), future imagination handlers.
        """
        lts = self._lts()
        if lts is None:
            return []
        concept = lts.get_concept(focal)
        if concept is None:
            return []
        bubbles = getattr(concept, 'bubbles', None) or []
        if not bubbles:
            return []
        # Count companion frequency across all bubbles.  Each
        # bubble's coactive set contributes its names; concepts
        # appearing in multiple bubbles' coactive sets bubble up.
        counts: Dict[str, int] = {}
        for b in bubbles:
            ck = getattr(b, 'context_key', None)
            if ck is None:
                continue
            coactive = getattr(ck, 'coactive_concepts', None) or set()
            for name in coactive:
                if name == focal or not name:
                    continue
                counts[name] = counts.get(name, 0) + 1
        if not counts:
            return []
        ranked = sorted(counts.items(),
                         key=lambda kv: (-kv[1], kv[0]))
        return [name for name, _ in ranked[:max(1, top_k)]]

    def _law_aware_context(self) -> str:
        """Phase G.4 (2026-05-16): return a brief chemistry-
        context footnote from discovered laws if any apply.

        A law "applies" when:
          - confidence ≥ 0.3 (not noise-level)
          - BOTH variables of the law currently deviate from
             their baselines in the direction the law predicts
             (e.g. law says "X rises with Y"; current chemistry
             shows both X and Y elevated)

        Returns the footnote string ("while noticing X rises with Y")
        or empty string.  Used by `_render_causal` and the schema
        branch to add lightweight meta-cognitive awareness.

        Only emits ONE footnote per thought — pick the highest-
        confidence applicable law.
        """
        if self._laws_provider is None:
            return ''
        try:
            lib = self._laws_provider()
        except Exception:
            return ''
        if lib is None or len(lib) == 0:
            return ''
        try:
            from ..chemistry_types import CHANNELS
        except Exception:
            return ''
        chem = None
        if self._chemistry_provider is not None:
            try:
                chem = self._chemistry_provider()
            except Exception:
                chem = None
        if chem is None:
            return ''
        state = getattr(chem, 'global_state', {}) or {}
        # Look at top laws (already sorted by confidence).
        try:
            top = lib.top_laws(n=5)
        except Exception:
            return ''
        for law in top:
            if law.confidence < 0.3:
                continue
            # Extract channel name from 'chem.X' prefix where
            # applicable.
            xv = self._law_variable_value(
                law.variable_x, state)
            yv = self._law_variable_value(
                law.variable_y, state)
            bx = self._law_variable_baseline(
                law.variable_x, CHANNELS)
            by = self._law_variable_baseline(
                law.variable_y, CHANNELS)
            if xv is None or yv is None or bx is None or by is None:
                continue
            # Direction check: both above OR both below baseline,
            # matching the correlation's sign.
            x_up = xv > bx + 0.01
            x_down = xv < bx - 0.01
            y_up = yv > by + 0.01
            y_down = yv < by - 0.01
            if law.correlation > 0:
                # Positive correlation: expect both up OR both down.
                if not ((x_up and y_up) or (x_down and y_down)):
                    continue
            else:
                # Negative correlation: expect opposing.
                if not ((x_up and y_down) or (x_down and y_up)):
                    continue
            # Render the footnote with short names.
            xn = self._law_variable_short(law.variable_x)
            yn = self._law_variable_short(law.variable_y)
            direction = ('rises with'
                            if law.correlation > 0
                            else 'falls as')
            return f' (while noticing {yn} {direction} {xn})'
        return ''

    def _law_variable_value(self, var_name: str, state):
        if var_name.startswith('chem.'):
            ch = var_name[5:]
            return state.get(ch)
        # body.* signals not tracked here; caller can extend.
        return None

    def _law_variable_baseline(self, var_name: str, channels):
        if var_name.startswith('chem.'):
            ch = var_name[5:]
            cfg = channels.get(ch)
            return cfg.get('baseline') if cfg else None
        return None

    def _law_variable_short(self, var_name: str) -> str:
        """Strip 'chem.' prefix for display."""
        if var_name.startswith('chem.'):
            return var_name[5:]
        if var_name.startswith('body.'):
            return var_name[5:]
        return var_name

    def _matching_schema_for_chain(self, relation: str):
        """Phase G.3 (2026-05-16): given a chain relation (the
        type of edge traversed in a multi-hop walk), return the
        matching established Schema record if any.  Used to
        confirm that 2-hop walks are doctrine-supported by a
        discovered pattern.

        Maps relation → schema kind:
          'causes' → SCHEMA_TRANSITIVE_CAUSES
          'enables' → SCHEMA_TRANSITIVE_ENABLES
          'is_a' → SCHEMA_TRANSITIVE_IS_A

        Requires schema.support ≥ 3 (matches the discovery
        promotion threshold; below that the pattern isn't
        confident enough to invoke).

        Returns the Schema, or None.
        """
        if self._schema_provider is None:
            return None
        try:
            lib = self._schema_provider()
        except Exception:
            return None
        if lib is None:
            return None
        try:
            from seagi.brain.capabilities.schema_discovery import (
                SCHEMA_TRANSITIVE_CAUSES, SCHEMA_TRANSITIVE_ENABLES,
                SCHEMA_TRANSITIVE_IS_A)
        except Exception:
            return None
        relation_to_kind = {
            'causes': SCHEMA_TRANSITIVE_CAUSES,
            # R.1: a composed causes-chain surfaces as 'leads_to';
            # it still has SCHEMA_TRANSITIVE_CAUSES backing.
            'leads_to': SCHEMA_TRANSITIVE_CAUSES,
            'enables': SCHEMA_TRANSITIVE_ENABLES,
            'is_a': SCHEMA_TRANSITIVE_IS_A,
        }
        kind = relation_to_kind.get(relation)
        if kind is None:
            return None
        try:
            matches = lib.by_kind(kind)
        except Exception:
            return None
        if not matches:
            return None
        # Pick highest-support schema of this kind.
        best = max(matches, key=lambda s: s.support)
        if best.support < 3:
            return None
        return best

    def _has_vivid_trace(self, focal: str) -> bool:
        """True when the focal's AWM bubble's accumulated trace
        has DEVIATED meaningfully from chemistry baselines on
        either M or I axis.  Deviation = lived experience;
        baseline = neutral untagged state.

        Only AWM-resident focals can be checked — dormant LTS
        concepts have no accessible bubble here.  This is
        brain-correct: only what's in active mind has accessible
        felt-state.  If a deeply-felt concept isn't currently in
        AWM, by the time cortical thinks about it (via attended
        percept), AWM has already promoted it — so the check
        works for the common path.
        """
        awm = self._awm()
        if awm is None:
            return False
        entry = awm.get(focal)
        if entry is None:
            return False
        bubble = entry.bubble
        m_dev = abs(bubble.m_polarity() - _BASELINE_M_POLARITY)
        i_dev = abs(bubble.i_polarity() - _BASELINE_I_POLARITY)
        return (m_dev + i_dev) >= VIVID_TRACE_DEVIATION

    # ---- Phase R.1: compositional multi-hop inference ----

    def set_engagement_ledger(self, ledger) -> None:
        """Wire the death-wall EngagementLedger (runtime)."""
        self._engagement_ledger = ledger

    def _note_slot_obstruction(self, source: str, relation: str,
                                top_target: Optional[str]) -> None:
        """M3 single shared obstruction predicate.  Notes an obstruction
        when the TOP-ranked edge of the (source, relation) slot is
        INCOHERENT (first_coherent_cycle == 0) — incoherent junk that
        out-ranks the slot and 'gets in the way'.  M2: counts at full
        weight even when the slot is junk-only (no coherent competitor),
        so an agent cannot dodge aging by keeping its slots junk-only.
        The ledger dedupes per-slot-per-episode."""
        ledger = self._engagement_ledger
        if ledger is None or top_target is None:
            return
        lts = self._lts()
        if lts is None:
            return
        e = lts.edge(source, relation, top_target)
        if e is None:
            return
        if (getattr(e, 'first_coherent_cycle', 0) or 0) <= 0:
            ledger.note_obstruction(source, relation)

    def _inference_chain(self,
                              focal: str,
                              max_hops: int = INFERENCE_MAX_HOPS
                              ) -> Optional[dict]:
        """Deductive walk: follow typed edges, composing the
        accumulated relation with each next edge's relation via
        RELATION_COMPOSITION.  A walk of >= 2 hops whose relations
        compose cleanly derives a conclusion the substrate does
        not directly state — that is inference, not association.

        Returns a dict:
          {'focal', 'relation' (composed), 'target' (terminal),
           'confidence', 'hops', 'chain' [(s,r,o), ...]}
        or None when no >= 2-hop composable chain exists.

        Bounded best-first search rather than a single greedy
        path: at each node it branches over the top
        INFERENCE_BRANCH_CAP composable edges.  Pure greedy
        dead-ends constantly on a real substrate — the strongest
        hop-1 edge usually lands on a node whose strongest onward
        edge doesn't compose.  Branching catches the chains a
        single path misses.  Cost is bounded at
        BRANCH_CAP**max_hops paths."""
        lts = self._lts()
        if lts is None:
            return None
        # Track the best chain.  A reasoning engine should surface
        # the DEEPEST valid derivation — depth IS the demonstration
        # of multi-step inference — so selection prefers more hops,
        # with confidence as the tie-break.  The confidence floor
        # keeps truly tenuous deep chains out; confidence is still
        # reported honestly (lower for deeper).
        best = {'hops': 0, 'conf': -1.0, 'result': None}

        def extend(node: str,
                       accumulated: Optional[str],
                       chain: list,
                       visited: set,
                       conf: float) -> None:
            # Record any chain of >= 2 hops as a candidate result.
            if len(chain) >= 2 and conf >= INFERENCE_MIN_CONFIDENCE:
                hops = len(chain)
                if (hops > best['hops']
                        or (hops == best['hops']
                            and conf > best['conf'])):
                    best['hops'] = hops
                    best['conf'] = conf
                    # A chain that crossed an analogy hop is an
                    # abductive conjecture, not a deduction — flag
                    # it so the renderer hedges and the thought is
                    # communicated as "may", not "is".
                    analogical = any(
                        r == _ANALOGY_RELATION for (_s, r, _o) in chain)
                    best['result'] = {
                        'focal': focal,
                        'relation': accumulated,
                        'target': node,
                        'confidence': conf,
                        'hops': len(chain),
                        'chain': list(chain),
                        'analogical': analogical,
                    }
            if len(chain) >= max_hops:
                return
            for rel, tgt, strength in self._composable_edges(
                    node, visited, accumulated):
                # compose() is the single composition step (shared with
                # family_grain_relations) — behaviour identical to the
                # inline table lookup this replaced.
                composed = compose(accumulated, rel)
                if composed is None:
                    continue
                if accumulated is None:
                    new_conf = conf * min(1.0, strength)
                else:
                    new_conf = (conf * MULTI_HOP_PENALTY
                                * min(1.0, strength))
                # Abductive discount: crossing an analogy edge costs
                # extra confidence (analogical transfer is a
                # conjecture).  Applied once per analogy hop.
                if rel == _ANALOGY_RELATION:
                    new_conf *= ANALOGY_TRANSFER_PENALTY
                extend(tgt, composed,
                           chain + [(node, rel, tgt)],
                           visited | {tgt}, new_conf)

        extend(focal, None, [], {focal}, 1.0)
        # Stamp engagement (the quarantine-tier "inference walked it"
        # bid-path) ONLY on edges that became part of the EMITTED
        # chain — the one that survived INFERENCE_MIN_CONFIDENCE and
        # ended up as best['result'].  Sub-threshold explorations
        # and abandoned branches do NOT stamp; otherwise every
        # neighbor of an active focal would be auto-engaged just by
        # being considered, immunizing whole neighborhoods from
        # quarantine.  Round-2 audit fix: "search looked at it" is
        # not engagement; "search returned a derivation that uses
        # it" is.
        result = best['result']
        if result is not None and lts is not None:
            cyc = self._cycle_provider()
            ledger = self._engagement_ledger
            for (s, r, o) in result.get('chain', ()):
                e = lts.edge(s, r, o)
                if e is not None:
                    e.last_engaged_cycle = int(cyc)
                    # Death-wall organ-c (SHADOW): a COHERENT edge USED in
                    # an emitted inference is MEANINGFUL use — it buys
                    # life.  (note_engaged ignores incoherent edges.)
                    if ledger is not None:
                        ledger.note_engaged(e)
        return result

    def _composable_edges(self,
                                node: str,
                                visited: set,
                                accumulated_rel: Optional[str]
                                ) -> list:
        """Top INFERENCE_BRANCH_CAP outgoing edges from `node`,
        as (relation, target, strength), ordered by chemistry-
        modulated score.  Filters revisits and — when
        `accumulated_rel` is given — relations that don't compose
        with it."""
        lts = self._lts()
        if lts is None:
            return []
        scored = []
        for rel in INFERENCE_RELATIONS:
            # Same single composition step as the walk (identical
            # semantics to the previous `(a, rel) not in
            # RELATION_COMPOSITION` membership test).
            if (accumulated_rel is not None
                    and compose(accumulated_rel, rel) is None):
                continue
            for tgt, _r, strength in lts.neighbors(
                    node, relation=rel):
                if tgt in visited or tgt == node:
                    continue
                score = self._modulated_score(
                    min(1.0, strength), tgt)
                scored.append((score, rel, tgt, strength))
        scored.sort(key=lambda t: -t[0])
        # Death-wall organ-b (SHADOW): at the FIRST hop (accumulated_rel
        # is None) each (node, rel) slot's top-ranked edge is the answer
        # the search would follow; an incoherent top is junk in the way.
        # Check the top edge per relation-slot (the ledger dedupes).
        # Deeper hops are search-internal (not a slot answer) → skipped,
        # keeping this off the hot recursion.
        if accumulated_rel is None and self._engagement_ledger is not None:
            seen_rel = set()
            for _s, rel, tgt, _st in scored:
                if rel in seen_rel:
                    continue
                seen_rel.add(rel)
                self._note_slot_obstruction(node, rel, tgt)
        return [(rel, tgt, strength)
                    for _s, rel, tgt, strength
                    in scored[:INFERENCE_BRANCH_CAP]]

    def _render_inference(self, result: dict) -> str:
        """Explain a composed inference as a single line: the
        derived conclusion followed by the chain that licenses
        it."""
        focal = result['focal']
        rel = result['relation'].replace('_', ' ')
        target = result['target']
        steps = []
        for (s, r, o) in result['chain']:
            steps.append(f"{s} {r.replace('_', ' ')} {o}")
        because = '; '.join(steps)
        if result.get('analogical'):
            # Abductive — hedge it.  "X may have P (by analogy)."
            return (f"I suspect {focal} {rel} {target} — by analogy: "
                      f"{because}.")
        return (f"I infer {focal} {rel} {target} — "
                  f"because {because}.")

    # ---- causal chain (1-2 hops on LTS) ----

    def _causal_chain(self,
                         focal: str,
                         max_hops: int = 2
                         ) -> Optional[tuple]:
        """Walk causes/leads_to/produces relations up to
        max_hops.  Returns (focal, relation, terminal, confidence,
        hops, evidence_strength, walked_hops) or None, where
        `walked_hops` is the list of REAL (s, r, o) edges actually
        traversed — 1 entry for a 1-hop walk, 2 for a 2-hop walk.

        The collapsed (focal, relation, terminal) is for RENDERING
        only.  Phase-2 per-hop reinforcement (2026-06-01) reinforces
        `walked_hops` — the genuine A->X->B evidence — instead of the
        fabricated collapsed end-to-end edge (which has no composing
        path of its own and never coheres).  See
        project_seagi_dry_reverie_igniter.

        Walk selection is chemistry-modulated: when multiple
        targets compete, the one whose accumulated NT trace
        resonates with current global chemistry wins.  Stored
        confidence stays raw — chemistry steers WHICH neighbor
        we pursue, but does not inflate the reasoning's own
        certainty downstream."""
        lts = self._lts()
        if lts is None:
            return None
        # 1-hop pass.
        best: Optional[Tuple[str, str, str, float, int, float]] = None
        best_hops: list = []
        best_modulated: float = -1e9
        for rel in CAUSAL_RELATIONS:
            edges = lts.neighbors(focal, relation=rel)
            for tgt, _r, strength in edges:
                conf = min(1.0, strength)
                modulated = self._modulated_score(conf, tgt)
                if modulated > best_modulated:
                    best_modulated = modulated
                    best = (focal, rel, tgt, conf, 1, strength)
                    best_hops = [(focal, rel, tgt)]
        # 2-hop pass — extend the best 1-hop.
        if best is not None and max_hops >= 2:
            f1, r1, mid, conf1, _, str1 = best
            for rel2 in CAUSAL_RELATIONS:
                edges2 = lts.neighbors(mid, relation=rel2)
                for tgt2, _r, str2 in edges2:
                    if tgt2 == focal:
                        continue  # no self-loops
                    conf2 = conf1 * MULTI_HOP_PENALTY * (
                        min(1.0, str2))
                    modulated2 = self._modulated_score(conf2, tgt2)
                    # Prefer 2-hop when it's near-as-good as the
                    # current best — multi-hop is more
                    # interesting if it matches energy.
                    if modulated2 > best_modulated * 0.9:
                        best_modulated = modulated2
                        best = (f1, r1, tgt2, conf2, 2,
                                  (str1 + str2) / 2.0)
                        # The REAL evidence is the two genuine hops,
                        # NOT the collapsed (f1, r1, tgt2).
                        best_hops = [(f1, r1, mid),
                                       (mid, rel2, tgt2)]
        if best is None:
            return None
        return (*best, best_hops)

    def _render_causal(self,
                          focal: str,
                          relation: str,
                          target: str,
                          hops: int,
                          evidence: float) -> str:
        rel = relation.replace('_', ' ')
        if hops == 1:
            return f"{focal} {rel} {target}."
        # 2-hop.
        return (f"Thinking about {focal} — through "
                  f"{rel} chain, I reach {target}.")

    def _first_neighbor(self,
                          focal: str,
                          relations: Tuple[str, ...]
                          ) -> Optional[Tuple[str, str, float]]:
        """Return the strongest (relation, target, strength) for
        the first relation that has any edges.  Within the chosen
        relation, picks by chemistry-modulated score — so
        mood-congruent identity/process targets beat tied-strength
        alternatives.  Relation-type priority is preserved
        (is_a beats has_property even when has_property has
        stronger raw edges)."""
        lts = self._lts()
        if lts is None:
            return None
        for rel in relations:
            edges = lts.neighbors(focal, relation=rel)
            if not edges:
                continue
            # Pick best by chemistry-modulated score within
            # this relation.
            best = max(
                edges,
                key=lambda t: self._modulated_score(t[2], t[0]))
            tgt, r, strength = best
            # Death-wall organ-b (SHADOW): the chosen answer for this
            # (focal, rel) slot; an incoherent answer = junk in the way.
            self._note_slot_obstruction(focal, rel, tgt)
            return (r, tgt, strength)
        return None

    # ---- counterfactual hint ----

    def _counterfactual_hint(self, focal: str) -> Optional[str]:
        """Sketch what would change without focal.  Lightweight
        version of v1 counterfactual_run — just counts neighbors
        that would lose their predicted source."""
        lts = self._lts()
        if lts is None:
            return None
        # Concepts focal causes/produces — would lose those.
        downstream = []
        for rel in CAUSAL_RELATIONS:
            for tgt, _r, _s in lts.neighbors(focal, relation=rel):
                downstream.append(tgt)
        if not downstream:
            return None
        sample = ', '.join(downstream[:3])
        return (f"If {focal} did not hold, I would lose my "
                  f"sense of {sample}.")

    # ---- schema matching on AWM ----

    def _match_schemas(self,
                          bus: EventBus,
                          trigger: str = ''
                          ) -> List[Tuple[Schema, Dict[str, str]]]:
        """Run STANDARD_SCHEMAS against current AWM contents.
        Each pattern is matched ONLY against concepts in AWM —
        sparse and fast.  When a match fires, emit substrate-
        write requests for the schema's inferences."""
        from seagi.core.substrate import PROVISIONAL_EDGE_STRENGTH
        awm = self._awm()
        lts = self._lts()
        if awm is None or lts is None:
            return []
        active = awm.active_concepts()
        if len(active) < 2:
            return []
        matches: List[Tuple[Schema, Dict[str, str]]] = []
        active_set = set(active)
        # Build a small relation-indexed view OVER AWM ONLY.
        # This is the speed win: not over all LTS edges, just
        # edges that touch current AWM.
        for schema in STANDARD_SCHEMAS:
            for binding in self._enumerate_bindings(
                    schema, active_set, lts):
                matches.append((schema, binding))
                self.schemas_matched += 1
                cycle = self._cycle_provider()
                # Emit substrate-write requests for inferences.
                for (s_slot, rel, o_slot) in schema.inferences:
                    s = binding.get(s_slot, '')
                    o = binding.get(o_slot, '')
                    if not (s and o):
                        continue
                    if lts.edge(s, rel, o) is not None:
                        continue  # already exists
                    bus.publish(SubstrateWriteQueuedEvent(
                        kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                        cycle=cycle,
                        timestamp=time.time(),
                        source_capability='cortical',
                        origin='internal',
                        origin_detail=schema.name,
                        subject=s, relation=rel, object=o,
                        strength=PROVISIONAL_EDGE_STRENGTH,
                        write_reason=(
                            f'schema:{schema.name}')))
                    self.inferences_emitted += 1
        return matches

    def _enumerate_bindings(self,
                                  schema: Schema,
                                  active: Set[str],
                                  lts: Any
                                  ) -> List[Dict[str, str]]:
        """Find slot bindings for `schema` constrained to AWM.

        Strategy: at least one slot must be in AWM (otherwise
        we'd consider patterns entirely in LTS — that's the v1
        cost we're avoiding).  For each AWM concept, attempt to
        bind it to each slot, then check premises via LTS edges.
        """
        slot_names: List[str] = []
        for (s, _r, o) in schema.premises:
            if s not in slot_names:
                slot_names.append(s)
            if o not in slot_names:
                slot_names.append(o)
        out: List[Dict[str, str]] = []
        # SAFETY (2026-07-24): bound total enumeration work per schema pass so a
        # densely-connected hub concept cannot make the _extend recursion explode
        # combinatorially (completes the len(out)>=20 intent, which was only
        # checked BETWEEN start-concepts, never inside the recursion).
        self._sched_budget = 4000
        # Try each AWM concept as a starting binding for slot 0.
        first_slot = slot_names[0]
        for first_val in active:
            bindings = {first_slot: first_val}
            self._extend(schema, slot_names, 1, bindings,
                            active, lts, out)
            if len(out) >= 20 or self._sched_budget <= 0:   # cap per pass
                break
        return out

    def _extend(self,
                  schema: Schema,
                  slot_names: List[str],
                  idx: int,
                  bindings: Dict[str, str],
                  active: Set[str],
                  lts: Any,
                  out: List[Dict[str, str]]) -> None:
        if self._sched_budget <= 0 or len(out) >= 20:
            return
        if idx >= len(slot_names):
            # All slots bound — verify all premises.
            for (s_slot, rel, o_slot) in schema.premises:
                s = bindings.get(s_slot, '')
                o = bindings.get(o_slot, '')
                if not lts.edge(s, rel, o):
                    return
            # Reject degenerate self-bindings.
            if len(set(bindings.values())) < len(bindings):
                return
            out.append(dict(bindings))
            return
        slot = slot_names[idx]
        # Determine candidates: walk premises to find what
        # values are constrained.
        for (s_slot, rel, o_slot) in schema.premises:
            if s_slot == slot and o_slot in bindings:
                # source slot unbound, object bound — query
                # backward.  Phase 3 doesn't have inverse
                # index; fall through to forward iteration.
                pass
            if (o_slot == slot and s_slot in bindings):
                # forward: get neighbors of bound source.
                for tgt, _r, _s in lts.neighbors(
                        bindings[s_slot], relation=rel):
                    self._sched_budget -= 1
                    if self._sched_budget <= 0 or len(out) >= 20:
                        return
                    new = dict(bindings)
                    new[slot] = tgt
                    if len(set(new.values())) == len(new):
                        self._extend(schema, slot_names, idx + 1,
                                        new, active, lts, out)
                return
        # No constraint on this slot from already-bound — try
        # any AWM concept.
        for v in active:
            self._sched_budget -= 1
            if self._sched_budget <= 0 or len(out) >= 20:
                return
            new = dict(bindings)
            new[slot] = v
            if len(set(new.values())) == len(new):
                self._extend(schema, slot_names, idx + 1,
                                new, active, lts, out)

    # ---- snapshot for SpeechRequest ----

    def _awm_snapshot(self) -> Dict[str, Any]:
        awm = self._awm()
        if awm is None:
            return {}
        return {
            'active_concepts': awm.active_concepts()[:16],
            'size': awm.size(),
        }

    # ---- diagnostics ----

    def stats(self) -> Dict[str, int]:
        return {
            'thoughts_produced': self.thoughts_produced,
            'responses_composed': self.responses_composed,
            'schemas_matched': self.schemas_matched,
            'inferences_emitted': self.inferences_emitted,
        }
