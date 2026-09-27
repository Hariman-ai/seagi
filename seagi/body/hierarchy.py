"""Layer 2 — Predictive coding hierarchy.

Per AGI_ENGINE_DESIGN.md §5. The engine that thinks. Six levels:

    L0 sensory token / L1 concept / L2 proposition /
    L3 scene / L4 intention / L5 self-model

Predictions flow down through substrate edges with M/I propagating
per Layer 0.3.5.2 (multiplicative through edge strength + probabilistic
OR with edge MIValue). Errors flow up, M/I-channel-separated, updating
substrate parameters via the prediction-ledger API.

Phase 1 implementation scope:
    - All six LatentState slots exist as data structures.
    - Active forward/backward pass: L2 ↔ L1 (proposition ↔ concept).
      This is where substrate-edges-as-parameters comes alive.
    - L0 is the input source (token strings → concept activations).
    - L3, L4, L5 latents exist as passive stubs (hold state but
      do not actively predict). Wiring expands when those layers
      are implemented (Layer 5 design done; Layer 3 design done).
    - Free-energy objective implemented for the active levels.
    - M/I-modulated decay per Layer 0.3.5.5.
    - Lateral recurrence with α=0.5.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (Dict, List, Tuple, Optional, Set, Iterable, Any,
                    Hashable)

from seagi.core.mi_value import MIValue, _clamp01
from seagi.core.substrate import (Substrate, Concept, Edge, EdgeKey,
                        Episode, Prediction)


# ---------------------------------------------------------------------
# Level identifiers
# ---------------------------------------------------------------------

L0_TOKEN = 0
L1_CONCEPT = 1
L2_PROPOSITION = 2
L3_SCENE = 3
L4_INTENTION = 4
L5_SELF = 5

NUM_LEVELS = 6


# ---------------------------------------------------------------------
# LatentState — per-level state
# ---------------------------------------------------------------------

@dataclass
class LatentState:
    """Sparse MIValue-tagged distribution over elements at one level.

    Per design §5.2:
        expected:    P(element active) tagged with M/I (the prediction)
        precision:   how confidently each element is predicted
        error:       M/I-tagged residual after observation, last cycle
        actuals:     what actually fired this cycle (set by observe());
                     authoritative source of "what's true now," distinct
                     from the error dict which captures residuals
        tick_last_updated: cycle counter
    """
    level: int
    expected: Dict[Hashable, MIValue] = field(default_factory=dict)
    precision: Dict[Hashable, float] = field(default_factory=dict)
    error: Dict[Hashable, MIValue] = field(default_factory=dict)
    actuals: Dict[Hashable, MIValue] = field(default_factory=dict)
    tick_last_updated: int = 0

    def clear_error(self) -> None:
        self.error = {}

    def clear_actuals(self) -> None:
        self.actuals = {}

    def clear_expected(self) -> None:
        self.expected = {}
        self.precision = {}

    def merge_expected(self, key: Hashable,
                       mi: MIValue, precision: float = 1.0) -> None:
        """Add an element to the expected distribution. If already
        present, combine via probabilistic OR per Layer 0.3.5.2.
        Defensive against callers that populated expected without
        precision (or vice versa)."""
        if key in self.expected:
            self.expected[key] = self.expected[key].combine_or(mi)
            self.precision[key] = max(
                self.precision.get(key, 0.0), precision)
        else:
            self.expected[key] = mi
            self.precision[key] = precision

    def magnitude(self) -> float:
        """Total expected-magnitude across all elements. Indicates
        how much 'is happening' at this level."""
        return sum(v.magnitude for v in self.expected.values())


# ---------------------------------------------------------------------
# Hierarchy — the six-level engine
# ---------------------------------------------------------------------

class Hierarchy:
    """The six-level predictive coding hierarchy.

    Holds a LatentState for each level. Provides:
        forward_pass(tone)              — L5 → L0 prediction flow
        observe(level, observation)     — set actual state at a level
        backward_pass(tone)              — L0 → L5 error flow
        free_energy()                    — total system loss
        decay_step(lifeforce)            — Layer 0.3.5.5 decay
        cycle(observation, tone, ...)    — full cycle in one call

    Phase 1: only the L2 ↔ L1 transition is active. Other transitions
    are pass-through (no-op).
    """

    # Lateral recurrence weight (Layer 2 §5.7). Initial 0.5.
    LATERAL_ALPHA = 0.5

    # L2-specific lateral persistence (separate from LATERAL_ALPHA).
    # Higher = propositions survive more turns. Cross-turn scene
    # formation needs propositions to outlive a single observation.
    # 0.5 → ~3 turns above threshold; 0.75 → ~8 turns. We use 0.75
    # so multi-turn dialog accumulates enough propositions for L3
    # scene formation without keeping stale propositions forever.
    L2_LATERAL_PERSISTENCE = 0.75

    # Decay constants (Layer 0.3.5.5). Initial values per spec.
    DECAY_BASE = 0.05
    DECAY_N_REF = 10.0
    DECAY_LIFEFORCE_ALPHA = 1.0

    # Edge update magnitudes for backward pass.
    EDGE_REINFORCE_DELTA = 0.05
    EDGE_WEAKEN_DELTA = 0.03

    def __init__(self, substrate: Substrate, num_levels: int = NUM_LEVELS):
        self.substrate = substrate
        self.num_levels = num_levels
        self.latents: List[LatentState] = [
            LatentState(level=k) for k in range(num_levels)
        ]
        # Cycle counter — independent of substrate, drives latent
        # decay and lateral recurrence. Caller advances via tick().
        self.cycle_counter: int = 0

    # ---- accessors ----

    def latent(self, level: int) -> LatentState:
        return self.latents[level]

    @property
    def L1(self) -> LatentState:
        return self.latents[L1_CONCEPT]

    @property
    def L2(self) -> LatentState:
        return self.latents[L2_PROPOSITION]

    @property
    def L3(self) -> LatentState:
        return self.latents[L3_SCENE]

    @property
    def L4(self) -> LatentState:
        return self.latents[L4_INTENTION]

    @property
    def L5(self) -> LatentState:
        return self.latents[L5_SELF]

    # ---- observation ----

    def observe(self,
                level: int,
                actuals: Dict[Hashable, MIValue]
                ) -> None:
        """Inject observed activations at a level. The forward pass's
        prediction at this level becomes ground-truth-comparable;
        backward pass will compute the error.

        At L1 (concepts), `actuals` is {concept_name: MIValue} for
        concepts that fired in the current observation.

        Stores actuals AND the M/I-channel-separated error dict.
        Actuals are the authoritative source of "what fired";
        error captures the residual against expected (both
        over-prediction — predicted but didn't fire — and
        under-prediction — fired but wasn't predicted).
        """
        latent = self.latents[level]
        latent.actuals = dict(actuals)

        err: Dict[Hashable, MIValue] = {}
        # Under-prediction: actual exceeds expected.
        for key, actual_mi in actuals.items():
            predicted_mi = latent.expected.get(key, MIValue.zero())
            err_m = max(0.0, actual_mi.m - predicted_mi.m)
            err_i = max(0.0, actual_mi.i - predicted_mi.i)
            if err_m > 0 or err_i > 0:
                err[key] = MIValue(err_m, err_i, max(actual_mi.n, 1))
        # Over-prediction: predicted but didn't fire.
        for key, predicted_mi in latent.expected.items():
            if key not in actuals:
                err[key] = predicted_mi
        latent.error = err
        latent.tick_last_updated = self.cycle_counter

    # ---- forward pass ----

    # Threshold below which L1 concepts don't anchor a scene candidate.
    SCENE_PROPOSITION_MIN_MAGNITUDE = 0.10
    # Minimum number of co-active propositions required to form a scene.
    SCENE_MIN_SIZE = 2
    # EWMA blend rate for scene MI updates from constituent propositions.
    SCENE_MI_LERP_RATE = 0.30

    def forward_pass(self, tone: MIValue,
                      perceive_edge_fn=None) -> None:
        """Top-down prediction flow. L3 → L2 → L1 active.
        Higher levels (L4, L5) are passive stubs.

        L3 → L2: each active scene at L3 emits its constituent
        propositions to L2 with M/I propagation per Layer 0.3.5.2.
        L2 → L1: each active proposition activates its subject and
        object concepts at L1.

        When `perceive_edge_fn` (engine.perceive_edge_weight) is
        supplied, edge MI in the L2→L1 propagation uses perception-
        time weight (endpoint perception drift × edge.strength)
        instead of the static substrate annotation. Same world fact
        propagates differently depending on agent state.
        """
        # L3 → L2: scene-driven proposition activation. Merges with
        # whatever L2 currently expects (lateral recurrence + new).
        self._forward_l3_to_l2(tone)

        # L2 → L1: existing proposition-driven concept activation.
        self._forward_l2_to_l1(tone, perceive_edge_fn=perceive_edge_fn)

    def _forward_l3_to_l2(self, tone: MIValue) -> None:
        """Each active scene contributes its propositions to L2's
        expected distribution. Pre-existing L2 expectations are
        preserved via merge_expected (probabilistic OR)."""
        for scene_id, scene_mi in list(self.L3.expected.items()):
            scene = self.substrate.scenes.get(scene_id)
            if scene is None:
                continue
            scene_precision = self.L3.precision.get(scene_id, 1.0)
            for prop in scene.propositions:
                # Scene MI propagates to its constituent propositions.
                self.L2.merge_expected(
                    prop, scene_mi,
                    precision=scene_precision * 0.5,
                )

    def _forward_l2_to_l1(self, tone: MIValue,
                            perceive_edge_fn=None) -> None:
        """L2 → L1 propagation (the existing core loop).
        When perceive_edge_fn is supplied, the propagation uses
        perception-time edge weight rather than the static
        edge.mi — felt-relevance shapes prediction flow."""
        # Lateral recurrence: blend previous L1 expected with the
        # incoming top-down predictions.
        prev_L1_expected = dict(self.L1.expected)
        prev_L1_precision = dict(self.L1.precision)

        # Reset L1 expected; will be populated from L2.
        self.L1.clear_expected()

        for prop_key, prop_mi in self.L2.expected.items():
            # prop_key is an EdgeKey: (source, relation, target).
            if not isinstance(prop_key, tuple) or len(prop_key) != 3:
                continue
            source, _relation, target = prop_key
            edge = self.substrate.edges.get(prop_key)
            if edge is None:
                continue
            # M/I propagation per §3.5.2:
            #   activation × strength × combine_or(prop.mi, edge.mi).
            # When perceive_edge_fn is supplied, edge.mi is replaced
            # by the perception-time weight — same fact, different
            # state, different propagation.
            if perceive_edge_fn is not None:
                try:
                    edge_mi = perceive_edge_fn(edge)
                except Exception:
                    edge_mi = edge.mi
            else:
                edge_mi = edge.mi
            propagated = prop_mi.scale(edge.strength).combine_or(edge_mi)
            # Both subject and object concepts get activated.
            self.L1.merge_expected(source, propagated,
                                    precision=edge.strength)
            self.L1.merge_expected(target, propagated,
                                    precision=edge.strength)

        # Apply lateral recurrence: blend with previous L1 state.
        if prev_L1_expected:
            blended_expected: Dict[Hashable, MIValue] = {}
            blended_precision: Dict[Hashable, float] = {}
            all_keys = (set(self.L1.expected.keys())
                        | set(prev_L1_expected.keys()))
            for key in all_keys:
                prev = prev_L1_expected.get(key, MIValue.zero())
                cur = self.L1.expected.get(key, MIValue.zero())
                blended_expected[key] = prev.lerp(
                    cur, 1.0 - self.LATERAL_ALPHA)
                p_prev = prev_L1_precision.get(key, 0.0)
                p_cur = self.L1.precision.get(key, 0.0)
                blended_precision[key] = (
                    p_prev * self.LATERAL_ALPHA
                    + p_cur * (1.0 - self.LATERAL_ALPHA))
            self.L1.expected = blended_expected
            self.L1.precision = blended_precision

        # Apply tone-modulated precision (Layer 0.3.5.3).
        for key, mi in self.L1.expected.items():
            base = self.L1.precision.get(key, 1.0)
            tone_factor = (
                (1.0 + tone.m * mi.m) * (1.0 + tone.i * mi.i))
            self.L1.precision[key] = base * tone_factor

        self.L1.tick_last_updated = self.cycle_counter

    # ---- backward pass ----

    def backward_pass(self, tone: MIValue) -> None:
        """Bottom-up error flow. Phase 1 implements L1 → L2 only.
        L1 actuals update edges (substrate parameters) and propose
        new L2 active propositions (the propositions whose subject
        and object both activated).

        Errors carry the M/I tag of the prediction they failed
        against; M-channel and I-channel updates run independently.
        """
        # Authoritative source of "what fired this cycle": L1.actuals,
        # NOT L1.error. Conflating these would treat over-predictions
        # (predicted but didn't fire) as activations — a bug class.
        actual_concepts = set(self.L1.actuals.keys())
        if not actual_concepts and not self.L1.error:
            return

        # For each L2 proposition that was predicted (even at low
        # weight), check whether its participants fired.
        for prop_key, prop_mi in list(self.L2.expected.items()):
            if not isinstance(prop_key, tuple) or len(prop_key) != 3:
                continue
            source, _relation, target = prop_key
            edge = self.substrate.edges.get(prop_key)
            if edge is None:
                continue

            both_fired = (source in actual_concepts
                           and target in actual_concepts)
            either_fired = (source in actual_concepts
                             or target in actual_concepts)

            # M/I tagging of the outcome update derives from the edge's
            # current M/I and the tone (under M-tone, M-channel updates
            # weight more heavily).
            m_weight = _clamp01(edge.mi.m * (1.0 + tone.m))
            i_weight = _clamp01(edge.mi.i * (1.0 + tone.i))

            if both_fired:
                # Confirmation: edge reinforces, evidence records hit.
                edge.reinforce(cycle=self.cycle_counter,
                                delta=self.EDGE_REINFORCE_DELTA,
                                origin='cognition')
                edge.evidence.record('hit', m_weight, i_weight,
                                       cycle=self.cycle_counter)
            elif either_fired:
                # Partial: inconclusive (one participant fired but not
                # the other). Don't reinforce or weaken; just log.
                edge.evidence.record('inconclusive', m_weight, i_weight,
                                       cycle=self.cycle_counter)
            else:
                # Predicted but didn't fire: weaken edge slightly.
                edge.weaken(delta=self.EDGE_WEAKEN_DELTA)
                edge.evidence.record('miss', m_weight, i_weight,
                                       cycle=self.cycle_counter)

        # New L2 propositions can form when concept pairs co-fire
        # along an existing substrate edge that wasn't already in L2.
        # This is where comprehension lives — input → propositions.
        new_L2_props: Dict[Hashable, MIValue] = {}
        # Lateral recurrence on L2: keep existing predictions with
        # L2_LATERAL_PERSISTENCE weight (higher than L1's α to enable
        # cross-turn scene formation).
        for prop_key, prop_mi in self.L2.expected.items():
            new_L2_props[prop_key] = prop_mi.scale(
                self.L2_LATERAL_PERSISTENCE)
        for src in actual_concepts:
            src_concept = self.substrate.concepts.get(src)
            if src_concept is None:
                continue
            for tgt_name, edge in src_concept.neighbors():
                if tgt_name not in actual_concepts:
                    continue
                key = edge.key
                # Both source and target fired — proposition is active.
                src_mi = self.L1.actuals.get(src, MIValue.zero())
                tgt_mi = self.L1.actuals.get(tgt_name, MIValue.zero())
                fired_mi = src_mi.combine_or(tgt_mi).combine_or(edge.mi)
                if key in new_L2_props:
                    new_L2_props[key] = new_L2_props[key].combine_or(fired_mi)
                else:
                    new_L2_props[key] = fired_mi
        self.L2.expected = new_L2_props
        # Recompute L2 precision from edge strengths and tone.
        self.L2.precision = {}
        for key in self.L2.expected:
            edge = self.substrate.edges.get(key) if isinstance(
                key, tuple) else None
            base = edge.strength if edge is not None else 0.5
            mi = self.L2.expected[key]
            self.L2.precision[key] = (
                base * (1.0 + tone.m * mi.m) * (1.0 + tone.i * mi.i))
        self.L2.tick_last_updated = self.cycle_counter

        # L2 → L3: form/reinforce scenes from co-active propositions.
        self._backward_l2_to_l3(tone)

    def _backward_l2_to_l3(self, tone: MIValue) -> None:
        """Identify the set of currently-active L2 propositions and
        promote them to a Scene at L3 (creating it if new).

        A scene is created/reinforced when ≥SCENE_MIN_SIZE
        propositions are above SCENE_PROPOSITION_MIN_MAGNITUDE.
        Scene MI is the EWMA of its constituent propositions' MI.
        """
        active_props = frozenset(
            prop for prop, mi in self.L2.expected.items()
            if isinstance(prop, tuple)
            and len(prop) == 3
            and mi.magnitude >= self.SCENE_PROPOSITION_MIN_MAGNITUDE
        )
        if len(active_props) < self.SCENE_MIN_SIZE:
            return
        try:
            scene = self.substrate.get_or_create_scene(
                active_props, cycle=self.cycle_counter)
        except ValueError:
            return  # empty propositions guard
        scene.activate(self.cycle_counter)

        # Aggregate scene MI from constituent propositions and
        # blend (EWMA) into scene's persistent MI.
        aggregate = MIValue.zero()
        for prop in active_props:
            aggregate = aggregate.combine_or(
                self.L2.expected.get(prop, MIValue.zero()))
        scene.update_mi(scene.mi.lerp(aggregate, self.SCENE_MI_LERP_RATE))

        # Add the scene to L3 expected so it conditions next cycle's
        # forward pass. Tone-modulated precision.
        precision = (1.0 + tone.m * aggregate.m) * (1.0 + tone.i * aggregate.i)
        self.L3.expected[scene.id] = aggregate
        self.L3.precision[scene.id] = precision
        self.L3.tick_last_updated = self.cycle_counter

    # ---- L4 intention level ----

    # Source labels for intentions populated into L4.
    INTENTION_SOURCE_VALUE = 'value'         # from SelfModel.value_priors
    INTENTION_SOURCE_THREAD = 'thread'       # from open_threads
    INTENTION_SOURCE_PEER_GOAL = 'peer'      # from peer.current_goal
    INTENTION_SOURCE_SPEECH_ACT = 'speech_act'  # from current input shape

    # Per-source default weight in L4.precision.
    _INTENTION_WEIGHTS = {
        'value': 1.0,         # standing values are durable
        'thread': 1.5,        # open conversation threads pull harder
        'peer': 0.8,          # modeled peer goals (lower confidence)
        'speech_act': 1.5,    # input shape is strong-signal but transient
    }

    # Identity-concept activation magnitude when L5 priors are
    # written into L1. Small but always-on — keeps self-related
    # concepts slightly active so introspection can fire.
    L5_IDENTITY_ACTIVATION = 0.15

    def condition_lower_levels_from_l5(self, l5_state) -> None:
        """L5 self-model conditions L1/L2 priors directly.

        Per Layer 2 §5.6: identity-related concepts (SEAGI, mortality,
        the user, etc.) get baseline activation at L1 even without
        explicit input. SelfModel.identity propositions also become
        always-active L2 priors.

        This is the always-on top-down conditioning the design
        committed to. Without it, self-state has no presence in the
        engine's prediction pass — only in EFE.
        """
        if l5_state is None:
            return
        sm = l5_state.self_model
        # Extract single-word concepts from identity / value priors
        # to inject as L1 baseline activation.
        identity_concepts: Set[str] = set()
        for prop_text in sm.identity:
            for token in str(prop_text).lower().split():
                tok = token.strip('.,!?;:')
                if len(tok) > 3 and tok in self.substrate.concepts:
                    identity_concepts.add(tok)
        for target_text, _mi in sm.value_priors:
            tok = str(target_text).lower().strip()
            if tok in self.substrate.concepts:
                identity_concepts.add(tok)
        for concept_name in identity_concepts:
            concept = self.substrate.concepts.get(concept_name)
            if concept is None:
                continue
            self.L1.merge_expected(
                concept_name,
                MIValue(0.0, self.L5_IDENTITY_ACTIVATION,
                         max(concept.mi.n, 1)),
                precision=0.3,
            )

    def populate_l4(self,
                    l5_state,
                    peer_id: Optional[str] = None,
                    speech_act_intentions: Optional[
                        List[Tuple]] = None) -> None:
        """Populate L4 from L5 — SelfModel.value_priors,
        open_threads on the active relationship, peer's modeled
        current_goal, and (when provided) speech-act intentions
        from the current input.

        speech_act_intentions: list of (mi, weight, topics) tuples
        from speech_act.speech_act_to_intentions(). Each becomes a
        new L4 entry under the 'speech_act' source. Topics encode
        the concepts the user is asking about; downstream Layer 3
        candidate generation can use them to boost actions touching
        those concepts.

        Each intention is keyed `(source, frozenset_of_propositions
        OR frozenset_of_topic_names)`. Multiple sources contribute
        independently; their pulls combine in aggregate_intention_pull.

        L5 is left untouched; L4 is rebuilt each call from current L5.
        """
        # Reset L4 each cycle. Intentions are ephemeral runtime
        # objects; persistence lives in L5.
        self.L4.clear_expected()
        self.L4.error = {}

        if l5_state is None:
            return

        # Source 1: standing value priors from SelfModel.
        for target_text, mi in (l5_state.self_model.value_priors or []):
            key = (self.INTENTION_SOURCE_VALUE, frozenset())
            existing = self.L4.expected.get(key, MIValue.zero())
            self.L4.expected[key] = existing.combine_or(mi)
            self.L4.precision[key] = self._INTENTION_WEIGHTS['value']

        # Source 2: open conversation threads (if peer specified).
        if peer_id is not None:
            rel = l5_state.relationships.get(
                (l5_state.seagi_id, peer_id))
            if rel is not None and rel.open_threads:
                # Treat each open thread as an intention with
                # neutral M/I (we want closure regardless of
                # polarity).
                key = (self.INTENTION_SOURCE_THREAD, frozenset())
                self.L4.expected[key] = MIValue(0.0, 0.3,
                                                 len(rel.open_threads))
                self.L4.precision[key] = self._INTENTION_WEIGHTS['thread']

            # Source 3: peer's modeled goal (if we have one). For
            # now, use peer.mi_pull as the directional component
            # (modeling "I want to satisfy what the peer pulls toward").
            peer = l5_state.peers.get(peer_id)
            if peer is not None and peer.confidence > 0.0:
                key = (self.INTENTION_SOURCE_PEER_GOAL, frozenset())
                self.L4.expected[key] = peer.mi_pull
                self.L4.precision[key] = (
                    self._INTENTION_WEIGHTS['peer'] * peer.confidence)

        # Source 4: speech-act intentions from current input.
        # Each contribution carries (mi, weight, topics). Topics are
        # encoded in the L4 key so candidate-scoring code can find
        # them; mi sets directional preference; weight sets pull magnitude.
        if speech_act_intentions:
            for mi, weight, topics in speech_act_intentions:
                key = (self.INTENTION_SOURCE_SPEECH_ACT,
                        frozenset(topics))
                self.L4.expected[key] = mi
                self.L4.precision[key] = (
                    self._INTENTION_WEIGHTS['speech_act'] * weight)

        self.L4.tick_last_updated = self.cycle_counter

    def aggregate_intention_pull(self) -> MIValue:
        """Return the aggregate MI pull across all active intentions.
        Layer 3 EFE uses this as an additional pragmatic preference
        (parallel to peer-pragmatic alignment from Layer 5)."""
        if not self.L4.expected:
            return MIValue.zero()
        pull = MIValue.zero()
        for key, mi in self.L4.expected.items():
            weight = self.L4.precision.get(key, 1.0)
            scaled = mi.scale(min(1.0, weight))
            pull = pull.combine_or(scaled)
        return pull

    def intention_alignment(self,
                            observation_mi: MIValue) -> float:
        """How aligned an action's expected observation is with the
        engine's active intentions. Returns NEGATIVE for aligned
        observations (preferred), zero for orthogonal/no intentions.

        Channel-wise dot product summed across intentions, weighted
        by precision. Same architectural pattern as
        PeerModel.pragmatic_alignment."""
        if not self.L4.expected:
            return 0.0
        total = 0.0
        for key, intention_mi in self.L4.expected.items():
            weight = self.L4.precision.get(key, 1.0)
            alignment = (observation_mi.m * intention_mi.m
                         + observation_mi.i * intention_mi.i)
            total += weight * alignment
        return -total

    # ---- free energy ----

    def free_energy(self) -> Dict[str, float]:
        """Sum of M/I-channel-separated losses across all levels.
        Returns {'m': ..., 'i': ..., 'total': ...}.

        Per Layer 2 §5.5:
            loss_m_k = sum_e precision_k(e) * error_k(e).m^2
            loss_i_k = sum_e precision_k(e) * error_k(e).i^2
        """
        loss_m = 0.0
        loss_i = 0.0
        for latent in self.latents:
            for key, err in latent.error.items():
                p = latent.precision.get(key, 1.0)
                loss_m += p * err.m * err.m
                loss_i += p * err.i * err.i
        return {'m': loss_m, 'i': loss_i, 'total': loss_m + loss_i}

    # ---- decay ----

    def decay_step(self, lifeforce: float) -> None:
        """Apply Layer 0.3.5.5 decay to all latent states.

            decay_rate(e) = base
                          · (1 / (1 + m + i))
                          · (1 / (1 + n / N_REF))
                          · (1 + alpha * (1 - L))

        Strongly-tagged elements decay slowest. Low lifeforce
        accelerates decay globally. Decay rate is never zero.
        """
        L = _clamp01(lifeforce)
        global_scale = 1.0 + self.DECAY_LIFEFORCE_ALPHA * (1.0 - L)

        for latent in self.latents:
            new_expected: Dict[Hashable, MIValue] = {}
            for key, mi in latent.expected.items():
                anchor = 1.0 / (1.0 + mi.m + mi.i)
                evidence = 1.0 / (1.0 + mi.n / self.DECAY_N_REF)
                rate = (self.DECAY_BASE * anchor * evidence
                        * global_scale)
                rate = min(0.95, rate)  # never wipe completely in one step
                # Decay each channel.
                new_m = mi.m * (1.0 - rate)
                new_i = mi.i * (1.0 - rate)
                # Drop elements that have decayed below a small floor.
                if new_m < 1e-3 and new_i < 1e-3:
                    continue
                new_expected[key] = MIValue(new_m, new_i, mi.n)
            latent.expected = new_expected
            # Precision survives but scales down with the same rate.
            latent.precision = {
                k: latent.precision.get(k, 0.0)
                for k in new_expected
            }
            # Errors don't accumulate across cycles without observation.
            latent.error = {}

    # ---- one full cycle ----

    def cycle(self,
              observation: Optional[Dict[Hashable, MIValue]] = None,
              tone: Optional[MIValue] = None,
              lifeforce: float = 0.7,
              ) -> Dict[str, Any]:
        """Run one cycle of the engine.

        Sequence:
            1. Forward pass (predictions flow down).
            2. Observe (if observation provided) — sets L1 actuals
               and computes L1 error against this cycle's predictions.
            3. Backward pass (errors propagate up; substrate updates).
            4. Compute free energy (against this cycle's residual).
            5. Decay step (latent states age, prepare for next cycle).
            6. Increment cycle counter.

        Returns a dict with the cycle's free-energy + counts.
        """
        if tone is None:
            tone = MIValue.zero()

        # Clear last cycle's actuals before forward pass.
        for lat in self.latents:
            lat.clear_actuals()

        self.forward_pass(tone)

        if observation is not None:
            self.observe(L1_CONCEPT, observation)
            self.backward_pass(tone)

        fe = self.free_energy()
        self.decay_step(lifeforce)

        result = {
            'cycle': self.cycle_counter,
            'free_energy': fe,
            'L1_active': len(self.L1.expected),
            'L2_active': len(self.L2.expected),
        }
        self.cycle_counter += 1
        return result

    # ---- introspection ----

    def stats(self) -> Dict[str, Any]:
        return {
            'cycle': self.cycle_counter,
            'levels': [
                {
                    'level': lat.level,
                    'expected': len(lat.expected),
                    'error': len(lat.error),
                    'tick_last_updated': lat.tick_last_updated,
                }
                for lat in self.latents
            ],
            'free_energy': self.free_energy(),
        }
