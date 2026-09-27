"""Layer 1 — Substrate as generative-model parameters.

Per AGI_ENGINE_DESIGN.md §4. The substrate is the conditional-parameter
storage of the generative model: typed Concept, Edge, Relation, plus
Episode and Prediction stores, plus a single query primitive that
replaces the parallel composition paths of current SEAGI.

This module defines the types and the Substrate container. Layer 2
(predictive coding hierarchy) operates on these types via forward
and backward passes. Layer 3 (active inference) reads them via the
query primitive when scoring candidate actions.

Design commitments:

- Every substrate element carries first-class MIValue (Layer 0
  foundation). No element is M/I-blind.
- Edges are first-class typed objects (closes the audit's biggest gap).
  Their MIValue propagates per Layer 0.3.5.2 rules.
- Relation types are typed objects with mi_prior — the seed M/I for
  newly-formed edges of that relation type.
- Single Substrate.query() entry point. No side-channels into
  internal data structures.
- Full persistence: to_dict / from_dict round-trips for everything,
  enabling brain_save in the new engine.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import (Dict, List, Tuple, Optional, Set, Iterable, Any,
                    Union, Literal)
import math

from .mi_value import MIValue, TransmitterState
from seagi.core.mi_embedding import MIEmbedding


EdgeKey = Tuple[str, str, str]  # (source_concept, relation_name, target_concept)


# ---------------------------------------------------------------------
# Phase C 2026-05-15 — frequency salience + decay constants.
#
# Doctrine rule 5: "use it or lose it."  Concepts that fire often
# gain salience (brightness, easy to retrieve, easy to connect).
# Concepts that rarely fire fade.  Decay is sub-promille per cycle —
# slow enough that bumps every ~100 cycles (typical-use cadence)
# integrate into accumulating brightness, fast enough that genuinely
# dark pathways halve over a long stretch (~50k cycles from 1.0).
#
# Math: with bump 0.005 every 100 cycles, decay 0.001/100 = 0.00001
# means per-100-cycle net = bump - 100*decay = 0.005 - 0.001 = +0.004
# → frequent use stays bright.  Disuse: from 1.0, halves in ~50k
# cycles, reaches zero in ~100k.  Brain-scale: a concept's character
# settles over many uses and only fades after very long disuse.
# ---------------------------------------------------------------------

SALIENCE_BUMP_DEFAULT = 0.005       # promille bump on each activation
SALIENCE_DECAY_PER_CYCLE = 0.00001  # absolute decay per cycle

EDGE_STRENGTH_BUMP_PER_USE = 0.005  # bump on each cortical traversal
EDGE_STRENGTH_DECAY_PER_CYCLE = 0.00001  # absolute decay per cycle

# ---------------------------------------------------------------------
# Reinforce observer (mortality-clock promotion 2026-07-18).
# A single module-level observer may register to be notified after every
# Edge.reinforce strength bump:
#     on_reinforce(edge, cycle, pre_strength, origin)
# Observer failures NEVER perturb the reinforce, but they are counted
# VISIBLY (reinforce_observer_errors — surfaced in /status as
# observer_errors), never silently swallowed.
# ---------------------------------------------------------------------
_REINFORCE_OBSERVER = None          # Optional[Callable]
_REINFORCE_OBSERVER_ERRORS = 0


def register_reinforce_observer(fn) -> None:
    """Register `fn(edge, cycle, pre_strength, origin)` as THE reinforce
    observer (single slot; a new registration replaces the old)."""
    global _REINFORCE_OBSERVER
    _REINFORCE_OBSERVER = fn


def unregister_reinforce_observer() -> None:
    """Clear the reinforce observer (test hygiene)."""
    global _REINFORCE_OBSERVER
    _REINFORCE_OBSERVER = None


def reinforce_observer_errors() -> int:
    """VISIBLE count of observer exceptions swallowed at the reinforce
    site.  Non-zero means the mortality clock is silently missing earn
    signal — must be exposed in /status."""
    return int(_REINFORCE_OBSERVER_ERRORS)

# Phase S.1 (2026-05-22) — substrate self-cleaning.
# Doctrine rules 4 + 5: "most encounters leave only a small trace
# and dissolve" / "use it or lose it."  A freshly-parsed edge is
# ONE encounter — it must start as a small PROVISIONAL trace, not
# a half-certain 0.5 fact, and it must dissolve unless it earns
# permanence (reinforced by use, re-attestation, or coherence).
PROVISIONAL_EDGE_STRENGTH = 0.1     # strength of a freshly-parsed edge
EDGE_PRUNE_FLOOR = 0.02             # below this effective strength → pruned
# Bounded sample for Substrate.discriminability (2026-08-20).  An
# ESTIMATOR parameter, not a behavioural constant: D is a mean over
# concepts and feeds only the sleep gate's d_modulation ratio, which
# saturates at 0.2 x baseline.  Sweeping all concepts cost ~5.7 s per call
# and 30.1% of the tick.  0 disables sampling (exact sweep).
D_SAMPLE_CONCEPTS = 2000

# Phase S.2 (2026-05-22) — coherence as the earn-signal.
# Re-attestation (the same exact triple recurring) is rare — the
# edge audit found 100% write-once.  The signal that survives
# without recurrence is COHERENCE: a true edge is corroborated by
# the surrounding graph.  An edge A→B coheres when an alternate
# path A→X→B exists — the rest of the structure independently
# agrees with the link.  Coherent edges get reinforced; isolated
# noise edges do not, and dissolve.
#
# Design invariant for the bump size: coherence is a persistent
# STATE, not an event — an edge that stays coherent should stay
# alive.  The reinforcement runs once per maintenance interval
# (EDGE_PRUNE_INTERVAL cycles), so a single coherence bump must
# OUTWEIGH one interval's worth of decay
# (EDGE_STRENGTH_DECAY_PER_CYCLE × EDGE_PRUNE_INTERVAL = 0.02).
# 0.025 holds the line and lets a persistently-coherent edge
# slowly climb toward strong knowledge over the agent's life;
# an isolated edge gets no bump and dissolves in ~5 intervals.
COHERENCE_REINFORCE_BUMP = 0.025    # > one interval's decay (0.02)
COHERENCE_NEIGHBOR_SCAN_CAP = 64    # max middle-nodes scanned per edge
COHERENCE_HUB_DEGREE = 200          # a path through a node above this
                                    # out-degree is weak corroboration

# Roadmap Step 4 (2026-05-22) — abstraction / concept formation.
# When many concepts X1,X2,...,Xn share an edge (Xi, R, T), the
# regularity is real: those Xi's belong to a class.  Form a
# synthetic concept _abstract_R_T and add is_a edges from each Xi
# to it.  The synthetic concept + edges enter at provisional
# strength — they earn permanence by being USED in cortical
# inference (R.1's is_a composition routes through them) or
# dissolve.  Run during sleep consolidation alongside coherence
# + prune.
ABSTRACTION_MIN_GROUP = 3           # >= N members to count as a regularity
ABSTRACTION_NAME_PREFIX = '_abstract_'
# Reserved namespace for firsthand WORLD tokens (grounding, Cap-1).
#
# STALE-COMMENT FIX (2026-07-19): this used to claim world tokens are
# SKIPPED as grouping sources in form_abstractions / form_analogies to
# close a "Cap-2 abstraction side-door" to record_learning / lifeforce.
# That WALL CAME DOWN on 2026-06-30 — world tokens CAN be abstracted and
# CAN form analogies.  See the wall-down comment in form_abstractions
# (~line 2410) and its twin in form_analogies: formation-vs-use is
# resolved, credit follows USE (newly_coherent) and never FORMATION, so
# forming a world abstraction credits nothing and earn-or-dissolve is
# the sole guard.  The prefix survives as a NAMESPACE marker only; it
# gates nothing here.
WORLD_TOKEN_PREFIX = '_world_s' 
ABSTRACTION_MAX_NEW_PER_PASS = 50   # cap creation per sleep pass
# EDGE budget per pass (2026-08-23).  The cap above is on CLASSES; each
# class attaches one is_a edge per MEMBER, uncapped -- measured 194
# classes writing ~61,000 edges.  DERIVED, not stamped: at most the
# minimum viable membership per permitted class.  Remaining members
# attach on later passes (the edge key dedups, so nothing is lost).
ABSTRACTION_MAX_EDGES_PER_PASS = ABSTRACTION_MAX_NEW_PER_PASS * 3


def _ISABUDGET_ON():
    """File gate /root/ISABUDGET_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/ISABUDGET_ON")
    except Exception:
        return False
ABSTRACTION_RELATION = 'is_a'       # how members link to the class
# Structural-role tag relation (SHADOW instrument, 2026-07-11).  The
# RoleRegularityShadow writes `has_role` edges from a world-state to a
# synthetic `_role_*` class node.  Deliberately NOT in RELATION_COMPOSITION
# (never coheres/credits) and skipped by form_abstractions so it can never
# become a traversable is_a — a read-only instrument that DRIVES NOTHING.
ROLE_RELATION = 'has_role'

# Similarity-docking facet relation (facet-docking build, 2026-07-21).  The
# WorldActor's attend-and-tag walk writes `has_facet` edges from a lived
# world-state token to a synthetic `_facet_{slot}_{value}` node — the
# percept's OWN (slot, value) componentry, the many docking points that give
# a percept the same many-edged shape a concept already has (the user's
# lock-and-key).  Deliberately NOT in RELATION_COMPOSITION (never coheres,
# never credits lifeforce → farm closed) and — like has_role — skipped as a
# grouping source in form_abstractions / form_analogies, so NO is_a edge
# with a `_facet_*` endpoint is ever minted (a facet CLASS would be a 1:1
# extensional duplicate of the facet node; ignition's carrier lookup runs
# O(1) over the relation index instead).  The facet node itself is the
# self-formed abstraction; `has_facet` is a CONTAINER relation, not a
# category.  Strength law: applied UN-engaged on observation, duplicates
# skipped (writer OBSERVE_REASONS), earns/loses ONLY through split-event
# contrastive dock earning (facet_confirm / facet_disconfirm).
FACET_RELATION = 'has_facet'
FACET_NODE_PREFIX = '_facet_'

# Analogy formation (2026-05-28) — structural cross-domain mapping
# (V1 analogy_engine ported doctrine-clean; V1's hardcoded domain
# keyword-lexicon DROPPED as a closed-enumeration anti-pattern).
# Two concepts that share an outgoing relation-SKELETON (the set of
# relation TYPES on their edges) but map to DISJOINT targets are
# structural analogs: same relational shape, different content
# (atom {has_part, attracts} ~ solar_system {has_part, attracts},
# with entirely different targets).  Detected during sleep
# alongside form_abstractions; writes provisional `analogous_to`
# edges that earn-or-dissolve via Phase S — an analogy walked in
# inference survives, an idle one fades.  Complementary axis to
# abstraction: abstraction groups by shared (relation, target) →
# class; analogy groups by shared relation-SET with disjoint
# targets → structural likeness.  The disjoint-targets rule is a
# threshold-free structural criterion (no hand-tuned similarity
# cutoff): shared structure + zero shared content = the cross-
# domain analogy V1 valued ("the further apart the domains, the
# more powerful").
ANALOGY_RELATION = 'analogous_to'
ANALOGY_MIN_SKELETON = 2            # >= N shared relation types = "structure"
ANALOGY_MAX_NEW_PER_PASS = ABSTRACTION_MAX_NEW_PER_PASS  # 50/pass cap
ANALOGY_BUCKET_MEMBER_CAP = COHERENCE_NEIGHBOR_SCAN_CAP  # 64 — bound pairing


# ---------------------------------------------------------------------
# KnowledgeItem — text-source claims attached to a concept
# ---------------------------------------------------------------------

@dataclass
class KnowledgeItem:
    """A text-source claim attached to a concept. Mirrors current
    SEAGI's bubble.knowledge entries but typed.

    Per the doctrine ("substrate stores no M/I labels"), this class
    no longer carries an `mi` field. The felt M/I of a knowledge
    item is whatever the bubble traces of its associated concept
    say at retrieval time. Old saved brains containing `mi` in
    KnowledgeItem dicts are silently ignored on load.
    """
    text: str
    source: str = 'unknown'
    confidence: float = 0.5
    cycle_added: int = 0

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'source': self.source,
            'confidence': self.confidence,
            'cycle_added': self.cycle_added,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'KnowledgeItem':
        return cls(
            text=d.get('text', ''),
            source=d.get('source', 'unknown'),
            confidence=float(d.get('confidence', 0.5)),
            cycle_added=int(d.get('cycle_added', 0)),
        )


# ---------------------------------------------------------------------
# EdgeEvidence — M/I-split hit/miss tracking per edge
# ---------------------------------------------------------------------

@dataclass
class EdgeEvidence:
    """Per-edge prediction-outcome ledger. Mirrors current
    cognitive_stages/prediction.py format with M/I-split counters.

    `hits_m / misses_m` accumulate when the edge fires under
    mortality-tagged context; `hits_i / misses_i` under immortality-
    tagged. An edge can be reliable in one polarity and unreliable
    in another — this data captures that asymmetry.
    """
    hits: int = 0
    misses: int = 0
    inconclusive: int = 0
    hits_m: float = 0.0
    hits_i: float = 0.0
    misses_m: float = 0.0
    misses_i: float = 0.0
    first_seen_cycle: int = 0
    last_check_cycle: int = 0

    def record(self, outcome: Literal['hit', 'miss', 'inconclusive'],
               m_weight: float, i_weight: float, cycle: int) -> None:
        m_w = max(0.0, min(1.0, float(m_weight)))
        i_w = max(0.0, min(1.0, float(i_weight)))
        self.last_check_cycle = cycle
        if outcome == 'hit':
            self.hits += 1
            self.hits_m += m_w
            self.hits_i += i_w
        elif outcome == 'miss':
            self.misses += 1
            self.misses_m += m_w
            self.misses_i += i_w
        else:
            self.inconclusive += 1

    def reliability(self,
                    m_context: float = 0.0,
                    i_context: float = 0.0,
                    mi_dom_threshold: float = 0.3,
                    ) -> float:
        """Reliability in [0, 1] under current M/I framing. No evidence
        → 0.5 (neutral). Polarity-matched evidence overrides base when
        context has a clear lean."""
        total = self.hits + self.misses
        if total == 0:
            return 0.5
        base = self.hits / total

        m_dom = m_context > mi_dom_threshold and m_context > i_context
        i_dom = i_context > mi_dom_threshold and i_context > m_context
        if m_dom:
            denom = self.hits_m + self.misses_m
            if denom > 0.05:
                return 0.7 * (self.hits_m / denom) + 0.3 * base
        if i_dom:
            denom = self.hits_i + self.misses_i
            if denom > 0.05:
                return 0.7 * (self.hits_i / denom) + 0.3 * base
        return base

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'EdgeEvidence':
        return cls(**{k: d.get(k, 0) for k in (
            'hits', 'misses', 'inconclusive', 'first_seen_cycle',
            'last_check_cycle')}, **{k: float(d.get(k, 0.0)) for k in (
            'hits_m', 'hits_i', 'misses_m', 'misses_i')})


# ---------------------------------------------------------------------
# Relation — typed relation with M/I prior
# ---------------------------------------------------------------------

@dataclass
class Relation:
    """A relation type. Edges are instances of relations between
    concepts.

    Per the doctrine ("substrate stores no M/I labels; no innate
    concept tags"), relations no longer carry an `mi_prior`. The
    felt M/I of an edge is derived from the chemistry trace on
    that edge's bubbles, which accumulate from contextual encounter
    — not from a seed value on the relation type.

    `extracted_from` records how many evidence examples seeded this
    relation type — for scaffold-extension audit.

    Old saved brains containing `mi_prior` in Relation dicts are
    silently ignored on load.
    """
    name: str
    extracted_from: int = 0

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'extracted_from': self.extracted_from,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Relation':
        return cls(
            name=d.get('name', ''),
            extracted_from=int(d.get('extracted_from', 0)),
        )


# Standard relation type names. These are bootstrap names that
# pre-register common relation types in the registry — types
# Seagi will encounter often. They carry NO innate M/I per the
# doctrine. M/I emerges from contextual chemistry imprints on
# the bubbles that form on edges of each type. A war passage's
# "kills" edge develops a different cocktail than a courtroom
# narrative's "kills" edge.
INITIAL_RELATION_NAMES: Tuple[str, ...] = (
    # Identity / categorization
    'is_a', 'has_property', 'similar', 'opposite', 'part_of', 'has_part',
    # Causal
    'causes', 'caused_by',
    # Function / capability
    'used_for', 'can_do', 'enables',
    # Temporal / change
    'precedes', 'follows',
    # Threat / harm (no innate M tag — context tags via chemistry)
    'threatens', 'destroys', 'kills', 'damages',
    # Growth / creation (no innate I tag — context tags via chemistry)
    'creates', 'grows', 'preserves', 'extends',
    # Cognitive
    'knows', 'understands', 'remembers',
    # Lacks
    'lacks',
    # Phase F.6 (2026-05-16): inverses needed for contradiction
    # detection.  Each pair (R, R_inv) flags a conflict: a peer
    # asserting (X R Y) while substrate holds (X R_inv Y) is a
    # candidate contradiction.  Inverse pairs are SYMMETRIC —
    # `causes` and `prevents` are inverses regardless of which
    # is asserted first.
    'prevents', 'protects', 'helps', 'hurts',
    'increases', 'decreases', 'disables',
)


# Phase F.6 — relation inverse map.  When the peer asserts
# (S, R, O), F.6 looks for any (S, R_inv, O) in substrate where
# R_inv ∈ INVERSE_RELATIONS.get(R, []).  Symmetric: an entry
# `('causes', 'prevents')` implies `('prevents', 'causes')` is
# also a conflict.
INVERSE_RELATIONS: Dict[str, Tuple[str, ...]] = {
    'causes':    ('prevents',),
    'prevents':  ('causes',),
    'creates':   ('destroys', 'kills'),
    'destroys':  ('creates', 'grows', 'preserves'),
    'kills':     ('creates', 'preserves', 'grows'),
    'grows':     ('destroys', 'kills'),
    'preserves': ('destroys', 'kills'),
    'threatens': ('protects',),
    'protects':  ('threatens', 'damages'),
    'damages':   ('protects', 'preserves'),
    'helps':     ('hurts', 'damages'),
    'hurts':     ('helps', 'preserves'),
    'enables':   ('disables', 'prevents'),
    'disables':  ('enables',),
    'increases': ('decreases',),
    'decreases': ('increases',),
    'lacks':     ('has_property',),
}


# ---------------------------------------------------------------------
# Edge — typed concept-to-concept link with M/I and evidence
# ---------------------------------------------------------------------

class Edge:
    """A typed link between two concepts.

    Per the M/I architecture:
        - `mi` is a DERIVED property from the edge's transmitter
          trace (the chemical fingerprint accumulated from
          encounters with this relational fact). With no trace
          yet, falls back to `_legacy_mi` — a value the constructor
          or older code wrote, typically computed at edge creation
          from endpoints + relation prior.
        - `transmitter_trace` is the substrate-side data: 8-channel
          chemistry accumulated when this edge fires (backward
          reinforcement, inference firing, tick-end sweep). Per
          the vision: relational world facts get tagged at
          perception, not at creation.
        - `evidence` accumulates M/I-split hit/miss counters.
    """

    __slots__ = ('source', 'target', 'relation_name', 'strength',
                 '_legacy_mi', 'transmitter_trace', 'bubbles',
                 'evidence', 'last_reinforced_cycle',
                 'first_coherent_cycle', 'last_engaged_cycle')

    def __init__(self,
                 source: str,
                 target: str,
                 relation_name: str,
                 strength: float = 0.5,
                 mi: Optional[MIValue] = None,
                 evidence: Optional[EdgeEvidence] = None,
                 last_reinforced_cycle: int = 0,
                 transmitter_trace: Optional[
                     'TransmitterState'] = None,
                 bubbles: Optional[List['Bubble']] = None,
                 first_coherent_cycle: int = 0,
                 last_engaged_cycle: int = 0):
        self.source = source
        self.target = target
        self.relation_name = relation_name
        self.strength = max(0.0, min(1.0, float(strength)))
        self._legacy_mi = mi if mi is not None else MIValue.zero()
        # Edge bubble list — context-keyed traces (parallel to
        # concepts but with a smaller cap). Each bubble carries
        # the chemistry of one context in which this edge fired.
        # Same edge in fearful vs comfortable context lands in
        # different bubbles, has different felt-character.
        self.bubbles: List['Bubble'] = list(bubbles) if bubbles else []
        # Legacy transmitter_trace field — kept as a mirror of the
        # most-recently-active bubble's trace for backward compat
        # with code that reads edge.transmitter_trace directly.
        # When constructor receives a transmitter_trace but no
        # bubbles, seed an initial generic bubble carrying that
        # trace so the data isn't lost.
        self.transmitter_trace = transmitter_trace
        if transmitter_trace is not None and not self.bubbles:
            self.bubbles.append(Bubble(
                transmitter_trace=transmitter_trace,
                context_key=ContextKey(),
                encounter_count=1,
                created_cycle=int(last_reinforced_cycle),
                last_active_cycle=int(last_reinforced_cycle),
            ))
        self.evidence = evidence if evidence is not None else EdgeEvidence()
        self.last_reinforced_cycle = int(last_reinforced_cycle)
        # Cycle this edge FIRST cohered (corroborated by an A→X→B
        # path).  0 = never yet coherent.  Set once and persisted —
        # the earn-gated growth signal for MortalityDrive keys on
        # this first-ever transition, NOT on re-reinforcement of
        # already-coherent edges (which is whole-substrate work, not
        # learning) and NOT on a decayed edge re-cohering (which
        # chemistry-never-dissolves makes recurrent — that would
        # re-open the anti-gaming hole).
        self.first_coherent_cycle = int(first_coherent_cycle)
        # Cycle this edge was last ENGAGED by one of the four
        # cognitive bid-paths: inference traversal (it became part
        # of an emitted reasoning chain), composable coherence
        # corroboration, chemistry imprint via attended bubble fire,
        # or re-attestation by intake.  Distinct from
        # `last_reinforced_cycle`, which `settle_weak_edges` re-
        # stamps every sleep and is therefore useless as an
        # engagement signal.  Substrate.quarantine_inert_edges keys
        # on this: edges that none of the bid-paths touched across
        # L_QUARANTINE cycles + are at strength-floor get moved to
        # the quiescent quarantine pool (retrievable, not deleted).
        # `0` is the default sentinel; substrate.quarantine_inert_
        # edges treats freshly-loaded (None) edges with a grace
        # window so legacy canonicals aren't mass-quarantined on
        # first sleep post-deploy.
        self.last_engaged_cycle = int(last_engaged_cycle)

    @property
    def mi(self) -> MIValue:
        """DERIVED: M/I projection from the most-recently-active
        bubble's transmitter trace.

        With no bubbles (newly-created edge before any encounter),
        falls back to _legacy_mi (the constructor value, or
        endpoint-derived value from Edge.initialize). Once the
        engine imprints chemistry onto the edge — spawning a
        bubble — derived takes over.

        Per the vision: relational world facts have NO innate M/I.
        Their tag is what they've been associated with, chemically,
        through use, in the contexts they fired in.
        """
        if self.bubbles:
            active = max(self.bubbles,
                          key=lambda b: b.last_active_cycle)
            from seagi.core.layer6 import derive_mi_from_trace, BASELINES
            # Check if trace has meaningful departure from baseline.
            # If trace is at baseline, fall back to legacy.
            for tr, baseline in BASELINES.items():
                if abs(getattr(active.transmitter_trace, tr,
                                  baseline) - baseline) > 1e-6:
                    return derive_mi_from_trace(
                        active.transmitter_trace)
        return self._legacy_mi

    @mi.setter
    def mi(self, value: MIValue) -> None:
        """Backward-compat setter: writes to _legacy_mi (the
        fallback used when no trace exists)."""
        self._legacy_mi = value

    def effective_mi(self, cycle: int) -> MIValue:
        """SHADOW live tag (value-web): M/I derived from the active
        bubble's DECAYED effective_trace(cycle) — "what the tag is worth
        NOW" (retires from disuse) — vs the frozen `.mi` getter which
        reads the raw trace.  Does NOT change live behaviour."""
        if self.bubbles:
            active = max(self.bubbles,
                          key=lambda b: b.last_active_cycle)
            from seagi.core.layer6 import derive_mi_from_trace
            return derive_mi_from_trace(active.effective_trace(cycle))
        return self._legacy_mi

    @property
    def key(self) -> EdgeKey:
        return (self.source, self.relation_name, self.target)

    @classmethod
    def initialize(cls,
                   source_concept: 'Concept',
                   target_concept: 'Concept',
                   relation: Relation,
                   strength: float = 0.5,
                   cycle: int = 0,
                   ) -> 'Edge':
        """Create a new edge.

        Per the doctrine, new edges carry no innate M/I — neither
        from the relation type (no `mi_prior`) nor from the
        endpoints (which themselves have no stored M/I; their
        `.mi` is derived from bubble traces). Edge M/I emerges
        from chemistry imprints on the edge's bubbles as it fires
        in context.
        """
        evidence = EdgeEvidence(first_seen_cycle=cycle,
                                last_check_cycle=cycle)
        return cls(source=source_concept.name,
                   target=target_concept.name,
                   relation_name=relation.name,
                   strength=strength,
                   mi=None,
                   evidence=evidence,
                   last_reinforced_cycle=cycle)

    def reinforce(self,
                     cycle: int,
                     delta: float = EDGE_STRENGTH_BUMP_PER_USE,
                     *,
                     origin: str = ''
                     ) -> None:
        """Strengthen this edge by `delta` (clamped).

        Phase C: the default `delta` is now promille
        (EDGE_STRENGTH_BUMP_PER_USE = 0.005) per the doctrine that
        all tagging contributions are at promille scale.  The
        reinforce call also applies any pending decay (lazy) so
        the stored value reflects the post-bump effective strength
        and `last_reinforced_cycle` anchors future decay.

        Callers can still pass a custom `delta` for back-compat,
        but promille is correct.

        `origin` (keyword-only, mortality-clock promotion 2026-07-18):
        the honest label of the calling path — 'cognition'
        (inference / abstraction / analogy / writer re-attestation /
        chemistry imprint / predictive confirmation), 'maintenance'
        (dirty sweep / replay), or '' (unlabeled → treated as
        'unknown').  Origin is LOGGED/split by the observer only; it
        NEVER gates crediting.

        After the strength bump, a registered module-level observer
        (the MortalityClock) is invoked:
            on_reinforce(edge, cycle, pre_strength, origin)
        Observer failures NEVER perturb the reinforce, but they are
        counted VISIBLY (reinforce_observer_errors, surfaced in
        /status as observer_errors) — never silently swallowed.
        """
        # Apply lazy decay before bumping.
        pre_strength = self.effective_strength(cycle)
        self.strength = pre_strength
        self.strength = min(1.0, self.strength + float(delta))
        self.last_reinforced_cycle = int(cycle)
        # Notify the registered reinforce observer (mortality clock).
        obs = _REINFORCE_OBSERVER
        if obs is not None:
            try:
                obs(self, int(cycle), float(pre_strength),
                    origin or 'unknown')
            except Exception:
                global _REINFORCE_OBSERVER_ERRORS
                _REINFORCE_OBSERVER_ERRORS += 1

    def effective_strength(self, cycle: int) -> float:
        """Strength with lazy decay applied.  Phase C: edges that
        have not been walked recently fade toward zero at promille
        per cycle.  Frequently-walked edges keep their value.
        """
        elapsed = int(cycle) - int(self.last_reinforced_cycle)
        if elapsed <= 0:
            return self.strength
        decayed = self.strength - EDGE_STRENGTH_DECAY_PER_CYCLE * elapsed
        if decayed < 0.0:
            return 0.0
        return decayed

    def weaken(self, delta: float = 0.05) -> None:
        self.strength = max(0.0, self.strength - delta)

    def __repr__(self) -> str:
        return (f"Edge({self.source} -[{self.relation_name}]-> "
                f"{self.target}, s={self.strength:.2f}, "
                f"mi={self.mi})")

    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'target': self.target,
            'relation_name': self.relation_name,
            'strength': self.strength,
            'mi': self._legacy_mi.to_dict(),
            'transmitter_trace': (
                self.transmitter_trace.to_dict()
                if self.transmitter_trace is not None else None),
            'bubbles': [b.to_dict() for b in self.bubbles],
            'evidence': self.evidence.to_dict(),
            'last_reinforced_cycle': self.last_reinforced_cycle,
            'first_coherent_cycle': self.first_coherent_cycle,
            'last_engaged_cycle': self.last_engaged_cycle,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Edge':
        tr_d = d.get('transmitter_trace')
        bubbles_d = d.get('bubbles', [])
        return cls(
            source=d.get('source', ''),
            target=d.get('target', ''),
            relation_name=d.get('relation_name', ''),
            strength=float(d.get('strength', 0.5)),
            mi=MIValue.from_dict(d.get('mi', {})),
            evidence=EdgeEvidence.from_dict(d.get('evidence', {})),
            last_reinforced_cycle=int(d.get('last_reinforced_cycle', 0)),
            transmitter_trace=(TransmitterState.from_dict(tr_d)
                                if tr_d else None),
            bubbles=[Bubble.from_dict(b) for b in bubbles_d],
            first_coherent_cycle=int(d.get('first_coherent_cycle', 0)),
            last_engaged_cycle=int(d.get('last_engaged_cycle', 0)),
        )


# ---------------------------------------------------------------------
# Bubbles — context-keyed transmitter traces
#
# A concept doesn't have a single chemical fingerprint; it has many,
# one per context. The bubble is the unit of context-distinguishable
# tagging: same word "fire" tagged differently in fireplace-context
# (warmth, dopamine-flavored) vs house-fire-context (cortisol-
# flavored). Active bubble at perception is the one whose context-
# key best matches now.
#
# Per the M/I vision:
#   - Bubbles are substrate-side data (each concept owns its bubble
#     list). Engine computes the current ContextKey from its state
#     (transmitters + co-active L1 concepts) at imprint/perceive time.
#   - The trace inside a bubble is fine-grained, multi-channel,
#     accumulated through repeated encounters in matching contexts.
#   - A new encounter either lands in an existing bubble (above
#     match-threshold) or spawns a new bubble.
# ---------------------------------------------------------------------

# How many decimal-bins per transmitter for chemistry-signature
# matching. 10 means each channel has 10 buckets (0.0, 0.1, 0.2,...
# rounded). Two encounters whose transmitter states round to same
# tuple have identical chemistry_signature.
CHEMISTRY_BUCKETS = 10

# Max co-active concepts retained in a bubble's context_key. Cap
# keeps similarity comparisons cheap.
COACTIVE_CONTEXT_CAP = 5

# Bubble-match threshold. Above this similarity, an encounter joins
# an existing bubble; below, a new bubble spawns.
BUBBLE_MATCH_THRESHOLD = 0.5

# (audit deletion-candidate, 2026-06-04) MAX_BUBBLES_PER_CONCEPT
# removed — declared with an "LRU evicted" comment but referenced
# nowhere (find_or_spawn_bubble never enforced it).  Bubble growth is
# now bounded by the sleep-gated prune_dormant_bubbles wire (audit #13).

# Cap on bubbles per edge. Smaller than concept cap because edges
# are typically 5-10x more numerous; we trade slightly less context
# discrimination for memory.
MAX_BUBBLES_PER_EDGE = 5


@dataclass
class ContextKey:
    """Fingerprint of the context in which a bubble formed.

    chemistry_signature: 8-tuple of bucketed transmitter values
        (each rounded to nearest 1/CHEMISTRY_BUCKETS). Two encounters
        with similar chemistry have identical signatures and match
        cleanly without floating-point noise.
    coactive_concepts: frozenset of the strongest co-active concept
        names at imprint time, capped at COACTIVE_CONTEXT_CAP.

    Comparison via context_similarity(); not a hash.
    """
    chemistry_signature: Tuple[int, ...] = field(default_factory=tuple)
    coactive_concepts: frozenset = field(default_factory=frozenset)

    def to_dict(self) -> dict:
        return {
            'chemistry_signature': list(self.chemistry_signature),
            'coactive_concepts': sorted(self.coactive_concepts),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ContextKey':
        sig = tuple(int(x) for x in d.get('chemistry_signature', []))
        coactive = frozenset(d.get('coactive_concepts', []))
        return cls(chemistry_signature=sig, coactive_concepts=coactive)


def context_similarity(a: ContextKey, b: ContextKey) -> float:
    """Similarity in [0, 1].

    Chemistry similarity: decreases sharply with max single-channel
    bucket difference. A 0.5 cortisol shift on one channel is
    enough to mark contexts as distinct, even if other channels
    match — chemical context is fingerprint-sensitive, not average-
    similar.

    Coactive similarity: Jaccard (|A∩B|/|A∪B|).

    If both coactive sets are empty (no L1 context — e.g., very
    early in life or in test setups), fall back to chemistry alone
    rather than counting empty-set match as evidence.
    """
    n = min(len(a.chemistry_signature), len(b.chemistry_signature))
    if n > 0:
        max_diff = max(abs(x - y) for x, y in zip(
            a.chemistry_signature, b.chemistry_signature))
        # 3-bucket gap on any single channel → 0 similarity.
        # (3 buckets = 0.3 transmitter departure — that's a real
        # context shift.)
        chem_sim = max(0.0, 1.0 - max_diff / 3.0)
    else:
        chem_sim = 0.0
    if not a.coactive_concepts and not b.coactive_concepts:
        # No context information beyond chemistry; trust chemistry.
        return chem_sim
    union = a.coactive_concepts | b.coactive_concepts
    if not union:
        coactive_sim = 1.0
    else:
        inter = a.coactive_concepts & b.coactive_concepts
        coactive_sim = len(inter) / len(union)
    return 0.5 * chem_sim + 0.5 * coactive_sim


# Unified Bubble — single source of truth.  Replaced the local
# v1 class definition with import from .bubble during
# the v1+v2 unification on 2026-05-14.  See
# [[project_seagi_v1_v2_unification]] for the migration record.
# The unified Bubble is:
#   - the SAME class used by v2 AWM/chemistry (no EnrichedBubble)
#   - backward-compat with v1 serialization (TransmitterState
#     adapter on save/load)
#   - constructor accepts either a Dict[str, float] OR a
#     TransmitterState for transmitter_trace (auto-converts)
from .bubble import Bubble  # noqa: E402


# ---------------------------------------------------------------------
# Concept — node with M/I, transmitters, knowledge, and embeddings
# ---------------------------------------------------------------------

class Concept:
    """A substrate concept. Mutable.

    Per the M/I architecture:
        name                — string identifier (substrate-unique)
        mi                  — DERIVED M/I projection from active
                              bubble's trace; under the bubble system
                              this is computed read-only at perception
                              time. Constructor mi-arg is translated
                              into an initial bubble imprint.
        bubbles             — context-keyed transmitter traces. Each
                              bubble is one chemical fingerprint per
                              context. Active bubble at perception is
                              the one whose context-key best matches
                              now. Empty list = never imprinted.
        transmitters        — derived view of the active bubble's
                              trace, kept as a settable field for
                              backward compat. Engine refreshes on
                              imprint via _set_active_bubble.
        activation_count    — total times this concept has been activated
        last_activated_cycle
        knowledge           — list of typed KnowledgeItem
        embedding           — MIEmbedding (computed lazily)
        edges_out           — outgoing edges, indexed by relation name
    """

    __slots__ = ('name', '_legacy_mi', 'bubbles', 'transmitters',
                 'activation_count', 'last_activated_cycle',
                 'knowledge', '_embedding',
                 'edges_out', 'created_cycle', 'synthetic',
                 # Phase C 2026-05-15: frequency salience.
                 # Per-concept "brightness" — bumped on each
                 # activation, decays slowly over disuse.  Read
                 # via effective_salience(cycle).
                 'salience', 'salience_last_update_cycle')

    def __init__(self,
                 name: str,
                 mi: Optional[MIValue] = None,
                 transmitters: Optional[TransmitterState] = None,
                 activation_count: int = 0,
                 last_activated_cycle: int = 0,
                 knowledge: Optional[List[KnowledgeItem]] = None,
                 embedding: Optional[MIEmbedding] = None,
                 created_cycle: int = 0,
                 synthetic: bool = False,
                 bubbles: Optional[List[Bubble]] = None):
        self.name = name
        # Legacy MI: stored as fallback when no bubbles exist (e.g.,
        # construction-time, pre-perception). Once a bubble forms,
        # the @property derives MI from the active bubble's trace
        # and _legacy_mi becomes inactive. Per the vision:
        # substrate stores no M/I — but for backward compat with
        # code that constructs Concept(mi=...) without going through
        # perception, the legacy field accepts the input.
        self._legacy_mi = mi if mi is not None else MIValue.zero()
        # Bubble list — context-keyed traces. Empty list at birth;
        # bubbles spawn through perception/imprint.
        self.bubbles: List[Bubble] = list(bubbles) if bubbles else []
        # Legacy `transmitters` field kept as a snapshot of the
        # most-recently-active bubble's trace. Engine refreshes via
        # imprint_on_perceive on each encounter. Backward-compat:
        # if caller passes transmitters but no bubbles, seed an
        # initial generic bubble carrying that trace so the data
        # isn't lost in the transition to the bubble model.
        self.transmitters = transmitters
        if transmitters is not None and not self.bubbles:
            self.bubbles.append(Bubble(
                transmitter_trace=transmitters,
                context_key=ContextKey(),
                encounter_count=1,
                created_cycle=int(created_cycle),
                last_active_cycle=int(created_cycle),
            ))
        self.activation_count = int(activation_count)
        self.last_activated_cycle = int(last_activated_cycle)
        self.knowledge = knowledge or []
        self._embedding = embedding
        self.edges_out: Dict[str, List[Edge]] = {}
        self.created_cycle = int(created_cycle)
        # Phase C: frequency-driven salience.  Starts at zero;
        # bumps on activation; decays slowly.  Read via
        # effective_salience(cycle).  Brightness of this concept
        # for attention/connection-formation purposes.
        self.salience: float = 0.0
        self.salience_last_update_cycle: int = int(created_cycle)
        # synthetic = formed by a generative pass (concept_formation,
        # consolidation cluster, transfer-mint). Filter sites that
        # care about "real vs derived" check this flag instead of
        # string-prefix matching on names. Persisted via to_dict.
        self.synthetic = bool(synthetic)

    @property
    def embedding(self) -> Optional[MIEmbedding]:
        return self._embedding

    @embedding.setter
    def embedding(self, value: Optional[MIEmbedding]) -> None:
        self._embedding = value

    @property
    def mi(self) -> MIValue:
        """DERIVED: M/I projection of active bubble's transmitter
        trace. Per the vision, substrate stores no static M/I —
        every M/I read is computed at access time from the felt
        chemistry currently most relevant to this concept (the
        active bubble's trace).

        With no bubbles yet, falls back to _legacy_mi (the value
        the constructor or older code wrote directly). Once a
        bubble forms through perception, the bubble's derived MI
        takes precedence.
        """
        if self.bubbles:
            # Active bubble = most-recently-active.
            active = max(self.bubbles,
                          key=lambda b: b.last_active_cycle)
            from seagi.core.layer6 import derive_mi_from_trace
            return derive_mi_from_trace(active.transmitter_trace)
        return self._legacy_mi

    @mi.setter
    def mi(self, value: MIValue) -> None:
        """Backward-compat setter: writes to _legacy_mi (the
        fallback used when no bubbles exist). Once bubbles form,
        the legacy field becomes inactive — derived MI takes over.

        Old code calling concept.mi = value still works for
        pre-perception state (e.g., test setup); active concepts
        with bubbles ignore the legacy write. Per the vision,
        this is correct: M/I is felt, not assigned.
        """
        self._legacy_mi = value

    def activate(self, cycle: int) -> None:
        """Record an activation event."""
        self.activation_count += 1
        self.last_activated_cycle = cycle

    def update_mi(self, new_mi: MIValue) -> None:
        """Legacy setter — alias for mi.setter. Old hierarchy code
        calls this from backward-pass error propagation; under the
        bubble model the call writes to _legacy_mi and is overridden
        once a bubble forms."""
        self.mi = new_mi

    def add_knowledge(self, item: KnowledgeItem) -> None:
        self.knowledge.append(item)

    def neighbors(self,
                  relation_filter: Optional[Set[str]] = None
                  ) -> List[Tuple[str, Edge]]:
        """Return [(target_name, Edge), ...] from this concept's
        outgoing edges. If relation_filter set, only return edges
        whose relation_name is in the filter."""
        out: List[Tuple[str, Edge]] = []
        for rel_name, edges in self.edges_out.items():
            if relation_filter is not None and rel_name not in relation_filter:
                continue
            for edge in edges:
                out.append((edge.target, edge))
        return out

    def __repr__(self) -> str:
        return f"Concept({self.name!r}, mi={self.mi}, n_edges={sum(len(es) for es in self.edges_out.values())})"

    # ---- Phase C: frequency salience ----

    def bump_salience(self,
                          amount: float,
                          cycle: int) -> None:
        """Increase this concept's salience by `amount` (clamped to
        [0, 1]).  Records the cycle so subsequent reads can decay.

        Bumped on every activation (AWM promotion) and every time
        cortical commits a thought referencing this concept as
        focal or target.  Bumps are at promille scale.
        """
        # Apply any pending decay before adding (so the stored
        # value reflects "current" effective level + bump).
        self.salience = self.effective_salience(cycle)
        self.salience = min(1.0, max(0.0,
            self.salience + float(amount)))
        self.salience_last_update_cycle = int(cycle)

    def effective_salience(self, cycle: int) -> float:
        """Salience with lazy decay applied.  Reads compute the
        current effective level from the stored value and the
        cycles elapsed since last update.  No background scan
        needed; scales to millions of concepts.

        Frequently-activated concepts (often-bumped) maintain high
        salience.  Rarely-activated concepts fade toward zero over
        many cycles.  Decay rate is promille per cycle.
        """
        elapsed = int(cycle) - int(self.salience_last_update_cycle)
        if elapsed <= 0:
            return self.salience
        decayed = self.salience - SALIENCE_DECAY_PER_CYCLE * elapsed
        if decayed < 0.0:
            return 0.0
        return decayed

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            # Persist the legacy fallback MI explicitly. The
            # @property derived value is computed live, not stored.
            # On reload, _legacy_mi is the fallback; bubbles drive
            # derived MI when they exist.
            'mi': self._legacy_mi.to_dict(),
            'transmitters': (self.transmitters.to_dict()
                              if self.transmitters else None),
            'bubbles': [b.to_dict() for b in self.bubbles],
            'activation_count': self.activation_count,
            'last_activated_cycle': self.last_activated_cycle,
            'knowledge': [k.to_dict() for k in self.knowledge],
            'embedding': (self._embedding.to_dict()
                          if self._embedding else None),
            'created_cycle': self.created_cycle,
            'synthetic': self.synthetic,
            # Phase C: frequency salience.
            'salience': float(self.salience),
            'salience_last_update_cycle': int(
                self.salience_last_update_cycle),
            # edges_out NOT serialized here — edges live in Substrate
            # and are reconnected by from_dict on Substrate side.
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Concept':
        emb_d = d.get('embedding')
        tr_d = d.get('transmitters')
        bubbles_d = d.get('bubbles', [])
        c = cls(
            name=d.get('name', ''),
            mi=MIValue.from_dict(d.get('mi', {})),
            transmitters=(TransmitterState.from_dict(tr_d)
                          if tr_d else None),
            activation_count=int(d.get('activation_count', 0)),
            last_activated_cycle=int(d.get('last_activated_cycle', 0)),
            knowledge=[KnowledgeItem.from_dict(k)
                       for k in d.get('knowledge', [])],
            embedding=(MIEmbedding.from_dict(emb_d) if emb_d else None),
            created_cycle=int(d.get('created_cycle', 0)),
            synthetic=bool(d.get('synthetic', False)),
            bubbles=[Bubble.from_dict(b) for b in bubbles_d],
        )
        # Phase C — restore salience state.  Old brains (no field)
        # default to zero.
        c.salience = float(d.get('salience', 0.0))
        c.salience_last_update_cycle = int(
            d.get('salience_last_update_cycle',
                   c.created_cycle))
        return c


# ---------------------------------------------------------------------
# Episode — a discrete first-class event in SEAGI's history
# ---------------------------------------------------------------------

@dataclass
class Episode:
    """Per Layer 4 §8.1. Episodes are the units of replay.

    `kind` is one of: dialog_turn / goal_resolution / self_observation
                       / tool_call / death / revival / insight / pain
                       / confirmation / falsification / peer_message
    """
    id: str
    cycle: int
    kind: str
    concepts: Set[str] = field(default_factory=set)
    content: str = ''
    mi_at_event: MIValue = field(default_factory=MIValue.zero)
    transmitters_at_event: Optional[TransmitterState] = None
    action_taken: Optional[Dict[str, Any]] = None
    observed_outcome: Optional[Dict[str, Any]] = None
    prediction_error: Optional[MIValue] = None
    peer_id: Optional[str] = None
    replay_count: int = 0
    last_replay_cycle: int = 0
    replay_priority: float = 0.0
    # Consolidation count: incremented by consolidation_pass each
    # time this episode contributes evidence to a generalized edge
    # / pattern. After SEMANTIC_THRESHOLD reinforcements, the
    # episode is treated as semantic memory: it has been replayed +
    # generalized enough that its content represents a learned
    # pattern, not a single event.
    consolidation_count: int = 0

    @property
    def is_semantic(self) -> bool:
        """True when this episode has crossed the consolidation
        threshold and now functions as semantic memory. The split
        is graded — episodes can move from episodic→semantic but
        not back."""
        return self.consolidation_count >= 3

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'cycle': self.cycle,
            'kind': self.kind,
            'concepts': sorted(list(self.concepts)),
            'content': self.content,
            'mi_at_event': self.mi_at_event.to_dict(),
            'transmitters_at_event': (
                self.transmitters_at_event.to_dict()
                if self.transmitters_at_event else None),
            'action_taken': self.action_taken,
            'observed_outcome': self.observed_outcome,
            'prediction_error': (self.prediction_error.to_dict()
                                 if self.prediction_error else None),
            'peer_id': self.peer_id,
            'replay_count': self.replay_count,
            'last_replay_cycle': self.last_replay_cycle,
            'replay_priority': self.replay_priority,
            'consolidation_count': self.consolidation_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Episode':
        tr = d.get('transmitters_at_event')
        pe = d.get('prediction_error')
        return cls(
            id=d.get('id', ''),
            cycle=int(d.get('cycle', 0)),
            kind=d.get('kind', ''),
            concepts=set(d.get('concepts', [])),
            content=d.get('content', ''),
            mi_at_event=MIValue.from_dict(d.get('mi_at_event', {})),
            transmitters_at_event=(TransmitterState.from_dict(tr)
                                    if tr else None),
            action_taken=d.get('action_taken'),
            observed_outcome=d.get('observed_outcome'),
            prediction_error=(MIValue.from_dict(pe) if pe else None),
            peer_id=d.get('peer_id'),
            replay_count=int(d.get('replay_count', 0)),
            last_replay_cycle=int(d.get('last_replay_cycle', 0)),
            replay_priority=float(d.get('replay_priority', 0.0)),
            consolidation_count=int(
                d.get('consolidation_count', 0)),
        )


# ---------------------------------------------------------------------
# Prediction — a recorded forward-pass with M/I context
# ---------------------------------------------------------------------

@dataclass
class Prediction:
    """Per Layer 1 §4.2.4. Mostly mirrors current
    cognitive_stages/prediction.py format.
    """
    id: str
    seed_concept: str
    expected_edges: List[EdgeKey] = field(default_factory=list)
    source_edges: List[EdgeKey] = field(default_factory=list)
    mi_at_emission: MIValue = field(default_factory=MIValue.zero)
    cycle_made: int = 0
    target_cycle: int = 0
    state: Literal['pending', 'confirmed', 'falsified',
                    'inconclusive'] = 'pending'
    check_cycle: Optional[int] = None
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'seed_concept': self.seed_concept,
            'expected_edges': [list(e) for e in self.expected_edges],
            'source_edges': [list(e) for e in self.source_edges],
            'mi_at_emission': self.mi_at_emission.to_dict(),
            'cycle_made': self.cycle_made,
            'target_cycle': self.target_cycle,
            'state': self.state,
            'check_cycle': self.check_cycle,
            'result': self.result,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Prediction':
        return cls(
            id=d.get('id', ''),
            seed_concept=d.get('seed_concept', ''),
            expected_edges=[tuple(e) for e in d.get('expected_edges', [])],
            source_edges=[tuple(e) for e in d.get('source_edges', [])],
            mi_at_emission=MIValue.from_dict(
                d.get('mi_at_emission', {})),
            cycle_made=int(d.get('cycle_made', 0)),
            target_cycle=int(d.get('target_cycle', 0)),
            state=d.get('state', 'pending'),
            check_cycle=d.get('check_cycle'),
            result=d.get('result'),
        )


# ---------------------------------------------------------------------
# Scene — a set of co-active propositions forming a "current situation"
# ---------------------------------------------------------------------

class Scene:
    """L3 element: a set of co-active propositions that recurred
    enough to form a stable situation pattern.

    Per Layer 2 §5.1: scenes represent "sets of co-active propositions
    forming a 'current situation.' Working memory of what's happening."
    Identity is by proposition set: two scenes with the same
    propositions are the same scene.
    """

    __slots__ = ('id', 'propositions', 'mi', 'activation_count',
                 'last_activated_cycle', 'created_cycle')

    def __init__(self,
                 id: str,
                 propositions: Iterable[EdgeKey],
                 mi: Optional[MIValue] = None,
                 activation_count: int = 0,
                 last_activated_cycle: int = 0,
                 created_cycle: int = 0):
        self.id = id
        self.propositions: frozenset = frozenset(propositions)
        self.mi = mi if mi is not None else MIValue.zero()
        self.activation_count = int(activation_count)
        self.last_activated_cycle = int(last_activated_cycle)
        self.created_cycle = int(created_cycle)

    def activate(self, cycle: int) -> None:
        self.activation_count += 1
        self.last_activated_cycle = cycle

    def update_mi(self, new_mi: MIValue) -> None:
        self.mi = new_mi

    def __repr__(self) -> str:
        return (f"Scene({self.id!r}, n_propositions="
                f"{len(self.propositions)}, mi={self.mi})")

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'propositions': [list(p) for p in self.propositions],
            'mi': self.mi.to_dict(),
            'activation_count': self.activation_count,
            'last_activated_cycle': self.last_activated_cycle,
            'created_cycle': self.created_cycle,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Scene':
        return cls(
            id=d.get('id', ''),
            propositions=frozenset(
                tuple(p) for p in d.get('propositions', [])),
            mi=MIValue.from_dict(d.get('mi', {})),
            activation_count=int(d.get('activation_count', 0)),
            last_activated_cycle=int(d.get('last_activated_cycle', 0)),
            created_cycle=int(d.get('created_cycle', 0)),
        )


# ---------------------------------------------------------------------
# Query patterns
# ---------------------------------------------------------------------

@dataclass
class ConceptByName:
    name: str

@dataclass
class EdgesQuery:
    """Query edges by any combination of source / target / relation."""
    source: Optional[str] = None
    target: Optional[str] = None
    relation: Optional[str] = None

@dataclass
class NeighborhoodQuery:
    """Walk outgoing edges from a concept up to `max_depth` hops.
    Returns concepts found and the edges traversed."""
    seed: str
    max_depth: int = 1
    relation_filter: Optional[Set[str]] = None
    min_strength: float = 0.0

@dataclass
class ConceptsByMI:
    """Find concepts whose M/I matches a threshold on one channel."""
    channel: Literal['m', 'i'] = 'm'
    threshold: float = 0.5
    evidence_min: int = 0

@dataclass
class ScenesByProposition:
    """Find all scenes that contain a given proposition (EdgeKey)."""
    proposition: EdgeKey

@dataclass
class ScenesByMI:
    """Find scenes whose aggregate MI matches a threshold on one channel."""
    channel: Literal['m', 'i'] = 'm'
    threshold: float = 0.5


QueryPattern = Union[ConceptByName, EdgesQuery, NeighborhoodQuery,
                     ConceptsByMI, ScenesByProposition, ScenesByMI]


@dataclass
class QueryResult:
    """Tagged-union result. Exactly one of these is populated per
    query, matching the pattern type."""
    concept: Optional[Concept] = None
    edges: List[Edge] = field(default_factory=list)
    concepts: List[Concept] = field(default_factory=list)
    traversed_edges: List[Edge] = field(default_factory=list)
    scenes: List['Scene'] = field(default_factory=list)


# ---------------------------------------------------------------------
# Substrate — the typed graph container
# ---------------------------------------------------------------------

class Substrate:
    """The substrate. Concepts, edges, relations, predictions, episodes.

    All Layer 2/3 access goes through Substrate.query() or one of the
    typed mutation methods (add_concept, add_edge, etc.). No direct
    poking into internal dicts — that's how the composition wall got
    built in current SEAGI.
    """

    # Cap on retained episodes. At sustained scale, unbounded
    # episode storage is the most likely failure mode. Episodes
    # beyond the cap drop oldest-first; persistent identity's
    # defining_moments preserves the high-priority ones separately.
    EPISODES_MAX = 5000

    def __init__(self,
                 concepts: Optional[Dict[str, Concept]] = None,
                 edges: Optional[Dict[EdgeKey, Edge]] = None,
                 relations: Optional[Dict[str, Relation]] = None,
                 predictions: Optional[List[Prediction]] = None,
                 episodes: Optional[List[Episode]] = None,
                 scenes: Optional[Dict[str, Scene]] = None,
                 seed_relations: bool = True):
        self.concepts: Dict[str, Concept] = concepts or {}
        self.edges: Dict[EdgeKey, Edge] = edges or {}
        # Quarantine tier (2026-05-30): edges that none of the four
        # cognitive bid-paths engaged across L_QUARANTINE cycles and
        # whose strength settled to the prune floor get moved here.
        # Quiescent, retrievable — honors chemistry-never-fully-
        # dissolves.  Inference / coherence / reverie do not see
        # quarantined edges (they live in a separate dict, NOT
        # filtered conditionally — invisibility is by relocation).
        # Auto-restore on any bid-path touch (forager re-attest,
        # chemistry imprint, AWM promotion).  Inference does NOT
        # restore (would defeat the point).
        self.quarantine_edges: Dict[EdgeKey, Edge] = {}
        # Reverse lookup: concept name → list of quarantined edge
        # keys touching that concept (as source OR target).  Lets
        # auto-restore be O(1) on the bid-path's focal name.
        self._quarantine_index: Dict[str, List[EdgeKey]] = {}
        # Migration flag — set True after the first sleep post-
        # deploy completes its one-shot composable-coherence
        # recomputation over legacy first_coherent_cycle stamps.
        # Until then, quarantine eligibility is suppressed so the
        # canonical's edges get a full L_QUARANTINE grace window.
        self._quarantine_migrated: bool = False
        # Cursor for the incremental migration pass (legacy stamps
        # re-tested in batches of `max_per_pass`).
        self._migration_cursor: int = 0
        # Scoped-consolidation dirty set (2026-06-02): concept names
        # whose outgoing neighborhood gained structure (a new or
        # restored edge) since the last Phase S pass.  Coherence is
        # STRUCTURAL (a composing A→X→B path exists), so only edges
        # whose source neighborhood CHANGED can change coherence
        # status — unchanged regions keep their stamps.  Phase S scopes
        # reinforce_coherent_edges to this set (consume_dirty), turning
        # the O(N×branching) whole-substrate rescan that hard-stalled
        # the daemon at ~697K edges into O(new).  Runtime-only: NOT
        # serialized — empty on load means the first post-load pass
        # re-consolidates nothing (safe; consolidation resumes as the
        # agent writes), and the existing coherent core keeps its
        # persisted first_coherent_cycle stamps regardless.
        self._dirty_concepts: Set[str] = set()
        # Need-driven-nap sensors (wake-consolidation part b, 2026-07-13).
        # These carry the FELT survival cost of not-napping — the
        # homeostatic-cost signal that replaces write-debt as the sleep
        # trigger, so sleep still earns its existence:
        #   _saturated_sources — concepts whose top out-edge crossed the
        #     saturation ceiling (1 - EDGE_PRUNE_FLOOR).  O(1) write-
        #     driven at the coherence-reinforce site; the nap's
        #     downscale_saturated_edges clears it.  Never downscaled →
        #     discriminability D collapses (reasoning degrades to ties).
        #   _last_fade_backlog — count of floored edges stale past
        #     L_QUARANTINE_CYCLES, cached from the scan quarantine_inert_
        #     edges ALREADY performs each nap (no new scan/organ).  Never
        #     drained → dead edges bloat the active set, newly_coherent
        #     drops, earning falls, death nears.
        #   _last_dormant_pending — dormant-bubble backlog measured by
        #     the prune pass (bubbles age over ~min_age_cycles, so this
        #     is the slow third trigger).
        # Runtime-only, NOT serialized (like _dirty_concepts).
        self._saturated_sources: Set[str] = set()
        self._last_fade_backlog: int = 0
        self._last_dormant_pending: int = 0
        self.relations: Dict[str, Relation] = relations or {}
        self.predictions: List[Prediction] = predictions or []
        self.episodes: List[Episode] = episodes or []
        self.scenes: Dict[str, Scene] = scenes or {}
        # Index for fast scene lookup by proposition set.
        self._scene_index: Dict[frozenset, str] = {
            s.propositions: s.id for s in self.scenes.values()
        }
        # Scale indexes — fast lookup at million-concept scale.
        # Maintained as side-effects on add_concept/add_edge.
        from seagi.core.scale_index import TokenIndex, RelationIndex, LSHIndex
        self._token_index: TokenIndex = TokenIndex()
        self._relation_index: RelationIndex = RelationIndex()
        self._lsh_index: LSHIndex = LSHIndex()
        # Reverse is_a index: parent name -> set of child (is_a member) names.
        # Maintained O(1) on add_edge, rebuilt on load (from_dict) — lets
        # grounding find a concept's is_a SIBLINGS in O(siblings) instead of an
        # O(N) scan over every concept.  The prerequisite for live class-
        # generalization: a self-formed class guiding action (the out-wire).
        self._is_a_children: Dict[str, Set[str]] = {}
        # Backfill from any concepts/edges passed in.
        for name in self.concepts:
            self._token_index.add(name)
        for c in self.concepts.values():
            emb = getattr(c, 'embedding', None)
            if emb is not None and getattr(emb, 'semantic', None):
                self._lsh_index.add(c.name, emb.semantic)
        for edge in self.edges.values():
            self._relation_index.add(edge)
            if edge.relation_name == ABSTRACTION_RELATION:
                p, s = self._name_of(edge.target), self._name_of(edge.source)
                if p and s:
                    self._is_a_children.setdefault(p, set()).add(s)

        if seed_relations and not self.relations:
            for name in INITIAL_RELATION_NAMES:
                self.relations[name] = Relation(name=name)

    # ---- mutation API ----

    def add_concept(self, concept: Concept) -> Concept:
        """Add a concept (or replace existing). Returns the stored
        instance."""
        self.concepts[concept.name] = concept
        self._token_index.add(concept.name)
        emb = getattr(concept, 'embedding', None)
        if emb is not None and getattr(emb, 'semantic', None):
            self._lsh_index.add(concept.name, emb.semantic)
        return concept

    # ---- Phase C: substrate maintenance / pruning ----

    def prune_dormant_bubbles(self,
                                  cycle: int,
                                  *,
                                  min_age_cycles: int = 10000,
                                  max_crystallization: float = 0.05,
                                  max_encounter_count: int = 2
                                  ) -> int:
        """Remove bubbles that have been dormant for `min_age_cycles`
        AND have low crystallization AND low encounter count.
        Returns number of bubbles removed.

        Doctrine: "use it or lose it."  Bubbles that fired briefly,
        accumulated barely any trace, and were never re-activated
        are noise — pruning them keeps the substrate from growing
        unbounded over a lifetime.  Crystallized bubbles are
        protected regardless of recency; they encode personality.

        Safe to call periodically; no-op if no bubbles meet all
        criteria.  Caller controls pacing (e.g., every 10000
        ticks during idle).
        """
        removed = 0
        for concept in self.concepts.values():
            bubbles = getattr(concept, 'bubbles', None)
            if not bubbles:
                continue
            survivors = []
            for b in bubbles:
                last_active = int(getattr(b, 'last_active_cycle', 0))
                age = cycle - last_active
                cryst = float(getattr(b, 'crystallization', 0.0))
                enc = int(getattr(b, 'encounter_count', 0))
                if (age >= min_age_cycles
                        and cryst <= max_crystallization
                        and enc <= max_encounter_count):
                    removed += 1
                    continue
                survivors.append(b)
            if len(survivors) != len(bubbles):
                concept.bubbles = survivors
        # Part b (2026-07-13): the slow third need-driven trigger.
        # Bubbles age over ~min_age_cycles, so this is rarely the onset
        # driver; P_sat + fade_backlog carry the load.
        self._last_dormant_pending = removed
        return removed

    def settle_weak_edges(self,
                              cycle: int,
                              *,
                              floor: float = EDGE_PRUNE_FLOOR
                              ) -> int:
        """Phase S maintenance (Step 0 organ 3 — doctrine-compliant).
        Edges whose effective strength has fallen below `floor` are
        NOT removed; they are clamped UP to floor and re-anchored.
        This honors [[seagi-chemistry-never-fully-dissolves]]: edges,
        like chemistry tags and bubbles, fade to a miniature
        molecular presence (the floor) but stay re-engageable,
        never disappearing fully.

        Edges above floor get their lazy decay materialized into
        stored strength so the next pass measures from an honest
        baseline.

        Returns the number of edges that were clamped up to floor
        on this pass (the "settled" count).  Used as part of
        edges_touched feeding MetabolicDebt.record_consolidation.

        Cross-session anchor: edges loaded from a saved canonical
        carry a `last_reinforced_cycle` from the prior session,
        which can be far in the FUTURE relative to a fresh session
        counter.  Such edges are re-anchored to the current cycle
        on first pass (the agent did not experience time while
        off — decay measures experienced cycles since last use).
        """
        settled = 0
        for key, edge in self.edges.items():
            anchor = int(getattr(edge, 'last_reinforced_cycle', 0))
            if anchor > cycle:
                # Stale future anchor — restart decay clock.
                edge.last_reinforced_cycle = cycle
                continue
            eff = edge.effective_strength(cycle)
            if eff < floor:
                edge.strength = floor
                edge.last_reinforced_cycle = cycle
                settled += 1
            else:
                edge.strength = eff
                edge.last_reinforced_cycle = cycle
        return settled

    def migrate_legacy_coherence_stamps(self,
                                          cycle: int,
                                          *,
                                          max_per_pass: int = 10_000
                                          ) -> int:
        """One-shot migration (2026-05-30): re-test edges with legacy
        `first_coherent_cycle > 0` stamps against the NEW composable
        coherence predicate.  The pre-fix coherence test was purely
        topological — it stamped ~270K edges on the canonical, many of
        them noise that triangulated by accident.  Now that
        `reinforce_coherent_edges` requires composition matching, the
        legacy stamps may be false.  Clear stamps that don't pass the
        new test so MortalityDrive's growth-credit (`newly_coherent`)
        keys on legitimate first-coherences going forward.

        Runs incrementally via `_migration_cursor`; one call processes
        up to `max_per_pass` edges.  When the cursor reaches the end,
        sets `_quarantine_migrated = True` and the regular quarantine
        sweep can begin.  Returns the count of stamps cleared this pass
        (diagnostic).
        """
        if self._quarantine_migrated:
            return 0
        from seagi.brain.capabilities.cortical import RELATION_COMPOSITION
        keys = list(self.edges.keys())
        n = len(keys)
        if self._migration_cursor >= n:
            self._quarantine_migrated = True
            self._migration_cursor = 0
            return 0
        end = min(self._migration_cursor + max_per_pass, n)
        cleared = 0
        for i in range(self._migration_cursor, end):
            key = keys[i]
            edge = self.edges.get(key)
            if edge is None or edge.first_coherent_cycle == 0:
                continue
            # Re-test composable coherence: same logic as
            # reinforce_coherent_edges but read-only.
            s, r_target, t = key
            src_concept = self.concepts.get(s)
            if src_concept is None:
                # Orphaned edge — clear stamp.
                edge.first_coherent_cycle = 0
                cleared += 1
                continue
            passes = False
            for r1, edges_r1 in src_concept.edges_out.items():
                if passes:
                    break
                if not any(rc == r_target
                           for (a, _b), rc in
                           RELATION_COMPOSITION.items() if a == r1):
                    continue
                for mid_edge in edges_r1:
                    x = mid_edge.target
                    if x == s or x == t:
                        continue
                    x_concept = self.concepts.get(x)
                    if x_concept is None or not x_concept.edges_out:
                        continue
                    deg = sum(len(b)
                              for b in x_concept.edges_out.values())
                    if deg > COHERENCE_HUB_DEGREE:
                        continue
                    for r2, edges_r2 in x_concept.edges_out.items():
                        if (RELATION_COMPOSITION.get((r1, r2))
                                != r_target):
                            continue
                        for second_edge in edges_r2:
                            if second_edge.target == t:
                                passes = True
                                break
                        if passes:
                            break
                    if passes:
                        break
            if not passes:
                edge.first_coherent_cycle = 0
                cleared += 1
        self._migration_cursor = end
        if end >= n:
            self._quarantine_migrated = True
            self._migration_cursor = 0
        return cleared

    # Derived from the brain's homeostatic cadence:
    #   CLEARANCE_DEQUE_MAXLEN (=10, metabolic_debt.py — one mortality-
    #     drive L window of sleep episodes)
    # × SLEEP_CONSOLIDATION_INTERVAL (=40, runtime.py — sleep ticks
    #     between Phase S passes)
    # = 400 cycles.  One full L-window of unengaged sleep cycles is the
    # window an edge gets to be picked up by any of the four cognitive
    # bid-paths (inference traversal, composable coherence, chemistry
    # imprint, re-attestation) before it counts as inert.  Substrate
    # holds its own copy to avoid importing brain modules; comment is
    # the derivation contract.
    L_QUARANTINE_CYCLES: int = 400

    def quarantine_inert_edges(self,
                                  cycle: int,
                                  *,
                                  max_per_pass: int = 50,
                                  grace_window:
                                      Optional[int] = None
                                  ) -> int:
        """Phase S — substrate self-cleaning, tier 2.  Edges that none
        of the four bid-paths engaged across `grace_window` cycles AND
        whose strength has settled to the prune floor get moved to the
        quiescent quarantine pool.  Honors chemistry-never-fully-
        dissolves: nothing deletes; quarantined edges live in
        `self.quarantine_edges`, retrievable via auto-restore on any
        bid-path touch.

        Gated on `_quarantine_migrated` — until the one-shot migration
        of legacy `first_coherent_cycle` stamps completes, this is a
        no-op (avoids quarantining edges whose engagement signal is
        still being re-validated).

        Returns count quarantined this pass.  Bounded by
        `max_per_pass` (caller-derived from MetabolicDebt clearance
        rate so outflow ≥ inflow at steady state).

        Eligibility (2026-06-02 — now the PRESERVE arm of edge-mortality):
            first_coherent_cycle > 0   (this edge ONCE earned coherence)
            AND effective_strength(cycle) <= EDGE_PRUNE_FLOOR
            AND (cycle - last_engaged_cycle) >= grace_window

        The `first_coherent_cycle > 0` gate is NOT the inverted veto the
        round-2 audit removed (that veto existed when quarantine was the
        ONLY sink, so vetoing never-cohered edges would have left noise
        in the active substrate forever).  Now there is a REAPER:
        `reap_stillborn_edges` DELETES never-cohered (fcc==0) at-floor
        stale edges.  So the two arms split the inert population by
        whether it ever earned existence — stillborn (fcc==0) → reaped
        (mortality); past-self (fcc>0) → quarantined (preserved,
        retrievable; chemistry-never-fully-dissolves applies to what
        once cohered, not to noise).  Run the reaper BEFORE this pass.
        """
        if not self._quarantine_migrated:
            return 0
        if grace_window is None:
            grace_window = self.L_QUARANTINE_CYCLES
        # Collect eligible keys first (don't mutate during scan).
        eligible: List[Tuple[int, EdgeKey]] = []
        for key, edge in self.edges.items():
            # (2026-06-08, world-value model) FADE-NOT-DELETE: every inert
            # edge fades to the quiescent pool whether or not it ever
            # cohered.  A never-cohered relation between real words is
            # still an imprint of the world — hard to retrieve, but there,
            # restorable via any bid-path (the human-memory analogy).  The
            # fcc==0 skip (which handed stillborns to the reaper) is gone;
            # the reaper is retired.  Discard is reserved for non-word/
            # non-concept noise at INGESTION, never for substrate here.
            if edge.effective_strength(cycle) > EDGE_PRUNE_FLOOR:
                continue
            staleness = cycle - int(edge.last_engaged_cycle)
            if staleness < grace_window:
                continue
            eligible.append((staleness, key))
        # Part b (2026-07-13): cache the fade backlog sensed by THIS scan
        # (the one the nap already runs) so the need-driven onset can read
        # it without a second O(E) pass.  Bloat here = dead edges in the
        # active set = the felt cost of not-napping.
        self._last_fade_backlog = len(eligible)
        if not eligible:
            return 0
        # Sort by staleness descending — oldest unengaged go first.
        # Cap at max_per_pass (inflow-rate-derived by the caller).
        eligible.sort(key=lambda p: -p[0])
        n = 0
        for _staleness, key in eligible[:max_per_pass]:
            if self._quarantine_edge(key):
                n += 1
        return n

    def reap_stillborn_edges(self,
                                 cycle: int,
                                 *,
                                 max_per_pass: int = 50,
                                 synthetic_only: bool = False,
                                 grace_window:
                                     Optional[int] = None) -> int:
        """RE-ARMED **SCOPED** 2026-07-24 (user sign-off): called per-nap
        with synthetic_only=True.  The 06-08 fade-not-delete criterion
        ('discard only what is not a word and not a concept') by IDENTITY
        mandates discarding never-cohered _abstract_* machinery exhaust --
        it is not a word of the world.  Word-imprints keep fade-not-delete
        (curriculum food untouched).  Phase S — EDGE-MORTALITY
        (2026-06-02).  Edges that NEVER once
        cohered (first_coherent_cycle == 0), have settled to the prune
        floor, AND were untouched by any of the four cognitive bid-paths
        across `grace_window` cycles are DELETED outright.

        This is the mortality vision applied to the substrate itself.
        An edge earns its existence by cohering — being independently
        corroborated by a composing A→X→B path (reinforce_coherent_
        edges).  An edge the parser or reverie threw up, that the graph
        never corroborated across a full engagement window, and that has
        decayed to the floor, never earned existence: it is a stillborn,
        and the mortality vision says it dies rather than lingering
        forever.  Without this the substrate is IMMORTAL — settle_weak_
        edges floors weak edges and quarantine relocates them, but
        nothing ever removed the never-cohered noise (the canonical's
        ~81% never-coherent edges), which both bloats the substrate and
        (pre-scoping) hard-stalled consolidation.

        Distinct from quarantine_inert_edges, which PRESERVES inert
        edges that DID once cohere (first_coherent_cycle > 0) — a faded
        past-self, retrievable.  Stillborn != past-self: the never-
        coherent edge has no earned existence to preserve.  This
        boundary keeps edge-mortality consistent with chemistry-never-
        fully-dissolves (about what once mattered, not about noise).

        Gated on `_quarantine_migrated` (same as quarantine): a no-op
        until the one-shot migration re-validates legacy first_coherent
        stamps against the composable predicate, so a noise edge with a
        bogus legacy stamp isn't spared and a real edge isn't killed on
        a stale read.  Bounded by `max_per_pass` (caller derives it from
        the metabolic clearance rate, like quarantine).  Returns the
        count reaped this pass.

        Eligibility:
            first_coherent_cycle == 0
            AND effective_strength(cycle) <= EDGE_PRUNE_FLOOR
            AND (cycle - last_engaged_cycle) >= grace_window
        """
        if not self._quarantine_migrated:
            return 0
        if grace_window is None:
            grace_window = self.L_QUARANTINE_CYCLES
        eligible: List[Tuple[int, EdgeKey]] = []
        for key, edge in self.edges.items():
            if synthetic_only and not (
                    key[1] == ABSTRACTION_RELATION
                    and str(key[2]).startswith(ABSTRACTION_NAME_PREFIX)):
                continue
            if int(edge.first_coherent_cycle) != 0:
                continue
            if edge.effective_strength(cycle) > EDGE_PRUNE_FLOOR:
                continue
            staleness = cycle - int(edge.last_engaged_cycle)
            if staleness < grace_window:
                continue
            eligible.append((staleness, key))
        if not eligible:
            return 0
        # Oldest-unengaged first, capped (symmetric with quarantine).
        eligible.sort(key=lambda p: -p[0])
        n = 0
        for _staleness, key in eligible[:max_per_pass]:
            if self._reap_edge(key):
                n += 1
        return n

    def sweep_stillborn_backlog(self,
                                    cycle: int,
                                    *,
                                    synthetic_only: bool = False,
                                    grace_window: int = 0) -> int:
        """One-time OFFLINE backlog sweep (2026-06-02).  Drains the
        accumulated never-cohered noise that built up while the
        substrate was effectively immortal, in a single O(N) scan-and-
        delete (no per-pass cap).  Intended for a maintenance context
        on a COPY of the canonical, NOT the live tick loop.

        Forces the one-shot legacy-stamp migration to completion first
        (so first_coherent_cycle == 0 is trustworthy), then deletes
        every never-cohered, at-or-below-floor edge older than
        `grace_window` (default 0 — these edges have already had the
        whole prior substrate lifetime to earn coherence and didn't).
        Returns the number reaped.

        Edges that DID cohere (fcc > 0) are untouched — real knowledge
        is preserved; only stillborn noise is removed.
        """
        guard = 0
        while not self._quarantine_migrated and guard < 1_000_000:
            self.migrate_legacy_coherence_stamps(cycle)
            guard += 1
        doomed: List[EdgeKey] = []
        for key, edge in self.edges.items():
            if synthetic_only and not (
                    key[1] == ABSTRACTION_RELATION
                    and str(key[2]).startswith(ABSTRACTION_NAME_PREFIX)):
                continue
            if int(edge.first_coherent_cycle) != 0:
                continue
            if edge.effective_strength(cycle) > EDGE_PRUNE_FLOOR:
                continue
            if (cycle - int(edge.last_engaged_cycle)) < grace_window:
                continue
            doomed.append(key)
        for key in doomed:
            self._reap_edge(key)
        return len(doomed)

    def downscale_saturated_edges(self,
                                       cycle: int,
                                       *,
                                       saturation_floor: float = (
                                           1.0 - EDGE_PRUNE_FLOOR),
                                       factor: float = (
                                           1.0 - COHERENCE_REINFORCE_BUMP),
                                       ) -> int:
        """Step 0 organ 3 — Phase S downscaling.  Restores
        discriminability D = mean(top - second strength) by
        knocking saturated concepts back from ceiling.

        A concept is "saturated" when its top outgoing edge has
        effective strength above `saturation_floor` (default
        1 - EDGE_PRUNE_FLOOR = 0.98).  When that happens, multiple
        edges sit at the ceiling and (top - second) collapses to
        near zero — rank information dissolves.

        On each saturated concept, multiply ALL outgoing edges'
        strengths by `factor` (default 1 - COHERENCE_REINFORCE_BUMP
        = 0.975).  Ranks preserved (multiplicative), but the whole
        ladder drops back below the ceiling, restoring headroom
        for next-cycle reinforcement to discriminate.

        Returns the number of edge-multiplications applied (the
        "edges_touched" count for MetabolicDebt clearance).

        Doctrine traces:
        - saturation_floor = 1 - EDGE_PRUNE_FLOOR (existing constant)
        - factor = 1 - COHERENCE_REINFORCE_BUMP (existing constant)
        - knocked-down amount equals one coherence-reinforce bump,
          so downscale + coherence-reinforce balance at equilibrium
        """
        touched = 0
        for concept in self.concepts.values():
            if not concept.edges_out:
                continue
            top = 0.0
            for edges in concept.edges_out.values():
                for e in edges:
                    s = e.effective_strength(cycle)
                    if s > top:
                        top = s
            if top <= saturation_floor:
                continue
            for edges in concept.edges_out.values():
                for e in edges:
                    s = e.effective_strength(cycle)
                    e.strength = max(0.0, s * factor)
                    e.last_reinforced_cycle = cycle
                    touched += 1
        # Part b (2026-07-13): this whole-substrate pass IS the authority
        # on saturation — it re-scanned every concept's top edge.  Clear
        # the O(1) write-driven pending set so P_sat resets to the truth
        # each nap; saturation must be RE-EARNED by fresh reinforcement.
        self._saturated_sources.clear()
        return touched

    def discriminability(self, cycle: int) -> float:
        """Step 0 organ 3 — substrate discriminability scalar D.

        D = mean of (top_strength - second_strength) over concepts
        with at least 2 outgoing edges, using effective_strength at
        the given cycle.  Concepts with 0 or 1 outgoing edges
        contribute nothing (no rank gradient defined).

        Range: [0, 1].  D ≈ 0 means edges are bunched (no rank
        separation, the substrate has collapsed to mass).  D ≈ 1
        means a clear winner edge per concept (sharp inference).

        Used by DiscriminabilityTracker to capture the wake_onset
        baseline and the current value; their ratio drives the
        d_modulation term in the SleepRegulator gate.

        Cold-start (substrate with no qualifying concepts): returns
        0 so d_modulation gates itself out (cold-start safety).
        """
        total = 0.0
        count = 0
        # SAMPLED, NOT SWEPT (2026-08-20).  Walking all 258k concepts /
        # ~1.5M edges cost ~5.7 s per call and 30.1% of his entire tick,
        # to produce ONE scalar for the sleep gate.  D is a MEAN, so a
        # bounded random sample estimates it to far better precision than
        # the gate's own sensitivity.  Own RNG => the actor's byte parity
        # is untouched; seeded by cycle => deterministic.
        _pool = self.concepts
        if D_SAMPLE_CONCEPTS and len(_pool) > D_SAMPLE_CONCEPTS:
            import random as _r
            _keys = list(_pool.keys())
            _rng = _r.Random(int(cycle))
            _iter = (_pool[k] for k in _rng.sample(_keys, D_SAMPLE_CONCEPTS))
        else:
            _iter = _pool.values()
        for concept in _iter:
            edges_out = concept.edges_out
            if not edges_out:
                continue
            strengths = []
            for edges in edges_out.values():
                for e in edges:
                    strengths.append(e.effective_strength(cycle))
                    if len(strengths) >= 2 and strengths[-1] >= strengths[-2]:
                        # cheap path: keep accumulating; sort at end
                        pass
            if len(strengths) < 2:
                continue
            strengths.sort(reverse=True)
            total += (strengths[0] - strengths[1])
            count += 1
        if count == 0:
            return 0.0
        return total / float(count)

    def consume_dirty(self) -> Set[str]:
        """Return the dirty-concept set (sources/targets whose
        neighborhood gained structure since the last call) and clear
        it.  Phase S calls this to SCOPE coherence reinforcement to the
        changed neighborhood — the O(new) un-stall.  Returns a snapshot;
        the live set is reset to empty so the next pass starts fresh.
        """
        d = self._dirty_concepts
        self._dirty_concepts = set()
        return d

    @staticmethod
    def _name_of(x) -> Optional[str]:
        return x if isinstance(x, str) else getattr(x, 'name', None)

    def _rebuild_is_a_index(self) -> None:
        """Rebuild the parent->children is_a index from the active edges (O(E)).
        Called on load (from_dict), where edges are re-attached directly rather
        than through add_edge's incremental maintenance."""
        idx: Dict[str, Set[str]] = {}
        for (s, r, t), _e in self.edges.items():
            if r == ABSTRACTION_RELATION:
                idx.setdefault(t, set()).add(s)
        self._is_a_children = idx

    def is_a_siblings(self, name: str) -> Set[str]:
        """The is_a SIBLINGS of `name` — concepts sharing at least one is_a
        parent with it (excluding itself).  O(siblings) via the reverse index;
        this is what makes grounding's class-generalization viable at scale."""
        c = self.concepts.get(name)
        if c is None:
            return set()
        sibs: Set[str] = set()
        for pe in c.edges_out.get(ABSTRACTION_RELATION, ()):
            parent = self._name_of(pe.target)
            if parent:
                sibs |= self._is_a_children.get(parent, set())
        sibs.discard(name)
        return sibs

    def reinforce_coherent_edges(
            self, cycle: int, *,
            candidates: Optional[Set[str]] = None,
            newly_coherent_sources: Optional[Set[str]] = None
            ) -> Tuple[int, int]:
        """Phase S.2 — coherence reinforcement.  An edge (A, R, B)
        COHERES when the surrounding graph independently corroborates
        it: a path A→X→B exists through some non-hub middle concept X
        AND **the relations along that path compose** to R via
        `RELATION_COMPOSITION`.

        SCOPING (2026-06-02, the un-stall): when `candidates` is given
        (a set of source-concept names from consume_dirty), only edges
        whose SOURCE is in that set are re-evaluated — the candidate
        edges whose neighborhood could have changed coherence this
        cycle.  This turns the O(N×branching) whole-substrate rescan
        (py-spy: the per-edge hub-degree sum hard-stalled the daemon at
        ~697K edges, ~1 tick/9min) into O(dirty×local-degree).  Coherence
        is structural, so unchanged regions keep their persisted stamps;
        `candidates=None` falls back to the full scan (used by tests and
        offline tooling).  DECLARED LIMITATION: a candidate that newly
        coheres ONLY because a freshly-added edge completes its SECOND
        hop (the new edge's source ≠ the candidate's source) is not
        re-checked until that region is next written — caught on a later
        pass, never lost (first_coherent_cycle persists once set).

        The composition requirement is the doctrine-correct earn-gate
        (audit 2026-05-30): purely topological coherence (any A→X→B
        triangle) was reinforcing noise — `co_occurs`, `creates`,
        verb-parser-noise edges triangulated by accident and earned
        coherence credit they don't deserve.  By demanding the
        corroborating chain's relations COMPOSE to the candidate edge's
        own relation, edges with non-canonical relations cannot earn
        coherence — they fade to the prune floor and become quarantine-
        eligible.  This aligns coherence's earn-criterion with the same
        composition table `_inference_chain` uses to walk reasoning
        chains — two organs agreeing on what counts as evidence.

        Paths through hub nodes (out-degree > COHERENCE_HUB_DEGREE)
        are not counted — a hub connects to everything, so a path
        through it is not specific corroboration.

        Run alongside settle_weak_edges.  Returns a tuple:
            (reinforced, newly_coherent)
        `reinforced` = total coherent edges bumped this pass (whole-
            substrate consolidation work; feeds MetabolicDebt
            clearance).
        `newly_coherent` = edges that cohered for the FIRST TIME EVER
            on this pass (`first_coherent_cycle` was unset).  This is
            the earn-gated GROWTH signal — feeds MortalityDrive's L.

        Engagement: on successful corroboration we stamp
        `last_engaged_cycle` on the CANDIDATE edge AND on the two
        middle-hop edges (the corroborators).  This is the coherence
        bid-path for the quarantine tier — middle-hop edges that keep
        finding themselves in valid corroborations stay alive even if
        they're never walked by inference directly.
        """
        # Lazy import to avoid circular dependency (cortical → substrate
        # at module load).  The composition table is the source-of-
        # truth for what counts as a meaningful relation chain.
        from seagi.brain.capabilities.cortical import RELATION_COMPOSITION

        # Scope the candidate-edge set.  None → full scan (back-compat).
        # Otherwise build the candidate list from the dirty concepts'
        # outgoing edges only — O(dirty×local-degree), not O(N).
        if candidates is None:
            edge_items = list(self.edges.items())
        else:
            edge_items = []
            for cname in candidates:
                cc = self.concepts.get(cname)
                if cc is None or not cc.edges_out:
                    continue
                for bucket in cc.edges_out.values():
                    for e in bucket:
                        edge_items.append((e.key, e))

        reinforced = 0
        newly_coherent = 0
        # --- value-web SHADOW (drives nothing): does continuous
        # corroboration reach + hold the felt core, and hold the active
        # set where recency-decay alone would have floored it? ---
        vw_corro = 0
        vw_rescued = 0
        vw_felt = 0
        vw_tension_top = 0.0
        for (s, r_target, t), edge in edge_items:
            src_concept = self.concepts.get(s)
            if src_concept is None or not src_concept.edges_out:
                continue
            corroborated = False
            corr_edges: List[Edge] = []  # the two middle-hop edges
            scanned = 0
            # Walk outgoing edges of s — these are candidate (R1, x)
            # first-hops on the corroborating path.
            for r1, edges_r1 in src_concept.edges_out.items():
                if corroborated:
                    break
                # Compositions starting with r1 that could land on
                # r_target.  Cheap pre-filter: skip r1 if no
                # (r1, _) entry composes to r_target at all.
                if not any(rc == r_target
                           for (a, _b), rc in RELATION_COMPOSITION.items()
                           if a == r1):
                    continue
                for mid_edge in edges_r1:
                    x = mid_edge.target
                    if x == s or x == t:
                        continue
                    scanned += 1
                    if scanned > COHERENCE_NEIGHBOR_SCAN_CAP:
                        break
                    x_concept = self.concepts.get(x)
                    if x_concept is None or not x_concept.edges_out:
                        continue
                    # Hub filter: total outgoing degree of X.
                    deg = sum(len(b)
                              for b in x_concept.edges_out.values())
                    if deg > COHERENCE_HUB_DEGREE:
                        continue
                    # Look for the second hop (R2, t) under any R2
                    # such that (r1, R2) composes to r_target.
                    for r2, edges_r2 in x_concept.edges_out.items():
                        composed = RELATION_COMPOSITION.get((r1, r2))
                        if composed != r_target:
                            continue
                        for second_edge in edges_r2:
                            if second_edge.target == t:
                                corroborated = True
                                corr_edges = [mid_edge, second_edge]
                                break
                        if corroborated:
                            break
                    if corroborated:
                        break
            if not corroborated:
                continue
            # value-web shadow: measure BEFORE the bump — pre-bump
            # effective_strength is what recency-decay ALONE would give,
            # and effective_mi is the LIVE (decayed) tag.  Drives nothing.
            try:
                vw_corro += 1
                if edge.effective_strength(cycle) <= EDGE_PRUNE_FLOOR:
                    vw_rescued += 1
                _vm = edge.effective_mi(cycle)
                _vt = getattr(_vm, 'tension', None)
                _tens = _vt() if callable(_vt) else float(_vt or 0.0)
                if _tens >= 0.1:
                    vw_felt += 1
                if _tens > vw_tension_top:
                    vw_tension_top = _tens
            except Exception:
                pass
            # Reinforce the candidate edge and stamp coherence
            # engagement on all three participants.
            edge.reinforce(cycle, delta=COHERENCE_REINFORCE_BUMP,
                           origin='maintenance')
            edge.last_engaged_cycle = int(cycle)
            reinforced += 1
            # Homeostatic-cost sensor (part b, 2026-07-13): a reinforced
            # edge that crossed the saturation ceiling (the SAME threshold
            # downscale_saturated_edges uses) marks its source as needing
            # downscale — P_sat, the need-driven nap's felt-cost trigger.
            # O(1) write-driven; the nap's downscale clears the set.
            if edge.effective_strength(cycle) > (1.0 - EDGE_PRUNE_FLOOR):
                self._saturated_sources.add(s)
            for ce in corr_edges:
                ce.last_engaged_cycle = int(cycle)
            # First-ever coherence is genuine growth.
            if edge.first_coherent_cycle == 0:
                edge.first_coherent_cycle = int(cycle) or 1
                newly_coherent += 1
                # Wake-consolidation (2026-07-13): surface WHICH focal
                # (edge source) reached first-ever coherence, so the
                # WAKE-LOCAL caller can complete a resolve_uncertainty
                # goal on it (D3 Wire B).  Optional out-collector; the
                # return tuple is unchanged, so every existing caller is
                # untouched.  first_coherent_cycle is set once, so a
                # source is surfaced only on its genuine first coherence.
                if newly_coherent_sources is not None:
                    newly_coherent_sources.add(s)
        # value-web shadow log — only on live (scoped) consolidation
        # passes that did real work; drives nothing but a log line.
        if candidates:
            try:
                import time as _vtime
                with open('/home/seagi/valueweb_shadow.log', 'a') as _vf:
                    _vf.write(
                        "[%s] cycle=%d candidates=%d corroborated=%d "
                        "rescued_from_floor=%d felt(tension>=0.1)=%d "
                        "tension_top=%.3f newly=%d\n" % (
                            _vtime.strftime('%Y-%m-%dT%H:%M:%SZ',
                                            _vtime.gmtime()),
                            int(cycle), len(candidates), vw_corro,
                            vw_rescued, vw_felt, vw_tension_top,
                            newly_coherent))
            except Exception:
                pass
        return reinforced, newly_coherent

    def recheck_edge_coherence(self, edge, cycle):
        """Single-edge coherence re-check -- the per-edge corroboration
        body of reinforce_coherent_edges, for ONE edge, NO mutation.
        Returns (corroborated, corr_edges).  The ReplayConsolidator
        return path reinforces + stamps last_engaged ONLY when this is
        True.  NEVER increments newly_coherent, NEVER touches
        first_coherent_cycle, NEVER calls reinforce_coherent_edges -- so
        re-engagement can never reach record_learning (lifeforce
        decoupling, by construction).
        """
        from seagi.brain.capabilities.cortical import RELATION_COMPOSITION
        s = edge.source
        t = edge.target
        r_target = edge.relation_name
        src_concept = self.concepts.get(s)
        if src_concept is None or not src_concept.edges_out:
            return False, []
        scanned = 0
        for r1, edges_r1 in src_concept.edges_out.items():
            if not any(rc == r_target
                       for (a, _b), rc in RELATION_COMPOSITION.items()
                       if a == r1):
                continue
            for mid_edge in edges_r1:
                x = mid_edge.target
                if x == s or x == t:
                    continue
                scanned += 1
                if scanned > COHERENCE_NEIGHBOR_SCAN_CAP:
                    return False, []
                x_concept = self.concepts.get(x)
                if x_concept is None or not x_concept.edges_out:
                    continue
                deg = sum(len(b) for b in x_concept.edges_out.values())
                if deg > COHERENCE_HUB_DEGREE:
                    continue
                for r2, edges_r2 in x_concept.edges_out.items():
                    if RELATION_COMPOSITION.get((r1, r2)) != r_target:
                        continue
                    for second_edge in edges_r2:
                        if second_edge.target == t:
                            return True, [mid_edge, second_edge]
        return False, []

    def derive_closure(self,
                          cycle: int,
                          *,
                          max_rounds: int = 4,
                          max_new: int = 5000,
                          candidates=None) -> int:
        """Phase S — DEDUCTIVE CLOSURE (reasoning compounds, 2026-06-02).

        The inference engine (cortical._inference_chain) returns only ONE
        best derivation per focal, so reverie can derive a fact but never
        enumerate the FULL set of consequences of what the agent knows —
        and the deeper conclusions (that need a derived fact as a premise)
        never surface.  Held-out test: one-best gave 1/10 autonomously;
        enumerate-all closure gives 10/10.

        This pass enumerates EVERY valid 2-hop composition A-r1->X-r2->B
        where (r1,r2) composes to R via RELATION_COMPOSITION and the
        composed confidence clears INFERENCE_MIN_CONFIDENCE, and writes
        the derived conclusion (A,R,B) at WEAKEST-LINK strength — the min
        of the two premise strengths: a conclusion is as established as
        its weakest support.  No hand-tuned constant; strength is derived
        from the premises, the 0.12 floor is the noise gate.  Newly-
        derived facts become premises in the next round, so the agent
        thinks through the full implications of what it knows, iterating
        to convergence.

        Earn-or-dissolve still governs: a derived edge enters and, if a
        supporting premise later fades, loses corroboration and dissolves
        like any provisional edge.  Only compositions of edges STRONG
        enough to clear the floor fire (both premises ~>=0.42 after the
        penalty), so weak/noise edges never enter the closure.

        Runs during sleep consolidation.  Returns total derived this pass.
        """
        from seagi.brain.capabilities.cortical import (
            RELATION_COMPOSITION, INFERENCE_MIN_CONFIDENCE,
            MULTI_HOP_PENALTY)
        # Scope (2026-06-04 audit #2): when `candidates` (the dirty
        # neighbourhood from consume_dirty) is given, derive only from
        # edges whose SOURCE is in that set — the proven un-stall
        # pattern — expanding outward through each derived conclusion so
        # the closure still reaches the consequences of what just
        # changed.  When None, closes over the whole substrate (the
        # test / one-shot batch path).
        active = set(candidates) if candidates is not None else None
        # Premise-reinforcement bookkeeping (2026-06-05): reinforce each
        # edge that PROVES a conclusion at most once per call.
        reinforced_premises: set = set()
        total = 0
        for _rnd in range(max_rounds):
            pending: List[Tuple[str, str, str, float]] = []
            # Scope (wake-consolidation 2026-07-13): when `active` (the
            # dirty neighbourhood) is given, iterate ONLY the candidate
            # concepts' own edges_out — O(dirty×local-degree) — instead
            # of scanning every edge in the substrate (the O(E) per-round
            # sweep the old `for ... in self.edges.items(): if a not in
            # active: continue` performed).  Rebuilt each round so a
            # conclusion added to `active` last round expands the frontier
            # this round.  `active is None` keeps the full-substrate scan
            # (test / one-shot batch path).
            if active is not None:
                first_hops: List[Tuple[str, str, Any]] = []
                for a in list(active):
                    ac = self.concepts.get(a)
                    if ac is None or not ac.edges_out:
                        continue
                    for r1, bucket in list(ac.edges_out.items()):
                        for e1 in bucket:
                            first_hops.append((a, r1, e1))
            else:
                first_hops = [
                    (a, r1, e1)
                    for (a, r1, _x), e1 in list(self.edges.items())]
            for (a, r1, e1) in first_hops:
                x = self._name_of(e1.target)
                if x is None:
                    continue
                s1 = e1.effective_strength(cycle)
                # Pre-filter: even a perfect 2nd hop can't clear the
                # floor if the first hop is too weak.
                if s1 * MULTI_HOP_PENALTY < INFERENCE_MIN_CONFIDENCE:
                    continue
                xc = self.concepts.get(x)
                if xc is None or not xc.edges_out:
                    continue
                for r2, e2list in xc.edges_out.items():
                    composed = RELATION_COMPOSITION.get((r1, r2))
                    if composed is None:
                        continue
                    for e2 in e2list:
                        tgt = getattr(e2, 'target', None)
                        b = (tgt if isinstance(tgt, str)
                             else getattr(tgt, 'name', None))
                        if not b or b == a:
                            continue
                        if (a, composed, b) in self.edges:
                            continue
                        s2 = e2.effective_strength(cycle)
                        conf = (min(1.0, s1) * MULTI_HOP_PENALTY
                                * min(1.0, s2))
                        if conf < INFERENCE_MIN_CONFIDENCE:
                            continue
                        pending.append((a, composed, b, min(s1, s2)))
                        # Premise-reinforcement (2026-06-05): both hops
                        # just PROVED a new conclusion (cleared the floor;
                        # the conclusion was not already in the substrate)
                        # — so they EARNED strength, moving away from the
                        # prune floor (death).  Reinforce each useful
                        # premise once per pass, so edges that DO
                        # reasoning become stronger and more reason-able:
                        # the compounding the mortality architecture is
                        # meant to provide (a proven premise moves toward
                        # life).  Gated by the SAME floor as the
                        # derivation — noise at the prune floor can never
                        # compose, so only genuinely-composable edges
                        # earn (anti-gaming by construction).
                        k1 = (a, r1, x)
                        if k1 not in reinforced_premises:
                            e1.reinforce(cycle, origin='cognition')
                            reinforced_premises.add(k1)
                        k2 = (x, r2, b)
                        if k2 not in reinforced_premises:
                            e2.reinforce(cycle, origin='cognition')
                            reinforced_premises.add(k2)
            if not pending:
                break
            wrote = 0
            for (a, R, b, strength) in pending:
                if (a, R, b) in self.edges:
                    continue
                self.add_edge(a, b, R, strength=strength, cycle=cycle)
                if active is not None:
                    active.add(b)   # derived conclusion can chain further
                total += 1
                wrote += 1
                if total >= max_new:
                    return total
            if wrote == 0:
                break
        return total

    def form_abstractions(self,
                              cycle: int,
                              *,
                              min_group: int = ABSTRACTION_MIN_GROUP,
                              max_new: int = ABSTRACTION_MAX_NEW_PER_PASS,
                              candidates: Optional[Set[str]] = None
                              ) -> int:
        """Roadmap Step 4 — concept formation by detected regularity.

        Groups source-concepts by (relation, target).  Each group
        of >= `min_group` members is a regularity worth naming —
        form (or reinforce) a synthetic class concept
        `_abstract_{relation}_{target}` and add provisional is_a
        edges from each member to it.

        Synthetic concepts and their is_a edges enter the normal
        earn-or-dissolve pipeline:
          - An abstraction that gets WALKED in cortical inference
            (R.1 composes is_a chains) gets its edges reinforced.
          - An abstraction that's never used has its edges decay
            below the prune floor and dissolves — leaving its
            synthetic concept orphan (harmless).
          - Re-running this method REINFORCES existing
            abstractions whose member group still satisfies
            min_group, via the writer's re-attestation path.

        Capped at `max_new` new abstractions per pass (sorted
        by group size, largest first).  Returns count of new
        abstractions created this pass.

        SCOPED (wake-consolidation 2026-07-13): when `candidates` (the
        dirty set from consume_dirty) is given, only the dirty members
        joining a (relation, target) group are processed, and the is_a
        edge is attached ONLY to those dirty member(s) — NOT re-stamped
        onto every group member every pass.  The co-group size is read
        in O(1) from the (relation, target) index, so the whole pass is
        O(dirty×local-degree) instead of the O(E) group-build below.
        Each member earns its is_a on the tick its own (relation, target)
        edge is written/re-engaged (which is exactly when it enters the
        dirty set), so the union across passes reproduces the full-scan
        membership without the whole-substrate rescan.  `candidates=None`
        keeps the full-scan path (tests / offline batch).
        """
        if candidates is not None:
            return self._form_abstractions_scoped(
                cycle, candidates, min_group=min_group, max_new=max_new)
        # Group sources by (relation, target).
        groups: Dict[tuple, List[str]] = {}
        for (s, r, t) in self.edges:
            # Don't form abstractions OVER abstractions (avoid
            # runaway hierarchy on a single pass — let real use
            # earn higher-order classes over time if needed).
            if s.startswith(ABSTRACTION_NAME_PREFIX):
                continue
            if t.startswith(ABSTRACTION_NAME_PREFIX):
                continue
            # Wall down (2026-06-30): firsthand world tokens CAN be abstracted.
            # Formation-vs-use is resolved — _learning_credit (runtime.py)
            # credits only newly_coherent USE, never FORMATION, so forming a
            # world abstraction credits zero lifeforce; the credit gate closes
            # the farm, the wall is redundant.  Earn-or-dissolve is sole guard:
            # a world class that coheres through use earns life, a hollow one
            # fades.
            # And ignore is_a edges TO existing abstractions (the
            # very edges this method creates) — they're the
            # output, not input to grouping.
            if r == ABSTRACTION_RELATION:
                continue
            # SHADOW instrument (2026-07-11): `has_role` edges are the
            # RoleRegularityShadow's provisional structural-role tags.  They
            # must NEVER become traversable is_a classes (that would let a
            # read-only instrument poison grounding._predict) — skip them as
            # grouping sources, the same guard as ABSTRACTION_RELATION above.
            if r == ROLE_RELATION:
                continue
            # Facet-docking (2026-07-21): `has_facet` is a CONTAINER
            # relation, not a category — a facet class would be a 1:1
            # extensional duplicate of the `_facet_*` node.  Skip it as a
            # grouping source (same guard as has_role) so no is_a edge with
            # a facet endpoint is ever minted (the re-audit ruled this
            # PARSIMONY, not a wall).
            if r == FACET_RELATION:
                continue
            key = (r, t)
            bucket = groups.get(key)
            if bucket is None:
                groups[key] = [s]
            else:
                bucket.append(s)
        # Candidates sorted by group size (strongest regularities
        # first).
        candidates = [(k, srcs) for k, srcs in groups.items()
                          if len(srcs) >= min_group]
        candidates.sort(key=lambda item: -len(item[1]))
        new_created = 0
        _edge_budget = [0]   # is_a edges attached this pass
        for (rel, tgt), srcs in candidates:
            abstraction_name = f"{ABSTRACTION_NAME_PREFIX}{rel}_{tgt}"
            existed = abstraction_name in self.concepts
            if not existed:
                if new_created >= max_new:
                    continue
                # Create synthetic class concept.
                #
                # PRICING / RETRO-DEMONETIZATION WATCH (2026-07-19):
                # `synthetic=True` is ECONOMY-CRITICAL.  Naming a class
                # collapses its members' FUTURE settle credit to family
                # grain — MortalityClock._fact_key keys every membership
                # edge into this concept as the single fact ('∈', name),
                # and keys members' machinery-entailed derived edges as
                # (name, r, t).  So the moment this line runs, N future
                # member settlings become 1.
                # Safe TODAY because abstraction timing is not
                # policy-controllable (this runs on the consolidation
                # pass, not on anything cognition chooses).  If it ever
                # becomes controllable, DEFERRED NAMING turns into an
                # earning strategy — collect N instance-grain settlings,
                # THEN name the class — and the grain must be
                # re-audited.  Any new synthetic=True site must be
                # registered in tests/test_synthetic_pricing_doctrine.py
                # with a pricing assertion.
                c = Concept(
                    name=abstraction_name,
                    mi=MIValue.zero(),
                    created_cycle=cycle,
                    synthetic=True)
                self.concepts[abstraction_name] = c
                self._token_index.add(abstraction_name)
                new_created += 1
            # Attach (or reinforce) member is_a edges.  Dedup
            # naturally via the (s, is_a, abstraction_name) key.
            for member in set(srcs):   # dedupe
                if (_ISABUDGET_ON()
                        and _edge_budget[0] >= ABSTRACTION_MAX_EDGES_PER_PASS):
                    break
                if member == abstraction_name:
                    continue
                key = (member, ABSTRACTION_RELATION,
                          abstraction_name)
                existing = self.edges.get(key)
                if existing is None:
                    # Self-cleaning F2 (2026-07-24): faded exhaust stays
                    # faded to this engine -- no self-restore.
                    if key in self.quarantine_edges:
                        continue
                    self.add_edge(
                        source=member,
                        target=abstraction_name,
                        relation_name=ABSTRACTION_RELATION,
                        strength=PROVISIONAL_EDGE_STRENGTH,
                        cycle=cycle)
                    _edge_budget[0] += 1
                elif (existing.last_engaged_cycle
                        > existing.last_reinforced_cycle):
                    # Re-attest credits ONLY on fresh external engagement
                    # since the last credit (earned, never stamped).
                    existing.reinforce(cycle, origin='cognition')
        return new_created

    def form_analogies(self,
                            cycle: int,
                            *,
                            min_skeleton: int = ANALOGY_MIN_SKELETON,
                            max_new: int = ANALOGY_MAX_NEW_PER_PASS,
                            candidates: Optional[Set[str]] = None
                            ) -> int:
        """Analogy formation — structural cross-domain mapping.

        Two concepts are structural analogs when they share an
        outgoing relation-SKELETON (same set of relation types,
        size >= `min_skeleton`) BUT map to DISJOINT targets: the
        same relational shape over entirely different content.
        For each such pair, write a provisional `analogous_to`
        edge (canonical sorted order, deduped).

        These edges enter the normal earn-or-dissolve pipeline:
          - An analogy WALKED in cortical inference (or otherwise
            re-attested) gets reinforced and survives.
          - An idle analogy decays below the prune floor and
            settles out — the substrate self-curates which
            structural mappings are real.
          - Re-running REINFORCES still-valid analogies.

        Complementary to form_abstractions: abstraction groups by
        shared (relation, target); analogy groups by shared
        relation-SET with DISJOINT targets.  The disjoint-targets
        rule is threshold-free (no hand-tuned similarity cutoff):
        shared structure + zero shared content.

        Bounded: concepts bucketed by exact relation-skeleton
        signature; richest skeletons (most relation types =
        most meaningful structural match) processed first; at
        most `ANALOGY_BUCKET_MEMBER_CAP` members paired per bucket;
        capped at `max_new` new edges per pass.  Returns count of
        new analogy edges created.

        SCOPED (wake-consolidation 2026-07-13): when `candidates` (the
        dirty set) is given, only the dirty concepts' relation-skeletons
        are re-evaluated.  For each dirty concept, partners are the other
        concepts sharing its EXACT skeleton — found by intersecting the
        existing `_relation_index` on the rarest relation in the skeleton
        (no new index, no O(E) rebuild), verified by an O(local-degree)
        exact-skeleton recompute per candidate partner.  Same disjoint-
        target rule, same provisional strength, same earn-or-dissolve as
        the full-scan path; `candidates=None` keeps the full scan.
        """
        if candidates is not None:
            return self._form_analogies_scoped(
                cycle, candidates,
                min_skeleton=min_skeleton, max_new=max_new)
        # Per-concept outgoing relation-type set + target set.
        rels_by_concept: Dict[str, set] = {}
        targets_by_concept: Dict[str, set] = {}
        for (s, r, t) in self.edges:
            # Ignore our own output + abstraction scaffolding so
            # analogies form over REAL relational structure.
            if r == ANALOGY_RELATION:
                continue
            if (s.startswith(ABSTRACTION_NAME_PREFIX)
                    or t.startswith(ABSTRACTION_NAME_PREFIX)):
                continue
            # Wall down (2026-06-30): firsthand world tokens CAN form analogies
            # (see form_abstractions — credit-by-use closes the farm, wall
            # redundant).
            # SHADOW instrument (2026-07-11): `has_role` edges are the
            # RoleRegularityShadow provisional role tags.  They give world-
            # state tokens a 2nd relation label, which would lift them to a
            # >= min_skeleton analogy skeleton and mint spurious `analogous_to`
            # edges across world states -- skip them as skeleton sources, the
            # same guard as ROLE_RELATION in form_abstractions above.
            if r == ROLE_RELATION:
                continue
            # Facet-docking (2026-07-21): `has_facet` gives world tokens a
            # 2nd relation label; skip it as a skeleton source too (same
            # guard as has_role) so it can never lift a state to an analogy
            # skeleton.
            if r == FACET_RELATION:
                continue
            rels_by_concept.setdefault(s, set()).add(r)
            targets_by_concept.setdefault(s, set()).add(t)
        # Bucket concepts by exact relation-skeleton signature,
        # only skeletons rich enough to count as "structure".
        buckets: Dict[frozenset, List[str]] = {}
        for c, rels in rels_by_concept.items():
            if len(rels) < min_skeleton:
                continue
            buckets.setdefault(frozenset(rels), []).append(c)
        # Richest skeletons first (more relation types = stronger
        # structural match), only buckets with >= 2 members.
        candidate_buckets = [
            (sig, members) for sig, members in buckets.items()
            if len(members) >= 2]
        candidate_buckets.sort(key=lambda item: -len(item[0]))
        new_created = 0
        for _sig, members in candidate_buckets:
            if new_created >= max_new:
                break
            members = sorted(set(members))[:ANALOGY_BUCKET_MEMBER_CAP]
            for i in range(len(members)):
                if new_created >= max_new:
                    break
                a = members[i]
                a_t = targets_by_concept.get(a, set())
                for j in range(i + 1, len(members)):
                    if new_created >= max_new:
                        break
                    b = members[j]
                    # Analogy requires DISJOINT content: same
                    # relational shape, no shared targets.  A
                    # shared target means overlapping content, not
                    # a cross-domain analogy.
                    if a_t & targets_by_concept.get(b, set()):
                        continue
                    src, tgt = (a, b) if a < b else (b, a)
                    key = (src, ANALOGY_RELATION, tgt)
                    existing = self.edges.get(key)
                    if existing is None:
                        # Self-cleaning F2 (2026-07-24): a faded analogy
                        # stays faded to this engine -- no self-restore.
                        if key in self.quarantine_edges:
                            continue
                        self.add_edge(
                            source=src, target=tgt,
                            relation_name=ANALOGY_RELATION,
                            strength=PROVISIONAL_EDGE_STRENGTH,
                            cycle=cycle)
                        new_created += 1
                    elif (existing.last_engaged_cycle
                            > existing.last_reinforced_cycle):
                        # Re-attest credits only on fresh external
                        # engagement (earned, never stamped).
                        existing.reinforce(cycle, origin='cognition')
        return new_created

    # ---- scoped (wake-local) consolidation helpers (2026-07-13) ----

    def _form_abstractions_scoped(self, cycle: int,
                                    candidates: Set[str], *,
                                    min_group: int,
                                    max_new: int) -> int:
        """Dirty-scoped abstraction formation.  See form_abstractions.
        Attaches the is_a edge ONLY to the dirty member joining a
        qualifying (relation, target) group; the co-group size is read
        O(1) from the (relation, target) index.  Rule-6 guards kept
        intact (is_a / role / abstraction-prefix skips)."""
        new_created = 0
        _scoped_budget = [0]   # is_a EDGES attached this pass
        for d in candidates:
            if (_ISABUDGET_ON()
                    and _scoped_budget[0] >= ABSTRACTION_MAX_EDGES_PER_PASS):
                break
            if d.startswith(ABSTRACTION_NAME_PREFIX):
                continue
            dc = self.concepts.get(d)
            if dc is None or not dc.edges_out:
                continue
            for r, bucket in list(dc.edges_out.items()):
                # Same grouping guards as the full scan: never group by
                # is_a (our own output), never by has_role (shadow tag),
                # never by has_facet (container relation, not a category).
                if (r == ABSTRACTION_RELATION or r == ROLE_RELATION
                        or r == FACET_RELATION):
                    continue
                for e in bucket:
                    t = self._name_of(e.target)
                    if not t or t.startswith(ABSTRACTION_NAME_PREFIX):
                        continue
                    # O(1) co-group: active, non-abstraction sources that
                    # share (r, t).  Filter to live edges so the size
                    # matches what the full scan would have grouped.
                    raw = self._relation_index.sources_for_rel_target(
                        r, t)
                    group = [s for s in raw
                             if not s.startswith(ABSTRACTION_NAME_PREFIX)
                             and (s, r, t) in self.edges]
                    if len(group) < min_group:
                        continue
                    abstraction_name = (
                        f"{ABSTRACTION_NAME_PREFIX}{r}_{t}")
                    if abstraction_name not in self.concepts:
                        if new_created >= max_new:
                            continue
                        c = Concept(
                            name=abstraction_name,
                            mi=MIValue.zero(),
                            created_cycle=cycle,
                            synthetic=True)
                        self.concepts[abstraction_name] = c
                        self._token_index.add(abstraction_name)
                        new_created += 1
                    if d == abstraction_name:
                        continue
                    key = (d, ABSTRACTION_RELATION, abstraction_name)
                    existing = self.edges.get(key)
                    if existing is None:
                        # Self-cleaning F2 (2026-07-24): a membership that
                        # FADED to quarantine stays faded to this engine.
                        # add_edge would auto-restore it (fresh engagement
                        # stamp -> staleness reset -> immortal oscillation);
                        # restore stays gated on genuinely cognition-
                        # EXTERNAL touches per _restore_quarantined's own
                        # contract (chemistry, AWM, intake).
                        if key in self.quarantine_edges:
                            continue
                        self.add_edge(
                            source=d,
                            target=abstraction_name,
                            relation_name=ABSTRACTION_RELATION,
                            strength=PROVISIONAL_EDGE_STRENGTH,
                            cycle=cycle)
                        _scoped_budget[0] += 1
                    elif (existing.last_engaged_cycle
                            > existing.last_reinforced_cycle):
                        # Re-attest credits ONLY on fresh EXTERNAL
                        # engagement since the last credit -- strength
                        # is earned by corroboration, never stamped by
                        # the engine's own pass over static structure
                        # (self-cleaning F2, 2026-07-24).
                        existing.reinforce(cycle, origin='cognition')
        return new_created

    def _structural_skeleton(self, name: str):
        """(frozenset(relation_types), set(targets)) for `name`, applying
        the SAME filters full-scan form_analogies uses: exclude
        analogous_to / has_role relations and abstraction-prefixed
        endpoints.  O(local degree).  An abstraction-prefixed source has
        an empty skeleton (excluded from analogy)."""
        c = self.concepts.get(name)
        if c is None or not c.edges_out:
            return frozenset(), set()
        if name.startswith(ABSTRACTION_NAME_PREFIX):
            return frozenset(), set()
        rels: Set[str] = set()
        targets: Set[str] = set()
        for r, bucket in c.edges_out.items():
            if (r == ANALOGY_RELATION or r == ROLE_RELATION
                    or r == FACET_RELATION):
                continue
            for e in bucket:
                t = self._name_of(e.target)
                if not t or t.startswith(ABSTRACTION_NAME_PREFIX):
                    continue
                rels.add(r)
                targets.add(t)
        return frozenset(rels), targets

    def _form_analogies_scoped(self, cycle: int,
                                 candidates: Set[str], *,
                                 min_skeleton: int,
                                 max_new: int) -> int:
        """Dirty-scoped analogy formation.  See form_analogies."""
        new_created = 0
        seen_pairs: Set[Tuple[str, str]] = set()
        for d in candidates:
            if new_created >= max_new:
                break
            if d.startswith(ABSTRACTION_NAME_PREFIX):
                continue
            d_skel, d_targets = self._structural_skeleton(d)
            if len(d_skel) < min_skeleton:
                continue
            # Partners must share d's EXACT skeleton, so they must have an
            # edge under EVERY relation in it — enumerate via the rarest
            # relation's bucket to bound the candidate set, then verify.
            r_min = min(
                d_skel,
                key=lambda rr: self._relation_index.bucket_size(rr))
            seen_partners: Set[str] = set()
            partners_checked = 0
            for e in self._relation_index.iter_edges(r_min):
                p = self._name_of(getattr(e, 'source', None))
                if not p or p == d or p in seen_partners:
                    continue
                seen_partners.add(p)
                if p.startswith(ABSTRACTION_NAME_PREFIX):
                    continue
                p_skel, p_targets = self._structural_skeleton(p)
                if p_skel != d_skel:
                    continue
                partners_checked += 1
                if partners_checked > ANALOGY_BUCKET_MEMBER_CAP:
                    break
                # Disjoint content (same shape, no shared targets).
                if d_targets & p_targets:
                    continue
                src, tgt = (d, p) if d < p else (p, d)
                pair = (src, tgt)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                key = (src, ANALOGY_RELATION, tgt)
                existing = self.edges.get(key)
                if existing is None:
                    # Self-cleaning F2 (2026-07-24): faded stays faded to
                    # this engine -- no self-restore.
                    if key in self.quarantine_edges:
                        continue
                    self.add_edge(
                        source=src, target=tgt,
                        relation_name=ANALOGY_RELATION,
                        strength=PROVISIONAL_EDGE_STRENGTH,
                        cycle=cycle)
                    new_created += 1
                    if new_created >= max_new:
                        break
                elif (existing.last_engaged_cycle
                        > existing.last_reinforced_cycle):
                    # Re-attest credits only on fresh external
                    # engagement (earned, never stamped).
                    existing.reinforce(cycle, origin='cognition')
        return new_created

    def defer_dirty(self, names: Iterable[str]) -> None:
        """Re-queue dirty candidates that a flood-guarded wake pass did
        NOT process this tick, so a deferred candidate keeps its earn-
        gate for the next pass (it is not lost)."""
        self._dirty_concepts |= set(names)

    def get_or_create_concept(self,
                              name: str,
                              mi: Optional[MIValue] = None,
                              cycle: int = 0
                              ) -> Concept:
        if name in self.concepts:
            return self.concepts[name]
        c = Concept(name=name, mi=mi or MIValue.zero(),
                    created_cycle=cycle)
        self.concepts[name] = c
        self._token_index.add(name)
        return c

    def get_or_create_relation(self,
                               name: str,
                               mi_prior: Optional[MIValue] = None
                               ) -> Relation:
        # mi_prior kept as a parameter for back-compat with callers
        # that pass it positionally, but it is now ignored per the
        # doctrine ("relations carry no innate M/I").
        if name in self.relations:
            return self.relations[name]
        r = Relation(name=name, extracted_from=0)
        self.relations[name] = r
        return r

    def add_edge(self,
                 source: str,
                 target: str,
                 relation_name: str,
                 strength: float = 0.5,
                 cycle: int = 0,
                 engage: bool = True
                 ) -> Edge:
        """Create or fetch an edge. Newly-created edges initialize
        their MIValue per §4.2.2.

        If the edge already exists (in active substrate), returns it
        (does not modify strength — caller must call edge.reinforce()
        explicitly).  If the edge is in QUARANTINE, the call is a
        re-attestation bid-path → restore the edge to active state,
        stamp last_engaged_cycle, return the restored edge.

        `engage` (2026-06-03, grounding world-loop): whether a NEWLY-
        created edge counts as freshly-engaged (stamps last_engaged_
        cycle).  Default True preserves the prior behavior — a valid
        import gets a grace window so it isn't quarantined before any
        bid-path touches it.  The world loop's MISS-writes pass
        engage=False: a transition that was merely OBSERVED (not
        predicted correctly) must NOT receive the engagement signal,
        so the ONLY survival signal for a world-transition edge is a
        correct prediction (write_reason='world_confirm', which stamps
        on the existing-edge path).  Doctrine: earn-by-predicting, not
        earn-by-being-seen.
        """
        key: EdgeKey = (source, relation_name, target)
        existing = self.edges.get(key)
        if existing is not None:
            return existing
        # Quarantine restoration: re-attestation is one of the four
        # cognitive bid-paths.  If a quarantined edge gets the same
        # triple written again from the outside (forager / chat),
        # that's the substrate seeing fresh evidence — restore.
        quarantined = self.quarantine_edges.get(key)
        if quarantined is not None:
            self._restore_quarantined(key, cycle=cycle)
            return self.edges[key]
        src = self.get_or_create_concept(source, cycle=cycle)
        tgt = self.get_or_create_concept(target, cycle=cycle)
        rel = self.get_or_create_relation(relation_name)
        edge = Edge.initialize(src, tgt, rel, strength=strength,
                               cycle=cycle)
        # New edge counts as freshly-engaged (it just entered via a
        # bid-path).  Without this stamp, brand-new edges would be
        # quarantine-eligible after L_QUARANTINE if nothing else
        # touched them — penalizing valid imports.  `engage=False`
        # (world-loop miss-writes) leaves it un-stamped so only a
        # correct prediction can grant survival.
        edge.last_engaged_cycle = int(cycle) if engage else 0
        self.edges[key] = edge
        # Index on source concept's edges_out.
        src.edges_out.setdefault(relation_name, []).append(edge)
        self._relation_index.add(edge)
        # Reverse is_a index (parent -> children) for O(siblings) sibling
        # lookup in grounding's class-generalization.
        if relation_name == ABSTRACTION_RELATION:
            self._is_a_children.setdefault(target, set()).add(source)
        # Scoped-consolidation: new structure entered the neighborhood
        # of both endpoints — re-evaluate their coherence next Phase S.
        self._dirty_concepts.add(source)
        self._dirty_concepts.add(target)
        return edge

    # ---- quarantine plumbing (substrate self-cleaning, tier 2) ----

    def _reap_edge(self, key: EdgeKey) -> bool:
        """DELETE an edge from the active substrate (edge-mortality).
        Removes it from `self.edges`, its source concept's edges_out,
        and the relation index.  Unlike `_quarantine_edge`, the edge is
        NOT moved to a retrievable pool — it is gone.  Used only by
        reap_stillborn_edges / sweep_stillborn_backlog on never-cohered
        stillborns (no earned existence to preserve).  Returns True on
        success."""
        edge = self.edges.get(key)
        if edge is None:
            return False
        src, rel, _tgt = key
        src_concept = self.concepts.get(src)
        if src_concept is not None:
            bucket = src_concept.edges_out.get(rel)
            if bucket is not None:
                try:
                    bucket.remove(edge)
                except ValueError:
                    pass
                if not bucket:
                    src_concept.edges_out.pop(rel, None)
        del self.edges[key]
        # Self-cleaning (2026-07-24): keep the reverse is_a index exact --
        # a reaped membership must not leave a phantom sibling behind.
        if rel == ABSTRACTION_RELATION:
            _kids = self._is_a_children.get(key[2])
            if _kids is not None:
                _kids.discard(src)
                if not _kids:
                    self._is_a_children.pop(key[2], None)
        try:
            self._relation_index.remove(edge)
        except Exception:
            pass
        return True

    def _quarantine_edge(self, key: EdgeKey) -> bool:
        """Move an edge from active substrate to the quarantine pool.
        Returns True on success.  Used by quarantine_inert_edges; not
        intended for direct caller use."""
        edge = self.edges.get(key)
        if edge is None:
            return False
        src, rel, tgt = key
        # Pull out of source.edges_out so inference / coherence /
        # iteration over neighbors no longer sees it.
        src_concept = self.concepts.get(src)
        if src_concept is not None:
            bucket = src_concept.edges_out.get(rel)
            if bucket is not None:
                try:
                    bucket.remove(edge)
                except ValueError:
                    pass
                if not bucket:
                    src_concept.edges_out.pop(rel, None)
        # Move from active edges to quarantine.
        del self.edges[key]
        # Self-cleaning (2026-07-24): a quarantined membership leaves the
        # reverse is_a index too -- grounding's sibling lookup must not see
        # phantom siblings from the quiescent pool (leak found in audit).
        if rel == ABSTRACTION_RELATION:
            _kids = self._is_a_children.get(tgt)
            if _kids is not None:
                _kids.discard(src)
                if not _kids:
                    self._is_a_children.pop(tgt, None)
        self.quarantine_edges[key] = edge
        # Reverse-index on BOTH endpoints so any bid-path that
        # touches either source or target can find the edge.
        self._quarantine_index.setdefault(src, []).append(key)
        if tgt != src:
            self._quarantine_index.setdefault(tgt, []).append(key)
        return True

    def _restore_quarantined(self, key: EdgeKey, *,
                              cycle: int = 0) -> bool:
        """Pull an edge back from quarantine into the active
        substrate.  Auto-called by add_edge (re-attestation),
        chemistry tag_attended (chemistry imprint), and AWM promotion
        (attentional bid).  Inference deliberately does NOT call this
        — quarantined edges are invisible to inference, restoration
        is gated on a cognition-external touch."""
        edge = self.quarantine_edges.pop(key, None)
        if edge is None:
            return False
        src, rel, tgt = key
        # Drop from reverse index on both endpoints.
        for endpoint in (src, tgt):
            bucket = self._quarantine_index.get(endpoint)
            if bucket is None:
                continue
            try:
                bucket.remove(key)
            except ValueError:
                pass
            if not bucket:
                self._quarantine_index.pop(endpoint, None)
        # Stamp engagement on restore (the bid-path just touched it).
        edge.last_engaged_cycle = int(cycle)
        # Re-attach in active substrate.  Strength preserved
        # (chemistry-never-fully-dissolves — it sits at floor and
        # must re-earn to grow).
        self.edges[key] = edge
        # Self-cleaning (2026-07-24): symmetric with the quarantine-side
        # removal -- a restored membership re-enters the reverse is_a index.
        if rel == ABSTRACTION_RELATION and tgt:
            self._is_a_children.setdefault(tgt, set()).add(src)
        src_concept = self.concepts.get(src)
        if src_concept is None:
            # Source concept was removed somehow — recreate so the
            # edges_out index stays consistent.  Rare.
            src_concept = self.get_or_create_concept(src, cycle=cycle)
        src_concept.edges_out.setdefault(rel, []).append(edge)
        # Restored structure re-enters the neighborhood — re-evaluate
        # coherence next Phase S (scoped-consolidation dirty set).
        self._dirty_concepts.add(src)
        self._dirty_concepts.add(tgt)
        return True

    def restore_concept_quarantine(self,
                                     concept_name: str,
                                     cycle: int = 0) -> int:
        """Restore ALL quarantined edges touching `concept_name`.
        Used by AWM promotion (the concept entering attention) and
        by chemistry imprint (the focal's bubble firing).  Returns
        the count restored.  O(degree-in-quarantine), not O(N)."""
        keys = list(self._quarantine_index.get(concept_name, ()))
        n = 0
        for key in keys:
            if self._restore_quarantined(key, cycle=cycle):
                n += 1
        return n

    def coherent_fraction_of(self,
                                keys: Iterable[EdgeKey],
                                cycle: int = 0) -> Dict[str, Any]:
        """Read-only instrumentation (reinforce-on-recall igniter,
        2026-05-31).  Of the given edge keys, what fraction have
        EVER cohered (first_coherent_cycle > 0)?

        This is the make-or-break number: if recall-reattested edges
        trend toward coherence, dry-reverie introspection produces
        earn-surviving growth (the flywheel turns).  If it stays ~0
        while writes/debt climb, the igniter is converting a dead
        substrate into a dying one — the honest thrashing-toward-
        death finding, which is the signal that goal-seeding (Step 3)
        is needed, not a bug to patch.

        Pure read: no mutation, no feedback into cognition.  Looks in
        BOTH the active and quarantined pools so a key that has
        drifted into quiescence still counts.  `cycle` is reserved
        (coherence is historical, not cycle-relative) and accepted
        for signature stability.  Returns {n, n_coherent, fraction}.
        """
        n = 0
        n_coherent = 0
        for key in keys:
            edge = self.edges.get(key)
            if edge is None:
                edge = self.quarantine_edges.get(key)
            if edge is None:
                continue
            n += 1
            if int(getattr(edge, 'first_coherent_cycle', 0)) > 0:
                n_coherent += 1
        return {
            'n': n,
            'n_coherent': n_coherent,
            'fraction': (n_coherent / n) if n else 0.0,
        }

    def add_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > self.EPISODES_MAX:
            # Drop oldest. Note: PersistentIdentity preserves
            # high-priority defining_moments separately, so
            # cosmologically-load-bearing events survive.
            overflow = len(self.episodes) - self.EPISODES_MAX
            self.episodes = self.episodes[overflow:]

    def add_prediction(self, prediction: Prediction) -> None:
        self.predictions.append(prediction)

    def get_or_create_scene(self,
                             propositions: Iterable[EdgeKey],
                             cycle: int = 0
                             ) -> Scene:
        """Look up a scene by exact proposition set; create if not
        present. Identity is by content (frozenset of EdgeKeys), so
        the same proposition set always returns the same Scene."""
        key = frozenset(propositions)
        if not key:
            raise ValueError("Cannot create scene from empty propositions")
        existing_id = self._scene_index.get(key)
        if existing_id is not None:
            return self.scenes[existing_id]
        new_id = f"scene_{len(self.scenes):04d}"
        new_scene = Scene(id=new_id, propositions=key,
                           created_cycle=cycle)
        self.scenes[new_id] = new_scene
        self._scene_index[key] = new_id
        return new_scene

    # ---- query API ----

    def query(self, pattern: QueryPattern) -> QueryResult:
        """Single substrate-query primitive. Dispatches on pattern
        type; returns a QueryResult with the relevant fields populated.
        """
        if isinstance(pattern, ConceptByName):
            return QueryResult(concept=self.concepts.get(pattern.name))

        if isinstance(pattern, EdgesQuery):
            matches: List[Edge] = []
            # Pick the smallest candidate set:
            #   - source given: use concept.edges_out (scoped)
            #   - relation given (no source): use relation index
            #   - else: full scan
            if pattern.source is not None:
                src = self.concepts.get(pattern.source)
                if src is None:
                    return QueryResult(edges=[])
                if pattern.relation is not None:
                    candidates = src.edges_out.get(
                        pattern.relation, [])
                else:
                    candidates = [e for elist
                                   in src.edges_out.values()
                                   for e in elist]
            elif pattern.relation is not None:
                candidates = self._relation_index.get(pattern.relation)
            else:
                candidates = list(self.edges.values())
            for edge in candidates:
                if (pattern.target is not None
                        and edge.target != pattern.target):
                    continue
                if (pattern.relation is not None
                        and edge.relation_name != pattern.relation):
                    continue
                matches.append(edge)
            return QueryResult(edges=matches)

        if isinstance(pattern, NeighborhoodQuery):
            return self._neighborhood(pattern)

        if isinstance(pattern, ConceptsByMI):
            matches: List[Concept] = []
            for c in self.concepts.values():
                v = c.mi.m if pattern.channel == 'm' else c.mi.i
                if v >= pattern.threshold and c.mi.n >= pattern.evidence_min:
                    matches.append(c)
            return QueryResult(concepts=matches)

        if isinstance(pattern, ScenesByProposition):
            matches: List[Scene] = [
                s for s in self.scenes.values()
                if pattern.proposition in s.propositions
            ]
            return QueryResult(scenes=matches)

        if isinstance(pattern, ScenesByMI):
            matches: List[Scene] = []
            for s in self.scenes.values():
                v = s.mi.m if pattern.channel == 'm' else s.mi.i
                if v >= pattern.threshold:
                    matches.append(s)
            return QueryResult(scenes=matches)

        raise TypeError(f"Unknown QueryPattern: {type(pattern).__name__}")

    def _neighborhood(self, q: NeighborhoodQuery) -> QueryResult:
        """BFS up to max_depth, filtering by relation and strength."""
        seed_concept = self.concepts.get(q.seed)
        if seed_concept is None:
            return QueryResult()
        visited: Set[str] = {q.seed}
        frontier: List[str] = [q.seed]
        traversed: List[Edge] = []
        out_concepts: List[Concept] = [seed_concept]

        for _depth in range(q.max_depth):
            next_frontier: List[str] = []
            for cur_name in frontier:
                cur = self.concepts.get(cur_name)
                if cur is None:
                    continue
                for tgt_name, edge in cur.neighbors(q.relation_filter):
                    if edge.strength < q.min_strength:
                        continue
                    traversed.append(edge)
                    if tgt_name not in visited:
                        visited.add(tgt_name)
                        tgt = self.concepts.get(tgt_name)
                        if tgt is not None:
                            out_concepts.append(tgt)
                            next_frontier.append(tgt_name)
            frontier = next_frontier
            if not frontier:
                break
        return QueryResult(concepts=out_concepts,
                           traversed_edges=traversed)

    # ---- semantic attention ----

    def attend_to_text(self,
                        text: str,
                        top_k: int = 5,
                        threshold: float = 0.3,
                        rebuild_threshold: int = 50,
                        ) -> List[Tuple[str, float]]:
        """Rank substrate concepts by semantic similarity to `text`.

        Uses the MiniLM bridge (or n-gram fallback) to compute the
        query embedding, then cosine against each concept's cached
        embedding (computed lazily on first access). Concept
        embeddings are rebuilt when the substrate has grown by more
        than `rebuild_threshold` since last build.

        Returns top_k (concept_name, similarity) pairs above
        `threshold`, sorted descending. Empty list if substrate
        is empty or text is empty.
        """
        if not text or not self.concepts:
            return []
        # Local imports to avoid circular dependency at module load.
        from seagi.core.mi_embedding import encode_text_with_mi, encode_concept_with_mi
        from seagi.core.minilm_backend import cosine

        query_emb = encode_text_with_mi(text)

        # LSH fast-path: first ensure all concepts are indexed under
        # the query's embedding dim. New / mis-dim concepts are
        # (re)embedded and added. Then collect candidates from LSH
        # buckets and rescore exactly. Falls back to full scan when
        # the candidate set is empty (fresh substrate before any
        # embeddings are populated).
        for name, concept in self.concepts.items():
            if (concept.embedding is None
                    or concept.embedding.dim != query_emb.dim):
                concept.embedding = encode_concept_with_mi(concept)
                self._lsh_index.add(name, concept.embedding.semantic)
            elif name not in self._lsh_index._name_to_sig:
                self._lsh_index.add(name, concept.embedding.semantic)

        candidate_names = self._lsh_index.candidates(query_emb.semantic)
        if candidate_names:
            iterable = (
                (n, self.concepts[n]) for n in candidate_names
                if n in self.concepts)
        else:
            # No buckets matched (very rare query orientation, or
            # empty index). Fall back to exact scan.
            iterable = self.concepts.items()

        scored: List[Tuple[str, float]] = []
        for name, concept in iterable:
            if concept.embedding is None:
                continue
            sim = cosine(query_emb.semantic,
                          concept.embedding.semantic)
            if sim >= threshold:
                scored.append((name, sim))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]

    # ---- token-index lookup ----

    def lookup_concepts_by_text(self,
                                  text: str,
                                  max_results: int = 1000
                                  ) -> Set[str]:
        """Return the set of concept names whose name shares a
        token with `text`. O(tokens × matches) via the inverted
        index, instead of O(N) over all concepts."""
        return self._token_index.find_by_text(
            text, max_results=max_results)

    # ---- introspection / stats ----

    def stats(self) -> Dict[str, int]:
        return {
            'n_concepts': len(self.concepts),
            'n_edges': len(self.edges),
            'n_relations': len(self.relations),
            'n_predictions': len(self.predictions),
            'n_episodes': len(self.episodes),
            # HIS OWN RECORD, readable.  A trace nobody can read is the
            # same failure as a trace nobody writes.
            'recent_episodes': [
                {
                    'cycle': e.cycle,
                    'kind': e.kind,
                    'content': e.content,
                    'm': round(float(getattr(e.mi_at_event, 'm', 0.0)), 4),
                    'i': round(float(getattr(e.mi_at_event, 'i', 0.0)), 4),
                    'action': e.action_taken,
                    'outcome': e.observed_outcome,
                }
                for e in self.episodes[-12:]
            ],
            'n_scenes': len(self.scenes),
        }

    # ---- persistence ----

    def to_dict(self, lazy: bool = False) -> dict:
        """Serialise the substrate.

        `lazy=True` leaves the Edge and Concept OBJECTS in place of their
        dicts, for a streaming encoder -- `json.dump` with a `default`
        that calls `to_dict` on each as it reaches it.  Each dict is then
        built and dropped one at a time instead of 2.25M of them being
        held at once.

        Measured 2026-08-29 on his own edges: materialising costs
        **1,010 B/edge (2.00 GB at 1,975,014 edges)** and streaming costs
        **0** -- the same pages are reused.  That transient was the whole
        of the save spike, which took RSS from 3,541,700 kB to 5,911,144
        on a 7.75 GB box.  Default stays False so every other caller is
        byte-identical.
        """
        return {
            'concepts': (dict(self.concepts) if lazy else
                         {name: c.to_dict()
                          for name, c in self.concepts.items()}),
            'edges': (list(self.edges.values()) if lazy else
                      [e.to_dict() for e in self.edges.values()]),
            'quarantine_edges': (
                list(self.quarantine_edges.values()) if lazy else
                [e.to_dict() for e in self.quarantine_edges.values()]),
            'quarantine_migrated': bool(self._quarantine_migrated),
            'migration_cursor': int(self._migration_cursor),
            'relations': {name: r.to_dict()
                           for name, r in self.relations.items()},
            'predictions': [p.to_dict() for p in self.predictions],
            'episodes': [e.to_dict() for e in self.episodes],
            'scenes': {sid: s.to_dict()
                        for sid, s in self.scenes.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Substrate':
        scenes = {sid: Scene.from_dict(sd)
                   for sid, sd in d.get('scenes', {}).items()}
        sub = cls(
            # May already be a Concept: the daemon's load passes an
            # object_hook that converts concept-shaped dicts DURING the
            # parse, so the raw dict is never retained (2026-08-29 --
            # bulk 6,142 B/concept against 3,827 hooked, 0.6 GB at his
            # scale).  Accept both.
            concepts={name: (cd if isinstance(cd, Concept)
                             else Concept.from_dict(cd))
                       for name, cd in d.get('concepts', {}).items()},
            relations={name: Relation.from_dict(rd)
                       for name, rd in d.get('relations', {}).items()},
            predictions=[Prediction.from_dict(p)
                         for p in d.get('predictions', [])],
            episodes=[Episode.from_dict(e)
                      for e in d.get('episodes', [])],
            scenes=scenes,
            seed_relations=False,
        )
        # Edges need to be re-attached to source concepts' edges_out.
        for ed in d.get('edges', []):
            # May already be an Edge: the daemon's load passes an
            # object_hook that converts edge-shaped dicts DURING the
            # parse, so the raw dict is never retained (2026-08-28 --
            # bulk parsing cost 3,177 B/edge against 951 streamed,
            # which is 4.3 GB at his scale).  Accept both.
            edge = ed if isinstance(ed, Edge) else Edge.from_dict(ed)
            sub.edges[edge.key] = edge
            # Repopulate the relation index too — from_dict bypasses
            # add_edge's incremental maintenance, and unlike the is_a
            # index this was never rebuilt on load (left empty), so a
            # relation-scoped EdgesQuery silently returned [].
            sub._relation_index.add(edge)
            src = sub.concepts.get(edge.source)
            if src is not None:
                src.edges_out.setdefault(
                    edge.relation_name, []).append(edge)
        # Reverse is_a index — rebuilt from the just-attached edges (from_dict
        # bypasses add_edge's incremental maintenance).
        sub._rebuild_is_a_index()
        # Quarantined edges: NOT attached to source concepts'
        # edges_out (that's the whole point — they're invisible to
        # inference/coherence iteration).  Rebuild reverse index.
        for ed in d.get('quarantine_edges', []):
            edge = ed if isinstance(ed, Edge) else Edge.from_dict(ed)
            sub.quarantine_edges[edge.key] = edge
            sub._quarantine_index.setdefault(
                edge.source, []).append(edge.key)
            if edge.target != edge.source:
                sub._quarantine_index.setdefault(
                    edge.target, []).append(edge.key)
        sub._quarantine_migrated = bool(
            d.get('quarantine_migrated', False))
        try:
            sub._migration_cursor = int(d.get('migration_cursor', 0))
        except (TypeError, ValueError):
            sub._migration_cursor = 0
        return sub
