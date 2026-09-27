"""Personality — the gestalt integrator above M/I-tagged memory.

Phase H.2 (2026-05-17).  Builds on H.1 persistence.

What this module is
-------------------
The PERSONALITY layer.  Not a registry of facts — an INTEGRATIVE
read across the M/I-tagged brain state that produces a coherent
multi-dimensional SHAPE.

Per doctrine (2026-05-17, user directive):

    "personality is a meta level of experience and concepts on the
     base of M/I nt tags.  it hovers and exists like a complete
     puzzle of all experiences formed over time.  it is separate
     from memory but at the same time connected to memory.
     personality is the sum and more than the sum of all
     experiences."

What this module is NOT
-----------------------
- Not stored as a copy of registry contents.  It is COMPUTED on
  demand from the persistent state.
- Not the source of facts.  Substrate + registries hold facts.
- Not the bias function (that is H.3, the consumer wiring).
  Personality SUPPLIES the gestalt; the puzzle-fit bias reads
  from it.

What we compute
---------------
Each `PersonalitySignature` carries:

  Single-dimensional reads (from each registry):
    - tone_label / tone_dominant_channels   (chemistry)
    - self_values / self_fears / self_categories / self_attention
                                              (identity)
    - anchor_concepts                          (top-crystallized
                                                substrate bubbles)
    - preferred_actions / avoided_actions    (action_credit
                                                top / bottom)
    - skill_verbs                              (reliable skills)
    - discovered_laws                          (top-confidence laws)
    - dominant_schemas                         (top-support schemas)
    - wondering_pairs                          (confirmed hypotheses)

  Cross-registry emergence ("more than the sum"):
    - themes — concepts appearing in ≥3 M/I-tagged contexts
                  (high-cryst bubble + identity edge + active
                  goal focal + reward-credit context)

Refresh dynamics
----------------
Personality has INERTIA.  `maybe_refresh()` only re-derives the
signature every N cycles (default 500, ~10x reflection interval).
This makes single experiences NOT shift the gestalt; sustained
patterns of fitting evidence over time do.

A signature persists between refreshes — callers see the same
shape until enough cycles have elapsed for re-derivation.

Doctrine alignment
------------------
- Reads the M/I-weighted state.  Anything below the persistence
  floor (H.1) was already not personality-bearing; here we go
  further and pick the TOP items per dimension to form the
  shape.
- "More than the sum" lives in the cross-registry theme detection
  — a concept is a personality theme only when it has converged
  in MULTIPLE M/I contexts.
- Slow-fire dynamic: refresh on a cycle interval, not on every
  event.  Low energy / fire when needed.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..events import (
    EventKind, BrainEvent, AttendedPerceptEvent, ChemistryEvent,
)
from ..bus import EventBus


# How often the gestalt re-derives.  500 cycles is ~10x the idle
# reflection interval — slow enough for inertia, fast enough to
# respond to sustained change.
DEFAULT_REFRESH_INTERVAL_CYCLES = 500

# How many items per signature dimension.  Top-K only — personality
# is a small coherent shape, not an exhaustive list.
DEFAULT_TOP_K = 5

# Number of M/I contexts a concept must appear in to count as a
# cross-registry theme ("more than the sum" threshold).  At 3, a
# concept that has high crystallization + a self-edge + an active
# goal becomes a theme.
DEFAULT_THEME_CONVERGENCE = 3

# Phase H.3 (2026-05-17): coherence scoring.  Per-dimension weights
# determine how much each signature membership contributes to the
# coherence score of an incoming focal.  Themes weigh most (they
# ARE the integrated convergence); anchors / categories / values
# weigh somewhat; self_fears contribute NEGATIVELY (fear-aligned
# input fires puzzle-stress).
COHERENCE_WEIGHT_THEME = 1.0
COHERENCE_WEIGHT_ANCHOR = 0.7
COHERENCE_WEIGHT_CATEGORY = 0.7
COHERENCE_WEIGHT_VALUE = 0.5
COHERENCE_WEIGHT_FEAR = -0.7
COHERENCE_WEIGHT_ATTENTION = 0.4

# Threshold above which coherence fires puzzle_fit, and below
# which it fires puzzle_stress.  Between the two, the percept is
# personality-neutral and nothing fires (low energy / fire when
# needed).
COHERENCE_FIT_THRESHOLD = 0.4
COHERENCE_STRESS_THRESHOLD = -0.4

# Chemistry-event magnitude scaling — the firing magnitude is
# `abs(coherence)` × this constant.  Keeps the puzzle-fit/stress
# events at promille scale even when coherence saturates.  See
# EVENT_DELTAS['puzzle_fit'] and ['puzzle_stress'] in chemistry.py
# for the channel multipliers.
COHERENCE_FIRE_SCALE = 0.5


@dataclass
class PersonalitySignature:
    """One snapshot of the gestalt at a moment in time.

    A signature is a READOUT, not the personality itself.  The
    personality is the integrated SHAPE; the signature reports it.
    """
    cycle: int = 0
    # ---- Tone / disposition ----
    tone_label: str = ''
    tone_valence: float = 0.0
    tone_arousal: float = 0.0
    tone_warmth: float = 0.0
    tone_dominant: str = ''
    # ---- Identity themes (from SelfModel) ----
    self_values: List[str] = field(default_factory=list)
    self_fears: List[str] = field(default_factory=list)
    self_categories: List[str] = field(default_factory=list)
    self_attention: List[str] = field(default_factory=list)
    # ---- Attentional anchors (top-crystallized substrate bubbles) ----
    anchor_concepts: List[str] = field(default_factory=list)
    # ---- Behavioral policy (action_credit signature) ----
    preferred_actions: List[str] = field(default_factory=list)
    avoided_actions: List[str] = field(default_factory=list)
    # ---- Reliable skills ----
    skill_verbs: List[str] = field(default_factory=list)
    # ---- Self-discovered laws ----
    discovered_laws: List[str] = field(default_factory=list)
    # ---- Schema awareness ----
    dominant_schemas: List[str] = field(default_factory=list)
    # ---- Wondering style (confirmed hypotheses) ----
    wondering_pairs: List[Tuple[str, str]] = field(default_factory=list)
    # ---- EMERGENT cross-registry themes ----
    # Concepts that appear in >= DEFAULT_THEME_CONVERGENCE M/I
    # contexts.  This is the "more than the sum" dimension.
    themes: List[str] = field(default_factory=list)
    # ---- Diagnostic: how much state was integrated ----
    inputs_total: int = 0


class Personality:
    """The gestalt-integrator capability.

    Reads across substrate + 6 registries + chemistry on a slow
    refresh interval.  Provides:
      - signature()                — current gestalt snapshot
      - maybe_refresh()            — interval-gated re-derivation
      - refresh_now()              — force re-derive
      - render_self_description()  — first-person "I am someone who..."
      - coherence(focal)           — Phase H.3: scalar [-1,1] fit
      - handle(event, bus)         — Phase H.3: fires puzzle_fit /
                                       puzzle_stress chemistry on
                                       non-neutral attended percepts
    """

    # Phase H.3: subscribe to ATTENDED_PERCEPT so the gestalt
    # can fire its bias chemistry as new experience comes in.
    SUBSCRIPTIONS = (
        EventKind.ATTENDED_PERCEPT,
    )

    def __init__(self,
                 chemistry_provider: Optional[Callable] = None,
                 identity_provider: Optional[Callable] = None,
                 lts_provider: Optional[Callable] = None,
                 reward_ledger_provider: Optional[Callable] = None,
                 skill_library_provider: Optional[Callable] = None,
                 symbolic_regressor_provider: Optional[Callable] = None,
                 creativity_provider: Optional[Callable] = None,
                 schema_library_provider: Optional[Callable] = None,
                 goal_tracker_provider: Optional[Callable] = None,
                 cycle_provider: Optional[Callable] = None,
                 refresh_interval_cycles: int = (
                     DEFAULT_REFRESH_INTERVAL_CYCLES),
                 top_k: int = DEFAULT_TOP_K):
        self._chemistry_provider = chemistry_provider
        self._identity_provider = identity_provider
        self._lts_provider = lts_provider
        self._reward_ledger_provider = reward_ledger_provider
        self._skill_library_provider = skill_library_provider
        self._symbolic_regressor_provider = (
            symbolic_regressor_provider)
        self._creativity_provider = creativity_provider
        self._schema_library_provider = schema_library_provider
        self._goal_tracker_provider = goal_tracker_provider
        self._cycle_provider = cycle_provider or (lambda: 0)
        self.refresh_interval = int(refresh_interval_cycles)
        self.top_k = int(top_k)
        # Current cached signature.  Starts empty.
        self._signature = PersonalitySignature()
        # When the last refresh ran.  Far in the past so first
        # call refreshes.
        self._last_refresh_cycle: int = -10**6
        # Diagnostics.
        self.refreshes: int = 0
        # Phase H.3: per-event counters.
        self.fits_fired: int = 0
        self.stresses_fired: int = 0
        self.neutral_percepts: int = 0

    # ---- public API ----

    def signature(self) -> PersonalitySignature:
        """Current cached gestalt.  Stable between refreshes —
        inertia."""
        return self._signature

    def maybe_refresh(self) -> bool:
        """Re-derive the signature if enough cycles have elapsed.
        Returns True if a refresh happened, False otherwise."""
        cycle = self._cycle_provider()
        if cycle - self._last_refresh_cycle < self.refresh_interval:
            return False
        self.refresh_now()
        return True

    def refresh_now(self) -> PersonalitySignature:
        """Force a refresh.  Always re-computes.  Returns the new
        signature.  Also marks last_refresh_cycle so a subsequent
        maybe_refresh respects the interval — manual refreshes
        are still "refreshes" for inertia purposes."""
        cycle = self._cycle_provider()
        self._last_refresh_cycle = cycle
        sig = PersonalitySignature(cycle=cycle)
        self._compute_tone(sig)
        self._compute_identity(sig)
        self._compute_anchors(sig)
        self._compute_policy(sig)
        self._compute_skills(sig)
        self._compute_laws(sig)
        self._compute_schemas(sig)
        self._compute_wonderings(sig)
        self._compute_themes(sig)
        self._signature = sig
        self.refreshes += 1
        return sig

    # ---- per-dimension computation ----

    def _compute_tone(self, sig: PersonalitySignature) -> None:
        chem = self._safe_get(self._chemistry_provider)
        if chem is None:
            return
        try:
            tone = chem.tone_summary()
        except Exception:
            return
        sig.tone_label = str(tone.get('label', ''))
        sig.tone_valence = float(tone.get('valence', 0.0))
        sig.tone_arousal = float(tone.get('arousal', 0.0))
        sig.tone_warmth = float(tone.get('warmth', 0.0))
        sig.tone_dominant = str(tone.get('dominant', ''))

    def _compute_identity(self,
                              sig: PersonalitySignature) -> None:
        idn = self._safe_get(self._identity_provider)
        if idn is None or len(idn) == 0:
            return
        # Bucket by relation.
        for rel, target_list in (
                ('value', sig.self_values),
                ('fear', sig.self_fears),
                ('attend_to', sig.self_attention),
                ('anchor_on', sig.self_attention)):
            edges = sorted(
                idn.edges_by_relation(rel),
                key=lambda e: (-e.crystallization, -e.strength))
            for e in edges[:self.top_k]:
                if e.object and e.object not in target_list:
                    target_list.append(e.object)
        # Identity claims ('is_a' / 'has_property' / 'is').
        for rel in ('is_a', 'has_property', 'is'):
            edges = sorted(
                idn.edges_by_relation(rel),
                key=lambda e: (-e.crystallization, -e.strength))
            for e in edges[:self.top_k]:
                if e.object and e.object not in sig.self_categories:
                    sig.self_categories.append(e.object)
        sig.inputs_total += len(idn)

    def _compute_anchors(self,
                            sig: PersonalitySignature) -> None:
        """Top-crystallized substrate concepts — the bubbles that
        have imprinted deepest.  These are what this Seagi has
        accumulated the most felt-weight on."""
        lts = self._safe_get(self._lts_provider)
        if lts is None:
            return
        substrate = getattr(lts, 'substrate', None)
        if substrate is None:
            return
        concepts = getattr(substrate, 'concepts', None) or {}
        if not concepts:
            return
        # Score each concept by the maximum crystallization across
        # its bubbles.  Concepts with no bubbles score 0.
        scored: List[Tuple[float, str]] = []
        for name, concept in concepts.items():
            bubbles = getattr(concept, 'bubbles', []) or []
            if not bubbles:
                continue
            max_cryst = max(
                float(getattr(b, 'crystallization', 0.0))
                for b in bubbles)
            if max_cryst <= 0.0:
                continue
            scored.append((max_cryst, name))
        scored.sort(reverse=True)
        sig.anchor_concepts = [n for _c, n in scored[:self.top_k]]
        sig.inputs_total += len(scored)

    def _compute_policy(self,
                            sig: PersonalitySignature) -> None:
        """Read action_credit and surface highest-positive
        (preferred) and lowest-negative (avoided) actions."""
        led = self._safe_get(self._reward_ledger_provider)
        if led is None:
            return
        # Aggregate credit across buckets per action (cross-context
        # "in general, does this tend to work" signal).
        agg: Dict[str, float] = defaultdict(float)
        for (action, _bucket), credit in led._action_credit.items():
            agg[action] += float(credit)
        if not agg:
            return
        ranked = sorted(agg.items(), key=lambda kv: -kv[1])
        sig.preferred_actions = [
            a for a, c in ranked[:self.top_k] if c > 0.0]
        sig.avoided_actions = [
            a for a, c in ranked[-self.top_k:] if c < 0.0]
        sig.inputs_total += len(agg)

    def _compute_skills(self,
                            sig: PersonalitySignature) -> None:
        lib = self._safe_get(self._skill_library_provider)
        if lib is None:
            return
        # Top reliable skills — extract their leading verbs.
        skills = list(lib.all_skills())
        skills.sort(key=lambda s: -s.reliability())
        seen = set()
        for s in skills[:self.top_k]:
            if not s.action_sequence:
                continue
            first = s.action_sequence[0]
            verb = first.split(':', 1)[0]
            if verb and verb not in seen:
                seen.add(verb)
                sig.skill_verbs.append(verb)
        sig.inputs_total += len(skills)

    def _compute_laws(self, sig: PersonalitySignature) -> None:
        reg = self._safe_get(self._symbolic_regressor_provider)
        if reg is None or len(reg) == 0:
            return
        for law in reg.top_laws(self.top_k):
            sig.discovered_laws.append(law.describe())
        sig.inputs_total += len(reg)

    def _compute_schemas(self,
                              sig: PersonalitySignature) -> None:
        lib = self._safe_get(self._schema_library_provider)
        if lib is None or len(lib) == 0:
            return
        for s in lib.top_schemas(self.top_k):
            sig.dominant_schemas.append(s.kind)
        sig.inputs_total += len(lib)

    def _compute_wonderings(self,
                                sig: PersonalitySignature) -> None:
        daemon = self._safe_get(self._creativity_provider)
        if daemon is None or len(daemon) == 0:
            return
        # Only confirmed wonderings — untested ones aren't
        # personality.
        confirmed = [h for h in daemon.recent(n=self.top_k * 2)
                     if h.confirmations > 0]
        for h in confirmed[:self.top_k]:
            sig.wondering_pairs.append((h.concept_a, h.concept_b))
        sig.inputs_total += len(daemon)

    # ---- EMERGENT cross-registry themes ----

    def _compute_themes(self,
                            sig: PersonalitySignature) -> None:
        """The "more than the sum" dimension.

        A concept becomes a THEME when it appears in at least
        DEFAULT_THEME_CONVERGENCE distinct M/I-tagged contexts:

          1. Anchor — among top-crystallized substrate concepts
          2. Identity — appears as object in any high-cryst
                          self-edge (already filtered by H.1
                          to be personality-bearing)
          3. Goal — focal of an active high-urgency goal
          4. Wondering — appears in a confirmed creative
                          hypothesis (concept_a or concept_b)
          5. Skill-context — appears as the focal portion of any
                          action_credit key (action:focal shape)

        Themes are what this agent IS ABOUT in the round —
        concepts that have converged across reflection, identity,
        intention, and curiosity.
        """
        contexts: Dict[str, int] = defaultdict(int)

        # 1. Anchors (already computed).
        for name in sig.anchor_concepts:
            contexts[name] += 1

        # 2. Identity objects.
        idn = self._safe_get(self._identity_provider)
        if idn is not None:
            counted = set()
            for e in idn.all_edges():
                if e.object and e.object not in counted:
                    counted.add(e.object)
                    contexts[e.object] += 1

        # 3. Active goal focals.
        goals = self._safe_get(self._goal_tracker_provider)
        if goals is not None:
            counted = set()
            for g in goals.active():
                if g.focal and g.focal not in counted:
                    counted.add(g.focal)
                    contexts[g.focal] += 1

        # 4. Confirmed wondering concepts.
        daemon = self._safe_get(self._creativity_provider)
        if daemon is not None:
            counted = set()
            for h in daemon.recent(n=20):
                if h.confirmations <= 0:
                    continue
                for c in (h.concept_a, h.concept_b):
                    if c and c not in counted:
                        counted.add(c)
                        contexts[c] += 1

        # 5. Skill-context focals (action_credit keys of shape
        # 'verb:focal' don't exist — action_kind is just the verb.
        # Instead, count concepts that are focal of skill action
        # sequences ending in ':focal').  In practice the bucket
        # tuple in action_credit doesn't carry the focal directly,
        # so we look at skills' action sequences.
        lib = self._safe_get(self._skill_library_provider)
        if lib is not None:
            counted = set()
            for s in lib.all_skills():
                if not s.action_sequence:
                    continue
                for action in s.action_sequence:
                    if ':' in action:
                        focal = action.split(':', 1)[1]
                        if focal and focal not in counted:
                            counted.add(focal)
                            contexts[focal] += 1

        # Promote concepts that crossed the convergence floor.
        themes: List[Tuple[int, str]] = []
        for name, count in contexts.items():
            if count >= DEFAULT_THEME_CONVERGENCE:
                themes.append((count, name))
        themes.sort(reverse=True)
        sig.themes = [n for _c, n in themes[:self.top_k * 2]]

    # ---- rendering ----

    def render_self_description(self) -> str:
        """First-person multi-clause description of the gestalt.

        Returns a coherent string like:
          'My disposition is calm.  I value patience, wisdom.  I am
           curious.  I think most about caesar, virtue.  I tend to
           reflect.  I have noticed I notice cortisol rises with
           norepinephrine.'

        Empty signature → empty string (no facts yet).
        """
        sig = self._signature
        clauses: List[str] = []

        if sig.tone_label:
            clauses.append(f'My disposition is {sig.tone_label}')

        if sig.self_categories:
            cats = ', '.join(sig.self_categories[:3])
            clauses.append(f'I am {cats}')

        if sig.self_values:
            vals = ', '.join(sig.self_values[:3])
            clauses.append(f'I value {vals}')

        if sig.self_fears:
            fears = ', '.join(sig.self_fears[:2])
            clauses.append(f'I fear {fears}')

        if sig.themes:
            ths = ', '.join(sig.themes[:3])
            clauses.append(f'I am drawn to {ths}')
        elif sig.anchor_concepts:
            anchors = ', '.join(sig.anchor_concepts[:3])
            clauses.append(f'I think most about {anchors}')

        if sig.preferred_actions:
            acts = ', '.join(sig.preferred_actions[:2])
            clauses.append(f'I tend to {acts}')

        if sig.discovered_laws:
            # First law only — keep it tight.
            clauses.append(f'I have noticed {sig.discovered_laws[0]}')

        if not clauses:
            return ''
        return '. '.join(clauses) + '.'

    # ---- Phase H.3: coherence + active gestalt feedback ----

    def coherence(self, focal: str) -> float:
        """Score how well `focal` fits the current personality
        gestalt.  Returns a value in [-1, 1]:

           > 0    coherent — focal aligns with positive identity /
                  themes / anchors / values.
           = 0    neutral — focal isn't in any signature dimension.
           < 0    incoherent — focal aligns with self_fears (the
                  agent's negative-pattern recognition); future
                  versions may also detect substrate-opposite
                  contradictions to identity claims.

        Cheap O(1) per dimension via set membership.  Safe to call
        on every attended percept.
        """
        if not focal:
            return 0.0
        sig = self._signature
        score = 0.0
        if focal in sig.themes:
            score += COHERENCE_WEIGHT_THEME
        if focal in sig.anchor_concepts:
            score += COHERENCE_WEIGHT_ANCHOR
        if focal in sig.self_categories:
            score += COHERENCE_WEIGHT_CATEGORY
        if focal in sig.self_values:
            score += COHERENCE_WEIGHT_VALUE
        if focal in sig.self_attention:
            score += COHERENCE_WEIGHT_ATTENTION
        if focal in sig.self_fears:
            score += COHERENCE_WEIGHT_FEAR
        # Clamp to [-1, 1] so callers can treat as a normalized
        # weight without rescaling.
        if score > 1.0:
            score = 1.0
        elif score < -1.0:
            score = -1.0
        return score

    def is_theme(self, focal: str) -> bool:
        """True if `focal` is a current emergent theme.  Used by
        callers that want the binary check rather than the
        coherence scalar."""
        return bool(focal) and focal in self._signature.themes

    # ---- Phase H.3: event handling (active gestalt) ----

    def handle(self,
                  event: BrainEvent,
                  bus: EventBus) -> None:
        """Subscribed to ATTENDED_PERCEPT.  For each focal that
        survived the thalamic gate, compute coherence against the
        cached gestalt.  When non-neutral, fire a chemistry event:
          coherence ≥  FIT_THRESHOLD   → 'puzzle_fit'
          coherence ≤ -STRESS_THRESHOLD → 'puzzle_stress'

        Doctrine: low energy / fire when needed.  Neutral percepts
        produce no chemistry event — the gestalt only colors the
        chemistry when there's a meaningful match or mismatch.
        """
        if not isinstance(event, AttendedPerceptEvent):
            return
        focals = getattr(event, 'focals', None) or []
        if not focals:
            return
        cycle = int(getattr(event, 'cycle', 0))
        for focal in focals:
            if not focal:
                continue
            score = self.coherence(focal)
            if score >= COHERENCE_FIT_THRESHOLD:
                self._fire(bus, focal, cycle, score, fit=True)
            elif score <= COHERENCE_STRESS_THRESHOLD:
                self._fire(bus, focal, cycle, -score, fit=False)
            else:
                self.neutral_percepts += 1

    def _fire(self,
                bus: EventBus,
                focal: str,
                cycle: int,
                magnitude: float,
                *,
                fit: bool) -> None:
        """Emit a puzzle_fit / puzzle_stress ChemistryEvent.

        The chemistry engine subscribes to CHEMISTRY_FIRE and
        applies EVENT_DELTAS[kind] to the focal's bubble.  The
        reward ledger also picks these up via REWARD_MAGNITUDES
        and propagates credit to the recent action trace.
        """
        kind = 'puzzle_fit' if fit else 'puzzle_stress'
        if fit:
            self.fits_fired += 1
        else:
            self.stresses_fired += 1
        try:
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=cycle,
                timestamp=time.time(),
                source_capability='personality',
                origin='internal',
                origin_detail=f'gestalt_bias:{focal}',
                chemistry_kind=kind,
                magnitude=max(0.0, min(1.0,
                                       magnitude * COHERENCE_FIRE_SCALE)),
                target_concepts=[focal]))
        except Exception:
            pass

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'refreshes': self.refreshes,
            'last_refresh_cycle': self._last_refresh_cycle,
            'cached_cycle': self._signature.cycle,
            'cached_inputs_total': self._signature.inputs_total,
            'cached_themes': len(self._signature.themes),
            'fits_fired': self.fits_fired,
            'stresses_fired': self.stresses_fired,
            'neutral_percepts': self.neutral_percepts,
        }

    # ---- helper ----

    @staticmethod
    def _safe_get(provider: Optional[Callable]) -> Any:
        if provider is None:
            return None
        try:
            return provider()
        except Exception:
            return None
