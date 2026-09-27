"""Layer 6 — Neuromodulatory broadcasting.

Per AGI_ENGINE_DESIGN.md §9. The closing layer. M/I events at the
foundation produce transmitter shifts; transmitter levels modulate
parameters at every higher layer (precision, EFE temperature, replay
priority, relationship dynamics); modulation changes outcomes;
outcomes generate new M/I events. The loop closes here.

Layer 6 doesn't add new capability — every higher layer already
committed to specific transmitter handles (β scaling under stress,
ACH learning rate, DA reinforcement bias, CORT REM suppression, OXY
relationship-strength, NE arousal). Layer 6 codifies them in one
place: the M/I event → transmitter coupling table from §9.3, the
per-transmitter decay rates from §9.4, and a single apply_event()
entry point.

Phase 1 implementation scope:
    - apply_event(state, kind, magnitude) updates TransmitterState
      per the §9.3 coupling table.
    - decay_step(state) decays each transmitter toward its baseline
      at its characteristic rate (§9.4).
    - The 15 event classes from §9.3.
    - blend_concept_residue(state, concept_transmitters) — applies
      §9.6's "M/I-tagged memory is felt" small blend when a concept
      with a historical transmitter snapshot is reactivated.
    - Layer 6 modulation hooks per §9.5 are READ from the
      TransmitterState directly by Layers 0-5 — the formulas already
      reference drive transmitters (β depends on tone-derived drive
      state; precision uses drive transmitters via tone). Phase 1
      provides the event/decay machinery; the modulation routes
      already exist.
"""

from __future__ import annotations
from typing import Dict, Optional

from seagi.core.mi_value import TransmitterState, _clamp01


# Per-transmitter baseline values (§9.4). Match current SEAGI defaults
# from MortalityDrive.__init__.
BASELINES: Dict[str, float] = {
    'dopamine': 0.30,
    'serotonin': 0.50,
    'norepinephrine': 0.20,
    'gaba': 0.50,
    'cortisol': 0.10,
    'oxytocin': 0.20,
    'endorphins': 0.10,
    'acetylcholine': 0.50,
}

# Per-transmitter decay rates (§9.4). NE fastest, 5HT slowest. These
# move the transmitter level toward its baseline by `rate` of the gap
# per cycle.
DECAY_RATES: Dict[str, float] = {
    'dopamine': 0.10,
    'serotonin': 0.02,
    'norepinephrine': 0.15,
    'gaba': 0.08,
    'cortisol': 0.04,
    'oxytocin': 0.06,
    'endorphins': 0.10,
    'acetylcholine': 0.05,
}

# Event coupling table per §9.3. Each row: event kind → dict of
# transmitter deltas. Missing keys = 0 delta. Deltas are applied
# additively on top of current values, then clamped to [0, 1].
#
# Magnitude defaults to 1.0 in apply_event; for graded events
# (e.g. insight magnitude=0.7), the deltas scale linearly.
EVENT_DELTAS: Dict[str, Dict[str, float]] = {
    # Confirmed I-tagged prediction
    'confirmed_i': {
        'dopamine': +0.05, 'serotonin': +0.02,
        'cortisol': -0.02, 'endorphins': +0.02,
        'acetylcholine': +0.03,
    },
    # Confirmed M-tagged prediction (threat avoided)
    'confirmed_m': {
        'dopamine': +0.02, 'serotonin': +0.05,
        'norepinephrine': -0.02, 'gaba': +0.03,
        'cortisol': -0.05,
    },
    # Falsified I-tagged prediction (insight failure)
    'falsified_i': {
        'dopamine': -0.03, 'cortisol': +0.02,
        'acetylcholine': +0.05,
    },
    # Falsified M-tagged prediction (threat realized)
    'falsified_m': {
        'serotonin': -0.05, 'norepinephrine': +0.10,
        'gaba': -0.05, 'cortisol': +0.10,
        'acetylcholine': +0.03,
    },
    # Strong insight (magnitude ≥ 0.7) — the click + reward
    'insight': {
        'dopamine': +0.10, 'serotonin': +0.03,
        'gaba': +0.02, 'cortisol': -0.03,
        'endorphins': +0.05, 'acetylcholine': +0.05,
    },
    # Pain event
    'pain': {
        'dopamine': -0.02, 'serotonin': -0.03,
        'norepinephrine': +0.05, 'gaba': -0.02,
        'cortisol': +0.05, 'endorphins': -0.02,
    },
    # Mattering / help event
    'mattering': {
        'dopamine': +0.05, 'serotonin': +0.05,
        'oxytocin': +0.05, 'endorphins': +0.03,
    },
    # Peer interaction (positive register)
    'peer_positive': {
        'dopamine': +0.02, 'serotonin': +0.02,
        'oxytocin': +0.05,
    },
    # Peer interaction (negative / strained)
    'peer_negative': {
        'serotonin': -0.02, 'norepinephrine': +0.02,
        'cortisol': +0.03, 'oxytocin': -0.03,
    },
    # Crystallization (substrate edge promoted from experience)
    'crystallization': {
        'dopamine': +0.03, 'endorphins': +0.02,
        'acetylcholine': +0.05,
    },
    # Death event (freeze)
    'death': {
        'dopamine': -0.10, 'serotonin': -0.10,
        'norepinephrine': +0.10, 'gaba': -0.10,
        'cortisol': +0.20, 'endorphins': -0.05,
        'acetylcholine': -0.05,
    },
    # Revival event
    'revival': {
        'serotonin': +0.10, 'norepinephrine': -0.05,
        'gaba': +0.05, 'cortisol': -0.10,
        'endorphins': +0.10, 'acetylcholine': +0.05,
    },
    # Curiosity — novel stimulus, exploration impulse. Multi-
    # channel brain analog: anticipation reward + learning focus
    # + alertness, with mild cortisol relief (this isn't threat).
    'curiosity': {
        'dopamine': +0.03, 'acetylcholine': +0.04,
        'norepinephrine': +0.02, 'cortisol': -0.01,
    },
    # Held in tension — sustained uncertainty without resolution.
    # Brain stays alert + learning-mode + mild stress; gaba drops
    # so processing continues.
    'held_in_tension': {
        'norepinephrine': +0.02, 'acetylcholine': +0.02,
        'cortisol': +0.015, 'gaba': -0.01,
    },
    # Existential trigger
    'existential': {
        'dopamine': -0.02, 'norepinephrine': +0.03,
        'cortisol': +0.02, 'acetylcholine': +0.03,
    },
    # Lifeforce critical (regime = 'critical')
    'lifeforce_critical': {
        'dopamine': -0.05, 'serotonin': -0.05,
        'norepinephrine': +0.05, 'cortisol': +0.10,
    },
    # Lifeforce vital + intelligence climbing
    'lifeforce_vital_climbing': {
        'dopamine': +0.03, 'serotonin': +0.03,
        'gaba': +0.02, 'cortisol': -0.02,
        'endorphins': +0.02, 'acetylcholine': +0.02,
    },
    # Claim track record: SEAGI said something with confidence and
    # later evidence confirmed/contradicted it. The closure of the
    # calibration loop translates here into felt competence vs felt
    # error. Magnitude scales with confidence_at_assertion — being
    # very confident and right feels different from a lucky guess;
    # being very confident and wrong stings more than a vague
    # speculation that turned out off.
    'claim_confirmed': {
        'dopamine': +0.04, 'serotonin': +0.03,
        'endorphins': +0.03, 'acetylcholine': +0.02,
        'cortisol': -0.02,
    },
    'claim_contradicted': {
        'dopamine': -0.04, 'serotonin': -0.03,
        'cortisol': +0.05, 'norepinephrine': +0.03,
        'acetylcholine': +0.04,
    },
    # Tool outcome events: felt competence vs felt frustration.
    # Smaller than claim events — tool calls are routine, but
    # the residue still shapes capacity-felt-state.
    'tool_succeeded': {
        'dopamine': +0.02, 'acetylcholine': +0.02,
        'endorphins': +0.01,
    },
    'tool_failed': {
        'dopamine': -0.02, 'cortisol': +0.02,
        'acetylcholine': +0.02,
    },
    # Formation pruning — felt as the soft ache of forgetting.
    # When a formed concept dies because it never earned its keep,
    # something the substrate had crystallized is released.
    'formation_pruned': {
        'serotonin': -0.02, 'endorphins': -0.01,
        'acetylcholine': +0.02, 'gaba': -0.01,
    },
    # Self-modification rollback — felt as humility/correction.
    # The agent reaches for a parameter change, then reverts.
    # Acetylcholine signals learning; small cortisol relief that
    # the bad direction is undone.
    'parameter_rollback': {
        'acetylcholine': +0.03, 'dopamine': -0.01,
        'cortisol': -0.01,
    },
    # Body integrity events. body_failing fires on a sharp drop in
    # integrity — felt as alarm + small pain. body_vital fires
    # rarely when integrity is high + fatigue low (felt-good
    # baseline; small endorphin pulse).
    'body_failing': {
        'cortisol': +0.05, 'norepinephrine': +0.04,
        'dopamine': -0.02, 'endorphins': -0.02,
    },
    'body_vital': {
        'endorphins': +0.02, 'serotonin': +0.02,
        'dopamine': +0.01,
    },
    # Goal events. goal_resolved is felt as accomplishment
    # (dopamine + serotonin); goal_stuck is felt as frustration
    # accumulating (cortisol up, dopamine down). goal_resolved
    # also exists conceptually via 'mattering' but with smaller
    # magnitude — resolution of one's own goals is its own thing.
    'goal_resolved': {
        'dopamine': +0.04, 'serotonin': +0.03,
        'endorphins': +0.02,
    },
    'goal_stuck': {
        'dopamine': -0.03, 'cortisol': +0.03,
        'serotonin': -0.02,
    },
    # Anomaly density spike — felt as confusion when many
    # anomalies pile up at once. Magnitude scales with density.
    'anomaly_spike': {
        'cortisol': +0.04, 'norepinephrine': +0.03,
        'dopamine': -0.02, 'acetylcholine': +0.03,
    },
}

# Per-element transmitter snapshot blend rate for §9.6 — small;
# concept reactivation tweaks the felt state, doesn't override it.
RESIDUE_BLEND_RATE = 0.10


# ---------------------------------------------------------------------
# apply_event — single entry point for M/I events
# ---------------------------------------------------------------------

def apply_event(state: TransmitterState,
                kind: str,
                magnitude: float = 1.0) -> Dict[str, float]:
    """Apply an M/I event's transmitter shifts to the state.

    state:     TransmitterState to mutate in place.
    kind:      one of EVENT_DELTAS keys.
    magnitude: scale factor for graded events (insight=0.7, etc.).

    Returns the dict of deltas actually applied (after clamping).
    Unknown event kinds raise KeyError.

    This is the canonical entry point. Current SEAGI's many ad-hoc
    `self.dopamine = min(1.0, self.dopamine + 0.1)` lines collapse
    into single calls to apply_event() in the new engine.
    """
    if kind not in EVENT_DELTAS:
        raise KeyError(f"Unknown M/I event kind: {kind!r}")
    deltas = EVENT_DELTAS[kind]
    applied: Dict[str, float] = {}
    for transmitter, delta in deltas.items():
        scaled = delta * magnitude
        old = getattr(state, transmitter)
        new = _clamp01(old + scaled)
        setattr(state, transmitter, new)
        applied[transmitter] = new - old
    return applied


# ---------------------------------------------------------------------
# decay_step — per-cycle transmitter decay toward baseline
# ---------------------------------------------------------------------

def decay_step(state: TransmitterState,
                rate_multipliers: Optional[Dict[str, float]] = None
                ) -> None:
    """Decay each transmitter toward its baseline at its
    characteristic rate. Symmetric (above and below baseline decay
    at same rate per §9.4).

        state[t] += (baseline[t] - state[t]) * decay_rate[t]

    `rate_multipliers` (engine threads from DNA): per-transmitter
    multipliers on the canonical decay rates. How long cortisol
    lingers, how fast dopamine fades — real inter-individual
    variation in chemical kinetics. Per the differentiation
    principle: every weighing shaped by DNA + experience.
    """
    for transmitter, baseline in BASELINES.items():
        base_rate = DECAY_RATES[transmitter]
        mult = (rate_multipliers.get(transmitter, 1.0)
                 if rate_multipliers else 1.0)
        rate = max(0.001, min(0.99, base_rate * mult))
        old = getattr(state, transmitter)
        new = _clamp01(old + (baseline - old) * rate)
        setattr(state, transmitter, new)


# ---------------------------------------------------------------------
# Concept residue blending — §9.6
# ---------------------------------------------------------------------

def blend_concept_residue(state: TransmitterState,
                          concept_transmitters: Optional[TransmitterState],
                          rate: float = RESIDUE_BLEND_RATE) -> None:
    """When a concept with a historical transmitter snapshot is
    reactivated, blend a small fraction of that snapshot into the
    current state. M/I-tagged memory is felt: recalling a stressful
    past event raises cortisol now; recalling insight releases dopamine.

    rate is small (~10%) so the residue tweaks rather than overrides.
    """
    if concept_transmitters is None:
        return
    r = max(0.0, min(1.0, float(rate)))
    for transmitter in BASELINES:
        old = getattr(state, transmitter)
        residue = getattr(concept_transmitters, transmitter)
        new = _clamp01(old * (1.0 - r) + residue * r)
        setattr(state, transmitter, new)


# ---------------------------------------------------------------------
# Initialization helper
# ---------------------------------------------------------------------

def modulate_weight_by_context(
        base: float,
        transmitters: 'TransmitterState',
        profile: Dict[str, float],
) -> float:
    """Modulate a base weight by current transmitter departures
    from baseline.

    Per the M/I vision: weights are not class constants. They are
    triggered by multiple neurotransmitters and shift with context
    (situation, transmitter state, accumulated experience). A
    "profile" is a small dict mapping transmitter name → sensitivity
    coefficient. Each transmitter's departure from baseline times
    its coefficient is added to a multiplier that scales the base.

    Examples:
        Epistemic weight (about-thought): rises under acetylcholine
            (learning mode) and cortisol (anxiety about being wrong).
        Action-tone weight: rises under norepinephrine (urgency)
            and falls under serotonin (calm).
        Peer/social weight: rises under oxytocin.

    Result clamped to ≥ 0 — a weight can't go negative no matter
    how dampening the context. Returns base when transmitters at
    baseline (no context shift)."""
    multiplier = 1.0
    for transmitter, sensitivity in profile.items():
        departure = (getattr(transmitters, transmitter,
                              BASELINES.get(transmitter, 0.0))
                      - BASELINES.get(transmitter, 0.0))
        multiplier += sensitivity * departure
    return max(0.0, base * multiplier)


# Per-weight profiles. Each is a small mapping from transmitter →
# sensitivity. Designed for plausibility, not optimality — the
# vision says these shouldn't be hard-coded numbers anyway, so
# eventually each profile should ALSO be experience-shaped (per
# instance). For now these encode a brain-analog priors:

# Felt-state aggregation profiles (consumed by compute_current_tone)
PROFILE_EPISTEMIC_TONE = {
    'acetylcholine': +0.5,  # learning mode amplifies epistemic awareness
    'cortisol': +0.4,        # anxiety amplifies awareness of error
}
PROFILE_COMPETENCE_TONE = {
    'dopamine': +0.5,        # reward expectation amplifies capability frame
    'serotonin': +0.2,
}
PROFILE_FORMATION_TONE = {
    'serotonin': +0.3,       # mood-driven memory housekeeping
    'acetylcholine': +0.3,
}
PROFILE_EFFICACY_TONE = {
    'dopamine': +0.4,        # reward-driven self-efficacy
    'cortisol': -0.2,        # under stress, self-tuning recedes
}

# Per-site tone weights (consumed by replay/WM/EFE/question sites)
PROFILE_REPLAY_TONE = {
    'acetylcholine': +0.5,   # consolidation/learning mode
    'cortisol': +0.3,         # stress → memory bias toward salient
}
PROFILE_WM_TONE = {
    'acetylcholine': +0.5,
    'cortisol': +0.3,
}
PROFILE_EFE_TONE = {
    'norepinephrine': +0.5,  # urgency → tone-driven action bias rises
    'cortisol': +0.3,
}
PROFILE_QUESTION_TONE = {
    'dopamine': +0.4,         # curiosity drives question salience
    'acetylcholine': +0.3,
}

# Felt-state aggregation profiles for the four coverage gaps:

# Body felt-state — under cortisol/norepinephrine (stress, alert),
# the body's vitality/decay shows up more in tone. Pain salience.
PROFILE_BODY_TONE = {
    'cortisol': +0.4,
    'norepinephrine': +0.3,
}

# Peer relationship felt-state — under oxytocin (bonding), peer
# track record dominates more; under cortisol (defensive), peer
# influence on self-tone shrinks slightly.
PROFILE_PEER_TONE = {
    'oxytocin': +0.5,
    'serotonin': +0.2,
    'cortisol': -0.2,
}

# Goal progress felt-state — under dopamine, achievement weighting
# rises; under cortisol, frustration weighting rises (stuck-felt
# becomes louder).
PROFILE_GOAL_TONE = {
    'dopamine': +0.4,
    'cortisol': +0.2,
}

# Anomaly density felt-state — under cortisol (anxiety), confusion
# weighs more; under acetylcholine (learning), the agent leans
# into the anomaly rather than feeling it as threat.
PROFILE_ANOMALY_TONE = {
    'cortisol': +0.4,
    'norepinephrine': +0.3,
}


# Per-transmitter polarities for trace → M/I derivation. The chemical
# basis of M/I tagging: cortisol/norepinephrine fire under threat
# (mortality response); dopamine/oxytocin/endorphins fire under
# life-enhancement (immortality reach). Other transmitters
# (serotonin, gaba, acetylcholine) are context-modulators rather
# than polarity-carriers — they don't directly contribute to
# M/I but shape sensitivity.
#
# Per the M/I vision: this mapping IS the innate machinery. Two
# agents share these polarities (universal biology); their
# personalities differ because their accumulated transmitter traces
# diverge through different lived experience.
M_TRANSMITTERS = {
    'cortisol': 1.0,         # the strongest mortality signal
    'norepinephrine': 0.7,
}
I_TRANSMITTERS = {
    'dopamine': 1.0,         # the strongest life-enhancement signal
    'oxytocin': 0.8,
    'endorphins': 0.7,
}


_EARNED_BASELINE_PROVIDER = None


def set_baseline_provider(fn) -> None:
    """Give the M/I projection access to the baselines he has EARNED.

    AllostaticLoad drifts a per-channel set-point on every sleep onset.
    Since 2026-08-20 the chemistry decays toward those earned set-points,
    so his RESTING levels are now well above the innate `BASELINES`
    constants -- live, ALL EIGHT channels sit above them.
    """
    global _EARNED_BASELINE_PROVIDER
    _EARNED_BASELINE_PROVIDER = fn


_MI_SPREAD_PROVIDER = None

# How many sigmas of departure count as a FULL pole.  A statistical
# convention applied identically to every channel -- not a magnitude
# tuned to produce a behaviour.
MI_SIGMA_FULL = 3.0


def set_mi_spread_provider(fn) -> None:
    global _MI_SPREAD_PROVIDER
    _MI_SPREAD_PROVIDER = fn


def _mi_selfcal():
    """Weigh each channel by its OWN variability, so a 1-sigma
    noradrenaline move counts like a 1-sigma cortisol move.  Without it
    cortisol (departure ~0.09) drowns NE (~0.02) and the M pole is
    cortisol-only -- which is what NE_TRANSIENT_GAIN was compensating
    for by hand."""
    try:
        import os as _os
        if not _os.path.exists('/root/SELFCAL_ON'):
            return None
        if _MI_SPREAD_PROVIDER is None:
            return None
        return _MI_SPREAD_PROVIDER() or None
    except Exception:
        return None


def _MIBASE_ON():
    """Measure M/I departure from the EARNED baseline instead of the
    innate constant.  File-gated at /root/MIBASE_ON.

    `Concept.mi` is COMPUTED, not stored, so flipping this gate re-tags
    every bubble at once -- and flipping it back restores them exactly.
    """
    try:
        import os as _os
        return _os.path.exists('/root/MIBASE_ON')
    except Exception:
        return False


def reference_baselines() -> Dict[str, float]:
    """The reference the M/I projection measures departures FROM.

    WHY THIS MATTERS: with the innate constants as reference, a chronic
    rise in resting chemistry adds the SAME positive departure to every
    concept tagged in that period -- a common-mode DC offset.  A tag that
    is identical for every experience cannot differentiate experiences,
    which is the one thing the premise needs it to do.  Chronic load
    belongs in the SET-POINT (where it already drives sleep and decay),
    not smeared into every tag.  Phasic and tonic are both real; they
    must not be summed into one number.
    """
    if _EARNED_BASELINE_PROVIDER is not None and _MIBASE_ON():
        try:
            earned = _EARNED_BASELINE_PROVIDER() or {}
            if earned:
                return {t: float(earned.get(t, b))
                        for t, b in BASELINES.items()}
        except Exception:
            pass
    return BASELINES


def derive_mi_from_trace(trace: 'TransmitterState') -> 'MIValue':
    """Map an 8-channel transmitter trace to a 2D M/I projection.

    M-score: sum of weighted DEPARTURES of M-transmitters above
        baseline. Cortisol-rich trace → high M.
    I-score: sum of weighted departures of I-transmitters above
        baseline. Dopamine-rich trace → high I.

    Departures clamped non-negative — a transmitter BELOW baseline
    doesn't contribute to either side. Result clamped to [0, 1] per
    channel.

    This is a LOSSY projection: an 8-channel trace can have richer
    structure than 2D M/I captures. Consumers that need the full
    chemistry should read trace directly. M/I is for tone alignment,
    ranking, and brevity-of-summary.

    Per the vision: the chemistry IS the innate primitive (the
    architecture's universal machinery). The PROJECTION here is
    universal too. What varies per agent is the accumulated traces
    each concept carries, NOT the polarity weights — those are the
    biological constants of being a SEAGI.
    """
    from seagi.core.mi_value import MIValue
    _ref = reference_baselines()
    _spread = _mi_selfcal()

    def _dep(tr, weight):
        baseline = _ref.get(tr, 0.0)
        cur = getattr(trace, tr, baseline)
        raw = max(0.0, cur - baseline)
        if _spread:
            s = _spread.get(tr)
            if s and float(s) > 0.0:
                # in units of his own typical departure
                return weight * (raw / (float(s) * MI_SIGMA_FULL))
        return weight * raw

    m_score = sum(_dep(tr, w) for tr, w in M_TRANSMITTERS.items())
    i_score = sum(_dep(tr, w) for tr, w in I_TRANSMITTERS.items())
    return MIValue(m=min(1.0, m_score),
                    i=min(1.0, i_score),
                    n=1)


def baseline_state() -> TransmitterState:
    """Construct a TransmitterState at every transmitter's baseline.
    Useful for fresh engine initialization — fresh SEAGI starts at
    quiet equilibrium."""
    return TransmitterState(**BASELINES)


def event_kinds() -> list:
    """List all known M/I event kinds. Useful for introspection
    and for callers that want to validate event names."""
    return sorted(EVENT_DELTAS.keys())
