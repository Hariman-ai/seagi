"""Primitive bodily-state lexicon — the innate-tagging layer
the doctrine explicitly permits.

Doctrine reference: `feedback_seagi_doctrine_2026_05_15` rule 2:

    "Innate tagging IS allowed and required, but only on primitive
    bodily states and actions — breathing, eating, hunger, thirst,
    pain, fatigue, suffocation, warmth, cold, satiation, injury,
    healing. Magnitude is promille. Additive, not overriding.
    Purpose: subconscious mortality awareness."

What this module is
-------------------
A named, queryable lexicon of universal-survival bodily primitives.
Each primitive maps to a small chemistry cocktail at promille scale.
The Insula consults this lexicon every tick: it reads body state,
identifies which primitives are currently active and at what
intensity, and contributes their cocktails additively to global
chemistry.

Why this is doctrine-correct
----------------------------
1. Substrate-side neutrality preserved — no innate tags on concepts
   like "death" or "kills".  The innate response lives in the body→
   chemistry pathway, NOT in substrate storage.

2. Promille magnitudes — even at maximum intensity (e.g. severe
   suffocation), a single primitive contributes <1‰ to any channel.
   Many primitives + sustained activation integrate over many ticks
   into perceptible bias, but no single tick swings chemistry.

3. Additive — primitives contribute INTO global chemistry; they
   don't override what perception, monitor activations, or cortical
   processing also contribute.  The cocktail of any moment is the
   SUM of all promille contributions firing simultaneously.

4. Always present — the agent's body always has SOME state, so
   there's always SOME primitive baseline contribution.  This is
   what "subconscious mortality awareness" means operationally —
   the body is constantly whispering its survival status into
   chemistry, even when conscious attention is elsewhere.
"""

from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------
# Primitive lexicon
# ---------------------------------------------------------------------
#
# Each entry maps a primitive name to a small per-channel cocktail
# at MAXIMUM intensity.  At intensity I ∈ [0, 1], the actual
# contribution is `delta × I`.  Magnitudes are promille — even
# fully-saturated primitives contribute at most 3-5‰ to any single
# channel.  Cocktails are real channel patterns, not just M/I axis
# nudges: fatigue is cortisol + low NE (tired-but-not-alarmed);
# pain is cortisol + NE + low dopamine (alarmed and aversive);
# satiation is endorphins + oxytocin (warm and connected to
# the world); rest is serotonin + endorphins (calm settling).


# M-side primitives — survival-threatened body states.  These
# contribute to the agent's continual subconscious awareness that
# something is amiss.
PRIMITIVE_M_STATES: Dict[str, Dict[str, float]] = {
    'fatigue':     {'cortisol': +0.002, 'norepinephrine': +0.001,
                     'serotonin': -0.001},
    'exhaustion':  {'cortisol': +0.003, 'norepinephrine': +0.002,
                     'dopamine': -0.002},
    'hunger':      {'cortisol': +0.002, 'norepinephrine': +0.002,
                     'dopamine': -0.001},
    'thirst':      {'cortisol': +0.002, 'norepinephrine': +0.001},
    'pain':        {'cortisol': +0.003, 'norepinephrine': +0.003,
                     'dopamine': -0.002, 'endorphins': -0.001},
    'injury':      {'cortisol': +0.003, 'norepinephrine': +0.003,
                     'gaba': -0.001},
    'suffocation': {'cortisol': +0.005, 'norepinephrine': +0.005,
                     'gaba': -0.002},   # the strongest M primitive
    'cold':        {'cortisol': +0.002, 'norepinephrine': +0.002},
}


# I-side primitives — survival-affirmed body states.  These
# contribute to the agent's continual subconscious sense of being
# OK, of presence, of safety.
PRIMITIVE_I_STATES: Dict[str, Dict[str, float]] = {
    'satiation':  {'endorphins': +0.002, 'oxytocin': +0.001,
                    'serotonin': +0.001},
    'rest':       {'serotonin': +0.003, 'gaba': +0.002,
                    'endorphins': +0.001},
    'healing':    {'endorphins': +0.002, 'serotonin': +0.002,
                    'dopamine': +0.001},
    'breathing':  {'gaba': +0.001},     # constant, very small
    'warmth':     {'oxytocin': +0.002, 'endorphins': +0.001},
}


# Convenience union — every primitive in either polarity.
ALL_PRIMITIVES: Dict[str, Dict[str, float]] = dict(
    PRIMITIVE_M_STATES, **PRIMITIVE_I_STATES)


# Thresholds for detection from BodyState.  Tunable; default
# values place the boundaries where felt-shift would be expected
# (a fatigue of 0.4 is "noticeable tired"; 0.7 is "deep exhaustion").
FATIGUE_NOTICE = 0.40
FATIGUE_EXHAUSTION = 0.70
HUNGER_ENERGY_FLOOR = 0.30   # energy below this = felt hunger
THIRST_ENERGY_FLOOR = 0.20
PAIN_INTEGRITY_DROP = 0.10   # >0.1 drop in integrity recently = pain
INJURY_INTEGRITY_FLOOR = 0.50
SUFFOCATION_LIFEFORCE = 0.10   # critical lifeforce = suffocation
COLD_INTEGRITY_FLOOR = 0.40

SATIATION_ENERGY = 0.70
REST_FATIGUE_CEIL = 0.30
HEALING_INTEGRITY = 0.85
WARMTH_INTEGRITY = 0.85


# ---------------------------------------------------------------------
# Detection — body state → active primitives
# ---------------------------------------------------------------------


def detect_primitive_states(body_state: Any,
                                lifeforce: float = 0.7,
                                prev_integrity: float = 1.0
                                ) -> Dict[str, float]:
    """Read a `BodyState` (or duck-typed body object) and return a
    dict `{primitive_name: intensity}` of currently-active
    primitives.

    Each intensity is in [0, 1].  Primitives not present in the
    return dict are not currently active.

    Args:
      body_state: object with attributes `energy`, `fatigue`,
        `integrity`.  Tolerates missing attributes (defaults
        applied).
      lifeforce: current lifeforce, used for suffocation detection.
      prev_integrity: previous tick's integrity, used for pain
        detection (rapid drop).
    """
    energy = float(getattr(body_state, 'energy', 0.7))
    fatigue = float(getattr(body_state, 'fatigue', 0.2))
    integrity = float(getattr(body_state, 'integrity', 0.9))

    active: Dict[str, float] = {}

    # --- M-side ---
    if fatigue >= FATIGUE_EXHAUSTION:
        active['exhaustion'] = min(1.0,
            (fatigue - FATIGUE_EXHAUSTION) / (1.0 - FATIGUE_EXHAUSTION))
    elif fatigue >= FATIGUE_NOTICE:
        active['fatigue'] = min(1.0,
            (fatigue - FATIGUE_NOTICE)
            / (FATIGUE_EXHAUSTION - FATIGUE_NOTICE))

    if energy <= HUNGER_ENERGY_FLOOR:
        active['hunger'] = min(1.0,
            (HUNGER_ENERGY_FLOOR - energy) / HUNGER_ENERGY_FLOOR)

    if energy <= THIRST_ENERGY_FLOOR:
        active['thirst'] = min(1.0,
            (THIRST_ENERGY_FLOOR - energy) / THIRST_ENERGY_FLOOR)

    integrity_drop = prev_integrity - integrity
    if integrity_drop >= PAIN_INTEGRITY_DROP:
        active['pain'] = min(1.0,
            integrity_drop / (1.0 - PAIN_INTEGRITY_DROP))

    if integrity <= INJURY_INTEGRITY_FLOOR:
        active['injury'] = min(1.0,
            (INJURY_INTEGRITY_FLOOR - integrity) / INJURY_INTEGRITY_FLOOR)

    if integrity <= COLD_INTEGRITY_FLOOR:
        active['cold'] = min(1.0,
            (COLD_INTEGRITY_FLOOR - integrity) / COLD_INTEGRITY_FLOOR
        ) * 0.5    # cold is subtler than injury at same integrity

    if lifeforce <= SUFFOCATION_LIFEFORCE:
        active['suffocation'] = min(1.0,
            (SUFFOCATION_LIFEFORCE - lifeforce) / SUFFOCATION_LIFEFORCE)

    # --- I-side ---
    if energy >= SATIATION_ENERGY:
        active['satiation'] = min(1.0,
            (energy - SATIATION_ENERGY) / (1.0 - SATIATION_ENERGY))

    if fatigue <= REST_FATIGUE_CEIL and energy >= 0.5:
        active['rest'] = min(1.0,
            (REST_FATIGUE_CEIL - fatigue) / REST_FATIGUE_CEIL)

    if integrity >= HEALING_INTEGRITY:
        active['healing'] = min(1.0,
            (integrity - HEALING_INTEGRITY) / (1.0 - HEALING_INTEGRITY))

    if integrity >= WARMTH_INTEGRITY and energy >= 0.5:
        active['warmth'] = min(1.0,
            (integrity - WARMTH_INTEGRITY) / (1.0 - WARMTH_INTEGRITY))

    # Breathing is the always-on primitive — present whenever the
    # agent is alive (lifeforce > 0).  Constant low-grade contribution.
    if lifeforce > 0.0:
        active['breathing'] = 1.0

    return active


# ---------------------------------------------------------------------
# Aggregate channel contribution
# ---------------------------------------------------------------------


def chemistry_contribution(active_primitives: Dict[str, float]
                              ) -> Dict[str, float]:
    """Sum each active primitive's intensity-scaled cocktail into
    a per-channel contribution dict.  Returned values are at
    promille scale and meant to be ADDED to global chemistry
    each tick — not multiplied through EVENT_DELTAS.

    Caller (Insula) clamps the global state after applying.
    """
    contrib: Dict[str, float] = {}
    for name, intensity in active_primitives.items():
        cocktail = ALL_PRIMITIVES.get(name)
        if not cocktail:
            continue
        for ch, delta in cocktail.items():
            contrib[ch] = contrib.get(ch, 0.0) + delta * intensity
    return contrib


def primitive_summary(active_primitives: Dict[str, float]
                          ) -> Dict[str, Any]:
    """Diagnostic summary of currently-active primitive states.

    Returns a dict with:
      - 'm_states' / 'i_states' — names of active M/I primitives
      - 'dominant' — strongest single primitive (or '' if none)
      - 'm_intensity' / 'i_intensity' — summed intensity per side
    """
    m_states = []
    i_states = []
    dominant = ''
    max_intensity = 0.0
    m_intensity = 0.0
    i_intensity = 0.0
    for name, intensity in active_primitives.items():
        if name in PRIMITIVE_M_STATES:
            m_states.append(name)
            m_intensity += intensity
        elif name in PRIMITIVE_I_STATES:
            i_states.append(name)
            i_intensity += intensity
        if intensity > max_intensity:
            max_intensity = intensity
            dominant = name
    return {
        'm_states': m_states,
        'i_states': i_states,
        'dominant': dominant,
        'm_intensity': m_intensity,
        'i_intensity': i_intensity,
    }
