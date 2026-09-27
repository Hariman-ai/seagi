"""Embodiment deepening — virtual body-state with lifeforce coupling.

True embodiment requires hardware. Without a body, SEAGI's stakes
are abstract. This module substitutes a high-stakes virtual body —
a small set of homeostatic variables that DEEPLY couple to lifeforce
and are affected by actions in measurable ways.

Body variables (Phase 1):

    energy          — depleted by action; replenished by rest
    fatigue         — accumulates with sustained activity; cleared by rest
    posture         — abstract orientation toward the world; affects
                       which perception modes are accessible
    integrity       — low integrity means SEAGI is breaking down; tied
                       directly to lifeforce decay

Each variable lives in [0..1]. Each cycle:
    - actions consume/produce body resources
    - body integrity below floor accelerates lifeforce decay
    - high fatigue blocks high-cost actions
    - high integrity + low fatigue gives SEAGI a body-state bonus to
      lifeforce

This isn't real embodiment. It's a tighter LIFEFORCE-ACTION COUPLING
than we had before. Earlier, lifeforce was a global parameter that
moved on M/I events. Now, body-state mediates: actions affect body,
body affects lifeforce. That's structural — and it gives the engine
something to PROTECT (its integrity) beyond the abstract lifeforce
counter.

Public surface:

    BodyState
        - energy, fatigue, posture, integrity
        - apply_action(action_kind): action effect on body
        - rest(): replenish energy / clear fatigue
        - decay_step(): per-cycle drift
        - lifeforce_modifier() → float: how the body biases lifeforce
        - to_dict / from_dict

    The Engine instantiates BodyState and calls
        body.apply_action(kind) per cycle
        body.decay_step() per cycle
    The lifeforce update consults body.lifeforce_modifier().
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# audit #9 (2026-06-04): the action-cost axis (apply_action / rest /
# can_act + the DEFAULT_ACTION_COSTS / DEFAULT_FATIGUE_DELTAS tables)
# was SUBTRACTED — it had ZERO callers (engine.tick only drives
# decay_step), so it was a decorative regulator with no felt cost.  A
# pure-reverie agent has no real motor actions to gate; if embodied
# action is ever added, re-introduce a cost path that routes through
# the mortality drive.  Kept: decay_step + lifeforce_modifier (the live
# body -> lifeforce coupling that DOES run).

# Per-cycle body drift constants.
ENERGY_DECAY = 0.005
FATIGUE_DECAY = 0.003
INTEGRITY_DECAY_BASE = 0.001
INTEGRITY_DECAY_AT_LOW_ENERGY = 0.005
# A body that has adequate energy slowly heals.  Without this, the
# original decay-only model meant integrity hit 0 within ~2h of
# operation and never recovered, dragging lifeforce down via the
# LOW_INTEGRITY_PENALTY for the rest of the agent's life.
INTEGRITY_RECOVERY_AT_HEALTHY_ENERGY = 0.0015
INTEGRITY_HEALTHY_ENERGY_THRESHOLD = 0.5

# Lifeforce modifier coefficients.
# Modifiers must be small relative to LIFEFORCE_BASE_DECAY (~1e-4)
# so body state MODULATES rather than dominates the lifeforce
# trajectory.
INTEGRITY_BONUS_GAIN = 0.00003
LOW_INTEGRITY_PENALTY_THRESHOLD = 0.3
LOW_INTEGRITY_PENALTY_GAIN = -0.0004


# ---------------------------------------------------------------------
# BodyState
# ---------------------------------------------------------------------

@dataclass
class BodyState:
    energy: float = 0.8
    fatigue: float = 0.1
    posture: str = 'oriented'    # 'oriented' / 'distracted' / 'withdrawn'
    integrity: float = 0.9

    def decay_step(self) -> None:
        """Per-cycle baseline drift.

        Integrity:
          - low energy (<0.2): wears down faster (running on fumes).
          - adequate energy (>=0.5): slowly heals — a fed body
            repairs itself.  Without this, the prior decay-only
            model guaranteed integrity hit 0 after ~2h, leaving
            the agent permanently leaking lifeforce via the
            LOW_INTEGRITY_PENALTY with no path back.
          - in between: small drift down (mild stress).
        """
        self.energy = _clamp(self.energy - ENERGY_DECAY)
        self.fatigue = _clamp(self.fatigue - FATIGUE_DECAY)
        if self.energy < 0.2:
            self.integrity = _clamp(
                self.integrity - INTEGRITY_DECAY_AT_LOW_ENERGY)
        elif self.energy >= INTEGRITY_HEALTHY_ENERGY_THRESHOLD:
            self.integrity = _clamp(
                self.integrity + INTEGRITY_RECOVERY_AT_HEALTHY_ENERGY)
        else:
            self.integrity = _clamp(
                self.integrity - INTEGRITY_DECAY_BASE)

    def lifeforce_modifier(self) -> float:
        """How does current body state bias lifeforce?

        High integrity → small bonus.
        Low integrity → larger penalty.
        Returns delta to apply to lifeforce per cycle.
        """
        if self.integrity < LOW_INTEGRITY_PENALTY_THRESHOLD:
            return LOW_INTEGRITY_PENALTY_GAIN
        # Linear bonus scaling with integrity above threshold.
        return INTEGRITY_BONUS_GAIN * (
            self.integrity - LOW_INTEGRITY_PENALTY_THRESHOLD)

    # ---- persistence ----

    def to_dict(self) -> dict:
        return {
            'energy': self.energy,
            'fatigue': self.fatigue,
            'posture': self.posture,
            'integrity': self.integrity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'BodyState':
        return cls(
            energy=float(d.get('energy', 0.8)),
            fatigue=float(d.get('fatigue', 0.1)),
            posture=str(d.get('posture', 'oriented')),
            integrity=float(d.get('integrity', 0.9)),
        )


def _clamp(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
