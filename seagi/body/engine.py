"""Seagi body — the homeostatic vessel.

Replaces the legacy `agi_engine.engine.Engine` (~5,000 lines
mixing body machinery with v1 cognitive code) with the minimal
body-machinery class.

What's here
-----------
  substrate    — Seagi's knowledge graph (concepts, edges,
                  bubbles).  The persistent state.
  hierarchy    — Time / cycle counter.  Drives decay and
                  lateral recurrence in latent levels.
  embodiment   — Body state: energy, fatigue, posture,
                  integrity.  Tightly coupled to lifeforce.
  lifeforce    — Mortality drive state, in [0..1].  Decays
                  with action, replenishes with rest.

What's NOT here
---------------
  No goal_tracker, identity, reward_ledger, skill_library,
  schema_library, dialog_manager, action_layer, replay_layer,
  L5State, Dna — these were v1 cognitive scaffolding and are
  replaced by seagi.brain.  Saved brain files from v1 that
  contained those fields just ignore them when loaded into
  this minimal Engine.

The brain reads from this vessel continuously (insula samples
body state every tick; chemistry decays each cycle).  The
brain does not own this state; this state IS the body.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from seagi.core.substrate import Substrate
from seagi.core.mi_value import _clamp01
from seagi.body.hierarchy import Hierarchy
from seagi.body.embodiment import BodyState


class Engine:
    """Seagi body.  Named 'Engine' for compatibility with
    callers (serve.py, v2 brain providers, test scaffolding,
    save_brain / load_brain) that already use this name.
    Future cleanup may rename to `Body`."""

    def __init__(self,
                 substrate: Optional[Substrate] = None,
                 lifeforce: float = 0.7,
                 seed: Optional[int] = None):
        # Substrate is the only thing that genuinely matters
        # for persistence.  Hierarchy and embodiment are
        # stateful but minimal; lifeforce is a scalar.
        self.substrate = substrate or Substrate()
        self.hierarchy = Hierarchy(self.substrate)
        self.embodiment = BodyState()
        self.lifeforce: float = _clamp01(lifeforce)

    # ---- persistence ----

    def to_dict(self, lazy: bool = False) -> Dict[str, Any]:
        """Serialize the body for save_brain.  Only the
        load-bearing state — the substrate and the four
        body-machinery values.  Brain-side transient state
        (chemistry, AWM, episodes) is NOT persisted here;
        that's brain-state and rebuilds from scratch on next
        session.  Personality persists via substrate
        (bubbles' transmitter_trace)."""
        return {
            'cycle': int(self.hierarchy.cycle_counter),
            'lifeforce': float(self.lifeforce),
            'substrate': self.substrate.to_dict(lazy=lazy),
            'embodiment': {
                'energy': float(self.embodiment.energy),
                'fatigue': float(self.embodiment.fatigue),
                'posture': str(self.embodiment.posture),
                'integrity': float(self.embodiment.integrity),
            },
        }

    @classmethod
    def from_dict(cls,
                     d: Dict[str, Any],
                     seed: Optional[int] = None) -> 'Engine':
        """Load from a save_brain dict.  Tolerates legacy v1
        brain files that contain many extra fields — they're
        silently ignored (those were v1 cognitive state that
        no longer exists in the unified architecture)."""
        sub = Substrate.from_dict(d.get('substrate', {}))
        e = cls(
            substrate=sub,
            lifeforce=float(d.get('lifeforce', 0.7)),
            seed=seed,
        )
        e.hierarchy.cycle_counter = int(d.get('cycle', 0))
        emb_d = d.get('embodiment') or {}
        if isinstance(emb_d, dict):
            for fname in ('energy', 'fatigue', 'integrity'):
                if fname in emb_d:
                    try:
                        setattr(e.embodiment, fname,
                                _clamp01(float(emb_d[fname])))
                    except Exception:
                        pass
            if 'posture' in emb_d:
                e.embodiment.posture = str(emb_d['posture'])
        return e

    # ---- per-cycle body update (called by serving code, not
    # by brain — brain has its own tick) ----

    def tick(self) -> None:
        """One cycle of body / lifeforce dynamics.  Brain has
        its own brain.tick() for chemistry / AWM / etc.  These
        are separate concerns and run at the caller's pace."""
        self.hierarchy.cycle_counter += 1
        self.embodiment.decay_step()
        # Lifeforce decay tempered by body's modifier.
        modifier = (self.embodiment.lifeforce_modifier()
                      if hasattr(self.embodiment,
                                  'lifeforce_modifier')
                      else 0.0)
        self.lifeforce = _clamp01(
            self.lifeforce + modifier)
