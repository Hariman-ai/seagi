"""Reward ledger — eligibility-trace credit assignment.

Phase F.8 (2026-05-16).  v1 had `reward.py`; v2 consolidation
dropped it.  The doctrine and audit both flag the gap: without
credit assignment, environmental feedback (rewards, prediction
confirmations, body replenishment) doesn't propagate back to
the action sequences that produced them.  Same actions keep
getting taken regardless of whether they led to good or bad
outcomes.

What this module is
-------------------
A small temporal-difference-style ledger that:

1. RECORDS each BG arbitration decision (action + chemistry
   context + cycle) into a bounded eligibility trace.
2. DETECTS reward signals:
     - confirmed_i chemistry → positive
     - falsified_i chemistry → negative
     - rest_replenish (body relief) → positive
     - threat (body alarm) → negative
3. PROPAGATES discounted credit backward through the recent
   trace on reward arrival.  Older actions get less credit
   (discount factor γ).
4. STORES `action_credit[(action_kind, chemistry_bucket)]` for
   BG to consult at arbitration time.

The chemistry_bucket is a coarse state fingerprint (the same
buckets as ContextKey.chemistry_signature in Phase B) — same
action in different chemistry contexts can have different
credit.  This is brain-correct: an action that works when calm
may not work when alarmed.

Doctrine alignment
------------------
- Promille magnitudes — single rewards nudge action_credit by
  small amounts.  Personality of "what tends to work" emerges
  from accumulation across many trials.
- No substrate writes — action_credit lives in brain working
  memory.  Bounded, decays over session.  Restart fresh each
  session.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import (Any, Callable, Deque, Dict, List, Optional,
                       Tuple)

from ..events import (
    EventKind, BrainEvent,
    ArbitrationDecidedEvent, ChemistryEvent,
    InteroceptionEvent,
)
from ..bus import EventBus


# Discount factor — credit assigned to older actions decays
# geometrically.  γ=0.9 means an action 5 steps back gets ~0.59
# of the reward; 10 steps back gets ~0.35.
DISCOUNT_FACTOR = 0.9

# Eligibility trace size — how many recent actions remain
# eligible for credit.  Beyond this, the action is "too old"
# to be plausibly responsible.
TRACE_SIZE = 16

# Reward magnitudes per signal kind.  Promille scale — many
# small signals integrate into stable preferences.
REWARD_MAGNITUDES: Dict[str, float] = {
    'confirmed_i':    +0.005,
    'falsified_i':    -0.003,
    'rest_replenish': +0.005,
    'threat':         -0.005,
    'mattering':      +0.003,
    'anomaly_spike':  -0.002,
    # Phase H.3 (2026-05-17): personality gestalt fit signals
    # propagate to action_credit so coherent percepts during
    # action sequences accumulate positive policy weight.  Closes
    # the gestalt → chemistry → reward → policy loop.
    'puzzle_fit':     +0.002,
    'puzzle_stress':  -0.001,
}

# Maximum |action_credit| per (action, context) pair.  Prevents
# runaway accumulation.  Saturating at ±0.5 keeps credit as a
# bias, never an override.
CREDIT_CAP = 0.5


def _MOTORSKILL_ON():
    """Let GAME actions earn skills and NT credit.  File-gated at
    /root/MOTORSKILL_ON.  Lifeforce credit stays off them
    regardless -- that is the real no-RL-on-survival boundary."""
    try:
        import os as _os
        return _os.path.exists('/root/MOTORSKILL_ON')
    except Exception:
        return False


@dataclass
class ActionTrace:
    """One entry in the eligibility trace."""
    cycle: int
    action: str               # the proposed_action that won
    capability: str           # who claimed it
    loop: str                 # 'cognitive'/'motivational'/'speech'
    chemistry_bucket: tuple   # coarse state at the moment


class RewardLedger:
    """Eligibility-trace credit assignment.

    Subscribes:
      ARBITRATION_DECIDED — record action into trace
      CHEMISTRY_FIRE     — detect reward / punishment signals
      INTEROCEPTION      — body-state rewards
    """

    SUBSCRIPTIONS = (
        EventKind.ARBITRATION_DECIDED,
        EventKind.CHEMISTRY_FIRE,
        EventKind.INTEROCEPTION,
    )

    def __init__(self,
                 chemistry_provider: Optional[Callable] = None,
                 cycle_provider: Optional[Callable] = None,
                 skill_library: Optional[Any] = None,
                 game_provider: Optional[Callable] = None):
        self._chemistry_provider = chemistry_provider
        self._cycle_provider = cycle_provider or (lambda: 0)
        # Phase F.9 (2026-05-16): skill_library lets the ledger
        # promote coherent reward-correlated sequences into
        # learned skills.  Optional — without it, only credit
        # accumulates; with it, repeated successful sequences
        # crystallize into reusable skill records.
        self._skill_library = skill_library
        # WHICH BOARD.  A motor action must not pool credit across games:
        # cross-game action->effect is 20.53% vs a 30.11% null.
        self._game_provider = game_provider
        self.motor_recorded: int = 0
        self.motor_skipped_body_credit: int = 0
        self._trace: Deque[ActionTrace] = deque(maxlen=TRACE_SIZE)
        # action_credit keyed by (action_kind, chemistry_bucket).
        # chemistry_bucket is a tuple of small ints (the same
        # bucketed signature ContextKey uses) — same action under
        # different chemistry contexts accumulates separately.
        self._action_credit: Dict[Tuple[str, tuple], float] = {}
        # Diagnostics.
        self.rewards_received: int = 0
        self.punishments_received: int = 0
        self.credits_applied: int = 0
        self.actions_recorded: int = 0

    # ---- event handling ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, ArbitrationDecidedEvent):
            self._record_action(event)
        elif isinstance(event, ChemistryEvent):
            self._maybe_apply_reward(event)
        elif isinstance(event, InteroceptionEvent):
            self._maybe_apply_body_reward(event)

    # ---- action recording ----

    def _record_action(self,
                          ev: ArbitrationDecidedEvent) -> None:
        # The 'motor' loop (WorldActor, the mortal-game test) must NEVER enter
        # the eligibility trace.  This ledger is a temporal-difference RL
        # engine; recording motor winners + later propagating lifeforce-delta
        # credit onto them would make the survival game quietly become
        # reinforcement learning on the score — the exact thing the
        # no-external-reward test forbids.  (Disabling the BG score-read alone
        # is insufficient: the leak is the LEDGER recording + body-reward
        # propagation, not just the arbitration read.)
        _is_motor = (getattr(ev, 'loop', '') == 'motor')
        if _is_motor and not _MOTORSKILL_ON():
            return
        bucket = self._current_bucket()
        if _is_motor:
            bucket = tuple(bucket) + (self._current_game(),)
            self.motor_recorded += 1
        # Reduce action to its kind: the verb part before ':' if
        # 'verb:focal' shape, otherwise the whole string.
        action_kind = self._action_kind(ev.winning_action)
        if not action_kind:
            return
        self._trace.append(ActionTrace(
            cycle=int(ev.cycle),
            action=action_kind,
            capability=ev.winning_capability,
            loop=ev.loop,
            chemistry_bucket=bucket,
        ))
        self.actions_recorded += 1
        # THE SKILL JUST RAN (2026-08-20).  Same match that promotes a
        # candidate -- last 3 actions + the oldest one's bucket -- applied
        # at EXECUTION time, so firing and success are defined by one rule
        # and neither is easier to earn than the other.  Until now nothing
        # recorded a sequence that ran and did NOT earn its reward, which
        # is why every skill's reliability was pinned at 1.0.
        try:
            _recent = list(self._trace)[-3:]
            # EXACTLY the slice `_observe_sequence` uses to promote: last 3,
            # minimum 2.  Requiring 3 here would let a 2-action skill be
            # promoted and never once be seen firing.
            if self._skill_library is not None and len(_recent) >= 2:
                for _sk in self._skill_library.find_by_sequence(
                        tuple(t.action for t in _recent),
                        _recent[0].chemistry_bucket):
                    self._skill_library.record_firing(_sk.id, int(ev.cycle))
                    # MANY may be in flight: a single slot would drop the
                    # first skill whenever a second ran before the first
                    # earned its reward, biasing reliability DOWN for
                    # skills that fire in quick succession.
                    if not hasattr(self, '_pending_skills'):
                        self._pending_skills = {}
                    self._pending_skills[_sk.id] = str(_sk.expected_reward)
                    # ...and the same firing against a WRONG reward, so the
                    # real rate has something to be compared with.
                    _nk = self._skill_library.null_reward_for(
                        _sk.expected_reward)
                    if _nk is not None:
                        if not hasattr(self, '_pending_null'):
                            self._pending_null = {}
                        self._pending_null[_sk.id] = str(_nk)
                        self._skill_library.record_null_firing()
        except Exception:
            pass

    # ---- reward detection ----

    def _maybe_apply_reward(self, ev: ChemistryEvent) -> None:
        """Check if a chemistry event is a reward signal.  If so,
        back-propagate credit through the trace."""
        # World/game tagging chemistry (confirmed_i / falsified_i fired by the
        # WorldActor to value game states) must NOT propagate action credit —
        # that would let game outcomes shape cognition by reward (RL).
        if getattr(ev, 'source_capability', '') == 'world_actor':
            return
        magnitude = REWARD_MAGNITUDES.get(ev.chemistry_kind)
        if magnitude is None:
            return
        # Scale by the event's own magnitude (chemistry fires
        # carry their own intensity).
        effective = magnitude * max(0.0, min(1.0, ev.magnitude))
        if effective == 0.0:
            return
        if effective > 0:
            self.rewards_received += 1
        else:
            self.punishments_received += 1
        self._propagate_credit(effective)
        # Phase F.9: feed positive-reward sequences to the
        # skill library so coherent action chains can be
        # promoted into learned skills.
        # The reward this skill PREDICTED actually arrived -> it worked.
        try:
            _pend = getattr(self, '_pending_skills', None) or {}
            _kind = str(ev.chemistry_kind)
            for _sid in [k for k, v in _pend.items() if v == _kind]:
                self._skill_library.record_success(_sid)
                _pend.pop(_sid, None)
            _pnull = getattr(self, '_pending_null', None) or {}
            for _sid in [k for k, v in _pnull.items() if v == _kind]:
                self._skill_library.record_null_success()
                _pnull.pop(_sid, None)
        except Exception:
            pass
        if (effective > 0
                and self._skill_library is not None
                and len(self._trace) >= 2):
            self._observe_sequence(ev.chemistry_kind, ev.cycle)

    def _observe_sequence(self,
                              reward_kind: str,
                              cycle: int) -> None:
        """Pass the recent trace's action sequence + context to
        the skill library for candidate promotion.

        Uses the last 3 actions (skill-sized window).  The bucket
        is taken from the OLDEST action in the slice — that's the
        precondition state at the start of the sequence.
        """
        if not self._trace or self._skill_library is None:
            return
        recent = list(self._trace)[-3:]
        if len(recent) < 2:
            return
        sequence = tuple(t.action for t in recent)
        precondition = recent[0].chemistry_bucket
        try:
            self._skill_library.observe_sequence(
                action_sequence=sequence,
                expected_reward=reward_kind,
                precondition_bucket=precondition,
                cycle=cycle,
                chem_raw=self._current_chem_raw())
        except Exception:
            pass

    def _maybe_apply_body_reward(self,
                                       ev: InteroceptionEvent) -> None:
        """Body deltas can be reward signals too.  Rising
        lifeforce = positive; sharp drops = negative.  Magnitudes
        scale promille."""
        # A lifeforce delta CAUSED BY the world / a game-death must not become
        # action credit (that would feed survival back into selection = RL).
        # The mortality wire flags such interoception world_caused=True.
        if getattr(ev, 'world_caused', False):
            return
        d = float(getattr(ev, 'delta_lifeforce', 0.0))
        # Use a smaller per-tick magnitude — body trends are
        # noisier signals than discrete chemistry events.
        if abs(d) < 0.05:
            return
        effective = d * 0.02   # promille scale
        if effective > 0:
            self.rewards_received += 1
        else:
            self.punishments_received += 1
        # LIFEFORCE CREDIT NEVER TOUCHES A GAME ACTION.  This is the real
        # no-RL-on-survival boundary -- not the trace exclusion, which
        # also blocked the harmless proxy/NT half.
        self._propagate_credit(effective, skip_motor=True)

    def _propagate_credit(self, reward: float,
                          skip_motor: bool = False) -> None:
        """Walk the eligibility trace from most recent backward,
        applying discounted reward to each (action, bucket) pair.

        Updates `action_credit[(action, bucket)] += reward × γ^age`,
        clamped to ±CREDIT_CAP.
        """
        gamma = 1.0
        for entry in reversed(self._trace):
            if skip_motor and getattr(entry, 'loop', '') == 'motor':
                self.motor_skipped_body_credit += 1
                continue
            key = (entry.action, entry.chemistry_bucket)
            delta = reward * gamma
            cur = self._action_credit.get(key, 0.0)
            new = cur + delta
            if new > CREDIT_CAP:
                new = CREDIT_CAP
            elif new < -CREDIT_CAP:
                new = -CREDIT_CAP
            self._action_credit[key] = new
            self.credits_applied += 1
            gamma *= DISCOUNT_FACTOR

    # ---- queries (BG calls this at arbitration time) ----

    def credit_for(self,
                      action: str,
                      bucket: Optional[tuple] = None) -> float:
        """Return accumulated credit for an action in current
        (or given) chemistry context.  Returns 0.0 if no
        history.  BG uses this as a small additive bias to the
        score during arbitration."""
        action_kind = self._action_kind(action)
        if not action_kind:
            return 0.0
        if bucket is None:
            bucket = self._current_bucket()
        return self._action_credit.get(
            (action_kind, bucket), 0.0)

    def total_credit_for(self, action: str) -> float:
        """Aggregate credit across all chemistry contexts for an
        action — the "in general, does this action tend to work"
        signal.  Used when context matching is sparse."""
        action_kind = self._action_kind(action)
        if not action_kind:
            return 0.0
        total = 0.0
        for (k, _b), v in self._action_credit.items():
            if k == action_kind:
                total += v
        return total

    # ---- helpers ----

    def _action_kind(self, action: str) -> str:
        """Reduce an action string to its 'kind' (verb part).
        e.g. 'attend_threat:wolf' → 'attend_threat',
        'reflect:fire' → 'reflect', 'speak' → 'speak'."""
        if not action:
            return ''
        if ':' in action:
            return action.split(':', 1)[0]
        return action

    def _current_game(self) -> str:
        """The board he is acting on, or empty when there is none."""
        if self._game_provider is None:
            return ''
        try:
            return str(self._game_provider() or '')
        except Exception:
            return ''

    def _current_chem_raw(self) -> tuple:
        """The raw channel values behind the current bucket, in the same
        stable order, so a stored key can be re-derived exactly."""
        if self._chemistry_provider is None:
            return ()
        try:
            chem = self._chemistry_provider()
            gs = getattr(chem, 'global_state', None) or {}
            from seagi.brain.chemistry_types import CHANNELS
            return tuple(float(gs.get(c, CHANNELS[c]['baseline']))
                         for c in sorted(CHANNELS.keys()))
        except Exception:
            return ()

    def _current_bucket(self) -> tuple:
        """Read current chemistry and bucket it (using the same
        bucketing as ContextKey.chemistry_signature in Phase B,
        for consistency across the architecture)."""
        if self._chemistry_provider is None:
            return ()
        try:
            chem = self._chemistry_provider()
        except Exception:
            return ()
        if chem is None:
            return ()
        try:
            from seagi.core.bubble import compute_chemistry_signature
            return compute_chemistry_signature(chem.global_state)
        except Exception:
            return ()

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'trace_size': len(self._trace),
            'rewards_received': self.rewards_received,
            'punishments_received': self.punishments_received,
            'credits_applied': self.credits_applied,
            'actions_recorded': self.actions_recorded,
            'unique_action_contexts': len(self._action_credit),
        }

    # ---- Phase H.1 (2026-05-17): M/I-weighted persistence ----

    # action_credit is heavily M/I-tagged by construction — it
    # IS the integrated reward signal.  Persist credits whose
    # absolute magnitude has crossed a noise threshold; the rest
    # are dialog-noise.  Eligibility trace itself is genuine
    # working memory — it does NOT persist (recent actions are
    # session-local).
    PERSIST_CREDIT_FLOOR = 0.1   # |c| >= 0.1 → personality-bearing

    def to_dict(self) -> Dict[str, Any]:
        out_credits = []
        for (action, bucket), c in self._action_credit.items():
            if abs(c) < self.PERSIST_CREDIT_FLOOR:
                continue
            out_credits.append({
                'action': action,
                'bucket': list(bucket),
                'credit': float(c),
            })
        return {
            'action_credit': out_credits,
            'rewards_received': self.rewards_received,
            'punishments_received': self.punishments_received,
            'motor_recorded': self.motor_recorded,
            'motor_skipped_body_credit': self.motor_skipped_body_credit,
            'motorskill_on': _MOTORSKILL_ON(),
        }

    def load_from_dict(self, d: Dict[str, Any]) -> None:
        """Restore action_credit from a saved snapshot.  Existing
        in-memory ledger is reset.  Eligibility trace is NOT
        restored — that's session-local working memory."""
        self._action_credit.clear()
        for entry in d.get('action_credit', []):
            action = str(entry.get('action', ''))
            bucket = tuple(entry.get('bucket', ()))
            credit = float(entry.get('credit', 0.0))
            if not action:
                continue
            self._action_credit[(action, bucket)] = max(
                -CREDIT_CAP, min(CREDIT_CAP, credit))
        self.rewards_received = int(d.get('rewards_received', 0))
        self.punishments_received = int(d.get(
            'punishments_received', 0))
