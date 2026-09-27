"""Skill library — learned action sequences with reliability.

Phase F.9 (2026-05-16).  v1 had `skills.py` + `skill_hierarchy.py`
+ `tool_learning.py`; v2 consolidation dropped them.  The audit
flagged the loss: no library of "what tends to work" sequences,
no way for repeated success to crystallize into a reusable skill.

What this module is
-------------------
A bounded library of `Skill` records.  Each skill is:

  precondition_bucket : the chemistry context where this skill
                          was discovered (same bucket format as
                          RewardLedger / ContextKey).  A skill
                          only fires when the current bucket
                          matches (approximately).

  action_sequence     : a tuple of action_kinds that, in order,
                          have led to positive outcomes.

  expected_reward     : the chemistry_kind that signals
                          success (typically 'confirmed_i' or
                          'mattering').

  firings             : how many times this skill has been
                          attempted.
  successes           : how many times it produced the expected
                          reward shortly after.
  reliability         : successes / firings, computed lazily.

How skills are born
-------------------
The `RewardLedger`'s eligibility trace already captures recent
action sequences in their chemistry context.  After a positive
reward, the most-recent N actions form a candidate sequence.
If the same sequence (or a prefix of it) is seen producing the
same reward kind repeatedly, the SkillLibrary promotes it.

How skills fire
---------------
A future BG deliberate-loop integration consults
`library.candidates_for(current_bucket)`.  Matching skills
contribute a capability claim with urgency proportional to
reliability.  When the skill wins arbitration, its action
sequence becomes the pending plan.

This module ships the LIBRARY + DISCOVERY.  BG integration is
intentionally deferred — the library is queryable and discovery
is automatic; consumers can call `candidates_for(bucket)` from
anywhere.

Doctrine alignment
------------------
- Skills live in working memory, not substrate.  Bounded
  registry.  Skills can be evicted by reliability when capacity
  fills.
- Promille thresholds — single co-occurrence doesn't promote
  a skill; many repetitions do.
- The library doesn't STORE actions taken — it stores patterns
  that have proven reliable.  Most action sequences never get
  promoted; only the ones that produce consistent reward.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


# Maximum skills retained.  Library is bounded; new skills with
# higher reliability evict less-reliable ones when full.
DEFAULT_LIBRARY_CAPACITY = 64

# Minimum sequence length to consider for promotion.  Single
# actions are not skills; they're already in BG's action-kind
# repertoire.  Two-step minimum captures the simplest plans.
MIN_SEQUENCE_LENGTH = 2

# Maximum sequence length captured for a single skill.  Longer
# sequences would need finer-grained matching at firing time.
MAX_SEQUENCE_LENGTH = 5


def _SKILL_LOOP_ON():
    """Let firing/success actually move reliability.  File-gated: with the
    loop closed, skills that fail fall below the 0.5 claim gate and go
    quiet -- which is the loop WORKING, but it should be measured first."""
    try:
        import os as _os
        return _os.path.exists('/root/SKILLLOOP_ON')
    except Exception:
        return False

# Threshold of repeated observation before a candidate becomes a
# real skill.  Three observations = "this is a pattern, not
# coincidence."
PROMOTION_THRESHOLD = 3


def _bucket_version() -> int:
    """1 = absolute int(val*10); 2 = departure scaled to each
    channel's own achievable excursion."""
    try:
        from seagi.core.bubble import _CHEMBUCKET_ON
        return 2 if _CHEMBUCKET_ON() else 1
    except Exception:
        return 1


def _migrate_bucket(bucket, stored_version, chem_raw=None):
    """Re-derive a stored key under the CURRENT scheme.

    A key written under scheme 1 means each position was int(val*10),
    so val sat in [b/10, (b+1)/10).  Re-derive from the midpoint.  No
    silent orphaning: a version mismatch is repaired, and an
    unrecognised one is left alone rather than corrupted.
    """
    cur = _bucket_version()
    have = int(stored_version or 1)
    # EXACT RE-DERIVATION when the raw observation was kept.  This is
    # the path that makes a bucketing change cost nothing: the index is
    # recomputed from what it was derived FROM, not guessed backwards
    # from a coarser index (which recovered only 3 of 9 positions).
    if chem_raw:
        try:
            from seagi.core.bubble import compute_chemistry_signature
            from seagi.brain.chemistry_types import CHANNELS
            order = sorted(CHANNELS.keys())
            if len(chem_raw) == len(order):
                state = {c: float(chem_raw[i])
                         for i, c in enumerate(order)}
                return tuple(compute_chemistry_signature(state))
        except Exception:
            pass
    if not bucket or have == cur:
        return tuple(bucket)
    if have == 1 and cur == 2:
        try:
            from seagi.core.bubble import _scaled_bucket
            from seagi.brain.chemistry_types import CHANNELS
            order = sorted(CHANNELS.keys())
            if len(bucket) != len(order):
                return tuple(bucket)
            out = []
            for i, ch in enumerate(order):
                mid = (float(bucket[i]) + 0.5) / 10.0
                base = CHANNELS[ch].get('baseline', 0.0)
                out.append(_scaled_bucket(ch, mid, base))
            return tuple(out)
        except Exception:
            return tuple(bucket)
    return tuple(bucket)


@dataclass
class Skill:
    """One learned action sequence."""
    id: str
    name: str                     # short human-readable label
    precondition_bucket: tuple    # chemistry signature at discovery
    action_sequence: Tuple[str, ...]
    expected_reward: str          # 'confirmed_i' / 'mattering' / etc.
    firings: int = 0
    successes: int = 0
    last_used_cycle: int = 0
    created_cycle: int = 0
    # THE OBSERVATION THE KEY CAME FROM, so `precondition_bucket` can be
    # re-derived EXACTLY when the bucketing changes.  Empty for skills
    # promoted before 2026-08-21.
    chem_raw: tuple = ()

    def reliability(self) -> float:
        if self.firings <= 0:
            return 0.0
        return self.successes / self.firings

    def shadow_reliability(self) -> float:
        """What reliability WOULD be with the loop closed."""
        f = getattr(self, 'shadow_firings', 0)
        if f <= 0:
            return 0.0
        return getattr(self, 'shadow_successes', 0) / f


class SkillLibrary:
    """Bounded registry of learned action sequences."""

    def __init__(self,
                 capacity: int = DEFAULT_LIBRARY_CAPACITY):
        self.capacity = int(capacity)
        # Index by canonical sequence string for dedup.
        self._skills: Dict[str, Skill] = {}
        self._next_id: int = 1
        # Candidate counter — sequences observed but not yet
        # promoted.  After PROMOTION_THRESHOLD observations, the
        # candidate becomes a real Skill.
        # Keyed by (sequence_tuple, expected_reward, bucket_key).
        self._candidates: Dict[
            Tuple[Tuple[str, ...], str, tuple], int] = defaultdict(int)
        # Diagnostics.
        self.candidates_seen: int = 0
        self.skills_promoted: int = 0
        self.skills_fired: int = 0
        # shuffle-null counters (see null_reward_for)
        self.shadow_null_firings: int = 0
        self.shadow_null_successes: int = 0
        self.skills_succeeded: int = 0

    # ---- queries ----

    def __len__(self) -> int:
        return len(self._skills)

    def all_skills(self) -> List[Skill]:
        return list(self._skills.values())

    def candidates_for(self,
                            current_bucket: tuple,
                            min_reliability: float = 0.5
                            ) -> List[Skill]:
        """Skills whose precondition_bucket approximately matches
        the current chemistry bucket, ranked by reliability.

        Matching is exact on bucket tuples for now — finer
        similarity (e.g. Hamming distance ≤ 1) can come later.
        """
        matches: List[Skill] = []
        for s in self._skills.values():
            if s.precondition_bucket != current_bucket:
                continue
            if s.reliability() < min_reliability:
                continue
            matches.append(s)
        matches.sort(key=lambda x: -x.reliability())
        return matches

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    # ---- discovery ----

    def observe_sequence(self,
                              action_sequence: Tuple[str, ...],
                              expected_reward: str,
                              precondition_bucket: tuple,
                              cycle: int,
                              chem_raw: tuple = ()) -> Optional[Skill]:
        """Called by the reward ledger (or other observer) when a
        coherent action sequence appears to have produced
        `expected_reward`.  Returns a Skill if this observation
        promoted a candidate.
        """
        if len(action_sequence) < MIN_SEQUENCE_LENGTH:
            return None
        if len(action_sequence) > MAX_SEQUENCE_LENGTH:
            action_sequence = action_sequence[:MAX_SEQUENCE_LENGTH]
        key = (action_sequence, expected_reward, precondition_bucket)
        self._candidates[key] += 1
        self.candidates_seen += 1
        count = self._candidates[key]
        if count < PROMOTION_THRESHOLD:
            return None
        # Promote.  Remove from candidates.
        del self._candidates[key]
        return self._promote(
            action_sequence, expected_reward,
            precondition_bucket, cycle, initial_successes=count,
            chem_raw=chem_raw)

    def _promote(self,
                    action_sequence: Tuple[str, ...],
                    expected_reward: str,
                    precondition_bucket: tuple,
                    cycle: int,
                    initial_successes: int = 1,
                    chem_raw: tuple = ()
                    ) -> Skill:
        """Add a new skill to the library.  Capacity-evicts the
        least-reliable existing skill if full."""
        # Dedup — check if this exact skill already exists.
        for s in self._skills.values():
            if (s.action_sequence == action_sequence
                    and s.expected_reward == expected_reward
                    and s.precondition_bucket == precondition_bucket):
                # Already known.  Bump SUCCESSES ONLY -- the firing is
                # counted at execution time by `record_firing`.  Bumping
                # both (the original) made the denominator grow exactly
                # with the numerator, pinning reliability at 1.0 forever
                # for all 405 skills.
                # WITH THE LOOP CLOSED, ALL CREDIT FLOWS THROUGH
                # record_firing / record_success.  Crediting here as well
                # would double-count: the reward that triggered this call
                # ALREADY credited the skill at reward time, and this path
                # adds `initial_successes` again every PROMOTION_THRESHOLD
                # repetitions -- successes would exceed firings and
                # reliability would climb above 1.0.
                # Gate OFF keeps the original behaviour byte-for-byte.
                if not _SKILL_LOOP_ON():
                    s.successes += initial_successes
                    s.firings += initial_successes
                return s
        # Capacity check.
        if len(self._skills) >= self.capacity:
            self._evict_least_reliable()
        sid = f's{self._next_id:04d}'
        self._next_id += 1
        name = '→'.join(a.split(':')[0] for a in action_sequence)
        skill = Skill(
            id=sid, name=name,
            precondition_bucket=precondition_bucket,
            action_sequence=tuple(action_sequence),
            expected_reward=expected_reward,
            firings=initial_successes,
            successes=initial_successes,
            last_used_cycle=cycle,
            created_cycle=cycle,
            chem_raw=tuple(chem_raw or ()),
        )
        self._skills[sid] = skill
        self.skills_promoted += 1
        return skill

    def _evict_least_reliable(self) -> None:
        """Drop the skill with the lowest reliability."""
        if not self._skills:
            return
        weakest = min(
            self._skills.values(), key=lambda s: s.reliability())
        del self._skills[weakest.id]

    # ---- firing feedback ----

    def find_by_sequence(self, action_sequence, precondition_bucket):
        """EVERY skill whose sequence just RAN, by the same match that
        promoted it.  Symmetric with `observe_sequence` by construction.

        Returns a LIST, not one skill: dedup keys on
        (sequence, expected_reward, bucket), so two skills may share a
        sequence and bucket while predicting DIFFERENT rewards.  Returning
        one would give an arbitrary skill all the firings and leave its
        twin permanently uncounted.
        """
        seq = tuple(action_sequence)
        return [s for s in self._skills.values()
                if (s.action_sequence == seq
                    and s.precondition_bucket == precondition_bucket)]

    def _adopt_shadow_units(self) -> None:
        """ONE UNIT, ONE CLOCK (2026-08-20).

        Historical `successes`/`firings` counted REWARDED REPETITIONS -- the
        dedup path bumped them once per PROMOTION_THRESHOLD rewards.  The
        closed loop counts EXECUTIONS, which are far more frequent.  Leaving
        the old stock in the numerator while the denominator moves to the
        faster clock makes reliability decay as N/(N+k) on a pure ARTIFACT:
        every skill drifts toward 0 and falls under the 0.5 candidate gate
        without having got any worse.  That is the "silencing" a correct
        loop must not do.

        So at gate-on, adopt the shadow counters -- they have been measured
        in EXECUTION units all along, and they are the same numbers the
        deploy gate was read from, so the post-enable state equals the
        measurement that justified enabling.

        Skills with no shadow evidence keep their prior untouched.
        """
        if getattr(self, '_loop_migrated', False):
            return
        self._loop_migrated = True
        for s in self._skills.values():
            sf = int(getattr(s, 'shadow_firings', 0) or 0)
            if sf > 0:
                s.firings = sf
                s.successes = int(getattr(s, 'shadow_successes', 0) or 0)

    def null_reward_for(self, kind: str):
        """THE SHUFFLE NULL for skill reliability.

        The shadow says skills succeed 6.2% of the time, but that number is
        unreadable on its own: with only a handful of ARC actions, any
        3-action window recurs constantly by coincidence, so the same reward
        may simply follow ANY window at that rate.

        So for every real firing we also test a WRONG reward -- the next
        distinct expected_reward in sorted order, cyclically.  Deterministic
        (no RNG, so it cannot drift or need seeding) and never equal to the
        true kind whenever two or more distinct kinds exist.  If the real
        rate matches this one, the skills carry no information and the
        defect is the PROMOTION rule, not the reliability floor.
        """
        kinds = sorted({str(x.expected_reward) for x in self._skills.values()
                        if x.expected_reward})
        if len(kinds) < 2:
            return None
        try:
            i = kinds.index(str(kind))
        except ValueError:
            return kinds[0]
        return kinds[(i + 1) % len(kinds)]

    def record_null_firing(self) -> None:
        self.shadow_null_firings += 1

    def record_null_success(self) -> None:
        self.shadow_null_successes += 1

    def record_firing(self, skill_id: str, cycle: int) -> None:
        s = self._skills.get(skill_id)
        if s is None:
            return
        if _SKILL_LOOP_ON():
            self._adopt_shadow_units()
        s.shadow_firings = getattr(s, 'shadow_firings', 0) + 1
        if _SKILL_LOOP_ON():
            s.firings += 1
        s.last_used_cycle = cycle
        self.skills_fired += 1

    def record_success(self, skill_id: str) -> None:
        s = self._skills.get(skill_id)
        if s is None:
            return
        if _SKILL_LOOP_ON():
            self._adopt_shadow_units()
        s.shadow_successes = getattr(s, 'shadow_successes', 0) + 1
        if _SKILL_LOOP_ON():
            s.successes += 1
        self.skills_succeeded += 1

    # ---- Phase G.2 (2026-05-16): emit BG claims ----

    def emit_claims(self,
                       bus: Any,
                       cycle: int,
                       current_bucket: tuple,
                       min_reliability: float = 0.5,
                       cap_per_pass: int = 3) -> int:
        """Publish CapabilityClaim events for skills whose
        precondition_bucket matches the current chemistry context.

        Doctrine (Phase G.2): skills are not just records of "what
        tends to work" — they should COMPETE for arbitration when
        their context fires.  BG considers them alongside cortical
        reflection claims, goal-driven motivational claims, and
        sentinel claims.

        Claim shape:
          loop = 'cognitive' (skills propose what to think/do)
          proposed_action = first action of the skill's sequence
          claim_strength = skill.reliability()
          payload carries the skill_id + full sequence

        Only fires for skills with reliability ≥ min_reliability
        AND precondition matching current_bucket exactly.  Capped
        at cap_per_pass to keep BG arbitration tractable.

        Returns number of claims published.
        """
        if bus is None:
            return 0
        try:
            from ..events import (
                CapabilityClaimEvent, EventKind)
        except Exception:
            return 0
        import time as _time
        matches = self.candidates_for(
            current_bucket, min_reliability)
        count = 0
        for s in matches:
            if count >= cap_per_pass:
                break
            if not s.action_sequence:
                continue
            try:
                bus.publish(CapabilityClaimEvent(
                    kind=EventKind.CAPABILITY_CLAIM,
                    cycle=int(cycle),
                    timestamp=_time.time(),
                    source_capability='skill_library',
                    origin='internal',
                    origin_detail=f'skill:{s.id}',
                    claim_strength=float(s.reliability()),
                    proposed_action=s.action_sequence[0],
                    loop='cognitive',
                    payload={'skill_id': s.id,
                                'full_sequence':
                                    list(s.action_sequence)}))
                count += 1
            except Exception:
                pass
        return count

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'skills': len(self._skills),
            'candidates': len(self._candidates),
            'candidates_seen': self.candidates_seen,
            'skills_promoted': self.skills_promoted,
            'skills_fired': self.skills_fired,
            'skills_succeeded': self.skills_succeeded,
        }

    # ---- Phase H.1 (2026-05-17): M/I-weighted persistence ----

    # A skill is "felt enough to matter for personality" when it
    # has demonstrated reliability AND been exercised enough times
    # that the demonstration isn't a fluke.  Candidates and
    # unfired skills don't persist — they're session-local.
    PERSIST_RELIABILITY_FLOOR = 0.5
    PERSIST_MIN_FIRINGS = 5

    def _persist_ok(self, s: Skill) -> bool:
        return (s.firings >= self.PERSIST_MIN_FIRINGS
                and s.reliability() >= self.PERSIST_RELIABILITY_FLOOR)

    def to_dict(self) -> Dict[str, Any]:
        out: List[Dict[str, Any]] = []
        for s in self._skills.values():
            if not self._persist_ok(s):
                continue
            out.append({
                'shadow_firings': getattr(s, 'shadow_firings', 0),
                'shadow_successes': getattr(s, 'shadow_successes', 0),
                'id': s.id, 'name': s.name,
                'precondition_bucket': list(s.precondition_bucket),
                # WHICH DERIVATION PRODUCED THIS KEY.  Without it a
                # change to compute_chemistry_signature orphans every
                # stored bucket in silence.
                'bucket_version': _bucket_version(),
                'chem_raw': list(getattr(s, 'chem_raw', ()) or ()),
                'action_sequence': list(s.action_sequence),
                'expected_reward': s.expected_reward,
                'firings': s.firings, 'successes': s.successes,
                'last_used_cycle': s.last_used_cycle,
                'created_cycle': s.created_cycle,
            })
        return {
            'next_id': self._next_id,
            'skills': out,
            # THE DEPLOY GATE FOR SKILLLOOP_ON (2026-08-20).  Counted over
            # ALL skills, not just the persisted ones above.  With the loop
            # closed `firings` comes ONLY from record_firing, and
            # _persist_ok demands >= 5 -- so if the execution-time match is
            # rare, NO skill clears the bar and the next save writes an
            # EMPTY library, permanently.  Enable only when
            # shadow_eligible approaches persist_eligible.
            'persist_eligible': sum(
                1 for x in self._skills.values() if self._persist_ok(x)),
            'shadow_eligible': sum(
                1 for x in self._skills.values()
                if getattr(x, 'shadow_firings', 0) >= self.PERSIST_MIN_FIRINGS
                and x.shadow_reliability() >= self.PERSIST_RELIABILITY_FLOOR),
            'shadow_any_firing': sum(
                1 for x in self._skills.values()
                if getattr(x, 'shadow_firings', 0) > 0),
            'shadow_total_firings': sum(
                int(getattr(x, 'shadow_firings', 0))
                for x in self._skills.values()),
            'shadow_total_successes': sum(
                int(getattr(x, 'shadow_successes', 0))
                for x in self._skills.values()),
            'skills_held': len(self._skills),
            'skills_fired': self.skills_fired,
            'shadow_null_firings': self.shadow_null_firings,
            'shadow_null_successes': self.shadow_null_successes,
            'loop_migrated': bool(getattr(self, '_loop_migrated', False)),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SkillLibrary':
        lib = cls()
        lib._next_id = int(d.get('next_id', 1))
        # Adopt-once must not re-run after a restart.
        lib._loop_migrated = bool(d.get('loop_migrated', False))
        for sd in d.get('skills', []):
            sid = str(sd.get('id', ''))
            if not sid:
                continue
            s = Skill(
                id=sid,
                name=str(sd.get('name', '')),
                precondition_bucket=_migrate_bucket(
                    tuple(sd.get('precondition_bucket', ())),
                    sd.get('bucket_version'),
                    tuple(sd.get('chem_raw', ()) or ())),
                chem_raw=tuple(sd.get('chem_raw', ()) or ()),
                action_sequence=tuple(sd.get('action_sequence', ())),
                expected_reward=str(sd.get('expected_reward', '')),
                firings=int(sd.get('firings', 0)),
                successes=int(sd.get('successes', 0)),
                last_used_cycle=int(sd.get('last_used_cycle', 0)),
                created_cycle=int(sd.get('created_cycle', 0)),
            )
            # The shadow must survive a restart or the gate measurement
            # silently restarts from zero every ~90s deploy.
            s.shadow_firings = int(sd.get('shadow_firings', 0) or 0)
            s.shadow_successes = int(sd.get('shadow_successes', 0) or 0)
            lib._skills[sid] = s
        return lib
