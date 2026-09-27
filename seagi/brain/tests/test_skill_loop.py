"""The skill reliability loop must be SYMMETRIC.

Pins the five defects found on 2026-08-20 while verifying the loop as
first written -- every one of which would have made the measurement lie:

  1. success double-counted (pending path + dedup path) -> reliability > 1.0
  2. the shadow replicated the original bug (dedup bumped BOTH shadow
     counters), so shadow reliability was pinned at 1.0 exactly like the
     real one and would have shown nothing
  3. a single `_pending_skill` slot dropped every skill but the most recent
  4. `find_by_sequence` returned ONE skill, so a twin sharing sequence and
     bucket but predicting a different reward never got a firing
  5. firing detection required 3 traced actions while promotion accepts 2
"""
import os
import pytest
from seagi.brain.capabilities.skill_library import SkillLibrary, Skill

BUCKET = ('neutral', 'mid')
OTHER = ('distress', 'low')


def _mk(lib, sid, seq, bucket, reward):
    s = Skill(id=sid, name=sid, precondition_bucket=bucket,
              action_sequence=tuple(seq), expected_reward=reward,
              created_cycle=0)
    lib._skills[sid] = s
    return s


@pytest.fixture
def lib():
    return SkillLibrary()


def test_shadow_reliability_falls_when_a_firing_earns_nothing(lib):
    """THE WHOLE POINT: a sequence that RAN and did not pay must show."""
    s = _mk(lib, 'a', ['L', 'R', 'L'], BUCKET, 'curiosity')
    for _ in range(4):
        lib.record_firing('a', 1)
    lib.record_success('a')
    assert (s.shadow_firings, s.shadow_successes) == (4, 1)
    assert s.shadow_reliability() == pytest.approx(0.25)


def test_shadow_is_not_pinned_at_one_by_the_dedup_path(lib):
    """Defect 2 -- the shadow must not advance its own denominator."""
    _mk(lib, 'a', ['L', 'R', 'L'], BUCKET, 'curiosity')
    for _ in range(4):
        lib.record_firing('a', 1)
    lib._promote(('L', 'R', 'L'), 'curiosity', BUCKET, 5, 3)
    s = lib._skills['a']
    assert s.shadow_firings == 4, 'dedup must NOT advance the denominator'
    assert s.shadow_reliability() < 1.0


def test_gate_off_leaves_the_real_counters_untouched(lib, monkeypatch):
    """Shadowing must be inert on the live counters."""
    # FORCE the gate -- never assert on which files exist on the box.
    import seagi.brain.capabilities.skill_library as _m
    monkeypatch.setattr(_m, '_SKILL_LOOP_ON', lambda: False)
    s = _mk(lib, 'a', ['L', 'R', 'L'], BUCKET, 'curiosity')
    f0, x0 = s.firings, s.successes
    for _ in range(5):
        lib.record_firing('a', 1)
    lib.record_success('a')
    assert (s.firings, s.successes) == (f0, x0)


def test_gate_off_preserves_the_original_dedup_behaviour(lib):
    """The old path must survive byte-for-byte while the gate is off."""
    s = _mk(lib, 'a', ['L', 'R', 'L'], BUCKET, 'curiosity')
    lib._promote(('L', 'R', 'L'), 'curiosity', BUCKET, 5, 3)
    assert (s.firings, s.successes) == (3, 3)


def test_reliability_can_never_exceed_one(lib):
    """Defect 1 -- pending path and dedup path both credited one reward."""
    _mk(lib, 'a', ['L', 'R', 'L'], BUCKET, 'curiosity')
    for _ in range(3):
        lib.record_firing('a', 1)
        lib.record_success('a')
    lib._promote(('L', 'R', 'L'), 'curiosity', BUCKET, 9, 3)
    s = lib._skills['a']
    assert s.shadow_successes <= s.shadow_firings
    assert s.shadow_reliability() <= 1.0


def test_find_by_sequence_matches_the_promotion_rule(lib):
    """Firing and success must be defined by ONE rule."""
    _mk(lib, 'a', ['L', 'R', 'L'], BUCKET, 'curiosity')
    assert [s.id for s in lib.find_by_sequence(('L', 'R', 'L'), BUCKET)] == ['a']
    assert lib.find_by_sequence(('L', 'R', 'L'), OTHER) == []
    assert lib.find_by_sequence(('L', 'R', 'R'), BUCKET) == []


def test_twins_sharing_a_sequence_both_get_the_firing(lib):
    """Defect 4 -- dedup keys on (sequence, reward, bucket), so two skills
    may share sequence+bucket and predict different rewards.  Returning one
    would leave its twin permanently uncounted."""
    _mk(lib, 'a', ['L', 'R', 'L'], BUCKET, 'curiosity')
    _mk(lib, 'b', ['L', 'R', 'L'], BUCKET, 'mattering')
    found = lib.find_by_sequence(('L', 'R', 'L'), BUCKET)
    assert {s.id for s in found} == {'a', 'b'}


def test_many_skills_may_be_pending_at_once():
    """Defect 3 -- one slot meant a skill whose reward arrived after a
    LATER skill fired could never be credited."""
    from seagi.brain.capabilities.reward_ledger import RewardLedger
    rl = RewardLedger()
    rl._pending_skills = {'a': 'curiosity', 'b': 'mattering'}
    assert len(rl._pending_skills) == 2


# ---- gate-ON semantics (unit consistency) ----

class _GateOn:
    """Turn the file gate on for one test without touching /root."""
    def __enter__(self):
        import seagi.brain.capabilities.skill_library as m
        self._orig = m._SKILL_LOOP_ON
        m._SKILL_LOOP_ON = lambda: True
        return self

    def __exit__(self, *a):
        import seagi.brain.capabilities.skill_library as m
        m._SKILL_LOOP_ON = self._orig
        return False


def test_gate_on_adopts_shadow_units_not_the_old_stock(lib):
    """Historical successes counted REWARDED REPETITIONS; the closed loop
    counts EXECUTIONS.  Mixing them decays reliability as N/(N+k) on a pure
    artifact -- every skill drifts under the 0.5 gate without getting worse.
    At gate-on the shadow counters (already in execution units) are adopted.
    """
    s = _mk(lib, 'a', ['L', 'R', 'L'], BUCKET, 'curiosity')
    s.firings, s.successes = 9, 9          # old units, reliability 1.0
    for _ in range(10):
        lib.record_firing('a', 1)          # shadow only, gate off
    for _ in range(8):
        lib.record_success('a')
    assert (s.firings, s.successes) == (9, 9), 'gate off stays inert'
    with _GateOn():
        lib.record_firing('a', 2)
    assert (s.firings, s.successes) == (11, 8), 'adopted 10/8, then +1 firing'
    assert s.reliability() == pytest.approx(8 / 11)


def test_adopt_is_idempotent_and_survives_a_restart(lib):
    s = _mk(lib, 'a', ['L', 'R', 'L'], BUCKET, 'curiosity')
    s.firings, s.successes = 9, 9
    for _ in range(6):
        lib.record_firing('a', 1)
    for _ in range(3):
        lib.record_success('a')
    with _GateOn():
        lib.record_firing('a', 2)
        lib.record_firing('a', 3)
    assert s.firings == 8 and s.successes == 3, 'adopted once, not twice'
    d = lib.to_dict()
    assert d['loop_migrated'] is True
    assert SkillLibrary.from_dict(d)._loop_migrated is True


def test_a_skill_with_no_shadow_evidence_keeps_its_prior(lib):
    """Untested skills must not be zeroed by the migration."""
    s = _mk(lib, 'a', ['L', 'R', 'L'], BUCKET, 'curiosity')
    s.firings, s.successes = 9, 9
    b = _mk(lib, 'b', ['U', 'D'], OTHER, 'mattering')
    b.firings, b.successes = 7, 7
    for _ in range(6):
        lib.record_firing('a', 1)
    with _GateOn():
        lib.record_firing('a', 2)
    assert (b.firings, b.successes) == (7, 7), 'no shadow evidence -> untouched'
    assert b.reliability() == 1.0


# ---- the shuffle null ----

def test_the_null_never_picks_the_true_reward(lib):
    """6.2% success is unreadable without a null: with few ARC actions, any
    3-action window recurs by coincidence, so the same reward may follow ANY
    window at that rate."""
    for i, k in enumerate(('curiosity', 'mattering', 'confirmed_i')):
        _mk(lib, str(i), ['L', 'R'], BUCKET, k)
    for k in ('curiosity', 'mattering', 'confirmed_i'):
        assert lib.null_reward_for(k) != k


def test_the_null_is_deterministic(lib):
    """No RNG: it cannot drift between runs or need seeding."""
    for i, k in enumerate(('curiosity', 'mattering', 'confirmed_i')):
        _mk(lib, str(i), ['L', 'R'], BUCKET, k)
    first = [lib.null_reward_for(k)
             for k in ('curiosity', 'mattering', 'confirmed_i')]
    for _ in range(5):
        assert [lib.null_reward_for(k)
                for k in ('curiosity', 'mattering', 'confirmed_i')] == first


def test_the_null_is_disabled_when_it_could_not_be_wrong(lib):
    """With a single reward kind there IS no wrong answer -- returning the
    true kind would manufacture a 100% null."""
    _mk(lib, 'a', ['L', 'R'], BUCKET, 'curiosity')
    assert lib.null_reward_for('curiosity') is None


def test_null_counters_are_independent_of_the_real_ones(lib):
    _mk(lib, 'a', ['L', 'R', 'L'], BUCKET, 'curiosity')
    lib.record_firing('a', 1)
    lib.record_null_firing()
    lib.record_null_success()
    s = lib._skills['a']
    assert (lib.shadow_null_firings, lib.shadow_null_successes) == (1, 1)
    # the shadow fields are set lazily, as the rest of the module reads them
    assert (getattr(s, 'shadow_firings', 0),
            getattr(s, 'shadow_successes', 0)) == (1, 0)
    assert s.shadow_reliability() == 0.0
