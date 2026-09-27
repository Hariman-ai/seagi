"""His calibrations survive a restart; his mood need not.

USER 2026-08-21: *"He shouldn't have to relearn who he is after every
deploy."*

His chemistry `global_state` is deliberately NOT persisted -- nobody wakes in
yesterday's exact mood, and only `allostatic_load` (the slow scar) crosses
the boundary.  But a NORM is not a mood: `_ins_exc` is built over hundreds of
settlings and `_dep_spread` over thousands of ticks.  Without them he wakes
unable to tell a large understanding from a small one, or to read a channel
at its own scale.
"""
import pytest
import seagi.brain.capabilities.world_actor as W


class _Stub:
    """Only the learning-persistence surface."""
    learning_to_dict = W.WorldActor.learning_to_dict
    learning_from_dict = W.WorldActor.learning_from_dict

    def __init__(self):
        self._skeys = {}
        self._vk = {}
        self._trans_world = {}
        self._cval = {}
        self._ins_exc = 0.0
        self._ins_exc_n = 0


def test_the_insight_norm_survives_a_restart():
    a = _Stub()
    a._ins_exc, a._ins_exc_n = 0.5860, 531
    b = _Stub()
    b.learning_from_dict(a.learning_to_dict())
    assert b._ins_exc == pytest.approx(0.5860)
    assert b._ins_exc_n == 531


def test_a_fresh_life_starts_without_one():
    b = _Stub()
    b.learning_from_dict({})
    assert b._ins_exc == 0.0 and b._ins_exc_n == 0


def test_a_malformed_norm_leaves_him_as_a_fresh_life_not_dead():
    """The standard learning_from_dict already sets: never fatal."""
    b = _Stub()
    b.learning_from_dict({'ins_exc': 'not-a-number', 'ins_exc_n': 'x'})
    assert b._ins_exc == 0.0


def test_a_zero_norm_is_not_restored_over_a_live_one():
    b = _Stub()
    b._ins_exc, b._ins_exc_n = 0.42, 100
    b.learning_from_dict({'ins_exc': 0.0, 'ins_exc_n': 0})
    assert b._ins_exc == pytest.approx(0.42)


def test_the_norm_is_what_sizes_the_dose():
    """Why it must persist: without it every insight is a flat 1.0 again."""
    class _M:
        _insight_magnitude = W.WorldActor._insight_magnitude

        def __init__(self, norm, n):
            self._ins_exc, self._ins_exc_n = norm, n
            self._pred_conf, self._tex = 0.48, 0.30

        def _tex_conf(self):
            return self._tex

    assert _M(0.10, 100)._insight_magnitude() == pytest.approx(1.8)
    assert _M(0.0, 0)._insight_magnitude() == 1.0        # relearning


def test_a_missing_attribute_silently_voids_ALL_learning():
    """`learning_to_dict` swallows every exception and returns {}.  So one
    missing attribute does not lose one field -- it loses skeys, vk, twld and
    the norms together, with no error.  Pinned so the hazard is visible."""
    class _Broken(_Stub):
        def __init__(self):
            _Stub.__init__(self)
            del self._cval
    assert _Broken().learning_to_dict() == {}
