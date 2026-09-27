"""HE SHOULD ALWAYS FEEL, AND WANT TO FEEL GOOD.

USER DOCTRINE 2026-08-29: *"most of the time playing a game and solving
problems step by step just solidify being alive, or confirm a solid
state of existence far away from mortality... he should always feel...
he wants to feel good... they will do anything to change something to
get back to feeling good."*

Measured before a line was written:
  - lifeforce RELAXES TOWARD baseline and NEVER exceeds it (129 tick
    samples: max +0.00000, every movement downward).  There was no
    "good" to return to, so wanting one had no referent.
  - ordinary competence is a real everyday signal, neither rare nor
    universal: **68.6% of L0 steps and 62.0% of L1 steps** are competent
    (n=141,871 / 17,180), against 19.3%/29.1% no-ops and 12.1%/8.9%
    walk-backs.
  - the felt state reached NO decision: every term in `propose` was
    navigational or novelty-based.

These pin the contract.  Lifeforce is untouched -- it is distance from
death and stays the law; this is how he is RIGHT NOW.
"""
import pytest

from seagi.brain.capabilities import world_actor as W
from seagi.world import arc_world as A
from seagi.world.arc_world import ARCWorld


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(ARCWorld, "_felt", 0.0, raising=False)
    monkeypatch.setattr(ARCWorld, "_felt_n", 0, raising=False)
    monkeypatch.setattr(A, "_FELT_ON", lambda: True)
    monkeypatch.setattr(A, "_FELTSTEER_ON", lambda: True)


class _W:
    """Only the felt readout, off the real class."""
    felt = ARCWorld.felt
    felt_low = ARCWorld.felt_low


def _settle(value, n=800):
    """Drive the EWMA the way step() does."""
    ARCWorld._felt = 0.0
    ARCWorld._felt_n = 0
    for _ in range(n):
        ARCWorld._felt += A._FELT_ALPHA * (value - ARCWorld._felt)
        ARCWorld._felt_n += 1


def test_he_has_no_opinion_until_he_has_played():
    """A felt state from four steps would be noise, not a feeling."""
    _settle(1.0, n=10)
    assert _W().felt() is None


def test_competent_steps_make_him_feel_good():
    _settle(1.0)
    f = _W().felt()
    assert f is not None and f > 0.5, f


def test_wasted_steps_make_him_feel_bad():
    _settle(-1.0)
    assert _W().felt() < -0.5


def test_he_can_be_both_which_lifeforce_cannot():
    """The whole point: lifeforce only relaxes toward baseline and never
    exceeds it, so it can express 'less bad' but never 'good'."""
    _settle(1.0)
    good = _W().felt()
    _settle(-1.0)
    bad = _W().felt()
    assert good > 0 > bad


def test_the_measured_mix_settles_positive_but_not_euphoric():
    """68.6% competent / 31.4% waste is an ordinary good day, not a
    Super Bowl.  It must not saturate.

    Interleaved, not blocked: a blocked run ends on 314 consecutive bad
    steps and correctly reads -0.60, which is the EWMA doing its job
    (300 wasted steps SHOULD feel bad) but is not the everyday mix.
    """
    ARCWorld._felt = 0.0
    ARCWorld._felt_n = 0
    for i in range(4000):
        d = 1.0 if ((i * 686) % 1000) < 686 else -1.0
        ARCWorld._felt += A._FELT_ALPHA * (d - ARCWorld._felt)
        ARCWorld._felt_n += 1
    f = _W().felt()
    assert 0.2 < f < 0.6, f


def test_the_gate_off_means_no_feeling_at_all(monkeypatch):
    _settle(1.0)
    monkeypatch.setattr(A, "_FELT_ON", lambda: False)
    assert _W().felt() is None


def test_feeling_good_is_not_a_demand_for_change():
    _settle(1.0)
    assert _W().felt_low() is False


def test_feeling_bad_is_a_demand_for_change():
    _settle(-1.0)
    assert _W().felt_low() is True


def test_the_steer_gate_can_be_shut_without_blinding_him(monkeypatch):
    """He keeps feeling; the feeling just stops ordering anything."""
    _settle(-1.0)
    monkeypatch.setattr(A, "_FELTSTEER_ON", lambda: False)
    assert _W().felt() < -0.5
    assert _W().felt_low() is False


# ---- and the half that acts on it ------------------------------------

class _Actor:
    _felt_change = W.WorldActor._felt_change
    _felt_err = W.WorldActor._felt_err
    _felt_pick = W.WorldActor._felt_pick

    def __init__(self, low, untried=None):
        # the REAL untried_here returns a LIST, not an int
        self._felt_last_action = {}

        class _World:
            def felt_low(_s):
                return low

            def untried_here(_s, n):
                return untried
        self.world = _World()


def test_a_repeat_is_broken_when_he_is_not_doing_well():
    a = _Actor(low=True, untried=[3])
    a._felt_last_action[7] = 2
    assert a._felt_change(7, 2, 6) == 3
    assert a.felt_changes == 1


def test_nothing_changes_while_he_is_doing_well():
    a = _Actor(low=False, untried=[3])
    a._felt_last_action[7] = 2
    assert a._felt_change(7, 2, 6) == 2
    assert getattr(a, "felt_changes", 0) == 0


def test_a_new_action_is_left_alone_even_when_he_feels_bad():
    """He is already changing something; do not change it twice."""
    a = _Actor(low=True, untried=[3])
    a._felt_last_action[7] = 2
    assert a._felt_change(7, 5, 6) == 5


def test_a_state_he_has_never_stood_on_is_left_alone():
    a = _Actor(low=True, untried=[3])
    assert a._felt_change(99, 2, 6) == 2


def test_it_falls_back_to_a_real_alternative_when_nothing_is_untried():
    """'Change something' has to mean something even with no untried."""
    a = _Actor(low=True, untried=None)
    a._felt_last_action[7] = 2
    assert a._felt_change(7, 2, 6) == 3


def test_a_single_action_world_cannot_change_anything():
    a = _Actor(low=True, untried=None)
    a._felt_last_action[7] = 0
    assert a._felt_change(7, 0, 1) == 0


def test_it_remembers_what_he_actually_did():
    a = _Actor(low=False, untried=None)
    a._felt_change(7, 4, 6)
    assert a._felt_last_action[7] == 4


def test_a_broken_world_cannot_break_the_move():
    class _Bad:
        def felt_low(self):
            raise RuntimeError("boom")
    a = _Actor(low=False)
    a.world = _Bad()
    assert a._felt_change(1, 2, 6) == 2
    assert a.felt_errors == 1


def test_the_failure_is_recorded_BY_KIND():
    """A bare `except: n += 1` said 224 things went wrong and nothing
    about what, which cost an evening of hypotheses."""
    class _Bad:
        def felt_low(self):
            raise IndexError("list index out of range")
    a = _Actor(low=False)
    a.world = _Bad()
    a._felt_change(1, 2, 6)
    a._felt_change(1, 2, 6)
    assert a.felt_err_kinds == {"IndexError": 2}


def test_the_funnel_counts_where_it_stops():
    """felt reached -0.46, ten times below the -0.05 threshold, and the
    steer still never fired.  These say which condition blocks it."""
    a = _Actor(low=True, untried=[3])
    a._felt_change(7, 2, 6)              # low, but never stood here
    assert a.felt_low_n == 1
    assert getattr(a, "felt_seen_n", 0) == 0
    a._felt_change(7, 5, 6)              # stood here, different action
    assert a.felt_seen_n == 1
    assert getattr(a, "felt_repeat_n", 0) == 0
    a._felt_change(7, 5, 6)              # stood here, SAME action
    assert a.felt_repeat_n == 1
    assert a.felt_changes == 1


def test_untried_here_returns_a_LIST_not_an_int():
    """The real signature: 'Empty list when everything here has been
    tried'.  The old stub returned an int, encoded the same wrong
    assumption as the code, and passed -- while live it threw TypeError
    2,621 times out of 2,621 attempts."""
    a = _Actor(low=True, untried=[4, 5])
    a._felt_last_action[7] = 2
    assert a._felt_change(7, 2, 6) == 4
    assert a.felt_changes == 1
    assert getattr(a, "felt_errors", 0) == 0


def test_an_empty_list_still_changes_something():
    a = _Actor(low=True, untried=[])
    a._felt_last_action[7] = 2
    assert a._felt_change(7, 2, 6) == 3
    assert a.felt_changes == 1


def test_the_repeated_action_is_never_picked_back():
    a = _Actor(low=True, untried=[2])
    a._felt_last_action[7] = 2
    assert a._felt_change(7, 2, 6) != 2


def test_out_of_range_candidates_are_ignored():
    a = _Actor(low=True, untried=[99, -1, 4])
    a._felt_last_action[7] = 2
    assert a._felt_change(7, 2, 6) == 4


def test_junk_from_the_world_cannot_throw():
    for junk in ("nonsense", {"a": 1}, [None, "x"], 3.7):
        a = _Actor(low=True, untried=junk)
        a._felt_last_action[7] = 2
        out = a._felt_change(7, 2, 6)
        assert isinstance(out, int) and 0 <= out < 6
        assert getattr(a, "felt_errors", 0) == 0, junk
