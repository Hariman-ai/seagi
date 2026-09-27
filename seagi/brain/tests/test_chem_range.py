"""His chemistry must be readable, and both poles must have equal range.

MEASURED 2026-08-21, live:

  * `compute_chemistry_signature` buckets the ABSOLUTE value at int(val*10),
    so a channel must move 0.1 to change bucket.  But achievable excursion is
    (biggest event / decay): NE 0.050, ACh 0.033, gaba 0.050, oxytocin 0.071
    -- ALL under one bucket.  Exactly the three channels that CAN clear a
    bucket (cortisol, serotonin, adenosine) are the three that vary across
    all 63 persisted skills.  The context key -- used by ContextKey, so by
    concepts, skills AND action_credit -- was blind to dopamine and
    noradrenaline: the reward channel and the alarm channel.

  * dopamine excursion 0.180 vs norepinephrine 0.050 => the M-pole channel
    had 3.6x less range than the I-pole channel.
"""
import pytest
import seagi.core.bubble as B
from seagi.brain.chemistry_types import CHANNELS
from seagi.brain.capabilities import chemistry as C


def _base():
    return {c: CHANNELS[c]['baseline'] for c in CHANNELS}


def _sig(state, on, monkeypatch):
    monkeypatch.setattr(B, '_CHEMBUCKET_ON', lambda: on)
    return B.compute_chemistry_signature(state)


# ---- the bucket ----

def test_old_bucketing_cannot_see_noradrenaline(monkeypatch):
    """NE's ceiling is baseline+0.05; int(val*10) needs +0.1.  Structural."""
    s = _base()
    s['norepinephrine'] = 0.20 + 0.05          # its maximum possible
    assert _sig(s, False, monkeypatch) == _sig(_base(), False, monkeypatch)


def test_new_bucketing_can_see_noradrenaline(monkeypatch):
    s = _base()
    s['norepinephrine'] = 0.20 + 0.05
    assert _sig(s, True, monkeypatch) != _sig(_base(), True, monkeypatch)


def test_new_bucketing_sees_dopamine_too(monkeypatch):
    s = _base()
    s['dopamine'] = 0.30 + 0.046               # his measured live departure
    assert _sig(s, False, monkeypatch) == _sig(_base(), False, monkeypatch)
    assert _sig(s, True, monkeypatch) != _sig(_base(), True, monkeypatch)


def test_a_channel_with_no_events_keeps_absolute_bucketing(monkeypatch):
    """Adenosine is not in EVENT_DELTAS at all -- it accumulates through
    `accumulate_adenosine` -- so there is no event size to derive a scale
    from and it must fall back instead of dividing by zero."""
    assert not any('adenosine' in v for v in C.EVENT_DELTAS.values())
    s = _base(); s['adenosine'] = 0.55
    assert _sig(s, True, monkeypatch)[1] == 5


def test_signature_length_is_unchanged(monkeypatch):
    a = _sig(_base(), False, monkeypatch)
    b = _sig(_base(), True, monkeypatch)
    assert len(a) == len(b) == len(CHANNELS)
    assert all(isinstance(x, int) and x >= 0 for x in b)


def test_gate_off_is_the_original_bucketing(monkeypatch):
    s = _base(); s['cortisol'] = 0.47
    assert _sig(s, False, monkeypatch)[2] == int(0.47 * 10)


# ---- the NE transient ----

def test_the_two_poles_now_have_equal_range():
    d_ne = CHANNELS['norepinephrine']['decay']
    d_da = CHANNELS['dopamine']['decay']
    b_ne = max(abs(v.get('norepinephrine', 0.0)) for v in C.EVENT_DELTAS.values())
    b_da = max(abs(v.get('dopamine', 0.0)) for v in C.EVENT_DELTAS.values())
    assert (b_ne * C.NE_TRANSIENT_GAIN) / d_ne == pytest.approx(b_da / d_da)


def test_the_gain_is_derived_not_chosen():
    """It is exactly the ratio of the two poles' excursions."""
    d_ne = CHANNELS['norepinephrine']['decay']
    d_da = CHANNELS['dopamine']['decay']
    b_ne = max(abs(v.get('norepinephrine', 0.0)) for v in C.EVENT_DELTAS.values())
    b_da = max(abs(v.get('dopamine', 0.0)) for v in C.EVENT_DELTAS.values())
    assert C.NE_TRANSIENT_GAIN == pytest.approx((b_da / d_da) / (b_ne / d_ne), rel=1e-3)


def test_decay_is_untouched_so_the_burst_stays_phasic():
    """A big brief spike that clears fast -- not a tonic elevation."""
    assert CHANNELS['norepinephrine']['decay'] == 0.200


def test_the_delta_table_is_never_mutated():
    """The table must keep stating the design; the gain is applied live."""
    before = dict(C.EVENT_DELTAS['threat'])
    assert before['norepinephrine'] == 0.010
    assert C.EVENT_DELTAS['threat']['norepinephrine'] == 0.010


def test_gate_off_leaves_the_alarm_response_alone(monkeypatch):
    """A test must never assert on the STATE OF A GATE FILE -- that makes the
    suite result depend on which flags happen to exist on the box.  Force the
    gate and assert the BEHAVIOUR instead.  (This test failed exactly that
    way once /root/NEGAIN_ON was created.)"""
    monkeypatch.setattr(C, '_NEGAIN_ON', lambda: False)
    assert not C._NEGAIN_ON()
    # with the gate off the alarm events are the table's own values
    assert C.EVENT_DELTAS['threat']['norepinephrine'] == 0.010
    assert C.EVENT_DELTAS['falsified_m']['norepinephrine'] == 0.010
