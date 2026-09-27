"""He calibrates his own channels; no constant a person chose.

USER DOCTRINE 2026-08-21: *"under the M/I theory where Seagi tries to avoid
to die, he should automatically calibrate his nt tags, and hormone
distribution."*

A channel that carries no information cannot warn him.  Biology solves this
with receptor up/down-regulation -- the channel adjusts its own sensitivity.
SEAGI has no evolutionary history to inherit a calibration from, so he must
find it within one lifetime.  These pin that the calibration comes from HIS
MEASURED EXPERIENCE, not from the config table and not from me.
"""
import pytest
import seagi.core.bubble as B
import seagi.core.layer6 as L
from seagi.brain.chemistry_types import CHANNELS

SPREAD = {'norepinephrine': 0.020, 'cortisol': 0.090, 'dopamine': 0.045,
          'oxytocin': 0.035, 'endorphins': 0.070, 'serotonin': 0.120,
          'acetylcholine': 0.020, 'gaba': 0.030, 'adenosine': 0.100}


def _base():
    return {c: CHANNELS[c]['baseline'] for c in CHANNELS}


class _T:
    pass


def _trace(**kw):
    t = _T()
    for c, v in _base().items():
        setattr(t, c, v)
    for k, v in kw.items():
        setattr(t, k, v)
    return t


# ---- the M/I projection ----

def test_one_sigma_counts_the_same_on_every_channel(monkeypatch):
    """THE POINT.  On raw departures cortisol (~0.09) drowned NE (~0.02) and
    the M pole was cortisol-only.  Normalised, what remains is the DESIGNED
    weight ratio (1.0 / 0.7), not an accident of swing size."""
    monkeypatch.setattr(L, '_mi_selfcal', lambda: SPREAD)
    monkeypatch.setattr(L, 'reference_baselines', _base)
    b = _base()
    ne = L.derive_mi_from_trace(
        _trace(norepinephrine=b['norepinephrine'] + SPREAD['norepinephrine']))
    co = L.derive_mi_from_trace(
        _trace(cortisol=b['cortisol'] + SPREAD['cortisol']))
    ratio = co.m / ne.m
    expected = L.M_TRANSMITTERS['cortisol'] / L.M_TRANSMITTERS['norepinephrine']
    assert ratio == pytest.approx(expected, rel=0.01)


def test_raw_departures_are_cortisol_dominated(monkeypatch):
    """The state it replaces: same two moves, wildly unequal contribution."""
    monkeypatch.setattr(L, '_mi_selfcal', lambda: None)
    monkeypatch.setattr(L, 'reference_baselines', _base)
    b = _base()
    ne = L.derive_mi_from_trace(
        _trace(norepinephrine=b['norepinephrine'] + SPREAD['norepinephrine']))
    co = L.derive_mi_from_trace(
        _trace(cortisol=b['cortisol'] + SPREAD['cortisol']))
    assert co.m / ne.m > 5.0


def test_a_channel_with_no_measured_spread_falls_back(monkeypatch):
    monkeypatch.setattr(L, '_mi_selfcal', lambda: {'cortisol': 0.0})
    monkeypatch.setattr(L, 'reference_baselines', _base)
    b = _base()
    v = L.derive_mi_from_trace(_trace(cortisol=b['cortisol'] + 0.09))
    assert v.m > 0.0


# ---- the bucketing ----

def test_bucket_width_is_his_measured_spread(monkeypatch):
    """One typical departure = one bucket, whatever the channel's scale."""
    monkeypatch.setattr(B, '_CHEMBUCKET_ON', lambda: True)
    monkeypatch.setattr(B, '_lived_spread', lambda ch: SPREAD.get(ch))
    b = _base()
    at_rest = B.compute_chemistry_signature(b)
    one_sigma = dict(b)
    one_sigma['norepinephrine'] += SPREAD['norepinephrine']
    moved = B.compute_chemistry_signature(one_sigma)
    i = sorted(CHANNELS).index('norepinephrine')
    assert moved[i] == at_rest[i] + 1


def test_the_same_holds_for_a_channel_of_a_different_scale(monkeypatch):
    """Self-calibration means scale-independence, so cortisol behaves the
    same as noradrenaline despite a 4.5x larger spread."""
    monkeypatch.setattr(B, '_CHEMBUCKET_ON', lambda: True)
    monkeypatch.setattr(B, '_lived_spread', lambda ch: SPREAD.get(ch))
    b = _base()
    at_rest = B.compute_chemistry_signature(b)
    moved = dict(b)
    moved['cortisol'] += SPREAD['cortisol']
    sig = B.compute_chemistry_signature(moved)
    i = sorted(CHANNELS).index('cortisol')
    assert sig[i] == at_rest[i] + 1


def test_gate_off_uses_no_measurement(monkeypatch):
    monkeypatch.setattr(B, '_SELFCAL_ON', lambda: False)
    B.set_spread_provider(lambda: SPREAD)
    assert B._lived_spread('cortisol') is None
    B.set_spread_provider(None)
