"""The M/I projection and the mortality pole.

Two repairs, 2026-08-20:

1. `derive_mi_from_trace` measured departures from the INNATE baselines
   while his chemistry rests at the EARNED (allostatic) set-points, so every
   channel sat permanently above reference and every concept tagged in a
   period carried the same DC offset.  A tag identical for every experience
   cannot differentiate experiences.

2. `confirmed_m` / `falsified_m` were defined in layer6.EVENT_DELTAS and had
   NO chemistry and NO emitter, so a mortality expectation could never be
   confirmed or refuted -- while `confirmed_i` / `falsified_i` both fired.
"""
import pytest
from seagi.core import layer6 as l6
from seagi.brain.capabilities import chemistry as chem


# ---- 1. the reference baseline ----

def _trace(**kw):
    vals = dict(l6.BASELINES)
    vals.update(kw)
    return l6.TransmitterState(**vals)


def test_gate_off_uses_the_innate_constants(monkeypatch):
    monkeypatch.setattr(l6, '_MIBASE_ON', lambda: False)
    l6.set_baseline_provider(lambda: {'cortisol': 0.9})
    assert l6.reference_baselines() is l6.BASELINES
    l6.set_baseline_provider(None)


def test_a_chronically_raised_resting_level_stops_inflating_every_tag(monkeypatch):
    """THE POINT.  He rests at cortisol 0.19 but the innate constant is
    0.10, so against the innate reference EVERY tag carries +0.09 of M that
    says nothing about the experience."""
    resting = dict(l6.BASELINES, cortisol=0.19)
    at_rest = _trace(cortisol=0.19)

    monkeypatch.setattr(l6, '_MIBASE_ON', lambda: False)
    l6.set_baseline_provider(lambda: resting)
    innate = l6.derive_mi_from_trace(at_rest)

    monkeypatch.setattr(l6, '_MIBASE_ON', lambda: True)
    earned = l6.derive_mi_from_trace(at_rest)
    l6.set_baseline_provider(None)

    assert innate.m > 0.08, 'innate reference invents M from resting level'
    assert earned.m == pytest.approx(0.0), 'at rest is NOT a mortality event'


def test_a_real_departure_still_registers(monkeypatch):
    """Fixing the reference must not deafen him -- a genuine spike above
    where he RESTS still reads as M."""
    resting = dict(l6.BASELINES, cortisol=0.19)
    monkeypatch.setattr(l6, '_MIBASE_ON', lambda: True)
    l6.set_baseline_provider(lambda: resting)
    spike = l6.derive_mi_from_trace(_trace(cortisol=0.29))
    l6.set_baseline_provider(None)
    assert spike.m == pytest.approx(0.10, abs=1e-6)


def test_reference_falls_back_when_the_provider_fails(monkeypatch):
    monkeypatch.setattr(l6, '_MIBASE_ON', lambda: True)
    l6.set_baseline_provider(lambda: (_ for _ in ()).throw(RuntimeError()))
    assert l6.reference_baselines() is l6.BASELINES
    l6.set_baseline_provider(lambda: {})
    assert l6.reference_baselines() is l6.BASELINES
    l6.set_baseline_provider(None)


# ---- 2. the mortality pole ----

def test_the_mortality_pole_has_a_chemistry_at_last():
    for k in ('confirmed_m', 'falsified_m'):
        assert k in chem.EVENT_DELTAS, '%s was a silent no-op' % k
        assert k in l6.EVENT_DELTAS


def test_foreseen_harm_settles_him_and_unforeseen_harm_alarms_him():
    """Not a typo: the SAME harm, predicted vs not, moves cortisol in
    OPPOSITE directions.  That is the predictability effect on the HPA
    axis, and it is the whole content of the mortality pole."""
    c = chem.EVENT_DELTAS['confirmed_m']
    f = chem.EVENT_DELTAS['falsified_m']
    assert c['cortisol'] < 0 < f['cortisol']
    assert c['serotonin'] > 0 > f['serotonin']
    assert f['norepinephrine'] > 0 > c['norepinephrine']


def test_being_wrong_about_harm_costs_more_than_being_right():
    f = chem.EVENT_DELTAS['falsified_m']
    c = chem.EVENT_DELTAS['confirmed_m']
    assert abs(f['cortisol']) > abs(c['cortisol'])


def test_both_poles_can_now_be_confirmed_and_refuted():
    """The asymmetry that made this half a substrate."""
    for k in ('confirmed_i', 'falsified_i', 'confirmed_m', 'falsified_m'):
        assert k in chem.EVENT_DELTAS


def test_the_m_pole_deltas_actually_move_the_m_score():
    """A kind whose deltas cancel in the projection would be decoration."""
    def poles(d):
        m = sum(w * max(0.0, d.get(t, 0.0))
                for t, w in l6.M_TRANSMITTERS.items())
        return m
    assert poles(chem.EVENT_DELTAS['falsified_m']) > 0.0
    assert poles(chem.EVENT_DELTAS['confirmed_m']) == 0.0
