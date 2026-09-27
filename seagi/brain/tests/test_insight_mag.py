"""Insight is a DOSE proportional to what was understood.

USER 2026-08-21: *"achievement is relative. seeing something, understanding
a move, reading the goal, everything he does is a small dose of achievement.
some are bigger than others."*

The deployed rule fired at a flat magnitude=1.0, which is why it could only
be all-or-nothing: it over-fired at 2.56x confirms and was switched off, and
the graded middle had nowhere to live.  Now the magnitude is the settling
key's confidence excess measured against HIS OWN typical excess.
"""
import pytest
import seagi.brain.capabilities.world_actor as W


class _WA:
    """Just the magnitude machinery, off the real class."""
    _insight_magnitude = W.WorldActor._insight_magnitude

    def __init__(self, norm, n, conf, tex):
        self._ins_exc = norm
        self._ins_exc_n = n
        self._pred_conf = conf
        self._tex = tex

    def _tex_conf(self):
        return self._tex


def _on(monkeypatch, val=True):
    import os
    real = os.path.exists
    monkeypatch.setattr(
        os.path, 'exists',
        lambda p: val if p == '/root/SELFCAL_ON' else real(p))


def test_a_typical_understanding_is_a_normal_dose(monkeypatch):
    _on(monkeypatch)
    w = _WA(norm=0.10, n=100, conf=0.40, tex=0.30)   # excess == his norm
    assert w._insight_magnitude() == pytest.approx(1.0)


def test_a_small_understanding_is_a_small_dose(monkeypatch):
    """Seeing something, understanding one move -- these should register,
    but small."""
    _on(monkeypatch)
    w = _WA(norm=0.10, n=100, conf=0.32, tex=0.30)   # excess a fifth of norm
    assert w._insight_magnitude() == pytest.approx(0.2)


def test_a_large_understanding_is_a_large_dose(monkeypatch):
    _on(monkeypatch)
    w = _WA(norm=0.10, n=100, conf=0.48, tex=0.30)   # 1.8x his norm
    assert w._insight_magnitude() == pytest.approx(1.8)


def test_the_dose_is_bounded(monkeypatch):
    """ChemistryEvent clamps at 2.0; do not hand it more."""
    _on(monkeypatch)
    w = _WA(norm=0.01, n=100, conf=0.90, tex=0.30)
    assert w._insight_magnitude() == 2.0


def test_no_excess_is_no_dose(monkeypatch):
    _on(monkeypatch)
    w = _WA(norm=0.10, n=100, conf=0.30, tex=0.30)
    assert w._insight_magnitude() == 0.0


def test_cold_start_falls_back_to_the_old_flat_dose(monkeypatch):
    """Before a norm exists he must not be graded against noise."""
    _on(monkeypatch)
    assert _WA(norm=0.10, n=5, conf=0.48, tex=0.30)._insight_magnitude() == 1.0
    assert _WA(norm=0.0, n=100, conf=0.48, tex=0.30)._insight_magnitude() == 1.0


def test_sizing_is_independent_of_the_chemistry_spread(monkeypatch):
    """`_ins_exc` and `chemistry._dep_spread` are two DIFFERENT measurements.
    Coupling them made a converged insight norm (n=531) wait on an
    unconverged chemistry spread (NE read 0.0195 then 0.00856 across
    restarts).  Each self-calibrating mechanism waits on its own evidence."""
    _on(monkeypatch, False)          # SELFCAL_ON absent
    w = _WA(norm=0.10, n=100, conf=0.48, tex=0.30)
    assert w._insight_magnitude() == pytest.approx(1.8)
    _on(monkeypatch, True)           # and identical with it present
    assert w._insight_magnitude() == pytest.approx(1.8)
