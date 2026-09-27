"""Interoception must scale to the body it is sensing.

MEASURED 2026-08-20 on the live process:
  insula samples_taken 750 -> interoception_emitted 0
  90 direct lifeforce reads / 180 s: largest step 2.45e-6, total swing 8.6e-5
  DELTA_THRESHOLD = 0.05  =>  20,404x larger than his largest step
  band pinned at 'settled' (edges 0.25/0.50, he lives at 0.88)

So nothing downstream of INTEROCEPTION has ever run.  The fix is an adaptive
gate -- k x his own recent mean |delta| -- because a just-noticeable
difference scales with the signal, and a smaller CONSTANT would break again
the moment his dynamics changed.
"""
import pytest
import seagi.brain.capabilities.insula as I


def _gate(scale, n):
    return I.ADAPT_K * scale if n >= I.ADAPT_MIN_SAMPLES else float('inf')


def _run(deltas, alpha=None):
    """Replay a delta sequence through the same EWMA the insula uses."""
    alpha = I.ADAPT_ALPHA if alpha is None else alpha
    scale, n, fired = 0.0, 0, []
    for d in deltas:
        ad = abs(d)
        n += 1
        scale = ad if scale <= 0.0 else (1 - alpha) * scale + alpha * ad
        if ad >= _gate(scale, n):
            fired.append(n)
    return fired, scale


def test_his_measured_dynamics_never_reach_the_fixed_threshold():
    """The premise, pinned: this is why the organ was silent."""
    largest_observed_step = 2.4505e-6
    assert largest_observed_step < I.DELTA_THRESHOLD
    assert I.DELTA_THRESHOLD / largest_observed_step > 10000


def test_steady_drift_at_his_real_scale_does_not_spam():
    """~1.2e-6 per sample, his measured median, must stay quiet."""
    fired, scale = _run([1.2e-6] * 500)
    assert fired == [], 'steady drift is not a shift'
    assert scale == pytest.approx(1.2e-6, rel=1e-6)


def test_a_real_shift_at_his_real_scale_DOES_fire():
    """20x his norm -- invisible to a 0.05 threshold, obvious to him."""
    deltas = [1.2e-6] * 200 + [2.4e-5]
    fired, _ = _run(deltas)
    assert fired == [201], 'a genuine excursion must register'


def test_the_gate_self_calibrates_to_any_scale():
    """The same rule works if his lifeforce dynamics change by orders of
    magnitude -- which is the whole reason not to pick a new constant."""
    for scale in (1e-7, 1e-6, 1e-3, 1e-1):
        deltas = [scale] * 200 + [scale * 20]
        fired, _ = _run(deltas)
        assert fired == [201], 'failed at scale %g' % scale


def test_no_judgement_before_a_scale_exists():
    """An early sample must not fire on a scale estimated from nothing."""
    early = I.ADAPT_MIN_SAMPLES // 5
    fired, _ = _run([1.2e-6] * early + [1.0])
    assert fired == [], 'no scale exists yet -- must not judge'
    # ...but once a scale exists, the same spike DOES register.
    fired, _ = _run([1.2e-6] * I.ADAPT_MIN_SAMPLES + [1.0])
    assert fired == [I.ADAPT_MIN_SAMPLES + 1]


def test_gate_off_is_the_original_behaviour(monkeypatch):
    # FORCE the gate -- never assert on which files exist on the box.
    monkeypatch.setattr(I, '_BODYSCALE_ON', lambda: False)
    assert not I._BODYSCALE_ON()
    assert I.DELTA_THRESHOLD == 0.05
