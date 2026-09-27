"""Being wrong is a signal, and its size is how surprising the error was.

USER 2026-08-22: "being wrong should give the signal to try again, try more,
be curious. it should be the signal, how can I solve this. how can I get it
right. everything always has a signal."

MEASURED before this: 1,336 falsifications in one night, ZERO chemistry
fired.  The falsification branches only incremented counters and cleared
insight banks.  A 65% error rate held steady all night because nothing in him
registered that anything needed solving.
"""
import pytest
import seagi.brain.capabilities.world_actor as W


class _WA:
    """Only the seeking machinery, off the real class."""
    _seek_on_error = W.WorldActor._seek_on_error

    def __init__(self, conf, norm=0.0, n=0):
        self._pred_conf = conf
        self._err_conf = norm
        self._err_conf_n = n
        self.seek_fires = 0
        self.published = []

        class _Bus:
            def __init__(self, outer):
                self.outer = outer

            def publish(self, ev):
                self.outer.published.append(ev)

        self.bus = _Bus(self)


def _on(monkeypatch, val=True):
    monkeypatch.setattr(W, '_ERRSEEK_ON', lambda: val)


def test_being_wrong_now_fires_something(monkeypatch):
    """The whole point: it fired nothing at all before."""
    _on(monkeypatch)
    w = _WA(conf=0.4, norm=0.4, n=100)
    w._seek_on_error('s', 1, 'p', 's2', 5)
    assert w.seek_fires == 1
    assert len(w.published) == 1


def test_what_it_fires_is_SEEKING_not_disappointment(monkeypatch):
    """`falsified_i` lowers dopamine; the ask is "try again, be curious",
    which is the seeking system.  Disappointment would suppress the drive it
    should provoke."""
    _on(monkeypatch)
    w = _WA(conf=0.4, norm=0.4, n=100)
    w._seek_on_error('s', 1, 'p', 's2', 5)
    ev = w.published[0]
    assert ev.chemistry_kind == 'curiosity'
    from seagi.brain.capabilities import chemistry as C
    assert C.EVENT_DELTAS['curiosity']['dopamine'] > 0
    assert C.EVENT_DELTAS['falsified_i']['dopamine'] < 0


def test_confidently_wrong_provokes_a_bigger_search(monkeypatch):
    _on(monkeypatch)
    big = _WA(conf=0.8, norm=0.4, n=100)
    big._seek_on_error('s', 1, 'p', 's2', 5)
    small = _WA(conf=0.1, norm=0.4, n=100)
    small._seek_on_error('s', 1, 'p', 's2', 5)
    # the norm absorbs the current sample before the dose is computed, so a
    # large error slightly damps its own magnitude -- mildly self-limiting
    assert big.published[0].magnitude > 1.9
    assert small.published[0].magnitude < 0.3
    assert big.published[0].magnitude > 6 * small.published[0].magnitude


def test_a_typical_error_is_a_typical_dose(monkeypatch):
    _on(monkeypatch)
    w = _WA(conf=0.4, norm=0.4, n=100)
    w._seek_on_error('s', 1, 'p', 's2', 5)
    assert w.published[0].magnitude == pytest.approx(1.0, rel=0.02)


def test_the_norm_is_learned_even_while_gated_off(monkeypatch):
    """So the size is right the moment the gate opens, not 20 errors later."""
    _on(monkeypatch, False)
    w = _WA(conf=0.5)
    for _ in range(5):
        w._seek_on_error('s', 1, 'p', 's2', 5)
    assert w._err_conf_n == 5
    assert w._err_conf == pytest.approx(0.5)
    assert w.seek_fires == 0          # but nothing fired


def test_cold_start_is_the_plain_event(monkeypatch):
    _on(monkeypatch)
    w = _WA(conf=0.9, norm=0.4, n=5)
    w._seek_on_error('s', 1, 'p', 's2', 5)
    assert w.published[0].magnitude == 1.0


def test_no_confidence_no_signal(monkeypatch):
    """He cannot be surprised by a prediction he never held."""
    _on(monkeypatch)
    w = _WA(conf=0.0, norm=0.4, n=100)
    w._seek_on_error('s', 1, 'p', 's2', 5)
    assert w.seek_fires == 0
