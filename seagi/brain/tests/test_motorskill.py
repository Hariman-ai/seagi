"""Game play may earn SKILLS and NT credit, but never LIFEFORCE credit.

USER DOCTRINE 2026-08-21: "no-RL-on-score doctrine is a misinterpretation of
M/I only. NT tags can work like RL on a meta level and as proxies to M/I."

The standing rule is that game events drive the PROXY layer, never lifeforce.
Chemistry/NT credit IS the proxy layer.  `_record_action` used to drop every
motor entry from the eligibility trace, which blocked the permitted proxy
half along with the forbidden lifeforce half -- so ARC play could never form
a skill, fire one, or earn NT credit.
"""
import pytest
import seagi.brain.capabilities.reward_ledger as R


class _Ev:
    def __init__(self, loop, action='L', cycle=1):
        self.loop = loop
        self.winning_action = action
        self.winning_capability = 'cap'
        self.cycle = cycle


@pytest.fixture
def led():
    l = R.RewardLedger(game_provider=lambda: 'g1')
    l._chemistry_provider = None          # bucket -> ()
    return l


def _on(monkeypatch, val=True):
    monkeypatch.setattr(R, '_MOTORSKILL_ON', lambda: val)


def test_gate_off_still_excludes_game_play(led, monkeypatch):
    """The original behaviour must survive untouched while gated off."""
    _on(monkeypatch, False)
    led._record_action(_Ev('motor'))
    assert len(led._trace) == 0
    assert led.motor_recorded == 0


def test_gate_on_lets_game_play_into_the_trace(led, monkeypatch):
    _on(monkeypatch)
    led._record_action(_Ev('motor'))
    assert len(led._trace) == 1
    assert led._trace[0].loop == 'motor'
    assert led.motor_recorded == 1


def test_a_motor_action_is_keyed_per_game(led, monkeypatch):
    """Cross-game action->effect is 20.53% vs a 30.11% null -- worse than
    chance -- so 25 games must not share one credit key."""
    _on(monkeypatch)
    led._record_action(_Ev('motor'))
    assert led._trace[0].chemistry_bucket == ('g1',)
    led2 = R.RewardLedger(game_provider=lambda: 'g2')
    led2._chemistry_provider = None
    led2._record_action(_Ev('motor'))
    assert led2._trace[0].chemistry_bucket == ('g2',)
    assert led._trace[0].chemistry_bucket != led2._trace[0].chemistry_bucket


def test_non_motor_buckets_are_unchanged(led, monkeypatch):
    """Only motor entries gain the game; internal ones must not shift."""
    _on(monkeypatch)
    led._record_action(_Ev('cognitive'))
    assert led._trace[0].chemistry_bucket == ()


def test_NT_credit_DOES_reach_game_actions(led, monkeypatch):
    """The proxy layer is exactly what the premise wants game events to
    drive."""
    _on(monkeypatch)
    led._record_action(_Ev('motor', action='L'))
    led._propagate_credit(0.01)
    assert led.credit_for('L', ('g1',)) > 0


def test_LIFEFORCE_credit_NEVER_reaches_a_game_action(led, monkeypatch):
    """THE REAL BOUNDARY.  Survival must not become RL on the score."""
    _on(monkeypatch)
    led._record_action(_Ev('motor', action='L'))
    led._record_action(_Ev('cognitive', action='THINK'))
    led._propagate_credit(-0.02, skip_motor=True)
    assert led.credit_for('L', ('g1',)) == 0, 'lifeforce credit hit a game action'
    assert led.credit_for('THINK', ()) < 0, 'internal action should still be credited'
    assert led.motor_skipped_body_credit == 1
