"""He must carry over what each action DOES, per game.

MEASURED LIVE 2026-08-21: dead_actions 986 / executions 3,512 = 28.1% of his
actions change nothing.  arc_world's own comment says 29% over 2,400
transitions; 08-13 measured 18.56% over n=15,662.  Those no-ops are 92.0%
predictable FROM THE ACTION ALONE (vs an 84.9% base rate), so (game, action)
is the right grain -- and it is the index THE RECURRENCE LAW says earns,
while everything keyed on his glance earned zero.

He already carries click_hits, paid, route, svalue, rules and self-locus, all
keyed by game.  He did NOT carry what each action does: `dead_actions` was a
bare counter that remembered none of the 986.
"""
import pytest
import seagi.world.arc_world as m


def _w(game_id='g1', table=None):
    w = m.ARCWorld.__new__(m.ARCWorld)
    w.game_id = game_id
    w._act_effect = dict(table or {})
    return w


def test_round_trip_preserves_the_table():
    w = _w(table={1: [10, 0], 2: [12, 9]})
    back = _w()
    assert back.act_effect_from_dict(w.act_effect_to_dict()) == 2
    assert back._act_effect == {1: [10, 0], 2: [12, 9]}


def test_a_different_game_imports_nothing():
    """Cross-game action->effect is REFUTED (20.53% vs a 30.11% null), so
    merging games would import noise, not knowledge."""
    w = _w('g1', {1: [10, 0]})
    other = _w('OTHER')
    assert other.act_effect_from_dict(w.act_effect_to_dict()) == 0
    assert other._act_effect == {}


def test_proved_inert_needs_enough_evidence():
    """3 tries proving nothing is not proof; 10 is."""
    assert _w(table={3: [3, 0]}).act_effect_report()['act_proved_inert'] == []
    assert _w(table={3: [10, 0]}).act_effect_report()['act_proved_inert'] == [3]


def test_an_action_that_sometimes_works_is_never_inert():
    r = _w(table={2: [40, 9]}).act_effect_report()
    assert r['act_proved_inert'] == []
    assert r['act_would_suppress'] == 0


def test_the_report_counts_what_a_steer_would_have_cost_him():
    r = _w(table={1: [10, 0], 2: [12, 9], 3: [3, 0]}).act_effect_report()
    assert r['act_proved_inert'] == [1]
    assert r['act_would_suppress'] == 10      # only the proved-inert action
    assert r['act_tried'] == 25 and r['act_changed'] == 9
    assert r['act_dead_share'] == pytest.approx(0.64)


def test_empty_is_reported_as_no_data_not_as_zero():
    r = _w().act_effect_report()
    assert r['act_dead_share'] is None
    assert r['act_keys'] == 0


def test_malformed_rows_are_skipped_not_fatal():
    back = _w()
    n = back.act_effect_from_dict({'g1': {'x': [1, 2], '4': 'bad', '5': [7, 3]}})
    assert n == 1 and back._act_effect == {5: [7, 3]}


def test_a_missing_key_restores_nothing():
    back = _w()
    assert back.act_effect_from_dict({}) == 0
    assert back.act_effect_from_dict(None) == 0
