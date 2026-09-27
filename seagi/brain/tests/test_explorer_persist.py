"""HE MUST NOT FORGET WHICH THING HE IS EVERY TIME HE RESTARTS.

ARCWorld._explorers was a class dict that nothing saved or loaded, while
IDENT_STEPS=400 frames must be spent per game before a self can be named.
Measured 2026-08-28: 8,508 arc steps over 25 games in 1h13m = 340
frames/game against that 400 budget, with restart intervals of 333s,
817s, 922s, 1695s, 2974s, 2982s, 5327s and 5918s -- so he was usually
killed before he finished identifying himself, and began again from
nothing.

These pin the contract, not the numbers.
"""
import json
import os
import tempfile

import pytest

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld
from seagi.world.explorer import Explorer


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    p = str(tmp_path / "explorer_state.json")
    monkeypatch.setattr(arc_world, "_EXPLORER_STATE_PATH", p)
    monkeypatch.setattr(ARCWorld, "_explorers", {}, raising=False)
    monkeypatch.setattr(ARCWorld, "_explorer_disk", None, raising=False)
    return p


def _named():
    e = Explorer()
    e.self_id = (9, 80)
    e.ident_n = 400
    e.model[(b"x" * 25, 1)][(0, 1)] = 7
    e.visited[0].add((3, 4))
    e.visited[1].add((7, 8))
    return e


def test_roundtrip_carries_self_model_and_visited():
    f = Explorer.from_dict(_named().to_dict())
    assert f.self_id == (9, 80)
    assert f.ident_n == 400
    assert f.model[(b"x" * 25, 1)][(0, 1)] == 7
    assert f.visited[0] == {(3, 4)}
    assert f.visited[1] == {(7, 8)}


def test_unnamed_game_resumes_with_a_fresh_attempt():
    """Persistence must never lock a game into permanent failure.

    A single-action game has cond == base by construction, so its lift is
    identically 0 and no extra evidence can ever name a self there
    (+0.000 measured in lp85/r11l/vc33/cd82 over three windows).  If a
    spent ident_n were carried, such a game could never retry.
    """
    g = Explorer.from_dict({"self_id": None, "ident_n": 399})
    assert g.self_id is None
    assert g.ident_n == 0


@pytest.mark.parametrize("bad", [
    {"model": "garbage"}, {"self_id": [1]}, {"visited": 7},
    {"model": [["nothex", 1, []]]}, {},
])
def test_malformed_state_degrades_to_a_fresh_explorer(bad):
    e = Explorer.from_dict(bad)
    assert isinstance(e, Explorer)
    assert e.self_id is None


def test_save_writes_only_games_worth_carrying(state_path):
    ARCWorld._explorers = {"named": _named(), "virgin": Explorer()}
    assert ARCWorld.save_explorers() == 1
    with open(state_path) as fh:
        d = json.load(fh)
    assert d["version"] == 1
    assert list(d["games"]) == ["named"]


def test_save_is_atomic_and_leaves_no_tmp(state_path):
    ARCWorld._explorers = {"named": _named()}
    ARCWorld.save_explorers()
    assert os.path.exists(state_path)
    assert not os.path.exists(state_path + ".tmp")


def test_missing_file_reads_as_empty_not_an_error(state_path):
    assert ARCWorld._explorer_saved() == {}


def test_unreadable_file_reads_as_empty_not_an_error(state_path):
    with open(state_path, "w") as fh:
        fh.write("{ this is not json")
    assert ARCWorld._explorer_saved() == {}


def test_saved_state_reloads_into_a_live_explorer(state_path):
    ARCWorld._explorers = {"g1": _named()}
    ARCWorld.save_explorers()
    ARCWorld._explorers = {}
    ARCWorld._explorer_disk = None
    e = Explorer.from_dict(ARCWorld._explorer_saved()["g1"])
    assert e.self_id == (9, 80)
    assert e.model[(b"x" * 25, 1)][(0, 1)] == 7


# --- the LOAD branch of ARCWorld._explorer ----------------------------
# The live daemon can only prove this by restarting.  Pinned here so a
# restart is not the test.


class _Stub(object):
    """_explorer() reads nothing but game_id."""
    def __init__(self, g):
        self.game_id = g


@pytest.fixture
def armed(tmp_path, monkeypatch):
    p = str(tmp_path / "explorer_state.json")
    monkeypatch.setattr(arc_world, "_EXPLORER_STATE_PATH", p)
    monkeypatch.setattr(arc_world, "_EXPLORER_ON", lambda: True)
    monkeypatch.setattr(arc_world, "_EXPLORERPERSIST_ON", lambda: True)
    monkeypatch.setattr(ARCWorld, "_explorers", {}, raising=False)
    monkeypatch.setattr(ARCWorld, "_explorer_disk", None, raising=False)
    monkeypatch.setattr(ARCWorld, "_explorer_loaded", 0, raising=False)
    return p


def _write(p, games):
    with open(p, "w") as fh:
        json.dump({"version": 1, "games": games}, fh)


def test_explorer_is_rebuilt_from_disk(armed):
    _write(armed, {"g1": {"self_id": [9, 80], "ident_n": 400,
                          "model": [["ab" * 25, 1, [[0, 1, 7]]]],
                          "visited": {"0": [[3, 4]]}}})
    e = ARCWorld._explorer(_Stub("g1"))
    assert e.self_id == (9, 80)
    assert e.model[(bytes.fromhex("ab" * 25), 1)][(0, 1)] == 7
    assert e.visited[0] == {(3, 4)}
    assert ARCWorld._explorer_loaded == 1


def test_game_absent_from_disk_starts_fresh(armed):
    _write(armed, {"other": {"self_id": [1, 2]}})
    e = ARCWorld._explorer(_Stub("g1"))
    assert e.self_id is None
    assert ARCWorld._explorer_loaded == 0


def test_gate_off_ignores_the_file_entirely(armed, monkeypatch):
    monkeypatch.setattr(arc_world, "_EXPLORERPERSIST_ON", lambda: False)
    _write(armed, {"g1": {"self_id": [9, 80]}})
    e = ARCWorld._explorer(_Stub("g1"))
    assert e.self_id is None, "gate off must not load"
    assert ARCWorld._explorer_loaded == 0


def test_second_call_returns_the_same_live_object(armed):
    _write(armed, {"g1": {"self_id": [9, 80]}})
    a = ARCWorld._explorer(_Stub("g1"))
    b = ARCWorld._explorer(_Stub("g1"))
    assert a is b
    assert ARCWorld._explorer_loaded == 1, "must not re-load per call"


def test_corrupt_file_does_not_break_the_world(armed):
    with open(armed, "w") as fh:
        fh.write("{ not json")
    e = ARCWorld._explorer(_Stub("g1"))
    assert e is not None and e.self_id is None


# --- a self named on 400 frames must still earn on 6,000 --------------
# Persisting self_id turned a marginal self from INERT into one he acts
# on for good.  Measured over the five he had actually named
# (n=4,991-6,884): ar25 +36.2, tr87 +28.3, ls20 +20.1, sc25 +14.0 and
# g50t -0.2 -- one in five explained nothing at all.

from seagi.world.explorer import EARN_MIN_LIFT, EARN_MIN_N


def _with(disp_all, disp_by, named=True):
    e = Explorer()
    if named:
        e.self_id = (9, 24)
        e.ident_n = 400
    for d, c in disp_all.items():
        e.disp_all[d] += c
    for a, cc in disp_by.items():
        for d, c in cc.items():
            e.disp_by[a][d] += c
    e.here = ((0.0, 0.0), b"e" * 25)
    return e


def _earning(n=EARN_MIN_N):
    h = n // 2
    return _with({(0, 1): h, (1, 0): h},
                 {0: {(0, 1): h}, 1: {(1, 0): h}})


def _not_earning(n=EARN_MIN_N):
    h = n // 2
    return _with({(0, 1): n}, {0: {(0, 1): h}, 1: {(0, 1): h}})


def test_an_unjudged_self_is_still_used():
    """Below EARN_MIN_N he behaves exactly as he did before."""
    e = _not_earning(n=EARN_MIN_N - 2)
    assert e.earns() is None
    assert e.usable() is True


def test_a_self_that_earns_is_kept():
    e = _earning()
    assert e.earns() > EARN_MIN_LIFT
    assert e.usable() is True


def test_a_self_that_earns_nothing_is_dropped():
    e = _not_earning()
    assert e.earns() == 0.0
    assert e.usable() is False
    assert e.unearned >= 1


def test_both_deciders_go_quiet_on_a_dead_self():
    """The safety property: he must fall back to behaving as though he
    had no self, which is what he did before persistence."""
    e = _not_earning()
    e.model[(b"e" * 25, 0)][(0, 1)] = 9
    assert e.toward_new(6, 0) is None
    assert e.moves_me(0) is None


def test_a_live_self_still_steers():
    e = _earning()
    e.model[(b"e" * 25, 0)][(0, 1)] = 9
    assert e.toward_new(6, 0) == 0
    assert e.moves_me(0) == 1.0


def test_no_self_at_all_is_unusable():
    e = _with({}, {}, named=False)
    assert e.usable() is False


def test_the_judgement_survives_a_restart():
    """Otherwise every restart re-earns the same verdict from zero."""
    e = _not_earning()
    f = Explorer.from_dict(e.to_dict())
    assert sum(f.disp_all.values()) == sum(e.disp_all.values())
    assert f.earns() == 0.0
    assert f.usable() is False


def test_stats_report_the_verdict():
    s = _not_earning().stats()
    assert s["usable"] is False
    assert s["earns"] == 0.0


# --- a save must not erase what this process never touched -------------
# He rotates: four minutes after a restart _explorers held 1 game of 25.
# Writing only those would have destroyed every self he had not happened
# to revisit, so the first save after each restart would undo most of
# what the previous one learned.

def test_save_keeps_games_this_process_never_touched(state_path):
    _write(state_path, {
        "untouched": {"self_id": [1, 2], "ident_n": 400,
                      "model": [["cd" * 25, 0, [[1, 0, 3]]]]},
    })
    ARCWorld._explorer_disk = None
    ARCWorld._explorers = {"fresh": _named()}
    ARCWorld.save_explorers()
    with open(state_path) as fh:
        g = json.load(fh)["games"]
    assert set(g) == {"untouched", "fresh"}, g.keys()
    assert g["untouched"]["self_id"] == [1, 2]


def test_a_revisited_game_is_overwritten_not_duplicated(state_path):
    _write(state_path, {"g1": {"self_id": [1, 2], "model": []}})
    ARCWorld._explorer_disk = None
    live = _named()
    ARCWorld._explorers = {"g1": live}
    ARCWorld.save_explorers()
    with open(state_path) as fh:
        g = json.load(fh)["games"]
    assert list(g) == ["g1"]
    assert g["g1"]["self_id"] == [9, 80], "live state must win"


def test_repeated_saves_are_stable(state_path):
    _write(state_path, {"untouched": {"self_id": [1, 2]}})
    ARCWorld._explorer_disk = None
    ARCWorld._explorers = {"fresh": _named()}
    for _ in range(3):
        ARCWorld.save_explorers()
    with open(state_path) as fh:
        g = json.load(fh)["games"]
    assert set(g) == {"untouched", "fresh"}


# --- the explorer, per level -----------------------------------------
# [[p_the_no_premise_experiment]] predicted in writing that the ego key
# helps MORE on L1 than L0 -- the reverse of every other organ, because
# board keys recur 0.0% across that boundary while the ego patch is
# translation invariant.  Nothing recorded a single explorer event by
# level, so it was never checkable.  First reading (sp80, the only game
# of six that names a self at all): L0 56.5% proposal rate, L1 49.1% --
# the prediction is NOT supported, though `no_self` and `unusable` are
# both 0 on L1, so the model itself does cross intact.

def test_observing_counts_by_level():
    e = Explorer()
    e.self_id = (9, 24)
    e.ident_n = 400
    flat = bytes(64 * 64)
    for lv in (0, 0, 1):
        e.observe(flat, 64, 64, 0, lv)
    assert e.by_level[0]["obs"] == 2
    assert e.by_level[1]["obs"] == 1


def test_a_self_he_cannot_locate_is_counted_per_level():
    e = Explorer()
    e.self_id = (7, 3)          # colour 7 is absent from a blank board
    e.ident_n = 400
    e.observe(bytes(64 * 64), 64, 64, 0, 1)
    assert e.by_level[1]["no_self"] == 1
    assert e.by_level[1]["obs"] == 1


def test_the_three_silences_are_told_apart():
    """no_model, all_visited and unusable are different diagnoses."""
    e = Explorer()
    e.self_id = (9, 24)
    e.ident_n = 400
    e.here = ((0.0, 0.0), b"e" * 25)

    # no model for this patch at all
    assert e.toward_new(6, 1) is None
    assert e.by_level[1]["no_model"] == 1
    assert e.by_level[1]["ask"] == 1

    # a model, and an unvisited landing -> a proposal
    e.model[(b"e" * 25, 2)][(1, 0)] = 9
    assert e.toward_new(6, 1) == 2
    assert e.by_level[1]["target"] == 1

    # same model, but that landing has now been stood on
    e.visited[1].add((1, 0))
    assert e.toward_new(6, 1) is None
    assert e.by_level[1]["all_visited"] == 1


def test_a_culled_self_is_counted_unusable_not_silent():
    e = Explorer()
    e.self_id = (9, 24)
    e.ident_n = 400
    e.here = ((0.0, 0.0), b"e" * 25)
    for _ in range(EARN_MIN_N):
        e.disp_all[(0, 1)] += 1
    e.disp_by[0][(0, 1)] = EARN_MIN_N // 2
    e.disp_by[1][(0, 1)] = EARN_MIN_N // 2
    assert e.usable() is False
    assert e.toward_new(6, 1) is None
    assert e.by_level[1]["unusable"] == 1


def test_by_level_survives_a_restart():
    """Otherwise the reading resets every time he is restarted, and he
    was restarted 12 times in one day."""
    e = Explorer()
    e.self_id = (9, 24)
    e.by_level[0]["ask"] = 5
    e.by_level[1]["target"] = 3
    f = Explorer.from_dict(e.to_dict())
    assert f.by_level[0]["ask"] == 5
    assert f.by_level[1]["target"] == 3


def test_stats_reports_by_level():
    e = Explorer()
    e.self_id = (9, 24)
    e.by_level[1]["obs"] = 4
    assert e.stats()["by_level"]["1"]["obs"] == 4
