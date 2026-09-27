"""KEEP THE WAY BACK TO HOW FAR HE GOT, NOT ONLY THE WAY OUT.

`_seal_path` fires only on a level clear.  He has never cleared level 1,
so `world_paths` held 239 boards, ALL level 0 and ZERO for level 1 --
each life re-derived the level from nothing.

Measured on his own corpus before this was written:
  * a median L1 life is 313-563 steps and LOOP-ERASES to 1-74 boards,
    an erase ratio of **0.06**.  Returns are only 8.9% of steps, but one
    return to an early board erases everything after it, so he makes
    forward progress and then throws it away.
  * **24.0% of the boards on a life's loop-erased path are stood on
    again the NEXT life** (m0r0 57.1%, r11l 28.2%, lp85 21.4%) -- so a
    stored board->action map does fire there.
  * cd82/sp80/vc33 erase to a MEDIAN OF 1: 275-400 steps ending on the
    board he started from.  Those lives must seal nothing.
"""
import pytest

from seagi.world import arc_world as A
from seagi.world.arc_world import ARCWorld


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(ARCWorld, "_cur_path", {}, raising=False)
    monkeypatch.setattr(ARCWorld, "_cur_seen", {}, raising=False)
    monkeypatch.setattr(ARCWorld, "_paths", {}, raising=False)
    monkeypatch.setattr(ARCWorld, "_depth_paths", {}, raising=False)
    monkeypatch.setattr(ARCWorld, "_depth_len", {}, raising=False)
    monkeypatch.setattr(ARCWorld, "_depth_sealed", 0, raising=False)
    monkeypatch.setattr(ARCWorld, "_depth_follow", 0, raising=False)
    monkeypatch.setattr(ARCWorld, "_aim_loops", 0, raising=False)
    monkeypatch.setattr(ARCWorld, "_path_follow", 0, raising=False)
    monkeypatch.setattr(ARCWorld, "_seed_loaded", True, raising=False)
    monkeypatch.setattr(A, "_DEPTHSEAL_ON", lambda: True)
    monkeypatch.setattr(A, "_KNOWNPATH_ON", lambda: True)


class _W:
    """Only the path machinery, off the real class."""
    _path_key = ARCWorld._path_key
    _seal_depth = ARCWorld._seal_depth
    known_action = ARCWorld.known_action
    known_aim = ARCWorld.known_aim
    depth_to_dict = ARCWorld.depth_to_dict
    depth_from_dict = ARCWorld.depth_from_dict

    def __init__(self, game="g", lv=1, here=None):
        self.game_id = game
        self._levels = lv
        self._here = here

    def _board_hash(self):
        return self._here


def _walk(*boards):
    """A life, as (board, (action, y, x)) the way _note_path_step keeps it."""
    return [(b, (i, None, None)) for i, b in enumerate(boards)]


def test_a_life_that_got_somewhere_is_sealed():
    w = _W()
    ARCWorld._cur_path[("g", 1)] = _walk(b"a", b"b", b"c")
    w._seal_depth(1)
    assert ARCWorld._depth_len[("g", 1)] == 3
    assert ARCWorld._depth_sealed == 1


def test_a_life_that_ended_where_it_started_seals_nothing():
    """cd82/sp80/vc33 erase to a median of 1 -- 275-400 steps of walking
    that returns to the starting board.  There is no frontier there."""
    w = _W()
    ARCWorld._cur_path[("g", 1)] = _walk(b"a", b"b", b"c", b"a")
    w._seal_depth(1)
    assert ("g", 1) not in ARCWorld._depth_paths
    assert ARCWorld._depth_sealed == 0


def test_further_wins_which_is_the_opposite_of_a_cleared_path():
    """`_seal_path` keeps the SHORTER route -- a cheaper way to a known
    exit.  Depth keeps the LONGER one -- a further frontier."""
    w = _W()
    ARCWorld._cur_path[("g", 1)] = _walk(b"a", b"b", b"c")
    w._seal_depth(1)
    ARCWorld._cur_path[("g", 1)] = _walk(b"a", b"b", b"c", b"d", b"e")
    w._seal_depth(1)
    assert ARCWorld._depth_len[("g", 1)] == 5
    ARCWorld._cur_path[("g", 1)] = _walk(b"a", b"b")
    w._seal_depth(1)
    assert ARCWorld._depth_len[("g", 1)] == 5, "a shallower life overwrote"
    assert ARCWorld._depth_sealed == 2


def test_the_gate_off_seals_nothing(monkeypatch):
    monkeypatch.setattr(A, "_DEPTHSEAL_ON", lambda: False)
    w = _W()
    ARCWorld._cur_path[("g", 1)] = _walk(b"a", b"b", b"c")
    w._seal_depth(1)
    assert ARCWorld._depth_paths == {}


def test_each_level_keeps_its_own_frontier():
    w = _W()
    ARCWorld._cur_path[("g", 0)] = _walk(b"a", b"b")
    ARCWorld._cur_path[("g", 1)] = _walk(b"x", b"y", b"z")
    w._seal_depth(0)
    w._seal_depth(1)
    assert ARCWorld._depth_len == {("g", 0): 2, ("g", 1): 3}


# ---- and reading it back ---------------------------------------------

def test_the_frontier_route_is_followed_when_nothing_else_answers():
    ARCWorld._depth_paths[("g", 1)] = {b"here": (4, None, None)}
    w = _W(here=b"here")
    assert w.known_action() == 4
    assert ARCWorld._depth_follow == 1
    assert ARCWorld._path_follow == 0, "must not be counted as a real path"


def test_a_route_that_actually_won_outranks_the_frontier():
    """A cleared path reached the EXIT; a depth path only reached the
    furthest he got.  The weaker evidence must never win."""
    ARCWorld._paths[("g", 1)] = {b"here": (1, None, None)}
    ARCWorld._depth_paths[("g", 1)] = {b"here": (4, None, None)}
    w = _W(here=b"here")
    assert w.known_action() == 1
    assert ARCWorld._path_follow == 1
    assert ARCWorld._depth_follow == 0


def test_an_unknown_board_still_returns_none():
    """The safety property: unknown board -> he plays normally."""
    ARCWorld._depth_paths[("g", 1)] = {b"elsewhere": (4, None, None)}
    w = _W(here=b"here")
    assert w.known_action() is None


def test_the_gate_off_hides_the_frontier_route(monkeypatch):
    monkeypatch.setattr(A, "_DEPTHSEAL_ON", lambda: False)
    ARCWorld._depth_paths[("g", 1)] = {b"here": (4, None, None)}
    w = _W(here=b"here")
    assert w.known_action() is None


def test_standing_here_twice_this_life_is_not_followed():
    """The stored path cannot repeat a board, so being on one twice means
    the WORLD sent him back -- following again would be a treadmill."""
    ARCWorld._depth_paths[("g", 1)] = {b"here": (4, None, None)}
    ARCWorld._cur_seen[("g", 1)] = {b"here"}
    w = _W(here=b"here")
    assert w.known_action() is None


# --- the frontier must survive a restart ------------------------------
# `_depth_paths` was a class attribute nothing saved -- the same disease
# the explorer had, and the one this organ exists to cure.  Caught with
# 79 seals across 30 paths and a mean frontier of 90.4 boards live, all
# of it process-local.

def test_the_frontier_survives_a_restart():
    w = _W()
    ARCWorld._depth_paths[("g", 1)] = {b"a": (2, None, None),
                                       b"b": (3, 4, 5)}
    ARCWorld._depth_len[("g", 1)] = 2
    blob = w.depth_to_dict()

    ARCWorld._depth_paths.clear()
    ARCWorld._depth_len.clear()
    assert w.depth_from_dict(blob) == 1
    assert ARCWorld._depth_paths[("g", 1)] == {b"a": (2, None, None),
                                              b"b": (3, 4, 5)}


def test_the_sealed_LENGTH_survives_too():
    """The comparison that admits a new frontier is 'longer than the
    stored one'.  A forgotten length would re-admit a shallower path."""
    w = _W()
    ARCWorld._depth_paths[("g", 1)] = {b"a": (1, None, None)}
    ARCWorld._depth_len[("g", 1)] = 97          # erased length, not dict size
    blob = w.depth_to_dict()
    ARCWorld._depth_paths.clear()
    ARCWorld._depth_len.clear()
    w.depth_from_dict(blob)
    assert ARCWorld._depth_len[("g", 1)] == 97

    # and a shallower life must still be refused after the reload
    ARCWorld._cur_path[("g", 1)] = _walk(b"x", b"y", b"z")
    w._seal_depth(1)
    assert ARCWorld._depth_len[("g", 1)] == 97


def test_only_this_game_is_written():
    w = _W(game="g")
    ARCWorld._depth_paths[("g", 1)] = {b"a": (1, None, None)}
    ARCWorld._depth_paths[("other", 1)] = {b"b": (2, None, None)}
    assert list(w.depth_to_dict()) == ["g"]


def test_each_level_round_trips_separately():
    w = _W()
    ARCWorld._depth_paths[("g", 0)] = {b"a": (1, None, None)}
    ARCWorld._depth_paths[("g", 1)] = {b"b": (2, None, None)}
    ARCWorld._depth_len.update({("g", 0): 1, ("g", 1): 1})
    blob = w.depth_to_dict()
    ARCWorld._depth_paths.clear()
    ARCWorld._depth_len.clear()
    assert w.depth_from_dict(blob) == 2
    assert set(ARCWorld._depth_paths) == {("g", 0), ("g", 1)}


def test_a_game_with_no_frontier_writes_nothing():
    assert _W(game="empty").depth_to_dict() == {}


@pytest.mark.parametrize("bad", [
    None, {}, {"g": None}, {"g": {"1": None}},
    {"g": {"1": {"route": {"nothex": [1, None, None]}}}},
    {"g": {"notalevel": {"route": {}}}},
])
def test_a_malformed_blob_cannot_break_the_load(bad):
    w = _W()
    assert w.depth_from_dict(bad) == 0
    assert ARCWorld._depth_paths == {}




# --- eleven days of frontiers, recovered ------------------------------
# `_seal_depth` only knows the lives it has watched and went live
# 2026-08-30 20:21Z.  The corpus holds the deepest loop-erased route he
# ever walked at each level: m0r0 108, cn04 74, lp85 LEVEL 2 65, r11l 60,
# cd82 52, ar25 48, sp80 42, ft09 30.  Without the seed he re-walks all
# of it.

import json as _json


@pytest.fixture
def seedfile(tmp_path, monkeypatch):
    """Point /root/depthseed.json at a temp file.

    The real `open` is captured BEFORE patching -- a patched `open` that
    calls `open` recurses into itself.
    """
    import builtins
    p = tmp_path / "depthseed.json"
    real_open = builtins.open

    def _open(path, *a, **k):
        if path == "/root/depthseed.json":
            return real_open(str(p), *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", _open)
    monkeypatch.setattr(ARCWorld, "_depth_seed_loaded", False, raising=False)
    monkeypatch.setattr(ARCWorld, "_depth_seeded", 0, raising=False)
    return p


def _write_seed(p, blob):
    # pathlib, so it does not go through the patched `open`
    p.write_text(_json.dumps(blob), encoding="utf-8")


def test_the_frontier_and_its_length_are_both_restored(seedfile):
    _write_seed(seedfile, {"g": {"1": {"len": 108,
                                       "route": {"ab" * 8: [0, 1, 2]}}}})
    ARCWorld._load_depth_seed()
    assert ARCWorld._depth_paths[("g", 1)] == {
        bytes.fromhex("ab" * 8): (0, 1, 2)}
    # the length matters: admission is "longer than stored", so a lost
    # length would let a shallower life overwrite a deep frontier
    assert ARCWorld._depth_len[("g", 1)] == 108
    assert ARCWorld._depth_seeded == 1


def test_a_frontier_he_has_beaten_himself_is_not_overwritten(seedfile):
    ARCWorld._depth_paths[("g", 1)] = {b"mine": (9, None, None)}
    ARCWorld._depth_len[("g", 1)] = 200
    _write_seed(seedfile, {"g": {"1": {"len": 108,
                                       "route": {"ab" * 8: [0, 1, 2]}}}})
    ARCWorld._load_depth_seed()
    assert ARCWorld._depth_paths[("g", 1)] == {b"mine": (9, None, None)}
    assert ARCWorld._depth_len[("g", 1)] == 200
    assert ARCWorld._depth_seeded == 0


def test_it_loads_once_per_process(seedfile):
    _write_seed(seedfile, {"g": {"1": {"len": 5,
                                       "route": {"cd" * 8: [1, None, None]}}}})
    ARCWorld._load_depth_seed()
    ARCWorld._depth_paths.clear()
    ARCWorld._load_depth_seed()
    assert ARCWorld._depth_paths == {}, "reloaded after being cleared"
    assert ARCWorld._depth_seeded == 1


def test_the_gate_off_seeds_nothing(seedfile, monkeypatch):
    monkeypatch.setattr(A, "_DEPTHSEAL_ON", lambda: False)
    _write_seed(seedfile, {"g": {"1": {"len": 5,
                                       "route": {"cd" * 8: [1, None, None]}}}})
    ARCWorld._load_depth_seed()
    assert ARCWorld._depth_paths == {}


def test_a_missing_file_is_simply_no_frontiers(seedfile):
    ARCWorld._load_depth_seed()          # file never written
    assert ARCWorld._depth_paths == {}


@pytest.mark.parametrize("blob", [
    {}, {"g": None}, {"g": {"1": None}}, {"g": {"1": {"route": None}}},
    {"g": {"notalevel": {"route": {"ab" * 8: [0, 1, 2]}}}},
    {"g": {"1": {"route": {"nothex": [0, 1, 2]}}}},
])
def test_a_malformed_seed_cannot_break_the_load(seedfile, blob):
    _write_seed(seedfile, blob)
    ARCWorld._load_depth_seed()
    assert ARCWorld._depth_paths == {}


def test_a_cleared_route_still_outranks_a_seeded_frontier(seedfile):
    """lp85 level 1 is CLEARED and seeded into _paths; its frontier must
    never shadow it."""
    ARCWorld._paths[("g", 1)] = {b"here": (1, None, None)}
    ARCWorld._depth_paths[("g", 1)] = {b"here": (4, None, None)}
    w = _W(here=b"here")
    assert w.known_action() == 1


# --- the aim must not pin him to a board he is standing on again -------
# `known_action` has carried a loop guard since it was written; `known_aim`
# never did, and nothing exposed it because no stored path had covered a
# board he could get stuck on.  Seeding his own recovered lp85 LEVEL 1
# route did: he stood on the route's first board, `known_action` correctly
# returned None (path_loops 2540, every single step), and the AIM stayed
# pinned to the stored (29,3).  Same click, board unchanged, same board
# next step -- 2,541 steps, 100% no-ops, zero terminals.

def test_the_aim_fires_on_a_board_he_has_not_stood_on():
    ARCWorld._paths[("g", 1)] = {b"here": (0, 29, 3)}
    w = _W(here=b"here")
    assert w.known_aim(b"here", 0) == (29, 3)


def test_the_aim_goes_quiet_on_a_board_he_is_repeating():
    """A route that cannot repeat a board must not aim at one twice."""
    ARCWorld._paths[("g", 1)] = {b"here": (0, 29, 3)}
    ARCWorld._cur_seen[("g", 1)] = {b"here"}
    w = _W(here=b"here")
    assert w.known_aim(b"here", 0) is None
    assert ARCWorld._aim_loops == 1


def test_the_action_and_the_aim_now_agree():
    """Before the fix these disagreed, which is exactly the trap: the
    action said 'play normally' and the aim said 'click here again'."""
    ARCWorld._paths[("g", 1)] = {b"here": (0, 29, 3)}
    ARCWorld._cur_seen[("g", 1)] = {b"here"}
    w = _W(here=b"here")
    assert w.known_action() is None
    assert w.known_aim(b"here", 0) is None


def test_a_different_action_still_gets_no_aim():
    ARCWorld._paths[("g", 1)] = {b"here": (0, 29, 3)}
    w = _W(here=b"here")
    assert w.known_aim(b"here", 4) is None


def test_an_unknown_board_gets_no_aim():
    ARCWorld._paths[("g", 1)] = {b"here": (0, 29, 3)}
    w = _W(here=b"elsewhere")
    assert w.known_aim(b"elsewhere", 0) is None
