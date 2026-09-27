"""KNOWNPATH: he remembers the way out of a level he has already cleared.

The gate is patched on for the duration, so these test the ORGAN rather
than whether it happens to be deployed right now.

Guards the properties the design depends on:
  * the stored path is LOOP-ERASED -- a board->action map loses order
    when a run revisits a board, which turned two of seven measured
    levels into cycles
  * the SHORTEST run wins, and a longer later run cannot overwrite it
  * an unknown board answers None -- ORDER ONLY, NEVER MEMBERSHIP: off
    the path he plays exactly as he would without this organ
  * the aim is bound to its action, since for a click the coordinate and
    not the index decides the outcome
  * a run that DIES leaves nothing behind
  * the map survives the restart boundary
"""
import unittest

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld


class _Stub(object):
    """Minimum surface the path methods touch."""

    game_id = 'testgame'

    def __init__(self):
        self._levels = 0
        self._grid = [[0] * 64 for _ in range(64)]

    def board(self, tag):
        self._grid[0][0] = tag
        return self._board_hash()

    _board_hash = ARCWorld._board_hash
    _board_hash_nb = ARCWorld._board_hash_nb
    _bar_of = ARCWorld._bar_of
    _path_key = ARCWorld._path_key
    _note_path_step = ARCWorld._note_path_step
    _seal_path = ARCWorld._seal_path
    _drop_path = ARCWorld._drop_path
    known_action = ARCWorld.known_action
    known_aim = ARCWorld.known_aim
    path_to_dict = ARCWorld.path_to_dict
    path_from_dict = ARCWorld.path_from_dict


class TestKnownPath(unittest.TestCase):

    def setUp(self):
        self._gate = arc_world._KNOWNPATH_ON
        arc_world._KNOWNPATH_ON = lambda: True
        self._saved = dict(ARCWorld._paths)
        self._savedcur = dict(ARCWorld._cur_path)
        ARCWorld._paths.clear()
        ARCWorld._cur_path.clear()
        self.w = _Stub()

    def tearDown(self):
        arc_world._KNOWNPATH_ON = self._gate
        ARCWorld._paths.clear()
        ARCWorld._paths.update(self._saved)
        ARCWorld._cur_path.clear()
        ARCWorld._cur_path.update(self._savedcur)

    def _run(self, steps):
        out = {}
        for tag, sent in steps:
            h = self.w.board(tag)
            out[tag] = h
            self.w._note_path_step(h, sent)
        return out

    def test_seal_erases_the_detour(self):
        h = self._run([(1, (0, 5, 7)), (2, (1, 8, 9)), (3, (2, 1, 1)),
                       (2, (9, 4, 4)), (4, (3, 2, 2))])
        self.w._seal_path(0)
        p = ARCWorld._paths[('testgame', 0)]
        self.assertEqual(len(p), 3, 'detour through 3 was not erased')
        self.assertNotIn(('testgame', 0), ARCWorld._cur_path)
        # board 2 keeps the LATER action, the one that led onward
        self.w.board(2)
        self.assertEqual(self.w.known_action(), 9)
        self.w.board(1)
        self.assertEqual(self.w.known_action(), 0)
        self.assertEqual(self.w.known_aim(h[1], 0), (5, 7))

    def test_unknown_board_is_silent(self):
        self._run([(1, (0, 0, 0)), (2, (1, 0, 0))])
        self.w._seal_path(0)
        self.w.board(42)
        self.assertIsNone(self.w.known_action(),
                          'an unknown board must not steer him')

    def test_aim_is_bound_to_its_action(self):
        h = self._run([(1, (4, 5, 7))])
        self.w._seal_path(0)
        self.assertEqual(self.w.known_aim(h[1], 4), (5, 7))
        self.assertIsNone(self.w.known_aim(h[1], 5),
                          'aim handed back for the wrong action')

    def test_shortest_run_is_kept(self):
        self._run([(1, (0, 0, 0)), (2, (0, 0, 0)), (3, (0, 0, 0))])
        self.w._seal_path(0)
        self.assertEqual(len(ARCWorld._paths[('testgame', 0)]), 3)
        ARCWorld._cur_path.clear()
        self._run([(1, (7, 0, 0)), (4, (8, 0, 0))])
        self.w._seal_path(0)
        self.assertEqual(len(ARCWorld._paths[('testgame', 0)]), 2,
                         'shorter run did not replace the longer one')
        ARCWorld._cur_path.clear()
        self._run([(1, (0, 0, 0)), (2, (0, 0, 0)),
                   (3, (0, 0, 0)), (4, (0, 0, 0))])
        self.w._seal_path(0)
        self.assertEqual(len(ARCWorld._paths[('testgame', 0)]), 2,
                         'a longer later run overwrote his best')

    def test_a_death_leaves_nothing(self):
        self._run([(1, (1, 1, 1)), (2, (2, 2, 2))])
        self.w._drop_path()
        self.assertFalse(ARCWorld._cur_path, 'a death left a partial path')
        self.w.board(1)
        self.assertIsNone(self.w.known_action())

    def test_survives_the_restart_boundary(self):
        self._run([(1, (7, 3, 4)), (2, (8, 0, 0))])
        self.w._seal_path(0)
        saved = dict(ARCWorld._paths)
        d = self.w.path_to_dict()
        ARCWorld._paths.clear()
        self.assertEqual(self.w.path_from_dict(d), 1)
        self.assertEqual(ARCWorld._paths, saved,
                         'round-trip through the save changed the map')
        self.w.board(1)
        self.assertEqual(self.w.known_action(), 7)
        self.assertEqual(self.w.known_aim(self.w.board(1), 7), (3, 4))

    def test_gate_off_means_no_steering(self):
        self._run([(1, (7, 3, 4)), (2, (8, 0, 0))])
        self.w._seal_path(0)
        arc_world._KNOWNPATH_ON = lambda: False
        self.w.board(1)
        self.assertIsNone(self.w.known_action(),
                          'the gate must switch the organ off completely')


if __name__ == '__main__':
    unittest.main()


class TestPathLoopGuard(TestKnownPath):
    """A remembered path must not become a treadmill.

    The stored path cannot repeat a board (loop-erasure guarantees it), so
    standing on one twice means the WORLD sent him back.  The world is
    only 67.1% deterministic, and 2 of the 7 measured maps cycle in the
    offline walk, so this is a real path and not a hypothetical one.
    """

    def test_second_visit_stops_following(self):
        self._run([(1, (5, 0, 0)), (2, (6, 0, 0))])
        self.w._seal_path(0)
        # a fresh attempt: he stands on 1 and follows
        self.w.board(1)
        self.assertEqual(self.w.known_action(), 5)
        self.w._note_path_step(self.w.board(1), (5, 0, 0))
        # the world sends him back to 1 -- do not follow it a second time
        self.w.board(1)
        self.assertIsNone(self.w.known_action(),
                          'followed the same board twice: a treadmill')

    def test_guard_does_not_leak_across_attempts(self):
        self._run([(1, (5, 0, 0)), (2, (6, 0, 0))])
        self.w._seal_path(0)
        self.w._note_path_step(self.w.board(1), (5, 0, 0))
        self.w.board(1)
        self.assertIsNone(self.w.known_action())
        # he dies; the next attempt must start clean
        self.w._drop_path()
        self.w.board(1)
        self.assertEqual(self.w.known_action(), 5,
                         'the guard survived a death and muted the organ')


class TestSealKeepsTheWinningMove(TestKnownPath):
    """The move that WINS must be in the path that gets sealed.

    `_absorb` advances self._levels before the step is recorded, so a
    winning move was filed under the NEXT level while `_seal_path` sealed
    the previous one.  The stored path lost its final, decisive move and
    replay would run out of map one step short of the win -- every time.
    """

    def test_the_winning_move_is_in_the_sealed_path(self):
        # two ordinary steps on level 0 ...
        for tag, sent in ((1, (0, 0, 0)), (2, (1, 0, 0))):
            self.w._note_path_step(self.w.board(tag), sent, 0)
        # ... then the move that completes it.  By now _absorb has already
        # advanced the level, exactly as it does live.
        self.w._levels = 1
        self.w._note_path_step(self.w.board(3), (9, 0, 0), 0)
        self.w._seal_path(0)
        p = ARCWorld._paths[('testgame', 0)]
        self.assertEqual(len(p), 3,
                         'the sealed path lost the move that won')
        self.assertEqual(p[self.w.board(3)][0], 9,
                         'the winning move is not the one stored')

    def test_the_next_level_does_not_inherit_a_stale_board(self):
        self.w._note_path_step(self.w.board(1), (0, 0, 0), 0)
        self.w._levels = 1
        self.w._note_path_step(self.w.board(2), (9, 0, 0), 0)
        self.assertNotIn(('testgame', 1), ARCWorld._cur_path,
                         'a board from the old level leaked into the new one')


class TestEnterStartsAFreshAttempt(TestKnownPath):
    """Rotating away and back must not glue two attempts together.

    `_cur_path` was cleared only on a seal or a death, so an abandoned
    stretch survived a rotation and the eventual seal credited its boards
    to a win they were never part of.  The first sealed path came to 71
    steps where the offline winning paths ran 3-60, median ~10.
    """

    def test_enter_drops_the_abandoned_stretch(self):
        for tag, sent in ((1, (0, 0, 0)), (2, (1, 0, 0))):
            self.w._note_path_step(self.w.board(tag), sent, 0)
        self.assertEqual(len(ARCWorld._cur_path[('testgame', 0)]), 2)
        ARCWorld.enter(self.w)          # he rotates away and comes back
        self.assertNotIn(('testgame', 0), ARCWorld._cur_path,
                         'the abandoned stretch survived the rotation')

    def test_a_win_after_returning_seals_only_the_new_run(self):
        self.w._note_path_step(self.w.board(7), (5, 0, 0), 0)   # abandoned
        ARCWorld.enter(self.w)
        for tag, sent in ((1, (0, 0, 0)), (2, (1, 0, 0))):
            self.w._note_path_step(self.w.board(tag), sent, 0)
        self.w._seal_path(0)
        p = ARCWorld._paths[('testgame', 0)]
        self.assertEqual(len(p), 2, 'sealed path absorbed the old stretch')
        self.assertNotIn(self.w.board(7), p,
                         'a board from the abandoned run was credited')
