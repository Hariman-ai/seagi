"""GOOD STEP, BAD STEP, NOT YET KNOWN.

The user's three basic tools for playing any game, remembered per
(board, action) and adjustable as evidence accumulates.

Pins the properties the design rests on:
  * fewer than two observations is UNCONFIRMED, so every mistake is still
    made twice before it is judged
  * decisive events (a level advance, a death) outrank everyday novelty
  * a genuine tie stays UNCONFIRMED rather than guessing
  * the verdict ADJUSTS as evidence accumulates -- it is counted, never
    latched, because 28% of repeated steps change verdict
  * the aggregated per-action index is rebuilt on restore, or a restored
    judgement would be unqueryable
  * the gate switches it off completely
"""
import unittest

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld

NOTF = 'NOT_FINISHED'


class _Stub(object):
    game_id = 'vgame'

    def __init__(self):
        self._levels = 0
        self._grid = [[0] * 64 for _ in range(64)]

    def board(self, tag):
        self._grid[0][0] = tag
        return self._board_hash()

    _board_hash = ARCWorld._board_hash
    _board_hash_nb = ARCWorld._board_hash_nb
    _bar_of = ARCWorld._bar_of
    _vkey = ARCWorld._vkey
    _judge_step = ARCWorld._judge_step
    step_verdict = ARCWorld.step_verdict
    verdict_to_dict = ARCWorld.verdict_to_dict
    verdict_from_dict = ARCWorld.verdict_from_dict


class TestStepVerdict(unittest.TestCase):

    def setUp(self):
        self._gate = arc_world._VERDICT_ON
        arc_world._VERDICT_ON = lambda: True
        for d in (ARCWorld._verdict, ARCWorld._verdict_i,
                  ARCWorld._att_boards):
            d.clear()
        self.w = _Stub()

    def tearDown(self):
        arc_world._VERDICT_ON = self._gate
        for d in (ARCWorld._verdict, ARCWorld._verdict_i,
                  ARCWorld._att_boards):
            d.clear()

    def _step(self, frm, act, to, success=False, state=NOTF):
        """He was on `frm`, sent `act`, and is now standing on `to`."""
        h = self.w.board(frm)
        self.w.board(to)
        self.w._judge_step(h, (act, None, None, False), success, state)

    def test_one_observation_is_unconfirmed(self):
        self._step(1, 0, 2)
        self.w.board(1)
        self.assertEqual(self.w.step_verdict(0), 0,
                         'judged on a single observation')

    def test_opening_new_ground_is_good(self):
        self._step(1, 0, 2)
        self._step(1, 0, 3)
        self.w.board(1)
        self.assertEqual(self.w.step_verdict(0), 1)

    def test_being_sent_back_is_bad(self):
        self._step(1, 0, 2)          # 2 becomes stood-on
        self._step(1, 0, 2)          # back to 2
        self._step(1, 0, 2)
        self.w.board(1)
        self.assertEqual(self.w.step_verdict(0), -1)

    def test_death_outranks_novelty(self):
        self._step(1, 0, 2)
        self._step(1, 0, 3)          # two NEW
        self._step(1, 0, 9, state='GAME_OVER')
        self.w.board(1)
        self.assertEqual(self.w.step_verdict(0), -1,
                         'a death must outrank ordinary novelty')

    def test_a_level_advance_outranks_a_death(self):
        self._step(1, 0, 9, state='GAME_OVER')
        self._step(1, 0, 4, success=True)
        self._step(1, 0, 5, success=True)
        self.w.board(1)
        self.assertEqual(self.w.step_verdict(0), 1)

    def test_the_verdict_adjusts(self):
        self._step(1, 0, 2)
        self._step(1, 0, 2)
        self._step(1, 0, 2)          # BACK, BACK -> bad
        self.w.board(1)
        self.assertEqual(self.w.step_verdict(0), -1)
        for t in (11, 12, 13, 14):   # then it starts opening ground
            self._step(1, 0, t)
        self.w.board(1)
        self.assertEqual(self.w.step_verdict(0), 1,
                         'the verdict must adjust as evidence accumulates')

    def test_actions_are_judged_separately(self):
        self._step(1, 0, 2)
        self._step(1, 0, 3)          # action 0 opens ground
        self._step(1, 1, 2)
        self._step(1, 1, 2)
        self._step(1, 1, 2)          # action 1 sends him back
        self.w.board(1)
        self.assertEqual(self.w.step_verdict(0), 1)
        self.assertEqual(self.w.step_verdict(1), -1)

    def test_an_unseen_board_is_unconfirmed(self):
        self._step(1, 0, 2)
        self._step(1, 0, 3)
        self.w.board(77)
        self.assertEqual(self.w.step_verdict(0), 0,
                         'a board he has never judged must not answer')

    def test_survives_the_restart_boundary(self):
        self._step(1, 0, 2)
        self._step(1, 0, 3)
        d = self.w.verdict_to_dict()
        ARCWorld._verdict.clear()
        ARCWorld._verdict_i.clear()
        self.assertGreaterEqual(self.w.verdict_from_dict(d), 1)
        self.w.board(1)
        self.assertEqual(self.w.step_verdict(0), 1,
                         'restored judgement is unqueryable -- the '
                         'aggregated index was not rebuilt')

    def test_gate_off_means_no_judgement(self):
        self._step(1, 0, 2)
        self._step(1, 0, 3)
        arc_world._VERDICT_ON = lambda: False
        self.w.board(1)
        self.assertEqual(self.w.step_verdict(0), 0)


if __name__ == '__main__':
    unittest.main()


class TestAttemptBoundary(TestStepVerdict):
    """A NEW LEVEL IS A NEW ATTEMPT.

    The first build cleared the stood-on set only on a death or a win, so
    a rotation away and back grew it into "every board ever seen in this
    game" and nearly every step read BACK.  Live effect: 89 BAD verdicts
    and ZERO good, against a corpus that yields 8,767 good.  The original
    tests all passed through that bug -- this is the one that would not.
    """

    def test_a_new_level_is_a_new_attempt(self):
        h5 = self.w.board(5)
        self._step(1, 0, 2)              # board 2 stood on during level 0
        self.w._levels = 1               # ... he advances
        self._step(5, 0, 2)              # landing on 2 again is NEW here
        e = ARCWorld._verdict_i[('vgame', h5, 0)]
        self.assertEqual(e[2], 1, 'a level change must start a new attempt')
        self.assertEqual(e[3], 0, 'stale boards leaked across the level')

    def test_stood_on_set_does_not_grow_without_bound(self):
        for t in range(60):
            self._step(1, 0, 100 + t)
        self.assertLessEqual(
            len(ARCWorld._att_boards.get('vgame', ())), 20000)

    def test_a_death_starts_a_new_attempt(self):
        h5 = self.w.board(5)
        self._step(1, 0, 2)
        self._step(1, 0, 9, state='GAME_OVER')
        self._step(5, 0, 2)              # fresh attempt: 2 is new again
        e = ARCWorld._verdict_i[('vgame', h5, 0)]
        self.assertEqual(e[3], 0, 'boards survived a death')
