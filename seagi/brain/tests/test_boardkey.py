"""THE WITHIN-LIFE BOARD ORGANS SEE THE BOARD, NOT THE CLOCK.

`_board_act` and `_board_noop` are written by `_note_board_move` and
read by `untried_here`, `board_inert_here` and `cycles_here`.  All were
keyed on the FULL board hash, which the budget line makes unique on
every step of a life -- so within a life "have I tried this here" and
"does this do nothing here" could never match.  cd82 level 2, measured:
75.4% of steps change nothing (random null 64.6%), 67.6% of them a
repeat of an action already seen to do nothing at that board (null
54.5%); live, 154 of 20,577 pairs dead (0.75%) after 54,186 steps.
The line is learned per process (~400 transitions per cell); until
then `_board_hash_nb` falls through to the full hash.

Pinned here:
  * gate absent is BYTE-IDENTICAL: a line tick makes the board "new",
    nothing is ever inert, nothing here counts as tried; no test reads
    the gate file
  * gate on: two line-only ticks make the action INERT here
  * gate on: a content change is NOT a no-op
  * gate on: the board is recognised across a line tick, so the
    untried list is what he has not tried HERE
  * gate on: a move that always lands on the same masked board CYCLES
  * an unlearned cell (no line) behaves exactly as gate off
  * the seed key (`_board_hash`) is untouched
"""
import unittest

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld

BAR_ROW = 63


def grid():
    return [[0] * 64 for _ in range(64)]


class _Stub(object):
    def __init__(self, game='vgame'):
        self.game_id = game
        self._levels = 0
        self._grid = grid()

    _bar_of = ARCWorld._bar_of
    _board_hash = ARCWorld._board_hash
    _board_hash_nb = ARCWorld._board_hash_nb
    _board_key = ARCWorld._board_key
    _note_board_move = ARCWorld._note_board_move
    untried_here = ARCWorld.untried_here
    board_inert_here = ARCWorld.board_inert_here
    cycles_here = ARCWorld.cycles_here


class _Base(unittest.TestCase):
    ON = True

    def setUp(self):
        self._saved = {}
        for name, val in (('_BOARDKEY_ON', lambda: self.ON),
                          ('_BARMASK_ON', lambda: True),
                          ('_CYCLEDEMOTE_ON', lambda: True),
                          ('_UNTRIEDHERE_ON', lambda: True)):
            self._saved[name] = getattr(arc_world, name)
            setattr(arc_world, name, val)
        for d in (ARCWorld._bar_row, ARCWorld._board_act,
                  ARCWorld._board_noop, ARCWorld._board_seen):
            d.clear()
        self.w = _Stub()
        self._t = 0

    def tearDown(self):
        for name, val in self._saved.items():
            setattr(arc_world, name, val)
        for d in (ARCWorld._bar_row, ARCWorld._board_act,
                  ARCWorld._board_noop, ARCWorld._board_seen):
            d.clear()

    def learn_line(self):
        ARCWorld._bar_row[('vgame', 0)] = BAR_ROW

    def step(self, action, tick=True, content=None):
        """One step: record the before-board, mutate, note the move."""
        prev = self.w._board_key()
        if tick:
            self.w._grid[BAR_ROW][self._t] = 5
            self._t += 1
        if content is not None:
            r, c, v = content
            self.w._grid[r][c] = v
        self.w._note_board_move(prev, action)


class TestGateOff(_Base):
    ON = False

    def test_key_is_the_full_hash(self):
        self.learn_line()
        self.assertEqual(self.w._board_key(), self.w._board_hash())

    def test_a_line_tick_makes_nothing_inert(self):
        self.learn_line()
        self.step(3)
        self.step(3)
        self.step(3)
        self.assertFalse(self.w.board_inert_here(3),
                         'gate off: the line makes every board new')

    def test_nothing_counts_as_tried_here(self):
        self.learn_line()
        self.step(1)
        self.step(2)
        self.assertEqual(self.w.untried_here(4), [],
                         'gate off: this board was never stood on')


class TestGateOn(_Base):

    def test_key_is_the_masked_hash(self):
        self.learn_line()
        self.assertEqual(self.w._board_key(), self.w._board_hash_nb())
        self.assertNotEqual(self.w._board_key(), self.w._board_hash())

    def test_two_line_only_ticks_make_the_action_inert_here(self):
        self.learn_line()
        self.step(3)
        self.assertFalse(self.w.board_inert_here(3),
                         'one observation is not enough')
        self.step(3)
        self.assertTrue(self.w.board_inert_here(3),
                        'spending budget twice with nothing else changing is dead')

    def test_a_content_change_is_not_a_noop(self):
        self.learn_line()
        self.step(3, content=(10, 10, 7))
        self.step(3, content=(11, 11, 7))
        self.assertFalse(self.w.board_inert_here(3))

    def test_the_board_is_recognised_across_a_line_tick(self):
        self.learn_line()
        self.step(1)          # tried 1 here (line ticked, board same)
        self.step(2)          # tried 2 here
        self.assertEqual(self.w.untried_here(4), [0, 3])

    def test_a_move_that_always_returns_here_cycles(self):
        # `_board_seen` only holds boards he has LANDED on, so the first
        # board of a life is not in it: arrive somewhere first, then a
        # no-op there lands on a seen board -- twice makes it a cycle.
        self.learn_line()
        self.step(1, content=(10, 10, 7))
        self.step(3)
        self.step(3)
        self.assertTrue(self.w.cycles_here(3))

    def test_a_noop_from_the_first_board_of_a_life_is_not_yet_a_cycle(self):
        # pins the semantics above so nobody "fixes" it into storming
        self.learn_line()
        self.step(3)
        self.step(3)
        self.assertFalse(self.w.cycles_here(3))

    def test_a_move_that_goes_somewhere_new_does_not_cycle(self):
        self.learn_line()
        self.step(3, content=(10, 10, 7))
        self.step(3, content=(11, 11, 7))
        self.assertFalse(self.w.cycles_here(3))

    def test_an_unlearned_cell_behaves_as_gate_off(self):
        # no line learned: the masked hash falls through to the full hash
        self.assertEqual(self.w._board_key(), self.w._board_hash())
        self.step(3)
        self.step(3)
        self.assertFalse(self.w.board_inert_here(3))

    def test_the_seed_key_is_untouched(self):
        self.learn_line()
        full = self.w._board_hash()
        self.w._grid[BAR_ROW][0] = 5
        self.assertNotEqual(self.w._board_hash(), full,
                            'the full hash still sees the line: seeds unchanged')


if __name__ == '__main__':
    unittest.main()
