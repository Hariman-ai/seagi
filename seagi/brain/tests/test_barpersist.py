"""THE BUDGET LINE HE LEARNED SURVIVES A RESTART.

`_bar_row` is a class dict decided after ~400 transitions per (game,
level) and was never saved, so after every restart NOTHING that masks
the line (FELT's walk-back, CLICKHIT, PROGMASK, BOARDKEY) can act in a
cell until he has spent ~400 steps there again.  `bar_to_dict` /
`bar_from_dict` mirror `depth_to_dict` / `depth_from_dict`.

Pinned here:
  * gate absent: nothing written, nothing loaded (byte-identical); no
    test reads the gate file
  * round trip restores exactly the decided lines of THIS game
  * "no line" (-1) is NOT persisted -- a no-bar cell re-learns, cheaply
  * a line already decided in this process is not overwritten
  * malformed input is ignored, never raised
"""
import unittest

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld


class _W(object):
    def __init__(self, game):
        self.game_id = game

    bar_to_dict = ARCWorld.bar_to_dict
    bar_from_dict = ARCWorld.bar_from_dict


class _Base(unittest.TestCase):
    ON = True

    def setUp(self):
        self._g = arc_world._BARPERSIST_ON
        arc_world._BARPERSIST_ON = lambda: self.ON
        ARCWorld._bar_row.clear()

    def tearDown(self):
        arc_world._BARPERSIST_ON = self._g
        ARCWorld._bar_row.clear()


class TestGateOff(_Base):
    ON = False

    def test_nothing_written(self):
        ARCWorld._bar_row[('cd82-x', 2)] = 63
        self.assertEqual(_W('cd82-x').bar_to_dict(), {})

    def test_nothing_loaded(self):
        n = _W('cd82-x').bar_from_dict({'cd82-x': {'2': 63}})
        self.assertEqual(n, 0)
        self.assertEqual(ARCWorld._bar_row, {})


class TestRoundTrip(_Base):

    def test_only_this_game_and_only_decided_lines(self):
        ARCWorld._bar_row[('cd82-x', 1)] = 63
        ARCWorld._bar_row[('cd82-x', 2)] = 63
        ARCWorld._bar_row[('r11l-y', 1)] = 64
        ARCWorld._bar_row[('lp85-z', 1)] = -1
        self.assertEqual(_W('cd82-x').bar_to_dict(),
                         {'cd82-x': {'1': 63, '2': 63}})
        self.assertEqual(_W('r11l-y').bar_to_dict(), {'r11l-y': {'1': 64}})
        self.assertEqual(_W('lp85-z').bar_to_dict(), {},
                         'a no-line decision re-learns; it is not persisted')

    def test_round_trip_restores_the_lines(self):
        ARCWorld._bar_row[('cd82-x', 2)] = 63
        ARCWorld._bar_row[('vc33-v', 1)] = 0
        blob = {}
        blob.update(_W('cd82-x').bar_to_dict())
        blob.update(_W('vc33-v').bar_to_dict())
        ARCWorld._bar_row.clear()
        d0 = ARCWorld._bar_decided
        self.assertEqual(_W('cd82-x').bar_from_dict(blob), 1)
        self.assertEqual(_W('vc33-v').bar_from_dict(blob), 1)
        self.assertEqual(ARCWorld._bar_row,
                         {('cd82-x', 2): 63, ('vc33-v', 1): 0})
        self.assertEqual(ARCWorld._bar_decided, d0 + 2,
                         'a loaded line is a decided line: the counter '
                         'every watcher prints must say so')

    def test_a_line_decided_this_process_is_not_overwritten(self):
        ARCWorld._bar_row[('cd82-x', 2)] = 63
        n = _W('cd82-x').bar_from_dict({'cd82-x': {'2': 5}})
        self.assertEqual(n, 0)
        self.assertEqual(ARCWorld._bar_row[('cd82-x', 2)] , 63)

    def test_malformed_input_is_ignored(self):
        for bad in (None, 7, [], {'cd82-x': None}, {'cd82-x': {'x': 63}},
                    {'cd82-x': {'2': 'row'}}, {'cd82-x': {'2': 999}},
                    {'cd82-x': {'2': -1}}):
            self.assertEqual(_W('cd82-x').bar_from_dict(bad), 0, bad)
        self.assertEqual(ARCWorld._bar_row, {})


if __name__ == '__main__':
    unittest.main()
