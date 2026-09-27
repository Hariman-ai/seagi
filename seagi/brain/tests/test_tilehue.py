"""PATCH 39 (2026-09-11) -- TILEHUE: a click target that wears the selected colour is
found by its shape.

MEASURED cd82 L5: every life stalls at mismatch 24 because the planner's last round
clicks the tile keyed with the colour it wore when learned; after the plan's own swatch
click the tile wears the new colour and `_find` returns None (no_tile_on_board 568).

Pinned here:
  * exact (colour, shape) still wins when present
  * a carrier shape (worn >= 2 hues) is found by shape alone when the colour differs
  * a shape that has worn ONE hue is not (a real recolour would be a different object)
  * two objects of a carrier shape on the board: ambiguous, None
  * gate off: byte-identical (None)
  * bg / None keys unchanged; the big-object third-row offset still applies
"""
import unittest
from unittest import mock

from seagi.world import relplan as R
from seagi.world.relplan import RelPlan

SIG = 'abcdef012345'


def _rp(hues, mutable=True):
    rp = RelPlan()
    rp.hues[SIG] = set(hues)
    if mutable:
        rp.mutable.add(SIG)
    return rp


class TestFind(unittest.TestCase):
    def setUp(self):
        self._g = mock.patch.object(R, '_TILEHUE_ON', lambda: True)
        self._g.start()

    def tearDown(self):
        self._g.stop()

    def test_exact_match_wins(self):
        rp = _rp([11, 14])
        outs = [(SIG, 11, (37, 13, 40, 15)), (SIG, 14, (10, 10, 13, 12))]
        self.assertEqual(rp._find(outs, (14, SIG, 1, 1)), (11, 11))
        self.assertEqual(getattr(rp, 'tilehue_finds', 0), 0)

    def test_a_carrier_is_found_by_shape_when_it_wears_another_hue(self):
        rp = _rp([8, 9, 11, 12, 14, 15])
        outs = [('other', 14, (0, 0, 2, 2)), (SIG, 11, (37, 13, 40, 15))]
        self.assertEqual(rp._find(outs, (14, SIG, 1, 1)), (38, 14))
        self.assertEqual(rp.tilehue_finds, 1)
        self.assertTrue(any(e.startswith('tilehue_find') for e in rp.events), rp.events)

    def test_an_immutable_shape_is_not_a_carrier(self):
        rp = _rp([11, 14], mutable=False)
        outs = [(SIG, 11, (37, 13, 40, 15))]
        self.assertIsNone(rp._find(outs, (14, SIG, 1, 1)))

    def test_a_one_hue_shape_is_not(self):
        rp = _rp([14])
        outs = [(SIG, 11, (37, 13, 40, 15))]
        self.assertIsNone(rp._find(outs, (14, SIG, 1, 1)))

    def test_two_of_the_shape_is_ambiguous(self):
        rp = _rp([11, 14])
        outs = [(SIG, 11, (37, 13, 40, 15)), (SIG, 8, (20, 20, 23, 22))]
        self.assertIsNone(rp._find(outs, (14, SIG, 1, 1)))

    def test_gate_off_is_byte_identical(self):
        rp = _rp([8, 11, 14])
        outs = [(SIG, 11, (37, 13, 40, 15))]
        with mock.patch.object(R, '_TILEHUE_ON', lambda: False):
            self.assertIsNone(rp._find(outs, (14, SIG, 1, 1)))
            self.assertEqual(rp._find(outs, (11, SIG, 1, 1)), (38, 14))

    def test_never_for_a_swatch_shape(self):
        rp = _rp([0, 8, 11, 14, 15])
        rp.swatch[14] = [(14, SIG, 1, 1), 5]        # SIG is the swatch shape
        outs = [(SIG, 11, (3, 40, 5, 42))]           # only one swatch visible, colour 11
        self.assertIsNone(rp._find(outs, (14, SIG, 1, 1)), 'a swatch of another colour was taken for colour 14')
        self.assertEqual(rp._find(outs, (11, SIG, 1, 1)), (4, 41))

    def test_bg_and_none_unchanged(self):
        rp = _rp([11, 14])
        self.assertIsNone(rp._find([(SIG, 11, (0, 0, 1, 1))], None))
        self.assertIsNone(rp._find([(SIG, 11, (0, 0, 1, 1))], ('bg',)))

    def test_big_object_offset_applies_on_the_fallback(self):
        rp = _rp([11, 14])
        big = R.BIG_OBJECT
        outs = [(SIG, 11, (10, 10, 10 + 3 * big - 1, 10 + 3 * big - 1))]
        y, x = rp._find(outs, (14, SIG, 2, 0))
        self.assertGreater(y, 10 + 2 * big - 1)
        self.assertLess(x, 10 + big)


if __name__ == '__main__':
    unittest.main()
