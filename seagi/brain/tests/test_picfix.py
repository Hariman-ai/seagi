"""PATCH 22 (2026-09-08) -- PICFIX.

Pinned here:
  * _rel_picture: a relation confirmed on ANY level of the game with the
    most to do this life beats the level's own pursued/confirmed small
    pair (gate on); gate off: the pursued one (old rule)
  * with nothing confirmed anywhere the old rule stands
  * a plan table asked about a picture of another shape is silent
    (`other_picture`), raises nothing, and learns nothing from it
  * pic_shape persists and an old blob without it loads as None
"""
import unittest
from unittest import mock
import numpy as np

from seagi.world import arc_world
from seagi.world import relplan as _rp
from seagi.world.arc_world import ARCWorld
from seagi.world.relplan import RelPlan
from seagi.world.relsense import RelSense
from seagi.world.relsense import segment
from seagi.brain.tests.test_relplan import Toy, teach, A, B

BIG = ('eq', (34, 27, 43, 36), (3, 3, 12, 12))
SMALL = ('eq', (20, 30, 22, 33), (3, 6, 5, 9))


class TestPicture(unittest.TestCase):
    def _rs(self):
        rs = RelSense()
        rs.v0 = {BIG: 72.0, SMALL: 12.0}
        rs.confirmed = set([SMALL]); rs._pursued = SMALL
        return rs

    def test_any_level_confirmation_wins_biggest_job(self):
        with mock.patch.object(arc_world, '_PICFIX_ON', lambda: True):
            self.assertEqual(ARCWorld._rel_picture(self._rs(), set([BIG, SMALL])), BIG)
            self.assertEqual(ARCWorld._rel_picture(self._rs(), set([BIG])), BIG)

    def test_gate_off_keeps_the_pursued_one(self):
        with mock.patch.object(arc_world, '_PICFIX_ON', lambda: False):
            self.assertEqual(ARCWorld._rel_picture(self._rs(), set([BIG, SMALL])), SMALL)
            self.assertEqual(ARCWorld._rel_picture(self._rs()), SMALL)

    def test_nothing_confirmed_anywhere_is_the_old_rule(self):
        rs = self._rs(); rs.confirmed = set(); rs._pursued = None
        with mock.patch.object(arc_world, '_PICFIX_ON', lambda: True):
            self.assertEqual(ARCWorld._rel_picture(rs, set()), BIG)   # most to do
            rs._pursued = SMALL
            self.assertEqual(ARCWorld._rel_picture(rs, None), SMALL)  # pursued first, as before


class TestTableShape(unittest.TestCase):
    def test_other_shape_is_silent_and_unlearned(self):
        rp = RelPlan()
        with mock.patch.object(_rp, '_PICFIX_ON', lambda: True):
            teach(rp)
            self.assertEqual(rp.pic_shape, (10, 10))
            n_eff = len(rp.effects); n_seen = dict(rp.seen)
            toy = Toy(); G = toy.board(); objs = segment(G, 63)[1]
            r = rp.plan(G, objs, (20, 30, 22, 33), (3, 6, 5, 9), 6)
            self.assertIsNone(r); self.assertEqual(rp.why, 'other_picture')
            toy.step(4); G2 = toy.board(); objs2 = segment(G2, 63)[1]
            rp.observe(G, objs, G2, objs2, 4, False, None, (20, 30, 22, 33), (3, 6, 5, 9))
            self.assertEqual(len(rp.effects), n_eff); self.assertEqual(dict(rp.seen), n_seen)
            # the right picture still plans
            self.assertIsNotNone(rp.plan(G, objs, A, B, 6))

    def test_gate_off_old_behaviour_including_the_error(self):
        rp = RelPlan()
        with mock.patch.object(_rp, '_PICFIX_ON', lambda: False):
            teach(rp)
            self.assertIsNone(rp.pic_shape)
            toy = Toy(); G = toy.board(); objs = segment(G, 63)[1]
            try:
                rp.plan(G, objs, (20, 30, 22, 33), (3, 6, 5, 9), 6)
            except IndexError:
                pass                                   # the measured live failure

    def test_an_unstamped_table_is_not_stamped_by_a_picture_its_cells_do_not_fit(self):
        rp = RelPlan()
        with mock.patch.object(_rp, '_PICFIX_ON', lambda: False):
            teach(rp)                                  # a pre-patch table: 10x10 cells, no stamp
        self.assertIsNone(rp.pic_shape)
        toy = Toy(); G = toy.board(); objs = segment(G, 63)[1]
        with mock.patch.object(_rp, '_PICFIX_ON', lambda: True):
            r = rp.plan(G, objs, (20, 30, 22, 33), (3, 6, 5, 9), 6)   # first ask after a restart
            self.assertIsNone(r); self.assertEqual(rp.why, 'other_picture')
            n_eff = len(rp.effects)
            toy.step(4); G2 = toy.board(); objs2 = segment(G2, 63)[1]
            rp.observe(G, objs, G2, objs2, 4, False, None, (20, 30, 22, 33), (3, 6, 5, 9))
            self.assertIsNone(rp.pic_shape); self.assertEqual(len(rp.effects), n_eff)
            rp.observe(G, objs, G2, objs2, 4, False, None, A, B)     # its own picture stamps it
            self.assertEqual(rp.pic_shape, (10, 10))

    def test_round_trip(self):
        rp = RelPlan()
        with mock.patch.object(_rp, '_PICFIX_ON', lambda: True):
            teach(rp)
        d = rp.to_dict(); self.assertEqual(d['pic_shape'], [10, 10])
        rp2 = RelPlan(); rp2.from_dict(d); self.assertEqual(rp2.pic_shape, (10, 10))
        d.pop('pic_shape'); rp3 = RelPlan(); rp3.from_dict(d); self.assertIsNone(rp3.pic_shape)


if __name__ == '__main__':
    unittest.main()
