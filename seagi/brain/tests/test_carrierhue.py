"""PATCH 17 (2026-09-07) -- CARRIERHUE: the selected-colour indicator must
have been seen in at least two colours.

Pinned here:
  * gate on: a constant-colour object with the most votes is NOT read as
    the selection; the two-colour object with fewer votes is
  * gate off: the old rule (most votes wins) -- byte-identical behaviour
  * no carrier qualifies -> falls back to his last colour click
  * observe() records hues from the unique objects of each frame
  * hues round-trip through to_dict/from_dict; an old blob without hues
    loads with hues empty
"""
import unittest
from unittest import mock

from seagi.world import relplan as _rp
from seagi.world.relplan import RelPlan


class TestCarrierHue(unittest.TestCase):
    def _rp(self):
        rp = RelPlan()
        rp.mutable.update(('bar', 'icon'))
        rp.carrier = {'bar': 126, 'icon': 37}
        rp.hues = {'bar': {0}, 'icon': {0, 15}}
        return rp

    def test_gate_on_ignores_the_constant_colour_object(self):
        rp = self._rp()
        with mock.patch.object(_rp, '_CARRIERHUE_ON', lambda: True):
            self.assertEqual(rp.selected(frozenset(), {'bar': 0, 'icon': 15}), 15)

    def test_gate_off_keeps_the_old_vote(self):
        rp = self._rp()
        with mock.patch.object(_rp, '_CARRIERHUE_ON', lambda: False):
            self.assertEqual(rp.selected(frozenset(), {'bar': 0, 'icon': 15}), 0)

    def test_no_qualifying_carrier_falls_back_to_the_last_click(self):
        rp = self._rp(); rp.hues = {}; rp.sel = 12
        with mock.patch.object(_rp, '_CARRIERHUE_ON', lambda: True):
            self.assertEqual(rp.selected(frozenset(), {'bar': 0, 'icon': 15}), 12)

    def test_observe_records_hues(self):
        from seagi.brain.tests.test_relplan import Toy, A, B
        LINE = 63
        from seagi.world.relsense import segment
        import numpy as np
        toy = Toy(); rp = RelPlan()
        G0 = np.array(toy.board()); toy.step(5, (4, 47)); G1 = np.array(toy.board())
        rp.observe(G0, segment(G0, LINE)[1], G1, segment(G1, LINE)[1], 5, True, (4, 47), A, B)
        self.assertTrue(rp.hues, 'no hues recorded')
        self.assertTrue(all(isinstance(v, set) for v in rp.hues.values()))

    def test_round_trip_and_old_blob(self):
        rp = self._rp()
        d = rp.to_dict()
        self.assertEqual(d['hues']['icon'], [0, 15])
        rp2 = RelPlan(); rp2.from_dict(d)
        self.assertEqual(rp2.hues, {'bar': {0}, 'icon': {0, 15}})
        d.pop('hues')
        rp3 = RelPlan(); rp3.from_dict(d)
        self.assertEqual(rp3.hues, {})
        self.assertEqual(rp3.carrier, {'bar': 126, 'icon': 37})


if __name__ == '__main__':
    unittest.main()
