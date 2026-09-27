"""PATCH 20 (2026-09-07) -- QTILE: a shape he clicks on is part of the
state however rarely it changes.

Pinned here:
  * gate on: a rare mutable part that is a click key of a learned effect
    stays in q(S); the same part without a click effect is dropped
  * gate off: the rare part is dropped either way (old rule)
  * a rare part that is NOT mostly arrow-changed is still dropped
  * the click-sig cache refreshes when the effects table grows
  * the views/effects keyed under the two rules differ only by that part
"""
import unittest
from unittest import mock

from seagi.world import relplan as _rp
from seagi.world.relplan import RelPlan


def _table(with_click_effect):
    rp = RelPlan()
    # primary: a shape that changes a lot; tile: present rarely, changes
    # when present (19 of 52), mostly by arrows (change_c 11 of 30)
    rp.mutable.update(('prim', 'tile'))
    rp.seen.update({'prim': 800, 'tile': 52})
    rp.change.update({'prim': 400, 'tile': 30})
    rp.change_c.update({'prim': 10, 'tile': 11})
    if with_click_effect:
        rp.effects[(5, (0, 'tile', 1, 1), frozenset([('prim', 10, 10, 20, 20)]))] = [set([1, 2, 3]), 4]
    return rp


S = frozenset([('prim', 10, 10, 20, 20), ('tile', 4, 15, 6, 17)])


class TestQTile(unittest.TestCase):
    def test_click_target_part_kept_under_the_gate(self):
        rp = _table(True)
        with mock.patch.object(_rp, '_QTILE_ON', lambda: True):
            q = rp.q(S)
        self.assertIn(('tile', -6, 5), q)
        self.assertIn(('prim', 0, 0), q)

    def test_without_a_click_effect_the_rare_part_is_dropped(self):
        rp = _table(False)
        with mock.patch.object(_rp, '_QTILE_ON', lambda: True):
            q = rp.q(S)
        self.assertEqual(q, frozenset([('prim', 0, 0)]))

    def test_gate_off_is_the_old_rule(self):
        rp = _table(True)
        with mock.patch.object(_rp, '_QTILE_ON', lambda: False):
            q = rp.q(S)
        self.assertEqual(q, frozenset([('prim', 0, 0)]))

    def test_a_click_moved_part_is_still_dropped(self):
        rp = _table(True)
        rp.change_c['tile'] = 25                      # mostly moved by clicks (a selector)
        with mock.patch.object(_rp, '_QTILE_ON', lambda: True):
            q = rp.q(S)
        self.assertEqual(q, frozenset([('prim', 0, 0)]))

    def test_cache_refreshes_when_effects_grow(self):
        rp = _table(False)
        with mock.patch.object(_rp, '_QTILE_ON', lambda: True):
            self.assertEqual(rp.q(S), frozenset([('prim', 0, 0)]))
            rp.effects[(5, (0, 'tile', 1, 1), frozenset([('prim', 10, 10, 20, 20)]))] = [set([1]), 1]
            self.assertIn(('tile', -6, 5), rp.q(S))


if __name__ == '__main__':
    unittest.main()
