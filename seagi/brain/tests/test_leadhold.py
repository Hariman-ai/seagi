"""PATCH 18 (2026-09-07) -- LEADHOLD helpers: the hand being played is
per (game, life, level); a new life, level or game starts with no lead.
The decision itself lives inline in WorldActor.propose() (no unit
harness exists for that method); it is covered by the live counters
rel_plan_lead_kept / rel_plan_lead_route and the reviewer's trace."""
import unittest

from seagi.brain.capabilities.world_actor import WorldActor


class _W(object):
    game_id = 'g1'; _lives_here = 3; _levels = 1


class _A(object):
    _lead_key = WorldActor._lead_key
    _lead_now = WorldActor._lead_now
    _lead_set = WorldActor._lead_set

    def __init__(self):
        self.world = _W()


class TestLeadHold(unittest.TestCase):
    def test_no_lead_until_someone_acts(self):
        a = _A()
        self.assertIsNone(a._lead_now())
        a._lead_set('route'); self.assertEqual(a._lead_now(), 'route')
        a._lead_set('plan'); self.assertEqual(a._lead_now(), 'plan')

    def test_a_new_life_level_or_game_clears_the_lead(self):
        a = _A(); a._lead_set('plan')
        a.world._lives_here = 4; self.assertIsNone(a._lead_now())
        a._lead_set('plan'); a.world._levels = 2; self.assertIsNone(a._lead_now())
        a._lead_set('plan'); a.world.game_id = 'g2'; self.assertIsNone(a._lead_now())

    def test_missing_world_attrs_are_tolerated(self):
        a = _A(); a.world = object()
        self.assertIsNone(a._lead_now())
        a._lead_set('route'); self.assertEqual(a._lead_now(), 'route')


if __name__ == '__main__':
    unittest.main()
