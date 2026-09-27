"""RELSTEER: the actor learns what each action does to the pursued
relation in a situation, and takes the best one.

Pinned here:
  * gate absent: nothing asked, nothing learned (no test reads the file)
  * with no situation from the world: nothing picked
  * learning is a running mean of (progress - regress) per
    (game, level, situation, action)
  * an action is taken only after two observations with a positive mean
  * the situation used for learning is the one BEFORE the step
"""
import unittest

from seagi.brain.capabilities import world_actor as wa_mod


class _World(object):
    game_id = "vgame"
    _levels = 1

    def __init__(self, ctx):
        self.ctx = ctx

    def rel_context(self):
        return self.ctx


class _Actor(object):
    _rel_key = wa_mod.WorldActor._rel_key if hasattr(wa_mod, "WorldActor") else None

    def __init__(self, world):
        self.world = world


def _bind(cls):
    # bind the three methods from whichever class defines them
    for name in ("_rel_key", "_rel_pick", "_rel_learn"):
        for candidate in vars(wa_mod).values():
            if isinstance(candidate, type) and hasattr(candidate, name):
                setattr(cls, name, getattr(candidate, name))
                break
    return cls


_Actor = _bind(_Actor)


class TestRelSteer(unittest.TestCase):

    def setUp(self):
        self._g = wa_mod._RELSTEER_ON
        wa_mod._RELSTEER_ON = lambda: True

    def tearDown(self):
        wa_mod._RELSTEER_ON = self._g

    def test_no_situation_no_pick(self):
        a = _Actor(_World(None))
        self.assertIsNone(a._rel_pick(4))
        self.assertEqual(getattr(a, "rel_asks", 0), 0)

    def test_learns_and_then_steers(self):
        a = _Actor(_World(((14, 9), 1, 1)))
        self.assertIsNone(a._rel_pick(4), "nothing learned yet")
        # action 3 moved the relation toward the goal twice
        a._rel_pick(4); a._rel_learn(3, {"rel_n": 1, "rel_progress": 0.1, "rel_regress": 0.0})
        a._rel_pick(4); a._rel_learn(3, {"rel_n": 1, "rel_progress": 0.1, "rel_regress": 0.0})
        # action 0 moved it away twice
        a._rel_pick(4); a._rel_learn(0, {"rel_n": 1, "rel_progress": 0.0, "rel_regress": 0.1})
        a._rel_pick(4); a._rel_learn(0, {"rel_n": 1, "rel_progress": 0.0, "rel_regress": 0.1})
        self.assertEqual(a._rel_pick(4), 3)
        # it steered on every pick after the second action-3 observation
        self.assertEqual(a.rel_steers, 3)
        e = a._rel_act[("vgame", 1, ((14, 9), 1, 1), 3)]
        self.assertEqual(e[1], 2)
        self.assertAlmostEqual(e[0], 0.1)

    def test_one_observation_is_not_enough(self):
        a = _Actor(_World(((14, 9), 0, 1)))
        a._rel_pick(4); a._rel_learn(3, {"rel_n": 1, "rel_progress": 0.5, "rel_regress": 0.0})
        self.assertIsNone(a._rel_pick(4))

    def test_negative_mean_is_not_taken(self):
        a = _Actor(_World(((14, 9), 0, 1)))
        for _ in range(3):
            a._rel_pick(4); a._rel_learn(3, {"rel_n": 1, "rel_progress": 0.0, "rel_regress": 0.2})
        self.assertIsNone(a._rel_pick(4))

    def test_learning_uses_the_situation_before_the_step(self):
        w = _World(((14, 9), 1, 1)); a = _Actor(w)
        a._rel_pick(4)
        w.ctx = ((14, 9), -1, 0)        # the world moved on
        a._rel_learn(3, {"rel_n": 1, "rel_progress": 0.1, "rel_regress": 0.0})
        self.assertIn(("vgame", 1, ((14, 9), 1, 1), 3), a._rel_act)

    def test_no_relation_nothing_learned(self):
        a = _Actor(_World(((14, 9), 1, 1)))
        a._rel_pick(4); a._rel_learn(4, {"rel_n": 0, "rel_progress": 0.5})
        self.assertEqual(a._rel_act, {})

    def test_a_picture_is_never_steered(self):
        # review 2026-09-05: "apply" on cd82 kept a positive-tiny mean over
        # hundreds of no-op presses and would have been taken on 96% of steps
        a = _Actor(_World(("eq",)))
        for _ in range(3):
            a._rel_pick(6); a._rel_learn(4, {"rel_n": 1, "rel_progress": 0.3, "rel_regress": 0.0})
        self.assertIsNone(a._rel_pick(6))
        self.assertEqual(getattr(a, "rel_asks", 0), 0)

    def test_an_action_known_inert_here_is_not_taken(self):
        w = _World(((14, 9), 1, 1)); w.board_inert_here = lambda x: x == 3
        a = _Actor(w)
        for _ in range(2):
            a._rel_pick(4); a._rel_learn(3, {"rel_n": 1, "rel_progress": 0.2, "rel_regress": 0.0})
            a._rel_pick(4); a._rel_learn(1, {"rel_n": 1, "rel_progress": 0.1, "rel_regress": 0.0})
        self.assertEqual(a._rel_pick(4), 1, "3 is better but known dead here")


if __name__ == "__main__":
    unittest.main()
