"""PATCH 14 (2026-09-07): the RELPLAN seed MERGES into a table that already
exists.

Pinned here:
  * the loader leaves a RelPlan behind for every persisted blob (empty
    for a dropped v1 table, small for a young one); the seed must still
    reach it -- `if rp is None` never did (measured: 20 of 83 effects
    after 14 h on cd82 L2)
  * the merge is a union: an effect the live table learned on its own
    survives, and merging twice changes nothing
  * the blob's life-state (sel, last_painted) does NOT overwrite the
    life he is in
  * once per (game, level) per process; a missing file or a game the
    seed does not know is a no-op with zero errors
"""
import json
import os
import tempfile
import unittest

from seagi.world.arc_world import ARCWorld
from seagi.world.relplan import RelPlan
from seagi.brain.tests.test_relplan_hook import _Base, _Stub, Toy, teach, CLICK


class TestSeedMerge(_Base):
    def setUp(self):
        super().setUp()
        self._path0 = ARCWorld._relplan_seed_path
        self._seeded0 = set(ARCWorld._relplan_seeded)
        ARCWorld._relplan_seeded.clear()
        # the seed = a fully taught table in to_dict form (his own frames)
        teach(self.w)
        full = ARCWorld._relplan[('vgame', 1)]
        self.assertEqual(len(full.effects), 2)
        self.blob = full.to_dict()
        self.assertEqual(self.blob['v'], 2)
        fd, self.path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as fh:
            json.dump({'vgame': {'1': self.blob}}, fh)
        ARCWorld._relplan_seed_path = self.path
        ARCWorld._relplan.clear()
        ARCWorld._relplan_seeded.clear()
        ARCWorld._relplan_errors = 0

    def tearDown(self):
        ARCWorld._relplan_seed_path = self._path0
        ARCWorld._relplan_seeded.clear()
        ARCWorld._relplan_seeded.update(self._seeded0)
        try:
            os.unlink(self.path)
        except OSError:
            pass
        super().tearDown()

    def _one_step(self):
        """One hook call on a board where the picture is pursued."""
        toy = Toy()
        self.w._grid = toy.board()
        prev = self.w._grid
        toy.step(0)
        self.w._grid = toy.board()
        self.w._steps_episode = 3
        self.w._relplan_step(prev, 1, 0, False, None)

    def test_merges_into_an_existing_empty_table(self):
        # what rel_from_dict leaves behind after dropping a v1 blob
        ARCWorld._relplan[('vgame', 1)] = RelPlan()
        self._one_step()
        rp = ARCWorld._relplan[('vgame', 1)]
        self.assertEqual(len(rp.effects), 2)
        # the hook also learns from THIS step after the merge, so the
        # automaton holds the seed's transitions plus at most one
        self.assertGreaterEqual(len(rp.auto), len(self.blob['auto']))
        self.assertIn(('vgame', 1), ARCWorld._relplan_seeded)
        self.assertEqual(ARCWorld._relplan_errors, 0)
        self.assertTrue(ARCWorld._relplan_last.startswith('seeded vgam lv1 effects=0->2'))

    def test_merges_into_a_new_table_too(self):
        self._one_step()
        self.assertEqual(len(ARCWorld._relplan[('vgame', 1)].effects), 2)
        self.assertEqual(ARCWorld._relplan_errors, 0)

    def test_union_keeps_what_he_learned_live(self):
        # a young table: only the first apply of the scripted life
        rp = RelPlan()
        ARCWorld._relplan[('vgame', 1)] = rp
        toy = Toy(); self.w._grid = toy.board(); prev = self.w._grid
        toy.step(4); self.w._grid = toy.board(); self.w._steps_episode = 1
        # seeding is deferred so the live effect exists before the merge
        ARCWorld._relplan_seeded.add(('vgame', 1))
        self.w._relplan_step(prev, 1, 4, False, None)
        self.assertEqual(len(rp.effects), 1)
        own = set(rp.effects)
        ARCWorld._relplan_seeded.discard(('vgame', 1))
        self._one_step()
        self.assertTrue(own <= set(rp.effects))
        self.assertEqual(len(rp.effects), 2)
        # idempotent: a second merge changes nothing
        ARCWorld._relplan_seeded.discard(('vgame', 1))
        self._one_step()
        self.assertEqual(len(rp.effects), 2)
        self.assertEqual(ARCWorld._relplan_errors, 0)

    def test_life_state_is_not_overwritten(self):
        rp = RelPlan()
        ARCWorld._relplan[('vgame', 1)] = rp
        rp.sel = 15; rp.last_painted = None
        self.assertEqual(self.blob['sel'], 12)      # the seed's stale moment
        self._one_step()
        self.assertEqual(rp.sel, 15)
        self.assertIsNone(rp.last_painted)

    def test_once_per_process(self):
        ARCWorld._relplan[('vgame', 1)] = RelPlan()
        self._one_step()
        ARCWorld._relplan_last = ''
        self._one_step()
        self.assertEqual(ARCWorld._relplan_last, '')
        self.assertEqual(len(ARCWorld._relplan[('vgame', 1)].effects), 2)

    def test_missing_file_and_unknown_game_are_silent(self):
        ARCWorld._relplan_seed_path = self.path + '.absent'
        ARCWorld._relplan[('vgame', 1)] = RelPlan()
        self._one_step()
        self.assertEqual(len(ARCWorld._relplan[('vgame', 1)].effects), 0)
        self.assertEqual(ARCWorld._relplan_errors, 0)
        ARCWorld._relplan_seed_path = self.path
        ARCWorld._relplan_seeded.clear()
        with open(self.path, 'w') as fh:
            json.dump({'othergame': {'1': self.blob}}, fh)
        self._one_step()
        self.assertEqual(len(ARCWorld._relplan[('vgame', 1)].effects), 0)
        self.assertEqual(ARCWorld._relplan_errors, 0)


if __name__ == '__main__':
    unittest.main()
