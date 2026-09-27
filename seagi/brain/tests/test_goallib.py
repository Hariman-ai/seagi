"""Patch 48 LIBRARY: the goal library (one across games), its learning from clear
lives against dry lives, its persistence, and the search organ's goal-guided
frontier and refutation.  Gate /root/LIBRARY_ON.

Pinned:
  * read/evaluate/reached on a toy board: an eq pair and a count- class, fixed boxes
  * a level's goal = the relation reaching its goal in clear lives and NOT in dry
    lives; the side-effect relation that reaches its goal in both is not paid
  * a clear nothing explains is counted UNEXPLAINED, once per level
  * the library round-trips through to_dict/from_dict (levels included)
  * Search: with a goal the frontier is best-first on the pursued value; without,
    nearest (the patch-45/47 path); at_goal at REACH_FRAC; refuted keys persist
  * the world hook: with the gate on the stub plays to a clear with lib_errors 0,
    the life-start board is kept, the clear is observed, PURSUE lines are written;
    with the gate off nothing of the library is touched
"""
import os
import unittest
import importlib.util

import numpy as np

from seagi.world import goal_library as L
from seagi.world import goal_search as S
from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld

W = 64


def board(pairs):
    G = np.zeros((W, W), dtype=int)
    for (r0, c0, r1, c1), col in pairs:
        G[r0:r1 + 1, c0:c1 + 1] = col
    return G


class TestLibrary(unittest.TestCase):
    def test_read_evaluate_reached(self):
        # two 3x3 boxes: A colour 2, B colour 3 (differ: 9 cells); three 1-cell marks colour 5
        G0 = board([((10, 10, 12, 12), 2), ((10, 30, 12, 32), 3), ((40, 5, 40, 5), 5), ((40, 7, 40, 7), 5), ((40, 9, 40, 9), 5)])
        bg, objs, inst = L.read(G0)
        rids = set(inst)
        self.assertIn(('eq', (10, 10, 12, 12), (10, 30, 12, 32)), rids)
        eq = inst[('eq', (10, 10, 12, 12), (10, 30, 12, 32))]
        self.assertEqual(eq.value, 9)
        self.assertEqual(eq.schema[0], 'eq')
        cm = inst[('count-', (5, 1))]
        self.assertEqual(cm.value, 3)
        # the boxes stay fixed when evaluated on another board
        G1 = board([((10, 10, 12, 12), 3), ((10, 30, 12, 32), 3), ((40, 5, 40, 5), 5)])
        v = L.evaluate(G1, inst)
        self.assertEqual(v[eq.rid], 0)
        self.assertEqual(v[cm.rid], 1)
        self.assertTrue(L.reached(9, 0)); self.assertTrue(L.reached(90, 15)); self.assertFalse(L.reached(90, 45))
        self.assertFalse(L.reached(0, 0))

    def test_goal_is_what_clears_have_and_dry_lives_lack(self):
        lib = L.Library()
        # start: A != B (the goal pair), C != D (a side pair that he always equalises)
        G0 = board([((10, 10, 12, 12), 2), ((10, 30, 12, 32), 3), ((30, 10, 32, 12), 4), ((30, 30, 32, 32), 6)])
        dry = board([((10, 10, 12, 12), 2), ((10, 30, 12, 32), 3), ((30, 10, 32, 12), 6), ((30, 30, 32, 32), 6)])
        win = board([((10, 10, 12, 12), 3), ((10, 30, 12, 32), 3), ((30, 10, 32, 12), 6), ((30, 30, 32, 32), 6)])
        for _ in range(4):
            lib.observe_life('gam1-x', 0, G0, dry, False)
        got = lib.learn_clear('gam1-x', 0, G0, win)
        self.assertEqual(got, ['eq:10,10,12,12~10,30,12,32'])
        paid = [k for k, v in lib.paid.items() if v > 0]
        self.assertEqual(len(paid), 1)
        self.assertTrue(paid[0].startswith('eq|'))
        self.assertEqual(lib.games[paid[0]], {'gam1'})
        self.assertEqual(lib.unexplained, 0)
        self.assertTrue(any(e.startswith('PAID') for e in lib.events))
        # a second game, same kind of goal: the schema is now known from two games
        for _ in range(L.MIN_DRY):
            lib.observe_life('gam2-y', 0, G0, dry, False)
        got2 = lib.learn_clear('gam2-y', 0, G0, win)
        self.assertTrue(got2)
        self.assertEqual(lib.games[paid[0]], {'gam1', 'gam2'})
        self.assertGreater(lib.prior(tuple(paid[0].split('|'))), 0.5)

    def test_unexplained_clear_is_counted_once(self):
        lib = L.Library()
        G0 = board([((10, 10, 12, 12), 2), ((10, 30, 12, 32), 3)])
        for _ in range(L.MIN_DRY):
            lib.observe_life('g', 0, G0, G0, False)                # dry lives: a null exists
        self.assertEqual(lib.learn_clear('g', 0, G0, G0), [])     # nothing moved
        self.assertEqual(lib.unexplained, 1)
        lib.learn_clear('g', 0, G0, G0)
        self.assertEqual(lib.unexplained, 1)

    def test_no_dry_life_pays_nothing(self):
        lib = L.Library()
        G0 = board([((10, 10, 12, 12), 2), ((10, 30, 12, 32), 3), ((30, 10, 32, 12), 4), ((30, 30, 32, 32), 6)])
        win = board([((10, 10, 12, 12), 3), ((10, 30, 12, 32), 3), ((30, 10, 32, 12), 6), ((30, 30, 32, 32), 6)])
        self.assertEqual(lib.learn_clear('g', 0, G0, win), [])     # two pairs reached: no null, no pay
        self.assertEqual(sum(lib.paid.values()), 0.0)
        self.assertEqual(lib.unexplained, 0)                       # not judged either
        self.assertEqual(lib.levels['g/0']['n_clear'], 1)
        dry = board([((10, 10, 12, 12), 2), ((10, 30, 12, 32), 3), ((30, 10, 32, 12), 6), ((30, 30, 32, 32), 6)])
        for _ in range(L.MIN_DRY):
            lib.observe_life('g', 0, G0, dry, False)
        got = lib.learn_clear('g', 0, G0, win)                     # now the null separates them
        self.assertEqual(got, ['eq:10,10,12,12~10,30,12,32'])

    def test_save_replaces_seed(self):
        seed = L.Library()
        seed.paid['eq|1|1|oo'] = 3.0; seed.games['eq|1|1|oo'].add('cd82')
        lib = L.Library(); lib.from_dict(seed.to_dict())
        lib.paid['eq|1|1|oo'] = 1.0                                # credit moved in-process
        save = lib.to_dict()
        lib2 = L.Library(); lib2.from_dict(seed.to_dict()); lib2.from_dict(save, replace=True)
        self.assertEqual(lib2.paid['eq|1|1|oo'], 1.0)

    def test_round_trip(self):
        lib = L.Library()
        G0 = board([((10, 10, 12, 12), 2), ((10, 30, 12, 32), 3)])
        win = board([((10, 10, 12, 12), 3), ((10, 30, 12, 32), 3)])
        for _ in range(L.MIN_DRY):
            lib.observe_life('g', 0, G0, G0, False)
        lib.learn_clear('g', 0, G0, win)
        lib.refute_key('near|1|0|0|0', 'gx')
        lib.refute_key('near|1|0|0|0', 'gx')
        self.assertEqual(len(lib.tried_games['near|1|0|0|0']), 1)   # once per (schema, game)
        d = lib.to_dict()
        lib2 = L.Library(); lib2.from_dict(d)
        self.assertEqual(lib2.paid, dict(lib.paid))
        self.assertEqual(lib2.tried, dict(lib.tried))
        self.assertEqual(lib2.tried_games, dict(lib.tried_games))
        self.assertEqual(lib2.clears, 1)
        self.assertIn('g/0', lib2.levels)
        self.assertEqual(lib2.levels['g/0']['paid_rid'], lib.levels['g/0']['paid_rid'])


class TestSearchGoal(unittest.TestCase):
    def _table(self):
        # r -> a (frontier, value 5) ; r -> b -> c (frontier, value 1)
        sr = S.Search()
        sr.begin_life(root='r')
        sr.see('r', [('A', 0), ('A', 1)])
        sr.observe('r', ('A', 0), 'a', False, False); sr.see('a', [('A', 0), ('A', 1)])
        sr.observe('r', ('A', 1), 'b', False, False); sr.see('b', [('A', 0)])
        sr.observe('b', ('A', 0), 'c', False, False); sr.see('c', [('A', 0), ('A', 1)])
        sr.observe('a', ('A', 0), 'a', False, False)     # a keeps ('A', 1) untried
        return sr

    def test_best_first_with_a_goal_nearest_without(self):
        sr = self._table()
        self.assertEqual(sr.untried('r', [('A', 0), ('A', 1)]), [])
        self.assertEqual(sr.act('r', [('A', 0), ('A', 1)]), ('A', 0))            # nearest frontier: a
        sr.set_goal('eq:x', 'eq|1|1|oo', 10.0)
        sr.note_value('a', 5.0); sr.note_value('c', 1.0); sr.note_value('b', 8.0)
        self.assertEqual(sr.act('r', [('A', 0), ('A', 1)]), ('A', 1))            # best-first: toward c
        sr.clear_goal()
        self.assertEqual(sr.act('r', [('A', 0), ('A', 1)]), ('A', 0))

    def test_plan_misses_fall_back_to_nearest(self):
        sr = self._table()
        sr.set_goal('eq:x', 'eq|1|1|oo', 10.0)
        sr.note_value('a', 5.0); sr.note_value('c', 1.0)
        self.assertEqual(sr.act('r', [('A', 0), ('A', 1)]), ('A', 1))       # best-first
        for i in range(3):
            a = sr.act('r', [('A', 0), ('A', 1)])                           # a planned step on a known edge
            self.assertIsNotNone(a)
            sr.observe('r', a, 'z%d' % i, False, False)                     # that lands somewhere else
        self.assertEqual(sr.plan_miss, 3)
        self.assertIsNotNone(sr.act('r', [('A', 0), ('A', 1)]))             # nearest for the rest of the life
        sr.begin_life(root='r')
        self.assertEqual(sr.plan_miss, 0)

    def test_flat_value_falls_back_to_nearest(self):
        sr = self._table()
        sr.set_goal('eq:x', 'eq|1|1|oo', 10.0)
        sr.note_value('a', 5.0); sr.note_value('c', 5.0)           # the relation does not vary
        self.assertEqual(sr.act('r', [('A', 0), ('A', 1)]), ('A', 0))   # nearest, as patch 45

    def test_at_goal_and_refuted_persist(self):
        sr = self._table()
        sr.set_goal('eq:x', 'eq|1|1|oo', 10.0)
        sr.note_value('c', 2.0)
        self.assertTrue(sr.at_goal('c'))
        sr.note_value('a', 3.0)
        self.assertFalse(sr.at_goal('a'))
        sr.lib_refuted.add('eq:x')
        d = sr.to_dict()
        sr2 = S.Search(); sr2.from_dict(d)
        self.assertIn('eq:x', sr2.lib_refuted)
        self.assertIsNone(sr2.pursued)


class TestWorldHook(unittest.TestCase):
    def _T(self):
        spec = importlib.util.spec_from_file_location(
            'test_search_mod', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_search.py'))
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        # the stub binds the world methods it knew; bind the library hooks the same way
        for name in ('_lib_line', '_lib_self', '_lib_life_start', '_lib_life_end', '_lib_goal', '_lib_refute'):
            setattr(T._Stub, name, getattr(ARCWorld, name))
        for name in ('_lib_get', '_lib_flush'):
            setattr(T._Stub, name, ARCWorld.__dict__[name])
        return T

    def _reset(self):
        ARCWorld._lib = None
        ARCWorld._lib_start = {}; ARCWorld._lib_sb0 = {}; ARCWorld._lib_insts = {}
        ARCWorld._lib_pursues = 0; ARCWorld._lib_refutes = 0; ARCWorld._lib_errors = 0

    def test_gate_on_plays_learns_and_persists(self):
        T = self._T(); T._reset_tables(); self._reset()
        arc_world._LIBRARY_ON = lambda: True
        try:
            w = T._Stub()
            clears, holds = T.play(w, lives=30)
            self.assertGreaterEqual(len(clears), 1)
            self.assertEqual(ARCWorld._lib_errors, 0)
            self.assertIsNotNone(ARCWorld._lib)
            self.assertIn(('sgame', 0), ARCWorld._lib_start)
            self.assertGreaterEqual(ARCWorld._lib.clears, 1)
            self.assertGreaterEqual(ARCWorld._lib_pursues, 1)
            sr = ARCWorld._srch[('sgame', 0)]
            self.assertTrue(sr.val or sr.pursued is None)
            d = w.rel_to_dict()
            self.assertIn('_goallib', d)
            self.assertEqual(int(d['_goallib']['clears']), ARCWorld._lib.clears)
            self._reset()
            w.rel_from_dict(d)
            self.assertIsNotNone(ARCWorld._lib)
            self.assertGreaterEqual(ARCWorld._lib.clears, 1)
        finally:
            arc_world._LIBRARY_ON = lambda: False
            T._reset_tables(); self._reset()

    def test_gate_off_touches_nothing(self):
        T = self._T(); T._reset_tables(); self._reset()
        arc_world._LIBRARY_ON = lambda: False
        try:
            w = T._Stub()
            clears, holds = T.play(w, lives=12)
            self.assertIsNone(ARCWorld._lib)
            self.assertEqual(ARCWorld._lib_start, {})
            self.assertEqual(ARCWorld._lib_pursues, 0)
            sr = ARCWorld._srch[('sgame', 0)]
            self.assertIsNone(sr.pursued)
            self.assertEqual(sr.val, {})
        finally:
            T._reset_tables(); self._reset()


if __name__ == '__main__':
    unittest.main()
