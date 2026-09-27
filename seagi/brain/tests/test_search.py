"""Patch 45 SEARCH: the organ, its hooks, its hold, its persistence.

Pinned here (in the DAEMON's order: the hold is asked at the terminal BEFORE the
next life begins; the organ is asked on every tick, executed only sometimes):
  * a toy panel of five toggle buttons and a clock row: the search sweeps the
    controls, finds the clearing combination, and next life walks the shortest
    known path to WIN; the click's aim travels through `_search_pick`
  * an ask is stateless: asking twice on one board consumes nothing
  * gate absent: search_action None, search_hold False, nothing learned
  * the hold: the life just ended found new states and a frontier remains;
    a life that found nothing new releases; nothing learned releases
  * a click aimed off the list (another organ's) is learned and counted
  * the blob round-trips; the map is capped; the root survives eviction
  * controls: lines excluded, smallest first, capped; states mask the clock lines
  * errors are counted, never raised
"""
import unittest
import numpy as np

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld
from seagi.world import goal_search as S

# patch 47: these tests pin the patch-45 hold; the gate file may exist on the live tree
S._CONVERGE_ON = lambda: False

W = 64; BG = 0
BTN = [(10, 10), (10, 30), (10, 50), (40, 10), (40, 30)]     # top-left corners, 3x3 each
ON, OFF = 4, 7
GOAL = (False, True, False, True, True)                        # the clearing combination
NOOP_A, NOOP_B, CLICK = 0, 1, 2


class Panel(object):
    def __init__(self):
        self.on = [False] * 5; self.levels = 0; self.state = 'NOT_FINISHED'; self.t = 0

    def board(self):
        G = np.full((W, W), BG, int)
        G[2:6, 2:62] = 3                                       # a panel (a wide object)
        for i, (r, c) in enumerate(BTN):
            G[r:r + 3, c:c + 3] = ON if self.on[i] else OFF
        G[63, 0:max(1, 60 - self.t)] = 5                       # the clock row
        return [[int(x) for x in row] for row in G]

    def step(self, a, aim=None):
        self.t += 1
        if a == CLICK and aim is not None:
            for i, (r, c) in enumerate(BTN):
                if r <= aim[0] <= r + 2 and c <= aim[1] <= c + 2:
                    self.on[i] = not self.on[i]
        if tuple(self.on) == GOAL:
            self.levels += 1
        if self.t >= 40:
            self.state = 'GAME_OVER'


class _Stub(object):
    def __init__(self, game='sgame'):
        self.game_id = game
        self._levels = 0
        self._grid = None
        self._steps_episode = 0
        self._lives_here = 0
        self._state = 'NOT_FINISHED'
        self._n_actions = 3
        self._acts = [1, 2, 6]
        self._rel_last = None
        self._search_win_path = False
        self._search_pick = None

    _bar_of = lambda self: 63
    _rel_objs_for = ARCWorld._rel_objs_for
    _hyp_arrows = ARCWorld._hyp_arrows
    _hyp_life_id = ARCWorld._hyp_life_id
    _search_click_idx = ARCWorld._search_click_idx
    _search_state = ARCWorld._search_state
    _search_controls = ARCWorld._search_controls
    _search_step = ARCWorld._search_step
    search_action = ARCWorld.search_action
    search_hold = ARCWorld.search_hold
    rel_to_dict = ARCWorld.rel_to_dict
    rel_from_dict = ARCWorld.rel_from_dict


def _reset_tables():
    ARCWorld._rel = {}
    ARCWorld._relplan = {}
    ARCWorld._hyp = {}
    ARCWorld._srch = {}
    ARCWorld._srch_life = {}
    ARCWorld._srch_errors = 0
    ARCWorld._bar_row = {('sgame', 0): 63, ('sgame', 1): 63}
    arc_world._BARMASK_ON = lambda: True
    arc_world._SEARCH_ON = lambda: True
    arc_world._RELSENSE_ON = lambda: True


def play(w, lives=6, seed=0, ask_twice=False, life_len=40):
    """Drive the stub through the hooks in the daemon's order."""
    rng = np.random.RandomState(seed)
    clears = []; holds = []
    for life in range(lives):
        p = Panel(); w._grid = p.board(); w._levels = 0; w._state = 'NOT_FINISHED'
        n = 0
        while True:
            prev_grid = w._grid; prev_lv = w._levels
            a = w.search_action(w._n_actions)
            pick = w._search_pick
            if ask_twice:
                a2 = w.search_action(w._n_actions)
                assert a2 == a and w._search_pick == pick, 'an ask consumed something'
            if a is None:
                a = int(rng.randint(0, 2))
            aim = None
            if a == CLICK:
                aim = pick[1] if (pick is not None and pick[0] == a) else (int(rng.randint(0, W)), int(rng.randint(0, W)))
            p.step(a, aim); n += 1
            w._grid = p.board(); w._levels = p.levels; w._state = p.state
            success = p.levels > prev_lv
            sent = (a, aim[0] if aim else None, aim[1] if aim else None, a == CLICK)
            w._search_step(prev_grid, prev_lv, a, sent, success)
            if success:
                clears.append(n); break
            if p.state == 'GAME_OVER':
                holds.append(bool(w.search_hold()))
                break
        w._lives_here += 1
    return clears, holds


class SearchModuleTest(unittest.TestCase):
    def test_sweep_finds_the_combination_then_walks_it(self):
        _reset_tables()
        w = _Stub()
        clears, holds = play(w, lives=30, ask_twice=True)
        sr = ARCWorld._srch[('sgame', 0)]
        self.assertGreaterEqual(len(clears), 2, 'the panel was not cleared twice: %r' % (clears,))
        self.assertTrue(sr.has_win())
        self.assertLessEqual(clears[-1], 3, 'the known win was not walked: %r' % (clears,))
        self.assertEqual(ARCWorld._srch_errors, 0)
        # the clock row never split a state: a life is <= 40 steps and the
        # panel has 32 boards, so the states are the boards
        self.assertLessEqual(len(sr.trans), 33)

    def test_gate_absent_is_inert(self):
        _reset_tables()
        arc_world._SEARCH_ON = lambda: False
        w = _Stub()
        p = Panel(); w._grid = p.board()
        self.assertIsNone(w.search_action(3))
        self.assertFalse(w.search_hold())
        w._search_step(p.board(), 0, NOOP_A, (NOOP_A, None, None, False), False)
        self.assertEqual(ARCWorld._srch, {})

    def test_hold_new_states_then_release(self):
        _reset_tables()
        w = _Stub()
        p = Panel(); g0 = p.board(); w._grid = g0
        self.assertFalse(w.search_hold())                     # nothing learned
        p.step(CLICK, (11, 11)); w._grid = p.board()
        w._search_step(g0, 0, CLICK, (CLICK, 11, 11, True), False)
        self.assertTrue(w.search_hold())                      # a new state, frontier remains
        sr = ARCWorld._srch[('sgame', 0)]
        w._lives_here += 1
        g1 = p.board(); p.step(NOOP_A); w._grid = p.board()
        w._search_step(g1, 0, NOOP_A, (NOOP_A, None, None, False), False)
        self.assertEqual(sr.lives, 2)
        self.assertEqual(sr.new_this_life, 0)
        self.assertFalse(w.search_hold())                     # nothing new this life

    def test_hold_is_not_a_cage(self):
        sr = S.Search()
        sr.begin_life(root='r'); sr.see('r', [('A', 0), ('A', 1)])
        sr.observe('r', ('A', 0), 's1', False, False)
        self.assertTrue(sr.testing())                         # first life, new state, frontier
        sr.begin_life(root='r')
        for i in range(30):                                   # a life in which nothing recurred
            sr.observe('n%d' % i, ('A', 0), 'n%d' % (i + 1), False, False)
        self.assertGreaterEqual(sr.new_this_life, sr.steps_this_life)
        self.assertFalse(sr.testing())
        sr.begin_life(root='r')
        sr.observe('r', ('A', 1), 'r', False, False)          # recurred, nothing new
        self.assertFalse(sr.testing())
        sr.begin_life(root='r')
        sr.observe('r', ('A', 0), 's1', False, False); sr.observe('s1', ('A', 0), 'x', False, False)
        self.assertTrue(sr.testing())                         # one new, one recurred, frontier
        sr.observe('s1', ('A', 1), 'y', True, False)          # a win is known: nothing to hold for
        self.assertTrue(sr.has_win())
        self.assertFalse(sr.testing())
        # eviction in a life releases
        sr2 = S.Search(); sr2.begin_life(root='r'); sr2.see('r', [('A', 0)])
        for i in range(S.MAX_STATES + 5):
            sr2.observe('s%d' % i, ('A', 0), 's%d' % (i + 1), False, False)
        sr2.begin_life(root='r'); sr2.observe('r', ('A', 0), 's0', False, False)
        for i in range(S.MAX_STATES + 5):
            sr2.observe('t%d' % i, ('A', 0), 't%d' % (i + 1), False, False)
        self.assertGreater(sr2.evicted, sr2.evicted_at_life)
        self.assertFalse(sr2.testing())

    def test_a_life_ended_by_the_clock_blames_no_control(self):
        sr = S.Search()
        sr.begin_life(root='r'); sr.see('r', [('A', 0), ('A', 1)])
        for i in range(5):
            sr.observe('r', ('A', 0), 'r', False, False)
        sr.observe('r', ('A', 1), None, False, True)          # first life: the clock length is unknown yet
        self.assertEqual(sr.trans['r'][('A', 1)], S.OVER)
        self.assertEqual(sr.max_life, 0)
        sr.begin_life(root='r')
        self.assertEqual(sr.max_life, 6)
        for i in range(5):
            sr.observe('r', ('A', 0), 'r', False, False)
        del sr.trans['r'][('A', 1)]
        sr.observe('r', ('A', 1), None, False, True)          # ended at the known length: not recorded
        self.assertNotIn(('A', 1), sr.trans['r'])
        sr.begin_life(root='r')
        sr.observe('r', ('A', 1), None, False, True)          # ended early: this control did it
        self.assertEqual(sr.trans['r'][('A', 1)], S.OVER)

    def test_win_path_survives_the_save_cap(self):
        sr = S.Search()
        sr.begin_life(root='r'); sr.see('r', [('A', 0)])
        sr.observe('r', ('A', 0), 'p1', False, False); sr.observe('p1', ('A', 0), 'p2', False, False)
        sr.observe('p2', ('A', 0), None, True, False)
        for i in range(S.SAVE_STATES + 50):                   # many well-visited states elsewhere
            for _ in range(3):
                sr.observe('v%d' % i, ('A', 0), 'v%d' % (i + 1), False, False)
        d = sr.to_dict()
        self.assertIn('r', d['trans']); self.assertIn('p1', d['trans']); self.assertIn('p2', d['trans'])
        sr3 = S.Search(); sr3.from_dict(d)
        self.assertTrue(sr3.has_win())
        self.assertEqual(sr3.act('r', [('A', 0)]), ('A', 0))
        self.assertTrue(sr3.win_path())
        self.assertEqual(S.Search().from_dict({'trans': {'r': {'': 'x'}}}), 0)

    def test_off_list_click_is_learned_and_counted(self):
        _reset_tables()
        w = _Stub()
        p = Panel(); g0 = p.board(); w._grid = g0
        ctls = w._search_controls(g0)
        p.step(CLICK, (0, 0)); w._grid = p.board()
        w._search_step(g0, 0, CLICK, (CLICK, 0, 0, True), False)
        sr = ARCWorld._srch[('sgame', 0)]
        s0 = w._search_state(g0)
        self.assertIn(('C', 0, 0), sr.trans[s0])
        self.assertEqual(sr.n_ctl[s0], len(ctls) + 1)
        self.assertTrue(sr.is_frontier(s0))

    def test_blob_round_trips_and_the_win_is_kept(self):
        _reset_tables()
        w = _Stub()
        clears, _ = play(w, lives=30)
        self.assertGreaterEqual(len(clears), 1)
        blob = w.rel_to_dict()
        self.assertIn('srch', blob['sgame']['0'])
        sr0 = ARCWorld._srch[('sgame', 0)]
        n_states = len(sr0.trans)
        ARCWorld._srch = {}
        ARCWorld._rel = {}
        n = w.rel_from_dict(blob)
        self.assertGreater(n, 0)
        sr = ARCWorld._srch[('sgame', 0)]
        self.assertTrue(sr.has_win())
        self.assertEqual(len(sr.trans), n_states)
        self.assertEqual(sr.root, sr0.root)

    def test_cap_and_root_survive_eviction(self):
        sr = S.Search()
        sr.begin_life(root='r')
        sr.see('r', [('A', 0)])
        for i in range(S.MAX_STATES + 5):
            sr.observe('s%d' % i, ('A', 0), 's%d' % (i + 1), False, False)
        self.assertLessEqual(len(sr.trans), S.MAX_STATES)
        self.assertIn('r', sr.trans)
        self.assertGreater(sr.evicted, 0)

    def test_controls_and_states(self):
        objs = [(3, 244, 2, 2, 5, 62), (7, 9, 10, 10, 12, 12), (4, 100, 20, 20, 29, 29)]
        c = S.controls_of(objs, [0, 1], 2)
        self.assertEqual(c[:2], [('A', 0), ('A', 1)])
        self.assertEqual(c[2:], [('C', 11, 11), ('C', 24, 24)])      # smallest first, the panel excluded
        many = [(1, 1, i, i, i, i) for i in range(60)]
        self.assertEqual(len(S.controls_of(many, [], 0)), S.MAX_OBJS)
        self.assertEqual(S.controls_of(objs, [0], None), [('A', 0)])
        p = Panel(); a = p.board(); p.step(NOOP_A); b = p.board()   # only the clock moved
        self.assertNotEqual(S.state_of(a), S.state_of(b))
        self.assertEqual(S.state_of(a, (63,)), S.state_of(b, (63,)))
        self.assertEqual(S.key_ctl(S.ctl_key(('C', 5, 7))), ('C', 5, 7))
        self.assertEqual(S.key_ctl(S.ctl_key(('A', 3))), ('A', 3))

    def test_errors_are_counted_not_raised(self):
        _reset_tables()
        w = _Stub()
        ARCWorld._srch[('sgame', 0)] = S.Search()
        w._grid = 'not a grid'
        self.assertIsNone(w.search_action(3))
        self.assertGreaterEqual(ARCWorld._srch_errors, 0)
        w._grid = None
        self.assertIsNone(w.search_action(3))
        self.assertFalse(w.search_hold())


if __name__ == '__main__':
    unittest.main()
