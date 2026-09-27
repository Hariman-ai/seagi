"""Patch 44 HYPOTHESIS: the organ, its hooks, its hold, its persistence.

Pinned here (in the DAEMON's order: the hold is asked at the terminal BEFORE the
next life begins; the organ is asked on every tick, executed only sometimes):
  * a toy maze: the organ maps it, refutes the decoy after one step at it, reaches
    the exit, confirms it at a clear on its own step, and next life walks the
    shortest known path to WIN
  * an ask is stateless: asking twice on one board consumes nothing
  * gate absent: hyp_action None, hyp_hold False, nothing learned (byte-identical)
  * the hook learns from executed arrow steps only, never clicks or terminal frames
  * a life begins once per terminal reset / re-entry / clear, not twice
  * the hold: the life in flight must set a record (or be the first of the process)
    and a frontier must remain; a known win does not hold; wa30's shape holds <= 1
  * a clear on a step the organ did not propose confirms nothing
  * refutation ends when every candidate is refuted (a new round)
  * a new body drops the map; the map is capped; the blob round-trips
  * errors are counted, never raised
"""
import unittest
import numpy as np

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld
from seagi.world.relsense import RelSense
from seagi.world import goal_hypothesis as H

W = 64; BG = 0
EXIT = (10, 20, 12, 22)          # colour 4, 3x3
DECOY = (31, 33, 32, 34)         # colour 7, 2x2, next to the start
START = (30, 30)                 # colour 9 pixel
UP, DOWN, LEFT, RIGHT, CLICK = 0, 1, 2, 3, 4


class Maze(object):
    def __init__(self):
        self.pos = START; self.levels = 0; self.state = 'NOT_FINISHED'; self.t = 0

    def board(self):
        G = np.full((W, W), BG, int)
        G[5:40, 15:45] = 5
        G[6:39, 16:44] = BG
        G[EXIT[0]:EXIT[2] + 1, EXIT[1]:EXIT[3] + 1] = 4
        G[DECOY[0]:DECOY[2] + 1, DECOY[1]:DECOY[3] + 1] = 7
        G[20, 16:40] = 5                        # a wall with a gap at cols 40-43
        G[self.pos[0], self.pos[1]] = 9
        G[63, 0:max(1, 60 - self.t // 2)] = 3   # budget bar
        return [[int(x) for x in row] for row in G]

    def step(self, a):
        self.t += 1
        r, c = self.pos
        d = {UP: (-2, 0), DOWN: (2, 0), LEFT: (0, -2), RIGHT: (0, 2)}.get(a)
        if d is not None:
            nr, nc = r + d[0], c + d[1]
            G = np.array(self.board())
            if 6 <= nr <= 38 and 16 <= nc <= 43 and G[nr, nc] in (BG, 4):
                self.pos = (nr, nc)
        if H._gap((self.pos[0], self.pos[1], self.pos[0], self.pos[1]), EXIT) == 0:
            self.levels += 1
        if self.t >= 120:
            self.state = 'GAME_OVER'


class _Stub(object):
    def __init__(self, game='hgame'):
        self.game_id = game
        self._levels = 0
        self._grid = None
        self._steps_episode = 0
        self._lives_here = 0
        self._state = 'NOT_FINISHED'
        self._n_actions = 5
        self._acts = [1, 2, 3, 4, 6]
        self._rel_last = None
        self._hyp_win_path = False

    _bar_of = lambda self: 63
    _rel_objs_for = ARCWorld._rel_objs_for
    _relsense_step = ARCWorld._relsense_step
    _hyp_arrows = ARCWorld._hyp_arrows
    _hyp_key = ARCWorld._hyp_key
    _hyp_life_id = ARCWorld._hyp_life_id
    _hyp_step = ARCWorld._hyp_step
    hyp_action = ARCWorld.hyp_action
    hyp_hold = ARCWorld.hyp_hold
    rel_to_dict = ARCWorld.rel_to_dict
    rel_from_dict = ARCWorld.rel_from_dict


def _reset_tables():
    ARCWorld._rel = {}
    ARCWorld._relplan = {}
    ARCWorld._hyp = {}
    ARCWorld._hyp_life = {}
    ARCWorld._hyp_errors = 0
    ARCWorld._self = {}


def play(w, lives=6, seed=0, ask_twice=False):
    """Drive the stub through the hooks in the daemon's order.  Returns
    (actions per clear, holds)."""
    rng = np.random.RandomState(seed)
    clears = []; holds = 0
    for life in range(lives):
        m = Maze(); w._grid = m.board(); w._levels = 0; w._state = 'NOT_FINISHED'
        w._steps_episode = 0
        n = 0
        while True:
            prev_grid = w._grid; prev_lv = w._levels
            a = w.hyp_action(w._n_actions)
            if ask_twice:
                a2 = w.hyp_action(w._n_actions)
                assert a2 == a, 'an ask consumed something: %r then %r' % (a, a2)
            if a is None:
                a = int(rng.randint(0, 4))
            m.step(a); n += 1
            w._grid = m.board(); w._levels = m.levels; w._state = m.state
            success = m.levels > prev_lv
            w._relsense_step(prev_grid, prev_lv, success)
            w._hyp_step(prev_grid, prev_lv, a, a == CLICK, success)
            w._steps_episode += 1
            if success:
                clears.append(n); break
            if m.state == 'GAME_OVER':
                if w.hyp_hold():                 # asked at the terminal, before the reset
                    holds += 1
                break
        w._lives_here += 1                       # the terminal reset (a clear here: a fresh maze)
    return clears, holds


class HypothesisModuleTest(unittest.TestCase):
    def test_refute_then_confirm_then_shortest(self):
        _reset_tables()
        arc_world._HYPOTHESIS_ON = lambda: True
        arc_world._RELSENSE_ON = lambda: True
        w = _Stub()
        clears, holds = play(w, lives=40, ask_twice=True)
        hy = ARCWorld._hyp[('hgame', 0)]
        self.assertGreaterEqual(len(clears), 2, 'the maze was not cleared twice: %r' % (clears,))
        self.assertIn((7, 4, DECOY[0], DECOY[1]), hy.refuted)
        self.assertEqual(hy.confirmed, (4, 9, EXIT[0], EXIT[1]))
        self.assertTrue(hy.has_win())
        # after the first clear the shortest known path is walked: the second
        # clear takes no more actions than the first
        self.assertLessEqual(clears[1], clears[0])
        self.assertEqual(ARCWorld._hyp_errors, 0)
        self.assertEqual(hy.ctrl, (9, 1))

    def test_gate_absent_is_inert(self):
        _reset_tables()
        arc_world._HYPOTHESIS_ON = lambda: False
        arc_world._RELSENSE_ON = lambda: True
        w = _Stub()
        m = Maze(); w._grid = m.board()
        self.assertIsNone(w.hyp_action(5))
        self.assertFalse(w.hyp_hold())
        w._hyp_step(m.board(), 0, RIGHT, False, False)
        self.assertEqual(ARCWorld._hyp, {})

    def test_clicks_and_terminals_teach_nothing(self):
        _reset_tables()
        arc_world._HYPOTHESIS_ON = lambda: True
        arc_world._RELSENSE_ON = lambda: True
        w = _Stub()
        m = Maze(); w._grid = m.board()
        rs = RelSense(); rs.ctrl = (9, 1); ARCWorld._rel[('hgame', 0)] = rs
        g0 = m.board(); m.step(RIGHT); w._grid = m.board()
        w._hyp_step(g0, 0, CLICK, True, False)
        self.assertEqual(ARCWorld._hyp[('hgame', 0)].auto, {})
        w._state = 'GAME_OVER'
        w._hyp_step(g0, 0, RIGHT, False, False)
        self.assertEqual(ARCWorld._hyp[('hgame', 0)].auto, {})
        w._state = 'NOT_FINISHED'
        w._hyp_step(g0, 0, RIGHT, False, False)
        self.assertEqual(len(ARCWorld._hyp[('hgame', 0)].auto), 1)

    def test_a_life_begins_once(self):
        _reset_tables()
        arc_world._HYPOTHESIS_ON = lambda: True
        arc_world._RELSENSE_ON = lambda: True
        w = _Stub()
        m = Maze(); w._grid = m.board()
        rs = RelSense(); rs.ctrl = (9, 1); ARCWorld._rel[('hgame', 0)] = rs
        g0 = m.board()
        for _ in range(3):                                   # three steps, one life
            m.step(RIGHT); w._grid = m.board(); w._hyp_step(g0, 0, RIGHT, False, False); g0 = m.board()
        hy = ARCWorld._hyp[('hgame', 0)]
        self.assertEqual(hy.lives, 1)
        w._lives_here += 1                                   # a terminal reset
        m.step(LEFT); w._grid = m.board(); w._hyp_step(g0, 0, LEFT, False, False); g0 = m.board()
        self.assertEqual(hy.lives, 2)
        w._hyp_new_life = True                               # a re-entry (enter())
        m.step(LEFT); w._grid = m.board(); w._hyp_step(g0, 0, LEFT, False, False)
        self.assertEqual(hy.lives, 3)
        m.step(LEFT); w._hyp_step(g0, 0, LEFT, False, False)
        self.assertEqual(hy.lives, 3)
        # a clear opens the new level's first life, once, even on a click
        w._levels = 1
        w._hyp_step(g0, 0, CLICK, True, True)
        self.assertEqual(ARCWorld._hyp[('hgame', 1)].lives, 1)

    def test_hold_reads_the_life_in_flight(self):
        hy = H.Hypothesis()
        arrows = [0, 1, 2, 3]
        k = (30, 30, 0, 0); k2 = (30, 32, 0, 0); near = (14, 30, 0, 0)
        cands = [((4, 9, 10, 20), EXIT)]
        hy.begin_life()                                      # life 1 (first of the process)
        hy.observe(k, 3, k2, False)
        hy.act(k2, cands, arrows)
        self.assertTrue(hy.testing(arrows))                  # first life: owed a second
        hy.begin_life()                                      # life 2: same gap -> no record
        hy.act(k2, cands, arrows)
        self.assertFalse(hy.testing(arrows))
        hy.begin_life()                                      # life 3: nearer -> a record
        hy.observe(k2, 0, near, False)
        hy.act(near, cands, arrows)
        self.assertTrue(hy.testing(arrows))
        hy.begin_life()                                      # life 4: no record, and then no frontier
        for kk in (k, k2, near):
            for a in arrows:
                hy.observe(kk, a, k, False)
        hy.act(near, cands, arrows)
        self.assertFalse(hy.testing(arrows))

    def test_known_win_does_not_hold_by_itself(self):
        hy = H.Hypothesis()
        hy.begin_life()
        hy.pursued = ((4, 9, 10, 20), EXIT)
        hy.last_pick = ((11, 19, 0, 0), 3)
        hy.observe((11, 19, 0, 0), 3, None, True, by_organ=True)
        self.assertEqual(hy.confirmed, (4, 9, 10, 20))
        self.assertTrue(hy.has_win())
        hy.begin_life(); hy.begin_life()                     # a later life with no record
        self.assertFalse(hy.testing([0, 1, 2, 3]))
        # ...but the win path is proposed from its first step
        self.assertEqual(hy.act((11, 19, 0, 0), [], [0, 1, 2, 3]), 3)
        self.assertTrue(hy.win_path())

    def test_reached_at_once_holds_at_most_one_extra_life(self):
        """wa30's shape: every candidate is reached and refuted in the life it
        is first pursued; a fresh candidate each life must not hold him."""
        hy = H.Hypothesis()
        arrows = [0, 1, 2, 3]
        k = (30, 30, 0, 0)
        holds = 0
        for life in range(6):
            hy.begin_life()
            cands = [((7, 4 + life, 29, 29), (29, 29, 31, 31))]   # around him: gap 0
            a = hy.act(k, cands, arrows)                          # pursued, not yet refuted
            self.assertIsNotNone(a)
            self.assertNotIn(cands[0][0], hy.refuted)
            hy.observe(k, a, (30, 32, 0, 0), False, by_organ=True)   # one executed step of HIS at it
            hy.observe((30, 32, 0, 0), 2, k, False)
            hy.act(k, cands, arrows)                              # now refuted
            self.assertIn(cands[0][0], hy.refuted)
            holds += 1 if hy.testing(arrows) else 0
        self.assertLessEqual(holds, 1)

    def test_clear_on_another_organs_step_confirms_nothing(self):
        hy = H.Hypothesis()
        hy.begin_life()
        hy.pursued = ((7, 4, 31, 33), DECOY)
        hy.observe((11, 19, 0, 0), 3, None, True, by_organ=False)
        self.assertIsNone(hy.confirmed)
        self.assertTrue(hy.has_win())

    def test_all_refuted_starts_a_new_round(self):
        hy = H.Hypothesis()
        hy.begin_life()
        hy._refute((7, 4, 31, 33)); hy._refute((4, 9, 10, 20))
        # a frame with no candidate at all (occluded) does not reset the round
        self.assertIsNone(hy.act((30, 30, 0, 0), [], [0, 1, 2, 3], n_all=0))
        self.assertEqual(len(hy.refuted), 2)
        # every candidate on the board refuted: a new round
        self.assertIsNone(hy.act((30, 30, 0, 0), [], [0, 1, 2, 3], n_all=2))
        self.assertEqual(hy.refuted, set())
        self.assertEqual(hy.rounds, 1)

    def test_a_replayed_route_does_not_refute(self):
        hy = H.Hypothesis()
        hy.begin_life()
        k = (30, 30, 0, 0)
        cands = [((7, 4, 29, 29), (29, 29, 31, 31))]
        hy.act(k, cands, [0, 1, 2, 3])
        for _ in range(3):                                    # three steps at it, none his
            hy.observe(k, 0, (30, 32, 0, 0), False, by_organ=False)
        hy.act(k, cands, [0, 1, 2, 3])
        self.assertNotIn(cands[0][0], hy.refuted)

    def test_an_unresolved_step_is_tried_but_no_transition(self):
        hy = H.Hypothesis()
        hy.observe((1, 1, 0, 0), 0, None, False)
        self.assertIn(0, hy.tried[(1, 1, 0, 0)])
        self.assertNotIn((1, 1, 0, 0), hy.auto)
        self.assertFalse(hy.has_frontier([0]))

    def test_eviction_keeps_reachable_tried_marks(self):
        hy = H.Hypothesis()
        for i in range(H.MAX_KEYS + 10):
            hy.observe((i, 0, 0, 0), 0, (i + 1, 0, 0, 0), False)
        for i in range(H.MAX_KEYS + 10):
            hy.observe((i, 0, 0, 0), 0, (i + 1, 0, 0, 0), False)   # visited twice: kept
        hy.observe((H.MAX_KEYS + 10, 0, 0, 0), 1, None, False)         # tried, unresolved, reachable
        for _ in range(5):                                             # ...from a well-visited row
            hy.observe((H.MAX_KEYS + 9, 0, 0, 0), 0, (H.MAX_KEYS + 10, 0, 0, 0), False)
        for i in range(20):
            hy.observe((900000 + i, 0, 0, 0), 0, (900001 + i, 0, 0, 0), False)   # once-visited filler
        hy._evict()
        self.assertIn((H.MAX_KEYS + 10, 0, 0, 0), hy.tried)
        self.assertNotIn((H.MAX_KEYS + 10, 0, 0, 0), hy.auto)
        d = hy.to_dict()
        self.assertIn('%d,0,0,0' % (H.MAX_KEYS + 10), d['tried'])

    def test_new_body_clears_confirmation(self):
        hy = H.Hypothesis()
        hy.set_ctrl((9, 1)); hy.confirmed = (4, 9, 10, 20)
        hy.observe((1, 1, 0, 0), 0, (1, 3, 0, 0), False)
        hy.set_ctrl((12, 8))
        self.assertIsNone(hy.confirmed)

    def test_save_is_capped(self):
        hy = H.Hypothesis()
        for i in range(H.SAVE_KEYS + 50):
            hy.observe((i, 0, 0, 0), 0, (i + 1, 0, 0, 0), False)
        hy.observe((0, 5, 0, 0), 1, None, True)               # a WIN row, visited once
        for i in range(H.SAVE_REFUTED + 20):
            hy._refute((7, 4, i, 0))
        d = hy.to_dict()
        self.assertEqual(len(d['auto']), H.SAVE_KEYS)
        self.assertIn('0,5,0,0', d['auto'])
        self.assertEqual(len(d['refuted']), H.SAVE_REFUTED)
        self.assertLessEqual(len(hy.refuted), H.SAVE_REFUTED)

    def test_new_body_drops_the_map(self):
        hy = H.Hypothesis()
        hy.set_ctrl((9, 1))
        hy.observe((1, 1, 0, 0), 0, (1, 3, 0, 0), False)
        hy.set_ctrl((9, 1))
        self.assertEqual(len(hy.auto), 1)
        hy.set_ctrl((12, 8))
        self.assertEqual(hy.auto, {})
        self.assertEqual(hy.ctrl, (12, 8))

    def test_map_is_capped(self):
        hy = H.Hypothesis()
        for i in range(H.MAX_KEYS + 10):
            hy.observe((i, 0, 0, 0), 0, (i + 1, 0, 0, 0), False)
        self.assertLessEqual(len(hy.auto), H.MAX_KEYS // 2 + 10)
        self.assertGreater(hy.evicted, 0)

    def test_gap_is_row_by_row_and_column_by_column(self):
        self.assertEqual(H._gap((10, 22, 10, 22), (10, 20, 12, 22)), 0)
        self.assertEqual(H._gap((10, 24, 10, 24), (10, 20, 12, 22)), 2)
        self.assertEqual(H._gap((16, 21, 16, 21), (10, 20, 12, 22)), 4)
        self.assertEqual(H._gap((0, 0, 1, 1), (5, 40, 7, 41)), 39)

    def test_persistence_roundtrip(self):
        _reset_tables()
        arc_world._HYPOTHESIS_ON = lambda: True
        arc_world._RELSENSE_ON = lambda: True
        w = _Stub()
        play(w, lives=12)
        hy = ARCWorld._hyp[('hgame', 0)]
        d = w.rel_to_dict()
        self.assertIn('hyp', d['hgame']['0'])
        _reset_tables()
        w2 = _Stub()
        self.assertGreater(w2.rel_from_dict(d), 0)
        hy2 = ARCWorld._hyp[('hgame', 0)]
        self.assertEqual(hy2.auto, hy.auto)
        self.assertEqual(hy2.refuted, hy.refuted)
        self.assertEqual(hy2.confirmed, hy.confirmed)
        self.assertEqual(hy2.tried, hy.tried)
        self.assertEqual(hy2.ctrl, hy.ctrl)
        self.assertEqual(hy2.best_gap, hy.best_gap)
        # gate off: the table still persists (rm HYPOTHESIS_ON does not erase it)
        arc_world._HYPOTHESIS_ON = lambda: False
        self.assertIn('hyp', w2.rel_to_dict()['hgame']['0'])
        # another body's blob is not merged
        d['hgame']['0']['hyp']['ctrl'] = [12, 8]
        _reset_tables(); w3 = _Stub()
        hy3 = H.Hypothesis(); hy3.ctrl = (9, 1); ARCWorld._hyp[('hgame', 0)] = hy3
        arc_world._HYPOTHESIS_ON = lambda: True
        w3.rel_from_dict(d)
        self.assertEqual(hy3.auto, {})

    def test_arrows_exclude_the_click(self):
        w = _Stub()
        self.assertEqual(w._hyp_arrows(), [0, 1, 2, 3])

    def test_candidates_are_unique_static_objects(self):
        hy = H.Hypothesis()
        m = Maze()
        from seagi.world.relsense import segment
        bg, objs = segment(m.board(), 63)
        c = hy.candidates(objs, (9, 1), set(), 63)
        ids = sorted(cid for cid, _ in c)
        self.assertIn((4, 9, EXIT[0], EXIT[1]), ids)
        self.assertIn((7, 4, DECOY[0], DECOY[1]), ids)
        self.assertFalse(any(cid[0] == 9 for cid in ids))     # never himself
        hy.refuted.add((7, 4, DECOY[0], DECOY[1]))
        c = hy.candidates(objs, (9, 1), set(), 63)
        self.assertFalse(any(cid[0] == 7 for cid, _ in c))

    def test_errors_are_counted_not_raised(self):
        _reset_tables()
        arc_world._HYPOTHESIS_ON = lambda: True
        arc_world._RELSENSE_ON = lambda: True
        w = _Stub()
        rs = RelSense(); rs.ctrl = (9, 1); ARCWorld._rel[('hgame', 0)] = rs
        ARCWorld._hyp[('hgame', 0)] = H.Hypothesis()
        m = Maze(); w._grid = m.board()

        def _boom():
            raise RuntimeError('boom')
        w._bar_of = _boom
        self.assertIsNone(w.hyp_action(5))
        self.assertGreaterEqual(ARCWorld._hyp_errors, 1)


if __name__ == '__main__':
    unittest.main()
