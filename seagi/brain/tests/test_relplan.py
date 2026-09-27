"""RELPLAN: a learned operator model and a planner for a picture.

Pinned here, on a toy paint world with NO cd82 knowledge in the module:
  * the indicator is discovered (objects that appear/disappear outside
    the canvas), lines are never indicators
  * apply's effect is learned per indicator state, the painted colour is
    read from the indicator, a swatch click is learned to set a colour
  * from a fresh life the plan is apply, click swatch, cycle, apply --
    and it completes the picture in four actions
  * the same choice EXECUTED on the same board three times without the
    board changing is a wrong model, not a plan (returns None)
  * the plan's own prediction is judged when its action runs
  * to_dict/from_dict round trip gives the same first action, and loading
    twice changes no count
  * an unconfirmed picture is acted on only for a full or half-way plan
"""
import unittest
from unittest import mock
import numpy as np

from seagi.world.relsense import segment
from seagi.world import relplan as _rp
from seagi.world.relplan import RelPlan, STUCK_AFTER

W = 64; BG = 5
A = (34, 27, 43, 36)      # canvas
B = (3, 3, 12, 12)        # target
SW = {15: (3, 40), 12: (3, 46)}   # swatch top-left corners (3x3)
MASKS = {0: [(r, c) for r in range(5) for c in range(10)],
         1: [(r, c) for r in range(5, 10) for c in range(10)]}
CLICK = 5


class Toy(object):
    """A canvas, a target (top half 15, bottom half 12), two swatches, an
    indicator rectangle of the selected colour whose width shows the
    mask, and a budget line on row 63 that shrinks one cell a step."""
    def __init__(self):
        self.shape = 0; self.colour = 15; self.canvas = np.zeros((10, 10), int); self.t = 0

    def board(self):
        G = np.full((W, W), BG, int)
        for r in range(10):
            for c in range(10):
                G[B[0] + r, B[1] + c] = 15 if r < 5 else 12
        G[A[0]:A[2] + 1, A[1]:A[3] + 1] = self.canvas
        for col, (r, c) in SW.items():
            G[r - 1:r + 4, c - 1:c + 4] = 4          # a frame every swatch shares
            G[r:r + 3, c:c + 3] = col
        w = 6 if self.shape == 0 else 9
        G[20:24, 40:40 + w] = self.colour
        G[63, 0:max(1, 60 - self.t)] = 4
        return G

    def step(self, a, aim=None):
        self.t += 1
        if a == 2:
            self.shape = 1 - self.shape
        elif a == 4:
            for r, c in MASKS[self.shape]:
                self.canvas[r, c] = self.colour
        elif a == CLICK and aim is not None:
            # the whole framed button selects, ring included
            for col, (r, c) in SW.items():
                if r - 1 <= aim[0] < r + 4 and c - 1 <= aim[1] < c + 4:
                    self.colour = col

    def mm(self):
        t = np.array([[15 if r < 5 else 12 for c in range(10)] for r in range(10)])
        return int((self.canvas != t).sum())


def run(toy, rp, script):
    """Feed a scripted life to the learner."""
    Gp = toy.board(); op = segment(Gp, 63)[1]
    for a, aim in script:
        toy.step(a, aim)
        Gc = toy.board(); oc = segment(Gc, 63)[1]
        rp.observe(Gp, op, Gc, oc, a, a == CLICK, aim, A, B)
        Gp, op = Gc, oc


def teach(rp):
    toy = Toy()
    # the colour click lands on the FRAME ring of the 12 swatch (2, 47).
    # CARRIERHUE (patch 17): an indicator must have WORN two colours
    # before it can be read, so the lesson opens by clicking 12 then 15
    # while the shape-0 indicator is showing (it wears 12, then 15).
    run(toy, rp, [(CLICK, (2, 47)), (CLICK, (2, 41)),
                  (4, None), (2, None), (CLICK, (2, 47)), (4, None)])
    assert toy.mm() == 0
    return toy


class TestLearning(unittest.TestCase):
    def test_indicator_found_and_lines_ignored(self):
        rp = RelPlan(); teach(rp)
        toy = Toy()
        G = toy.board(); outs = rp.outside(G, segment(G, 63)[1], A, B)
        self.assertFalse(any(b[0] == 63 for s, c, b in outs))
        # the indicator rectangle plus the two swatches (one shared shape)
        S, cols = rp.state(outs)
        self.assertEqual(len(S), 5)     # rect, two swatch blocks, two frames -- placed
        # only the unique object can carry the selected colour
        self.assertEqual(len(cols), 1)
        self.assertTrue(all(rp._carrier_ok(s) for s in rp.carrier))

    def test_effects_and_colour(self):
        rp = RelPlan(); teach(rp)
        self.assertEqual(len(rp.effects), 2)
        for (a, ck, S), (cells, n) in rp.effects.items():
            self.assertEqual(a, 4); self.assertIsNone(ck); self.assertEqual(len(cells), 50)
        self.assertIn(12, rp.swatch)
        # the key names an object OF colour 12, not the shared frame
        self.assertEqual(rp.swatch[12][0][0], 12)
        self.assertEqual(rp.click_index, CLICK)
        self.assertTrue(rp.carrier)

    def test_plan_completes_a_fresh_life(self):
        rp = RelPlan(); teach(rp)
        toy = Toy(); acts = []
        for _ in range(8):
            G = toy.board(); objs = segment(G, 63)[1]
            r = rp.plan(G, objs, A, B, 6, board_key=str(toy.t))
            if r is None:
                break
            a, aim, info = r
            acts.append(a)
            toy.step(a, aim)
            if toy.mm() == 0:
                break
        self.assertEqual(toy.mm(), 0, acts)
        self.assertEqual(acts, [4, CLICK, 2, 4])

    def test_plan_none_when_done_or_untaught(self):
        rp = RelPlan()
        toy = Toy(); G = toy.board()
        self.assertIsNone(rp.plan(G, segment(G, 63)[1], A, B, 6))
        teach(rp)
        toy = Toy(); toy.canvas[:5] = 15; toy.canvas[5:] = 12
        G = toy.board()
        self.assertIsNone(rp.plan(G, segment(G, 63)[1], A, B, 6))

    def test_stuck_guard_counts_executions(self):
        rp = RelPlan(); teach(rp)
        toy = Toy(); G = toy.board(); objs = segment(G, 63)[1]
        # asking again and again is free; EXECUTING the pick on a board
        # that does not change is what counts
        for _ in range(STUCK_AFTER + 2):
            self.assertIsNotNone(rp.plan(G, objs, A, B, 6, board_key=str(_)))
        for k in range(STUCK_AFTER):
            r = rp.plan(G, objs, A, B, 6, board_key=str(k))   # a changing hash must not matter
            self.assertIsNotNone(r, k)
            a, aim, info = r
            rp.observe(G, objs, G, objs, a, a == CLICK, aim, A, B)   # nothing happened
        self.assertIsNone(rp.plan(G, objs, A, B, 6, board_key="new"))
        self.assertEqual(rp.plan_judged, STUCK_AFTER); self.assertEqual(rp.plan_reached, 0)
        rp.begin_life()
        self.assertIsNotNone(rp.plan(G, objs, A, B, 6, board_key="new"))

    def test_plan_prediction_is_judged_on_execution(self):
        rp = RelPlan(); teach(rp)
        toy = Toy()
        for _ in range(6):
            G = toy.board(); objs = segment(G, 63)[1]
            r = rp.plan(G, objs, A, B, 6, board_key=str(toy.t))
            if r is None:
                break
            a, aim, info = r
            toy.step(a, aim)
            G2 = toy.board(); objs2 = segment(G2, 63)[1]
            rp.observe(G, objs, G2, objs2, a, a == CLICK, aim, A, B)
        self.assertEqual(toy.mm(), 0)
        self.assertEqual(rp.plan_judged, 4)
        self.assertEqual(rp.plan_reached, 4)
        self.assertEqual(rp.exec_n, 4)

    def test_from_dict_idempotent(self):
        rp = RelPlan(); teach(rp)
        d = rp.to_dict()
        rp2 = RelPlan(); rp2.from_dict(d); snap = {k: (set(v[0]), v[1]) for k, v in rp2.effects.items()}
        auto = {k: dict(v) for k, v in rp2.auto.items()}; seen = dict(rp2.seen)
        rp2.from_dict(d)
        self.assertEqual({k: (set(v[0]), v[1]) for k, v in rp2.effects.items()}, snap)
        self.assertEqual({k: dict(v) for k, v in rp2.auto.items()}, auto)
        self.assertEqual(dict(rp2.seen), seen)

    def test_unconfirmed_needs_a_real_plan(self):
        rp = RelPlan(); teach(rp)
        # keep only a 20-cell fragment of the top-half effect
        keep = None
        for k, e in list(rp.effects.items()):
            if keep is None and len(e[0]) == 50 and min(e[0]) < 50:
                keep = k
            else:
                rp.effects.pop(k)
        rp.effects[keep][0] = set(sorted(rp.effects[keep][0])[:20])
        toy = Toy(); G = toy.board(); objs = segment(G, 63)[1]
        self.assertIsNone(rp.plan(G, objs, A, B, 6, confirmed=False))
        self.assertIsNotNone(rp.plan(G, objs, A, B, 6, confirmed=True))

    def test_empty_state_effect_never_applies(self):
        rp = RelPlan(); teach(rp)
        rp.effects[(4, None, frozenset())] = [set(range(100)), 5]
        eff, auto = rp.views()
        self.assertFalse(any(not k[2] for k in eff))

    def test_subset_match_must_be_unique(self):
        rp = RelPlan(); teach(rp)
        # two effects learned under richer states that share the current
        # state's only shape: neither may apply through the subset
        toy = Toy(); G = toy.board(); objs = segment(G, 63)[1]
        outs = rp.outside(G, objs, A, B); Sfull, cols = rp.state(outs); S = rp.q(Sfull)
        psig = next(iter(S))[0]
        o = next(t for t in Sfull if t[0] == psig)
        rp.effects.clear()
        # f1/f2 sit just beside the primary and change as often: parts of its state
        f1 = ("f1", o[1] + 1, o[4] + 1, o[1] + 2, o[4] + 2)     # just right of the primary
        f2 = ("f2", o[1] + 1, o[4] + 1, o[1] + 2, o[4] + 2)
        rp.effects[(4, None, frozenset([o, f1]))] = [set(range(50)), 5]
        rp.effects[(4, None, frozenset([o, f2]))] = [set(range(50, 100)), 5]
        rp.mutable.update(["f1", "f2"])
        rp.change["f1"] = rp.change["f2"] = rp.change[psig] - rp.change_c.get(psig, 0)
        # an ambiguous match yields NO PLAN; with CURIOUS on it may yield a probe instead,
        # which is a different, legitimate answer -- pin the planner, not the gate
        with mock.patch.object(_rp, "_CURIOUS_ON", lambda: False):
            self.assertIsNone(rp.plan(G, objs, A, B, 6, confirmed=True))
        # with only one richer state the subset applies
        rp.effects.pop((4, None, frozenset([o, f2])))
        self.assertIsNotNone(rp.plan(G, objs, A, B, 6, confirmed=True))

    def test_exact_evidence_beats_pooled(self):
        rp = RelPlan()
        rp.mutable.update(["p", "x", "zz", "nw"]); rp.change.update({"p": 12, "x": 10, "zz": 10, "nw": 10})
        P = ("p", 0, 0, 3, 3); X = ("x", 5, 5, 6, 6); Z = ("zz", 0, 5, 1, 6); NW = ("nw", 0, 0, 1, 1)
        S = frozenset([P]); T = frozenset([X])
        rp.auto[(S, 0)] = {T: 3}                              # exact evidence, MIN_EXACT samples
        rp.auto[(frozenset([P, Z]), 0)] = {frozenset([NW]): 50}   # a comparable superset, loudly wrong
        rp._auto_n += 2
        eff, auto = rp.views()
        self.assertEqual(rp._next(auto, rp.q(S), 0), rp.q(T))
        rp.auto[(S, 0)] = {T: 1}; rp._auto_n += 1           # too little exact evidence: pooled wins
        eff, auto = rp.views()
        self.assertEqual(rp._next(auto, rp.q(S), 0), rp.q(frozenset([NW])))

    def test_spoiled_effect_burned_for_the_life(self):
        rp = RelPlan(); teach(rp)
        toy = Toy(); G = toy.board(); objs = segment(G, 63)[1]
        r = rp.plan(G, objs, A, B, 6, confirmed=True)
        self.assertEqual(r[0], 4)
        op = rp._pending["op"]; self.assertIsNotNone(op)
        # the bottom half was already right (12); the apply painted it 15
        good = Toy(); good.canvas[5:] = 12
        G = good.board(); objs = segment(G, 63)[1]
        r = rp.plan(G, objs, A, B, 6, confirmed=True); op = rp._pending["op"]
        bad = Toy(); bad.canvas[5:] = 15
        rp.observe(G, objs, bad.board(), segment(bad.board(), 63)[1], 4, False, None, A, B)
        self.assertIn(op, rp._burned)
        r2 = rp.plan(G, objs, A, B, 6, confirmed=True)
        self.assertTrue(r2 is None or rp._pending["op"] != op)
        rp.begin_life()
        self.assertEqual(rp._burned, set())

    def test_a_part_that_moves_is_a_different_state(self):
        rp = RelPlan(); teach(rp)
        toy = Toy(); G = np.array(toy.board())
        # a two-cell mark inside the indicator rectangle, left vs right
        G1 = G.copy(); G1[25:27, 41] = 4
        G2 = G.copy(); G2[25:27, 44] = 4
        for Gx in (G1, G2):
            outs = rp.outside(Gx, segment(Gx, 63)[1], A, B)
            for s_, c_, b_ in outs:
                if c_ == 4 and (b_[2] - b_[0] + 1) * (b_[3] - b_[1] + 1) == 2:
                    rp.mutable.add(s_); rp.change[s_] = 4     # the mark: a part, changed by arrows
                if c_ == 15 and (b_[2] - b_[0] + 1) * (b_[3] - b_[1] + 1) >= 20:
                    rp.mutable.add(s_); rp.change[s_] = 6     # the rectangle (not a swatch): the primary
        q1 = rp.q(rp.state(rp.outside(G1, segment(G1, 63)[1], A, B))[0])
        q2 = rp.q(rp.state(rp.outside(G2, segment(G2, 63)[1], A, B))[0])
        self.assertNotEqual(q1, q2)
        self.assertEqual(len(q1), len(q2))

    def test_life_start_resets_selection(self):
        rp = RelPlan(); teach(rp)
        self.assertIsNotNone(rp.sel)
        rp.begin_life()
        self.assertIsNone(rp.sel); self.assertIsNone(rp.last_painted); self.assertIsNone(rp._pending)

    def test_round_trip(self):
        rp = RelPlan(); teach(rp)
        d = rp.to_dict()
        import json
        d = json.loads(json.dumps(d))
        rp2 = RelPlan(); self.assertGreater(rp2.from_dict(d), 0)
        toy = Toy(); G = toy.board(); objs = segment(G, 63)[1]
        self.assertEqual(rp.plan(G, objs, A, B, 6)[0], rp2.plan(G, objs, A, B, 6)[0])
        self.assertEqual(rp2.click_index, CLICK)
        self.assertEqual(set(rp2.effects), set(rp.effects))

    def test_prediction_counter(self):
        rp = RelPlan(); teach(rp)
        # a second identical life: every apply is now predicted
        teach(rp)
        self.assertEqual(rp.judged, 2)
        self.assertEqual(rp.reached, 2)


if __name__ == "__main__":
    unittest.main()
