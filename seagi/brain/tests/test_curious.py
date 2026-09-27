"""PATCH 23 (2026-09-08) -- CURIOUS: when nothing known gains, go and try a
mask never applied.  PATCH 25 -- the walk to it is LATCHED and may cross a
known apply; the probe is marked on execution; a floor gate.

Toy: two masks (shape 0 = top half, shape 1 = bottom half), cycled by
action 2.  Teach ONLY the top-half apply (plus the cycle, so the
automaton knows shape 1 exists), never the bottom-half apply.  Paint the
top half correctly; now no known effect gains.

Pinned here:
  * gate on: the plan steps toward the untried mask (the cycle arrow),
    then applies there as a probe; the probe teaches the bottom-half
    effect and the next plan completes the picture
  * gate off: the plan is silent (`no_gain`) at the same point
  * a state is probed once per life, marked when the probe RAN (patch
    25): a pick the world refused is asked again; begin_life resets
  * a probe is not scored as a prediction (plan_judged unchanged)

Toy3 (patch 25): three masks; APPLY advances the mask 0->1->2->0 as cd82's
indicator does when he paints; arrow 2 cycles 0<->1 and 2->0, so mask 2
sits BEHIND an apply from 1.  Target rows 0-5 = 15, rows 6-9 = 12.
  * the stalled life walks 0 -(2)-> 1 -(apply, SPOILS row 5)-> 2 and the
    NEXT plan still serves the walk (the final probe at 2), not the
    repair a greedy round would take; the probe teaches mask 2 and the
    picture then completes
  * floor gate: at mm0 above the floor the fallback is silent
  * a refused walk step is asked again without a second latch
  * a walk over budget is dropped and its target not retried this life
"""
import unittest
from unittest import mock
import numpy as np

from seagi.world import relplan as _rp
from seagi.world.relplan import RelPlan, MAX_WALK_STEPS, STUCK_AFTER
from seagi.world.relsense import segment
from seagi.brain.tests.test_relplan import Toy, run, A, B, CLICK, SW, BG, W


def _teach_half(rp):
    """Top-half apply in shape 0, click 12, click 15, cycle to shape 1
    (seen, never applied from), cycle back."""
    toy = Toy()
    run(toy, rp, [(CLICK, (2, 47)), (CLICK, (2, 41)), (4, None), (2, None), (2, None)])
    return toy


def _step(rp, toy, a, aim=None):
    Gp = toy.board(); op = segment(Gp, 63)[1]
    toy.step(a, aim)
    Gc = toy.board(); oc = segment(Gc, 63)[1]
    rp.observe(Gp, op, Gc, oc, a, a == CLICK, aim, A, B)


def _plan(rp, toy, **kw):
    G = toy.board(); objs = segment(G, 63)[1]
    return rp.plan(G, objs, A, B, 6, **kw)


_ON = {'_CURIOUS_ON': lambda: True, '_NAVPAINT_ON': lambda: True,
       # later gates read as OFF here whatever /root holds; their own
       # test classes switch them on
       '_PROBEHUE_ON': lambda: False, '_SAMEMASK_ON': lambda: False,
       '_LOOKAHEAD_ON': lambda: False, '_SEQHUE_ON': lambda: False,
       '_REFUTE_ON': lambda: False, '_LAFLOOR_ON': lambda: False}


def _gates(**over):
    d = dict(_ON); d.update(over)
    return [mock.patch.object(_rp, k, v) for k, v in d.items()]


class _Gated(unittest.TestCase):
    """CURIOUS and NAVPAINT read as ON whatever /root holds."""
    def setUp(self):
        self._ps = _gates()
        for p in self._ps:
            p.start()

    def tearDown(self):
        for p in self._ps:
            p.stop()


class TestCurious(_Gated):
    def _ready(self):
        rp = RelPlan(); _teach_half(rp)
        toy = Toy()                                     # fresh life, shape 0, colour 15
        rp.begin_life()
        r = _plan(rp, toy); self.assertEqual(r[0], 4)   # the known top-half paint gains
        _step(rp, toy, 4)
        self.assertEqual(toy.mm(), 50)
        return rp, toy

    def test_gate_off_is_silent(self):
        rp, toy = self._ready()
        with mock.patch.object(_rp, '_CURIOUS_ON', lambda: False):
            self.assertIsNone(_plan(rp, toy)); self.assertEqual(rp.why, 'no_gain')

    def test_probe_reaches_applies_and_teaches(self):
        rp, toy = self._ready()
        r = _plan(rp, toy)
        self.assertIsNotNone(r); self.assertEqual(r[0], 2); self.assertTrue(r[2].get('probe'))
        self.assertFalse(r[2].get('final')); self.assertEqual(rp.walks, 1)
        judged0 = rp.plan_judged
        _step(rp, toy, 2)                            # now in shape 1
        r = _plan(rp, toy)
        self.assertIsNotNone(r); self.assertEqual(r[0], 4); self.assertTrue(r[2].get('probe'))
        self.assertTrue(r[2].get('final'))
        self.assertEqual(rp.probe_applies, 1)
        n_eff = len(rp.effects)
        _step(rp, toy, 4)                            # the probe paints the bottom half (15)
        self.assertEqual(rp.plan_judged, judged0)    # not scored as a prediction
        self.assertGreater(len(rp.effects), n_eff)   # a new effect learned
        self.assertEqual(rp.probe_exec, 1); self.assertIsNone(rp._walk)
        # the next plan knows the bottom-half mask: click 12, apply -> done
        acts = []
        for _ in range(6):
            r = _plan(rp, toy)
            if r is None:
                break
            acts.append(r[0]); _step(rp, toy, r[0], r[1])
            if toy.mm() == 0:
                break
        self.assertEqual(toy.mm(), 0, acts)

    def test_probe_marked_on_execution_once_per_life(self):
        rp, toy = self._ready()
        _step(rp, toy, 2)
        r = _plan(rp, toy); self.assertEqual(r[0], 4); self.assertTrue(r[2].get('probe'))
        # the world refuses (nothing executed): the mask is still open and
        # is asked for again (patch 25 -- marked on execution)
        self.assertEqual(len(rp._probed), 0)
        r = _plan(rp, toy); self.assertIsNotNone(r); self.assertEqual(r[0], 4)
        self.assertEqual(rp.probe_exec, 0)
        _step(rp, toy, 4)                            # executed
        self.assertEqual(rp.probe_exec, 1); self.assertEqual(len(rp._probed), 1)
        rp.begin_life()
        self.assertEqual(len(rp._probed), 0); self.assertIsNone(rp._walk)


class TestCuriousFixes(_Gated):
    def test_no_probe_on_an_unconfirmed_picture(self):
        rp = RelPlan(); _teach_half(rp); toy = Toy(); rp.begin_life()
        _step(rp, toy, 4)                                # top half painted
        G = toy.board(); objs = segment(G, 63)[1]
        self.assertIsNone(rp.plan(G, objs, A, B, 6, confirmed=False))
        self.assertIsNotNone(rp.plan(G, objs, A, B, 6, confirmed=True))

    def test_apply_action_is_the_best_supported_paint(self):
        rp = RelPlan(); _teach_half(rp)
        # a stray arrow 'paint' with support 1 beside apply effects
        eff, auto = rp.views()
        S_any = next(iter(k[2] for k in eff if k[1] is None))
        rp.effects[(2, None, frozenset([('stray', 0, 0, 1, 1)]))] = [set([0]), 1]
        toy = Toy(); rp.begin_life(); _step(rp, toy, 4)
        _step(rp, toy, 2)
        r = _plan(rp, toy)
        self.assertIsNotNone(r); self.assertTrue(r[2].get('probe'))
        self.assertEqual(r[0], 4, 'the probe pressed the stray arrow instead of applying')

    def test_unpriced_masks_are_probed_first(self):
        rp = RelPlan()
        S = frozenset([('p', 0, 0)])
        priced = frozenset([('p', 0, 0), ('x', 1, 1)])   # a superset of a known Sreq: already priced
        fresh = frozenset([('z', 0, 0)])
        rp.effects[(4, None, frozenset([('p', 3, 3, 5, 5)]))] = [set([1, 2]), 5]
        rp.mutable.update(('p', 'x', 'z')); rp.change.update({'p': 10, 'x': 10, 'z': 10}); rp.seen.update({'p': 10, 'x': 10, 'z': 10})
        eff, auto = rp.views()
        dist0 = {S: (0, None), priced: (1, 1), fresh: (2, 3)}
        canvas = np.zeros((10, 10), int)
        r = rp._probe(S, dist0, eff, 6, canvas, None, 50, True)
        self.assertIsNotNone(r)
        self.assertEqual(r[0], 3, 'the priced mask was probed before the unmatched one')
        self.assertEqual(r[2]['dist'], 2)


# ---------------------------------------------------------------- patch 25
MASKS3 = {0: [(r, c) for r in range(5) for c in range(10)],        # top half
          1: [(5, c) for c in range(10)],                           # row 5
          2: [(r, c) for r in range(6, 10) for c in range(10)]}     # rows 6-9
WIDTH3 = {0: 6, 1: 9, 2: 12}
NEXT_ARROW3 = {0: 1, 1: 0, 2: 0}


class Toy3(Toy):
    """Three masks; APPLY paints and advances the mask 0->1->2->0; arrow 2
    cycles 0<->1 and 2->0, so mask 2 is reached only by applying from 1.
    Target rows 0-5 = 15, rows 6-9 = 12."""
    def board(self):
        G = Toy.board(self)
        for r in range(10):
            for c in range(10):
                G[B[0] + r, B[1] + c] = 15 if r < 6 else 12
        G[20:24, 40:52] = BG
        G[20:24, 40:40 + WIDTH3[self.shape]] = self.colour
        return G

    def step(self, a, aim=None):
        self.t += 1
        if a == 2:
            self.shape = NEXT_ARROW3[self.shape]
        elif a == 4:
            for r, c in MASKS3[self.shape]:
                self.canvas[r, c] = self.colour
            self.shape = (self.shape + 1) % 3
        elif a == CLICK and aim is not None:
            for col, (r, c) in SW.items():
                if r - 1 <= aim[0] < r + 4 and c - 1 <= aim[1] < c + 4:
                    self.colour = col

    def mm(self):
        t = np.array([[15 if r < 6 else 12 for c in range(10)] for r in range(10)])
        return int((self.canvas != t).sum())


def _teach3(rp):
    """Applies from masks 0 and 1 known (and the arrows 0<->1, 2->0); mask
    2 seen once, never applied from."""
    toy = Toy3()
    run(toy, rp, [(CLICK, (2, 47)), (CLICK, (2, 41)),      # hues: 12 then 15
                  (4, None),                                # apply@0: top half 15 -> mask 1
                  (2, None), (2, None),                     # 1 -> 0 -> 1
                  (CLICK, (2, 47)),                         # colour 12
                  (4, None),                                # apply@1: row 5 = 12 -> mask 2
                  (2, None)])                               # 2 -> 0
    return toy


def _stalled3(rp):
    """A fresh life at the floor: rows 0-5 already right, colour 12
    selected, mask 0 showing; no known effect gains (mm 40)."""
    toy = Toy3()
    toy.canvas[0:6, :] = 15
    rp.begin_life()
    _step(rp, toy, CLICK, (2, 47))                          # select 12
    assert toy.mm() == 40
    return toy


class TestWalk(_Gated):
    def test_walk_through_a_spoiling_apply_still_probes(self):
        rp = RelPlan(); _teach3(rp); toy = _stalled3(rp)
        # 1. the stall: the fallback latches a walk toward mask 2
        r = _plan(rp, toy)
        self.assertIsNotNone(r, rp.why); self.assertTrue(r[2].get('probe'))
        self.assertEqual(r[0], 2); self.assertFalse(r[2].get('final'))
        self.assertEqual(r[2]['dist'], 2); self.assertEqual(rp.walks, 1)
        _step(rp, toy, 2)                                    # mask 1
        # 2. the apply on the way: a probe step, a known apply, counted
        r = _plan(rp, toy)
        self.assertIsNotNone(r, rp.why); self.assertTrue(r[2].get('probe'))
        self.assertEqual(r[0], 4); self.assertFalse(r[2].get('final'))
        self.assertEqual(rp.probe_paint_applies, 1); self.assertEqual(rp.probe_applies, 0)
        judged0 = rp.plan_judged
        _step(rp, toy, 4)                                    # row 5 -> 12: SPOILS, mask 2
        self.assertEqual(toy.mm(), 50)
        self.assertEqual(rp.plan_judged, judged0)            # unjudged
        self.assertEqual(len(rp.spoiled), 0); self.assertEqual(len(rp._burned), 0)
        # 3. THE LATCH: a repair (+10: arrow, arrow, click 15, apply) is
        #    available and a greedy round would take it; the walk is served
        self.assertIsNotNone(rp._walk)
        r = _plan(rp, toy)
        self.assertIsNotNone(r, rp.why); self.assertTrue(r[2].get('probe'))
        self.assertEqual(r[0], 4); self.assertTrue(r[2].get('final'))
        self.assertEqual(rp.probe_applies, 1)
        n_eff = len(rp.effects)
        _step(rp, toy, 4)                                    # rows 6-9 -> 12: mask 2 learned
        self.assertEqual(toy.mm(), 10)
        self.assertGreater(len(rp.effects), n_eff)
        self.assertEqual(rp.probe_exec, 1); self.assertIsNone(rp._walk)
        self.assertEqual(len(rp._probed), 1)
        # 4. with the walk over, the rounds repair row 5 and finish
        acts = []
        for _ in range(8):
            r = _plan(rp, toy)
            if r is None:
                break
            self.assertFalse(r[2].get('probe'), 'a second walk was latched with a repair at hand')
            acts.append(r[0]); _step(rp, toy, r[0], r[1])
            if toy.mm() == 0:
                break
        self.assertEqual(toy.mm(), 0, acts)

    def test_without_the_latch_the_repair_wins(self):
        """The control: the same board after the spoil, latch cleared --
        the greedy round takes the repair (what killed patch 24)."""
        rp = RelPlan(); _teach3(rp); toy = _stalled3(rp)
        _plan(rp, toy); _step(rp, toy, 2)
        _plan(rp, toy); _step(rp, toy, 4)
        rp._walk = None
        r = _plan(rp, toy)
        self.assertIsNotNone(r, rp.why); self.assertFalse(r[2].get('probe'))
        self.assertGreater(r[2]['rounds'], 0)

    def test_floor_gate(self):
        rp = RelPlan(); _teach3(rp); toy = _stalled3(rp)
        self.assertIsNone(_plan(rp, toy, floor=39)); self.assertEqual(rp.why, 'no_gain')
        self.assertEqual(rp.floor_held, 1); self.assertIsNone(rp._walk); self.assertEqual(rp.walks, 0)
        r = _plan(rp, toy, floor=40)
        self.assertIsNotNone(r); self.assertTrue(r[2].get('probe')); self.assertEqual(rp.walks, 1)

    def test_a_refused_walk_step_is_asked_again(self):
        rp = RelPlan(); _teach3(rp); toy = _stalled3(rp)
        r = _plan(rp, toy); self.assertEqual(r[0], 2)
        # a route overruled it: nothing executed, the board is the same
        r = _plan(rp, toy); self.assertIsNotNone(r); self.assertEqual(r[0], 2)
        self.assertTrue(r[2].get('probe')); self.assertEqual(rp.walks, 1)

    def test_a_stuck_first_step_tries_the_next_candidate(self):
        """25b (review F3): the stall board recurs byte-identically after
        every repair; when the best mask's first arrow is stuck there, the
        next mask is walked instead of exploring ending for the life."""
        rp = RelPlan()
        S = frozenset([('p', 0, 0)])
        near = frozenset([('y', 0, 0)]); far = frozenset([('z', 0, 0)])
        rp.effects[(4, None, frozenset([('p', 3, 3, 5, 5)]))] = [set([1, 2]), 5]
        rp.mutable.update(('p', 'y', 'z')); rp.change.update({'p': 10, 'y': 10, 'z': 10}); rp.seen.update({'p': 10, 'y': 10, 'z': 10})
        eff, auto = rp.views()
        dist0 = {S: (0, None), near: (1, 3), far: (2, 1)}
        canvas = np.zeros((10, 10), int)
        rp._stuck[(S, canvas.tobytes(), None, 3, None)] = STUCK_AFTER
        r = rp._probe(S, dist0, eff, 6, canvas, None, 50, True)
        self.assertIsNotNone(r, 'exploring ended on one stuck first step')
        self.assertEqual(r[0], 1); self.assertEqual(r[2]['dist'], 2)
        self.assertEqual(rp._walk['X'], far); self.assertEqual(rp.walks, 1)
        self.assertEqual(rp.events[-1][:5], 'latch')

    def test_an_unreachable_drop_keeps_the_target_eligible(self):
        """26b (review of 26, finding 7): a route moved him somewhere the
        target cannot be reached from -- the latch is dropped but the
        target is NOT failed for the life; it is re-latched on return."""
        rp = RelPlan(); _teach3(rp); toy = _stalled3(rp)
        r = _plan(rp, toy); self.assertTrue(r[2].get('probe')); X = rp._walk['X']
        with mock.patch.object(rp, '_dist', lambda auto, S0, n, navpaint=True: {S0: (0, None)}):
            self.assertIsNone(_plan(rp, toy)); self.assertEqual(rp.why, 'no_gain')
        self.assertIsNone(rp._walk); self.assertEqual(rp.walks_dropped_why, {'unreachable': 1})
        self.assertEqual(len(rp._walk_failed), 0)
        r = _plan(rp, toy)
        self.assertIsNotNone(r); self.assertTrue(r[2].get('probe')); self.assertEqual(rp._walk['X'], X)
        self.assertEqual(rp.walks, 2)

    def test_a_walk_over_budget_is_dropped_for_the_life(self):
        rp = RelPlan(); _teach3(rp); toy = _stalled3(rp)
        r = _plan(rp, toy); self.assertTrue(r[2].get('probe'))
        X = rp._walk['X']
        rp._walk['n'] = MAX_WALK_STEPS
        self.assertIsNone(_plan(rp, toy)); self.assertEqual(rp.why, 'no_gain')
        self.assertEqual(rp.walks_dropped, 1); self.assertIn(X, rp._walk_failed)
        self.assertEqual(rp.walks_dropped_why, {'budget': 1})
        self.assertIsNone(rp._walk); self.assertEqual(rp.walks, 1)
        rp.begin_life()
        self.assertEqual(len(rp._walk_failed), 0)


# ---------------------------------------------------------------- patch 26
SW4 = {15: (3, 40), 12: (3, 46), 9: (3, 52)}   # three swatches


class Toy4(Toy3):
    """Toy3 with a third swatch (9).  `null2`: the apply at mask 2 paints
    NOTHING (a null mask) but still advances the indicator."""
    def __init__(self, null2=False):
        Toy3.__init__(self); self.null2 = null2

    def board(self):
        G = np.full((W, W), BG, int)
        for r in range(10):
            for c in range(10):
                G[B[0] + r, B[1] + c] = 15 if r < 6 else 12
        G[A[0]:A[2] + 1, A[1]:A[3] + 1] = self.canvas
        for col, (r, c) in SW4.items():
            G[r - 1:r + 4, c - 1:c + 4] = 4
            G[r:r + 3, c:c + 3] = col
        G[20:24, 40:40 + WIDTH3[self.shape]] = self.colour
        G[63, 0:max(1, 60 - self.t)] = 4
        return G

    def step(self, a, aim=None):
        self.t += 1
        if a == 2:
            self.shape = NEXT_ARROW3[self.shape]
        elif a == 4:
            if not (self.null2 and self.shape == 2):
                for r, c in MASKS3[self.shape]:
                    self.canvas[r, c] = self.colour
            self.shape = (self.shape + 1) % 3
        elif a == CLICK and aim is not None:
            for col, (r, c) in SW4.items():
                if r - 1 <= aim[0] < r + 4 and c - 1 <= aim[1] < c + 4:
                    self.colour = col


def _teach4(rp, null2=False):
    toy = Toy4(null2)
    run(toy, rp, [(CLICK, (2, 47)), (CLICK, (2, 41)),
                  (4, None), (2, None), (2, None), (CLICK, (2, 47)), (4, None), (2, None),
                  (CLICK, (2, 53))])                        # swatch 9 learned
    return toy


def _stalled4(rp, null2=False):
    """Rows 0-5 right, rows 6-9 painted 9 and 9 SELECTED: the untried mask
    (rows 6-9) would paint 9 over 9 -- nothing -- unless the probe changes
    colour first."""
    toy = Toy4(null2)
    toy.canvas[0:6, :] = 15; toy.canvas[6:10, :] = 9
    rp.begin_life()
    _step(rp, toy, CLICK, (2, 53))                          # select 9
    assert toy.mm() == 40
    return toy


def _walk_to_mask2(rp, toy):
    r = _plan(rp, toy); assert r[0] == 2 and r[2].get('probe'), (r, rp.why)
    _step(rp, toy, 2)
    r = _plan(rp, toy); assert r[0] == 4 and r[2].get('probe'), (r, rp.why)
    _step(rp, toy, 4)                                        # row 5 -> 9 (spoil), mask 2
    assert toy.shape == 2 and toy.mm() == 50


class TestProbeHue(_Gated):
    def setUp(self):
        _Gated.setUp(self)
        self._hp = mock.patch.object(_rp, '_PROBEHUE_ON', lambda: True); self._hp.start()

    def tearDown(self):
        self._hp.stop(); _Gated.tearDown(self)

    def test_gate_off_probes_with_the_selection_and_learns_nothing(self):
        rp = RelPlan(); _teach4(rp); toy = _stalled4(rp); _walk_to_mask2(rp, toy)
        with mock.patch.object(_rp, '_PROBEHUE_ON', lambda: False):
            r = _plan(rp, toy)
            self.assertEqual(r[0], 4); self.assertTrue(r[2].get('final'))
            n_eff = len(rp.effects)
            _step(rp, toy, 4)                                # 9 over 9: nothing
            self.assertEqual(toy.mm(), 50); self.assertEqual(len(rp.effects), n_eff)
            self.assertEqual(rp.probe_exec, 1); self.assertEqual(len(rp.null_masks), 0)

    def test_hue_click_then_apply_reveals_the_mask(self):
        rp = RelPlan(); _teach4(rp); toy = _stalled4(rp); _walk_to_mask2(rp, toy)
        # the final step is first a CLICK on a swatch of an absent colour --
        # 12, which the target needs -- not the apply
        r = _plan(rp, toy)
        self.assertEqual(r[0], CLICK, (r, rp.why)); self.assertTrue(r[2].get('probe'))
        self.assertFalse(r[2].get('final')); self.assertEqual(r[2].get('hue'), 12)
        self.assertEqual(rp.probe_hue_clicks, 1); self.assertEqual(rp.probe_applies, 0)
        _step(rp, toy, CLICK, r[1])
        self.assertEqual(toy.colour, 12); self.assertIsNotNone(rp._walk)
        r = _plan(rp, toy)
        self.assertEqual(r[0], 4, (r, rp.why)); self.assertTrue(r[2].get('final'))
        self.assertIsNone(r[2].get('hue')); self.assertEqual(rp.probe_hue_clicks, 1)
        n_eff = len(rp.effects)
        _step(rp, toy, 4)                                    # rows 6-9 -> 12: revealed
        self.assertEqual(toy.mm(), 10); self.assertGreater(len(rp.effects), n_eff)
        self.assertEqual(rp.probe_exec, 1); self.assertEqual(len(rp.null_masks), 0)
        acts = []
        for _ in range(8):
            r = _plan(rp, toy)
            if r is None:
                break
            acts.append(r[0]); _step(rp, toy, r[0], r[1])
            if toy.mm() == 0:
                break
        self.assertEqual(toy.mm(), 0, acts)

    def test_no_click_when_the_selection_is_already_absent(self):
        rp = RelPlan(); _teach4(rp); toy = _stalled4(rp); _walk_to_mask2(rp, toy)
        _step(rp, toy, CLICK, (2, 47))                       # he selects 12 himself
        r = _plan(rp, toy)
        self.assertEqual(r[0], 4); self.assertTrue(r[2].get('final')); self.assertEqual(rp.probe_hue_clicks, 0)

    def test_a_null_mask_is_remembered_after_two_absent_probes(self):
        rp = RelPlan(); _teach4(rp, null2=True)
        for life in (1, 2):
            toy = _stalled4(rp, null2=True); _walk_to_mask2(rp, toy)
            r = _plan(rp, toy); self.assertEqual(r[0], CLICK); _step(rp, toy, CLICK, r[1])
            r = _plan(rp, toy); self.assertEqual(r[0], 4); self.assertTrue(r[2].get('final'))
            X = rp._walk['X']
            _step(rp, toy, 4)                                # absent colour, nothing painted
            self.assertEqual(toy.mm(), 50)
            self.assertEqual(rp.null_masks.get(X), life); self.assertIsNone(rp._walk)
            self.assertEqual(rp.events[-1][:9], 'null_mask')
        # third life: the null mask counts as known -- nothing left to walk to
        toy = _stalled4(rp, null2=True)
        w0 = rp.walks
        self.assertIsNone(_plan(rp, toy)); self.assertEqual(rp.why, 'no_gain'); self.assertEqual(rp.walks, w0)
        # persisted, and idempotent
        rp2 = RelPlan(); rp2.from_dict(rp.to_dict()); rp2.from_dict(rp.to_dict())
        self.assertEqual(rp2.null_masks, rp.null_masks)
        self.assertEqual(rp2._null_known(), {X})

    def test_hue_click_gives_up_after_tries(self):
        rp = RelPlan(); _teach4(rp); toy = _stalled4(rp); _walk_to_mask2(rp, toy)
        for i in range(STUCK_AFTER):
            r = _plan(rp, toy); self.assertEqual(r[0], CLICK, i)   # never executed (overruled)
        r = _plan(rp, toy)
        self.assertEqual(r[0], 4); self.assertTrue(r[2].get('final'))
        self.assertEqual(rp.probe_hue_clicks, STUCK_AFTER)


# ---------------------------------------------------------------- patch 27
class Toy5(Toy4):
    """Toy4 whose indicator changes SHAPE with its colour: in colour 12 it
    loses its top-left cell (cd82: the recoloured stamp icon segments
    differently -- 30/76 L3 clicks changed the q-state, canvas unchanged)."""
    def board(self):
        G = Toy4.board(self)
        if self.colour == 12:
            G[20, 40] = BG
        return G


def _teach5(rp, null2=False):
    toy = Toy5(null2)
    run(toy, rp, [(CLICK, (2, 47)), (CLICK, (2, 41)),
                  (4, None), (2, None), (2, None), (CLICK, (2, 47)), (4, None), (2, None),
                  (CLICK, (2, 53))])
    return toy


def _stalled5(rp, null2=False):
    toy = Toy5(null2)
    toy.canvas[0:6, :] = 15; toy.canvas[6:10, :] = 9
    rp.begin_life()
    _step(rp, toy, CLICK, (2, 53))
    assert toy.mm() == 40
    return toy


def _q_now(rp, toy):
    G = toy.board(); outs = rp.outside(G, segment(G, 63)[1], A, B)
    return rp.q(rp.state(outs)[0])


class TestSameMask(_Gated):
    def setUp(self):
        _Gated.setUp(self)
        self._ps2 = [mock.patch.object(_rp, '_PROBEHUE_ON', lambda: True),
                     mock.patch.object(_rp, '_SAMEMASK_ON', lambda: True)]
        for p in self._ps2:
            p.start()

    def tearDown(self):
        for p in self._ps2:
            p.stop()
        _Gated.tearDown(self)

    def test_a_swatch_click_that_changes_the_state_records_a_pair(self):
        rp = RelPlan(); _teach5(rp); toy = _stalled5(rp)      # colour 9, mask 0
        q0 = _q_now(rp, toy); n0 = len(rp.same_mask)
        _step(rp, toy, CLICK, (2, 47))                          # swatch 12: the icon notches
        q1 = _q_now(rp, toy)
        self.assertNotEqual(q0, q1); self.assertEqual(len(rp.same_mask), n0 + 1)
        self.assertIn(frozenset([q0, q1]), rp.same_mask); self.assertEqual(rp.events[-1][:9], 'same_mask')
        _step(rp, toy, CLICK, (2, 41))                          # swatch 15: back to the plain icon
        self.assertEqual(len(rp.same_mask), n0 + 1)             # the same pair, once
        _step(rp, toy, CLICK, (30, 30))                         # background: no pair
        self.assertEqual(len(rp.same_mask), n0 + 1)
        self.assertEqual(rp._twins(q0), set([q0, q1]))
        # 27b: a pair is NOT an equivalence -- a twin of a known state is not known
        self.assertEqual(rp._known_closure(set([q1])), set([q1]))

    def test_arrival_via_a_twin_serves_the_final_apply(self):
        rp = RelPlan(); _teach5(rp); toy = _stalled5(rp)
        X = _q_now(rp, toy)
        _step(rp, toy, CLICK, (2, 47))                          # now at the twin, colour 12
        S = _q_now(rp, toy); self.assertIn(frozenset([X, S]), rp.same_mask)
        rp._walk = {"X": X, "act": 4, "d": 1, "n": 0}
        G = toy.board(); objs = segment(G, 63)[1]; outs = rp.outside(G, objs, A, B)
        eff, auto = rp.views(); canvas = G[A[0]:A[2] + 1, A[1]:A[3] + 1]; target = G[B[0]:B[2] + 1, B[1]:B[3] + 1]
        # (X = mask 0 is known: the closure clears the latch instead) -> use an unknown twin pair
        Xu = frozenset([('u', 0, 0)]); Su = frozenset([('u12', 0, 0)])
        # 27b: only the twin THIS walk's own hue click produced counts as arrival
        rp._walk = {"X": Xu, "act": 4, "d": 1, "n": 0, "twin": Su}
        r = rp._serve_walk(Su, eff, auto, 6, canvas, 12, 40, outs=outs, target=target)
        self.assertIsNotNone(r); self.assertEqual(r[0], 4); self.assertTrue(r[2].get('final'))
        # a HISTORICAL pair alone is not trusted (73/78 pairs join different masks)
        rp.same_mask.add(frozenset([Xu, Su])); rp._walk = {"X": Xu, "act": 4, "d": 1, "n": 0}
        self.assertIsNone(rp._serve_walk(Su, eff, auto, 6, canvas, 12, 40, outs=outs, target=target))
        self.assertEqual(rp.walks_dropped_why.get('unreachable', 0), 1)

    def test_the_walk_pools_a_twins_evidence(self):
        rp = RelPlan()
        S1 = frozenset([('a', 0, 0)]); S1n = frozenset([('a12', 0, 0)]); S2 = frozenset([('b', 0, 0)])
        auto = {(S1n, 4): {S2: 1}}
        self.assertIsNone(rp._next(auto, S1, 4))
        rp.same_mask.add(frozenset([S1, S1n]))
        self.assertIsNone(rp._next(auto, S1, 4))                # the plan's own view: untouched
        self.assertEqual(rp._next(auto, S1, 4, tw=True), S2)    # the walk's view
        self.assertEqual(rp._dist(auto, S1, 6, navpaint=False).get(S2), (1, 4))

    def test_the_plans_reach_ignores_pairs(self):
        """27b: the rounds never treat a twin as reached (L0: 9/26 and L2:
        12/22 segments changed picks when they did)."""
        rp = RelPlan()
        S = frozenset([('p', 0, 0)]); S2 = frozenset([('p12', 0, 0)])
        rp.same_mask.add(frozenset([S, S2]))
        self.assertIsNone(rp._reach({S: (0, None)}, S2))

    def test_the_hue_click_sets_the_walks_twin(self):
        rp = RelPlan(); _teach5(rp); toy = _stalled5(rp)
        r = _plan(rp, toy); self.assertTrue(r[2].get('probe')); X = rp._walk['X']
        _step(rp, toy, r[0], r[1])
        r = _plan(rp, toy)
        if r[0] == CLICK:                                   # the hue click at the target
            self.assertIsNone(rp._walk.get('twin'))
            _step(rp, toy, CLICK, r[1])
            self.assertEqual(rp._walk.get('twin'), _q_now(rp, toy))
            self.assertNotEqual(rp._walk['twin'], X)

    def test_same_mask_persists(self):
        rp = RelPlan(); _teach5(rp); toy = _stalled5(rp)
        _step(rp, toy, CLICK, (2, 47)); self.assertEqual(len(rp.same_mask), 1)
        rp2 = RelPlan(); rp2.from_dict(rp.to_dict()); rp2.from_dict(rp.to_dict())
        self.assertEqual(rp2.same_mask, rp.same_mask)

    def test_end_to_end_the_probe_learns_the_mask_behind_a_colour(self):
        """From the stall the walk may target the known mask's other-colour
        state first; after the hue click the pair is learned, the closure
        clears that latch, the walk goes on to the true untried mask and
        the probe paints it.  No latch dies on budget."""
        rp = RelPlan(); _teach5(rp); toy = _stalled5(rp)
        n_eff = len(rp.effects); mm_min = toy.mm(); trace = []
        for _ in range(14):
            r = _plan(rp, toy)
            if r is None:
                trace.append(rp.why); break
            trace.append((r[0], r[2].get('hue'), r[2].get('final')))
            _step(rp, toy, r[0], r[1]); mm_min = min(mm_min, toy.mm())
            if rp.probe_exec >= 1 and toy.shape == 0:
                break
        self.assertGreaterEqual(rp.probe_exec, 1, trace)
        self.assertGreater(len(rp.effects), n_eff, trace)
        self.assertGreaterEqual(len(rp.same_mask), 1, trace)
        self.assertEqual(rp.walks_dropped_why.get('budget', 0), 0, trace)
        self.assertLessEqual(mm_min, 10, trace)


# ---------------------------------------------------------------- patch 28
MASKS6 = {0: [(r, c) for r in range(10) for c in range(10)],        # the whole canvas
          1: [(r, c) for r in range(5) for c in range(10)],         # rows 0-4
          2: [(r, c) for r in range(8, 10) for c in range(10)]}     # rows 8-9
WIDTH6 = {0: 6, 1: 9, 2: 12}
T6 = np.array([[15 if r < 5 else (12 if r < 8 else 9) for c in range(10)] for r in range(10)])


class Toy6(Toy4):
    """Three masks cycled by arrow 2 (0->1->2->0); APPLY keeps the mask.
    Target rows 0-4 = 15, rows 5-7 = 12, rows 8-9 = 9.  From a canvas
    with rows 0-4 right and rows 5-9 = 9 (mm 30) no single known paint
    gains; painting EVERYTHING 12 (mm 70), then rows 0-4 15 (20), then
    rows 8-9 9 (0) clears -- the cd82 L3 shape of the problem."""
    def board(self):
        G = np.full((W, W), BG, int)
        G[B[0]:B[2] + 1, B[1]:B[3] + 1] = T6
        G[A[0]:A[2] + 1, A[1]:A[3] + 1] = self.canvas
        for col, (r, c) in SW4.items():
            G[r - 1:r + 4, c - 1:c + 4] = 4
            G[r:r + 3, c:c + 3] = col
        G[20:24, 40:40 + WIDTH6[self.shape]] = self.colour
        G[63, 0:max(1, 60 - self.t)] = 4
        return G

    def step(self, a, aim=None):
        self.t += 1
        if a == 2:
            self.shape = (self.shape + 1) % 3
        elif a == 4:
            for r, c in MASKS6[self.shape]:
                self.canvas[r, c] = self.colour
        elif a == CLICK and aim is not None:
            for col, (r, c) in SW4.items():
                if r - 1 <= aim[0] < r + 4 and c - 1 <= aim[1] < c + 4:
                    self.colour = col

    def mm(self):
        return int((self.canvas != T6).sum())


def _teach6(rp):
    toy = Toy6()
    run(toy, rp, [(CLICK, (2, 47)), (CLICK, (2, 41)),          # hues 12, 15
                  (4, None),                                    # mask 0 @15
                  (2, None), (CLICK, (2, 47)), (4, None),       # mask 1 @12
                  (2, None), (CLICK, (2, 53)), (4, None),       # mask 2 @9
                  (2, None)])                                   # back to 0
    return toy


def _stalled6(rp):
    toy = Toy6()
    toy.canvas[0:5, :] = 15; toy.canvas[5:10, :] = 9
    rp.begin_life()
    _step(rp, toy, CLICK, (2, 53))                              # select 9
    assert toy.mm() == 30
    return toy


class TestLookahead(_Gated):
    def setUp(self):
        _Gated.setUp(self)
        self._lp = mock.patch.object(_rp, '_LOOKAHEAD_ON', lambda: True); self._lp.start()

    def tearDown(self):
        self._lp.stop(); _Gated.tearDown(self)

    def test_gate_off_is_silent(self):
        rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
        with mock.patch.object(_rp, '_LOOKAHEAD_ON', lambda: False):
            self.assertIsNone(_plan(rp, toy)); self.assertEqual(rp.why, 'no_gain')
        self.assertEqual(rp.lookaheads, 0)

    def test_three_step_sequence_found_first_step_spoils_unjudged(self):
        rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
        r = _plan(rp, toy)
        self.assertIsNotNone(r, rp.why); self.assertEqual(r[2].get('lookahead'), 3)
        self.assertEqual(r[2]['mm0'], 30); self.assertEqual(r[2]['mm1'], 0)
        self.assertEqual(rp.lookahead_hits, 1); self.assertEqual(rp.lookaheads, 1)
        self.assertEqual(r[0], CLICK)                             # colour 12 first
        self.assertTrue(any(e.startswith('lookahead depth=3') for e in rp.events[-2:]), rp.events)
        _step(rp, toy, CLICK, r[1]); self.assertEqual(toy.colour, 12)
        r = _plan(rp, toy)
        self.assertEqual(r[0], 4); self.assertEqual(r[2].get('lookahead'), 3)
        self.assertIsNone(rp._pending['op'])                      # the spoil is unjudged
        judged0 = rp.plan_judged
        _step(rp, toy, 4)
        self.assertEqual(toy.mm(), 70)
        self.assertEqual(rp.plan_judged, judged0 + 1)             # but still scored
        self.assertEqual(len(rp._burned), 0); self.assertEqual(len(rp.spoiled), 0)
        # from here the greedy rounds finish: rows 0-4 15, rows 8-9 9
        acts = []
        for _ in range(10):
            r = _plan(rp, toy)
            if r is None:
                break
            acts.append((r[0], r[2].get('lookahead')))
            _step(rp, toy, r[0], r[1])
            if toy.mm() == 0:
                break
        self.assertEqual(toy.mm(), 0, acts)

    def test_floor_gate_holds_the_lookahead(self):
        rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
        self.assertIsNone(_plan(rp, toy, floor=29)); self.assertEqual(rp.lookaheads, 0)
        self.assertIsNotNone(_plan(rp, toy, floor=30)); self.assertEqual(rp.lookaheads, 1)

    def test_cache_is_per_board(self):
        rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
        _plan(rp, toy); _plan(rp, toy)
        self.assertEqual(rp.lookaheads, 1)                        # the same board: cached

    def test_the_sequence_is_latched_and_served_ahead_of_the_greedy(self):
        """28c (review of 28, K2): after the spoiling paint the greedy would
        take a repair; the latch serves step 2, then step 3, to the end."""
        rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
        r = _plan(rp, toy); self.assertIsNotNone(rp._seq); self.assertEqual(rp._seq['i'], 0)
        self.assertEqual(rp.events[-1][:9], 'seq_latch')
        _step(rp, toy, r[0], r[1])                                # click 12
        r = _plan(rp, toy); self.assertEqual(r[0], 4); self.assertEqual(rp._pending['seq'], 0)
        _step(rp, toy, 4)                                         # step 1: spoils to 70
        self.assertEqual(rp._seq['i'], 1); self.assertEqual(toy.mm(), 70)
        # the latch serves the REMAINING two steps (the greedy loop is skipped)
        r = _plan(rp, toy)
        self.assertEqual(r[2].get('lookahead'), 2); self.assertEqual(r[2]['mm1'], 0)
        acts = []
        for _ in range(10):
            r = _plan(rp, toy)
            if r is None:
                break
            acts.append(r[0]); _step(rp, toy, r[0], r[1])
            if toy.mm() == 0:
                break
        self.assertEqual(toy.mm(), 0, acts)
        self.assertIsNone(rp._seq); self.assertEqual(rp.lookahead_done, 1); self.assertEqual(rp.lookahead_dropped, 0)

    def test_a_diverged_board_drops_the_latch(self):
        rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
        r = _plan(rp, toy); self.assertIsNotNone(rp._seq)
        toy.canvas[0, 0] = 9                                      # someone else painted
        _plan(rp, toy)
        self.assertIsNone(rp._seq); self.assertEqual(rp.lookahead_dropped, 1)
        self.assertTrue(any(e.startswith('seq_drop why=diverged') for e in rp.events))

    def test_a_mispredicted_step_drops_the_latch(self):
        rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
        r = _plan(rp, toy); _step(rp, toy, r[0], r[1])            # click 12
        r = _plan(rp, toy); self.assertEqual(rp._pending['seq'], 0)
        toy.shape = 1                                             # the apply lands on another mask
        _step(rp, toy, 4)
        self.assertIsNone(rp._seq); self.assertEqual(rp.lookahead_dropped, 1)
        self.assertTrue(any(e.startswith('seq_drop why=misprediction') for e in rp.events))



class Toy7(Toy6):
    """Toy6 whose indicator LOSES ITS TOP-LEFT CELL in colour 15 (Toy5's trick;
    cd82: the recoloured stamp icon segments differently -- on L4 20/41 of
    the plan's own swatch clicks changed the q-state, canvas unchanged).  The
    colour-15 shapes are other states: worn once in the lesson (one arrow
    between them), never applied from, never reached by an arrow from the
    plain shapes."""
    def board(self):
        G = Toy6.board(self)
        if self.colour == 15:
            G[20, 40] = BG
        return G


def _teach7(rp):
    """Every mask learned with the icon in its plain shape (12 or 9)."""
    toy = Toy7()
    run(toy, rp, [(CLICK, (2, 47)), (CLICK, (2, 41)), (2, None), (CLICK, (2, 47)),   # 12; 15 (notched 0 -> notched 1); 12
                  (2, None), (2, None),                                              # plain 2, plain 0
                  (4, None),                                                         # mask 0 @12 (all)
                  (2, None), (CLICK, (2, 53)), (4, None),                            # mask 1 @9 (rows 0-4)
                  (2, None), (4, None),                                              # mask 2 @9 (rows 8-9: 12 -> 9)
                  (2, None)])                                                        # back to 0
    return toy


def _stalled7(rp):
    toy = Toy7()
    toy.canvas[0:5, :] = 15; toy.canvas[5:10, :] = 9
    toy.colour = 12
    rp.begin_life()
    _step(rp, toy, CLICK, (2, 53))                              # select 9
    assert toy.mm() == 30
    return toy


def _drive(rp, toy, n=14):
    """plan() and execute until mm 0, silence or n steps; the picks."""
    acts = []
    for _ in range(n):
        r = _plan(rp, toy)
        if r is None:
            acts.append(('silent', rp.why)); break
        acts.append((r[0], r[1])); _step(rp, toy, r[0], r[1])
        if toy.mm() == 0:
            break
    return acts


class TestSeqHue(_Gated):
    """PATCH 29: a latched step survives its own colour click."""
    def setUp(self):
        _Gated.setUp(self)
        self._ps2 = [mock.patch.object(_rp, '_LOOKAHEAD_ON', lambda: True),
                     mock.patch.object(_rp, '_SAMEMASK_ON', lambda: True),
                     mock.patch.object(_rp, '_SEQHUE_ON', lambda: True)]
        for p in self._ps2:
            p.start()

    def tearDown(self):
        for p in self._ps2:
            p.stop()
        _Gated.tearDown(self)

    def test_gate_off_the_colour_click_first_loses_the_mask(self):
        """28c on Toy7 = the live L4 drop: step 2 clicks 15 at mask 0, the
        icon shifts, arrows from the new state reach nothing."""
        with mock.patch.object(_rp, '_SEQHUE_ON', lambda: False):
            rp = RelPlan(); _teach7(rp); toy = _stalled7(rp)
            r = _plan(rp, toy); self.assertEqual(r[2].get('lookahead'), 3, rp.why)
            _step(rp, toy, r[0], r[1])                            # click 12
            r = _plan(rp, toy); self.assertEqual(r[0], 4); _step(rp, toy, 4)
            self.assertEqual(toy.mm(), 70); self.assertEqual(rp._seq['i'], 1)
            r = _plan(rp, toy)
            self.assertEqual(r[0], CLICK)                             # colour 15 FIRST, at mask 0
            _step(rp, toy, r[0], r[1]); self.assertEqual(toy.colour, 15)
            _plan(rp, toy)
            self.assertIsNone(rp._seq); self.assertEqual(rp.lookahead_dropped, 1)
            self.assertTrue(any(e.startswith('seq_drop why=unreachable i=1') for e in rp.events), rp.events)

    def test_navigate_first_colour_at_the_mask_twin_then_paint(self):
        rp = RelPlan(); _teach7(rp); toy = _stalled7(rp)
        r = _plan(rp, toy); self.assertEqual(r[2].get('lookahead'), 3, rp.why)
        self.assertEqual(r[0], CLICK)                                 # step 1 is AT its mask: colour 12
        self.assertEqual(rp._pending['seq_hue'], 0)
        _step(rp, toy, r[0], r[1])
        self.assertEqual(rp._seq.get('twin_i'), 0); self.assertEqual(rp.seq_twins, 0)   # 12 keeps the shape
        r = _plan(rp, toy); self.assertEqual(r[0], 4); _step(rp, toy, 4)
        self.assertEqual(toy.mm(), 70); self.assertEqual(rp._seq['i'], 1)
        r = _plan(rp, toy)
        self.assertEqual(r[0], 2, (r, rp.why))                        # the ARROW first, not colour 15
        self.assertIsNone(rp._pending.get('seq_hue'))
        _step(rp, toy, 2); self.assertEqual(toy.shape, 1)
        r = _plan(rp, toy)
        self.assertEqual(r[0], CLICK); self.assertEqual(rp._pending['seq_hue'], 1)
        self.assertTrue(any(e.startswith('seq_hue i=1 c=15') for e in rp.events), rp.events)
        _step(rp, toy, r[0], r[1]); self.assertEqual(toy.colour, 15)  # the icon loses a cell
        self.assertEqual(rp.seq_twins, 1); self.assertEqual(rp._seq.get('twin_i'), 1)
        self.assertTrue(any(e.startswith('seq_twin i=1') for e in rp.events), rp.events)
        self.assertTrue(any(e.startswith('same_mask') for e in rp.events), rp.events)
        r = _plan(rp, toy)
        self.assertEqual(r[0], 4, (r, rp.why))                        # at the twin: the paint
        self.assertEqual(rp._pending['seq'], 1)
        _step(rp, toy, 4)
        self.assertEqual(toy.mm(), 20); self.assertEqual(rp._seq['i'], 2)
        self.assertEqual(rp.lookahead_dropped, 0)

    def test_from_a_variant_the_twins_arrows_serve_then_one_colour_click_then_the_end(self):
        rp = RelPlan(); _teach7(rp); toy = _stalled7(rp)
        acts = _drive(rp, toy)
        self.assertEqual(toy.mm(), 0, acts)
        self.assertEqual(rp.lookahead_done, 1); self.assertEqual(rp.lookahead_dropped, 0)
        self.assertIsNone(rp._seq)
        # picks: click 12, paint, arrow, click 15, paint, arrow (pooled from the
        # twin), colour 9 first (no path from the notched mask 2), paint
        self.assertEqual([a for a, _ in acts], [CLICK, 4, 2, CLICK, 4, 2, CLICK, 4], acts)
        self.assertEqual(rp.seq_hues, 2); self.assertEqual(rp.seq_twins, 1); self.assertEqual(rp.seq_prehues, 1)

    def test_a_second_colour_click_is_not_tried_the_latch_drops(self):
        rp = RelPlan(); _teach7(rp); toy = _stalled7(rp)
        acts = _drive(rp, toy, n=6)                                   # ... up to the arrow into notched mask 2
        self.assertEqual([a for a, _ in acts], [CLICK, 4, 2, CLICK, 4, 2], acts)
        r = _plan(rp, toy); self.assertEqual(r[0], CLICK)             # the one colour click
        self.assertEqual(rp._pending['seq_prehue'], 2)
        # ... but the click changes nothing on the board: the icon stays notched
        Gp = toy.board(); op = segment(Gp, 63)[1]
        rp.observe(Gp, op, Gp, op, CLICK, True, r[1], A, B)
        _plan(rp, toy)
        self.assertIsNone(rp._seq); self.assertEqual(rp.lookahead_dropped, 1)
        self.assertTrue(any(e.startswith('seq_drop why=unreachable i=2') for e in rp.events), rp.events)

    def test_toy6_still_completes_with_the_gate(self):
        rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
        acts = _drive(rp, toy)
        self.assertEqual(toy.mm(), 0, acts)
        self.assertEqual(rp.lookahead_done, 1); self.assertEqual(rp.lookahead_dropped, 0)


class TestRefute(_Gated):
    """PATCH 30: a refuted latched step is burned for the life; the lookahead
    is not gated off by its own record."""
    def setUp(self):
        _Gated.setUp(self)
        self._ps2 = [mock.patch.object(_rp, '_LOOKAHEAD_ON', lambda: True),
                     mock.patch.object(_rp, '_REFUTE_ON', lambda: True),
                     mock.patch.object(_rp, '_LAFLOOR_ON', lambda: True)]
        for p in self._ps2:
            p.start()

    def tearDown(self):
        for p in self._ps2:
            p.stop()
        _Gated.tearDown(self)

    def _mispredict(self, rp, toy):
        """Latch Toy6's sequence, click 12, and let the first apply land on
        another mask (the 28c misprediction test); the latched step's op."""
        r = _plan(rp, toy); self.assertIsNotNone(rp._seq)
        st = rp._seq['steps'][0]; op = (st[0], st[1], st[2])
        _step(rp, toy, r[0], r[1])                                # click 12
        r = _plan(rp, toy); self.assertEqual(rp._pending['seq'], 0)
        toy.shape = 1
        _step(rp, toy, 4)
        self.assertIsNone(rp._seq); self.assertEqual(rp.lookahead_dropped, 1)
        return op

    def test_gate_off_a_mispredicted_step_is_not_burned(self):
        with mock.patch.object(_rp, '_REFUTE_ON', lambda: False):
            rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
            op = self._mispredict(rp, toy)
            self.assertNotIn(op, rp._burned); self.assertEqual(rp.seq_refuted, 0)
            self.assertFalse(any(e.startswith('seq_refute') for e in rp.events), rp.events)

    def test_a_mispredicted_step_is_burned_for_the_life(self):
        rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
        op = self._mispredict(rp, toy)
        self.assertIn(op, rp._burned); self.assertEqual(rp.seq_refuted, 1)
        self.assertTrue(any(e.startswith('seq_refute i=0 why=misprediction') for e in rp.events), rp.events)
        # the burned effect is out of the plan's candidates: whatever the next
        # plan does, it does not latch a sequence starting with that op
        r = _plan(rp, toy)
        if rp._seq is not None:
            self.assertNotEqual(tuple(rp._seq['steps'][0][:3]), op)
        # a new life forgives
        rp.begin_life(); self.assertEqual(len(rp._burned), 0)

    def test_a_stuck_drop_burns_the_step(self):
        rp = RelPlan(); rp.begin_life()
        S0 = frozenset([('x', 1, 2)])
        rp._seq = {'steps': [(4, None, S0, 12), (4, None, S0, 15)], 'i': 1, 'canvases': [None, None, None]}
        rp._drop_seq('stuck')
        self.assertIn((4, None, S0), rp._burned); self.assertEqual(rp.seq_refuted, 1)
        self.assertIsNone(rp._seq)
        self.assertTrue(any(e.startswith('seq_refute i=1 why=stuck') for e in rp.events), rp.events)

    def test_other_drops_burn_nothing(self):
        rp = RelPlan(); rp.begin_life()
        S0 = frozenset([('x', 1, 2)])
        for why in ('unreachable', 'effect_gone', 'diverged', 'budget'):
            rp._seq = {'steps': [(4, None, S0, 12)], 'i': 0, 'canvases': [None, None]}
            rp._drop_seq(why)
        self.assertEqual(len(rp._burned), 0); self.assertEqual(rp.seq_refuted, 0)

    def test_a_burn_re_runs_the_cached_lookahead(self):
        rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
        _plan(rp, toy); rp._seq = None; _plan(rp, toy)
        self.assertEqual(rp.lookaheads, 1)                        # the same board: cached
        rp._seq = None; rp._burned.add(('never', None, frozenset()))
        _plan(rp, toy)
        self.assertEqual(rp.lookaheads, 2)                        # the candidates changed: searched again
        with mock.patch.object(_rp, '_REFUTE_ON', lambda: False):
            rp2 = RelPlan(); _teach6(rp2); toy2 = _stalled6(rp2)
            _plan(rp2, toy2); rp2._seq = None; rp2._burned.add(('never', None, frozenset()))
            _plan(rp2, toy2)
            self.assertEqual(rp2.lookaheads, 1)                   # gate off: the stale entry is served

    def test_above_the_floor_on_an_unwon_level_the_lookahead_runs(self):
        rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
        r = _plan(rp, toy, floor=13)                              # the record a lookahead set; the stall is 30
        self.assertIsNotNone(r, rp.why); self.assertEqual(r[2].get('lookahead'), 3)
        self.assertEqual(rp.lookaheads, 1); self.assertEqual(rp.la_unfloored, 1)

    def test_la_unfloored_counts_lookahead_runs_not_plans(self):
        """Review of 30, finding 3: a plan above the floor that still has a
        greedy round never runs the lookahead and must not count."""
        rp = RelPlan(); _teach_half(rp); toy = Toy(); rp.begin_life()
        r = _plan(rp, toy, floor=10)                              # mm 100: the top-half paint gains
        self.assertEqual(r[0], 4); self.assertIsNone(r[2].get('lookahead'))
        self.assertEqual(rp.lookaheads, 0); self.assertEqual(rp.la_unfloored, 0)

    def test_a_won_level_keeps_the_lookahead_off(self):
        rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
        self.assertIsNone(_plan(rp, toy, floor=0)); self.assertEqual(rp.lookaheads, 0)
        self.assertEqual(rp.la_unfloored, 0)

    def test_gate_off_the_floor_holds_the_lookahead(self):
        with mock.patch.object(_rp, '_LAFLOOR_ON', lambda: False):
            rp = RelPlan(); _teach6(rp); toy = _stalled6(rp)
            self.assertIsNone(_plan(rp, toy, floor=13)); self.assertEqual(rp.lookaheads, 0)
            self.assertEqual(rp.la_unfloored, 0)

    def test_the_probe_keeps_its_floor_gate(self):
        """LAFLOOR frees the lookahead only: with nothing to find, the
        CURIOUS fallback above the floor stays silent."""
        rp = RelPlan(); _teach_half(rp); toy = Toy(); rp.begin_life()
        r = _plan(rp, toy); _step(rp, toy, 4); self.assertEqual(toy.mm(), 50)
        held = rp.floor_held
        self.assertIsNone(_plan(rp, toy, floor=10)); self.assertEqual(rp.why, 'no_gain')
        self.assertEqual(rp.floor_held, held + 1)


if __name__ == '__main__':
    unittest.main()
