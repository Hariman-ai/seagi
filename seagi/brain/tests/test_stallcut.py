"""LEAVING A PATCH THAT YIELDS NO PROGRESS -- WITHOUT A ROTATION STORM.

`depleted` rule (d) has never been armed, and in its original state
arming it went wrong in both directions:

  * `enter()` resets `_won_here`, `_term_here`, `_since_novel`,
    `_novel_here`, `_novel_at_term` and `_steps_here` -- and FORGOT
    `_prog_here`.  A lifetime numerator over a per-visit denominator
    inflates `_ph`, and the rule is silently near-dead.
  * Reset that counter and nothing else, and it storms: `_prog_here` is
    0 at the start of every visit, so `_ph = 0 < _pe` says LEAVE on the
    ABSENCE of evidence.  That is precisely how rule (a) produced the
    rotation storm.

MEASURED over 36 visits in 24 h (median 233 steps): 40.3% contain NO
progress fire at all, and 55.6% are shorter than the 285 steps in which
one is expected at his measured 3.5 per 1,000.

So the rule refuses to judge until the visit has run long enough that a
fire was EXPECTED -- `steps_here * _pe >= 1`, from his own rate, no
constant.  These tests pin BOTH failure directions, because fixing one
of them is what creates the other.
"""
import unittest

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld


class _Stub(object):
    """Everything `depleted` reads, and nothing else."""

    def __init__(self):
        self.game_id = 'vgame'
        self._levels = 0
        self._dep_why = ''
        self._won_here = 0          # YIELDSTAY off
        self._steps_here = 1000
        self._novel_here = 1000     # rule (a): here >= env, so silent
        self._max_gap = 0           # rule (b) silent
        self._since_novel = 0
        self._gap_n = 0.0           # rule (c) silent
        self._gap_mean = 0.0
        self._prog_here = 0

    depleted = ARCWorld.depleted
    enter = ARCWorld.enter

    def visit(self, steps, prog):
        """Set the visit length WITHOUT waking rule (a).

        Rule (a) is checked BEFORE rule (d) and returns immediately, so
        a test that changes `_steps_here` alone drives the novelty rate
        below the environment and measures rule (a) instead.  That is
        what these tests caught on their first run.
        """
        self._steps_here = steps
        self._novel_here = steps        # novelty rate 1.0 == env
        self._prog_here = prog


class _Base(unittest.TestCase):
    STALL = True

    def setUp(self):
        self._s = arc_world._STALLCUT_ON
        self._y = arc_world._YIELDSTAY_ON
        arc_world._STALLCUT_ON = lambda: self.STALL
        arc_world._YIELDSTAY_ON = lambda: False
        self._novel_all = ARCWorld._novel_all
        self._steps_all = ARCWorld._steps_all
        self._prog_all = getattr(ARCWorld, '_prog_all', 0)
        # environment: 1 progress fire per 1,000 steps
        ARCWorld._steps_all = 1000000
        ARCWorld._novel_all = 1000000       # env novelty rate 1.0
        ARCWorld._prog_all = 1000
        self.w = _Stub()

    def tearDown(self):
        arc_world._STALLCUT_ON = self._s
        arc_world._YIELDSTAY_ON = self._y
        ARCWorld._novel_all = self._novel_all
        ARCWorld._steps_all = self._steps_all
        ARCWorld._prog_all = self._prog_all


class TestTheForgottenCounter(_Base):

    def test_enter_resets_prog_here_with_its_siblings(self):
        self.w._prog_here = 40
        self.w._novel_here = 40
        self.w._steps_here = 40
        self.w.enter()
        self.assertEqual(self.w._prog_here, 0,
                         '_prog_here is the fourth "here" counter and must '
                         'reset with the other three, or rule (d) divides a '
                         'LIFETIME numerator by a PER-VISIT denominator')
        self.assertEqual(self.w._novel_here, 0)
        self.assertEqual(self.w._steps_here, 0)


class TestNoStorm(_Base):
    """The direction that fixing the counter would otherwise create."""

    def test_a_fresh_visit_is_not_judged(self):
        # 50 steps in: _pe is 1/1000, so 0.05 fires were expected.
        self.w.visit(50, 0)
        self.assertFalse(self.w.depleted,
                         'leaving on the absence of evidence IS the '
                         'rotation storm')

    def test_a_visit_just_under_the_evidence_bar_is_not_judged(self):
        self.w.visit(999, 0)
        self.assertFalse(self.w.depleted)

    def test_the_median_visit_is_not_judged(self):
        # his measured median visit is 233 steps
        self.w.visit(233, 0)
        self.assertFalse(self.w.depleted,
                         '40.3%% of his visits carry no progress fire at '
                         'all; judging them would fire on 40%% of visits')


class TestItStillWorks(_Base):
    """The rule must not be neutered into uselessness either."""

    def test_a_long_barren_visit_IS_judged(self):
        self.w.visit(5000, 0)   # none arrived
        self.assertTrue(self.w.depleted,
                        'a patch given a fair chance and yielding nothing '
                        'must still be leavable')
        self.assertIn('d:progress', self.w._dep_why)

    def test_a_productive_visit_is_kept(self):
        self.w.visit(5000, 50)   # 10x the environment rate
        self.assertFalse(self.w.depleted)

    def test_exactly_at_the_environment_rate_is_kept(self):
        self.w.visit(5000, 5)   # == env rate
        self.assertFalse(self.w.depleted)


class TestGateOff(_Base):
    STALL = False

    def test_the_rule_does_nothing_at_all(self):
        self.w.visit(5000, 0)
        self.assertFalse(self.w.depleted,
                         'gate absent must be byte-identical')

    def test_enter_still_resets_the_counter(self):
        # the reset is NOT gated -- it makes the counter mean what its
        # siblings mean, whatever rule (d) is doing
        self.w._prog_here = 12
        self.w.enter()
        self.assertEqual(self.w._prog_here, 0)


if __name__ == '__main__':
    unittest.main()
