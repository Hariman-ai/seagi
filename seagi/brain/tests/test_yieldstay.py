"""HE STAYS WHERE HE WINS -- BUT HE IS NEVER CAGED.

The first version of this rule returned False for the WHOLE depleted
predicate.  While his only win was the one in this patch,
`_won_here == _won_all`, so the test reduced to
    1/_steps_here > 1/_steps_all
which is true forever -- a visit's steps are a subset of all steps.  He
would have been caged in the first game he ever won in, unable to escape
because escaping needs a win somewhere else.

These pin the corrected behaviour: the yield hold suppresses ONLY the
novelty MVT rule (a).  The gap rules still eject him, so a cage is
impossible whatever the yield arithmetic says.
"""
import unittest

from seagi.world import arc_world
from seagi.world.arc_world import ARCWorld

DEPLETED = ARCWorld.__dict__['depleted'].fget


class _W(object):
    def __init__(self):
        self._won_here = 0
        self._steps_here = 0
        self._novel_here = 0
        self._prog_here = 0
        self._since_novel = 0
        self._max_gap = 0
        self._gap_n = 0.0
        self._gap_mean = 0.0
        self._dep_why = ''


class TestYieldStay(unittest.TestCase):

    def setUp(self):
        self._g = arc_world._YIELDSTAY_ON
        arc_world._YIELDSTAY_ON = lambda: True
        self._saved = (ARCWorld._won_all, ARCWorld._steps_all,
                       ARCWorld._novel_all)
        self.w = _W()

    def tearDown(self):
        arc_world._YIELDSTAY_ON = self._g
        (ARCWorld._won_all, ARCWorld._steps_all,
         ARCWorld._novel_all) = self._saved

    def _paying_patch(self):
        """This patch out-pays a REAL environment: other patches have won
        too, so the comparison is not vacuous."""
        self.w._won_here = 1
        self.w._steps_here = 50
        self.w._novel_here = 0      # novelty rate 0: rule (a) would eject
        ARCWorld._won_all = 3       # ... and others have paid as well
        ARCWorld._steps_all = 5000
        ARCWorld._novel_all = 1000

    def test_a_paying_patch_survives_the_novelty_rule(self):
        self._paying_patch()
        self.assertFalse(DEPLETED(self.w),
                         'a patch that just paid was ejected on novelty')
        self.assertIn('stay:yield', self.w._dep_why)

    def test_he_can_still_leave_on_the_record_gap(self):
        self._paying_patch()
        self.w._max_gap = 10
        self.w._since_novel = 99      # a record drought
        self.assertTrue(DEPLETED(self.w),
                        'CAGED: the yield hold blocked every way out')

    def test_he_can_still_leave_on_the_mean_gap(self):
        self._paying_patch()
        self.w._gap_n = 5.0
        self.w._gap_mean = 4.0
        self.w._since_novel = 99
        if arc_world._PATCH_ROTATE:
            self.assertTrue(DEPLETED(self.w),
                            'CAGED: mean-gap exit was blocked too')

    def test_a_patch_that_never_paid_is_untouched(self):
        self.w._won_here = 0          # no win here
        self.w._steps_here = 50
        self.w._novel_here = 0
        ARCWorld._won_all = 1
        ARCWorld._steps_all = 5000
        ARCWorld._novel_all = 1000
        self.assertTrue(DEPLETED(self.w),
                        'the hold fired without a win in this visit')

    def test_gate_off_restores_the_old_behaviour(self):
        self._paying_patch()
        arc_world._YIELDSTAY_ON = lambda: False
        self.assertTrue(DEPLETED(self.w))


if __name__ == '__main__':
    unittest.main()


class TestNoVacuousEnvironment(TestYieldStay):
    """The environment must be more than this patch.

    Live at 02:05Z the rule fired 605 times from ONE win: with
    won_here == won_all the test is 1/steps_here > 1/steps_all, true
    forever.  Not a cage any more, but not calibrated either.
    """

    def test_his_only_win_does_not_hold_him(self):
        self._paying_patch()
        ARCWorld._won_all = 1         # his ONLY win is the one here
        self.assertTrue(DEPLETED(self.w),
                        'held on a vacuous environment: this patch IS '
                        'the environment, so the comparison means nothing')

    def test_a_second_earner_makes_the_comparison_real(self):
        self._paying_patch()          # won_all 3 > won_here 1
        self.assertFalse(DEPLETED(self.w),
                         'a genuinely out-paying patch was ejected')
