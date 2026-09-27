"""IMMORTALITY TRIPWIRE (2026-09-04).

The suite could not tell a HEALTHY wall from an IMMORTAL one:

  * `test_advance_at_zero_E_always_at_least_D` hardcodes `E = 0.0`, so it is
    structurally incapable of observing any behaviour at `E > 0`.
  * `test_wall_clamped_at_floor` drives `E = 1e6` and PASSES ONLY IF the wall
    sits at `BASELINE_FLOOR` -- which is precisely what immortality looks like.

That gap is not hypothetical.  The pre-2026-07-14 law was

    advance = aging * (1.0 - math.tanh(episode_earned))

and `1.0 - math.tanh(x)` is EXACTLY `0.0` in float64 for `x >~ 19.1`.  A
strong learner then aged not slowly but *not at all*, and every test in the
suite still passed.

DOCTRINE (f_mortality_is_the_selection_pressure, user 2026-09-04): mortality
is the baseline everything else is measured against -- "if mortality is not
the baseline evolution works against, nothing will be more important than
anything else."  A wall that can never reach lifeforce is therefore as
broken as one that reaches it on a fixed timer.

These tests assert the one property that must survive ANY future change to
the aging law: DEATH REMAINS REACHABLE.
"""

from __future__ import annotations

import unittest

from seagi.core.substrate import EDGE_STRENGTH_DECAY_PER_CYCLE
from seagi.brain import EventBus
from seagi.brain.capabilities.mortality_drive import (
    MortalityDrive,
    BASELINE_FLOOR,
)

D = EDGE_STRENGTH_DECAY_PER_CYCLE


class _Body:
    def __init__(self, lifeforce=0.7):
        self.lifeforce = lifeforce


def _drive(body, cycle_box):
    bus = EventBus()
    md = MortalityDrive(
        bus=bus,
        cycle_provider=lambda: cycle_box[0],
        lifeforce_get=lambda: body.lifeforce,
        lifeforce_set=lambda v: setattr(body, 'lifeforce', v))
    return bus, md


class TestDeathRemainsReachable(unittest.TestCase):
    """The anti-immortality half.  Complements the existing clamp tests,
    which can only observe that the wall IS at the floor -- never that it
    can leave again."""

    def test_advance_never_underflows_to_zero_below_breakeven(self):
        """THE TANH BUG, GENERALISED.  For any E strictly below break-even
        the advance must be STRICTLY positive -- never exactly 0.0.  An
        advance of exactly zero is immortality by arithmetic, which is how
        the deleted law failed."""
        for o in (0.0, 0.5, 1.0, 5.0, 1e3):
            breakeven = D * (1.0 + o)
            for frac in (0.0, 0.5, 0.9, 0.999, 0.999999):
                E = breakeven * frac
                adv = MortalityDrive.compute_pertick_advance(o, E)
                self.assertGreater(
                    adv, 0.0,
                    'advance underflowed to %r at o=%r E=%r' % (adv, o, E))

    def test_advance_is_sign_reachable_in_both_directions(self):
        """The invariant the 09-04 doctrine audit demanded: the wall must be
        able to CLIMB and to RECEDE as a function of his earning, and sit
        null when earning exactly matches upkeep.  A term that can only
        subtract is immortality; one that can only be zero is the death
        loop."""
        for o in (0.0, 0.5, 1.0, 5.0):
            breakeven = D * (1.0 + o)
            self.assertGreater(
                MortalityDrive.compute_pertick_advance(o, 0.0), 0.0)
            self.assertLess(
                MortalityDrive.compute_pertick_advance(o, breakeven * 10.0),
                0.0)
            self.assertAlmostEqual(
                MortalityDrive.compute_pertick_advance(o, breakeven),
                0.0, places=12)

    def test_wall_is_driven_down_then_resumes_climbing(self):
        """THE TRIPWIRE.  Three phases, and each one matters:

          1. wall starts ABOVE the floor (not at it -- otherwise "pinned at
             the floor" is trivially true and proves nothing);
          2. huge E must DRIVE IT DOWN to the floor -- a law that can only
             slow aging, never reverse it, fails here;
          3. with earning stopped the wall must CLIMB AGAIN -- any latch,
             saturation or underflow surviving phase 2 shows up here.

        Validated 2026-09-04 against the deleted tanh law, which fails
        phase 2 (its advance is aging*(1-slowdown) >= 0, so the wall can
        never recede)."""
        body = _Body(0.9)
        cyc = [0]
        _, md = _drive(body, cyc)
        E_box = [1e6]
        md.set_clock_providers(E_provider=lambda: E_box[0],
                               cf_provider=lambda: 0.0)
        md.tick()
        md.wall = 0.5                      # start well above the floor
        for _ in range(200):
            cyc[0] += 1
            md.tick()
        self.assertEqual(
            md.wall, BASELINE_FLOOR,
            'sustained earning did not drive the wall back down to the '
            'floor -- the aging term cannot reverse, only slow')

        E_box[0] = 0.0
        before = md.wall
        for _ in range(200):
            cyc[0] += 1
            md.tick()
        self.assertGreater(
            md.wall, before,
            'wall did not resume climbing after earning stopped -- '
            'death has become unreachable (immortality)')

    def test_time_to_death_is_finite_when_not_earning(self):
        """Death must be reachable in BOUNDED time, not merely eventually.
        With E = 0 the advance is at least D, so the wall closes any gap in
        at most (lifeforce - floor)/D ticks.  If this ever fails, the wall
        can be outrun by construction."""
        for o in (0.0, 1.0, 5.0):
            adv = MortalityDrive.compute_pertick_advance(o, 0.0)
            self.assertGreaterEqual(adv, D)
            worst_case = (1.0 - BASELINE_FLOOR) / adv
            self.assertLess(worst_case, float('inf'))


if __name__ == '__main__':
    unittest.main()


# =====================================================================
# SUSTAIN (2026-09-04): living as he usually does must not age him.
# NOTE: these monkeypatch the gate.  They must NEVER create
# /root/SUSTAIN_ON -- that would arm the LIVE daemon.
# =====================================================================
import seagi.brain.capabilities.mortality_drive as _md_mod


class _Counts:
    """Stand-in for grounding: raw cumulative (confirms, predictions)."""
    def __init__(self, rate):
        self.c = 0.0
        self.p = 0.0
        self.rate = rate

    def step(self, n=100.0, rate=None):
        r = self.rate if rate is None else rate
        self.p += n
        self.c += n * r

    def __call__(self):
        return (self.c, self.p)


class TestSustainHoldsTheWall(unittest.TestCase):

    def setUp(self):
        self._real_gate = _md_mod._SUSTAIN_ON
        _md_mod._SUSTAIN_ON = lambda: True

    def tearDown(self):
        _md_mod._SUSTAIN_ON = self._real_gate

    def _run(self, rates, start_wall=0.5, warm=400):
        body = _Body(0.9)
        cyc = [0]
        _, md = _drive(body, cyc)
        md.set_clock_providers(E_provider=lambda: 0.0,
                               cf_provider=lambda: 0.0)
        counts = _Counts(rates[0])
        md.set_sustain_provider(counts)
        md.tick()
        md.wall = start_wall
        for _ in range(warm):                 # establish the habit
            counts.step()
            cyc[0] += 1
            md.tick()
        before = md.wall
        for r in rates[1:]:
            for _ in range(400):
                counts.step(rate=r)
                cyc[0] += 1
                md.tick()
        return before, md.wall, md

    def test_gate_off_is_byte_identical(self):
        # HERMETIC: must NOT call the real gate -- /root/SUSTAIN_ON now
        # exists in production, so the real one returns True and this
        # test silently inverted (2026-09-04).
        _md_mod._SUSTAIN_ON = lambda: False
        before, after, md = self._run([0.5, 0.5], warm=200)
        self.assertGreater(after, before,
                           'with the gate off the wall must age exactly as '
                           'it did before')
        self.assertEqual(md._last_sustain, 0.0)

    def test_living_at_his_habit_does_not_age_him(self):
        before, after, _ = self._run([0.5, 0.5])
        self.assertLessEqual(
            after, before + 1e-9,
            'holding his habitual confirmation rate still aged him -- '
            'that is a lifespan timer, not mortality')

    def test_doing_better_than_usual_pushes_the_wall_down(self):
        before, after, _ = self._run([0.4, 0.9])
        self.assertLess(after, before)

    def test_genuine_decline_still_ages_him(self):
        before, after, _ = self._run([0.8, 0.05])
        self.assertGreater(
            after, before,
            'a real decline in what he knows must still age him -- '
            'that is old age, and it must stay reachable')

    def test_no_new_predictions_leaves_the_law_untouched(self):
        """Asleep / not predicting: no interval, no sustain, incumbent law."""
        body = _Body(0.9)
        cyc = [0]
        _, md = _drive(body, cyc)
        md.set_clock_providers(E_provider=lambda: 0.0,
                               cf_provider=lambda: 0.0)
        md.set_sustain_provider(lambda: (7.0, 21.0))   # frozen counts
        md.tick()
        md.wall = 0.5
        before = md.wall
        for _ in range(300):
            cyc[0] += 1
            md.tick()
        self.assertGreater(after_ := md.wall, before)
        self.assertEqual(md._last_sustain, 0.0)


class TestDeathStaysReachableWithSustain(unittest.TestCase):
    """The sustain term lets the wall HOLD.  It must not let him hold
    forever: mortality is the baseline, so a long life of ordinary
    fluctuation around his habit must still drift the wall up and kill
    him.  The asymmetry that provides this is the floor clamp -- surplus
    is discarded at BASELINE_FLOOR, deficit is not."""

    def setUp(self):
        self._real_gate = _md_mod._SUSTAIN_ON
        _md_mod._SUSTAIN_ON = lambda: True

    def tearDown(self):
        _md_mod._SUSTAIN_ON = self._real_gate

    def test_fluctuation_around_habit_still_drifts_the_wall_up(self):
        import random
        rng = random.Random(20260904)
        body = _Body(0.9)
        cyc = [0]
        _, md = _drive(body, cyc)
        md.set_clock_providers(E_provider=lambda: 0.0,
                               cf_provider=lambda: 0.0)
        counts = _Counts(0.5)
        md.set_sustain_provider(counts)
        md.tick()
        for _ in range(2000):                      # settle the habit
            counts.step()
            cyc[0] += 1
            md.tick()
        md.wall = BASELINE_FLOOR
        for _ in range(60000):                     # a long, ordinary life
            counts.step(rate=max(0.0, min(1.0, rng.gauss(0.5, 0.15))))
            cyc[0] += 1
            md.tick()
        self.assertGreater(
            md.wall, BASELINE_FLOOR,
            'a whole life of ordinary fluctuation left the wall pinned at '
            'the floor -- death has become unreachable')
