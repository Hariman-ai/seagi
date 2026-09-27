"""LIVENESS ASSERTIONS — catch organs that go SILENTLY DEAD in a refactor.

WHY THIS FILE EXISTS (2026-07-30).  In one day, six separate mechanisms were
found to be doing nothing, and **1,079 passing tests caught none of them**:

  * `note_attended` was DEAD CODE with no call site, so three "leave this patch"
    rules read frozen zeros for weeks;
  * the BASELINE DRIFT was deleted by the 2026-07-14 refactor, leaving
    `MORTALITY_DRIFT_RATE` and `L_SUSTAIN` declared but used in NO computation —
    the mortality pole had no moving parts while its docstring still promised it;
  * `Arcade.make()` was called without a `scorecard_id`, so every "score 0.0" in
    the project's history was an UNSCORED run;
  * a `depleted` branch changed patch with no logging and no hold check.

The reason the suite missed all of it: **a unit test that constructs its own
inputs cannot catch a wiring error.**  It proves a function works when called;
it never asks whether anything calls it, or whether its result reaches anything.

So these tests assert the two properties normal tests do not:
  1. STATIC   — a declared thing is actually REFERENCED somewhere (no orphans,
                no uncalled hooks).
  2. DYNAMIC  — a value that is supposed to MOVE actually moves when driven.

They must all PASS on current code, so that a future refactor which silently
kills a wire turns the suite RED instead of staying green.
"""
import ast
import os
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))


def _bus():
    from seagi.brain.bus import EventBus
    return EventBus()


def _read(rel):
    with open(os.path.join(_PKG, rel), encoding='utf-8') as f:
        return f.read()


def _iter_sources():
    for root, dirs, files in os.walk(_PKG):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for fn in files:
            if fn.endswith('.py'):
                p = os.path.join(root, fn)
                try:
                    with open(p, encoding='utf-8') as f:
                        yield p, f.read()
                except OSError:
                    continue


class TestNoOrphanedConstants(unittest.TestCase):
    """A constant declared but used in NO computation means the code that used
    it was deleted.  That is exactly how the baseline drift died: the refactor
    removed the assignment and left the constants behind, so the module still
    *described* an organ that no longer existed."""

    # (module, constant) pairs that are LOAD-BEARING: if one stops being
    # referenced, the mechanism it parameterises has been deleted.
    LOAD_BEARING = [
        ('seagi/brain/capabilities/mortality_drive.py', 'MORTALITY_DRIFT_RATE'),
        # L_SUSTAIN removed 2026-08-10 — the constant is DELETED, not
        # orphaned.  The reference it stood for is now `_L_habit`/`_b_habit`,
        # which are per-agent state rather than module constants, so the AST
        # check cannot cover them.  TestHabitIsTheReference below guards them
        # dynamically instead, which is a stronger check than this one.
        ('seagi/brain/capabilities/mortality_drive.py', 'MORTALITY_RELAX_RATE'),
        ('seagi/brain/capabilities/mortality_drive.py', 'BASELINE_FLOOR'),
    ]

    def test_load_bearing_constants_are_actually_used(self):
        for rel, name in self.LOAD_BEARING:
            src = _read(rel)
            tree = ast.parse(src)
            assigned = loaded = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == name:
                    if isinstance(node.ctx, ast.Store):
                        assigned += 1
                    else:
                        loaded += 1
            self.assertGreater(
                assigned, 0, '%s: %s is not declared at all' % (rel, name))
            self.assertGreater(
                loaded, 0,
                '%s: %s is DECLARED BUT NEVER READ -- the code that used it has '
                'been deleted, so the mechanism it parameterises is gone even '
                'though the module still documents it. This is exactly how the '
                'baseline drift died.' % (rel, name))


class TestDeclaredHooksHaveCallSites(unittest.TestCase):
    """A method whose docstring says "the runtime calls this" must actually be
    called somewhere.  `note_attended` carried all the state-novelty
    bookkeeping and had NO call site anywhere in the package, so three patch-
    depletion rules silently read frozen zeros."""

    # DEFERRED 2026-08-17: `note_attended` was implemented and called ONLY
    # inside ARCWorld, archived to /root/seagi_archive_arc_layer/code when the
    # benchmark was removed from the architecture.  The organ is GENERAL
    # (state-novelty bookkeeping feeding depleted()) and is owed an extraction
    # onto the ladder worlds -- but it is ALSO the organ diagnosed as the cause
    # of the rotation storm (08-14 one-counter finding), so it is NOT ported
    # blind.  RE-ADD THIS HOOK the moment it is extracted.
    HOOKS = [
        # RE-ADDED 2026-08-19: ARC is restored, so `note_attended` has its
        # call site back.  Asserts the CALL SITE exists -- it does NOT arm
        # the depletion rules the 08-14 one-counter finding blamed for the
        # rotation storm; those stay behind SEAGI_ARCROTATE.
        ('note_attended', 'state-novelty bookkeeping feeding depleted()'),
        ('paid_from_dict', 'restores which games have paid him across restarts'),
        ('paid_to_dict', 'persists which games have paid him'),
        ('record_learning', 'the immortality pole of the mortality drive'),
    ]

    def test_hooks_are_called_somewhere(self):
        bodies = list(_iter_sources())
        for hook, why in self.HOOKS:
            call_sites = 0
            for path, src in bodies:
                try:
                    tree = ast.parse(src)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        fn = node.func
                        nm = getattr(fn, 'attr', None) or getattr(fn, 'id', None)
                        if nm == hook:
                            call_sites += 1
            self.assertGreater(
                call_sites, 0,
                '%s() has NO CALL SITE anywhere in the package -- it is DEAD '
                'CODE. %s. Anything reading its output is seeing frozen '
                'values.' % (hook, why))


class TestBaselineActuallyDrifts(unittest.TestCase):
    """DYNAMIC: the mortality baseline must MOVE when episodes close.

    This is the assertion whose absence let the drift stay deleted for six
    weeks while `lf=1.000/1.000 horizon=inf` was printed every tick and the
    suite stayed green."""

    def test_baseline_moves_on_episode_close(self):
        from seagi.brain.capabilities.mortality_drive import MortalityDrive

        d = MortalityDrive(bus=_bus())
        d.baseline = 1.0
        start = float(d.baseline)

        # A non-learner (no earned growth) must ERODE.  Same guard, new
        # units (2026-08-10): the reference is his own habit rather than a
        # stamped L_SUSTAIN, so he is given one -- an agent who HAS learned
        # at 0.4 and then stops.  A cold-start drive with no history seeds
        # its habit to "learns nothing" and correctly drifts by zero, which
        # is the doctrine (idleness is answered by restlessness, not by
        # erosion) but would silently vacate this liveness check.
        d._L_habit = 0.4
        d._b_habit = 1.0
        for _ in range(10):
            d._close_episode()

        self.assertLess(
            float(d.baseline), start,
            'baseline did NOT move across 10 episode closes. The drift '
            'assignment has been deleted again -- the mortality pole has no '
            'moving parts and NOTHING can make dying matter.')
        self.assertNotEqual(
            float(getattr(d, '_last_drift', 0.0)), 0.0,
            '_last_drift stayed 0.0, so the drift never computed.')

    def test_a_learner_does_not_erode(self):
        """The other pole: sustained earning must NOT erode him. Guards against
        a 'fix' that makes the baseline fall unconditionally."""
        from seagi.brain.capabilities.mortality_drive import MortalityDrive

        # Maintenance scales with what he HOLDS (2026-08-09).  New units
        # (2026-08-10): the altitude is priced against his HABITUAL life
        # rather than the suffocation floor, so this learner sits BELOW his
        # habit (0.2 against a 0.5 habit = cheap) and earns ABOVE it.
        d = MortalityDrive(bus=_bus())
        d.baseline = 0.2
        d._L_habit = 0.4
        d._b_habit = 0.5
        start = float(d.baseline)
        for _ in range(10):
            d.record_learning(0.8)          # double his habitual rate
            d._close_episode()
        self.assertGreater(
            float(d.baseline), start,
            'a learner earning MORE than its maintenance ERODED. The '
            'drift has lost its sign -- learning must raise the '
            'set-point.')

    def test_a_learner_living_beyond_its_means_erodes(self):
        """M IS THE UNDERLYING RULE: earning at his habitual rate is not
        enough on its own -- what he HOLDS must be paid for.  A learner
        carrying more than its earning supports must lose altitude.

        UNCHANGED IN SUBSTANCE 2026-08-10, only in units: the old form
        priced the altitude against SUFFOCATION and the earning against a
        stamped L_SUSTAIN.  Both references are now his own history, so the
        same rule reads: carrying ABOVE his habitual life while earning only
        his habitual rate must erode.  The failure this catches is identical
        -- a rich life costing no more than a poor one, so I sits at the top
        for free and M is no longer underneath anything."""
        from seagi.brain.capabilities.mortality_drive import MortalityDrive

        d = MortalityDrive(bus=_bus())
        d.baseline = 0.8          # well above his habitual 0.5
        d._L_habit = 0.4
        d._b_habit = 0.5
        start = float(d.baseline)
        for _ in range(10):
            d.record_learning(0.4)          # exactly his habitual rate
            d._close_episode()
        self.assertLess(
            float(d.baseline), start,
            'a learner holding well above its habitual life on merely '
            'habitual earning did NOT erode -- maintenance has stopped '
            'scaling with what he holds, so I can win permanently and M '
            'is no longer underneath anything.')

    def test_the_poor_life_is_cheap(self):
        """The mirror of the rule above, which was never asserted: carrying
        BELOW his habitual life on habitual earning must RISE.  Without this
        the previous test passes trivially for a law that simply erodes
        everything, which is exactly the bug shipped on 2026-08-10 (every
        reachable L put the equilibrium at or under the floor)."""
        from seagi.brain.capabilities.mortality_drive import MortalityDrive

        d = MortalityDrive(bus=_bus())
        d.baseline = 0.3
        d._L_habit = 0.4
        d._b_habit = 0.5
        start = float(d.baseline)
        for _ in range(10):
            d.record_learning(0.4)
            d._close_episode()
        self.assertGreater(
            float(d.baseline), start,
            'living BELOW his habit on habitual earning did not recover. '
            'The law only erodes, so the floor is the sole attractor.')


class TestHabitIsTheReference(unittest.TestCase):
    """The reference the maintenance cost is measured against must be HIS
    OWN and must actually be consulted.  Replaces the AST orphan-check that
    covered the deleted `L_SUSTAIN` constant."""

    def test_changing_the_habit_changes_the_drift(self):
        from seagi.brain.capabilities.mortality_drive import MortalityDrive

        drifts = []
        for habit in (0.2, 0.8):
            d = MortalityDrive(bus=_bus())
            d.baseline = 0.5
            d._L_habit = habit
            d._b_habit = 0.5
            d.record_learning(0.4)
            d._close_episode()
            drifts.append(float(d._last_drift))
        self.assertNotEqual(
            drifts[0], drifts[1],
            'the habit does not enter the drift -- the cost reference is '
            'not being consulted, so it is decorative.')
        self.assertGreater(
            drifts[0], drifts[1],
            'earning 0.4 against a 0.2 habit must pay BETTER than against '
            'a 0.8 habit; the sign of the reference is inverted.')

    def test_mean_preserving_noise_does_not_erode(self):
        """Regression, 2026-08-10: with a direction-dependent satiation term
        a step down near the ceiling was ~17x a step up, so L wobbling
        +/-30% around an UNCHANGED mean walked the baseline 0.951 -> 0.591
        with no drought at all.  Erosion must require an actual shortfall."""
        from seagi.brain.capabilities.mortality_drive import MortalityDrive

        d = MortalityDrive(bus=_bus())
        d.baseline = 0.95
        d._L_habit = 0.4
        d._b_habit = 0.95
        start = float(d.baseline)
        for i in range(2000):
            d.record_learning(0.4 * (1.3 if (i // 20) % 2 == 0 else 0.7))
            d._close_episode()
        self.assertGreater(
            float(d.baseline), start - 0.02,
            'mean-preserving noise eroded the baseline by more than 2%%: '
            'the satiation term has become direction-dependent again.')

    def test_idleness_can_never_reach_the_suffocation_floor(self):
        """DOCTRINE (2026-08-10, user): he may die of old age or of his own
        deliberate act, never of being idle.  A total learning drought must
        bottom out well clear of the floor -- and structurally, because his
        habit falls with him, not because a clamp catches him there."""
        from seagi.brain.capabilities.mortality_drive import MortalityDrive
        from seagi.body.primitive_states import SUFFOCATION_LIFEFORCE

        d = MortalityDrive(bus=_bus())
        d.baseline = 0.5
        d._L_habit = 0.4
        d._b_habit = 0.5
        for _ in range(20000):
            d._close_episode()          # earning nothing, ever
        self.assertGreater(
            float(d.baseline), SUFFOCATION_LIFEFORCE + 0.02,
            'a total learning drought walked him onto the suffocation '
            'floor. Idleness has become lethal by construction again.')


if __name__ == '__main__':
    unittest.main()
