"""EVERY HOOK THAT CHANGES THE ACTION MUST REACH THE PUBLISH.

THE BUG THIS EXISTS FOR (2026-08-27):
`propose` does not act by returning -- runtime.py calls it and DISCARDS
the value.  The action reaches the world only through the
CapabilityClaimEvent published at the END of `propose`, which the basal
ganglia then arbitrates.

The KNOWNPATH hook returned ~500 lines BEFORE that publish.  So whenever
it matched a board: no claim, no arbitration, HE DID NOT MOVE -- and he
stayed on that board, so it matched again next tick.  A permanent stall
on any board a sealed path covered.  He lost hours to it.

34 tests passed through that bug.  Every one tested the ORGAN in
isolation -- seal, loop-erase, shortest-kept, persistence.  NONE tested
that a chosen action reaches the world.

These are source-level: they read `propose` and assert its structure,
because the failure was structural and no behavioural test of an organ
could have seen it.
"""
import inspect
import unittest

from seagi.brain.capabilities import world_actor




class TestActionReachesWorld(unittest.TestCase):

    def setUp(self):
        self.src = inspect.getsource(world_actor.WorldActor.propose)

    def test_propose_publishes_a_claim(self):
        self.assertIn('bus.publish(CapabilityClaimEvent', self.src,
                      'propose must publish a claim -- that is the ONLY '
                      'way its action reaches the world')

    def test_no_organ_hook_returns_before_the_publish(self):
        """The bug: a hook returned before the publish, so no claim.

        Every gate that can change `a` must sit before the publish and
        fall through to it.  A `return` between the first organ hook and
        the publish silently suppresses his action.
        """
        lines = self.src.splitlines()
        pub = next((i for i, l in enumerate(lines)
                    if 'CapabilityClaimEvent' in l
                    and not l.strip().startswith('#')), None)
        self.assertIsNotNone(pub, 'no publish found in propose')
        first_hook = next(
            (i for i, l in enumerate(lines)
             if any(g in l for g in ('_KNOWNPATH_ON()', '_VERDICT_ON()',
                                     '_CYCLEALL_ON()', '_UNTRIEDHERE_ON()',
                                     '_EXPLORER_ON()'))),
            None)
        if first_hook is None:
            self.skipTest('no organ hooks present')
        offenders = [(i + 1, l.strip())
                     for i, l in enumerate(lines[first_hook:pub],
                                           start=first_hook)
                     if l.strip().startswith('return ')]
        self.assertEqual(
            offenders, [],
            'a hook returns before the publish -- his action would be '
            'SUPPRESSED, not chosen: %r' % (offenders,))

    def test_every_gate_is_applied_before_the_publish(self):
        """A gate mentioned only AFTER the publish cannot affect the
        action it was written to change."""
        lines = self.src.splitlines()
        pub = next(i for i, l in enumerate(lines)
                   if 'CapabilityClaimEvent' in l
                   and not l.strip().startswith('#'))
        for gate in ('_VERDICT_ON()', '_CYCLEALL_ON()',
                     '_UNTRIEDHERE_ON()', '_EXPLORER_ON()'):
            hits = [i for i, l in enumerate(lines) if gate in l]
            if not hits:
                continue
            self.assertTrue(
                any(i < pub for i in hits),
                '%s is only referenced after the publish' % gate)


if __name__ == '__main__':
    unittest.main()
