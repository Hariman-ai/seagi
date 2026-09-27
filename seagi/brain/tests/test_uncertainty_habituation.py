"""Uncertainty habituation = MORTALITY on the uncertainty (2026-06-04).

The autonomous loop stayed stuck even after goals were made mortal,
because the UNCERTAINTY was still immortal: a self-referential focal
("doubts") re-records itself every reflection, filling the sliding
window with itself and holding full pressure forever — so a dead goal
was instantly re-spawned. The fix: an uncertainty earns continued grip
ONLY by being RESOLVED (a non-thin "thought-well" thought); one you
keep failing to resolve HABITUATES (net pressure = occurrences minus
habituation), and once every focal has habituated to zero the monitor
fires NOTHING — attention falls through to exploration. Habituation is
itself mortal (resets when a focal leaves the window).

See feedback_stop_manufacturing_problems (immortality is the disease).
"""

from __future__ import annotations

import unittest

from seagi.brain.capabilities.uncertainty_monitor import (
    UncertaintyMonitor, REFRACTORY_CYCLES,
)
from seagi.brain.events import (
    EventKind, ThoughtProducedEvent, CapabilityClaimEvent,
)


class _Bus:
    def __init__(self):
        self.claims = []

    def publish(self, ev):
        if isinstance(ev, CapabilityClaimEvent):
            self.claims.append(ev)


def _thin(cycle, focal):
    return ThoughtProducedEvent(
        kind=EventKind.THOUGHT_PRODUCED, cycle=cycle,
        source_capability='cortical', origin='internal',
        focal=focal, relation='reflects_on', target='x',
        method='reflection', confidence=0.6, thin_substrate=True)


def _confident(cycle, focal):
    return ThoughtProducedEvent(
        kind=EventKind.THOUGHT_PRODUCED, cycle=cycle,
        source_capability='cortical', origin='internal',
        focal=focal, relation='is_a', target='x',
        method='inference', confidence=0.9, thin_substrate=False)


class TestUncertaintyHabituation(unittest.TestCase):

    def _monitor(self):
        self.cyc = [0]
        m = UncertaintyMonitor(
            bus=None, cycle_provider=lambda: self.cyc[0])
        return m, _Bus()

    def _hammer(self, m, bus, focal, n, start=0):
        c = start
        for _ in range(n):
            self.cyc[0] = c
            m.handle(_thin(c, focal), bus)
            c += REFRACTORY_CYCLES + 1
        return c

    def test_persistent_unresolved_focal_habituates_to_silence(self):
        m, bus = self._monitor()
        self._hammer(m, bus, 'doubts', 40)
        # fired early, then habituated and went silent
        self.assertGreater(len(bus.claims), 0)
        self.assertGreater(m.habituated_suppressions, 0)
        # and it has dropped out of the goal-layer read entirely
        self.assertEqual(m.top_unresolved_focals(), [])

    def test_all_habituated_fires_nothing(self):
        m, bus = self._monitor()
        end = self._hammer(m, bus, 'doubts', 30)
        before = len(bus.claims)
        self._hammer(m, bus, 'doubts', 6, start=end)
        self.assertEqual(len(bus.claims), before)   # no new claims

    def test_resolution_clears_and_resets(self):
        m, bus = self._monitor()
        end = self._hammer(m, bus, 'doubts', 20)
        self.assertEqual(m.top_unresolved_focals(), [])   # habituated
        # a confident (non-thin) thought RESOLVES it
        self.cyc[0] = end
        m.handle(_confident(end, 'doubts'), bus)
        self.assertNotIn(
            'doubts', [f for (_c, f) in m._unresolved])    # cleared
        self.assertNotIn('doubts', m._habituation)         # reset

    def test_fresh_focal_still_drives(self):
        # habituation must not suppress legitimate NEW uncertainty
        m, bus = self._monitor()
        self._hammer(m, bus, 'newthing', 7)
        self.assertTrue(
            any('newthing' in cl.proposed_action for cl in bus.claims))

    def test_habituation_resets_when_focal_leaves_window(self):
        # mortality on the habituation itself: once a focal is no
        # longer pressing, its habituation is forgotten.
        m, bus = self._monitor()
        end = self._hammer(m, bus, 'doubts', 15)
        self.assertIn('doubts', m._habituation)
        # flood the window with a DIFFERENT focal so 'doubts' ages out
        self._hammer(m, bus, 'rivers', 12, start=end)
        self.assertNotIn('doubts', m._habituation)


if __name__ == '__main__':
    unittest.main()
