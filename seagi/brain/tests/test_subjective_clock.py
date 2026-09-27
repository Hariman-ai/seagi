"""Tests for SubjectiveClock (2026-06-12) — Part A: the persistent
monotonic subjective-time organ.

_internal_cycle is the brain's sole clock (advances +1 per brain.tick).
It was never serialized, so every restart reset 'now' to ~0 while the
substrate carried stamps far in the future, inverting every
(now - stamp) reader (decay, quarantine grace, replay staleness).

Part A persists it and restores it FORWARD-ONLY:
  - to_dict writes 'subjective_clock' = int(_internal_cycle)
  - load_personality restores it but NEVER rewinds the running counter
  - a legacy save lacking the key leaves the clock untouched (no crash)

Part A prevents the inversion from RECURRING on future restarts.  It
does NOT repair the current corrupted save's future-dated stamps —
that is a separate, gated step (Part B).
"""

from __future__ import annotations

import unittest

from seagi.body.engine import Engine
from seagi.brain.runtime import Brain


class TestSubjectiveClockPersistence(unittest.TestCase):

    def test_to_dict_persists_subjective_clock(self):
        brain = Brain(engine=Engine())
        brain._internal_cycle = 5000
        snap = brain.to_dict()
        self.assertIn('subjective_clock', snap)
        self.assertEqual(snap['subjective_clock'], 5000)

    def test_restore_jumps_clock_forward(self):
        brain = Brain(engine=Engine())
        brain._internal_cycle = 5000
        snap = brain.to_dict()

        fresh = Brain(engine=Engine())
        # Fresh brain starts at 0 (or low); restore must pull it forward.
        fresh._internal_cycle = 100
        fresh.load_personality(snap)
        self.assertEqual(fresh._internal_cycle, 5000)

    def test_restore_is_forward_only_never_rewinds(self):
        # The clock is strictly monotonic: a saved value LOWER than the
        # running counter must NOT rewind it.  This is what makes
        # 'subjective time' time — it only moves forward.
        brain = Brain(engine=Engine())
        brain._internal_cycle = 5000
        snap = brain.to_dict()

        ahead = Brain(engine=Engine())
        ahead._internal_cycle = 9000
        ahead.load_personality(snap)  # snap carries 5000 < 9000
        self.assertEqual(ahead._internal_cycle, 9000)

    def test_legacy_save_without_key_leaves_clock_untouched(self):
        # Backward compatibility: a pre-Part-A save has no
        # 'subjective_clock' key.  load_personality must not crash and
        # must leave the running counter where it is.
        brain = Brain(engine=Engine())
        brain._internal_cycle = 5000
        snap = brain.to_dict()
        del snap['subjective_clock']

        legacy = Brain(engine=Engine())
        legacy._internal_cycle = 250
        legacy.load_personality(snap)
        self.assertEqual(legacy._internal_cycle, 250)


if __name__ == '__main__':
    unittest.main()
