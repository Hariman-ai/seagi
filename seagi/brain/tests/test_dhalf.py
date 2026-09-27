"""PATCH 16 (2026-09-07) -- DHALF: the discriminability modulator HALVES
the sleep gate at full modulation (its documented contract) instead of
zeroing it.

Pinned here:
  * gate on, d_mod = 1.0: theta = full_scale / 2 (> 0)
  * gate off, d_mod = 1.0: theta = 0 (the measured 71%-asleep regime)
  * gate on, d_mod = 0: theta = full_scale (nothing else changes)
  * the load term is untouched (multiplies as before)
  * no transition is provoked by the evaluation itself at zero excess
"""
import unittest
from unittest import mock

from seagi.brain.capabilities import sleep_wake
from seagi.brain.capabilities.sleep_wake import SleepRegulator


class TestDHalf(unittest.TestCase):
    def _reg(self, dmod, load=0.0):
        reg = SleepRegulator(bus=None, cycle_provider=lambda: 1,
                             d_modulation_provider=lambda: dmod,
                             load_provider=lambda: load,
                             adenosine_provider=lambda: reg._adenosine_baseline())
        return reg

    def _theta(self, on, dmod, load=0.0):
        reg = self._reg(dmod, load)
        with mock.patch.object(sleep_wake, '_DHALF_ON', lambda: on):
            reg._maybe_sleep(1)
        return reg

    def test_full_modulation_halves_the_gate(self):
        r = self._theta(True, 1.0)
        self.assertGreater(r.last_full_scale, 0.0)
        self.assertAlmostEqual(r.last_effective_threshold, 0.5 * r.last_full_scale)
        self.assertEqual(r.state, 'wake')

    def test_gate_off_still_zeroes_it(self):
        r = self._theta(False, 1.0)
        self.assertEqual(r.last_effective_threshold, 0.0)

    def test_no_modulation_is_unchanged(self):
        r_on = self._theta(True, 0.0); r_off = self._theta(False, 0.0)
        self.assertAlmostEqual(r_on.last_effective_threshold, r_on.last_full_scale)
        self.assertAlmostEqual(r_on.last_effective_threshold, r_off.last_effective_threshold)

    def test_load_term_untouched(self):
        r = self._theta(True, 1.0, load=0.5)
        self.assertAlmostEqual(r.last_effective_threshold, 0.5 * r.last_full_scale * 0.5)

    def test_half_modulation(self):
        r = self._theta(True, 0.5)
        self.assertAlmostEqual(r.last_effective_threshold, 0.75 * r.last_full_scale)


if __name__ == '__main__':
    unittest.main()
