"""Tests for Step 0 organ 4b — Thalamic gate debt attenuation.

Covers:
- Threshold safety ceiling: chemistry-modulated threshold cannot
  exceed min(W_PEER, W_M, W_I, W_NOVELTY) × 1.0 = 0.5.
- Above-threshold survival bypass: percepts with salience >=
  threshold are NEVER attenuated by debt.
- Below-threshold attenuation formula:
  s × (1 − attenuation_strength × (1 − s/threshold)).
- Cold-start (no debt providers): no attenuation (baseline path
  preserved).
- Attenuation_strength clipping [0, 1].
- AttendedPerceptEvent's salience reflects debt-attenuation when
  it fires.
"""

from __future__ import annotations

import time
import unittest

from seagi.brain import EventKind, EventBus
from seagi.brain.events import (
    RawPerceptEvent, AttendedPerceptEvent,
)
from seagi.brain.capabilities.thalamic_gate import (
    ThalamicGate,
    THRESHOLD_SAFETY_CEILING,
    SALIENCE_THRESHOLD_BASE,
    W_PEER, W_M_CONTENT, W_I_CONTENT, W_NOVELTY,
)


class _DebtKnobs:
    def __init__(self, debt=0.0, baseline=0.0, full_scale=50.0):
        self.debt = debt
        self.baseline = baseline
        self.full_scale = full_scale

    def debt_provider(self):
        return self.debt

    def baseline_provider(self):
        return self.baseline

    def debt_full_scale_provider(self):
        return self.full_scale


def _make_raw(origin: str = 'peer',
                  payload=None,
                  cycle: int = 0) -> RawPerceptEvent:
    return RawPerceptEvent(
        kind=EventKind.RAW_PERCEPT,
        cycle=cycle,
        timestamp=time.time(),
        source_capability='test',
        origin=origin,
        origin_detail='test',
        modality='text',
        raw_text='danger',
        payload=(payload if payload is not None else {'danger': 1.0}),
    )


class TestSafetyCeiling(unittest.TestCase):

    def test_ceiling_matches_min_weight(self):
        expected = min(W_PEER, W_M_CONTENT, W_I_CONTENT, W_NOVELTY)
        self.assertEqual(THRESHOLD_SAFETY_CEILING, expected)

    def test_threshold_capped_by_ceiling(self):
        # Force a giant chemistry modulator that would push the
        # threshold well above the ceiling.
        gate = ThalamicGate(
            engine=None,
            threshold=1.0,  # base way above ceiling
            chemistry_provider=lambda: 1.0)
        # threshold_base × (0.6 + 0.8 × 1.0) = 1.4 — capped at 0.5.
        self.assertLessEqual(
            gate._threshold_now(), THRESHOLD_SAFETY_CEILING + 1e-9)


class TestAttenuationStrength(unittest.TestCase):

    def test_no_providers_returns_zero(self):
        gate = ThalamicGate()
        self.assertEqual(gate._attenuation_strength(), 0.0)

    def test_debt_at_baseline_zero(self):
        knobs = _DebtKnobs(debt=10.0, baseline=10.0, full_scale=50.0)
        gate = ThalamicGate(
            debt_provider=knobs.debt_provider,
            baseline_provider=knobs.baseline_provider,
            debt_full_scale_provider=knobs.debt_full_scale_provider)
        self.assertEqual(gate._attenuation_strength(), 0.0)

    def test_mid_attenuation(self):
        knobs = _DebtKnobs(debt=35.0, baseline=10.0, full_scale=50.0)
        gate = ThalamicGate(
            debt_provider=knobs.debt_provider,
            baseline_provider=knobs.baseline_provider,
            debt_full_scale_provider=knobs.debt_full_scale_provider)
        self.assertAlmostEqual(
            gate._attenuation_strength(), 0.5, places=6)

    def test_clamps_at_one(self):
        knobs = _DebtKnobs(debt=10000.0, full_scale=50.0)
        gate = ThalamicGate(
            debt_provider=knobs.debt_provider,
            baseline_provider=knobs.baseline_provider,
            debt_full_scale_provider=knobs.debt_full_scale_provider)
        self.assertEqual(gate._attenuation_strength(), 1.0)

    def test_zero_full_scale_safe(self):
        knobs = _DebtKnobs(debt=100.0, full_scale=0.0)
        gate = ThalamicGate(
            debt_provider=knobs.debt_provider,
            baseline_provider=knobs.baseline_provider,
            debt_full_scale_provider=knobs.debt_full_scale_provider)
        self.assertEqual(gate._attenuation_strength(), 0.0)


class TestAttenuationCurve(unittest.TestCase):

    def _make_gate(self, debt):
        knobs = _DebtKnobs(debt=debt, baseline=0.0, full_scale=50.0)
        return ThalamicGate(
            debt_provider=knobs.debt_provider,
            baseline_provider=knobs.baseline_provider,
            debt_full_scale_provider=knobs.debt_full_scale_provider)

    def test_above_threshold_passes_at_full_magnitude(self):
        # debt at max → attenuation_strength = 1.0.  s = 0.8 with
        # threshold = 0.4 means s > threshold; survival bypass.
        gate = self._make_gate(debt=100.0)
        out = gate._apply_debt_attenuation(0.8, threshold=0.4)
        self.assertEqual(out, 0.8)

    def test_at_threshold_no_attenuation(self):
        gate = self._make_gate(debt=100.0)
        # s == threshold → factor = 1 - 1.0 × 0 = 1.0
        out = gate._apply_debt_attenuation(0.4, threshold=0.4)
        self.assertAlmostEqual(out, 0.4, places=6)

    def test_deep_below_pegs_at_one_minus_strength(self):
        # s = 0, attenuation = 1.0, factor = 1 - 1 × 1 = 0 → output 0.
        gate = self._make_gate(debt=100.0)
        out = gate._apply_debt_attenuation(0.0, threshold=0.4)
        self.assertEqual(out, 0.0)

    def test_mid_below_threshold(self):
        # s = 0.2, threshold = 0.4 → depth = 0.5
        # strength = 0.5 → factor = 1 - 0.5 × 0.5 = 0.75
        # output = 0.2 × 0.75 = 0.15
        gate = self._make_gate(debt=25.0)
        out = gate._apply_debt_attenuation(0.2, threshold=0.4)
        self.assertAlmostEqual(out, 0.15, places=6)

    def test_zero_debt_no_attenuation(self):
        # strength = 0 → factor = 1 always.
        gate = self._make_gate(debt=0.0)
        out = gate._apply_debt_attenuation(0.2, threshold=0.4)
        self.assertAlmostEqual(out, 0.2, places=6)


class TestGateEmission(unittest.TestCase):
    """End-to-end: the gate's AttendedPerceptEvent.salience reflects
    debt attenuation when applicable."""

    def _collect_attended(self, bus, gate, event):
        collected = []

        class _Sink:
            SUBSCRIPTIONS = (EventKind.ATTENDED_PERCEPT,)

            def handle(self, ev, bus):
                if isinstance(ev, AttendedPerceptEvent):
                    collected.append(ev)
        sink = _Sink()
        bus.subscribe(sink.SUBSCRIPTIONS, sink)
        bus.subscribe(gate.SUBSCRIPTIONS, gate)
        bus.publish(event)
        return collected

    def test_above_threshold_survival_bypass_under_debt(self):
        knobs = _DebtKnobs(debt=1000.0, full_scale=50.0)
        gate = ThalamicGate(
            engine=None,
            chemistry_provider=lambda: 0.5,  # threshold_now ≈ 0.30
            debt_provider=knobs.debt_provider,
            baseline_provider=knobs.baseline_provider,
            debt_full_scale_provider=knobs.debt_full_scale_provider)
        # peer origin → peer_proximity = 1.0 → salience = W_PEER ×
        # 1.0 = 1.0 (well above any threshold).
        bus = EventBus()
        attended = self._collect_attended(
            bus, gate, _make_raw(origin='peer'))
        self.assertEqual(len(attended), 1)
        # Salience should be the un-attenuated 1.0 (survival bypass).
        self.assertEqual(attended[0].salience, 1.0)
        # Diagnostic: debt_attenuated_count did NOT bump on this
        # above-threshold percept.
        self.assertEqual(gate.debt_attenuated_count, 0)

    def test_below_threshold_attenuated_under_debt(self):
        # Use a fake substrate with the concept already well-known
        # so novelty doesn't dominate salience.
        from types import SimpleNamespace

        class _FakeBubble:
            encounter_count = 100
            transmitter_trace = SimpleNamespace(
                cortisol=0.0, norepinephrine=0.0,
                dopamine=0.0, oxytocin=0.0, endorphins=0.0)

        class _FakeConcept:
            bubbles = [_FakeBubble()]

        engine = SimpleNamespace(
            substrate=SimpleNamespace(
                concepts={'danger': _FakeConcept()}))
        knobs = _DebtKnobs(debt=1000.0, full_scale=50.0)
        gate = ThalamicGate(
            engine=engine,
            chemistry_provider=lambda: 0.5,
            debt_provider=knobs.debt_provider,
            baseline_provider=knobs.baseline_provider,
            debt_full_scale_provider=knobs.debt_full_scale_provider)
        bus = EventBus()
        # forager origin → peer_proximity = 0.2.  novelty = 0
        # (encounter_count >= 3).  m/i_content = 0 (zero trace).
        # salience = max(0, 0, 0, 0.2) = 0.2 — below threshold ~0.30.
        attended = self._collect_attended(
            bus, gate, _make_raw(origin='forager'))
        self.assertEqual(len(attended), 1)
        # Attenuated downward from 0.2.
        self.assertLess(attended[0].salience, 0.2)
        self.assertEqual(gate.debt_attenuated_count, 1)


if __name__ == '__main__':
    unittest.main()
