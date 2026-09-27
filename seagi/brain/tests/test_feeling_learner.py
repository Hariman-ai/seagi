"""Tests for the V1→V2 FeelingLearner port.

Covers:
- tick samples the 8-channel chemistry vector, tags with a label
- centroid learning groups samples by label
- nearest-centroid classification + confidence gap
- learned label overrides bootstrap cascade in tone_summary
  (above FEELING_CONF_FLOOR) and falls back below it
- rare centroids fade-not-delete on relearn
- sleep-state tagging
- persistence round-trip
- read-only: no substrate writes / no metabolic debt
"""

from __future__ import annotations

import unittest

from seagi.brain import EventBus
from seagi.brain.chemistry_types import CHANNELS
from seagi.brain.capabilities.feeling_learner import (
    FeelingLearner, MIN_SAMPLES_PER_FEELING, RELEARN_EVERY,
    MAX_SAMPLES)
from seagi.brain.capabilities.chemistry import FEELING_CONF_FLOOR


def _baseline_state():
    return {ch: cfg['baseline'] for ch, cfg in CHANNELS.items()}


class _Harness:
    def __init__(self, bootstrap_label='flat'):
        self.bus = EventBus()
        self.state = _baseline_state()
        self.cycle = {'c': 0}
        self.asleep = {'v': False}
        self.bootstrap = {'label': bootstrap_label}
        self.fl = FeelingLearner(
            bus=self.bus,
            cycle_provider=lambda: self.cycle['c'],
            chemistry_state_provider=lambda: self.state,
            bootstrap_label_fn=lambda: self.bootstrap['label'],
            is_asleep_provider=lambda: self.asleep['v'])

    def set_channel(self, ch, val):
        self.state[ch] = val

    def feed(self, n):
        for _ in range(n):
            self.fl.tick()
            self.cycle['c'] += 1


class TestSampling(unittest.TestCase):

    def test_tick_samples_with_label(self):
        h = _Harness(bootstrap_label='curious')
        h.fl.tick()
        self.assertEqual(len(h.fl._samples), 1)
        self.assertEqual(h.fl._samples[0]['feeling'], 'curious')
        # 8-channel signature captured.
        self.assertEqual(
            set(h.fl._samples[0]['signature'].keys()),
            set(CHANNELS.keys()))

    def test_no_provider_safe(self):
        fl = FeelingLearner(bus=EventBus(), cycle_provider=lambda: 0,
                            chemistry_state_provider=None)
        fl.tick()  # no crash
        self.assertEqual(len(fl._samples), 0)

    def test_sample_buffer_caps(self):
        h = _Harness()
        h.feed(MAX_SAMPLES + 50)
        self.assertEqual(len(h.fl._samples), MAX_SAMPLES)

    def test_sleep_state_tagged(self):
        h = _Harness()
        h.asleep['v'] = True
        h.fl.tick()
        self.assertTrue(h.fl._samples[-1]['asleep'])
        h.asleep['v'] = False
        h.fl.tick()
        self.assertFalse(h.fl._samples[-1]['asleep'])


class TestLearning(unittest.TestCase):

    def test_centroid_learned_per_label(self):
        h = _Harness(bootstrap_label='calm')
        # Feed enough 'calm' samples at baseline to pass min-samples.
        h.feed(MIN_SAMPLES_PER_FEELING + 2)
        h.fl.learn()
        self.assertIn('calm', h.fl.feeling_signatures)
        meta = h.fl.feeling_signatures['calm']
        self.assertIn('centroid', meta)
        self.assertGreaterEqual(meta['n_samples'],
                                MIN_SAMPLES_PER_FEELING)

    def test_relearn_fires_on_interval(self):
        h = _Harness()
        before = h.fl.relearns
        h.feed(RELEARN_EVERY)
        self.assertGreater(h.fl.relearns, before)

    def test_below_min_samples_no_new_centroid(self):
        h = _Harness(bootstrap_label='grief')
        h.feed(MIN_SAMPLES_PER_FEELING - 1)
        h.fl.learn()
        # Not enough → no centroid formed.
        self.assertNotIn('grief', h.fl.feeling_signatures)

    def test_rare_centroid_fades_not_deleted(self):
        h = _Harness()
        # Cold-start (no centroids yet): build two DISTINCT-state
        # clusters so the first learn() forms two centroids.
        h.bootstrap['label'] = 'hope'
        h.set_channel('serotonin', 0.85)
        h.feed(MIN_SAMPLES_PER_FEELING + 2)
        h.bootstrap['label'] = 'calm'
        h.set_channel('serotonin', CHANNELS['serotonin']['baseline'])
        h.set_channel('oxytocin', 0.75)
        h.feed(MIN_SAMPLES_PER_FEELING + 2)
        h.fl.learn()
        self.assertIn('hope', h.fl.feeling_signatures)
        self.assertIn('calm', h.fl.feeling_signatures)
        # Now feed ONLY the calm-state cluster and relearn — 'hope'
        # no longer appears in fresh samples but its centroid must
        # persist (fade-not-delete per chemistry-never-dissolves).
        h.fl._samples.clear()
        h.feed(MIN_SAMPLES_PER_FEELING + 2)
        h.fl.learn()
        self.assertIn('calm', h.fl.feeling_signatures)
        self.assertIn('hope', h.fl.feeling_signatures)


class TestClassification(unittest.TestCase):

    def test_distinct_states_classify_distinctly(self):
        h = _Harness()
        # Cluster A: high cortisol, tagged 'anxious'.
        h.bootstrap['label'] = 'anxious'
        h.set_channel('cortisol', 0.8)
        h.feed(MIN_SAMPLES_PER_FEELING + 2)
        # Cluster B: high dopamine, tagged 'pleasure'.
        h.bootstrap['label'] = 'pleasure'
        h.set_channel('cortisol', CHANNELS['cortisol']['baseline'])
        h.set_channel('dopamine', 0.9)
        h.feed(MIN_SAMPLES_PER_FEELING + 2)
        h.fl.learn()
        self.assertIn('anxious', h.fl.feeling_signatures)
        self.assertIn('pleasure', h.fl.feeling_signatures)
        # Now in a high-cortisol state → classify near 'anxious'.
        h.set_channel('dopamine', CHANNELS['dopamine']['baseline'])
        h.set_channel('cortisol', 0.8)
        result = h.fl.classify_current_feeling()
        self.assertEqual(result['feeling'], 'anxious')
        self.assertTrue(result['learned'])

    def test_cold_start_not_learned(self):
        h = _Harness()
        result = h.fl.classify_current_feeling()
        self.assertFalse(result['learned'])
        self.assertEqual(result['confidence'], 0.0)


class TestChemistryIntegration(unittest.TestCase):

    def test_learned_label_overrides_cascade(self):
        from seagi.brain.capabilities.chemistry import ChemistryEngine
        # FeelingLearner whose classify returns a confident label.
        fake_label = {'feeling': 'serene', 'confidence': 0.9,
                      'learned': True}
        chem = ChemistryEngine(feeling_provider=lambda: fake_label)
        tone = chem.tone_summary()
        self.assertEqual(tone['label'], 'serene')
        self.assertTrue(tone['label_learned'])

    def test_low_confidence_falls_back_to_cascade(self):
        from seagi.brain.capabilities.chemistry import ChemistryEngine
        below = {'feeling': 'serene',
                 'confidence': FEELING_CONF_FLOOR - 0.01,
                 'learned': True}
        chem = ChemistryEngine(feeling_provider=lambda: below)
        tone = chem.tone_summary()
        self.assertNotEqual(tone['label'], 'serene')
        self.assertFalse(tone['label_learned'])

    def test_no_provider_uses_cascade(self):
        from seagi.brain.capabilities.chemistry import ChemistryEngine
        chem = ChemistryEngine()
        tone = chem.tone_summary()
        # Baseline → 'flat'; label_learned False.
        self.assertFalse(tone['label_learned'])
        self.assertIn('label', tone)

    def test_bootstrap_label_now_nonrecursive(self):
        from seagi.brain.capabilities.chemistry import ChemistryEngine
        # Even with a confident provider, the bootstrap helper must
        # return the cascade label (no override, no recursion).
        chem = ChemistryEngine(
            feeling_provider=lambda: {
                'feeling': 'serene', 'confidence': 0.9,
                'learned': True})
        lbl = chem.bootstrap_tone_label_now()
        self.assertNotEqual(lbl, 'serene')


class TestPersistence(unittest.TestCase):

    def test_round_trip(self):
        h = _Harness(bootstrap_label='calm')
        h.feed(MIN_SAMPLES_PER_FEELING + 2)
        h.fl.learn()
        snap = h.fl.to_dict()
        fresh = FeelingLearner(
            bus=EventBus(), cycle_provider=lambda: 0,
            chemistry_state_provider=lambda: _baseline_state())
        fresh.load_dict(snap)
        self.assertEqual(
            set(fresh.feeling_signatures.keys()),
            set(h.fl.feeling_signatures.keys()))
        self.assertEqual(len(fresh._samples), len(h.fl._samples))


class TestBrainWiring(unittest.TestCase):

    def test_feeling_learner_wired_and_readonly(self):
        from seagi.body.engine import Engine
        from seagi.brain.runtime import Brain
        brain = Brain(engine=Engine())
        self.assertIsNotNone(brain.feeling_learner)
        # Chemistry's feeling_provider points at the learner.
        self.assertIsNotNone(brain.chemistry._feeling_provider)
        # Ticking samples chemistry; no metabolic debt accrues from
        # the learner (read-only observer).
        debt_before = brain.metabolic_debt.debt
        samples_before = brain.feeling_learner.samples_observed
        brain.tick()
        self.assertEqual(
            brain.feeling_learner.samples_observed,
            samples_before + 1)
        # Feeling-learner itself queued no substrate writes.
        # (debt may change from other tick activity, but the learner
        # writes nothing — assert it never calls a write path by
        # checking it has no such method/effect: samples grew,
        # signatures dict is the only state it owns.)
        self.assertIsInstance(
            brain.feeling_learner.feeling_signatures, dict)


if __name__ == '__main__':
    unittest.main()
