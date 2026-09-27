"""Exogenous-input grounding loop — Cap-3 SHADOW phase (Step 5).

Proves the disarmed shadow wire is SAFE and NON-FARMABLE over exogenous
input: trivial/repeated input earns ~0, surprise pays once per edge,
precision kills lucky guesses, self-authored origins are excluded, the
shadow wire never writes back to lifeforce, and `anticipates` stays
decoupled from composition.
"""

from __future__ import annotations

import types
import unittest

from seagi.core.substrate import Substrate
from seagi.brain.capabilities.exogenous_grounding import (
    ExogenousGroundingLoop, EXO_RELATION, EXOGENOUS_ORIGINS)
from seagi.brain.capabilities.writer import JournaledSubstrateWriter
from seagi.brain.events import EventKind, RawPerceptEvent


class _DeliverBus:
    def __init__(self, writer):
        self.writer = writer

    def publish(self, event):
        self.writer.handle(event, self)


def _rig(shadow=True, record=None):
    sub = Substrate()
    sub._quarantine_migrated = True
    engine = types.SimpleNamespace(substrate=sub)
    writer = JournaledSubstrateWriter(engine=engine)
    loop = ExogenousGroundingLoop(
        engine=engine, bus=_DeliverBus(writer),
        shadow_mode=shadow, record_world_learning=record)
    return sub, loop


class TestExogenousGroundingShadow(unittest.TestCase):

    def test_trivial_repeated_stream_earns_zero(self):
        # A deterministic always-right stream is never SURPRISING ->
        # surprise gate stays 0 -> zero credit.  Trivial input pays nothing.
        _, loop = _rig()
        for i in range(60):
            loop.observe('a', 'b', cycle=i, origin='ingestion')
        self.assertGreater(loop.confirms, 0)        # it does predict+confirm
        self.assertEqual(loop.surprise_fires, 0)    # but never surprised
        self.assertEqual(loop.shadow_credit_total, 0.0)

    def test_surprise_pays_once_per_edge(self):
        # First correct anticipation of a PREVIOUSLY-WRONG transition pays;
        # the same transition never pays again (one-shot per edge).
        _, loop = _rig()
        key = ('a', EXO_RELATION, 'b')
        loop._registry[key] = [0, 1]                 # was_wrong, unpaid
        loop._rel['a'] = [10.0, 1.0, 0.0]            # fully reliable -> precision 1
        c1 = loop._credit(key, 'a', 'ingestion')
        c2 = loop._credit(key, 'a', 'ingestion')
        self.assertGreater(c1, 0.0)
        self.assertEqual(c2, 0.0)                    # paid -> no second credit
        self.assertEqual(loop.surprise_fires, 1)

    def test_no_credit_without_prior_wrong(self):
        # A transition that was right from the start (never was_wrong)
        # earns nothing even when confirmed reliably.
        _, loop = _rig()
        key = ('a', EXO_RELATION, 'b')
        loop._registry[key] = [0, 0]                 # NOT was_wrong
        loop._rel['a'] = [10.0, 1.0, 0.0]
        self.assertEqual(loop._credit(key, 'a', 'ingestion'), 0.0)

    def test_precision_kills_lucky_guess(self):
        # A genuinely-uncertain (noisy 50/50) source: surprise may fire,
        # but precision_weight ~0 -> credit ~0.  Luck pays nothing.
        _, loop = _rig()
        key = ('q', EXO_RELATION, 'y')
        loop._registry[key] = [0, 1]
        # 50/50 reliability: mean 0.5, var 0.25 -> normalized_var 1 -> weight 0
        loop._rel['q'] = [20.0, 0.5, 5.0]            # m2/n = 0.25
        credit = loop._credit(key, 'q', 'ingestion')
        self.assertLess(credit, 0.05)
        # contrast: a reliable source with the same surprise pays ~full
        key2 = ('d', EXO_RELATION, 'x')
        loop._registry[key2] = [0, 1]
        loop._rel['d'] = [20.0, 1.0, 0.0]
        self.assertGreater(loop._credit(key2, 'd', 'ingestion'), 0.5)

    def test_nonexo_origin_skipped_and_breaks_chain(self):
        # Self-authored / ambiguous input is not grounded and breaks the
        # prediction chain (so a self-percept can't sit between two
        # exogenous ones and be credited).
        _, loop = _rig()
        loop._last_state = 'prev'
        ev = RawPerceptEvent(
            kind=EventKind.RAW_PERCEPT, cycle=1, timestamp=0.0,
            source_capability='test', origin='internal',
            payload={'x': 0.9}, modality='text')
        loop.handle(ev, loop.bus)
        self.assertEqual(loop.skipped_nonexo, 1)
        self.assertIsNone(loop._last_state)          # chain broken
        self.assertEqual(loop.predictions_made, 0)
        # 'tool' (agent-initiated) and 'sensor' (ambiguous) are excluded too
        self.assertNotIn('tool', EXOGENOUS_ORIGINS)
        self.assertNotIn('sensor', EXOGENOUS_ORIGINS)
        self.assertNotIn('internal', EXOGENOUS_ORIGINS)

    def test_exogenous_origin_grounds(self):
        # A genuinely-external percept IS processed.
        _, loop = _rig()
        ev = RawPerceptEvent(
            kind=EventKind.RAW_PERCEPT, cycle=1, timestamp=0.0,
            source_capability='test', origin='curiosity_feeder',
            payload={'apple': 0.9, 'fruit': 0.3}, modality='text')
        loop.handle(ev, loop.bus)
        self.assertEqual(loop._last_state, 'apple')   # dominant concept

    def test_shadow_never_writes_back(self):
        # shadow_mode=True: credit is computed but record_world_learning
        # is NEVER called and applied_total stays 0.
        applied = []
        _, loop = _rig(shadow=True, record=lambda c: applied.append(c))
        key = ('a', EXO_RELATION, 'b')
        loop._registry[key] = [0, 1]
        loop._rel['a'] = [10.0, 1.0, 0.0]
        c = loop._credit(key, 'a', 'ingestion')
        self.assertGreater(c, 0.0)               # credit computed
        self.assertEqual(applied, [])            # but NOT applied
        self.assertEqual(loop.applied_total, 0.0)

    def test_armed_would_write_back(self):
        # Flipping shadow_mode=False (the arming commit) routes credit to
        # record_world_learning — proves the wire is only one flag away,
        # and that shadow is genuinely disarming it.
        applied = []
        _, loop = _rig(shadow=False, record=lambda c: applied.append(c))
        key = ('a', EXO_RELATION, 'b')
        loop._registry[key] = [0, 1]
        loop._rel['a'] = [10.0, 1.0, 0.0]
        loop._credit(key, 'a', 'ingestion')
        self.assertEqual(len(applied), 1)
        self.assertGreater(loop.applied_total, 0.0)

    def test_credit_attributed_by_origin(self):
        _, loop = _rig()
        for org in ('ingestion', 'curiosity_feeder'):
            key = (f's_{org}', EXO_RELATION, 't')
            loop._registry[key] = [0, 1]
            loop._rel[f's_{org}'] = [10.0, 1.0, 0.0]
            loop._credit(key, f's_{org}', org)
        self.assertIn('ingestion', loop.shadow_credit_by_origin)
        self.assertIn('curiosity_feeder', loop.shadow_credit_by_origin)

    def test_exo_relation_decoupled_from_composition(self):
        # anticipates must never be a composition output, or world edges
        # could cohere -> newly_coherent -> record_learning (a backdoor).
        from seagi.brain.capabilities.cortical import RELATION_COMPOSITION
        self.assertNotIn(EXO_RELATION, set(RELATION_COMPOSITION.values()))
        self.assertNotIn(EXO_RELATION, set(RELATION_COMPOSITION.keys()))


if __name__ == '__main__':
    unittest.main()
