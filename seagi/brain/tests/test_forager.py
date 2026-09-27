"""Tests for the V1→V2 Forager port.

Covers:
- inbox-first then reading_list pick order
- sleep-gating (no read while asleep)
- pace-gating (one read per FORAGE_INTERVAL_TICKS)
- feeds via intake_fn with origin='forager'
- thin-substrate THOUGHT_PRODUCED → information_request → biases pick
- read_manifest prevents re-reading reading_list files
- inbox files archived after ingest
- reconsider_all clears manifest (re-engageable)
- persistence round-trip
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest

from seagi.brain import EventKind, EventBus
from seagi.brain.events import ThoughtProducedEvent
from seagi.brain.capabilities.forager import (
    Forager, FORAGE_INTERVAL_TICKS, REQUEST_CAP)


def _thin_thought(focal: str, cycle: int = 0) -> ThoughtProducedEvent:
    return ThoughtProducedEvent(
        kind=EventKind.THOUGHT_PRODUCED,
        cycle=cycle,
        timestamp=time.time(),
        source_capability='cortical',
        origin='internal',
        origin_detail=focal,
        focal=focal, relation='', target='',
        confidence=0.2, method='inference', text='',
        thin_substrate=True)


class _ForagerHarness:
    """Forager + a captured-intake stub + controllable cycle/sleep."""

    def __init__(self, root):
        self.bus = EventBus()
        self.cycle = {'c': 10**6}   # start high so pace gate is open
        self.asleep = {'v': False}
        self.intaken = []   # (text, origin, origin_detail)
        self.forager = Forager(
            bus=self.bus,
            intake_fn=self._intake,
            cycle_provider=lambda: self.cycle['c'],
            is_asleep_provider=lambda: self.asleep['v'],
            root=root)
        self.forager.ensure_dirs()
        self.bus.subscribe(self.forager.SUBSCRIPTIONS, self.forager)

    def _intake(self, payload, modality='text', origin='internal',
                origin_detail=''):
        self.intaken.append((payload, origin, origin_detail))

    def advance(self, n):
        self.cycle['c'] += n


def _write(path, name, text):
    with open(os.path.join(path, name), 'w', encoding='utf-8') as f:
        f.write(text)


class TestForagerPick(unittest.TestCase):

    def test_inbox_first(self):
        with tempfile.TemporaryDirectory() as root:
            h = _ForagerHarness(root)
            _write(h.forager.paths['reading_list'], 'rl.txt',
                   'reading list content here')
            _write(h.forager.paths['inbox'], 'in.txt',
                   'inbox content here')
            summary = h.forager.tick()
            self.assertIsNotNone(summary)
            self.assertEqual(summary['location'], 'inbox')
            self.assertEqual(len(h.intaken), 1)
            text, origin, detail = h.intaken[0]
            self.assertEqual(origin, 'forager')
            self.assertEqual(detail, 'in.txt')

    def test_reading_list_when_inbox_empty(self):
        with tempfile.TemporaryDirectory() as root:
            h = _ForagerHarness(root)
            _write(h.forager.paths['reading_list'], 'rl.txt',
                   'reading list content here')
            summary = h.forager.tick()
            self.assertIsNotNone(summary)
            self.assertEqual(summary['location'], 'reading_list')

    def test_nothing_to_read_returns_none(self):
        with tempfile.TemporaryDirectory() as root:
            h = _ForagerHarness(root)
            self.assertIsNone(h.forager.tick())
            self.assertEqual(len(h.intaken), 0)


class TestForagerGating(unittest.TestCase):

    def test_no_read_while_asleep(self):
        with tempfile.TemporaryDirectory() as root:
            h = _ForagerHarness(root)
            _write(h.forager.paths['inbox'], 'in.txt', 'content here')
            h.asleep['v'] = True
            self.assertIsNone(h.forager.tick())
            self.assertEqual(len(h.intaken), 0)
            # Wakes → reads.
            h.asleep['v'] = False
            self.assertIsNotNone(h.forager.tick())

    def test_pace_gate(self):
        with tempfile.TemporaryDirectory() as root:
            h = _ForagerHarness(root)
            _write(h.forager.paths['inbox'], 'a.txt', 'aaaa content')
            _write(h.forager.paths['inbox'], 'b.txt', 'bbbb content')
            first = h.forager.tick()
            self.assertIsNotNone(first)
            # Immediately again — pace gate blocks.
            self.assertIsNone(h.forager.tick())
            # After the interval, reads the second.
            h.advance(FORAGE_INTERVAL_TICKS)
            second = h.forager.tick()
            self.assertIsNotNone(second)


class TestInformationRequests(unittest.TestCase):

    def test_thin_thought_creates_request(self):
        with tempfile.TemporaryDirectory() as root:
            h = _ForagerHarness(root)
            # Use a word whose lemma == itself so the test verifies
            # the channel mechanism, not the lemmatizer's quirks.
            h.bus.publish(_thin_thought('rosencrantz'))
            self.assertIn('rosencrantz',
                          list(h.forager.information_requests))

    def test_request_biases_pick(self):
        with tempfile.TemporaryDirectory() as root:
            h = _ForagerHarness(root)
            rl = h.forager.paths['reading_list']
            # Two reading-list files; one matches the request.
            _write(rl, 'aaa_general.txt', 'general background text')
            _write(rl, 'zzz_photosynthesis.txt', 'about plants and light')
            # Request photosynthesis (would otherwise pick aaa first).
            h.bus.publish(_thin_thought('photosynthesis'))
            summary = h.forager.tick()
            self.assertIsNotNone(summary)
            self.assertEqual(summary['file'], 'zzz_photosynthesis.txt')

    def test_request_dedup_and_cap(self):
        with tempfile.TemporaryDirectory() as root:
            h = _ForagerHarness(root)
            for i in range(REQUEST_CAP + 10):
                h.forager.request_information(f'concept{i}')
            self.assertLessEqual(
                len(h.forager.information_requests), REQUEST_CAP)
            # Re-requesting moves to front, no dup.
            h.forager.request_information('concept50')
            h.forager.request_information('concept50')
            self.assertEqual(
                list(h.forager.information_requests).count('concept50'),
                1)


class TestManifestAndArchive(unittest.TestCase):

    def test_reading_list_not_reread(self):
        with tempfile.TemporaryDirectory() as root:
            h = _ForagerHarness(root)
            _write(h.forager.paths['reading_list'], 'rl.txt',
                   'reading content here')
            h.forager.tick()
            self.assertIn('rl.txt', h.forager.read_manifest)
            # Next eligible tick: nothing new → None (already read).
            h.advance(FORAGE_INTERVAL_TICKS)
            self.assertIsNone(h.forager.tick())

    def test_inbox_archived_after_ingest(self):
        with tempfile.TemporaryDirectory() as root:
            h = _ForagerHarness(root)
            _write(h.forager.paths['inbox'], 'in.txt', 'inbox content')
            h.forager.tick()
            # File moved out of inbox.
            self.assertFalse(
                os.path.exists(
                    os.path.join(h.forager.paths['inbox'], 'in.txt')))

    def test_reconsider_all_clears_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            h = _ForagerHarness(root)
            _write(h.forager.paths['reading_list'], 'rl.txt',
                   'reading content here')
            h.forager.tick()
            self.assertEqual(len(h.forager.read_manifest), 1)
            cleared = h.forager.reconsider_all()
            self.assertEqual(cleared, 1)
            self.assertEqual(len(h.forager.read_manifest), 0)
            # Now re-readable.
            h.advance(FORAGE_INTERVAL_TICKS)
            self.assertIsNotNone(h.forager.tick())


class TestPersistence(unittest.TestCase):

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            h = _ForagerHarness(root)
            _write(h.forager.paths['inbox'], 'in.txt', 'content here')
            h.forager.tick()
            h.forager.request_information('gravity')
            snap = h.forager.to_dict()
            fresh = Forager(
                bus=EventBus(),
                intake_fn=lambda *a, **k: None,
                cycle_provider=lambda: 0,
                root=root)
            fresh.load_dict(snap)
            self.assertEqual(
                fresh.read_manifest, h.forager.read_manifest)
            self.assertIn('gravity',
                          list(fresh.information_requests))
            self.assertEqual(fresh.ingest_count, h.forager.ingest_count)


class TestBrainWiring(unittest.TestCase):

    def test_forager_wired_into_brain(self):
        from seagi.body.engine import Engine
        from seagi.brain.runtime import Brain
        brain = Brain(engine=Engine())
        self.assertIsNotNone(brain.forager)
        # Re-root the brain's forager at an empty temp dir so the
        # test doesn't depend on the CWD's reading_list contents.
        with tempfile.TemporaryDirectory() as root:
            brain.forager = Forager(
                bus=brain.bus,
                intake_fn=brain.intake,
                cycle_provider=lambda: brain._cycle_provider(),
                is_asleep_provider=(
                    lambda: brain.sleep_regulator.is_asleep()),
                root=root)
            before = brain.forager.ingest_count
            brain.tick()
            # Empty dirs → no read; tick path doesn't crash.
            self.assertEqual(brain.forager.ingest_count, before)


class TestSatisfactionLoop(unittest.TestCase):
    """Satisfaction-clearance loop: when forager-origin ingest writes
    a subject matching a pending information_request, the agent FEELS
    the resolution (insight chemistry on the focal) and that question
    leaves the channel — moving on to the next.  Curiosity-as-drive
    persists.
    """

    def setUp(self):
        # Forager has to be active (paths set) for the satisfaction
        # loop tests, but no files actually need to exist — the loop
        # is event-driven (SubstrateWriteQueuedEvent), not file-read.
        self._tmp = tempfile.TemporaryDirectory()
        self.h = _ForagerHarness(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _write_event(subject, origin='forager', cycle=10**6):
        from seagi.brain.events import SubstrateWriteQueuedEvent
        return SubstrateWriteQueuedEvent(
            kind=EventKind.SUBSTRATE_WRITE_QUEUED,
            cycle=cycle, timestamp=time.time(),
            source_capability='test', origin=origin,
            origin_detail='test',
            subject=subject, relation='is', object='thing',
            strength=0.1, write_reason='test')

    def _insight_recorder(self):
        from seagi.brain.events import ChemistryEvent
        fired = []
        class R:
            def handle(self, e, bus):
                if (isinstance(e, ChemistryEvent)
                        and e.chemistry_kind == 'insight'):
                    fired.append(e)
        rec = R()
        self.h.bus.subscribe((EventKind.CHEMISTRY_FIRE,), rec)
        return fired

    def test_request_key_is_lemmatized_on_store(self):
        # "kings" → "king".  Critical for matching ingest "king"
        # against a request originally posed as "kings".
        self.h.forager.request_information('kings')
        self.assertIn('king', list(self.h.forager.information_requests))

    def test_forager_ingest_matching_request_fires_insight_and_clears(self):
        self.h.forager.request_information('king')
        fired = self._insight_recorder()
        self.h.bus.publish(self._write_event('king'))
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].target_concepts, ['king'])
        self.assertNotIn(
            'king', list(self.h.forager.information_requests))
        self.assertEqual(self.h.forager.resolutions_fired, 1)
        self.assertEqual(self.h.forager.last_resolved, 'king')

    def test_lemma_match_kings_resolves_king_request(self):
        # Ingest subject "kings" → lemma "king" → resolves the
        # "king" request (or vice versa).  This is the D2 audit
        # concern made concrete.
        self.h.forager.request_information('king')
        fired = self._insight_recorder()
        self.h.bus.publish(self._write_event('kings'))
        self.assertEqual(len(fired), 1)
        self.assertNotIn(
            'king', list(self.h.forager.information_requests))

    def test_non_forager_origin_write_does_not_clear(self):
        # Self-percepts (inner_voice, origin='self') and reverie
        # internal writes (origin='internal') must NOT fire insight
        # — that would be self-answering / gaming.
        self.h.forager.request_information('king')
        fired = self._insight_recorder()
        for ori in ('self', 'internal', 'peer', ''):
            self.h.bus.publish(self._write_event('king', origin=ori))
        self.assertEqual(len(fired), 0)
        self.assertIn(
            'king', list(self.h.forager.information_requests))

    def test_non_matching_subject_does_not_fire(self):
        self.h.forager.request_information('king')
        fired = self._insight_recorder()
        self.h.bus.publish(self._write_event('something_else'))
        self.assertEqual(len(fired), 0)
        self.assertEqual(self.h.forager.resolutions_fired, 0)

    def test_resolution_fires_only_once_per_request(self):
        # A multi-sentence answer ingests many triples with the same
        # subject; the FIRST write clears the request and fires
        # insight, the rest must be silent on that subject.
        self.h.forager.request_information('king')
        fired = self._insight_recorder()
        for _ in range(5):
            self.h.bus.publish(self._write_event('king'))
        self.assertEqual(len(fired), 1)
        self.assertEqual(self.h.forager.resolutions_fired, 1)

    def test_multiple_distinct_requests_resolve_independently(self):
        # Curiosity-as-drive: clearing one question doesn't shut
        # the channel; other pending requests stay live and resolve
        # on their own evidence.
        for c in ('rosencrantz', 'lew', 'chap'):
            self.h.forager.request_information(c)
        fired = self._insight_recorder()
        self.h.bus.publish(self._write_event('rosencrantz'))
        self.h.bus.publish(self._write_event('chap'))
        self.assertEqual(len(fired), 2)
        self.assertIn(
            'lew', list(self.h.forager.information_requests))
        self.assertEqual(self.h.forager.resolutions_fired, 2)

    def test_status_surfaces_resolutions(self):
        self.h.forager.request_information('king')
        self.h.bus.publish(self._write_event('king'))
        s = self.h.forager.status()
        self.assertEqual(s['resolutions_fired'], 1)
        self.assertEqual(s['last_resolved'], 'king')


if __name__ == '__main__':
    unittest.main()
