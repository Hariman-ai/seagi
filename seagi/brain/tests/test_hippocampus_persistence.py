"""Episodic memory must survive a restart.

Measured 2026-08-17/18 on the live VPS: `Hippocampus._episodes`
is a bounded deque built fresh in `__init__` and NEVER serialized,
while `world_route` (9,133), `world_svalue` (9,434) and
`world_rules` (308) all persist.  The asymmetry is exact:
PROCEDURAL memory survives restarts, FELT memory never has.  He
keeps the routes and loses why they mattered -- and he restarts
often (10 distinct PIDs inside three hours on 2026-08-18).  A live
snapshot at 15.6 h uptime held 486 episodes, 457 of them tagged at
encoding; all of it was discarded on restart.

THERE ARE TWO EPISODIC STORES, and this file is about the one that
is written.  `substrate.episodes` (`seagi/core/substrate.py:1431`,
serialized at `:3363`, `add_episode()` at `:3130`) persists
correctly and is EMPTY -- `grep -rn add_episode` finds only its own
definition, so nothing has ever written it.  Its `Episode`
(`substrate.py:1082`) is the richer one: `mi_at_event`,
`transmitters_at_event`, `action_taken`, `observed_outcome`,
`prediction_error`, `replay_priority`, semantic graduation.  The
`"episodes": []` in the save is that store, not debris.

So: one episodic store is written every step and never persisted;
the other persists and is never written.  These tests cover the
first.  Neither store has a reader -- nothing outside
`hippocampus.py` calls `recent_episodes()`, and `dmn.py:234` holds
a `hippocampus_provider` it never calls.

These tests fail until `Hippocampus.to_dict()` / `load_dict()`
exist and are wired into `Brain.to_dict()` / `load_personality()`.
"""

from __future__ import annotations

import time
import unittest

from seagi.brain import EventKind, EventBus
from seagi.brain.events import (
    AttendedPerceptEvent,
    ReflectionFiredEvent,
)
from seagi.brain.capabilities.hippocampus import Hippocampus


def _percept(cycle: int,
             focals=None,
             text: str = 'a thing happened',
             salience: float = 0.7,
             novelty: float = 0.6,
             m: float = 0.4,
             i: float = 0.2) -> AttendedPerceptEvent:
    return AttendedPerceptEvent(
        kind=EventKind.ATTENDED_PERCEPT,
        cycle=cycle,
        timestamp=time.time(),
        source_capability='gate',
        origin='sensor',
        origin_detail='world',
        focals=list(focals or ['door', 'key']),
        raw_text=text,
        salience=salience,
        novelty=novelty,
        m_content=m,
        i_content=i,
    )


def _reflection(cycle: int) -> ReflectionFiredEvent:
    return ReflectionFiredEvent(
        kind=EventKind.REFLECTION_FIRED,
        cycle=cycle,
        timestamp=time.time(),
        source_capability='idle',
        origin='internal',
        origin_detail='test',
        trigger='manual',
        reflection_kind='general',
    )


def _live(n: int = 5, cycle0: int = 100):
    """A hippocampus that has actually lived a little."""
    bus = EventBus()
    h = Hippocampus(bus=bus, cycle_provider=lambda: cycle0 + 1000)
    bus.subscribe(h.SUBSCRIPTIONS, h)
    for k in range(n):
        bus.publish(_percept(
            cycle=cycle0 + k,
            focals=[f'focal{k}', 'shared'],
            text=f'episode {k}',
            m=0.1 * k,
            i=0.05 * k))
    return bus, h


class TestHippocampusRoundTrip(unittest.TestCase):

    def test_buffer_survives_round_trip(self):
        """The felt record itself -- not just its counters."""
        _, h = _live(5)
        self.assertEqual(h.buffer_size(), 5)
        state = h.to_dict()

        fresh = Hippocampus(bus=EventBus())
        fresh.load_dict(state)

        self.assertEqual(fresh.buffer_size(), 5)
        for before, after in zip(h.recent_episodes(limit=5),
                                 fresh.recent_episodes(limit=5)):
            self.assertEqual(after.episode_id, before.episode_id)
            self.assertEqual(after.cycle, before.cycle)
            self.assertEqual(after.focals, before.focals)
            self.assertEqual(after.raw_text, before.raw_text)
            self.assertAlmostEqual(after.salience, before.salience)
            self.assertAlmostEqual(after.novelty, before.novelty)
            self.assertAlmostEqual(after.m_polarity, before.m_polarity)
            self.assertAlmostEqual(after.i_polarity, before.i_polarity)
            self.assertEqual(after.origin, before.origin)
            self.assertEqual(after.origin_detail, before.origin_detail)

    def test_episode_ids_do_not_collide_after_restore(self):
        """A restored buffer whose next id restarts at 1 would give
        two different memories the same name."""
        _, h = _live(5)
        state = h.to_dict()

        bus = EventBus()
        fresh = Hippocampus(bus=bus)
        bus.subscribe(fresh.SUBSCRIPTIONS, fresh)
        fresh.load_dict(state)
        bus.publish(_percept(cycle=200, focals=['new']))

        ids = [e.episode_id for e in fresh.recent_episodes(limit=10)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(max(ids), 6)

    def test_consolidated_flag_survives(self):
        """Without the flag, every restart re-consolidates the same
        episodes and re-writes the same co_occurs edges."""
        bus, h = _live(5)
        bus.publish(_reflection(cycle=110))
        self.assertGreater(h.episodes_consolidated, 0)
        n_consolidated = sum(
            1 for e in h.recent_episodes(limit=512) if e.consolidated)

        fresh = Hippocampus(bus=EventBus())
        fresh.load_dict(h.to_dict())

        self.assertEqual(
            sum(1 for e in fresh.recent_episodes(limit=512)
                if e.consolidated),
            n_consolidated)

    def test_counters_survive(self):
        bus, h = _live(6)
        bus.publish(_reflection(cycle=110))
        # A second, much later reflection so the episodes that did NOT
        # consolidate age past DECAY_AGE_CYCLES and actually decay --
        # otherwise episodes_decayed is 0 on both sides and the key is
        # pinned vacuously.
        bus.publish(_reflection(cycle=800))
        self.assertGreater(h.episodes_consolidated, 0)
        self.assertGreater(h.episodes_decayed, 0)
        self.assertGreater(h.edges_requested, 0)
        self.assertGreater(h.episodes_formed, 0)

        fresh = Hippocampus(bus=EventBus())
        fresh.load_dict(h.to_dict())

        self.assertEqual(fresh.episodes_formed, h.episodes_formed)
        self.assertEqual(fresh.episodes_consolidated,
                         h.episodes_consolidated)
        self.assertEqual(fresh.episodes_decayed, h.episodes_decayed)
        self.assertEqual(fresh.edges_requested, h.edges_requested)
        self.assertEqual(fresh.tagged_at_encoding, h.tagged_at_encoding)

    def test_tagged_at_encoding_survives(self):
        """`tagged_at_encoding` only moves when the percept carries no
        polarity of its own and the live chemistry supplies it -- the
        2026-08-17 felt-tag path.  Pin it with that path exercised, not
        at 0 on both sides."""
        bus = EventBus()
        h = Hippocampus(bus=bus, tone_provider=lambda: (0.3, 0.1))
        bus.subscribe(h.SUBSCRIPTIONS, h)
        for k in range(4):
            bus.publish(_percept(cycle=100 + k, m=0.0, i=0.0))
        self.assertEqual(h.tagged_at_encoding, 4)
        self.assertAlmostEqual(
            h.recent_episodes(limit=1)[0].m_polarity, 0.3)

        fresh = Hippocampus(bus=EventBus())
        fresh.load_dict(h.to_dict())
        self.assertEqual(fresh.tagged_at_encoding, 4)
        self.assertAlmostEqual(
            fresh.recent_episodes(limit=1)[0].m_polarity, 0.3)

    def test_capacity_is_respected_on_load(self):
        """A save written under a larger capacity must not blow the
        buffer open -- keep the most recent `capacity`."""
        _, h = _live(12)
        small = Hippocampus(bus=EventBus(), capacity=4)
        small.load_dict(h.to_dict())
        self.assertEqual(small.buffer_size(), 4)
        ids = [e.episode_id for e in small.recent_episodes(limit=10)]
        self.assertEqual(max(ids), 12)

    def test_load_dict_tolerates_junk(self):
        """Never crash the restore path on a malformed save."""
        fresh = Hippocampus(bus=EventBus())
        for junk in (None, {}, [], 'nope',
                     {'episodes': 'nope'},
                     {'episodes': [None, 3, {'episode_id': 'x'}]},
                     {'next_id': 'x', 'episodes_formed': None}):
            fresh.load_dict(junk)
        self.assertEqual(fresh.buffer_size(), 0)

    def test_serialized_state_is_json_safe(self):
        """The canonical is gzipped JSON; a non-finite float or a
        set would break the whole save, not just this organ."""
        import json
        _, h = _live(5)
        json.dumps(h.to_dict(), allow_nan=False)


class TestBrainWiring(unittest.TestCase):

    def test_brain_save_carries_episodes(self):
        from seagi.body.engine import Engine
        from seagi.brain.runtime import Brain
        brain = Brain(engine=Engine())
        for k in range(3):
            brain.bus.publish(_percept(
                cycle=k, focals=[f'thing{k}', 'shared']))
        self.assertGreaterEqual(brain.hippocampus.buffer_size(), 3)

        d = brain.to_dict()
        self.assertIn('hippocampus', d)
        self.assertGreaterEqual(len(d['hippocampus']['episodes']), 3)

    def test_brain_restore_rehydrates_episodes(self):
        from seagi.body.engine import Engine
        from seagi.brain.runtime import Brain
        brain = Brain(engine=Engine())
        for k in range(3):
            brain.bus.publish(_percept(
                cycle=k, focals=[f'thing{k}', 'shared']))
        n = brain.hippocampus.buffer_size()
        d = brain.to_dict()

        fresh = Brain(engine=Engine())
        self.assertEqual(fresh.hippocampus.buffer_size(), 0)
        fresh.load_personality(d)
        self.assertEqual(fresh.hippocampus.buffer_size(), n)


if __name__ == '__main__':
    unittest.main()
