"""Hippocampus — episode formation + consolidation.

Brain analog: hippocampal formation + entorhinal cortex.  Binds
multi-feature snapshots into episodes at experience time.
During rest / DMN cycles, replays episodes in compressed form
and decides which graduate into long-term substrate (LTS) vs
which decay out of the hippocampal buffer.

Replay is weighted by emotional intensity + novelty, mirroring
the brain's bias to consolidate what mattered.

Phase 4a: FULL implementation.

What landed
-----------
1. EPISODE FORMATION
   Subscribes to ATTENDED_PERCEPT.  Each gate-passing percept
   becomes an Episode bound to the focals + raw_text + the
   M/I tag of the moment + novelty.  Episodes live in a
   bounded ring (hippocampal buffer, capacity 512).

2. REPLAY + CONSOLIDATION
   Subscribes to REFLECTION_FIRED (idle driver fires this).
   Scores each non-consolidated episode by:
       replay_weight = emotional_intensity * novelty * recency
   where emotional_intensity = max(m_polarity, i_polarity).
   Top-K (default 4) get consolidated:
       - For every pair of focals in the episode, request a
         co_occurs edge in LTS via SubstrateWriteQueuedEvent.
       - Emit EPISODE_CONSOLIDATED.

3. DECAY
   Bottom episodes whose replay_weight is below DECAY_FLOOR
   and whose age exceeds DECAY_AGE_CYCLES get marked decayed
   (still readable in buffer for diagnostics; EPISODE_DECAYED
   emitted so other capabilities can react).

Subscribes
----------
ATTENDED_PERCEPT     — form episode
REFLECTION_FIRED     — run consolidation pass
EPISODE_FORMED       — (self) bookkeeping count

Emits
-----
EPISODE_FORMED          — every new episode
EPISODE_CONSOLIDATED    — episodes that graduated to LTS
EPISODE_DECAYED         — episodes that dropped out
SUBSTRATE_WRITE_QUEUED  — co_occurs edges from consolidation
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from ..events import (
    EventKind, BrainEvent,
    AttendedPerceptEvent,
    ReflectionFiredEvent,
    EpisodeFormedEvent,
    EpisodeConsolidatedEvent,
    EpisodeDecayedEvent,
    SubstrateWriteQueuedEvent,
)
from ..bus import EventBus


HIPPOCAMPUS_BUFFER_CAPACITY = 512
DEFAULT_TOP_K_PER_REFLECTION = 4
DECAY_FLOOR = 0.10
DECAY_AGE_CYCLES = 500
RECENCY_TAU = 200.0
CO_OCCURS_RELATION = 'co_occurs'
CO_OCCURS_STRENGTH = 0.30


@dataclass
class Episode:
    """A single hippocampal episode.  Mutable — replay_weight
    updates over its life and the consolidated/decayed flags
    flip as reflection processes it."""
    episode_id: int
    cycle: int
    focals: List[str]
    raw_text: str
    salience: float
    m_polarity: float
    i_polarity: float
    novelty: float
    origin: str
    origin_detail: str
    replay_weight: float = 0.0
    consolidated: bool = False
    decayed: bool = False

    def emotional_intensity(self) -> float:
        """Felt-magnitude of the moment.  Either polarity
        counts — fear and joy both consolidate strongly in
        real brains."""
        return max(self.m_polarity, self.i_polarity)


class Hippocampus:
    """Episode formation + consolidation.  Continuous on
    attended percepts; batch on reflection."""

    SUBSCRIPTIONS = (
        EventKind.ATTENDED_PERCEPT,
        EventKind.REFLECTION_FIRED,
    )

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None,
                 capacity: int = HIPPOCAMPUS_BUFFER_CAPACITY,
                 top_k_per_reflection: int = (
                     DEFAULT_TOP_K_PER_REFLECTION),
                 tone_provider=None):
        self.bus = bus
        # HOW HE FELT WHEN HE ENCODED IT.  Returns (m_polarity,
        # i_polarity) or None.  Optional, so every existing caller and
        # every test constructs unchanged.
        self._tone_provider = tone_provider
        self.tagged_at_encoding = 0
        self._cycle_provider = cycle_provider or (lambda: 0)
        self.capacity = int(capacity)
        self.top_k = int(top_k_per_reflection)
        self._episodes: Deque[Episode] = deque(maxlen=self.capacity)
        self._next_id: int = 1
        # Diagnostics.
        self.episodes_formed: int = 0
        self.episodes_consolidated: int = 0
        self.episodes_decayed: int = 0
        self.edges_requested: int = 0

    # ---- event handling ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, AttendedPerceptEvent):
            self._on_attended(event, bus)
        elif isinstance(event, ReflectionFiredEvent):
            self._on_reflection(event, bus)

    # ---- formation ----

    def _on_attended(self,
                          ev: AttendedPerceptEvent,
                          bus: EventBus) -> None:
        """Each gate-passing percept becomes an Episode.  Don't
        filter further here — the gate already filtered."""
        if not ev.focals:
            return
        # M/I polarities — Phase 4a reads them off the salience
        # components (m_content / i_content) since AWM holds
        # the per-bubble polarity but isn't necessarily updated
        # yet when this fires.  Good enough proxy.
        # TAG AT ENCODING.  The percept fields are hard-coded 0.0 by
        # `world_actor`, and they must STAY 0.0: `chemistry.py:939/952`
        # fire anomaly_spike / mattering on any non-zero content, so
        # filling them would self-excite every step.  Read the live
        # chemistry here instead, where the tag is actually consumed.
        # An event carrying real content of its own still wins.
        _mp = float(getattr(ev, "m_content", 0.0) or 0.0)
        _ip = float(getattr(ev, "i_content", 0.0) or 0.0)
        if _mp <= 0.0 and _ip <= 0.0 and self._tone_provider is not None:
            try:
                _t = self._tone_provider() or (0.0, 0.0)
                _mp, _ip = float(_t[0] or 0.0), float(_t[1] or 0.0)
                if _mp > 0.0 or _ip > 0.0:
                    self.tagged_at_encoding += 1
            except Exception:
                _mp, _ip = 0.0, 0.0
        ep = Episode(
            episode_id=self._next_id,
            cycle=ev.cycle,
            focals=list(ev.focals),
            raw_text=ev.raw_text[:240],   # cap stored text
            salience=ev.salience,
            m_polarity=_mp,
            i_polarity=_ip,
            novelty=ev.novelty,
            origin=ev.origin,
            origin_detail=ev.origin_detail,
        )
        self._next_id += 1
        self._episodes.append(ep)
        self.episodes_formed += 1
        # Emit the EPISODE_FORMED event so dmDMN / vmDMN can
        # react opportunistically (peer-origin → ToM update;
        # internal-origin self-observation → autobiographical
        # signal).
        bus.publish(EpisodeFormedEvent(
            kind=EventKind.EPISODE_FORMED,
            cycle=ev.cycle,
            timestamp=time.time(),
            source_capability='hippocampus',
            origin=ev.origin,
            origin_detail=ev.origin_detail,
            episode_id=ep.episode_id,
            focals=list(ep.focals),
            raw_text=ep.raw_text,
            salience=ep.salience,
            m_polarity=ep.m_polarity,
            i_polarity=ep.i_polarity,
            novelty=ep.novelty,
        ))

    # ---- consolidation on reflection ----

    def _on_reflection(self,
                            ev: ReflectionFiredEvent,
                            bus: EventBus) -> None:
        """Score episodes, consolidate top-K, decay the rest
        that have aged past DECAY_AGE_CYCLES."""
        now = ev.cycle if ev.cycle else self._cycle_provider()
        # Score every non-consolidated, non-decayed episode.
        scored: List[Tuple[float, Episode]] = []
        for ep in self._episodes:
            if ep.consolidated or ep.decayed:
                continue
            ep.replay_weight = self._score(ep, now)
            scored.append((ep.replay_weight, ep))
        if not scored:
            return
        scored.sort(key=lambda t: -t[0])

        # Consolidate the top-K above a small floor.
        top = scored[:self.top_k]
        for weight, ep in top:
            if weight < DECAY_FLOOR:
                break
            self._consolidate(ep, bus)

        # Decay aged-out low-weight episodes.
        for weight, ep in scored:
            age = now - ep.cycle
            if (weight < DECAY_FLOOR
                    and age > DECAY_AGE_CYCLES
                    and not ep.consolidated):
                self._decay(ep, bus, age=age)

    def _score(self, ep: Episode, now: int) -> float:
        emotional = ep.emotional_intensity()
        novelty = ep.novelty
        age = max(0, now - ep.cycle)
        recency = pow(2.718281828, -(age / RECENCY_TAU))
        # Salience-at-formation acts as a small floor — even
        # low-emotion episodes that were highly salient at
        # the time deserve weight.
        salience_floor = 0.2 * ep.salience
        return (emotional * 0.5
                  + novelty * 0.3
                  + salience_floor) * recency

    def _consolidate(self,
                          ep: Episode,
                          bus: EventBus) -> None:
        """Promote episode content into LTS via substrate writes.

        FOCAL-CENTRIC BINDING (brain-correct sparse coding):
        bind the primary focal (`focals[0]`, what attention
        concentrated on) to each OTHER focal in the episode.
        n − 1 edges per episode instead of n × (n − 1) / 2.

        Why this matters: v1 and early-v2 wrote every-pair-to-
        every-pair, producing up to 120 co_occurs writes per
        16-focal episode.  Over many reflections of live
        operation the substrate becomes co_occurs noise — every
        concept ends up co_occurs-related to nearly every other
        it's ever shared an episode with.  At scale this drowns
        out signal.

        Real hippocampus doesn't bind every-pair-to-every-pair
        either — pattern completion happens around the salient
        focal.  Secondary co-occurring features bind weakly to
        each other; strongly to the focus.  Focal-centric
        approximates that.

        Strength stays scaled by emotional_intensity so highly-
        felt episodes still carry more weight per edge — they
        just don't multiply the EDGE COUNT.
        """
        n_requested = 0
        strength = min(1.0, CO_OCCURS_STRENGTH * (
            1.0 + ep.emotional_intensity()))
        if ep.focals:
            primary = ep.focals[0]
            if primary:
                for other in ep.focals[1:]:
                    if not other or other == primary:
                        continue
                    bus.publish(SubstrateWriteQueuedEvent(
                        kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                        cycle=ep.cycle,
                        timestamp=time.time(),
                        source_capability='hippocampus',
                        origin='internal',
                        origin_detail=(
                            f'episode:{ep.episode_id}'),
                        subject=primary,
                        relation=CO_OCCURS_RELATION,
                        object=other,
                        strength=strength,
                        write_reason='consolidation',
                    ))
                    n_requested += 1
        self.edges_requested += n_requested
        ep.consolidated = True
        # audit #12 (2026-06-04): only COUNT a consolidation and publish
        # the replay event when real co_occurs edges were written.  A
        # single-focal episode (the dominant dry-reverie input from
        # inner_voice / idle_motivation) writes zero edges — counting it
        # inflated episodes_consolidated and fed vmDMN empty replay.  The
        # episode is still marked processed (no re-attempt); it just is
        # not a consolidation.  Deeper fix (widen reverie beyond single
        # focals) is tracked under audit #1, not here.
        if n_requested > 0:
            self.episodes_consolidated += 1
            bus.publish(EpisodeConsolidatedEvent(
                kind=EventKind.EPISODE_CONSOLIDATED,
                cycle=ep.cycle,
                timestamp=time.time(),
                source_capability='hippocampus',
                origin=ep.origin,
                origin_detail=ep.origin_detail,
                episode_id=ep.episode_id,
                focals=list(ep.focals),
                n_edges_strengthened=n_requested,
                replay_weight=ep.replay_weight,
            ))

    def _decay(self,
                 ep: Episode,
                 bus: EventBus,
                 age: int) -> None:
        ep.decayed = True
        self.episodes_decayed += 1
        bus.publish(EpisodeDecayedEvent(
            kind=EventKind.EPISODE_DECAYED,
            cycle=self._cycle_provider(),
            timestamp=time.time(),
            source_capability='hippocampus',
            origin='internal',
            origin_detail=f'episode:{ep.episode_id}',
            episode_id=ep.episode_id,
            age_cycles=age,
        ))

    # ---- queries ----

    def recent_episodes(self,
                                limit: int = 10
                                ) -> List[Episode]:
        return list(self._episodes)[-limit:][::-1]

    def consolidated_count(self) -> int:
        return sum(1 for e in self._episodes if e.consolidated)

    def buffer_size(self) -> int:
        return len(self._episodes)

    # ---- persistence ----

    # EPISODIC MEMORY SURVIVES RESTART (2026-08-18).  The buffer was
    # built fresh in __init__ and never serialized, while world_route
    # / world_svalue / world_rules all persisted -- procedural memory
    # survived every restart and felt memory never had.  He kept the
    # routes and lost why they mattered.

    @staticmethod
    def _finite(x, default: float = 0.0) -> float:
        """The canonical is JSON; inf/nan would break the whole save."""
        try:
            v = float(x)
        except (TypeError, ValueError):
            return default
        if v != v or v in (float('inf'), float('-inf')):
            return default
        return v

    def _episode_to_dict(self, ep: Episode) -> Dict[str, Any]:
        return {
            'episode_id': int(ep.episode_id),
            'cycle': int(ep.cycle),
            'focals': [str(f) for f in ep.focals],
            'raw_text': str(ep.raw_text),
            'salience': self._finite(ep.salience),
            'm_polarity': self._finite(ep.m_polarity),
            'i_polarity': self._finite(ep.i_polarity),
            'novelty': self._finite(ep.novelty),
            'origin': str(ep.origin),
            'origin_detail': str(ep.origin_detail),
            'replay_weight': self._finite(ep.replay_weight),
            'consolidated': bool(ep.consolidated),
            'decayed': bool(ep.decayed),
        }

    def _episode_from_dict(self, d: Dict[str, Any]) -> Optional[Episode]:
        if not isinstance(d, dict):
            return None
        try:
            eid = int(d['episode_id'])
            cyc = int(d.get('cycle', 0))
        except (TypeError, ValueError, KeyError):
            return None
        focals = d.get('focals')
        if not isinstance(focals, (list, tuple)):
            focals = []
        return Episode(
            episode_id=eid,
            cycle=cyc,
            focals=[str(f) for f in focals],
            raw_text=str(d.get('raw_text', ''))[:240],
            salience=self._finite(d.get('salience')),
            m_polarity=self._finite(d.get('m_polarity')),
            i_polarity=self._finite(d.get('i_polarity')),
            novelty=self._finite(d.get('novelty')),
            origin=str(d.get('origin', '')),
            origin_detail=str(d.get('origin_detail', '')),
            replay_weight=self._finite(d.get('replay_weight')),
            consolidated=bool(d.get('consolidated', False)),
            decayed=bool(d.get('decayed', False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'episodes': [self._episode_to_dict(e)
                         for e in self._episodes],
            'next_id': int(self._next_id),
            'episodes_formed': int(self.episodes_formed),
            'episodes_consolidated': int(self.episodes_consolidated),
            'episodes_decayed': int(self.episodes_decayed),
            'edges_requested': int(self.edges_requested),
            'tagged_at_encoding': int(
                getattr(self, 'tagged_at_encoding', 0)),
        }

    def load_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        def _int(key: str, current: int) -> int:
            try:
                return int(state[key])
            except (KeyError, TypeError, ValueError):
                return current

        eps = state.get('episodes')
        if isinstance(eps, (list, tuple)):
            # Keep the MOST RECENT `capacity` -- a save written under a
            # larger buffer must not blow this one open.
            restored = deque(maxlen=self.capacity)
            for ed in list(eps)[-self.capacity:]:
                ep = self._episode_from_dict(ed)
                if ep is not None:
                    restored.append(ep)
            self._episodes = restored
        # Ids must never collide: two different memories with the same
        # name would corrupt consolidation bookkeeping.
        highest = max((e.episode_id for e in self._episodes), default=0)
        self._next_id = max(_int('next_id', self._next_id), highest + 1)
        self.episodes_formed = _int('episodes_formed',
                                    self.episodes_formed)
        self.episodes_consolidated = _int('episodes_consolidated',
                                          self.episodes_consolidated)
        self.episodes_decayed = _int('episodes_decayed',
                                     self.episodes_decayed)
        self.edges_requested = _int('edges_requested',
                                    self.edges_requested)
        self.tagged_at_encoding = _int('tagged_at_encoding',
                                       self.tagged_at_encoding)

    # ---- diagnostics ----

    def stats(self) -> Dict[str, int]:
        return {
            'episodes_formed': self.episodes_formed,
            'tagged_at_encoding': getattr(self, 'tagged_at_encoding', 0),
            'episodes_consolidated': self.episodes_consolidated,
            'episodes_decayed': self.episodes_decayed,
            'edges_requested': self.edges_requested,
            'buffer_size': self.buffer_size(),
        }
