"""Default Mode Network — reflection during quiet.

Brain analog: the brain's reflective network — most active
when NOT engaged with external task.  Two sub-networks:

    DORSOMEDIAL (dmDMN):
        Social cognition.  Theory of Mind.  Refines models of
        peers from accumulated peer-origin interactions.

    VENTROMEDIAL (vmDMN):
        Self-reference.  Autobiographical replay.  Consolidates
        "what kind of agent am I becoming" from recent
        consolidated episodes + AWM's high-crystallization
        bubbles.

Both activate on REFLECTION_FIRED events.  Brain's idle driver
(in runtime.py) fires REFLECTION_FIRED when:
    - N idle cycles elapse without peer input  (idle_timer)
    - Chemistry global state spikes              (chemistry_spike)
    - Manually triggered                          (manual / tests)

Phase 4a: FULL implementation.

Subscribes
----------
dmDMN: ATTENDED_PERCEPT (peer-origin only), REFLECTION_FIRED,
        EPISODE_FORMED (peer-origin only)
vmDMN: REFLECTION_FIRED, EPISODE_CONSOLIDATED

Emits
-----
dmDMN: SUBSTRATE_WRITE_QUEUED (peer-association edges)
vmDMN: THOUGHT_PRODUCED (reflective self-narrative),
        SUBSTRATE_WRITE_QUEUED (self-concept anchors when
                                  crystallization crosses
                                  threshold)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..events import (
    EventKind, BrainEvent,
    AttendedPerceptEvent,
    EpisodeFormedEvent,
    EpisodeConsolidatedEvent,
    ReflectionFiredEvent,
    SubstrateWriteQueuedEvent,
    ThoughtProducedEvent,
)
from ..bus import EventBus


# Minimum peer interaction count before dmDMN considers a peer
# model "worth committing to substrate" during reflection.
PEER_COMMIT_THRESHOLD = 2

# Per-peer topic association decay.  Each reflection halves
# old peer-topic weights so dmDMN tracks recent rather than
# all-time.
DMDMN_TOPIC_DECAY = 0.85

# vmDMN: how many consolidated episodes feed into a single
# self-narrative reflection.
VMDMN_REPLAY_DEPTH = 6


# ---------------------------------------------------------------
# dmDMN — social cognition / ToM
# ---------------------------------------------------------------


@dataclass
class PeerModel:
    """The agent's running model of one peer (by origin_detail
    string — e.g. 'harald').  Updated on every peer-origin
    attended percept."""
    peer_id: str
    interactions: int = 0
    topic_weights: Dict[str, float] = field(default_factory=dict)
    last_seen_cycle: int = 0
    last_focals: List[str] = field(default_factory=list)

    def observe(self,
                  cycle: int,
                  focals: List[str],
                  salience: float) -> None:
        self.interactions += 1
        self.last_seen_cycle = cycle
        self.last_focals = list(focals)
        weight = max(0.05, salience)
        for f in focals:
            self.topic_weights[f] = self.topic_weights.get(
                f, 0.0) + weight


class DorsomedialDMN:
    """Social cognition / Theory of Mind layer."""

    SUBSCRIPTIONS = (
        EventKind.ATTENDED_PERCEPT,
        EventKind.REFLECTION_FIRED,
        EventKind.EPISODE_FORMED,
    )

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._peers: Dict[str, PeerModel] = {}
        self.peer_observations: int = 0
        self.tom_writes: int = 0

    # ---- public ----

    def model_for(self, peer_id: str) -> Optional[PeerModel]:
        return self._peers.get(peer_id)

    def peers_known(self) -> List[str]:
        return list(self._peers.keys())

    # ---- event handler ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, AttendedPerceptEvent):
            if event.origin == 'peer' and event.focals:
                self._observe(event)
        elif isinstance(event, EpisodeFormedEvent):
            if event.origin == 'peer' and event.focals:
                # Reinforce peer model from formed episodes.
                pid = event.origin_detail or 'anonymous'
                model = self._peers.setdefault(
                    pid, PeerModel(peer_id=pid))
                model.observe(
                    event.cycle, event.focals, event.salience)
                self.peer_observations += 1
        elif isinstance(event, ReflectionFiredEvent):
            if event.reflection_kind in ('general', 'social'):
                self._reflect(event, bus)

    def _observe(self, ev: AttendedPerceptEvent) -> None:
        pid = ev.origin_detail or 'anonymous'
        model = self._peers.setdefault(
            pid, PeerModel(peer_id=pid))
        model.observe(ev.cycle, ev.focals, ev.salience)
        self.peer_observations += 1

    def _reflect(self,
                       ev: ReflectionFiredEvent,
                       bus: EventBus) -> None:
        """Commit strong peer-topic associations to substrate.
        Decay topic weights so the model tracks recent
        rather than all-time."""
        cycle = ev.cycle if ev.cycle else self._cycle_provider()
        for pid, model in self._peers.items():
            if model.interactions < PEER_COMMIT_THRESHOLD:
                continue
            # Find the strongest topics for this peer.
            if not model.topic_weights:
                continue
            top = sorted(
                model.topic_weights.items(),
                key=lambda kv: -kv[1])[:3]
            peer_node = f'peer:{pid}'
            for topic, w in top:
                if w < 0.1:
                    continue
                strength = min(0.9, 0.3 + 0.1 * w)
                try:
                    bus.publish(SubstrateWriteQueuedEvent(
                        kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                        cycle=cycle,
                        timestamp=time.time(),
                        source_capability='dmdmn',
                        origin='internal',
                        origin_detail=pid,
                        subject=peer_node,
                        relation='talks_about',
                        object=topic,
                        strength=strength,
                        write_reason='tom_update',
                    ))
                    self.tom_writes += 1
                except Exception:
                    pass
            # Decay topic weights so we track recent rather
            # than all-time.
            for topic in list(model.topic_weights.keys()):
                model.topic_weights[topic] *= DMDMN_TOPIC_DECAY
                if model.topic_weights[topic] < 0.02:
                    del model.topic_weights[topic]

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'peers_known': len(self._peers),
            'peer_observations': self.peer_observations,
            'tom_writes': self.tom_writes,
            'peer_sample': list(self._peers.keys())[:5],
        }


# ---------------------------------------------------------------
# vmDMN — self-reference / autobiographical
# ---------------------------------------------------------------


class VentromedialDMN:
    """Self-reference reflection.  Replays recent consolidated
    episodes + reads AWM crystallization to produce a self-
    narrative thought."""

    SUBSCRIPTIONS = (
        EventKind.REFLECTION_FIRED,
        EventKind.EPISODE_CONSOLIDATED,
    )

    def __init__(self,
                 bus: EventBus,
                 awm_provider: Callable,
                 hippocampus_provider: Callable,
                 chemistry_provider: Optional[Callable] = None,
                 insula_provider: Optional[Callable] = None,
                 cycle_provider: Optional[Callable] = None,
                 self_model_provider: Optional[Callable] = None):
        self.bus = bus
        self._awm_provider = awm_provider
        self._hippocampus_provider = hippocampus_provider
        self._chemistry_provider = chemistry_provider
        self._insula_provider = insula_provider
        self._cycle_provider = cycle_provider or (lambda: 0)
        # Phase G.5 (2026-05-16): self_model_provider lets vmDMN
        # feed self-edges it identifies in its own narrative into
        # the SelfModel registry.  Identity grows from internal
        # reflection, not just peer assertions.
        self._self_model_provider = self_model_provider
        # Most-recent self-narrative.  Speech can read this
        # without listening on the bus.
        self.last_narrative: str = ''
        self.last_narrative_cycle: int = 0
        # Cache of last consolidated focals seen — used to
        # detect "what kind of agent am I becoming" shifts.
        self._recent_consolidated: List[List[str]] = []
        self.reflections_fired: int = 0
        self.narratives_emitted: int = 0
        # Phase G.5: count of self-edges fed into SelfModel.
        self.self_edges_recorded: int = 0

    # ---- event handler ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, EpisodeConsolidatedEvent):
            self._recent_consolidated.append(list(event.focals))
            if len(self._recent_consolidated) > 32:
                self._recent_consolidated.pop(0)
        elif isinstance(event, ReflectionFiredEvent):
            if event.reflection_kind in ('general', 'autobiographical'):
                self._reflect(event, bus)

    def _reflect(self,
                       ev: ReflectionFiredEvent,
                       bus: EventBus) -> None:
        """Compose a reflective self-narrative.  Combines:
          1. Recent consolidated-episode focal frequency
          2. AWM crystallization peaks
          3. Insula band (felt body state colors the framing)
        Emits a THOUGHT_PRODUCED (method='reflection') so the
        next chat / speech layer can surface it as voice."""
        cycle = ev.cycle if ev.cycle else self._cycle_provider()
        self.reflections_fired += 1

        focal_freq: Dict[str, int] = defaultdict(int)
        replay_slice = self._recent_consolidated[-VMDMN_REPLAY_DEPTH:]
        for focals in replay_slice:
            for f in focals:
                focal_freq[f] += 1

        # Top AWM crystallization anchors (the agent's most
        # settled bubbles).
        awm = self._awm_provider()
        crystal_anchors: List[Tuple[str, float]] = []
        if awm is not None:
            for name in awm.active_concepts():
                entry = awm.get(name)
                if entry is None:
                    continue
                crystal_anchors.append(
                    (name, entry.bubble.crystallization))
            crystal_anchors.sort(key=lambda t: -t[1])
            crystal_anchors = crystal_anchors[:3]

        # Body framing from insula if available.
        body_frame = ''
        if self._insula_provider is not None:
            try:
                insula = self._insula_provider()
                if insula is not None:
                    felt = insula.felt_state()
                    body_frame = felt.get('narrative', '') or ''
            except Exception:
                pass

        narrative = self._compose_narrative(
            focal_freq, crystal_anchors, body_frame, replay_slice)
        if not narrative:
            return
        self.last_narrative = narrative
        self.last_narrative_cycle = cycle

        # Phase G.5 (2026-05-16): feed self-edges from the
        # narrative's ingredients into SelfModel.  Recurrent
        # focals (focal_freq) → 'attend_to' self-edges;
        # crystallized anchors → 'anchor_on' self-edges.
        # Source='narrative' distinguishes from peer-asserted
        # identity ('peer:alice') in the registry.
        self._record_self_edges_from_narrative(
            focal_freq, crystal_anchors, cycle)

        # Surface as a THOUGHT_PRODUCED so cortical/speech layers
        # can pick it up.  origin='internal' / detail='reflection'
        # so source monitoring can distinguish reflective from
        # peer-driven thoughts.
        try:
            bus.publish(ThoughtProducedEvent(
                kind=EventKind.THOUGHT_PRODUCED,
                cycle=cycle,
                timestamp=time.time(),
                source_capability='vmdmn',
                origin='internal',
                origin_detail='reflection',
                focal='self',
                relation='reflects_on',
                target=(crystal_anchors[0][0]
                          if crystal_anchors else ''),
                confidence=0.6,
                method='reflection',
                text=narrative,
                thin_substrate=False,
            ))
            self.narratives_emitted += 1
        except Exception:
            pass

        # If a focal has dominated recent consolidation strongly,
        # write a self_attends_to anchor.  Note: substrate-side
        # relation uses 3rd-person 'attends_to' (substrate stores
        # facts ABOUT the agent: "self attends_to X").  SelfModel
        # edges below use 1st-person base-form 'attend_to' for
        # clean introspective rendering ("I attend to X").
        if focal_freq:
            top_focal, count = max(
                focal_freq.items(), key=lambda kv: kv[1])
            if count >= 3:
                try:
                    bus.publish(SubstrateWriteQueuedEvent(
                        kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                        cycle=cycle,
                        timestamp=time.time(),
                        source_capability='vmdmn',
                        origin='internal',
                        origin_detail='reflection',
                        subject='self',
                        relation='attends_to',
                        object=top_focal,
                        strength=min(0.9, 0.3 + 0.1 * count),
                        write_reason='self_anchor',
                    ))
                except Exception:
                    pass

    def _record_self_edges_from_narrative(self,
                                                 focal_freq: Dict[str, int],
                                                 crystal_anchors: List[Tuple[str, float]],
                                                 cycle: int) -> None:
        """Phase G.5 (2026-05-16): extract self-edges from the
        narrative-ingredient lists and feed them to SelfModel.

        Heuristic:
          - Recurrent focals (focal_freq, top 3 by count) become
             self-edges with relation 'attend_to'.  These are
             concepts the agent has been returning to.
          - Crystallized anchors (already top 3 by
             crystallization) become 'anchor_on' self-edges.
             These are the settled focals the agent is most
             stable around.

        Both source='narrative' (vs 'peer:X' for assertions).
        The SelfModel's own dedup + crystallization handles
        repetition; calling record() many times for the same
        (relation, focal) just reinforces.

        Returns silently if no SelfModel is wired or no anchors
        worth recording.
        """
        if self._self_model_provider is None:
            return
        try:
            model = self._self_model_provider()
        except Exception:
            return
        if model is None:
            return
        # attend_to: top 3 recurrent focals (focals that came
        # back across consolidated episodes).
        if focal_freq:
            top = sorted(
                focal_freq.items(),
                key=lambda kv: -kv[1])[:3]
            for name, _count in top:
                if name and name != 'self':
                    try:
                        model.record(
                            relation='attend_to',
                            obj=name,
                            cycle=cycle,
                            strength=0.5,
                            source='narrative')
                        self.self_edges_recorded += 1
                    except Exception:
                        pass
        # anchor_on: crystallized AWM bubbles — the agent's
        # current settled ground.
        for name, _cryst in crystal_anchors:
            if name and name != 'self':
                try:
                    model.record(
                        relation='anchor_on',
                        obj=name,
                        cycle=cycle,
                        strength=0.5,
                        source='narrative')
                    self.self_edges_recorded += 1
                except Exception:
                    pass

    def _compose_narrative(self,
                                focal_freq: Dict[str, int],
                                crystal_anchors: List[Tuple[str, float]],
                                body_frame: str,
                                replay_slice: List[List[str]]
                                ) -> str:
        """Build a first-person reflective sentence from the
        ingredients gathered.  Honest about thin material —
        no narrative if nothing to say."""
        parts: List[str] = []
        if focal_freq:
            top = sorted(
                focal_freq.items(), key=lambda kv: -kv[1])[:3]
            names = ', '.join(n for n, _ in top)
            parts.append(
                f'Lately I have been returning to {names}.')
        elif not crystal_anchors and not body_frame:
            return ''  # nothing reflective to say
        if crystal_anchors:
            anchor_names = ', '.join(n for n, _ in crystal_anchors)
            parts.append(
                f'My settled ground holds {anchor_names}.')
        if body_frame:
            parts.append(body_frame)
        if len(replay_slice) >= 3 and focal_freq:
            parts.append(
                f'This is the pattern I see in myself across '
                f'{len(replay_slice)} recent episodes.')
        return ' '.join(parts).strip()

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'reflections_fired': self.reflections_fired,
            'narratives_emitted': self.narratives_emitted,
            'recent_consolidated_kept': len(
                self._recent_consolidated),
            'last_narrative_cycle': self.last_narrative_cycle,
            'last_narrative': self.last_narrative[:120],
        }
