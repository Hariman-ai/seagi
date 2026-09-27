"""Forager — autonomous external input (V1→V2 port, Tier 1).

Brain analog: the drive to seek novel stimulation — an organism
that stops sampling its environment starves.  V2's daemon ran 3
days of pure reverie over a frozen 47k-concept substrate and never
ate.  Without new information no architecture can grow toward AGI;
the forager is the input channel that lets SEAGI keep reading even
with no human in the loop.

Ported from V1 `forager.py`, with three deliberate divergences:

1. **In-band, not threaded.** V1 ran on its own thread with a
   shared lock.  V2 capabilities tick inside `Brain.tick()` under
   the daemon's single `brain_lock`; a background thread would
   race the tick loop.  So `tick()` is called once per brain tick
   and does at most one pick→ingest→archive on a pace gate.

2. **Sleep-gated.** Foraging while asleep would inject fresh
   provisional edges mid-consolidation and corrupt the Step 0
   `debt_full_scale` calibration.  The forager pauses while the
   sleep regulator reports sleep.  This gives the natural duty
   cycle the 3-day-no-sleep bug was missing: eat awake → accrue
   substrate-write debt → sleep → consolidate → wake → eat.

3. **Feeds through the existing intake path.** It calls
   `brain.intake(text, origin='forager', origin_detail=...)`,
   so every percept flows sensory → gate → AWM → chemistry →
   cortical and writes provisional SVO edges that face Phase S.
   The forager never bypasses the gate and emits no new event type.

Information-request loop (the "wanting to know more" signal):
subscribes to THOUGHT_PRODUCED filtered on `thin_substrate=True`;
the focal of a thin-substrate thought is pushed into
`information_requests`, and the next reading-list pick prefers a
file whose name or first bytes mention a requested concept.  So
the same epistemic-gap signal that drives curiosity also shapes
what gets read next (rule 4 — capabilities compound).

File lifecycle (disk-based, polling, zero deps):
  inbox/foo.txt        → ingested → archive/YYYY-MM-DD/foo.txt
  reading_list/bar.txt → ingested → stays put, recorded in
                          read_manifest (cycle read)

Per [[seagi-chemistry-never-fully-dissolves]] the archive is
RE-INGESTABLE: a fully-read reading_list does not hard-stop
forever — `reconsider_all()` clears the manifest so faded
material can be re-encountered.  Smallest version idles when all
read; re-engagement is an explicit call, not automatic.

Deferred (per [[feedback_seagi_vision_proportion]], one channel at
a time): RSS/wiki fetching (V1 `rss_fetcher`/`wiki_extractor`) —
different permission boundary, and Wikipedia overlaps the tool-use
channel.

Cross-refs: [[project_seagi_v1_port_tier1_wiring]] (plan),
[[project_seagi_step0_wiring_plan]] (sleep/debt economy).
"""

from __future__ import annotations

import datetime
import os
import shutil
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from ..events import (
    EventKind, BrainEvent, ThoughtProducedEvent,
    SubstrateWriteQueuedEvent, ChemistryEvent,
)
from ..bus import EventBus
from seagi.ingestion.lemmatizer import lemmatize as _lemmatize


# Pace: at most one document per FORAGE_INTERVAL_TICKS.  DECLARED
# MEASUREMENT DEBT (auditor): the derivation is a structural
# placeholder ("one document per maintenance-window fraction so a
# doc can be perceived + partially consolidated before the next
# arrives"), NOT a causally-grounded cadence.  Re-derive
# empirically on the daemon (confirm real-world rate at 5Hz)
# before treating as final.
#
# The implied edge-prune interval (the conceptual EDGE_PRUNE_INTERVAL
# referenced in substrate.py comments) is the number of cycles a
# provisional edge survives unreinforced before hitting the floor:
# EDGE_PRUNE_FLOOR / EDGE_STRENGTH_DECAY_PER_CYCLE.  Both are real
# substrate constants, so this traces rather than being a bare
# magnitude.
from ..capabilities.awm import AWM_CAPACITY_COLD_START_FLOOR
from seagi.core.substrate import (
    EDGE_PRUNE_FLOOR, EDGE_STRENGTH_DECAY_PER_CYCLE)

_IMPLIED_PRUNE_INTERVAL = int(
    EDGE_PRUNE_FLOOR / EDGE_STRENGTH_DECAY_PER_CYCLE)   # 0.02/1e-5 = 2000
FORAGE_INTERVAL_TICKS = (
    _IMPLIED_PRUNE_INTERVAL // AWM_CAPACITY_COLD_START_FLOOR)  # 2000//3 = 666

# Cap on the information-request queue (V1 parity).
REQUEST_CAP = 32


def _list_text_files(path: str) -> List[str]:
    """All *.txt files directly under `path`, oldest-first."""
    if not os.path.isdir(path):
        return []
    entries = []
    for name in os.listdir(path):
        if not name.lower().endswith('.txt'):
            continue
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            continue
        entries.append((mtime, full))
    entries.sort()
    return [p for _, p in entries]


class Forager:
    """Autonomous reader.  Ticks in-band; sleep-gated; feeds the
    existing intake path."""

    SUBSCRIPTIONS = (
        EventKind.THOUGHT_PRODUCED,
        # Satisfaction-clearance loop: when forager-origin ingest
        # writes an edge whose subject matches a pending information
        # request, the agent FELT satisfaction (insight chemistry on
        # the focal) and the question clears from the channel — so
        # the agent can move on to the next question.  Curiosity-as-
        # drive persists; this only retires the specific resolved
        # question.
        EventKind.SUBSTRATE_WRITE_QUEUED,
    )

    def __init__(self,
                 bus: EventBus,
                 intake_fn: Callable,
                 cycle_provider: Optional[Callable] = None,
                 is_asleep_provider: Optional[Callable] = None,
                 root: Optional[str] = None):
        """
        intake_fn: Brain.intake — (payload, modality, origin,
            origin_detail) -> None.  The universal input entry.
        is_asleep_provider: () -> bool.  Forager pauses while True.
        root: directory holding inbox/ reading_list/ archive/.
            Defaults to CWD.
        """
        self.bus = bus
        self._intake_fn = intake_fn
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._is_asleep_provider = is_asleep_provider
        # INERT BY DEFAULT.  root=None → the forager does nothing
        # (tick no-ops).  This is critical: a fresh Brain() must NOT
        # silently forage from its CWD — every test instantiates
        # Brain, and reading stray .txt files from the repo's
        # reading_list/ contaminates unrelated tests (observed
        # 2026-05-28: it perturbed a chemistry-distinguishing test).
        # The daemon calls activate(root) at boot to turn it on;
        # tests pass an explicit temp root to exercise it.
        self.paths: Optional[Dict[str, str]] = (
            self._resolve_paths(root) if root is not None else None)

        # Persisted state.
        self.read_manifest: Dict[str, int] = {}     # filename → cycle
        self.last_forage_cycle: int = -1
        self.information_requests: Deque[str] = deque(maxlen=REQUEST_CAP)

        # Diagnostics (not persisted).
        self.ingest_count: int = 0
        self.last_ingest_summary: str = ''
        self._last_forage_tick: int = -10**9
        # Satisfaction-loop diagnostics (not persisted).  Insight
        # fires once per resolved request; the question then leaves
        # the channel.  See handle() / SUBSTRATE_WRITE_QUEUED branch.
        self.resolutions_fired: int = 0
        self.last_resolved: str = ''

    # ---- dir setup ----

    @staticmethod
    def _resolve_paths(root: Optional[str]) -> Dict[str, str]:
        r = root or os.getcwd()
        return {
            'root': r,
            'inbox': os.path.join(r, 'inbox'),
            'reading_list': os.path.join(r, 'reading_list'),
            'archive': os.path.join(r, 'archive'),
        }

    def activate(self, root: str) -> None:
        """Turn the forager ON, rooted at `root`, creating its
        inbox/ reading_list/ archive/ dirs.  Called by the daemon at
        boot.  Until this (or an explicit root at construction) the
        forager is inert."""
        self.paths = self._resolve_paths(root)
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """Create inbox/ reading_list/ archive/ under the resolved
        root.  No-op if inert (paths is None)."""
        if self.paths is None:
            return
        for key in ('inbox', 'reading_list', 'archive'):
            try:
                os.makedirs(self.paths[key], exist_ok=True)
            except OSError:
                pass

    # ---- event handler (information-request loop) ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, ThoughtProducedEvent):
            if event.thin_substrate and event.focal:
                self.request_information(event.focal)
        elif isinstance(event, SubstrateWriteQueuedEvent):
            # Restrict the satisfaction loop to EXTERNAL content
            # (forager-origin ingest).  Self-percepts (inner_voice)
            # and reverie-internal writes must not clear questions
            # the agent posed honestly — that would be self-answering
            # gaming.
            if event.origin != 'forager':
                return
            subj = getattr(event, 'subject', '') or ''
            key = _lemmatize(subj.strip().lower()) if subj else ''
            if not key or key not in self.information_requests:
                return
            # Match.  The agent is about to FEEL the answer to a
            # question it had been asking.
            try:
                self.information_requests.remove(key)
            except ValueError:
                pass
            self.resolutions_fired += 1
            self.last_resolved = subj
            # Fire `insight` — the dormant chemistry event finally
            # earning its keep: the felt pulse that marks
            # request→resolution.  Targets the resolved focal so its
            # bubble imprints the satisfaction tied to THIS concept
            # (not a global mood shift).  Doctrine: curiosity-as-
            # drive persists (idle_motivation, cortical thin-checks
            # keep firing for new questions); satisfaction is per-
            # question, not a curiosity-off switch.
            import time as _t
            try:
                bus.publish(ChemistryEvent(
                    kind=EventKind.CHEMISTRY_FIRE,
                    cycle=event.cycle,
                    timestamp=_t.time(),
                    source_capability='forager',
                    origin='internal',
                    origin_detail=f'resolved:{key}',
                    chemistry_kind='insight',
                    magnitude=1.0,
                    target_concepts=[subj]))
            except Exception:
                pass

    def request_information(self, concept: str) -> None:
        """Push a concept the agent couldn't think well about onto
        the want-to-read queue.  Move-to-front dedup; capped.

        The key is LEMMATIZED so that requests and ingest-subject
        matches use one canonical form — without this, a request for
        'king' silently misses a write of 'kings', and the loop never
        closes.  (The fix verified by the curiosity-loop audit, D2.)
        """
        if not concept or not isinstance(concept, str):
            return
        c = _lemmatize(concept.strip().lower())
        if len(c) < 3:
            return
        try:
            # deque has no remove-by-value cheap path that preserves
            # order beyond .remove; it's O(n) but n<=32.
            if c in self.information_requests:
                self.information_requests.remove(c)
        except ValueError:
            pass
        self.information_requests.append(c)

    # ---- per-tick ----

    def tick(self) -> Optional[Dict[str, Any]]:
        """One pace-gated, sleep-gated pick→ingest→archive pass.
        Returns a summary dict when a file was consumed, else None."""
        # Inert unless activated with a root.
        if self.paths is None:
            return None
        # Sleep gate: don't read during consolidation.
        if self._is_asleep_provider is not None:
            try:
                if self._is_asleep_provider():
                    return None
            except Exception:
                pass
        cycle = int(self._cycle_provider())
        # Pace gate.
        if cycle - self._last_forage_tick < FORAGE_INTERVAL_TICKS:
            return None
        path = self._pick_next_file()
        if path is None:
            return None

        in_inbox = os.path.dirname(path) == self.paths['inbox']
        name = os.path.basename(path)
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except OSError:
            return None
        if not text.strip():
            # Empty file — record so we don't re-pick it, skip.
            self.read_manifest[name] = cycle
            return None

        # Feed through the universal intake path.  origin='forager'
        # is already a first-class source tag; downstream gate /
        # chemistry / cortical / SVO-writer handle it.
        try:
            self._intake_fn(
                text, modality='text', origin='forager',
                origin_detail=name)
        except Exception:
            return None

        self.read_manifest[name] = cycle
        self.last_forage_cycle = cycle
        self._last_forage_tick = cycle
        self.ingest_count += 1
        if in_inbox:
            self._archive_inbox_file(path)
        self.last_ingest_summary = (
            f"{name} ({'inbox' if in_inbox else 'reading_list'}), "
            f"{len(text)} chars")
        return {
            'file': name,
            'location': 'inbox' if in_inbox else 'reading_list',
            'chars': len(text),
            'cycle': cycle,
        }

    # ---- file selection ----

    def _pick_next_file(self) -> Optional[str]:
        """Inbox (freshest) first; then a reading_list file matching
        an information_request; then oldest unread reading_list file.
        None when inbox empty and all reading_list files read."""
        inbox_files = _list_text_files(self.paths['inbox'])
        if inbox_files:
            return inbox_files[0]
        rl_files = _list_text_files(self.paths['reading_list'])
        if not rl_files:
            return None
        unread = [p for p in rl_files
                  if os.path.basename(p) not in self.read_manifest]
        if not unread:
            return None
        wanted = list(self.information_requests)
        if wanted:
            for p in unread:
                if self._file_matches(p, wanted):
                    return p
        return unread[0]

    @staticmethod
    def _file_matches(path: str, concepts: List[str],
                       probe_bytes: int = 2048) -> bool:
        name = os.path.basename(path).lower()
        for c in concepts:
            if c in name:
                return True
        try:
            with open(path, 'r', encoding='utf-8',
                      errors='ignore') as f:
                head = f.read(probe_bytes).lower()
        except OSError:
            return False
        return any(c in head for c in concepts)

    def _archive_inbox_file(self, src: str) -> str:
        today = datetime.date.today().isoformat()
        dst_dir = os.path.join(self.paths['archive'], today)
        try:
            os.makedirs(dst_dir, exist_ok=True)
        except OSError:
            return src
        dst = os.path.join(dst_dir, os.path.basename(src))
        base, ext = os.path.splitext(dst)
        n = 1
        while os.path.exists(dst):
            dst = f"{base}.{n}{ext}"
            n += 1
        try:
            shutil.move(src, dst)
            return dst
        except OSError:
            return src

    # ---- re-engagement (chemistry-never-dissolves) ----

    def reconsider_all(self) -> int:
        """Clear the read manifest so reading_list material can be
        re-encountered.  A faded trace stays re-engageable per
        [[seagi-chemistry-never-fully-dissolves]] — this is the
        explicit re-engagement hook (not automatic).  Returns the
        number of entries cleared."""
        n = len(self.read_manifest)
        self.read_manifest = {}
        return n

    # ---- persistence ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            'read_manifest': dict(self.read_manifest),
            'last_forage_cycle': int(self.last_forage_cycle),
            'information_requests': list(self.information_requests),
            'ingest_count': int(self.ingest_count),
        }

    def load_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        rm = state.get('read_manifest')
        if isinstance(rm, dict):
            self.read_manifest = {
                str(k): int(v) for k, v in rm.items()
                if isinstance(v, (int, float))}
        try:
            self.last_forage_cycle = int(
                state.get('last_forage_cycle', -1))
        except (TypeError, ValueError):
            self.last_forage_cycle = -1
        reqs = state.get('information_requests')
        if isinstance(reqs, (list, tuple)):
            self.information_requests = deque(
                (str(r) for r in reqs), maxlen=REQUEST_CAP)
        try:
            self.ingest_count = int(state.get('ingest_count', 0))
        except (TypeError, ValueError):
            self.ingest_count = 0

    # ---- diagnostics ----

    def status(self) -> Dict[str, Any]:
        if self.paths is None:
            return {
                'active': False,
                'ingest_count': int(self.ingest_count),
                'information_requests':
                    list(self.information_requests)[:8],
            }
        inbox = _list_text_files(self.paths['inbox'])
        rl = _list_text_files(self.paths['reading_list'])
        return {
            'inbox_pending': len(inbox),
            'reading_list_total': len(rl),
            'reading_list_read': sum(
                1 for p in rl
                if os.path.basename(p) in self.read_manifest),
            'information_requests':
                list(self.information_requests)[:8],
            'ingest_count': int(self.ingest_count),
            'last_ingest': self.last_ingest_summary,
            'last_forage_cycle': int(self.last_forage_cycle),
            'resolutions_fired': int(self.resolutions_fired),
            'last_resolved': self.last_resolved,
        }
