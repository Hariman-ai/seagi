"""Knowledge curriculum prober — RIDDLES over his consumed knowledge, run in
the LIVE reverie loop (Step 6, 2026-06-26).

Why this exists
---------------
~182K of his edges sit in the quarantine pool: once-cohered facts that faded
from disuse, now invisible to inference (the one-way gate — a fact below the
inference floor can't be walked, so it can't be re-used, so it can't earn its
way back).  This prober gives faded-but-true facts a SECOND CHANCE by
re-presenting them: it picks a faded fact, restores its subject's quarantined
edges (the same `restore_concept_quarantine` AWM already calls on attention),
and opens a low-urgency goal on the subject so reverie attends it.  The EXISTING
background then does the work: restore marks the edges dirty -> Phase S
`reinforce_coherent_edges` re-corroborates any that still have a composable
non-hub support path -> their strength climbs back past the gate.

What it is NOT
--------------
A FLASHLIGHT, not a pump.  It writes no strength and credits no lifeforce.  The
riddle only selects WHICH faded facts get re-tested; the substrate's earn-gate
decides which actually climb.  A fact with no surviving support never coheres no
matter how often it is cued.  It targets `first_coherent_cycle > 0` facts only,
so re-corroboration never stamps a NEW first-coherence -> never increments
`newly_coherent` -> never reaches `record_learning` -> cannot farm lifeforce.

Doctrine
--------
This is curriculum/exercise (the user's "create these problems, easy->hard"), a
PRECURSOR to the paused mattering layer, not a substitute: a climbed fact stays
up only while it keeps genuinely corroborating.  Observable by construction —
every cue is logged and every cued fact's strength is tracked, so climb-vs-
refade is visible on the one true Seagi.  Subordinate goal lane (urgency 0.3 <
the 0.5 persistence floor) so it only ever fills a goal slot intrinsic reverie
left empty.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

GOAL_CURRICULUM = 'curriculum'
CURRICULUM_SPAWN_INTERVAL = 150       # cycles between cues (gentle, gives Phase S
                                      # time between presentations)
CURRICULUM_URGENCY = 0.3              # < PERSIST_URGENCY_FLOOR(0.5): strict
                                      # subordinate — never evicts a real goal
CURRICULUM_BATTERY = 40               # faded facts to track + re-cue round-robin
_HUB_DEGREE = 40                      # skip hub subjects (drowned riddles + junk)
_CHAINABLE = 0.41                     # a hop must reach ~this to compose past the
                                      # 0.12 inference gate (sqrt(0.12/0.7))
_CLIMB_EPS = 0.02
_WORD = re.compile(r"^[a-z][a-z'\-]{2,}$")     # a real word, not _world_s/_abstract_
_PROX = {'co_occurs', 'co_occur', 'related_to', 'relates_to'}   # proximity, not knowledge


class CurriculumProber:
    """Spawns riddle-goals over faded consumed facts and re-presents them to the
    earn-gate.  Mirrors GoalSpawner's shape (maybe_spawn each reflection pass)."""

    def __init__(self,
                 tracker_provider: Callable[[], Any],
                 lts_provider: Callable[[], Any],
                 cycle_provider: Callable[[], int]):
        self._tracker_provider = tracker_provider
        self._lts_provider = lts_provider
        self._cycle = cycle_provider
        self._battery: List[Tuple[str, str, str]] = []   # (subject, relation, answer)
        self._idx = 0
        self._last_spawn = -10 ** 9
        self._built = False
        # observability: (s,r,t) -> {first, best, cur, cues}
        self._track: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.cues = 0
        self.spawns = 0
        self.restores = 0

    # ---- battery construction (one-time, lazy, defensive) ----

    def _degree(self, concepts: dict, name: str) -> int:
        c = concepts.get(name)
        if c is None:
            return 0
        eo = getattr(c, 'edges_out', {}) or {}
        return sum(len(v) for v in eo.values())

    def _build_battery(self) -> None:
        self._built = True
        try:
            lts = self._lts_provider()
            sub = getattr(lts, 'substrate', None) if lts else None
            if sub is None:
                return
            concepts = getattr(sub, 'concepts', {}) or {}
            qedges = getattr(sub, 'quarantine_edges', {}) or {}
            cand: List[Tuple[float, str, str, str]] = []
            scanned = 0
            for e in qedges.values():
                scanned += 1
                if scanned > 400000:
                    break
                s = str(getattr(e, 'source', '') or '')
                t = str(getattr(e, 'target', '') or '')
                r = str(getattr(e, 'relation_name', '') or '')
                if r in _PROX:
                    continue
                if not (_WORD.match(s) and _WORD.match(t)):
                    continue
                if int(getattr(e, 'first_coherent_cycle', 0) or 0) <= 0:
                    continue                       # fcc>0 only -> farm-safe
                if self._degree(concepts, s) > _HUB_DEGREE:
                    continue
                st = float(getattr(e, 'strength', 0.0) or 0.0)
                cand.append((st, s, r, t))
            cand.sort()                            # faintest first
            seen = set()
            for st, s, r, t in cand:
                if s in seen:                      # diverse subjects
                    continue
                seen.add(s)
                self._battery.append((s, r, t))
                if len(self._battery) >= CURRICULUM_BATTERY:
                    break
            print('[curriculum] battery built: %d faded fcc>0 content facts '
                  '(from %d quarantined edges)' % (len(self._battery), len(qedges)),
                  flush=True)
        except Exception as e:                     # never crash the daemon
            print('[curriculum] battery build error: %s: %s'
                  % (type(e).__name__, e), flush=True)

    # ---- strength read (active edge, else quarantine) ----

    def _current_strength(self, s: str, r: str, t: str) -> Optional[float]:
        try:
            lts = self._lts_provider()
            if lts is None:
                return None
            e = lts.edge(s, r, t)
            if e is not None:
                return float(getattr(e, 'strength', 0.0) or 0.0)
            sub = getattr(lts, 'substrate', None)
            if sub is not None:
                qe = getattr(sub, 'quarantine_edges', {}) or {}
                ed = qe.get((s, r, t))
                if ed is not None:
                    return float(getattr(ed, 'strength', 0.0) or 0.0)
        except Exception:
            pass
        return None

    # ---- the live pass ----

    def maybe_spawn(self) -> None:
        try:
            cyc = int(self._cycle())
            if cyc - self._last_spawn < CURRICULUM_SPAWN_INTERVAL:
                return
            if not self._built:
                self._build_battery()
            if not self._battery:
                return
            s, r, t = self._battery[self._idx % len(self._battery)]
            self._idx += 1
            self._last_spawn = cyc
            self.cues += 1

            before = self._current_strength(s, r, t)
            tr = self._track.setdefault((s, r, t),
                                        {'first': before, 'best': before, 'cues': 0})
            tr['cues'] += 1
            if before is not None and (tr['best'] is None or before > tr['best']):
                tr['best'] = before

            # 1) PRESENT the material: restore the subject's quarantined edges so
            #    the support triangle is visible to the next Phase S pass.  Same
            #    public "cognition-external touch" AWM uses on attention; bounded
            #    by the non-hub filter above.  Restore writes NO strength.
            lts = self._lts_provider()
            sub = getattr(lts, 'substrate', None) if lts else None
            restored = 0
            if sub is not None and hasattr(sub, 'restore_concept_quarantine'):
                try:
                    restored = int(sub.restore_concept_quarantine(s, cycle=cyc) or 0)
                except TypeError:
                    restored = int(sub.restore_concept_quarantine(s) or 0)
                except Exception as e:
                    print('[curriculum] restore error %s: %s'
                          % (type(e).__name__, e), flush=True)
                self.restores += restored

            # 2) OPEN the riddle goal so reverie attends the subject.
            tracker = self._tracker_provider()
            g = None
            if tracker is not None:
                g = tracker.spawn(kind=GOAL_CURRICULUM, focal=s, target=t,
                                  urgency=CURRICULUM_URGENCY, source='curriculum',
                                  cycle=cyc,
                                  notes='riddle: %s -%s-> ? (consumed-knowledge recall)'
                                        % (s, r))
            if g is not None:
                self.spawns += 1

            climbed = self._count_climbed()
            print("[curriculum] cue %d: '%s -%s-> %s' str=%s restored=%d "
                  "goal=%s climbed=%d/%d"
                  % (self.cues, s, r, t,
                     ('%.3f' % before) if before is not None else 'NA',
                     restored, g is not None, climbed, len(self._battery)),
                  flush=True)
        except Exception as e:                     # never crash the daemon
            print('[curriculum] error: %s: %s' % (type(e).__name__, e), flush=True)

    def _count_climbed(self) -> int:
        n = 0
        for (s, r, t), v in self._track.items():
            cur = self._current_strength(s, r, t)
            if cur is not None and v['best'] is not None and cur > v['best']:
                v['best'] = cur
            f, b = v['first'], v['best']
            if f is not None and b is not None and b - f > _CLIMB_EPS:
                n += 1
        return n

    # ---- telemetry ----

    def stats(self) -> Dict[str, Any]:
        climbed: List[Tuple] = []
        refaded = 0
        for (s, r, t), v in self._track.items():
            cur = self._current_strength(s, r, t)
            f, b = v['first'], v['best']
            if f is not None and b is not None and b - f > _CLIMB_EPS:
                climbed.append((s, r, t, round(f, 3), round(b, 3),
                                round(cur, 3) if cur is not None else None))
                if cur is not None and cur < b - _CLIMB_EPS:
                    refaded += 1
        return {
            'battery': len(self._battery),
            'cues': self.cues,
            'goals_spawned': self.spawns,
            'edges_restored': self.restores,
            'tracked': len(self._track),
            'climbed': len(climbed),
            'refaded': refaded,
            'climbed_sample': climbed[:12],
        }
