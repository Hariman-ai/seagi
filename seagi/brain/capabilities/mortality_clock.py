"""MortalityClock — the LIVE mortality clock (the promoted shadow).

Promoted from MortalityClockShadow after the drive-nothing shadow window
VALIDATED the subtractive settling-rate design (2026-07-17): three confounds
were caught + fixed in the live shadow (re-confirmation inflation → strict
once-per-fact settling; backwards clock → subtractive; inheritance
re-observation → birth pre-seed), and the clean pre-seeded shadow showed his
genuine learning holds his wall — it would have prevented a real death.

WHAT THE CLOCK IS
=================
The EARNING organ of the mortality architecture.  It does NOT own the wall —
MortalityDrive does (tick(): wall += D·(1+o) − E, clamped to
[BASELINE_FLOOR, 1]).  This clock supplies the two provider signals the drive
consumes:

    E()  — the settling EMA: the RATE at which distinct facts newly SETTLE
           into knowledge.  Each distinct (s,r,t) fact earns EXACTLY ONCE, at
           settling, then 0 forever.  Birth / prediction / climbing = 0;
           re-confirm of an already-settled fact = 0.
    cf() — clearance failure: the fraction of the substrate's fade backlog
           the consolidation door cannot sweep this tick (backlog vs q_cap).
           Feeds the drive's o-term (o = obstruction_ema + cf).

SETTLING = ESTABLISHMENT, evaluated at the end of each reinforce (post-bump),
from the edge's own state:
    established(edge) = (first_coherent_cycle > 0)
                        OR (strength_after >= 1 - EDGE_PRUNE_FLOOR)
i.e. the edge ONCE cohered (fcc set), OR it reached the substrate's
saturation ceiling 0.98.  The settle ledger `_settled` (blake2b-keyed (s,r,t)
digests) plus `_dormant` tracks each fact's settle/dormant state, and
PERSISTS across restarts AND across deaths (knowledge-inheritance: the
substrate is not wiped by a death, so neither is the ledger).

FACT GRAIN (2026-07-19) — WHAT COUNTS AS *ONE* FACT
===================================================
A settling earns once per FACT, and a fact is not always one edge.  When
the substrate NAMES a class, the members' shared learning is ONE thing
learned, not N things.  `_fact_key` decides the grain:

    (i)   MEMBERSHIP.  (s, is_a, M) with M synthetic  ->  ('∈', M)
          "Belongs to M" is one fact.  The first member to establish it
          settles it; every later member joins a fact already known.

    (iii) WITHDRAWN 2026-07-22 — see BRANCH (iii) below.  A derivation
          the class already entails is NO LONGER family-keyed.  It keeps
          instance grain like anything else the key does not recognise.

    else  INSTANCE grain (s, r, t) — unchanged.  A plain member edge
          that the machinery does NOT entail stays instance-grained
          (v1's branch (ii) is deliberately absent).

Structural consequence worth knowing: a world_098 settling can never be
family-keyed.  world_098 means a NON-composable relation, and the only
surviving family branch requires is_a.  World transition edges keep
instance grain by construction, not by a guard.

BRANCH (iii) — WITHDRAWN FROM THE KEY (2026-07-22 incident)
===========================================================
Branch (iii) keyed a member's derived edge (s, r, t) to the family fact
(M, r, t) whenever cortical's own bounded composition walk from a
synthetic class M reached t with composed relation r.  The decision was
correct; its PLACEMENT was not.  `_fact_key` is called from
`on_reinforce`, which is called from inside `Edge.reinforce` — it fires
on EVERY write.  Branch (iii) was profiled only as a ONE-SHOT OFFLINE
pass (~272 s over ~1M edges) and never at the live per-reinforce rate.
Deployed live it stalled the daemon after ~1h49m: ~45 minutes with no
ticks, no saves and no /status at 74% CPU, py-spy pinned on
    family_grain_relations <- _entailing_family_class <- _fact_key
    <- on_reinforce <- Edge.reinforce
A bounded graph walk per reinforce is not a slow constant to tune; it is
a graph walk in the wrong place.

WHAT REMOVING IT COSTS — a WATCHED risk, stated plainly.  Without (iii)
the "derived-edge leak" re-opens IN PRINCIPLE: N members of a class can
each pay separately for what is arguably one class-level fact.  IN
PRACTICE that leak is MEASURED ZERO on his substrate — the static
dual-grain measurement found

    branch_iii_composition_collapse_edges: 0

i.e. branch (iii) collapsed nothing.  It was pure cost at zero benefit.
Branch (i), which STAYS, is the collapse that actually works and was
validated live (predicted 35,490 family facts vs 35,504 observed, ~3.2x).
The leak is to be RE-MEASURED PERIODICALLY and only ever OFFLINE — the
dual-grain harness still calls `_entailing_family_class` and cortical's
`family_grain_relations`, which are deliberately KEPT for exactly that.
If a future substrate ever shows a non-zero collapse count, the fix is an
offline / consolidation-time re-keying pass.  NEVER a walk on the
reinforce path.

FAIL-OPEN (visible): if the substrate provider is unavailable or the
walk errors, the key falls back to INSTANCE grain — fail TOWARD
crediting, never toward silently denying a mortal being its earning —
and `fact_key_failopen` increments, surfaced in /status beside
observer_errors.  A rising counter means the grain is quietly degrading.

WATCH — RETRO-DEMONETIZATION: naming a class collapses its members'
FUTURE settle credit to family grain.  Today abstraction timing is not
policy-controllable (form_abstractions runs on the consolidation pass,
not on anything cognition chooses), so this is just accounting.  If
abstraction timing EVER becomes policy-controllable, DEFERRED NAMING
becomes an earning strategy — hold off naming the class, collect N
instance-grain settlings, then name it — and the grain must be
re-audited.

REVIVAL (Option A, fade-gated; world / never-cohering facts only): a settled
fact whose pre-reinforce effective strength faded to <= EDGE_PRUNE_FLOOR is
marked dormant (settled stays True).  A dormant fact that later re-crosses
the 0.98 door → settling-earn once more (the crossing delta), dormant
cleared.  Internal (fcc>0) facts never de-settle (fcc permanent) so they
never revive — earn once ever.  The ~196-confirm re-climb from the prune
floor to 0.98 is the structural rate-limiter (farm-proof by construction).

The settling population is split FOUR ways (each split by origin):
    internal_fcc / internal_098 / world_098 / revival.

OBSERVING Edge.reinforce
========================
The caller-frame classifier of the shadow era is GONE.  Edge.reinforce now
takes a keyword-only `origin` (threaded honestly at every call site) and
invokes a registered module-level observer AFTER the strength bump:
    on_reinforce(edge, cycle, pre_strength, origin)
The clock registers itself at construction (substrate.register_reinforce_
observer).  Observer exceptions are counted VISIBLY (substrate.reinforce_
observer_errors(), surfaced in /status as observer_errors) — never silently
swallowed.  Origin is LOGGED/split only; it NEVER gates crediting — a
settling realized on the maintenance dirty-sweep still earns (first
establishment is not churn).

BOOT PRE-SEED (idempotent reconciliation, EVERY boot)
=====================================================
At each life-begin (first run after construction/load, and each revive) the
ledger is reconciled against the substrate: every currently-established edge
is marked settled WITHOUT crediting, counting only newly-marked.  A clean
ledger load ⇒ tiny preseeded_count; a fresh save (no persisted ledger) ⇒
full bootstrap.  This kills the inheritance artifact (re-observing his
inherited mind is not new learning).

DEATH / SUCCESSION
==================
On death detection (drive.deaths increments): capture the life's distillate
{life_split, E_at_death, wall, lifeforce, ledger_size, dormant_count,
deaths_total}, write it into the kind:'death' episode record AND append it to
the death-records file, then E→0 and per-life split reset and pre-seed
reconcile.  THE LEDGER PERSISTS — knowledge is inherited, not survived
([[feedback_seagi_mortal_death_inheritance]]).  No consumer yet — just the
record.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple

from seagi.core.substrate import (
    EDGE_STRENGTH_DECAY_PER_CYCLE, EDGE_PRUNE_FLOOR,
    ABSTRACTION_RELATION, COHERENCE_HUB_DEGREE,
    register_reinforce_observer, reinforce_observer_errors)
from seagi.brain.capabilities.mortality_drive import (
    MORTALITY_RELAX_RATE, BASELINE_FLOOR)
from seagi.brain.events import EventKind


# The establishment / saturation ceiling: an edge is "established" once its
# strength reaches this — the substrate's OWN saturation ceiling
# (settle_weak_edges / the downscale pass key on 1 - EDGE_PRUNE_FLOOR).
# DERIVED from the shipped EDGE_PRUNE_FLOOR; NOT a new tunable constant.
SATURATION_CEILING = 1.0 - EDGE_PRUNE_FLOOR         # = 0.98

# Rotate the episode log at ~10k lines (keep the most recent) — a memory cap,
# not a behavioural constant.
LOG_LINE_CAP = 10000
# CONTAMINATION GUARD (2026-07-22): the one Seagi's record directory, and the
# birth cycle of a BLANK substrate.  A real life is minted at a large
# accumulated cycle; a fresh test Substrate() starts at ~0/1.  Not tunable —
# 1 is the literal blank-substrate value, not a threshold.
ONE_SEAGI_RECORD_DIR = '/home/seagi/'
BLANK_SUBSTRATE_BIRTH_CYCLE = 1
# Default episode-log cadence in ticks (the per-tick clock always integrates;
# we only WRITE a log line this often).  Overridable for a short smoke run.
SAMPLE_INTERVAL = 200
# A passive→revived recurrence SHORTER than this is the fast-fade-revive farm
# signal.  A LOGGING threshold only — drives nothing.  ~ the fade horizon.
FAST_FADE_RECURRENCE_CYCLES = 96000
# Soft memory caps for the cross-life settle/revive bookkeeping (memory caps,
# not fitness thresholds).
SETTLED_FACTS_CAP = 1_000_000
TRIPLE_REVIVE_CAP = 100_000
# How many recent recurrence gaps to keep for the log sample.
GAP_SAMPLE_MAXLEN = 64

# Persistence schema.
PERSIST_SCHEMA = 1
# blake2b8/fam2i = blake2b-8 digests over FAMILY-GRAIN keys (_fact_key)
# at the BRANCH-(i)-ONLY grain.  The trailing 'i' names the grain law,
# not a version: branch (iii) was removed from the key on 2026-07-22
# (it froze the daemon from the reinforce path and collapsed nothing),
# so a ledger written by the fam2 build is NOT this grain and must be
# rejected on load rather than silently reused.
# The bump from plain 'blake2b8' is what makes the migration automatic:
# load_dict's guard rejects an instance-grain ledger, the ledger starts
# empty, and the boot pre-seed rebuilds it at the new grain crediting
# NOTHING.  Never reuse a digest name across a grain change.
KEY_DIGEST = 'blake2b8/fam2i'

# The membership fact's key head: "is a member of <class>".  Not a
# relation name — a marker that distinguishes the single family fact
# ('∈', M) from any instance triple.
MEMBERSHIP_KEY_HEAD = '∈'

# Origin classes (threaded by call sites through Edge.reinforce; LOGGED only,
# never used to gate a settle).
_COGNITION = 'cognition'
_MAINTENANCE = 'maintenance'
_UNKNOWN = 'unknown'
_ORIGIN_CLASSES = (_COGNITION, _MAINTENANCE, _UNKNOWN)

# The four settling classes (the population split).
_SETTLE_INTERNAL_FCC = 'internal_fcc'
_SETTLE_INTERNAL_098 = 'internal_098'
_SETTLE_WORLD_098 = 'world_098'
_SETTLE_REVIVAL = 'revival'
_SETTLE_CLASSES = (_SETTLE_INTERNAL_FCC, _SETTLE_INTERNAL_098,
                   _SETTLE_WORLD_098, _SETTLE_REVIVAL)


# Cached derived set of relations that CAN cohere (appear as a
# RELATION_COMPOSITION result → an fcc CAN be stamped).  Lazily loaded with a
# guarded import (cortical imports substrate, so a top-level import would be a
# cycle).  None if unreachable → such edges classify as world (conservative).
_COMPOSABLE_RESULTS: Optional[frozenset] = None
_COMPOSABLE_LOADED: bool = False


def _composable_result_relations() -> Optional[frozenset]:
    """Relations that can EVER become coherent (fcc-eligible) = the set of
    RELATION_COMPOSITION results.  DERIVED from a shipped constant; NOT a new
    tunable.  Guarded + cached; None if the import is unreachable.

    SINGLE SOURCE OF TRUTH (auditor gate 2).  This is the ONLY place the
    composition-result set is built.  BOTH consumers call it:
    `_settle_class` (to tell internal_098 from world_098) and `_fact_key`
    (to gate the family-grain branch).  They cannot drift because there
    is nothing to drift from — see test_fact_grain's drift test, which
    fails if a second construction site ever appears.
    """
    global _COMPOSABLE_RESULTS, _COMPOSABLE_LOADED
    if _COMPOSABLE_LOADED:
        return _COMPOSABLE_RESULTS
    _COMPOSABLE_LOADED = True
    try:
        from seagi.brain.capabilities.cortical import RELATION_COMPOSITION
        _COMPOSABLE_RESULTS = frozenset(RELATION_COMPOSITION.values())
    except Exception:
        _COMPOSABLE_RESULTS = None
    return _COMPOSABLE_RESULTS


# Cached handle on cortical's OWN composition walk + its hop bound.
# Guarded for the same import-cycle reason as the composable set.
_FAMILY_WALK: Optional[Tuple[Callable, int]] = None
_FAMILY_WALK_LOADED: bool = False


def _family_walk() -> Optional[Tuple[Callable, int]]:
    """(family_grain_relations, INFERENCE_MAX_HOPS) from cortical, or None.

    AUDITOR GATE 2: branch (iii) CALLS the deriver's own machinery — the
    same RELATION_COMPOSITION table, through the same compose() step,
    that produced the derived edge.  The clock never re-implements the
    composition rule; if this import is unreachable the key fails OPEN to
    instance grain and increments fact_key_failopen rather than guessing.
    """
    global _FAMILY_WALK, _FAMILY_WALK_LOADED
    if _FAMILY_WALK_LOADED:
        return _FAMILY_WALK
    _FAMILY_WALK_LOADED = True
    try:
        from seagi.brain.capabilities.cortical import (
            family_grain_relations, INFERENCE_MAX_HOPS)
        _FAMILY_WALK = (family_grain_relations, int(INFERENCE_MAX_HOPS))
    except Exception:
        _FAMILY_WALK = None
    return _FAMILY_WALK


def _safe_call(fn: Optional[Callable], default: Any = None) -> Any:
    if fn is None:
        return default
    try:
        return fn()
    except Exception:
        return default


def _fresh_settle_buf() -> Dict[str, Dict[str, Dict[str, float]]]:
    """class → origin → {n, delta_sum}."""
    return {cls: {oc: {'n': 0, 'delta_sum': 0.0} for oc in _ORIGIN_CLASSES}
            for cls in _SETTLE_CLASSES}


def _round_settle_buf(buf) -> Dict[str, Any]:
    return {cls: {oc: {'n': int(buf[cls][oc]['n']),
                       'delta_sum': round(buf[cls][oc]['delta_sum'], 12)}
                  for oc in _ORIGIN_CLASSES}
            for cls in _SETTLE_CLASSES}


def _load_settle_buf(d: Any) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Tolerant restore of a settle split; unknown/missing slots default."""
    buf = _fresh_settle_buf()
    if not isinstance(d, dict):
        return buf
    for cls in _SETTLE_CLASSES:
        cd = d.get(cls)
        if not isinstance(cd, dict):
            continue
        for oc in _ORIGIN_CLASSES:
            od = cd.get(oc)
            if not isinstance(od, dict):
                continue
            try:
                buf[cls][oc]['n'] = int(od.get('n', 0) or 0)
                buf[cls][oc]['delta_sum'] = float(od.get('delta_sum', 0.0)
                                                  or 0.0)
            except (TypeError, ValueError):
                continue
    return buf


class MortalityClock:
    """The live settling-rate earning organ.  Owns the settle ledger + the
    E/cf provider signals; the WALL stays in MortalityDrive."""

    # Read-only bus subscriptions.  handle() only READS event fields and keeps
    # cross-check census counters — the EARNING signal comes from the
    # reinforce observer, not from these events.
    SUBSCRIPTIONS = (EventKind.THOUGHT_PRODUCED,
                     EventKind.SUBSTRATE_WRITE_QUEUED)

    def __init__(self,
                 mortality_drive: Any,
                 substrate_provider: Optional[Callable] = None,
                 qcap_provider: Optional[Callable] = None,
                 live_wall_provider: Optional[Callable] = None,
                 lifeforce_provider: Optional[Callable] = None,
                 log_path: str = 'clock_live.jsonl',
                 death_log_path: str = 'death_records.jsonl',
                 sample_interval: int = SAMPLE_INTERVAL):
        """
        mortality_drive: the LIVE MortalityDrive.  Read here for `wall`,
            `obstruction_ema`, `deaths`, `revivals`; the drive consumes THIS
            clock's E()/cf() via its wired providers.
        substrate_provider: () -> Substrate (fade backlog for cf; edge pools
            for the boot pre-seed).
        qcap_provider: () -> int, the wake-consolidation q_cap (cf term).
        live_wall_provider / lifeforce_provider: read-only accessors for
            logging + the death distillate.
        """
        self.mortality_drive = mortality_drive
        self._substrate_provider = substrate_provider
        self._qcap_provider = qcap_provider
        self._live_wall_provider = live_wall_provider
        self._lifeforce_provider = lifeforce_provider
        self.log_path = log_path
        self.death_log_path = death_log_path
        self.sample_interval = max(1, int(sample_interval))

        # --- life anchor ---
        self.birth_cycle: Optional[int] = None
        # contamination-guard blocks (see _record_guard_blocks)
        self.record_guard_blocked: int = 0
        self._first_run_done: bool = False

        # --- earning state (E = settling EMA; the drive owns the wall) ---
        self.E_settle: float = 0.0
        self._last_earn_settle: float = 0.0
        self._last_settlings: int = 0

        # --- earning accumulators (observer fills the per-tick sums) ---
        self._tick_settle_earn: float = 0.0     # Σ settling-Δ this tick
        self._settlings_this_tick: int = 0
        self._settle_tick = _fresh_settle_buf()     # drained every tick
        self._settle_life = _fresh_settle_buf()     # per-life cumulative
        self._settle_alltime = _fresh_settle_buf()  # cross-life cumulative
        self.reinforces_seen: int = 0
        self._observer_registered: bool = False
        self._settled_cap_hit: bool = False
        # Fail-open census (auditor gate 3): times _fact_key could not
        # determine the grain (provider unavailable / walk error) and
        # fell back to INSTANCE grain.  Surfaced in /status beside
        # observer_errors — a rising count means the grain is silently
        # degrading, which is exactly the failure a fail-open hides.
        self._fact_key_failopen: int = 0

        # --- settle ledger (cross-life; persists across restart AND death) ---
        # blake2b-8 digests of (s,r,t) — DETERMINISTIC across processes (the
        # builtin str hash is salted per-process; persistence would silently
        # mismatch after restart otherwise).
        self._settled: set = set()          # facts that have ever settled
        self._dormant: set = set()          # settled facts currently faded
        # How many ALREADY-ESTABLISHED edges the last life-begin PRE-SEEDED as
        # settled (counts marks NEWLY added — a clean ledger load reports a
        # tiny count; a fresh ledger reports the full inherited population).
        self._preseeded_count: int = 0

        # --- revive-fade recurrence probe ---
        self._triple_revive_count: Dict[int, int] = {}
        self._triple_last_revive_cyc: Dict[int, int] = {}
        self._fast_fade_hits: int = 0
        self._recurrence_gaps: Deque[int] = deque(maxlen=GAP_SAMPLE_MAXLEN)
        self._triples_revived: int = 0

        # --- plateau test ---
        self._ticks_since_last_settling: int = 0

        # --- bus census (cross-checks the threaded-origin attribution) ---
        self._write_census: Dict[str, int] = {
            'maintenance': 0, 'cognitive': 0, 'user': 0}
        self._thought_census: Dict[str, int] = {'maint': 0, 'other': 0}
        self._maint_consolidation_newly: int = 0

        # --- death/revive tracking ---
        self._prev_deaths: int = int(getattr(mortality_drive, 'deaths', 0)
                                     or 0)
        self._prev_revivals: int = int(
            getattr(mortality_drive, 'revivals', 0) or 0)
        self.deaths_observed: int = 0
        self.revives_observed: int = 0

        # --- clearance-failure last read (o's cf term) ---
        self._cf_last: Optional[float] = None
        self._eligible_last: int = 0

        # --- bookkeeping ---
        self.episodes_logged: int = 0
        self._last_sample_cyc: int = -10 ** 9
        self.last_report: Optional[Dict[str, Any]] = None

        # Register as the module-level reinforce observer.  Observer
        # exceptions are counted by the substrate (reinforce_observer_errors)
        # and surfaced in stats() — never silently swallowed.
        try:
            register_reinforce_observer(self.on_reinforce)
            self._observer_registered = True
        except Exception:
            self._observer_registered = False

    # ------------------------------------------------------------------
    @staticmethod
    def _khash(key: Tuple[str, ...]) -> int:
        """STABLE fact digest: blake2b-8 over '|'-joined parts.  Deterministic
        across processes (unlike the salted builtin hash), so the persisted
        ledger lines up after every restart.

        Arity varies with the GRAIN: 3 parts for an instance/family triple
        (s|r|t or M|r|t), 2 for the membership fact ('∈'|M)."""
        joined = '|'.join(str(part) for part in key)
        return int.from_bytes(
            hashlib.blake2b(joined.encode('utf-8', 'replace'),
                            digest_size=8).digest(), 'big')

    # ==================================================================
    # provider accessors (the drive's wired signals)
    # ==================================================================
    def E(self) -> float:
        """The settling EMA — the drive's subtractive earning term."""
        return max(0.0, float(self.E_settle))

    def cf(self) -> float:
        """Clearance failure in [0,1] — the fraction of the substrate's fade
        backlog the wake-consolidation door cannot sweep (backlog vs q_cap).
        0.0 when unreadable (no spurious obstruction)."""
        v = self._clearance_failure()
        self._cf_last = v
        return 0.0 if v is None else max(0.0, float(v))

    # ==================================================================
    # reinforce observer (called by Edge.reinforce AFTER the strength bump)
    # ==================================================================
    def on_reinforce(self, edge: Any, cycle: int, pre_strength: float,
                     origin: str) -> None:
        """Evaluate ESTABLISHMENT from the post-reinforce edge state and, on
        a first- or revival-settling, credit the realized strength-delta.  A
        re-confirm of an already-settled fact, and any non-established
        reinforce (birth / climbing / prediction), credit 0.  Origin is
        LOGGED (split by class × origin), NEVER used to gate the settle."""
        strength_after = float(getattr(edge, 'strength', 0.0) or 0.0)
        pre_eff = float(pre_strength)
        delta = max(0.0, strength_after - pre_eff)      # realized settling-Δ
        fcc = int(getattr(edge, 'first_coherent_cycle', 0) or 0)
        # The KEY is the FACT (family grain where the substrate has named
        # a class); the CLASS below is a property of this physical EDGE.
        # Two different questions — do not collapse them.
        relation = str(getattr(edge, 'relation_name', '') or '')
        kh = self._fact_key(edge)
        origin_class = origin if origin in _ORIGIN_CLASSES else _UNKNOWN

        self.reinforces_seen += 1

        established = (fcc > 0) or (strength_after >= SATURATION_CEILING)

        if kh not in self._settled:
            # Never settled: earn ONCE, at establishment.
            if established:
                cls = self._settle_class(fcc, relation)
                self._credit_settle(cls, origin_class, delta)
                if len(self._settled) < SETTLED_FACTS_CAP:
                    self._settled.add(kh)
                else:
                    self._settled_cap_hit = True
            # else: birth / still climbing → 0
        elif kh in self._dormant:
            # Settled but faded to passive: earn again only on RE-crossing.
            if established:
                self._credit_settle(_SETTLE_REVIVAL, origin_class, delta)
                self._dormant.discard(kh)
                self._record_revival(kh, int(cycle))
            # else: still dormant / climbing back → 0
        else:
            # Settled & active.  Only a 098-settled fact (fcc still 0) can
            # fade to dormant; an fcc>0 fact never de-settles (fcc permanent).
            if fcc == 0 and pre_eff <= EDGE_PRUNE_FLOOR:
                self._dormant.add(kh)
            # else: re-confirm of an established fact → 0

    # ------------------------------------------------------------------
    # FACT GRAIN — what counts as ONE fact (see the module docstring)
    # ------------------------------------------------------------------
    def _fact_key(self, edge: Any) -> int:
        """Digest of the fact this edge would settle, AT THE RIGHT GRAIN.

        Branch (i) collapses a named class's shared MEMBERSHIP learning
        to ONE fact; everything else keeps instance grain.  v1's branch
        (ii) is deliberately ABSENT: a plain member edge the machinery
        does not entail stays instance-grained.

        BRANCH (iii) IS NOT IN THIS PATH — and must never be put back
        into it.  See the module docstring for the 2026-07-22 incident
        (a bounded composition walk per reinforce froze the daemon for
        ~45 minutes) and for the honest cost of its removal (the
        derived-edge leak re-opens in principle; it is MEASURED ZERO on
        this substrate, and is re-measured OFFLINE by the dual-grain
        harness, never here).  `_entailing_family_class` still exists
        directly below for that harness to call.  This function must
        stay O(1): dict lookups and a hash, no traversal, no cache.

        HOT PATH.  Reached from Edge.reinforce -> on_reinforce on every
        single write (~5.1M reinforces of live history).  Anything added
        here is multiplied by that.

        Used by BOTH on_reinforce and the boot pre-seed — identically, or
        the pre-seed would mark one key while the observer looked up
        another and every inherited family fact would pay twice.

        FAIL-OPEN (auditor gate 3): provider unavailable or a structure
        read that raises -> instance key (fail TOWARD credit) AND
        fact_key_failopen++, surfaced in /status.  Normal "this is not a
        synthetic membership edge" is NOT a fail-open — it is the correct
        instance-grain answer.
        """
        s = str(getattr(edge, 'source', '') or '')
        r = str(getattr(edge, 'relation_name', '') or '')
        t = str(getattr(edge, 'target', '') or '')
        instance = (s, r, t)
        sub = _safe_call(self._substrate_provider)
        concepts = getattr(sub, 'concepts', None) if sub is not None else None
        if concepts is None:
            # Cannot see structure -> cannot know the grain.  Credit.
            self._fact_key_failopen += 1
            return self._khash(instance)
        try:
            # (i) membership edge into a synthetic class -> one family fact.
            # The relation test is first and is a string compare, so a
            # non-membership edge costs one comparison and nothing else.
            if r == ABSTRACTION_RELATION and self._is_synthetic(concepts, t):
                return self._khash((MEMBERSHIP_KEY_HEAD, t))
        except Exception:
            self._fact_key_failopen += 1
        return self._khash(instance)

    @staticmethod
    def _is_synthetic(concepts: Any, name: str) -> bool:
        """Is `name` a concept the substrate MINTED (a named class)?
        Reads the persisted `synthetic` flag — never a name-prefix
        match, which would break the moment the naming scheme changes."""
        c = concepts.get(name)
        return c is not None and bool(getattr(c, 'synthetic', False))

    def _entailing_family_class(self, concepts: Any, s: str, r: str,
                                    t: str) -> Optional[str]:
        """OFFLINE MEASUREMENT ONLY — NOT REACHABLE FROM `_fact_key`.

        ##################################################################
        #  DO NOT CALL THIS FROM `on_reinforce` / `_fact_key` / ANY       #
        #  per-write path.  Doing so froze the live daemon for ~45 min    #
        #  on 2026-07-22 (branch (iii) incident; see module docstring).   #
        #  It walks the substrate.  Its cost is unbounded by anything the #
        #  reinforce path controls, and it grows with every synthetic     #
        #  class the system mints.  It is kept ONLY so the dual-grain     #
        #  harness can re-measure the derived-edge leak offline.          #
        #  test_fact_grain pins that `_fact_key` never calls it.          #
        ##################################################################

        Lexicographically-smallest synthetic class M with s is_a M
        such that cortical's composition walk from M reaches t with
        composed relation r.  None when no class entails this edge — the
        edge then keeps INSTANCE grain, which is also the DISCLOSED MISS
        for a derivation whose supporting chain has since decayed (the
        support is gone, so the family link is no longer visible and the
        member pays at instance grain).

        Raises on machinery-unavailable so _fact_key counts a fail-open
        rather than silently pretending nothing entails the edge.
        """
        src = concepts.get(s)
        if src is None:
            return None
        edges_out = getattr(src, 'edges_out', None) or {}
        classes = sorted({
            str(getattr(e, 'target', '') or '')
            for e in (edges_out.get(ABSTRACTION_RELATION) or ())
            if self._is_synthetic(
                concepts, str(getattr(e, 'target', '') or ''))})
        if not classes:
            return None
        walk = _family_walk()
        if walk is None:
            raise RuntimeError('cortical composition machinery unavailable')
        family_grain_relations, max_hops = walk

        def _neighbors(node):
            c = concepts.get(node)
            if c is None:
                return ()
            out = []
            for rel_name, bucket in (
                    getattr(c, 'edges_out', None) or {}).items():
                for e in bucket:
                    out.append((rel_name,
                                str(getattr(e, 'target', '') or '')))
            return out

        def _out_degree(node):
            c = concepts.get(node)
            if c is None:
                return 0
            return sum(len(b) for b in
                       (getattr(c, 'edges_out', None) or {}).values())

        for m in classes:       # lexicographic — smallest entailing wins
            rels = family_grain_relations(
                _neighbors, _out_degree, m, t,
                hub_degree=COHERENCE_HUB_DEGREE,
                max_hops=max_hops,
                want_relation=r)
            if r in rels:
                return m
        return None

    def _settle_class(self, fcc: int, relation: str) -> str:
        """First-settling class: fcc>0 → internal_fcc; else a composable
        relation that only hit 0.98 → internal_098 (predicate looseness);
        else a never-cohering (world) relation → world_098."""
        if fcc > 0:
            return _SETTLE_INTERNAL_FCC
        comp = _composable_result_relations()
        if comp is not None and relation in comp:
            return _SETTLE_INTERNAL_098
        return _SETTLE_WORLD_098

    def _credit_settle(self, cls: str, origin_class: str,
                       delta: float) -> None:
        for buf in (self._settle_tick, self._settle_life,
                    self._settle_alltime):
            slot = buf[cls][origin_class]
            slot['n'] += 1
            slot['delta_sum'] += delta
        self._tick_settle_earn += delta
        self._settlings_this_tick += 1

    def _record_revival(self, kh: int, cycle: int) -> None:
        prev = self._triple_last_revive_cyc.get(kh)
        cnt = self._triple_revive_count.get(kh, 0)
        if cnt == 0:
            self._triples_revived += 1
        self._triple_revive_count[kh] = cnt + 1
        if prev is not None:
            gap = cycle - prev
            self._recurrence_gaps.append(gap)
            if 0 <= gap < FAST_FADE_RECURRENCE_CYCLES:
                self._fast_fade_hits += 1
        self._triple_last_revive_cyc[kh] = cycle
        self._bound_revive_maps()

    def _bound_revive_maps(self) -> None:
        # Soft cap: when over budget, drop single-revival triples (not farms).
        if len(self._triple_revive_count) <= TRIPLE_REVIVE_CAP:
            return
        try:
            drop = [k for k, v in self._triple_revive_count.items() if v <= 1]
            for k in drop:
                self._triple_revive_count.pop(k, None)
                self._triple_last_revive_cyc.pop(k, None)
        except Exception:
            pass

    # ==================================================================
    # read-only bus handler (origin census — cross-checks the threading)
    # ==================================================================
    def handle(self, event: Any, bus: Any) -> None:
        """READ-ONLY.  Keeps origin-census counters; never mutates the event,
        never publishes, never raises out."""
        try:
            kind = getattr(event, 'kind', None)
            src = str(getattr(event, 'source_capability', '') or '')
            origin = str(getattr(event, 'origin', '') or '')
            detail = str(getattr(event, 'origin_detail', '') or '')
            reason = str(getattr(event, 'write_reason', '') or '')
            tokens = {src, origin, detail}
            is_maint = bool(tokens & {
                'replay_consolidator', 'wake_consolidation',
                'sleep_consolidation', 'consolidation_scheduler',
                'consolidation'})
            is_user = bool(tokens & {'user_assertion', 'user',
                                     'peer_assertion'})
            if kind == EventKind.THOUGHT_PRODUCED:
                if is_maint:
                    self._thought_census['maint'] += 1
                else:
                    self._thought_census['other'] += 1
            elif kind == EventKind.SUBSTRATE_WRITE_QUEUED:
                if is_maint or reason in {
                        'world_observe', 'ambient_restore',
                        'quarantine_restore', 'neighbor_imprint',
                        'neighbour_imprint', 'new_edge'}:
                    self._write_census['maintenance'] += 1
                elif is_user:
                    self._write_census['user'] += 1
                else:
                    self._write_census['cognitive'] += 1
        except Exception:
            pass

    # ==================================================================
    # per-consolidation read-only hook (maintenance door diagnostic)
    # ==================================================================
    def observe_consolidation(self, cyc: int, candidates: Any,
                              newly_coherent: int) -> None:
        """Called immediately AFTER the live reinforce_coherent_edges
        returns.  Only tallies the maintenance-door throughput as a
        diagnostic; the settlings it realizes are credited via the reinforce
        observer (origin is logged, not excluded).  READS ONLY."""
        try:
            self._maint_consolidation_newly += int(newly_coherent or 0)
        except Exception:
            pass

    # ==================================================================
    # per-tick entry point (end of brain.tick(); cheap + fully guarded)
    # ==================================================================
    def run(self, cyc: int) -> None:
        try:
            cyc = int(cyc)
            if not self._first_run_done:
                self._begin_life(cyc)
                self._first_run_done = True
            self._detect_death_revive(cyc)
            self._pertick(cyc)
            if cyc - self._last_sample_cyc >= self.sample_interval:
                self._last_sample_cyc = cyc
                self._episode_sample(cyc)
        except Exception as exc:
            self._log({'kind': 'error',
                       'cycle': int(cyc) if isinstance(cyc, (int, float))
                       else None, 'error': repr(exc)})

    # ------------------------------------------------------------------
    def _begin_life(self, cyc: int) -> None:
        """Anchor a new life: reset the per-life settle totals + the E EMA +
        the plateau anchor, then PRE-SEED-reconcile the ledger.  Cross-life
        bookkeeping (the settle ledger, revive recurrence, all-time settle
        split) PERSISTS — it tracks substrate-edge dynamics, which neither a
        restart nor a death wipes."""
        self.birth_cycle = int(cyc)
        self.E_settle = 0.0
        self._settle_life = _fresh_settle_buf()
        self._ticks_since_last_settling = 0
        # Adopt the drive's CURRENT death/revival counters as this life's
        # baseline.  Detection is for LIVE transitions only: the counters
        # captured at construction predate load_personality, so a save with
        # historical deaths>0 would otherwise register a PHANTOM death on
        # every boot — an empty-distillate record polluting the succession
        # chain (caught by the promotion smoke, 2026-07-18).
        md = self.mortality_drive
        self._prev_deaths = int(getattr(md, 'deaths', 0) or 0)
        self._prev_revivals = int(getattr(md, 'revivals', 0) or 0)
        # PRE-SEED reconciliation: mark everything ALREADY established, so
        # only genuinely-NEW post-birth settlings ever earn.
        self._preseed_settled(int(cyc))

    # ------------------------------------------------------------------
    def _preseed_settled(self, cyc: int) -> None:
        """Idempotent boot/life-begin PRE-SEED: mark every edge that is
        ALREADY ESTABLISHED as settled, WITHOUT crediting; count only
        newly-marked.

        WHY: with an empty ledger (fresh save / pre-ledger save), the first
        reinforce of an edge established in a PRIOR life would otherwise
        score as a fresh settling — the INHERITANCE ARTIFACT (re-observing
        his inherited mind is not new learning).  A clean persisted ledger
        makes this a near-no-op (tiny preseeded_count) — exactly the
        reconciliation semantics.

        HOW: iterate BOTH edge pools (substrate.edges + quarantine_edges),
        compute establishment with the SAME predicate the settling logic uses
        (fcc>0 OR effective_strength(cyc) >= SATURATION_CEILING) and mark the
        (s,r,t) digest in _settled using the SAME key scheme as on_reinforce.
        Credits NO earning — pure marking.  An established edge is by
        definition not faded, so it is cleared from _dormant.

        GUARDED: a failure logs and continues.  O(edges) once per life-begin
        — an accepted one-time cost."""
        marked = 0
        try:
            sub = _safe_call(self._substrate_provider)
            if sub is None:
                self._preseeded_count = 0
                return
            for pool_name in ('edges', 'quarantine_edges'):
                pool = getattr(sub, pool_name, None)
                if pool is None:
                    continue
                try:
                    edges_iter = (pool.values() if hasattr(pool, 'values')
                                  else pool)
                except Exception:
                    continue
                for edge in edges_iter:
                    try:
                        fcc = int(getattr(edge, 'first_coherent_cycle', 0)
                                  or 0)
                        try:
                            eff = float(edge.effective_strength(int(cyc)))
                        except Exception:
                            eff = float(getattr(edge, 'strength', 0.0) or 0.0)
                        established = (fcc > 0) or (eff >= SATURATION_CEILING)
                        if not established:
                            continue
                        # IDENTICAL derivation to on_reinforce — same
                        # method, no parallel key build.  An inherited
                        # family fact must pre-seed under the SAME digest
                        # the observer will look up, or it pays twice.
                        kh = self._fact_key(edge)
                        if kh not in self._settled:
                            if len(self._settled) < SETTLED_FACTS_CAP:
                                self._settled.add(kh)
                                marked += 1
                            else:
                                self._settled_cap_hit = True
                        # established ⇒ not faded: settled=True, dormant=False
                        self._dormant.discard(kh)
                    except Exception:
                        continue
            self._preseeded_count = int(marked)
        except Exception as exc:
            self._preseeded_count = int(marked)
            try:
                self._log({'kind': 'preseed_error',
                           'cycle': (int(cyc)
                                     if isinstance(cyc, (int, float))
                                     else None),
                           'preseeded_count': int(marked),
                           'error': repr(exc)})
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _detect_death_revive(self, cyc: int) -> None:
        md = self.mortality_drive
        deaths = int(getattr(md, 'deaths', 0) or 0)
        revivals = int(getattr(md, 'revivals', 0) or 0)
        if deaths != self._prev_deaths:
            self.deaths_observed += 1
            # SUCCESSION: distill the closing life BEFORE any reset.
            distillate = self._death_distillate(deaths)
            rec = self._life_event_record('death', cyc, deaths, revivals)
            rec['distillate'] = distillate
            self._log(rec)
            self._append_death_record(rec)
            self._prev_deaths = deaths
            # E→0, per-life split reset, pre-seed reconcile.  LEDGER PERSISTS
            # (knowledge is inherited, not survived).
            self.E_settle = 0.0
            self._settle_life = _fresh_settle_buf()
            self._preseed_settled(int(cyc))
        if revivals != self._prev_revivals:
            self.revives_observed += 1
            self._log(self._life_event_record('revive', cyc, deaths,
                                              revivals))
            self._prev_revivals = revivals
            # a revive is a NEW life — fresh birth anchor + per-life resets
            # (+ another idempotent pre-seed reconcile).
            self._begin_life(cyc)

    def _death_distillate(self, deaths_total: int) -> Dict[str, Any]:
        """The life's distillate — captured at death, before resets.  No
        consumer yet; just the record (succession doctrine)."""
        lf = _safe_call(self._lifeforce_provider)
        if lf is None:
            try:
                lf_get = getattr(self.mortality_drive, '_lf_get', None)
                lf = float(lf_get()) if lf_get is not None else None
            except Exception:
                lf = None
        return {
            'life_split': _round_settle_buf(self._settle_life),
            'E_at_death': round(float(self.E_settle), 12),
            'wall': self._live_wall(),
            'lifeforce': (None if lf is None else float(lf)),
            'ledger_size': int(len(self._settled)),
            'dormant_count': int(len(self._dormant)),
            'deaths_total': int(deaths_total),
        }

    def _life_event_record(self, kind: str, cyc: int,
                           deaths: int, revivals: int) -> Dict[str, Any]:
        return {
            'kind': kind,
            'cycle': int(cyc),
            'deaths': int(deaths),
            'revivals': int(revivals),
            'E_at_event': round(float(self.E_settle), 12),
            'wall_at_event': self._live_wall(),
        }

    # ==================================================================
    # per-tick: drain earn → E (EMA); the DRIVE integrates the wall
    # ==================================================================
    def _pertick(self, cyc: int) -> None:
        earn_settle = self._tick_settle_earn
        settlings = self._settlings_this_tick
        # drain for next tick
        self._tick_settle_earn = 0.0
        self._settlings_this_tick = 0
        self._settle_tick = _fresh_settle_buf()
        self._last_earn_settle = earn_settle
        self._last_settlings = settlings

        # E = EMA_λ(earn_tick), λ = MORTALITY_RELAX_RATE.
        self.E_settle += MORTALITY_RELAX_RATE * (earn_settle - self.E_settle)

        # refresh the cf read (also caches for the episode log).
        self.cf()

        # plateau tracking: a settling resets the stretch; else the stretch
        # grows (the wall creeps with no facts settling — by design).
        if settlings > 0:
            self._ticks_since_last_settling = 0
        else:
            self._ticks_since_last_settling += 1

    # ==================================================================
    # per-episode: write a full log line
    # ==================================================================
    def _episode_sample(self, cyc: int) -> None:
        gaps = list(self._recurrence_gaps)
        rec = {
            'kind': 'episode',
            'model': 'settling_rate',
            'cycle': int(cyc),
            'birth_cycle': int(self.birth_cycle or 0),
            'age': int(cyc - (self.birth_cycle or cyc)),
            'deaths': int(getattr(self.mortality_drive, 'deaths', 0) or 0),
            'revivals': int(getattr(self.mortality_drive, 'revivals', 0)
                            or 0),
            'clock': {
                'D': EDGE_STRENGTH_DECAY_PER_CYCLE,
                'obstruction_ema': round(self._obstruction_ema(), 10),
                'cf': (None if self._cf_last is None
                       else round(self._cf_last, 6)),
                'cf_eligible': int(self._eligible_last),
                'lambda': MORTALITY_RELAX_RATE,
                'saturation_ceiling': SATURATION_CEILING,
                'E': round(self.E_settle, 12),
                'earn_settle_last_tick': round(self._last_earn_settle, 12),
                'last_wall_advance': float(getattr(
                    self.mortality_drive, '_last_wall_advance', 0.0) or 0.0),
            },
            'wall': self._live_wall(),
            'lifeforce': (None if _safe_call(self._lifeforce_provider) is None
                          else float(_safe_call(self._lifeforce_provider))),
            'settle': {
                'credited_E': 'every settling-Δ (all origins); once per fact',
                'settlings_this_tick': int(self._last_settlings),
                'cumulative_distinct_settled':
                    self._cumulative_distinct_counts(),
                'cumulative_split': _round_settle_buf(self._settle_alltime),
                'life_split': _round_settle_buf(self._settle_life),
                'reinforces_seen': int(self.reinforces_seen),
                'observer_registered': bool(self._observer_registered),
                'observer_errors': int(reinforce_observer_errors()),
                'fact_key_failopen': int(self._fact_key_failopen),
                'key_digest': KEY_DIGEST,
                'preseeded_count': int(self._preseeded_count),
                'settled_ledger_size': int(len(self._settled)),
                'dormant_now': int(len(self._dormant)),
                'settled_cap_hit': bool(self._settled_cap_hit),
                'composable_set_available':
                    bool(_composable_result_relations() is not None),
                'origin_method': 'threaded_kwarg',
            },
            'plateau': {
                'ticks_since_last_settling':
                    int(self._ticks_since_last_settling),
            },
            'revive_fade': {
                'triples_revived': int(self._triples_revived),
                'triples_revived_ge2': int(sum(
                    1 for v in self._triple_revive_count.values() if v >= 2)),
                'fast_fade_hits_sub_threshold': int(self._fast_fade_hits),
                'threshold_cycles': FAST_FADE_RECURRENCE_CYCLES,
                'min_recurrence_gap': (min(gaps) if gaps else None),
                'recent_gaps_sample': gaps[-10:],
            },
            'census': {
                'write_census': dict(self._write_census),
                'thought_census': dict(self._thought_census),
                'maint_consolidation_newly': int(
                    self._maint_consolidation_newly),
            },
        }
        self.last_report = rec
        self.episodes_logged += 1
        self._log(rec)

    def _cumulative_distinct_counts(self) -> Dict[str, int]:
        """Per-class cumulative counts (summed over origin).  The three
        first-settle classes count DISTINCT facts (each once); revival counts
        re-settle EVENTS."""
        return {cls: int(sum(self._settle_alltime[cls][oc]['n']
                             for oc in _ORIGIN_CLASSES))
                for cls in _SETTLE_CLASSES}

    # ==================================================================
    # read-only live signal helpers
    # ==================================================================
    def _live_wall(self) -> Optional[float]:
        if self._live_wall_provider is not None:
            v = _safe_call(self._live_wall_provider)
            return None if v is None else float(v)
        w = getattr(self.mortality_drive, 'wall', None)
        return None if w is None else float(w)

    def _obstruction_ema(self) -> float:
        try:
            return max(0.0, float(
                getattr(self.mortality_drive, 'obstruction_ema', 0.0) or 0.0))
        except Exception:
            return 0.0

    def _clearance_failure(self) -> Optional[float]:
        sub = _safe_call(self._substrate_provider)
        if sub is None:
            self._eligible_last = 0
            return None
        eligible = getattr(sub, '_last_fade_backlog', None)
        if eligible is None:
            self._eligible_last = 0
            return None
        eligible = int(eligible)
        self._eligible_last = eligible
        if eligible <= 0:
            return 0.0
        qcap = _safe_call(self._qcap_provider)
        if qcap is None:
            return None
        unswept = max(0, eligible - int(qcap))
        return unswept / float(eligible)

    # ==================================================================
    # persistence
    # ==================================================================
    def to_dict(self) -> Dict[str, Any]:
        """Persist the cross-life ledger + all-time split + revive probe.
        E / per-life split / census are per-boot transients — the boot
        pre-seed + a fresh EMA rebuild them (E resets at every life-begin
        anyway)."""
        return {
            'schema': PERSIST_SCHEMA,
            'key_digest': KEY_DIGEST,
            'settled': [int(k) for k in self._settled],
            'dormant': [int(k) for k in self._dormant],
            'settle_alltime': _round_settle_buf(self._settle_alltime),
            'triple_revive': [
                [int(k), int(v),
                 int(self._triple_last_revive_cyc.get(k, 0))]
                for k, v in self._triple_revive_count.items()],
            'fast_fade_hits': int(self._fast_fade_hits),
        }

    def load_dict(self, state: Dict[str, Any]) -> None:
        """Tolerant restore.  Absence / mismatch → empty ledger (the boot
        pre-seed bootstraps).  A key_digest other than blake2b8 means the
        persisted digests don't line up with this keying — start fresh."""
        if not isinstance(state, dict):
            return
        if str(state.get('key_digest', KEY_DIGEST)) != KEY_DIGEST:
            return
        try:
            settled = state.get('settled')
            if isinstance(settled, (list, tuple)):
                self._settled = {int(k) for k in settled}
            dormant = state.get('dormant')
            if isinstance(dormant, (list, tuple)):
                self._dormant = {int(k) for k in dormant}
                # dormant ⊆ settled by construction; enforce on load.
                self._dormant &= self._settled
            self._settle_alltime = _load_settle_buf(
                state.get('settle_alltime'))
            tr = state.get('triple_revive')
            if isinstance(tr, (list, tuple)):
                self._triple_revive_count = {}
                self._triple_last_revive_cyc = {}
                for row in tr:
                    try:
                        kh, cnt, last_cyc = (int(row[0]), int(row[1]),
                                             int(row[2]))
                    except (TypeError, ValueError, IndexError):
                        continue
                    self._triple_revive_count[kh] = cnt
                    self._triple_last_revive_cyc[kh] = last_cyc
                self._triples_revived = len(self._triple_revive_count)
            try:
                self._fast_fade_hits = int(state.get('fast_fade_hits', 0)
                                           or 0)
            except (TypeError, ValueError):
                self._fast_fade_hits = 0
        except Exception:
            # Tolerant: a partially-restored ledger is still consistent
            # enough (pre-seed reconciles the rest at next life-begin).
            pass

    # ==================================================================
    def stats(self) -> Dict[str, Any]:
        counts = self._cumulative_distinct_counts()

        def _origin_total(oc):
            return round(sum(self._settle_alltime[cls][oc]['delta_sum']
                             for cls in _SETTLE_CLASSES), 12)
        return {
            'model': 'settling_rate',
            'birth_cycle': self.birth_cycle,
            'episodes_logged': int(self.episodes_logged),
            'wall': self._live_wall(),
            'E': round(self.E_settle, 12),
            'cf_last': (None if self._cf_last is None
                        else round(self._cf_last, 6)),
            'reinforces_seen': int(self.reinforces_seen),
            'observer_registered': bool(self._observer_registered),
            'observer_errors': int(reinforce_observer_errors()),
            # Grain health: non-zero means _fact_key could not read
            # structure and fell back to instance grain (fail-toward-
            # credit).  Watch it exactly like observer_errors.
            'fact_key_failopen': int(self._fact_key_failopen),
            'key_digest': KEY_DIGEST,
            'preseeded_count': int(self._preseeded_count),
            'settlings_last_tick': int(self._last_settlings),
            'settled_ledger_size': int(len(self._settled)),
            'dormant_now': int(len(self._dormant)),
            'cumulative_distinct_settled': counts,
            'earn_delta_sum_cognition': _origin_total(_COGNITION),
            'earn_delta_sum_maintenance': _origin_total(_MAINTENANCE),
            'earn_delta_sum_unknown': _origin_total(_UNKNOWN),
            'triples_revived': int(self._triples_revived),
            'fast_fade_hits_sub_threshold': int(self._fast_fade_hits),
            'ticks_since_last_settling': int(self._ticks_since_last_settling),
            'write_census': dict(self._write_census),
            'deaths_observed': int(self.deaths_observed),
            'revives_observed': int(self.revives_observed),
            'record_guard_blocked': int(self.record_guard_blocked),
        }

    # ==================================================================
    # logging (own files — NOT the personality/substrate save)
    # ==================================================================
    def _record_guard_blocks(self, path: str) -> bool:
        """CONTAMINATION GUARD (2026-07-22): a BLANK-SUBSTRATE clock must never
        write into the one Seagi's records.  Three times a test process that
        booted the runtime without SEAGI_CLOCK_LOG set fell back to the
        absolute default and appended birth_cycle:1 episodes into the live
        record -- corrupting the very instrument the flip gates are read from.
        The env-var wrapper cannot close this (a test can unset the var), so
        the writer itself refuses.  A real life is minted at a large
        accumulated cycle, so this can never fire for the live daemon; the
        block is COUNTED (surfaced in stats), never silent."""
        bc = self.birth_cycle
        if bc is None or int(bc) > BLANK_SUBSTRATE_BIRTH_CYCLE:
            return False
        if not str(path).startswith(ONE_SEAGI_RECORD_DIR):
            return False
        self.record_guard_blocked += 1
        return True

    def _log(self, rec: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        if self._record_guard_blocks(self.log_path):
            return
        try:
            line = json.dumps(rec, default=str)
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
            self._rotate()
        except Exception:
            pass

    def _append_death_record(self, rec: Dict[str, Any]) -> None:
        """Append the death record (with distillate) to the death-records
        file (created if absent).  NEVER rotated — deaths are rare and the
        succession chain must stay complete."""
        if not self.death_log_path:
            return
        if self._record_guard_blocks(self.death_log_path):
            return
        try:
            line = json.dumps(rec, default=str)
            with open(self.death_log_path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:
            pass

    def _rotate(self) -> None:
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) > LOG_LINE_CAP:
                with open(self.log_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines[-LOG_LINE_CAP:])
        except Exception:
            pass
