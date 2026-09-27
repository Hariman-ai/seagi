"""RoleRegularityShadow — a read-only STRUCTURAL-role instrument (SHADOW).

Pattern: GroundingLoop's precision block.  It OWNS its state, computes, writes
PROVISIONAL edges, logs, and DRIVES NOTHING.  It is NOT bus-subscribed and has
NO per-tick hook; the runtime calls ``run_pass(cyc)`` once per sleep-
consolidation pass, immediately AFTER form_abstractions / form_analogies and
INSIDE the ``is_asleep()`` + ``SLEEP_CONSOLIDATION_INTERVAL`` gate.

What it measures
----------------
Every world-state the WorldActor has touched is given four integer STRUCTURAL
ROLES, read straight off the actor's own map (no new signal invented):

  * ``dist``   — route-distance bucket ``d ~ round(log(route[s]) / log(0.9))``.
                 The success value-gradient the actor propagates is ``0.9**d``,
                 so ``d`` is "how many moves from the goal".  No route -> 'nr'.
  * ``indeg``  — how many ``(s', a)`` transitions LEAD INTO ``s`` (a convergence
                 point in his learned map).
  * ``outdeg`` — how many distinct actions have a known effect FROM ``s`` (a
                 branch point).
  * ``visit``  — how many times ``s`` has been reached (raw ``_visits`` count).

For each family it writes provisional ``s --has_role--> _role_{family}_{bucket}``
edges (strength 0.1) through the single writer, and asks ONE question: do these
purely STRUCTURAL role-classes PREDICT the actor's VALUE map (``value_of``,
primary) and route (secondary)?  It answers with a one-way-ANOVA effect size
(eta^2) against a label-permutation null (K=50): real eta^2, null mean, null
p95, z.  ``dist -> Y_route`` is a POSITIVE CONTROL (route == 0.9**d by
construction, so eta^2 there should be high); the informative cells are
``indeg / outdeg / visit -> Y_val``.

Anti-contamination (why writing into the live graph is still safe)
------------------------------------------------------------------
``has_role`` is absent from ``RELATION_COMPOSITION`` (it can never cohere ->
never credits lifeforce -> never reaches ``record_learning``) and is traversed
by NEITHER ``grounding._predict`` (which walks only ``transitions_to`` and is_a
siblings) NOR ``is_a_siblings``.  The single path that could turn a ``has_role``
edge into a traversable is_a class — the live ``form_abstractions`` — is closed
by an explicit ``if r == ROLE_RELATION: continue`` guard beside the existing
``ABSTRACTION_RELATION`` guard.  So the instrument writes into the graph yet
cannot change what the mind predicts, credits, or does.  Role edges are
provisional 0.1, recomputed each pass from the live ``_route`` / ``_trans`` /
``_visits``; stable roles are re-attested, stale ones decay and settle out
(earn-or-dissolve).
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..events import EventKind, SubstrateWriteQueuedEvent
from seagi.core.substrate import (PROVISIONAL_EDGE_STRENGTH, ROLE_RELATION,
                                  ABSTRACTION_MIN_GROUP)
from .world_actor import PATH_CREDIT_DECAY


# K label-permutations for the chance baseline (preserving class-size dist).
NULL_PERMUTATIONS = 50
# Rotate/cap the shadow log at ~10k lines (keep the most recent).
LOG_LINE_CAP = 10000
# The structural-role families this instrument buckets each state into.
ROLE_FAMILIES = ('dist', 'indeg', 'outdeg', 'visit')


class RoleRegularityShadow:
    """Read-only structural-role regularity instrument.  DRIVES NOTHING."""

    def __init__(self,
                 bus: Any,
                 world_actor: Any,
                 value_landscape: Any,
                 log_path: str = 'shadow_role_metric.jsonl',
                 min_group: int = ABSTRACTION_MIN_GROUP):
        self.bus = bus
        self.world_actor = world_actor
        self.value_landscape = value_landscape
        self.log_path = log_path
        self.min_group = int(min_group)
        # read-only diagnostics
        self.passes: int = 0
        self.role_edges_written: int = 0          # cumulative across passes
        self.last_report: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # read the actor's own map (READ-ONLY snapshot)
    # ------------------------------------------------------------------
    def _gather(self):
        wa = self.world_actor
        visits = dict(getattr(wa, '_visits', {}) or {})
        route = dict(getattr(wa, '_route', {}) or {})
        trans = dict(getattr(wa, '_trans', {}) or {})
        # invert _trans ONCE per pass: in-degree per target, out-degree per src.
        indeg: Dict[Any, int] = defaultdict(int)
        outdeg: Dict[Any, int] = defaultdict(int)
        targets = set()
        for (s, a), nxt in trans.items():
            indeg[nxt] += 1
            outdeg[s] += 1
            targets.add(nxt)
        # domain (per spec) = visited states | routed states | transition targets
        domain = set(visits) | set(route) | targets
        return domain, route, indeg, outdeg, visits

    def _bucket_state(self, s, route, indeg, outdeg, visits) -> Dict[str, Any]:
        # dist: route[s] = 0.9**d (value spread back from the goal) -> d moves
        # from the goal.  No route -> 'nr' (not on any known route).
        r = route.get(s, 0.0)
        if r and r > 0.0:
            try:
                d: Any = int(round(math.log(r) / math.log(PATH_CREDIT_DECAY)))
            except (ValueError, ZeroDivisionError):
                d = 'nr'
        else:
            d = 'nr'
        return {
            'dist': d,
            'indeg': int(indeg.get(s, 0)),
            'outdeg': int(outdeg.get(s, 0)),
            'visit': int(visits.get(s, 0)),
        }

    @staticmethod
    def _role_node(family: str, bucket: Any) -> str:
        return f'_role_{family}_{bucket}'

    def _value_of(self, s) -> float:
        try:
            return float(self.value_landscape.value_of(s))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # the pass
    # ------------------------------------------------------------------
    def run_pass(self, cyc: int) -> Dict[str, Any]:
        # WHOLE-PASS GATE (2026-08-23).  This is a K=50 permutation
        # ANOVA over every state the actor has touched, run once per
        # sleep-consolidation pass.  QUARVIS_ON took that domain from
        # 10,486 to 30,653 states and a clean sleep window measured
        # 0.000 ticks/s.  The instrument drives nothing and its result
        # is already extracted, so the pass is pure cost.
        try:
            import os as _os
            if _os.path.exists("/root/ROLESHADOW_OFF"):
                self.skipped = getattr(self, "skipped", 0) + 1
                return {"skipped": True,
                        "reason": "ROLESHADOW_OFF",
                        "passes_skipped": self.skipped}
        except Exception:
            pass
        cyc = int(cyc)
        self.passes += 1
        domain, route, indeg, outdeg, visits = self._gather()
        # Deterministic order (independent of PYTHONHASHSEED) so the null
        # permutation is reproducible from `cyc` alone.
        states: List[Any] = sorted(domain, key=str)

        # per-family integer bucket labels
        labels: Dict[str, Dict[Any, Any]] = {f: {} for f in ROLE_FAMILIES}
        for s in states:
            bk = self._bucket_state(s, route, indeg, outdeg, visits)
            for fam in ROLE_FAMILIES:
                labels[fam][s] = bk[fam]

        # outcomes over the domain
        y_val = {s: self._value_of(s) for s in states}
        y_route = {s: float(route.get(s, 0.0)) for s in states}

        # --- write provisional has_role edges (one per state per family) ---
        active_nodes = set()
        edges_written = 0
        for fam in ROLE_FAMILIES:
            fam_labels = labels[fam]
            for s in states:
                node = self._role_node(fam, fam_labels[s])
                active_nodes.add(node)
                self._write_role_edge(s, node, cyc)
                edges_written += 1
        self.role_edges_written += edges_written

        # --- metric per family: eta^2 vs a label-permutation null ---
        perm_rng = random.Random(cyc ^ 0x5EA61C)
        families_report: List[Dict[str, Any]] = []
        for fam in ROLE_FAMILIES:
            fam_labels = labels[fam]
            sizes: Dict[Any, int] = defaultdict(int)
            for s in states:
                sizes[fam_labels[s]] += 1
            # a recurring role-CLASS = a bucket with in-degree >= min_group
            n_classes = sum(1 for c in sizes.values() if c >= self.min_group)
            ev, nmv, p95v, zv = self._anova_with_null(
                states, fam_labels, y_val, perm_rng)
            er, nmr, p95r, zr = self._anova_with_null(
                states, fam_labels, y_route, perm_rng)
            families_report.append({
                'family': fam,
                'n_classes': n_classes,
                'class_sizes': dict(sorted(
                    sizes.items(), key=lambda kv: (-kv[1], str(kv[0])))),
                'class_value_top': self._top_class_value(
                    states, fam_labels, y_val, sizes),
                'eta2_val': round(ev, 6),
                'null_mean_val': round(nmv, 6),
                'null_p95_val': round(p95v, 6),
                'z_val': round(zv, 4),
                'eta2_route': round(er, 6),
                'null_mean_route': round(nmr, 6),
                'null_p95_route': round(p95r, 6),
                'z_route': round(zr, 4),
            })

        report = {
            'cycle': cyc,
            'n_states': len(states),
            'shadow_staged': True,
            'families': families_report,
            'role_edges_written': edges_written,
            'role_nodes_active': len(active_nodes),
            'role_write_gated': __import__('os').path.exists('/root/ROLEWRITE_OFF'),
        }
        self.last_report = report
        self._log(report)
        return report

    # ------------------------------------------------------------------
    # writing (through the single writer, via the bus)
    # ------------------------------------------------------------------
    def _write_role_edge(self, subject, obj, cyc: int) -> None:
        # GATED 2026-08-23: these edges are 42.0% of the live
        # substrate and have NO reader -- seven code sites exist
        # solely to exclude them, and has_role appears zero times
        # in RELATION_COMPOSITION.  run_pass recomputes roles from
        # live _route/_trans/_visits and never reads these back, so
        # the instrument is unaffected: only the STORAGE stops.
        try:
            import os as _os
            if _os.path.exists('/root/ROLEWRITE_OFF'):
                return
        except Exception:
            pass
        if self.bus is None:
            return
        try:
            self.bus.publish(SubstrateWriteQueuedEvent(
                kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                cycle=int(cyc),
                source_capability='role_shadow',
                origin='internal',
                origin_detail='role_regularity_shadow',
                subject=str(subject),
                relation=ROLE_RELATION,
                object=str(obj),
                strength=PROVISIONAL_EDGE_STRENGTH,
                write_reason='role_shadow'))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------
    def _anova_with_null(self, states, fam_labels, y, rng
                         ) -> Tuple[float, float, float, float]:
        ys = [y[s] for s in states]
        groups: Dict[Any, List[float]] = defaultdict(list)
        for s in states:
            groups[fam_labels[s]].append(y[s])
        real = self._eta2(groups, ys)
        # NULL: permute the label list across states (preserves the class-size
        # distribution exactly — it is the same multiset of labels, reassigned).
        label_list = [fam_labels[s] for s in states]
        nulls: List[float] = []
        for _ in range(NULL_PERMUTATIONS):
            perm = label_list[:]
            rng.shuffle(perm)
            g: Dict[Any, List[float]] = defaultdict(list)
            for lbl, v in zip(perm, ys):
                g[lbl].append(v)
            nulls.append(self._eta2(g, ys))
        if nulls:
            nmean = sum(nulls) / len(nulls)
            var = sum((x - nmean) ** 2 for x in nulls) / len(nulls)
            nstd = math.sqrt(var)
        else:
            nmean = nstd = 0.0
        p95 = self._percentile(nulls, 95.0)
        z = (real - nmean) / nstd if nstd > 1e-12 else 0.0
        return real, nmean, p95, z

    @staticmethod
    def _eta2(groups, ys) -> float:
        n = len(ys)
        if n == 0:
            return 0.0
        grand = sum(ys) / n
        ss_total = sum((v - grand) ** 2 for v in ys)
        if ss_total <= 1e-12:
            return 0.0
        ss_between = 0.0
        for vals in groups.values():
            m = len(vals)
            if m == 0:
                continue
            gm = sum(vals) / m
            ss_between += m * (gm - grand) ** 2
        return max(0.0, min(1.0, ss_between / ss_total))

    @staticmethod
    def _percentile(values, p: float) -> float:
        if not values:
            return 0.0
        xs = sorted(values)
        if len(xs) == 1:
            return xs[0]
        k = (len(xs) - 1) * (p / 100.0)
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return xs[lo]
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    def _top_class_value(self, states, fam_labels, y_val, sizes
                         ) -> Optional[Dict[str, Any]]:
        sums: Dict[Any, float] = defaultdict(float)
        for s in states:
            sums[fam_labels[s]] += y_val[s]
        best = None
        for bucket, cnt in sizes.items():
            if cnt < self.min_group:
                continue
            mean_v = sums[bucket] / cnt
            if best is None or mean_v > best['mean_val']:
                best = {'bucket': str(bucket),
                        'size': int(cnt),
                        'mean_val': round(mean_v, 6)}
        return best

    # ------------------------------------------------------------------
    # logging (own file, NOT the personality/substrate save)
    # ------------------------------------------------------------------
    def _log(self, report: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        try:
            line = json.dumps(report, default=str)
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
            self._rotate()
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
