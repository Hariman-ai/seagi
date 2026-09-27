"""RoleRegularityShadow — the SHADOW structural-role instrument test battery.

Pins the instrument's three behaviours (metric validity, provisional-edge
writing, earn-or-dissolve stale decay) AND the three SAFETY-CRITICAL falsifiers
(propose byte-identical, record_learning call-count unchanged, grounding
._predict unchanged with has_role edges present — the anti-poison guard holds).
"""

from __future__ import annotations

import os
import tempfile
import types
import unittest

from seagi.core.substrate import (
    Substrate, ROLE_RELATION, ABSTRACTION_RELATION, ANALOGY_RELATION,
    ABSTRACTION_NAME_PREFIX, PROVISIONAL_EDGE_STRENGTH,
)
from seagi.brain.bus import EventBus
from seagi.brain.capabilities.writer import JournaledSubstrateWriter
from seagi.brain.capabilities.role_regularity_shadow import (
    RoleRegularityShadow, ROLE_FAMILIES,
)
from seagi.brain.capabilities.world_actor import WorldActor, PATH_CREDIT_DECAY
from seagi.brain.capabilities.grounding import GroundingLoop, WORLD_RELATION
from seagi.world.goal_world import GoalWorld
from seagi.brain.capabilities.world_transducer import WorldTransducer


# --------------------------------------------------------------------------
# fakes / rigs
# --------------------------------------------------------------------------
class _CollectBus:
    """Collects published events without dispatching (for metric-only tests)."""

    def __init__(self):
        self.events = []

    def publish(self, ev):
        self.events.append(ev)


class _DeliverBus:
    """Delivers published writes to the single writer (end-to-end contract)."""

    def __init__(self, writer):
        self.writer = writer
        self.published = []

    def publish(self, event):
        self.published.append(event)
        self.writer.handle(event, self)


class _FakeWA:
    """Minimal stand-in exposing only the three maps the instrument reads."""

    def __init__(self, visits=None, route=None, trans=None):
        self._visits = dict(visits or {})
        self._route = dict(route or {})
        self._trans = dict(trans or {})


class _FakeValue:
    def __init__(self, d=None):
        self.d = dict(d or {})

    def value_of(self, tok):
        return self.d.get(tok, 0.0)


def _tmp_log():
    fd, path = tempfile.mkstemp(suffix='.jsonl')
    os.close(fd)
    return path


def _family(report, name):
    for f in report['families']:
        if f['family'] == name:
            return f
    raise AssertionError(f'family {name} not in report')


# --------------------------------------------------------------------------
# (1) metric: planted role->value coupling detected; scrambled at chance
# --------------------------------------------------------------------------
class TestMetricValidity(unittest.TestCase):

    def test_planted_visit_coupling_above_null_p95(self):
        # 30 states in 3 visit-buckets {1,5,9}; value == visit-bucket.
        visits = {f's{i}': [1, 5, 9][i // 10] for i in range(30)}
        value = {f's{i}': [0.0, 0.5, 1.0][i // 10] for i in range(30)}
        wa = _FakeWA(visits=visits)
        sh = RoleRegularityShadow(_CollectBus(), wa, _FakeValue(value),
                                  log_path=_tmp_log())
        rep = sh.run_pass(cyc=7)
        fam = _family(rep, 'visit')
        # perfect separation => eta^2 ~ 1 and WAY above the permutation null.
        self.assertGreater(fam['eta2_val'], fam['null_p95_val'])
        self.assertGreater(fam['eta2_val'], 0.9)
        self.assertGreater(fam['z_val'], 3.0)
        self.assertEqual(fam['n_classes'], 3)

    def test_scrambled_coupling_at_chance(self):
        # same visit buckets, but value is INDEPENDENT of the bucket
        # (value cycles by i%3 inside each visit-bucket of 10 states).
        visits = {f's{i}': [1, 5, 9][i // 10] for i in range(30)}
        value = {f's{i}': [0.0, 0.5, 1.0][i % 3] for i in range(30)}
        wa = _FakeWA(visits=visits)
        sh = RoleRegularityShadow(_CollectBus(), wa, _FakeValue(value),
                                  log_path=_tmp_log())
        rep = sh.run_pass(cyc=7)
        fam = _family(rep, 'visit')
        # no real structure -> real eta^2 does NOT exceed the null p95.
        self.assertLessEqual(fam['eta2_val'], fam['null_p95_val'])
        self.assertLess(fam['z_val'], 2.0)

    def test_dist_route_positive_control(self):
        # route[s] = 0.9**d by construction -> dist family PERFECTLY predicts
        # Y_route (the documented positive control).
        route = {}
        for d in range(4):
            for j in range(5):
                route[f'd{d}_{j}'] = PATH_CREDIT_DECAY ** d
        wa = _FakeWA(route=route)
        sh = RoleRegularityShadow(_CollectBus(), wa, _FakeValue({}),
                                  log_path=_tmp_log())
        rep = sh.run_pass(cyc=3)
        fam = _family(rep, 'dist')
        self.assertGreater(fam['eta2_route'], fam['null_p95_route'])
        self.assertGreater(fam['eta2_route'], 0.9)


# --------------------------------------------------------------------------
# (2) role edges written at 0.1 via the writer journal; absent from is_a
# --------------------------------------------------------------------------
class TestProvisionalEdgeWriting(unittest.TestCase):

    def test_edges_are_has_role_at_provisional_strength_not_is_a(self):
        bus = EventBus()
        writer = JournaledSubstrateWriter(engine=None, memory_mode=True)
        bus.subscribe(writer.SUBSCRIPTIONS, writer)
        wa = _FakeWA(
            visits={'a': 2, 'b': 2, 'c': 3},
            route={'a': 1.0, 'b': 0.9},
            trans={('a', 0): 'b', ('b', 1): 'c', ('a', 1): 'c'})
        sh = RoleRegularityShadow(bus, wa, _FakeValue({'a': 0.5}),
                                  log_path=_tmp_log())
        rep = sh.run_pass(cyc=100)

        self.assertGreater(rep['role_edges_written'], 0)
        self.assertTrue(writer.memory_edges)
        for (subj, rel, obj), strength in writer.memory_edges.items():
            self.assertEqual(rel, ROLE_RELATION)          # 'has_role'
            self.assertNotEqual(rel, ABSTRACTION_RELATION)  # never 'is_a'
            self.assertTrue(obj.startswith('_role_'))
            self.assertAlmostEqual(strength, PROVISIONAL_EDGE_STRENGTH, 6)
        # journal records the shadow provenance
        by_source = dict(writer.by_source)
        self.assertEqual(by_source.get('role_shadow', 0),
                         rep['role_edges_written'])


# --------------------------------------------------------------------------
# (3) a state whose role changes leaves a decaying (un-re-attested) stale edge
# --------------------------------------------------------------------------
class TestEarnOrDissolveStaleDecay(unittest.TestCase):

    def test_role_change_leaves_decaying_stale_edge(self):
        sub = Substrate()
        engine = types.SimpleNamespace(substrate=sub)
        writer = JournaledSubstrateWriter(engine=engine)
        bus = _DeliverBus(writer)
        wa = _FakeWA(visits={'X': 2})
        sh = RoleRegularityShadow(bus, wa, _FakeValue({}), log_path=_tmp_log())

        sh.run_pass(cyc=10)                    # writes X -> _role_visit_2 @c10
        wa._visits['X'] = 6                    # the role changes
        sh.run_pass(cyc=100)                   # writes X -> _role_visit_6 @c100

        stale = ('X', ROLE_RELATION, '_role_visit_2')
        fresh = ('X', ROLE_RELATION, '_role_visit_6')
        stable = ('X', ROLE_RELATION, '_role_indeg_0')
        self.assertIn(stale, sub.edges)
        self.assertIn(fresh, sub.edges)
        # stale role was NOT re-attested in pass 2 (left to decay);
        # the current role WAS just written; a stable role was re-attested.
        self.assertEqual(sub.edges[stale].last_reinforced_cycle, 10)
        self.assertEqual(sub.edges[fresh].last_reinforced_cycle, 100)
        self.assertEqual(sub.edges[stable].last_reinforced_cycle, 100)
        # => at a later cycle the un-re-attested stale role is weaker (decaying)
        later = 400
        self.assertLess(sub.edges[stale].effective_strength(later),
                        sub.edges[fresh].effective_strength(later))
        self.assertLess(sub.edges[stale].effective_strength(later),
                        PROVISIONAL_EDGE_STRENGTH)


# --------------------------------------------------------------------------
# shared rig for the falsifiers (real substrate + grounding + world)
# --------------------------------------------------------------------------
def _make_actor_rig(credit_sink=None):
    sub = Substrate()
    sub._quarantine_migrated = True
    engine = types.SimpleNamespace(substrate=sub)
    writer = JournaledSubstrateWriter(engine=engine)
    bus = _DeliverBus(writer)
    grounding = GroundingLoop(engine=engine, bus=bus)
    world = GoalWorld(grid=4, seed=1, stochastic=False, mastery_threshold=1)
    tx = WorldTransducer()
    wa = WorldActor(
        bus, world, tx, grounding,
        value_provider=lambda: _FakeValue({}),
        credit_learning=(credit_sink if credit_sink is not None else None),
        cycle_provider=lambda: 0, seed=0)
    return sub, bus, grounding, world, tx, wa


def _run(wa, n, shadow=None):
    actions = []
    for step in range(n):
        a = wa.propose(cyc=step)
        actions.append(a)
        wa.execute(a, cyc=step)
        if shadow is not None:
            shadow.run_pass(cyc=step)
    return actions


# --------------------------------------------------------------------------
# (4) FALSIFIER: propose output is BYTE-IDENTICAL with the instrument on/off
# --------------------------------------------------------------------------
class TestFalsifierProposeIdentical(unittest.TestCase):

    def test_propose_sequence_identical_with_instrument(self):
        # baseline: no instrument
        _s1, _b1, _g1, _w1, _t1, wa1 = _make_actor_rig()
        actions_off = _run(wa1, 60)

        # identical rig, but the instrument runs every step (writes has_role
        # edges into the SAME substrate grounding._predict reads from)
        _s2, bus2, _g2, _w2, _t2, wa2 = _make_actor_rig()
        shadow = RoleRegularityShadow(bus2, wa2, _FakeValue({}),
                                      log_path=_tmp_log())
        actions_on = _run(wa2, 60, shadow=shadow)

        self.assertEqual(actions_off, actions_on)      # byte-identical
        self.assertGreater(shadow.role_edges_written, 0)  # instrument DID write


# --------------------------------------------------------------------------
# (5) FALSIFIER: mortality record_learning call-count UNCHANGED (zero lifeforce)
# --------------------------------------------------------------------------
class TestFalsifierNoLifeforce(unittest.TestCase):

    def test_record_learning_call_count_unchanged(self):
        credited_off = []
        _s1, _b1, _g1, _w1, _t1, wa1 = _make_actor_rig(
            credit_sink=lambda g: credited_off.append(g))
        _run(wa1, 500)

        credited_on = []
        _s2, bus2, _g2, _w2, _t2, wa2 = _make_actor_rig(
            credit_sink=lambda g: credited_on.append(g))
        shadow = RoleRegularityShadow(bus2, wa2, _FakeValue({}),
                                      log_path=_tmp_log())
        _run(wa2, 500, shadow=shadow)

        # the instrument adds ZERO record_learning calls
        self.assertEqual(len(credited_off), len(credited_on))
        # and the scenario actually exercises mastery (so the test is real)
        self.assertGreater(len(credited_off), 0)
        self.assertGreater(shadow.role_edges_written, 0)


# --------------------------------------------------------------------------
# (6) FALSIFIER: grounding._predict UNCHANGED with has_role edges present
#     (the form_abstractions anti-poison guard holds)
# --------------------------------------------------------------------------
class TestFalsifierNoPredictPoison(unittest.TestCase):

    def test_has_role_edges_never_become_is_a_siblings(self):
        sub = Substrate()
        sub._quarantine_migrated = True
        engine = types.SimpleNamespace(substrate=sub)
        writer = JournaledSubstrateWriter(engine=engine)
        bus = _DeliverBus(writer)
        loop = GroundingLoop(engine=engine, bus=bus)

        # b1,b2,b3 -> Y (own transitions); q has NOTHING (predict = None).
        for m in ('b1', 'b2', 'b3'):
            sub.add_edge(m, 'Y', WORLD_RELATION, strength=0.6, cycle=5)

        PC = 100
        base_q = loop._predict('q', cycle=PC)
        base_b = {m: loop._predict(m, cycle=PC) for m in ('b1', 'b2', 'b3')}
        self.assertEqual(base_q, (None, ''))          # q predicts nothing

        # Plant a shared structural role on b1,b2,b3 AND q (4 >= min_group).
        for m in ('b1', 'b2', 'b3', 'q'):
            sub.add_edge(m, '_role_visit_3', ROLE_RELATION,
                         strength=PROVISIONAL_EDGE_STRENGTH, cycle=PC)

        # The live consolidation step that WOULD poison _predict if unguarded.
        sub.form_abstractions(cycle=PC + 1)

        # positive control: a NON-role shared target DOES form an abstraction
        # (so form_abstractions is genuinely active this pass) ...
        self.assertIn(f'{ABSTRACTION_NAME_PREFIX}{WORLD_RELATION}_Y',
                      sub.concepts)
        # ... but has_role is SKIPPED — no role abstraction, no is_a poison.
        self.assertFalse(any(
            n.startswith(f'{ABSTRACTION_NAME_PREFIX}{ROLE_RELATION}')
            for n in sub.concepts))
        # q gained NO is_a siblings via the shared role node.
        self.assertEqual(set(sub.is_a_siblings('q')), set())
        # grounding._predict is byte-identical to baseline.
        self.assertEqual(loop._predict('q', cycle=PC), base_q)
        for m in ('b1', 'b2', 'b3'):
            # The PREDICTION must stay byte-identical.  The `via` LABEL
            # may legitimately differ: since 2026-08-13 own and
            # inherited are co-present, and the _abs_transitions_to_Y
            # abstraction that this test's own positive control
            # REQUIRES to form makes b1/b2/b3 is_a siblings -- so their
            # own transition and their siblings' now AGREE, and the
            # label becomes 'both'.  That agreement comes from the
            # transitions_to abstraction, NOT from has_role.  The poison
            # this falsifier exists to catch is still pinned exactly by
            # the three assertions above: q predicts nothing, no role
            # abstraction forms, no role-derived is_a siblings.
            self.assertEqual(loop._predict(m, cycle=PC)[0],
                             base_b[m][0])


# --------------------------------------------------------------------------
# (7) FALSIFIER: form_analogies UNCHANGED by has_role skeletons
#     (the analogy anti-poison guard mirrors the form_abstractions one)
# --------------------------------------------------------------------------
class TestFalsifierNoAnalogyPoison(unittest.TestCase):

    def test_has_role_does_not_make_world_tokens_analogy_eligible(self):
        sub = Substrate()
        sub._quarantine_migrated = True
        PC = 100

        # Two world-like state tokens.  Each has ONE real structural
        # relation (transitions_to) -> a size-1 skeleton, below
        # ANALOGY_MIN_SKELETON (2), so NOT analogy-eligible on its own.
        sub.add_edge('s1', 't1', WORLD_RELATION, strength=0.6, cycle=5)
        sub.add_edge('s2', 't2', WORLD_RELATION, strength=0.6, cycle=5)
        # The instrument tags each with a has_role edge (DISJOINT role
        # targets).  UNGUARDED, this 2nd relation label would lift both to
        # a shared {transitions_to, has_role} skeleton with disjoint
        # targets -- exactly the shape form_analogies mints over.
        sub.add_edge('s1', '_role_visit_2', ROLE_RELATION,
                     strength=PROVISIONAL_EDGE_STRENGTH, cycle=PC)
        sub.add_edge('s2', '_role_visit_6', ROLE_RELATION,
                     strength=PROVISIONAL_EDGE_STRENGTH, cycle=PC)

        # Positive control: two tokens with a GENUINE 2-relation non-role
        # skeleton and disjoint targets DO form an analogy this pass, so
        # form_analogies is provably active.
        sub.add_edge('p1', 'u1', WORLD_RELATION, strength=0.6, cycle=5)
        sub.add_edge('p1', 'u2', 'r_extra', strength=0.6, cycle=5)
        sub.add_edge('p2', 'u3', WORLD_RELATION, strength=0.6, cycle=5)
        sub.add_edge('p2', 'u4', 'r_extra', strength=0.6, cycle=5)

        sub.form_analogies(cycle=PC + 1)

        # positive control fired: p1<->p2 analogy exists ...
        pa, pb = sorted(('p1', 'p2'))
        self.assertIn((pa, ANALOGY_RELATION, pb), sub.edges)
        # ... but the shared has_role tag did NOT make s1<->s2
        # analogy-eligible: no analogous_to edge between the world tokens.
        sa, sb = sorted(('s1', 's2'))
        self.assertNotIn((sa, ANALOGY_RELATION, sb), sub.edges)
        # and NO analogous_to edge touches s1 or s2 at all (guard held).
        self.assertFalse(any(
            r == ANALOGY_RELATION and (s in ('s1', 's2') or t in ('s1', 's2'))
            for (s, r, t) in sub.edges))


if __name__ == '__main__':
    unittest.main()
