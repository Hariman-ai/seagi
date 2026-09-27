"""Facet-docking foundation — mechanical build gates (2026-07-21, RESCOPED
for the ignition build per the final APPROVE-WITH-CONDITIONS re-audit).

What changed vs the first facet build:
  * C1 RESCOPE — OBSERVE_REASONS = {'facet_observe'} ONLY.  world_observe
    keeps its incumbent live semantics (un-engaged on creation, duplicates
    STILL reinforce) pending the Flag-C recovery design pass.  Pinned here.
  * TOKEN-ARGMAX DELETED — dock_predict / the token-vote dock is GONE from
    the live path (G2 showed it predicts the wrong quantity).  Earning is
    now SPLIT-EVENT CONTRASTIVE on the 2-class signature (C2 v2), tested
    here; ignition/fusion tests live in test_ignition_fusion.py.
  * S3 facet tiebreak RETIRED with its substrate (the argmax).

Covers:
  - has_facet pinned OUT of RELATION_COMPOSITION (keys AND values)   [farm]
  - has_facet pinned OUT of INFERENCE_RELATIONS                      [farm]
  - co_occurs pinned OUT of both (the fusion bond must never compose)[farm]
  - facet_observe duplicate is a strength no-op (no perception ratchet)
  - world_observe duplicate STILL reinforces (C1 rescope pin — Flag-C)
  - facet_disconfirm: symmetric promille weaken, floors, never creates
  - split-event contrastive earning: unanimity = no writes; split moves
    matching electors up / mismatching down; a correct MINORITY never bleeds
  - facets ON is directed-exploration SEED-EXACT vs incumbent (zero actor-
    RNG draws)                                                      [C3/G1]
  - invariant  facet_edges == 9 x facet-written-states               [C4]
  - no edge with a `_facet_*` endpoint has relation != has_facet     [skip]
"""
from __future__ import annotations

import types
import unittest

from seagi.core.substrate import (
    Substrate, FACET_RELATION, FACET_NODE_PREFIX, PROVISIONAL_EDGE_STRENGTH,
    EDGE_STRENGTH_BUMP_PER_USE)
from seagi.brain.capabilities.writer import (
    JournaledSubstrateWriter, OBSERVE_REASONS)
from seagi.brain.capabilities.grounding import GroundingLoop
from seagi.brain.capabilities.world_actor import WorldActor
from seagi.brain.capabilities.world_transducer import WorldTransducer
from seagi.brain.capabilities.cortical import (
    RELATION_COMPOSITION, INFERENCE_RELATIONS)
from seagi.brain.events import EventKind, SubstrateWriteQueuedEvent
from seagi.world.structured_territory_world import StructuredTerritoryWorld


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

class _SimBus:
    """Delivers substrate-write events to the single writer; ignores the
    rest (claims / attends / chemistry are not modelled here)."""
    def __init__(self, writer):
        self.writer = writer
    def publish(self, ev):
        if isinstance(ev, SubstrateWriteQueuedEvent):
            self.writer.handle(ev, self)


class _Rig:
    pass


def _build(level, seed, enable_facets=False, facet_seed=0):
    sub = Substrate()
    sub._quarantine_migrated = True
    engine = types.SimpleNamespace(substrate=sub)
    writer = JournaledSubstrateWriter(engine=engine)
    bus = _SimBus(writer)
    grounding = GroundingLoop(engine=engine, bus=bus)
    tx = WorldTransducer()
    world = StructuredTerritoryWorld(level=level, seed=seed, k=8)
    actor = WorldActor(bus, world, tx, grounding,
                       value_provider=None, seed=seed,
                       enable_facets=enable_facets,
                       facet_seed=facet_seed)
    r = _Rig()
    r.sub, r.writer, r.bus, r.grounding = sub, writer, bus, grounding
    r.tx, r.world, r.actor = tx, world, actor
    return r


def _run(actor, ticks, c0=1):
    c = c0
    actions = []
    for _ in range(ticks):
        a = actor.propose(c)
        if a is None:
            break
        actions.append(a)
        actor.execute(a, c)
        c += 1
    return actions, c


def _write(bus, subj, rel, obj, reason, strength=PROVISIONAL_EDGE_STRENGTH,
           cycle=1):
    bus.publish(SubstrateWriteQueuedEvent(
        kind=EventKind.SUBSTRATE_WRITE_QUEUED, cycle=cycle,
        source_capability='world_actor', origin='internal',
        subject=subj, relation=rel, object=obj,
        strength=strength, write_reason=reason))


# --------------------------------------------------------------------------
# farm-closure pins
# --------------------------------------------------------------------------

class TestFarmPins(unittest.TestCase):

    def test_has_facet_not_in_relation_composition_keys_or_values(self):
        for (a, b) in RELATION_COMPOSITION.keys():
            self.assertNotEqual(a, FACET_RELATION)
            self.assertNotEqual(b, FACET_RELATION)
        for v in RELATION_COMPOSITION.values():
            self.assertNotEqual(v, FACET_RELATION)

    def test_has_facet_not_in_inference_relations(self):
        self.assertNotIn(FACET_RELATION, INFERENCE_RELATIONS)

    def test_co_occurs_not_in_composition_or_inference(self):
        # The ignition fusion bond: co_occurs must never compose (it would
        # open first_coherent_cycle -> lifeforce for episode bonds).
        for (a, b) in RELATION_COMPOSITION.keys():
            self.assertNotEqual(a, 'co_occurs')
            self.assertNotEqual(b, 'co_occurs')
        for v in RELATION_COMPOSITION.values():
            self.assertNotEqual(v, 'co_occurs')
        self.assertNotIn('co_occurs', INFERENCE_RELATIONS)


# --------------------------------------------------------------------------
# C1 (rescoped) — the facet perception ratchet is closed; world_observe
# keeps its incumbent semantics (the Flag-C rider stays OUT)
# --------------------------------------------------------------------------

class TestObserveSemantics(unittest.TestCase):

    def test_observe_reasons_is_facet_only(self):
        # THE RESCOPE PIN: the writer's observe-law covers the facet channel
        # ONLY.  world_observe joins it only after the Flag-C recovery pass.
        self.assertEqual(OBSERVE_REASONS, frozenset({'facet_observe'}))

    def test_facet_observe_duplicate_is_strength_no_op(self):
        r = _build(level=1, seed=1)
        key = ('s1', FACET_RELATION, '_facet_0_3')
        # first write: applied, UN-engaged (earn-by-predicting)
        _write(r.bus, 's1', FACET_RELATION, '_facet_0_3', 'facet_observe',
               strength=0.1, cycle=1)
        e = r.sub.edges[key]
        self.assertEqual(int(e.last_engaged_cycle), 0)
        s0 = float(e.strength)
        reinf0 = r.writer.writes_reinforced
        # second write with HIGHER strength: skipped, no reinforce, no
        # engagement stamp, NO strength move.
        _write(r.bus, 's1', FACET_RELATION, '_facet_0_3', 'facet_observe',
               strength=0.9, cycle=2)
        e2 = r.sub.edges[key]
        self.assertEqual(float(e2.strength), s0)
        self.assertEqual(int(e2.last_engaged_cycle), 0)
        self.assertEqual(r.writer.writes_reinforced, reinf0)
        self.assertEqual(r.writer.journal[-1]['status'], 'skipped')
        self.assertEqual(r.writer.journal[-1]['detail'], 'observe_duplicate')

    def test_world_observe_duplicate_still_reinforces(self):
        # C1 RESCOPE PIN (Flag-C): the live world_observe duplicate-
        # reinforce behavior is UNCHANGED by this build — closing it removes
        # recovery-by-re-observation and needs its own design pass first.
        r = _build(level=1, seed=1)
        key = ('a', 'transitions_to', 'b')
        _write(r.bus, 'a', 'transitions_to', 'b', 'world_observe',
               strength=0.1, cycle=1)
        e = r.sub.edges[key]
        self.assertEqual(int(e.last_engaged_cycle), 0)   # creation un-engaged
        reinf0 = r.writer.writes_reinforced
        _write(r.bus, 'a', 'transitions_to', 'b', 'world_observe',
               strength=0.1, cycle=2)
        self.assertEqual(r.writer.writes_reinforced, reinf0 + 1)
        # duplicate reinforces strength but does NOT stamp engagement
        # (world_observe is not in the writer's engagement whitelist).
        e2 = r.sub.edges[key]
        self.assertGreater(float(e2.strength), 0.1 - 1e-9)
        self.assertEqual(int(e2.last_engaged_cycle), 0)


# --------------------------------------------------------------------------
# facet_disconfirm — the subtractive half of contrastive earning
# --------------------------------------------------------------------------

class TestFacetDisconfirm(unittest.TestCase):

    def test_disconfirm_weakens_symmetric_promille(self):
        r = _build(level=1, seed=1)
        key = ('sp', FACET_RELATION, '_facet_0_3')
        _write(r.bus, 'sp', FACET_RELATION, '_facet_0_3', 'facet_observe',
               strength=0.1, cycle=1)
        s0 = float(r.sub.edges[key].strength)
        _write(r.bus, 'sp', FACET_RELATION, '_facet_0_3', 'facet_disconfirm',
               cycle=1)
        e = r.sub.edges[key]
        self.assertAlmostEqual(float(e.strength),
                               s0 - EDGE_STRENGTH_BUMP_PER_USE, places=9)
        self.assertEqual(int(e.last_engaged_cycle), 0)   # no stamp
        self.assertEqual(r.writer.writes_weakened, 1)
        self.assertEqual(r.writer.journal[-1]['status'], 'weakened')

    def test_confirm_and_disconfirm_are_symmetric(self):
        r = _build(level=1, seed=1)
        key = ('sp', FACET_RELATION, '_facet_0_3')
        _write(r.bus, 'sp', FACET_RELATION, '_facet_0_3', 'facet_observe',
               strength=0.1, cycle=1)
        s0 = float(r.sub.edges[key].strength)
        _write(r.bus, 'sp', FACET_RELATION, '_facet_0_3', 'facet_confirm',
               cycle=1)
        up = float(r.sub.edges[key].strength) - s0
        _write(r.bus, 'sp', FACET_RELATION, '_facet_0_3', 'facet_disconfirm',
               cycle=1)
        back = float(r.sub.edges[key].strength)
        self.assertAlmostEqual(up, EDGE_STRENGTH_BUMP_PER_USE, places=9)
        self.assertAlmostEqual(back, s0, places=9)
        # confirm stamps engagement (the survival signal); disconfirm never.
        _write(r.bus, 'sp', FACET_RELATION, '_facet_0_3', 'facet_confirm',
               cycle=7)
        self.assertEqual(int(r.sub.edges[key].last_engaged_cycle), 7)

    def test_disconfirm_floors_at_zero_never_negative(self):
        r = _build(level=1, seed=1)
        key = ('sp', FACET_RELATION, '_facet_0_3')
        _write(r.bus, 'sp', FACET_RELATION, '_facet_0_3', 'facet_observe',
               strength=0.1, cycle=1)
        for i in range(40):     # 40 x 0.005 = 0.2 > 0.1
            _write(r.bus, 'sp', FACET_RELATION, '_facet_0_3',
                   'facet_disconfirm', cycle=1)
        e = r.sub.edges[key]
        self.assertGreaterEqual(float(e.strength), 0.0)
        self.assertLessEqual(float(e.strength), 0.02)   # under the prune floor

    def test_disconfirm_never_creates_an_edge(self):
        r = _build(level=1, seed=1)
        key = ('ghost', FACET_RELATION, '_facet_0_9')
        _write(r.bus, 'ghost', FACET_RELATION, '_facet_0_9',
               'facet_disconfirm', cycle=1)
        self.assertNotIn(key, r.sub.edges)
        self.assertEqual(r.writer.journal[-1]['status'], 'skipped')
        self.assertEqual(r.writer.journal[-1]['detail'], 'disconfirm_missing')


# --------------------------------------------------------------------------
# C2 v2 — split-event contrastive earning
# --------------------------------------------------------------------------

class TestSplitEventEarning(unittest.TestCase):
    """The elector rule: strength moves ONLY where elector signatures
    DISAGREE.  sig(x, a) = (trans[(x, a)] == x): self-loop vs move."""

    def _rig_electors(self):
        r = _build(level=1, seed=1)
        sub = r.sub
        # facet carriers (mimic facet_observe: un-engaged, provisional).
        for (src, F) in (('spWALL', '_facet_F1'), ('spWALL', '_facet_F2'),
                         ('spMOVE', '_facet_F1'), ('spLONE', '_facet_F9')):
            sub.add_edge(src, F, FACET_RELATION,
                         strength=PROVISIONAL_EDGE_STRENGTH, cycle=1,
                         engage=False)
        # elector transitions for action 0: spWALL self-loops (wall);
        # spMOVE moves.  A SPLIT population.
        trans = {('spWALL', 0): 'spWALL', ('spMOVE', 0): 'spX'}
        facet_nodes = ['_facet_F1', '_facet_F2']
        return r, trans, facet_nodes

    def test_unanimous_is_uninformative_no_writes(self):
        r, trans, fns = self._rig_electors()
        trans = {('spWALL', 0): 'spWALL'}     # only one elector -> unanimous
        w0 = (r.writer.writes_reinforced, r.writer.writes_weakened)
        r.grounding.dock_observe('s', 0, 's', 10, r.bus, fns, trans)
        self.assertEqual(r.grounding.dock_uninformative, 1)
        self.assertEqual(r.grounding.dock_split, 0)
        self.assertEqual(
            (r.writer.writes_reinforced, r.writer.writes_weakened), w0)

    def test_split_moves_matching_up_mismatching_down(self):
        r, trans, fns = self._rig_electors()
        # truth: s self-loops (sig_now True) -> spWALL matches, spMOVE not.
        r.grounding.dock_observe('s', 0, 's', 10, r.bus, fns, trans)
        self.assertEqual(r.grounding.dock_split, 1)
        self.assertEqual(r.grounding.dock_elector_confirms, 1)
        self.assertEqual(r.grounding.dock_elector_disconfirms, 1)
        eW1 = r.sub.edges[('spWALL', FACET_RELATION, '_facet_F1')]
        eW2 = r.sub.edges[('spWALL', FACET_RELATION, '_facet_F2')]
        eM = r.sub.edges[('spMOVE', FACET_RELATION, '_facet_F1')]
        self.assertGreater(float(eW1.strength), PROVISIONAL_EDGE_STRENGTH)
        self.assertGreater(float(eW2.strength), PROVISIONAL_EDGE_STRENGTH)
        self.assertEqual(int(eW1.last_engaged_cycle), 10)
        self.assertLess(float(eM.strength), PROVISIONAL_EDGE_STRENGTH)
        self.assertEqual(int(eM.last_engaged_cycle), 0)
        # the uninvolved carrier is untouched.
        eL = r.sub.edges[('spLONE', FACET_RELATION, '_facet_F9')]
        self.assertEqual(float(eL.strength), PROVISIONAL_EDGE_STRENGTH)

    def test_correct_minority_never_bleeds(self):
        # 1 elector says MOVE, 2 say WALL; truth is MOVE.  The minority
        # matching elector EARNS; the majority loses — disconfirm tracks the
        # OBSERVED outcome, never the consensus.
        r = _build(level=1, seed=1)
        sub = r.sub
        for (src, F) in (('m1', '_facet_F1'), ('w1', '_facet_F1'),
                         ('w2', '_facet_F1')):
            sub.add_edge(src, F, FACET_RELATION,
                         strength=PROVISIONAL_EDGE_STRENGTH, cycle=1,
                         engage=False)
        trans = {('m1', 0): 'elsewhere', ('w1', 0): 'w1', ('w2', 0): 'w2'}
        r.grounding.dock_observe('s', 0, 'snext', 10, r.bus,
                                 ['_facet_F1'], trans)   # truth: moved
        self.assertGreater(
            float(sub.edges[('m1', FACET_RELATION, '_facet_F1')].strength),
            PROVISIONAL_EDGE_STRENGTH)
        for w in ('w1', 'w2'):
            self.assertLess(
                float(sub.edges[(w, FACET_RELATION, '_facet_F1')].strength),
                PROVISIONAL_EDGE_STRENGTH)

    def test_no_electors_abstains(self):
        r = _build(level=1, seed=1)
        r.grounding.dock_observe('s', 0, 's', 10, r.bus,
                                 ['_facet_NONE'], {})
        self.assertEqual(r.grounding.dock_abstains, 1)
        self.assertEqual(r.grounding.dock_split, 0)

    def test_counters_surface_in_stats(self):
        # C-A: the split/uninformative counters MUST be readable in the
        # grounding stats block (runtime /status includes grounding.stats()).
        r = _build(level=1, seed=1)
        st = r.grounding.stats()
        for k in ('dock_events', 'dock_split', 'dock_uninformative',
                  'dock_abstains', 'dock_elector_confirms',
                  'dock_elector_disconfirms'):
            self.assertIn(k, st)


# --------------------------------------------------------------------------
# C3 / G1 — exploration structural protection + RNG isolation
# --------------------------------------------------------------------------

class TestExplorationSeedExact(unittest.TestCase):

    def test_facets_draw_nothing_and_never_remove_an_action(self):
        """Re-pinned 2026-07-26 for the ordering wire.

        Dropped: "facets ON vs OFF give an identical action sequence" and
        the matching directed/fallback counts.  Those pinned facets as
        perception-only; the wire deliberately makes recognition set the
        ORDER untried actions are tried in, so an identical sequence would
        now mean the feature is absent.

        Kept, and checked FIRST: the facet machinery draws NOTHING from
        the actor RNG.  In the old order that assertion sat behind the
        sequence check and would never run once the sequence differed -- a
        guarantee you cannot reach is not a guarantee.  It survives because
        the wire draws from `novel` BEFORE consulting recognition, so
        consumption is identical either way.

        Added: MEMBERSHIP.  `novel` is _trans-only, so no untried action
        can be removed -- only deferred.  Checked operationally below.
        """
        import random as _random
        inc = _build(level=3, seed=7, enable_facets=False)
        fac = _build(level=3, seed=7, enable_facets=True, facet_seed=99)
        _run(inc.actor, 1500)
        _run(fac.actor, 1500)
        inc_d = fac.actor.explorations_directed
        inc_f = fac.actor.explorations_fallback

        # 1) THE WIRE ADDS NO RNG DRAW.  Cross-config byte parity cannot
        #    survive a behaviour-changing feature -- once one action
        #    differs the trajectories diverge and the novel branch is
        #    entered a different number of times, so the streams separate
        #    for reasons that have nothing to do with extra draws.  The
        #    guarantee that IS meaningful, and that the wire must honour,
        #    is that it consumes nothing of its own: exactly one draw per
        #    proposal that reaches a selection branch, wire or no wire.
        class _CountingRNG(object):
            def __init__(self, r):
                self._r = r
                self.n = 0

            def choice(self, seq):
                self.n += 1
                return self._r.choice(seq)

            def __getattr__(self, k):
                return getattr(self._r, k)

        probe = _CountingRNG(fac.actor._rng)
        fac.actor._rng = probe
        _run(fac.actor, 600)
        self.assertEqual(
            probe.n,
            fac.actor.explorations_directed + fac.actor.explorations_fallback
            - (inc_d + inc_f),
            'the ordering wire consumed extra actor-RNG draws')

        # 2) the isolated facet generator drew nothing either.
        self.assertEqual(fac.actor._facet_rng.getstate(),
                         _random.Random(99).getstate())
        # 3) not vacuous: facets ACTUALLY got written.
        n_facet = sum(1 for (s, rel, t) in fac.sub.edges
                      if rel == FACET_RELATION)
        self.assertGreater(n_facet, 0, 'facets never wrote — test is vacuous')
        # 4) MEMBERSHIP / every-mistake-once: wherever he has had enough
        #    visits to try everything, he HAS tried everything.  Ordering
        #    may defer an action; it may never delete one.
        a = fac.actor
        n = int(getattr(a.world, 'n_actions', 4))
        tried = {}
        for (s, act) in a._trans:
            tried.setdefault(s, set()).add(act)
        checked = 0
        for s, acts in tried.items():
            if a._visits.get(s, 0) >= n:
                checked += 1
                self.assertEqual(
                    len(acts), n,
                    'state visited %d times but only %d/%d actions tried — '
                    'recognition removed one' % (a._visits.get(s, 0),
                                                 len(acts), n))
        self.assertGreater(checked, 0, 'no state was visited enough to check')


# --------------------------------------------------------------------------
# standing invariants (C4 + grouping skip)
# --------------------------------------------------------------------------

class TestFacetInvariants(unittest.TestCase):

    def test_c4_facet_edges_are_9x_written_states(self):
        # vector length is 9 (radius-1 3x3 window) -> the C4 literal; this
        # is ALSO the derived ignition focal bound |focals| <= 1 + 9.
        fac = _build(level=3, seed=7, enable_facets=True)
        vec = fac.world.percept()['world_vector']
        self.assertEqual(len(WorldTransducer.facets(vec)), 9)
        _run(fac.actor, 4000)
        facet_edges = [(s, rel, t) for (s, rel, t) in fac.sub.edges
                       if rel == FACET_RELATION]
        sources = {s for (s, rel, t) in facet_edges}
        self.assertGreater(len(sources), 0, 'no facet-written states')
        self.assertEqual(len(facet_edges), 9 * len(sources))

    def test_no_facet_endpoint_outside_has_facet(self):
        fac = _build(level=3, seed=7, enable_facets=True)
        _run(fac.actor, 4000)
        # run BOTH consolidation passes, full-scan AND scoped, that COULD
        # mint is_a / analogous_to over facet nodes were the skip absent.
        allc = set(fac.sub.concepts)
        fac.sub.form_abstractions(cycle=99999)
        fac.sub.form_analogies(cycle=99999)
        fac.sub.form_abstractions(cycle=99999, candidates=allc)
        fac.sub.form_analogies(cycle=99999, candidates=allc)
        bad = [(s, rel, t) for (s, rel, t) in fac.sub.edges
               if (str(s).startswith(FACET_NODE_PREFIX)
                   or str(t).startswith(FACET_NODE_PREFIX))
               and rel != FACET_RELATION]
        self.assertEqual(bad, [], f'facet endpoint leaked: {bad[:5]}')

    def test_token_argmax_is_deleted_from_the_live_path(self):
        # The G2-refuted token-vote dock must not linger dormant (the audit:
        # retired machinery is DELETED, not switched off).
        self.assertFalse(hasattr(GroundingLoop, 'dock_predict'))
        self.assertFalse(hasattr(WorldActor, '_facet_ordered_tie'))


if __name__ == '__main__':
    unittest.main()
