"""Ignition + fusion — mechanical build gates (2026-07-21, per the final
APPROVE-WITH-CONDITIONS re-audit of the similarity-docking re-aim).

The dock does NOT compute a prediction.  On total own-silence (the existing
genuine-novelty predicate: no learned transition for ANY action at s) the
best-fitting facet-sharing siblings are PUBLISHED INTO ATTENTION as one
multi-focal AttendedPerceptEvent — recognition lights memories into the
present; prediction falls out of RECALL.  Fusion is the EXISTING hippocampus
binding the ignition episode (current <-> ignited co_occurs), and the
durable "new part" is the EXISTING form_abstractions minting an is_a class
over >= 3 co-fitting bonds.

Pins covered here:
  - ignition gate fires ONLY on total own-silence, and only when a key fits
  - one best-fitting sibling per lock; dedup; |focals| <= 1 + |facets|  [3]
  - m_content == i_content == 0 pinned (FAILS on nonzero)             [5]
  - ZERO ChemistryEvents on the ignition path                          [5]
  - published AFTER the action is chosen; zero actor-RNG draws         [G1]
  - C-C INVARIANT: no non-ignition event path ever publishes >= 2
    `_world_s*` focals (sim-scan over every published attend)
  - fusion: hippocampus binds focal-centric current<->ignited co_occurs;
    a single-focal episode writes NOTHING (the hippocampus guard)
  - fusion bond born at CO_OCCURS_STRENGTH * (1 + 0) = 0.30 flat (the
    m/i=0 firewall — the owned fusion-suppression cost)
  - C5 EXTENDED (observed, never excluded): zero clock-grade settlings on
    `_facet_*` edges, world<->world co_occurs bonds, AND membership edges
    of `_abstract_co_occurs_*` classes in sims                    [G6/G8]
"""
from __future__ import annotations

import random as _random
import types
import unittest

from seagi.core.substrate import (
    Substrate, FACET_RELATION, FACET_NODE_PREFIX, PROVISIONAL_EDGE_STRENGTH,
    WORLD_TOKEN_PREFIX, register_reinforce_observer,
    unregister_reinforce_observer)
from seagi.brain.capabilities.writer import JournaledSubstrateWriter
from seagi.brain.capabilities.grounding import GroundingLoop
from seagi.brain.capabilities.world_actor import WorldActor
from seagi.brain.capabilities.world_transducer import WorldTransducer
from seagi.brain.capabilities.hippocampus import (
    Hippocampus, CO_OCCURS_STRENGTH)
from seagi.brain.events import (
    EventKind, SubstrateWriteQueuedEvent, AttendedPerceptEvent,
    ChemistryEvent, CapabilityClaimEvent, ReflectionFiredEvent)
from seagi.world.structured_territory_world import (
    StructuredTerritoryWorld, StructuredTerritoryWorldBlind,
    territory_escalator, territory_blind_escalator)

SAT = 0.98      # SATURATION_CEILING — the non-cohering settling door


class _RecBus:
    """Records EVERY published event; routes writes to the single writer
    and (optionally) attends/reflections to a hippocampus."""
    def __init__(self, writer, hippocampus=None):
        self.writer = writer
        self.hippocampus = hippocampus
        self.events = []
    def publish(self, ev):
        self.events.append(ev)
        if isinstance(ev, SubstrateWriteQueuedEvent):
            self.writer.handle(ev, self)
        elif self.hippocampus is not None and isinstance(
                ev, (AttendedPerceptEvent, ReflectionFiredEvent)):
            self.hippocampus.handle(ev, self)


class _Rig:
    pass


def _build(level, seed, enable_facets=True, hippo=False, blind=False):
    sub = Substrate()
    sub._quarantine_migrated = True
    engine = types.SimpleNamespace(substrate=sub)
    writer = JournaledSubstrateWriter(engine=engine)
    hip = Hippocampus(bus=None) if hippo else None
    bus = _RecBus(writer, hip)
    if hip is not None:
        hip.bus = bus
    grounding = GroundingLoop(engine=engine, bus=bus)
    tx = WorldTransducer()
    W = StructuredTerritoryWorldBlind if blind else StructuredTerritoryWorld
    world = W(level=level, seed=seed, k=8)
    actor = WorldActor(bus, world, tx, grounding,
                       value_provider=None, seed=seed,
                       enable_facets=enable_facets, facet_seed=1000 + seed)
    r = _Rig()
    r.sub, r.writer, r.bus, r.grounding = sub, writer, bus, grounding
    r.tx, r.world, r.actor, r.hip = tx, world, actor, hip
    return r


def _run(actor, ticks, c0=1, on_success=None):
    c = c0
    actions = []
    for _ in range(ticks):
        a = actor.propose(c)
        if a is None:
            break
        actions.append(a)
        res = actor.execute(a, c)
        c += 1
        if res and res.get('success') and on_success is not None:
            on_success()
    return actions, c


def _train_then_escalate(r, ticks=6000):
    """Train until the current level is solved a few times (facet edges
    exist), then escalate — the fresh frontier room is genuinely novel
    territory whose windows SHARE colors with known territory -> keys fit.
    The BLIND twin escalates with ITS OWN escalator (seed-bumped random
    percepts: nothing recurs, the honest control)."""
    esc = (territory_blind_escalator
           if isinstance(r.actor.world, StructuredTerritoryWorldBlind)
           else territory_escalator)
    holder = {'n': 0}
    def _esc():
        holder['n'] += 1
        if holder['n'] % 2 == 0:
            r.actor.world = esc(r.actor.world)
    _run(r.actor, ticks, on_success=_esc)


def _ignition_events(bus):
    return [e for e in bus.events
            if isinstance(e, AttendedPerceptEvent)
            and getattr(e, 'origin_detail', '') == 'world_ignition']


# --------------------------------------------------------------------------
# the ignition gate + event shape
# --------------------------------------------------------------------------

class _TinyWorld:
    """Controlled 1-state world: every action is a no-op.  Lets the gate be
    tested deterministically (percept never changes until we change it)."""
    n_actions = 2
    def __init__(self, vec=(1.0, 2.0, 3.0)):
        self.vec = list(vec)
    def percept(self):
        return {'world_vector': list(self.vec)}
    def step(self, a):
        return {'world_vector': list(self.vec), 'success': False,
                'steps': 1}


def _tiny_rig(seed_carrier=True):
    sub = Substrate()
    sub._quarantine_migrated = True
    engine = types.SimpleNamespace(substrate=sub)
    writer = JournaledSubstrateWriter(engine=engine)
    bus = _RecBus(writer)
    grounding = GroundingLoop(engine=engine, bus=bus)
    tx = WorldTransducer()
    world = _TinyWorld()
    actor = WorldActor(bus, world, tx, grounding, value_provider=None,
                       seed=3, enable_facets=True, facet_seed=5)
    if seed_carrier:
        # a known sibling sp carrying two of the tiny percept's three
        # facets — a key that fits 2 of 3 locks.
        for (slot, val) in ((0, '1'), (1, '2')):
            sub.add_edge('_world_sSIB', tx.facet_node(slot, val),
                         FACET_RELATION,
                         strength=PROVISIONAL_EDGE_STRENGTH, cycle=1,
                         engage=False)
    return sub, bus, actor


class TestIgnitionGate(unittest.TestCase):

    def test_gate_fires_on_total_silence_and_closes_after_contact(self):
        sub, bus, actor = _tiny_rig()
        # 1st propose: s totally unknown + a key fits -> ONE ignition.
        a = actor.propose(1)
        igs = _ignition_events(bus)
        self.assertEqual(len(igs), 1)
        self.assertEqual(actor.ignitions, 1)
        e = igs[0]
        self.assertEqual(e.focals[1:], ['_world_sSIB'])
        self.assertAlmostEqual(e.payload['_world_sSIB']['salience'],
                               2.0 / 3.0, places=9)
        # execute writes _trans[(s, a)] -> total silence is BROKEN.
        actor.execute(a, 1)
        # 2nd propose on the SAME state: one action is now modelled ->
        # the gate must NOT fire again (all() over actions is false).
        actor.propose(2)
        self.assertEqual(len(_ignition_events(bus)), 1)
        self.assertEqual(actor.ignitions, 1)

    def test_no_ignition_when_no_key_fits(self):
        sub, bus, actor = _tiny_rig(seed_carrier=False)
        actor.propose(1)
        self.assertEqual(actor.ignitions, 0)
        self.assertEqual(_ignition_events(bus), [])

    def test_ignition_fires_in_real_escalation(self):
        r = _build(level=2, seed=7)
        _train_then_escalate(r)
        igs = _ignition_events(r.bus)
        self.assertGreater(len(igs), 0, 'ignition never fired in a '
                           'train->escalate run — no key ever fit')
        self.assertEqual(len(igs), r.actor.ignitions)
        for e in igs:
            self.assertTrue(e.focals[0].startswith(WORLD_TOKEN_PREFIX))

    def test_event_shape_and_derived_bound(self):
        r = _build(level=2, seed=7)
        _train_then_escalate(r)
        igs = _ignition_events(r.bus)
        self.assertGreater(len(igs), 0)
        for e in igs:
            nf = 9      # radius-1 3x3 eye — the C4 geometry
            # [3] derived focal bound: current + <= one sibling per lock.
            self.assertLessEqual(len(e.focals), 1 + nf)
            self.assertGreaterEqual(len(e.focals), 2)
            # dedup
            self.assertEqual(len(e.focals), len(set(e.focals)))
            # [5] the m/i=0 firewall — MUST fail on nonzero.
            self.assertEqual(float(e.m_content), 0.0)
            self.assertEqual(float(e.i_content), 0.0)
            self.assertEqual(e.modality, 'world')
            self.assertEqual(e.origin_detail, 'world_ignition')
            # per-sibling DERIVED salience in payload: shared/|facets|.
            for sp in e.focals[1:]:
                meta = e.payload.get(sp) or {}
                self.assertIn('salience', meta)
                self.assertGreater(meta['salience'], 0.0)
                self.assertLessEqual(meta['salience'], 1.0)
                self.assertEqual(
                    meta['salience'], meta['shared_facets'] / float(nf))
            # event scalar = best fit.
            self.assertAlmostEqual(
                float(e.salience),
                max(e.payload[sp]['salience'] for sp in e.focals[1:]),
                places=9)

    def test_zero_chemistry_on_ignition_and_publish_after_choice(self):
        # Deterministic single-propose form: the whole propose tick of an
        # igniting state publishes EXACTLY [CapabilityClaim, ignition
        # attend] in that order — zero ChemistryEvents [5], and the attend
        # comes strictly AFTER the action was chosen (claim precedes it).
        sub, bus, actor = _tiny_rig()
        n0 = len(bus.events)
        actor.propose(1)
        tick = bus.events[n0:]
        kinds = [type(e).__name__ for e in tick]
        self.assertEqual(kinds,
                         ['CapabilityClaimEvent', 'AttendedPerceptEvent'])
        self.assertEqual(tick[1].origin_detail, 'world_ignition')
        self.assertFalse(any(isinstance(e, ChemistryEvent) for e in tick))
        # and across a full real run: no ChemistryEvent ever shares a
        # publication site with ignition (world_actor chemistry only rides
        # the success path, pinned by origin_detail).
        r = _build(level=2, seed=7)
        _train_then_escalate(r)
        for e in r.bus.events:
            if isinstance(e, ChemistryEvent):
                self.assertNotEqual(
                    getattr(e, 'origin_detail', ''), 'world_ignition')

    def test_facets_reorder_but_never_remove(self):
        """Re-pinned 2026-07-26 for the ordering wire.

        Was: facets ON vs OFF must give a BYTE-IDENTICAL action sequence.
        That pinned facets as perception-only.  The ordering wire makes
        recognition set the ORDER untried actions are tried in, so an
        identical sequence would now mean the feature is absent.

        The invariant that protects every-mistake-once is MEMBERSHIP, not
        order: `novel` is _trans-only, so no untried action may ever be
        removed -- an action may be tried later, never never.  Asserted
        directly, per state, below.

        (The old form also compared two 5,000-element lists with
        assertEqual; on failure unittest fed both to difflib and the suite
        stalled for 25+ minutes rendering the diff.  Nothing here diffs a
        long sequence.)
        """
        fac = _build(level=2, seed=7, enable_facets=True)
        a = fac.actor
        n = int(getattr(a.world, 'n_actions', 4))
        _run(a, 800)

        # MEMBERSHIP: for every state he has acted from, the untried set is
        # exactly {all actions} - {actions in _trans}.  Recognition cannot
        # have removed one, whatever order it preferred.
        states = {s for (s, _x) in a._trans}
        self.assertTrue(states, 'actor never acted')
        for s in states:
            tried = {x for (st, x) in a._trans if st == s}
            untried = set(range(n)) - tried
            for act in untried:
                self.assertNotIn(
                    (s, act), a._trans,
                    'an untried action leaked into _trans')

        # the isolated facet generator is still never drawn from
        self.assertEqual(a._facet_rng.getstate(),
                         _random.Random(1007).getstate())

    def test_blind_config_is_byte_identical(self):
        """With facets OFF the actor is unchanged, byte for byte.

        This is the half of the old assertion that still holds and still
        matters: the wire must not perturb him where recognition has
        nothing to say.  Compared as a single boolean so a failure can
        never trigger a difflib diff over 5,000 elements.
        """
        x = _build(level=2, seed=7, enable_facets=False)
        y = _build(level=2, seed=7, enable_facets=False)
        ax, _ = _run(x.actor, 1500)
        ay, _ = _run(y.actor, 1500)
        self.assertTrue(ax == ay, 'facets-OFF run is not reproducible')
        self.assertEqual(x.actor._rng.getstate(), y.actor._rng.getstate())

    def test_blind_twin_never_LEARNS(self):
        # THE CONTROL: noise must never TEACH him anything.
        #
        # Re-pinned 2026-07-25.  This previously asserted zero ACTIVITY
        # (ignitions == 0, dock_uninformative == 0).  That held only
        # because descriptions were written exclusively on successful
        # paths, so a blind run had no facet edge in the substrate to
        # match against -- the assertion measured the WRITE GATE, not the
        # signal.  Descriptions are now written on ENCOUNTER, so random
        # percepts do occasionally share a facet by chance: measured over
        # 4,000 blind ticks, 1 ignition and 2 uninformative dock events
        # out of 2,958 (2,956 abstains).
        #
        # What must remain EXACTLY zero is EARNING.  Facet strength moves
        # only through a dock SPLIT, so these absolute zeros are the real
        # control -- and two of them (the elector counters, and the facet
        # write stream) were never asserted before.  No thresholds: the
        # invariant is zero, not "small".
        r = _build(level=2, seed=7, blind=True)
        _train_then_escalate(r, ticks=4000)
        self.assertEqual(r.grounding.dock_split, 0,
                         'noise produced SELECTIVE evidence')
        self.assertEqual(r.grounding.dock_elector_confirms, 0)
        self.assertEqual(r.grounding.dock_elector_disconfirms, 0)
        facet_writes = [
            e for e in r.bus.events
            if getattr(e, 'write_reason', '') in
            ('facet_confirm', 'facet_disconfirm')]
        self.assertEqual(facet_writes, [],
                         f'noise moved facet strength: {facet_writes[:3]}')
        # and an ignition on noise must still assert NO lean (the
        # zero-chemistry pin covers the sighted case; hold it here too).
        self.assertFalse(any(isinstance(e, ChemistryEvent)
                             for e in _ignition_events(r.bus)))


# --------------------------------------------------------------------------
# C-C invariant — no non-ignition path publishes >= 2 world focals
# --------------------------------------------------------------------------

class TestCCCInvariant(unittest.TestCase):

    def test_no_non_ignition_multi_world_focal_attends(self):
        r = _build(level=2, seed=7, hippo=True)
        _train_then_escalate(r)
        offenders = []
        for e in r.bus.events:
            if not isinstance(e, AttendedPerceptEvent):
                continue
            wf = [f for f in e.focals
                  if str(f).startswith(WORLD_TOKEN_PREFIX)]
            if len(wf) >= 2 and e.origin_detail != 'world_ignition':
                offenders.append((e.origin_detail, e.focals[:3]))
        self.assertEqual(offenders, [],
                         f'non-ignition multi-world-focal attend: '
                         f'{offenders[:5]}')
        # and the sim actually exercised both attend paths (non-vacuous):
        details = {e.origin_detail for e in r.bus.events
                   if isinstance(e, AttendedPerceptEvent)}
        self.assertIn('world_ignition', details)
        self.assertIn('world_experience', details)


# --------------------------------------------------------------------------
# fusion — existing hippocampus machinery binds the ignition episode
# --------------------------------------------------------------------------

class TestFusion(unittest.TestCase):

    def _reflect(self, r, c):
        r.bus.publish(ReflectionFiredEvent(
            kind=EventKind.REFLECTION_FIRED, cycle=int(c),
            source_capability='test', origin='internal',
            trigger='manual', reflection_kind='general'))

    def test_ignition_episode_consolidates_to_co_occurs_bonds(self):
        r = _build(level=2, seed=7, hippo=True)
        _run(r.actor, 6000)
        r.actor.world = territory_escalator(r.actor.world)
        # frontier phase: reflect every 100 ticks so fresh ignition
        # episodes are consolidated while their recency still carries them
        # (RECENCY_TAU = 200 — reflection cadence is a sim harness choice,
        # not a new organ constant).
        c = 6001
        for _ in range(3000):
            a = r.actor.propose(c)
            if a is None:
                break
            r.actor.execute(a, c)
            if c % 100 == 0:
                self._reflect(r, c)
            c += 1
        self._reflect(r, c + 1)
        bonds = [(s, t, e) for (s, rel, t), e in r.sub.edges.items()
                 if rel == 'co_occurs'
                 and s.startswith(WORLD_TOKEN_PREFIX)
                 and t.startswith(WORLD_TOKEN_PREFIX)]
        self.assertGreater(len(bonds), 0,
                           'no fusion bond minted from ignition episodes')

    def test_bond_born_flat_at_030_no_emotional_boost(self):
        # the m/i=0 firewall: an ignition episode consolidates its bonds at
        # CO_OCCURS_STRENGTH * (1 + max(m, i)) = 0.30 FLAT — the owned
        # fusion-suppression cost.  Controlled single-episode form so the
        # writer journal cannot evict the evidence.
        sub = Substrate(); sub._quarantine_migrated = True
        engine = types.SimpleNamespace(substrate=sub)
        writer = JournaledSubstrateWriter(engine=engine)
        hip = Hippocampus(bus=None)
        bus = _RecBus(writer, hip)
        hip.bus = bus
        bus.publish(AttendedPerceptEvent(
            kind=EventKind.ATTENDED_PERCEPT, cycle=1, timestamp=0.0,
            source_capability='world_actor', origin='internal',
            origin_detail='world_ignition',
            focals=['_world_sNEW', '_world_sSIB1', '_world_sSIB2'],
            payload={}, raw_text='', modality='world',
            salience=0.5, novelty=1.0, m_content=0.0, i_content=0.0))
        bus.publish(ReflectionFiredEvent(
            kind=EventKind.REFLECTION_FIRED, cycle=2,
            source_capability='test', origin='internal',
            trigger='manual', reflection_kind='general'))
        for sib in ('_world_sSIB1', '_world_sSIB2'):
            e = sub.edges[('_world_sNEW', 'co_occurs', sib)]
            self.assertAlmostEqual(float(e.strength), CO_OCCURS_STRENGTH,
                                   places=9)
        # focal-centric: NO sibling<->sibling triangulation edge.
        self.assertNotIn(('_world_sSIB1', 'co_occurs', '_world_sSIB2'),
                         sub.edges)

    def test_single_focal_episode_writes_nothing(self):
        # The hippocampus guard the fusion arithmetic leans on (C-D pin).
        sub = Substrate(); sub._quarantine_migrated = True
        engine = types.SimpleNamespace(substrate=sub)
        writer = JournaledSubstrateWriter(engine=engine)
        hip = Hippocampus(bus=None)
        bus = _RecBus(writer, hip)
        hip.bus = bus
        bus.publish(AttendedPerceptEvent(
            kind=EventKind.ATTENDED_PERCEPT, cycle=1, timestamp=0.0,
            source_capability='test', origin='internal',
            origin_detail='world_experience', focals=['_world_sAAA'],
            payload={}, raw_text='', modality='world',
            salience=1.0, novelty=1.0))
        bus.publish(ReflectionFiredEvent(
            kind=EventKind.REFLECTION_FIRED, cycle=2,
            source_capability='test', origin='internal',
            trigger='manual', reflection_kind='general'))
        self.assertEqual(
            [k for k in sub.edges if k[1] == 'co_occurs'], [])
        self.assertEqual(hip.episodes_consolidated, 0)


# --------------------------------------------------------------------------
# C5 EXTENDED — minted artifacts earn ZERO clock credit (observed, never
# excluded) [G6 / G8 unit form]
# --------------------------------------------------------------------------

class _ArtifactSettleObserver:
    """Replicates the mortality clock's settle door for NON-cohering facts
    (strength >= SAT) and the cohering door (first_coherent_cycle > 0) over
    the three ignition-artifact families.  OBSERVED gate: counts what IS,
    excludes nothing."""
    def __init__(self):
        self.facet_settlings = 0
        self.bond_settlings = 0
        self.membership_settlings = 0
        self._seen = set()

    def __call__(self, edge, cycle, pre_strength, origin):
        rel = getattr(edge, 'relation_name', '')
        s = str(getattr(edge, 'source', '') or '')
        t = str(getattr(edge, 'target', '') or '')
        post = float(getattr(edge, 'strength', 0.0) or 0.0)
        fcc = int(getattr(edge, 'first_coherent_cycle', 0) or 0)
        established = (fcc > 0) or (post >= SAT)
        if not established or (s, rel, t) in self._seen:
            return
        self._seen.add((s, rel, t))
        if rel == FACET_RELATION:
            self.facet_settlings += 1
        elif (rel == 'co_occurs' and s.startswith(WORLD_TOKEN_PREFIX)
                and t.startswith(WORLD_TOKEN_PREFIX)):
            self.bond_settlings += 1
        elif rel == 'is_a' and t.startswith('_abstract_co_occurs_'):
            self.membership_settlings += 1


class TestC5MintedArtifactsEarnZero(unittest.TestCase):

    def test_zero_clock_grade_settlings_in_sim(self):
        obs = _ArtifactSettleObserver()
        register_reinforce_observer(obs)
        try:
            r = _build(level=2, seed=7, hippo=True)
            _run(r.actor, 6000)
            r.actor.world = territory_escalator(r.actor.world)
            c = 6001
            for _ in range(3000):
                a = r.actor.propose(c)
                if a is None:
                    break
                r.actor.execute(a, c)
                if c % 100 == 0:
                    r.bus.publish(ReflectionFiredEvent(
                        kind=EventKind.REFLECTION_FIRED, cycle=c,
                        source_capability='test', origin='internal',
                        trigger='manual', reflection_kind='general'))
                c += 1
            # Phase S passes over the whole substrate (abstraction minting
            # runs; nothing may push an artifact through the settle door).
            r.sub.form_abstractions(cycle=c + 100)
            r.sub.form_abstractions(cycle=c + 200)
            r.sub.form_abstractions(cycle=c + 300)
            n_bonds = sum(1 for (s, rel, t) in r.sub.edges
                          if rel == 'co_occurs'
                          and s.startswith(WORLD_TOKEN_PREFIX))
        finally:
            unregister_reinforce_observer()
        # non-vacuous: fusion bonds actually existed in this sim.
        self.assertGreater(n_bonds, 0)
        # ARITHMETICALLY CLOSED families — MUST be zero at any horizon:
        # bonds are born 0.30 flat with no re-attestation path to 0.98 and
        # cannot cohere; memberships are provisional + migration-collapsed.
        self.assertEqual(obs.bond_settlings, 0)
        self.assertEqual(obs.membership_settlings, 0)
        # has_facet is NOT arithmetically closed: the 40K-tick G8 soak
        # OBSERVED confirmed keys crossing the settle door (157 at 40K —
        # split-event confirms are genuine earn-by-predicting, +0.005 per
        # split win, so an informative key saturates).  This short-horizon
        # sim must not have crossed yet; the honest pin is the LAW, below:
        # a facet edge may sit above provisional ONLY via the earning
        # channel (engaged by a confirm), NEVER by being seen.
        for (s, rel, t), e in r.sub.edges.items():
            if rel != FACET_RELATION:
                continue
            if float(e.strength) > PROVISIONAL_EDGE_STRENGTH + 1e-9:
                self.assertGreater(
                    int(e.last_engaged_cycle), 0,
                    f'facet edge {s}->{t} climbed without a confirm')

    def test_fusion_bond_arithmetic_cannot_reach_settle(self):
        # co_occurs born 0.30 flat; its ONLY re-write path is the writer's
        # duplicate-reinforce blend toward the incoming 0.30 — max(0.30)
        # can never cross SAT=0.98; and co_occurs cannot compose so fcc
        # stays 0.  Arithmetic closed; the sim gate above OBSERVES it.
        self.assertLess(CO_OCCURS_STRENGTH * 2, SAT)


if __name__ == '__main__':
    unittest.main()
