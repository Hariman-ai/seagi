"""FACT-GRAIN tests — what counts as ONE fact (design v2, 2026-07-19).

Behavioural tests on a micro-substrate + the live MortalityClock, plus
the two auditor-gate structural tests.

The grain under test (MortalityClock._fact_key):
  (i)   (s, is_a, M) with M synthetic          -> ('∈', M)
  else                                         -> (s, r, t) instance

v1's branch (ii) must NOT appear: a plain member edge the machinery does
not entail stays INSTANCE grain.

BRANCH (iii) WAS WITHDRAWN FROM THE KEY (2026-07-22 incident).  It ran
cortical's composition walk from inside Edge.reinforce and froze the live
daemon for ~45 minutes; it also collapsed a MEASURED ZERO edges on the
real substrate.  The tests below are NOT deleted — they are inverted to
pin the new law:

  * an entailed derivation now keeps INSTANCE grain (was: family grain),
    and each member therefore pays separately.  That is the re-opened
    derived-edge leak, accepted knowingly because it is measured zero.
  * `_fact_key` must NEVER reach `_entailing_family_class` or the walk.
    TestBranchThreeIsNotInTheHotPath enforces that structurally, on the
    AST, so a future edit cannot quietly reintroduce it.
  * `_entailing_family_class` and cortical's `family_grain_relations`
    must STILL EXIST and still be correct — the offline dual-grain
    harness re-measures the leak with them.
"""

from __future__ import annotations

import ast
import os
import unittest

from seagi.core.substrate import (
    Substrate,
    ABSTRACTION_RELATION,
    unregister_reinforce_observer,
)
from seagi.brain.capabilities import cortical as cortical_mod
from seagi.brain.capabilities import mortality_clock as mc_mod
from seagi.brain.capabilities.mortality_clock import (
    MortalityClock,
    MEMBERSHIP_KEY_HEAD,
    _SETTLE_WORLD_098,
    _composable_result_relations,
)

_MC_PATH = os.path.abspath(mc_mod.__file__).replace('.pyc', '.py')


class _DriveStub:
    """Read-only stand-in for MortalityDrive."""

    def __init__(self):
        self.deaths = 0
        self.revivals = 0
        self.wall = 0.1
        self.obstruction_ema = 0.0
        self._lf_get = lambda: 0.7
        self._last_wall_advance = 0.0


def _clock(sub=None, **kw):
    kw.setdefault('log_path', '')
    kw.setdefault('death_log_path', '')
    return MortalityClock(
        mortality_drive=_DriveStub(),
        substrate_provider=((lambda: sub) if sub is not None else None),
        **kw)


def _settlings(clock) -> int:
    """Total credited settlings across every class x origin."""
    return int(sum(
        clock._settle_alltime[cls][oc]['n']
        for cls in clock._settle_alltime
        for oc in clock._settle_alltime[cls]))


def _synth(sub: Substrate, name: str) -> str:
    """Mint a synthetic class concept the way form_abstractions does."""
    c = sub.get_or_create_concept(name, cycle=0)
    c.synthetic = True
    return name


def _edge(sub: Substrate, s, r, t, *, fcc=0, strength=0.5, cycle=0):
    e = sub.add_edge(source=s, target=t, relation_name=r,
                     strength=strength, cycle=cycle)
    e.first_coherent_cycle = int(fcc)
    return e


class _GrainBase(unittest.TestCase):

    def tearDown(self):
        unregister_reinforce_observer()


# =====================================================================
# (i) MEMBERSHIP — naming a class makes membership ONE fact
# =====================================================================
class TestMembershipGrain(_GrainBase):

    def test_three_members_credit_exactly_once(self):
        sub = Substrate()
        cls = _synth(sub, '_abstract_has_property_red')
        for m in ('alpha', 'beta', 'gamma'):
            _edge(sub, m, ABSTRACTION_RELATION, cls, fcc=5)
        clock = _clock(sub=sub)
        for m in ('alpha', 'beta', 'gamma'):
            sub.edges[(m, ABSTRACTION_RELATION, cls)].reinforce(
                1, origin='cognition')
        self.assertEqual(_settlings(clock), 1,
                         'three memberships of one named class are ONE fact')
        self.assertEqual(len(clock._settled), 1)

    def test_membership_key_is_the_family_key(self):
        sub = Substrate()
        cls = _synth(sub, '_abstract_causes_rain')
        e1 = _edge(sub, 'alpha', ABSTRACTION_RELATION, cls, fcc=5)
        e2 = _edge(sub, 'beta', ABSTRACTION_RELATION, cls, fcc=5)
        clock = _clock(sub=sub)
        expected = clock._khash((MEMBERSHIP_KEY_HEAD, cls))
        self.assertEqual(clock._fact_key(e1), expected)
        self.assertEqual(clock._fact_key(e2), expected)

    def test_is_a_into_a_NON_synthetic_target_stays_instance(self):
        """Only a MINTED class collapses.  An ordinary is_a is a fact."""
        sub = Substrate()
        e = _edge(sub, 'socrates', ABSTRACTION_RELATION, 'man', fcc=5)
        clock = _clock(sub=sub)
        self.assertEqual(
            clock._fact_key(e),
            clock._khash(('socrates', ABSTRACTION_RELATION, 'man')))

    def test_real_form_abstractions_pricing_end_to_end(self):
        """Drive the REAL abstraction pass: 3 members sharing (r, t) mint
        a class, and their memberships then price as one fact."""
        sub = Substrate()
        for m in ('robin', 'crow', 'wren'):
            _edge(sub, m, 'has_property', 'feathered', strength=0.5)
        created = sub.form_abstractions(cycle=1, min_group=3)
        self.assertGreaterEqual(created, 1)
        cls = '_abstract_has_property_feathered'
        self.assertIn(cls, sub.concepts)
        self.assertTrue(sub.concepts[cls].synthetic)
        clock = _clock(sub=sub)
        keys = set()
        for m in ('robin', 'crow', 'wren'):
            e = sub.edges.get((m, ABSTRACTION_RELATION, cls))
            self.assertIsNotNone(e, 'member %s should have an is_a' % m)
            keys.add(clock._fact_key(e))
        self.assertEqual(len(keys), 1,
                         'all memberships of the minted class share one key')


# =====================================================================
# (ii) ABSENT — a plain member edge stays INSTANCE grain
# =====================================================================
class TestNoBranchTwo(_GrainBase):

    def test_learned_non_membership_edge_from_a_member_is_instance(self):
        sub = Substrate()
        cls = _synth(sub, '_abstract_has_property_red')
        _edge(sub, 'alpha', ABSTRACTION_RELATION, cls, fcc=5)
        # 'alpha' learns something the class does NOT entail.
        e = _edge(sub, 'alpha', 'has_property', 'shiny', fcc=7)
        clock = _clock(sub=sub)
        self.assertEqual(
            clock._fact_key(e),
            clock._khash(('alpha', 'has_property', 'shiny')),
            'membership alone must not collapse unrelated learning')

    def test_two_members_learning_different_facts_credit_twice(self):
        sub = Substrate()
        cls = _synth(sub, '_abstract_has_property_red')
        for m in ('alpha', 'beta'):
            _edge(sub, m, ABSTRACTION_RELATION, cls, fcc=5)
        _edge(sub, 'alpha', 'has_property', 'shiny', fcc=7)
        _edge(sub, 'beta', 'has_property', 'rough', fcc=7)
        clock = _clock(sub=sub)
        sub.edges[('alpha', 'has_property', 'shiny')].reinforce(
            1, origin='cognition')
        sub.edges[('beta', 'has_property', 'rough')].reinforce(
            1, origin='cognition')
        self.assertEqual(_settlings(clock), 2)


# =====================================================================
# (iii) WITHDRAWN — a machinery-entailed derivation now keeps INSTANCE
#       grain.  These were the branch-(iii) behavioural tests; they are
#       INVERTED rather than deleted, so the accepted cost of the
#       2026-07-22 removal is a pinned, visible fact and not a silence.
# =====================================================================
class TestEntailedDerivationKeepsInstanceGrain(_GrainBase):

    def _entailed_world(self):
        """M --has_property--> warm ; alpha,beta,gamma is_a M.

        Under branch (iii) all three member edges collapsed to the single
        family fact (M, has_property, warm).  They no longer do."""
        sub = Substrate()
        cls = _synth(sub, '_abstract_is_a_mammal')
        _edge(sub, cls, 'has_property', 'warm', strength=0.6)
        for m in ('alpha', 'beta', 'gamma'):
            _edge(sub, m, ABSTRACTION_RELATION, cls, fcc=5)
            _edge(sub, m, 'has_property', 'warm', fcc=9)
        return sub, cls

    def test_entailed_edge_keeps_instance_grain(self):
        """THE ACCEPTED COST, stated as an assertion.  This is the
        derived-edge leak: measured ZERO on the real substrate
        (branch_iii_composition_collapse_edges: 0), re-measured OFFLINE
        by the dual-grain harness, never by a walk on the hot path."""
        sub, cls = self._entailed_world()
        clock = _clock(sub=sub)
        family = clock._khash((cls, 'has_property', 'warm'))
        for m in ('alpha', 'beta', 'gamma'):
            e = sub.edges[(m, 'has_property', 'warm')]
            self.assertEqual(clock._fact_key(e),
                             clock._khash((m, 'has_property', 'warm')),
                             '%s must keep instance grain' % m)
            self.assertNotEqual(clock._fact_key(e), family,
                                'branch (iii) must not be keying anything')

    def test_each_member_now_credits_separately(self):
        """Was: 'one class-level fact = one settling'.  Now three."""
        sub, _cls = self._entailed_world()
        clock = _clock(sub=sub)
        for m in ('alpha', 'beta', 'gamma'):
            sub.edges[(m, 'has_property', 'warm')].reinforce(
                1, origin='cognition')
        self.assertEqual(_settlings(clock), 3,
                         'instance grain -> one settling per member')

    def test_membership_still_collapses_around_the_same_class(self):
        """Branch (i) is UNTOUCHED by the removal — the collapse that
        actually works (validated live: predicted 35,490 vs 35,504)."""
        sub, cls = self._entailed_world()
        clock = _clock(sub=sub)
        expected = clock._khash((MEMBERSHIP_KEY_HEAD, cls))
        for m in ('alpha', 'beta', 'gamma'):
            e = sub.edges[(m, ABSTRACTION_RELATION, cls)]
            self.assertEqual(clock._fact_key(e), expected)

    def test_composable_relation_from_a_member_is_instance(self):
        """A composition-RESULT relation out of a class member used to
        trigger the walk.  It must now short-circuit at the relation
        test and never look at the class at all."""
        sub, cls = self._entailed_world()
        clock = _clock(sub=sub)
        comp = _composable_result_relations()
        self.assertIsNotNone(comp)
        rel = sorted(comp)[0]
        _edge(sub, cls, rel, 'somewhere', strength=0.6)
        e = _edge(sub, 'alpha', rel, 'somewhere', fcc=9)
        self.assertEqual(clock._fact_key(e),
                         clock._khash(('alpha', rel, 'somewhere')))
        self.assertEqual(clock._fact_key_failopen, 0)

    def test_non_composable_relation_is_instance(self):
        sub = Substrate()
        cls = _synth(sub, '_abstract_is_a_thing')
        _edge(sub, cls, 'zz_not_composable', 'somewhere', strength=0.6)
        _edge(sub, 'alpha', ABSTRACTION_RELATION, cls, fcc=5)
        e = _edge(sub, 'alpha', 'zz_not_composable', 'somewhere', fcc=9)
        clock = _clock(sub=sub)
        self.assertEqual(
            clock._fact_key(e),
            clock._khash(('alpha', 'zz_not_composable', 'somewhere')))


# =====================================================================
# THE HOT-PATH LAW (2026-07-22 incident)
# =====================================================================
class TestBranchThreeIsNotInTheHotPath(_GrainBase):
    """`_fact_key` runs inside Edge.reinforce, on every write.  Nothing
    that traverses the substrate may be reachable from it.  These tests
    are STRUCTURAL (AST + monkeypatch), because a behavioural test only
    catches the walk when a fixture happens to entail something."""

    def _fact_key_ast(self):
        src = open(_MC_PATH, encoding='utf-8').read()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == '_fact_key':
                return node
        self.fail('_fact_key not found in the clock source')

    def test_fact_key_source_never_names_the_walk(self):
        node = self._fact_key_ast()
        names = {n.attr for n in ast.walk(node)
                 if isinstance(n, ast.Attribute)}
        names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        for banned in ('_entailing_family_class', 'family_grain_relations',
                       '_family_walk'):
            self.assertNotIn(
                banned, names,
                'branch (iii) is back in the hot path: _fact_key names %s'
                % banned)

    def test_fact_key_calls_nothing_that_can_traverse(self):
        """Whitelist the callees.  A new call into _fact_key is a
        deliberate act and must be added here on purpose."""
        node = self._fact_key_ast()
        called = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name):
                    called.add(f.id)
                elif isinstance(f, ast.Attribute):
                    called.add(f.attr)
        allowed = {'str', 'getattr', '_safe_call', '_khash',
                   '_is_synthetic', 'get'}
        self.assertTrue(
            called <= allowed,
            'unexpected call(s) in the reinforce hot path: %s'
            % sorted(called - allowed))

    def test_walk_is_never_invoked_during_a_reinforce(self):
        """Live proof: detonate cortical's walk, then reinforce an edge
        that branch (iii) WOULD have keyed.  It must not fire."""
        sub = Substrate()
        cls = _synth(sub, '_abstract_is_a_mammal')
        _edge(sub, cls, 'has_property', 'warm', strength=0.6)
        _edge(sub, 'alpha', ABSTRACTION_RELATION, cls, fcc=5)
        _edge(sub, 'alpha', 'has_property', 'warm', fcc=9)
        clock = _clock(sub=sub)
        calls = []

        def _boom(*a, **kw):
            calls.append(a)
            raise AssertionError('the walk ran on the reinforce path')

        orig_mod = cortical_mod.family_grain_relations
        orig_cached = mc_mod._FAMILY_WALK
        cortical_mod.family_grain_relations = _boom
        mc_mod._FAMILY_WALK = (_boom, 4)
        mc_mod._FAMILY_WALK_LOADED = True
        try:
            sub.edges[('alpha', 'has_property', 'warm')].reinforce(
                1, origin='cognition')
            clock._preseed_settled(0)
        finally:
            cortical_mod.family_grain_relations = orig_mod
            mc_mod._FAMILY_WALK = orig_cached
        self.assertEqual(calls, [], 'the walk was called %d time(s)'
                         % len(calls))
        self.assertEqual(clock._fact_key_failopen, 0,
                         'and it must not have been swallowed as a failopen')

    def test_the_walk_is_still_available_for_OFFLINE_measurement(self):
        """Removal is from the KEY, not from the codebase.  The
        dual-grain harness must still be able to re-measure the leak."""
        self.assertTrue(hasattr(cortical_mod, 'family_grain_relations'))
        self.assertTrue(callable(
            getattr(MortalityClock, '_entailing_family_class', None)))
        sub = Substrate()
        cls = _synth(sub, '_abstract_is_a_mammal')
        _edge(sub, cls, 'has_property', 'warm', strength=0.6)
        _edge(sub, 'alpha', ABSTRACTION_RELATION, cls, fcc=5)
        clock = _clock(sub=sub)
        self.assertEqual(
            clock._entailing_family_class(
                sub.concepts, 'alpha', 'has_property', 'warm'),
            cls,
            'the offline measurement path must still be CORRECT')

    def test_entailing_family_class_is_marked_offline_only(self):
        src = open(_MC_PATH, encoding='utf-8').read()
        i = src.index('def _entailing_family_class')
        self.assertIn('OFFLINE MEASUREMENT ONLY', src[i:i + 400],
                      'the banner is the only thing stopping the next '
                      'reader from wiring it back into the hot path')


# =====================================================================
# WORLD_098 — structurally unreachable by either family branch
# =====================================================================
class TestWorldNeverFamilyKeyed(_GrainBase):

    def test_transitions_to_hub_member_settles_at_instance_grain(self):
        sub = Substrate()
        cls = _synth(sub, '_abstract_transitions_to_s_goal')
        # a transitions_to "hub": many world states share (transitions_to,
        # s_goal) and are members of the minted class
        for i in range(5):
            w = '_world_s%02d' % i
            _edge(sub, w, ABSTRACTION_RELATION, cls, fcc=3)
            _edge(sub, w, 'transitions_to', 's_goal', strength=0.977)
        _edge(sub, cls, 'transitions_to', 's_goal', strength=0.6)
        clock = _clock(sub=sub)
        # the world edge classifies world_098 ...
        self.assertEqual(clock._settle_class(0, 'transitions_to'),
                         _SETTLE_WORLD_098)
        # ... and keeps INSTANCE grain, by construction not by a guard
        for i in range(5):
            w = '_world_s%02d' % i
            e = sub.edges[(w, 'transitions_to', 's_goal')]
            self.assertEqual(clock._fact_key(e),
                             clock._khash((w, 'transitions_to', 's_goal')))

    def test_world_098_settlings_each_credit_separately(self):
        sub = Substrate()
        cls = _synth(sub, '_abstract_transitions_to_s_goal')
        for i in range(3):
            w = '_world_s%02d' % i
            _edge(sub, w, ABSTRACTION_RELATION, cls, fcc=3)
            _edge(sub, w, 'transitions_to', 's_goal', strength=0.977)
        clock = _clock(sub=sub)
        for i in range(3):
            sub.edges[('_world_s%02d' % i, 'transitions_to',
                       's_goal')].reinforce(1, origin='maintenance')
        world_n = int(sum(
            clock._settle_alltime[_SETTLE_WORLD_098][oc]['n']
            for oc in clock._settle_alltime[_SETTLE_WORLD_098]))
        self.assertEqual(world_n, 3,
                         'world facts are individual; no family collapse')

    def test_invariant_world_098_implies_instance_grain(self):
        """The structural claim, asserted directly: world_098 means a
        NON-composable relation, and both family branches require is_a or
        a composition result -- so the two can never intersect."""
        comp = _composable_result_relations()
        self.assertIsNotNone(comp)
        self.assertNotIn('transitions_to', comp)
        self.assertIn(ABSTRACTION_RELATION, comp)


# =====================================================================
# PRE-SEED PARITY
# =====================================================================
class TestPreseedParity(_GrainBase):

    def test_preseeded_entailed_edge_credits_zero_on_next_reinforce(self):
        """Pre-seed/observer parity survives the branch-(iii) removal:
        both sides now key these edges at INSTANCE grain, and they still
        agree, so an inherited mind still earns nothing for what it
        already knows."""
        sub = Substrate()
        cls = _synth(sub, '_abstract_is_a_mammal')
        _edge(sub, cls, 'has_property', 'warm', strength=0.6)
        for m in ('alpha', 'beta'):
            _edge(sub, m, ABSTRACTION_RELATION, cls, fcc=5)
            _edge(sub, m, 'has_property', 'warm', fcc=9)
        clock = _clock(sub=sub)
        clock._preseed_settled(0)             # inherited mind, no credit
        self.assertEqual(_settlings(clock), 0, 'pre-seed credits nothing')
        self.assertGreater(clock._preseeded_count, 0)
        sub.edges[('alpha', 'has_property', 'warm')].reinforce(
            1, origin='cognition')
        self.assertEqual(_settlings(clock), 0,
                         're-observing an inherited fact earns 0')
        sub.edges[('beta', 'has_property', 'warm')].reinforce(
            2, origin='cognition')
        self.assertEqual(_settlings(clock), 0)

    def test_preseed_and_observer_agree_on_every_key(self):
        """The pre-seed must mark the SAME digest the observer looks up;
        any divergence double-pays the whole inherited population.

        Scope note: the pre-seed marks ESTABLISHED edges only (fcc>0 or
        saturated), so that is the set the parity claim covers.  Before
        the branch-(iii) removal the unestablished class-level SUPPORT
        edge also appeared marked -- not because the pre-seed marked it,
        but because branch (iii) mapped the member's derived edge onto
        exactly that instance key.  That coincidence is gone; the parity
        property it was standing in for is asserted directly here."""
        sub = Substrate()
        cls = _synth(sub, '_abstract_is_a_mammal')
        _edge(sub, cls, 'has_property', 'warm', strength=0.6)   # NOT est.
        _edge(sub, 'alpha', ABSTRACTION_RELATION, cls, fcc=5)
        _edge(sub, 'alpha', 'has_property', 'warm', fcc=9)
        _edge(sub, 'alpha', 'zz_world', 'elsewhere', strength=0.99)
        clock = _clock(sub=sub)
        clock._preseed_settled(0)
        established, unestablished = [], []
        for e in sub.edges.values():
            fcc = int(getattr(e, 'first_coherent_cycle', 0) or 0)
            try:
                eff = float(e.effective_strength(0))
            except Exception:
                eff = float(getattr(e, 'strength', 0.0) or 0.0)
            (established if (fcc > 0 or eff >= mc_mod.SATURATION_CEILING)
             else unestablished).append(e)
        self.assertTrue(established and unestablished, 'fixture is degenerate')
        for e in established:
            self.assertIn(clock._fact_key(e), clock._settled,
                          'observer key must match what the pre-seed marked')
        # and an established edge must therefore earn NOTHING on re-observe
        before = _settlings(clock)
        for e in established:
            e.reinforce(1, origin='cognition')
        self.assertEqual(_settlings(clock), before,
                         'inherited facts must not re-earn')


# =====================================================================
# FAIL-OPEN (auditor gate 3)
# =====================================================================
class _RaisingConcepts(dict):
    def get(self, *a, **kw):
        raise RuntimeError('substrate read blew up')


class _BadSub:
    concepts = _RaisingConcepts()
    edges = {}
    quarantine_edges = {}


class TestFailOpen(_GrainBase):

    def test_provider_unavailable_credits_instance_and_counts(self):
        clock = _clock(sub=None)          # no substrate provider at all
        sub = Substrate()
        e = _edge(sub, 'alpha', 'has_property', 'warm', fcc=9)
        self.assertEqual(clock._fact_key(e),
                         clock._khash(('alpha', 'has_property', 'warm')))
        self.assertEqual(clock._fact_key_failopen, 1)
        self.assertEqual(clock.stats()['fact_key_failopen'], 1)

    def test_structure_read_error_credits_instance_and_counts(self):
        """A structure read that BLOWS UP still fails open + counts.

        Was test_walk_error_*: the raising read used to be the branch
        (iii) walk, reachable from any composable relation.  The only
        structure read left is `_is_synthetic` on a membership edge, so
        that is where the fail-open is now exercised."""
        clock = _clock(sub=_BadSub())
        sub = Substrate()
        e = _edge(sub, 'alpha', ABSTRACTION_RELATION, 'some_class', fcc=9)
        self.assertEqual(clock._fact_key(e),
                         clock._khash(('alpha', ABSTRACTION_RELATION,
                                       'some_class')))
        self.assertEqual(clock._fact_key_failopen, 1)

    def test_non_membership_edge_reads_no_structure_at_all(self):
        """The NARROWED read surface, asserted rather than assumed: a
        non-membership edge takes the instance answer without touching
        the substrate, so a broken substrate cannot even be noticed --
        and nothing is degraded, because instance grain is the correct
        answer for that edge either way.  This is also why the hot path
        is cheap."""
        clock = _clock(sub=_BadSub())
        sub = Substrate()
        e = _edge(sub, 'alpha', 'has_property', 'warm', fcc=9)
        self.assertEqual(clock._fact_key(e),
                         clock._khash(('alpha', 'has_property', 'warm')))
        self.assertEqual(clock._fact_key_failopen, 0)

    def test_failopen_is_surfaced_for_status(self):
        clock = _clock(sub=None)
        self.assertIn('fact_key_failopen', clock.stats())
        self.assertIn('key_digest', clock.stats())

    def test_normal_no_entailment_is_NOT_a_failopen(self):
        sub = Substrate()
        e = _edge(sub, 'alpha', 'has_property', 'warm', fcc=9)
        clock = _clock(sub=sub)
        clock._fact_key(e)
        self.assertEqual(clock._fact_key_failopen, 0,
                         'instance grain is an ANSWER, not a failure')


# =====================================================================
# MIGRATION GUARD
# =====================================================================
class TestDigestMigration(_GrainBase):

    def test_key_digest_is_fam2i(self):
        """The digest names the GRAIN LAW.  Branch (iii) leaving the key
        changed the law, so the name changed with it -- a fam2 ledger is
        a different grain and must not be silently reused."""
        self.assertEqual(mc_mod.KEY_DIGEST, 'blake2b8/fam2i')

    def test_instance_grain_ledger_is_rejected_on_load(self):
        """An old-grain ledger must NOT load into the new grain -- the
        digest guard is what makes the migration automatic."""
        clock = _clock(sub=None)
        clock.load_dict({'schema': 1, 'key_digest': 'blake2b8',
                         'settled': [1, 2, 3], 'dormant': []})
        self.assertEqual(len(clock._settled), 0,
                         'old-grain digests must be refused')

    def test_branch_three_ledger_is_also_rejected_on_load(self):
        """A ledger written by the STALLED fam2 build carries family keys
        this build would never mint.  Refuse it too."""
        clock = _clock(sub=None)
        clock.load_dict({'schema': 1, 'key_digest': 'blake2b8/fam2',
                         'settled': [1, 2, 3], 'dormant': []})
        self.assertEqual(len(clock._settled), 0,
                         'branch-(iii) digests must be refused')

    def test_new_grain_ledger_round_trips(self):
        sub = Substrate()
        cls = _synth(sub, '_abstract_has_property_red')
        _edge(sub, 'alpha', ABSTRACTION_RELATION, cls, fcc=5)
        clock = _clock(sub=sub)
        sub.edges[('alpha', ABSTRACTION_RELATION, cls)].reinforce(
            1, origin='cognition')
        state = clock.to_dict()
        self.assertEqual(state['key_digest'], 'blake2b8/fam2i')
        clock2 = _clock(sub=sub)
        clock2.load_dict(state)
        self.assertEqual(clock2._settled, clock._settled)
        # and a re-observation earns nothing
        before = _settlings(clock2)
        sub.edges[('alpha', ABSTRACTION_RELATION, cls)].reinforce(
            2, origin='cognition')
        self.assertEqual(_settlings(clock2), before)


# =====================================================================
# AUDITOR GATE 2 — one composition set, one composition machinery
# =====================================================================
class TestCompositionSetDrift(_GrainBase):

    def test_composable_set_is_a_single_cached_object(self):
        a = _composable_result_relations()
        b = _composable_result_relations()
        self.assertIs(a, b, 'the set must be built once, not per-call')

    def test_composable_set_is_derived_from_cortical(self):
        self.assertEqual(
            _composable_result_relations(),
            frozenset(cortical_mod.RELATION_COMPOSITION.values()),
            'the clock must not carry its own copy of the results')

    def test_exactly_one_construction_site_in_the_clock(self):
        src = open(_MC_PATH, encoding='utf-8').read()
        self.assertEqual(
            src.count('frozenset(RELATION_COMPOSITION.values())'), 1,
            'a second construction site is exactly how the two sets drift')
        self.assertEqual(
            src.count('RELATION_COMPOSITION: Dict'), 0,
            'the clock must never define its own composition table')

    def test_settle_class_and_fact_key_use_the_SAME_accessor(self):
        """Both consumers must call _composable_result_relations().  If
        either ever inlines its own set, this fails."""
        tree = ast.parse(open(_MC_PATH, encoding='utf-8').read())
        seen = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name in ('_settle_class', '_fact_key')):
                seen[node.name] = {
                    c.func.id for c in ast.walk(node)
                    if isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Name)}
        self.assertIn('_settle_class', seen)
        self.assertIn('_composable_result_relations', seen['_settle_class'],
                      '_settle_class must use the shared accessor')
        # `_fact_key` was the SECOND consumer -- branch (iii) gated on the
        # composable set.  With (iii) gone there is exactly one consumer,
        # so the drift this gate guarded against is now structurally
        # impossible.  Assert the absence, so re-adding a consumer is a
        # deliberate act that has to come back through this test.
        self.assertIn('_fact_key', seen)
        self.assertNotIn(
            '_composable_result_relations', seen['_fact_key'],
            'the reinforce hot path must not consult the composable set '
            '-- that gate was branch (iii), which was removed 2026-07-22')

    def test_offline_walk_delegates_to_corticals_machinery(self):
        """The OFFLINE walk must be CORTICAL's, never re-implemented
        here.  (No longer a hot-path concern -- `_fact_key` does not
        reach this -- but the harness's answer must still be the
        deriver's own answer.)"""
        self.assertTrue(hasattr(cortical_mod, 'family_grain_relations'))
        self.assertTrue(hasattr(cortical_mod, 'compose'))
        walk = mc_mod._family_walk()
        self.assertIsNotNone(walk, 'clock must reach the walk')
        self.assertIs(walk[0], cortical_mod.family_grain_relations)
        self.assertEqual(walk[1], cortical_mod.INFERENCE_MAX_HOPS)

    def test_compose_is_the_single_composition_step(self):
        src = open(os.path.abspath(cortical_mod.__file__).replace(
            '.pyc', '.py'), encoding='utf-8').read()
        self.assertEqual(
            src.count('RELATION_COMPOSITION.get('), 1,
            'compose() must be the only place the table is consulted')

    def test_walk_bounds_reuse_shipped_constants(self):
        """No new constants: the walk is bounded by INFERENCE_MAX_HOPS
        and COHERENCE_HUB_DEGREE with their original meanings."""
        from seagi.core.substrate import COHERENCE_HUB_DEGREE
        self.assertEqual(cortical_mod.INFERENCE_MAX_HOPS, 4)
        self.assertEqual(COHERENCE_HUB_DEGREE, 200)

    def test_walk_is_side_effect_free(self):
        """The walk runs inside a reinforce observer; if it stamped
        engagement it would feed cognition from the accountant."""
        sub = Substrate()
        cls = _synth(sub, '_abstract_is_a_mammal')
        _edge(sub, cls, 'has_property', 'warm', strength=0.6)
        _edge(sub, 'alpha', ABSTRACTION_RELATION, cls, fcc=5)
        e = _edge(sub, 'alpha', 'has_property', 'warm', fcc=9)
        support = sub.edges[(cls, 'has_property', 'warm')]
        before = (support.strength, support.last_engaged_cycle,
                  support.last_reinforced_cycle)
        clock = _clock(sub=sub)
        clock._fact_key(e)
        after = (support.strength, support.last_engaged_cycle,
                 support.last_reinforced_cycle)
        self.assertEqual(before, after, 'the walk must mutate nothing')

    def test_walk_hop_bound_is_respected(self):
        """A chain longer than INFERENCE_MAX_HOPS must not entail."""
        sub = Substrate()
        cls = _synth(sub, '_abstract_is_a_chain')
        # M is_a n1 is_a n2 is_a n3 is_a far  -> 5 hops incl. membership
        _edge(sub, cls, 'is_a', 'n1', strength=0.6)
        _edge(sub, 'n1', 'is_a', 'n2', strength=0.6)
        _edge(sub, 'n2', 'is_a', 'n3', strength=0.6)
        _edge(sub, 'n3', 'is_a', 'far', strength=0.6)
        _edge(sub, 'alpha', ABSTRACTION_RELATION, cls, fcc=5)
        e = _edge(sub, 'alpha', 'is_a', 'far', fcc=9)
        clock = _clock(sub=sub)
        self.assertEqual(clock._fact_key(e),
                         clock._khash(('alpha', 'is_a', 'far')),
                         'beyond the hop bound -> instance grain')


if __name__ == '__main__':
    unittest.main()
