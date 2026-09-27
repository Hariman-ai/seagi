"""Grounding over the firsthand WORLD (Capability 1, Step 5).

End-to-end: WorldDriver -> WorldTransducer -> GroundingLoop.  Proves the
loop learns the firsthand world's token-transitions, RECOVERS when the
hidden law changes (N-robust — recovery, not tied to a grid size), that
it does not hallucinate predictions, and that the precision instrument
discriminates predictable states from noisy ones.  NO lifeforce is staked
(Capability 1): transitions_to stays decoupled from record_learning.
"""

from __future__ import annotations

import sys
import types
import unittest

from seagi.core.substrate import Substrate, WORLD_TOKEN_PREFIX
from seagi.brain.capabilities.grounding import GroundingLoop, WORLD_RELATION
from seagi.brain.capabilities.writer import JournaledSubstrateWriter
from seagi.world.world_driver import WorldDriver
from seagi.brain.capabilities.world_transducer import WorldTransducer


class _DeliverBus:
    def __init__(self, writer):
        self.writer = writer

    def publish(self, event):
        self.writer.handle(event, self)


def _rig():
    sub = Substrate()
    sub._quarantine_migrated = True
    engine = types.SimpleNamespace(substrate=sub)
    writer = JournaledSubstrateWriter(engine=engine)
    loop = GroundingLoop(engine=engine, bus=_DeliverBus(writer))
    return sub, loop


def _recent_confirm(loop, w=40):
    h = [x for x in list(loop.history)[-w:] if x[2] is not None]
    return (sum(1 for (_, _, sp, sa) in h if sp == sa) / len(h)) if h else 0.0


def _drive(loop, trans, driver, steps, c0):
    """Run the world through the transducer into the loop under a
    deterministic policy (action = pure function of the current token,
    so the token->token map is deterministic and learnable)."""
    c = c0
    prev = None
    for _ in range(steps):
        tok = trans.encode(driver.percept()['world_vector'])
        if prev is not None:
            loop.observe(prev, tok, cycle=c)
            c += 1
        action = int(tok[len(WORLD_TOKEN_PREFIX):]) % driver.n_actions
        driver.step(action)
        prev = tok
    return c


class TestGroundingOverWorld(unittest.TestCase):

    def _learn_and_recover(self, grid):
        driver = WorldDriver(grid=grid, k=8, stochastic=False, seed=grid)
        trans = WorldTransducer()
        _, loop = _rig()
        c = _drive(loop, trans, driver, steps=grid * grid * 60, c0=1)
        self.assertGreater(loop.predictions_made, 10)
        self.assertGreater(_recent_confirm(loop), 0.6,
                           f'grid{grid}: did not learn the world')
        misses_before = loop.misses
        driver.reset_law(seed=grid + 100)
        c = _drive(loop, trans, driver, steps=grid * grid * 4, c0=c)
        self.assertGreater(loop.misses, misses_before,
                           f'grid{grid}: law change produced no dip')
        c = _drive(loop, trans, driver, steps=grid * grid * 150, c0=c)
        self.assertGreater(_recent_confirm(loop), 0.6,
                           f'grid{grid}: did not recover after law change')

    def test_recovery_is_n_robust(self):
        # Recovery holds across world sizes -> not tuned to a specific N.
        self._learn_and_recover(4)
        self._learn_and_recover(6)

    def test_not_a_scheduler_no_hallucinated_prediction(self):
        # A never-seen state yields no_prediction, not a hallucinated
        # transition: prediction is earned from structure, not scheduled.
        _, loop = _rig()
        before = loop.predictions_made
        loop.observe('never_seen_state', 'whatever', cycle=1)
        self.assertEqual(loop.predictions_made, before)
        self.assertEqual(loop.no_prediction, 1)

    def test_nontrivial_world_yields_precision_spread(self):
        # Cap-1.5: with the hidden phase-flip ON (derived period =
        # n_cells), transitions from phase-sensitive cells are unreliable
        # while the rest stay reliable -> a REAL precision spread, NOT the
        # trivial all-1.0 / spread-0 world.  This is the proven-grounding
        # signal the 1b mortality gate needs.
        driver = WorldDriver(grid=5, k=8, seed=3)   # stochastic by default
        trans = WorldTransducer()
        _, loop = _rig()
        c = 1
        prev = None
        for _ in range(driver.n_cells() * 400):
            tok = trans.encode(driver.percept()['world_vector'])
            if prev is not None:
                loop.observe(prev, tok, cycle=c)
                c += 1
            # fixed reactive policy: deterministic per token, so stable
            # cells stay perfectly learnable while phase-sensitive cells
            # go ~50/50 across the unobservable phase.
            driver.step(int(tok[len(WORLD_TOKEN_PREFIX):]) % driver.n_actions)
            prev = tok
        rep = loop.precision_report()
        rels = [st[1] for st in loop._rel.values() if st[0] > 0]
        print(f"\n[cap1.5] states={rep['precision_states']} "
              f"mean={rep['reliability_mean']:.3f} "
              f"spread={rep['reliability_spread']:.4f} "
              f"min={min(rels):.3f} max={max(rels):.3f}", file=sys.stderr)
        self.assertGreater(rep['precision_states'], 3)
        self.assertGreater(rep['reliability_spread'], 0.02,
                           f"world still trivial: {rep}")
        # some states reliable, some not -> the spread is real
        self.assertGreater(max(rels), 0.8)
        self.assertLess(min(rels), 0.85)

    @unittest.skip("Wall removed 2026-06-30: world tokens ARE allowed to "
                   "abstract. The architecture is the guard -- credit-by-use "
                   "(runtime._learning_credit) pays only for coherent USE; "
                   "forming credits zero lifeforce, so the farm stays shut "
                   "without a wall. This test asserted the wall, not the guard.")
    def test_world_tokens_cannot_farm_abstractions(self):
        # The deploy guard (wall UP on the canonical merge): world-token
        # concepts entering the substrate must NOT mint abstractions (which
        # would credit record_learning / lifeforce).  A real text group still
        # abstracts — the guard is source-scoped, not a blanket off-switch.
        sub = Substrate()
        sub._quarantine_migrated = True
        # 5 world tokens transition to the same target (a strong
        # (transitions_to, target) regularity that WOULD group if unguarded)
        for i in range(1, 6):
            sub.add_edge(f'{WORLD_TOKEN_PREFIX}{i}', f'{WORLD_TOKEN_PREFIX}9',
                         WORLD_RELATION, strength=0.5, cycle=0)
        # control: 5 text concepts share (kind_of, thing) -> SHOULD abstract
        for nm in ('alpha', 'beta', 'gamma', 'delta', 'epsilon'):
            sub.add_edge(nm, 'thing', 'kind_of', strength=0.5, cycle=0)
        sub.form_abstractions(cycle=10)
        absn = [c for c in sub.concepts if c.startswith('_abstract_')]
        # no abstraction over the world transitions_to regularity ...
        self.assertFalse(
            any(WORLD_RELATION in c for c in absn),
            "world tokens farmed a transitions_to abstraction")
        # ... but the real text regularity DID form one
        self.assertTrue(any('kind_of' in c for c in absn),
                        "guard wrongly blocked a real text abstraction")

    def test_forming_an_abstraction_credits_no_lifeforce(self):
        # The earn-or-dissolve gate that REPLACES the wall: lifeforce credit
        # from a consolidation is newly_coherent ONLY.  Forming abstractions or
        # analogies (even many) credits nothing — they earn lifeforce only when
        # their edges later cohere through USE.  So world-abstractions can form
        # freely and still farm no lifeforce.
        from seagi.brain.runtime import Brain
        self.assertEqual(Brain._learning_credit(0, 50, 50), 0)   # formation only
        # USE CREDITS -- but as a RATE, not a tally (2026-08-10).  The
        # old `== 7` asserted the count semantics that made credit 0 or
        # ~300 with nothing in between, and made a bigger substrate earn
        # more for the same living.  These three assertions pin what the
        # rate law must satisfy, which the equality could not express.
        self.assertGreater(
            Brain._learning_credit(7, 50, 50, reinforced=0), 0.0,
            'use must still credit something -- if this is 0 the earn '
            'path is dead and nothing can raise the set-point.')
        self.assertLessEqual(
            Brain._learning_credit(10000, 50, 50, reinforced=0), 1.0,
            'credit must be BOUNDED -- an unbounded credit pins the '
            'baseline at the ceiling and ends the M/I tension.')
        self.assertAlmostEqual(
            Brain._learning_credit(10, 50, 50, reinforced=90),
            Brain._learning_credit(1000, 50, 50, reinforced=9000),
            places=9,
            msg='credit must be SIZE-INDEPENDENT -- a bigger brain must '
                'not earn more for the same quality of living.')
        # And credit depends ONLY on newly_coherent: the same coherence
        # earns the same whether 0 or 50 abstractions formed alongside
        # it.  (Was `== 3`, the tally semantics; independence is what the
        # line was actually guarding and it survives the rate law.)
        self.assertEqual(
            Brain._learning_credit(3, 0, 0, reinforced=7),
            Brain._learning_credit(3, 50, 50, reinforced=7),
            'formation counts leaked into the credit -- only coherence '
            'through USE may earn.')

    def test_used_class_earns_lifeforce_by_coherence(self):
        # The COMPLEMENT of the no-farm contract: an abstraction that is USED
        # (its is_a edges drive inheritance) derives edges that COHERE, which DO
        # credit lifeforce.  So credit-by-use doesn't render abstractions unable
        # to earn — they earn exactly when used, not when formed.
        from seagi.brain.runtime import Brain
        sub = Substrate()
        sub._quarantine_migrated = True
        for m in ('cat', 'dog', 'cow'):
            sub.add_edge(m, 'mammal', 'is_a', strength=0.7, cycle=0)
        sub.add_edge('mammal', 'warm', 'has_property', strength=0.7, cycle=0)
        sub.derive_closure(1)                          # USE: inheritance
        _, newly_coherent = sub.reinforce_coherent_edges(2)
        self.assertGreater(newly_coherent, 0,
                           "using the class should cohere derived edges")
        self.assertGreater(Brain._learning_credit(newly_coherent, 0, 0), 0,
                           "a used abstraction must still earn lifeforce")

    def test_is_a_index_finds_siblings_and_rebuilds(self):
        # The reverse-is_a index that wakes the out-wire: O(siblings) sibling
        # lookup, maintained incrementally on add_edge and rebuilt on load.
        s = Substrate()
        s._quarantine_migrated = True
        for m in ('a', 'b', 'c'):
            s.add_edge(m, 'K', 'is_a', strength=0.5, cycle=0)
        s.add_edge('x', 'OTHER', 'is_a', strength=0.5, cycle=0)
        self.assertEqual(s.is_a_siblings('a'), {'b', 'c'})   # share parent K
        self.assertEqual(s.is_a_siblings('x'), set())        # different parent
        s._is_a_children = {}                                # simulate post-load
        s._rebuild_is_a_index()                              # the load rebuild
        self.assertEqual(s.is_a_siblings('a'), {'b', 'c'})

    def test_precision_discriminates_predictable_from_noisy(self):
        _, loop = _rig()
        c = 1
        for _ in range(30):           # deterministic state: d -> x always
            loop.observe('d', 'x', cycle=c)
            c += 1
        for i in range(30):           # noisy state: q -> y / z alternating
            loop.observe('q', 'y' if i % 2 == 0 else 'z', cycle=c)
            c += 1
        rel = {k: st[1] for k, st in loop._rel.items()}
        self.assertGreater(rel['d'], rel['q'])           # predictable > noisy
        rep = loop.precision_report()
        self.assertGreater(rep['reliability_spread'], 0.0)
        self.assertGreaterEqual(rep['precision_states'], 2)


if __name__ == '__main__':
    unittest.main()
