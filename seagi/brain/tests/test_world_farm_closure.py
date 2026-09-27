"""FARM-CLOSURE GUARD (seagi-doctrine-auditor must-fix, 2026-06-25).

INVARIANT: world experience must NEVER reach `newly_coherent` -- the ONLY
signal that credits lifeforce (runtime `_learning_credit`: earned =
int(newly_coherent)).  If a world-token edge ever sets `first_coherent_cycle`,
the agent can extend its life by looping the world = the RL/farm backdoor the
whole world-isolation exists to prevent.

WHY THIS GUARD IS NEEDED NOW: world transition edges use `transitions_to`,
deliberately excluded from RELATION_COMPOSITION (grounding.py:62), so they
cannot cohere.  BUT with the abstraction wall DOWN, world states form `is_a`
edges to a shared `_abstract_*` class, and `('is_a','is_a')->'is_a'` IS
composable (cortical.py:142).  So farm-closure no longer rests on the
`transitions_to` firewall alone -- it rests on world abstractions being is_a
TARGETS only, with NO outgoing composing hop to complete an A->X->B chain.
That is a STRUCTURAL ACCIDENT, not an explicit guard.  These tests pin it:
they fail loudly the moment a future second-hop wiring (meta-abstraction over
abstractions, a composable world relation, etc.) lets a world chain complete
and reach newly_coherent.
"""
import unittest

from seagi.core.substrate import Substrate, WORLD_TOKEN_PREFIX


class TestWorldFarmClosure(unittest.TestCase):

    def _fcc(self, sub, s, r, t):
        e = sub.edges.get((s, r, t))
        return int(getattr(e, 'first_coherent_cycle', 0) or 0) if e is not None else -1

    def _world_cohered(self, sub):
        WP = WORLD_TOKEN_PREFIX
        return [(s, r, t) for (s, r, t), e in sub.edges.items()
                if (str(s).startswith(WP) or str(t).startswith(WP))
                and int(getattr(e, 'first_coherent_cycle', 0) or 0) > 0]

    def test_walldown_structure_never_coheres(self):
        """The canonical wall-down structure (world states transitions_to a
        chain + is_a a shared abstraction) must not cohere -- with a live
        positive control proving the coherence pass actually fires here."""
        WP = WORLD_TOKEN_PREFIX
        sub = Substrate()
        sub._quarantine_migrated = True

        # POSITIVE CONTROL: a real is_a chain that MUST cohere (non-vacuous).
        sub.add_edge('dog', 'mammal', 'is_a')
        sub.add_edge('mammal', 'animal', 'is_a')
        sub.add_edge('dog', 'animal', 'is_a')            # candidate: dog is_a animal

        # WORLD STRUCTURE with the WALL DOWN:
        #  (a) the world's own transitions_to chain
        sub.add_edge(f'{WP}0', f'{WP}1', 'transitions_to')
        sub.add_edge(f'{WP}1', f'{WP}2', 'transitions_to')
        sub.add_edge(f'{WP}0', f'{WP}2', 'transitions_to')   # candidate world edge
        #  (b) the wall-down abstraction: world states is_a one shared class
        for i in range(3):
            sub.add_edge(f'{WP}{i}', '_abstract_world', 'is_a')

        reinforced, newly = sub.reinforce_coherent_edges(cycle=1000)

        self.assertGreater(self._fcc(sub, 'dog', 'is_a', 'animal'), 0,
                           "positive control must cohere -- else test proves nothing")
        self.assertGreaterEqual(newly, 1)

        self.assertEqual(
            self._world_cohered(sub), [],
            "FARM OPEN: a world edge reached newly_coherent -> lifeforce. A "
            "second-hop/composition wiring has opened the credit path.")
        self.assertEqual(
            self._fcc(sub, f'{WP}0', 'transitions_to', f'{WP}2'), 0,
            "transitions_to must never cohere (excluded from RELATION_COMPOSITION)")
        self.assertEqual(
            self._fcc(sub, f'{WP}0', 'is_a', '_abstract_world'), 0,
            "world->abstraction is_a must never cohere (abstractions stay is_a "
            "TARGETS with no outgoing composing hop)")

    def test_real_abstraction_organ_output_never_coheres(self):
        """Drive the REAL form_abstractions organ (wall down) on a realistic
        world and assert whatever it builds still never coheres -- catches
        shapes a hand-built test would miss (e.g. meta-abstraction)."""
        WP = WORLD_TOKEN_PREFIX
        sub = Substrate()
        sub._quarantine_migrated = True

        # realistic world: >=3 states sharing (transitions_to, same target) is
        # exactly the (relation, target) group form_abstractions abstracts over.
        for i in range(6):
            sub.add_edge(f'{WP}{i}', f'{WP}99', 'transitions_to')
        sub.add_edge(f'{WP}99', f'{WP}100', 'transitions_to')

        # let the real organ build world abstractions (wall is down locally)
        sub.form_abstractions(cycle=500)

        # positive control proves the coherence pass is live in this test too
        sub.add_edge('dog', 'mammal', 'is_a')
        sub.add_edge('mammal', 'animal', 'is_a')
        sub.add_edge('dog', 'animal', 'is_a')

        reinforced, newly = sub.reinforce_coherent_edges(cycle=1000)

        self.assertGreater(self._fcc(sub, 'dog', 'is_a', 'animal'), 0,
                           "positive control must cohere -- else test proves nothing")
        self.assertEqual(
            self._world_cohered(sub), [],
            "FARM OPEN: an edge built by form_abstractions over world tokens "
            "reached newly_coherent -> lifeforce. The wall-down abstraction "
            "path now credits survival.")


class _NullBus:
    def publish(self, ev):
        pass


class _NullGrounding:
    def observe(self, *a, **k):
        pass

    def _predict(self, sv, cyc):
        return (None, None)


class _NullTransducer:
    def encode(self, vec):
        return f"{WORLD_TOKEN_PREFIX}{int(vec[0])}" if vec else f"{WORLD_TOKEN_PREFIX}0"


class _ScriptWorld:
    """Returns scripted step results so the credit gate can be tested exactly."""
    n_actions = 4

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self._state = 0

    def percept(self):
        return {'world_vector': [float(self._state)]}

    def step(self, a):
        r = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        self._state = int(r.get('world_vector', [self._state + 1])[0])
        return r


class TestMasteryCreditFarm(unittest.TestCase):
    """FARM GUARD #2 (auditor must-fix): the SECOND lifeforce-credit door.
    `_on_mastery -> credit_learning(1.0) -> record_learning` is NOT covered by
    the newly_coherent guard above.  Pin the invariant the safety rests on:
    credit fires ONLY on goal-permutation mastery (goal_changed), NEVER on
    re-solving a stationary goal — so the agent cannot bank lifeforce-credit by
    looping the same solve.  Makes un-farmability a TESTED fact, not a tanh
    accident."""

    def test_credit_only_on_goal_change_never_on_resolve(self):
        from seagi.brain.capabilities.world_actor import WorldActor
        credits = []
        # 8 successes; only ONE permutes the goal (mastery).  The other 7 are
        # re-solves of a stationary goal and MUST credit nothing.
        script = (
            [{'success': True, 'goal_changed': False,
              'world_vector': [1.0], 'steps': 3}] * 4
            + [{'success': True, 'goal_changed': True,
                'world_vector': [2.0], 'steps': 3}]      # the mastery
            + [{'success': True, 'goal_changed': False,
                'world_vector': [3.0], 'steps': 3}] * 3)
        wa = WorldActor(
            bus=_NullBus(), world=_ScriptWorld(script),
            transducer=_NullTransducer(), grounding=_NullGrounding(),
            value_provider=None,
            credit_learning=lambda g: credits.append(float(g)),
            cycle_provider=lambda: 0, seed=1)
        for t in range(len(script)):
            a = wa.propose(t)
            wa.execute(a if a is not None else 0, t)

        self.assertEqual(wa.successes, 8)
        self.assertEqual(wa.challenges_mastered, 1)
        # THE INVARIANT: one credit, on the goal-change only — not 8.
        self.assertEqual(
            credits, [1.0],
            "lifeforce-credit must fire ONLY on goal-permutation mastery, "
            "never on re-solving a stationary goal (the farm door)")

    def test_pure_resolve_loop_credits_zero(self):
        """A goal that NEVER permutes (always goal_changed=False) — i.e. the
        pure farm attempt — credits exactly zero no matter how many solves."""
        from seagi.brain.capabilities.world_actor import WorldActor
        credits = []
        script = [{'success': True, 'goal_changed': False,
                   'world_vector': [1.0], 'steps': 2}] * 25
        wa = WorldActor(
            bus=_NullBus(), world=_ScriptWorld(script),
            transducer=_NullTransducer(), grounding=_NullGrounding(),
            value_provider=None,
            credit_learning=lambda g: credits.append(float(g)),
            cycle_provider=lambda: 0, seed=2)
        for t in range(len(script)):
            a = wa.propose(t)
            wa.execute(a if a is not None else 0, t)
        self.assertEqual(wa.successes, 25)
        self.assertEqual(credits, [], "re-solving a stationary goal must NEVER credit")


if __name__ == '__main__':
    unittest.main()
