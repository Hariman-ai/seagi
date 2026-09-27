"""WorldActor — the goal-pursuer that learns to SOLVE a task (variation x
selection).  Pins: a motor claim is published; the policy leans on the success
assumption but stays stochastic (curiosity never off); the action-conditioned
transition is learned; reaching the goal fires the immortality NT-lean back
along the path; and MASTERING a challenge credits learning (lifeforce follows).
Grounding/value are faked so the actor's own logic is isolated."""

from __future__ import annotations

import unittest

from seagi.world.goal_world import GoalWorld
from seagi.brain.capabilities.world_transducer import WorldTransducer
from seagi.brain.capabilities.world_actor import WorldActor
from seagi.brain.events import (
    EventKind, CapabilityClaimEvent, ChemistryEvent, ArbitrationDecidedEvent)


class _Bus:
    def __init__(self):
        self.events = []

    def publish(self, ev):
        self.events.append(ev)


class _FakeGrounding:
    def __init__(self, mapping=None):
        self.map = dict(mapping or {})
        self.observed = []

    def _predict(self, subj, cyc):
        return (self.map.get(subj), 'own' if subj in self.map else '')

    def observe(self, s_prev, s_actual, cyc, bus, valence=None):
        self.observed.append((s_prev, s_actual))


class _FakeValue:
    def __init__(self, d):
        self.d = dict(d)

    def value_of(self, tok):
        return self.d.get(tok, 0.0)


def _encode_start(world, tx):
    return tx.encode(world.percept()['world_vector'])


class TestWorldActor(unittest.TestCase):

    def setUp(self):
        self.bus = _Bus()
        self.tx = WorldTransducer()

    def test_publishes_motor_claim(self):
        w = GoalWorld(grid=5, seed=1, stochastic=False)
        wa = WorldActor(self.bus, w, self.tx, _FakeGrounding(),
                        value_provider=lambda: _FakeValue({}), seed=0)
        wa.propose(cyc=1)
        claims = [e for e in self.bus.events
                  if isinstance(e, CapabilityClaimEvent)]
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].loop, 'motor')
        self.assertTrue(claims[0].proposed_action.startswith('move:'))

    def test_execute_learns_action_conditioned_transition(self):
        w = GoalWorld(grid=5, seed=1, stochastic=False)
        g = _FakeGrounding()
        wa = WorldActor(self.bus, w, self.tx, g,
                        value_provider=lambda: _FakeValue({}), seed=0)
        s = _encode_start(w, self.tx)
        a = wa.propose(cyc=1)
        wa.execute(a, cyc=1)
        self.assertEqual(len(g.observed), 1)
        subj, nxt = g.observed[0]
        self.assertEqual(subj, f'{s}|a{a}')      # composite (state, action)
        self.assertTrue(isinstance(nxt, str) and nxt)

    def test_success_fires_life_lean_and_mastery_credits(self):
        """Reaching the goal fires the confirmed_i (immortality) life-lean on
        the path; mastering the challenge (threshold 1) credits learning."""
        w = GoalWorld(grid=5, seed=1, stochastic=False, mastery_threshold=1)
        w.goal = (1, 0)            # odd-parity, deterministic; Down from start
        w._pos = (0, 0)
        credited = []
        wa = WorldActor(self.bus, w, self.tx, _FakeGrounding(),
                        value_provider=lambda: _FakeValue({}),
                        credit_learning=lambda gain: credited.append(gain),
                        seed=0)
        wa._pending = {'s': _encode_start(w, self.tx), 'a': 1}
        wa.execute(1, cyc=1)       # action 1 = Down -> (1,0) = goal
        tags = [e for e in self.bus.events
                if isinstance(e, ChemistryEvent)
                and e.chemistry_kind == 'confirmed_i']
        self.assertGreaterEqual(len(tags), 1)
        self.assertEqual(tags[0].source_capability, 'world_actor')
        self.assertEqual(wa.successes, 1)
        self.assertEqual(wa.challenges_mastered, 1)    # goal escalated
        self.assertEqual(len(credited), 1)             # lifeforce followed

    def test_timeout_counts_as_failure_no_mortality_tag(self):
        """RE-EXPRESSED 2026-08-10 on the user's ruling: *"loss makes him
        curious"*, *"losing is only annoying"*.  This asserted `tags == []`
        -- NO chemistry at all on a loss -- which made the single most common
        event in his world felt as nothing, and also denied it the companion
        curiosity `_feel` gives every other feeling.

        THE GUARD UNDERNEATH IS UNCHANGED AND NOW EXPLICIT: a game loss must
        never fire mortality/threat chemistry.  The blanket `== []` form hid
        that intent behind a stronger claim than the doctrine ever made
        ([[given-baseline-energy]]: failing a puzzle is not mortal danger).
        So: no threat, no death, no chronic stress -- and the annoyance +
        curiosity that the ruling requires."""
        w = GoalWorld(grid=5, seed=1, stochastic=False, mastery_threshold=1)
        w.goal = (2, 1)
        w._pos = (0, 0)
        wa = WorldActor(self.bus, w, self.tx, _FakeGrounding(),
                        value_provider=lambda: _FakeValue({}), seed=0)
        w.steps_this_episode = w.step_budget - 1
        wa._pending = {'s': _encode_start(w, self.tx), 'a': 0}
        wa.execute(0, cyc=1)
        self.assertEqual(wa.failures, 1)
        self.assertEqual(wa.successes, 0)
        kinds = [e.chemistry_kind for e in self.bus.events
                 if isinstance(e, ChemistryEvent)]
        # THE FIREWALL: a lost game is not a threat to his life.
        for lethal in ('threat', 'death', 'revival', 'chronic_stress'):
            self.assertNotIn(
                lethal, kinds,
                'a game loss fired %r -- the game/life firewall is breached '
                'and losing has become mortal danger.' % lethal)
        # THE RULING: it is felt, and being felt makes him curious.
        self.assertIn('puzzle_stress', kinds,
                      'a loss fired no annoyance -- defeat is unfelt again.')
        self.assertIn('curiosity', kinds,
                      'a loss fired no curiosity -- the companion every '
                      'other feeling gets is missing from the one event '
                      'that happens most.')

    def test_failure_never_credits_lifeforce(self):
        """The half of the old blanket assertion that must NOT relax: whatever
        a loss is allowed to FEEL, it can never pay the immortality pole.
        `confirmed_i` is the kind that reaches record_learning, so a loss
        firing it would make failing profitable."""
        w = GoalWorld(grid=5, seed=1, stochastic=False, mastery_threshold=1)
        w.goal = (2, 1)
        w._pos = (0, 0)
        wa = WorldActor(self.bus, w, self.tx, _FakeGrounding(),
                        value_provider=lambda: _FakeValue({}), seed=0)
        w.steps_this_episode = w.step_budget - 1
        wa._pending = {'s': _encode_start(w, self.tx), 'a': 0}
        wa.execute(0, cyc=1)
        self.assertEqual(wa.failures, 1)
        kinds = [e.chemistry_kind for e in self.bus.events
                 if isinstance(e, ChemistryEvent)]
        self.assertNotIn(
            'confirmed_i', kinds,
            'a LOSS fired the success kind -- failing now earns lifeforce.')
        self.assertEqual(wa.challenges_mastered, 0)

    def test_motor_decision_executes_the_world(self):
        w = GoalWorld(grid=5, seed=1, stochastic=False)
        wa = WorldActor(self.bus, w, self.tx, _FakeGrounding(),
                        value_provider=lambda: _FakeValue({}), seed=0)
        wa.propose(cyc=1)
        ev = ArbitrationDecidedEvent(
            kind=EventKind.ARBITRATION_DECIDED, cycle=1,
            source_capability='basal_ganglia', origin='internal',
            loop='motor', winning_action=wa._pending and 'move:1' or 'move:0',
            winning_capability='world_actor', winning_strength=1.0,
            n_competing=1)
        before = wa.executions
        wa.handle(ev, self.bus)
        self.assertEqual(wa.executions, before + 1)


if __name__ == '__main__':
    unittest.main()
