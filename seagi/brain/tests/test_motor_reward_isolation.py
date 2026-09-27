"""The mortal-game test must not become reinforcement learning ON SURVIVAL.

REVISED 2026-08-21 after the user's doctrinal correction: *"no-RL-on-score
doctrine is a misinterpretation of M/I only.  NT tags can work like RL on a
meta level and as proxies to M/I."*

The standing rule is that game events drive the PROXY layer, never lifeforce.
Chemistry/NT credit IS the proxy layer, so a game action earning it is what
the premise wants.  The original tests pinned a WIDER prohibition than the
doctrine required -- they blocked the permitted proxy half along with the
forbidden lifeforce half, which cost the skill system any contact with ARC.

What is pinned now:
  1. gate OFF  -> the original behaviour, byte for byte
  2. gate ON   -> a motor winner MAY enter the trace (proxy/NT credit)
  3. ALWAYS    -> lifeforce-delta credit NEVER reaches a motor action
  4. ALWAYS    -> a world-caused lifeforce delta is never action credit
"""

from __future__ import annotations

import unittest

import seagi.brain.capabilities.reward_ledger as RL
from seagi.brain.events import (
    EventKind, ArbitrationDecidedEvent, InteroceptionEvent)
from seagi.brain.capabilities.reward_ledger import RewardLedger


def _arb(loop, action='move:2'):
    return ArbitrationDecidedEvent(
        kind=EventKind.ARBITRATION_DECIDED, cycle=0,
        source_capability='basal_ganglia', origin='internal',
        loop=loop, winning_action=action,
        winning_capability='world_actor', winning_strength=1.0,
        n_competing=1)


def _body(delta, world_caused):
    return InteroceptionEvent(
        kind=EventKind.INTEROCEPTION, cycle=0,
        source_capability='insula', origin='internal',
        delta_lifeforce=delta, world_caused=world_caused)


class _Gate:
    """Force the motor gate for one test, whatever the box's /root holds."""

    def __init__(self, val):
        self.val = val

    def __enter__(self):
        self._orig = RL._MOTORSKILL_ON
        RL._MOTORSKILL_ON = lambda: self.val

    def __exit__(self, *a):
        RL._MOTORSKILL_ON = self._orig
        return False


class TestMotorRewardIsolation(unittest.TestCase):

    def test_gate_off_keeps_motor_out_entirely(self):
        """The original behaviour must survive untouched while gated off."""
        with _Gate(False):
            rl = RewardLedger(cycle_provider=lambda: 0)
            rl.handle(_arb('motor', 'move:1'), None)
            self.assertEqual(rl.actions_recorded, 0)
            self.assertEqual(len(rl._trace), 0)

    def test_gate_on_lets_a_game_action_earn_proxy_credit(self):
        """THE CORRECTION: NT credit is the proxy layer, and game events are
        supposed to drive it."""
        with _Gate(True):
            rl = RewardLedger(cycle_provider=lambda: 0)
            rl.handle(_arb('motor', 'move:1'), None)
            self.assertEqual(len(rl._trace), 1)
            self.assertEqual(rl._trace[0].loop, 'motor')

    def test_lifeforce_credit_NEVER_reaches_a_game_action(self):
        """THE REAL BOUNDARY, and it holds with the gate ON."""
        with _Gate(True):
            rl = RewardLedger(cycle_provider=lambda: 0)
            rl._chemistry_provider = None
            rl.handle(_arb('motor', 'move:1'), None)
            rl.handle(_arb('cognitive', 'pursue:x'), None)
            rl.handle(_body(-0.2, world_caused=False), None)
            motor = [k for k in rl._action_credit if k[0] == 'move']
            self.assertEqual(motor, [], 'lifeforce credit hit a game action')
            self.assertGreater(rl.motor_skipped_body_credit, 0)

    def test_nonmotor_winner_still_recorded(self):
        """Regression: cognitive/motivational/speech recording is unchanged."""
        rl = RewardLedger(cycle_provider=lambda: 0)
        rl.handle(_arb('cognitive', 'pursue:x'), None)
        self.assertEqual(rl.actions_recorded, 1)
        self.assertEqual(len(rl._trace), 1)

    def test_world_caused_body_delta_gives_no_credit(self):
        rl = RewardLedger(cycle_provider=lambda: 0)
        rl.handle(_arb('cognitive', 'pursue:x'), None)   # something to credit
        before = dict(rl._action_credit)
        rl.handle(_body(-0.2, world_caused=True), None)
        self.assertEqual(rl._action_credit, before)       # untouched
        self.assertEqual(rl.punishments_received, 0)

    def test_normal_body_delta_still_gives_credit(self):
        """Regression: an ordinary (non-world) body delta still propagates."""
        rl = RewardLedger(cycle_provider=lambda: 0)
        rl.handle(_arb('cognitive', 'pursue:x'), None)
        rl.handle(_body(-0.2, world_caused=False), None)
        self.assertGreater(rl.punishments_received, 0)


if __name__ == '__main__':
    unittest.main()
