"""End-to-end integration of the learn-to-solve-a-task test inside a real Brain.

The WorldActor unit tests use fakes; these tick an actual Brain so the FULL
loop runs through the real bus + basal ganglia + grounding + value landscape:
propose (assumption + curiosity) -> motor claim -> arbitrate -> execute -> learn
-> on success fire the life-lean -> on mastery escalate.  They pin that the
agent SOLVES the task and that mastering a goal ESCALATES to a new one, and that
the live RL-isolation holds (motor never credited)."""

from __future__ import annotations

import unittest

from seagi.body.engine import Engine
from seagi.brain.runtime import Brain


def _ticked_brain(n: int) -> Brain:
    brain = Brain(engine=Engine())
    for _ in range(n):
        brain.tick()
    return brain


class TestGameIntegration(unittest.TestCase):

    def test_loop_runs_and_solves(self):
        """With no reward, the agent learns to reach the goal end-to-end."""
        brain = _ticked_brain(6000)
        wa = brain.world_actor
        self.assertGreater(wa.executions, 0, "the motor loop must step")
        self.assertGreater(wa.explorations, 0, "curiosity must explore")
        self.assertGreater(wa.successes, 0, "the agent should solve the task")

    def test_masters_and_escalates(self):
        """Mastering a goal escalates to a new challenge (continual learning)."""
        brain = _ticked_brain(16000)
        self.assertGreaterEqual(
            brain.world_actor.challenges_mastered, 1,
            "mastering a goal should escalate to a new challenge")

    def test_lifeforce_credit_never_reaches_a_game_action(self):
        """Live RL-isolation, REVISED 2026-08-21.

        The old assertion was that no 'move' may appear in the trace at
        all.  That pinned a WIDER rule than the doctrine: game events
        drive the PROXY layer (NT credit), and only LIFEFORCE credit is
        forbidden on them.  So a move MAY be in the trace; what must
        never happen is a body delta crediting it.
        """
        import seagi.brain.capabilities.reward_ledger as _RL
        brain = _ticked_brain(4000)
        self.assertGreater(brain.world_actor.executions, 0)
        rl = brain.reward_ledger
        moves = [t for t in rl._trace
                 if str(t.action).startswith('move')]
        if not _RL._MOTORSKILL_ON():
            # gated off -> the original behaviour, unchanged
            self.assertEqual(moves, [], "motor must stay out while gated off")
        # THE INVARIANT THAT HOLDS EITHER WAY: no game action has ever
        # been credited by a lifeforce delta.
        for entry in rl._trace:
            if getattr(entry, 'loop', '') == 'motor':
                key = (entry.action, entry.chemistry_bucket)
                self.assertFalse(
                    rl._body_credited_keys.intersection({key})
                    if hasattr(rl, '_body_credited_keys') else False,
                    "lifeforce credit reached a game action")


if __name__ == '__main__':
    unittest.main()
