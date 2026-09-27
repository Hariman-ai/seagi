"""THE FULL SWEEP (2026-09-25): under /root/SWEEP_ON every ARC game gets one life on the card
before any hold or draw; then `_allocate` runs as before.  Pinned here:
  * one move per terminal, to the first game not yet visited in this sweep, in ladder order
  * a hold (organ / record / commitment) does not keep him while games remain
  * after the last game the sweep is done: state says so, no sweep move, normal rules decide
  * a gate file newer than the state starts a new sweep; gate absent: no sweep at all
  * ladder worlds (no _won_ever) are never sweep targets
"""
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from seagi.world import curriculum_world as _cw_mod
from seagi.world.curriculum_world import CurriculumWorld
from seagi.brain.tests.test_alloc import _Fake, _patches, REC


class TestSweep(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gate = os.path.join(self.tmp, 'SWEEP_ON')
        self.state = os.path.join(self.tmp, 'sweep_state.json')
        open(self.gate, 'w').close()
        self._ps = [mock.patch.object(_cw_mod, '_SWEEP_GATE', self.gate),
                    mock.patch.object(_cw_mod, '_SWEEP_STATE', self.state)]
        for p in self._ps:
            p.start()

    def tearDown(self):
        for p in self._ps:
            p.stop()

    def _run(self, cw, n, **kw):
        ps = _patches(**kw)
        for p in ps:
            p.start()
        try:
            for _ in range(n):
                cw.step(0)
        finally:
            for p in ps:
                p.stop()

    def test_one_life_per_game_then_done_and_holds_do_not_keep_him(self):
        a = _Fake('aaaa-1', [True] * 3, organ=[True, True, True], record=True)   # would be held
        b = _Fake('bbbb-1', [True] * 3)
        c = _Fake('cccc-1', [True] * 3)
        cw = CurriculumWorld([a, b, c], mastery_threshold=5)
        self._run(cw, 1, rec=REC)                       # terminal on a -> sweep move to b
        self.assertEqual(cw._idx, 1)
        self.assertEqual(cw.sweep_moves, 1)
        self.assertEqual(cw.alloc_why, {'sweep': 1})
        st = json.load(open(self.state))
        self.assertEqual(st['visited'], {'aaaa': 1})
        self.assertIsNone(st['done'])
        self._run(cw, 1, rec=REC)                       # terminal on b -> c
        self.assertEqual(cw._idx, 2)
        self._run(cw, 1, rec=REC)                       # terminal on c -> sweep complete; dry -> normal move
        st = json.load(open(self.state))
        self.assertEqual(sorted(st['visited']), ['aaaa', 'bbbb', 'cccc'])
        self.assertTrue(st['done'])
        self.assertEqual(st['lives'], 3)
        self.assertEqual(cw.sweep_moves, 2)
        self.assertEqual(cw.alloc_why.get('sweep'), 2)
        # sweep done: the normal rules decided this terminal (a dry move, not a sweep move)
        self.assertEqual(cw.patch_moves, 3)
        self.assertIn('dry', cw.alloc_why)
        # and the next terminal is normal too: the organ hold on a keeps him now
        cw._idx = 0
        self._run(cw, 1, rec=REC)
        self.assertEqual(cw.alloc_stays, 1)
        self.assertEqual(cw.sweep_moves, 2)

    def test_gate_absent_is_byte_identical_and_a_newer_gate_restarts(self):
        a = _Fake('aaaa-1', [True] * 2, organ=[True, True])
        b = _Fake('bbbb-1', [True] * 2)
        cw = CurriculumWorld([a, b], mastery_threshold=5)
        os.remove(self.gate)
        self._run(cw, 1, rec=REC)
        self.assertEqual(cw.patch_moves, 0)
        self.assertEqual(getattr(cw, 'sweep_moves', 0), 0)
        self.assertFalse(os.path.exists(self.state))
        # a finished sweep, then the gate is touched anew: the state resets
        json.dump({'started': 1.0, 'visited': {'aaaa': 1, 'bbbb': 1}, 'done': 2.0, 'lives': 2},
                  open(self.state, 'w'))
        open(self.gate, 'w').close()
        os.utime(self.gate, (time.time() + 5, time.time() + 5))
        self._run(cw, 1, rec=REC)
        st = json.load(open(self.state))
        self.assertEqual(st['visited'], {'aaaa': 1})
        self.assertIsNone(st['done'])
        self.assertEqual(cw.sweep_moves, 1)

    def test_ladder_worlds_are_not_sweep_targets(self):
        class _Ladder(object):
            game_id = 'ladd-1'
            def step(self, a):
                return {'success': False, 'done': True, 'timed_out': False}
            def enter(self):
                pass
        a = _Fake('aaaa-1', [True] * 2)
        b = _Fake('bbbb-1', [True] * 2)
        cw = CurriculumWorld([a, _Ladder(), b], mastery_threshold=5)
        self._run(cw, 1, rec=REC)
        self.assertEqual(cw._idx, 2)     # skipped the ladder world


if __name__ == '__main__':
    unittest.main()
