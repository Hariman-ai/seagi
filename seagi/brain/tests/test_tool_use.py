"""Tests for the V1→V2 ToolUse port (calculator + algebra).

Covers:
- calculator: safe AST eval, word-operator normalization, rejects
  non-arithmetic
- algebra: solve / differentiate / simplify; security whitelist
  rejects dangerous tokens
- registry priority (algebra before calculator)
- dispatch returns evidence + voice; absorb re-enters via intake
  (origin='tool') — the V3-clean weighed path
- autonomous BG path: thin-substrate thought with a tool-shaped
  text emits a tool claim; arbitration win triggers execution
- security: no eval of names/calls; banned tokens rejected
- persistence round-trip
- chat dispatches a tool on explicit request
"""

from __future__ import annotations

import time
import unittest

from seagi.brain import EventKind, EventBus
from seagi.brain.events import (
    ThoughtProducedEvent, ArbitrationDecidedEvent)
from seagi.brain.capabilities.tool_use import (
    ToolUse, _safe_arith_eval, _algebra_is_safe, _calc_trigger,
    _algebra_trigger, _load_sympy)

# SymPy is present on alpha (pulled in by torch) but not in every
# local env.  Algebra EXECUTION assertions skip when absent; the
# trigger/priority/security checks run regardless.
_HAS_SYMPY = _load_sympy()


class _ToolHarness:
    def __init__(self):
        self.bus = EventBus()
        self.cycle = {'c': 0}
        self.intaken = []
        self.tu = ToolUse(
            bus=self.bus,
            intake_fn=self._intake,
            cycle_provider=lambda: self.cycle['c'])
        self.bus.subscribe(self.tu.SUBSCRIPTIONS, self.tu)

    def _intake(self, payload, modality='text', origin='internal',
                origin_detail=''):
        self.intaken.append((payload, origin, origin_detail))


def _thin_thought(text, focal='x', cycle=0):
    return ThoughtProducedEvent(
        kind=EventKind.THOUGHT_PRODUCED, cycle=cycle,
        timestamp=time.time(), source_capability='cortical',
        origin='internal', origin_detail=focal,
        focal=focal, relation='', target='', confidence=0.2,
        method='inference', text=text, thin_substrate=True)


class TestCalculatorSafety(unittest.TestCase):

    def test_basic_arithmetic(self):
        self.assertEqual(_safe_arith_eval('7 * 342'), 2394)
        self.assertEqual(_safe_arith_eval('12 + 19'), 31)
        self.assertAlmostEqual(_safe_arith_eval('10 / 4'), 2.5)

    def test_rejects_names_and_calls(self):
        self.assertIsNone(_safe_arith_eval('__import__("os")'))
        self.assertIsNone(_safe_arith_eval('open("x")'))
        self.assertIsNone(_safe_arith_eval('x + 1'))
        self.assertIsNone(_safe_arith_eval('[].__class__'))

    def test_rejects_div_by_zero(self):
        self.assertIsNone(_safe_arith_eval('1/0'))

    def test_trigger_needs_operator(self):
        self.assertIsNone(_calc_trigger('the number 42'))
        self.assertIsNotNone(_calc_trigger('what is 6 * 7'))


class TestAlgebraSafety(unittest.TestCase):

    def test_safe_chars_pass(self):
        self.assertTrue(_algebra_is_safe('x**2 + 3*x - 4'))
        self.assertTrue(_algebra_is_safe('2*x = 10'))

    def test_banned_tokens_rejected(self):
        self.assertFalse(_algebra_is_safe('__import__'))
        self.assertFalse(_algebra_is_safe('x; import os'))
        self.assertFalse(_algebra_is_safe('eval(x)'))
        self.assertFalse(_algebra_is_safe("x.__class__"))

    def test_algebra_trigger_matches_ops(self):
        self.assertEqual(
            _algebra_trigger('differentiate x^2')['op'],
            'differentiate')
        self.assertEqual(
            _algebra_trigger('solve x**2 = 9')['op'], 'solve')


class TestDispatch(unittest.TestCase):

    def test_calculator_dispatch(self):
        h = _ToolHarness()
        ev = h.tu.dispatch('what is 7 * 342')
        self.assertIsNotNone(ev)
        self.assertEqual(ev['tool'], 'calculator')
        self.assertTrue(ev['success'])
        self.assertEqual(ev['output'], 2394)
        self.assertIn('2394', ev['voice'])

    def test_algebra_priority_over_calculator(self):
        # "solve x**2 = 9" has arithmetic-shaped substrings but
        # algebra (explicit 'solve') must win at the TRIGGER level
        # regardless of whether sympy can execute it.
        h = _ToolHarness()
        ev = h.tu.dispatch('solve x**2 = 9')
        self.assertIsNotNone(ev)
        self.assertEqual(ev['tool'], 'algebra')
        if _HAS_SYMPY:
            self.assertTrue(ev['success'])
        else:
            # Graceful degradation: reports sympy-unavailable.
            self.assertFalse(ev['success'])
            self.assertIn('sympy', (ev.get('error') or '').lower())

    @unittest.skipUnless(_HAS_SYMPY, 'sympy not installed')
    def test_differentiate(self):
        h = _ToolHarness()
        ev = h.tu.dispatch('differentiate x**2')
        self.assertIsNotNone(ev)
        self.assertEqual(ev['tool'], 'algebra')
        self.assertTrue(ev['success'])
        # d/dx x^2 = 2*x
        self.assertIn('2*x', ev['voice'].replace(' ', '')
                      .replace('2x', '2*x'))

    def test_no_tool_returns_none(self):
        h = _ToolHarness()
        self.assertIsNone(h.tu.dispatch('tell me about your day'))

    def test_absorb_reenters_via_intake_origin_tool(self):
        h = _ToolHarness()
        ev = h.tu.dispatch('what is 2 + 2')
        h.tu.absorb(ev)
        self.assertEqual(len(h.intaken), 1)
        payload, origin, detail = h.intaken[0]
        self.assertEqual(origin, 'tool')
        self.assertEqual(detail, 'calculator')

    def test_absorb_skips_failure(self):
        h = _ToolHarness()
        ev = {'tool': 'calculator', 'success': False, 'voice': 'x'}
        h.tu.absorb(ev)
        self.assertEqual(len(h.intaken), 0)


class TestAutonomousArbitration(unittest.TestCase):

    def test_thin_thought_with_math_emits_claim(self):
        h = _ToolHarness()
        claims = []

        class _Sink:
            SUBSCRIPTIONS = (EventKind.CAPABILITY_CLAIM,)

            def handle(self, ev, bus):
                claims.append(ev)
        h.bus.subscribe(_Sink.SUBSCRIPTIONS, _Sink())
        h.bus.publish(_thin_thought('what is 8 * 9', focal='math'))
        self.assertEqual(len(claims), 1)
        self.assertTrue(
            claims[0].proposed_action.startswith('tool:calculator:'))

    def test_thin_thought_without_math_no_claim(self):
        h = _ToolHarness()
        claims = []

        class _Sink:
            SUBSCRIPTIONS = (EventKind.CAPABILITY_CLAIM,)

            def handle(self, ev, bus):
                claims.append(ev)
        h.bus.subscribe(_Sink.SUBSCRIPTIONS, _Sink())
        h.bus.publish(_thin_thought('justice relates to fairness',
                                     focal='justice'))
        self.assertEqual(len(claims), 0)

    def test_arbitration_win_executes_and_absorbs(self):
        h = _ToolHarness()
        # First raise the claim (stashes the pending text).
        h.bus.publish(_thin_thought('what is 5 * 5', focal='math'))
        # Then simulate BG selecting it.
        h.bus.publish(ArbitrationDecidedEvent(
            kind=EventKind.ARBITRATION_DECIDED, cycle=1,
            timestamp=time.time(), source_capability='basal_ganglia',
            origin='internal', origin_detail='',
            loop='cognitive',
            winning_action='tool:calculator:math',
            winning_capability='tool_use',
            winning_strength=0.5, n_competing=1))
        # Executed → result absorbed via intake(origin='tool').
        self.assertEqual(len(h.intaken), 1)
        self.assertEqual(h.intaken[0][1], 'tool')
        self.assertEqual(h.tu.successes, 1)

    def test_non_tool_arbitration_ignored(self):
        h = _ToolHarness()
        h.bus.publish(ArbitrationDecidedEvent(
            kind=EventKind.ARBITRATION_DECIDED, cycle=1,
            timestamp=time.time(), source_capability='basal_ganglia',
            origin='internal', origin_detail='',
            loop='cognitive', winning_action='pursue:justice',
            winning_capability='goals', winning_strength=0.5,
            n_competing=1))
        self.assertEqual(len(h.intaken), 0)


class TestPersistence(unittest.TestCase):

    def test_round_trip(self):
        h = _ToolHarness()
        h.tu.dispatch('what is 3 * 3')
        snap = h.tu.to_dict()
        fresh = ToolUse(bus=EventBus(),
                        intake_fn=lambda *a, **k: None,
                        cycle_provider=lambda: 0)
        fresh.load_dict(snap)
        self.assertEqual(fresh.dispatches, h.tu.dispatches)
        self.assertEqual(fresh.successes, h.tu.successes)


class TestBrainIntegration(unittest.TestCase):

    def test_chat_dispatches_calculator(self):
        from seagi.body.engine import Engine
        from seagi.brain.runtime import Brain
        brain = Brain(engine=Engine())
        resp = brain.chat('what is 7 * 342', peer_id='test')
        # The tool voice leads the response.
        self.assertIn('2394', resp)
        self.assertGreaterEqual(brain.tool_use.successes, 1)

    def test_chat_normal_message_no_tool(self):
        from seagi.body.engine import Engine
        from seagi.brain.runtime import Brain
        brain = Brain(engine=Engine())
        before = brain.tool_use.dispatches
        brain.chat('hello there friend', peer_id='test')
        # matches_a_tool returns None → dispatch returns None, but
        # dispatch() is still called once (returns None, no success).
        self.assertEqual(brain.tool_use.successes, 0)


if __name__ == '__main__':
    unittest.main()
