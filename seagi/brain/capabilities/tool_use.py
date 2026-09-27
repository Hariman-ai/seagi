"""Tool use — external grounding as EVIDENCE channels (V1→V2 port,
Tier 1).

V2 reasons from substrate alone — it cannot verify arithmetic,
manipulate symbols, or check a fact it never ingested.  R.1
composition happily produces garbage when the substrate is wrong
("fear enables child").  Tools are the grounding primitive that
lets SEAGI check itself against something outside its own graph.

V1's framing, preserved exactly: **a tool result is a weighed
claim, not an oracle.**  The result re-enters through the normal
percept path (`intake(origin='tool')`) and writes provisional SVO
edges that face Phase S earn-or-dissolve like any other percept —
so a correct calculator result still has to EARN longevity by being
re-walked.  Tool `confidence` shapes the spoken voice; it never
sets absolute edge strength (this is the V3 fix — a near-ceiling
tool edge is structurally impossible).

Divergences from V1 (the doctrine improvements the auditor required):
- **Routed through BG arbitration, not bypassed.**  V1 fired tools
  inline in the cognitive cycle.  Here a tool-need raises a
  `CapabilityClaimEvent('tool:<name>:<focal>')` that competes in the
  cognitive loop against thinking-it-out and goals; the tool fires
  only when arbitration picks it.  (Explicit peer requests in chat
  dispatch directly — answering a direct "what is 7*342" is
  responding to a request, not autonomous action.)
- **Calculator + algebra only.**  Wikipedia is deferred (different
  permission boundary; overlaps the forager's external-input role;
  and a 5s network fetch under the daemon's brain_lock would stall
  ~25 ticks — when wiki lands it MUST be off-thread).

Security is the load-bearing part of the port and is copied
verbatim from V1: calculator uses a whitelisted-AST evaluator
(never eval of names/calls/attributes); algebra pre-validates with
a char-whitelist + banned-token list BEFORE SymPy's parse_expr
(parse_expr's own transformations are NOT a security boundary, per
SymPy docs) and runs with an empty global dict.

Cross-refs: [[project_seagi_v1_port_tier1_wiring]] (V3 resolution),
[[feedback_doctrine_vision_guardian]], [[feedback_homeostatic_cost_doctrine]].
"""

from __future__ import annotations

import ast
import re
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from ..events import (
    EventKind, BrainEvent, ThoughtProducedEvent,
    ArbitrationDecidedEvent, CapabilityClaimEvent,
)
from ..bus import EventBus
# claim_strength for a tool-need is an uncertainty-resolution claim;
# reuse the UncertaintyMonitor floor rather than inventing a new
# magnitude (auditor: derive, don't hand-tune).
from .uncertainty_monitor import UNCERTAINTY_CLAIM_STRENGTH_FLOOR


# ------------------------------------------------------------------
# CALCULATOR — safe AST-based evaluator (ported verbatim from V1)
# ------------------------------------------------------------------

_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.USub, ast.UAdd, ast.Load,
)


def _safe_arith_eval(expr: str) -> Optional[float]:
    expr = (expr or '').strip()
    if not expr or len(expr) > 200:
        return None
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            return None
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                return None
    try:
        return eval(compile(tree, '<calc>', 'eval'),  # noqa: S307
                    {'__builtins__': {}}, {})
    except (ZeroDivisionError, OverflowError, ValueError):
        return None


_ARITH_WORD_OPS = {
    'plus': '+', 'minus': '-', 'times': '*',
    'divided by': '/', 'divided': '/', 'over': '/',
    'to the power of': '**', 'squared': '**2', 'cubed': '**3',
}
_ARITH_EXPR_RE = re.compile(
    r'([-+]?\d+(?:\.\d+)?'
    r'(?:\s*(?:\*\*|//|[\+\-\*\/\%])\s*[-+]?\d+(?:\.\d+)?)+)')


def _arith_normalize(text: str) -> str:
    s = (text or '').lower()
    for w, op in sorted(_ARITH_WORD_OPS.items(),
                        key=lambda kv: -len(kv[0])):
        s = s.replace(w, f' {op} ')
    return re.sub(r'\s+', ' ', s).strip()


def _calc_trigger(text: str) -> Optional[Dict[str, Any]]:
    norm = _arith_normalize(text)
    m = _ARITH_EXPR_RE.search(norm)
    if not m:
        return None
    expr = m.group(1)
    if not re.search(r'[\+\-\*\/\%]', expr):
        return None
    return {'expr': expr}


def _calc_execute(args: Dict[str, Any]) -> Dict[str, Any]:
    expr = args.get('expr', '')
    result = _safe_arith_eval(expr)
    if result is None:
        return {'tool': 'calculator', 'args': args, 'output': None,
                'success': False, 'confidence': 0.0,
                'concepts': ['number', 'calculation'],
                'error': f'could not evaluate {expr!r}'}
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return {'tool': 'calculator', 'args': args, 'output': result,
            'success': True, 'confidence': 1.0,
            'concepts': ['number', 'calculation'], 'error': None}


def _calc_voice(ev: Dict[str, Any]) -> str:
    if not ev.get('success'):
        return ("I tried to work that out but could not — the "
                "expression did not resolve.")
    return (f"I worked it out: {ev.get('args', {}).get('expr', '')} "
            f"= {ev.get('output')}.")


# ------------------------------------------------------------------
# ALGEBRA — symbolic reasoning via SymPy (ported verbatim from V1)
# ------------------------------------------------------------------

_SYMPY_LOADED = False
_SYMPY_LOAD_ERROR: Optional[str] = None
_SYMPY_MODULES: Dict[str, Any] = {}


def _load_sympy() -> bool:
    global _SYMPY_LOADED, _SYMPY_LOAD_ERROR, _SYMPY_MODULES
    if _SYMPY_LOADED:
        return True
    if _SYMPY_LOAD_ERROR is not None:
        return False
    try:
        import sympy  # noqa: F401
        from sympy import (symbols, solve, simplify, diff, integrate,
                           factor)
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application, convert_xor)
        _SYMPY_MODULES = {
            'symbols': symbols, 'solve': solve, 'simplify': simplify,
            'diff': diff, 'integrate': integrate, 'factor': factor,
            'parse_expr': parse_expr,
            'transformations': (standard_transformations
                                + (implicit_multiplication_application,
                                   convert_xor)),
        }
        _SYMPY_LOADED = True
        return True
    except Exception as ex:
        _SYMPY_LOAD_ERROR = f'{type(ex).__name__}: {ex}'
        return False


_ALGEBRA_OP_PATTERNS = [
    ('differentiate',
     re.compile(r'^(?:differentiate|what\s+is\s+the\s+derivative\s+of|'
                r'find\s+the\s+derivative\s+of|d/dx\s+(?:of\s+)?)'
                r'\s*(.+?)'
                r'(?:\s+with\s+respect\s+to\s+(\w+))?'
                r'[\?\.\!]*$', re.IGNORECASE)),
    ('integrate',
     re.compile(r'^(?:integrate|find\s+the\s+integral\s+of|'
                r'what\s+is\s+the\s+integral\s+of)'
                r'\s+(.+?)'
                r'(?:\s+(?:with\s+respect\s+to|d)\s*(\w+))?'
                r'[\?\.\!]*$', re.IGNORECASE)),
    ('solve',
     re.compile(r'^(?:solve|find)\s+(?:the\s+equation\s+)?(?:for\s+'
                r'\w+\s+in\s+)?(.+?)[\?\.\!]*$', re.IGNORECASE)),
    ('simplify',
     re.compile(r'^(?:simplify|what\s+is\s+the\s+simplified\s+form\s+of)'
                r'\s+(.+?)[\?\.\!]*$', re.IGNORECASE)),
    ('factor',
     re.compile(r'^(?:factor|factorize|factorise)\s+(.+?)'
                r'[\?\.\!]*$', re.IGNORECASE)),
]


def _normalize_math_text(text: str) -> str:
    s = (text or '').strip()
    s = re.sub(r'\bsquared\b', '**2', s, flags=re.IGNORECASE)
    s = re.sub(r'\bcubed\b', '**3', s, flags=re.IGNORECASE)
    s = re.sub(r'\bto\s+the\s+power\s+of\b', '**', s, flags=re.IGNORECASE)
    s = re.sub(r'\btimes\b', '*', s, flags=re.IGNORECASE)
    s = re.sub(r'\bplus\b', '+', s, flags=re.IGNORECASE)
    s = re.sub(r'\bminus\b', '-', s, flags=re.IGNORECASE)
    s = re.sub(r'\bdivided\s+by\b', '/', s, flags=re.IGNORECASE)
    s = re.sub(r'\bover\b', '/', s, flags=re.IGNORECASE)
    s = re.sub(r'\bequals\b', '=', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+d[a-z]\b\s*$', '', s)
    return s.strip()


def _algebra_trigger(text: str) -> Optional[Dict[str, Any]]:
    s = (text or '').lower().strip()
    if not s:
        return None
    for prefix in ('please ', 'can you ', 'could you ', 'would you '):
        if s.startswith(prefix):
            s = s[len(prefix):]
    for op, pat in _ALGEBRA_OP_PATTERNS:
        m = pat.match(s)
        if m:
            expr_raw = m.group(1).strip()
            expr = _normalize_math_text(expr_raw)
            if len(expr) < 1 or len(expr) > 400:
                return None
            var = None
            if m.lastindex and m.lastindex >= 2 and m.group(2):
                var = m.group(2).strip()
            args = {'op': op, 'expr': expr, 'raw': expr_raw}
            if var:
                args['var'] = var
            return args
    return None


# Security boundary (verbatim V1): char-whitelist + banned tokens.
_ALGEBRA_SAFE_CHARS = re.compile(r'^[A-Za-z0-9_ +\-*/%^()=,.\t]+$')
_ALGEBRA_BANNED_TOKENS = (
    '__', 'import', 'exec', 'eval', 'open', 'os.', 'sys.',
    'globals', 'locals', 'compile', 'getattr', 'setattr', 'lambda',
)


def _algebra_is_safe(expr: str) -> bool:
    if not expr or len(expr) > 400:
        return False
    if not _ALGEBRA_SAFE_CHARS.match(expr):
        return False
    low = expr.lower()
    return not any(b in low for b in _ALGEBRA_BANNED_TOKENS)


def _alg_ok(args, output, concepts):
    return {'tool': 'algebra', 'args': args, 'output': output,
            'success': True, 'confidence': 0.95,
            'concepts': concepts, 'error': None}


def _alg_fail(args, err):
    return {'tool': 'algebra', 'args': args, 'output': None,
            'success': False, 'confidence': 0.0,
            'concepts': [], 'error': err}


def _algebra_execute(args: Dict[str, Any]) -> Dict[str, Any]:
    if not _load_sympy():
        return _alg_fail(args, f'sympy not available: {_SYMPY_LOAD_ERROR}')
    op = args.get('op', '')
    expr_text = args.get('expr', '')
    var_text = args.get('var')
    if not _algebra_is_safe(expr_text):
        return _alg_fail(args, 'expression contains disallowed tokens')
    if var_text and not re.match(r'^[A-Za-z_]\w{0,20}$', var_text):
        return _alg_fail(args, 'invalid variable name')
    parse_expr = _SYMPY_MODULES['parse_expr']
    transforms = _SYMPY_MODULES['transformations']
    symbols = _SYMPY_MODULES['symbols']
    try:
        if op == 'solve':
            if '=' in expr_text:
                left, right = expr_text.split('=', 1)
                expr = parse_expr(f'({left.strip()}) - ({right.strip()})',
                                  transformations=transforms)
            else:
                expr = parse_expr(expr_text, transformations=transforms)
            var_sym = symbols(var_text) if var_text else None
            if var_sym is None:
                free = list(expr.free_symbols)
                if len(free) == 1:
                    var_sym = free[0]
                elif len(free) == 0:
                    return _alg_fail(args, 'no free symbol to solve for')
                else:
                    names = {str(s): s for s in free}
                    var_sym = names.get('x') or sorted(free, key=str)[0]
            solutions = _SYMPY_MODULES['solve'](expr, var_sym)
            return _alg_ok(args, {'variable': str(var_sym),
                                  'solutions': [str(s) for s in solutions]},
                           ['algebra', 'equation', str(var_sym)])
        elif op == 'simplify':
            expr = parse_expr(expr_text, transformations=transforms)
            return _alg_ok(args,
                           {'result': str(_SYMPY_MODULES['simplify'](expr))},
                           ['algebra', 'simplification'])
        elif op == 'factor':
            expr = parse_expr(expr_text, transformations=transforms)
            return _alg_ok(args,
                           {'result': str(_SYMPY_MODULES['factor'](expr))},
                           ['algebra', 'factorization'])
        elif op == 'differentiate':
            expr = parse_expr(expr_text, transformations=transforms)
            if var_text:
                var_sym = symbols(var_text)
            else:
                free = list(expr.free_symbols)
                if not free:
                    return _alg_fail(args, 'nothing to differentiate')
                names = {str(s): s for s in free}
                var_sym = names.get('x') or sorted(free, key=str)[0]
            result = _SYMPY_MODULES['diff'](expr, var_sym)
            return _alg_ok(args, {'variable': str(var_sym),
                                  'result': str(result)},
                           ['calculus', 'derivative', str(var_sym)])
        elif op == 'integrate':
            expr = parse_expr(expr_text, transformations=transforms)
            if var_text:
                var_sym = symbols(var_text)
            else:
                free = list(expr.free_symbols)
                if not free:
                    return _alg_fail(args, 'nothing to integrate')
                names = {str(s): s for s in free}
                var_sym = names.get('x') or sorted(free, key=str)[0]
            result = _SYMPY_MODULES['integrate'](expr, var_sym)
            return _alg_ok(args, {'variable': str(var_sym),
                                  'result': str(result)},
                           ['calculus', 'integral', str(var_sym)])
        else:
            return _alg_fail(args, f'unknown op {op!r}')
    except (SyntaxError, TypeError, ValueError) as ex:
        return _alg_fail(args, f'parse: {ex}')
    except Exception as ex:
        return _alg_fail(args, f'{type(ex).__name__}: {ex}')


def _algebra_voice(ev: Dict[str, Any]) -> str:
    if not ev.get('success'):
        op = ev.get('args', {}).get('op', 'that')
        return (f"I tried to {op} that but it did not resolve — "
                f"{ev.get('error', '')}.")
    op = ev.get('args', {}).get('op')
    raw = ev.get('args', {}).get('raw', '')
    out = ev.get('output') or {}
    if op == 'solve':
        var = out.get('variable', 'x')
        sols = out.get('solutions') or []
        if not sols:
            return f"I worked through {raw} and found no solutions."
        return (f"I worked it out symbolically: {var} = "
                f"{', '.join(sols)}.")
    if op == 'simplify':
        return f"Simplifying {raw}, I get {out.get('result')}."
    if op == 'factor':
        return f"Factoring {raw}, I get {out.get('result')}."
    if op == 'differentiate':
        return (f"The derivative of {raw} with respect to "
                f"{out.get('variable', 'x')} is {out.get('result')}.")
    if op == 'integrate':
        return (f"The integral of {raw} with respect to "
                f"{out.get('variable', 'x')} is {out.get('result')} + C.")
    return str(out)


# Registry — algebra first (checks explicit op keywords so "solve
# x^2=9" isn't stolen by the calculator's plain-arithmetic trigger).
_REGISTRY = [
    {'name': 'algebra', 'trigger': _algebra_trigger,
     'execute': _algebra_execute, 'voice': _algebra_voice},
    {'name': 'calculator', 'trigger': _calc_trigger,
     'execute': _calc_execute, 'voice': _calc_voice},
]


class ToolUse:
    """Tool registry + arbitrated dispatch.  Tools are evidence
    channels: results re-enter via intake(origin='tool') and earn
    via Phase S like any percept."""

    SUBSCRIPTIONS = (
        EventKind.THOUGHT_PRODUCED,
        EventKind.ARBITRATION_DECIDED,
    )

    def __init__(self,
                 bus: EventBus,
                 intake_fn: Callable,
                 cycle_provider: Optional[Callable] = None):
        self.bus = bus
        self._intake_fn = intake_fn
        self._cycle_provider = cycle_provider or (lambda: 0)
        # Pending tool-need text keyed by the claim's focal, so an
        # arbitration win can recover what to run.
        self._pending: Dict[str, str] = {}
        # Diagnostics + persisted counts.
        self.tool_call_log: Deque[Dict[str, Any]] = deque(maxlen=32)
        self.dispatches: int = 0
        self.successes: int = 0
        self.claims_emitted: int = 0

    # ---- core dispatch (synchronous, no side effects) ----

    def dispatch(self, text: str) -> Optional[Dict[str, Any]]:
        """Run the first matching tool's trigger→execute→voice.
        Returns the evidence dict, or None when no tool applies.
        Does NOT intake — caller decides (chat uses the voice;
        autonomous path calls absorb())."""
        for tool in _REGISTRY:
            try:
                args = tool['trigger'](text)
            except Exception:
                continue
            if args is None:
                continue
            try:
                evidence = tool['execute'](args)
            except Exception as ex:
                evidence = {'tool': tool['name'], 'args': args,
                            'output': None, 'success': False,
                            'confidence': 0.0, 'concepts': [],
                            'error': f'{type(ex).__name__}: {ex}'}
            evidence['voice'] = tool['voice'](evidence)
            self.dispatches += 1
            if evidence.get('success'):
                self.successes += 1
            self.tool_call_log.append({
                'tool': evidence.get('tool'),
                'success': bool(evidence.get('success')),
                'cycle': int(self._cycle_provider()),
            })
            return evidence
        return None

    def absorb(self, evidence: Dict[str, Any]) -> None:
        """Re-enter a successful tool result as a weighed percept
        (origin='tool').  V3-clean: it flows through the normal
        intake path and writes provisional SVO edges that face
        Phase S — confidence shapes the voice, never absolute edge
        strength."""
        if not evidence or not evidence.get('success'):
            return
        voice = evidence.get('voice') or ''
        if not voice:
            return
        try:
            self._intake_fn(
                voice, modality='text', origin='tool',
                origin_detail=str(evidence.get('tool', 'tool')))
        except Exception:
            pass

    @staticmethod
    def matches_a_tool(text: str) -> Optional[str]:
        """Return the name of the tool that would fire for `text`,
        or None.  Cheap trigger-only check (no execution)."""
        for tool in _REGISTRY:
            try:
                if tool['trigger'](text) is not None:
                    return tool['name']
            except Exception:
                continue
        return None

    # ---- arbitrated (autonomous) path ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, ThoughtProducedEvent):
            # A thin-substrate thought whose rendered text contains a
            # tool-triggerable expression raises a tool claim that
            # competes in BG against thinking-it-out.  Rare in pure
            # reverie (substrate thoughts aren't arithmetic), but the
            # doctrine-clean autonomous channel.
            if not event.thin_substrate:
                return
            text = event.text or event.focal or ''
            tool_name = self.matches_a_tool(text)
            if tool_name is None:
                return
            focal = event.focal or tool_name
            self._pending[focal] = text
            bus.publish(CapabilityClaimEvent(
                kind=EventKind.CAPABILITY_CLAIM,
                cycle=int(event.cycle),
                timestamp=time.time(),
                source_capability='tool_use',
                origin='internal',
                origin_detail=tool_name,
                claim_strength=UNCERTAINTY_CLAIM_STRENGTH_FLOOR,
                proposed_action=f'tool:{tool_name}:{focal}',
                loop='cognitive',
                payload={'text': text}))
            self.claims_emitted += 1
        elif isinstance(event, ArbitrationDecidedEvent):
            action = event.winning_action or ''
            if not action.startswith('tool:'):
                return
            parts = action.split(':', 2)
            if len(parts) < 3:
                return
            _verb, _name, focal = parts
            text = self._pending.pop(focal, '')
            if not text:
                return
            evidence = self.dispatch(text)
            if evidence is not None:
                self.absorb(evidence)

    # ---- persistence ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dispatches': int(self.dispatches),
            'successes': int(self.successes),
            'claims_emitted': int(self.claims_emitted),
            'tool_call_log': list(self.tool_call_log),
        }

    def load_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        try:
            self.dispatches = int(state.get('dispatches', 0))
            self.successes = int(state.get('successes', 0))
            self.claims_emitted = int(state.get('claims_emitted', 0))
        except (TypeError, ValueError):
            pass
        log = state.get('tool_call_log')
        if isinstance(log, (list, tuple)):
            self.tool_call_log = deque(
                (x for x in log if isinstance(x, dict)), maxlen=32)

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'registered': [t['name'] for t in _REGISTRY],
            'dispatches': int(self.dispatches),
            'successes': int(self.successes),
            'claims_emitted': int(self.claims_emitted),
        }
