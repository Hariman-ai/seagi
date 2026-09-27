"""Exogenous-input grounding loop (Cap-3 SHADOW phase, Step 5).

Predicts the next INCOMING percept in the EXOGENOUS stream Seagi is FED
(text via intake — ingestion / forager / peer / curiosity_feeder), over
the `anticipates` relation, held OUT of RELATION_COMPOSITION so it can
never reach coherence/lifeforce except via the explicit world-credit
term (which is DISARMED in this phase).

"Modelling reality extends life": correctly anticipating exogenous
experience would EARN lifeforce — and it is NON-FARMABLE by construction,
because the agent does NOT author the stream.  It cannot manufacture a
surprise in input it does not generate (the toy-action-world farming
lever does not exist here — the user's correction: "the agent never
fully dictates input; most of it he is exposed to and makes sense of").

SHADOW PHASE (shadow_mode=True default): the would-be credit is computed,
attributed BY ORIGIN, and surfaced in stats() — but record_world_learning
is NEVER called.  Nothing touches lifeforce.  We observe what the wire
WOULD do on real input before it can ever be fatal.  Arming (a separate
later commit) adds the record_world_learning wire + an ABSOLUTE per-
episode cap independent of the deduction signal.

Credit per confirm = surprise_gate × precision_weight:
  - surprise_gate ∈ {0,1}: 1 only on the FIRST correct anticipation of a
    transition that was PREVIOUSLY WRONG, once per edge.  Already-mastered
    / never-surprised → 0, so trivial/repeated input earns ~0.  Non-
    farmable: the agent can't manufacture the prior miss in an exogenous
    stream.
  - precision_weight ∈ [0,1]: the state's self-derived reliability
    (Welford) — an irreducibly-uncertain source earns ~0 (luck pays
    nothing); a stably-modelled one earns full.

ORIGIN PROVENANCE (positive allowlist, not a blocklist): only genuinely-
external origins are credited.  `internal` (reverie), `tool` (agent-
initiated), and `sensor` (the ambiguous caller-supplied /observation
default) are EXCLUDED.  Arming additionally requires hardening the
caller-supplied /observation origin so an exogenous tag is a structural
guarantee, not an assertion — declared as a pre-arm must-fix.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from ..events import EventKind, BrainEvent, SubstrateWriteQueuedEvent
from seagi.core.substrate import PROVISIONAL_EDGE_STRENGTH

# Relation the exogenous loop predicts over.  Like `transitions_to`, it is
# deliberately ABSENT from RELATION_COMPOSITION (never coheres, never
# reaches newly_coherent / record_learning).
EXO_RELATION = 'anticipates'

# Positive allowlist of genuinely-EXTERNAL origins.  internal/tool/sensor
# excluded (self-authored / agent-initiated / ambiguous).
EXOGENOUS_ORIGINS = frozenset(
    {'ingestion', 'forager', 'peer', 'curiosity_feeder'})

PREDICTION_HISTORY = 512


class ExogenousGroundingLoop:
    """Predict the next exogenous percept; compute (but in shadow mode do
    NOT apply) the lifeforce credit for correctly anticipating reality."""

    SUBSCRIPTIONS = (EventKind.RAW_PERCEPT,)

    def __init__(self, engine: Any = None, bus: Any = None,
                 shadow_mode: bool = True,
                 record_world_learning: Optional[Callable] = None):
        self.engine = engine
        self.bus = bus
        self.shadow_mode = bool(shadow_mode)
        self._record = record_world_learning
        self._last_state: Optional[str] = None
        self.history: Deque[Tuple[int, str, Optional[str], str]] = deque(
            maxlen=PREDICTION_HISTORY)
        # per-(subject, relation, object) registry: [paid, was_wrong].
        self._registry: Dict[Tuple[str, str, str], List[int]] = {}
        # per-state predictive reliability (Welford over hit/miss): [n,mean,m2]
        self._rel: Dict[str, List[float]] = {}
        self.predictions_made = 0
        self.confirms = 0
        self.misses = 0
        self.no_prediction = 0
        self.skipped_nonexo = 0
        self.surprise_fires = 0
        # shadow credit accounting — COMPUTED, NOT APPLIED in shadow mode.
        self.shadow_credit_total = 0.0
        self.shadow_credit_by_origin: Dict[str, float] = {}
        self.applied_total = 0.0   # stays 0 while shadow_mode is True

    # ---- bus entry ----
    def handle(self, event: BrainEvent, bus: Any) -> None:
        if getattr(event, 'kind', None) != EventKind.RAW_PERCEPT:
            return
        if getattr(event, 'modality', 'text') != 'text':
            return
        origin = getattr(event, 'origin', '') or ''
        if origin not in EXOGENOUS_ORIGINS:
            # Self-authored / ambiguous input does NOT ground or chain —
            # and breaks the prediction chain so a self-percept can't sit
            # between two exogenous ones and be credited.
            self.skipped_nonexo += 1
            self._last_state = None
            return
        payload = getattr(event, 'payload', None) or {}
        state = self._dominant(payload)
        if state is None:
            return
        self.observe(self._last_state, state,
                     int(getattr(event, 'cycle', 0)), origin, bus)
        self._last_state = state

    @staticmethod
    def _magnitude(val: Any) -> float:
        try:
            return float(getattr(val, 'magnitude', val))
        except Exception:
            return 0.0

    def _dominant(self, payload: Dict[str, Any]) -> Optional[str]:
        best, best_m = None, -1.0
        for name, val in payload.items():
            m = self._magnitude(val)
            if m > best_m:
                best_m, best = m, str(name)
        return best

    # ---- core loop (also test-driven) ----
    def observe(self, s_prev: Optional[str], s_actual: str,
                cycle: int, origin: str = 'ingestion',
                bus: Any = None) -> float:
        bus = bus if bus is not None else self.bus
        if not s_prev:
            return 0.0
        s_pred = self._predict(s_prev, cycle)
        self.history.append((int(cycle), s_prev, s_pred, s_actual))
        key = (s_prev, EXO_RELATION, s_actual)
        if s_pred is None:
            self.no_prediction += 1
            self._write(s_prev, s_actual, 'world_observe', cycle, bus)
            return 0.0
        self.predictions_made += 1
        hit = (s_pred == s_actual)
        self._record_reliability(s_prev, 1.0 if hit else 0.0)
        if hit:
            self.confirms += 1
            credit = self._credit(key, s_prev, origin)
            self._write(s_prev, s_actual, 'world_confirm', cycle, bus)
            return credit
        self.misses += 1
        # mark the TRUE transition previously-wrong (we predicted otherwise).
        self._registry.setdefault(key, [0, 0])[1] = 1
        self._write(s_prev, s_actual, 'world_observe', cycle, bus)
        return 0.0

    # ---- credit (surprise-gated, precision-scaled) — DISARMED in shadow ----
    def _credit(self, key: Tuple[str, str, str],
                s_prev: str, origin: str) -> float:
        reg = self._registry.setdefault(key, [0, 0])
        paid, was_wrong = reg[0], reg[1]
        surprise = 1.0 if (was_wrong and not paid) else 0.0
        if surprise:
            reg[0] = 1
            self.surprise_fires += 1
        credit = surprise * self._precision_weight(s_prev)
        if credit > 0.0:
            self.shadow_credit_total += credit
            self.shadow_credit_by_origin[origin] = (
                self.shadow_credit_by_origin.get(origin, 0.0) + credit)
            if not self.shadow_mode and self._record is not None:
                # ARMED path (NOT taken in the shadow phase).
                self._record(credit)
                self.applied_total += credit
        return credit

    def _precision_weight(self, s_prev: str) -> float:
        st = self._rel.get(s_prev)
        if st is None or st[0] < 1:
            return 0.0
        n, mean, m2 = st
        var = m2 / n
        denom = mean * (1.0 - mean) + 1e-9
        norm_var = var / denom
        if norm_var > 1.0:
            norm_var = 1.0
        return mean * (1.0 - norm_var)

    def _record_reliability(self, s_prev: str, hit: float) -> None:
        st = self._rel.get(s_prev)
        if st is None:
            st = [0.0, 0.0, 0.0]
            self._rel[s_prev] = st
        st[0] += 1.0
        delta = hit - st[1]
        st[1] += delta / st[0]
        st[2] += delta * (hit - st[1])

    # ---- prediction (own strongest anticipates edge) ----
    def _strongest_transition(self, concept: Any, cycle: int) -> Optional[str]:
        best, best_s = None, -1.0
        for e in concept.edges_out.get(EXO_RELATION, ()):
            tgt = (e.target if isinstance(e.target, str)
                   else getattr(e.target, 'name', None))
            es = e.effective_strength(cycle)
            if es > best_s:
                best_s, best = es, tgt
        return best

    def _predict(self, s_prev: str, cycle: int) -> Optional[str]:
        sub = getattr(self.engine, 'substrate', None)
        if sub is None:
            return None
        c = sub.concepts.get(s_prev)
        if c is None:
            return None
        return self._strongest_transition(c, cycle)

    # ---- write (single-writer contract) ----
    def _write(self, subj: str, obj: str, reason: str,
               cycle: int, bus: Any) -> None:
        if bus is None:
            return
        bus.publish(SubstrateWriteQueuedEvent(
            kind=EventKind.SUBSTRATE_WRITE_QUEUED,
            cycle=int(cycle),
            source_capability='exogenous_grounding',
            origin='internal',
            subject=subj, relation=EXO_RELATION, object=obj,
            strength=PROVISIONAL_EDGE_STRENGTH, write_reason=reason))

    def stats(self) -> Dict[str, Any]:
        n = self.predictions_made
        return {
            'shadow_mode': self.shadow_mode,
            'predictions_made': n,
            'confirms': self.confirms,
            'misses': self.misses,
            'no_prediction': self.no_prediction,
            'skipped_nonexo': self.skipped_nonexo,
            'confirm_rate': (self.confirms / n) if n else 0.0,
            'surprise_fires': self.surprise_fires,
            'paid_transitions': sum(1 for v in self._registry.values()
                                    if v[0]),
            'shadow_credit_total': round(self.shadow_credit_total, 4),
            'shadow_credit_by_origin': {
                k: round(v, 4)
                for k, v in self.shadow_credit_by_origin.items()},
            'applied_total': round(self.applied_total, 4),
        }
