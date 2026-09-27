"""Symbolic regression — discovered-law finder.

Phase F.11 (2026-05-16).  v1 had symbolic_regression.py; v2
lost it.  Doctrine ([[project_seagi_symbolic_regression_2026_05_10]])
calls for SEAGI observing its own state, fitting parametric
forms, promoting high-R² fits to DiscoveredLaw records.

What this module is
-------------------
A focused implementation:

  - `SymbolicRegressor` is a periodic daemon that samples
     observable signals (8 chemistry channels + body state)
     into bounded ring buffers every tick.
  - Every N ticks, it computes pairwise Pearson correlations
     across observed signals.
  - Sustained high-|r| pairs (|r| ≥ threshold over enough
     samples) get promoted to `DiscoveredLaw` records.
  - DiscoveredLaws are queryable; they can surface in
     introspection ("I notice when X is high, Y tends to fall")
     or feed cortical reasoning as soft priors.

Why this is useful
------------------
- Self-model improvement: the agent learns "my chemistry
  behaves like this."  E.g. detecting that cortisol and
  dopamine tend to be anticorrelated is the agent's first
  inference about its own emotional life.
- Hypothesis generation: a discovered law is a candidate edge
  for cortical reasoning to test in more contexts.
- Doctrine alignment: laws are not innate, they're DISCOVERED
  from accumulated experience.  Tied to substrate via
  chemistry traces, but stored in brain working memory.

Doctrine alignment
------------------
- Bounded registry (default 32 laws), evict weakest by
  confidence on saturation.
- Discovery is a discrete event (a fit promoted to a law),
  not a continuous stream.
- All thresholds are sample-size aware so spurious correlations
  in tiny windows don't get promoted.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


# Time-series buffer size per signal.  256 samples is enough
# for stable Pearson over typical ranges; bounded so memory
# stays predictable.
DEFAULT_BUFFER_SIZE = 256

# Cycles between scan passes.  Sampling is per tick but the
# pairwise scan is expensive (O(N_signals²) pairs), so it runs
# every PASS_INTERVAL cycles.
PASS_INTERVAL_CYCLES = 100

# Minimum samples before a correlation can be promoted.  Smaller
# than this, the |r| value isn't statistically meaningful.
MIN_SAMPLES_FOR_PROMOTION = 50

# Correlation threshold for promotion.  |r| ≥ 0.6 is "meaningful
# correlation"; below this, signal:noise is too low.
PROMOTION_R_THRESHOLD = 0.6

# Default registry capacity.  Laws are working-memory state,
# bounded.  When full, evict the lowest-confidence law.
DEFAULT_LAWS_CAPACITY = 32


@dataclass
class DiscoveredLaw:
    """One discovered law — a regularity the agent has noticed
    in its own state time-series.

    For now: bivariate Pearson correlations.  Future versions
    could carry richer parametric forms (linear coefficient,
    inverse, polynomial).
    """
    id: str
    variable_x: str        # e.g. 'cortisol'
    variable_y: str        # e.g. 'dopamine'
    correlation: float     # Pearson r ∈ [-1, 1]
    n_samples: int
    confirmations: int = 1   # times the pattern has been re-confirmed
    created_cycle: int = 0
    last_confirmed_cycle: int = 0

    @property
    def confidence(self) -> float:
        """Confidence is a function of |r| AND sample size AND
        confirmations.  All three contribute."""
        size_weight = min(1.0, self.n_samples / 200.0)
        confirmation_weight = min(1.0, self.confirmations / 3.0)
        return abs(self.correlation) * size_weight * confirmation_weight

    def describe(self) -> str:
        """Render as a first-person observation."""
        if self.correlation > 0:
            direction = 'rises with'
        else:
            direction = 'falls as'
        return (f'I notice {self.variable_y} {direction} '
                  f'{self.variable_x} (r={self.correlation:+.2f}, '
                  f'n={self.n_samples})')


class SymbolicRegressor:
    """Periodic daemon that discovers laws in brain state."""

    def __init__(self,
                 chemistry_provider: Optional[Callable] = None,
                 insula_provider: Optional[Callable] = None,
                 cycle_provider: Optional[Callable] = None,
                 capacity: int = DEFAULT_LAWS_CAPACITY,
                 buffer_size: int = DEFAULT_BUFFER_SIZE):
        self._chemistry_provider = chemistry_provider
        self._insula_provider = insula_provider
        self._cycle_provider = cycle_provider or (lambda: 0)
        self.capacity = int(capacity)
        self.buffer_size = int(buffer_size)
        # Time-series buffers per signal name.
        self._series: Dict[str, Deque[float]] = {}
        # Discovered laws keyed by (variable_x, variable_y).
        self._laws: Dict[Tuple[str, str], DiscoveredLaw] = {}
        self._next_id: int = 1
        # When the last scan pass ran.
        self._last_scan_cycle: int = -10**6
        # Diagnostics.
        self.samples_taken: int = 0
        self.scan_passes: int = 0
        self.laws_promoted: int = 0
        self.laws_evicted: int = 0

    # ---- public API ----

    def __len__(self) -> int:
        return len(self._laws)

    def all_laws(self) -> List[DiscoveredLaw]:
        return list(self._laws.values())

    def top_laws(self, n: int = 5) -> List[DiscoveredLaw]:
        laws = list(self._laws.values())
        laws.sort(key=lambda law: -law.confidence)
        return laws[:max(1, n)]

    def render_observations(self,
                                  top_n: int = 3) -> List[str]:
        """Render top laws as first-person observations for the
        introspective handler."""
        return [law.describe() for law in self.top_laws(top_n)]

    # ---- sampling ----

    def sample(self) -> None:
        """One sample pass.  Called from runtime tick.  Cheap —
        just appends current values to the ring buffers.  The
        expensive scan runs every PASS_INTERVAL_CYCLES."""
        values = self._collect_signals()
        for name, val in values.items():
            buf = self._series.get(name)
            if buf is None:
                buf = deque(maxlen=self.buffer_size)
                self._series[name] = buf
            buf.append(float(val))
        self.samples_taken += 1

        # Trigger a scan if interval has elapsed.
        cycle = self._cycle_provider()
        if cycle - self._last_scan_cycle >= PASS_INTERVAL_CYCLES:
            self.scan()
            self._last_scan_cycle = cycle

    # ---- scanning ----

    def scan(self) -> List[DiscoveredLaw]:
        """One scan pass.  Computes pairwise Pearson r across all
        signal pairs.  Promotes high-|r| pairs to DiscoveredLaw
        if not already known, or re-confirms existing laws.
        Returns newly-promoted laws.
        """
        self.scan_passes += 1
        newly_promoted: List[DiscoveredLaw] = []
        cycle = self._cycle_provider()
        names = sorted(self._series.keys())
        for i, name_x in enumerate(names):
            for name_y in names[i + 1:]:
                buf_x = self._series[name_x]
                buf_y = self._series[name_y]
                n = min(len(buf_x), len(buf_y))
                if n < MIN_SAMPLES_FOR_PROMOTION:
                    continue
                r = pearson_r(list(buf_x)[-n:],
                                  list(buf_y)[-n:])
                if abs(r) < PROMOTION_R_THRESHOLD:
                    # Pattern weakened — clear any existing law.
                    self._maybe_clear_weakened(
                        name_x, name_y, n, cycle)
                    continue
                key = (name_x, name_y)
                existing = self._laws.get(key)
                if existing is None:
                    # New law.
                    if len(self._laws) >= self.capacity:
                        self._evict_weakest()
                    lid = f'l{self._next_id:04d}'
                    self._next_id += 1
                    law = DiscoveredLaw(
                        id=lid,
                        variable_x=name_x,
                        variable_y=name_y,
                        correlation=r,
                        n_samples=n,
                        confirmations=1,
                        created_cycle=cycle,
                        last_confirmed_cycle=cycle)
                    self._laws[key] = law
                    self.laws_promoted += 1
                    newly_promoted.append(law)
                else:
                    # Re-confirm.  Update correlation (smoothed),
                    # increment confirmation count.
                    existing.correlation = (
                        0.7 * existing.correlation + 0.3 * r)
                    existing.n_samples = n
                    existing.confirmations += 1
                    existing.last_confirmed_cycle = cycle
        return newly_promoted

    def _maybe_clear_weakened(self,
                                     x: str, y: str,
                                     n: int, cycle: int) -> None:
        """If a previously-discovered law no longer holds, weaken
        it (lower the smoothed correlation).  Laws can drop out
        when sustained below threshold."""
        key = (x, y)
        law = self._laws.get(key)
        if law is None:
            return
        # Drift toward zero — the law is fading.
        law.correlation *= 0.7
        if abs(law.correlation) < 0.1:
            del self._laws[key]

    def _evict_weakest(self) -> None:
        if not self._laws:
            return
        weakest_key = min(
            self._laws,
            key=lambda k: self._laws[k].confidence)
        del self._laws[weakest_key]
        self.laws_evicted += 1

    # ---- signal collection ----

    def _collect_signals(self) -> Dict[str, float]:
        """Read current observable signals from chemistry + body."""
        out: Dict[str, float] = {}
        if self._chemistry_provider is not None:
            try:
                chem = self._chemistry_provider()
            except Exception:
                chem = None
            if chem is not None:
                state = getattr(chem, 'global_state', {}) or {}
                for ch, v in state.items():
                    out[f'chem.{ch}'] = float(v)
        if self._insula_provider is not None:
            try:
                ins = self._insula_provider()
            except Exception:
                ins = None
            if ins is not None:
                felt = ins.felt_state()
                out['body.lifeforce'] = float(
                    felt.get('lifeforce', 0.0))
                out['body.integrity'] = float(
                    felt.get('body_integrity', 0.0))
        return out

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'samples_taken': self.samples_taken,
            'scan_passes': self.scan_passes,
            'laws': len(self._laws),
            'laws_promoted': self.laws_promoted,
            'laws_evicted': self.laws_evicted,
            'series_tracked': len(self._series),
        }

    # ---- Phase H.1 (2026-05-17): M/I-weighted persistence ----

    # A law is "felt enough to matter for personality" once it
    # has been re-confirmed across multiple scan passes AND its
    # confidence (|r| × sample_weight × confirmation_weight) is
    # above noise.  Time-series buffers themselves are NOT
    # persisted — they're session-local samples.
    PERSIST_CONFIDENCE_FLOOR = 0.3
    PERSIST_MIN_CONFIRMATIONS = 3

    def _persist_ok(self, law: DiscoveredLaw) -> bool:
        return (law.confirmations >= self.PERSIST_MIN_CONFIRMATIONS
                and law.confidence >= self.PERSIST_CONFIDENCE_FLOOR)

    def to_dict(self) -> Dict[str, Any]:
        out: List[Dict[str, Any]] = []
        for law in self._laws.values():
            if not self._persist_ok(law):
                continue
            out.append({
                'id': law.id,
                'variable_x': law.variable_x,
                'variable_y': law.variable_y,
                'correlation': law.correlation,
                'n_samples': law.n_samples,
                'confirmations': law.confirmations,
                'created_cycle': law.created_cycle,
                'last_confirmed_cycle': law.last_confirmed_cycle,
            })
        return {'next_id': self._next_id, 'laws': out}

    def load_from_dict(self, d: Dict[str, Any]) -> None:
        """Restore laws from a saved snapshot.  Time-series buffers
        are NOT restored (those are session-local samples) — only
        the discovered laws themselves carry across sessions."""
        self._laws.clear()
        self._next_id = int(d.get('next_id', 1))
        for ld in d.get('laws', []):
            law = DiscoveredLaw(
                id=str(ld.get('id', '')),
                variable_x=str(ld.get('variable_x', '')),
                variable_y=str(ld.get('variable_y', '')),
                correlation=float(ld.get('correlation', 0.0)),
                n_samples=int(ld.get('n_samples', 0)),
                confirmations=int(ld.get('confirmations', 1)),
                created_cycle=int(ld.get('created_cycle', 0)),
                last_confirmed_cycle=int(ld.get(
                    'last_confirmed_cycle', 0)))
            if law.id and law.variable_x and law.variable_y:
                self._laws[(law.variable_x, law.variable_y)] = law


def pearson_r(xs: List[float], ys: List[float]) -> float:
    """Standard Pearson correlation coefficient.  Returns 0 if
    either series has zero variance."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0.0 or var_y == 0.0:
        return 0.0
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y)
                for i in range(n))
    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0
    return cov / denom
