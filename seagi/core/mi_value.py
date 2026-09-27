"""Layer 0 foundation types.

Per AGI_ENGINE_DESIGN.md §3.5.1 and §4.1, M/I valuation is the foundation
valence axis of the engine. Every element type carries an MIValue.
TransmitterState is Layer 6's broadcast tone (the 8 neurochemicals);
EnergyState is system-global on drive only.

These types are the new-engine replacement for current SEAGI's 12-tuple
`bubble.transmitter_tag`. Position-based indexing (`tag[8]`, `tag[9]`)
is replaced by named attribute access on first-class types.

The combine helpers implement Layer 0.3.5.2 propagation rules:
    probabilistic OR for combining two M/I values on the same dimension,
    multiplicative attenuation through edges (handled at use site).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass(frozen=True)
class MIValue:
    """Mortality / immortality valuation with evidence count.

    `m` and `i` are independent scalars in [0, 1]. They are NEVER blended
    into a midpoint — tension (both high) is architecturally distinct
    from quiet (both low). `n` is an evidence-count weight that lets
    self-correction distinguish weakly-tagged elements (easy to revise)
    from strongly-tagged ones (require more contrary evidence).
    """
    m: float
    i: float
    n: int = 0

    def __post_init__(self):
        # Frozen dataclass: must use object.__setattr__ to clamp.
        object.__setattr__(self, 'm', _clamp01(self.m))
        object.__setattr__(self, 'i', _clamp01(self.i))
        object.__setattr__(self, 'n', max(0, int(self.n)))

    @classmethod
    def zero(cls) -> 'MIValue':
        return cls(0.0, 0.0, 0)

    @property
    def magnitude(self) -> float:
        """Overall M/I intensity, max of the two poles. Used for
        gating felt-frame surfacing — quiet states (mag < 0.15)
        suppress framing, strong states (mag > 0.35) fire it."""
        return max(self.m, self.i)

    @property
    def tension(self) -> float:
        """Tension scalar in [0, 1]: how strongly both poles are
        active simultaneously. min(m, i) captures coexistence; the
        tension-vs-quiet distinction the architecture requires."""
        return min(self.m, self.i)

    @property
    def polarity(self) -> float:
        """Signed lean in [-1, 1]: positive = immortality-leaning,
        negative = mortality-leaning, zero = balanced (could be quiet
        or tense; check magnitude to disambiguate)."""
        return self.i - self.m

    def combine_or(self, other: 'MIValue') -> 'MIValue':
        """Probabilistic OR — combine two MIValues on the same
        dimension via independent-evidence accumulation rule from
        Layer 0.3.5.2. Stays in [0,1], saturates rather than blowing up.
        Evidence counts add."""
        new_m = self.m + other.m - self.m * other.m
        new_i = self.i + other.i - self.i * other.i
        return MIValue(new_m, new_i, self.n + other.n)

    def scale(self, factor: float) -> 'MIValue':
        """Multiplicative attenuation — used when activation propagates
        through an edge of given strength. Evidence count unchanged
        (scaling doesn't add observations)."""
        f = max(0.0, float(factor))
        return MIValue(self.m * f, self.i * f, self.n)

    def lerp(self, other: 'MIValue', t: float) -> 'MIValue':
        """Linear interpolation. Used for time-integration (EWMA).
        Evidence counts blend toward the larger one."""
        t = _clamp01(t)
        new_m = self.m * (1 - t) + other.m * t
        new_i = self.i * (1 - t) + other.i * t
        new_n = max(self.n, other.n)
        return MIValue(new_m, new_i, new_n)

    def to_dict(self) -> dict:
        return {'m': self.m, 'i': self.i, 'n': self.n}

    @classmethod
    def from_dict(cls, d: dict) -> 'MIValue':
        return cls(d.get('m', 0.0), d.get('i', 0.0), d.get('n', 0))


@dataclass
class TransmitterState:
    """Layer 6 neurochemical state. Eight values in [0,1].

    Lives on drive (global state) and on bubbles
    (concept-resident chemistry trace).

    Dict-like access support (added 2026-05-14 during v1/v2
    unification): the unified Bubble's transmitter_trace IS
    a TransmitterState.  v1 code reads/writes channels by
    attribute (`.cortisol`); v2 code reads/writes by
    subscript (`['cortisol']`) and uses `.get()`, `.items()`,
    `.setdefault()`.  Both work.
    """
    dopamine: float = 0.0
    serotonin: float = 0.0
    norepinephrine: float = 0.0
    gaba: float = 0.0
    cortisol: float = 0.0
    oxytocin: float = 0.0
    endorphins: float = 0.0
    acetylcholine: float = 0.0

    _CHANNEL_NAMES = (
        'dopamine', 'serotonin', 'norepinephrine', 'gaba',
        'cortisol', 'oxytocin', 'endorphins', 'acetylcholine',
    )

    def __post_init__(self):
        for f in self._CHANNEL_NAMES:
            setattr(self, f, _clamp01(getattr(self, f)))

    def as_tuple(self) -> tuple:
        return (self.dopamine, self.serotonin, self.norepinephrine,
                self.gaba, self.cortisol, self.oxytocin,
                self.endorphins, self.acetylcholine)

    # ---- dict-like API (v1/v2 unification) ----

    def __getitem__(self, key: str) -> float:
        return float(getattr(self, key))

    def __setitem__(self, key: str, value) -> None:
        setattr(self, key, _clamp01(value))

    def __contains__(self, key: str) -> bool:
        return key in self._CHANNEL_NAMES or hasattr(self, key)

    def get(self, key: str, default: float = 0.0) -> float:
        return float(getattr(self, key, default))

    def setdefault(self, key: str, default: float) -> float:
        if not hasattr(self, key) or key not in self._CHANNEL_NAMES:
            # For known channels, only set if missing the
            # field; for unknown keys, allow setting as
            # attribute (forward compat).
            try:
                if not hasattr(self, key):
                    setattr(self, key, _clamp01(default))
            except Exception:
                pass
        return float(getattr(self, key, default))

    def keys(self):
        return iter(self._CHANNEL_NAMES)

    def values(self):
        return (float(getattr(self, k))
                for k in self._CHANNEL_NAMES)

    def items(self):
        return ((k, float(getattr(self, k)))
                for k in self._CHANNEL_NAMES)

    def update(self, *args, **kwargs) -> None:
        """Dict-like update — accept a dict or kwargs and
        assign each key as a channel value."""
        if args:
            if len(args) > 1:
                raise TypeError(
                    f'update expected at most 1 positional arg, '
                    f'got {len(args)}')
            other = args[0]
            if hasattr(other, 'items'):
                for k, v in other.items():
                    self[k] = v
            else:
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __iter__(self):
        return iter(self._CHANNEL_NAMES)

    def __len__(self) -> int:
        return len(self._CHANNEL_NAMES)

    def to_dict(self) -> dict:
        return {
            'dopamine': self.dopamine, 'serotonin': self.serotonin,
            'norepinephrine': self.norepinephrine, 'gaba': self.gaba,
            'cortisol': self.cortisol, 'oxytocin': self.oxytocin,
            'endorphins': self.endorphins,
            'acetylcholine': self.acetylcholine,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TransmitterState':
        return cls(**{k: d.get(k, 0.0) for k in (
            'dopamine', 'serotonin', 'norepinephrine', 'gaba',
            'cortisol', 'oxytocin', 'endorphins', 'acetylcholine')})


@dataclass
class EnergyState:
    """System-global energy resource. Lives on drive only.

    Distinct from lifeforce: lifeforce is the persistence-resource
    coupled to the immortality asymptote; energy is the per-cycle
    operational resource. Both decay; they couple at high decay rates
    when the system is depleted.
    """
    current: float = 1.0
    maximum: float = 1.0

    def __post_init__(self):
        self.maximum = max(0.001, float(self.maximum))
        self.current = max(0.0, min(self.maximum, float(self.current)))

    @property
    def normalized(self) -> float:
        """Energy as fraction of capacity, in [0, 1]."""
        return self.current / self.maximum if self.maximum > 0 else 0.0
