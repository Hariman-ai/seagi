"""Unified Bubble — single source of truth for concept-level
chemistry trace.

Doctrinal alignment
-------------------
This is the post-merge unified type.  It replaces both the
legacy v1 `Bubble` (in agi_engine/substrate.py) and the v2
`EnrichedBubble` (in agi_engine/brain/chemistry_types.py).
Same class used by:
  - substrate persistence: `concept.bubbles: List[Bubble]`
  - v2 AWM: AWMEntry.bubble references the same object
  - v2 chemistry: imprints write directly to it
  - Layer 3 ingestion: trace accumulates here

The key insight that eliminates an entire class of bugs:
**the AWM Bubble IS the substrate Bubble.**  Modifying it in
the brain IS modifying it in storage.  Persistence by
construction.  No translation layer.  No flush step.  The
73-minute-wasted persistence-gap bug cannot recur.

Design feature provenance
-------------------------
From v2 (brain-correct chemistry features):
  - transmitter_trace as Dict[str, float] (flexible
    multi-channel)
  - receptor_sensitivity per channel (desensitization
    over use)
  - refractory_until (post-firing dampening)
  - is_refractory / m_polarity / i_polarity helpers

From v1 (persistence + context):
  - Typed ContextKey for context-distinguishable bubbles
    (same word "fire" in different contexts)
  - created_cycle (for crystallization-by-age dynamics)
  - to_dict / from_dict serialization via TransmitterState
    adapter (backward-compat with existing brain files)
  - crystallization (settled-feeling)

The transmitter_trace is INTERNALLY a dict (v2 style); on
serialization it converts to v1's TransmitterState named-field
format so existing seagi_brain.json files keep loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Channel configuration — imported lazily inside methods to
# avoid a circular import (brain/chemistry_types.py defines
# CHANNELS but also will eventually import Bubble for type
# hints; lazy import breaks the cycle).
def _channels():
    from seagi.brain.chemistry_types import CHANNELS
    return CHANNELS


def _m_channels():
    from seagi.brain.chemistry_types import M_CHANNELS
    return M_CHANNELS


def _i_channels():
    from seagi.brain.chemistry_types import I_CHANNELS
    return I_CHANNELS


def _default_refractory_cycles():
    from seagi.brain.chemistry_types import (
        DEFAULT_REFRACTORY_CYCLES)
    return DEFAULT_REFRACTORY_CYCLES


def _default_context_key():
    from .substrate import ContextKey
    return ContextKey()


@dataclass
class Bubble:
    """One context-keyed transmitter trace on a concept.

    A concept doesn't have a single chemical fingerprint; it
    has many — one per context.  This Bubble is one such
    fingerprint, mutable through chemistry firings, persisted
    via the substrate's normal serialization path.

    Lifecycle:
      created — `__post_init__` fills transmitter_trace and
                  receptor_sensitivity from channel baselines
      promoted to AWM — AWMEntry references THIS object;
                  modifications during AWM residency are
                  immediately visible to substrate
      imprinted — chemistry events update transmitter_trace
                  via _apply_to_bubble; encounter_count++
      evicted from AWM — Bubble stays where it is on the
                  Concept; only the AWMEntry wrapper is dropped
      persisted — via Concept.to_dict → Bubble.to_dict
    """
    # === Transmitter state ===
    # A TransmitterState (dataclass with 8 channel fields)
    # that ALSO supports dict-like access (via __getitem__,
    # .get, .items, etc.).  Both v1 attribute access
    # (`.cortisol`) and v2 dict access (`['cortisol']`) work.
    # Constructor also accepts a plain dict and auto-converts.
    transmitter_trace: Any = None

    # === v2 brain-correct chemistry features ===
    # Per-channel receptor sensitivity.  Decreases with
    # repeated strong firings on a channel — desensitization
    # over use.  Brain-correct synaptic plasticity.
    receptor_sensitivity: Dict[str, float] = field(
        default_factory=dict)
    # Cycle through which this bubble's response to chemistry
    # is dampened after a strong firing.  Brain-correct
    # synaptic refractory.
    refractory_until: int = -10**9

    # === v1 context support ===
    # ContextKey distinguishes "same concept in different
    # context" — "fire" in fireplace context vs house-fire
    # context get different bubbles even though same concept.
    context_key: Optional[Any] = None
    # Optional concept name — convenience for AWM and
    # diagnostics; not always populated.
    concept_name: str = ''

    # === Shared lifecycle ===
    encounter_count: int = 0
    created_cycle: int = 0
    last_active_cycle: int = 0
    crystallization: float = 0.0

    def __post_init__(self) -> None:
        # Normalize transmitter_trace to a TransmitterState (the
        # dict-like dataclass).  Accepts: TransmitterState,
        # dict, None.  Both v1 attribute-style and v2 dict-style
        # access work after this.
        from .mi_value import TransmitterState
        if self.transmitter_trace is None:
            self.transmitter_trace = TransmitterState()
        elif isinstance(self.transmitter_trace, TransmitterState):
            pass  # already correct
        elif isinstance(self.transmitter_trace, dict):
            self.transmitter_trace = TransmitterState(
                **{k: float(v)
                    for k, v in self.transmitter_trace.items()
                    if k in TransmitterState._CHANNEL_NAMES})
        else:
            # Some object with channel attributes (v1 Bubble
            # might pass something duck-typed) — copy channels.
            channels = _channels() if _channels() else {}
            new = TransmitterState()
            for ch in TransmitterState._CHANNEL_NAMES:
                if hasattr(self.transmitter_trace, ch):
                    setattr(new, ch, float(
                        getattr(self.transmitter_trace, ch)))
            self.transmitter_trace = new
        # Default-fill any channel still at 0.0 with its
        # baseline (only when caller hasn't set it).
        try:
            channels = _channels()
        except Exception:
            channels = {}
        for ch, cfg in channels.items():
            cur = self.transmitter_trace.get(ch, 0.0)
            if cur == 0.0:
                self.transmitter_trace[ch] = cfg['baseline']
            self.receptor_sensitivity.setdefault(ch, 1.0)
        # Default the context_key to a fresh ContextKey if not
        # provided.  Lazy because importing ContextKey at module
        # load time creates a circular import with substrate.py.
        if self.context_key is None:
            try:
                self.context_key = _default_context_key()
            except Exception:
                self.context_key = None

    def effective_trace(self, cycle: int):
        """Transmitter trace with lazy decay toward baseline -- the
        chemistry mirror of Edge.effective_strength / effective_salience.
        Unre-activated bubbles relax each channel multiplicatively toward
        baseline so derived M/I (mattering) FADES from disuse and is RE-
        EVOKED on re-anchor (last_active_cycle reset by the return path).
        Rate DERIVED from the substrate-memory horizon -- no new constant.
        SHADOW-STAGED: not yet authoritative in Concept.mi.
        """
        from .mi_value import TransmitterState
        elapsed = int(cycle) - int(self.last_active_cycle)
        if elapsed <= 0:
            return self.transmitter_trace
        from .substrate import (EDGE_PRUNE_FLOOR,
                                 EDGE_STRENGTH_DECAY_PER_CYCLE)
        horizon = max(1.0, EDGE_PRUNE_FLOOR / EDGE_STRENGTH_DECAY_PER_CYCLE)
        frac = 0.5 ** (elapsed / horizon)
        channels = _channels() if _channels() else {}
        out = TransmitterState()
        for ch in TransmitterState._CHANNEL_NAMES:
            base = channels.get(ch, {}).get("baseline", 0.0)
            cur = self.transmitter_trace.get(ch, base)
            out[ch] = base + (cur - base) * frac
        return out

    # ---- chemistry helpers ----

    def is_refractory(self, cycle: int) -> bool:
        return cycle < self.refractory_until

    def m_polarity(self) -> float:
        """M-side projection — averaged cortisol + NE relative
        to channels.  Doctrine: M/I is the tension expressed
        BY the NT state, not stored separately."""
        try:
            channels = _channels()
            m_chans = _m_channels()
        except Exception:
            return 0.0
        return sum(
            self.transmitter_trace.get(
                ch, channels.get(ch, {}).get('baseline', 0.0))
            for ch in m_chans) / max(1, len(m_chans))

    def i_polarity(self) -> float:
        """I-side projection — averaged dopamine + oxytocin +
        endorphins."""
        try:
            channels = _channels()
            i_chans = _i_channels()
        except Exception:
            return 0.0
        return sum(
            self.transmitter_trace.get(
                ch, channels.get(ch, {}).get('baseline', 0.0))
            for ch in i_chans) / max(1, len(i_chans))

    # ---- persistence ----

    def to_dict(self) -> dict:
        """Backward-compat serialization.  transmitter_trace is
        a TransmitterState (with dict-like methods), so
        .to_dict() produces the v1-compatible named-field shape
        existing brain files expect.  Receptor sensitivity +
        refractory + context_key written as extension fields
        (read by from_dict, ignored by old v1-only loaders).
        """
        d = {
            'transmitter_trace': self.transmitter_trace.to_dict(),
            'encounter_count': int(self.encounter_count),
            'created_cycle': int(self.created_cycle),
            'last_active_cycle': int(self.last_active_cycle),
            'crystallization': float(self.crystallization),
            # v2 extensions
            'receptor_sensitivity': dict(self.receptor_sensitivity),
            'refractory_until': int(self.refractory_until),
        }
        # ContextKey serialization (if available).
        if self.context_key is not None:
            ck_to_dict = getattr(self.context_key, 'to_dict', None)
            if callable(ck_to_dict):
                d['context_key'] = ck_to_dict()
        if self.concept_name:
            d['concept_name'] = self.concept_name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Bubble':
        """Backward-compat deserialization.  Reads TransmitterState-
        style named fields directly.  v2 extension fields
        (receptor_sensitivity, refractory_until) default if
        absent (old v1-format files)."""
        from .mi_value import TransmitterState
        from .substrate import ContextKey
        try:
            channels = _channels()
        except Exception:
            channels = {}
        tr_d = d.get('transmitter_trace', {})
        ts = TransmitterState.from_dict(tr_d)
        # Receptor sensitivity defaults to 1.0 per channel for
        # files written before this field existed.
        sens = dict(d.get('receptor_sensitivity', {}))
        for ch in channels:
            sens.setdefault(ch, 1.0)
        ck_d = d.get('context_key')
        if ck_d is not None:
            ck = ContextKey.from_dict(ck_d)
        else:
            ck = ContextKey()
        return cls(
            transmitter_trace=ts,
            receptor_sensitivity=sens,
            refractory_until=int(
                d.get('refractory_until', -10**9)),
            context_key=ck,
            concept_name=str(d.get('concept_name', '') or ''),
            encounter_count=int(d.get('encounter_count', 0)),
            created_cycle=int(d.get('created_cycle', 0)),
            last_active_cycle=int(d.get('last_active_cycle', 0)),
            crystallization=float(d.get('crystallization', 0.0)),
        )


# ---------------------------------------------------------------------
# Context capture + bubble-pool selection — the meaning-making core.
#
# Doctrine (feedback_seagi_doctrine_2026_05_15):
#   "Same word, different context, different cocktail."  A concept
#   doesn't have ONE chemical fingerprint; it has many — one per
#   context.  Children's-book alligator vs golf-course alligator vs
#   river-swim alligator are three different bubbles, each tagged
#   with the chemistry that was firing at the moment of encounter.
#
#   "Low energy / fire when needed."  Co-activation is cheap.
#   Spawning bubbles is expensive.  Most encounters route their
#   small imprint into the closest existing bubble.  A new bubble
#   spawns only when (a) context-similarity to all existing bubbles
#   is low AND (b) the chemistry magnitude is high enough to justify
#   the cost.
# ---------------------------------------------------------------------


# Number of buckets to quantize each chemistry channel into.
# 10 buckets across [0..1] → 0.1 width.  context_similarity in
# substrate.py treats a 3-bucket gap on any single channel as a
# context shift; that matches "0.3 transmitter departure marks
# real change."
CHEMISTRY_SIGNATURE_BUCKETS = 10

# Maximum number of coactive concept names captured in a ContextKey.
# Doctrine: a few high-salience neighbors are the context, not the
# whole AWM.
COACTIVE_CONTEXT_CAP = 8

# Above this context_similarity, an encounter reuses the closest
# existing bubble.  Below it, the encounter is in a "different
# enough" context that the existing bubble doesn't represent.
SPAWN_SIMILARITY_THRESHOLD = 0.5

# Below this magnitude, even a low-similarity context doesn't
# justify spawning a new bubble.  Mundane novel contexts don't
# proliferate bubbles; only vivid novel contexts do.
SPAWN_INTENSITY_THRESHOLD = 0.10


_SPREAD_PROVIDER = None


def set_spread_provider(fn) -> None:
    """His own lived spread per channel, so the bucket width is measured
    rather than chosen."""
    global _SPREAD_PROVIDER
    _SPREAD_PROVIDER = fn


def _SELFCAL_ON():
    """Calibrate from his measured experience instead of the config."""
    try:
        import os as _os
        return _os.path.exists('/root/SELFCAL_ON')
    except Exception:
        return False


def _lived_spread(ch):
    """EWMA of |departure| he has actually experienced on this channel."""
    if _SPREAD_PROVIDER is None or not _SELFCAL_ON():
        return None
    try:
        v = (_SPREAD_PROVIDER() or {}).get(ch)
        return float(v) if v and float(v) > 0.0 else None
    except Exception:
        return None


def _CHEMBUCKET_ON():
    """Bucket chemistry by DEPARTURE scaled to each channel's own
    achievable excursion, so dopamine and noradrenaline can
    discriminate.  File-gated at /root/CHEMBUCKET_ON."""
    try:
        import os as _os
        return _os.path.exists('/root/CHEMBUCKET_ON')
    except Exception:
        return False


# How many levels each side of baseline a channel gets.
CHEM_LEVELS = 4
_CHEM_SCALE_CACHE = {}


def _chem_scale(ch):
    """The channel's own achievable excursion = biggest event /
    decay.  None when it does not decay (adenosine), which then
    keeps a fixed absolute width.  Cached; derived from the live
    channel config and event table, never hand-set."""
    if ch in _CHEM_SCALE_CACHE:
        return _CHEM_SCALE_CACHE[ch]
    scale = None
    try:
        from seagi.brain.chemistry_types import CHANNELS as _C
        from seagi.brain.capabilities.chemistry import (
            EVENT_DELTAS as _E)
        dec = float(_C.get(ch, {}).get('decay', 0.0) or 0.0)
        evs = [abs(float(v.get(ch, 0.0)))
               for v in _E.values() if ch in v]
        big = max(evs) if evs else 0.0
        if dec > 0.0 and big > 0.0:
            scale = big / dec
    except Exception:
        scale = None
    _CHEM_SCALE_CACHE[ch] = scale
    return scale


def _scaled_bucket(ch, val, baseline):
    """Signed departure in units of scale/CHEM_LEVELS, offset so
    the result stays a small non-negative int."""
    # HIS measurement first, the config only as a fallback.
    lived = _lived_spread(ch)
    if lived:
        steps = int(round((val - baseline) / lived))
        if steps < -CHEM_LEVELS:
            steps = -CHEM_LEVELS
        elif steps > CHEM_LEVELS:
            steps = CHEM_LEVELS
        return steps + CHEM_LEVELS
    scale = _chem_scale(ch)
    if not scale or scale <= 0.0:
        # undamped channel: keep absolute bucketing
        b = int(val * CHEMISTRY_SIGNATURE_BUCKETS)
        return min(b, CHEMISTRY_SIGNATURE_BUCKETS - 1)
    width = scale / float(CHEM_LEVELS)
    if width <= 0.0:
        return CHEM_LEVELS
    steps = int(round((val - baseline) / width))
    if steps < -CHEM_LEVELS:
        steps = -CHEM_LEVELS
    elif steps > CHEM_LEVELS:
        steps = CHEM_LEVELS
    return steps + CHEM_LEVELS


def compute_chemistry_signature(global_state: Dict[str, float]
                                  ) -> tuple:
    """Bucket each channel's current global value into a small int.
    Returns a tuple in stable channel order — comparable by
    `context_similarity()` in substrate.py.

    The bucketing makes contexts compare as "same enough" or
    "different enough" without floating-point sensitivity.  Channels
    that depart by <0.1 stay in the same bucket; departures of >=0.3
    cross 3 buckets and mark the context as distinct.
    """
    try:
        channels = _channels()
    except Exception:
        return tuple()
    channel_order = sorted(channels.keys())
    out = []
    for ch in channel_order:
        baseline = channels[ch].get('baseline', 0.0)
        val = float(global_state.get(ch, baseline))
        # Clamp to [0, 1] then bucket.
        if val < 0.0:
            val = 0.0
        elif val > 1.0:
            val = 1.0
        if _CHEMBUCKET_ON():
            bucket = _scaled_bucket(ch, val, baseline)
        else:
            bucket = int(val * CHEMISTRY_SIGNATURE_BUCKETS)
            if bucket >= CHEMISTRY_SIGNATURE_BUCKETS:
                bucket = CHEMISTRY_SIGNATURE_BUCKETS - 1
        out.append(bucket)
    return tuple(out)


def compute_context_key(global_chemistry: Dict[str, float],
                          coactive_names) -> Any:
    """Build a ContextKey from the current global chemistry state
    and an iterable of currently-coactive concept names.

    Caller is responsible for ordering `coactive_names` by salience
    (most-relevant first) — this function caps at
    `COACTIVE_CONTEXT_CAP`.
    """
    from .substrate import ContextKey
    sig = compute_chemistry_signature(global_chemistry)
    # Cap.  Caller's order is honored; frozenset loses order but
    # that's fine for Jaccard comparison.
    capped = []
    for name in coactive_names:
        if len(capped) >= COACTIVE_CONTEXT_CAP:
            break
        capped.append(name)
    return ContextKey(
        chemistry_signature=sig,
        coactive_concepts=frozenset(capped))


def find_or_spawn_bubble(concept,
                            current_context,
                            magnitude: float,
                            cycle: int) -> 'Bubble':
    """Find the closest-context bubble in concept.bubbles, or spawn
    a new one if context drift is large AND chemistry magnitude is
    significant.

    Returns the Bubble that should receive the upcoming imprint.
    The bubble is guaranteed to be in `concept.bubbles` (appended
    if newly spawned).

    Decision tree:
      - concept has no bubbles → spawn first bubble
      - best-match similarity >= SPAWN_SIMILARITY_THRESHOLD → reuse
      - elif magnitude >= SPAWN_INTENSITY_THRESHOLD → spawn new
      - else (low similarity but low intensity) → reuse closest
        (low-significance encounter; cheap to merge into closest)

    This is the low-energy convergence point.  Most encounters
    reuse.  Only distinctive-AND-vivid moments spawn.
    """
    from .substrate import context_similarity
    bubbles = getattr(concept, 'bubbles', None)
    if bubbles is None:
        # Caller didn't supply a real Concept (or it has no bubble
        # list).  Return a fresh detached bubble — caller may or
        # may not attach it.
        return Bubble(
            context_key=current_context,
            concept_name=getattr(concept, 'name', ''),
            created_cycle=cycle,
            last_active_cycle=cycle,
        )

    # No bubbles yet → spawn the first one.
    if not bubbles:
        new_bubble = Bubble(
            context_key=current_context,
            concept_name=getattr(concept, 'name', ''),
            created_cycle=cycle,
            last_active_cycle=cycle,
        )
        bubbles.append(new_bubble)
        return new_bubble

    # Find best-match bubble by context similarity.
    best_sim = -1.0
    best_bubble = bubbles[0]
    for b in bubbles:
        bck = getattr(b, 'context_key', None)
        if bck is None:
            continue
        try:
            sim = context_similarity(current_context, bck)
        except Exception:
            sim = 0.0
        if sim > best_sim:
            best_sim = sim
            best_bubble = b

    # Reuse if similarity is high enough.
    if best_sim >= SPAWN_SIMILARITY_THRESHOLD:
        return best_bubble

    # Spawn if intensity justifies it.
    if magnitude >= SPAWN_INTENSITY_THRESHOLD:
        new_bubble = Bubble(
            context_key=current_context,
            concept_name=getattr(concept, 'name', ''),
            created_cycle=cycle,
            last_active_cycle=cycle,
        )
        bubbles.append(new_bubble)
        return new_bubble

    # Low similarity, low intensity → reuse closest existing.
    # Mundane novel contexts don't proliferate bubbles.
    return best_bubble
