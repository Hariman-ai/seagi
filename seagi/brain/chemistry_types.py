"""Chemistry data types — the substrate for v2's M/I tagging.

These are the building blocks the Chemistry Engine and AWM use.
Kept separate from the capability implementations so other
capabilities can import them without cyclic dependencies.

Three core types:

  ChemistrySample      — one point-in-time chemistry reading
  ChemistryRingBuffer  — bounded history of samples (the time-series)
  EnrichedBubble       — bubble + receptor sensitivities + refractory

Plus the channel config (decay rates + baselines per NT).

Channels — per-channel time constants
-------------------------------------
Brain reality: different neurotransmitters have very different
half-lives.  NE acts in seconds; cortisol in minutes-to-hours.
v1 collapsed everything to one decay rate.  Phase 2 fixes this.

These constants are per-cycle decay multipliers: `new = baseline +
(cur - baseline) * (1 - decay)`.  Higher decay = faster return
to baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Per-channel chemistry config.
#
# Decay rates are PER TICK (chemistry.decay_tick).  At
# ingestion regime (1 sentence/tick), the brain's tick rate
# compresses biological time radically — a real cortisol
# response lingering for hours in biology can't have a
# 0.005-per-tick decay rate without saturating instantly
# during sustained reading.
#
# Calibrated 2026-05-14 (Session 7) after the 114-minute
# foundational ingestion saturated cortisol on every bubble.
# These rates produce balanced chemistry trajectories under
# both ingestion-tick and live-chat regimes.  The relative
# fast/medium/slow ordering is preserved (NE faster than
# dopamine faster than cortisol), just shifted.
CHANNELS: Dict[str, Dict[str, float]] = {
    # Fast: NE, acetylcholine, gaba
    'norepinephrine': {'decay': 0.20, 'baseline': 0.20},
    'acetylcholine':  {'decay': 0.15, 'baseline': 0.50},
    'gaba':           {'decay': 0.12, 'baseline': 0.50},
    # Medium: dopamine, endorphins, oxytocin
    'dopamine':       {'decay': 0.10, 'baseline': 0.30},
    'endorphins':     {'decay': 0.08, 'baseline': 0.10},
    'oxytocin':       {'decay': 0.07, 'baseline': 0.20},
    # Slow: serotonin, cortisol — but NOT so slow they ratchet.
    # Cortisol bumped 6x (0.005 → 0.03) so chronic low-grade
    # M-content exposure can't drive saturation.  Still slowest
    # channel; reflects that mortality response lingers longer
    # than dopamine reward.
    'serotonin':      {'decay': 0.05, 'baseline': 0.50},
    'cortisol':       {'decay': 0.03, 'baseline': 0.10},
    # Adenosine (sleep-pressure S, part-b v2 2026-07-13): NOT a valence
    # NT — it is the two-process fatigue scalar.  decay ==
    # EDGE_STRENGTH_DECAY_PER_CYCLE (substrate.py:61 = 0.00001, the
    # smallest system constant) → essentially ZERO wake clearance, so it
    # only RISES with cognitive effort (accumulate_adenosine) and only a
    # NAP discharges it (discharge_adenosine).  baseline 0.10 = the
    # rested floor.  EXCLUDED from valence use via IMPRINT_CHANNELS below
    # (never imprinted on bubbles, never a feeling dimension).
    'adenosine':      {'decay': 0.00001, 'baseline': 0.10},
}

# Channel-axis projection to M/I.  Mortality channels +
# immortality channels.  Same shape v1 used.  Adenosine is deliberately
# NOT in either axis — it is fatigue, not valence.
M_CHANNELS = ('cortisol', 'norepinephrine')
I_CHANNELS = ('dopamine', 'oxytocin', 'endorphins')

# The valence channels — every channel EXCEPT adenosine.  Bubble
# imprinting and feeling-signature learning iterate THIS, so the sleep-
# pressure scalar can never leak into affect / personality (auditor MF).
IMPRINT_CHANNELS = tuple(ch for ch in CHANNELS if ch != 'adenosine')

# Default receptor sensitivity per bubble.  Modified by
# experience: repeated firing of a channel down-regulates its
# receptor on that bubble (real brain: receptor desensitization).
DEFAULT_RECEPTOR_SENSITIVITY = 1.0

# Refractory window in cycles.  After a strong firing, the
# bubble's response to further events is dampened for this many
# cycles.  Brain-correct: synapses have refractory periods.
DEFAULT_REFRACTORY_CYCLES = 5

# Strength of lateral effects between co-active bubbles.  When
# bubble X fires strongly, similar co-active bubbles get a small
# chemistry nudge in the same direction.  Brain analog:
# associative binding (Hebbian co-firing).
LATERAL_COUPLING_STRENGTH = 0.10


# ---------------------------------------------------------------------
# ChemistrySample — one timestamped reading
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ChemistrySample:
    """One snapshot of chemistry state at a specific cycle.

    Used inside ring buffers to track trajectory.  Immutable.
    `values` is a dict of channel → level (0..1).
    `m_polarity` and `i_polarity` are derived; stored explicitly
    so the time-series can be queried without recomputing.
    """
    cycle: int
    timestamp: float
    values: Dict[str, float] = field(default_factory=dict)
    m_polarity: float = 0.0
    i_polarity: float = 0.0
    triggering_event: str = ''  # event kind that produced this

    @classmethod
    def from_state(cls,
                    cycle: int,
                    timestamp: float,
                    state: Dict[str, float],
                    triggering_event: str = '') -> 'ChemistrySample':
        """Build a sample from a current chemistry state dict.
        Computes M/I polarities from the canonical channels."""
        m = sum(state.get(ch, CHANNELS[ch]['baseline'])
                  for ch in M_CHANNELS) / len(M_CHANNELS)
        i = sum(state.get(ch, CHANNELS[ch]['baseline'])
                  for ch in I_CHANNELS) / len(I_CHANNELS)
        return cls(
            cycle=cycle,
            timestamp=timestamp,
            values=dict(state),
            m_polarity=m,
            i_polarity=i,
            triggering_event=triggering_event,
        )

    def get(self, channel: str) -> float:
        """Channel value with baseline fallback."""
        return self.values.get(
            channel, CHANNELS.get(channel, {}).get('baseline', 0.0))


# ---------------------------------------------------------------------
# ChemistryRingBuffer — bounded history per concept
# ---------------------------------------------------------------------


class ChemistryRingBuffer:
    """Bounded ring of `ChemistrySample` per active concept.

    The "chemistry time-series" the architecture document calls
    for.  Gives reflection real temporal context: *"my cortisol
    rose over the last N cycles while attending to this focal."*

    Default capacity 100 — covers ~minute-scale at typical event
    rates while staying small.

    Operations
    ----------
    append(sample)       O(1) amortized
    latest()             most recent sample
    trajectory(window)   last N samples
    mean(channel)        average value of a channel over buffer
    delta(channel)       (latest - oldest) for that channel
                          — useful for "rising / falling"
    """

    def __init__(self, capacity: int = 100):
        self.capacity = int(capacity)
        self._samples: List[ChemistrySample] = []

    def append(self, sample: ChemistrySample) -> None:
        self._samples.append(sample)
        if len(self._samples) > self.capacity:
            # Drop oldest in batches to amortize.
            drop = len(self._samples) - self.capacity
            del self._samples[:drop]

    def latest(self) -> Optional[ChemistrySample]:
        return self._samples[-1] if self._samples else None

    def oldest(self) -> Optional[ChemistrySample]:
        return self._samples[0] if self._samples else None

    def trajectory(self, window: int = 10) -> List[ChemistrySample]:
        if not self._samples:
            return []
        return list(self._samples[-window:])

    def mean(self, channel: str) -> float:
        if not self._samples:
            return CHANNELS.get(channel, {}).get('baseline', 0.0)
        return sum(s.get(channel) for s in self._samples) / len(
            self._samples)

    def delta(self, channel: str) -> float:
        """Latest - oldest.  Positive = rising; negative = falling."""
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1].get(channel) - self._samples[0].get(
            channel)

    def __len__(self) -> int:
        return len(self._samples)


# ---------------------------------------------------------------------
# EnrichedBubble — unified with the substrate's Bubble.
#
# Pre-merge: v2 EnrichedBubble lived here as a separate type,
# with translation methods to/from v1's Bubble.  That two-type
# split caused a 73-minute Layer 3 ingestion to produce ZERO
# persisted chemistry — accumulating imprints lived only on
# v2 bubbles, evaporated on AWM eviction, never reached v1
# substrate before save_brain.
#
# Post-merge (2026-05-14): EnrichedBubble is an ALIAS for the
# unified Bubble in seagi.core.bubble.  Same class used by:
#   - substrate persistence (concept.bubbles)
#   - v2 AWM (AWMEntry.bubble references it)
#   - v2 chemistry (imprints write to it)
# Modifying through AWM IS modifying substrate.  Persistence
# gap structurally impossible.
#
# Brain code continues to import `EnrichedBubble` for now;
# subsequent cleanup sessions will rename to `Bubble`.
# ---------------------------------------------------------------------

from seagi.core.bubble import Bubble as EnrichedBubble  # noqa: E402
