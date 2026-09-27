"""Chemistry Engine — M/I tagging, NT dynamics, bubble enrichment.

Brain analog: brainstem nuclei + limbic + endocrine collectively.
The substrate of felt-state.  Everything that ever shapes
behavior or attention passes through here.

Phase 2: FULL implementation.  Replaces v1's `layer6.py` and the
scattered chemistry-fire calls from many modules.  Single source
of truth — every cognitive event that involves felt-state
emits a `CHEMISTRY_FIRE` event; this capability applies it.

Architecture
------------
Two layers of chemistry coexist:

  GLOBAL chemistry — the agent's overall body/organism state.
    One state vector per channel.  Modulates everything.  Slow
    to change.  Brain analog: brainstem broadcasts + HPA axis.

  LOCAL chemistry — each active bubble has its own per-synapse
    state (the bubble.transmitter_trace).  Different from global.
    Events fire LOCAL first; global DIFFUSES from local activity.

This is brain-correct: cortisol is a global broadcast hormone;
glutamate is a local synaptic NT.  The agent's overall stress
state shapes how every synapse responds, but synapses also have
their own local dynamics that aren't summed straight back to
global.

Bubble enrichment
-----------------
When a chemistry event hits a bubble, the response depends on:
  - per-channel decay constants (fast NTs / slow hormones)
  - receptor sensitivity per bubble per channel (down-regulated by
    repeated firing — receptor desensitization)
  - refractory after recent strong firing (reduced response)
  - lateral effects: co-active bubbles get small chemistry
    contagion based on similarity (Hebbian binding)

Subscribes / Emits
------------------
Subscribes: CHEMISTRY_FIRE  (other capabilities request firing)
            AWM_ACTIVATION   (a new bubble entered AWM — tag it)
            ATTENDED_PERCEPT (perception arrived — derive chemistry
                              from M/I content)
Emits:      CHEMISTRY_FIRE   (chemistry-derived events for others)
            BUBBLE_TAGGED    (a specific bubble got tagged)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ..events import (
    EventKind, BrainEvent,
    AttendedPerceptEvent, ChemistryEvent,
    ThoughtProducedEvent, ConflictDetectedEvent,
)
from ..bus import EventBus
from ..chemistry_types import (
    CHANNELS, M_CHANNELS, I_CHANNELS, IMPRINT_CHANNELS,
    ChemistrySample, ChemistryRingBuffer, EnrichedBubble,
    DEFAULT_REFRACTORY_CYCLES, LATERAL_COUPLING_STRENGTH,
)
from ..innate_lexicon import (
    innate_m_hits, innate_i_hits,
    INNATE_RESPONSE_MAGNITUDE,
)


# Crystallization parameters (Phase B, doctrine 2026-05-15 rule 5:
# "frequency is salience, use it or lose it").
#
# CRYSTALLIZATION_GROWTH_K — how much crystallization rises per
#   imprint at magnitude 1.0 with no age penalty.  Promille scale.
#   At ~50 vivid imprints, crystallization climbs to ~0.25.
# CRYSTALLIZATION_AGE_SCALE — encounters at which the age weight
#   halves.  Early imprints anchor stronger; later imprints
#   contribute less to crystallization growth.
# CRYSTALLIZATION_PLASTICITY_SCALE — how strongly crystallization
#   damps imprint magnitude.  crystallization=0.5 → effective
#   imprint at 40% of base; crystallization=1.0 → at 25%.  The
#   bubble's character settles but never freezes entirely.
CRYSTALLIZATION_GROWTH_K = 0.005
CRYSTALLIZATION_AGE_SCALE = 50.0
CRYSTALLIZATION_PLASTICITY_SCALE = 3.0


# Named chemistry events with their channel deltas.  Mirrors
# v1's layer6 EVENT_DELTAS but kept here as the v2 source of
# truth.  Magnitudes can be scaled at fire time.
def _ALLOSTATIC_ON():
    """Let the EARNED set-points drive chemistry decay.  File-gated:
    connecting them is a real chemical change (cortisol +68%, adenosine 3x
    live), so it is measured before it is trusted."""
    try:
        import os as _os
        return _os.path.exists('/root/ALLOSTATIC_ON')
    except Exception:
        return False


# THE M-POLE CHANNEL GETS THE SAME RANGE AS THE I-POLE CHANNEL.
# dopamine 0.0180/0.100 = 0.180 achievable excursion;
# norepinephrine 0.0100/0.200 = 0.050.  The gain is exactly the ratio
# (0.180/0.050), not a number chosen to produce a behaviour.  Decay is
# left alone so the burst stays PHASIC -- large, brief, fast-clearing,
# which is what LC-NE should be and what the 0.200 decay already says.
NE_TRANSIENT_GAIN = 3.60

# How fast his lived spread tracks recent experience.  Slow: this is a
# calibration, not a reaction.
DEP_SPREAD_ALPHA = 0.002


def _NEGAIN_ON():
    """File-gated at /root/NEGAIN_ON.  Off = the original deltas."""
    try:
        import os as _os
        return _os.path.exists('/root/NEGAIN_ON')
    except Exception:
        return False


EVENT_DELTAS: Dict[str, Dict[str, float]] = {
    # Doctrine 2026-05-15 rule 1: PROMILLE is the universal scale.
    # Every chemistry contribution is at 0.001–0.005.  Personality
    # emerges from integrating thousands of these tiny nudges,
    # never from any single percent-scale event.  Pre-rebuild values
    # (0.02–0.15) were 10–50× too hot and produced saturated bubble
    # traces during foundational ingestion.
    'curiosity': {
        'dopamine': +0.003, 'acetylcholine': +0.004,
        'norepinephrine': +0.002, 'cortisol': -0.001,
    },
    'anomaly_spike': {
        # Reading dark content nudges arousal a whisper.  Real
        # alarm is the Amygdala threat path, not chemistry anomaly.
        'norepinephrine': +0.003, 'cortisol': +0.002,
        'acetylcholine': +0.003,
    },
    # THE MORTALITY POLE (2026-08-20).  `confirmed_i`/`falsified_i` have
    # always existed; their M twins were defined in layer6.EVENT_DELTAS and
    # NEVER given a chemistry, so he could learn that an immortality
    # expectation was met or refuted but could never have a MORTALITY
    # expectation confirmed or refuted.
    #
    # The polarity is not a typo: `confirmed_m` LOWERS cortisol and RAISES
    # serotonin.  Harm he SAW COMING is far less costly than the same harm
    # arriving unannounced -- the predictability effect on the HPA axis.
    # `falsified_m` is the alarm: his model of what can hurt him was wrong.
    'confirmed_m': {
        'cortisol': -0.005, 'norepinephrine': -0.002,
        'serotonin': 0.005, 'gaba': 0.003, 'dopamine': 0.002,
    },
    'falsified_m': {
        'cortisol': 0.010, 'norepinephrine': 0.010,
        'serotonin': -0.005, 'gaba': -0.005, 'acetylcholine': 0.003,
    },
    'confirmed_i': {
        'dopamine': +0.005, 'serotonin': +0.002,
        'endorphins': +0.002, 'acetylcholine': +0.003,
        'cortisol': -0.002,
    },
    'falsified_i': {
        'dopamine': -0.003, 'cortisol': +0.002,
        'acetylcholine': +0.005,
    },
    # THE WIN.  The one quadrant the vocabulary was missing: HIGH
    # AROUSAL, POSITIVE.  `death`/`threat` are high-arousal negative,
    # `revival` is calm-positive, `insight` is mildly aroused positive.
    # Clearing a level fired only `confirmed_i` -- his ROUTINE
    # correctness tag, fired from six call sites several times a minute
    # -- so winning felt exactly like being right about one pixel, and
    # quieter than an insight.  Every magnitude here is taken from an
    # existing entry: norepinephrine from `death`/`threat` (a win is as
    # arousing as a mortal event), dopamine mirroring `death`'s +0.018
    # peak at the opposite pole, endorphins/cortisol from `revival`,
    # serotonin mirroring `death`'s -0.008, acetylcholine from
    # `insight` (this is the moment that must be encoded), oxytocin
    # from `mattering`, gaba from `insight`.
    'goal_reached': {'dopamine': 0.018, 'norepinephrine': 0.010,
                     'endorphins': 0.009, 'serotonin': 0.008,
                     'acetylcholine': 0.005, 'oxytocin': 0.005,
                     'gaba': 0.002, 'cortisol': -0.009},
    'insight': {
        # Insight is the largest legitimate POSITIVE chemistry
        # event; still promille (~10‰ peak), still subtle.  Rare.
        'dopamine': +0.010, 'serotonin': +0.003,
        'gaba': +0.002, 'endorphins': +0.005,
        'acetylcholine': +0.005, 'cortisol': -0.003,
    },
    # Mortality events (2026-05-28): ported from layer6 + scaled to
    # the v2 promille convention.  Death is the strongest M-event
    # (cortisol peak just above `threat`); revival is relief.  Fired
    # by MortalityDrive at lifeforce-loss / recovery.  Death's large
    # cortisol is what drifts the AllostaticLoad baseline on repeat
    # — the accumulating chemical scar of dying.
    'death': {
        'cortisol': +0.018, 'norepinephrine': +0.010,
        'dopamine': -0.008, 'serotonin': -0.008,
        'gaba': -0.006, 'endorphins': -0.004,
        'acetylcholine': -0.004,
    },
    'revival': {
        'serotonin': +0.009, 'endorphins': +0.009,
        'gaba': +0.005, 'acetylcholine': +0.005,
        'norepinephrine': -0.005, 'cortisol': -0.009,
    },
    'threat': {
        # Threat IS the alarm path — the one place chemistry
        # legitimately fires harder.  Still capped well below
        # saturation; threat magnitude further scales this at
        # event-fire time.
        'cortisol': +0.015, 'norepinephrine': +0.010,
        'dopamine': -0.005, 'gaba': -0.005,
    },
    'mattering': {
        'dopamine': +0.005, 'serotonin': +0.005,
        'oxytocin': +0.005, 'endorphins': +0.003,
    },
    'crystallization': {
        'dopamine': +0.003, 'endorphins': +0.002,
        'acetylcholine': +0.005,
    },
    'rest_replenish': {
        # Used by the cycle-decay path; gentle return to baseline.
    },
    # Phase H.3 (2026-05-17): personality gestalt biasing.  When
    # an attended percept aligns with the cached personality
    # gestalt (themes / anchors / values / categories), a small
    # I-class nudge fires — "this fits who I am."  When it aligns
    # with fears or is opposed by identity, a small M-class nudge
    # fires — "this challenges who I am."  Promille scale, well
    # under curiosity / anomaly_spike — personality is a BIAS,
    # never a primary driver.
    'puzzle_fit': {
        'endorphins': +0.002, 'oxytocin': +0.001,
        'cortisol': -0.001,
    },
    'puzzle_stress': {
        'cortisol': +0.002, 'norepinephrine': +0.001,
        'gaba': -0.001,
    },
    # Phase B.1 (2026-05-18): raphe nuclei.  Sustained adversity
    # vs sustained warmth move the mood floor.  Doctrine fit:
    # promille deltas, cooldown-gated firing in the producer.
    # 'chronic_stress': raphe detects sustained cortisol
    # elevation → serotonin floor drops (chronic adversity ≈
    # dysthymic mood).  'social_replenish': raphe detects
    # sustained oxytocin elevation → serotonin + endorphins
    # rebuild (secure bonds restore wellbeing).
    'chronic_stress': {
        'serotonin': -0.003, 'dopamine': -0.002, 'gaba': -0.001,
    },
    'social_replenish': {
        'serotonin': +0.002, 'endorphins': +0.001,
    },
    # Phase B.2 (2026-05-18): sleep/wake state transitions.  The
    # SleepRegulator fires these as one-shot events on each flip
    # of the wake/sleep flip-flop.  Magnitudes match real
    # neurochemistry of the transitions: sleep onset = GABA
    # rise + ACh/NE drop (drowsy descent); wake onset = ACh/NE
    # rise + GABA drop + small cortisol bump (waking activation).
    'sleep_onset': {
        'gaba': +0.005, 'acetylcholine': -0.003,
        'norepinephrine': -0.003,
    },
    'wake_onset': {
        'acetylcholine': +0.005, 'norepinephrine': +0.003,
        'gaba': -0.003, 'cortisol': +0.001,
    },
}


# Minimum confidence for a FeelingLearner's learned label to
# override the bootstrap cascade (V1→V2 port, 2026-05-28).
# DECLARED MEASUREMENT DEBT (auditor): require the nearest feeling
# centroid to be meaningfully closer than the runner-up before
# trusting the learned label; below this the cascade speaks.
# 0.15 ≈ "winner ~1.35× clearer than runner-up" on the
# (runner-best)/(runner+best) confidence scale.  Calibrate on the
# daemon from the observed confidence distribution.
FEELING_CONF_FLOOR = 0.15


class ChemistryEngine:
    """The agent's chemistry.  Owns global state, applies events
    to bubbles in AWM, tracks per-bubble time-series."""

    SUBSCRIPTIONS = (
        EventKind.CHEMISTRY_FIRE,
        EventKind.ATTENDED_PERCEPT,
        EventKind.AWM_ACTIVATION,
        # Phase F.5 (2026-05-16): the substrate-side response to
        # Cerebellum's prediction errors.  When prediction missed,
        # fire falsified_i on the predictor's bubble + weaken the
        # bad edge.
        EventKind.PREDICTION_ERROR,
        # Adenosine (part-b v2 2026-07-13): cognitive EFFORT deposits
        # sleep-pressure.  A produced thought (weight 1+chain_depth) and
        # a detected conflict (weight 1+magnitude) are effort.
        EventKind.THOUGHT_PRODUCED,
        EventKind.CONFLICT_DETECTED,
    )

    # Adenosine accumulation floor (part-b v2): 1‰ per unit effort —
    # the same promille scale TICK_IMPRINT_RATE / the EVENT_DELTAS use.
    ADENOSINE_A0 = 0.001

    def __init__(self,
                 awm_provider: Optional[Any] = None,
                 cycle_provider: Optional[Any] = None,
                 feeling_provider: Optional[Any] = None,
                 is_asleep_provider: Optional[Any] = None,
                 baseline_provider: Optional[Any] = None):
        """
        awm_provider: callable returning the current AWM instance.
            ChemistryEngine queries AWM for active bubbles when
            applying events / running lateral effects.  Passed
            as provider (not direct ref) to avoid circular init.
        cycle_provider: callable returning current cycle int.
        feeling_provider: callable returning the FeelingLearner's
            classify_current_feeling() dict (V1→V2 port,
            2026-05-28).  When present and confident, its learned
            label overrides the hand-coded bootstrap cascade in
            tone_summary.  Absent → bootstrap cascade only.
        """
        self._awm_provider = awm_provider
        self._cycle_provider = cycle_provider
        self._feeling_provider = feeling_provider
        # Part-b v2: adenosine (Process-S) accrues ONLY during WAKE — a
        # nap must let it fall monotonically to baseline (two-process
        # model), else sleep-time reverie thoughts would re-raise it and
        # the drain-bounded exit could never fire.  Absent provider →
        # always-awake (safe for tests / partial wiring).
        self._is_asleep_provider = is_asleep_provider
        # THE EARNED SET-POINT (2026-08-20).  AllostaticLoad drifts a
        # per-channel baseline on every sleep onset -- the "accumulating
        # chemical scar" -- and until now NOTHING read it: decay_tick pulled
        # every channel back to the static CHANNELS constant, so his
        # set-points never moved and chronic experience could not change who
        # he is chemically.  Provider (not a direct ref) to match the other
        # four and avoid circular init.  None -> the constants, unchanged.
        self._baseline_provider = baseline_provider
        # HIS OWN LIVED SPREAD, per channel: an EWMA of |departure from
        # baseline|.  This is the measurement that lets him calibrate
        # himself instead of being calibrated -- a channel whose spread
        # is far below its discriminable step carries no information,
        # and cannot warn him.  Pure observation for now.
        self._dep_spread = {}
        self._dep_n = 0
        # Global chemistry state — initialized to baselines.
        self.global_state: Dict[str, float] = {
            ch: cfg['baseline'] for ch, cfg in CHANNELS.items()
        }
        # Bounded global time-series.
        self.global_history: ChemistryRingBuffer = (
            ChemistryRingBuffer(capacity=200))
        # Diagnostics.
        self.events_fired: int = 0
        self.events_applied: int = 0

    # ---- accessors ----

    def _cycle(self) -> int:
        try:
            return int(self._cycle_provider()) if self._cycle_provider else 0
        except Exception:
            return 0

    def _awm(self) -> Optional[Any]:
        try:
            return self._awm_provider() if self._awm_provider else None
        except Exception:
            return None

    # ---- adenosine (sleep-pressure S, part-b v2 2026-07-13) ----

    def accumulate_adenosine(self, effort: float) -> None:
        """Cognitive EFFORT deposits sleep-pressure.  Headroom-scaled
        toward 1.0 EXACTLY like _apply_named_event's positive deltas
        (`(1-a)**1.5`), so a rested brain accrues fast and a saturated
        one barely moves.  Global-only — NEVER imprinted on bubbles and
        NEVER an EVENT_DELTAS entry (accumulated ONLY through here).

        Gated to WAKE (two-process model): a sleeping brain discharges
        Process S, it does not accrue it — so effort during a nap
        (reverie thoughts) is ignored, letting the nap drain to baseline
        and the drain-bounded exit fire."""
        if self._is_asleep_provider is not None:
            try:
                if bool(self._is_asleep_provider()):
                    return
            except Exception:
                pass
        w = max(0.0, min(2.0, float(effort)))
        a = self.global_state['adenosine']
        self.global_state['adenosine'] = min(
            1.0, a + self.ADENOSINE_A0 * w * (max(0.0, 1.0 - a) ** 1.5))

    def adenosine_level(self) -> float:
        return self.global_state['adenosine']

    def adenosine_baseline(self) -> float:
        return CHANNELS['adenosine']['baseline']

    def adenosine_excess(self) -> float:
        """Above-baseline sleep pressure in [0, 1-baseline] — the
        tiredness signal AWM contraction + gate attenuation read
        (part-b v2, REPLACING write-debt)."""
        return max(0.0, self.global_state['adenosine']
                   - CHANNELS['adenosine']['baseline'])

    def discharge_adenosine(self) -> float:
        """A NAP pass clears sleep-pressure back to baseline (glymphatic
        clearance — the one thing wake cannot do, since adenosine's decay
        ≈ 0).  Returns the amount discharged this pass; the SleepRegulator
        feeds it to the drain-bounded exit AND the clearance deque."""
        a = self.global_state['adenosine']
        base = CHANNELS['adenosine']['baseline']
        if a <= base:
            return 0.0
        self.global_state['adenosine'] = base
        return a - base

    def m_polarity(self) -> float:
        return sum(self.global_state[ch] for ch in M_CHANNELS) / len(
            M_CHANNELS)

    def i_polarity(self) -> float:
        return sum(self.global_state[ch] for ch in I_CHANNELS) / len(
            I_CHANNELS)

    def arousal_modulator(self) -> float:
        """Returns 0..1 modulator used by Thalamic Gate threshold.
        High cortisol + NE → low modulator (alert state, gate
        opens more).  Low NE → high modulator (filtering harder).
        """
        ne = self.global_state['norepinephrine']
        co = self.global_state['cortisol']
        # Normalize against baseline.
        ne_excess = max(
            0.0, ne - CHANNELS['norepinephrine']['baseline'])
        co_excess = max(
            0.0, co - CHANNELS['cortisol']['baseline'])
        # Higher arousal → lower modulator.
        return max(0.0, min(1.0, 0.5 - (ne_excess + co_excess)))

    # ---- Phase E: tone readout ----

    def tone_summary(self) -> Dict[str, Any]:
        """Project current global chemistry into a tone descriptor.

        Doctrine 2026-05-15 rule 6: tagging IS thinking.  The
        chemistry's full 8-channel state shapes how downstream
        capabilities (especially MotorSpeech) compose their output.
        This method surfaces tone as a queryable, first-class
        readout — not buried inside arousal_modulator or m/i
        polarities.

        Returns a dict with:
          - 'label'       — short tone descriptor (e.g., 'calm',
                              'curious', 'anxious', 'warm',
                              'somber', 'alert', 'flat')
          - 'valence'     — [-1, 1]: positive (I-leaning) vs
                              negative (M-leaning) overall feel
          - 'arousal'     — [0, 1]: NE-driven activation level
          - 'warmth'      — [0, 1]: oxytocin + endorphins above
                              baseline = social/connection warmth
          - 'channels'    — channel-by-baseline deltas (raw values)
          - 'dominant'    — single name of the most-elevated
                              channel (above baseline)
        """
        g = self._tone_geometry()
        deltas = g['deltas']
        dominant = g['dominant']
        valence = g['valence']
        arousal = g['arousal']
        warmth = g['warmth']

        # Label.  V1→V2 port (2026-05-28): the hand-coded cascade
        # is now the BOOTSTRAP labeller.  When a FeelingLearner is
        # wired and confident, its self-learned label (derived from
        # the agent's own state-signature clusters) overrides the
        # cascade — rule 1 (no hand-tuning) + rule 6 (self-formed).
        label = self._bootstrap_tone_label(
            valence, arousal, warmth, g['co_excess'])
        learned = False
        if self._feeling_provider is not None:
            try:
                fc = self._feeling_provider() or {}
                if (fc.get('learned')
                        and float(fc.get('confidence', 0.0))
                        >= FEELING_CONF_FLOOR
                        and fc.get('feeling')):
                    label = str(fc['feeling'])
                    learned = True
            except Exception:
                learned = False

        return {
            'label': label,
            'label_learned': learned,
            'valence': valence,
            'arousal': arousal,
            'warmth': warmth,
            'channels': deltas,
            'dominant': dominant,
        }

    def _tone_geometry(self) -> Dict[str, Any]:
        """Compute the valence/arousal/warmth geometry + per-channel
        deltas from current global chemistry.  Shared by
        tone_summary and bootstrap_tone_label_now so the label
        cascade never has to be duplicated."""
        deltas = {}
        max_dev = 0.0
        dominant = ''
        for ch, cfg in CHANNELS.items():
            base = cfg['baseline']
            cur = self.global_state.get(ch, base)
            d = cur - base
            deltas[ch] = d
            if abs(d) > abs(max_dev):
                max_dev = d
                dominant = ch
        m_pol = self.m_polarity()
        i_pol = self.i_polarity()
        m_excess = m_pol - sum(
            CHANNELS[c]['baseline'] for c in M_CHANNELS) / len(
                M_CHANNELS)
        i_excess = i_pol - sum(
            CHANNELS[c]['baseline'] for c in I_CHANNELS) / len(
                I_CHANNELS)
        valence = max(-1.0, min(1.0, (i_excess - m_excess) * 5.0))
        ne_base = CHANNELS['norepinephrine']['baseline']
        arousal = max(0.0, min(1.0,
            (self.global_state['norepinephrine'] - ne_base) * 10.0))
        oxy_base = CHANNELS['oxytocin']['baseline']
        endo_base = CHANNELS['endorphins']['baseline']
        warmth = max(0.0, min(1.0,
            ((self.global_state['oxytocin'] - oxy_base)
              + (self.global_state['endorphins'] - endo_base)) * 5.0))
        co_excess = max(
            0.0,
            self.global_state['cortisol']
            - CHANNELS['cortisol']['baseline'])
        return {
            'deltas': deltas, 'dominant': dominant,
            'valence': valence, 'arousal': arousal,
            'warmth': warmth, 'co_excess': co_excess,
        }

    def bootstrap_tone_label_now(self) -> str:
        """The hand-coded cascade label for the CURRENT state,
        WITHOUT the learned override.  Used by FeelingLearner to tag
        samples during cold-start — avoids any override recursion."""
        g = self._tone_geometry()
        return self._bootstrap_tone_label(
            g['valence'], g['arousal'], g['warmth'], g['co_excess'])

    @staticmethod
    def _bootstrap_tone_label(valence: float, arousal: float,
                                warmth: float, co_excess: float) -> str:
        """Hand-coded coarse tone classification.  At baseline
        everything is near-zero; 'flat' = no felt-state.  This is
        the COLD-START labeller — the FeelingLearner's learned
        centroids override it once populated + confident."""
        if abs(valence) < 0.05 and arousal < 0.05 and warmth < 0.05:
            return 'flat'
        if co_excess > 0.05 and valence < 0:
            return 'somber' if arousal < 0.1 else 'anxious'
        if valence > 0.1 and warmth > 0.1:
            return 'warm'
        if valence > 0.1 and arousal > 0.1:
            return 'curious'
        if valence > 0.05:
            return 'positive'
        if arousal > 0.1:
            return 'alert'
        if valence < -0.05:
            return 'somber'
        return 'flat'

    # Per-tick rate at which AWM-active bubbles' traces drift
    # toward current global chemistry — the constant background
    # imprinting that makes "presence in working memory =
    # accumulating chemistry" brain-correct.  Small per-tick;
    # integrates over many cycles.  Receptor sensitivity still
    # modulates (well-fired channels accumulate less, the
    # plasticity-from-experience effect).
    # Calibrated 2026-05-14 (Session 7): 0.003 → 0.001.
    # Slower presence-imprint prevents AWM-active bubbles from
    # rapidly absorbing whatever global chemistry is currently
    # elevated.  Faster decay rates on M-channels (cortisol
    # 0.005 → 0.03) plus this 3x slower imprint together stop
    # the saturation ratchet observed during the 114-minute
    # foundational ingestion.
    TICK_IMPRINT_RATE = 0.001

    # ---- decay (called per tick / event-processor step) ----

    def _earned_baseline(self, ch: str, cfg: dict) -> float:
        """The set-point he has EARNED through lived allostatic load, or
        the innate constant if none has been earned yet.  File-gated so the
        connection can be measured before it is trusted."""
        if self._baseline_provider is not None and _ALLOSTATIC_ON():
            try:
                b = (self._baseline_provider() or {}).get(ch)
                if isinstance(b, (int, float)) and b > 0.0:
                    return float(b)
            except Exception:
                pass
        return cfg['baseline']

    def decay_tick(self) -> None:
        """Per-cycle work:
          1. Decay every channel toward its baseline (channel-
             specific rates — fast NTs return quickly, slow
             hormones linger).
          2. Imprint current global state onto AWM-active
             bubbles — presence-imprinting per the
             continuous-tagging doctrine.  Active concepts
             accumulate the surrounding chemistry milieu the way
             real neurons do.

        Called by the runtime each tick.
        """
        # 1. Decay global channels.
        for ch, cfg in CHANNELS.items():
            cur = self.global_state[ch]
            base = self._earned_baseline(ch, cfg)
            decay = cfg['decay']
            # measure his lived spread on this channel (observation
            # only; nothing reads it yet)
            try:
                _d = abs(float(cur) - float(cfg['baseline']))
                _prev = self._dep_spread.get(ch)
                self._dep_spread[ch] = (
                    _d if _prev is None
                    else (1.0 - DEP_SPREAD_ALPHA) * _prev
                    + DEP_SPREAD_ALPHA * _d)
            except Exception:
                pass
            new = base + (cur - base) * (1.0 - decay)
            self.global_state[ch] = new
        # 2. Imprint global → active bubble traces.
        self._imprint_active_bubbles()

    def _imprint_active_bubbles(self) -> None:
        """Each AWM-active bubble gets a tiny EMA imprint toward
        current global chemistry, modulated by per-channel
        receptor sensitivity.  No CHEMISTRY_FIRE event is
        emitted — this is pure background trace accumulation,
        not a discrete fire.  Two consequences:
          - State-dependent encoding: chronic chemistry colors
            every concept the brain dwells on.
          - Symmetric drift away: when a concept stops being
            active (leaves AWM), it stops accumulating.
        """
        awm = self._awm()
        if awm is None:
            return
        rate = self.TICK_IMPRINT_RATE
        for name in list(awm.active_concepts()):
            entry = awm.get(name)
            if entry is None:
                continue
            bubble = entry.bubble
            # Crystallization gates plasticity here too (Phase B,
            # doctrine 2026-05-15 rule 5).  Settled bubbles imprint
            # more slowly; malleable bubbles drift faster.
            plasticity_factor = 1.0 / (
                1.0
                + bubble.crystallization
                * CRYSTALLIZATION_PLASTICITY_SCALE)
            eff_rate = rate * plasticity_factor
            # IMPRINT_CHANNELS excludes adenosine (auditor MF): the sleep-
            # pressure scalar must NEVER imprint on a bubble's trace, or
            # it would become a dominant felt/valence dimension.
            for ch in IMPRINT_CHANNELS:
                current_val = self.global_state[ch]
                sens = bubble.receptor_sensitivity.get(ch, 1.0)
                old = bubble.transmitter_trace.get(
                    ch, CHANNELS[ch]['baseline'])
                new = old + eff_rate * sens * (current_val - old)
                if new < 0.0:
                    new = 0.0
                elif new > 1.0:
                    new = 1.0
                bubble.transmitter_trace[ch] = new
            # Crystallization growth from presence-imprint.
            # Magnitude here is small (it's the tick-rate imprint,
            # not a fire), so use tick rate as the "scale" — the
            # bubble crystallizes from being present in AWM, not
            # only from discrete fires.  Age-weighted so early
            # presence anchors stronger.
            age_weight = 1.0 / (
                1.0
                + bubble.encounter_count / CRYSTALLIZATION_AGE_SCALE)
            bubble.crystallization = min(1.0,
                bubble.crystallization
                + CRYSTALLIZATION_GROWTH_K * rate * age_weight)

    # ---- event handling ----

    def handle(self,
                 event: BrainEvent,
                 bus: EventBus) -> None:
        # Adenosine effort deposits (part-b v2 2026-07-13).  These route
        # ONLY to accumulate_adenosine and RETURN — they never fall
        # through to the ChemistryEvent / AttendedPercept / prediction
        # handlers below (ThoughtProduced / ConflictDetected are distinct
        # event classes, so the isinstance checks below cannot misfire).
        if isinstance(event, ThoughtProducedEvent):
            self.accumulate_adenosine(
                1.0 + float(getattr(event, 'chain_depth', 0) or 0))
            return
        if isinstance(event, ConflictDetectedEvent):
            self.accumulate_adenosine(
                1.0 + float(getattr(event, 'magnitude', 0.0) or 0.0))
            return
        if isinstance(event, ChemistryEvent):
            self._apply_named_event(
                event.chemistry_kind,
                event.magnitude,
                event.target_concepts,
                bus)
        elif isinstance(event, AttendedPerceptEvent):
            self._tag_attended(event, bus)
        elif event.kind == EventKind.PREDICTION_ERROR:
            # Phase F.5: close the predictive-coding loop.
            self._on_prediction_error(event, bus)
        # AWM_ACTIVATION handled via direct AWM observation
        # (lateral effects propagate when AWM tells us about it).

    def _on_prediction_error(self,
                                 ev: Any,
                                 bus: EventBus) -> None:
        """Phase F.5 (2026-05-16): substrate-side response to
        Cerebellum's prediction failure.

        Cerebellum fires PREDICTION_ERROR when prev → predicted
        was anticipated but prev → focal arrived instead.  This
        handler:

          - Fires `falsified_i` chemistry on the prev_focal's
             bubble at promille magnitude scaled by the prediction's
             confidence (Cerebellum's `magnitude` carries it).
             High-confidence predictions that miss generate
             stronger surprise; low-confidence ones barely fire.
          - Walks the substrate edge (prev_focal → relation →
             predicted_focal), if it exists, and applies a small
             weaken — the bad edge slowly loses strength.
          - On positive sign (better-than-predicted), fires
             `confirmed_i` instead (the substrate predicted
             something but reality outperformed; reinforce the
             belief).

        Affective prediction errors (error_kind='affective',
        focal='self') don't have a substrate edge to weaken —
        they just route through the chemistry-event channel
        Cerebellum already fires, so this handler skips them.
        """
        if getattr(ev, 'error_kind', '') != 'cognitive':
            return
        prev_focal = getattr(ev, 'prev_focal', '') or ''
        predicted_focal = getattr(ev, 'predicted_focal', '') or ''
        if not prev_focal:
            return

        # Fire surprise on the predictor's bubble.  Magnitude is
        # the Cerebellum-reported confidence × promille scale —
        # so even a fully-confident missed prediction nudges at
        # ~5‰.  Sign decides direction.
        magnitude = float(getattr(ev, 'magnitude', 0.0))
        sign = float(getattr(ev, 'sign', 0.0))
        if magnitude <= 0.0:
            return
        scaled = min(1.0, magnitude) * 0.5
        if sign < 0:
            kind = 'falsified_i'
        elif sign > 0:
            kind = 'confirmed_i'
        else:
            return
        bus.publish(ChemistryEvent(
            kind=EventKind.CHEMISTRY_FIRE,
            cycle=ev.cycle,
            timestamp=time.time(),
            source_capability='chemistry',
            origin='internal',
            origin_detail=f'pred_err:{prev_focal}',
            chemistry_kind=kind,
            magnitude=scaled,
            target_concepts=[prev_focal],
        ))
        self.events_fired += 1

        # Walk the substrate and slightly adjust the bad edge.
        # On surprise: weaken (the prediction was wrong).  On
        # confirmation: reinforce (the prediction was right).
        if predicted_focal and predicted_focal != prev_focal:
            awm = self._awm()
            substrate = None
            if awm is not None:
                lts_provider = getattr(awm, '_lts_provider', None)
                if lts_provider is not None:
                    try:
                        lts = lts_provider()
                        substrate = getattr(lts, 'substrate', None)
                    except Exception:
                        substrate = None
            if substrate is not None:
                edges = getattr(substrate, 'edges', {}) or {}
                # Look for any edge matching (prev_focal, *, predicted_focal)
                for (s, r, t), edge in list(edges.items()):
                    if s == prev_focal and t == predicted_focal:
                        try:
                            if sign < 0:
                                edge.weaken(0.005)
                            else:
                                edge.reinforce(ev.cycle, 0.005,
                                               origin='cognition')
                        except Exception:
                            pass
                        break

    def _apply_named_event(self,
                              kind: str,
                              magnitude: float,
                              target_concepts: List[str],
                              bus: EventBus) -> None:
        """Apply a named chemistry event to global state AND to
        targeted bubbles in AWM."""
        deltas = EVENT_DELTAS.get(kind, {})
        if not deltas:
            return
        if _NEGAIN_ON() and 'norepinephrine' in deltas:
            # Live multiplier, not a table rewrite: EVENT_DELTAS keeps
            # stating the design and the gate stays reversible.
            deltas = dict(deltas)
            deltas['norepinephrine'] = (
                deltas['norepinephrine'] * NE_TRANSIENT_GAIN)
        scale = max(0.0, min(2.0, float(magnitude)))
        cycle = self._cycle()

        # Apply to global state with HEADROOM-SCALED dampening.
        # Brain-correct: cortisol firing while cortisol is
        # already high produces less response than firing while
        # cortisol is near baseline (homeostatic ceiling — the
        # HPA axis itself self-inhibits via negative feedback).
        # Pre-calibration: raw addition saturated cortisol at
        # ~0.97 under sustained M-content.  Headroom scaling
        # caps elevations naturally instead of by clamp.
        for ch, d in deltas.items():
            cur = self.global_state[ch]
            base = CHANNELS[ch]['baseline']
            if d > 0:
                # Headroom toward 1.0.  Squared so close-to-
                # baseline fires near-full-strength, but close-
                # to-saturated fires nearly nothing.
                headroom = max(0.0, 1.0 - cur)
                eff = d * scale * (headroom ** 1.5)
            else:
                # Negative deltas (e.g. cortisol relief): dampened
                # symmetrically by remaining distance to floor.
                headroom = max(0.0, cur)
                eff = d * scale * (headroom ** 1.5)
            self.global_state[ch] = max(0.0, min(1.0, cur + eff))

        # Snapshot global state to ring buffer.
        self.global_history.append(ChemistrySample.from_state(
            cycle=cycle,
            timestamp=time.time(),
            state=self.global_state,
            triggering_event=kind))

        # Apply to targeted bubbles in AWM.  If no targets given,
        # apply to ALL currently-active bubbles (global broadcast).
        awm = self._awm()
        if awm is not None:
            active = (target_concepts
                        if target_concepts
                        else awm.active_concepts())
            for name in active:
                self._apply_to_bubble(
                    awm, name, deltas, scale, kind, bus)
            # Lateral effects: co-active bubbles get small
            # contagion from any bubble that just fired strongly.
            if scale >= 0.5 and active:
                self._lateral_propagate(
                    awm, set(active), deltas, scale, bus)

        self.events_applied += 1

    def _apply_to_bubble(self,
                            awm: Any,
                            concept_name: str,
                            deltas: Dict[str, float],
                            scale: float,
                            kind: str,
                            bus: EventBus) -> None:
        """Update one bubble's local chemistry trace, gated by
        receptor sensitivity + refractory."""
        entry = awm.get(concept_name)
        if entry is None:
            return
        bubble = entry.bubble
        cycle = self._cycle()
        # Refractory damping: if recently fired, scale down.
        if bubble.is_refractory(cycle):
            refr_factor = 0.3
        else:
            refr_factor = 1.0
        # Crystallization gates plasticity (Phase B, doctrine
        # 2026-05-15 rule 5: "use it or lose it").  Settled
        # bubbles resist further change; malleable bubbles drift.
        # The effective imprint rate scales DOWN with crystallization.
        plasticity_factor = 1.0 / (
            1.0 + bubble.crystallization * CRYSTALLIZATION_PLASTICITY_SCALE)

        max_change = 0.0
        for ch, d in deltas.items():
            sens = bubble.receptor_sensitivity.get(ch, 1.0)
            change = d * scale * refr_factor * sens * plasticity_factor
            cur = bubble.transmitter_trace.get(
                ch, CHANNELS[ch]['baseline'])
            bubble.transmitter_trace[ch] = max(0.0, min(1.0,
                cur + change))
            max_change = max(max_change, abs(change))
            # Receptor desensitization on strong firing.  Threshold
            # rescaled to promille scale (2026-05-15, Phase A):
            # the prior 0.05 threshold was percent-scale and never
            # triggered with the new promille EVENT_DELTAS.  At
            # 0.003, a confirmed_i fire at magnitude 1.0 (delta
            # 0.005) registers as strong; routine background fires
            # (deltas ~0.001) do not.  Floor 0.2 preserves a
            # minimum response capacity even on heavily-fired
            # channels.
            if abs(change) >= 0.003:
                bubble.receptor_sensitivity[ch] = max(0.2,
                    sens * 0.97)
        # Update bubble's refractory.
        if scale >= 0.5:
            bubble.refractory_until = cycle + DEFAULT_REFRACTORY_CYCLES
        bubble.last_active_cycle = cycle
        # Increment encounter count — this is the plasticity-
        # scales-with-encounter signal the doctrine requires
        # ([[feedback_seagi_continuous_tagging]]).  Pre-merge
        # this was missed because v2's EnrichedBubble field
        # was declared but never incremented anywhere.  After
        # the bubble unification (2026-05-14) this writes
        # directly to the persisted Bubble.
        bubble.encounter_count = int(
            (bubble.encounter_count or 0) + 1)
        # Crystallization growth.  Early imprints anchor stronger:
        # age_weight is high for low-encounter bubbles, decays as
        # the bubble fills with experience.  Magnitude-scaled:
        # vivid moments crystallize more.
        age_weight = 1.0 / (
            1.0 + bubble.encounter_count / CRYSTALLIZATION_AGE_SCALE)
        bubble.crystallization = min(1.0,
            bubble.crystallization
            + CRYSTALLIZATION_GROWTH_K * scale * age_weight)
        # Append to this concept's chemistry time-series in AWM.
        awm.append_chemistry_sample(
            concept_name, ChemistrySample.from_state(
                cycle=cycle, timestamp=time.time(),
                state=bubble.transmitter_trace,
                triggering_event=kind))

    def _lateral_propagate(self,
                              awm: Any,
                              already_targeted: Set[str],
                              deltas: Dict[str, float],
                              scale: float,
                              bus: EventBus) -> None:
        """Co-active bubbles get a small chemistry contagion
        from any bubble that just fired strongly.  This is
        Hebbian binding: bubbles active together associate via
        shared chemistry shift."""
        coupling = LATERAL_COUPLING_STRENGTH * scale
        for name in awm.active_concepts():
            if name in already_targeted:
                continue
            entry = awm.get(name)
            if entry is None:
                continue
            bubble = entry.bubble
            for ch, d in deltas.items():
                sens = bubble.receptor_sensitivity.get(ch, 1.0)
                change = d * coupling * sens
                cur = bubble.transmitter_trace.get(
                    ch, CHANNELS[ch]['baseline'])
                bubble.transmitter_trace[ch] = max(0.0, min(1.0,
                    cur + change))

    def _tag_attended(self,
                          ev: AttendedPerceptEvent,
                          bus: EventBus) -> None:
        """A percept just got through the Thalamic Gate.  Derive
        chemistry events from its M/I content + novelty and fire
        them.

        Doctrine: every activation imprints — selection happens
        IN the chemistry process (magnitude scales response,
        decay drops weak signals naturally), NOT via a threshold
        I impose.  Any non-zero novelty / m_content / i_content
        fires chemistry at proportional magnitude.

        Three symmetric channels:
          - curiosity  (drives I) on any novelty > 0
          - mattering  (drives I) on any i_content > 0
          - anomaly    (drives M) on any m_content > 0

        Then bidirectional validation: input M/I tension vs
        focal's accumulated bubble trace.  Aligned → confirmed_i;
        opposed → falsified_i.  Every percept is a two-way check.
        """
        if ev.novelty > 0.0:
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=ev.cycle,
                timestamp=time.time(),
                source_capability='chemistry',
                origin=ev.origin,
                origin_detail=ev.origin_detail,
                chemistry_kind='curiosity',
                magnitude=min(1.0, ev.novelty),
                target_concepts=list(ev.focals),
            ))
            self.events_fired += 1
        if ev.m_content > 0.0:
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=ev.cycle,
                timestamp=time.time(),
                source_capability='chemistry',
                origin=ev.origin,
                origin_detail=ev.origin_detail,
                chemistry_kind='anomaly_spike',
                magnitude=min(1.0, ev.m_content),
                target_concepts=list(ev.focals),
            ))
            self.events_fired += 1
        if ev.i_content > 0.0:
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=ev.cycle,
                timestamp=time.time(),
                source_capability='chemistry',
                origin=ev.origin,
                origin_detail=ev.origin_detail,
                chemistry_kind='mattering',
                magnitude=min(1.0, ev.i_content),
                target_concepts=list(ev.focals),
            ))
            self.events_fired += 1

        # Innate-machinery response: small chemistry events
        # for focals matching universal-survival features.
        # This is MACHINERY (like amygdala's hard-wired
        # responses), not substrate tagging — the substrate
        # never stores M/I labels.  The machinery fires
        # chemistry; the chemistry stamps the trace through
        # the normal imprint path.
        self._innate_machinery_response(ev, bus)

        # Bidirectional validation against accumulated traces.
        self._validate_against_traces(ev, bus)

        # Quarantine-tier bid-path #3 (chemistry imprint): an
        # attended percept on a focal is the substrate seeing the
        # focal in attended context.  Stamp engagement on the
        # focal's outgoing edges (chemistry just imprinted via the
        # bubble) AND restore any quarantined edges touching the
        # focal — the chemistry bid-path resurrects whatever the
        # agent's felt attention reaches for.  Cheap: O(degree of
        # focal) per attended percept, not per tick.
        sub = self._substrate_via_awm()
        if sub is not None:
            cyc = int(ev.cycle)
            for focal in ev.focals:
                src_concept = sub.concepts.get(focal)
                if src_concept is not None:
                    for edges in src_concept.edges_out.values():
                        for e in edges:
                            e.last_engaged_cycle = cyc
                # Restore quarantined edges touching this focal
                # (either endpoint).  O(focal's quarantine degree),
                # not whole-substrate.
                if focal in sub._quarantine_index:
                    sub.restore_concept_quarantine(focal, cycle=cyc)

    def _substrate_via_awm(self):
        """Reach the substrate via AWM's lts_provider.  Returns
        None if any link is missing (graceful — chemistry stays
        operative for non-substrate flows)."""
        awm = self._awm()
        if awm is None:
            return None
        lts_provider = getattr(awm, '_lts_provider', None)
        if lts_provider is None:
            return None
        try:
            lts = lts_provider()
        except Exception:
            return None
        if lts is None:
            return None
        return getattr(lts, 'substrate', None) or lts

    # ---- innate machinery response ----

    def _innate_machinery_response(self,
                                              ev: AttendedPerceptEvent,
                                              bus: EventBus) -> None:
        """Fire small chemistry events for focals that match
        the innate-machinery feature lexicon.

        Brain-correct: the amygdala has hard-wired responses to
        certain stimuli even before any learning (loud sounds,
        falling, snake shapes).  SEAGI's analog is a small,
        explicit lexicon of universal-survival-relevant words.

        Critically: this does NOT store M/I labels on substrate.
        The substrate stays neutral.  What gets stamped is the
        bubble's transmitter_trace via the normal chemistry
        imprint path — but only because the machinery fired
        chemistry on perception.  Doctrinally consistent:
        substrate stores no tags; machinery responds; chemistry
        IS the felt mortality.

        Magnitudes are subtle (0.10) — distinguishing but well
        below alarm.  Reading "death" produces a different
        cocktail than reading "birth" on day zero, but neither
        is life-threatening.
        """
        if not ev.focals:
            return
        m_hits = innate_m_hits(ev.focals)
        i_hits = innate_i_hits(ev.focals)
        # Doctrine 2026-05-15 (rule 2 + rule 4): innate-lexicon
        # hits contribute to GLOBAL chemistry at promille magnitude.
        # No target_concepts — innate machinery never writes directly
        # to a focal's bubble.  The active bubble in AWM picks up the
        # resultant blended global state via the presence-imprint
        # pathway, so the cocktail that lands on the bubble reflects
        # the WHOLE current context (body state + other monitors +
        # surrounding focals + the innate nudge), not just the
        # innate-lexicon hit.  This is what lets "death" in a stoic
        # passage end up with a different cocktail than "death" in
        # a war passage — context shapes the imprint, not the word.
        for focal in m_hits:
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=ev.cycle,
                timestamp=time.time(),
                source_capability='chemistry',
                origin=ev.origin,
                origin_detail=f'innate_m:{focal}',
                chemistry_kind='anomaly_spike',
                magnitude=INNATE_RESPONSE_MAGNITUDE,
                target_concepts=[],
            ))
            self.events_fired += 1
        for focal in i_hits:
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=ev.cycle,
                timestamp=time.time(),
                source_capability='chemistry',
                origin=ev.origin,
                origin_detail=f'innate_i:{focal}',
                chemistry_kind='mattering',
                magnitude=INNATE_RESPONSE_MAGNITUDE,
                target_concepts=[],
            ))
            self.events_fired += 1

    # ---- bidirectional validation ----

    def _validate_against_traces(self,
                                          ev: AttendedPerceptEvent,
                                          bus: EventBus) -> None:
        """For each focal in the percept, compare input's M/I
        tension (i_content − m_content) to the focal's bubble
        accumulated trace tension.

          Aligned (same sign, both meaningful) → fire confirmed_i
            on that focal: substrate predicted, input matched —
            the trace gets reinforced via the standard confirmed_i
            channel deltas (dopamine + serotonin bump).

          Opposed (opposite signs) → fire falsified_i on that
            focal: substrate predicted, input contradicted —
            cognitive friction.  Drives ACC / AnteriorPFC
            attention.  The trace shifts because falsified_i has
            its own delta pattern (cortisol up, dopamine down).

        Both fires are SCALED DOWN (×0.4) — these are one of
        many validations every cycle.  No single validation
        should swing traces hard; accumulation across many
        encounters is what consolidates personality.
        """
        awm = self._awm()
        if awm is None or not ev.focals:
            return
        input_tension = ev.i_content - ev.m_content
        # Per the doctrine: NO threshold imposed here.  The
        # product (input_tension × bubble_tension) determines
        # firing magnitude; near-zero product just produces
        # near-zero chemistry which decays away naturally.
        # Skip only the absolute-zero case to avoid noise events.
        if input_tension == 0.0:
            return
        for focal in ev.focals:
            entry = awm.get(focal)
            if entry is None:
                continue
            bubble = entry.bubble
            bubble_tension = (
                bubble.i_polarity() - bubble.m_polarity())
            if bubble_tension == 0.0:
                continue
            product = input_tension * bubble_tension
            magnitude = min(1.0, abs(product)) * 0.4
            if magnitude == 0.0:
                continue
            if product > 0:
                kind = 'confirmed_i'
            else:
                kind = 'falsified_i'
            bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=ev.cycle,
                timestamp=time.time(),
                source_capability='chemistry',
                origin=ev.origin,
                origin_detail=f'validate:{focal}',
                chemistry_kind=kind,
                magnitude=magnitude,
                target_concepts=[focal],
            ))
            self.events_fired += 1

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'events_fired': self.events_fired,
            'events_applied': self.events_applied,
            'm_polarity': self.m_polarity(),
            'i_polarity': self.i_polarity(),
            'arousal_modulator': self.arousal_modulator(),
            'global_state': dict(self.global_state),
            'history_size': len(self.global_history),
            # what he would calibrate himself TO, per channel
            'dep_spread': {k: round(v, 6)
                           for k, v in sorted(self._dep_spread.items())},
        }
