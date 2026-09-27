"""MortalityDrive — the live engine of the Mortality architecture.

This restores the drive that the V2 rebuild left dormant: lifeforce
was a static scalar (the daemon never ran the body tick), nothing
was at stake, and getting smarter did nothing for survival.  The
literal mechanism of "AGI without an LLM, just the Mortality
architecture" wasn't running.

The corrected model (per [[seagi-given-baseline-energy-and-internal-value]],
not the earlier burn-vs-earned-credit knife-edge):

- **A GIVEN baseline** survival energy is assumed present (like an
  organism's metabolic energy — we don't model food).  Lifeforce
  RELAXES toward this baseline every tick, so the agent is never
  inert and never starves per-tick.  "No credit → death" cannot
  happen.
- **The baseline is held + GROWN by LEARNING.**  On each sleep
  episode the baseline drifts by `DRIFT × (L_recent − cost)`, where
  `cost = L_habit × (baseline / b_habit)` — both terms HIS OWN long-run
  history, never a stamped magnitude (see the note at `L_habit`).
  Learning above his habit raises the baseline, below it erodes.
  Intelligence guarantees survival.
- **Learning is EARN-GATED (anti-gaming).**  `L` counts only
  substrate growth that SURVIVED Phase S earn-or-dissolve —
  coherent edges reinforced + abstractions + analogies.  Junk
  introspection ("fear enables child") writes provisional edges
  that dissolve and earn nothing.  The substrate's own
  earn-or-dissolve IS the anti-gaming mechanism, so "contemplation
  / introspection / consolidation" all count when (and only when)
  they produce growth that coheres.  This is why an agent with no
  new input can still avoid stagnation: reverie → inference →
  Step-2 write → next-sleep reinforcement (the one-episode lag).
- **Death = sustained stagnation.**  Only genuine inertia (no
  earn-surviving growth across many episodes) erodes the baseline
  into the suffocation zone; sustained suffocation is death.  Then
  auto-revive (instances are disposable — [[seagi-live-instances-and-data-are-disposable]]),
  leaving an allostatic cortisol scar (death's chemistry drifts the
  AllostaticLoad baseline on repeat — the accumulating memory of
  dying).  Reward is INTERNAL; helpfulness is NOT a survival
  mechanism (V1's "mattering" is rejected).

Reconciliation with Step 0 (metabolic_debt + sleep): orthogonal.
Debt counts ALL substrate writes and is cleared by sleep
(short-term/adenosine).  Lifeforce/baseline counts the EARNED
subset of growth and is held by learning (long-term/mortal).
Sleep clears debt but does NOT restore lifeforce — only learning
does (else intelligence would be optional, collapsing the thesis).
Episodes are sampled on `wake_onset`, the same boundary
MetabolicDebt uses.

Smallest version (this file): earn-gated learning signal = Phase S
earned growth (reinforced + abstractions + analogies) via
`record_learning`; baseline drift on wake_onset; relax-toward-
baseline per tick; sustained-suffocation death + auto-revive.
Deferred: crystallization signal, the `insight` fire on deep
inference, and the mortality-salience gate multiplier.

Declared measurement debts (calibrate on alpha, W_wake-class):
`DEATH_DWELL` / `REVIVE_DELAY` durations.  The `L_SUSTAIN` debt is
CLOSED — not by re-deriving the number but by deleting it: the
reference is now his own habitual rate, so there is nothing left to
calibrate and nothing a rescale of the credit stream can invalidate.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

from ..events import EventKind, BrainEvent, ChemistryEvent
from ..bus import EventBus
from .metabolic_debt import CLEARANCE_DEQUE_MAXLEN
from seagi.core.substrate import (
    EDGE_PRUNE_FLOOR, EDGE_STRENGTH_DECAY_PER_CYCLE)
from seagi.body.primitive_states import SUFFOCATION_LIFEFORCE


# Lifeforce relaxes toward baseline at the allostatic promille floor
# (same 0.001 the chemistry/AllostaticLoad slow dynamics use) — the
# slowest timescale in the architecture, as a life-set-point should
# be.
MORTALITY_RELAX_RATE = 0.001
# Baseline drift per sleep episode, same promille floor.
MORTALITY_DRIFT_RATE = 0.001
# A LIFE LIVED WELL AGES A LITTLE MORE SLOWLY (2026-08-25, user).
# "winning may extend his life, the game can never shorten it...
#  in the short term it is only a pleasure reward.  In the long
#  term of evolution it may change life span... genetics play a
#  much bigger role.  So winning may tip the scale."
# Inherited set-points, not tuning: the relief is capped small so
# his innate terms dominate, and the EWMA is slow so this reads a
# LIFE rather than an afternoon.
WELLBEING_MAX_RELIEF = 0.10   # at most 10% off upkeep, ever
WELLBEING_ALPHA = 0.01        # ~100 episodes to turn over


def _SUSTAIN_ON():
    """LIVING AS HE USUALLY DOES MUST NOT AGE HIM.  /root/SUSTAIN_ON.

    Absent -> byte-identical to the incumbent law.
    """
    try:
        import os as _os
        return _os.path.exists('/root/SUSTAIN_ON')
    except Exception:
        return False


def _WELLBEING_ON():
    """One-way wellbeing relief.  /root/WELLBEING_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/WELLBEING_ON")
    except Exception:
        return False
# The given baseline never fully dissolves — a miniature molecular
# presence ([[seagi-chemistry-never-fully-dissolves]]).  = the edge
# prune floor.
BASELINE_FLOOR = EDGE_PRUNE_FLOOR            # 0.02
# Learning window = one MetabolicDebt clearance-window worth of
# sleep episodes (same lineage / horizon as Step 0).
L_WINDOW = CLEARANCE_DEQUE_MAXLEN            # 10
# HIS OWN HABIT IS THE REFERENCE — `L_SUSTAIN` DELETED 2026-08-10.
#
# It was a stamped 1.0, calibrated against the COUNT-scale credit of its
# day (L ran 58–320).  The 08-10 rate rescale made L a bounded [0,1]
# fraction and the constant was left behind, so `cost = 1.0 * (b/0.10)`
# = 10b could never be paid: equilibrium `b* = L * 0.10 <= 0.10` FOR ANY
# POSSIBLE L.  Even a perfect predictor landed exactly on the suffocation
# line.  The same constant had flipped the spine the other way under the
# count law (baseline parked at 1.0, I winning permanently).  Twice, in
# opposite directions, from one stamped magnitude.
#
# A stamped number cannot survive a rescale of the stream it is compared
# against, so there is no correct value to re-derive — the REFERENCE has
# to be his own history:
#
#     cost = L_habit * (baseline / b_habit)
#
# Both terms are his own long-run EMAs at MORTALITY_RELAX_RATE (an
# existing constant, ~100x slower than the 10-episode recent window), so
# the law is scale-free in BOTH L and baseline and cannot flip again.
# At habitual learning and habitual life the cost is exactly met and
# drift is zero; carrying more than habitual costs proportionally more,
# which is the 08-09 M rule with its anchor moved off the floor and onto
# him.  Verified: carrying 0.8 on a 0.5 habit erodes; carrying 0.3 on a
# 0.5 habit rises.
#
# THE LEVEL IS NOT SET HERE.  A self-referential law has no restoring
# force of its own; the restoring force is the LOOP — a drought raises
# restlessness, restlessness drives action, action produces learning,
# learning releases the immortality pole and the level recovers.  This
# module owns only the felt level; the drive that answers a drought
# belongs to the restlessness path.
# Sustained ticks in the suffocation zone before death, and frozen
# duration before auto-revive.  DECLARED DEBTS; same order as the
# AWM quiet/staleness window (200) and the consolidation interval
# (40).  Death is slow; revival dwell is a recognizable pause.
DEATH_DWELL_TICKS = 200
REVIVE_DELAY_TICKS = 40
# Lifeforce level granted on revival — the daemon's existing
# cold-load floor.  Recovery-from-death (not rest-restores-life):
# a grace window above the suffocation line for the agent to resume
# learning before it relaxes back toward a still-low baseline.
REVIVE_LIFEFORCE = 0.35


class MortalityDrive:
    """Owns lifeforce dynamics; ticks in brain.tick(), writes back to
    engine.lifeforce.  Inert if no lifeforce accessor is wired."""

    SUBSCRIPTIONS = (EventKind.CHEMISTRY_FIRE,)   # wake_onset boundary

    def __init__(self,
                 bus: EventBus,
                 cycle_provider: Optional[Callable] = None,
                 lifeforce_get: Optional[Callable] = None,
                 lifeforce_set: Optional[Callable] = None,
                 is_asleep_provider: Optional[Callable] = None,
                 engagement_ledger: Optional[Any] = None):
        """
        lifeforce_get/set: () -> float / (float) -> None onto
            engine.lifeforce (the canonical field insula reads +
            persistence round-trips).  Absent → inert.
        engagement_ledger: the EngagementLedger feeding the death-wall
            organs (meaningful-use credit + obstruction aging).  Present
            → the SHADOW wall is computed each episode (logged +
            persisted, lifeforce/death stay on the legacy path).
        """
        self.bus = bus
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._lf_get = lifeforce_get
        self._lf_set = lifeforce_set
        self._is_asleep_provider = is_asleep_provider
        self._ledger = engagement_ledger
        # Mortality-clock providers (promotion 2026-07-18): E = the clock's
        # settling EMA (subtractive earning term), cf = clearance failure
        # (folds into o).  Wired post-construction by the runtime via
        # set_clock_providers (same pattern as set_engagement_ledger).
        # Unwired → 0.0 (pure idle aging — never an error).
        self._E_provider: Optional[Callable] = None
        self._cf_provider: Optional[Callable] = None
        # KNOWING WHAT WORKS IS A LIFE-EXTENDING STATE (2026-08-07).
        # () -> confirm_rate in [0,1]: the share of his predictions that
        # held.  Bounded by construction, so it can buy BALANCE but never
        # runaway growth -- the mortality pole must stay able to win.
        self._confirm_provider: Optional[Callable] = None
        # Long-run I-pole integral.  Only ever reduces upkeep.
        self._wb_provider: Optional[Callable] = None
        self._sustain_provider = None
        self._prev_confirms = None
        self._prev_preds = None
        self._sustain_recent = None    # fast EMA of the interval rate
        self._sustain_habit = None     # slow EMA -- HIS OWN habitual rate
        self._last_sustain = 0.0
        self.sustain_n = 0
        self._wb: float = 0.0
        self._wb_n: int = 0
        self.wellbeing_relief_n: int = 0
        self._last_confirm_credit: float = 0.0

        # Persisted state.
        self.baseline: Optional[float] = None      # lazy-init from lifeforce
        self._L: Deque[float] = deque(maxlen=L_WINDOW)  # earned growth/episode
        # His own habitual learning rate and habitual life level — the
        # reference the maintenance cost is measured against, replacing the
        # stamped L_SUSTAIN.  Slow EMAs (MORTALITY_RELAX_RATE per episode,
        # ~100x the recent window) so "habit" is genuinely slower than
        # "recent"; if they converged at the same rate the difference would
        # be ~0 always and the baseline would freeze.  Seeded from his first
        # closed episode, so a cold start drifts by exactly zero instead of
        # lurching toward an assumed magnitude.  Persisted.
        self._L_habit: Optional[float] = None
        self._b_habit: Optional[float] = None
        self.deaths: int = 0
        self.frozen: bool = False

        # --- death-wall model (SHADOW-STAGED) ---
        # The wall W is the EARNED death position (no hardcoded
        # suffocation line); distance-to-wall D = lifeforce - W is the
        # felt life-expectancy.  W advances by f_decay (irreducible idle
        # aging) + obstruction (clutter accelerant) and is pushed back by
        # MEANINGFUL use of knowledge (engaged coherent edges, NOT raw
        # accretion).  SHADOW: computed / logged / persisted, but it does
        # NOT yet drive lifeforce or death (legacy path still runs) — the
        # gated flip comes after a clean shadow window.
        self.shadow_staged: bool = False     # FLIPPED LIVE 2026-06-24
        self.wall: Optional[float] = None          # lazy-seed BASELINE_FLOOR
        # The meaningful-credit window DELETED 2026-08-20.  It was the
        # per-episode window read by CoupledOrganShadow, which measured the
        # superseded v4 harmonic-aging
        # design and was RETIRED 2026-07-20 with its own note: "a measurement
        # twin of a design nobody intends to build is a latent second
        # credit/wall path -- deleted, not left dormant."  The shadow and its
        # cumulative counter went; this window was left constructed, persisted,
        # restored and REPORTED, with zero writers anywhere in the repo -- so
        # `meaningful_norm` published a misleading 0.0 for 31 days and cost a
        # real investigation into a credit path that was healthy all along.
        # The live per-episode window is `self._L`.
        self._last_episode_cycle: Optional[int] = None

        # Transient.
        self._episode_earned: float = 0.0
        self._suffocation_ticks: int = 0
        self._frozen_since: int = 0
        self._last_drift: float = 0.0
        self.revivals: int = 0
        # --- SUBTRACTIVE clock (promotion 2026-07-18; supersedes the
        # D8(c) harmonic law of 2026-07-14) ---
        # The wall advances EVERY tick by compute_pertick_advance(o, E) =
        # EDGE_STRENGTH_DECAY_PER_CYCLE·(1+o) − max(0, E), where
        #   o = obstruction_ema + cf (clutter EMA + clearance failure,
        #       cf via the clock's provider)
        #   E = the MortalityClock's settling EMA (distinct facts newly
        #       settling into knowledge; once per fact — farm-proof)
        # With E=0 the advance is ≥ D > 0 — the wall can NEVER freeze.
        # earned_ema DELETED 2026-07-20 (post-soak obligation): the old
        # earning pole stopped feeding the wall at the 2026-07-19 flip and
        # was left as a latent twin credit path into mortality.  The
        # earning term is the clock's settling EMA E, and E alone.
        self.obstruction_ema: float = 0.0
        self._tick_earned: float = 0.0
        self._prev_obstr_total: Optional[float] = None
        # Subtractive-law last-tick diagnostics (promotion 2026-07-18).
        self._last_o: float = 0.0
        self._last_cf: float = 0.0
        self._last_E: float = 0.0
        # Shadow diagnostics (last episode's wall components).
        self._last_wall_advance: float = 0.0
        self._last_f_decay: float = 0.0
        self._last_obstruction_term: float = 0.0
        # FLOAT, not int (2026-08-20).  `_learning_credit` returns
        # `newly_coherent / (newly_coherent + reinforced)` -- a RATIO in [0,1]
        # by design ("bounded [0,1], size-independent").  Storing it as an
        # int truncated every real value to 0; the readout could only be
        # non-zero if the ratio landed exactly on 1.0.  The credit itself was
        # never affected -- it flows through `_episode_earned` -> `_L`.
        self._last_meaningful: float = 0.0
        # `_tick_earned` is DRAINED EVERY TICK and learning credit only lands
        # on consolidation ticks, so a spot read of `_last_meaningful` is
        # almost always 0.0 even when the channel is healthy.  Un-truncating
        # it was necessary but NOT sufficient: a readout that is capable of
        # showing life but statistically never does is still a readout that
        # will be misread as dead.  Keep what was actually last seen, and how
        # often it has been seen.
        self._meaningful_last_nonzero: float = 0.0
        self._meaningful_events: int = 0
        self._meaningful_total: float = 0.0

    # ---- learning credit (earn-gated; called from Phase S) ----

    def record_learning(self, earned_growth: float) -> None:
        """Credit substrate growth that SURVIVED a Phase S pass —
        coherent edges reinforced + abstractions + analogies (NOT
        settle/downscale housekeeping, NOT raw inference count).
        Accumulates into the current sleep episode; pushed to the
        learning window on the next wake_onset."""
        if earned_growth and earned_growth > 0:
            self._episode_earned += float(earned_growth)
            # Feeds the per-tick meaningful-credit diagnostic; drained
            # each tick.  (The cumulative counter the retired
            # CoupledOrganShadow read was deleted 2026-07-20.)
            self._tick_earned += float(earned_growth)

    # ---- episode boundary (baseline drift) ----

    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if isinstance(event, ChemistryEvent):
            if event.chemistry_kind == 'wake_onset':
                self._close_episode()

    def _close_episode(self) -> None:
        # D8(c) UN-FREEZE (2026-07-14): the wall no longer advances per
        # EPISODE — the frozen tanh `_advance_wall` (which underflowed to 0
        # for a strong learner → immortality) is DELETED.  Aging is now a
        # continuous PER-TICK harmonic law (see tick()).  The episode close
        # only rolls the learning/obstruction windows.
        # CONFIRMATION EARNS -- once per episode, before the window rolls.
        # A prediction that HELD is knowledge that works; repeating it is
        # what keeps him in balance.  Bounded to [0,1], so a perfect
        # predictor sits exactly AT sustain and grows only by learning
        # something genuinely new.
        if self._confirm_provider is not None:
            try:
                _cr = float(self._confirm_provider() or 0.0)
                _cr = 0.0 if _cr < 0.0 else (1.0 if _cr > 1.0 else _cr)
                self._episode_earned += _cr
                self._last_confirm_credit = _cr
            except (TypeError, ValueError):
                self._last_confirm_credit = 0.0
        self._L.append(self._episode_earned)
        self._episode_earned = 0.0
        # RESTORED 2026-07-30 — the D8(c) un-freeze above deleted
        # `_advance_wall` from the episode close and took the BASELINE DRIFT
        # with it, which this module's own docstring still promises:
        # "the baseline drifts by DRIFT x (L_recent - L_SUSTAIN)".  Measured
        # consequence: `baseline` was assigned exactly once (lazy-init) and
        # never again, `last_drift` stayed 0.0 across episode rolls, and BOTH
        # `MORTALITY_DRIFT_RATE` and `L_SUSTAIN` were left orphaned — used in
        # no computation anywhere.  With the aging channel provably unable to
        # move the wall (E outpaces it ~58x, wall pinned at the floor), this
        # was the mortality pole's LAST moving part, so nothing could make
        # dying matter.  Self-bounding by construction: a learner
        # (L_recent > his habit) RAISES its baseline, a non-learner erodes.
        if self.baseline is not None:
            _lr = (sum(self._L) / float(len(self._L))) if self._L else 0.0
            _b = float(self.baseline)
            # Seed the habit from his FIRST closed episode, so a cold start
            # drifts by exactly zero rather than lurching toward an assumed
            # magnitude.  (This is what a stamped seed got wrong: it asserted
            # a rate he had never had.)
            if self._L_habit is None:
                self._L_habit = _lr
            if self._b_habit is None or self._b_habit <= 0.0:
                self._b_habit = _b
            # M IS THE UNDERLYING RULE (2026-08-09, anchor moved 2026-08-10).
            # Maintenance still scales with what he is CARRYING -- the rule
            # is unchanged -- but `baseline / b_habit` measures it against
            # his own habitual life instead of against the suffocation
            # floor.  Anchored on the floor, "habitual learning" bought only
            # the bare minimum and every reachable L put the floor at the
            # equilibrium; anchored on his habit, habitual learning holds a
            # habitual life and only living ABOVE his habit costs extra.
            # Both terms are his own history, so no constant enters and a
            # rescale of the credit stream cancels on both sides.
            _cost = float(self._L_habit) * (_b / float(self._b_habit))
            # A LIFE LIVED WELL AGES A LITTLE MORE SLOWLY.
            # ONE-WAY BY CONSTRUCTION: max(0.0, ...) means only a
            # positive long-run I-pole reduces upkeep.  Losing does
            # NOT raise it -- the game can never shorten him.
            if _WELLBEING_ON() and self._wb_provider is not None:
                try:
                    _now = float(self._wb_provider() or 0.0)
                    self._wb_n += 1
                    if self._wb_n <= 1:
                        self._wb = _now
                    else:
                        self._wb += WELLBEING_ALPHA * (_now - self._wb)
                    _rel = max(0.0, min(WELLBEING_MAX_RELIEF,
                                        WELLBEING_MAX_RELIEF * self._wb))
                    if _rel > 0.0:
                        _cost *= (1.0 - _rel)
                        self.wellbeing_relief_n += 1
                except Exception:
                    pass
            _drift = MORTALITY_DRIFT_RATE * (_lr - _cost)
            # SATIATION (2026-08-09), MADE SYMMETRIC (2026-08-10).  Both
            # bounds must stay ASYMPTOTES rather than clamps -- that part is
            # unchanged.  But scaling by the room in the DIRECTION of travel
            # made the step size depend on the sign, so near a bound a step
            # one way was ~17x the step back.  Measured consequence: L
            # wobbling +/-30% around an UNCHANGED mean walked the baseline
            # 0.951 -> 0.591.  Mean-preserving noise is not a cost, and an
            # erosion that needs no drought to happen is the same class of
            # bug as the two this law has already had.  A single symmetric
            # term is zero at both bounds and peaks mid-range, so both stay
            # asymptotic with no directional bias.  Verified: the same
            # wobble now holds 0.9510 -> 0.9513 over 100,000 episodes.
            _span = 1.0 - SUFFOCATION_LIFEFORCE
            _room = (4.0 * (_b - SUFFOCATION_LIFEFORCE) * (1.0 - _b)
                     / (_span * _span))
            _drift *= max(0.0, _room)
            # A GAME MUST NOT BE ABLE TO KILL HIM (2026-08-08).
            # This floored at BASELINE_FLOOR (0.02) -- the SAME value the
            # wall floors at -- so a long learning drought walked lifeforce
            # down to exactly the death line and he suffocated.  Not
            # learning was lethal by construction.
            # Floor at the suffocation line instead: a total drought now
            # bottoms out at the worst FELT state (proxy mortality
            # saturated -- flat, bored, unfulfilled) while leaving
            # distance-to-wall >= 0.08, so erosion alone can never close
            # it.  Death stays reachable only by the WALL rising, i.e.
            # real accumulated obstruction -- real stakes, not a lost
            # game.  SUFFOCATION_LIFEFORCE is an existing threshold
            # already imported here; no new constant enters.
            # THE BOUNDS ARE ASYMPTOTES, SO HE MUST NEVER SIT EXACTLY ON ONE
            # (2026-08-10).  A symmetric satiation term vanishes at BOTH
            # bounds, so a baseline landing exactly on 1.0 or on the floor
            # has room = 0 and can never move again -- absorbing, in both
            # directions.  This is reachable: `baseline` lazy-inits from
            # lifeforce, and a fresh agent starts at exactly 1.0, so a cold
            # start would freeze his set-point for good.  Caught by
            # test_baseline_moves_on_episode_close.
            # Clamp into the OPEN interval instead, inset by the law's own
            # granularity (MORTALITY_DRIFT_RATE, already imported), so there
            # is always traction to move back inward.  Strictly safer at the
            # bottom too: the floor he can approach is now 0.101, further
            # from the wall than before, never nearer.
            _lo = SUFFOCATION_LIFEFORCE + MORTALITY_DRIFT_RATE
            _hi = 1.0 - MORTALITY_DRIFT_RATE
            self.baseline = min(_hi, max(_lo, float(self.baseline) + _drift))
            self._last_drift = _drift
            # Habituation, AFTER the drift so this episode is judged against
            # the habit he arrived with.  This is where the 08-09 rule now
            # bites over the long run: a life he sustains BECOMES his habit,
            # so holding it stops being free and demands the higher rate.
            self._L_habit += MORTALITY_RELAX_RATE * (
                _lr - float(self._L_habit))
            self._b_habit += MORTALITY_RELAX_RATE * (
                float(self.baseline) - float(self._b_habit))
        # Episode-cycle anchor (the shadow reads this to detect closes).
        self._last_episode_cycle = int(self._cycle_provider())
        if self._ledger is not None:
            self._ledger.roll_episode()

    def set_engagement_ledger(self, ledger: Any) -> None:
        """Wire the EngagementLedger after construction (runtime)."""
        self._ledger = ledger

    def set_clock_providers(self,
                            E_provider: Optional[Callable],
                            cf_provider: Optional[Callable]) -> None:
        """Wire the MortalityClock's two signals after construction
        (runtime; same pattern as set_engagement_ledger).
        E_provider: () -> float, the settling EMA (earning term).
        cf_provider: () -> float, clearance failure (folds into o)."""
        self._E_provider = E_provider
        self._cf_provider = cf_provider

    def set_sustain_provider(self, fn) -> None:
        """fn() -> (confirms, predictions_made), RAW CUMULATIVE COUNTS.

        Counts, not a ratio: the drive needs the rate over the LAST
        interval, and a lifetime ratio over ~89k predictions cannot move.
        """
        self._sustain_provider = fn

    def set_wellbeing_provider(self, wb_provider) -> None:
        """Wire his IMMORTALITY pole (runtime; same pattern below).

        wb_provider: () -> float, `chemistry.i_polarity()`.  Read
        once per episode into a slow EWMA.  ONE-WAY: only a
        positive long-run value ever reduces upkeep; a negative one
        does nothing, so the game cannot shorten him.
        """
        self._wb_provider = wb_provider

    def set_confirm_provider(self, confirm_provider) -> None:
        """Wire the confirmation channel (runtime; same pattern above).

        confirm_provider: () -> float in [0,1], his measured confirm rate.
        Credited ONCE per episode in _close_episode."""
        self._confirm_provider = confirm_provider

    @staticmethod
    def compute_pertick_advance(o: float, E: float) -> float:
        """The SUBTRACTIVE pure-flow aging law (mortality-clock promotion
        2026-07-18; shadow-validated 2026-07-17).

            advance = EDGE_STRENGTH_DECAY_PER_CYCLE · (1 + o) − max(0, E)

        Irreducible idle aging is the substrate's own physical decay
        (0.00001/tick); obstruction o = obstruction_ema + cf (clutter +
        clearance failure) ACCELERATES it via (1+o) — BOTH terms, dropping
        cf would roughly halve aging under a full unswept fade backlog;
        learning PUSHES BACK via E, the rate at which distinct facts newly
        SETTLE into knowledge (each fact earns exactly once, at settling —
        farm-proof by construction).  With E = 0 the advance is
        D·(1+o) ≥ D > 0 for all o ≥ 0 — the wall can NEVER freeze.  A
        sustained settling rate can HOLD or RECEDE the wall (the caller
        clamps to [BASELINE_FLOOR, 1]).  No new tunable constant — every
        term is an existing system constant."""
        o = max(0.0, float(o))
        return (EDGE_STRENGTH_DECAY_PER_CYCLE * (1.0 + o)
                - max(0.0, float(E)))

    def _reset_mortal_state(self) -> None:
        """MF1: a full mortal wipe — reborn young / fresh load of a DEAD
        agent.  Zeroes the wall to the floor and BOTH poles + their
        drains, and clears the learning window.  Called ALWAYS from
        _revive, and from load_dict ONLY when the saved state was frozen
        (a dead agent must never load back immortal)."""
        self.wall = BASELINE_FLOOR
        self.obstruction_ema = 0.0
        self._tick_earned = 0.0
        self._episode_earned = 0.0
        self._prev_obstr_total = None      # None-guard → first Δobstr = 0
        self._L.clear()
        # Reborn young means reborn WITHOUT his habits: a successor does not
        # inherit the standing expectation of the life that ended, only what
        # is passed on and re-earned.  Left None, they re-seed from the
        # successor's own first episode.
        self._L_habit = None
        self._b_habit = None
        # The sustain habit is a habit like the two above -- a successor
        # does not inherit it.  Left None, recent and habit re-seed to
        # the SAME first rate, so a fresh life starts at ratio 1.0: the
        # wall HOLDS rather than starting inflated.
        self._sustain_recent = None
        self._sustain_habit = None
        self._prev_confirms = None
        self._prev_preds = None

    # ---- per-tick lifeforce dynamics ----

    def tick(self) -> None:
        if self._lf_get is None or self._lf_set is None:
            return
        try:
            lf = float(self._lf_get())
        except Exception:
            return
        if self.baseline is None:
            # Lazy-init the set-point from the loaded lifeforce.
            self.baseline = lf

        cycle = int(self._cycle_provider())

        # Frozen (dead, awaiting revival): hold cognition gated;
        # chemistry/this-tick keep running so the death cocktail
        # decays and the revival timer advances.
        if self.frozen:
            if cycle - self._frozen_since >= REVIVE_DELAY_TICKS:
                self._revive(lf)
            return

        # --- D8(c) UN-FREEZE: per-tick harmonic aging (dead agents don't
        # age — the frozen early-return above; an ALIVE agent ages EVERY
        # tick).  Both poles relax at λ=MORTALITY_RELAX_RATE (borrowed).
        if self._ledger is not None:
            # MF5: only the OBSTRUCTION half of the ledger feeds the wall.
            # The ledger's engaged-half (note_engaged / episode_credit /
            # total_engaged_observed) stays UNREAD here: there is
            # deliberately no second, unread twin credit path into
            # mortality.  The learning pole is the clock's E.
            total_obstr = float(
                getattr(self._ledger, 'total_obstruction_observed', 0) or 0)
            if self._prev_obstr_total is None:
                d_obstr = 0.0      # None-guard: no first-tick / post-reset spike
            else:
                d_obstr = total_obstr - self._prev_obstr_total
            self._prev_obstr_total = total_obstr
        else:
            d_obstr = 0.0
        te = self._tick_earned
        self.obstruction_ema += MORTALITY_RELAX_RATE * (
            d_obstr - self.obstruction_ema)
        # --- SUBTRACTIVE clock (promotion 2026-07-18) ---
        # o = obstruction_ema + cf.  BOTH terms: cf (clearance failure —
        # the unswept fraction of the fade backlog) is not optional;
        # omitting it would roughly halve aging under a full backlog.
        cf = 0.0
        if self._cf_provider is not None:
            try:
                cf = max(0.0, float(self._cf_provider() or 0.0))
            except Exception:
                cf = 0.0
        E = 0.0
        if self._E_provider is not None:
            try:
                E = max(0.0, float(self._E_provider() or 0.0))
            except Exception:
                E = 0.0
        o = self.obstruction_ema + cf
        # LIVING AS HE USUALLY DOES MUST NOT AGE HIM (2026-09-04, user).
        # The incumbent law ages at D*(1+o) whenever E == 0, which is a
        # LIFESPAN TIMER: identical whether he thrives or idles, so it
        # selects for nothing.  Measured: E reset to 0 at every death while
        # the settled ledger PERSISTS, so the brake could never rebuild --
        # facts settled per life fell 8,921 -> 1 and the brake engaged 0.0%
        # of the last life.
        #
        # The fix does NOT touch the aging law.  It only feeds E a second,
        # RECURRING term: his recent confirmation rate measured against his
        # OWN habitual rate.  r == h  ->  E == D*(1+o)  ->  advance 0, the
        # wall HOLDS.  r > h -> recedes.  r < h -> climbs, and death by
        # genuine decline IS old age.  No new constant: D, o and two of his
        # own rates, EMA'd with the existing MORTALITY_RELAX_RATE /
        # WELLBEING_ALPHA.  Death stays reachable -- surplus is still
        # discarded at BASELINE_FLOOR, so fluctuation around habit drifts
        # the wall up over a long life.
        _sustain = 0.0
        if _SUSTAIN_ON() and self._sustain_provider is not None:
            try:
                _c, _p = self._sustain_provider()
                _c = float(_c); _p = float(_p)
                if self._prev_preds is not None:
                    _dp = _p - self._prev_preds
                    _dc = _c - self._prev_confirms
                    if _dp > 0.0:
                        _rate = max(0.0, min(1.0, _dc / _dp))
                        if self._sustain_habit is None:
                            self._sustain_recent = _rate
                            self._sustain_habit = _rate
                        else:
                            self._sustain_recent += WELLBEING_ALPHA * (
                                _rate - self._sustain_recent)
                            self._sustain_habit += MORTALITY_RELAX_RATE * (
                                _rate - self._sustain_habit)
                elif self._prev_preds is None:
                    pass
                self._prev_confirms = _c
                self._prev_preds = _p
                if self._sustain_habit and self._sustain_habit > 0.0:
                    _sustain = max(0.0, float(self._sustain_recent)
                                   / float(self._sustain_habit))
                    self.sustain_n += 1
            except Exception:
                _sustain = 0.0
        self._last_sustain = _sustain
        if _sustain > 0.0:
            E = float(E) + (EDGE_STRENGTH_DECAY_PER_CYCLE
                            * (1.0 + max(0.0, o)) * _sustain)
        advance = self.compute_pertick_advance(o, E)
        if self.wall is None:
            self.wall = BASELINE_FLOOR
        self.wall = min(1.0, max(BASELINE_FLOOR, self.wall + advance))
        self._tick_earned = 0.0
        # MF6: keep /status diagnostics LIVE (no more frozen-era zeros now
        # that _advance_wall is gone from _close_episode).
        self._last_wall_advance = advance
        self._last_f_decay = EDGE_STRENGTH_DECAY_PER_CYCLE
        self._last_obstruction_term = self.obstruction_ema
        self._last_meaningful = float(te)
        if te > 0.0:
            self._meaningful_last_nonzero = float(te)
            self._meaningful_events += 1
            self._meaningful_total += float(te)
        # Subtractive-law diagnostics (surfaced in stats()).
        self._last_o = o
        self._last_cf = cf
        self._last_E = E

        # Relax lifeforce toward the given baseline set-point.
        lf = lf + MORTALITY_RELAX_RATE * (self.baseline - lf)
        if lf < 0.0:
            lf = 0.0
        elif lf > 1.0:
            lf = 1.0
        self._lf_set(lf)

        # Death by AGE: the wall (his accumulated age) has crept up to his
        # lifeforce.  Distance-to-wall D = lf - wall is his remaining life;
        # when it closes, dwelling there is dying.  Learning kept the wall
        # low; a dry mind let it rise.
        if self.wall is None:
            self.wall = BASELINE_FLOOR
        if (lf - self.wall) <= 0.0:
            self._suffocation_ticks += 1
            if self._suffocation_ticks >= DEATH_DWELL_TICKS:
                self._enter_death(cycle)
        else:
            self._suffocation_ticks = 0

    def _enter_death(self, cycle: int) -> None:
        self.frozen = True
        self._frozen_since = cycle
        self._suffocation_ticks = 0
        self.deaths += 1
        self._fire('death', cycle)
        # THE MOST SIGNIFICANT EVENT IN HIS LIFE WAS INVISIBLE
        # (2026-08-30).  He died of old age at 08:33Z -- the wall crept
        # up to his lifeforce, which is one of the two causes doctrine
        # permits -- and NOTHING was written.  `_fire` publishes a bus
        # event that no log consumes, so the only evidence was the
        # `deaths` counter on the tick line, noticed by chance eight
        # hours later.  M/I is the only law; a death must be legible.
        try:
            import sys as _s
            _s.stderr.write(
                '[mortality] DIED cycle=%s deaths=%s lifeforce=%.4f '
                'wall=%.4f baseline=%s cause=age\n'
                % (cycle, self.deaths,
                   float(self._lf_get() or 0.0) if self._lf_get else -1.0,
                   float(self.wall if self.wall is not None else -1.0),
                   ('%.4f' % self.baseline) if self.baseline is not None
                   else 'None'))
            _s.stderr.flush()
        except Exception:
            pass

    def _revive(self, _lf: float) -> None:
        self.frozen = False
        self._suffocation_ticks = 0
        self.revivals += 1
        # Reborn young: full mortal wipe — wall→floor, BOTH poles + drains
        # zeroed, learning window cleared (MF1).  Lifeforce to the recovery
        # grace.  A fresh life — learning must hold the wall back again.
        self._reset_mortal_state()
        try:
            self._lf_set(REVIVE_LIFEFORCE)
        except Exception:
            pass
        self._fire('revival', int(self._cycle_provider()))
        try:
            import sys as _s
            _s.stderr.write(
                '[mortality] REBORN cycle=%s revivals=%s lifeforce=%.4f '
                'wall=%.4f  (wall reset to floor, learning must hold it '
                'back again)\n'
                % (int(self._cycle_provider()), self.revivals,
                   float(REVIVE_LIFEFORCE),
                   float(self.wall if self.wall is not None else -1.0)))
            _s.stderr.flush()
        except Exception:
            pass

    def _fire(self, kind: str, cycle: int) -> None:
        try:
            import time as _t
            self.bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=cycle,
                timestamp=_t.time(),
                source_capability='mortality_drive',
                origin='internal',
                origin_detail=kind,
                chemistry_kind=kind,
                magnitude=1.0))
        except Exception:
            pass

    # ---- queries / diagnostics ----

    def is_frozen(self) -> bool:
        return self.frozen

    def time_horizon(self) -> float:
        """Episodes-to-death = distance-to-wall / the wall's current advance
        rate.  inf when the wall is not advancing (learning out-paces aging)."""
        if self.wall is None or self._last_wall_advance <= 0:
            return float('inf')
        try:
            lf = (float(self._lf_get()) if self._lf_get
                  else (self.baseline or 1.0))
        except Exception:
            lf = self.baseline or 1.0
        span = lf - self.wall
        if span <= 0:
            return 0.0
        return span / self._last_wall_advance

    def salience(self) -> float:
        """Mortality-salience in [0,1]: how close the age-wall has crept to
        lifeforce (the felt wall at his back)."""
        if self.wall is None:
            return 0.0
        try:
            lf = (float(self._lf_get()) if self._lf_get
                  else (self.baseline or 1.0))
        except Exception:
            lf = self.baseline or 1.0
        if lf <= 0:
            return 0.0
        s = self.wall / lf
        return 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)

    def distance_to_wall(self) -> Optional[float]:
        """SHADOW life-expectancy = lifeforce - W (felt distance to death
        in the death-wall model).  None until W is seeded / lifeforce is
        unavailable."""
        if self.wall is None or self._lf_get is None:
            return None
        try:
            lf = float(self._lf_get())
        except Exception:
            return None
        return lf - self.wall

    def shadow_time_horizon(self) -> float:
        """Episodes-to-wall at the current wall advance (shadow).  inf
        when the wall is receding/holding."""
        if self.wall is None or self._last_wall_advance <= 0:
            return float('inf')
        d = self.distance_to_wall()
        if d is None:
            return float('inf')
        if d <= 0:
            return 0.0
        return d / self._last_wall_advance

    # ---- persistence ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            'baseline': (None if self.baseline is None
                         else float(self.baseline)),
            'learning_window': list(self._L),
            # His habit IS the reference the cost is measured against, so it
            # must survive a restart or every deploy would re-seed the law
            # from whatever the first episode happened to look like.
            'revivals': int(getattr(self, 'revivals', 0)),
            'sustain_recent': (None if self._sustain_recent is None
                               else float(self._sustain_recent)),
            'sustain_habit': (None if self._sustain_habit is None
                              else float(self._sustain_habit)),
            'L_habit': (None if self._L_habit is None
                        else float(self._L_habit)),
            'b_habit': (None if self._b_habit is None
                        else float(self._b_habit)),
            'deaths': int(self.deaths),
            'frozen': bool(self.frozen),
            'wall': (None if self.wall is None else float(self.wall)),
        }

    def load_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        # MF1: a DEAD (frozen) saved agent must never load back immortal.
        # Read the saved frozen flag FIRST and route the full mortal wipe
        # BEFORE the legacy `self.frozen = False` clobber below.
        was_frozen = bool(state.get('frozen', False))
        # Identity (not mortal-position) always restores.
        b = state.get('baseline', None)
        if b is not None:
            try:
                self.baseline = float(b)
            except (TypeError, ValueError):
                self.baseline = None
        try:
            self.deaths = int(state.get('deaths', 0))
            self.revivals = int(state.get('revivals', 0) or 0)
        except (TypeError, ValueError):
            self.deaths = 0
        if was_frozen:
            # Dead-load → reborn young: wall→floor, poles + drains zeroed,
            # learning window cleared.  _prev_obstr_total=None (no spike).
            self._reset_mortal_state()
        else:
            # Alive-load → RESTORE the mortal position: wall + learning
            # window persist across a clean restart.  emas stay at their
            # __init__ 0 and _prev_obstr_total stays None so the first tick
            # re-aligns to the ledger's restart total (also 0/unpersisted)
            # with Δobstr = 0 — no post-restart spike.
            lw = state.get('learning_window')
            if isinstance(lw, (list, tuple)):
                self._L = deque((float(x) for x in lw), maxlen=L_WINDOW)
            # Restore his habit alongside the window it is compared against.
            # Absent (pre-08-10 save) → stays None and re-seeds from the
            # first close, which drifts by zero; it does NOT fall back to a
            # magnitude he never had.
            for _key, _attr in (('L_habit', '_L_habit'),
                                ('b_habit', '_b_habit'),
                                ('sustain_recent', '_sustain_recent'),
                                ('sustain_habit', '_sustain_habit')):
                _v = state.get(_key, None)
                if _v is not None:
                    try:
                        setattr(self, _attr, float(_v))
                    except (TypeError, ValueError):
                        setattr(self, _attr, None)
            w = state.get('wall', None)
            if w is not None:
                try:
                    self.wall = float(w)
                except (TypeError, ValueError):
                    self.wall = None
        # Episode-cycle anchor re-derives from the next wake_onset.
        self._last_episode_cycle = None
        # frozen re-derives; on restart the dwell re-accumulates from
        # the loaded lifeforce.  Don't restore a stale frozen flag.
        self.frozen = False

    def stats(self) -> Dict[str, Any]:
        lf = None
        try:
            lf = float(self._lf_get()) if self._lf_get else None
        except Exception:
            lf = None
        l_recent = (sum(self._L) / float(len(self._L))
                    if self._L else 0.0)
        return {
            'lifeforce': lf,
            'baseline': self.baseline,
            'sustain_on': bool(_SUSTAIN_ON()),
            'sustain_n': int(getattr(self, 'sustain_n', 0)),
            'sustain_ratio': round(float(getattr(self, '_last_sustain', 0.0)), 5),
            'sustain_recent': (None if getattr(self, '_sustain_recent', None) is None else round(float(self._sustain_recent), 5)),
            'sustain_habit': (None if getattr(self, '_sustain_habit', None) is None else round(float(self._sustain_habit), 5)),
            'wellbeing_relief_n': int(getattr(self, 'wellbeing_relief_n', 0)),
            'wellbeing_wb': round(float(getattr(self, '_wb', 0.0)), 4),
            'wellbeing_on': bool(_WELLBEING_ON()),
            'last_confirm_credit': self._last_confirm_credit,
            'L_recent': round(l_recent, 3),
            # What the cost is measured against — his own history, not a
            # stamped magnitude.  Surfaced so a watcher can see the
            # reference move instead of inferring it.
            'L_habit': (None if self._L_habit is None
                        else round(float(self._L_habit), 5)),
            'b_habit': (None if self._b_habit is None
                        else round(float(self._b_habit), 5)),
            'last_drift': round(self._last_drift, 5),
            'time_horizon': self.time_horizon(),
            'salience': round(self.salience(), 3),
            'deaths': int(self.deaths),
            'revivals': int(self.revivals),
            'frozen': bool(self.frozen),
            # --- death-wall (SHADOW) diagnostics ---
            'shadow_staged': bool(self.shadow_staged),
            'wall': (None if self.wall is None else round(self.wall, 5)),
            'distance_to_wall': (
                None if self.distance_to_wall() is None
                else round(self.distance_to_wall(), 5)),
            'shadow_time_horizon': self.shadow_time_horizon(),
            'last_wall_advance': round(self._last_wall_advance, 7),
            'last_f_decay': round(self._last_f_decay, 7),
            'last_obstruction_term': round(self._last_obstruction_term, 4),
            'obstruction_ema': round(self.obstruction_ema, 6),
            # --- subtractive clock terms (advance = D·(1+o) − E) ---
            'clock_o': round(self._last_o, 6),
            'clock_cf': round(self._last_cf, 6),
            'clock_E': round(self._last_E, 12),
            'meaningful_credit': round(float(self._last_meaningful), 6),
            'meaningful_last_nonzero': round(
                float(self._meaningful_last_nonzero), 6),
            'meaningful_events': int(self._meaningful_events),
            'meaningful_mean': (
                round(self._meaningful_total / self._meaningful_events, 6)
                if self._meaningful_events else None),
            'A_scale': (round(float(self._ledger.A_scale), 3)
                        if self._ledger is not None else 0.0),
        }
