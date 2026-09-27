"""WorldActor — learns to SOLVE a task by variation x selection, from scratch.

The agent is set a problem it has never solved (reach a goal in an unknown
world) and we observe the architecture learn to solve it.  The loop, all from
machinery already present:

  * ASSUMPTION + PERSISTENT CURIOSITY (the action policy).  The agent acts on
    whatever it currently has tagged as success-leading — value_of(the state an
    action is predicted to reach) — but CURIOSITY NEVER SWITCHES OFF: a
    count-based novelty bonus keeps under-tried actions attractive, so the agent
    keeps varying and re-learning instead of freezing on its first guess.
    score(a) = value_of(predicted_next) + 1/(1+visits(state,a)).  Unknown
    consequences (visits 0) get the full curiosity bonus = explored.  This is
    the VARIATION.

  * SUCCESS FIRES THE LIFE-LEAN (the selection).  Reaching the goal fires the
    immortality / confirmed_i NT-lean back along the path that got there
    (decaying by recency, an eligibility trace), so value_of those states rises
    and the agent leans toward them next time.  Solving the task and "the NT
    says this prolongs life" are the same event.  This is the SELECTION.

  * LIFEFORCE FOLLOWS.  Genuine improvement (reaching the goal in fewer steps
    than ever before) credits record_learning — the intelligence->life set-point
    the doctrine already wires.  It is gated to real improvement, so it cannot
    be farmed by re-solving and it plateaus at mastery.

Action-conditioning: the world model is learned action-conditioned, keying the
transition on a composite (state, action) token and reusing GroundingLoop's
predict-confirm-or-dissolve.  It is NOT reinforcement learning: nothing is
maximised; curiosity varies and the intrinsic survival NT-lean selects.  The
'world_actor'-sourced chemistry is ignored by the reward ledger (no RL backdoor).
"""

from __future__ import annotations

import math
import os
import random
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from ..events import (EventKind, CapabilityClaimEvent, ChemistryEvent,
                      AttendedPerceptEvent, SubstrateWriteQueuedEvent,
                      PredictionErrorEvent)
from ..bus import EventBus
from seagi.core.substrate import (EDGE_PRUNE_FLOOR, PROVISIONAL_EDGE_STRENGTH,
                                  FACET_RELATION, WORLD_TOKEN_PREFIX)
from seagi.brain.capabilities.grounding import WORLD_RELATION


# The architecture's own credit-decay (reward_ledger.DISCOUNT_FACTOR) — the
# eligibility trace along the successful path AND the dis-corroboration of a
# route that led to failure, not a new constant.
PATH_CREDIT_DECAY = 0.9
# Bounded history of attempt lengths — a measurement ring (the learning curve),
# never a behavioural signal.
SOLVE_HISTORY = 256
# SURVIVAL VALENCE (2026-07-29), OPT-IN.  Unset -> byte-identical to before.
_SURVIVAL = os.environ.get(
    'SEAGI_SURVIVAL', '').strip().lower() in ('1', 'true', 'yes', 'on')
# Memory bound on the per-life (s, a, s_next) ring ONLY — not a behavioural
# constant.  Observed lives are ~700-1,300 steps; saturation is COUNTED and
# surfaced as `survival_saturated` so a truncated walk can never masquerade
# as a complete one.
PATH_SA_CAP = 65536
# THE FLIP (2026-07-29, USER AUTHORISED), OPT-IN and independent of
# SEAGI_SURVIVAL.  Unset -> byte-identical scoring to before.
def _TONE_STEER_ON():
    """File-gated so half B can be turned on/off WITHOUT a restart --
    a restart costs hours of `_tried` rebuild."""
    try:
        return os.path.exists('/root/TONE_STEER_ON')
    except Exception:
        return False


def _WORLD_PE_ON():
    """Give the WORLD a route to his dopamine.  File-gated so it can be
    interleaved without a restart, like the tone steer."""
    try:
        return os.path.exists('/root/WORLDPE_ON')
    except Exception:
        return False


EXPLORE_EVERY = 300      # ticks between recomputes (cost guard)
EXPLORE_SWEEPS = 40      # bounded value-iteration sweeps


def _EGOMOVE_ON():
    """Demote a move predicted not to move him.  /root/EGOMOVE_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/EGOMOVE_ON")
    except Exception:
        return False


def _RELSENSE_ON():
    """The world reports the pursued goal relation.  /root/RELSENSE_ON.
    Here it only decides what PROGRESS means for the insight channel."""
    try:
        import os as _os
        return _os.path.exists("/root/RELSENSE_ON")
    except Exception:
        return False


def _SEARCH_ON():
    """Take the search organ's control.  /root/SEARCH_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/SEARCH_ON")
    except Exception:
        return False


def _HYPOTHESIS_ON():
    """Take the hypothesis organ's arrow.  /root/HYPOTHESIS_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/HYPOTHESIS_ON")
    except Exception:
        return False


def _RELPLAN_ON():
    """Take the first action of the world's paint plan.  /root/RELPLAN_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/RELPLAN_ON")
    except Exception:
        return False


def _RELSTEER_ON():
    """Act on the pursued relation.  /root/RELSTEER_ON.

    Keyed on (game, level, situation, action): the situation is the
    direction from him to the marker, or "the picture" -- an index that
    recurs across lives and levels, where a board hash does not.  The
    mean movement of the relation per action in that situation is
    learned from his own steps; the action with the best mean, once it
    has been seen twice and is positive, is taken.  Absent ->
    byte-identical.
    """
    try:
        import os as _os
        return _os.path.exists("/root/RELSTEER_ON")
    except Exception:
        return False


def _NOOPALL_ON():
    """Drop an action known to change nothing HERE, asked about the
    action he actually chose.  /root/NOOPALL_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/NOOPALL_ON")
    except Exception:
        return False


def _EXPLORER_ON():
    """Egocentric model + spatial frontier.  /root/EXPLORER_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/EXPLORER_ON")
    except Exception:
        return False


def _UNTRIEDHERE_ON():
    """Prefer an action never tried FROM THIS BOARD.
    /root/UNTRIEDHERE_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/UNTRIEDHERE_ON")
    except Exception:
        return False


def _VERDICT_ON():
    """Avoid a step he has judged BAD.  /root/VERDICT_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/VERDICT_ON")
    except Exception:
        return False


def _VERDICTGOOD_ON():
    """Also PREFER a step judged GOOD over one not yet known.
    Trades exploration for exploitation -- judged separately.
    /root/VERDICTGOOD_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/VERDICTGOOD_ON")
    except Exception:
        return False


def _CYCLEALL_ON():
    """Ask the cycle demotion about the action he actually chose,
    against every action -- not only those untried at his glance
    state.  /root/CYCLEALL_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/CYCLEALL_ON")
    except Exception:
        return False


def _KNOWNPATH_ON():
    """He may follow his own winning path on a level he has already
    completed.  /root/KNOWNPATH_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/KNOWNPATH_ON")
    except Exception:
        return False


def _EXPLROUTE_ON():
    """Lookahead curiosity where no route exists. /root/EXPLROUTE_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/EXPLROUTE_ON")
    except Exception:
        return False


def _NOOPDEMOTE_ON():
    """Demote moves already known to do nothing.  /root/NOOPDEMOTE_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/NOOPDEMOTE_ON")
    except Exception:
        return False


def _ROUTELOCAL_ON():
    """Trust the route only where he has been IN THIS GAME.
    /root/ROUTELOCAL_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/ROUTELOCAL_ON")
    except Exception:
        return False


def _TRIEDRESTORE_ON():
    """Restore _tried from _trans at boot.  /root/TRIEDRESTORE_ON.

    _trans[(s,a)] is ONLY written after executing a, so its presence IS
    the record that he tried it.  _tried is not persisted (line 358,
    and the class docstring already notes "a restart costs hours of
    _tried rebuild"), so every restart makes him believe he has tried
    NOTHING -- novel then contains every action, the untried-preference
    branch takes ~94% of decisions, and the whole scoring path (M/I,
    route, novelty weighting, no-op memory) never runs.
    MEASURED: trans_keys 145,128 restored while unknown_pref and
    mi_earned_steers both sat at 0."""
    try:
        import os as _os
        return _os.path.exists("/root/TRIEDRESTORE_ON")
    except Exception:
        return False


def _QUAREV_ON():
    """Let evidence lookups see the quarantine pool too.

    _quarantine_edge does del self.edges[key] (substrate.py:3060), and
    196,549 transitions_to edges are quarantined against 166,412 live.
    BOTH _deflection (polarity) and _evidence_n (count) look up
    sub.edges, so for a quarantined edge both return 0 -- the M/I
    confidence weight is zero and the score reduces to pure novelty.
    MEASURED: mi_earned_steers 2 in 1,215 executions with
    SEAGI_MI_EARNED=1 set.  Reading the pool does not un-quarantine.
    /root/QUAREV_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/QUAREV_ON")
    except Exception:
        return False


def _QUARVIS_ON():
    """Let _rehydrate_trans read the quarantine pool.  196,549 of his
    world transitions live there vs 166,412 active, and they are the
    more deterministic half.  File-gated at /root/QUARVIS_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/QUARVIS_ON")
    except Exception:
        return False


def _LEADSTO_ON():
    """Write bare->bare `leads_to` so his inference engine can chain
    world reachability.  File-gated at /root/LEADSTO_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/LEADSTO_ON")
    except Exception:
        return False


def _ERRSEEK_ON():
    """Being wrong drives seeking.  File-gated at /root/ERRSEEK_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/ERRSEEK_ON")
    except Exception:
        return False


def _LEADHOLD_ON():
    """Within a life, the controller that made the last executed choice
    keeps precedence while it still has a proposal.  /root/LEADHOLD_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/LEADHOLD_ON")
    except Exception:
        return False


def _SHORTCUT_ON():
    """SHORTCUT (patch 40, 2026-09-11): on a level he has CLEARED, the full
    search runs once at the attempt's first plan; a sequence reaching the
    picture in fewer actions than the rest of the WON route is latched and
    outranks the route for that attempt.  cd82 replayed L3 in 28 and L4 in 42
    actions forever while his own search from the start boards reaches 0 in
    14 and 13.  /root/SHORTCUT_ON."""
    try:
        import os as _os
        return _os.path.exists('/root/SHORTCUT_ON')
    except Exception:
        return False


def _TRACE_ON():
    """Write the reasoning trace: what he saw, what he did, what
    happened.  File-gated at /root/TRACE_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/TRACE_ON")
    except Exception:
        return False


def _INSIGHT_ON():
    """THE PROBLEM YIELDING FEELS GOOD (2026-08-19).  File-gated like
    `_TONE_STEER_ON` so the felt pulse can be interleaved against current
    behaviour without a restart."""
    try:
        return os.path.exists('/root/INSIGHT_ON')
    except Exception:
        return False


_MI_BASE = os.environ.get(
    'SEAGI_MI_BASE', '').strip().lower() in ('1', 'true', 'yes', 'on')
# THE FLIP, WEIGHTED BY WHAT M/I HAS EARNED (2026-07-29, USER AUTHORISED).
# Independent gate; unset -> byte-identical scoring.
_MI_EARNED = os.environ.get(
    'SEAGI_MI_EARNED', '').strip().lower() in ('1', 'true', 'yes', 'on')
# SURVIVAL SEEDS THE ROUTE (2026-07-29, USER AUTHORISED).  Own gate;
# unset -> byte-identical.
_SROUTE = os.environ.get(
    'SEAGI_SROUTE', '').strip().lower() in ('1', 'true', 'yes', 'on')


class WorldActor:
    """Acts to SOLVE a task: leans on success-tagged assumptions, keeps
    curiosity on, fires the life-lean on success, lets lifeforce follow.  The
    output-to-behaviour organ for the 'motor' arbitration loop."""

    SUBSCRIPTIONS = (EventKind.ARBITRATION_DECIDED,)

    def __init__(self,
                 bus: EventBus,
                 world: Any,
                 transducer: Any,
                 grounding: Any,
                 value_provider: Optional[Callable] = None,
                 credit_learning: Optional[Callable] = None,
                 cycle_provider: Optional[Callable] = None,
                 fatigue_effort: Optional[Callable] = None,
                 tone_provider: Optional[Callable] = None,
                 seed: int = 0,
                 enable_facets: bool = False,
                 facet_seed: int = 0):
        self.bus = bus
        self.world = world
        self.transducer = transducer
        self.grounding = grounding
        # HALF B (2026-08-02): his live CHEMICAL state, so the mood the
        # felt-play wire builds can reach the choice.  None -> byte-
        # identical old behaviour.
        self._tone = tone_provider
        self._role_seen = {}
        self._count_seen = {}
        # --- facet docking + IGNITION (similarity-docking build, re-aimed
        # per the 2026-07-21 ignition revision) ---
        # enable_facets: (a) write `has_facet` edges in the attend-and-tag
        #   walk ("tag what HELPS" reused as the write gate, +1.9% edges);
        #   (b) run split-event contrastive earning at first-contact (s,a)
        #   (grounding.dock_observe — the sole strength channel for facet
        #   edges); (c) on TOTAL own-silence (no learned transition for ANY
        #   action at s — the existing genuine-novelty predicate), IGNITE:
        #   publish the best-fitting facet-sharing siblings into attention
        #   as ONE multi-focal AttendedPerceptEvent so their memories are
        #   PRESENT — prediction falls out of RECALL, not out of a computed
        #   token vote (the token-argmax dock was DELETED from the live
        #   path after G2 showed it predicts the wrong quantity).
        self._enable_facets = bool(enable_facets)
        # RNG ISOLATION (C3): facet/ignition machinery owns its OWN
        # generator and — being fully deterministic (argmax + lexicographic
        # ties) — draws from NEITHER it nor the actor's _rng.  Enabling
        # facets therefore consumes zero actor-RNG draws and the directed-
        # exploration sequence stays seed-exact vs the incumbent (pinned by
        # test + gate G1).
        self._facet_rng = random.Random(facet_seed)
        # token -> its percept vector (facets() needs the vector; the paths
        # store tokens).  Populated only when facets are enabled, so the
        # incumbent path is untouched.
        self._state_vec: Dict[str, tuple] = {}
        # proposals where a recognised sibling actually ordered the
        # frontier (a borrowed prior broke the tie).
        self.frontier_ordered: int = 0
        # memoised recognition, keyed by state token.  Without this the
        # carrier scan runs per tick and costs ~7x (measured).
        self._sib_cache = {}
        # the borrowed prior is computed ONCE per state: it is a
        # PRIOR, so a slightly stale one is still a prior.
        self._prior_cache = {}
        # VISIBLE count of exceptions swallowed by the frontier-prior
        # guard.  Non-zero means recognition is silently not reaching
        # the hands -- same contract as reinforce_observer_errors.
        self.frontier_errors: int = 0
        # states whose description (has_facet edges) has been written.
        # NOT _state_vec: execute() also fills that for s_next, so a
        # state first met as a destination would never be described.
        self._faceted = set()
        # steps the world answered with nothing at all.
        self.dead_actions: int = 0
        self._value_provider = value_provider     # () -> ValueLandscape
        self._credit_learning = credit_learning    # (earned: float) -> None
        # Adenosine (part-b v2 2026-07-13): every solved STEP is cognitive
        # EFFORT that deposits sleep-pressure — the felt cost that makes a
        # nap earn its existence during QuestWorld solving.  (weight: float
        # in [1.0,1.5]) -> None.  Wrapped like _credit_learning so a
        # chemistry exception can never kill the solving/bus path.
        self._fatigue_effort = fatigue_effort
        self._cycle_provider = cycle_provider or (lambda: 0)
        self._rng = random.Random(seed)
        self._pending: Optional[Dict[str, Any]] = None
        self._path: List[str] = []          # states visited this attempt
        # SURVIVAL VALENCE (2026-07-29): the (s, a, s_next) triples he actually
        # walked THIS life, and a ring of his OWN past lifespans.  When the end
        # comes it is the only fact about persistence he ever gets; these two
        # let it reach the valuation.  Empty ring = no experience = no lean.
        self._path_sa: Deque[tuple] = deque(maxlen=PATH_SA_CAP)
        self._lives: Deque[int] = deque(maxlen=SOLVE_HISTORY)
        # action -> [n_lives, mean share of a life's steps].  Incremental
        # mean (1/n), so his own history is the only scale.
        self._share: Dict[int, list] = {}
        # state -> [n_lives, mean survival lean of the lives through it].
        # The GRADIENT the route field is seeded from; a flat seed would be a
        # plateau with nothing to follow.
        self._svalue: Dict[str, list] = {}
        self.sroute_seeds = 0
        self.sroute_ms_last = 0.0
        self.survival_walks = 0
        self.survival_steps_tagged = 0
        self.survival_saturated = 0
        self.last_lifespan = 0
        self.last_lean = 0.0
        self._seen: set = set()
        # per-state visit counts -> honest count-based novelty (a revisited
        # state is NOT novel); bookkeeping only, fires no chemistry.
        self._visits: Dict[str, int] = {}
        # The agent's own MAP: learned transitions + the propagated route.
        # _trans[(state, action)] = the state that action led to (his model).
        # WHERE-STREAM (2026-08-13).  The same map, keyed on (state,
        # PLACE) -- because the same 3x3 in two different places is not
        # the same situation, and measuring said so: +12.82 points.
        self._trans_pos = {}
        # PLACE-ONLY: what happens HERE, whatever it looks like.  Small
        # key space (~256 places per game), so it earns fast where the
        # composite key is still cold.  Keyed by GAME because position
        # (3,5) exists on every board.
        self._trans_place = {}
        # THE WORLD MODEL: (state, action) -> what THIS PLACE became.
        # Same key as `_trans`, so both are tested the same number of
        # times and the race between them is fair.
        self._trans_world = {}
        self._world_hits = 0.0
        # the null this model has to beat, and whether it merely IS
        # the null (see patch_world_null.py)
        self._world_static = 0.0
        # THE ONLY NUMBER THAT CAN SHOW SKILL: accuracy restricted to
        # events where the place ACTUALLY CHANGED.  The static null
        # scores exactly 0 there, so it cannot be borrowed.  This is
        # the live analogue of the harness PRED_CHG.
        self._world_chg_n = 0.0
        self._world_chg_hits = 0.0
        self._world_says_static = 0.0
        self._world_n = 0.0
        self.world_guess_used = 0
        self._place_hits = 0.0
        self._place_n = 0.0
        self.place_guess_used = 0
        self._where_grain = 4
        self._where_last = None
        # earned precedence, per game: hit rate x n/(1+n), the same
        # self-scaling form the self-model confidence already uses
        self._pos_hits = 0.0
        self._pos_n = 0.0
        self._tex_hits = 0.0
        self._tex_n = 0.0
        self.pos_guess_used = 0
        # _route[state] = value-toward-goal: the success NT-signal SPREAD back
        # through _trans so a followable route emerges (not just a point tag).
        self._trans: Dict[tuple, str] = {}
        # CORROBORATION, NOT RECENCY (2026-08-01, MEASURED).  A (s,a) pair
        # has a mean of 4.29 distinct observed successors; the bare overwrite
        # above kept whichever arrived LAST.  This counts them so the STORED
        # successor is the most-corroborated one.  Not persisted: it rebuilds
        # within a life, and rehydration seeds it at 1.
        self._trans_n: Dict[tuple, Dict[str, int]] = {}
        self.trans_writes: int = 0
        self.trans_flips: int = 0          # NEW rule: stored successor changed
        self.trans_recency_flips: int = 0  # OLD rule: s_next != stored
        # REACHABILITY, kept apart from PREDICTION (2026-07-31).  `_trans` is
        # his single best guess for "where will this action land me" and must
        # stay single-valued.  Value propagation is a different question: a
        # path he has LIVED exists whether or not it was the most recent thing
        # that (s,a) did.  Measured: collapsing the two drops the goal-reaching
        # set from 5,677 states to 6.
        self._trans_all: Dict[str, set] = {}
        self._route: Dict[str, float] = {}
        # His map is PERSISTENT: _trans is refilled once from the
        # substrate, so a life does not begin by re-exploring a
        # world he has already mapped.
        # WHAT HE HAS TRIED THIS LIFE -- distinct from what he
        # KNOWS (_trans, which is rehydrated from the substrate).
        # `novel` keys on THIS, so 'every mistake once' is per-life
        # and survives the rehydration of his map.
        self._tried: set = set()
        # WAS THIS (s,a) PREDICTED CORRECTLY LAST TIME?  Process-local and
        # bounded by the (s,a) count; it rebuilds within a life.  Only used
        # to detect a regularity that was FAILING and now HOLDS.
        self._pred_ok: Dict[tuple, bool] = {}
        self.insight_fires: int = 0
        # SHADOW COUNTERS (2026-08-20): candidate fire conditions, counted
        # and discarded, so the true live rate on ARC can be read before
        # anything is switched on.  Same pattern as tone_scored/_engaged.
        self._ins_A: int = 0        # deployed rule: conf beats his global
        self._ins_B: int = 0        # repetition + dominance, no global avg
        self._ins_C: int = 0        # B, and his GLANCE moved too
        self._ins_stasis: int = 0   # the sb26 case: world moved, gaze did not
        # HIS TYPICAL INSIGHT SIZE.  An EWMA of how far a settling key's
        # confidence beats his global average, so the magnitude of an
        # insight can be measured against his own norm instead of being
        # a flat 1.0.  Small understandings then give small doses.
        self.leads_written: int = 0
        self.leads_errors: int = 0
        self._leads_seen = set()
        self.seek_fires: int = 0
        self._err_conf: float = 0.0
        self._err_conf_n: int = 0
        self.trace_written: int = 0
        # a readable window onto the record; the persistent one is
        # substrate.episodes
        self._trace_ring = __import__("collections").deque(maxlen=12)
        self.trace_errors: int = 0
        self._ins_exc: float = 0.0
        self._ins_exc_n: int = 0
        # RULE-LAYER candidates (2026-08-20): about the WORLD, not his gaze.
        self._ins_R1: int = 0       # a new rule was born
        self._ins_R2: int = 0       # a rule learned ELSEWHERE first held here
        self._prev_rules: int = -1
        self._prev_transfers: int = -1
        # WORLD prediction error -> VTA (2026-08-20).  Shadow first.
        self._wpe_pos: int = 0
        self._wpe_neg: int = 0
        self._wpe_pos_mag: float = 0.0
        self._wpe_neg_mag: float = 0.0
        self._wpe_fired: int = 0
        self._ins_bankA: dict = {}
        self._ins_bankB: dict = {}
        self._ins_bankC: dict = {}
        # VALUE KEYS (2026-07-28): (s,a) -> sign of its M/I lean.
        # A state's keys are its folders; membership is partial and
        # overlapping, so two states can share 'action 2 pays here'
        # and nothing else.  Measured held-out lift 1.6x-6.0x vs
        # 1.06x for appearance keys (which have a MEDIAN OF ONE
        # MEMBER and were never categories).
        # INDEXED BY STATE: a full scan of every recorded lean on
        # every step slowed the tick loop enough to break a
        # time-based chemistry test (29 vs 28 baseline).  Keys per
        # state is O(actions), so both paths stay O(1)-ish.
        self._skeys: Dict[str, dict] = {}
        self._vk: Dict[tuple, list] = {}
        self.vkey_priors: int = 0
        # class -> [sum of leans experienced there, n].  The value
        # of a FOLDER, which exists without any completion.
        self._cval: Dict[frozenset, list] = {}
        self.class_value_steers: int = 0
        # what a state is worth given the BEST it can reach --
        # class value carried backward along what he has walked.
        self._chain: Dict[str, float] = {}
        self.chain_backups: int = 0
        # states whose value keys were INHERITED from past lives.
        self.learning_restored: int = 0
        self._rehydrated: bool = False
        # states whose goal-value was inherited from a past life.
        self.route_restored: int = 0
        self.trans_rehydrated: int = 0
        self.trans_rehydrated_quar: int = 0
        # (state, action) -> [tries, times the BOARD actually moved].
        # Keyed on world_changed, NOT on his glance token: the token
        # follows his locus and changes even when nothing happened.
        self._noop_seen = {}
        self._prog_n: int = 0
        self._prog_mean: float = 0.0
        self.progress_fires: int = 0
        self._explore_route = {}
        self._explore_cyc: int = -10 ** 9
        self._explore_level: int = -1
        self.explore_seeds: int = 0
        self.explore_recomputes: int = 0
        self.rehydrate_errors: int = 0
        # --- diagnostics / the learning curve (read-only) ---
        self.proposals: int = 0
        self.executions: int = 0
        # Exploration counter SPLIT (Fix 2, 2026-07-21): a genuine-novelty
        # step (acted on an un-modelled (s,a)) vs a memory-directed FALLBACK
        # step (route-less, least-visited).  The legacy `explorations` =
        # their sum (property below) — so an A/B gate can detect an
        # exploration regression the old single counter (incremented in
        # BOTH branches) could not.
        self.explorations_directed: int = 0
        self.explorations_fallback: int = 0
        # The branch that actually USES the gradient had no counter, so
        # "does he follow the route?" was unanswerable.  `route_ties` counts
        # decisions where the top score is shared -- a field that is present
        # but FLAT steers nothing even when the branch runs.
        self.explorations_route: int = 0
        self.route_ties: int = 0
        # Same three counters, restricted to DEPTH -- decisions taken while
        # `levels_completed > 0`, i.e. on level 2+.  The globals mix level 1
        # (where he succeeds) with level 2 (where he never does), so they
        # cannot show which branch is driving the failure.
        self._cur_level: int = 0
        self._deep_directed: int = 0
        self._deep_fallback: int = 0
        self._deep_route: int = 0
        # GOAL-SEED PROBE (read-only).  `_route[s_next]=1.0` fires on success
        # and s_next is the POST-step state, while arc_world advances the
        # level in the SAME step -- so a level-completing win may stamp the
        # goal on the FIRST STATE OF THE NEW LEVEL, and he would then route-
        # follow back to the doorway he came in through.  `_route` PERSISTS,
        # so old seeds count too: track the whole seed set, not just the
        # freshly stamped one.  Nothing but to_dict reads these.
        self._goal_seed = None
        self._goal_seed_was_levelup: bool = False
        self._goal_seed_level: int = -1
        self._seed_states = set()
        # levels for which he actually holds a doorway stamp.  A seed
        # only steers on the level it opens (see _route gating below).
        self._seed_levels = set()
        self._deep_seed_revisits: int = 0
        self._deep_anyseed_revisits: int = 0
        # Ignition diagnostics (read-only).
        self.ignitions: int = 0
        self.ignition_siblings_last: int = 0
        self.successes: int = 0
        self.failures: int = 0
        self.best_steps: Optional[int] = None
        self.challenges_mastered: int = 0
        self.solve_steps: Deque[int] = deque(maxlen=SOLVE_HISTORY)

    # ---- composite (state, action) token ----
    @staticmethod
    def _comp(state: str, action: int) -> str:
        return f'{state}|a{int(action)}'

    def _value_of(self, token: Optional[str]) -> float:
        if not token or self._value_provider is None:
            return 0.0
        try:
            vp = self._value_provider()
            return float(vp.value_of(token)) if vp is not None else 0.0
        except Exception:
            return 0.0

    def _tone_blend(self, sc, nov, damp, n):
        """Blend ANY branch's score toward the unvisited in proportion
        to his mortality lean.  `nov` is rescaled to the score's own
        range so the blend is unit-free; damp=0 returns sc unchanged.
        Always computed, only USED when the gate is on -- so the wire
        measures whether it can flip the argmax before it steers."""
        if damp <= 0.0 or not sc:
            return sc
        _mx = max(sc)
        if _mx <= 0.0:
            return sc
        return [(1.0 - damp) * sc[a] + damp * max(0.0, nov[a]) * _mx
                for a in range(n)]

    def _tone_observe(self, sc, sc_t, n, damp):
        self.tone_scored = getattr(self, 'tone_scored', 0) + 1
        if damp > 0.0:
            self.tone_engaged = getattr(self, 'tone_engaged', 0) + 1
        try:
            if sc and sc_t:
                a0 = min(i for i in range(n) if sc[i] == max(sc))
                a1 = min(i for i in range(n) if sc_t[i] == max(sc_t))
                if a0 != a1:
                    self.tone_flips = getattr(self, 'tone_flips', 0) + 1
        except Exception:
            pass

    def _feel(self, kind, magnitude, target, cyc, detail):
        """Fire a feeling -- and ALWAYS let curiosity in with it.

        User, 2026-08-03: *"all of the emotions let in curiosity to move
        forward and change emotional perception of the situation"*.
        Curiosity is not one feeling among others; it is what stops any
        feeling becoming TERMINAL -- the opening through which the
        appraisal can be revised and he can move.  So it is a companion
        to EVERY affective fire, structurally, and any feeling added
        later inherits it for free instead of needing its own branch.

        It cannot flood: the companion's strength is how much room there
        is to see this place differently -- `1/(1+visits)` -- so in worn
        territory it is a whisper and on the frontier it is loud.  Same
        self-referential form used everywhere else; no constant enters.
        """
        try:
            self.bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE, cycle=int(cyc),
                source_capability='world_actor', origin='internal',
                origin_detail=detail, chemistry_kind=kind,
                magnitude=float(magnitude), target_concepts=[target]))
            if kind != 'curiosity':
                self.bus.publish(ChemistryEvent(
                    kind=EventKind.CHEMISTRY_FIRE, cycle=int(cyc),
                    source_capability='world_actor', origin='internal',
                    origin_detail=detail + '+curious',
                    chemistry_kind='curiosity',
                    magnitude=float(
                        1.0 / (1.0 + self._visits.get(target, 0))),
                    target_concepts=[target]))
                self.companion_curiosity = getattr(
                    self, 'companion_curiosity', 0) + 1
        except Exception:
            pass

    # ---- 1. propose: lean on the assumption, but curiosity STAYS ----
    # What he tried last time he stood here -- so "change something"
    # can mean something concrete rather than "act at random".
    _felt_last_action: dict = {}

    def _felt_change(self, s, a, n):
        """When he is not doing well, do something different here.

        USER DOCTRINE 2026-08-29: *"he wants to feel good... they will do
        anything to change something to get back to feeling good."*  The
        felt state is the error signal of that loop; before this it
        ordered nothing -- every term in `propose` was navigational or
        novelty-based, and `_onward_valence`, the only thing here with
        "valence" in its name, counts visits rather than feeling.

        Changes the action ONLY when all of: the felt state is below
        good, he has stood on this state before, and he is about to
        repeat what he did last time.  Prefers something untried here;
        falls back to the next action so "change something" always means
        something.  Returns the action to take -- unchanged on any doubt,
        so this can only ever re-order, never remove an option.
        """
        low = False
        try:
            _fn = getattr(self.world, "felt_low", None)
            if _fn is not None:
                low = bool(_fn())
        except Exception as exc:
            # `getattr(obj, name, default)` swallows ONLY AttributeError.
            # CurriculumWorld.__getattr__ ends in `worlds[idx]`, so a
            # rotation can raise IndexError straight through the default
            # -- 224 errors in 16,820 steps (1.3%) with no other
            # explanation.  Recorded by type rather than guessed at.
            self._felt_err(exc)
            return a
        if low:
            self.felt_low_n = getattr(self, "felt_low_n", 0) + 1
        try:
            last = self._felt_last_action.get(s)
        except Exception as exc:
            self._felt_err(exc)
            return a
        if low and last is not None:
            self.felt_seen_n = getattr(self, "felt_seen_n", 0) + 1
        if low and last is not None and n > 1 and last == a:
            self.felt_repeat_n = getattr(self, "felt_repeat_n", 0) + 1
            try:
                a = self._felt_pick(a, n)
                self.felt_changes = getattr(self, "felt_changes", 0) + 1
            except Exception as exc:
                self._felt_err(exc)
        try:
            self._felt_last_action[s] = a
        except Exception as exc:
            self._felt_err(exc)
        return a

    def _rel_key(self):
        """(game, level, situation) from the world, or None."""
        _rc = getattr(self.world, 'rel_context', None)
        _ctx = _rc() if _rc is not None else None
        if _ctx is None:
            return None
        _g = str(getattr(self.world, 'game_id', ''))
        _lv = int(getattr(self.world, '_levels', 0) or 0)
        return (_g, _lv, tuple(_ctx))

    def _rel_pick(self, n):
        """The action with the best learned movement of the pursued
        relation in this situation, or None (fall through)."""
        if not hasattr(self, '_rel_act'):
            self._rel_act = {}
        key = self._rel_key()
        self._rel_ctx = key
        if key is None or int(n) < 2:
            return None
        # A PICTURE IS NOT STEERED HERE.  Review 2026-09-05: on cd82 the
        # only action that ever moves the picture is "apply", whose mean
        # stays positive-tiny on hundreds of no-op presses (a running
        # mean fed zeros never crosses zero), so this would have pressed
        # apply on 96% of steps.  Completing a picture needs an operator
        # model (colour x mask -> canvas), not a per-action mean.
        if key[2] and key[2][0] == 'eq':
            return None
        self.rel_asks = getattr(self, 'rel_asks', 0) + 1
        _inert = getattr(self.world, 'board_inert_here', None)
        best = None
        for x in range(int(n)):
            e = self._rel_act.get(key + (int(x),))
            if e is None or e[1] < 2 or e[0] <= 0.0:
                continue
            # what he already knows does nothing HERE is not taken,
            # whatever it did elsewhere in this situation
            try:
                if _inert is not None and _inert(int(x)):
                    continue
            except Exception:
                pass
            if best is None or e[0] > best[1]:
                best = (int(x), e[0])
        if best is not None and best[1] > 0.0:
            self.rel_steers = getattr(self, 'rel_steers', 0) + 1
            return best[0]
        return None

    def _rel_learn(self, a, result):
        """Running mean of (progress - regress) of the pursued relation
        for the action taken in the situation he was in."""
        if not hasattr(self, '_rel_act'):
            self._rel_act = {}
        key = getattr(self, '_rel_ctx', None)
        if key is None or int(result.get('rel_n') or 0) <= 0:
            return
        d = (float(result.get('rel_progress') or 0.0)
             - float(result.get('rel_regress') or 0.0))
        k = key + (int(a),)
        e = self._rel_act.get(k)
        if e is None:
            if len(self._rel_act) > 20000:
                self._rel_act.clear()
            e = [0.0, 0]
            self._rel_act[k] = e
        e[1] += 1
        e[0] += (d - e[0]) / float(e[1])
        self.rel_learned = getattr(self, 'rel_learned', 0) + 1

    def _felt_pick(self, a, n):
        """Something other than `a` to try here.

        `untried_here` returns a LIST -- "empty list when everything here
        has been tried" -- and this code called `int()` on it.  Measured
        live: `felt_repeat_n` 2,621 and `felt_err_kinds
        {'TypeError': 2621}`, exactly equal, so the steer reached its
        innermost branch 2,621 times and threw every single time.
        `felt_changes` stayed 0 -- it has never once completed.

        ⚠ The unit test stubbed `untried_here` as returning an int, so it
        encoded the same wrong assumption and passed.  A stub must match
        the real signature or it tests the mistake.
        """
        cand = None
        try:
            cand = self.world.untried_here(n)
        except Exception:
            cand = None
        if isinstance(cand, (list, tuple, set)):
            opts = [int(c) for c in cand
                    if isinstance(c, int) and 0 <= int(c) < n and int(c) != a]
            if opts:
                return opts[0]
        elif isinstance(cand, int) and 0 <= cand < n and cand != a:
            return int(cand)
        # "Change something" must always mean something.
        return (int(a) + 1) % int(n)

    def _felt_err(self, exc):
        """Count the failure AND remember what kind it was.

        A bare `except Exception: n += 1` told me 224 things went wrong
        and nothing about what, which cost an evening of hypotheses.
        """
        self.felt_errors = getattr(self, "felt_errors", 0) + 1
        kinds = getattr(self, "felt_err_kinds", None)
        if kinds is None:
            kinds = self.felt_err_kinds = {}
        k = type(exc).__name__
        kinds[k] = kinds.get(k, 0) + 1

    def propose(self, cyc: int) -> Optional[int]:
        try:
            vec = self.world.percept().get('world_vector')
            s = self.transducer.encode(vec)
        except Exception:
            return None
        if self._enable_facets:
            self._state_vec[s] = tuple(vec)     # facets() needs the vector
        # WHERE HE IS LOOKING, coarsened.  Captured with the state it
        # belongs to so a later learn cannot pair it with a different
        # one; the pairing is checked by name before use.
        _w = self._where_code()
        self._where_last = (s, _w)
        n = int(getattr(self.world, 'n_actions', 4))
        # HE REMEMBERS THE WAY OUT.  On a level he has ALREADY
        # completed, this board may be one he stood on during a run
        # that finished it -- so take what worked.  The world is
        # 94.4% deterministic given the action AND its aim, so the
        # next board is the next board on the path.
        #
        # ORDER ONLY, NEVER MEMBERSHIP: off the path this returns
        # None and everything below runs untouched.  Only completed
        # levels have a path, so the frontier is always played fresh.
        # REMEMBERED, NOT RETURNED.  `propose` does not act by
        # returning -- runtime.py discards the value; the action is
        # executed via the CapabilityClaimEvent published at the END.
        # Returning here skipped the publish, so he made no claim and
        # DID NOT MOVE, then matched again next tick: a permanent
        # stall on any board a sealed path covers.  Applied below,
        # just before the publish.
        _known_a = None
        if _KNOWNPATH_ON():
            try:
                _kf = getattr(self.world, "known_action", None)
                _ka = _kf() if _kf is not None else None
                if _ka is not None and 0 <= int(_ka) < n:
                    _known_a = int(_ka)
            except Exception:
                _known_a = None
        # Fix 2 (2026-07-21) — EXPLORATION IS STRUCTURALLY PROTECTED: the
        # guess for each action is his OWN learned transition ONLY.  The
        # class out-wire (grounding._predict) has LEFT propose ENTIRELY, so
        # nothing external can make guess[a] non-None and thereby empty
        # `novel` — the failure mode that would have killed "every mistake
        # once" (the property that cracked Hanoi-6).  Facets/ignition touch
        # this choice NOWHERE: ignition publishes AFTER the action is
        # chosen, into ATTENTION, not into the policy.
        if not self._rehydrated:
            self._rehydrate_trans()
            self._rehydrate_doorways()
        guess = [self._trans.get((s, a)) for a in range(n)]
        # THE WHERE-STREAM ANSWERS FIRST where it has an answer AND has
        # earned precedence in this game; elsewhere the texture-only map
        # replies exactly as it does today.  `guess[a] is None` stays
        # None if NEITHER knows -- so "unknown == maximally novel" and
        # `novel`/`_tried` below are byte-identical.
        if _w is not None:
            _tc = self._tex_conf()
            _pc = self._pos_conf()
            _lc = self._place_conf()
            _gk = self._game_key()
            # Best-earned source answers where it HAS an answer.  Order
            # matters only through the earned rates, and a source with
            # no measurement scores 0, so nothing untested displaces
            # anything tested.  guess[a] stays None when NO source
            # knows, so "unknown == maximally novel" is untouched.
            _wc = self._world_conf()
            if _wc > _tc and _wc >= _lc and _wc >= _pc:
                # THE WORLD ANSWERS.  Only where it has earned a better
                # measured rate than his gaze map, per the same rule
                # every other source here follows.
                for _a in range(n):
                    _wg = self._trans_world.get((s, _a))
                    if _wg is not None:
                        if guess[_a] != _wg:
                            self.world_guess_used += 1
                        guess[_a] = _wg
            elif _lc > _tc and _lc >= _pc:
                for _a in range(n):
                    _qg = self._trans_place.get(((_gk, _w), _a))
                    if _qg is not None:
                        if guess[_a] != _qg:
                            self.place_guess_used += 1
                        guess[_a] = _qg
            elif _pc > _tc:
                for _a in range(n):
                    _pg = self._trans_pos.get(((s, _w), _a))
                    if _pg is not None:
                        if guess[_a] != _pg:
                            self.pos_guess_used += 1
                        guess[_a] = _pg
        # Genuine novelty = no learned transition of his OWN.  Nothing can
        # empty this set from the outside.
        # THE MOOD DECIDES HOW MUCH HE TRUSTS HIS OWN MODEL (half B).
        # `m_polarity`/`i_polarity` are the mortality- and immortality-
        # leaning halves of his LIVE chemistry, so this is M/I steering
        # through the chemical proxy -- not a term anyone invented.
        # While the I-lean dominates (his normal state) _damp is 0 and
        # every score below is BYTE-IDENTICAL to today's.  Only when he
        # is genuinely more mortality-leaning than immortality-leaning --
        # stuck, cortisol built by puzzle_stress -- does he discount his
        # model and swing toward the unvisited: the 'flip a switch from
        # idle to explore again'.  Scale-free: gap over total.
        _damp = 0.0
        try:
            if self._tone is not None:
                _t = self._tone() or {}
                _ip = float(_t.get('i_polarity', 0.0))
                _mp = float(_t.get('m_polarity', 0.0))
                # MEASURED 2026-08-02: `_mp > _ip` NEVER held (0.157 vs
                # 0.230, tone_engaged 0 over 22 scored choices).  The two
                # aggregates have DIFFERENT RESTING LEVELS -- the I-channels
                # (dopamine/serotonin/oxytocin/endorphins) simply sit higher
                # than the M-channels -- so comparing them absolutely asks a
                # structural question, not an informative one.  The right
                # question is self-referential, the same shape puzzle_stress
                # already uses: is he more mortality-leaning THAN HE USUALLY
                # IS?  Exact running mean + mean-absolute-deviation (Welford
                # form, count-based), so no constant and no tuned window.
                _d = _mp - _ip
                self._lean_n = getattr(self, '_lean_n', 0.0) + 1.0
                _lm = getattr(self, '_lean_mean', 0.0)
                _lm += (_d - _lm) / self._lean_n
                self._lean_mean = _lm
                _ld = getattr(self, '_lean_mad', 0.0)
                _ld += (abs(_d - _lm) - _ld) / self._lean_n
                self._lean_mad = _ld
                if self._lean_n >= 2.0 and _ld > 0.0:
                    _damp = min(1.0, max(0.0, (_d - _lm) / _ld))
        except Exception:
            _damp = 0.0
        novel = [a for a in range(n) if (s, a) not in self._tried]
        if novel:
            # FRONTIER ORDERING (2026-07-25) — recognition reaches the
            # hands.  It sets the ORDER untried actions are tried in,
            # NEVER which are untried: `novel` above is _trans-only and
            # untouched, so the membership failure that retired the old
            # class->action wire is structurally closed and every
            # mistake is still made once (at most deferred).
            # Prior = sum_sp fit(sp)*polarity(sp's own (sp,a) edge);
            # fit = shared/|facets(s)|, derived, no threshold.  POLARITY
            # ONLY — magnitude is NOT rented from effective_strength,
            # which the world_observe ratchet inflates by re-visiting.
            # Cheapest path: only computed when there is actually an
            # ordering decision (>= 2 untried).  No recognition or no
            # grounded evidence -> empty -> the EXACT prior draw.
            # DRAW FIRST, always from `novel`: random.choice consumes a
            # variable number of bits for different list lengths, so
            # choosing from a recognition-narrowed list would perturb his
            # generator and break the standing guarantee that the facet
            # machinery draws NOTHING from the actor RNG.  Drawing first
            # keeps that byte parity for free.
            a = self._rng.choice(novel)
            pri = (self._frontier_prior(s, vec, novel, int(cyc))
                   if len(novel) >= 2 else {})
            vpri = self._value_prior(s, novel) if len(novel) >= 2 else {}
            if pri or vpri:
                # TIE-BREAK, NEVER A SUM.  Summing the value prior (a polarity
                # in [-1,1]) into `pri` (whose units are its own) let one
                # swamp or invert the other: cortisol ROSE 0.120 -> 0.159
                # where it must decay, i.e. he experienced MORE mortality.
                # So `pri` decides first in its own units, and `vpri` only
                # separates actions `pri` ranks EQUAL.  With vpri empty this
                # is byte-identical to the previous behaviour.
                _pb = max((float(pri.get(x, 0.0)) for x in novel), default=0.0)
                _cand = [x for x in novel if float(pri.get(x, 0.0)) == _pb]
                if len(_cand) > 1 and vpri:
                    _vb = max(float(vpri.get(x, 0.0)) for x in _cand)
                    _cand = [x for x in _cand if float(vpri.get(x, 0.0)) == _vb]
                if a not in _cand:
                    # deterministic, no extra draw.  Membership is untouched
                    # -- every untried action is still tried, at most later.
                    a = min(_cand)
            # BODY NOVELTY ORDERS THE FRONTIER TOO (2026-08-03).
            # The blend was first put only in the fallback branch -- which
            # runs ~2.4% of steps in steady state and almost never just after
            # a restart, when `_tried` is empty and THIS branch takes ~94%.
            # Measured consequence: body_nav_steers stayed 0 with self
            # confidence already at 0.927.  Same error as the tone-steer,
            # twice.
            #
            # ORDER ONLY, NEVER MEMBERSHIP: `novel` is untouched above, so
            # every untried action is still tried -- at most later.  That
            # keeps "every mistake once", the property that cracked Hanoi-6.
            # Weighted by what the self-model has EARNED in this game, so a
            # murky self changes nothing.
            try:
                _nb = getattr(self, '_nov_body', None)
                _cf = float(getattr(self, '_self_conf', 0.0) or 0.0)
                if _nb and _cf > 0.0 and len(novel) >= 2:
                    _bb = max(float(_nb.get(x, 0.0)) for x in novel)
                    _bcand = [x for x in novel
                              if float(_nb.get(x, 0.0)) >= _bb - 1e-9]
                    if _bcand and a not in _bcand:
                        a = min(_bcand)
                        self.body_nav_steers = getattr(
                            self, 'body_nav_steers', 0) + 1
                    _rest = _bcand or novel
            except Exception:
                pass
            # HE ALREADY LEARNED THIS DOES NOTHING HERE.
            # `_trans[(s,a)] == s` is a self-loop -- his own record
            # that the move changed nothing.  MEASURED: 19.7% of his
            # actions re-do exactly such a move.  ORDER ONLY: never
            # removed, so every mistake is still made ONCE -- just not
            # twenty times.  Untried actions have guess None, so
            # exploration is untouched.
            try:
                if _NOOPDEMOTE_ON():
                    _np = (locals().get("_rest") or novel)
                    if len(_np) >= 2:
                        # inert here == tried at least twice and the
                        # BOARD never moved.  Untried pairs are absent
                        # from the store, so exploration is untouched.
                        _alive = [x for x in _np
                                  if not self._is_inert_here(s, x)]
                        if _alive and len(_alive) < len(_np):
                            if a not in _alive:
                                a = min(_alive)
                                self.noop_demotes = getattr(
                                    self, "noop_demotes", 0) + 1
                            _rest = _alive
            except Exception:
                pass
            # DO NOT WALK BACK INTO A ROOM YOU HAVE BEEN IN.
            # MEASURED: he returns to an already-visited BOARD on 65% of
            # moves; cheap completions run 0.95 distinct boards per action,
            # expensive ones 0.11.  The board recurs 65% where his glance
            # token recurs 13%, so this is keyed on the board.  A pair that
            # has ALWAYS led backwards accounts for 10.9% of all actions.
            # ORDER ONLY: >=2 observations required, never removed, so
            # every mistake is still made once and he always has a move.
            try:
                if hasattr(self.world, "cycles_here"):
                    _cp = (locals().get("_rest") or novel)
                    if len(_cp) >= 2:
                        _fwd = [x for x in _cp
                                if not self.world.cycles_here(x)]
                        if _fwd and len(_fwd) < len(_cp):
                            if a not in _fwd:
                                a = min(_fwd)
                                self.cycle_demotes = getattr(
                                    self, "cycle_demotes", 0) + 1
                            _rest = _fwd
            except Exception:
                self.cycle_errors = getattr(self, "cycle_errors", 0) + 1
            # LOOKAHEAD CURIOSITY ORDERS THE FRONTIER TOO.
            # On a level he has never completed the route is silent,
            # and local novelty has ONE step of lookahead -- it
            # oscillates once every neighbour is visited.  The
            # explore route carries 0.9**d back from the frontier,
            # so a distant unexplored region pulls him across known
            # ground.  ORDER ONLY, NEVER MEMBERSHIP -- same contract
            # as the body-novelty stage above.
            try:
                if _EXPLROUTE_ON() and self._explore_route:
                    _ep = (locals().get("_rest") or novel)
                    if len(_ep) >= 2:
                        _er = self._explore_route
                        _eb = max(float(_er.get(guess[x], 0.0))
                                  for x in _ep)
                        if _eb > 0.0:
                            _ec = [x for x in _ep
                                   if float(_er.get(guess[x], 0.0))
                                   >= _eb - 1e-9]
                            if _ec and a not in _ec:
                                a = min(_ec)
                                self.explore_steers_live = getattr(
                                    self, "explore_steers_live", 0) + 1
                            _rest = _ec or _ep
            except Exception:
                self.explore_errors = getattr(
                    self, "explore_errors", 0) + 1
            # THE FRONTIER BREAKS WHAT BODY NOVELTY LEAVES EQUAL
            # (2026-08-15).  Scored HERE because this is the only point
            # at which the executed aim is knowable: `_locus` is set by
            # the `percept()` in `propose()` and nothing reassigns it
            # before `step()` reads it.  v2 scored inside `step()`,
            # before its own `percept()`, and matched the executed aim
            # 0.7% of the time.
            #
            # ORDER ONLY, NEVER MEMBERSHIP -- every untried action is
            # still tried, at most later.  And it narrows only the set
            # body novelty is INDIFFERENT about, so it cannot take over
            # the way v2 did (it overrode the body channel on ~75% of
            # decisions while its own gauge read healthy).
            try:
                _iw = getattr(self.world, 'world', None)
                _fs = (_iw.frontier_scores(novel)
                       if hasattr(_iw, 'frontier_scores') else None)
                # REAL CASCADE: what body novelty left, else what the
                # pri/vpri stage left, else `novel`.  The comment used
                # to claim the first and the code did the last.
                _pool = (locals().get('_rest')
                         or locals().get('_cand') or novel)
                if _fs and len(_pool) >= 2:
                    _fb = max(float(_fs.get(x, 0.0)) for x in _pool)
                    _fc = [x for x in _pool
                           if float(_fs.get(x, 0.0)) >= _fb - 1e-9]
                    if _fc and a not in _fc:
                        a = min(_fc)
                        self.frontier_steers = getattr(
                            self, 'frontier_steers', 0) + 1
                        # WHICH WAY did it steer?  The click key carries
                        # the varying aim, so it recurs far less (61.6%
                        # vs 83.4%) and sits at 1.0 more often -- and
                        # `min()` favours low indices, where index 0 IS
                        # the coordinate action two thirds of the time.
                        # So the stage can drift onto clicking while a
                        # count of steers reads perfectly healthy.
                        try:
                            _ac = getattr(_iw, '_acts', None) or []
                            if 0 <= a < len(_ac) and int(_ac[a]) == 6:
                                self.frontier_steers_click = getattr(
                                    self, 'frontier_steers_click', 0) + 1
                            else:
                                self.frontier_steers_move = getattr(
                                    self, 'frontier_steers_move', 0) + 1
                        except Exception:
                            pass
            except Exception:
                pass
            self.explorations_directed += 1
            if self._cur_level > 0:
                self._deep_directed += 1
        else:
            # FOLLOW THE ROUTE, now STEERED by grounded M/I on the (s,a)
            # transition edge (Stage 3): score = base * (1 + deflection),
            # deflection = polarity(edge.evidence)*effective_strength.  M-edges
            # (known walls) suppress -> curiosity redirects to other actions;
            # I-edges (frontier) amplify.  defl=0 when the edge has no grounded
            # evidence -> reduces to the exact prior selection.  Untried actions
            # live in `novel` (handled above) so steering NEVER suppresses an
            # unexplored action (every-mistake-once intact).
            defl = [self._deflection(s, a, guess[a], int(cyc))
                    for a in range(n)]
            # A DOORWAY OPENS ONE LEVEL.  He is playing level
            # `_cur_level + 1`, so only a stamp earned by COMPLETING
            # that level describes where its exit is.  On a level he has
            # never finished he holds no such stamp, the route is silent,
            # and `max(vals) <= 0` hands him to curiosity -- correct on
            # ground he has never seen.  Without this the level-1 stamp
            # (which sits at the level-2 SPAWN) pins him to his own feet
            # and the curiosity fallback can never fire.
            if _EXPLROUTE_ON():
                self._maybe_propagate_explore(int(self._cur_level))
            _route_live = int(self._cur_level) in self._seed_levels
            # HIS ROUTE IS GLOBAL; HIS GAMES ARE NOT.  `_route` is one
            # dict keyed by state token, and the token is a 3x3 glance
            # that recurs across games -- so a gradient earned in sp80
            # steers him in r11l where it means nothing.  MEASURED: the
            # same game+level costs 41 actions once and 871 another,
            # 21-39x spread with NO trend across five repeats.  Trust a
            # route value only where he has actually been IN THIS GAME.
            # This removes misinformation; it adds no signal.
            _rl = False
            try:
                _rl = (_ROUTELOCAL_ON()
                       and hasattr(self.world, "seen_here"))
            except Exception:
                _rl = False
            if _rl:
                _blocked = 0
                for _i in range(n):
                    _gi = guess[_i]
                    if (_gi is not None
                            and self._route.get(_gi, 0.0) > 0.0
                            and not self.world.seen_here(_gi)):
                        _blocked += 1
                if _blocked:
                    self.route_local_blocks = getattr(
                        self, "route_local_blocks", 0) + _blocked
            vals = [self._route.get(guess[a], 0.0)
                    if (_route_live and guess[a] is not None
                        and (not _rl
                             or self.world.seen_here(guess[a])))
                    else 0.0 for a in range(n)]
            # THE FULL PICTURE COMES FIRST (2026-08-06).  A route is
            # worth following only through ground he actually knows; a
            # state with an unmodelled action is a hole in the map, and
            # holes are what curiosity is FOR.  Self-terminating: once
            # every action here has an effect he knows, this is False
            # for this state forever and the route governs again.
            _hole_here = any(guess[a] is None for a in range(n))
            if _hole_here and max(vals) > 0.0:
                self.wander_incomplete = getattr(
                    self, 'wander_incomplete', 0) + 1
            if max(vals) <= 0.0 or _hole_here:
                # MEMORY-DIRECTED FALLBACK, STEERED: least-visited AND not a
                # known wall.  At defl=0 this is argmax(1/(1+visits)) =
                # min-visits = the prior fallback.
                # AN UNKNOWN ACTION IS THE MOST INTERESTING ONE, NOT THE
                # LEAST (2026-08-05).  `guess[a] is None` means he has
                # never learned what this action does here -- and it was
                # scored -1.0, the LOWEST value, so the thing he knew
                # nothing about was the thing he most avoided.  That is
                # inverted curiosity, and it is why he hammered a single
                # button in 7 of 25 games: with no action variety,
                # contingency had nothing to correlate, so no self, no
                # action->effect model, and no navigation.
                # 1.0 is not a new constant -- it is what this very
                # formula yields for a destination visited zero times.
                # Unknown == maximally novel.
                nov = [(1.0 / (1.0 + self._visits.get(guess[a], 0)))
                       if guess[a] is not None else 1.0 for a in range(n)]
                # KNOWN SELF-LOOP -> lowest rank among the scored.
                # Not zeroed and not removed: it stays selectable if
                # every option here is inert, so nothing is closed off.
                if _NOOPDEMOTE_ON():
                    _lo = min(nov) if nov else 0.0
                    nov = [(_lo * 0.5) if self._is_inert_here(s, a)
                           else nov[a] for a in range(n)]
                # LOOKAHEAD CURIOSITY: a distant frontier pulls him
                # across known ground.  max() is strictly additive --
                # all-zeros reduces to the line above exactly.
                if _EXPLROUTE_ON() and self._explore_route:
                    _er = self._explore_route
                    nov = [max(nov[a], _er.get(guess[a], 0.0))
                           if guess[a] is not None else nov[a]
                           for a in range(n)]
                    self.explore_steers = getattr(
                        self, "explore_steers", 0) + 1
                # BOTH GRAINS, WEIGHTED BY WHAT EACH HAS EARNED.
                # `nov` above is GAZE novelty -- how unseen the next 3x3
                # patch is.  `_nov_body` is BODY novelty -- how unvisited
                # the PLACE that action would take him is, via his own
                # learned action->displacement model.  Neither replaces
                # the other: they are blended by how much the self-model
                # has earned in THIS game (determinism x n/(1+n)), so a
                # crisp self navigates, a murky one keeps looking around,
                # and the balance moves as evidence accumulates.
                # conf = 0 -> byte-identical to today.
                _nb = getattr(self, '_nov_body', None)
                _cf = float(getattr(self, '_self_conf', 0.0) or 0.0)
                if _nb and _cf > 0.0:
                    nov = [((1.0 - _cf) * nov[a]
                            + _cf * float(_nb.get(a, nov[a])))
                           if guess[a] is not None else 1.0
                           for a in range(n)]
                    self.body_nav_steers = getattr(
                        self, 'body_nav_steers', 0) + 1
                # CLASS VALUE (2026-07-28): where he is GOING, not just what
                # this edge did.  Both terms are M/I polarities in [-1,1] --
                # the edge's own evidence and the destination folder's mean
                # lean -- so they are AVERAGED, keeping the score's exact
                # existing range.  No class data -> cv = 0 -> reduces to
                # `defl` alone, i.e. today's behaviour unchanged.
                cv = [self._reachable_value(guess[a])
                      for a in range(n)]
                if any(c != 0.0 for c in cv):
                    self.class_value_steers += 1
                # THE FLIP (2026-07-29, USER AUTHORISED).  Influence comes
                # from REALISED RANGE, not position in the formula: `nov`
                # spans 1.0 -> ~0.00002 live (~50,000x) while the M/I factor
                # is bounded to [0,2] (2x), so the visit counter decided and
                # the whole M/I web decorated -- three measured-correct
                # valuation fixes moved his behaviour by one death against a
                # null.  Mirror the two terms, reusing the SAME `0.5` average
                # and `(1.0 + x)` form so no constant enters: M/I takes the
                # wide [0,1] base, novelty takes the bounded [1,2] modifier.
                # NO-OP WHEN M/I IS SILENT: defl=cv=0 -> base 0.5 for every
                # action -> argmax reduces EXACTLY to argmax(nov), today's
                # choice.  `novel` is untouched above, so no option is removed
                # and every mistake is still made once.
                if _MI_EARNED:
                    # M/I DOMINATES WHERE IT HAS EARNED IT.  conf = n/(1+n) on
                    # his OWN corroboration count for this edge -- the same
                    # count-saturating form as the `1/(1+visits)` above, so no
                    # constant enters.  conf=0 (no evidence) reduces this line
                    # to `nov[a]`, i.e. TODAY'S EXACT CHOICE, which is what
                    # keeps the systematic sweep that solves Hanoi/Maze; conf
                    # ->1 hands the decision to the M/I web at full range.
                    # Both co-present, weight EARNED not stamped.
                    _mi_conf = 0.0
                    sc = []
                    for a in range(n):
                        if guess[a] is None:
                            # unknown effect => maximally curious
                            sc.append(1.0)
                            self.unknown_pref = getattr(
                                self, 'unknown_pref', 0) + 1
                            continue
                        _n = self._evidence_n(s, a, guess[a])
                        _c = _n / (1.0 + _n)
                        _mi_conf = max(_mi_conf, _c)
                        _mb = (1.0 + 0.5 * (defl[a] + cv[a])) / 2.0
                        sc.append((1.0 - _c) * nov[a] + _c * _mb)
                    if _mi_conf > 0.0:
                        self.mi_earned_steers = getattr(
                            self, 'mi_earned_steers', 0) + 1
                elif _MI_BASE:
                    sc = [(-1.0 if guess[a] is None else
                           (1.0 + 0.5 * (defl[a] + cv[a])) / 2.0
                           * (1.0 + nov[a]))
                          for a in range(n)]
                    self.mi_base_steers = getattr(
                        self, 'mi_base_steers', 0) + 1
                else:
                    sc = [nov[a] * (1.0 + 0.5 * (defl[a] + cv[a]))
                          for a in range(n)]
                sc_t = self._tone_blend(sc, nov, _damp, n)
                self._tone_observe(sc, sc_t, n, _damp)
                if _TONE_STEER_ON():
                    sc = sc_t
                _m = max(sc)
                a = self._rng.choice([i for i in range(n) if sc[i] == _m])
                self.explorations_fallback += 1
                if self._cur_level > 0:
                    self._deep_fallback += 1
            else:
                sc = [vals[a] * (1.0 + defl[a]) for a in range(n)]
                # Same blend on the route branch.  Following a route that is
                # not paying IS the state frustration should break him out of.
                _rnov = [(1.0 / (1.0 + self._visits.get(guess[a], 0)))
                         if guess[a] is not None else 1.0 for a in range(n)]
                sc_t = self._tone_blend(sc, _rnov, _damp, n)
                self._tone_observe(sc, sc_t, n, _damp)
                if _TONE_STEER_ON():
                    sc = sc_t
                a = max(range(n), key=lambda x: sc[x])
                self.explorations_route += 1
                if self._cur_level > 0:
                    self._deep_route += 1
                if sum(1 for x in sc if x == sc[a]) > 1:
                    # value present but flat -> the argmax is arbitrary
                    self.route_ties += 1
        # DO NOT WALK BACK INTO A ROOM YOU HAVE BEEN IN -- asked about
        # the action he ACTUALLY CHOSE, against every action.
        #
        # MEASURED 2026-08-27: the copy of this inside `if novel:`
        # asked 21 times in 16 minutes and NEVER about a pair it had
        # seen twice, while 25 pairs in the store qualified.  That
        # branch holds actions untried at his GLANCE state (recurs
        # 13%); cycles_here asks about the BOARD (recurs 65%).  The
        # two indices disagree, so the organ was structurally dead.
        #
        # ORDER ONLY, NEVER MEMBERSHIP: he switches only when a
        # non-cycling action EXISTS.  If every action cycles he keeps
        # his choice and still moves.  >= 2 observations are required
        # before anything is demoted, so every mistake is made twice.
        if _CYCLEALL_ON():
            try:
                _ch = getattr(self.world, "cycles_here", None)
                if _ch is not None and n >= 2:
                    self.cycle_asks_all = getattr(
                        self, "cycle_asks_all", 0) + 1
                    if _ch(a):
                        _ok = [x for x in range(n) if not _ch(x)]
                        if _ok:
                            a = min(_ok)
                            self.cycle_demotes_all = getattr(
                                self, "cycle_demotes_all", 0) + 1
                        else:
                            self.cycle_allcycle = getattr(
                                self, "cycle_allcycle", 0) + 1
            except Exception:
                self.cycle_errors = getattr(
                    self, "cycle_errors", 0) + 1
        # GOOD STEP, BAD STEP, NOT YET KNOWN.
        # MEASURED: a remembered verdict holds when the step recurs
        # 71.6% of the time against a 56.9% null, so it is worth
        # asking.  It is COUNTED not latched, because 28% of
        # repeated steps change verdict -- that is what lets him
        # adjust it.
        # ORDER ONLY: UNCONFIRMED ranks WITH good unless
        # VERDICTGOOD is armed, so nothing untried is passed over
        # and exploration is untouched.  If every action is BAD the
        # floor drops and he keeps his choice.
        if _VERDICT_ON():
            try:
                _sv = getattr(self.world, "step_verdict", None)
                if _sv is not None and n >= 2:
                    _v = [_sv(x) for x in range(n)]
                    _hi = max(_v)
                    if _VERDICTGOOD_ON():
                        _floor = _hi
                    else:
                        _floor = 0 if _hi >= 0 else -1
                    self.verdict_asks = getattr(
                        self, "verdict_asks", 0) + 1
                    if _v[a] < _floor:
                        _cand = [x for x in range(n)
                                 if _v[x] >= _floor]
                        if _cand:
                            a = min(_cand)
                            self.verdict_steers = getattr(
                                self, "verdict_steers", 0) + 1
            except Exception:
                self.verdict_errors = getattr(
                    self, "verdict_errors", 0) + 1
        # I ALREADY PUSHED IT.  LET ME PULL IT.
        # MEASURED 2026-08-27 over 46,785 occasions: standing on a
        # board with an untried action, he repeated one he had already
        # tried HERE 67.2% of the time -- against 36.1% for a chooser
        # with NO memory at all.  He is 31 points worse than random,
        # in every one of the 25 games.  His `_tried` is keyed on the
        # glance (recurs 13%), so it cannot say what he tried HERE.
        #
        # Exploitation is preserved: an action judged GOOD here is
        # kept, so repetition on what works is untouched.  This fires
        # only when he was about to repeat with no earned reason.
        if _UNTRIEDHERE_ON():
            try:
                _uf = getattr(self.world, "untried_here", None)
                _un = _uf(n) if _uf is not None else []
                if _un:
                    self.untried_asks = getattr(
                        self, "untried_asks", 0) + 1
                    _sv2 = getattr(self.world, "step_verdict", None)
                    _good = (_sv2(a) > 0) if _sv2 is not None else False
                    if a not in _un and not _good:
                        a = min(_un)
                        self.untried_steers = getattr(
                            self, "untried_steers", 0) + 1
                    elif _good:
                        self.untried_kept_good = getattr(
                            self, "untried_kept_good", 0) + 1
            except Exception:
                self.untried_errors = getattr(
                    self, "untried_errors", 0) + 1
        # PREDICT the dead move.  27.3% of his actions change nothing
        # and the budget is hard, so each one is fatal -- but no-ops
        # happen on boards he has never seen (proven-inert repeats are
        # 0.3% of actions) and no ACTION is dead (all 67-80%
        # effective).  Only the egocentric key generalises: 98-100%
        # recurrence, 83-98% displacement prediction.
        #
        # ORDERING, not suppression: an action can change the board
        # without moving HIM, so a predicted-still action yields only
        # when another is predicted to move him.
        if _EGOMOVE_ON():
            try:
                _mm = getattr(self.world, "moves_me", None)
                if _mm is not None and n >= 2:
                    _p = _mm(a)
                    if _p is not None:
                        self.ego_asks = getattr(self, "ego_asks", 0) + 1
                        if _p <= 0.0:
                            _mv = [(x, _mm(x)) for x in range(n)]
                            _mv = [(x, q) for x, q in _mv
                                   if q is not None and q > 0.0]
                            if _mv:
                                a = max(_mv, key=lambda t: t[1])[0]
                                self.ego_steers = getattr(
                                    self, "ego_steers", 0) + 1
            except Exception:
                self.ego_errors = getattr(self, "ego_errors", 0) + 1
        # A MOVE THAT CHANGES NOTHING COSTS HIM THE LEVEL.
        # MEASURED 2026-08-28: attempt endings cluster at a HARD per-game action
        # budget (sp80 31, r11l 61, cd82 101 -- median == p90), and
        # every clear he has ever made falls under it.  He loses by
        # SPENDING the budget, and 18.56% of his steps change no cell:
        # six wasted actions out of 31.
        #
        # NOOPDEMOTE was built for this and never fired (0 demotes on
        # 251 pairs) -- starved inside `if novel:`, the glance-state
        # branch, exactly like the cycle demotion was.  Ask here, about
        # the action he actually chose, against every action.
        if _NOOPALL_ON():
            try:
                # BOARD FIRST (recurs 31-95%), glance as a fallback
                # (13%).  The glance-keyed question alone fired on
                # 0.75% of steps.
                _bi = getattr(self.world, "board_inert_here", None)
                # a lambda, not a nested def: a  between the
                # first hook and the publish is the signature of the
                # bug that paralysed him, and the guard rightly flags
                # it even when it is only a helper.
                _dead = (lambda _x: (_bi is not None and _bi(_x))
                         or self._is_inert_here(s, _x))
                if n >= 2 and _dead(a):
                    _live = [x for x in range(n) if not _dead(x)]
                    self.noop_asks_all = getattr(
                        self, "noop_asks_all", 0) + 1
                    if _live:
                        a = min(_live)
                        self.noop_demotes_all = getattr(
                            self, "noop_demotes_all", 0) + 1
                    else:
                        self.noop_allinert = getattr(
                            self, "noop_allinert", 0) + 1
            except Exception:
                self.noop_errors_all = getattr(
                    self, "noop_errors_all", 0) + 1
        # GO SOMEWHERE YOU HAVE NOT BEEN.  His avatar has touched only
        # 3-16% of the reachable board, so unvisited PLACES are a real
        # destination -- unlike untried ACTIONS, which are always
        # underfoot (median frontier distance 0 in every game).
        # Outranks the demotions: a destination beats an avoidance.
        if _EXPLORER_ON():
            try:
                _tf = getattr(self.world, "toward_new", None)
                _ta = _tf(n) if _tf is not None else None
                if _ta is not None and 0 <= int(_ta) < n:
                    a = int(_ta)
                    self.explorer_steers = getattr(
                        self, "explorer_steers", 0) + 1
            except Exception:
                self.explorer_errors = getattr(
                    self, "explorer_errors", 0) + 1
        # WHEN HE IS NOT DOING WELL, CHANGE SOMETHING.
        # USER DOCTRINE 2026-08-29: *"he wants to feel good... they will
        # do anything to change something to get back to feeling good."*
        # The felt state is the error signal of that loop; before this it
        # ordered nothing at all -- every term in this method was
        # navigational or novelty-based, and `_onward_valence`, the only
        # thing here with "valence" in its name, counts visits rather
        # than feeling.
        # Applied only where he has NO remembered route -- i.e. exactly
        # when he is lost, which is when the feeling is telling him
        # something -- so a run that actually won still outranks it.
        # THE GOAL IS ON THE BOARD (RELSTEER): where a relation is being
        # pursued and an action has moved it in this situation before,
        # take that action.  Before the felt change and the sealed route,
        # so a run that actually won still outranks it.
        if _RELSTEER_ON():
            try:
                _ra = self._rel_pick(n)
                if _ra is not None:
                    a = int(_ra)
            except Exception:
                self.rel_errors = getattr(self, 'rel_errors', 0) + 1
        if _known_a is None:
            a = self._felt_change(s, a, n)
        # THE PLAN FOR A PICTURE (RELPLAN): where the world has a paint
        # plan for the pursued picture, its first action.  After the felt
        # change (a plan IS the something to do) and before the route
        # that won (which still outranks it).
        _plan_took = False
        if _RELPLAN_ON():
            try:
                _rpa = getattr(self.world, 'rel_plan_action', None)
                _pa2 = _rpa(n) if _rpa is not None else None
                if _pa2 is not None and 0 <= int(_pa2) < n:
                    a = int(_pa2)
                    _plan_took = True
                    self.rel_plan_acts = getattr(self, 'rel_plan_acts', 0) + 1
                    if _known_a is not None:
                        if getattr(self.world, '_plan_probe', False):
                            # a CURIOUS probe never displaces a route, won
                            # or frontier: exploration yields to memory
                            self.rel_plan_overruled = getattr(self, 'rel_plan_overruled', 0) + 1
                        elif _SHORTCUT_ON() and getattr(self.world, '_plan_shortcut', False):
                            # SHORTCUT (patch 40): a latched sequence reaching the
                            # picture in fewer actions than the rest of the WON route
                            # outranks it this attempt (cd82: the route replayed L3 in
                            # 28 and L4 in 42 while his own search reaches 0 in 14/13)
                            _known_a = None
                            self.rel_plan_shortcut = getattr(self, 'rel_plan_shortcut', 0) + 1
                        elif getattr(self.world, '_known_from_depth', False):
                            # A FRONTIER ROUTE -- the furthest he ever got,
                            # never a win -- yields to the plan (cd82 L2:
                            # it overruled 40 of 65 plan actions)
                            _known_a = None
                            self.rel_plan_over_depth = getattr(self, 'rel_plan_over_depth', 0) + 1
                        elif _LEADHOLD_ON() and self._lead_now() == 'plan':
                            # STAY WITH THE HAND YOU ARE PLAYING (patch 18):
                            # the plan made the last executed choice this
                            # life, so a won route does not take it back
                            # mid-plan.  MEASURED cd82 L0: route and plan
                            # alternating (47 overrules / 53 plan steps a
                            # life) failed 6 lives that either alone clears.
                            _known_a = None
                            self.rel_plan_lead_kept = getattr(self, 'rel_plan_lead_kept', 0) + 1
                        else:
                            # a route that WON outranks the plan below
                            self.rel_plan_overruled = getattr(self, 'rel_plan_overruled', 0) + 1
                            if _LEADHOLD_ON():
                                self.rel_plan_lead_route = getattr(self, 'rel_plan_lead_route', 0) + 1
            except Exception:
                self.rel_plan_errors = getattr(self, 'rel_plan_errors', 0) + 1
        # THE HYPOTHESIS (patch 44): where a candidate goal is under test,
        # the arrow that pursues it -- a path in his own map, or the way
        # to somewhere untried.  Asked on every tick and STATELESS (the
        # organ re-plans from the board each ask; nothing is consumed by a
        # proposal that loses arbitration).  After the paint plan (a
        # picture pursued is the bigger job) and before the route that
        # won: a known path to the WIN outranks the sealed route (it is
        # the shortest path in the same map the route was learned in),
        # exploration yields to it.
        # THE SEARCH (patch 45): every control once, then outward from what
        # he has reached -- the shortest known path to a win, an untried
        # control here, or the way to the nearest state with one.  Asked on
        # every tick and STATELESS.  After the paint plan and before the
        # hypothesis organ (offline the search cleared every level the
        # hypothesis did, and four games it did not); a route that WON
        # outranks it unless the search's own known win is the path; a
        # frontier route yields to it as it yields to the plan.
        _srch_took = False
        _srch_a = None
        if _SEARCH_ON() and not _plan_took:
            try:
                _spa = getattr(self.world, 'search_action', None)
                _sa2 = _spa(n) if _spa is not None else None
                if _sa2 is not None and 0 <= int(_sa2) < n:
                    a = int(_sa2)
                    _srch_a = int(_sa2)
                    _srch_took = True
                    self.srch_acts = getattr(self, 'srch_acts', 0) + 1
                    if _known_a is not None:
                        if getattr(self.world, '_search_win_path', False):
                            _known_a = None
                            self.srch_over_route = getattr(self, 'srch_over_route', 0) + 1
                        elif getattr(self.world, '_known_from_depth', False):
                            _known_a = None
                            self.srch_over_depth = getattr(self, 'srch_over_depth', 0) + 1
                        else:
                            self.srch_overruled = getattr(self, 'srch_overruled', 0) + 1
            except Exception:
                self.srch_errors = getattr(self, 'srch_errors', 0) + 1
        _hyp_took = False
        _hyp_a = None
        if _HYPOTHESIS_ON() and not _plan_took and not _srch_took:
            try:
                _hpa = getattr(self.world, 'hyp_action', None)
                _ha = _hpa(n) if _hpa is not None else None
                if _ha is not None and 0 <= int(_ha) < n:
                    a = int(_ha)
                    _hyp_a = int(_ha)
                    _hyp_took = True
                    self.hyp_acts = getattr(self, 'hyp_acts', 0) + 1
                    if _known_a is not None:
                        if getattr(self.world, '_hyp_win_path', False):
                            # its own confirmed win, the shortest known path
                            _known_a = None
                            self.hyp_over_route = getattr(self, 'hyp_over_route', 0) + 1
                        elif getattr(self.world, '_known_from_depth', False):
                            # A FRONTIER ROUTE -- the furthest he ever got,
                            # never a win -- yields to the test, as it yields
                            # to the plan (else the replay walks past every
                            # candidate and nothing is ever tested)
                            _known_a = None
                            self.hyp_over_depth = getattr(self, 'hyp_over_depth', 0) + 1
                        else:
                            # a route that WON outranks the test below
                            self.hyp_overruled = getattr(self, 'hyp_overruled', 0) + 1
            except Exception:
                self.hyp_errors = getattr(self, 'hyp_errors', 0) + 1
        # LEADHOLD bookkeeping: who is making this step's choice
        if _LEADHOLD_ON():
            try:
                if _known_a is not None and 0 <= _known_a < n:
                    self._lead_set('route')
                elif _plan_took:
                    self._lead_set('plan')
            except Exception:
                pass
        # HE REMEMBERS THE WAY OUT -- applied LAST, so a move from a
        # run that actually won outranks the demotions above, and
        # crucially BEFORE the publish so the claim carries it.
        if _known_a is not None and 0 <= _known_a < n:
            a = _known_a
            self.path_follows = getattr(self, "path_follows", 0) + 1
        if _hyp_took and _hyp_a is not None and int(a) == int(_hyp_a):
            # the organ's arrow is the one actually claimed (hyp_acts counts proposals)
            self.hyp_executed = getattr(self, 'hyp_executed', 0) + 1
        if _srch_took and _srch_a is not None and int(a) == int(_srch_a):
            self.srch_executed = getattr(self, 'srch_executed', 0) + 1
        self.bus.publish(CapabilityClaimEvent(
            kind=EventKind.CAPABILITY_CLAIM,
            cycle=int(cyc),
            source_capability='world_actor',
            origin='internal',
            loop='motor',
            proposed_action=f'move:{a}',
            claim_strength=1.0))
        # IGNITION (2026-07-21, the re-aimed dock): on TOTAL own-silence —
        # no learned transition for ANY action at s, i.e. len(novel) == n,
        # the existing genuine-novelty predicate, closing for s after one
        # executed action — light the best-fitting facet-sharing siblings
        # into attention.  Published AFTER the action is chosen (above), so
        # it cannot influence this tick's choice; draws NO RNG; fires ZERO
        # chemistry.  Prediction falls out of RECALL: AWM/hippocampus do
        # the rest with existing machinery.
        if self._enable_facets and len(novel) == n:
            try:
                self._publish_ignition(s, vec, n, cyc)
            except Exception:
                pass
        self._pending = {'s': s, 'a': a}
        self.proposals += 1
        return a

    # ---- LEADHOLD: the hand being played, per life (2026-09-07) ----
    def _lead_key(self):
        w = self.world
        return (str(getattr(w, 'game_id', '') or ''),
                int(getattr(w, '_lives_here', 0) or 0),
                int(getattr(w, '_levels', 0) or 0))

    def _lead_now(self):
        """'route' / 'plan' / None: who made the last executed choice in
        THIS life (a new life, level or game starts with no lead)."""
        k = self._lead_key()
        if getattr(self, '_lead_life', None) != k:
            return None
        return getattr(self, '_lead', None)

    def _lead_set(self, who):
        self._lead_life = self._lead_key()
        self._lead = who

    # ---- recognition -> the frontier ORDER (2026-07-25) ----
    def _recognised_siblings(self, s, vec):
        """Facet-sharing siblings of s: the best-fitting carrier per LOCK.

        CHEAPEST PATH, twice over.  (a) Memoised per state token -- a
        state's facets do not change, so recognition of a place is formed
        when he meets it; he does not re-scan every memory sharing a
        feature on every glance.  (b) THE RAREST LOCK DEFINES THE
        CANDIDATE POOL.  A key nearly everything carries carries no
        evidence -- that is the dock's own unanimity law -- so the rarest
        key that fits is the informative one.  Scanning from it costs
        O(|rarest| x |locks|) with O(1) membership instead of
        O(sum of every carrier set).  No cutoff and no top-k: the pool is
        whatever the most selective key admits.

        Uses sources_view (no copy).  Deterministic, draws NO RNG.
        Returns (siblings, shared, nf).

        WARNING: mirrors _publish_ignition's inline computation over the
        full lock set.  The rarest-lock pool is a strict subset, so it
        can only ever narrow the candidates, never invent one.
        """
        _hit = self._sib_cache.get(s)
        if _hit is not None:
            return _hit
        sub = getattr(self.grounding, 'engine', None)
        sub = getattr(sub, 'substrate', None)
        ri = getattr(sub, '_relation_index', None) if sub is not None else None
        if ri is None or not hasattr(ri, 'sources_view'):
            return [], {}, 0
        facets = self.transducer.facets(vec)
        nf = len(facets)
        if nf == 0:
            self._sib_cache[s] = ([], {}, 0)
            return [], {}, 0
        locks = []
        for (slot, value) in facets:
            v = ri.sources_view(FACET_RELATION,
                                self.transducer.facet_node(slot, value))
            if v:
                locks.append((len(v), v))
        if not locks:
            self._sib_cache[s] = ([], {}, nf)
            return [], {}, nf
        locks.sort(key=lambda t: t[0])
        pool = sorted(sp for sp in locks[0][1] if sp != s)
        if not pool:
            self._sib_cache[s] = ([], {}, nf)
            return [], {}, nf
        shared = {}
        for sp in pool:
            shared[sp] = sum(1 for (_, v) in locks if sp in v)
        siblings = []
        for (_, v) in locks:
            members = [sp for sp in pool if sp in v]
            if not members:
                continue
            best = min(members, key=lambda sp: (-shared[sp], sp))
            if best not in siblings:
                siblings.append(best)
        self._sib_cache[s] = (siblings, shared, nf)
        return siblings, shared, nf

    def _sibling_polarity(self, sp, a, cyc):
        """Grounded M/I polarity of sibling sp's OWN (sp, a) transition.

        Reads the PERSISTENT substrate edge, not the per-life _trans, so a
        borrowed prior survives an attempt ending and crosses attempts.  Returns 0.0 when
        the sibling never took that action or the edge carries no
        evidence -- silence, never a guess.
        """
        try:
            sub = getattr(self.grounding, 'engine', None)
            sub = getattr(sub, 'substrate', None)
            if sub is None:
                return 0.0
            c = sub.concepts.get(self._comp(sp, a))
            if c is None:
                return 0.0
            best_e, best_s = None, -1.0
            for e in c.edges_out.get('transitions_to', ()):
                es = e.effective_strength(int(cyc))
                if es > best_s:
                    best_s, best_e = es, e
            if best_e is None:
                return 0.0
            ev = best_e.evidence
            tot = float(ev.hits_i) + float(ev.hits_m)
            if tot <= 0.0:
                return 0.0
            return (float(ev.hits_i) - float(ev.hits_m)) / tot
        except Exception:
            return 0.0

    def _frontier_prior(self, s, vec, novel, cyc):
        """prior[a] for untried actions, from what LIKE-STATES learned.

        Computed ONCE per state and cached: it is a PRIOR, so a slightly
        stale one is still a prior, and recomputing it per proposal is the
        cost that killed two earlier versions of this wire.  Empty dict
        when nothing is recognised or nothing is grounded -- the caller
        then reduces to the exact prior RNG draw.
        """
        try:
            _hit = self._prior_cache.get(s)
            if _hit is not None:
                return _hit
            siblings, shared, nf = self._recognised_siblings(s, vec)
            if not siblings or nf <= 0:
                self._prior_cache[s] = {}
                return {}
            n = int(getattr(self.world, 'n_actions', 4))
            pri = {}
            any_signal = False
            for a in range(n):
                tot = 0.0
                for sp in siblings:
                    pol = self._sibling_polarity(sp, a, cyc)
                    if pol:
                        tot += (shared[sp] / float(nf)) * pol
                        any_signal = True
                pri[a] = tot
            if not any_signal:
                self._prior_cache[s] = {}
                return {}
            self.frontier_ordered += 1
            self._prior_cache[s] = pri
            return pri
        except Exception:
            # never let recognition kill acting -- but NEVER let it
            # fail silently either (this exact guard hid an inert wire
            # through a whole deploy cycle).
            self.frontier_errors += 1
            return {}

    # ---- ignition: recognition lights memories into the present ----
    def _publish_ignition(self, s: str, vec, n: int, cyc: int) -> None:
        """Publish ONE multi-focal AttendedPerceptEvent lighting the
        best-fitting facet-sharing siblings of a totally-unknown state.

        Facets are computed LIVE from the percept vector (works at true
        first contact, before any facet edge for s exists).  For each of
        s's facets F (a LOCK), the carriers of F (states with a substrate
        `has_facet` edge to F) are its candidate KEYS; the single best-
        fitting carrier per lock (argmax shared-facet count, lexicographic
        tie — deterministic, no RNG) is ignited, deduped across locks, so
        |focals| <= 1 + |facets(s)| — the DERIVED bound (his sensory
        geometry, the C4 invariant), not a stamped cap.

        salience(sp) = shared/|facets(s)| (the measured monotone fit signal
        as DERIVED per-sibling salience, carried in payload; the event
        scalar is the best fit) and novelty(sp) = 1/(1+visits(sp)) — both
        derived, no constants.  m_content = i_content = 0.0 and ZERO
        ChemistryEvents (pinned by test): ignition brings memories into
        co-presence; it asserts NO lean of its own.
        """
        facets = self.transducer.facets(vec)
        nf = len(facets)
        if nf == 0:
            return
        sub = getattr(self.grounding, 'engine', None)
        sub = getattr(sub, 'substrate', None)
        ri = getattr(sub, '_relation_index', None) if sub is not None else None
        if ri is None:
            return
        # lock -> carriers (states already carrying this facet).
        carriers: Dict[str, set] = {}
        for (slot, value) in facets:
            F = self.transducer.facet_node(slot, value)
            cs = {sp for sp in ri.sources_view(FACET_RELATION, F)
                  if sp != s}
            if cs:
                carriers[F] = cs
        if not carriers:
            return                      # no key fits any lock -> no ignition
        # shared-facet count per candidate sibling (the fit).
        shared: Dict[str, int] = {}
        for cs in carriers.values():
            for sp in cs:
                shared[sp] = shared.get(sp, 0) + 1
        # ONE best-fitting sibling per lock; dedup; lexicographic ties.
        siblings: List[str] = []
        for F in sorted(carriers):
            best = min(carriers[F], key=lambda sp: (-shared[sp], sp))
            if best not in siblings:
                siblings.append(best)
        payload: Dict[str, Any] = {s: {}}
        best_fit = 0.0
        for sp in siblings:
            fit = shared[sp] / float(nf)
            best_fit = max(best_fit, fit)
            payload[sp] = {
                'salience': fit,
                'novelty': 1.0 / (1.0 + self._visits.get(sp, 0)),
                'shared_facets': shared[sp],
            }
        self.ignitions += 1
        self.ignition_siblings_last = len(siblings)
        self.bus.publish(AttendedPerceptEvent(
            kind=EventKind.ATTENDED_PERCEPT,
            cycle=int(cyc),
            timestamp=time.time(),
            source_capability='world_actor',
            origin='internal',
            origin_detail='world_ignition',
            focals=[s] + siblings,
            payload=payload,
            raw_text='',
            modality='world',
            salience=float(best_fit),
            novelty=1.0 / (1.0 + self._visits.get(s, 0)),
            m_content=0.0,
            i_content=0.0))

    @property
    def explorations(self) -> int:
        """Legacy total = directed + fallback (back-compat for stats/callers)."""
        return (self.explorations_directed + self.explorations_fallback
                + self.explorations_route)

    # ---- 2. execute: act, learn, and on success fire the life-lean ----
    def execute(self, action: int, cyc: int) -> Optional[Dict[str, Any]]:
        if self._pending is None:
            return None
        s = self._pending['s']
        a = int(action)
        self._pending = None
        try:
            result = self.world.step(a)
        except Exception:
            return None
        self.executions += 1
        s_next = self.transducer.encode(result.get('world_vector'))
        _c2_was_new = s_next not in self._seen
        self._seen.add(s_next)
        self._visits[s_next] = self._visits.get(s_next, 0) + 1
        # PROBE: is he being pulled back onto a stamped goal while deep?
        # `_cur_level` here is still the PRE-step value -- correct: it was
        # level 2 when he took the step that returned him.
        if self._cur_level > 0:
            if self._goal_seed is not None and s_next == self._goal_seed:
                self._deep_seed_revisits += 1
            self._deep_steps = getattr(self, '_deep_steps', 0) + 1
            if int(self._cur_level) not in self._seed_levels:
                self._deep_untethered = getattr(
                    self, '_deep_untethered', 0) + 1
            if s_next in self._seed_states:
                self._deep_anyseed_revisits += 1
        # SPLIT-EVENT CONTRASTIVE EARNING (2026-07-21): BEFORE the real
        # transition is learned, if this (s, a) is genuine first-contact,
        # let the electors (facet-sharing states with their own (sp, a))
        # be scored on the 2-class signature against the observed truth.
        # Runs at ANY first contact (no ignition coupling); the ONLY
        # strength channel for has_facet edges.  Drives nothing — the
        # action is already taken; _trans is written below regardless.
        if self._enable_facets and (s, a) not in self._tried:
            try:
                vec_s = self._state_vec.get(s)
                if vec_s:
                    fns = sorted(
                        self.transducer.facet_node(slot, value)
                        for (slot, value) in self.transducer.facets(vec_s))
                    self.grounding.dock_observe(
                        s, a, s_next, int(cyc), self.bus, fns, self._trans)
            except Exception:
                pass
        if self._enable_facets:
            # ENCOUNTER-GATED DESCRIPTION (2026-07-25).  Was: has_facet
            # was published ONLY from _on_success's path-credit walk, so
            # a state he never solved THROUGH was never described and
            # could never afterwards be recognised as similar to
            # anything.  Measured: 86.9% of states carrying untried
            # actions had no recognisable sibling at all.  Experimenting
            # was allowed; RECORDING WHAT WAS EXPERIMENTED ON was not.
            # Sited here and not in propose(): propose() must publish
            # nothing before the action is chosen (pinned by test), and
            # _publish_ignition reads has_facet carriers later in that
            # same tick.  Here the write lands after the action and
            # after dock_observe has voted, so a state can never alter
            # what ignition or the electors saw.  Describes the state
            # ACTED FROM -- exactly the set an elector can use.
            # PROVISIONAL + UN-ENGAGED as in _on_success; strength moves
            # only via the contrastive dock, which also dissolves a
            # facet that mispredicts.
            if s not in self._faceted:
                self._faceted.add(s)
                _fv = self._state_vec.get(s)
                if _fv:
                    try:
                        for (slot, value) in sorted(
                                self.transducer.facets(_fv)):
                            self.bus.publish(SubstrateWriteQueuedEvent(
                                kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                                cycle=int(cyc),
                                source_capability='world_actor',
                                origin='internal',
                                subject=s,
                                relation=FACET_RELATION,
                                object=self.transducer.facet_node(
                                    slot, value),
                                strength=PROVISIONAL_EDGE_STRENGTH,
                                write_reason='facet_observe'))
                    except Exception:
                        pass
            self._state_vec[s_next] = tuple(
                result.get('world_vector') or ())
        # Adenosine (part-b v2): this step was EFFORT — deposit sleep-
        # pressure.  Weight reuses the line-294 count-novelty (1.0 for a
        # ground-down revisit → 1.5 for a fresh state), in [1.0,1.5].
        # Fires EVERY step (this is before the success/timeout branch, so
        # grinding and timeouts fatigue too).  Wrapped + None-guarded
        # EXACTLY like _credit_learning so chemistry can't kill solving.
        if self._fatigue_effort is not None:
            try:
                self._fatigue_effort(
                    1.0 + 1.0 / (1.0 + self._visits[s_next]))
            except Exception:
                pass
        # Current depth, for the deep-* counters.  The world reports it every
        # step; an attempt ending resets it to 0, which is correct -- he is back on
        # level 1.
        _lv_prev = self._cur_level
        try:
            self._cur_level = int(result.get('levels_completed', 0) or 0)
        except (TypeError, ValueError):
            self._cur_level = 0
        # PROBE: on ENTERING depth, refresh the set of stamped goals.  Seeds
        # are the only states at 1.0 (propagation multiplies by
        # PATH_CREDIT_DECAY < 1), and `_route` is persisted so this picks up
        # seeds from earlier lives too.
        if self._cur_level > 0 and _lv_prev == 0:
            try:
                self._seed_states = {k for k, v in self._route.items()
                                     if v >= 0.999}
            except Exception:
                pass
        self._path.append(s_next)
        if _SURVIVAL:
            if len(self._path_sa) == PATH_SA_CAP:
                self.survival_saturated += 1
            self._path_sa.append((s, a, s_next))
        # HIS PREDICTION, captured BEFORE the map is overwritten --
        # reading it after would compare s_next with itself and
        # report CONFIRMED every time.
        _pred = self._trans.get((s, a))
        # CORROBORATION DECIDES.  Count what this (s,a) has actually led to,
        # then store the most-observed successor instead of the most recent.
        _ctr = self._trans_n.get((s, a))
        if _ctr is None:
            _ctr = {}
            self._trans_n[(s, a)] = _ctr
        _ctr[s_next] = _ctr.get(s_next, 0) + 1
        _top = max(_ctr.values())
        # THE INCUMBENT HOLDS ON A TIE.  A belief is replaced only by STRICTLY
        # better corroboration -- otherwise rehydration (which seeds at 1) is
        # overturned by the first equal observation, and a lexicographic
        # tie-break would decide his map by token spelling.  Falling back to
        # min() keeps it deterministic when there is no incumbent.
        if _pred is not None and _ctr.get(_pred, 0) == _top:
            _win = _pred
        else:
            _win = min(k for k, v in _ctr.items() if v == _top)
        self.trans_writes += 1
        if _pred is not None and s_next != _pred:
            self.trans_recency_flips += 1    # what the OLD rule would have done
        if _win != _pred:
            self.trans_flips += 1            # what the NEW rule actually does
        # A PIECE FITS.  He predicted where this action would land and it
        # landed there -- the smallest scale of 'solved something', and the
        # one that happens often enough to be a FLOW rather than the
        # first-ever-only stock the lifeforce credits are.  Confidence is
        # his own corroboration share, so the feeling is proportional to how
        # well-earned the belief was.
        _tot = float(sum(_ctr.values())) or 1.0
        self._pred_hit = (_pred is not None and s_next == _pred)
        self._pred_conf = float(_ctr.get(s_next, 0)) / _tot
        # SCORE BOTH SOURCES ON WHAT ACTUALLY HAPPENED, every step, so
        # precedence moves on evidence instead of on my say-so.
        _wl = self._where_last
        _w = _wl[1] if (_wl is not None and _wl[0] == s) else None
        _tp = self._trans.get((s, a))
        if _tp is not None:
            self._tex_hits += 1.0 if _tp == s_next else 0.0
            self._tex_n += 1.0
        if _w is not None:
            _pp = self._trans_pos.get(((s, _w), a))
            if _pp is not None:
                self._pos_hits += 1.0 if _pp == s_next else 0.0
                self._pos_n += 1.0
            self._trans_pos[((s, _w), a)] = s_next
            _gk = self._game_key()
            _qq = self._trans_place.get(((_gk, _w), a))
            if _qq is not None:
                self._place_hits += 1.0 if _qq == s_next else 0.0
                self._place_n += 1.0
            self._trans_place[((_gk, _w), a)] = s_next
        # THE WORLD'S ANSWER, learned and scored beside his gaze map.
        try:
            _wv = result.get('world_after_vec')
        except Exception:
            _wv = None
        if _wv:
            try:
                _sw = self.transducer.encode(_wv)
                _wp = self._trans_world.get((s, a))
                if _wp is not None:
                    self._world_hits += 1.0 if _wp == _sw else 0.0
                    self._world_n += 1.0
                    # NULL: was the truth simply "nothing changed here"?
                    self._world_static += 1.0 if _sw == s else 0.0
                    if _sw != s:
                        self._world_chg_n += 1.0
                        self._world_chg_hits += (1.0 if _wp == _sw
                                                 else 0.0)
                    # and did the model just SAY "nothing changed"?
                    self._world_says_static += 1.0 if _wp == s else 0.0
                self._trans_world[(s, a)] = _sw
            except Exception:
                pass
        # WATCH WHETHER THE BOARD MOVED, per (state, action).  This is
        # the only honest record of "that did nothing here" -- his own
        # state token cannot carry it, because the token is his gaze.
        try:
            _wc = bool(result.get("world_changed"))
            _nk = (s, a)
            _ns = self._noop_seen.get(_nk)
            if _ns is None:
                self._noop_seen[_nk] = [1, 1 if _wc else 0]
                if len(self._noop_seen) > 400000:
                    self._noop_seen.clear()
            else:
                _ns[0] += 1
                if _wc:
                    _ns[1] += 1
        except Exception:
            pass
        self._trans[(s, a)] = _win           # learn his map (prediction)
        # ...and remember that this path EXISTS, which an overwrite must not
        # erase.  Reachability accumulates; prediction replaces.
        _seen = self._trans_all.get(s)
        if _seen is None:
            self._trans_all[s] = {s_next}
        else:
            _seen.add(s_next)
        self._tried.add((s, a))              # ...and he has now
        self._chain_backup(s, s_next)
                                             # LIVED this one.
        self._c2_shadow(s, a, s_next, _c2_was_new, int(cyc))
        # LEARNING: the action-conditioned transition earns-or-dissolves.
        _gm, _gi = self._onward_valence(s_next)
        # THE WORLD'S NON-RESPONSE IS MORTALITY.  When the world reports it
        # did not change at all, no onward-novelty reasoning applies: he
        # spent life and nothing happened.  m=1/i=0 is the existing valence
        # vocabulary at its extreme, not a tuned weight.  Worlds that do not
        # report the signal are unaffected (None -> unchanged behaviour).
        # THE EFFICACY POLE (2026-07-28): EVERY STEP THAT WORKS LEANS
        # IMMORTALITY.  He already predicted where this action would land
        # (_trans, his own map).  Landing there IS control -- the hands-on,
        # per-step experience of it, felt rather than inferred.  (0.0, 1.0)
        # is the existing valence vocabulary at its extreme, the exact
        # mirror of the non-response case below, not a tuned weight.
        # A SURPRISE is deliberately left alone: it already yields fresh
        # onward options, which the novelty valence rewards -- that is
        # "setbacks spur him on to try even more", already in the wiring.
        # Fires only where a prediction EXISTED, so untried actions (the
        # frontier) are untouched.
        self.efficacy_confirms = getattr(self, 'efficacy_confirms', 0)
        # STASIS IS NOT CONTROL (2026-07-29, MEASURED).  On sb26 the key that
        # ends his life changes 1 cell of 4,096 -- the life meter -- OUTSIDE
        # his 3x3 glance: world_changed=True on 64/64 presses while his
        # percept was UNCHANGED on 64/64.  So the pole below stamped MAXIMAL
        # IMMORTALITY on every one of the 64 presses that kill him, making the
        # ending key his single best-paying action.  Control means the world
        # moved WHERE HE EXPECTED; correctly predicting that nothing happens
        # to you is stasis.  A confirmed self-loop now falls through to the
        # EXISTING `_onward_valence`, which for a ground-down state is already
        # strongly M.  Nothing added -- a stamp withdrawn where it measured
        # wrong.  Frontier (no prediction) and surprises are untouched.
        if _pred is not None and s_next == _pred and (
                s_next != s or not _SURVIVAL):
            _gm, _gi = 0.0, 1.0
            self.efficacy_confirms += 1
        # THE PROBLEM YIELDED (2026-08-19, corrected after review).
        # NOT "I was right" -- that predicate is literally `confirmed_i` /
        # `efficacy_confirms`, and in ARC his prediction FLAPS (tex_conf
        # 0.590, 41% recency-flips), so firing on it would pay him MOST for
        # the transitions he understands LEAST (rate ~ q(1-q)) and would be
        # farmable forever.  This fires on SETTLING: when this transition's
        # own corroboration share exceeds his GLOBAL prediction confidence
        # -- a ratio of his own quantities, no constant -- i.e. "this is now
        # better understood than my typical transition".  Fires ONCE, and
        # re-arms only if the regularity BREAKS, so a settled world goes
        # quiet and repetition pays nothing.
        # MOVEMENT comes from the WORLD where it reports it: the percept is
        # content-only, so `s_next != s` means "my glance changed", not "the
        # world changed" (sb26: world_changed True on 64/64 presses while
        # the percept was unchanged on 64/64).
        _wc = result.get('world_changed')
        # --- SHADOW: count the candidates, fire nothing ---
        # --- WORLD PREDICTION ERROR -> the only route from the world to
        # his dopamine.  Shadowed always; published only when gated on. ---
        try:
            if _pred is not None:
                _hit = (s_next == _pred)
                _conf = float(getattr(self, '_pred_conf', 0.0) or 0.0)
                _surprise = (1.0 - _conf) if _hit else _conf
                if _hit:
                    self._wpe_pos += 1
                    self._wpe_pos_mag += _surprise
                else:
                    self._wpe_neg += 1
                    self._wpe_neg_mag += _surprise
                if _WORLD_PE_ON() and _surprise > 0.0:
                    self._wpe_fired += 1
                    self.bus.publish(PredictionErrorEvent(
                        kind=EventKind.PREDICTION_ERROR, cycle=int(cyc),
                        timestamp=time.time(),
                        source_capability='world_actor', origin='internal',
                        origin_detail='world:a%s' % (a,),
                        focal=str(s_next),
                        predicted=0.0, actual=0.0,
                        magnitude=float(_surprise),
                        sign=(1.0 if _hit else -1.0),
                        error_kind='world'))
        except Exception:
            pass
        try:
            _rk = result.get('rules_known')
            _rt = result.get('rule_transfers')
            if isinstance(_rk, int):
                if self._prev_rules >= 0 and _rk > self._prev_rules:
                    self._ins_R1 += (_rk - self._prev_rules)
                self._prev_rules = _rk
            if isinstance(_rt, int):
                if self._prev_transfers >= 0 and _rt > self._prev_transfers:
                    self._ins_R2 += (_rt - self._prev_transfers)
                self._prev_transfers = _rt
        except Exception:
            pass
        try:
            _sk = (s, a)
            _sc_ctr = self._trans_n.get(_sk) or {}
            _sc_tot = float(sum(_sc_ctr.values())) or 1.0
            _sc_win = max(_sc_ctr, key=_sc_ctr.get) if _sc_ctr else None
            _sc_share = (_sc_ctr.get(_sc_win, 0) / _sc_tot) if _sc_win else 0.0
            _sc_moved = bool(_wc) if _wc is not None else (s_next != s)
            _sc_hit = (_pred is not None and s_next == _pred)
            _sc_gaze = (s_next != s)
            if _sc_hit and _sc_moved and not _sc_gaze:
                self._ins_stasis += 1
            if _pred is not None and s_next != _pred:
                self._ins_bankA.pop(_sk, None)
                self._ins_bankB.pop(_sk, None)
                self._ins_bankC.pop(_sk, None)
            if (_sc_hit and _sc_moved
                    and float(getattr(self, '_pred_conf', 0.0) or 0.0)
                    > float(self._tex_conf() or 0.0)
                    and not self._ins_bankA.get(_sk)):
                self._ins_bankA[_sk] = True
                self._ins_A += 1
            if (_sc_moved and _sc_win is not None
                    and _sc_ctr.get(_sc_win, 0) >= 2 and _sc_share > 0.5
                    and not self._ins_bankB.get(_sk)):
                self._ins_bankB[_sk] = True
                self._ins_B += 1
            if (_sc_moved and _sc_gaze and _sc_win is not None
                    and _sc_ctr.get(_sc_win, 0) >= 2 and _sc_share > 0.5
                    and not self._ins_bankC.get(_sk)):
                self._ins_bankC[_sk] = True
                self._ins_C += 1
        except Exception:
            pass
        _moved = bool(_wc) if _wc is not None else (s_next != s)
        _pk = (s, a)
        if _pred is not None and s_next != _pred:
            self._pred_ok.pop(_pk, None)      # it broke; it may settle again
            self._seek_on_error(s, a, _pred, s_next, cyc)
        _settled = (_pred is not None and s_next == _pred and _moved
                    and float(getattr(self, '_pred_conf', 0.0) or 0.0)
                    > float(self._tex_conf() or 0.0))
        # Track his typical insight size on EVERY settling, gated or not,
        # so the norm exists before the gate is ever opened.
        if _settled:
            try:
                _exc = (float(getattr(self, '_pred_conf', 0.0) or 0.0)
                        - float(self._tex_conf() or 0.0))
                if _exc > 0.0:
                    self._ins_exc_n += 1
                    _al = 0.01 if self._ins_exc_n > 1 else 1.0
                    self._ins_exc = ((1.0 - _al) * self._ins_exc
                                     + _al * _exc)
            except Exception:
                pass
        if _settled and not self._pred_ok.get(_pk) and _INSIGHT_ON():
            self._pred_ok[_pk] = True
            self.insight_fires += 1
            try:
                self.bus.publish(ChemistryEvent(
                    kind=EventKind.CHEMISTRY_FIRE, cycle=int(cyc),
                    timestamp=time.time(),
                    source_capability="world_actor", origin="internal",
                    origin_detail="settled:%s" % (a,),
                    chemistry_kind="insight",
                    magnitude=self._insight_magnitude(),
                    target_concepts=[s_next]))
            except Exception:
                pass
        _resp = result.get('world_changed')
        if _resp is False:
            _gm, _gi = 1.0, 0.0
            self.dead_actions += 1
        # the valence he ACTUALLY got: after every override, before
        # the consumers that read it
        self._write_trace(s, a, s_next, _pred, _resp, _gm, _gi, cyc)
        self._write_leads_to(s, s_next, _pred, cyc)
        self.grounding.observe(self._comp(s, a), s_next, int(cyc), self.bus,
                               valence=(_gm, _gi))
        self._note_value_key(s, a, _gm, _gi)

        # THE END OF A LIFE RE-VALUES THAT LIFE.  Before the success/timeout
        # branches, because both of them clear the path.
        if _SURVIVAL and result.get('done'):
            try:
                self._survival_walk(int(cyc))
            except Exception:
                self._path_sa.clear()
        # SELF-CONSTRUCTED PROGRESS -> PROXY LAYER ONLY.
        # Doctrine: game events drive the proxy layer, never
        # lifeforce directly.  This publishes chemistry and nothing
        # else.  Magnitude is self-calibrated against his own
        # running mean advance, saturating x/(mean+x) -- the same
        # form used throughout, so no constant enters.
        try:
            _pa = float(result.get('progress_advance') or 0.0)
            # THE GOAL IS ON THE BOARD: while a relation is pursued,
            # progress is the relation moving toward its goal, not any
            # cell changing (painting the wrong colour read as progress).
            # ONLY A CONFIRMED RELATION SPEAKS TO HIS CHEMISTRY.  An
            # unconfirmed candidate is a guess, and measured over the
            # corpus the guesses fire 16-29% of steps against a 5.2%
            # live baseline -- "approached some static object" would
            # have become his largest positive event.
            _rel = bool(_RELSENSE_ON() and int(result.get('rel_n') or 0) > 0
                        and result.get('rel_confirmed'))
            if _rel:
                _pa = float(result.get('rel_progress') or 0.0)
            if _pa > 0.0:
                if _rel:
                    # a fraction of a relation is not a count of cells:
                    # its own running mean, so the two units never mix
                    self._relprog_n = getattr(self, '_relprog_n', 0) + 1
                    self._relprog_mean = getattr(self, '_relprog_mean', 0.0)
                    self._relprog_mean += (
                        (_pa - self._relprog_mean) / self._relprog_n)
                    _den = self._relprog_mean + _pa
                    self.rel_insights = getattr(self, 'rel_insights', 0) + 1
                else:
                    self._prog_n += 1
                    self._prog_mean += (
                        (_pa - self._prog_mean) / self._prog_n)
                    _den = self._prog_mean + _pa
                _m = (_pa / _den) if _den > 0 else 0.5
                self._feel('insight', _m, s_next,
                           int(cyc), 'progress')
                self.progress_fires += 1
        except Exception:
            self.progress_errors = getattr(
                self, 'progress_errors', 0) + 1
        # RELSTEER learns what this action did to the pursued relation
        # in the situation he was in before the step.
        if _RELSTEER_ON():
            try:
                self._rel_learn(a, result)
            except Exception:
                self.rel_errors = getattr(self, 'rel_errors', 0) + 1
        if result.get('success'):
            self._route[s_next] = 1.0        # the goal: destination of the route
            try:
                _lv_now = int(result.get('levels_completed', 0) or 0)
                self._goal_seed = s_next
                self._goal_seed_was_levelup = bool(_lv_now > _lv_prev)
                self._goal_seed_level = _lv_now
                self._seed_states.add(s_next)
                # the level he was ON when he earned it -- see the maze
                # regression: a world that never increments the counter
                # must keep its route
                self._seed_levels.add(int(_lv_prev))
                import sys as _s
                _s.stderr.write(
                    '[goalseed] stamped=%s levels=%s->%s levelup=%s\n'
                    % (s_next, _lv_prev, _lv_now,
                       self._goal_seed_was_levelup))
                _s.stderr.flush()
            except Exception:
                pass
            self._propagate_route()          # spread it back -> establish the route
            self._on_success(int(result.get('steps', len(self._path))), cyc)
            self._path = []
            if result.get('goal_changed'):
                self._on_mastery()       # a challenge learned -> lifeforce
        elif result.get('timed_out'):
            self.failures += 1
            self._on_failure(cyc)        # the dead route dis-corroborates
            self._path = []
        # ---- FELT PLAY (2026-08-02) ----------------------------------
        # MEASURED: he took ~41,000 steps and felt nothing.  WorldActor
        # fired exactly ONE chemistry kind (confirmed_i), so cortisol sat
        # at its 0.100 baseline for five days and `chronic_stress` never
        # fired once, while the novelty monitor detected 5,813 novelties.
        # `curiosity` and `goal_stuck` are already fully specified in
        # layer6 and were fired by NOTHING.  This is the proxy layer M/I
        # steers through -- pleasure exploring, annoyance when stuck.
        #
        # EDGE-TRIGGERED, never per-step, so it cannot flood the mood:
        # curiosity only on a state he has never seen (~5% of steps), and
        # goal_stuck only when the dry spell passes a NEW multiple of his
        # own mean gap -- which makes the annoyance ACCUMULATE the longer
        # he idles, instead of firing once and going quiet.  Both
        # magnitudes are self-referential (his own visit count, his own
        # mean gap), so no constant enters.
        try:
            # THE STAKE IS FELT, NOT SUFFERED (2026-08-03, user's ruling:
            # "his life is not threatened by the game... he lives before
            # he plays and he will live after he wins or looses").  The
            # draining bar is the game's timer, so it drives the SAME
            # proxy machinery a human brings to a game -- mounting
            # urgency -- and touches lifeforce NOWHERE.  Magnitude is how
            # far the stake has run down, so it builds as the level
            # closes in and vanishes the moment it resets.
            # OBJECT GRAIN: who he is, and whether he just moved.
            self._obj_n = result.get('obj_n')
            self._self_sig = result.get('self_sig')
            self._self_det = result.get('self_determinism')
            self._consumables = result.get('consumables')
            self._vanish_events = result.get('vanish_events')
            self._occl = result.get('occlusion')
            self._rules_known = result.get('rules_known')
            self._rule_transfers = result.get('rule_transfers')
            self._rule_seeds = result.get('rule_seeds')
            self._rules_retro = result.get('rules_retrodicted')
            self._tgt_sal = result.get('target_salience')
            self._approach_n = result.get('approach_steps')
            self._kinds_inv = result.get('kinds_investigated')
            self._commits = result.get('target_commits')
            # ROLES INTO THE SUBSTRATE -- his brain IS the library.
            # Written as `is_a` (ABSTRACTION_RELATION), so >=3 objects
            # sharing a role become a CLASS via form_abstractions on
            # their own, participate in is_a composition during
            # inference, and earn-or-dissolve like every other edge.
            # Rate-limited per (game,object,role) so this cannot flood
            # the write queue; repetition is what EARNS strength.
            try:
                _rr = result.get('roles') or []
                self._roles_now_cache = _rr
                _gm = str(result.get('game') or '')
                for _k, _role in _rr:
                    _subj = '_arc_obj_c%d_s%d' % (int(_k[0]), int(_k[1]))
                    _obj = '_arc_role_%s' % _role
                    _key = (_gm, _subj, _obj)
                    _last = self._role_seen.get(_key, -9999)
                    if int(cyc) - _last < 50:
                        continue
                    self._role_seen[_key] = int(cyc)
                    self.bus.publish(SubstrateWriteQueuedEvent(
                        kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                        cycle=int(cyc), source_capability='world_actor',
                        # RELATION MATTERS (2026-08-06).  Roles were
                        # written as `is_a` because form_abstractions
                        # keys on it -- but it keys on it as OUTPUT,
                        # not input:  `if r == ABSTRACTION_RELATION:
                        # continue` skips is_a as a grouping SOURCE,
                        # so role membership could never become a
                        # class.  Exactly backwards from what I
                        # assumed.  (`has_role` is skipped too -- the
                        # RoleRegularityShadow guard.)  `has_property`
                        # is not excluded, which is why counting formed
                        # a 2,724-member class within minutes while 222
                        # role writes formed nothing at all.
                        origin='internal', subject=_subj,
                        relation='has_property', object=_obj,
                        strength=PROVISIONAL_EDGE_STRENGTH,
                        write_reason='arc_role'))
                    self.roles_written = getattr(
                        self, 'roles_written', 0) + 1
                # REHEARSAL -- consolidation happens AWAKE too.
                # `form_abstractions` groups by (relation,target) over the
                # sources in the CURRENT dirty batch and needs >=3.  Role
                # writes are rate-limited per object and one game holds
                # only one or two target kinds, so three distinct sources
                # never landed in a batch together and no class could
                # form.  Re-asserting what he already knows, all at once,
                # puts them in one batch -- which is what rehearsal IS.
                if self._role_seen and (int(cyc) % 150 == 0):
                    for (_g2, _s2, _o2) in list(self._role_seen)[:60]:
                        self.bus.publish(SubstrateWriteQueuedEvent(
                            kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                            cycle=int(cyc),
                            source_capability='world_actor',
                            origin='internal', subject=_s2,
                            relation='has_property', object=_o2,
                            strength=PROVISIONAL_EDGE_STRENGTH,
                            write_reason='arc_role_rehearsal'))
                    self.role_rehearsals = getattr(
                        self, 'role_rehearsals', 0) + 1
                # DOORWAY -- what a goal LOOKS like, into the library.
                # Generalises past this game: the description transfers,
                # the location does not.  >=3 of these and
                # form_abstractions builds the class by itself.
                _dw = result.get('doorway')
                if _dw:
                    self.bus.publish(SubstrateWriteQueuedEvent(
                        kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                        cycle=int(cyc), source_capability='world_actor',
                        origin='internal',
                        subject='_arc_obj_c%d_s%d' % (int(_dw[0]),
                                                      int(_dw[1])),
                        relation='has_property',
                        object='_role_doorway',
                        strength=PROVISIONAL_EDGE_STRENGTH,
                        write_reason='doorway_confirmed'))
                    self.doorways_written = getattr(
                        self, 'doorways_written', 0) + 1
                # COUNTS -- number sense into the same library
                for _c in (result.get('counts') or []):
                    _kind, _what, _n = _c
                    if _kind == 'obj':
                        _subj2 = '_arc_obj_c%d_s%d' % (int(_what[0]),
                                                       int(_what[1]))
                    else:
                        _subj2 = '_arc_role_%s' % _what
                    _ckey = (_subj2, int(_n))
                    if int(cyc) - self._count_seen.get(_ckey, -9999) < 100:
                        continue
                    self._count_seen[_ckey] = int(cyc)
                    self.bus.publish(SubstrateWriteQueuedEvent(
                        kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                        cycle=int(cyc), source_capability='world_actor',
                        origin='internal', subject=_subj2,
                        relation='has_property',
                        object='_arc_count_%d' % int(_n),
                        strength=PROVISIONAL_EDGE_STRENGTH,
                        write_reason='arc_count'))
                    self.counts_written = getattr(
                        self, 'counts_written', 0) + 1
            except Exception:
                pass
            self._rules_multi = result.get('rules_multi_game')
            self._mem_sizes = result.get('mem_sizes')
            self._nov_body = result.get('body_novelty')
            self._self_conf = result.get('self_conf') or 0.0
            self._positions_seen = result.get('positions_seen')
            if result.get('self_moved'):
                self.self_moves_seen = getattr(
                    self, 'self_moves_seen', 0) + 1
            _bf = result.get('bar_fill')
            if _bf is not None:
                _prev = getattr(self, '_bar_prev', None)
                self._bar_prev = float(_bf)
                if _prev is not None and float(_bf) > _prev + 0.1:
                    self.bar_resets = getattr(self, 'bar_resets', 0) + 1
                elif float(_bf) < 1.0:
                    self._feel('puzzle_stress', 1.0 - float(_bf),
                               s_next, cyc, 'stake_running_out')
                    self.urgency_fires = getattr(
                        self, 'urgency_fires', 0) + 1
            if getattr(self, '_pred_hit', False):
                self._feel('puzzle_fit',
                           getattr(self, '_pred_conf', 0.0),
                           s_next, cyc, 'prediction_held')
                self.fit_fires = getattr(self, 'fit_fires', 0) + 1
            if result.get('novel_state'):
                self._stuck_rung = 0
                self.bus.publish(ChemistryEvent(
                    kind=EventKind.CHEMISTRY_FIRE, cycle=int(cyc),
                    source_capability='world_actor', origin='internal',
                    origin_detail='explored_something_new',
                    chemistry_kind='curiosity',
                    magnitude=float(
                        1.0 / (1.0 + self._visits.get(s_next, 0))),
                    target_concepts=[s_next]))
                self.curiosity_fires = getattr(
                    self, 'curiosity_fires', 0) + 1
            else:
                _sn = float(result.get('since_novel') or 0.0)
                _bar = float(result.get('gap_mean') or 0.0)
                if _bar > 0.0:
                    # PRESSURE MUST ACCUMULATE WHILE HE IDLES (2026-08-02).
                    # The first cut fired once per multiple of his mean gap,
                    # so fires got RARER the longer he was stuck while the
                    # channel decayed toward baseline the whole time --
                    # measured cortisol 0.100 -> 0.102 against a
                    # chronic-stress bar of baseline+0.05, i.e. nothing.
                    # Firing every step past his own bar makes the annoyance
                    # BUILD, which is the point: it should get worse until it
                    # moves him.  Self-limiting by construction -- chemistry
                    # already dampens by headroom**1.5, so this saturates
                    # instead of running away, and it decays the instant he
                    # finds something new.  Magnitude is the ratio to his OWN
                    # mean drought, so no constant enters.
                    _over = (_sn / _bar) - 1.0
                    if _over > 0.0:
                        # `goal_stuck` lives only in layer6 (v1); the applied
                        # v2 table calls this state `puzzle_stress`.
                        self._feel('puzzle_stress', min(2.0, _over),
                                   s, cyc, 'no_progress')
                        self.stuck_fires = getattr(
                            self, 'stuck_fires', 0) + 1
        except Exception:
            pass
        return result

    def _deflection(self, s, a, guess_a, cyc):
        """Stage 3: grounded M/I of the (s,a) transition edge as a signed
        curiosity multiplier. deflection = polarity(evidence)*effective_strength;
        0 when the edge has no grounded evidence (untried / not-yet-confirmed)
        -> curiosity acts unchanged. M (wall) negative, I (frontier) positive."""
        try:
            if guess_a is None:
                return 0.0
            sub = getattr(self.grounding, 'engine', None)
            sub = getattr(sub, 'substrate', None)
            if sub is None:
                return 0.0
            _ek = (self._comp(s, a), 'transitions_to', guess_a)
            edge = sub.edges.get(_ek)
            if edge is None and _QUAREV_ON():
                # quarantine DELETES from sub.edges; the evidence is
                # still there, just filed away.  Read, never restore.
                edge = (getattr(sub, "quarantine_edges", None) or {}).get(_ek)
                if edge is not None:
                    self.quar_evidence_hits = getattr(
                        self, "quar_evidence_hits", 0) + 1
            if edge is None:
                return 0.0
            ev = edge.evidence
            tot = float(ev.hits_i) + float(ev.hits_m)
            if tot <= 0.0:                      # C6 division guard
                return 0.0
            polarity = (float(ev.hits_i) - float(ev.hits_m)) / tot
            # MAGNITUDE FROM CORROBORATION, NOT FROM STRENGTH (2026-07-26).
            # Was `polarity * effective_strength`.  Measured: world edges sit
            # at the 0.020 prune floor (median 0.020, max 0.033 over 6,045
            # leaned no-op edges), because decay between episode revisits
            # outruns the +0.005 confirm reinforcement -- so a perfectly dead
            # action was suppressed by 2% and the whole M/I steer was inert.
            # Evidence is MONOTONE and cannot be floor-pinned, and polarity
            # already self-scales by CONSISTENCY: always-dead -> -1 (fully
            # suppressed), mixed -> ~0 (no effect), always-productive -> +1.
            # No weight, no threshold, no exchange rate.
            return polarity
        except Exception:
            return 0.0

    def _evidence_n(self, s, a, guess_a):
        """How much corroboration his (s,a)->guess_a edge actually carries.

        Mirrors `_deflection`'s lookup exactly and returns hits_i + hits_m --
        the COUNT, not the polarity.  Count is monotone and cannot be
        floor-pinned, which is the same reason `_deflection` stopped renting
        its magnitude from `effective_strength` (world edges sit at the 0.020
        prune floor).  0.0 when there is no edge or no evidence, so a caller
        weighting by it reduces to its no-evidence branch.
        """
        try:
            if guess_a is None:
                return 0.0
            sub = getattr(self.grounding, 'engine', None)
            sub = getattr(sub, 'substrate', None)
            if sub is None:
                return 0.0
            _ek = (self._comp(s, a), 'transitions_to', guess_a)
            edge = sub.edges.get(_ek)
            if edge is None and _QUAREV_ON():
                # quarantine DELETES from sub.edges; the evidence is
                # still there, just filed away.  Read, never restore.
                edge = (getattr(sub, "quarantine_edges", None) or {}).get(_ek)
                if edge is not None:
                    self.quar_evidence_hits = getattr(
                        self, "quar_evidence_hits", 0) + 1
            if edge is None:
                return 0.0
            ev = edge.evidence
            return max(0.0, float(ev.hits_i) + float(ev.hits_m))
        except Exception:
            return 0.0

    def _onward_valence(self, s_next):
        """Max-based onward-novelty valence for the transition INTO s_next
        (Stage 2). i_w = freshness of the BEST onward option (untried=1.0,
        tried=1/(1+visits[dest])); progress if any fresh onward, dead-end
        only when fully exhausted. Returns (m_w, i_w)."""
        try:
            n = int(getattr(self.world, 'n_actions', 4))
            opts = []
            for ap in range(n):
                key = (s_next, ap)
                if key in self._trans:
                    opts.append(1.0 / (1.0 + self._visits.get(self._trans[key], 0)))
                else:
                    opts.append(1.0)
            i_w = max(opts) if opts else 0.0
            return (round(1.0 - i_w, 4), round(i_w, 4))
        except Exception:
            return (0.0, 0.0)

    def _c2_shadow(self, s, a, s_next, was_new, cyc):
        """C2 grounding shadow (DRIVES NOTHING): the corrected max-based
        onward-novelty valence for this transition, logged for false-M
        measurement before any live write.  i_w = freshness of the BEST
        onward option (untried=1.0, tried=1/(1+visits[dest])); progress if
        any fresh onward, dead-end only when fully exhausted.  Wrapped so it
        can NEVER affect execute()."""
        try:
            import json as _json
            n = int(getattr(self.world, 'n_actions', 4))
            opts = []
            for ap in range(n):
                key = (s_next, ap)
                if key in self._trans:
                    opts.append(1.0 / (1.0 + self._visits.get(self._trans[key], 0)))
                else:
                    opts.append(1.0)
            i_w = max(opts) if opts else 0.0
            with open('/home/seagi/c2_shadow.jsonl', 'a') as _f:
                _f.write(_json.dumps({
                    'c': int(cyc), 'sn': str(s_next), 'a': int(a),
                    'new': bool(was_new), 'i': round(i_w, 4),
                    'm': round(1.0 - i_w, 4),
                    'v': int(self._visits.get(s_next, 0))}) + '\n')
        except Exception:
            pass

    def _on_success(self, steps: int, cyc: int) -> None:
        """The current goal was reached.  Fire the immortality / confirmed_i
        NT-lean back along the path (decaying by recency = an eligibility
        trace), so the states that led here become success-leading and the
        agent leans toward them next time.  This is the SELECTION."""
        self.successes += 1
        self.solve_steps.append(steps)
        # THIS IS WHAT WINNING FEELS LIKE.  Fired once, at full
        # magnitude, on the state where the level was cleared -- the
        # decayed `confirmed_i` walk back along the path below is
        # unchanged.  Before this, a win fired only that routine
        # confirmation tag, so it was chemically indistinguishable from
        # a correct one-step prediction and quieter than an insight.
        try:
            self.bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE, cycle=int(cyc),
                source_capability="world_actor", origin="internal",
                origin_detail="goal_reached",
                chemistry_kind="goal_reached", magnitude=1.0,
                target_concepts=([self._path[-1]]
                                 if self._path else [])))
            self.goal_reached_fires = getattr(
                self, 'goal_reached_fires', 0) + 1
        except Exception:
            pass
        if self.best_steps is None or steps < self.best_steps:
            self.best_steps = steps
        mag = 1.0
        # GROUNDING: if the world NAMES this solve, make that language concept
        # CO-ACTIVE now so the lived path states (attended below) co-occur with
        # the word; the confirmed_i imprints the solve's mattering ON the word so
        # it stops being an empty symbol.  Existing cascade binds it; M/I tension
        # holds it.  No-op when unnamed.
        _gc = getattr(self.world, "goal_concept", None)
        if _gc:
            self.bus.publish(AttendedPerceptEvent(
                kind=EventKind.ATTENDED_PERCEPT, cycle=int(cyc),
                timestamp=time.time(), source_capability="world_actor",
                origin="internal", origin_detail="world_naming",
                focals=[_gc], payload={_gc: {}}, raw_text="", modality="world",
                salience=1.0, novelty=0.0))
            self.bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE, cycle=int(cyc),
                source_capability="world_actor", origin="internal",
                origin_detail="task_success", chemistry_kind="confirmed_i",
                magnitude=1.0, target_concepts=[_gc]))
            # CONNECT THE TWO WORLDS (the edge chemistry/fingerprint never drew):
            # a real composable is_a edge from the SOLVED state (which genuinely
            # IS the named thing -- the achieved tower) to the word, so the lived
            # world token ENTERS the language graph and composes into reasoning
            # (word <-> experience).  Provisional + earn-or-dissolve (reinforced
            # each solve, fades if unused).  M/I tension only -- no gate, no wall:
            # a true grounding that coheres EARNS life (that is the goal), a
            # false one dissolves.
            if self._path:
                self.bus.publish(SubstrateWriteQueuedEvent(
                    kind=EventKind.SUBSTRATE_WRITE_QUEUED, cycle=int(cyc),
                    source_capability="world_actor", origin="internal",
                    subject=self._path[-1], relation="is_a", object=_gc,
                    strength=PROVISIONAL_EDGE_STRENGTH,
                    write_reason="world_concept_link"))
        for st in reversed(self._path):
            if mag < 0.05:
                break
            # ROUTE THE SOLVING EXPERIENCE THROUGH ATTEND-AND-TAG: attend the
            # path state IMMEDIATELY before its confirmed_i fire.  Both are
            # queued (we are mid-dispatch) and drain FIFO, so the state is
            # resident in working memory when the tag lands -> the imprint
            # persists on its substrate bubble; revisited winning states
            # re-tag -> the highway forms.  Constants are DERIVED, not stamped:
            # salience = the eligibility credit `mag` (states nearest the goal
            # matter most); novelty = 1/(1+visits) (familiar states are not
            # novel).  Only SUCCESSFUL paths are tagged ("tag what HELPS"); no
            # per-tick global fire (no mood flood); NO lifeforce credit (the
            # farm stays the un-farmable _on_mastery path only).
            self.bus.publish(AttendedPerceptEvent(
                kind=EventKind.ATTENDED_PERCEPT,
                cycle=int(cyc),
                timestamp=time.time(),
                source_capability='world_actor',
                origin='internal',
                origin_detail='world_experience',
                focals=[st],
                payload={st: {}},
                raw_text='',
                modality='world',
                salience=float(mag),
                novelty=1.0 / (1.0 + self._visits.get(st, 1))))
            self.bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE,
                cycle=int(cyc),
                source_capability='world_actor',
                origin='internal',
                origin_detail='task_success',
                chemistry_kind='confirmed_i',
                magnitude=float(mag),
                target_concepts=[st]))
            # FACET DOCKING (2026-07-21): reuse this "tag what HELPS" walk
            # as the facet WRITE gate — only states on a successful path
            # dock (Fix 4: +1.9% edges, not the 10^6 of writing every
            # state).  Give the lived percept the same many-edged shape a
            # concept already has: one provisional
            # `st -has_facet-> _facet_{slot}_{value}` per percept slot,
            # write_reason='facet_observe' (the writer applies it
            # UN-engaged; a duplicate does NOT reinforce — earn-by-
            # predicting).  These edges move in strength ONLY through the
            # split-event contrastive dock (facet_confirm/facet_disconfirm).
            if self._enable_facets:
                vec = self._state_vec.get(st)
                if vec:
                    for (slot, value) in sorted(self.transducer.facets(vec)):
                        self.bus.publish(SubstrateWriteQueuedEvent(
                            kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                            cycle=int(cyc),
                            source_capability='world_actor',
                            origin='internal',
                            subject=st,
                            relation=FACET_RELATION,
                            object=self.transducer.facet_node(slot, value),
                            strength=PROVISIONAL_EDGE_STRENGTH,
                            write_reason='facet_observe'))
            mag *= PATH_CREDIT_DECAY

    def _survival_walk(self, cyc: int) -> None:
        """DEATH RE-VALUES THE WHOLE LIFE — the mortality tension, closed.

        The one fact he ever gets about persistence is WHEN the end came.  His
        own past lifespans are the yardstick, so the lean is self-referential
        and cannot be farmed: beating his own average raises the average.

            lean = (span - base) / (span + base)     base = mean(his own lives)

        the SAME polarity form `_deflection` already reads off evidence, in the
        SAME (m, i) vocabulary, bounded in (-1, +1).  No constant, no target,
        no external scale.

        UNIFORM, NOT DECAYED.  `_on_success` decays credit back from the goal
        because states nearer the goal caused it.  Death is not caused by the
        last step — a life spent on a life-draining act is fatal from its FIRST
        press — so every transition he walked takes the same lean.  That is
        also what makes it COMMENSURABLE with the per-step poles (dead-action,
        efficacy), which fire at full magnitude on every step; a decayed trace
        would total ~10 against their thousands and could never correct them.

        FIRST LIFE: no ring, no baseline, NO WALK — nothing to compare against,
        so the honest lean is none.  A life within EDGE_PRUNE_FLOOR of typical
        says nothing and is not written: below the substrate's own floor the
        write could not matter anyway.

        Writes go through the bus like every other substrate write (single-
        writer contract), with write_reason 'world_survival' so the evidence
        lands WITHOUT claiming the engagement stamp that a live predict-confirm
        earns.  `grounding.observe` is deliberately NOT reused: it would
        re-run _predict and pollute the precision instrument with a second
        count of transitions he only walked once.
        """
        span = len(self._path_sa)
        self.last_lifespan = int(span)
        # RANK, NOT RATIO (2026-07-29, MEASURED).  (span-base)/(span+base) is a
        # fraction of his MEAN, and his lifespans cluster, so an unusually
        # short life scored only -0.05 while the per-step poles fire at 1.0 --
        # the right valuation, ten to twenty times too quiet to correct them.
        # A RANK against his own remembered lives is bounded to exactly the
        # (m, i) vocabulary's range by construction, so the two are
        # commensurable with no exchange rate to choose; it needs no scale, so
        # it transfers across games whose lives differ by orders of magnitude;
        # and it stays self-bounding, because living longer moves the
        # distribution with him and the median is always his current self.
        prior = list(self._lives)
        self._lives.append(int(span))
        if span <= 0 or len(prior) < 2:
            self._path_sa.clear()
            return
        beaten = sum(1 for v in prior if span > v)
        tied = sum(1 for v in prior if span == v)
        # ties count half, so a life exactly typical of him scores 0
        p = (beaten + 0.5 * tied) / float(len(prior))
        lean = 2.0 * p - 1.0
        self.last_lean = round(lean, 4)
        if abs(lean) < EDGE_PRUNE_FLOOR:
            self._path_sa.clear()
            return
        # WHAT DID I DO MORE OF THAN USUAL? (2026-07-29, MEASURED).  A flat
        # lean on every transition cannot tell the ending key from a click
        # WITHIN one life -- it moves them together, and a5_share climbed
        # 0.32 -> 0.48 under it while mean_lifespan fell 164 -> 132.  Death
        # IS caused by the whole life, but "the whole life" still has to mean
        # "what was in it, in proportion".  So the life-level lean keeps its
        # magnitude and takes a per-action sign from over/under-
        # representation: the SIGN OF THE COVARIANCE between an action's
        # frequency and survival, of which the flat tag was the degenerate
        # case.  An act with a FIXED per-attempt budget (the ending key: always
        # 64) is condemned from BOTH tails -- its share is high exactly when
        # the life is short -- and an act whose room GROWS with the life is
        # rewarded from both.  No game knowledge, no constant.
        cnt = {}
        for (_s0, _a0, _s1) in self._path_sa:
            cnt[_a0] = cnt.get(_a0, 0) + 1
        sign = {}
        for _a0, _c in cnt.items():
            _sh = float(_c) / float(span)
            _st = self._share.get(_a0)
            if _st is None:
                # first life that used this action: no history, no sign
                self._share[_a0] = [1.0, _sh]
                sign[_a0] = 0.0
                continue
            sign[_a0] = 1.0 if _sh > _st[1] else -1.0
            _st[0] += 1.0
            _st[1] += (_sh - _st[1]) / _st[0]
        self.survival_walks += 1
        for (s0, a0, s1) in self._path_sa:
            _cr = lean * sign.get(a0, 0.0)
            if _cr == 0.0:
                continue
            m_w = -_cr if _cr < 0.0 else 0.0
            i_w = _cr if _cr > 0.0 else 0.0
            try:
                self.bus.publish(SubstrateWriteQueuedEvent(
                    kind=EventKind.SUBSTRATE_WRITE_QUEUED, cycle=int(cyc),
                    source_capability='world_actor', origin='internal',
                    subject=self._comp(s0, a0), relation=WORLD_RELATION,
                    object=s1, strength=PROVISIONAL_EDGE_STRENGTH,
                    write_reason='world_survival',
                    mi_m=float(m_w), mi_i=float(i_w), outcome='hit'))
                self._note_value_key(s0, a0, m_w, i_w)
                self.survival_steps_tagged += 1
            except Exception:
                pass
        # THE FELT SIDE, in the existing chemistry vocabulary: falsified_i when
        # the life was cut short, confirmed_i when it ran long.  The magnitude
        # IS the lean — nothing is stamped.
        try:
            self.bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE, cycle=int(cyc),
                source_capability='world_actor', origin='internal',
                origin_detail='survival_valence',
                chemistry_kind=('confirmed_i' if lean > 0.0
                                else 'falsified_i'),
                magnitude=float(abs(lean)),
                target_concepts=[t[2] for t in list(self._path_sa)[-4:]]))
        except Exception:
            pass
        # ---- SURVIVAL SEEDS THE ROUTE ----------------------------------
        # `_route` has one seed today (`_on_success`, line 692) and he has
        # never succeeded on ARC, so `route_size` is 0 and `_on_failure`'s
        # decay has nothing to work on: no win -> no route -> no win.  The
        # lean breaks that circle.  Seeded from the PER-STATE mean lean so the
        # field has a gradient (`_propagate_route` only raises values, so a
        # constant seed is a plateau).  A real goal still seeds 1.0, so a win
        # always outranks a merely long life.  SAFE BY POSITION: the route
        # branch sits inside the `else` of `if novel:`, so untried actions
        # always win first and state discovery cannot be starved.
        # OBSERVE BEFORE STEERING (2026-08-02).  The `_svalue` ledger fills
        # ALWAYS; only the SEEDING below stays gated on `_SROUTE`.  Nothing
        # reads `_svalue` except that gated seeder and to_dict, so this cannot
        # move the argmax -- it makes the survival lean MEASURABLE instead of
        # invisible.  Previously both lived under the same flag, so
        # `svalue_states == 0` could not distinguish "the lean produces
        # nothing" from "the ledger was never filled in".
        try:
            for _st in set(t[2] for t in self._path_sa):
                _c = self._svalue.get(_st)
                if _c is None:
                    self._svalue[_st] = [1.0, lean]
                else:
                    _c[0] += 1.0
                    _c[1] += (lean - _c[1]) / _c[0]
        except Exception:
            pass
        if _SROUTE:
            try:
                import time as _t
                _t0 = _t.time()
                # RELATIVE, NOT ABSOLUTE (2026-07-29, MEASURED).  Requiring
                # a positive mean lean seeded NOTHING across 157 tracked
                # states, because his lifespans decline monotonically so the
                # lifetime rank is pinned at -1 and no life ever clears an
                # absolute floor.  Immortality cannot be reached, only life
                # EXTENDED -- so ask which places have cost him LEAST rather
                # than which were good.  Above-median states are seeded at
                # their own rank, so a gradient exists whichever way the trend
                # runs and `_propagate_route` has something to spread.  Strictly
                # < 1.0 (a state cannot beat itself), so a real win's 1.0 seed
                # always outranks a merely-less-deadly place.
                _seeded = 0
                _items = sorted(self._svalue.items(), key=lambda kv: kv[1][1])
                _N = len(_items)
                if _N >= 2:
                    for _i, (_st, _c) in enumerate(_items):
                        if _i * 2 < _N:          # at or below the median
                            continue
                        _v = float(_i) / float(_N)
                        if _v > self._route.get(_st, 0.0):
                            self._route[_st] = _v
                            _seeded += 1
                if _seeded:
                    self.sroute_seeds += _seeded
                    self._propagate_route()
                self.sroute_ms_last = round((_t.time() - _t0) * 1000.0, 1)
            except Exception:
                pass
        self._path_sa.clear()

    def _on_mastery(self) -> None:
        """A CHALLENGE was mastered (the goal escalated).  THIS is genuine new
        learning, so LIFEFORCE FOLLOWS: credit record_learning, the
        intelligence->life set-point.  Life is prolonged only by continually
        mastering NEW challenges; stop learning and the credit stops and the
        life-set-point erodes.  Un-farmable: a mastered goal moves away, so the
        same challenge cannot be re-banked.  Reset best_steps — the new goal is
        a new problem to learn."""
        self.challenges_mastered += 1
        self.best_steps = None
        if self._credit_learning is not None:
            try:
                self._credit_learning(1.0)
            except Exception:
                pass

    def _on_failure(self, cyc: int = 0) -> None:
        """The attempt ended without reaching the goal.  Solving is NOT literally
        life-or-death — failing a problem threatens nothing directly, exactly as
        a human who can't crack a puzzle is in no mortal danger (the given
        baseline of [given-baseline-energy]).  The stakes are a META-level PROXY
        to the underlying mortality architecture, never the architecture itself.

        So failure fires NO threat/mortality chemistry.  What it does is the one
        thing the earn-or-dissolve LAW demands: the route he FOLLOWED did not
        earn the goal, so it is DIS-CORROBORATED and its value falls.  This is
        the mortality architecture's PATTERN (strength held only by sustained
        corroboration, never stamped and defended against decay) applied to the
        PROXY value — not a death signal.  A route that keeps leading nowhere
        erodes until it drops below the substrate's own prune floor and is let
        go, so he abandons a dead approach on his own, with no re-aim rule.

        Heavily re-tread (stuck) states erode FASTEST for free: they recur most
        in the path, so being stuck IS the abandonment signal — no stuck-detector
        bolted on.  The opposing pole, success, RE-earns the route — the two are
        co-present, so a route in active use stays strong and only a dead one
        fades.

        A LOSS IS FELT, AND FEELING IT MAKES HIM CURIOUS (2026-08-10, user:
        *"loss makes him curious"*, *"losing is only annoying but also spurs
        his interest to do better and keep at it"*).  Until now this method
        fired NOTHING -- it adjusted route numbers in silence, so the single
        most common event in his world (a loss every 2-3 minutes) produced no
        felt state at all, and therefore also missed the companion curiosity
        that `_feel` attaches to every other feeling.  He was annoyed by
        droughts and unmoved by defeat.

        `puzzle_stress` is the annoyance channel already live in the v2 table
        and already used for being stuck: 7.5x LESS cortisol than `threat`
        and, unlike threat, it does not suppress dopamine.  Annoying, not
        frightening -- which is the whole ruling.  `_feel` then fires
        companion curiosity at `1/(1+visits)`, so a defeat on fresh ground
        opens him up loudly and one on worn ground is a whisper.

        MAGNITUDE IS HIS OWN INVESTMENT: the share of the path he just lost
        that had actually earned route value.  Dying on a route he had
        established stings; dying while wandering barely registers.  A ratio
        of his own quantities -- no constant enters."""
        _held = 0
        for st in self._path:
            v = self._route.get(st)
            if v is None:
                continue
            _held += 1
            v *= PATH_CREDIT_DECAY
            if v < EDGE_PRUNE_FLOOR:
                self._route.pop(st, None)      # let the dead route go
            else:
                self._route[st] = v
        if self._path:
            try:
                self._feel('puzzle_stress',
                           float(_held) / float(len(self._path)),
                           self._path[-1], int(cyc), 'lost')
                self.loss_fires = getattr(self, 'loss_fires', 0) + 1
            except Exception:
                pass

    def _rehydrate_doorways(self) -> None:
        """What a goal LOOKS like, read back from his own library.

        Doorways are written as `<obj> has_property _role_doorway`, so
        they already survive the process boundary; only the fast-path
        cache in the world is per-life.  Rebuild it from the substrate
        instead of persisting the same fact twice.

        Walks `_arc_obj_*` concepts only (a few hundred), never the edge
        table.  Degrades to today's behaviour on any failure."""
        self.doorways_rehydrated = 0
        try:
            seed = getattr(self.world, 'seed_doorways', None)
            if seed is None:
                return                      # not an ARC world
            sub = getattr(self.grounding, 'engine', None)
            sub = getattr(sub, 'substrate', None)
            concepts = getattr(sub, 'concepts', None)
            if not concepts:
                return
            found = []
            for name, c in list(concepts.items()):
                if not name.startswith('_arc_obj_c'):
                    continue
                hit = False
                for e in (getattr(c, 'edges_out', {}) or {}).get(
                        'has_property', ()):
                    if getattr(e, 'target', '') == '_role_doorway':
                        hit = True
                        break
                if not hit:
                    continue
                try:
                    _c, _s = name[len('_arc_obj_c'):].split('_s')
                    found.append(((int(_c), int(_s)), 1))
                except (ValueError, IndexError):
                    continue
            if found:
                self.doorways_rehydrated = int(seed(found) or 0)
        except Exception:
            self.doorways_rehydrate_errors = getattr(
                self, 'doorways_rehydrate_errors', 0) + 1

    def _rehydrate_trans(self) -> None:
        """Refill the per-life map from the PERSISTENT substrate (once).

        His substrate already records every action he has ever taken as a
        `_world_s<state>|a<N> -transitions_to-> <next>` edge.  Leaving
        `_trans` empty at startup makes every one of those actions count as
        untried, so he re-explores a world he has already mapped.  This is
        the same substrate read `_sibling_polarity` performs -- no new
        knowledge, no constant, just continuity across lives.

        Failures are COUNTED, never swallowed: a substrate that cannot be
        read leaves him exactly as he was before (empty map, full
        exploration), which is the safe degradation.
        """
        self._rehydrated = True
        try:
            sub = getattr(self.grounding, 'engine', None)
            sub = getattr(sub, 'substrate', None)
            if sub is None:
                return
            concepts = getattr(sub, 'concepts', None)
            if not concepts:
                return
            cyc = int(getattr(sub, 'cycle', 0) or 0)
            n = 0
            for name, c in list(concepts.items()):
                if not name.startswith(WORLD_TOKEN_PREFIX):
                    continue
                sep = name.rfind('|a')
                if sep <= 0:
                    continue
                try:
                    a = int(name[sep + 2:])
                except ValueError:
                    continue
                bare = name[:sep]
                best, best_s = None, -1.0
                for e in c.edges_out.get(WORLD_RELATION, ()):
                    tgt = (e.target if isinstance(e.target, str)
                           else getattr(e.target, 'name', None))
                    if tgt is None:
                        continue
                    try:
                        es = float(e.effective_strength(cyc))
                    except Exception:
                        es = 0.0
                    if es > best_s:
                        best_s, best = es, tgt
                if best is not None and (bare, a) not in self._trans:
                    self._trans[(bare, a)] = best
                    # HE HAS TRIED THIS.  _trans is only written after
                    # executing the action, so restoring the map without
                    # the tried-set makes him re-explore what he knows.
                    if _TRIEDRESTORE_ON():
                        self._tried.add((bare, a))
                        self.tried_restored = getattr(
                            self, "tried_restored", 0) + 1
                    # a persisted belief is evidence: seed it at 1 so ONE new
                    # observation ties rather than overturns it
                    self._trans_n[(bare, a)] = {best: 1}
                    n += 1
                # Every observed target feeds reachability, not just the
                # strongest -- and with ~99% of world edges pinned at the
                # prune floor, "strongest" is a coin flip anyway.
                for e in c.edges_out.get(WORLD_RELATION, ()):
                    _t = (e.target if isinstance(e.target, str)
                          else getattr(e.target, 'name', None))
                    if not _t:
                        continue
                    _bag = self._trans_all.get(bare)
                    if _bag is None:
                        self._trans_all[bare] = {_t}
                    else:
                        _bag.add(_t)
            # QUARANTINE VISIBILITY (2026-08-23, gate /root/QUARVIS_ON).
            # He already learned these transitions; quarantine merely
            # filed them where edges_out -- the only thing this method
            # and grounding._predict read -- cannot see them.  Reading
            # the pool does NOT un-quarantine: it only lets his map see
            # what he earned.  Live entries always win.
            if _QUARVIS_ON():
                _q = getattr(sub, "quarantine_edges", None) or {}
                _best = {}
                for _k, _e in list(_q.items()):
                    try:
                        _s, _r, _t = _k
                    except Exception:
                        continue
                    if _r != WORLD_RELATION or not _t:
                        continue
                    _s = str(_s)
                    _sep = _s.rfind("|a")
                    if _sep <= 0:
                        continue
                    try:
                        _a = int(_s[_sep + 2:])
                    except ValueError:
                        continue
                    _bare = _s[:_sep]
                    # NOT _trans_all: _propagate_route value-iterates
                    # over it, and both its per-sweep cost and its sweep
                    # bound scale with len(_trans_all).  Feeding it the
                    # quarantine pool cost ticks/s 2.467 -> 1.98 and
                    # breached the 2.0 gate.  Measured separately or
                    # not at all.
                    if (_bare, _a) in self._trans:
                        continue
                    try:
                        _es = float(_e.effective_strength(cyc))
                    except Exception:
                        _es = 0.0
                    _cur = _best.get((_bare, _a))
                    if _cur is None or _es > _cur[0]:
                        _best[(_bare, _a)] = (_es, str(_t))
                for _key, _v in _best.items():
                    self._trans[_key] = _v[1]
                    self._trans_n[_key] = {_v[1]: 1}
                    n += 1
                    self.trans_rehydrated_quar += 1
            self.trans_rehydrated = n
        except Exception:
            self.rehydrate_errors += 1

    def _note_value_key(self, s, a, gm, gi) -> None:
        """Fold this outcome into "what does action a do in situations that
        feel like this".

        The state's OTHER actions are its folders; this outcome teaches those
        folders about action `a`.  Recording under the state's own key for
        `a` would be circular, so that key is skipped -- the same guard the
        held-out measurement used.
        """
        tot = float(gm) + float(gi)
        if tot <= 0.0:
            return
        p = (float(gi) - float(gm)) / tot
        own = self._skeys.get(s)
        if own is None:
            own = self._skeys[s] = {}
        for aa, sign in own.items():
            if aa == a:
                continue
            cell = self._vk.get((aa, sign, a))
            if cell is None:
                self._vk[(aa, sign, a)] = [p, 1.0]
            else:
                cell[0] += p
                cell[1] += 1.0
        own[a] = 1 if p > 0 else (-1 if p < 0 else 0)
        # the folder this state now belongs to learns what being
        # here is worth.
        _c = frozenset(own.items())
        _cell = self._cval.get(_c)
        if _cell is None:
            self._cval[_c] = [p, 1.0]
        else:
            _cell[0] += p
            _cell[1] += 1.0

    def _write_trace(self, s, a, s_next, pred, resp, gm, gi, cyc):
        """One line of his own record: what he saw, what he did, what he
        expected, what happened, and how it felt.

        `substrate.Episode` was built for exactly this and had never been
        written to.  Without it his thinking cannot be followed: the
        hippocampal episode keeps how a moment FELT but not what he did
        or what the world did back.
        """
        if not _TRACE_ON():
            return
        try:
            from seagi.core.substrate import Episode
            from seagi.core.mi_value import MIValue
            sub = getattr(self.grounding, "engine", None)
            sub = getattr(sub, "substrate", None)
            if sub is None or not hasattr(sub, "add_episode"):
                return
            if pred is None:
                kind = "self_observation"
            elif s_next == pred:
                kind = "confirmation"
            else:
                kind = "falsification"
            try:
                game = str(getattr(self.world, "game_id", "") or "")
            except Exception:
                game = ""
            content = (
                "in %s I saw %s, did %s, expected %s, got %s -- the world %s"
                % (game or "?", s, a,
                   pred if pred is not None else "nothing",
                   s_next,
                   "moved" if resp else ("did not move"
                                        if resp is False
                                        else "did not say")))
            ep = Episode(
                id="ep_%d_%d" % (int(cyc), int(a)),
                cycle=int(cyc),
                kind=kind,
                concepts={str(s), str(s_next)},
                content=content,
                mi_at_event=MIValue(m=float(gm), i=float(gi), n=1),
                action_taken={"action": int(a), "game": game,
                              "from": str(s)},
                observed_outcome={"to": str(s_next),
                                  "world_changed": resp},
            )
            try:
                ep.transmitters_at_event = dict(self._tone_provider() or {})
            except Exception:
                pass
            sub.add_episode(ep)
            self._trace_ring.append({
                "cycle": int(cyc), "kind": kind, "content": content,
                "m": round(float(gm), 4), "i": round(float(gi), 4),
                "world_changed": resp,
            })
            self.trace_written += 1
        except Exception:
            self.trace_errors = getattr(self, "trace_errors", 0) + 1

    def _write_leads_to(self, s, s_next, pred, cyc):
        """`s leads_to s_next` -- bare to bare, so it CHAINS.

        The world edge grounding writes is `state|aN transitions_to
        bare`: composite source, bare target, so it can never compose.
        `leads_to` is already in RELATION_COMPOSITION and bare->bare is
        already legal, so this hands his existing inference engine the
        world without touching the farming table.

        EARNED ONLY: written when he PREDICTED this transition and was
        right.  Being surprised by a transition does not entitle him to
        assert it leads anywhere.  Deduplicated, because ~1M latent
        corroborations into `newly_coherent` is a real inflation risk.
        """
        if not _LEADSTO_ON():
            return
        if pred is None or s_next != pred:
            return
        if not s or not s_next or s == s_next:
            return
        key = (s, s_next)
        if key in self._leads_seen:
            return
        self._leads_seen.add(key)
        if len(self._leads_seen) > 200000:
            self._leads_seen.clear()
        try:
            self.bus.publish(SubstrateWriteQueuedEvent(
                kind=EventKind.SUBSTRATE_WRITE_QUEUED, cycle=int(cyc),
                source_capability="world_actor", origin="internal",
                subject=str(s), relation="leads_to", object=str(s_next),
                strength=PROVISIONAL_EDGE_STRENGTH,
                write_reason="world_reachability"))
            self.leads_written += 1
        except Exception:
            self.leads_errors = getattr(self, "leads_errors", 0) + 1

    def _seek_on_error(self, s, a, pred, s_next, cyc):
        """He was wrong.  That is not nothing and it is not punishment --
        it is the question "how do I get this right", which is the
        seeking system.

        Sized by how CONFIDENT he was while being wrong, against his own
        norm: confidently wrong is a big surprise and should provoke a
        big search; tentatively wrong a small one.  Symmetric with
        `_insight_magnitude`, which sizes by how far a settling beat his
        norm.
        """
        try:
            conf = float(getattr(self, "_pred_conf", 0.0) or 0.0)
        except Exception:
            return
        if conf <= 0.0:
            return
        # his typical confidence when wrong -- tracked always, so the
        # norm exists before the gate is ever opened
        self._err_conf_n += 1
        al = 0.01 if self._err_conf_n > 1 else 1.0
        self._err_conf = (1.0 - al) * self._err_conf + al * conf
        if not _ERRSEEK_ON():
            return
        norm = float(self._err_conf or 0.0)
        if norm <= 0.0 or self._err_conf_n < 20:
            mag = 1.0
        else:
            mag = max(0.0, min(2.0, conf / norm))
        if mag <= 0.0:
            return
        try:
            self.bus.publish(ChemistryEvent(
                kind=EventKind.CHEMISTRY_FIRE, cycle=int(cyc),
                timestamp=time.time(),
                source_capability="world_actor", origin="internal",
                origin_detail="wrong_about:%s" % (a,),
                chemistry_kind="curiosity", magnitude=mag,
                target_concepts=[s_next]))
            self.seek_fires += 1
        except Exception:
            pass

    def _insight_magnitude(self) -> float:
        """How big THIS understanding was, against his own norm.

        A flat 1.0 made every insight the same size, so the mechanism
        could only be all-or-nothing.  Achievement is relative:
        seeing something, understanding a move and reading the goal
        are all small doses, and some are bigger than others.
        Falls back to 1.0 until a norm exists, so gate-off behaviour
        and the cold start are unchanged.
        """
        # Sized from its own norm, independent of the chemistry spread:
        # these are two different measurements and coupling them would
        # make a converged calibration wait on an unconverged one.
        try:
            norm = float(getattr(self, '_ins_exc', 0.0) or 0.0)
            if norm <= 0.0 or int(getattr(self, '_ins_exc_n', 0)) < 20:
                return 1.0
            exc = (float(getattr(self, '_pred_conf', 0.0) or 0.0)
                   - float(self._tex_conf() or 0.0))
            if exc <= 0.0:
                return 0.0
            return max(0.0, min(2.0, exc / norm))
        except Exception:
            return 1.0

    def _value_prior(self, s, novel) -> Dict[int, float]:
        """Expected lean of each UNTRIED action, from the folders `s` is in.

        Mean over the state's keys -- no threshold, no weight.  Silent
        (empty) when a state has no keys yet or none of its folders have
        met the action, so a cold start behaves exactly as before.
        """
        keys = list((self._skeys.get(s) or {}).items())
        if not keys:
            return {}
        out = {}
        for x in novel:
            tot = n = 0.0
            for (aa, sign) in keys:
                cell = self._vk.get((aa, sign, x))
                if cell and cell[1] > 0.0:
                    tot += cell[0] / cell[1]
                    n += 1.0
            if n > 0.0:
                out[x] = tot / n
        if out:
            self.vkey_priors += 1
        return out

    def _class_value(self, s) -> float:
        """How life-ward the FOLDER of state `s` has felt, in [-1, 1].

        Same units as `_deflection` (an M/I polarity), so the two are two
        measurements of one quantity and can be averaged.  Silent (0.0) for
        an unknown or unclassified state, which makes selection reduce to
        its previous behaviour exactly.
        """
        if s is None:
            return 0.0
        own = self._skeys.get(s)
        if not own:
            return 0.0
        cell = self._cval.get(frozenset(own.items()))
        if not cell or cell[1] <= 0.0:
            return 0.0
        v = cell[0] / cell[1]
        return 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)

    # ---- where-stream ------------------------------------------------
    def _where_code(self):
        """His glance locus, coarsened.  None when the world has no
        locus, in which case the where-stream simply does not apply and
        everything falls through to the texture-only map."""
        try:
            lo = getattr(self.world, "_locus", None)
            if lo is None:
                return None
            g = self._where_grain
            return (int(lo[0]) // g, int(lo[1]) // g)
        except Exception:
            return None

    @staticmethod
    def _earned(hits, n):
        """Measured hit rate discounted by how much has been measured --
        hit_rate x n/(1+n).  Same self-scaling form as the self-model
        confidence; no threshold, no constant, and an untested source
        scores 0 so it can never displace a tested one."""
        if n <= 0.0:
            return 0.0
        return (hits / n) * (n / (1.0 + n))

    def _pos_conf(self):
        return self._earned(self._pos_hits, self._pos_n)

    def _place_conf(self):
        return self._earned(self._place_hits, self._place_n)

    def _world_conf(self):
        return self._earned(self._world_hits, self._world_n)

    def _game_key(self):
        try:
            return str(getattr(self.world, "game_id", "") or "")
        except Exception:
            return ''

    def _tex_conf(self):
        return self._earned(self._tex_hits, self._tex_n)

    def learning_to_dict(self) -> Dict[str, Any]:
        """Everything he has learned by PLAYING.  Substrate-independent."""
        try:
            return {
                # HIS SENSE OF A TYPICAL INSIGHT, built over hundreds of
                # settlings.  Structure, not mood: without it he wakes
                # unable to tell a large understanding from a small one.
                'ins_exc': float(getattr(self, '_ins_exc', 0.0) or 0.0),
                'ins_exc_n': int(getattr(self, '_ins_exc_n', 0) or 0),
                'skeys': {str(s): {str(a): int(v) for a, v in d.items()}
                          for s, d in self._skeys.items() if d},
                'vk': {'%s|%s|%s' % k: [float(c[0]), float(c[1])]
                       for k, c in self._vk.items()},
                # THE WORLD MODEL: (state, action) -> what that PLACE
                # became.  Persisted here rather than as substrate
                # edges because the gaze map already owns that subject
                # key and only the target differs -- writing both would
                # corrupt `_trans` on rehydrate.
                'twld': {'%s|%s' % (k[0], k[1]): str(v)
                         for k, v in self._trans_world.items()},
                'cval': {','.join('%s:%s' % (a, s)
                                  for a, s in sorted(c)): [float(v[0]),
                                                           float(v[1])]
                         for c, v in self._cval.items()},
            }
        except Exception:
            return {}

    def learning_from_dict(self, d) -> int:
        """Inherit it.  Tolerant: a malformed entry is skipped, never fatal
        -- a bad payload must leave him exactly as a fresh life, not dead."""
        n = 0
        try:
            _ie = d.get('ins_exc')
            if _ie is not None and float(_ie) > 0.0:
                self._ins_exc = float(_ie)
                self._ins_exc_n = int(d.get('ins_exc_n') or 0)
        except (TypeError, ValueError):
            pass
        try:
            for s, dd in (d.get('skeys') or {}).items():
                try:
                    self._skeys[str(s)] = {int(a): int(v)
                                           for a, v in dd.items()}
                    n += 1
                except (TypeError, ValueError):
                    continue
            for k, c in (d.get('vk') or {}).items():
                try:
                    aa, sign, x = str(k).split('|')
                    self._vk[(int(aa), int(sign), int(x))] = [float(c[0]),
                                                              float(c[1])]
                except (TypeError, ValueError, IndexError):
                    continue
            for k, v in (d.get('twld') or {}).items():
                try:
                    _st, _a = str(k).rsplit('|', 1)
                    self._trans_world[(_st, int(_a))] = str(v)
                    n += 1
                except (TypeError, ValueError, IndexError):
                    continue
            for k, v in (d.get('cval') or {}).items():
                try:
                    if not k:
                        continue
                    cls = frozenset((int(p.split(':')[0]), int(p.split(':')[1]))
                                    for p in str(k).split(',') if ':' in p)
                    self._cval[cls] = [float(v[0]), float(v[1])]
                except (TypeError, ValueError, IndexError):
                    continue
        except Exception:
            pass
        self.learning_restored = n
        return n

    def _chain_backup(self, s, s_next) -> None:
        """One Bellman backup, backward, per step.

        A state is worth the best it can REACH, not merely what it is.  This
        is the same backward spread `_propagate_route` performs after a win,
        but seeded continuously by how situations FEEL -- so it works before
        any completion has ever happened, which on this game is the whole
        problem (`_route` seeds only on a win he has never had).

        O(1): no sweep, no schedule.  Uses the decay the route already uses,
        so a distant good place is worth less than a near one without any
        new constant.  Clamped to the deflection range so a chain can never
        outgrow the scale it is averaged into.
        """
        try:
            nxt = self._chain.get(s_next)
            if nxt is None:
                nxt = self._class_value(s_next)
            v = PATH_CREDIT_DECAY * float(nxt)
            own = self._class_value(s)
            best = own if own > v else v
            if best > 1.0:
                best = 1.0
            elif best < -1.0:
                best = -1.0
            if best != self._chain.get(s):
                self._chain[s] = best
                self.chain_backups += 1
        except Exception:
            self.chain_errors = getattr(self, 'chain_errors', 0) + 1

    def _reachable_value(self, s) -> float:
        """Chained value if he has one, else what the folder alone says."""
        if s is None:
            return 0.0
        v = self._chain.get(s)
        return float(v) if v is not None else self._class_value(s)

    def _maybe_propagate_explore(self, level: int) -> None:
        """Value-iterate UNEXPLOREDNESS backward over `_trans_all`.

        Same shape as `_propagate_route`, different seed: the goal is
        the FRONTIER -- never visited, or still holding an untried
        action.  Gives him a gradient on a level he has never
        completed, which is the only kind that matters for depth.
        """
        _c = int(getattr(self, "trans_writes", 0) or 0)
        if (self._explore_level == level
                and _c - self._explore_cyc < EXPLORE_EVERY):
            return
        self._explore_level = level
        self._explore_cyc = _c
        try:
            er = {}
            for s in self._trans_all:
                if self._visits.get(s, 0) == 0:
                    er[s] = 1.0
            for (s, a) in self._trans:
                if (s, a) not in self._tried:
                    er[s] = 1.0
            self.explore_seeds = len(er)
            if not er:
                self._explore_route = {}
                return
            for _ in range(EXPLORE_SWEEPS):
                changed = False
                for s, nxts in self._trans_all.items():
                    best = 0.0
                    for nxt in nxts:
                        v = PATH_CREDIT_DECAY * er.get(nxt, 0.0)
                        if v > best:
                            best = v
                    if best > er.get(s, 0.0):
                        er[s] = best
                        changed = True
                if not changed:
                    break
            self._explore_route = er
            self.explore_recomputes = getattr(
                self, "explore_recomputes", 0) + 1
        except Exception:
            self.explore_errors = getattr(self, "explore_errors", 0) + 1

    def _is_inert_here(self, s, a) -> bool:
        """Has he watched this move fail to move the board HERE?

        Requires >= 2 observations, so one unlucky step never condemns
        an action -- "every mistake is made once".  Absent (untried)
        returns False, so exploration is never narrowed.
        """
        try:
            _r = self._noop_seen.get((s, a))
            return bool(_r is not None and _r[0] >= 2 and _r[1] == 0)
        except Exception:
            return False

    def _propagate_route(self) -> None:
        """Establish the route: spread the goal's value backward through his
        learned transitions, so each state knows the way to the goal.
        route[s] = max over actions of PATH_CREDIT_DECAY * route[next(s,a)].
        Sweep until the values stop changing (the fixpoint).  The number of
        sweeps needed is the goal's reach DIAMETER — which the WORLD's structure
        decides, not a stamped constant: a shallow grid converges in a few
        sweeps, a deep puzzle (a route many moves long) takes as many as the
        problem is deep.  Bounded by the learned-edge count (the convergence
        bound for this backward value spread) only to guarantee termination.
        This is the success NT-signal ESTABLISHING a followable route (not just
        a point tag) — turning his scattered memories of past attempts into a
        plan he can follow, so each solve gets shorter."""
        # Walk `_trans_all` (every lived transition), NOT `_trans` (the single
        # best guess).  Measured 2026-07-31: over `_trans` a seeded goal
        # reaches a median of 359 states and his three real goals reach 6;
        # over the full relation the median is 5,689 of 9,552.  Value has to
        # be able to flow to him or there is no gradient to follow.
        for _ in range(len(self._trans_all) + 1):
            changed = False
            for s, nxts in self._trans_all.items():
                best_v = 0.0
                for nxt in nxts:
                    v = PATH_CREDIT_DECAY * self._route.get(nxt, 0.0)
                    if v > best_v:
                        best_v = v
                if best_v > self._route.get(s, 0.0):
                    self._route[s] = best_v
                    changed = True
            if not changed:
                break

    # ---- bus entry: execute the arbitrated motor decision ----
    @staticmethod
    def _decode(action: str) -> Optional[int]:
        if not action or ':' not in action:
            return None
        try:
            return int(action.split(':', 1)[1])
        except (ValueError, IndexError):
            return None

    def handle(self, event: Any, bus: Any) -> None:
        if getattr(event, 'kind', None) != EventKind.ARBITRATION_DECIDED:
            return
        if getattr(event, 'loop', '') != 'motor':
            return
        a = self._decode(getattr(event, 'winning_action', ''))
        if a is not None:
            self.execute(a, int(getattr(event, 'cycle', 0)))

    # ---- diagnostics (read-only) ----
    @property
    def states_seen(self) -> int:
        return len(self._seen)

    def mean_solve(self) -> float:
        return (sum(self.solve_steps) / len(self.solve_steps)
                if self.solve_steps else 0.0)

    def stats(self) -> Dict[str, Any]:
        attempts = self.successes + self.failures
        return {
            'trans_pos_size': len(self._trans_pos),
            'pos_conf': self._pos_conf(),
            'tex_conf': self._tex_conf(),
            'pos_n': self._pos_n,
            'tex_n': self._tex_n,
            'pos_guess_used': self.pos_guess_used,
            'trans_place_size': len(self._trans_place),
            'place_n': self._place_n,
            'place_conf': self._place_conf(),
            'place_guess_used': self.place_guess_used,
            'trans_world_size': len(self._trans_world),
            'world_n': self._world_n,
            'world_conf': self._world_conf(),
            'world_guess_used': self.world_guess_used,
            'world_static_rate': (self._world_static / self._world_n)
                                 if self._world_n else None,
            'world_says_static_rate': (self._world_says_static
                                       / self._world_n)
                                      if self._world_n else None,
            'world_chg_n': self._world_chg_n,
            'world_chg_rate': (self._world_chg_hits / self._world_chg_n)
                              if self._world_chg_n else None,
            'world_raw_rate': (self._world_hits / self._world_n)
                              if self._world_n else None,
            'executions': self.executions,
            'explorations': self.explorations,
            'explorations_directed': self.explorations_directed,
            'explorations_fallback': self.explorations_fallback,
            'explorations_route': self.explorations_route,
            'route_ties': self.route_ties,
            'cur_level': self._cur_level,
            'deep_directed': self._deep_directed,
            'deep_fallback': self._deep_fallback,
            'deep_route': self._deep_route,
            'goal_seed_was_levelup': bool(self._goal_seed_was_levelup),
            'goal_seed_level': int(self._goal_seed_level),
            'goal_seed_route_val': float(
                self._route.get(self._goal_seed, 0.0)
                if self._goal_seed is not None else 0.0),
            'seed_states_n': len(self._seed_states),
            'deep_seed_revisits': int(self._deep_seed_revisits),
            'deep_anyseed_revisits': int(self._deep_anyseed_revisits),
            'ignitions': self.ignitions,
            'dead_actions': self.dead_actions,
            'insight_fires': getattr(self, 'insight_fires', 0),
            'ins_shadow_A': getattr(self, '_ins_A', 0),
            'ins_shadow_B': getattr(self, '_ins_B', 0),
            'ins_shadow_C': getattr(self, '_ins_C', 0),
            'ins_stasis': getattr(self, '_ins_stasis', 0),
            'ins_rule_new': getattr(self, '_ins_R1', 0),
            'ins_rule_transfer': getattr(self, '_ins_R2', 0),
            'wpe_pos': getattr(self, '_wpe_pos', 0),
            'wpe_neg': getattr(self, '_wpe_neg', 0),
            'wpe_pos_mean': (getattr(self, '_wpe_pos_mag', 0.0)
                             / max(1, getattr(self, '_wpe_pos', 0))),
            'wpe_neg_mean': (getattr(self, '_wpe_neg_mag', 0.0)
                             / max(1, getattr(self, '_wpe_neg', 0))),
            'wpe_fired': getattr(self, '_wpe_fired', 0),
            'wpe_on': bool(_WORLD_PE_ON()),
            'insight_on': bool(_INSIGHT_ON()),
            'trace_written': int(getattr(self, 'trace_written', 0)),
            'trace_errors': int(getattr(self, 'trace_errors', 0)),
            'trace_on': bool(_TRACE_ON()),
            'seek_fires': int(getattr(self, 'seek_fires', 0)),
            'err_conf_norm': round(float(getattr(self, '_err_conf', 0.0)), 6),
            'err_conf_n': int(getattr(self, '_err_conf_n', 0)),
            'errseek_on': bool(_ERRSEEK_ON()),
            'leads_written': int(getattr(self, 'leads_written', 0)),
            'leads_errors': int(getattr(self, 'leads_errors', 0)),
            'leads_pairs': len(getattr(self, '_leads_seen', ()) or ()),
            'leadsto_on': bool(_LEADSTO_ON()),
            'quarvis_on': bool(_QUARVIS_ON()),
            'quarev_on': bool(_QUAREV_ON()),
            'quar_evidence_hits': int(getattr(self, 'quar_evidence_hits', 0)),
            'triedrestore_on': bool(_TRIEDRESTORE_ON()),
            'tried_restored': int(getattr(self, 'tried_restored', 0)),
            'tried_pairs': len(getattr(self, '_tried', ()) or ()),
            'routelocal_on': bool(_ROUTELOCAL_ON()),
            'route_local_blocks': int(getattr(self, 'route_local_blocks', 0)),
            'noopdemote_on': bool(_NOOPDEMOTE_ON()),
            'noop_demotes': int(getattr(self, 'noop_demotes', 0)),
            'cycle_demotes': int(getattr(self, 'cycle_demotes', 0)),
            'cycleall_on': bool(_CYCLEALL_ON()),
            'cycle_asks_all': int(getattr(self, 'cycle_asks_all', 0)),
            'cycle_demotes_all': int(getattr(
                self, 'cycle_demotes_all', 0)),
            'cycle_allcycle': int(getattr(self, 'cycle_allcycle', 0)),
            'verdict_on': bool(_VERDICT_ON()),
            'verdictgood_on': bool(_VERDICTGOOD_ON()),
            'verdict_asks': int(getattr(self, 'verdict_asks', 0)),
            'verdict_steers': int(getattr(self, 'verdict_steers', 0)),
            'verdict_errors': int(getattr(self, 'verdict_errors', 0)),
            'untriedhere_on': bool(_UNTRIEDHERE_ON()),
            'untried_asks': int(getattr(self, 'untried_asks', 0)),
            'untried_steers': int(getattr(self, 'untried_steers', 0)),
            'untried_kept_good': int(getattr(
                self, 'untried_kept_good', 0)),
            'untried_errors': int(getattr(self, 'untried_errors', 0)),
            'felt_changes': int(getattr(self, 'felt_changes', 0)),
            'felt_errors': int(getattr(self, 'felt_errors', 0)),
            'felt_err_kinds': dict(getattr(self, 'felt_err_kinds', {}) or {}),
            # THE FUNNEL.  felt reached -0.46, ten times below the -0.05
            # threshold, and the steer still never fired.  These say which
            # of the three conditions is the one that blocks it.
            'felt_low_n': int(getattr(self, 'felt_low_n', 0)),
            'felt_seen_n': int(getattr(self, 'felt_seen_n', 0)),
            'felt_repeat_n': int(getattr(self, 'felt_repeat_n', 0)),
            'explorer_on': bool(_EXPLORER_ON()),
            'explorer_steers': int(getattr(self, 'explorer_steers', 0)),
            'explorer_errors': int(getattr(self, 'explorer_errors', 0)),
            'noopall_on': bool(_NOOPALL_ON()),
            'egomove_on': bool(_EGOMOVE_ON()),
            'ego_asks': int(getattr(self, 'ego_asks', 0)),
            'ego_steers': int(getattr(self, 'ego_steers', 0)),
            'ego_errors': int(getattr(self, 'ego_errors', 0)),
            'noop_asks_all': int(getattr(self, 'noop_asks_all', 0)),
            'noop_demotes_all': int(getattr(
                self, 'noop_demotes_all', 0)),
            'noop_allinert': int(getattr(self, 'noop_allinert', 0)),
            'noop_errors_all': int(getattr(self, 'noop_errors_all', 0)),
            'cycle_errors': int(getattr(self, 'cycle_errors', 0)),
            'noop_pairs': len(getattr(self, '_noop_seen', ()) or ()),
            'noop_inert_pairs': sum(
                1 for _v in (getattr(self, '_noop_seen', {}) or {}).values()
                if _v[0] >= 2 and _v[1] == 0),
            'progress_fires': int(getattr(self, 'progress_fires', 0)),
            'relsteer_on': bool(_RELSTEER_ON()),
            'relplan_on': bool(_RELPLAN_ON()),
            'hyp_acts': int(getattr(self, 'hyp_acts', 0)),
            'hyp_executed': int(getattr(self, 'hyp_executed', 0)),
            'hyp_over_depth': int(getattr(self, 'hyp_over_depth', 0)),
            'hyp_over_route': int(getattr(self, 'hyp_over_route', 0)),
            'hyp_overruled': int(getattr(self, 'hyp_overruled', 0)),
            'hyp_errors': int(getattr(self, 'hyp_errors', 0)),
            'srch_acts': int(getattr(self, 'srch_acts', 0)),
            'srch_executed': int(getattr(self, 'srch_executed', 0)),
            'srch_over_depth': int(getattr(self, 'srch_over_depth', 0)),
            'srch_over_route': int(getattr(self, 'srch_over_route', 0)),
            'srch_overruled': int(getattr(self, 'srch_overruled', 0)),
            'srch_errors': int(getattr(self, 'srch_errors', 0)),
            'rel_plan_acts': int(getattr(self, 'rel_plan_acts', 0)),
            'rel_plan_overruled': int(getattr(self, 'rel_plan_overruled', 0)),
            'rel_plan_shortcut': int(getattr(self, 'rel_plan_shortcut', 0)),
            'leadhold_on': bool(_LEADHOLD_ON()),
            'rel_plan_lead_kept': int(getattr(self, 'rel_plan_lead_kept', 0)),
            'rel_plan_lead_route': int(getattr(self, 'rel_plan_lead_route', 0)),
            'rel_plan_lead': str(self._lead_now() or ''),
            'rel_plan_over_depth': int(getattr(self, 'rel_plan_over_depth', 0)),
            'rel_plan_errors': int(getattr(self, 'rel_plan_errors', 0)),
            'rel_asks': int(getattr(self, 'rel_asks', 0)),
            'rel_steers': int(getattr(self, 'rel_steers', 0)),
            'rel_learned': int(getattr(self, 'rel_learned', 0)),
            'rel_errors': int(getattr(self, 'rel_errors', 0)),
            'rel_act_keys': len(getattr(self, '_rel_act', {}) or {}),
            'rel_insights': int(getattr(self, 'rel_insights', 0)),
            'progress_errors': int(getattr(self, 'progress_errors', 0)),
            'prog_mean': round(float(getattr(self, '_prog_mean', 0.0)), 3),
            'explroute_on': bool(_EXPLROUTE_ON()),
            'explore_seeds': int(getattr(self, 'explore_seeds', 0)),
            'explore_recomputes': int(getattr(self, 'explore_recomputes', 0)),
            'explore_steers': int(getattr(self, 'explore_steers', 0)),
            'explore_steers_live': int(getattr(self, 'explore_steers_live', 0)),
            'explore_errors': int(getattr(self, 'explore_errors', 0)),
            'explore_size': len(getattr(self, '_explore_route', ()) or ()),
            'trans_rehydrated_quar': int(getattr(self, 'trans_rehydrated_quar', 0)),
            'trace': list(getattr(self, '_trace_ring', []) or []),
            'ins_exc_norm': round(float(getattr(self, '_ins_exc', 0.0)), 6),
            'ins_exc_n': int(getattr(self, '_ins_exc_n', 0)),
            'ins_mag_now': round(self._insight_magnitude(), 4),
            'trans_rehydrated': self.trans_rehydrated,
            'trans_writes': self.trans_writes,
            'trans_flips': self.trans_flips,
            'trans_recency_flips': self.trans_recency_flips,
            'trans_multi': sum(1 for v in self._trans_n.values()
                               if len(v) > 1),
            'trans_keys': len(self._trans_n),
            'vkey_priors': self.vkey_priors,
            'class_value_steers': self.class_value_steers,
            'chain_backups': self.chain_backups,
            'chain_size': len(self._chain),
            'chain_errors': getattr(self, 'chain_errors', 0),
            'learning_restored': self.learning_restored,
            'class_count': len(self._cval),
            'efficacy_confirms': getattr(self, 'efficacy_confirms', 0),
            'vkey_cells': len(self._vk),
            'route_restored': self.route_restored,
            'route_size': len(self._route),
            'trans_all_states': len(self._trans_all),
            'trans_all_edges': sum(len(v) for v in self._trans_all.values()),
            'rehydrate_errors': self.rehydrate_errors,
            'frontier_ordered': self.frontier_ordered,
            'recognised_states': len(self._sib_cache),
            'frontier_errors': self.frontier_errors,
            'successes': self.successes,
            'goal_reached_fires': getattr(
                self, 'goal_reached_fires', 0),
            'failures': self.failures,
            'success_rate': round(self.successes / attempts, 3)
            if attempts else 0.0,
            'best_steps': self.best_steps,
            'challenges_mastered': self.challenges_mastered,
            'curiosity_fires': getattr(self, 'curiosity_fires', 0),
            'stuck_fires': getattr(self, 'stuck_fires', 0),
            'loss_fires': getattr(self, 'loss_fires', 0),
            'fit_fires': getattr(self, 'fit_fires', 0),
            'urgency_fires': getattr(self, 'urgency_fires', 0),
            'obj_n': getattr(self, '_obj_n', None),
            'self_sig': getattr(self, '_self_sig', None),
            'self_determinism': getattr(self, '_self_det', None),
            'self_moves_seen': getattr(self, 'self_moves_seen', 0),
            'consumables': getattr(self, '_consumables', None),
            'vanish_events': getattr(self, '_vanish_events', 0),
            'occlusion': getattr(self, '_occl', None),
            'rules_known': getattr(self, '_rules_known', None),
            'rule_transfers': getattr(self, '_rule_transfers', None),
            'rule_seeds': getattr(self, '_rule_seeds', None),
            'rules_retrodicted': getattr(self, '_rules_retro', None),
            'target_salience': getattr(self, '_tgt_sal', None),
            'approach_steps': getattr(self, '_approach_n', None),
            'kinds_investigated': getattr(self, '_kinds_inv', None),
            'target_commits': getattr(self, '_commits', None),
            'seed_levels': sorted(getattr(self, '_seed_levels', set())),
            'doorways_written': getattr(self, 'doorways_written', 0),
            'doorways_rehydrated': getattr(self, 'doorways_rehydrated', 0),
            'deep_steps': getattr(self, '_deep_steps', 0),
            'deep_untethered': getattr(self, '_deep_untethered', 0),
            'wander_incomplete': getattr(self, 'wander_incomplete', 0),
            'deep_seed_revisits': getattr(self, '_deep_seed_revisits', 0),
            'roles_written': getattr(self, 'roles_written', 0),
            'counts_written': getattr(self, 'counts_written', 0),
            'role_rehearsals': getattr(self, 'role_rehearsals', 0),
            'roles_now': getattr(self, '_roles_now_cache', None),
            'rules_multi_game': getattr(self, '_rules_multi', None),
            'mem_sizes': getattr(self, '_mem_sizes', None),
            'self_conf': getattr(self, '_self_conf', 0.0),
            'frontier_steers': getattr(self, 'frontier_steers', 0),
            # DEPLOY GATE: the click share of steers must not exceed the
            # click share of the actions on offer.  If it does, the
            # frontier has collapsed onto the coordinate action.
            'frontier_steers_click': getattr(
                self, 'frontier_steers_click', 0),
            'frontier_steers_move': getattr(
                self, 'frontier_steers_move', 0),
            'positions_seen': getattr(self, '_positions_seen', None),
            'body_nav_steers': getattr(self, 'body_nav_steers', 0),
            'unknown_pref': getattr(self, 'unknown_pref', 0),
            'companion_curiosity': getattr(
                self, 'companion_curiosity', 0),
            'bar_resets': getattr(self, 'bar_resets', 0),
            'bar_fill': getattr(self, '_bar_prev', None),
            'tone_scored': getattr(self, 'tone_scored', 0),
            'tone_engaged': getattr(self, 'tone_engaged', 0),
            'tone_flips': getattr(self, 'tone_flips', 0),
            'tone_steering': bool(_TONE_STEER_ON()),
            'lean_mean': round(getattr(self, '_lean_mean', 0.0), 5),
            'lean_mad': round(getattr(self, '_lean_mad', 0.0), 5),
            'mean_solve_recent': round(self.mean_solve(), 1),
            'states_seen': len(self._seen),
            'survival_on': bool(_SURVIVAL),
            'mi_base_on': bool(_MI_BASE),
            'mi_earned_on': bool(_MI_EARNED),
            'sroute_on': bool(_SROUTE),
            'sroute_seeds': getattr(self, 'sroute_seeds', 0),
            'sroute_ms_last': getattr(self, 'sroute_ms_last', 0.0),
            'svalue_states': len(getattr(self, '_svalue', ())),
            'svalue_lean': (lambda _v: {
                'n': len(_v),
                'min': round(min(_v), 5) if _v else None,
                'med': round(sorted(_v)[len(_v) // 2], 5) if _v else None,
                'max': round(max(_v), 5) if _v else None,
                'distinct': len(set(round(x, 6) for x in _v)),
            })([c[1] for c in getattr(self, '_svalue', {}).values()]),
            'mi_earned_steers': getattr(self, 'mi_earned_steers', 0),
            'mi_base_steers': getattr(self, 'mi_base_steers', 0),
            'survival_walks': getattr(self, 'survival_walks', 0),
            'survival_steps_tagged': getattr(self, 'survival_steps_tagged', 0),
            'survival_saturated': getattr(self, 'survival_saturated', 0),
            'lives_recorded': len(getattr(self, '_lives', ())),
            'last_lifespan': getattr(self, 'last_lifespan', 0),
            'last_lean': getattr(self, 'last_lean', 0.0),
            'mean_lifespan': (round(sum(self._lives) / len(self._lives), 1)
                              if getattr(self, '_lives', None) else 0.0),
        }
