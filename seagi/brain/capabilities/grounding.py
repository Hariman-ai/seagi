"""Grounding world-model loop — BUILD 1 of the world+imagination
keystone (2026-06-03).

The first organ that makes substrate edges earn by PREDICTING A WORLD
rather than by internal text-coherence.  A stream of world-states
arrives (modality='world'); for each step the loop predicts the next
state from the substrate, then sees the actual next state:

  - CONFIRM (predicted == actual): the transition edge EARNED.  It is
    written with write_reason='world_confirm', which reinforces it AND
    stamps last_engaged_cycle — the SINGLE survival signal.  Under
    edge-mortality a `transitions_to` edge can never cohere (it is NOT
    in RELATION_COMPOSITION), so it is permanently first_coherent_cycle
    ==0 and survives ONLY by being re-predicted correctly: predict
    right = engaged = lives.
  - MISS (predicted != actual): the predicted edge gets NOTHING (no
    stamp) and ages toward the reaper; the correct transition is
    written provisionally with write_reason='world_observe', which the
    writer applies with engage=False — observation alone does NOT grant
    survival.

The reaper IS the fitness function (no new fitness constant): a
transition the world stopped producing, or that the agent keeps
mis-predicting, goes stale (last_engaged not refreshed), decays to the
prune floor, and is reaped.  Earn-or-dissolve via prediction.

DECOUPLED from lifeforce (audit must-fix 1): `transitions_to` is kept
OUT of RELATION_COMPOSITION, so world prediction never sets
first_coherent_cycle and so never reaches mortality's record_learning.
This is a precursor; the lifeforce stake ("modelling reality extends
life") is a separate, separately-audited 1b build.

GENERALIZATION, not lookup (the unfakeable bar): when a state has no
transition of its own, the loop predicts by INHERITANCE — the majority
transition of its is_a siblings (states of the same kind).  A held-out
state never observed transitioning is predicted correctly by what its
kind does.  A lookup table scores zero on held-out states (see
test_grounding); that gap is what separates grounding from memorisation.

PRECISION-WEIGHTING (Capability 1, now BUILT — read-only instrument):
each state's predictive reliability is tracked online (Welford over
hit/miss) — a self-derived inverse-variance, no hand-tuned confidence
threshold.  In Capability 1 it ONLY instruments (reported in stats());
the proven-grounding gate reads its cross-state spread to decide
grounding is real before any lifeforce stake is wired (Capability 3).
Adaptive use on a stochastic world is the Phase-2 build.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from ..events import EventKind, BrainEvent, SubstrateWriteQueuedEvent
from ..bus import EventBus
from seagi.core.substrate import PROVISIONAL_EDGE_STRENGTH, FACET_RELATION


# The relation the world loop predicts over.  Deliberately NOT a member
# of RELATION_COMPOSITION (that would let transitions cohere -> reach
# lifeforce; the decoupling depends on its absence).
WORLD_RELATION = 'transitions_to'

# Diagnostic history ring — NOT a behavioural constant.  Each observe()
# makes AND resolves its prediction within the same call, so this deque
# never gates behaviour; it only bounds the recent-prediction history
# used to report a windowed confirm-rate (audit reservation 4: a memory
# cap, not a fitness threshold).
PREDICTION_HISTORY = 512


class GroundingLoop:
    """Predict the next world-state from the substrate; earn-or-dissolve
    transition edges by predictive success.  Decoupled from lifeforce."""

    # Live world source (2026-06-03): the attended-percept stream —
    # Seagi's own spotlight of attention.  The loop predicts the NEXT
    # concept he attends to from the current one, earning transition
    # edges that anticipate his experience.  Perception becomes a
    # predictive act, and the substrate earns against real (if still
    # secondhand-text) input rather than a toy world.
    SUBSCRIPTIONS = (EventKind.ATTENDED_PERCEPT,)

    def __init__(self, engine: Any = None, bus: Any = None,
                 generalize: bool = True):
        self.engine = engine
        self.bus = bus
        # Sibling-inheritance generalization (the held-out-state path).
        # The reverse-is_a index this once waited on NOW EXISTS
        # (substrate._is_a_children -> substrate.is_a_siblings), and the
        # toggle is ON live (runtime.py passes generalize=True), so the
        # sibling lookup is O(siblings), not an O(concepts) scan.
        # MEASURED 2026-07-24: the path is on but NEAR-EMPTY — only 0.2%
        # of world-state concepts have a sibling that owns a transition,
        # and 92.8% own a transition themselves so `_predict` returns
        # 'own' before reaching here.  `generalized` therefore reads 0.
        # The blocker is class membership on LIVE world states, not this
        # toggle.  Do not re-read this flag as the reason.
        self.generalize = generalize
        self._last_state: Optional[str] = None
        self.history: Deque[Tuple[int, str, Optional[str], str]] = deque(
            maxlen=PREDICTION_HISTORY)
        self.predictions_made: int = 0
        self.confirms: int = 0
        self.misses: int = 0
        self.no_prediction: int = 0
        self.generalized: int = 0   # confirms that came from inheritance
        # inheritances served by the BARE-grain fallback specifically,
        # so its contribution is visible rather than merged away.
        self.generalized_bare: int = 0
        # WHY generalization never fires (2026-08-06).  Separates
        # 'never reached' from 'reached but the state had no class'
        # from 'had siblings, none owning that action'.
        self.inherit_reached: int = 0
        self.inherit_no_siblings: int = 0
        # was only ever reached via getattr(..., 0), so it never had
        # an initialiser; it is a real counter now.
        self.inherit_served: int = 0
        # predictions where the facet grain supplied the voters
        self.facet_grain_used: int = 0
        self.inherit_sibs_no_trans: int = 0
        # a like-state candidate EXISTED (whether or not it won), so
        # 'never offered' is distinguishable from 'offered and lost'.
        self.inherit_offered: int = 0
        # running global hit rate: what a prediction of his is
        # typically worth.  Prices a source that has no measurement of
        # its own, so a structurally-rarely-measured limb is not
        # silently switched off.
        self._rel_hits: float = 0.0
        self._rel_n: float = 0.0
        # the INHERIT CHANNEL's own measured hit rate.  An unmeasured
        # sibling is priced at this, not at his global average --
        # pricing it at the global average made inheritance win on
        # assumption and cost him 8.6% vs 33.4% accuracy.
        self._inh_hits: float = 0.0
        self._inh_n: float = 0.0
        # Per-state predictive reliability (Welford over hit/miss) — the
        # self-derived precision the proven-grounding gate will read.
        self._rel: Dict[str, list] = {}
        # PER-SOURCE RELIABILITY (2026-08-13).  `_rel` is now the
        # state's OWN-transition record ONLY; inherited outcomes go
        # here instead, so an inherited miss can never blacken the
        # state's own rate and strand it on the inherit channel.
        self._rel_inh: Dict[str, list] = {}
        # which source actually served each prediction
        self.pred_own: int = 0
        self.pred_inherit: int = 0
        self.pred_both: int = 0
        # co-presence outcome: they agreed / they did not
        self.agree: int = 0
        self.disagree: int = 0
        # confirms where BOTH sources said the same thing and were
        # right -- corroborated knowledge, the thing worth having
        self.corroborated: int = 0
        # inherit scored WITHOUT being acted on (the un-starving)
        self.shadow_inherit_n: int = 0
        # --- facet dock: SPLIT-EVENT CONTRASTIVE earning (ignition build,
        # 2026-07-21; C2 v2).  The token-argmax dock (dock_predict) was
        # DELETED from the live path — prediction falls out of RECALL (the
        # ignition publication in world_actor), never out of a computed
        # vote.  What remains here is the EARNING channel for has_facet
        # edges: at any first-contact (s, a), electors (facet-sharing
        # states holding their OWN (sp, a) transition) are scored on the
        # 2-class signature sig(x, a) = (next == x) (self-loop vs move).
        # Strength moves ONLY where elector signatures DISAGREE (a split):
        # unanimity carries zero selective evidence → no writes.  C-A: the
        # split / uninformative counters below are surfaced in stats() →
        # /status, and G3 FAILS on flat elector strength or ~zero splits
        # (starvation detected, never a vacuous pass).
        self.dock_events: int = 0           # first-contact earning calls
        self.dock_abstains: int = 0         # no elector -> nothing to score
        self.dock_split: int = 0            # elector signatures disagreed
        self.dock_uninformative: int = 0    # unanimous -> no writes
        self.dock_elector_confirms: int = 0     # electors that matched truth
        self.dock_elector_disconfirms: int = 0  # electors that mismatched

    # ---- bus entry point (live wiring) ----
    def handle(self, event: BrainEvent, bus: EventBus) -> None:
        if getattr(event, 'kind', None) != EventKind.ATTENDED_PERCEPT:
            return
        # The mortal-game world publishes its tokens as modality='world'
        # percepts for chemistry tagging; they must NOT pollute the attention-
        # stream transition model (WorldActor learns the world separately, in
        # an action-conditioned way via composite tokens).  Skip them here.
        if getattr(event, 'modality', '') == 'world':
            return
        focals = getattr(event, 'focals', None) or []
        if not focals:
            return
        # The top (most salient) attended concept is the current world-
        # state; the stream of top-focals is the experience trajectory.
        state = str(focals[0])
        self.observe(self._last_state, state,
                     int(getattr(event, 'cycle', 0)), bus)
        self._last_state = state

    # ---- core loop (also driven directly by tests) ----
    def observe(self, s_prev: Optional[str], s_actual: str,
                cycle: int, bus: Any = None, valence=None) -> None:
        bus = bus if bus is not None else self.bus
        if not s_prev:
            return
        s_pred, via, _own, _cand = self._predict2(s_prev, cycle)
        self.history.append((int(cycle), s_prev, s_pred, s_actual))
        if s_pred is None:
            # First encounter — nothing to predict from.  Observe the
            # transition provisionally, UN-engaged: being seen is not
            # earning.  It must be predicted correctly next time to live.
            self.no_prediction += 1
            self._write(s_prev, s_actual, 'world_observe', cycle, bus)
            return
        self.predictions_made += 1
        _hit = 1.0 if s_pred == s_actual else 0.0
        self._record_reliability(s_prev, _hit, via)
        # THE INHERIT CHANNEL IS SCORED ON EVERY PREDICTION, not only
        # the ones it served.  When own won, the sibling candidate is
        # still checked against what actually happened -- a shadow.
        # This is what ends the starvation: the channel accumulates a
        # real track record continuously, so it can earn its way in on
        # evidence instead of waiting for a condition (own is None)
        # that is true 50 times in 6,052.
        if _cand is not None:
            _ch = 1.0 if _cand == s_actual else 0.0
            self._inh_hits += _ch
            self._inh_n += 1.0
            if via == 'own':
                self.shadow_inherit_n += 1
                # per-state inherit record, but NOT the global typical
                # rate -- a shadow was never acted on, so it must not
                # move the number that prices his predictions.
                self._record_reliability(s_prev, _ch, 'inherit', False)
        if s_pred == s_actual:
            self.confirms += 1
            if via in ('inherit', 'both'):
                self.generalized += 1
            if via == 'both':
                self.corroborated += 1
            # CONFIRM: reinforce + stamp engaged (the survival signal).
            # If the edge did not yet exist (inherited prediction), the
            # write materialises it AND engages it (write_reason is not
            # 'world_observe', so add_edge stamps it).
            self._write(s_prev, s_actual, 'world_confirm', cycle, bus,
                        valence=valence)
        else:
            self.misses += 1
            # MISS: the predicted edge gets nothing (ages to the reaper).
            # Record the correct transition provisionally, un-engaged.
            self._write(s_prev, s_actual, 'world_observe', cycle, bus)

    # ---- prediction ----
    def _strongest_transition(self, concept: Any,
                              cycle: int) -> Optional[str]:
        best, best_s = None, -1.0
        for e in concept.edges_out.get(WORLD_RELATION, ()):
            tgt = (e.target if isinstance(e.target, str)
                   else getattr(e.target, 'name', None))
            es = e.effective_strength(cycle)
            if es > best_s:
                best_s, best = es, tgt
        return best

    def _inherit_rel(self):
        """What an INHERITED prediction of his is actually worth, measured.
        None until the channel has been tested at all -- and an unproven
        source must not displace a working one."""
        if self._inh_n <= 0.0:
            return None
        return self._inh_hits / self._inh_n

    def _typical_rel(self):
        """His global hit rate so far, or None before anything is tested.
        Derived from his own history; not a tuned prior."""
        if self._rel_n <= 0.0:
            return None
        return self._rel_hits / self._rel_n

    def _rel_of(self, name, src='own'):
        """His measured hit rate for `name` FROM ONE SOURCE, or None if
        that source has never been tested there.
        _rel[name] = [n, mean, m2] and mean is the hit rate."""
        st = (self._rel if src == 'own' else self._rel_inh).get(name)
        if st is None or st[0] <= 0.0:
            return None
        return float(st[1])

    def _inherit_candidate(self, s_prev, cycle, sub):
        """The best transition his LIKE-STATES would predict, with the mean
        measured reliability of the states that voted for it.

        Two grains, both consulted:
          * siblings of s_prev itself (composites almost never carry a
            class -- measured 10 of 7,193 -- so this rarely fires), and
          * siblings of the BARE state, voting over `SIBLING|aN`: the same
            action in a place of the same kind, which is the grain at
            which his abstraction machinery actually deposits classes.

        Returns (candidate, mean_reliability) or None.  Reliability is
        averaged over the voters for the winning candidate ONLY, so a
        confident minority is not diluted by unrelated siblings.
        """
        votes = {}
        backers = {}
        # diagnostics kept from the limb this replaces, so /status
        # keeps reporting why generalization did or did not have
        # anything to say.
        self.inherit_reached += 1
        _saw_sibling = False

        def _tally(node_name, weight=1.0):
            node = sub.concepts.get(node_name)
            if node is None:
                return
            t = self._strongest_transition(node, cycle)
            if t is None:
                return
            votes[t] = votes.get(t, 0.0) + float(weight)
            backers.setdefault(t, []).append(node_name)

        if hasattr(sub, 'is_a_siblings'):
            for sib in sub.is_a_siblings(s_prev):
                if sib != s_prev:
                    _saw_sibling = True
                    _tally(sib)
            sep = s_prev.rfind('|a')
            if sep > 0:
                bare, act = s_prev[:sep], s_prev[sep:]
                for sib in sub.is_a_siblings(bare):
                    if sib != bare:
                        _saw_sibling = True
                        _tally(sib + act)
        if not votes:
            # FACET GRAIN (2026-08-13) -- similarity docking at the
            # PREDICTION layer.  Reached only here, where his is_a
            # classes had nothing to say, which is the measured norm.
            self._facet_votes(s_prev, cycle, sub, _tally)
        if not votes:
            if _saw_sibling:
                self.inherit_sibs_no_trans += 1
            else:
                self.inherit_no_siblings += 1
            return None
        self.inherit_served += 1
        cand = max(votes, key=votes.get)
        rels = [r for r in (self._rel_of(b) for b in backers.get(cand, ()))
                if r is not None]
        return cand, (sum(rels) / len(rels) if rels else None)

    # ---- similarity grain: states built from the same parts -------
    # Compute bound only.  It caps how much of a huge facet bucket is
    # walked per prediction; it asserts nothing about similarity.
    FACET_NEIGHBOUR_BUDGET = 256

    def _facet_votes(self, s_prev, cycle, sub, tally):
        """Vote with states that SHARE COMPONENTRY with s_prev,
        weighted by how many facets they share -- lock-and-key, many
        docking points.  Rarest facets first: a facet held by few
        states is the one that actually discriminates.

        Facets live on the BARE state (that is where dock_observe
        deposits them), while s_prev is a `STATE|aN` composite, so the
        vote is taken over `NEIGHBOUR|aN` -- the same action, in a
        place built of the same parts.
        """
        ri = getattr(sub, '_relation_index', None)
        if ri is None:
            return
        sep = s_prev.rfind('|a')
        bare = s_prev[:sep] if sep > 0 else s_prev
        act = s_prev[sep:] if sep > 0 else ''
        node = sub.concepts.get(bare)
        if node is None:
            return
        facets = []
        for e in node.edges_out.get(FACET_RELATION, ()):
            tgt = (e.target if isinstance(e.target, str)
                   else getattr(e.target, 'name', None))
            if tgt is not None:
                facets.append(tgt)
        if not facets:
            return
        try:
            facets.sort(key=lambda f: ri.bucket_size(FACET_RELATION, f))
        except Exception:
            # no bucket_size on this index -- order is then arbitrary,
            # and the budget below still bounds the work
            pass
        shared = {}
        walked = 0
        for F in facets:
            try:
                srcs = ri.sources_view(FACET_RELATION, F)
            except Exception:
                continue
            for sp in srcs:
                if sp == bare:
                    continue
                shared[sp] = shared.get(sp, 0) + 1
                walked += 1
                if walked >= self.FACET_NEIGHBOUR_BUDGET:
                    break
            if walked >= self.FACET_NEIGHBOUR_BUDGET:
                break
        if not shared:
            return
        self.facet_grain_used += 1
        for sp, k in shared.items():
            tally(sp + act, float(k))

    def _predict(self, s_prev: str,
                 cycle: int) -> Tuple[Optional[str], str]:
        """(prediction, via) -- the two-value form callers and tests
        already use.  `_predict2` carries the full picture."""
        p, via, _own, _cand = self._predict2(s_prev, cycle)
        return p, via

    def _predict2(self, s_prev: str, cycle: int):
        """Predict s_prev's next state from BOTH sources at once.

        Returns (prediction, via, own, cand) where via is
        'own' | 'inherit' | 'both' | '' and own/cand are the two raw
        candidates -- the caller scores the one that did not win.

        MANY THINGS CAN BE TRUE AT ONCE.  The specific (his own
        transition for exactly this state) and the general (what
        states of this kind do) are not alternatives to arbitrate
        between once and for all; they are two readings of the same
        moment.  When they agree that is corroboration and both earn.
        When they disagree the better-measured one serves THIS state,
        and the other is still scored, so the ranking keeps moving.
        """
        sub = getattr(self.engine, "substrate", None)
        if sub is None:
            return None, "", None, None
        c = sub.concepts.get(s_prev)
        if c is None:
            return None, "", None, None
        own = self._strongest_transition(c, cycle)
        cand = None
        if self.generalize:
            alt = self._inherit_candidate(s_prev, cycle, sub)
            if alt is not None:
                self.inherit_offered += 1
                cand = alt[0]
        if own is None and cand is None:
            return None, "", None, None
        if cand is None:
            self.pred_own += 1
            return own, "own", own, None
        if own is None:
            # the old own-absent path, unchanged in effect
            self.generalized_bare += 1
            self.pred_inherit += 1
            return cand, "inherit", None, cand
        if own == cand:
            self.agree += 1
            self.pred_both += 1
            return own, "both", own, cand
        self.disagree += 1
        # DISAGREEMENT: measured rate decides, per state first, then
        # per channel, then his typical rate.  No threshold and no
        # constant -- and the incumbent keeps ties, so a source only
        # displaces the other on evidence.
        # BOTH must be measured AT THIS STATE.  Falling back to a
        # channel average priced the incumbent with a number the
        # challenger deflates -- measured: confirm_rate 0.298 -> 0.170,
        # inheritance winning 195 of 203 disagreements by dragging the
        # global rate under its own.  An average can no longer displace
        # anything; only this state's own history can.
        own_r = self._rel_of(s_prev, "own")
        inh_r = self._rel_of(s_prev, "inherit")
        if (inh_r is not None and own_r is not None and inh_r > own_r):
            self.pred_inherit += 1
            return cand, "inherit", own, cand
        self.pred_own += 1
        return own, "own", own, cand

    # ---- facet dock: split-event contrastive earning (2026-07-21) ----
    def dock_observe(self, s: str, a: int, s_actual: str, cycle: int,
                     bus: Any = None, facet_nodes: Any = (),
                     trans: Any = None) -> None:
        """SPLIT-EVENT CONTRASTIVE earning for has_facet edges (C2 v2).

        Called by the WorldActor at a genuine FIRST-CONTACT (s, a), with
        `facet_nodes` = the facet-node names of s's LIVE percept vector
        (computed by the transducer — works at true first contact, before
        any facet edge for s exists) and `trans` = the actor's own learned
        transition map.

        ELECTORS: every state sp ≠ s that (a) carries at least one of s's
        facets in the substrate (`sp -has_facet-> F`, O(1) via the relation
        index) and (b) holds its OWN transition for the same action
        (`(sp, a) ∈ trans`).  Each elector's remembered 2-class signature
            sig(sp, a) = (trans[(sp, a)] == sp)        (self-loop vs move)
        is compared with the observed truth
            sig_now    = (s_actual == s).

        If ALL elector signatures agree (unanimity), the event carries ZERO
        selective evidence — journal `uninformative`, NO writes (this kills
        vacuous earning structurally: near-ubiquitous facets whose carriers
        all agree can never ratchet).  If the electors SPLIT, strength moves
        symmetrically at promille through the single writer: each MATCHING
        elector's shared has_facet edges are re-attested (+0.005, engaged,
        write_reason='facet_confirm'); each MISMATCHING elector's shared
        edges are weakened (−0.005, floored at prune,
        write_reason='facet_disconfirm').  A correct MINORITY key never
        bleeds — the comparison is against the OBSERVED outcome, not the
        consensus.

        PURE apart from the bus writes; draws NO RNG (C3).  Deterministic
        iteration order (sorted) so runs are reproducible.
        """
        bus = bus if bus is not None else self.bus
        self.dock_events += 1
        sub = getattr(self.engine, 'substrate', None)
        ri = getattr(sub, '_relation_index', None) if sub is not None else None
        if sub is None or ri is None or trans is None or not facet_nodes:
            self.dock_abstains += 1
            return
        electors: Dict[str, list] = {}
        for F in facet_nodes:
            for sp in ri.sources_view(FACET_RELATION, F):
                if sp == s:
                    continue
                if (sp, a) not in trans:
                    continue
                electors.setdefault(sp, []).append(F)
        if not electors:
            self.dock_abstains += 1
            return
        sig_now = (s_actual == s)
        sigs = {sp: (trans[(sp, a)] == sp) for sp in electors}
        if len(set(sigs.values())) == 1:
            # Unanimous — zero selective evidence.  No writes, by law.
            self.dock_uninformative += 1
            return
        self.dock_split += 1
        for sp in sorted(electors):
            match = (sigs[sp] == sig_now)
            if match:
                self.dock_elector_confirms += 1
            else:
                self.dock_elector_disconfirms += 1
            reason = 'facet_confirm' if match else 'facet_disconfirm'
            for F in sorted(set(electors[sp])):
                self._write_facet(sp, F, reason, cycle, bus)

    def _write_facet(self, subj: str, facet_node: str, reason: str,
                     cycle: int, bus: Any) -> None:
        if bus is None:
            return
        bus.publish(SubstrateWriteQueuedEvent(
            kind=EventKind.SUBSTRATE_WRITE_QUEUED,
            cycle=int(cycle),
            source_capability='grounding',
            origin='internal',
            subject=subj, relation=FACET_RELATION, object=facet_node,
            strength=PROVISIONAL_EDGE_STRENGTH, write_reason=reason))

    # ---- write (single-writer contract: go through the bus) ----
    def _write(self, subj: str, obj: str, reason: str,
               cycle: int, bus: Any, valence=None) -> None:
        if bus is None:
            return
        _mm, _mi, _oc = 0.0, 0.0, ''
        if valence is not None:
            _mm, _mi, _oc = float(valence[0]), float(valence[1]), 'hit'
        bus.publish(SubstrateWriteQueuedEvent(
            kind=EventKind.SUBSTRATE_WRITE_QUEUED,
            cycle=int(cycle),
            source_capability='grounding',
            origin='internal',
            subject=subj, relation=WORLD_RELATION, object=obj,
            strength=PROVISIONAL_EDGE_STRENGTH, write_reason=reason,
            mi_m=_mm, mi_i=_mi, outcome=_oc))

    # ---- precision instrument (Capability 1, read-only) ----
    def _record_reliability(self, s_prev: str, hit: float,
                            via: str = 'own',
                            count_global: bool = True) -> None:
        """Credit the SOURCE that predicted, not just the state.

        `via='both'` means own and inherited agreed, so the outcome
        belongs to both records.  `count_global=False` is for shadow
        scoring: it updates the per-state record without moving the
        typical rate, which prices predictions he actually made.
        """
        if count_global:
            self._rel_hits += float(hit)
            self._rel_n += 1.0
        if via == "own":
            targets = (self._rel,)
        elif via == "inherit":
            targets = (self._rel_inh,)
        else:
            targets = (self._rel, self._rel_inh)
        for _d in targets:
            st = _d.get(s_prev)
            if st is None:
                st = [0.0, 0.0, 0.0]   # n, mean, m2
                _d[s_prev] = st
            st[0] += 1.0
            delta = hit - st[1]
            st[1] += delta / st[0]
            st[2] += delta * (hit - st[1])

    def precision_report(self) -> Dict[str, Any]:
        """Per-state reliability summary the proven-grounding gate reads.
        reliability_mean = mean over states of their hit-rate;
        reliability_spread = variance ACROSS states of that hit-rate (a
        grounded world has a spread; a degenerate one-attractor world
        does not).  All self-derived, no typed threshold."""
        rels = [st[1] for st in self._rel.values() if st[0] > 0]
        if not rels:
            return {'reliability_mean': 0.0, 'reliability_spread': 0.0,
                    'precision_states': 0}
        rmean = sum(rels) / len(rels)
        rspread = sum((r - rmean) ** 2 for r in rels) / len(rels)
        return {'reliability_mean': rmean,
                'reliability_spread': rspread,
                'precision_states': len(rels)}

    def stats(self) -> Dict[str, Any]:
        n = self.predictions_made
        out = {
            'predictions_made': n,
            'confirms': self.confirms,
            'misses': self.misses,
            'no_prediction': self.no_prediction,
            'generalized': self.generalized,
            'generalized_bare': self.generalized_bare,
            'inherit_reached': self.inherit_reached,
            'facet_grain_used': self.facet_grain_used,
            'pred_own': self.pred_own,
            'pred_inherit': self.pred_inherit,
            'pred_both': self.pred_both,
            'agree': self.agree,
            'disagree': self.disagree,
            'corroborated': self.corroborated,
            'shadow_inherit_n': self.shadow_inherit_n,
            'inherit_no_siblings': self.inherit_no_siblings,
            'inherit_sibs_no_trans': self.inherit_sibs_no_trans,
            'inherit_served': getattr(self, 'inherit_served', 0),
            'inherit_offered': self.inherit_offered,
            'inherit_rel': (self._inh_hits / self._inh_n)
            if self._inh_n > 0 else None,
            'confirm_rate': (self.confirms / n) if n else 0.0,
            # C-A: split-event earning counters, surfaced in /status via
            # runtime's grounding stats block.  G3 reads these: ~zero
            # dock_split on a live soak = STARVATION (gate FAIL), never a
            # vacuous pass.
            'dock_events': self.dock_events,
            'dock_abstains': self.dock_abstains,
            'dock_split': self.dock_split,
            'dock_uninformative': self.dock_uninformative,
            'dock_elector_confirms': self.dock_elector_confirms,
            'dock_elector_disconfirms': self.dock_elector_disconfirms,
        }
        out.update(self.precision_report())
        return out
