"""ReplayConsolidator — the RETURN PATH (Step 0 substrate retrofit).

The architecture had the forward flow (input -> concept -> edge -> strength)
and the FADE (decay / quarantine), but never the RETURN: nothing re-engaged
the established core, so strength rotted to the floor and chemistry froze.
Every existing loop is centrifugal (reverie by recency, goals by uncertainty,
Phase-S by the dirty set).  This is the missing centripetal loop the
doctrine's mortality clause assumes -- "strength continually re-earned by
MATTERING (engaged -> strong)."

Each sleep consolidation pass it:
  1. SELECTS the mattering core -- top-k concepts by TENSION = min(M, I),
     the chemistry coincidence the architecture declares to be mattering:
     a genuinely-mattering concept departed BOTH poles together (mortality
     cortisol/NE AND immortality dopamine/oxytocin).  magnitude (= max)
     tracked raw FREQUENCY -- heavy reading inflates the I-side ambient
     mood, so a high-touch function word ("than") scored as high as the
     felt core; tension is ~81% less frequency-contaminated, surfacing the
     genuinely-felt core (justice/worship/reverence) instead.  Staleness is
     NOT a score term: at the live ~1M-cycle staleness plateau any recency
     factor saturates to a dead constant (it discriminated nothing, or
     discriminated toward ancient parser junk) -- staleness survives only as
     the restart GUARD below (staleness>0), because the internal cycle resets
     to 0 each restart while last_active_cycle holds absolute history.  The
     currently-frozen M/I is a usable decay-INDEPENDENT historical-mattering
     seed (breaks the chicken-and-egg).
  2. RE-WALKS each selected concept's edges through earn-or-dissolve via the
     single-edge `recheck_edge_coherence`: a still-corroborated edge earns
     COHERENCE_REINFORCE_BUMP (held off the floor); a faded one earns
     nothing and keeps fading (mortality still bites).
  3. RE-ANCHORS the concept's bubble (last_active_cycle = cycle) IFF at least
     one of its edges re-corroborated this pass -- the immortality stamp is
     EARNED, not granted on selection (earn-or-dissolve applied to the
     re-anchor itself; a concept whose edges all fade gets no recency credit
     and keeps fading, closing the self-licking-stamp / stale-clique holes).
     Re-imprint is by re-anchoring, NOT by firing a synthetic chemistry event
     (no mood pollution, no encounter_count inflation, no interoception write).

Decay (Bubble.effective_trace) is the fade-complement: what this organ does
NOT reach relaxes toward baseline -> M/I -> 0 (quiescent).  Together:
mattering drives re-engagement; re-engagement re-earns strength + re-evokes
mattering; decay fades the rest.  One tension.

LIFEFORCE-DECOUPLED BY CONSTRUCTION (audited, provable): re-walk uses the
single-edge recheck (never increments newly_coherent, never touches
first_coherent_cycle, never calls reinforce_coherent_edges); re-anchor fires
no chemistry event.  Nothing here can reach record_learning.

SHADOW-STAGED (2026-06-10 audit build-condition): the M/I-routing through
effective_trace is NOT yet authoritative -- Concept.mi still reads the raw
trace, so selection uses the frozen historical-mattering seed and there is
NO inversion shock.  This organ re-walks strength + re-anchors LIVE (both
safe) and COMPUTES + LOGS the would-be decayed M/I (shadow), so we watch
what the fade would do before flipping it authoritative.
"""

from __future__ import annotations

from typing import Any, Dict, List

from seagi.core.substrate import COHERENCE_REINFORCE_BUMP


class ReplayConsolidator:

    def __init__(self, engine: Any = None):
        self.engine = engine
        self.passes = 0
        self.concepts_selected = 0
        self.edges_rewalked = 0
        self.edges_faded = 0
        self.reanchored = 0
        # shadow accounting (the decay-complement, not yet authoritative)
        self.shadow_mi_drop_total = 0.0
        self.last_selection_top: List[str] = []
        self.last_rewalked_mean_strength = 0.0

    @staticmethod
    def _tension(mi: Any) -> float:
        """min(M, I) -- both poles departed from baseline together.  This is
        the mattering signal (vs magnitude = max, which tracks frequency)."""
        try:
            return float(mi.tension)
        except Exception:
            try:
                return min(float(mi.m), float(mi.i))
            except Exception:
                return 0.0

    def run_pass(self, cycle: int, k: int) -> int:
        """One sleep replay pass.  Returns the count of edges re-walked
        (folds into MetabolicDebt).  k = the inflow-derived budget (q_cap).
        """
        sub = getattr(self.engine, 'substrate', None)
        if sub is None or int(k) < 1:
            return 0
        self.passes += 1
        # --- selection: top-k by TENSION = min(M, I) ---
        # tension surfaces concepts that departed BOTH the mortality and the
        # immortality pole together (the architecture's declared mattering
        # signal), unlike magnitude (= max) which tracked raw frequency.
        # Staleness is the restart GUARD only, NOT a score term: at the live
        # ~1M staleness plateau any recency factor saturates to a constant.
        scored = []
        for name, c in sub.concepts.items():
            bs = getattr(c, 'bubbles', None)
            if not bs:
                continue
            active = max(bs, key=lambda b: b.last_active_cycle)
            staleness = int(cycle) - int(active.last_active_cycle)
            if staleness <= 0:
                continue
            tension = self._tension(c.mi)   # raw/frozen mi (shadow-staged seed)
            if tension <= 0.0:
                continue
            scored.append((tension, name, c, active))
        if not scored:
            return 0
        scored.sort(key=lambda r: r[0], reverse=True)
        rewalked = 0
        names: List[str] = []
        rewalked_strengths: List[float] = []
        for tension, name, c, active in scored[:int(k)]:
            self.concepts_selected += 1
            names.append(name)
            # --- strength re-walk through earn-or-dissolve ---
            rewalked_this_concept = 0
            for bucket in list(getattr(c, 'edges_out', {}).values()):
                for e in list(bucket):
                    ok, corr = sub.recheck_edge_coherence(e, cycle)
                    if ok:
                        e.reinforce(cycle, delta=COHERENCE_REINFORCE_BUMP,
                                    origin='maintenance')
                        e.last_engaged_cycle = int(cycle)
                        for ce in corr:
                            ce.last_engaged_cycle = int(cycle)
                        rewalked += 1
                        rewalked_this_concept += 1
                        self.edges_rewalked += 1
                        rewalked_strengths.append(float(e.strength))
                    else:
                        self.edges_faded += 1
            # --- shadow: what WOULD the decayed tension be? (log only) ---
            try:
                from seagi.core.layer6 import derive_mi_from_trace
                shadow_tension = self._tension(
                    derive_mi_from_trace(active.effective_trace(cycle)))
                self.shadow_mi_drop_total += max(0.0, tension - shadow_tension)
            except Exception:
                pass
            # --- re-imprint = RE-ANCHOR ONLY, EARN-GATED (no event, no
            # encounter++): the immortality stamp is earned by a real re-walk,
            # not granted on selection.  A concept whose edges all faded gets
            # no recency credit and keeps fading. ---
            if rewalked_this_concept > 0:
                active.last_active_cycle = int(cycle)
                self.reanchored += 1
        self.last_selection_top = names[:20]
        if rewalked_strengths:
            self.last_rewalked_mean_strength = round(
                sum(rewalked_strengths) / len(rewalked_strengths), 4)
        return rewalked

    def stats(self) -> Dict[str, Any]:
        return {
            'shadow_staged': True,
            'passes': self.passes,
            'concepts_selected': self.concepts_selected,
            'edges_rewalked': self.edges_rewalked,
            'edges_faded': self.edges_faded,
            'reanchored': self.reanchored,
            'shadow_mi_drop_total': round(self.shadow_mi_drop_total, 3),
            'last_selection_top': list(self.last_selection_top),
            'last_rewalked_mean_strength': self.last_rewalked_mean_strength,
        }
