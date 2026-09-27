"""EngagementLedger — open intelligence-credit + obstructive-burden
accumulator (MortalityDrive death-wall organs b/c helper).

Mirrors MetabolicDebt's self-calibrating-deque pattern, but accumulates two
per-episode quantities the death wall reads at wake_onset:

  * ENGAGEMENT (organ c — the OPEN intelligence credit): distinct COHERENT
    edges (first_coherent_cycle > 0) whose last_engaged_cycle advanced this
    episode.  Reasoning / derive_closure / replay re-engagement of the felt
    core / recall ALL stamp last_engaged on coherent edges, so every faculty
    funnels through this ONE count WITHOUT being enumerated (Rule 6: no
    closed list of intelligence-types).  This is the pushback that holds the
    death wall back — "anything an intelligent mind does holds it back."

  * OBSTRUCTION (organ b — aging-as-clutter): per-(source, relation)-slot
    events where an INCOHERENT edge (first_coherent_cycle == 0) out-ranks the
    coherent material in its slot — incoherent junk that STAYS and GETS IN
    THE WAY (proteinopathy / glymphatic-clearance-failure aging, NOT healthy
    fade or discard).  Per-slot-per-episode deduped (one plaque, not one
    collision).

Both are read by MortalityDrive._close_episode, which then calls
roll_episode() to append the obstruction count to the self-calibrating
deque and reset the per-episode transients.  The ledger does NOT subscribe
to wake_onset itself — the drive owns the close ordering so it reads the
values BEFORE they reset (no event-handler-ordering race).

Doctrine: open credit (no enumerated faculty list) serves Rule 6; clutter-
aging serves Rule 2 (felt cost) and Rule 3 (earn-or-dissolve is the sole
gate — junk relieves aging ONLY by fading or cohering).  Zero new constants:
deque length and the mean-scale reuse MetabolicDebt's lineage
(CLEARANCE_DEQUE_MAXLEN).

Doctrine cross-refs
-------------------
* [[feedback_seagi_death_wall_no_ceiling_model]] — the model this serves
* [[feedback_homeostatic_cost_doctrine]] — the felt-cost requirement
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Set, Tuple

from .metabolic_debt import CLEARANCE_DEQUE_MAXLEN


class EngagementLedger:
    """Per-episode intelligence-engagement (credit) and obstructive-
    incoherent-burden (aging) accumulator for the MortalityDrive death
    wall.  Passive: fed by direct calls from Cortical's ranking sites;
    rolled once per episode by MortalityDrive."""

    # No event subscriptions: MortalityDrive drives the episode roll so it
    # reads the per-episode values before they reset (avoids a wake_onset
    # handler-ordering race).

    def __init__(self) -> None:
        # Self-calibrating obstruction history (persisted).  Its mean is the
        # A_scale the salience clutter-term normalizes against.  Starts
        # empty -> A_scale 0.0 until the first episode rolls; consumers guard
        # with max(A_scale, 1.0).
        self._obstruction_deque: Deque[float] = deque(
            maxlen=CLEARANCE_DEQUE_MAXLEN)

        # Per-episode transients (re-derive across restarts; not persisted).
        self._episode_obstruction: int = 0
        self._slots_seen_this_episode: Set[Tuple[Any, Any]] = set()
        self._engaged_this_episode: Set[Any] = set()

        # Diagnostics (not persisted).
        self.episodes_rolled: int = 0
        self.total_obstruction_observed: int = 0
        self.total_engaged_observed: int = 0

    # ---- fed by Cortical's ranking sites (direct calls) ----

    def note_obstruction(self, source: Any, relation: Any) -> None:
        """Register that an incoherent edge out-ranked coherent material in
        the (source, relation) slot.  Per-slot-per-episode deduped: a junk
        fragment that hijacks the same slot a thousand times in one episode
        is ONE unit of burden (one plaque, not one collision)."""
        key = (source, relation)
        if key in self._slots_seen_this_episode:
            return
        self._slots_seen_this_episode.add(key)
        self._episode_obstruction += 1
        self.total_obstruction_observed += 1

    def note_engaged(self, edge: Any) -> None:
        """Register that a COHERENT edge (first_coherent_cycle > 0) was
        engaged this episode.  Distinct-edge deduped (a tight re-walk loop
        over one edge = ONE unit — engagement count saturates, never
        multiplies).  Incoherent edges are ignored: they cannot buy life;
        if active and in-the-way they are aging, counted elsewhere."""
        fcc = getattr(edge, 'first_coherent_cycle', 0) or 0
        if fcc <= 0:
            return
        key = self._edge_key(edge)
        if key in self._engaged_this_episode:
            return
        self._engaged_this_episode.add(key)
        self.total_engaged_observed += 1

    @staticmethod
    def _edge_key(edge: Any) -> Any:
        # Prefer a stable identity; fall back to the (s, r, t) triple.
        k = getattr(edge, 'key', None)
        if k is not None:
            return k
        return (getattr(edge, 'source', None),
                getattr(edge, 'relation', None),
                getattr(edge, 'target', None))

    # ---- read by MortalityDrive._close_episode ----

    @property
    def episode_obstruction(self) -> int:
        """A_episode — obstructive-incoherent-burden accrued this episode
        (deduped per slot).  The death-wall accelerant."""
        return self._episode_obstruction

    def episode_credit(self) -> int:
        """The organ-c open intelligence credit accrued this episode =
        distinct coherent edges engaged.  Pushes the death wall back."""
        return len(self._engaged_this_episode)

    @property
    def A_scale(self) -> float:
        """Mean of recent episodes' obstruction burden (self-calibrating,
        debt_full_scale lineage).  Used by the salience clutter-term
        normalizer; consumers guard with max(A_scale, 1.0).  0.0 until the
        first episode rolls."""
        if not self._obstruction_deque:
            return 0.0
        return (sum(self._obstruction_deque) /
                float(len(self._obstruction_deque)))

    def roll_episode(self) -> None:
        """Close the episode: append this episode's obstruction burden to
        the self-calibrating deque and reset the per-episode transients.
        Called by MortalityDrive._close_episode AFTER it has read
        episode_obstruction / episode_credit (the drive owns ordering)."""
        self._obstruction_deque.append(float(self._episode_obstruction))
        self._episode_obstruction = 0
        self._slots_seen_this_episode.clear()
        self._engaged_this_episode.clear()
        self.episodes_rolled += 1

    # ---- per-tick symmetry (no-op) ----

    def tick(self) -> None:
        """No per-tick activity; method present for Brain.tick() symmetry."""
        return

    # ---- persistence ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            'obstruction_deque': list(self._obstruction_deque),
        }

    def load_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        vals = state.get('obstruction_deque')
        if isinstance(vals, (list, tuple)):
            try:
                self._obstruction_deque = deque(
                    (float(v) for v in vals),
                    maxlen=CLEARANCE_DEQUE_MAXLEN)
            except (TypeError, ValueError):
                pass
        # Transients re-derive across restarts.
        self._episode_obstruction = 0
        self._slots_seen_this_episode = set()
        self._engaged_this_episode = set()

    # ---- diagnostics ----

    def stats(self) -> Dict[str, Any]:
        return {
            'episode_obstruction': int(self._episode_obstruction),
            'episode_credit': int(self.episode_credit()),
            'A_scale': float(self.A_scale),
            'obstruction_deque': list(self._obstruction_deque),
            'episodes_rolled': int(self.episodes_rolled),
            'total_obstruction_observed':
                int(self.total_obstruction_observed),
            'total_engaged_observed': int(self.total_engaged_observed),
        }
