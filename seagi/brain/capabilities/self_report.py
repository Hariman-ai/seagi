"""SELF-REPORT -- he tells us which of his organs is not paying.

User, 2026-08-06: "I am also looking for a way for Seagi to self improve
instead of getting tuned and guided all the way. This is the idea behind the
whole AGI concept. He needs to flag to you what he needs and wants as much
as we mold him."

Every architectural discovery on 2026-08-06 was made from OUTSIDE him: a
human wrote a probe, read a counter, and named the fault.  He held every one
of those signals internally the whole time and could not say any of them.
That asymmetry is the thing to remove.

THE SIGNATURE.  All six findings that day had one shape:

    replay.edges_rewalked              0  from 468,274 faded
    reasoning.consolidations_emitted   0  from 310 recalls
    mortality_clock.deaths_observed    0  while mortality_drive.deaths = 14
    curriculum battery                 0  facts from 270,416 quarantined
    raphe.social_replenish_fires       0
    roles abstracted                   0  from 222 written

A counter pinned at zero while the organ around it did a great deal of work.
That is a dead output on a live input -- an organ spending his life and
returning nothing -- and it is detectable mechanically, with no
understanding of what any particular counter MEANS.

NO THRESHOLDS, NO WINDOWS, NO CONSTANTS.  A key is not judged against a
number chosen by a human.  It is judged against ITS OWN SIBLINGS: if the
other counters of the same organ moved while this one never left zero, the
organ demonstrably ran and this output demonstrably did not.  The evidence
is the siblings' own movement, which also supplies the ranking -- the more
work went on around a dead output, the louder the complaint.  Same idiom as
`depleted()` rule (a), which already judges a patch against the environment
rather than a constant.

M/I GROUNDING.  A complaint is legitimate when an organ consumes and does
not return: that is a mortality lean with no immortality credit.  This is
the companion to the homeostatic-cost doctrine (an organ that does not
engage must be FELT); here, an organ that does not PAY must be SAID.

It flags, it never repairs.  A wrong flag costs a human glance, and a flag
that turns out to be correct-by-design gets silenced explicitly.
"""
from typing import Any, Dict, List, Tuple

# Keys that are legitimately zero and are not complaints: identifiers,
# clocks, and flags.  Substring match, deliberately small -- an over-eager
# silence list would hide exactly the faults this exists to surface.
_NOT_A_COMPLAINT = (
    'error', 'last_', '_id', 'mode', 'enabled', 'seed', 'cycle',
    'timestamp', 'version',
    # A REJECTION THAT NEVER FIRED IS NOT A DEAD OUTPUT.  These count
    # things he declined to do; zero means nothing needed declining,
    # which is success.  (His own first report flagged four
    # inner_voice.skipped_* counters -- all false.)
    'skipped', 'suppressed', 'rejected', 'dropped', 'aborted',
    # A DEFECT THAT NEVER OCCURRED IS HEALTH, not a dead output.  These
    # count things that should never happen; zero is the system working.
    # (`source_monitor.untagged_count` = bus events missing origin or
    # source_capability -- zero means everything is properly tagged.)
    'untagged', 'malformed', 'invalid', 'orphan', 'corrupt',
)


def _flatten(o: Any, pre: str = '') -> Dict[str, float]:
    """Numeric leaves of the stats tree, dotted-path keyed."""
    out: Dict[str, float] = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flatten(v, '%s.%s' % (pre, k) if pre else str(k)))
    elif isinstance(o, bool):
        pass                      # a flag is a state, not an amount
    elif isinstance(o, (int, float)):
        out[pre] = float(o)
    return out


class SelfReport:
    """Watches his own stats and names the outputs that never move."""

    def __init__(self) -> None:
        self._first: Dict[str, float] = {}     # value when first seen
        self._now: Dict[str, float] = {}
        self._seen_n: int = 0
        self._last_top: List[str] = []
        self.observations: int = 0

    # ------------------------------------------------------------------
    def observe(self, tree: Dict[str, Any]) -> None:
        flat = _flatten(tree)
        if not flat:
            return
        self.observations += 1
        for k, v in flat.items():
            if k not in self._first:
                self._first[k] = v
            self._now[k] = v

    # ------------------------------------------------------------------
    def _organ(self, key: str) -> str:
        return key.split('.', 1)[0]

    def needs(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Dead outputs, loudest first.

        Loud = how much work the rest of the organ did while this output
        stayed at zero.  Requires at least two observations, so nothing is
        reported from a single sample.
        """
        if self.observations < 2:
            return []
        moved: Dict[str, float] = {}
        n_moved: Dict[str, int] = {}
        n_total: Dict[str, int] = {}
        # the biggest mover in each organ -- the best available proxy
        # for 'the work that flowed in', with no code model needed
        top_in: Dict[str, tuple] = {}
        for k, now in self._now.items():
            org = self._organ(k)
            n_total[org] = n_total.get(org, 0) + 1
            d = now - self._first.get(k, now)
            if d > 0:
                moved[org] = moved.get(org, 0.0) + d
                n_moved[org] = n_moved.get(org, 0) + 1
                if d > top_in.get(org, ('', 0.0))[1]:
                    top_in[org] = (k, d)

        out: List[Tuple[float, Dict[str, Any]]] = []
        for k, now in self._now.items():
            if now != 0.0 or self._first.get(k, 0.0) != 0.0:
                continue
            kl = k.lower()
            if any(t in kl for t in _NOT_A_COMPLAINT):
                continue
            org = self._organ(k)
            work = moved.get(org, 0.0)
            if work <= 0.0:
                continue          # the organ never ran; that is not ITS fault
            # SCALE-FREE: what share of this organ's own counters moved
            # while this one did not.  A single huge cycle-counter can no
            # longer carry its whole organ to the top of the list.
            share = float(n_moved.get(org, 0)) / float(
                max(1, n_total.get(org, 1)))
            _in_k, _in_d = top_in.get(org, ('', 0.0))
            out.append((share, {
                'key': k,
                'organ': org,
                'siblings_moved': '%d/%d' % (n_moved.get(org, 0),
                                             n_total.get(org, 1)),
                'share': round(share, 3),
                'sibling_work': round(work, 2),
                'fed_by': _in_k,
                'opportunities': round(_in_d, 2),
                # the DIAGNOSIS: a dead output next to the live input
                # that fed it.  This is the liveness assertion a human
                # had to construct by hand for every fault found so far.
                'says': (('%s never fired across %.0f of %s'
                          % (k, _in_d, _in_k)) if _in_k else
                         ('%s never moved off zero while %d of %d '
                          'counters in %s did'
                          % (k, n_moved.get(org, 0),
                             n_total.get(org, 1), org))),
            }))
        out.sort(key=lambda r: -r[0])
        self._last_top = [r[1]['key'] for r in out[:limit]]
        return [r[1] for r in out[:limit]]

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        n = self.needs()
        return {
            'observations': self.observations,
            'keys_watched': len(self._now),
            'dead_outputs': len(n),
            'needs': n,
        }
