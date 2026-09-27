"""THE PLAN FOR A PICTURE -- a learned operator model and a planner (2026-09-06).

Measured before this was written (memory p_apply_is_a_learnable_operator):
on cd82 the effect of APPLY on the canvas is a function of the indicator
drawn on the board -- the stamp icon's shape and its fill colour -- on
3,946 of 3,946 applies (the null, "nothing changes", is 70%); the arrow
actions form a closed, deterministic automaton over the icon shapes
(98.8% modal); a swatch click sets the colour (814 of 814); and a greedy
planner over (shape, colour) clears level 1 in 6 steps.  Nothing in him
composed those steps: in 20 level-1 lives today he did the first apply
right every time and never once clicked the colour the second layer
needs.  Five earlier 1->2 clears were a step-locked coincidence of a
stored route, gone since he stopped repeating no-ops.

One RelPlan per (game, level), attached to the EQ relation the relation
sense pursues (canvas A, target B).  NO PER-GAME KNOWLEDGE.  Learned from
his own frames:

  INDICATOR  the objects outside A and B (lines excluded).  A state is
             the set of their normalised shapes (colour-blind), STORED IN
             FULL and READ as up to TWO shapes: among the shapes present
             that CHANGE MOST OFTEN under his actions (appear, disappear
             or recolour at least a fifth as often as the most changing
             one), the two most often seen -- an icon's outline and its
             fill (an outline shared by two masks changes twice as often
             as either fill).  A static palette never changes; a selector bar that
             the floating icon occasionally covers changes a handful of
             times against the icon's hundreds.  Four half-masks share
             one square outline and differ in the fill, so the fill is
             read too; a diagonal icon's fill can merge with same-
             coloured canvas cells and vanish as a shape, so states are
             matched when COMPARABLE (one contains the other): the
             outline alone still finds what was learned with its fill.
             Change and frequency counts are learned, so old states are
             re-read.  The
             SELECTED COLOUR is read off the object that is unique on the
             board, CHANGES (it appears or disappears under an action, or
             changes colour under a click) and whose colour matched what
             got painted most often (a swatch never changes, so it can
             never pose as the selection).
  EFFECT     (action, click target, indicator state) -> the canvas cells
             that changed (their union), and votes for which indicator
             object carries the painted colour.
  AUTOMATON  (indicator state, non-click action) -> next state (modal).
  SWATCH     a click on an object of colour c that introduces c into the
             indicator -> c can be selected by clicking that object.

The planner: from the current canvas, target, indicator state and
selected colour, greedily pick the (effect, colour) with the best
gain per step, cost = automaton distance + colour click + the action
itself, until the picture matches or nothing helps; hand back the FIRST
action of that plan, with the aim if it is a click.  Every action is a
prediction; a wrong one is learned from like any other step.

Constants are shape limits, stated once here.
"""
import hashlib
import numpy as np
from collections import deque

W = 64
MAX_ROUNDS = 12            # paints a plan may chain
MAX_STATES = 400           # indicator states kept in the automaton
MAX_PLACES = 600           # places kept per level (cd82 off-route: 49-56 seen)
MAX_EFFECTS = 400
STUCK_AFTER = 3            # the same plan action EXECUTED on the same board, no progress
MAX_WALK_STEPS = 12        # executed steps a latched walk may take toward one mask
                           # (twice the longest path measured, l3_probe8: 1-6)
LA_DEPTH = 3               # paints a lookahead sequence may chain (cd82 L3 needs 3)
LA_BEAM = 32               # distinct depth-2 canvases the third step is searched from
LA_DEPTH_FULL = 4          # paints the FULL search may chain (LAFULL; cd82 L4 needs 4)
LA_MAX_CANVASES = 60000    # canvases the FULL search may hold (cd82 L4 from mm 8: 27,106)
LA_TIME_FULL = 1.5         # seconds a FULL search may run (review of 33: an unsolvable
                           # board ran 12 s at the 120k cap, on EVERY arrow step)
LA_MAX_PAIRS = 160         # (effect, colour) pairs the search may hold (cd82 L3: 15 x 7)
BIG_OBJECT = 6             # a click target at least this tall AND wide keeps
                           # its within-object offset (thirds)
import time as _time


def _EXACTFIRST_ON():
    """EXACT-FIRST (patch 34, 2026-09-09): a state with its OWN entry for
    (a, ck) whose cells differ from a step's entry is not a place to execute
    that step, in either direction of the subset match.  cd82 L4 23:05Z: the
    tile click learned at the T slot (top block) was executed at the Bo slot
    (a one-part SUBSET state with its own 3-sample bottom-block entry), 4/4
    latched routes refuted at that step.  /root/EXACTFIRST_ON."""
    try:
        import os as _os
        return _os.path.exists('/root/EXACTFIRST_ON')
    except Exception:
        return False


def _LAFULL_ON():
    """FULL lookahead search (patch 33, 2026-09-09): distinct masks x target
    colours, canvases deduplicated across the whole search, LA_DEPTH_FULL
    paints.  The beam kept the lowest-mismatch partials and pruned every
    route that must spoil first; cd82 L4 needs four paints whose first two
    make the picture worse (beam 8192 at depth 4: still 8->5; full: 8->0
    in 27k canvases, 0.45 s).  /root/LAFULL_ON."""
    try:
        import os as _os
        return _os.path.exists('/root/LAFULL_ON')
    except Exception:
        return False


def _CURIOUS_ON():
    """When nothing known gains, try a mask never applied.  /root/CURIOUS_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/CURIOUS_ON")
    except Exception:
        return False


def _LOOKAHEAD_ON():
    """At the floor, a bounded search over known masks may take a spoiling
    first step when the sequence ends better.  /root/LOOKAHEAD_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/LOOKAHEAD_ON")
    except Exception:
        return False


def _SEQHUE_ON():
    """A latched lookahead step navigates first and clicks its colour AT the
    mask; the state its own click produced is where the mask now is; arrows
    pool twin evidence; with no path, one colour click is tried before the
    latch drops.  /root/SEQHUE_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/SEQHUE_ON")
    except Exception:
        return False


def _REFUTE_ON():
    """A latched step whose paint mispredicted, or whose pick the stuck
    guard refused, is not planned on again this life.  /root/REFUTE_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/REFUTE_ON")
    except Exception:
        return False


def _LAFLOOR_ON():
    """The lookahead runs at any greedy stall on a level without a won
    route, not only at the level's best.  /root/LAFLOOR_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/LAFLOOR_ON")
    except Exception:
        return False


def _SAMEMASK_ON():
    """Two indicator states a colour click swaps between are the same
    mask.  /root/SAMEMASK_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/SAMEMASK_ON")
    except Exception:
        return False


def _PROBEHUE_ON():
    """A probe paints in a colour the canvas does not hold; a mask that
    still paints nothing is remembered as null.  /root/PROBEHUE_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/PROBEHUE_ON")
    except Exception:
        return False


def _PICFIX_ON():
    """A plan table acts only on the picture shape it learned on.
    /root/PICFIX_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/PICFIX_ON")
    except Exception:
        return False


def _NAVPAINT_ON():
    """Navigation uses only actions that have never painted the canvas.
    /root/NAVPAINT_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/NAVPAINT_ON")
    except Exception:
        return False


def _QTILE_ON():
    """A shape he clicks on is part of the state however rarely it
    changes.  /root/QTILE_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/QTILE_ON")
    except Exception:
        return False


def _SPOILMEM_ON():
    """A spoil counts against the effect across lives until the effect
    learns something new.  /root/SPOILMEM_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/SPOILMEM_ON")
    except Exception:
        return False


def _PLACEKEY_ON():
    """PLACEKEY (patch 41, 2026-09-12): a latched shortcut navigates on a key for WHERE he
    stands that canvas paint cannot change -- the board with canvas, target and budget row
    blanked, hashed.  Measured off-route on cd82 (3 walks, 3.1k arrow steps each): arrows have
    ONE successor 94-96% of the time here against 70-77% under his quotient state, which moves
    whenever paint lands under the stamp.  The quotient still says what a paint DOES (96-98%).
    /root/PLACEKEY_ON."""
    try:
        import os as _os
        return _os.path.exists('/root/PLACEKEY_ON')
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


def _TILEHUE_ON():
    """A CLICK TARGET THAT WEARS THE SELECTED COLOUR IS FOUND BY ITS SHAPE.
    /root/TILEHUE_ON.  MEASURED cd82 L5 (2026-09-11): every life stalls at
    mismatch 24 because the plan's last round is a click on the tile, keyed
    with the colour the tile wore when the effect was learned (14); after the
    plan's own swatch click the tile wears the new colour (11), `_find` finds
    nothing -> `no_tile_on_board` (568) -> arrows for the rest of the life.
    The tile's shape has worn six hues (carrier 271): by the CARRIERHUE rule
    in this file it is a selection indicator, and an indicator's colour
    names nothing.  Under this gate `_find` falls back from the exact
    (colour, shape) to the shape alone for such objects."""
    try:
        import os as _os
        return _os.path.exists("/root/TILEHUE_ON")
    except Exception:
        return False


def _CARRIERHUE_ON():
    """A colour indicator must have worn >= 2 colours.  /root/CARRIERHUE_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/CARRIERHUE_ON")
    except Exception:
        return False


MIN_EXACT = 3              # samples before a state's own transitions outrank pooled ones


def _sig(G, o):
    """Normalised, translation-invariant shape of one object."""
    col, size, r0, c0, r1, c1 = o
    m = (np.asarray(G)[r0:r1 + 1, c0:c1 + 1] == col)
    return hashlib.blake2b(bytes([r1 - r0 + 1, c1 - c0 + 1]) + np.packbits(m).tobytes(),
                           digest_size=6).hexdigest()


def _inside(o, R):
    """The object's box lies within R (a canvas blob, a target blob)."""
    return o[2] >= R[0] and o[4] <= R[2] and o[3] >= R[1] and o[5] <= R[3]


class RelPlan(object):
    def __init__(self):
        self.mutable = set()      # sigs that change: appear/disappear, or recolour
        self.seen = {}            # sig -> frames it appeared in
        self.change = {}          # sig -> steps it appeared, disappeared, moved or recoloured
        self.change_c = {}        # ... of which on CLICK steps (a selector bar moves on clicks)
        self.cochange = {}        # "s1|s2" (sorted) -> steps both changed together
        self.effects = {}         # (a, ck, S) -> [set(cells), n]
        self.auto = {}            # (S, a) -> {S2: n}
        self.carrier = {}         # sig -> votes: this object's colour is what gets painted
        self.hues = {}            # sig -> colours a unique object has been seen in (CARRIERHUE)
        self.pic_shape = None     # (h, w) of the picture this table learned on (PICFIX)
        self.swatch = {}          # colour -> [ck, votes]
        self.sel = None           # the colour the last colour-setting click chose
        self.last_painted = None  # the colour of the last paint he made
        self.plans = 0            # plans formed
        self.acts = 0             # actions handed out
        self.reached = 0          # steps whose canvas diff the full-state key predicted exactly
        self.judged = 0
        self.plan_reached = 0     # executed plan actions whose predicted canvas change came true
        self.plan_judged = 0
        self.exec_n = 0           # plan actions actually executed
        self.probes = 0           # curiosity picks handed out (CURIOUS)
        self.probe_applies = 0    # ... of which the apply at the mask (final), stuck guard passed
        self.probe_paint_applies = 0  # ... applies issued ON THE WAY to a mask (patch 25)
        self.probe_exec = 0       # final probes that RAN (marked on execution, patch 25)
        self.walks = 0            # walks latched toward a mask (patch 25)
        self.walks_dropped = 0    # ... abandoned: unreachable, over budget, stuck
        self.walks_dropped_why = {}  # ... by reason (patch 25b)
        self.events = []          # walk events for the journal, drained by the world (25b)
        self.probe_hue_clicks = 0 # swatch clicks issued so a probe paints a colour the canvas lacks (26)
        self.null_masks = {}      # q-state -> times an absent-colour probe there painted NOTHING (26, persisted)
        self.same_mask = set()    # frozenset({q1, q2}): a swatch click swapped them, canvas unchanged (27, persisted)
        self.floor_held = 0       # stalls the floor gate kept from exploring (patch 25)
        self.lookaheads = 0       # searches run (patch 28)
        self.lookahead_hits = 0   # ... that found a sequence beating the present picture
        self.lookahead_acts = 0   # first actions handed out from a lookahead sequence
        self._la_cache = {}       # (canvas, state, sel, table) -> sequence or None (bounded)
        self._seq = None          # the latched lookahead sequence: {"steps": [(a, ck, Sreq, c)], "i"} (28c)
        self.lookahead_done = 0   # sequences executed to their end
        self.lookahead_dropped = 0  # ... abandoned: misprediction, unreachable, stuck
        self.seq_hues = 0         # colour clicks a latched step issued AT its mask (SEQHUE, 29)
        self.seq_twins = 0        # ... that changed the q-state: the twin is where the mask now is
        self.seq_prehues = 0      # colour clicks issued because no arrow path reached the step's mask (29)
        self.seq_refuted = 0      # latched steps burned for the life after a misprediction or stuck drop (REFUTE, 30)
        self.la_full = 0          # FULL lookahead searches run (LAFULL, 33)
        self.la_full_capped = 0   # ... that hit LA_MAX_CANVASES or LA_TIME_FULL
        self._la_found = {}       # FULL search results per (canvas, table, burned, colours)
        self.shortcuts = 0        # shortcut sequences latched on a WON level (SHORTCUT, 40)
        self.shortcut_done = 0    # ... executed to their end
        self.shortcut_dropped = 0 # ... abandoned mid-attempt (misprediction, stuck, ...)
        self.shortcut_won = 0     # attempts the world judged: cleared in fewer actions
        self.shortcut_refuted = 0 # ... not shorter, or ended without a clear
        self._sc_try = False      # this attempt's first plan may try a shortcut (begin_life)
        self._sc_life = None      # the route length a latched shortcut is judged against
        self._sc_refuted = None   # the route length a shortcut failed against (persisted)
        self._sc_off = False      # a shortcut dropped this attempt: he is off the won route
        self.place_auto = {}      # (place, a) -> {place2: n}: where the arrows lead (PLACEKEY, 41)
        self.place_of = {}        # (a, ck, S) -> {place: n}: where an effect was applied from
        self.place_paint = {}     # (place, a) -> {mask: n}: what a paint from here puts down
        self.place_navs = 0       # navigations attempted on the place automaton
        self.place_hits = 0       # ... that found a target and a path (or stood on it)
        self.place_explores = 0   # ... that pressed an untried arrow to learn the way
        self._sc_refuted_places = None  # places known when the route length was refuted
        self._sc_retries = 0            # shortcut retries against the refuted route length
        self.shortcut_yields = 0  # latches that yielded to a greedy plan completing sooner
        self._sc_yield = False    # this attempt's shortcut yielded: its greedy keeps precedence
        self.la_unfloored = 0     # lookaheads run above the level's best on an unwon level (LAFLOOR, 30)
        self._probed = set()      # mask states probed this life (marked when the probe RAN)
        self._walk = None         # the latched walk: {"X", "act", "d", "n"} (patch 25)
        self._walk_failed = set() # targets whose walk was abandoned this life
        self._stuck = {}
        self._pending = None      # the last pick: what it predicted, to be judged when executed
        self._burned = set()      # effects whose paint WORSENED the picture this life
        self.spoiled = {}         # (state at pick, op) -> [times the executed pick WORSENED the picture, cells known then] (SPOILMEM)
        self.worked = {}          # (state at pick, op) -> times the executed pick IMPROVED the picture
        self.why = ""             # why the last plan() said nothing
        self.why_counts = {}
        self._last_ck = None
        self._logged = False
        self._auto_n = 0
        self._view = None
        self._steps = 0

    # ------------------------------------------------------------ perception
    def outside(self, G, objs, A, B):
        """(sig, colour, box) of every object outside the canvas and target."""
        out = []
        for o in objs or []:
            # what lies INSIDE the canvas or the target is the picture,
            # not an indicator; an icon that floats over the canvas's
            # edge is still an indicator (cd82's stamp does)
            if _inside(o, A) or _inside(o, B):
                continue
            if o[1] > 0.40 * W * W:
                continue
            h, w = o[4] - o[2] + 1, o[5] - o[3] + 1
            # a line (a budget bar shrinking by one cell a step, a border)
            # is not an indicator: its shape would change every step and
            # no state would ever recur
            if (h == 1 and w >= W // 4) or (w == 1 and h >= W // 4):
                continue
            out.append((_sig(G, o), int(o[0]), (o[2], o[3], o[4], o[5])))
        return out

    def state(self, outs):
        """(state, colour of every sig that is unique on the board).  A
        state is the set of (sig, r0, c0, r1, c1): shape AND place."""
        S = frozenset((s, b[0], b[1], b[2], b[3]) for s, c, b in outs)
        cnt = {}
        for s, c, b in outs:
            cnt[s] = cnt.get(s, 0) + 1
        cols = dict((s, c) for s, c, b in outs if cnt[s] == 1)
        return S, cols

    def click_key(self, outs, aim, objs, A, B):
        """The innermost object under the aim, as (colour, sig, third-row,
        third-col); offsets only for objects at least BIG_OBJECT big."""
        if aim is None:
            return None
        y, x = int(aim[0]), int(aim[1])
        best = None
        for s, c, b in outs:
            if b[0] <= y <= b[2] and b[1] <= x <= b[3]:
                area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
                if best is None or area < best[0]:
                    best = (area, s, c, b)
        if best is None:
            return ("bg",)
        _, s, c, b = best
        h, w = b[2] - b[0] + 1, b[3] - b[1] + 1
        if h >= BIG_OBJECT and w >= BIG_OBJECT:
            return (c, s, (y - b[0]) * 3 // h, (x - b[1]) * 3 // w)
        return (c, s, 1, 1)

    def _carrier_ok(self, s):
        if s not in self.mutable:
            return False
        if _CARRIERHUE_ON():
            # a selection indicator has WORN at least two colours; a
            # constant-colour object that merely moves (cd82's black
            # selector bar, 126 votes on level 0) names nothing
            return len(self.hues.get(s, ())) >= 2
        return True

    def colour_key(self, outs, aim, c):
        """The thing you click to get colour c is the thing OF colour c at
        or next to the aim: the smallest object of colour c whose box
        lies within 2 cells of the aim.  A frame around a swatch is
        shared by every swatch and names no colour."""
        if aim is None:
            return None
        y, x = int(aim[0]), int(aim[1])
        best = None
        for s, col, b in outs:
            if col != c:
                continue
            if b[0] - 2 <= y <= b[2] + 2 and b[1] - 2 <= x <= b[3] + 2:
                area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
                if best is None or area < best[0]:
                    best = (area, (c, s, 1, 1))
        return best[1] if best is not None else None

    def selected(self, S, cols):
        """The colour he has selected: the unique, colour-changing object
        with the most carrier votes; else his last colour-setting click;
        else the colour he painted last."""
        best = None
        for s, c in cols.items():
            v = self.carrier.get(s, 0)
            if v > 0 and self._carrier_ok(s) and (best is None or v > best[0]):
                best = (v, c)
        if best is not None:
            return best[1]
        if self.sel is not None:
            return self.sel
        return self.last_painted

    # --------------------------------------------------------------- learning
    def begin_life(self):
        """A life resets the board, and with it whatever colour was
        selected: the selection is re-read from the carrier, or learned
        again from his next click."""
        self._stuck = {}
        self._logged = False
        self._pending = None
        self._burned = set()
        self._probed = set()
        self._walk = None
        self._walk_failed = set()
        self._seq = None
        self.sel = None
        self.last_painted = None
        self._last_ck = None
        self._sc_try = True
        self._sc_off = False
        self._sc_yield = False
        self._sc_life = None

    def observe(self, Gp, objs_p, Gc, objs_c, a, is_click, aim, A, B,
                place=None, place_next=None):
        """One step: what changed outside the canvas, what changed in it."""
        Gp = np.asarray(Gp); Gc = np.asarray(Gc)
        if _PICFIX_ON():
            # PICFIX: this table belongs to ONE picture; another shape is
            # another table's business (cd82 L2: a 3x4 tile-region pair
            # confirmed at the win would otherwise be learned into the
            # 10x10 table and asked of it)
            _shape = (int(A[2] - A[0] + 1), int(A[3] - A[1] + 1))
            if self.pic_shape is None:
                # stamp only a picture the table's own cells FIT (an old
                # blob has no stamp; a wrong first picture must not
                # become its shape for good)
                if self._max_cell() >= _shape[0] * _shape[1]:
                    return
                self.pic_shape = _shape
            elif tuple(self.pic_shape) != _shape:
                return
        op = self.outside(Gp, objs_p, A, B)
        oc = self.outside(Gc, objs_c, A, B)
        Sp, colp = self.state(op)
        Sc, colc = self.state(oc)
        self._steps += 1
        # CARRIERHUE: which colours has each unique object worn?  An
        # object of one colour only cannot indicate a colour.
        for _cols in ((colp, colc) if self._steps == 1 else (colc,)):
            for _s, _c in _cols.items():
                _h = self.hues.get(_s)
                if _h is None:
                    if len(self.hues) >= 4 * MAX_STATES:
                        continue
                    _h = self.hues[_s] = set()
                if len(_h) < 16:
                    _h.add(int(_c))
        if self._steps == 1:
            for t in Sp:
                self.seen[t[0]] = self.seen.get(t[0], 0) + 1
        for t in Sc:
            self.seen[t[0]] = self.seen.get(t[0], 0) + 1
        # a part that appeared, disappeared, MOVED or recoloured changed
        changed = set(t[0] for t in (Sp ^ Sc))
        for s in colp:
            if s in colc and colp[s] != colc[s]:
                changed.add(s)
        for s in changed:
            self.mutable.add(s)
            self.change[s] = self.change.get(s, 0) + 1
            if is_click:
                self.change_c[s] = self.change_c.get(s, 0) + 1
        if 2 <= len(changed) <= 8:
            cl = sorted(changed)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    k = cl[i] + "|" + cl[j]
                    self.cochange[k] = self.cochange.get(k, 0) + 1
        a = int(a)
        if is_click:
            self.click_index = a
        if _PLACEKEY_ON() and place is not None and place_next is not None and not is_click:
            # PLACEKEY (41): the arrows of THIS place.  Paint under the stamp cannot
            # move it, so the same walk from here always lands on the same board.
            try:
                _pd = self.place_auto.get((place, a))
                if _pd is None and len(self.place_auto) < MAX_PLACES * 6:
                    _pd = self.place_auto.setdefault((place, a), {})
                if _pd is not None:
                    _pd[place_next] = _pd.get(place_next, 0) + 1
            except Exception:
                pass
        ck = self.click_key(op, aim, objs_p, A, B) if is_click else None
        # the canvas
        cp = Gp[A[0]:A[2] + 1, A[1]:A[3] + 1]
        cc = Gc[A[0]:A[2] + 1, A[1]:A[3] + 1]
        diff = cp != cc
        # THE PLAN IS JUDGED BY WHAT HAPPENED WHEN ITS ACTION RAN: the
        # pick it handed out, if this is the step that executed it
        pend = self._pending
        self._pending = None
        if pend is not None and pend["a"] == a and (
                not is_click or pend["aim"] is None or
                (aim is not None and (int(aim[0]), int(aim[1])) == tuple(pend["aim"]))):
            self.exec_n += 1
            self._stuck[pend["key"]] = self._stuck.get(pend["key"], 0) + 1
            got = set(int(i) for i in np.flatnonzero(diff))
            if pend.get("probe"):
                # a probe predicts nothing; it is judged by what it teaches.
                # MARKED ON EXECUTION (patch 25): a pick a route overruled
                # never tried anything (20% of probe picks live), so the
                # mask stays open until the apply actually ran
                _X = pend.get("walk")
                if _X is None:
                    _X = pend["S"]
                _w = getattr(self, "_walk", None)
                if pend.get("final", True):
                    self._probed.add(_X)
                    self.probe_exec += 1
                    self._event("probe_exec painted=%d effects=%d" % (len(got), len(self.effects)))
                    if not got and pend.get("sel_absent") and _X:
                        # PROBEHUE (26): the colour was absent from the whole
                        # canvas, so any mask would have shown -- there is none
                        _nm = getattr(self, "null_masks", None)
                        if _nm is not None and (_X in _nm or len(_nm) < MAX_STATES):
                            _nm[_X] = _nm.get(_X, 0) + 1
                            self._event("null_mask state=%d n=%d" % (len(_X), _nm[_X]))
                    if _w is not None and _w.get("X") == _X:
                        self._walk = None
                elif _w is not None and _w.get("X") == _X:
                    _w["n"] = int(_w.get("n", 0)) + 1
                    if pend.get("hue") is not None:
                        # SAMEMASK (27b): the state the walk's OWN colour
                        # click produced is where its target now is
                        try:
                            _w["twin"] = self.q(Sc)
                        except Exception:
                            pass
                pend = None
        if pend is not None and pend["a"] == a and (
                not is_click or pend["aim"] is None or
                (aim is not None and (int(aim[0]), int(aim[1])) == tuple(pend["aim"]))):
            self.plan_judged += 1
            if got == pend["pred"] and (not got or set(int(x) for x in cc[diff]) == {pend["colour"]}):
                self.plan_reached += 1
            _sh_i = pend.get("seq_hue") if pend.get("seq_hue") is not None else pend.get("seq_prehue")
            if _sh_i is not None and getattr(self, "_seq", None) is not None \
                    and int(_sh_i) == int(self._seq["i"]) and not got:
                # SEQHUE (29): the q-state this step's own colour click produced.
                # At the mask it is the step's TWIN -- where the mask now is (the
                # walk's twin, 27b); anywhere, the pair is one mask in two colours.
                try:
                    _hc = pend.get("hue_c")
                    if _hc is not None and int(_hc) in set(int(v) for v in colc.values()):
                        # the plan clicked colour c on purpose and a unique object
                        # now wears it: c is selected.  Neither selection rule
                        # fires when the click re-segments the icon (no same-
                        # signature recolour) and the swatch was known by KIND
                        # only (no entry to match) -- the stale selection would
                        # have clicked c again until the stuck guard dropped the latch
                        self.sel = int(_hc)
                        if int(_hc) not in self.swatch and pend.get("hue_ck") is not None:
                            self.swatch[int(_hc)] = [pend["hue_ck"], 0]
                    _q0 = pend.get("S"); _q1 = self.q(Sc)
                    if _q1 and pend.get("seq_hue") is not None:
                        self._seq["twin"] = _q1; self._seq["twin_i"] = int(_sh_i)
                        self._seq["twin_from"] = _q0
                    if _q0 and _q1 and _q1 != _q0:
                        if pend.get("seq_hue") is not None:
                            self.seq_twins += 1
                            self._event("seq_twin i=%d" % int(_sh_i))
                        _pr = frozenset([_q0, _q1])
                        if _SAMEMASK_ON() and _pr not in self.same_mask and len(self.same_mask) < MAX_STATES:
                            self.same_mask.add(_pr)
                            self._event("same_mask n=%d" % len(self.same_mask))
                except Exception:
                    pass
            if pend.get("seq") is not None and getattr(self, "_seq", None) is not None:
                # a latched sequence advances on its own paint, as predicted
                if got == pend["pred"] and int(pend["seq"]) == int(self._seq["i"]):
                    self._seq["i"] += 1
                    if self._seq["i"] >= len(self._seq["steps"]):
                        self.lookahead_done += 1
                        if self._seq.get("shortcut") is not None:
                            self.shortcut_done += 1
                            self._event("shortcut_done route=%d" % int(self._seq["shortcut"]))
                        self._event("seq_done")
                        self._seq = None
                else:
                    self._drop_seq("misprediction")
            # A PAINT THAT MADE THE PICTURE WORSE is not planned on again
            # this life: its learned cells were a subset of its mask (live:
            # the same +10 prediction lost 25 three times in one life)
            if pend.get("op") is not None and got:
                tg = Gc[B[0]:B[2] + 1, B[1]:B[3] + 1]
                _m1 = int((cc != tg).sum()); _m0 = int((cp != tg).sum())
                if _m1 > _m0:
                    self._burned.add(pend["op"])
                    # SPOILMEM: remembered across lives, with how much of
                    # the mask was known -- new cells forgive the spoil
                    # keyed on the PAIR (state he planned from, effect): the
                    # same effect works from its exact state and spoils from
                    # a subset match (cd82 L0: +50 at step 27, -50 at step 35,
                    # every life) -- the match is the fault, not the effect
                    _op = pend["op"]; _pk = (pend.get("S"), _op)
                    _known = 0
                    try:
                        _eff, _ = self.views()
                        _known = len(_eff.get(_op, [(), 0])[0])
                    except Exception:
                        _known = 0
                    _sp = self.spoiled.get(_pk)
                    if _sp is None:
                        if len(self.spoiled) < MAX_EFFECTS:
                            self.spoiled[_pk] = [1, int(_known)]
                    else:
                        _sp[0] += 1; _sp[1] = int(_known)
                elif _m1 < _m0:
                    _pk = (pend.get("S"), pend["op"])
                    if _pk in self.worked or len(self.worked) < MAX_EFFECTS:
                        self.worked[_pk] = self.worked.get(_pk, 0) + 1
        if diff.any():
            painted = set(int(x) for x in cc[diff])
            key = (a, ck, Sp)
            if _PLACEKEY_ON() and place is not None:
                # PLACEKEY (41): what a paint FROM HERE puts down, and that this effect
                # (keyed as the planner asks for it -- views() re-keys on the quotient)
                # has been applied from here
                try:
                    _cells = frozenset(int(i) for i in np.flatnonzero(diff.ravel()))
                    _pp = self.place_paint.get((place, a, ck))
                    if _pp is None and len(self.place_paint) < MAX_PLACES * 6:
                        _pp = self.place_paint.setdefault((place, a, ck), {})
                    if _pp is not None and (_cells in _pp or len(_pp) < 8):
                        _pp[_cells] = _pp.get(_cells, 0) + 1
                    _pk = (a, ck, self.q(Sp))
                    _pw = self.place_of.get(_pk)
                    if _pw is None and len(self.place_of) < MAX_EFFECTS:
                        _pw = self.place_of.setdefault(_pk, {})
                    if _pw is not None and (place in _pw or len(_pw) < 16):
                        _pw[place] = _pw.get(place, 0) + 1
                except Exception:
                    pass
            e = self.effects.get(key)
            if e is None:
                if len(self.effects) >= MAX_EFFECTS:
                    # the table is full: the least-supported entry goes;
                    # nothing else in this step is skipped
                    victim = min(self.effects, key=lambda k: self.effects[k][1])
                    self.effects.pop(victim, None)
                    self._click_sigs_cache = None   # QTILE: the table changed at equal length
                    self._paint_actions_cache = None  # NAVPAINT: same
                e = [set(), 0]
                self.effects[key] = e
            cells = set(int(i) for i in np.flatnonzero(diff))
            # was this step predicted?  (judged before the union grows)
            if e[1] > 0 and len(painted) == 1:
                p = next(iter(painted))
                pred = set(i for i in e[0] if int(cp.flat[i]) != p)
                self.judged += 1
                if pred == cells:
                    self.reached += 1
            e[0] |= cells
            e[1] += 1
            if len(painted) == 1:
                p = next(iter(painted))
                self.last_painted = p
                for s, c in colp.items():
                    if c == p:
                        self.carrier[s] = self.carrier.get(s, 0) + 1
                if self._last_ck is not None and self._last_ck[1] == p:
                    sw = self.swatch.setdefault(p, [self._last_ck[0], 0])
                    sw[1] += 1
        # the indicator
        if not is_click:
            if Sp or Sc:
                d = self.auto.setdefault((Sp, a), {})
                if len(self.auto) < MAX_STATES * 6 or (Sp, a) in self.auto:
                    d[Sc] = d.get(Sc, 0) + 1
                    self._auto_n += 1
        else:
            # a click that changed the colour of a unique object set the
            # selection to that object's new colour
            new = set(colc[s] for s in colc if s in colp and colp[s] != colc[s])
            if len(new) == 1 and ck is not None and ck != ("bg",):
                c = next(iter(new))
                self.sel = c
                cky = self.colour_key(op, aim, c)
                if cky is not None:
                    self._last_ck = (cky, c)
                    sw = self.swatch.setdefault(c, [cky, 0])
                    sw[0] = cky
            if _SAMEMASK_ON() and ck is not None and ck != ("bg",) and not diff.any():
                # SAMEMASK (27): a colour click cannot change the stamp's
                # MASK, yet 30/76 L3 clicks changed the q-state (the
                # recoloured icon segments differently: one signature
                # swapped, canvas unchanged) -- the two states are one
                # mask in two colours.  Only a click on a SWATCH KIND
                # (or one that recoloured a same-signature object) says
                # so; a tile click changes the mask and is not one.
                try:
                    _kinds = set(sw[0][1] for sw in self.swatch.values()
                                 if sw[0] is not None and len(sw[0]) == 4)
                    _y, _x = int(aim[0]), int(aim[1])
                    # the aim sits on or beside an object of a swatch's
                    # SHAPE (a click on the frame ring selects too)
                    _near = any(sg in _kinds and b[0] - 2 <= _y <= b[2] + 2
                                and b[1] - 2 <= _x <= b[3] + 2 for sg, col, b in op)
                    if len(new) == 1 or _near:
                        _q0 = self.q(Sp); _q1 = self.q(Sc)
                        if _q0 and _q1 and _q0 != _q1:
                            _pr = frozenset([_q0, _q1])
                            if _pr not in self.same_mask and len(self.same_mask) < MAX_STATES:
                                self.same_mask.add(_pr)
                                self._event("same_mask n=%d" % len(self.same_mask))
                except Exception:
                    pass
            if not new and ck is not None and ck != ("bg",):
                # A CLICK THAT CHANGED NOTHING still says what is selected:
                # clicking a known swatch of colour c while the indicator
                # already shows c (live 21:48Z: nine clicks on the 15
                # swatch, the remembered selection stuck at 9)
                for c, sw in self.swatch.items():
                    if sw[0] is not None and len(sw[0]) == 4 and sw[0][1] == ck[1] and ck[0] == c:
                        if c in set(colc.values()):
                            self.sel = c
                        break

    # ---------------------------------------------------------------- planning
    def _click_sigs(self):
        """Sigs of the shapes his learned effects CLICK on (QTILE)."""
        _ce = getattr(self, "_click_sigs_cache", None)
        if _ce is None or _ce[0] != len(self.effects):
            _ce = (len(self.effects),
                   frozenset(k[1][1] for k in self.effects
                             if k[1] is not None and len(k[1]) >= 2))
            self._click_sigs_cache = _ce
        return _ce[1]

    def q(self, S):
        """A state as the planner reads it: the PRIMARY (the most changing
        shape present) at offset (0, 0), plus every changing part that
        lies within the primary's box grown by its own size, each with
        its box offset from the primary.  Empty when nothing present ever
        changed.  A part that only MOVES (cd82 level 2's two-cell mark
        that says which half) is a different state in a different place."""
        # change under NON-CLICK actions: what the arrows do to the board
        # is the mask indicator; what clicks move (a selector bar) is not
        ca = dict((t[0], self.change.get(t[0], 0) - self.change_c.get(t[0], 0)) for t in S)
        m = 0
        for t in S:
            if t[0] in self.mutable:
                m = max(m, ca[t[0]])
        if m <= 0:
            return frozenset()
        pt = max((t for t in S if t[0] in self.mutable),
                 key=lambda t: (ca[t[0]], self.seen.get(t[0], 0), t))
        p = pt[0]
        pr0, pc0, pr1, pc1 = pt[1], pt[2], pt[3], pt[4]
        ph, pw = pr1 - pr0 + 1, pc1 - pc0 + 1
        parts = [(p, 0, 0)]
        for t in S:
            if t == pt or t[0] not in self.mutable:
                continue
            c = ca[t[0]]
            if c * 5 < m and not (_QTILE_ON() and t[0] in self._click_sigs()):
                # a rare part is dropped -- unless it is a shape he CLICKS
                # on (QTILE, 2026-09-07): cd82 L2's tile `1fe92e` changes
                # in 19 of the 52 frames it is present, under a fifth of
                # the primary's count, and without it in the state an
                # effect learned with one tile shape matched a board
                # showing another (83/83 silent asks)
                continue
            if c * 2 < self.change.get(t[0], 0):
                continue
            if not (pr0 - ph <= t[1] <= pr1 + ph and pc0 - pw <= t[2] <= pc1 + pw):
                continue
            parts.append((t[0], t[1] - pr0, t[2] - pc0))
        return frozenset(parts)

    @staticmethod
    def comparable(S, T):
        return S <= T or T <= S

    def views(self):
        """Effects and automaton re-keyed on the mutable part of their
        states, rebuilt when anything they depend on has changed."""
        key = (len(self.mutable), len(self.effects), self._auto_n, self._steps // 50)
        if self._view is not None and self._view[0] == key:
            return self._view[1], self._view[2]
        eff = {}
        for (a, ck, S), (cells, n) in self.effects.items():
            if not cells:
                continue
            k = (a, ck, self.q(S))
            if not k[2]:
                # learned when nothing present was known to change: it
                # would match EVERY state (the empty set is comparable
                # with all) -- not a model, a guess
                continue
            e = eff.get(k)
            if e is None:
                eff[k] = [set(cells), n]
            else:
                e[0] |= cells; e[1] += n
        auto = {}
        for (S, a), d in self.auto.items():
            k = (self.q(S), a)
            dd = auto.setdefault(k, {})
            for T, n in d.items():
                qt = self.q(T)
                dd[qt] = dd.get(qt, 0) + n
        self._view = (key, eff, auto)
        return eff, auto

    def _next(self, auto, S, a, tw=False):
        """Modal successor of S under a.  EXACT evidence for S first, when
        it has at least MIN_EXACT samples; only then the pooled evidence
        of every comparable state (an outline alone pools both fills'
        futures -- live that made one arrow right 10 times in 25).
        `tw` (SAMEMASK, the walk only): a twin's evidence pools too."""
        d = auto.get((S, a))
        if d and sum(d.values()) >= MIN_EXACT:
            return max(d.items(), key=lambda kv: kv[1])[0]
        d = {}
        _tw = self._twins(S) if (tw and _SAMEMASK_ON()) else None
        for (T, aa), dd in auto.items():
            if aa == a and (self.comparable(S, T) or (_tw is not None and T in _tw)):
                for U, n in dd.items():
                    d[U] = d.get(U, 0) + n
        if not d:
            return None
        return max(d.items(), key=lambda kv: kv[1])[0]

    def _max_cell(self):
        """The largest canvas cell index any effect holds (-1 if none)."""
        m = -1
        for e in self.effects.values():
            if e[0]:
                v = max(e[0])
                if v > m:
                    m = v
        return m

    def _paint_actions(self):
        """Actions that have ever painted the canvas (NAVPAINT)."""
        _pc = getattr(self, "_paint_actions_cache", None)
        if _pc is None or _pc[0] != len(self.effects):
            _pc = (len(self.effects), frozenset(int(k[0]) for k in self.effects))
            self._paint_actions_cache = _pc
        return _pc[1]

    def _dist(self, auto, S0, n_actions, navpaint=True, tw=None):
        """BFS over modal transitions: state -> (distance, first action).
        `navpaint=False` walks the FULL automaton, applies included -- the
        exploration walk of a stalled life (CURIOUS, patch 25)."""
        if _NAVPAINT_ON() and navpaint:
            # NAVPAINT (2026-09-08): an action that PAINTS is an operator,
            # not a way of getting somewhere.  cd82 L3: the automaton knew
            # that apply moves the indicator, so the reach branch issued
            # an unpredicted apply that spoiled the picture every 4 steps
            # (19 of 20 spoils); such transitions are not navigation.
            _paint = self._paint_actions()
            if _paint:
                auto = dict(((X, a), d) for (X, a), d in auto.items() if a not in _paint)
        out = {S0: (0, None)}
        q = deque([S0])
        while q:
            S = q.popleft()
            for a in range(int(n_actions)):
                T = self._next(auto, S, a, tw=((not navpaint) if tw is None else bool(tw)))
                if T is None or T in out:
                    continue
                first = out[S][1] if out[S][1] is not None else a
                out[T] = (out[S][0] + 1, first)
                q.append(T)
        return out

    def _reach(self, dist, Sreq, skip=None):
        """The nearest reached state comparable with Sreq: (distance,
        first action, state) or None.  `skip(X)` (EXACTFIRST, 34): a state
        whose own entry contradicts Sreq's is not a place to apply from."""
        best = None
        for X, (d, first) in dist.items():
            if self.comparable(X, Sreq) and (best is None or d < best[0]):
                if skip is not None and X != Sreq and skip(X):
                    continue
                best = (d, first, X)
        return best

    def _places_known(self):
        try:
            return len(set(P for P, _a in getattr(self, "place_auto", {})))
        except Exception:
            return 0

    def _sc_is_refuted(self, route_len):
        """SHORTCUT (40) refuted this route length after one failed attempt.  PLACEKEY
        (41): a retry is EARNED by new evidence -- the number of places he knows has
        grown since the refutation (his first plays after a deploy teach the 8 slots).
        Bounded by the places there are.  Gate off: refuted stays refuted."""
        if getattr(self, "_sc_refuted", None) != int(route_len):
            return False
        if not _PLACEKEY_ON():
            return True
        if int(getattr(self, "_sc_retries", 0)) >= 3:
            return True                      # review of 41: retries are capped per route length
        if not getattr(self, "place_paint", None):
            return True                      # nothing to navigate to yet
        _then = getattr(self, "_sc_refuted_places", None)
        if _then is None:
            return False                     # a refutation from before places existed
        return self._places_known() <= int(_then)


    def _reach_place(self, place, op, n_actions):
        """PLACEKEY (41): BFS over the PLACE automaton -- modal successors of places,
        painting actions excluded -- from where he stands to a place where the step's
        action PAINTS THE STEP'S CELLS (fallback: a place the effect was applied from).
        (distance, first action) or None.  Measured off-route on cd82: arrows have ONE
        successor 94-96% of the time here against 70-77% under the quotient state, and
        a 3-step path holds at ~98% against ~83%."""
        try:
            self.place_navs += 1
            self._nav_explore = False
            if place is None:
                return None
            tgt = set()
            _want = None
            try:
                _e = self.views()[0].get(op)
                _want = frozenset(_e[0]) if _e and _e[0] else None
            except Exception:
                _want = None
            if _want:
                # a place that PAINTS THESE CELLS is the target: learnable from any paint,
                # ordinary route play included -- "places this effect was applied from" had
                # no target on 7 of 9 navigations (offline cd82 L3)
                for (P, aa, cck), d in getattr(self, "place_paint", {}).items():
                    if int(aa) != int(op[0]) or cck != op[1]:
                        continue
                    for _m in d:
                        # a paint from the right place is a SUBSET of the mask (cells
                        # already that colour do not change) and at least half of it
                        if _m and _m <= _want and 2 * len(_m) >= len(_want):
                            tgt.add(P); break
            _n_paint = len(tgt)
            if not tgt:
                tgt = set(getattr(self, "place_of", {}).get(op) or ())
            if not tgt:
                self._event("place_miss why=no_target want=%s" % (len(_want) if _want else None))
                return None
            if place in tgt:
                self.place_hits += 1
                return (0, None)
            _paint = self._paint_actions() if _NAVPAINT_ON() else frozenset()
            out = {place: (0, None)}
            dq = deque([place])
            while dq and len(out) <= MAX_PLACES:
                X = dq.popleft()
                for a in range(int(n_actions)):
                    if a in _paint:
                        continue
                    d = self.place_auto.get((X, a))
                    if not d:
                        continue
                    T = max(d.items(), key=lambda kv: kv[1])[0]
                    if T in out:
                        continue
                    out[T] = (out[X][0] + 1, out[X][1] if out[X][1] is not None else a)
                    if T in tgt:
                        self.place_hits += 1
                        return (out[T][0], out[T][1])
                    dq.append(T)
            # EXPLORE (41): no known arrow leads there.  Press an arrow never pressed
            # FROM HERE (else from the nearest place that has one) so the automaton
            # learns the way -- offline cd82 L3: without this, four retries oscillated
            # between the same two places and taught 0 new ones (explored=2 every time)
            for X, (d, first) in sorted(out.items(), key=lambda kv: kv[1][0]):
                for a in range(int(n_actions)):
                    if a in _paint or (X, a) in self.place_auto:
                        continue
                    self.place_explores += 1
                    self._nav_explore = True
                    _act = a if first is None else first
                    self._event("place_explore d=%d a=%d tgt=%d explored=%d" % (d, _act, len(tgt), len(out)))
                    return (d + 1, _act)
            self._event("place_miss why=unreachable tgt=%d paint_tgt=%d explored=%d" % (
                len(tgt), _n_paint, len(out)))
            return None
        except Exception:
            return None

    def _reach_raw(self, Sfull, Sreq, n_actions, a_ck=None, skip=None):
        """SHORTCUT (40): navigation on the RAW automaton.  BFS over modal
        successors of FULL states (shape AND place of every part; painting
        actions excluded) from where he stands.  Target: the nearest raw
        state where this very effect was applied before (its quotient is
        Sreq); else any raw state whose quotient is comparable with Sreq.
        (distance, first action, quotient state) or None -- the caller then
        falls back to the quotient automaton."""
        try:
            _paint = self._paint_actions() if _NAVPAINT_ON() else frozenset()
            out = {Sfull: (0, None)}
            dq = deque([Sfull])
            while dq and len(out) <= MAX_STATES:
                X = dq.popleft()
                for a in range(int(n_actions)):
                    if a in _paint:
                        continue
                    d = self.auto.get((X, a))
                    if not d:
                        continue
                    T = max(d.items(), key=lambda kv: kv[1])[0]
                    if T in out:
                        continue
                    out[T] = (out[X][0] + 1, out[X][1] if out[X][1] is not None else a)
                    dq.append(T)
            exact = set()
            if a_ck is not None:
                exact = set(T for (a2, ck2, T) in self.effects
                            if (a2, ck2) == tuple(a_ck) and self.q(T) == Sreq)
            for want_exact in (True, False):
                best = None
                for T, (d, first) in out.items():
                    if first is None:
                        continue
                    qT = self.q(T)
                    if want_exact:
                        ok = T in exact
                    else:
                        ok = bool(qT) and self.comparable(qT, Sreq) and not (
                            skip is not None and qT != Sreq and skip(qT))
                    if ok and (best is None or d < best[0]):
                        best = (d, first, qT)
                if best is not None:
                    return best
            return None
        except Exception:
            return None

    def _find(self, outs, ck):
        """A cell to click for a click key on the current board, or None.

        Exact (colour, shape) first.  TILEHUE (39): when nothing wears the
        learned colour and the shape is a carrier -- it has worn two or
        more hues, the CARRIERHUE definition of a selection indicator --
        the object of that shape is taken whatever colour it wears now:
        an indicator's colour names the selection, not the object.  Never
        for a swatch shape (the swatches share one shape across colours)."""
        if ck is None or ck == ("bg",):
            return None
        c, s = ck[0], ck[1]
        _hit = None
        for sg, col, b in outs:
            if sg == s and col == c:
                _hit = b
                break
        if (_hit is None and _TILEHUE_ON()
                and s in self.mutable and len(self.hues.get(s, ())) >= 2
                and not any(sw and sw[0] and len(sw[0]) > 1 and sw[0][1] == s
                            for sw in self.swatch.values())):
            # never for a SWATCH shape: the swatches share one shape across
            # every colour and there a colour IS the identity
            _cands = [(col, b) for sg, col, b in outs if sg == s]
            if len(_cands) == 1:
                _hit = _cands[0][1]
                self.tilehue_finds = int(getattr(self, "tilehue_finds", 0)) + 1
                self._event("tilehue_find shape=%s wore=%s key=%s" % (str(s)[:6], _cands[0][0], c))
        if _hit is None:
            return None
        b = _hit
        h, w = b[2] - b[0] + 1, b[3] - b[1] + 1
        if h >= BIG_OBJECT and w >= BIG_OBJECT and len(ck) == 4:
            y = b[0] + (ck[2] * h) // 3 + max(0, (h // 3 - 1) // 2)
            x = b[1] + (ck[3] * w) // 3 + max(0, (w // 3 - 1) // 2)
        else:
            y = (b[0] + b[2]) // 2; x = (b[1] + b[3]) // 2
        return (int(y), int(x))

    def _none(self, why):
        self.why = why
        self.why_counts[why] = self.why_counts.get(why, 0) + 1
        return None

    def plan(self, G, objs, A, B, n_actions, board_key=None, confirmed=True, floor=None,
             route_left=None, place=None):
        """Returns (action, aim, info) or None.  info = dict(rounds,
        steps, mm0, mm1, ...) for the whole plan.

        `floor` (patch 25): the best mismatch any life on this level has
        reached; the curiosity fallback fires only at or below it.

        Acts only where the picture is CONFIRMED by a clear, or the plan
        completes it, or the plan does at least half the job: on an
        unconfirmed candidate pair (a maze cell that matches another) a
        small "paint" would otherwise override the explorer."""
        G = np.asarray(G)
        outs = self.outside(G, objs, A, B)
        Sfull, cols = self.state(outs)
        S = self.q(Sfull)
        sel = self.selected(Sfull, cols)
        eff, auto = self.views()
        canvas = G[A[0]:A[2] + 1, A[1]:A[3] + 1].copy()
        target = G[B[0]:B[2] + 1, B[1]:B[3] + 1]
        if canvas.shape != target.shape:
            return self._none("shape")
        if _PICFIX_ON():
            if self.pic_shape is not None:
                if tuple(self.pic_shape) != tuple(int(x) for x in canvas.shape):
                    return self._none("other_picture")
            elif self._max_cell() >= int(canvas.size):
                return self._none("other_picture")
        mm0 = int((canvas != target).sum())
        if mm0 <= 0:
            return self._none("done")
        if _CURIOUS_ON() and getattr(self, "_walk", None) is not None:
            # A LATCHED WALK IS SERVED AHEAD OF THE ROUNDS (patch 25): the
            # apply on the way spoils the picture, and a greedy round
            # takes the repair over the walk every time (review of patch
            # 24: 65/65 walk applies were followed by a repair pick, no
            # probe ever ran).  The canvas resets each life; the mask
            # learned does not.
            _wk = self._serve_walk(S, eff, auto, n_actions, canvas, sel, mm0,
                                   outs=outs, target=target)
            if _wk is not None:
                return _wk
        ops = [(k, e) for k, e in eff.items() if e[0] and e[1] > 0 and k not in self._burned]
        _ops_all = list(ops)
        if _SPOILMEM_ON() and self.spoiled:
            # an effect that has HURT more often than it has worked, and
            # has learned nothing new since, is not planned on
            ops = [(k, e) for k, e in ops
                   if not (self.spoiled.get((S, k), [0, 0])[0] > self.worked.get((S, k), 0)
                           and len(e[0]) <= self.spoiled.get((S, k), [0, 0])[1])]
        # AN EFFECT THAT KNOWS MORE THAN THE STATE DOES applies only if it
        # is the only one: an outline alone is a subset of every state
        # that outline shares, and the union of two half-masks paints
        # nothing he wants (live 21:44Z, pred=100)
        by_ak = {}
        for (a, ck, Sreq), e in ops:
            by_ak.setdefault((a, ck), []).append(Sreq)

        def ambiguous(a, ck, X, Sreq):
            if X == Sreq:
                return False
            if _EXACTFIRST_ON():
                # EXACT-FIRST (34): his own repeated experience at X outranks
                # a sub-/super-state match with an entry learned elsewhere
                _ex = eff.get((a, ck, X))
                if _ex and _ex[0] and int(_ex[1]) >= MIN_EXACT and                         set(_ex[0]) != set(eff.get((a, ck, Sreq), [(), 0])[0]):
                    return True
            if Sreq < X:
                return False
            return sum(1 for T in by_ak.get((a, ck), ()) if X < T) > 1
        if not ops:
            return self._none("no_ops")
        colours = self._colour_map(outs, sel)
        if not colours:
            return self._none("no_colours")
        dist0 = self._dist(auto, S, n_actions)
        cur = canvas.copy(); cur_S = S; cur_sel = sel
        rounds = []; steps = 0
        _la = False
        if getattr(self, "_sc_try", False) and S:
            # SHORTCUT (patch 40): once per attempt, at its first plan WHERE HE
            # SEES HIS INDICATOR, on a level with a WON route (`route_left` =
            # (actions the route still needs from this board, its length)).  The
            # route replays the same boards forever -- a shorter clear can never
            # happen -- while his own full search reaches the picture in half the
            # actions (cd82 L3 14 vs 28, L4 13 vs 42).  At the level-start board
            # the indicator is not on the board (S empty, an alias of every
            # indicator-less board): the first step's arrow changed only the bar,
            # three times, and the latch died stuck (offline cd82, L0/L3/L4/L5);
            # the route walks him to where he can see it.  A route length a
            # shortcut was refuted against (the world's verdict) is not retried.
            self._sc_try = False
            if (_SHORTCUT_ON() and _LOOKAHEAD_ON() and route_left is not None
                    and getattr(self, "_seq", None) is None
                    and not self._sc_is_refuted(int(route_left[1]))):
                _sc = self._lookahead(_ops_all, colours, canvas, target, S, sel, auto, n_actions,
                                      dist0, ambiguous)
                if _sc and int((_sc[1] != target).sum()) == 0 and int(_sc[2]) < int(route_left[0]):
                    _cvs = [canvas.copy()]; _cv = canvas.copy()
                    for r_ in _sc[0]:
                        _cells = None
                        for (aa, cc_, Sr), (cl, n_) in _ops_all:
                            if aa == r_[0] and cc_ == r_[1] and Sr == r_[2]:
                                _cells = cl; break
                        _cv = _cv.copy()
                        if _cells:
                            _cv.flat[np.fromiter(_cells, dtype=np.int64)] = r_[3]
                        _cvs.append(_cv)
                    self._seq = {"steps": [(r_[0], r_[1], r_[2], r_[3]) for r_ in _sc[0]], "i": 0,
                                 "canvases": _cvs, "shortcut": int(route_left[1])}
                    self._sc_life = int(route_left[1])
                    if getattr(self, "_sc_refuted", None) == int(route_left[1]):
                        self._sc_retries = int(getattr(self, "_sc_retries", 0)) + 1
                    self.shortcuts += 1
                    self._event("shortcut_latch steps=%d route_left=%d route=%d mm=%d->0" % (
                        int(_sc[2]), int(route_left[0]), int(route_left[1]), mm0))
                else:
                    self._event("shortcut_none steps=%s end=%s route_left=%d" % (
                        (int(_sc[2]) if _sc else None),
                        (int((_sc[1] != target).sum()) if _sc else None), int(route_left[0])))
        if _LOOKAHEAD_ON() and getattr(self, "_seq", None) is not None:
            # A LATCHED SEQUENCE IS SERVED AHEAD OF THE GREEDY ROUNDS (28c):
            # after its spoiling first paint the greedy would take the
            # repair back to the floor every time (review of 28, K2)
            _sv = self._serve_seq(_ops_all, canvas, target, S, sel, auto, n_actions, dist0, ambiguous)
            if _sv is not None:
                rounds, cur, steps = _sv
                _la = True
        _sc_live = bool(_la and _SHORTCUT_ON() and getattr(self, "_seq", None) is not None
                        and self._seq.get("shortcut") is not None)
        _lat = (rounds, cur, steps) if _sc_live else None
        if _sc_live:
            # SHORTCUT (40): the greedy is priced on this board too (below)
            rounds = []; cur = canvas.copy(); cur_S = S; cur_sel = sel; steps = 0
        dist = dist0
        for _ in (range(MAX_ROUNDS) if not rounds else ()):
            best = None
            for (a, ck, Sreq), (cells, n) in ops:
                dd = self._reach(dist, Sreq, skip=((lambda X: ambiguous(a, ck, X, Sreq)) if _EXACTFIRST_ON() else None))
                if dd is None or ambiguous(a, ck, dd[2], Sreq):
                    continue
                idx = np.fromiter(cells, dtype=np.int64)
                for c in colours:
                    new = cur.copy()
                    new.flat[idx] = c
                    gain = int((cur != target).sum()) - int((new != target).sum())
                    if gain <= 0:
                        continue
                    cost = dd[0] + (1 if c != cur_sel else 0) + 1
                    score = (gain / float(cost), gain)
                    if best is None or score > best[0]:
                        best = (score, gain, cost, a, ck, Sreq, c, new)
            if best is None:
                break
            _, gain, cost, a, ck, Sreq, c, new = best
            rounds.append((a, ck, Sreq, c, gain, cost))
            steps += cost
            cur = new; cur_S = Sreq; cur_sel = c
            dist = self._dist(auto, cur_S, n_actions)
        if _sc_live:
            if rounds and int((cur != target).sum()) == 0 and int(steps) < int(_lat[2]):
                # SHORTCUT (40): a greedy plan that COMPLETES the picture in
                # fewer steps than the latch has left outranks it (review of
                # 40: offline cd82 L4 the latch walked away from a board with
                # a 2-4-action finish three times and died stuck; yielding
                # clears L4 in 13 actions, route 42).  Still the shortcut's
                # attempt: it keeps precedence and the world's verdict.
                self._event("shortcut_yield greedy_steps=%d latch_steps=%d" % (int(steps), int(_lat[2])))
                self.shortcut_yields += 1
                self._seq = None; self._sc_yield = True; self._sc_off = True; _la = False
            else:
                rounds, cur, steps = _lat
        _la_ok = bool(floor is None or mm0 <= int(floor))
        _la_free = bool(not _la_ok and _LAFLOOR_ON() and int(floor) > 0)
        if _la_free:
            # LAFLOOR (30): the floor gate is the probe's; the lookahead
            # searches what he KNOWS.  cd82 L4: the lookahead set the
            # level's best (20->13) and was then locked out of every
            # later life by that record (4 lives, 0 lookaheads, all dry).
            # A won level (floor 0) keeps it off.
            _la_ok = True
        if not _la_ok and getattr(self, "_sc_off", False) and _SHORTCUT_ON():
            # SHORTCUT (40): a dropped shortcut left the won route; "he can win
            # by the route" (floor 0) no longer holds -- search what he knows
            _la_ok = True
        if not rounds and _LOOKAHEAD_ON() and _la_ok:
            if _la_free:
                self.la_unfloored += 1   # a lookahead RUN above the floor (review of 30)
            # LOOKAHEAD (patch 28): nothing gains in one paint, but at the
            # floor a SEQUENCE of masks he knows may -- cd82 L3: paint 50
            # cells (spoils), then 55, then 15 -> 0.  Exploit before
            # explore: before the CURIOUS fallback.
            _seq = self._lookahead(_ops_all, colours, canvas, target, S, sel, auto, n_actions,
                                   dist0, ambiguous)
            if _seq:
                rounds, cur, steps = _seq
                _la = True
                # the latch carries the canvas expected before each step (28c,
                # review: drop when the board diverges between steps)
                _cvs = [canvas.copy()]; _cv = canvas.copy()
                for r in rounds:
                    _cells = None
                    for (aa, cc_, Sr), (cl, n) in _ops_all:
                        if aa == r[0] and cc_ == r[1] and Sr == r[2]:
                            _cells = cl; break
                    _cv = _cv.copy()
                    if _cells:
                        _cv.flat[np.fromiter(_cells, dtype=np.int64)] = r[3]
                    _cvs.append(_cv)
                self._seq = {"steps": [(r[0], r[1], r[2], r[3]) for r in rounds], "i": 0, "canvases": _cvs}
                self._event("seq_latch n=%d mm=%d->%d" % (len(rounds), mm0, int((cur != target).sum())))
        if not rounds:
            if _CURIOUS_ON():
                if floor is not None and mm0 > int(floor):
                    # THE FLOOR GATE (patch 25): a stall above the best this
                    # level has reached is not the edge of what he knows
                    # (L1: 4/4 no_gain lives cleared; L2: 22/29 improved)
                    self.floor_held += 1
                else:
                    _pk = self._probe(S, dist0, eff, n_actions, canvas, sel, mm0, confirmed,
                                      auto=auto, outs=outs, target=target)
                    if _pk is not None:
                        return _pk
            return self._none("no_gain")
        mm1 = int((cur != target).sum())
        if not (confirmed or mm1 == 0 or (mm0 - mm1) * 2 >= mm0):
            return self._none("gate")
        a, ck, Sreq, c, gain, cost = rounds[0]
        info = {"rounds": len(rounds), "steps": steps, "mm0": mm0, "mm1": mm1,
                "sel": sel, "state": len(S),
                "plan": [(int(r[0]), r[1], len(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in rounds]}
        # the first action of the plan, and what it predicts for the canvas
        pred = set(); pcol = None; _is_paint = False; _seq_hue = None; _seq_prehue = None; _hue_c = None; _hue_ck = None
        _sq = getattr(self, "_seq", None) if _la else None
        _sh = bool(_la and _sq is not None and _SEQHUE_ON())
        _at_here = bool(self.comparable(S, Sreq) and not (_EXACTFIRST_ON() and ambiguous(a, ck, S, Sreq)))
        _at_twin = bool(_sh and _sq.get("twin") is not None and _sq.get("twin") == S
                        and int(_sq.get("twin_i", -1)) == int(_sq["i"])
                        and not (_EXACTFIRST_ON() and ambiguous(a, ck, _sq.get("twin_from") or S, Sreq)))
        _pr = None
        if (_PLACEKEY_ON() and _SHORTCUT_ON() and _sh and place is not None
                and _sq.get("shortcut") is not None and not _at_here and not _at_twin):
            _pr = self._reach_place(place, (a, ck, Sreq), n_actions)
            if _pr is not None and _pr[1] is None:
                # PLACEKEY (41): the place says he is THERE -- this action has painted
                # these cells from here -- though the paint-sensitive quotient reads as
                # another state.  Go to the paint; walking away was the oscillation
                # (offline cd82 L3: place_nav d=1 at a4, a6, a8, a10, then stuck).
                _at_here = True
                self._event("place_here")
        _nav_first = bool(_sh and not _at_twin and not _sq.get("pre_hue")
                          and not _at_here)
        if _nav_first:
            # SEQHUE (29): a latched step NAVIGATES FIRST and clicks its colour
            # at the mask.  His own swatch click changes the indicator's state
            # half the time (cd82 L4: 20/41 plan clicks, 3/4 with a sequence
            # latched) and arrows from the recoloured state reached nothing
            # (21:38:34Z, replayed) while from here they reach it in 2.  Twin
            # evidence pools, as in the walk's BFS (27b).
            rr = None
            _pr_explore = bool(getattr(self, "_nav_explore", False))
            if _PLACEKEY_ON() and _SHORTCUT_ON() and _pr is not None:
                if _pr[1] is not None and not _pr_explore:
                    rr = (int(_pr[0]), int(_pr[1]), Sreq)
                    self._event("place_nav d=%d a=%d" % (int(_pr[0]), int(_pr[1])))
            if rr is None and _SHORTCUT_ON() and _sq.get("shortcut") is not None:
                # SHORTCUT (40): a latched shortcut NAVIGATES on the raw
                # automaton -- full states, the place of every part -- and
                # meets its mask in the quotient.  The quotient drops where
                # the stamp is: offline cd82, the modal successor of (S, a)
                # was another place than this board's (L3 150 vs 25, L4
                # 169+164 vs 3) and the latch oscillated until it died stuck
                rr = self._reach_raw(Sfull, Sreq, n_actions, a_ck=(a, ck),
                                     skip=((lambda X: ambiguous(a, ck, X, Sreq)) if _EXACTFIRST_ON() else None))
                if rr is not None:
                    self._event("shortcut_nav raw d=%d a=%d" % (int(rr[0]), int(rr[1])))
            if (rr is None and _PLACEKEY_ON() and _pr is not None and _pr[1] is not None
                    and bool(getattr(self, "_nav_explore", False))):
                # PLACEKEY (41): EXPLORE is the last resort before the quotient -- the raw
                # navigation won L4/L5/L0 live and keeps precedence (review of 41)
                rr = (int(_pr[0]), int(_pr[1]), Sreq)
                self._event("place_nav explore d=%d a=%d" % (int(_pr[0]), int(_pr[1])))
            if rr is None:
                rr = self._reach(self._dist(auto, S, n_actions, tw=True), Sreq, skip=((lambda X: ambiguous(a, ck, X, Sreq)) if _EXACTFIRST_ON() else None))
            if rr is None or rr[1] is None:
                self._drop_seq("unreachable")
                return self._none("unreachable")
            pick = (int(rr[1]), None)
            _sq["nav"] = int(_sq.get("nav", 0)) + 1
        elif c != sel:
            sck = colours.get(c)
            aim = self._find(outs, sck)
            if aim is None:
                return self._none("no_swatch_on_board")
            act = self._click_action(n_actions)
            if act is None:
                return self._none("no_click_index")
            pick = (act, aim)
            if _sh:
                _hue_c = int(c); _hue_ck = sck
                if _sq.get("pre_hue"):
                    _seq_prehue = int(_sq["i"])
                else:
                    _seq_hue = int(_sq["i"])
        elif not _at_twin and not _at_here:
            rr = self._reach(dist0, Sreq, skip=((lambda X: ambiguous(a, ck, X, Sreq)) if _EXACTFIRST_ON() else None))
            if rr is None or rr[1] is None:
                return self._none("unreachable")
            pick = (int(rr[1]), None)
        else:
            aim = self._find(outs, ck) if ck is not None else None
            if ck is not None and aim is None:
                return self._none("no_tile_on_board")
            pick = (int(a), aim)
            _is_paint = True
            cells = None
            # the list the latch was searched and priced on (review of 33: a
            # SPOILMEM-filtered step found no cells here, pred was empty and
            # a correct paint was dropped as a misprediction and burned)
            for (aa, cc_, Sr), (cl, n) in (_ops_all if _la else ops):
                if aa == a and cc_ == ck and Sr == Sreq:
                    cells = cl; break
            if cells:
                pred = set(i for i in cells if int(canvas.flat[i]) != c); pcol = c
        # the same choice EXECUTED in the same situation more than
        # STUCK_AFTER times this life is a wrong model, not a plan.  The
        # situation is the planner's own (state, canvas, selection): a
        # board hash carries the budget row wherever the row is unknown
        # and would change every step.
        sk = (S, canvas.tobytes(), sel, pick[0], pick[1])
        if self._stuck.get(sk, 0) >= STUCK_AFTER:
            if _la and getattr(self, "_seq", None) is not None:
                self._drop_seq("stuck")
            return self._none("stuck")
        self.why = ""
        if _seq_prehue is not None:
            # counted past the stuck guard (review of 29, K2)
            _sq["hue_i"] = int(_seq_prehue); self.seq_prehues += 1
            self._event("seq_prehue i=%d c=%d" % (_seq_prehue, int(c)))
        elif _seq_hue is not None:
            self.seq_hues += 1
            self._event("seq_hue i=%d c=%d" % (_seq_hue, int(c)))
        # a spoiling lookahead step is the plan's own doing: predicted and
        # scored, but never burned or remembered as a spoil (patch 28)
        _unjudged = bool(_la and rounds and int(rounds[0][4]) <= 0)
        if _la:
            self.lookahead_acts += 1
            info["lookahead"] = len(rounds)
        self._pending = {"key": sk, "a": int(pick[0]), "aim": pick[1], "pred": pred, "colour": pcol,
                         "op": ((a, ck, Sreq) if (pred and not _unjudged) else None), "S": S,
                         "seq": (int(self._seq["i"]) if (_la and _is_paint and getattr(self, "_seq", None) is not None) else None),
                         "seq_hue": _seq_hue, "seq_prehue": _seq_prehue, "hue_c": _hue_c, "hue_ck": _hue_ck}
        if _la and _sq is not None and _sq.get("shortcut") is not None:
            info["shortcut"] = int(_sq["shortcut"])
        elif getattr(self, "_sc_yield", False) and getattr(self, "_sc_life", None) is not None and _SHORTCUT_ON():
            info["shortcut"] = int(self._sc_life)   # SHORTCUT (40): the yielded attempt keeps precedence
        info["pred"] = len(pred)
        self.plans += 1
        return pick[0], pick[1], info

    def _colour_map(self, outs, sel):
        """colour -> click key for every colour he can select on this
        board: the swatches he has learned, every object of a learned
        swatch's SHAPE (a swatch is a kind of object), and the current
        selection (key None: nothing to click)."""
        colours = dict((c, sw[0]) for c, sw in self.swatch.items())
        # a swatch is a KIND of object: every object of that shape on the
        # board selects its own colour, clicked or not yet
        kinds = set(sw[0][1] for sw in self.swatch.values()
                    if sw[0] is not None and len(sw[0]) == 4)
        if kinds:
            for sg, col, b in outs:
                if sg in kinds and col not in colours:
                    colours[col] = (col, sg, 1, 1)
        if sel is not None:
            colours.setdefault(sel, None)
        return colours

    def _twins(self, X):
        """X and every state a swatch click has swapped it with (SAMEMASK)."""
        out = set([X])
        if _SAMEMASK_ON():
            try:
                for pr in self.same_mask:
                    if X in pr:
                        out |= set(pr)
            except Exception:
                pass
        return out

    def _known_closure(self, known):
        """RETIRED (27b): a colour pair is not an equivalence -- 73/78
        recorded pairs join states with different learned masks -- so a
        twin of a known state is NOT known.  Kept for the record."""
        return known

    def _null_known(self):
        """Mask states an absent-colour probe found empty at least twice
        (PROBEHUE): known to paint nothing."""
        try:
            return set(X for X, n in self.null_masks.items() if int(n) >= 2)
        except Exception:
            return set()

    def _probe(self, S, dist0, eff, n_actions, canvas, sel, mm0, confirmed=True, auto=None,
               outs=None, target=None):
        """CURIOUS: the nearest reachable mask state he has never applied
        from, not yet probed this life -- the next step toward it, or the
        apply itself when he is there.  None when there is nothing to
        learn that way.  Only on a CONFIRMED picture (a probe is a blind
        paint; the gate that keeps blind paints off an unconfirmed pair
        applies to it too).  Masks the planner already prices (a known
        apply effect comparable with the state) go last.

        THROUGH APPLIES (patch 25): when nothing arrow-reachable is left
        to try, the walk crosses the full automaton, a KNOWN apply on the
        way included (cd82 L3: every untried mask sits behind one), and
        is LATCHED -- served ahead of the rounds until the mask is
        probed, learned, lost or the walk runs out of steps."""
        try:
            if not confirmed:
                return None
            support = {}
            for k, e in eff.items():
                if k[1] is None:
                    support[int(k[0])] = support.get(int(k[0]), 0) + int(e[1])
            if not support:
                return None
            act = max(sorted(support), key=lambda a: support[a])
            known = set(k[2] for k in eff if k[1] is None and int(k[0]) == act)
            if _PROBEHUE_ON():
                known |= self._null_known()
            skip = set(self._probed) | set(getattr(self, "_walk_failed", ()))

            def _cands(dist):
                out = []
                for X, (d, first) in dist.items():
                    if not X or X in known or X in skip:
                        continue
                    priced = 1 if any(self.comparable(X, Sr) for Sr in known) else 0
                    out.append(((priced, d, sorted(X)), X, first, d))
                out.sort(key=lambda t: t[0])
                return out
            cands = _cands(dist0)
            if not cands and auto is not None:
                cands = _cands(self._dist(auto, S, n_actions, navpaint=False))
            # the best candidate whose first step the stuck guard allows
            # (25b: the stall board recurs byte-identically after every
            # repair, so one stuck first arrow must not end exploring)
            r = None
            for _, X, first, d in cands:
                r = self._probe_step(S, X, act, d, first, canvas, sel, mm0,
                                     outs=outs, target=target, n_actions=n_actions)
                if r is not None:
                    break
            if r is None:
                return None
            _hue = r[2].get("hue") is not None
            if int(d) > 0 or _hue:
                # a direct probe that first clicks a colour is latched too
                # (26): the apply follows the click
                self._walk = {"X": X, "act": int(act), "d": int(d), "n": 0,
                              "hue_tries": 1 if _hue else 0}
                self.walks += 1
                self._event("latch d=%d target=%d cands=%d" % (int(d), len(X), len(cands)))
            return r
        except Exception:
            return None

    def _probe_hue(self, canvas, sel, colours, outs, target, n_actions, w):
        """PROBEHUE (26): the swatch click that makes the coming probe paint
        a colour the canvas does not hold, as (click action, aim, colour);
        None when the selection is already absent from the canvas, no such
        swatch is on the board, or this latch has spent its tries.
        MEASURED: 52/52 no-change L3 applies painted a colour already on
        the canvas (median 55% of cells) while a swatch of an absent
        colour sat on the board every time."""
        try:
            if w is not None and int(w.get("hue_tries", 0)) >= STUCK_AFTER:
                return None
            hist = {}
            for v in canvas.flat:
                hist[int(v)] = hist.get(int(v), 0) + 1
            if sel is not None and hist.get(int(sel), 0) == 0:
                return None
            act = self._click_action(n_actions)
            if act is None:
                return None
            absent = [c for c, ck in colours.items()
                      if ck is not None and hist.get(int(c), 0) == 0 and c != sel]
            if not absent:
                return None
            need = {}
            if target is not None:
                for v in target.flat:
                    need[int(v)] = need.get(int(v), 0) + 1
            absent.sort(key=lambda c: (-need.get(int(c), 0), int(c)))
            for c in absent:
                aim = self._find(outs, colours[c])
                if aim is not None:
                    return (int(act), aim, int(c), len(absent))
            return None
        except Exception:
            return None

    def _probe_step(self, S, X, act, d, first, canvas, sel, mm0,
                    outs=None, target=None, n_actions=None, w=None):
        """One step of a probe toward mask X: the apply itself at X (final),
        the swatch click that first makes it paint a colour the canvas
        lacks (PROBEHUE), or the next transition toward it -- an arrow, or
        a known apply on the way.  None when the stuck guard refuses it.
        Counted only past that guard."""
        final = (int(d) == 0 or first is None)
        hue = None
        if final and _PROBEHUE_ON() and outs is not None:
            hue = self._probe_hue(canvas, sel, self._colour_map(outs, sel), outs, target,
                                  n_actions, w)
        if hue is not None:
            pick = (int(hue[0]), (int(hue[1][0]), int(hue[1][1])))
        else:
            pick = (int(act), None) if final else (int(first), None)
        sk = (S, canvas.tobytes(), sel, pick[0], pick[1])
        if self._stuck.get(sk, 0) >= STUCK_AFTER:
            return None
        sel_absent = False
        if hue is not None:
            self.probe_hue_clicks += 1
            if w is not None:
                w["hue_tries"] = int(w.get("hue_tries", 0)) + 1
            self._event("hue c=%d absent=%d" % (hue[2], hue[3]))
        elif final:
            self.probe_applies += 1
            try:
                sel_absent = bool(sel is not None and not (canvas == int(sel)).any())
            except Exception:
                sel_absent = False
        elif pick[0] in self._paint_actions():
            self.probe_paint_applies += 1
        self.why = ""
        self._pending = {"key": sk, "a": int(pick[0]), "aim": pick[1], "pred": set(),
                         "colour": None, "op": None, "S": S, "probe": True,
                         "walk": X, "final": bool(final and hue is None),
                         "sel_absent": bool(sel_absent),
                         "hue": (hue[2] if hue is not None else None)}
        self.probes += 1
        info = {"rounds": 0, "steps": int(d) + 1 + (1 if hue is not None else 0), "mm0": mm0,
                "mm1": mm0, "sel": sel, "state": len(S), "plan": [], "probe": True,
                "target": len(X), "dist": int(d), "final": bool(final and hue is None),
                "hue": (hue[2] if hue is not None else None)}
        return pick[0], pick[1], info

    def _serve_walk(self, S, eff, auto, n_actions, canvas, sel, mm0, outs=None, target=None):
        """The next step of the latched walk, or None once the latch is
        dropped: the mask was probed or learned meanwhile (no loss), it
        is no longer reachable, the walk is over budget, or the stuck
        guard refused the step (a loss: the target is not retried this
        life).  Arrows first; the full automaton only when they do not
        reach."""
        w = self._walk
        try:
            X = w["X"]; act = int(w["act"])
            known = set(k[2] for k in eff if k[1] is None and int(k[0]) == act)
            if _PROBEHUE_ON():
                known |= self._null_known()
            if X in self._probed or X in known:
                self._walk = None
                return None
            if int(w.get("n", 0)) >= MAX_WALK_STEPS:
                return self._drop_walk("budget")
            if S != X and w.get("twin") is not None and S == w.get("twin"):
                # ARRIVED (27b): this walk's own hue click recoloured the
                # icon and the planner reads it as another state -- the
                # same mask (a HISTORICAL pair is not trusted: 73/78 pairs
                # join different masks that look alike in one colour)
                r = self._probe_step(S, X, act, 0, None, canvas, sel, mm0,
                                     outs=outs, target=target, n_actions=n_actions, w=w)
                if r is None:
                    return self._drop_walk("stuck")
                return r
            dist = self._dist(auto, S, n_actions)
            if X not in dist:
                dist = self._dist(auto, S, n_actions, navpaint=False)
            if X not in dist:
                # transient (26b): a route or another organ moved him; the
                # target stays eligible for a new latch when it is reachable
                return self._drop_walk("unreachable", fail=False)
            d, first = dist[X]
            r = self._probe_step(S, X, act, d, first, canvas, sel, mm0,
                                 outs=outs, target=target, n_actions=n_actions, w=w)
            if r is None:
                return self._drop_walk("stuck")
            return r
        except Exception:
            self._walk = None
            return None

    def _drop_walk(self, why="?", fail=True):
        w = self._walk
        self._walk = None
        if w is not None:
            if fail:
                try:
                    self._walk_failed.add(w["X"])
                except Exception:
                    pass
            self.walks_dropped += 1
            self.walks_dropped_why[why] = self.walks_dropped_why.get(why, 0) + 1
            self._event("drop why=%s n=%d d=%d" % (why, int(w.get("n", 0)), int(w.get("d", 0))))
        return None

    def _event(self, msg):
        """A walk event for the journal (drained by the world)."""
        try:
            ev = self.events
            if len(ev) < 64:
                ev.append(str(msg))
        except Exception:
            pass

    def _lookahead(self, ops, colours, canvas, target, S, sel, auto, n_actions, dist0, ambiguous):
        """A sequence of up to LA_DEPTH (effect, colour) paints whose FINAL
        picture beats the present one, as rounds [(a, ck, Sreq, c, gain,
        cost)], the final canvas and the summed cost -- or None.  Depth 1
        and 2 exhaustive, depth 3 from the LA_BEAM best depth-2 partials.
        Each step's state must be reachable: the first by arrows from
        here; each next by arrows from the predicted post-apply state of
        the step before, or over the full automaton when that prediction
        is unknown.  Cached per (canvas, state, selection, table)."""
        try:
            key = (canvas.tobytes(), S, sel, len(self.effects), self._auto_n)
            if _REFUTE_ON():
                # a burned effect leaves the candidates: the search must re-run
                key = key + (len(self._burned),)
            if key in self._la_cache:
                return self._la_cache[key]
            if len(self._la_cache) >= 16:
                self._la_cache.clear()
            self.lookaheads += 1
            mm0 = int((canvas != target).sum())
            cols = [c for c in colours]
            cand = []
            for (a, ck, Sreq), (cells, n) in ops:
                if not cells:
                    continue
                cand.append((int(n), a, ck, Sreq, np.fromiter(cells, dtype=np.int64)))
            cand.sort(key=lambda t: -t[0])
            _ncol = len(cols)
            if _LAFULL_ON():
                _ncol = len([c for c in cols if c in set(int(v) for v in np.unique(target))])
            while cand and len(cand) * max(1, _ncol) > LA_MAX_PAIRS:
                cand.pop()
            if not cand or not cols:
                self._la_cache[key] = None
                return None

            def paint(cv, idx, c):
                new = cv.copy(); new.flat[idx] = c
                return new, int((new != target).sum())
            def expand(layer):
                # every (effect, colour) after every partial; a step that
                # changes nothing is never taken; one partial per resulting
                # canvas (28b: 16 no-op pairs filled the whole beam)
                out = {}
                for m0_, seq0, cv0 in layer:
                    for n, a, ck, Sreq, idx in cand:
                        for c in cols:
                            new, m = paint(cv0, idx, c)
                            kb = new.tobytes()
                            if kb == cv0.tobytes() or kb in out:
                                continue
                            out[kb] = (m, seq0 + [(a, ck, Sreq, c)], new)
                return list(out.values())
            if _LAFULL_ON():
                # FULL SEARCH (patch 33): distinct masks x TARGET colours,
                # canvases deduplicated across the search.  A paint no cell of
                # whose mask wants its colour is repainted in full by any
                # solution that uses it, so it is never taken.  Every entry of
                # a mask is a candidate for its step: the pricing below takes
                # the first reachable, unambiguous one.
                _tcols = set(int(v) for v in np.unique(target))
                _tc = [c for c in cols if c in _tcols]
                _tflat = target.ravel()
                # the search does not depend on where he stands: cached per
                # (canvas, table, burned, colours) so a stalled board is
                # PRICED, not re-searched, on every arrow step (_auto_n
                # grows per step and would miss _la_cache each time)
                _fkey = (canvas.tobytes(), len(self.effects), len(self._burned), tuple(sorted(_tc)))
                _lf = getattr(self, "_la_found", None)
                if _lf is None:
                    _lf = self._la_found = {}
                if _fkey in _lf:
                    found = _lf[_fkey]
                else:
                  self.la_full += 1
                  _masks = {}
                  for n_, a_, ck_, Sreq_, idx_ in cand:
                      _masks.setdefault(idx_.tobytes(), (idx_, []))[1].append((n_, a_, ck_, Sreq_))
                  _ops = []
                  for _k, (idx_, ents) in _masks.items():
                      ents.sort(key=lambda t: -t[0])
                      for c in _tc:
                          if bool((_tflat[idx_] == c).any()):
                              _ops.append((idx_, c, ents))
                  _tb = target.tobytes()
                  _seen = {canvas.tobytes(): None}
                  _layer = [canvas]; _final = []; _sols = []; _solved = False; _capped = False
                  _t0 = _time.time()
                  for _d in range(int(LA_DEPTH_FULL)):
                      _next_layer = []
                      for cv0 in _layer:
                          for idx_, c, ents in _ops:
                              new = cv0.copy(); new.flat[idx_] = c
                              kb = new.tobytes()
                              if kb in _seen:
                                  if kb == _tb:
                                      # another way to the picture: kept, so a
                                      # burned or unreachable first step does
                                      # not hide the rest (review of 33)
                                      _sols.append((cv0.tobytes(), ents, c))
                                  continue
                              _seen[kb] = (cv0.tobytes(), ents, c)
                              m = int((new != target).sum())
                              _final.append((m, kb, new)); _next_layer.append(new)
                              if m == 0:
                                  _solved = True
                              if len(_seen) > LA_MAX_CANVASES or                                  ((len(_seen) & 1023) == 0 and _time.time() - _t0 > LA_TIME_FULL):
                                  _capped = True; break
                          if _capped:
                              break
                      _layer = _next_layer
                      if _solved or _capped:
                          break
                  if _capped:
                      self.la_full_capped += 1
                  def _unwind(kb):
                      out = []
                      while _seen[kb] is not None:
                          pk, ents, c = _unwind_step = _seen[kb]
                          out.append((ents, c)); kb = pk
                      return out[::-1]
                  found = [(m, _unwind(kb), cv) for m, kb, cv in _final if m < mm0]
                  for pk, ents, c in _sols:
                      found.append((0, _unwind(pk) + [(ents, c)], target.copy()))
                  if len(_lf) >= 8:
                      _lf.clear()
                  _lf[_fkey] = found
                  self._event("la_full canvases=%d ops=%d best=%d alt=%d %.2fs" % (len(_seen), len(_ops), min([t[0] for t in _final] or [mm0]), len(_sols), _time.time() - _t0))
            else:
                L1 = expand([(mm0, [], canvas)])
                L2 = expand(L1) if LA_DEPTH >= 2 else []
                L3 = []
                if LA_DEPTH >= 3 and L2:
                    L2.sort(key=lambda t: t[0])
                    L3 = expand(L2[:LA_BEAM])
                found = [(m, [([(0, a, ck, Sreq)], c) for (a, ck, Sreq, c) in seq], cv)
                         for m, seq, cv in L1 + L2 + L3 if m < mm0]
            found.sort(key=lambda t: (t[0], len(t[1])))
            _dists = {}

            def dist_from(X, navpaint=True):
                k = (X, navpaint)
                if k not in _dists:
                    _dists[k] = self._dist(auto, X, n_actions, navpaint=navpaint)
                return _dists[k]
            for m, seq, cv in found[:64]:
                rounds = []; steps = 0; cur = canvas; cur_sel = sel; ok = True
                d_here = dist0
                for i, (ents, c) in enumerate(seq):
                    dd = None
                    for (n_, a, ck, Sreq) in ents:
                        dd = self._reach(d_here, Sreq, skip=((lambda X: ambiguous(a, ck, X, Sreq)) if _EXACTFIRST_ON() else None))
                        if dd is not None and not ambiguous(a, ck, dd[2], Sreq):
                            break
                        dd = None
                    if dd is None:
                        ok = False
                        break
                    idx = next(t[4] for t in cand if t[1] == a and t[2] == ck and t[3] == Sreq)
                    new, mnew = paint(cur, idx, c)
                    gain = int((cur != target).sum()) - mnew
                    cost = dd[0] + (1 if c != cur_sel else 0) + 1
                    rounds.append((a, ck, Sreq, c, gain, cost))
                    steps += cost; cur = new; cur_sel = c
                    # where will he stand after this paint?  the apply's own
                    # transition, else the full automaton from Sreq
                    if ck is not None:
                        d_here = dist_from(Sreq)        # a click leaves the indicator
                    else:
                        post = self._next(auto, Sreq, a)
                        d_here = dist_from(post) if post is not None else dist_from(Sreq, navpaint=False)
                if ok:
                    self.lookahead_hits += 1
                    self._event("lookahead depth=%d mm=%d->%d" % (len(rounds), mm0, m))
                    res = (rounds, cur, steps)
                    self._la_cache[key] = res
                    return res
            self._la_cache[key] = None
            return None
        except Exception:
            return None

    def _serve_seq(self, ops, canvas, target, S, sel, auto, n_actions, dist0, ambiguous):
        """The latched sequence's remaining steps, re-priced from here:
        rounds, the predicted final canvas and the summed cost -- or None
        after dropping the latch (a step's state no longer reachable, or
        nothing left)."""
        sq = self._seq
        try:
            steps_left = sq["steps"][int(sq["i"]):]
            if not steps_left:
                self._seq = None
                return None
            _exp = sq.get("canvases", [None] * 8)[int(sq["i"])]
            if _exp is not None and _exp.tobytes() != canvas.tobytes():
                return self._drop_seq("diverged")
            cells_of = {}
            for (a, ck, Sreq), (cells, n) in ops:
                cells_of[(a, ck, Sreq)] = np.fromiter(cells, dtype=np.int64)
            _dists = {}
            _sh = _SEQHUE_ON()
            sq["pre_hue"] = False

            def dist_from(X, navpaint=True):
                k = (X, navpaint)
                if k not in _dists:
                    # SEQHUE (29): a colour variant of a known state shares its
                    # arrows (the walk's BFS, 27b)
                    _dists[k] = self._dist(auto, X, n_actions, navpaint=navpaint,
                                           tw=(True if _sh else None))
                return _dists[k]
            rounds = []; total = 0; cur = canvas; cur_sel = sel
            d_here = dist_from(S) if _sh else dist0
            for k_, (a, ck, Sreq, c) in enumerate(steps_left):
                idx = cells_of.get((a, ck, Sreq))
                if idx is None:
                    return self._drop_seq("effect_gone")
                dd = None
                if k_ == 0 and _sh and sq.get("twin") is not None and sq.get("twin") == S \
                        and int(sq.get("twin_i", -1)) == int(sq["i"]) \
                        and not ambiguous(a, ck, sq.get("twin_from") or S, Sreq):
                    # AT THE MASK, in the colour this step's own click chose:
                    # the recoloured icon is another q-state (cd82 L4: the
                    # state collapsed from two parts to one, 21:38:34Z).  The
                    # "effect knows more than the state" guard still applies
                    # to the state he clicked from (review of 29, K3)
                    dd = (0, None, S)
                if dd is None:
                    dd = self._reach(d_here, Sreq, skip=((lambda X: ambiguous(a, ck, X, Sreq)) if _EXACTFIRST_ON() else None))
                    if dd is not None and ambiguous(a, ck, dd[2], Sreq):
                        dd = None
                if k_ == 0 and _sh and int(sq.get("nav", 0)) > MAX_WALK_STEPS:
                    # a latch is not a licence to walk out the level clock
                    # (review of 29, K2): the walk's budget applies
                    return self._drop_seq("budget")
                if dd is None:
                    if k_ == 0 and _sh and c != cur_sel and int(sq.get("hue_i", -1)) != int(sq["i"]):
                        # NO ARROW PATH FROM HERE: the icon may be a colour
                        # variant the automaton never saw; the step's own colour
                        # click can bring it back.  One try per step.
                        dd = (1, None, S); sq["pre_hue"] = True
                    else:
                        return self._drop_seq("unreachable")
                new = cur.copy(); new.flat[idx] = c
                gain = int((cur != target).sum()) - int((new != target).sum())
                cost = dd[0] + (1 if c != cur_sel else 0) + 1
                rounds.append((a, ck, Sreq, c, gain, cost))
                total += cost; cur = new; cur_sel = c
                if ck is not None:
                    d_here = dist_from(Sreq)
                else:
                    post = self._next(auto, Sreq, a)
                    d_here = dist_from(post) if post is not None else dist_from(Sreq, navpaint=False)
            return (rounds, cur, total)
        except Exception:
            self._seq = None
            return None

    def shortcut_outcome(self, cleared, actions=None):
        """SHORTCUT (40): the world's verdict on an attempt that latched a
        shortcut -- the level cleared in `actions` realised actions, or the
        attempt ended without a clear.  Fewer actions than the route it was
        judged against = it worked (the world seals the shorter route);
        anything else refutes that route length, persisted, so it is never
        retried against the same route (doctrine audit of 40: a clear that
        is not shorter was neither sealed nor refuted and would have fired on
        every attempt).  Returns the journal line, or None."""
        rl = getattr(self, "_sc_life", None)
        self._sc_life = None
        if rl is None:
            return None
        if cleared and actions is not None and int(actions) < int(rl):
            self.shortcut_won += 1
            return "shortcut_won actions=%d route=%d" % (int(actions), int(rl))
        if getattr(self, "_sc_refuted", None) != int(rl):
            self._sc_retries = 0
        self._sc_refuted = int(rl)
        self._sc_refuted_places = self._places_known()   # SHORTCUT/PLACEKEY: the evidence then
        self.shortcut_refuted += 1
        return "shortcut_refuted cleared=%d actions=%s route=%d" % (int(bool(cleared)), actions, int(rl))

    def _drop_seq(self, why="?"):
        sq = getattr(self, "_seq", None)
        if sq is not None:
            self.lookahead_dropped += 1
            if sq.get("shortcut") is not None:
                # SHORTCUT (40): off the won route for the rest of the attempt;
                # the verdict is the world's, at the attempt's end
                self._sc_off = True
                self.shortcut_dropped += 1
                self._event("shortcut_drop why=%s i=%d route=%d" % (why, int(sq.get("i", 0)), int(sq["shortcut"])))
            self._event("seq_drop why=%s i=%d n=%d" % (why, int(sq.get("i", 0)), len(sq.get("steps", ()))))
            if _REFUTE_ON() and why in ("misprediction", "stuck"):
                # REFUTE (30): the step's effect did not do what the table said
                # here (or its pick was refused STUCK_AFTER times) -- the same
                # sequence was latched three times frame for frame, then found
                # again by ~20 fresh searches after the stuck drop (cd82 L4,
                # 75/100 steps).  A spoiling step carries op=None and is never
                # judged-burned; burning it here takes it out of the candidates.
                try:
                    _i = int(sq.get("i", 0))
                    _st = sq["steps"][_i]
                    _op = (_st[0], _st[1], _st[2])
                    if _op not in self._burned:
                        self._burned.add(_op)
                    self.seq_refuted += 1
                    self._event("seq_refute i=%d why=%s" % (_i, why))
                except Exception:
                    pass
        self._seq = None
        return None

    def _click_action(self, n_actions):
        """The coordinate action's index, as observed."""
        ci = getattr(self, "click_index", None)
        if ci is None or not (0 <= int(ci) < int(n_actions)):
            return None
        return int(ci)

    # ------------------------------------------------------------- persistence
    @staticmethod
    def _S_str(S):
        return ";".join(sorted("%s@%d,%d,%d,%d" % t if len(t) == 5 else "%s@%d,%d" % t for t in S))

    @staticmethod
    def _S_parse(s):
        out = []
        for tok in s.split(";"):
            if not tok:
                continue
            sig, _, pos = tok.partition("@")
            nums = [int(v) for v in pos.split(",") if v != ""] if pos else []
            if len(nums) == 4:
                out.append((sig, nums[0], nums[1], nums[2], nums[3]))
            elif len(nums) == 2:
                out.append((sig, nums[0], nums[1]))
            else:
                out.append((sig, 0, 0, 0, 0))
        return frozenset(out)

    @staticmethod
    def _ck_str(ck):
        if ck is None:
            return ""
        return "|".join(str(x) for x in ck)

    @staticmethod
    def _ck_parse(s):
        if not s:
            return None
        parts = s.split("|")
        if parts == ["bg"]:
            return ("bg",)
        try:
            return (int(parts[0]), parts[1], int(parts[2]), int(parts[3]))
        except (ValueError, IndexError):
            return None

    def to_dict(self):
        out = {
            "v": 2,
            "mutable": sorted(self.mutable),
            "seen": dict((s, int(v)) for s, v in self.seen.items() if s in self.mutable),
            "change": dict((s, int(v)) for s, v in self.change.items()),
            "change_c": dict((s, int(v)) for s, v in self.change_c.items()),
            "cochange": dict((k, int(v)) for k, v in self.cochange.items()),
            "effects": [[int(a), self._ck_str(ck), self._S_str(S), sorted(e[0]), int(e[1])]
                        for (a, ck, S), e in self.effects.items()],
            "auto": [[self._S_str(S), int(a), [[self._S_str(T), int(n)] for T, n in d.items()]]
                     for (S, a), d in self.auto.items()],
            "carrier": dict(self.carrier),
            "hues": dict((s, sorted(int(c) for c in v)) for s, v in self.hues.items()),
            "spoiled": [[self._S_str(Sp_), int(a), self._ck_str(ck), self._S_str(S), int(v[0]), int(v[1])]
                        for (Sp_, (a, ck, S)), v in self.spoiled.items() if Sp_ is not None],
            "worked": [[self._S_str(Sp_), int(a), self._ck_str(ck), self._S_str(S), int(v)]
                       for (Sp_, (a, ck, S)), v in self.worked.items() if Sp_ is not None],
            "swatch": [[int(c), self._ck_str(sw[0]), int(sw[1])] for c, sw in self.swatch.items()],
            "sel": self.sel,
            "last_painted": self.last_painted,
            "click_index": getattr(self, "click_index", None),
            "pic_shape": (list(self.pic_shape) if self.pic_shape is not None else None),
            "null_masks": [[self._S_str(X), int(n)] for X, n in getattr(self, "null_masks", {}).items()],
            "same_mask": [sorted(self._S_str(X) for X in pr) for pr in getattr(self, "same_mask", ())],
        }
        if getattr(self, "place_auto", None):
            out["place_auto"] = [[str(P), int(a), [[str(T), int(n)] for T, n in d.items()]]
                                 for (P, a), d in self.place_auto.items()]
            out["place_of"] = [[int(k[0]), self._ck_str(k[1]), self._S_str(k[2]),
                                [[str(P), int(n)] for P, n in d.items()]]
                               for k, d in self.place_of.items()]
            out["place_paint"] = [[str(P), int(a), self._ck_str(ck), [[sorted(m), int(n)] for m, n in d.items()]]
                                  for (P, a, ck), d in getattr(self, "place_paint", {}).items()]
        if getattr(self, "_sc_refuted", None) is not None:
            out["sc_refuted"] = int(self._sc_refuted)   # SHORTCUT (40): an observation, kept
            if getattr(self, "_sc_refuted_places", None) is not None:
                out["sc_refuted_places"] = int(self._sc_refuted_places)
            out["sc_retries"] = int(getattr(self, "_sc_retries", 0))
        return out

    def from_dict(self, d):
        n = 0
        if int((d or {}).get("v") or 0) != 2:
            return 0          # an older, shape-only table: dropped, the seed re-teaches
        try:
            for s in (d or {}).get("mutable") or []:
                self.mutable.add(str(s)); n += 1
            for s, v in ((d or {}).get("seen") or {}).items():
                self.seen[str(s)] = max(self.seen.get(str(s), 0), int(v))
            for s, v in ((d or {}).get("change") or {}).items():
                self.change[str(s)] = max(self.change.get(str(s), 0), int(v))
            for s, v in ((d or {}).get("change_c") or {}).items():
                self.change_c[str(s)] = max(self.change_c.get(str(s), 0), int(v))
            for k, v in ((d or {}).get("cochange") or {}).items():
                self.cochange[str(k)] = max(self.cochange.get(str(k), 0), int(v))
            # IDEMPOTENT: loading the same blob twice changes nothing
            # (counts take the max, never the sum)
            for a, ck, S, cells, cnt in (d or {}).get("effects") or []:
                key = (int(a), self._ck_parse(ck), self._S_parse(S))
                e = self.effects.setdefault(key, [set(), 0])
                e[0] |= set(int(i) for i in cells); e[1] = max(e[1], int(cnt)); n += 1
            for S, a, outs in (d or {}).get("auto") or []:
                dd = self.auto.setdefault((self._S_parse(S), int(a)), {})
                for T, cnt in outs:
                    T = self._S_parse(T)
                    dd[T] = max(dd.get(T, 0), int(cnt))
                n += 1
            self._auto_n += 1
            for s, v in ((d or {}).get("carrier") or {}).items():
                self.carrier[str(s)] = max(self.carrier.get(str(s), 0), int(v))
            for s, v in ((d or {}).get("hues") or {}).items():
                self.hues.setdefault(str(s), set()).update(int(c) for c in (v or []))
            for Sp_, a, ck, S, n_, cells_ in (d or {}).get("spoiled") or []:
                key = (self._S_parse(Sp_), (int(a), self._ck_parse(ck), self._S_parse(S)))
                old = self.spoiled.get(key)
                if old is None:
                    self.spoiled[key] = [int(n_), int(cells_)]
                else:
                    old[0] = max(old[0], int(n_)); old[1] = max(old[1], int(cells_))
            for Sp_, a, ck, S, n_ in (d or {}).get("worked") or []:
                key = (self._S_parse(Sp_), (int(a), self._ck_parse(ck), self._S_parse(S)))
                self.worked[key] = max(self.worked.get(key, 0), int(n_))
            for c, ck, v in (d or {}).get("swatch") or []:
                cku = self._ck_parse(ck)
                if cku is None or cku == ("bg",) or int(cku[0]) != int(c):
                    continue     # a frame, not a swatch: names no colour
                old = self.swatch.get(int(c))
                self.swatch[int(c)] = [cku, max(int(v), old[1] if old else 0)]; n += 1
            if (d or {}).get("sel") is not None:
                self.sel = int(d["sel"])
            if (d or {}).get("last_painted") is not None:
                self.last_painted = int(d["last_painted"])
            if (d or {}).get("click_index") is not None:
                self.click_index = int(d["click_index"])
            if (d or {}).get("pic_shape") and self.pic_shape is None:
                self.pic_shape = (int(d["pic_shape"][0]), int(d["pic_shape"][1]))
            for X, n_ in (d or {}).get("null_masks") or []:
                key = self._S_parse(X)
                if key:
                    self.null_masks[key] = max(self.null_masks.get(key, 0), int(n_))
            for pr in (d or {}).get("same_mask") or []:
                if len(pr) == 2 and len(self.same_mask) < MAX_STATES:
                    a_, b_ = self._S_parse(pr[0]), self._S_parse(pr[1])
                    if a_ and b_ and a_ != b_:
                        self.same_mask.add(frozenset([a_, b_]))
            for P, a, outs in (d or {}).get("place_auto") or []:
                dd = self.place_auto.setdefault((str(P), int(a)), {})
                for T, n_ in outs:
                    dd[str(T)] = max(dd.get(str(T), 0), int(n_))
            for a, ck, S, ps in (d or {}).get("place_of") or []:
                dd = self.place_of.setdefault((int(a), self._ck_parse(ck), self._S_parse(S)), {})
                for P, n_ in ps:
                    dd[str(P)] = max(dd.get(str(P), 0), int(n_))
            for P, a, ck, ms in (d or {}).get("place_paint") or []:
                dd = self.place_paint.setdefault((str(P), int(a), self._ck_parse(ck)), {})
                for m, n_ in ms:
                    _m = frozenset(int(i) for i in m)
                    dd[_m] = max(dd.get(_m, 0), int(n_))
            if (d or {}).get("sc_refuted") is not None:
                self._sc_refuted = int(d["sc_refuted"])
            if (d or {}).get("sc_refuted_places") is not None:
                self._sc_refuted_places = int(d["sc_refuted_places"])
            if (d or {}).get("sc_retries") is not None:
                self._sc_retries = int(d["sc_retries"])
        except Exception:
            return n
        return n
