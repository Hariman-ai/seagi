"""THE GOAL IS ON THE BOARD -- a relation sense (2026-09-05).

Measured before this was written (see memory p_the_goal_is_on_the_board):
in every ARC game he plays that could be read, the goal and a cell-exact
progress number are drawn on the board on every frame, in one of three
shapes -- make region A look like region B; put things in the order
shown elsewhere; bring an object to a marked place.  His state is a 3x3
glance and his memories are keyed on whole boards, so a relation between
two regions is invisible to him by construction.  On cd82 level 2 he
reaches 57 wrong cells of 100 and paints over them; on tu93, a small
maze with the goal cell drawn, he moves one cell in 50 steps.

This module gives him two of the three shapes, discovered from his own
frames with no per-game knowledge:

  EQ   two rectangles of the same shape, one of which never changes (the
       TARGET) while the other changes under his actions (the CANVAS).
       value = cells that differ.
  NEAR the object he controls and a static object class with few
       instances (a MARKER).  value = Chebyshev gap between bounding
       boxes to the nearest instance.  The controlled object is detected
       HERE, as the class whose single instance TRANSLATES most under
       his actions (same size and box, new position).  His older self
       organ is not consulted: adversarial review found it names the
       cd82 CANVAS as him (colour 0, size 84, three "moves" -- a blob
       being painted in place), which would have excluded the canvas as
       his body and confirmed a spurious marker at every clear.

Every step reports the largest fractional improvement of any candidate
relation (progress) and the largest worsening (regress), each relative
to the relation's value at the start of the life.  A level clear
CONFIRMS the candidate that was nearest its goal on the board before the
winning move, and from then on only confirmed relations count on that
level -- that is how the information from a right decision is kept.

A clear confirms only a relation at its LIFE MINIMUM on the pre-win
board, and a confirmation is REVOKED when a later clear finds it away
from its minimum, or when it sits satisfied (value 0) for longer than
its patience without a clear.

Constants: none that tune behaviour.  The few numbers below are shape
limits (a region must be at least 3x3; a marker class has at most 8
instances -- re86 draws 8 targets, a textured maze has 20+ cells; a
region counts as static after 30 unchanged observations; a controlled
object needs 3 translations) and are stated once, here.
"""
import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:            # pragma: no cover - scipy is present on the VPS
    _ndi = None

W = 64
MIN_AREA = 9                 # a relation needs at least a 3x3 region
MAX_AREA_FRAC = 0.40         # walls, panels and backgrounds are not regions
MAX_MARKER_INSTANCES = 8     # a marker class: few instances, not a texture
STATIC_AFTER = 30            # unchanged observations before "never changes"
MAX_REGIONS = 80
CTRL_AFTER = 3               # translations before a class is "the object he controls"


def segment(G, exclude_line=None):
    """Connected same-colour components, background excluded.

    Returns (bg, objects); each object is (colour, size, r0, c0, r1, c1).
    `exclude_line`: a row (0-63) or column (64-127) to ignore -- the
    budget line, when he has learned it."""
    if G is None or _ndi is None:
        return None, []
    try:
        G = np.asarray(G, dtype=np.int16)
    except (TypeError, ValueError):
        return None, []
    if G.shape != (W, W):
        return None, []
    mask = np.ones((W, W), dtype=bool)
    if exclude_line is not None:
        if 0 <= exclude_line < W:
            mask[exclude_line, :] = False
        elif W <= exclude_line < 2 * W:
            mask[:, exclude_line - W] = False
    vals, cts = np.unique(G[mask], return_counts=True)
    if len(vals) == 0:
        return None, []
    bg = int(vals[int(np.argmax(cts))])
    out = []
    for v in vals:
        v = int(v)
        if v == bg:
            continue
        lab, n = _ndi.label((G == v) & mask)
        if n <= 0 or n > 120:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        for k, sl in enumerate(_ndi.find_objects(lab)):
            if sl is None:
                continue
            out.append((v, int(sizes[k]), sl[0].start, sl[1].start,
                        sl[0].stop - 1, sl[1].stop - 1))
        if len(out) > 200:
            return bg, []
    return bg, out


def _gap(A, B):
    r0, c0, r1, c1 = A
    s0, d0, s1, d1 = B
    return max(max(0, max(r0, s0) - min(r1, s1)),
               max(0, max(c0, d0) - min(c1, d1)))


def propose_regions(G, bg, objs):
    """Candidate rectangles: object boxes; the interior of a hollow
    object (a frame); and the box of the CONTENT inside that interior
    (a picture inside a frame is smaller than the frame)."""
    G = np.asarray(G)
    regs = set()
    lim = MAX_AREA_FRAC * W * W
    for col, size, r0, c0, r1, c1 in objs:
        h, w = r1 - r0 + 1, c1 - c0 + 1
        area = h * w
        if MIN_AREA <= area <= lim:
            regs.add((r0, c0, r1, c1))
        if size < area and h >= 5 and w >= 5:
            ir = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
            if (ir[2] - ir[0] + 1) * (ir[3] - ir[1] + 1) >= MIN_AREA:
                regs.add(ir)
                sub = G[ir[0]:ir[2] + 1, ir[1]:ir[3] + 1]
                # the picture inside a frame: neither background nor
                # the frame's own colour (a two-cell border reaches
                # inside the interior box)
                nz = np.argwhere((sub != bg) & (sub != col))
                if len(nz):
                    a0, b0 = nz.min(0); a1, b1 = nz.max(0)
                    cr = (ir[0] + int(a0), ir[1] + int(b0),
                          ir[0] + int(a1), ir[1] + int(b1))
                    if (cr[2] - cr[0] + 1) * (cr[3] - cr[1] + 1) >= MIN_AREA:
                        regs.add(cr)
    return regs


def _content(G, R):
    return np.asarray(G)[R[0]:R[2] + 1, R[1]:R[3] + 1].tobytes()


class RelSense(object):
    """One per (game, level).  Cumulative across lives and, through
    to_dict/from_dict, across restarts."""

    def __init__(self):
        self.regs = {}          # rect -> [ref bytes, changed, seen]
        self.markers = {}       # class (colour,size) -> instance boxes at life start
        self.confirmed = set()  # relation ids confirmed by a clear
        self.v0 = {}            # rel id -> value at life start
        self.last = {}          # rel id -> last value
        self.self_box = None
        self.self_prev = None
        self.lives = 0
        self.fires = 0
        self.revoked = 0
        self._pursued = None
        self._since = {}
        self._lifemin = {}       # rel id -> lowest value this life
        self._moved = set()      # rel ids whose value moved this life
        self._zero_since = 0     # steps the pursued CONFIRMED relation has sat at 0
        self.mobile = set()      # marker classes seen to move: never markers
        self.ctrl = None         # (colour, size) of the object he controls
        self.body = set()        # classes attached to it in the current frame
        self._moves = {}         # class -> translation events
        self._prev_objs = None

    # -------------------------------------------------- the object he controls
    def _detect_ctrl(self, objs, count=True):
        """A translation: a class with exactly one instance in both
        frames whose box keeps its size and dimensions and moves.  A blob
        painted in place (a canvas) changes size and keeps its box; a
        mask cycling shape changes dimensions; neither counts."""
        prev = self._prev_objs
        self._prev_objs = objs
        if not count or prev is None:
            return
        pa = {}; pb = {}
        for o in prev:
            pa.setdefault((o[0], o[1]), []).append(o)
        for o in objs:
            pb.setdefault((o[0], o[1]), []).append(o)
        for k, la in pa.items():
            lb = pb.get(k)
            if len(la) != 1 or not lb or len(lb) != 1:
                continue
            a, b = la[0], lb[0]
            if (a[2], a[3]) != (b[2], b[3]) and (a[4] - a[2], a[5] - a[3]) == (b[4] - b[2], b[5] - b[3]):
                self._moves[k] = self._moves.get(k, 0) + 1
        if self._moves:
            # the most-translated class; a tie (a ring and its centre
            # pixel move together) goes to the LARGEST, his whole body
            k, n = max(self._moves.items(), key=lambda kv: (kv[1], kv[0][1]))
            if n >= CTRL_AFTER:
                self.ctrl = k

    # ------------------------------------------------------------ perception
    def _self_from(self, objs, sig):
        """His body: the controlled object's box, grown to include every
        object attached to it (touching or overlapping) -- a ring around
        a centre pixel, a cap on a block.  Their passage is his passage."""
        if not sig or not objs:
            return None
        col, sz = int(sig[0]), int(sig[1])
        tol = max(1, sz // 4)
        cands = [o for o in objs if o[0] == col and abs(o[1] - sz) <= tol]
        if not cands:
            return None
        if self.self_prev is not None and len(cands) > 1:
            pr = ((self.self_prev[0] + self.self_prev[2]) / 2.0,
                  (self.self_prev[1] + self.self_prev[3]) / 2.0)
            cands.sort(key=lambda o: abs((o[2] + o[4]) / 2.0 - pr[0])
                       + abs((o[3] + o[5]) / 2.0 - pr[1]))
        o = cands[0]
        box = (o[2], o[3], o[4], o[5])
        self.body = {(o[0], o[1])}
        lim = MAX_AREA_FRAC * W * W
        for p in objs:
            if p is o or p[1] > lim:
                continue
            pb = (p[2], p[3], p[4], p[5])
            if _gap(box, pb) == 0:
                self.body.add((p[0], p[1]))
                box = (min(box[0], pb[0]), min(box[1], pb[1]),
                       max(box[2], pb[2]), max(box[3], pb[3]))
        return box

    def begin_life(self, G, bg, objs, sig=None, line=None):
        """Called on the first board of a life (a reset, or a new level).
        `sig` is accepted for API stability and ignored: the controlled
        object is detected here."""
        self.lives += 1
        self._detect_ctrl(objs, count=False)      # a reset teleports; do not count it
        self.self_prev = None
        self.self_box = self._self_from(objs, self.ctrl)
        for R in propose_regions(G, bg, objs):
            if R not in self.regs and len(self.regs) < MAX_REGIONS:
                over = self.self_box is not None and _gap(R, self.self_box) == 0
                # a region he stands on at the start gets its reference
                # content once he has left it
                self.regs[R] = [None if over else _content(G, R), False, 0]
        # markers: static classes with few instances, not him, not the line
        cls = {}
        for o in objs:
            cls.setdefault((o[0], o[1]), []).append((o[2], o[3], o[4], o[5]))
        lim = MAX_AREA_FRAC * W * W
        self.markers = {}
        for k, boxes in cls.items():
            if len(boxes) > MAX_MARKER_INSTANCES or k in self.mobile:
                continue
            if self.ctrl and k[0] == int(self.ctrl[0]) and abs(k[1] - int(self.ctrl[1])) <= max(1, int(self.ctrl[1]) // 4):
                continue
            if k in self.body:
                continue
            if k[1] > lim:
                continue
            if line is not None and any(
                    (0 <= line < W and b[0] == b[2] == line)
                    or (line >= W and b[1] == b[3] == line - W) for b in boxes):
                continue
            # a full-width row or full-height column is a line (a
            # budget or a border), not a place to go
            if any((b[0] == b[2] and b[3] - b[1] + 1 >= W)
                   or (b[1] == b[3] and b[2] - b[0] + 1 >= W) for b in boxes):
                continue
            self.markers[k] = boxes
        self.v0 = {}
        self.last = {}
        self._since = {}
        self._lifemin = {}
        self._moved = set()
        self._zero_since = 0
        vals = self.values(G)
        for rid, v in vals.items():
            if v > 0:
                self.v0[rid] = float(v)
                self.last[rid] = float(v)
                self._lifemin[rid] = float(v)
        self._pursued = None
        self._pursued = self._choose([r for r in vals if r in self.v0], vals)

    def _update_regs(self, G, boxes):
        """A region is judged only when he is NOT on it (before, during or
        after the step).  Passage restores a region's content once he
        has left, so a maze cell never becomes a canvas; paint stays, so
        a canvas he stamped while standing over it is dynamic the moment
        he moves off.  A region he stood on at life start takes its
        reference content once he has left it."""
        boxes = [b for b in boxes if b is not None]
        for R, rec in self.regs.items():
            rec[2] += 1
            over = any(_gap(R, b) == 0 for b in boxes)
            if rec[0] is None:
                if not over:
                    rec[0] = _content(G, R)
                continue
            if rec[1] or over:
                continue
            if _content(G, R) != rec[0]:
                rec[1] = True

    def _update_markers(self, objs):
        """A marker that moves is not a place -- it is part of him (the
        cap on wa30's avatar) or another moving thing (re86's second
        crosshair).  Dropped for this life and remembered as mobile."""
        if not self.markers:
            return
        now = {}
        for o in objs:
            now.setdefault((o[0], o[1]), []).append((o[2], o[3], o[4], o[5]))
        for k in list(self.markers):
            cur = now.get(k)
            if cur is None or sorted(cur) != sorted(self.markers[k]):
                if cur is not None and len(cur) == len(self.markers[k]):
                    self.mobile.add(k)
                    del self.markers[k]
                    self.v0.pop(("near", k), None)
                    self.last.pop(("near", k), None)

    def pairs(self):
        """EQ candidates: (dynamic, static) same shape, disjoint, and the
        dynamic one is not where he stands."""
        dyn = [R for R, rec in self.regs.items() if rec[1]]
        sta = [R for R, rec in self.regs.items()
               if not rec[1] and rec[0] is not None and rec[2] >= STATIC_AFTER]
        out = []
        for A in dyn:
            sh = (A[2] - A[0], A[3] - A[1])
            for B in sta:
                if (B[2] - B[0], B[3] - B[1]) == sh and _gap(A, B) > 0:
                    out.append((A, B))
        return out

    def values(self, G):
        """Current value of every candidate relation (0 = goal reached)."""
        G = np.asarray(G)
        out = {}
        for A, B in self.pairs():
            a = G[A[0]:A[2] + 1, A[1]:A[3] + 1]
            b = G[B[0]:B[2] + 1, B[1]:B[3] + 1]
            out[("eq", A, B)] = int((a != b).sum())
        if self.self_box is not None:
            for k, boxes in self.markers.items():
                out[("near", k)] = min(_gap(self.self_box, b) for b in boxes)
        return out

    # --------------------------------------------------------------- the step
    def observe(self, G, bg, objs, sig, new_life=False, line=None):
        """After a board update.  Returns dict(progress, regress, n, view)."""
        if G is None or bg is None:
            return {"progress": 0.0, "regress": 0.0, "n": 0, "view": None}
        if new_life or self.lives == 0:
            self.begin_life(G, bg, objs, sig, line)
            return {"progress": 0.0, "regress": 0.0, "n": len(self.v0), "view": None}
        had_ctrl = self.ctrl
        self._detect_ctrl(objs)
        if self.ctrl != had_ctrl:
            # he has just learned what he controls.  Everything judged
            # before that was judged without knowing his body: his own
            # starting square was a region whose content "changed" when
            # he walked away, and paired with every same-shape cell (the
            # 122 spurious pairs on tu93).  Region bookkeeping restarts
            # with the body known; markers are re-derived at the next
            # life start.
            for rec in self.regs.values():
                rec[0] = None; rec[1] = False; rec[2] = 0
            self.self_box = self._self_from(objs, self.ctrl)
            self.self_prev = None
            for k in list(self.markers):
                if k == self.ctrl or k in self.body:
                    del self.markers[k]
                    self.v0.pop(("near", k), None); self.last.pop(("near", k), None)
        new_box = self._self_from(objs, self.ctrl) or self.self_box
        # where he was before this step and where he is after it: a
        # region he just left changed because his body left it
        self._update_regs(G, (self.self_box, new_box))
        self.self_prev = self.self_box
        self.self_box = new_box
        self._update_markers(objs)
        vals = self.values(G)
        # a relation discovered mid-life gets its baseline now
        for rid, v in vals.items():
            if rid not in self.v0 and v > 0:
                self.v0[rid] = float(v)
                self.last[rid] = float(v)
                self._since[rid] = 0
                self._lifemin[rid] = float(v)
                # it exists because its dynamic side just changed
                self._moved.add(rid)
        for rid, v in vals.items():
            if rid in self._lifemin and float(v) < self._lifemin[rid]:
                self._lifemin[rid] = float(v)
            if rid in self.last and float(v) != self.last[rid]:
                self._moved.add(rid)
        use = [r for r in vals if r in self.v0]
        # ONE relation is pursued at a time, like a person with a goal:
        # the confirmed one where a clear has taught him, else the
        # candidate nearest its goal.  Progress and regress are judged on
        # the relation he was pursuing BEFORE this step, so "something
        # somewhere got nearer" cannot fire (on wa30 it fired on 67% of
        # steps when every candidate counted).
        pursued = self._pursued if self._pursued in use else None
        prog = 0.0; reg = 0.0
        if pursued is not None:
            v = float(vals[pursued])
            d = (self.last.get(pursued, v) - v) / self.v0[pursued]
            prog = max(0.0, d); reg = max(0.0, -d)
        for rid in use:
            # patience runs only while a relation is PURSUED: giving up
            # is for what he tried, not for what he never attempted
            if rid != pursued:
                self._since[rid] = 0
            elif float(vals[rid]) < self.last.get(rid, float(vals[rid])):
                self._since[rid] = 0
            else:
                self._since[rid] = self._since.get(rid, 0) + 1
            self.last[rid] = float(vals[rid])
        if prog > 0:
            self.fires += 1
        # a CONFIRMED relation that sits satisfied without a clear for
        # longer than its patience was not the goal after all: revoke it
        if pursued is not None and pursued in self.confirmed:
            if float(vals[pursued]) <= 0.0:
                self._zero_since += 1
                if self._zero_since > 2 * self.v0.get(pursued, 1.0) + 5:
                    self.confirmed.discard(pursued)
                    self.revoked += 1
                    self._zero_since = 0
            else:
                self._zero_since = 0
        self._pursued = self._choose(use, vals)
        view = None
        if self._pursued is not None:
            rid = self._pursued
            view = (self._name(rid), float(vals[rid]), self.v0[rid], rid in self.confirmed)
        return {"progress": min(1.0, prog), "regress": min(1.0, reg),
                "n": len(use), "view": view, "best": pursued if prog > 0 else None,
                "confirmed": bool(self._pursued is not None
                                  and self._pursued in self.confirmed)}

    def _choose(self, use, vals):
        """Which relation to pursue.  Confirmed ones first.  Otherwise the
        one with the most to do at the start of the life (a fragment of
        the target is also a same-shape pair, and it must not win; nor
        must a 3x3 maze cell that happens to match another).  A relation
        that has not improved for longer than twice its remaining
        distance (plus a few steps) is stuck -- unreachable from here, or
        not what the level wants -- and is set aside for this life; a
        confirmed relation is never set aside."""
        if not use:
            return None
        conf = [r for r in use if r in self.confirmed]
        pool = conf or use

        def stuck(r):
            if r in self.confirmed:
                return False
            return self._since.get(r, 0) > 2 * vals[r] + 5

        live = [r for r in pool if vals[r] > 0 and not stuck(r)]
        if not live:
            live = [r for r in pool if vals[r] > 0] or pool
        if self._pursued in live:
            return self._pursued

        # THE BIGGEST JOB ON THE BOARD IS THE GOAL: the relation that had
        # the most to do at the start of the life.  A 100-cell picture
        # outranks a 28-cell walk (cd82); a 28-cell walk outranks a
        # spurious 3x3 "picture" made of maze cells (tu93, ka59).
        return max(live, key=lambda r: (self.v0.get(r, 0.0), -vals[r]))

    def confirm(self, G):
        """The board BEFORE the winning move: which candidate was the goal?

        A CLEAR IS POSITIVE EVIDENCE AND NEVER REVOKES ANYTHING.
        Measured 2026-09-06: one of the five cd82 level-1 clears came by
        a spoil-then-fix route whose pre-win board sat ABOVE its life
        minimum, and an earlier "must be at its minimum" rule therefore
        revoked the correct canvas/target pair and confirmed a marker
        instead.  Spoil-then-fix is how the level is played.

        Only a candidate whose value MOVED at some point this life can
        be the goal: a relation that never moved carries no evidence
        about what he did, and on cd82 every marker is constant (his
        body does not move there), so a constant would otherwise win
        every time.  Among those, the one nearest its goal.
        """
        vals = self.values(G)
        use = [r for r in vals if r in self.v0 and r in self._moved]
        if not use:
            # nothing moved toward a goal this life: confirm nothing
            # rather than something wrong
            return None
        rid = min(use, key=lambda r: vals[r] / self.v0[r])
        self.confirmed.add(rid)
        self._pursued = rid
        return rid

    def context(self, G):
        """For a steer: the best NEAR relation's direction from him to the
        nearest marker instance, as (sign dr, sign dc), or None."""
        if self.self_box is None:
            return None
        rid = self._pursued
        if rid is None or rid[0] != "near":
            return None
        boxes = self.markers.get(rid[1]) or []
        if not boxes:
            return None
        b = min(boxes, key=lambda bx: _gap(self.self_box, bx))
        sr = (self.self_box[0] + self.self_box[2]) / 2.0
        sc = (self.self_box[1] + self.self_box[3]) / 2.0
        mr = (b[0] + b[2]) / 2.0; mc = (b[1] + b[3]) / 2.0
        dr = int(np.sign(round(mr - sr))); dc = int(np.sign(round(mc - sc)))
        return (rid[1], dr, dc)

    @staticmethod
    def _name(rid):
        if rid[0] == "eq":
            A, B = rid[1], rid[2]
            return "eq[%d-%d,%d-%d]~[%d-%d,%d-%d]" % (A[0], A[2], A[1], A[3], B[0], B[2], B[1], B[3])
        return "near c%d/s%d" % rid[1]

    # ------------------------------------------------------------ persistence
    def to_dict(self):
        return {
            "static": [list(R) for R, rec in self.regs.items()
                       if not rec[1] and rec[0] is not None and rec[2] >= STATIC_AFTER],
            "dynamic": [list(R) for R, rec in self.regs.items() if rec[1]],
            "confirmed": [self._rid_str(r) for r in self.confirmed],
            "ctrl": (list(self.ctrl) if self.ctrl else None),
            "mobile": [list(k) for k in self.mobile],
        }

    def from_dict(self, d):
        n = 0
        try:
            for R in (d or {}).get("static") or []:
                R = tuple(int(x) for x in R)
                if len(R) == 4 and R not in self.regs:
                    self.regs[R] = [None, False, STATIC_AFTER]; n += 1
            for R in (d or {}).get("dynamic") or []:
                R = tuple(int(x) for x in R)
                if len(R) == 4:
                    if R in self.regs:
                        self.regs[R][1] = True
                    else:
                        self.regs[R] = [None, True, 0]
                    n += 1
            for s in (d or {}).get("confirmed") or []:
                rid = self._rid_parse(s)
                if rid is not None:
                    self.confirmed.add(rid); n += 1
            c = (d or {}).get("ctrl")
            if c and len(c) == 2:
                self.ctrl = (int(c[0]), int(c[1]))
                self._moves[self.ctrl] = max(self._moves.get(self.ctrl, 0), CTRL_AFTER)
                n += 1
            for k in (d or {}).get("mobile") or []:
                if len(k) == 2:
                    self.mobile.add((int(k[0]), int(k[1])))
        except Exception:
            return n
        return n

    @staticmethod
    def _rid_str(rid):
        if rid[0] == "eq":
            return "eq:" + ",".join(str(x) for x in rid[1]) + ":" + ",".join(str(x) for x in rid[2])
        return "near:%d,%d" % rid[1]

    @staticmethod
    def _rid_parse(s):
        try:
            kind, rest = s.split(":", 1)
            if kind == "eq":
                a, b = rest.split(":")
                return ("eq", tuple(int(x) for x in a.split(",")), tuple(int(x) for x in b.split(",")))
            if kind == "near":
                c, z = rest.split(",")
                return ("near", (int(c), int(z)))
        except Exception:
            return None
        return None
