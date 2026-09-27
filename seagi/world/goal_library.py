"""THE GOAL LIBRARY (patch 48, 2026-09-16).  ONE for him, across every game.

A mechanic who has fixed twelve cars does not read the thirteenth from nothing:
he knows the KINDS of thing a car wants.  Until now his only goal organ knew one
kind ("bring the self to a unique object", written for tu93) and had no way to
get a second.  This module is the library of kinds, filled from HIS OWN clears
and read on every board he meets.

  kind      a relation any board can instantiate, colour-free and position-free:
              eq       a region made equal to another region of the same shape
                       (value = mismatching cells, goal 0)
              near     the object he controls brought to a class of objects
                       (value = Chebyshev gap, goal 0)
              count-   a class of objects made to vanish (value = instances, goal 0)
            each with a descriptor of the things involved (size bucket, hollow,
            how many siblings, shape kind), never their colour or place
  schema    (kind, descriptor).  What the library keeps: how often each schema
            was the thing that dropped to its goal on the board before a winning
            move (paid), how often it was pursued to its goal without a clear
            (tried), and on which games
  learn     at every level clear: the instances present on the life's first board
            that reached their goal on the pre-win board explain the clear and
            their schemas are paid; if nothing explains it the clear is counted
            UNEXPLAINED -- the library's own record of what it cannot yet read
  rank      on any board: every instance of every kind, ordered by the schema's
            record (paid+1)/(paid+tried+2), then by how much there is to do;
            refuted instances (this level) skipped
  use       the search organ pursues the top candidate: the value of the pursued
            relation at every state he stands on is what a GOOD STATE LOOKS LIKE,
            and its frontier choice becomes best-first on that value instead of
            nearest; a candidate that reached its goal under his own step without
            a clear is REFUTED for the level and the next is pursued

No per-game knowledge.  The kinds are the vocabulary; which kinds matter, on what
sort of objects, is learned from what paid.  A new game meets the whole library.
"""
from collections import defaultdict
import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:              # pragma: no cover - scipy is present on the VPS
    _ndi = None

W = 64
MIN_REGION = 3                 # a region is at least 3x3 (the relation sense's bound)
MAX_MARKER = 8                 # a marker class: few instances, not a texture (relsense)
MAX_REGIONS = 60               # regions considered per board (largest first)
MAX_PAIRS = 80                 # eq pairs per board
MAX_SCHEMAS = 400              # schemas kept (least paid dropped)
MAX_RIDS = 300                 # relation instances tracked per (game, level)
SAVE_RIDS = 60                 # of them written to the save
MIN_DRY = 3                    # dry lives a level needs before a clear can be judged (the null)
MAX_OBJS = 200                 # a board with more objects is a texture: no reading
KINDS = ('eq', 'near', 'count-')
REACH_FRAC = 0.2               # a relation is at its goal when <= this fraction of its job remains


def segment(G, line=None):
    """(bg, objects) with objects (colour, size, r0, c0, r1, c1); the clock
    line masked.  Same reading as the relation sense's."""
    if G is None or _ndi is None:
        return None, []
    try:
        G = np.asarray(G, dtype=np.int16)
    except (TypeError, ValueError):
        return None, []
    if G.shape != (W, W):
        return None, []
    mask = np.ones((W, W), dtype=bool)
    if line is not None:
        try:
            line = int(line)
            if 0 <= line < W:
                mask[line, :] = False
            elif W <= line < 2 * W:
                mask[:, line - W] = False
        except (TypeError, ValueError):
            pass
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
        if len(out) > MAX_OBJS:
            return bg, []
    return bg, out


def _gap(A, B):
    return max(max(0, max(A[0], B[0]) - min(A[2], B[2])),
               max(0, max(A[1], B[1]) - min(A[3], B[3])))


def size_b(s):
    return 0 if s <= 1 else 1 if s <= 4 else 2 if s <= 9 else 3 if s <= 24 else 4 if s <= 63 else 5


def sib_b(n):
    return 0 if n <= 1 else 1 if n <= 3 else 2 if n <= 8 else 3


def _hollow(o):
    h = o[4] - o[2] + 1
    w = o[5] - o[3] + 1
    return bool(o[1] < h * w and h >= 3 and w >= 3)


def class_desc(o, n):
    """Colour-free descriptor of an object class: size bucket, hollow, siblings, shape."""
    h = o[4] - o[2] + 1
    w = o[5] - o[3] + 1
    shape = 0 if h == w else 1 if max(h, w) <= 2 * min(h, w) else 2
    return (size_b(o[1]), int(_hollow(o)), sib_b(n), shape)


def regions_of(objs):
    """Regions: object boxes of at least MIN_REGION on a side, plus the boxes of
    clusters of touching objects.  Each tagged 'o' (one object) or 'c' (cluster)."""
    boxes = [(o[2], o[3], o[4], o[5]) for o in objs]
    n = len(boxes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if _gap(boxes[i], boxes[j]) == 0:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(boxes[i])
    out = {}
    for b in boxes:
        if b[2] - b[0] + 1 >= MIN_REGION and b[3] - b[1] + 1 >= MIN_REGION:
            out[b] = 'o'
    for g in groups.values():
        if len(g) < 2:
            continue
        b = (min(x[0] for x in g), min(x[1] for x in g), max(x[2] for x in g), max(x[3] for x in g))
        if b[2] - b[0] + 1 >= MIN_REGION and b[3] - b[1] + 1 >= MIN_REGION and b not in out:
            out[b] = 'c'
    lim = 0.4 * W * W
    regs = [(b, k) for b, k in out.items() if (b[2] - b[0] + 1) * (b[3] - b[1] + 1) <= lim]
    regs.sort(key=lambda bk: -((bk[0][2] - bk[0][0] + 1) * (bk[0][3] - bk[0][1] + 1)))
    return regs[:MAX_REGIONS]


def schema_key(schema):
    return '|'.join(str(x) for x in schema)


def rid_key(rid):
    if rid[0] == 'eq':
        return 'eq:%d,%d,%d,%d~%d,%d,%d,%d' % (rid[1] + rid[2])
    return '%s:%d,%d' % (rid[0], rid[1][0], rid[1][1])


class Instance(object):
    __slots__ = ('rid', 'schema', 'value', 'start')

    def __init__(self, rid, schema, value):
        self.rid = rid          # ('eq', A, B) | ('near', cls) | ('count-', cls) | ('count+', cls)
        self.schema = schema    # (kind, descriptor...)
        self.value = value      # lower is better; <= 0 = goal reached (eq, near, count-)
        self.start = None


def read(G, line=None, self_box=None, self_cls=None):
    """Every instance of every kind on this board.  Returns (bg, objs, {rid: Instance})."""
    bg, objs = segment(G, line)
    out = {}
    if bg is None or not objs:
        return bg, objs, out
    Ga = np.asarray(G, dtype=np.int16)
    cls = defaultdict(list)
    for o in objs:
        cls[(int(o[0]), int(o[1]))].append(o)
    # eq: same-shape disjoint region pairs
    regs = regions_of(objs)
    npairs = 0
    for i in range(len(regs)):
        A, ka = regs[i]
        sh = (A[2] - A[0], A[3] - A[1])
        for j in range(i + 1, len(regs)):
            B, kb = regs[j]
            if (B[2] - B[0], B[3] - B[1]) != sh or _gap(A, B) <= 0:
                continue
            a, b = (A, B) if A < B else (B, A)
            ka2, kb2 = (ka, kb) if A < B else (kb, ka)
            v = int((Ga[a[0]:a[2] + 1, a[1]:a[3] + 1] != Ga[b[0]:b[2] + 1, b[1]:b[3] + 1]).sum())
            rid = ('eq', a, b)
            sch = ('eq', size_b(sh[0] + 1), size_b(sh[1] + 1), ''.join(sorted(ka2 + kb2)))
            out[rid] = Instance(rid, sch, v)
            npairs += 1
            if npairs >= MAX_PAIRS:
                break
        if npairs >= MAX_PAIRS:
            break
    # near: the self to each small class
    if self_box is not None:
        for k, inst in cls.items():
            if k == self_cls or len(inst) > MAX_MARKER:
                continue
            g = min(_gap(self_box, (o[2], o[3], o[4], o[5])) for o in inst)
            out[('near', k)] = Instance(('near', k), ('near',) + class_desc(inst[0], len(inst)), g)
    # count-: a class made to vanish
    for k, inst in cls.items():
        if k == self_cls:
            continue
        d = class_desc(inst[0], len(inst))
        out[('count-', k)] = Instance(('count-', k), ('count-',) + d, len(inst))
    return bg, objs, out


def evaluate(G, insts, line=None, self_box=None, self_cls=None):
    """The value of instances READ ON ANOTHER BOARD (a life's first board),
    measured on this one: eq on the same fixed boxes, near from the self box
    to the class's instances here, count- of the class here.  {rid: value};
    a relation whose things are gone reads as None."""
    if G is None:
        return {}
    try:
        Ga = np.asarray(G, dtype=np.int16)
    except (TypeError, ValueError):
        return {}
    if Ga.shape != (W, W):
        return {}
    need_objs = any(i.rid[0] != 'eq' for i in insts.values())
    cls = defaultdict(list)
    if need_objs:
        _, objs = segment(G, line)
        for o in objs:
            cls[(int(o[0]), int(o[1]))].append(o)
    out = {}
    for rid, i in insts.items():
        if rid[0] == 'eq':
            a, b = rid[1], rid[2]
            out[rid] = int((Ga[a[0]:a[2] + 1, a[1]:a[3] + 1] != Ga[b[0]:b[2] + 1, b[1]:b[3] + 1]).sum())
        elif rid[0] == 'near':
            inst = cls.get(rid[1])
            if self_box is None or not inst:
                out[rid] = None
            else:
                out[rid] = min(_gap(self_box, (o[2], o[3], o[4], o[5])) for o in inst)
        elif rid[0] == 'count-':
            out[rid] = len(cls.get(rid[1], ()))
    return out


def reached(v0, v1):
    """At its goal: at most REACH_FRAC of the job remains (a clear comes one
    move after the picture is complete on cd82: 90 -> 15, never 0)."""
    if v0 is None or v1 is None or v0 <= 0:
        return False
    return v1 <= REACH_FRAC * v0


def goal_reached(inst):
    return inst.value <= 0


class Library(object):
    """The schemas and their record.  ONE instance, across games; persisted."""

    def __init__(self):
        self.paid = defaultdict(float)     # schema key -> credit from clears
        self.tried = defaultdict(float)    # schema key -> refutation events (readout only)
        self.tried_games = defaultdict(set)  # schema key -> games it was refuted on (the record)
        self.games = defaultdict(set)      # schema key -> games it paid on
        self.levels = {}                   # 'game/lv' -> per-level evidence (clear vs dry)
        self.clears = 0
        self.unexplained = 0
        self.events = []

    # ----------------------------------------------------------- record
    def prior(self, schema):
        """(paid + 1) / (paid + games refuted on + 2).  A refutation counts once
        per (schema, game): refutations scale with steps, payments with first
        clears, so per-event counting sank every paid schema below a virgin one
        within hours (review, 2026-09-16)."""
        k = schema_key(schema)
        return (self.paid[k] + 1.0) / (self.paid[k] + len(self.tried_games[k]) + 2.0)

    def known(self, schema):
        return self.paid.get(schema_key(schema), 0.0) > 0

    # ------------------------------------------------------------ learn
    def _lvl(self, game, level):
        k = '%s/%s' % (str(game)[:4], int(level))
        L = self.levels.get(k)
        if L is None:
            L = {'n_clear': 0, 'n_dry': 0, 'reach': {}, 'schema': {}, 'paid_rid': None, 'credit': 0.0}
            self.levels[k] = L
        return k, L

    def observe_life(self, game, level, start, end, cleared, line=None, self_box0=None,
                     self_box1=None, self_cls=None):
        """A life ended: its first board and its last board (the board before
        the winning move, or the last ordinary board of a dry life).  Which
        instances reached their goal?  A goal is what reaches its goal in
        CLEAR lives and not in DRY lives of the same level -- a clear alone
        cannot tell the goal from a side effect (measured 2026-09-16: crediting
        every relation at 0 paid small spurious pairs 36 times on cd82).
        Returns the level's current best rid key (or None)."""
        k, L = self._lvl(game, level)
        _, _, i0 = read(start, line, self_box0, self_cls)
        v1 = evaluate(end, i0, line, self_box1, self_cls)
        if cleared:
            L['n_clear'] += 1
            self.clears += 1
        else:
            L['n_dry'] += 1
        for rid, a in i0.items():
            if not reached(a.value, v1.get(rid)):
                continue
            rk = rid_key(rid)
            st = L['reach'].get(rk)
            if st is None:
                if len(L['reach']) >= MAX_RIDS:
                    # evict the least-seen rid (never the paid one), not the newcomer
                    victim = min((r for r in L['reach'] if r != L['paid_rid']),
                                 key=lambda r: L['reach'][r][0] + L['reach'][r][1], default=None)
                    if victim is None:
                        continue
                    L['reach'].pop(victim, None)
                    L['schema'].pop(victim, None)
                st = [0, 0]
                L['reach'][rk] = st
                L['schema'][rk] = schema_key(a.schema)
            st[0 if cleared else 1] += 1
        if cleared:
            return self._pay(game, level, k, L)
        return L['paid_rid']

    def level_best(self, L):
        """(rid key, difference) of the relation most specific to clears."""
        if L['n_clear'] <= 0 or L['n_dry'] < MIN_DRY:
            # without dry lives there is no null: every relation that reached
            # its goal ties, and the first in dict order would be paid (review)
            return None, 0.0
        best = None
        bd = 0.0
        bn = -1
        nd = L['n_dry']
        for rk, (rc, rd) in L['reach'].items():
            pc = rc / float(L['n_clear'])
            pd = rd / float(nd)
            d = pc - pd
            if d > bd or (d == bd and best is not None and rc + rd > bn):
                bd = d
                bn = rc + rd          # ties: the relation with the most evidence
                best = rk
        return best, bd

    def _pay(self, game, level, k, L):
        best, d = self.level_best(L)
        if best is None or d < 0.5:
            if L['paid_rid'] is None and not L.get('unexplained') and L['n_dry'] >= MIN_DRY:
                L['unexplained'] = True          # counted once per level, once a null exists
                self.unexplained += 1
                self.events.append('UNEXPLAINED game=%s lv=%s clears=%d best=%s d=%.2f'
                                   % (str(game)[:4], level, L['n_clear'], best, d))
            return L['paid_rid']
        sch = L['schema'][best]
        g = str(game)[:4]
        if L['paid_rid'] != best or abs(L['credit'] - d) > 1e-9:
            if L['paid_rid'] is not None:
                old = L['schema'].get(L['paid_rid'])
                if old is not None:
                    self.paid[old] = max(0.0, self.paid[old] - L['credit'])
            self.paid[sch] += d
            self.games[sch].add(g)
            L['paid_rid'] = best
            L['credit'] = d
            self.events.append('PAID schema=%s credit=%.2f game=%s lv=%s rid=%s paid=%.1f games=%d'
                               % (sch, d, g, level, best, self.paid[sch], len(self.games[sch])))
            self._cap()
        return best

    def learn_clear(self, game, level, start, prewin, line=None, self_box0=None,
                    self_box1=None, self_cls=None):
        """A clear: observe_life with cleared=True.  Returns [rid key] or []."""
        r = self.observe_life(game, level, start, prewin, True, line, self_box0, self_box1, self_cls)
        return [r] if r else []

    def refute(self, schema, game=None):
        self.refute_key(schema_key(schema), game)

    def refute_key(self, k, game=None):
        k = str(k)
        self.tried[k] += 1.0
        if game is not None:
            self.tried_games[k].add(str(game)[:4])

    def _cap(self):
        if len(self.paid) <= MAX_SCHEMAS:
            return
        keep = sorted(self.paid, key=lambda k: -self.paid[k])[:MAX_SCHEMAS]
        keep = set(keep)
        for k in list(self.paid):
            if k not in keep:
                self.paid.pop(k, None)
                self.tried.pop(k, None)
                self.games.pop(k, None)

    # ------------------------------------------------------------- rank
    def candidates(self, G, line=None, self_box=None, self_cls=None, refuted=()):
        """Instances on this board, best first: by the schema's record, then by
        the size of the job.  Only schemas that ever paid lead; the rest follow
        in the same order (a new kind is tried after every known kind)."""
        bg, objs, inst = read(G, line, self_box, self_cls)
        refuted = set(refuted or ())
        out = [i for i in inst.values() if rid_key(i.rid) not in refuted and not goal_reached(i)]
        out.sort(key=lambda i: (-self.prior(i.schema), -abs(i.value)))
        return out, inst

    # ------------------------------------------------------ persistence
    def to_dict(self):
        lv = {}
        for k, L in self.levels.items():
            top = sorted(L['reach'].items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:SAVE_RIDS]
            lv[k] = {'n_clear': L['n_clear'], 'n_dry': L['n_dry'], 'paid_rid': L['paid_rid'],
                     'credit': L['credit'], 'reach': dict(top),
                     'schema': {rk: L['schema'][rk] for rk, _ in top if rk in L['schema']}}
        return {'v': 2, 'paid': dict(self.paid), 'tried': dict(self.tried),
                'tried_games': {k: sorted(v) for k, v in self.tried_games.items()},
                'games': {k: sorted(v) for k, v in self.games.items()},
                'clears': int(self.clears), 'unexplained': int(self.unexplained), 'levels': lv}

    def from_dict(self, d, replace=False):
        """Merge a blob.  `replace`: the blob is HIS save, which already
        contains everything the seed gave him -- it replaces the record
        rather than merging with max (max re-inflated a credit that _pay
        had moved; review, 2026-09-16).  The seed is loaded without it."""
        try:
            if replace:
                self.paid = defaultdict(float)
                self.tried = defaultdict(float)
                self.tried_games = defaultdict(set)
                self.games = defaultdict(set)
                self.levels = {}
            for k, v in (d.get('paid') or {}).items():
                self.paid[str(k)] = max(self.paid.get(str(k), 0.0), float(v))
            for k, v in (d.get('tried') or {}).items():
                self.tried[str(k)] = max(self.tried.get(str(k), 0.0), float(v))
            for k, v in (d.get('tried_games') or {}).items():
                self.tried_games[str(k)].update(str(x) for x in (v or []))
            for k, v in (d.get('games') or {}).items():
                self.games[str(k)].update(str(x) for x in (v or []))
            self.clears = max(self.clears, int(d.get('clears') or 0)) if not replace else int(d.get('clears') or 0)
            self.unexplained = max(self.unexplained, int(d.get('unexplained') or 0)) if not replace else int(d.get('unexplained') or 0)
            for k, L in (d.get('levels') or {}).items():
                if k in self.levels:
                    continue
                self.levels[str(k)] = {'n_clear': int(L.get('n_clear') or 0), 'n_dry': int(L.get('n_dry') or 0),
                                       'paid_rid': L.get('paid_rid'), 'credit': float(L.get('credit') or 0.0),
                                       'reach': {str(a): [int(b[0]), int(b[1])] for a, b in (L.get('reach') or {}).items()},
                                       'schema': {str(a): str(b) for a, b in (L.get('schema') or {}).items()}}
        except (TypeError, ValueError, AttributeError):
            return 0
        return len(self.paid)

    def summary(self, top=12):
        rows = sorted(self.paid.items(), key=lambda kv: -kv[1])[:top]
        return [(k, round(p, 1), len(self.tried_games.get(k, ())), sorted(self.games.get(k, ()))) for k, p in rows]
