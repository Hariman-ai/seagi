"""THE HYPOTHESIS ORGAN (patch 44, 2026-09-14).  One per (game, level).

A human reads a candidate goal off the picture -- "the exit of the maze", "the
socket shaped like the ring" -- and plays to TEST it.  The relation sense can see
such candidates (a unique static object near the object he controls) but can only
CONFIRM one from a win, so on the fifteen games he has never won nothing is ever
tested and nothing is ever refuted (measured 2026-09-11: ka59's pursued pair reached
0 with no clear and was pursued again the next life; sc25 147 plans, mm 9 -> 9).

This organ is the general form of what relsense + relplan do on cd82 after a win:

  self       the object he controls (relsense.ctrl, found by translation)
  map        (self box) -arrow-> (self box), learned from every EXECUTED arrow step,
             persisted across lives and restarts; the winning move is a transition
             too (to the WIN node of this level); the map is dropped if the body
             he controls changes identity
  candidates unique static objects on the current board -- one instance of its
             (colour, size) class, not a line, not a panel (relsense's own area
             bound), not his own colour, not mobile -- nearest first; a REFUTED
             candidate is skipped until every candidate on the level has been
             refuted, then the level starts a new round with what he knows now
  ask        STATELESS, like relplan: every ask re-plans from the current board
             (world_actor asks on every tick, step() executes only the winner of
             arbitration), so nothing is consumed by an ask that is not executed
  pursue     the shortest path in the map to a box at gap <= 1 of the candidate;
             a known path to WIN outranks everything; with no known path, the
             reachable box with an untried arrow nearest the candidate, then that
             arrow (frontier); with no frontier, nothing (his ordinary actor plays)
  refute     reached (gap <= 1) and an executed step there brought no clear
  confirm    the level increments on a step THIS organ proposed while pursuing

MEASURED OFFLINE before this was written (/root/self_seek.py, then /root/hyp_loop.py
through the patched code on local games, nothing of his touched): tu93 -- his own
ring refuted, the one unreached object (14, 9) pursued, level 0 cleared on life 2
and then every visit at 18 actions (human baseline 19).  Live he had 990 tu93 lives
and no win.  g50t, ka59, dc22, ls20, bp35, re86: the candidate is unreachable by
arrows alone; wa30, sc25: every candidate reached and refuted.  Those games need a
second operator (a click, a push); this organ gives them an honest refutation and
nothing else.

No per-game knowledge.  Arrows only; a click is never proposed here.
"""
from collections import deque

W = 64
REACH_GAP = 1            # adjacent or overlapping = reached (his step is 1-2 cells)
WIN = ('WIN',)
MAX_KEYS = 4000          # per (game, level); least-visited half evicted past it (the _selfhyp lesson)
SAVE_KEYS = 1000         # rows written to the save per (game, level): WIN rows + the most visited
SAVE_REFUTED = 256       # refutations kept, most recent first

try:
    from .relsense import MAX_AREA_FRAC as _AREA_FRAC     # the incumbent's "not a panel" bound
except Exception:                                          # pragma: no cover
    _AREA_FRAC = 0.40
MAX_CAND_AREA = int(_AREA_FRAC * W * W)


def _gap(a, b):
    """Chebyshev gap between boxes (r0, c0, r1, c1); 0 = touching/overlap."""
    return max(max(b[0] - a[2], a[0] - b[2], 0), max(b[1] - a[3], a[1] - b[3], 0))


def self_box(objs, ctrl):
    """The controlled object's own box from relsense-format objects
    (colour, size, r0, c0, r1, c1), or None unless exactly one instance."""
    if not ctrl or not objs:
        return None
    col, sz = int(ctrl[0]), int(ctrl[1])
    tol = max(1, sz // 4)
    c = [o for o in objs if int(o[0]) == col and abs(int(o[1]) - sz) <= tol]
    if len(c) != 1:
        return None
    o = c[0]
    return (int(o[2]), int(o[3]), int(o[4]), int(o[5]))


def box_key(box):
    """The map key: where the self is and how big."""
    if box is None:
        return None
    return (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))


def key_box(key):
    return (key[0], key[1], key[0] + key[2], key[1] + key[3])


class Hypothesis(object):
    def __init__(self):
        self.auto = {}          # key -> {action: key | WIN}
        self.tried = {}         # key -> set(actions)
        self.visits = {}        # key -> executed steps from it (for eviction)
        self.refuted = set()    # candidate ids (colour, size, r0, c0), this round
        self.refuted_order = [] # the same, oldest first (for the cap)
        self.rounds = 0         # times every candidate was refuted and the level restarted
        self.confirmed = None   # candidate id, from a clear on a step this organ proposed
        self.ctrl = None        # the body the map belongs to
        self.lives = 0          # persisted
        self.clears = 0
        self.asks = 0
        self.plans = 0          # asks answered with a path step
        self.frontier = 0       # asks answered with a step toward untried ground
        self.lost = 0           # asks with nothing to propose
        self.reached = 0
        self.evicted = 0
        self.best_gap = None    # nearest any pursued candidate has come, ever (this level)
        # life state (never persisted)
        self.proc_lives = 0     # lives begun in this process
        self.pursued = None     # (cid, box)
        self.life_gap = None    # nearest the pursued candidate came this life
        self.reached_steps = 0  # executed steps taken while at gap <= REACH_GAP of the pursued
        self.last_pick = None   # (key, action) of the last proposal
        self._win_path = False
        self.events = []
        self.errors = 0
        self._obs = 0
        self._dis = 0

    # ------------------------------------------------------------ lives
    def begin_life(self):
        """The first step of a life.  Folds the ended life's record in."""
        self.lives += 1
        self.proc_lives += 1
        lg = self.life_gap
        if lg is not None and (self.best_gap is None or lg < self.best_gap):
            self.best_gap = lg
        self.life_gap = None
        self.reached_steps = 0
        self.pursued = None
        self.last_pick = None
        self._win_path = False

    def set_ctrl(self, ctrl):
        """The map belongs to one body: a new one starts a new map."""
        c = tuple(int(x) for x in ctrl) if ctrl else None
        if c is None or c == self.ctrl:
            return
        if self.ctrl is not None and self.auto:
            self.events.append('NEWBODY %s->%s keys_dropped=%d' % (self.ctrl, c, len(self.auto)))
            self.auto = {}
            self.tried = {}
            self.visits = {}
            self._obs = 0
            self._dis = 0
            self.pursued = None
            self.confirmed = None       # confirmed with another body's map
        self.ctrl = c

    # ------------------------------------------------------- candidates
    def candidates(self, objs, ctrl, mobile=(), line=None, skip_refuted=True):
        """Unique static objects on this board, as (cid, box)."""
        cls = {}
        for o in objs:
            cls.setdefault((int(o[0]), int(o[1])), []).append(
                (int(o[2]), int(o[3]), int(o[4]), int(o[5])))
        out = []
        ccol = int(ctrl[0]) if ctrl else None
        for k, inst in cls.items():
            if len(inst) != 1 or k in mobile:
                continue
            if ccol is not None and k[0] == ccol:
                continue
            r0, c0, r1, c1 = inst[0]
            if r1 - r0 + 1 >= W or c1 - c0 + 1 >= W:
                continue
            if (r1 - r0 + 1) * (c1 - c0 + 1) > MAX_CAND_AREA:
                continue
            if line is not None and ((0 <= line < W and r0 == r1 == line)
                                     or (line >= W and c0 == c1 == line - W)):
                continue
            cid = (k[0], k[1], r0, c0)
            if skip_refuted and cid in self.refuted:
                continue
            out.append((cid, inst[0]))
        return out

    # --------------------------------------------------------- learning
    def observe(self, key, action, key2, cleared, by_organ=False):
        """One EXECUTED arrow step from `key` (before) to `key2` (after)."""
        self.last_pick = None           # consumed by this executed step, whatever it was
        if key is None:
            return
        a = int(action)
        if self.pursued is not None and by_organ:
            # only a step HE chose at the candidate can refute it: a replayed
            # route brushing past it is not a test (review 2, finding 1)
            if _gap(key_box(key), self.pursued[1]) <= REACH_GAP:
                self.reached_steps += 1
        if cleared:
            self.tried.setdefault(key, set()).add(a)
            self.auto.setdefault(key, {})[a] = WIN
            self.visits[key] = self.visits.get(key, 0) + 1
            self.clears += 1
            if self.pursued is not None and by_organ:
                self.confirmed = self.pursued[0]
                self.events.append('CONFIRMED cand=%s' % (self.pursued[0],))
            self.pursued = None
            return
        self.tried.setdefault(key, set()).add(a)     # tried, whatever came of it
        self.visits[key] = self.visits.get(key, 0) + 1
        if key2 is None:
            return                      # nowhere resolvable: no transition (a tried arrow
                                        # with no row is left alone by the frontier)
        row = self.auto.setdefault(key, {})
        if a in row:
            self._obs += 1
            if row[a] != key2 and row[a] != WIN:
                self._dis += 1
        row[a] = key2
        if len(self.auto) > MAX_KEYS:
            self._evict()

    def _evict(self):
        """Drop the least-visited half of the map (never a WIN row)."""
        keep = sorted(self.auto, key=lambda k: (any(v == WIN for v in self.auto[k].values()),
                                                self.visits.get(k, 0)), reverse=True)[:MAX_KEYS // 2]
        keep = set(keep)
        for k in list(self.auto):
            if k not in keep:
                self.auto.pop(k, None)
                self.evicted += 1
        reach = self._reachable_keys()
        for k in list(self.tried):
            if k not in reach:
                self.tried.pop(k, None)
        for k in list(self.visits):
            if k not in reach:
                self.visits.pop(k, None)

    def _reachable_keys(self):
        """Keys a surviving row can stand on: its own key or a successor.
        A key that is tried but has no row of its own (its arrow led
        nowhere resolvable) stays known as tried only while reachable."""
        out = set(self.auto)
        for row in self.auto.values():
            for v in row.values():
                if v != WIN:
                    out.add(v)
        return out

    def determinism(self):
        """Share of re-observed (key, action) steps that agreed with the map."""
        return 1.0 if self._obs == 0 else 1.0 - float(self._dis) / float(self._obs)

    # ----------------------------------------------------------- acting
    def _bfs(self, start, goal):
        seen = {start: None}
        q = deque([start])
        while q:
            k = q.popleft()
            if goal(k):
                path = []
                while seen[k] is not None:
                    pk, a = seen[k]
                    path.append(a)
                    k = pk
                return path[::-1]
            if k == WIN:
                continue
            for a, k2 in self.auto.get(k, {}).items():
                if k2 not in seen:
                    seen[k2] = (k, a)
                    q.append(k2)
        return None

    def _frontier_path(self, start, tb, arrows):
        """Path to the reachable key with an untried arrow nearest `tb`
        (ties: the shorter path), or None."""
        arrows = set(int(a) for a in arrows)
        seen = {start: None}
        dist = {start: 0}
        q = deque([start])
        best = None
        while q:
            k = q.popleft()
            if k != WIN and (arrows - self.tried.get(k, set())):
                cand = (_gap(key_box(k), tb), dist[k], k)
                if best is None or cand[:2] < best[:2]:
                    best = cand
            if k == WIN:
                continue
            for a, k2 in self.auto.get(k, {}).items():
                if k2 not in seen:
                    seen[k2] = (k, a)
                    dist[k2] = dist[k] + 1
                    q.append(k2)
        if best is None:
            return None
        k = best[2]
        path = []
        while seen[k] is not None:
            pk, a = seen[k]
            path.append(a)
            k = pk
        return path[::-1]

    def has_win(self):
        return any(WIN in row.values() for row in self.auto.values())

    def has_frontier(self, arrows):
        arrows = set(int(a) for a in arrows)
        for k in self.auto:
            if k != WIN and (arrows - self.tried.get(k, set())):
                return True
        return False

    def _refute(self, cid):
        self.refuted.add(cid)
        self.refuted_order.append(cid)
        if len(self.refuted_order) > SAVE_REFUTED:
            old = self.refuted_order.pop(0)
            if old not in self.refuted_order:
                self.refuted.discard(old)
        self.reached += 1
        self.events.append('REFUTED cand=%s' % (cid,))

    def act(self, key, cands, arrows, n_all=None, _after_refute=False):
        """The arrow to take from `key` now, or None.  STATELESS: nothing
        here is consumed; the same board asks the same answer.  `cands`
        from candidates() (refuted skipped); `n_all` how many candidates
        the board has before the refuted are skipped; `arrows` the arrow
        indices."""
        self.asks += 1
        self._win_path = False
        self.last_pick = None
        if key is None or not arrows:
            self.lost += 1
            return None
        arrows = [int(a) for a in arrows]
        # A KNOWN WIN IS A KNOWN WIN: the shortest path to it, whatever is pursued
        p = self._bfs(key, lambda k: k == WIN)
        if p:
            self._win_path = True
            self.plans += 1
            self.last_pick = (key, int(p[0]))
            return int(p[0])
        # the pursued candidate must still be on the board (a mover or an
        # occluded object is not chased at where it was)
        if self.pursued is not None:
            if self.pursued[0] in self.refuted or not any(c[0] == self.pursued[0] for c in cands):
                self.pursued = None
        if self.pursued is None:
            if not cands:
                if self.refuted and not _after_refute and (n_all is None or n_all > 0):
                    # every candidate on this level has been refuted: a new
                    # round with what he knows now (refutation is evidence,
                    # not a verdict -- "true is momentary")
                    self.events.append('ROUND %d: all %d candidates refuted, cleared'
                                       % (self.rounds + 1, len(self.refuted)))
                    self.refuted = set()
                    self.refuted_order = []
                    self.rounds += 1
                self.lost += 1
                return None
            conf = [cb for cb in cands if cb[0] == self.confirmed]
            pool = conf or sorted(cands, key=lambda cb: (_gap(key_box(key), cb[1]), cb[0]))
            self.pursued = pool[0]
            self.reached_steps = 0
            self.events.append('PURSUE cand=%s gap=%d refuted=%d'
                               % (self.pursued[0], _gap(key_box(key), self.pursued[1]),
                                  len(self.refuted)))
        cid, tb = self.pursued
        g = _gap(key_box(key), tb)
        if self.life_gap is None or g < self.life_gap:
            self.life_gap = g
        if g <= REACH_GAP and self.reached_steps >= 1 and cid != self.confirmed:
            # he stood at it, stepped, and the level did not end: not the goal
            self._refute(cid)
            self.pursued = None
            return self.act(key, [cb for cb in cands if cb[0] != cid], arrows, n_all, _after_refute=True)
        p = self._bfs(key, lambda k: k != WIN and _gap(key_box(k), tb) <= REACH_GAP)
        if p:
            self.plans += 1
            self.last_pick = (key, int(p[0]))
            return int(p[0])
        if p is not None and g <= REACH_GAP:
            # already there (an empty path): try something untried here so
            # the next executed step can refute it
            untried = [x for x in arrows if x not in self.tried.get(key, set())]
            a = untried[0] if untried else arrows[0]
            self.frontier += 1
            self.last_pick = (key, a)
            return a
        p = self._frontier_path(key, tb, arrows)
        if p is None:
            self.lost += 1
            return None
        if p:
            self.frontier += 1
            self.last_pick = (key, int(p[0]))
            return int(p[0])
        untried = [x for x in arrows if x not in self.tried.get(key, set())]
        if not untried:
            self.lost += 1
            return None
        self.frontier += 1
        self.last_pick = (key, untried[0])
        return untried[0]

    def proposed(self, key, action):
        """Was (key, action) this organ's last proposal?  For confirmation."""
        return self.last_pick is not None and self.last_pick[0] == key and self.last_pick[1] == int(action)

    def win_path(self):
        """True when the last act() was a step on a known path to WIN."""
        return bool(self._win_path)

    def record_now(self):
        """The life in flight brought a candidate nearer than any life before."""
        lg = self.life_gap
        return lg is not None and (self.best_gap is None or lg < self.best_gap)

    def testing(self, arrows):
        """A hypothesis is under test here, asked AT the terminal of the life
        in flight: this life set a record (or is the first of the process),
        and somewhere untried is reachable.  A known win does not hold by
        itself: it is walked from the first step of the next life anyway."""
        if self.proc_lives <= 1 and self.life_gap is not None:
            return self.has_frontier(arrows)
        if not self.record_now():
            return False
        return self.has_frontier(arrows)

    # ------------------------------------------------------ persistence
    def to_dict(self):
        def ks(k):
            return 'WIN' if k == WIN else ','.join(str(int(x)) for x in k)
        keep = sorted(self.auto, key=lambda k: (any(v == WIN for v in self.auto[k].values()),
                                                self.visits.get(k, 0)), reverse=True)[:SAVE_KEYS]
        auto = {}
        for k in keep:
            auto[ks(k)] = {str(int(a)): ks(k2) for a, k2 in self.auto[k].items()}
        reach = set(keep)
        for k in keep:
            for v in self.auto[k].values():
                if v != WIN:
                    reach.add(v)
        tried = {ks(k): sorted(int(a) for a in self.tried[k]) for k in reach if k in self.tried}
        return {'v': 2, 'auto': auto, 'tried': tried,
                'refuted': [list(c) for c in self.refuted_order[-SAVE_REFUTED:]],
                'rounds': int(self.rounds),
                'confirmed': list(self.confirmed) if self.confirmed else None,
                'ctrl': list(self.ctrl) if self.ctrl else None,
                'lives': int(self.lives), 'clears': int(self.clears),
                'best_gap': (int(self.best_gap) if self.best_gap is not None else None),
                'obs': int(self._obs), 'dis': int(self._dis)}

    def from_dict(self, d):
        """Union merge; returns the number of transitions taken."""
        def kp(s):
            if s == 'WIN':
                return WIN
            return tuple(int(x) for x in s.split(','))
        n = 0
        try:
            if d.get('ctrl'):
                c = tuple(int(x) for x in d['ctrl'])
                if self.ctrl is not None and c != self.ctrl:
                    return 0                # another body's map: not merged
                self.ctrl = c
            for ks_, row in (d.get('auto') or {}).items():
                k = kp(ks_)
                r = self.auto.setdefault(k, {})
                for a, k2 in row.items():
                    if int(a) not in r:
                        r[int(a)] = kp(k2)
                        n += 1
            for ks_, acts in (d.get('tried') or {}).items():
                self.tried.setdefault(kp(ks_), set()).update(int(a) for a in acts)
            for c in (d.get('refuted') or [])[-SAVE_REFUTED:]:
                c = tuple(int(x) for x in c)
                if c not in self.refuted:
                    self.refuted.add(c)
                    self.refuted_order.append(c)
            if len(self.refuted_order) > SAVE_REFUTED:
                drop = self.refuted_order[:-SAVE_REFUTED]
                self.refuted_order = self.refuted_order[-SAVE_REFUTED:]
                for c in drop:
                    if c not in self.refuted_order:
                        self.refuted.discard(c)
            self.rounds = max(self.rounds, int(d.get('rounds') or 0))
            if d.get('confirmed') and self.confirmed is None:
                self.confirmed = tuple(int(x) for x in d['confirmed'])
            if d.get('best_gap') is not None and (self.best_gap is None or int(d['best_gap']) < self.best_gap):
                self.best_gap = int(d['best_gap'])
            self.lives = max(self.lives, int(d.get('lives') or 0))
            self.clears = max(self.clears, int(d.get('clears') or 0))
            self._obs += int(d.get('obs') or 0)
            self._dis += int(d.get('dis') or 0)
            if len(self.auto) > MAX_KEYS:
                self._evict()
        except (TypeError, ValueError, AttributeError):
            self.errors += 1
        return n
