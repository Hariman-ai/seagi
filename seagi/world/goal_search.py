"""THE SEARCH ORGAN (patch 45, 2026-09-15).  One per (game, level).

A mechanic on a machine he has never seen presses every control once, remembers
what each did, and works outward from what he has already reached: not at random,
and not twice.  This organ is that method, with no knowledge of any game.

  state      the board with the clock lines masked (a hash); the level's root
             is the board he stands on after a reset or an entry
  controls   every non-click action, plus a click on the centre of every object
             on the board (the smallest first, capped) -- objects from the
             relation sense's own segmentation
  map        state -control-> state | WIN | OVER, learned from every EXECUTED
             step whoever proposed it, persisted across lives, visits, restarts
  ask        STATELESS: from the state he is in, the shortest known path to WIN;
             else an untried control here; else the shortest known path to the
             nearest state with an untried control; else nothing
  hold       at a terminal, while the life just ended found states never seen
             before and somewhere untried is still reachable from the root

MEASURED OFFLINE before this was written (/root/bfs_click.py, 6000 steps, three
orderings): tu93 levels 0-3 (paths 18/10/21-31/17; human 19/16/34/42), s5i5 L0
(3/3), tn36 L0 (3/3; the object-centre action set alone does that one), su15 L0
(2/3), ls20 L0, dc22 L0, tr87 L0 -- games he had never cleared in his life, and
three levels of tu93 beyond the one he had.  Searched and failed: sb26, ka59,
bp35, sk48, sc25; not searchable (nondeterministic across resets): g50t, wa30.
THROUGH THE PATCHED CODE (/root/search_loop.py, 6000 steps): tu93 L0 18 / L1 10
actions once found, tn36 7, ls20 21, dc22 20, s5i5 once in 31 (life 56), su15
none in 6000.  Random play over random cells clears none of them.

Why the earlier click tools failed and this does not: a click's effect could not
be keyed on the object clicked (ka59: 3116 keys, none re-observed), nor learned
from random clicks (tn36, s5i5: a random click is a no-op except at the buttons).
The board itself recurs (key_recur: sb26 0.88, su15 0.98) and a systematic sweep
finds the buttons in one life.  The state is the world's, not his gaze's.
"""
from collections import deque
import hashlib
import os
import sys

W = 64
WIN = ('WIN',)
OVER = ('OVER',)
MAX_STATES = 2000        # per (game, level); least-visited half evicted past it (review: 35 MB per 4000)
SAVE_STATES = 1000       # rows written to the save per (game, level)
MAX_OBJS = 48            # clicks offered per state: the smallest objects first
LINE_SPAN = 60           # an object spanning this many cells is a line/panel, not a button
HIST_MAX = 400           # lives of one visit the walk judgement looks back over
WALK_MIN_LIVES = 6       # the first judgement: three lives against three
WALK_RUN = 12            # consecutive walk judgements before a release (patch 47, measured)
REACH_FRAC = 0.2         # the library's "at its goal": this fraction of the job or less remains (patch 48)


def _CONVERGE_ON():
    """THE SEARCH JUDGES ITS OWN PROGRESS.  /root/CONVERGE_ON (patch 47).
    MEASURED 2026-09-16 (/root/bfs4_out.txt, /root/live_series.json): the
    patch-45 hold released a life that found no new state and held one that
    did.  The games the sweep clears offline (s5i5 176 lives, su15 150, dc22
    193, tu93 43) have streaks of zero-new lives that still consume untried
    controls; the games it never clears (sc25, sk48, sb26, ka59, bp35) find
    new states at a CONSTANT rate per step, life after life.  Live: sk48 held
    60 lives, sc25 134, 0 clears; no converging search was ever held past its
    first dry life.  Gate on: a life with progress (a new state or an untried
    control executed) holds while a frontier is reachable, unless the visit is
    a WALK: from six lives on, the pooled new-per-step rate of the later half
    of the visit is not below half the earlier half's, WALK_RUN terminals in a
    row.  Gate off: the patch-45 body, unchanged."""
    try:
        return os.path.exists('/root/CONVERGE_ON')
    except Exception:
        return False


def state_of(grid, lines=()):
    """The state key: the board with the clock lines masked, hashed.
    `grid` is a 64x64 sequence of ints; `lines` the world's clock lines
    (row index, or 64 + column index)."""
    if grid is None:
        return None
    try:
        rs = set(int(x) for x in lines if 0 <= int(x) < W)
        cs = set(int(x) - W for x in lines if int(x) >= W)
        b = bytearray()
        for i, row in enumerate(grid):
            if i in rs:
                continue
            if cs:
                b.extend(int(v) & 0xFF for j, v in enumerate(row) if j not in cs)
            else:
                b.extend(int(v) & 0xFF for v in row)
        return sys.intern(hashlib.blake2b(bytes(b), digest_size=8).hexdigest())
    except Exception:
        return None


def controls_of(objs, arrows, click_idx):
    """The controls at a board: ('A', i) for every non-click index, and
    ('C', r, c) for the centre of every object (relsense format
    (colour, size, r0, c0, r1, c1)), smallest first, capped."""
    out = [('A', int(a)) for a in arrows]
    if click_idx is None or not objs:
        return out
    seen = set()
    cand = []
    for o in objs:
        try:
            r0, c0, r1, c1 = int(o[2]), int(o[3]), int(o[4]), int(o[5])
        except (TypeError, ValueError, IndexError):
            continue
        if r1 - r0 >= LINE_SPAN or c1 - c0 >= LINE_SPAN:
            continue
        rc = ((r0 + r1) // 2, (c0 + c1) // 2)
        if rc in seen:
            continue
        seen.add(rc)
        cand.append(((r1 - r0 + 1) * (c1 - c0 + 1), rc))
    cand.sort()
    out.extend(('C', rc[0], rc[1]) for _, rc in cand[:MAX_OBJS])
    return out


def ctl_key(c):
    return 'A%d' % c[1] if c[0] == 'A' else 'C%d,%d' % (c[1], c[2])


def key_ctl(s):
    if s[0] == 'A':
        return ('A', int(s[1:]))
    r, c = s[1:].split(',')
    return ('C', int(r), int(c))


class Search(object):
    def __init__(self):
        self.trans = {}         # state -> {control: state | WIN | OVER}
        self.n_ctl = {}         # state -> how many controls the board offered
        self.visits = {}        # state -> executed steps from it
        self.root = None        # the level's root state (after a reset / entry)
        self.lives = 0
        self.clears = 0
        self.asks = 0
        self.plans = 0          # asks answered with a path step (to WIN or to the frontier)
        self.tries = 0          # asks answered with an untried control here
        self.lost = 0
        self.evicted = 0
        self.errors = 0
        self.wins = 0           # WIN edges known (the win BFS runs only when > 0)
        self.max_life = 0       # the longest life seen here: the clock's length
        # life state (never persisted)
        self.proc_lives = 0
        self.new_this_life = 0  # states first seen this life
        self.steps_this_life = 0
        self.untried_this_life = 0  # controls executed here for the first time (patch 47)
        self.evicted_at_life = 0
        self.hist = []          # (steps, new) per ended life of THIS visit (patch 47)
        self.walk_run = 0       # consecutive walk judgements, last computed
        self._walk_logged = -1
        # the goal from the library (patch 48)
        self.pursued = None      # (rid key, schema key, v0) or None
        self.lib_refuted = set() # rid keys refuted on this level (persisted)
        self.val = {}            # state -> value of the pursued relation there
        self.plan_miss = 0       # this life: his own planned steps that landed elsewhere
        self.last_pick = None   # (state, control) of the last proposal
        self._win_path = False
        self.events = []

    # ------------------------------------------------------------ lives
    def begin_life(self, root=None, visit=False):
        self.lives += 1
        self.proc_lives += 1
        self.max_life = max(self.max_life, self.steps_this_life)
        if visit:
            # a new visit: the lives before it were another visit's (patch 47)
            self.hist = []
            self.walk_run = 0
        elif self.steps_this_life > 0:
            self.hist.append((int(self.steps_this_life), int(self.new_this_life)))
            if len(self.hist) > HIST_MAX:
                del self.hist[:len(self.hist) - HIST_MAX]
        self.new_this_life = 0
        self.steps_this_life = 0
        self.untried_this_life = 0
        self.plan_miss = 0
        self.evicted_at_life = self.evicted
        self.last_pick = None
        self._win_path = False
        if root is not None:
            if self.root is not None and root != self.root and root not in self.trans:
                self.events.append('NEWROOT %s->%s' % (self.root[:6], root[:6]))
            self.root = root

    # --------------------------------------------------------- learning
    def see(self, state, controls):
        """A board he stands on: register its controls once."""
        if state is None:
            return
        if state not in self.n_ctl:
            known = state in self.trans
            row = self.trans.setdefault(state, {})
            # controls he already tried here off the list (a click aimed
            # elsewhere by another organ) stay counted: the list plus them
            self.n_ctl[state] = len(controls) + sum(1 for c in row if c not in controls)
            if not known:
                self.new_this_life += 1
        if self.root is None:
            self.root = state

    def untried(self, state, controls):
        row = self.trans.get(state, {})
        return [c for c in controls if c not in row]

    def observe(self, state, control, state2, cleared, over, listed=True):
        """One EXECUTED step from `state` by `control`.  `listed`: the
        control was one this organ offers at that board (an off-list click,
        aimed by another organ, is learned too and counted as one more)."""
        if (self.last_pick is not None and state is not None and control is not None
                and self.last_pick == (state, control) and state2 is not None):
            _exp = self.trans.get(state, {}).get(control)
            if _exp is not None and _exp != WIN and _exp != OVER and _exp != state2:
                # a step he planned on a known edge landed elsewhere: the world
                # is not what the map says here (a non-Markov game)
                self.plan_miss += 1
        self.last_pick = None
        if state is None or control is None:
            return
        self.steps_this_life += 1
        row = self.trans.setdefault(state, {})
        fresh = control not in row    # patch 47: counted below, only where the row is WRITTEN
        if not listed and control not in row and state in self.n_ctl:
            self.n_ctl[state] += 1
        self.visits[state] = self.visits.get(state, 0) + 1
        if cleared:
            if row.get(control) != WIN:
                self.wins += 1
            if fresh:
                self.untried_this_life += 1
            row[control] = WIN
            self.clears += 1
            self.events.append('WIN via %s states=%d' % (ctl_key(control), len(self.trans)))
            return
        if over:
            if self.max_life and self.steps_this_life >= self.max_life:
                # the life ended at the clock's known length, not on this
                # control: its effect is unknown, it stays untried (review 2, 4)
                return
            if fresh:
                self.untried_this_life += 1
            row[control] = OVER
            return
        if state2 is None:
            return
        if fresh:
            self.untried_this_life += 1
        row[control] = state2
        if state2 not in self.trans:
            # first seen as a successor: its controls are learned when he stands on it
            self.trans[state2] = {}
            self.new_this_life += 1
        if len(self.trans) > MAX_STATES:
            self._evict()

    def _evict(self):
        keep = sorted(self.trans, key=lambda s: (self.has_win_from(s), self.visits.get(s, 0)),
                      reverse=True)[:MAX_STATES // 2]
        keep = set(keep)
        if self.root is not None:
            keep.add(self.root)
        for s in list(self.trans):
            if s not in keep:
                self.trans.pop(s, None)
                self.n_ctl.pop(s, None)
                self.visits.pop(s, None)
                self.val.pop(s, None)
                self.evicted += 1

    def has_win_from(self, s):
        return any(v == WIN for v in self.trans.get(s, {}).values())

    def has_win(self):
        return self.wins > 0

    def is_frontier(self, s):
        """A state with a control not yet tried (its controls known), or a
        state never stood on (controls unknown, so something is untried)."""
        n = self.n_ctl.get(s)
        if n is None:
            return True
        return len(self.trans.get(s, {})) < n

    def has_frontier(self, start=None):
        """Is a frontier state reachable from `start` (default: the root)?"""
        return self._bfs(start if start is not None else self.root, self.is_frontier) is not None

    # ----------------------------------------------------------- acting
    def _bfs(self, start, goal):
        if start is None:
            return None
        seen = {start: None}
        q = deque([start])
        while q:
            s = q.popleft()
            if goal(s):
                path = []
                while seen[s] is not None:
                    ps, c = seen[s]
                    path.append(c)
                    s = ps
                return path[::-1]
            for c, s2 in self.trans.get(s, {}).items():
                if s2 in (WIN, OVER):
                    continue
                if s2 not in seen:
                    seen[s2] = (s, c)
                    q.append(s2)
        return None

    # ------------------------------------------------- the goal (patch 48)
    def set_goal(self, rk, schema, v0):
        """Pursue the relation `rk` (its schema, its value on the life-start
        board).  A new relation starts a new value map."""
        if self.pursued is not None and self.pursued[0] == rk:
            return
        self.pursued = (str(rk), str(schema), float(v0))
        self.val = {}

    def clear_goal(self):
        self.pursued = None
        self.val = {}

    def note_value(self, state, v):
        if self.pursued is not None and state is not None and v is not None:
            self.val[state] = float(v)

    def at_goal(self, state):
        """The pursued relation is at its goal on `state` (REACH_FRAC of the
        job or less remains)."""
        if self.pursued is None or state is None:
            return False
        v = self.val.get(state)
        return v is not None and self.pursued[2] > 0 and v <= REACH_FRAC * self.pursued[2]

    def _best_frontier(self, start):
        """With a goal: the reachable frontier state where the pursued relation
        is nearest its goal (ties: the nearest state); a state never valued
        ranks last.  The path of controls to it, or None."""
        if start is None:
            return None
        seen = {start: None}
        q = deque([(start, [])])
        best = None
        inf = float('inf')
        while q:
            s, p = q.popleft()
            if s != start and self.is_frontier(s):
                v = self.val.get(s, inf)
                if best is None or (v, len(p)) < (best[0], best[1]):
                    best = (v, len(p), p)
            for c, s2 in self.trans.get(s, {}).items():
                if s2 == WIN or s2 == OVER or s2 in seen:
                    continue
                seen[s2] = None
                q.append((s2, p + [c]))
        return best[2] if best is not None else None

    def act(self, state, controls):
        """The control to take from `state` now, or None.  STATELESS."""
        self.asks += 1
        self._win_path = False
        self.last_pick = None
        if state is None or not controls:
            self.lost += 1
            return None
        self.see(state, controls)
        # a known win: the shortest path to it (searched only when one is known)
        p = self._bfs(state, lambda s: self.has_win_from(s)) if self.wins else None
        if p is not None:
            if p:
                c = p[0]
            else:
                c = next(k for k, v in self.trans[state].items() if v == WIN)
            self._win_path = True
            self.plans += 1
            self.last_pick = (state, c)
            return c
        # something untried here
        un = self.untried(state, controls)
        if un:
            self.tries += 1
            self.last_pick = (state, un[0])
            return un[0]
        # the nearest state with something untried -- or, with a goal from
        # the library, the reachable one where the goal is nearest (patch 48)
        if self.pursued is not None and self.plan_miss < 3 and len(set(self.val.values())) >= 2:
            # best-first toward the goal -- unless three of his planned steps
            # this life landed elsewhere (a non-Markov game: fixating on one
            # target would bounce him until the clock; nearest, as patch 45)
            p = self._best_frontier(state)
        else:
            p = self._bfs(state, lambda s: s != state and self.is_frontier(s))
        if p:
            self.plans += 1
            self.last_pick = (state, p[0])
            return p[0]
        self.lost += 1
        return None

    def proposed(self, state, control):
        return self.last_pick is not None and self.last_pick[0] == state and self.last_pick[1] == control

    def win_path(self):
        return bool(self._win_path)

    def walking(self):
        """Is this visit a WALK?  From WALK_MIN_LIVES lives on, judge each
        terminal: the pooled new-states-per-step rate of the later half of the
        visit's lives against the earlier half's; 'walk' when it has not
        halved.  True when the last WALK_RUN judgements all said walk.  The
        life in flight counts as the latest life.  (patch 47, measured)"""
        ser = list(self.hist)
        if self.steps_this_life > 0:
            ser.append((int(self.steps_this_life), int(self.new_this_life)))
        n = len(ser)
        run = 0
        if n >= WALK_MIN_LIVES:
            cs = [0]
            cn = [0]
            for st, nw in ser:
                cs.append(cs[-1] + st)
                cn.append(cn[-1] + nw)
            for i in range(WALK_MIN_LIVES, n + 1):
                h = i // 2
                sa = cs[h] - cs[0]
                sb = cs[i] - cs[h]
                ra = (cn[h] - cn[0]) / float(sa) if sa > 0 else 0.0
                rb = (cn[i] - cn[h]) / float(sb) if sb > 0 else 0.0
                if ra > 0 and rb >= 0.5 * ra:
                    run += 1
                else:
                    run = 0
        self.walk_run = run
        return run >= WALK_RUN

    def _testing_converge(self):
        """The patch-47 hold: progress this life and a frontier reachable,
        unless the visit is a walk.  Same exits as the patch-45 body for a
        known win, an evicting life and a life in which nothing recurred."""
        if self.has_win():
            return False
        if self.evicted > self.evicted_at_life:
            return False
        if self.steps_this_life <= 0:
            return False
        if self.proc_lives <= 1:
            return self.has_frontier()
        if self.new_this_life >= self.steps_this_life:
            return False
        if self.new_this_life <= 0 and self.untried_this_life <= 0:
            return False
        if self.walking():
            if self._walk_logged != self.lives:
                self._walk_logged = self.lives
                self.events.append('WALK run=%d hist=%d' % (self.walk_run, len(self.hist)))
            return False
        return self.has_frontier()

    def testing(self):
        """Hold him through a terminal: the life just ended found states
        never seen before (or is the first of the process with anything
        learned), and somewhere untried is reachable from the root.
        Never a cage (review 2, finding 1): a known win is walked from the
        first step of the next visit, nothing to hold for; a life in which
        the table evicted is thrashing, its 'new' states are eviction, not
        discovery; a life in which no state recurred at all is a key that
        cannot earn here (the recurrence law), not a search."""
        if _CONVERGE_ON():
            return self._testing_converge()
        if self.has_win():
            return False
        if self.evicted > self.evicted_at_life:
            return False
        if self.proc_lives <= 1 and self.steps_this_life > 0:
            return self.has_frontier()
        if self.new_this_life <= 0 or self.new_this_life >= self.steps_this_life:
            return False
        return self.has_frontier()

    # ------------------------------------------------------ persistence
    def to_dict(self):
        keep = sorted(self.trans, key=lambda s: (self.has_win_from(s), self.visits.get(s, 0)),
                      reverse=True)[:SAVE_STATES]
        if self.wins and self.root is not None:
            # the states along the path root -> WIN are kept whatever their visits
            p = self._bfs(self.root, lambda s: self.has_win_from(s))
            if p is not None:
                s = self.root
                path_states = [s]
                for c in p:
                    s = self.trans.get(s, {}).get(c)
                    if s in (None, WIN, OVER):
                        break
                    path_states.append(s)
                keep = list(dict.fromkeys(path_states + keep))[:SAVE_STATES + len(path_states)]
        trans = {}
        for s in keep:
            row = {}
            for c, v in self.trans[s].items():
                row[ctl_key(c)] = 'WIN' if v == WIN else ('OVER' if v == OVER else v)
            trans[s] = row
        n_ctl = {s: int(self.n_ctl[s]) for s in keep if s in self.n_ctl}
        return {'v': 1, 'trans': trans, 'n_ctl': n_ctl, 'root': self.root,
                'lives': int(self.lives), 'clears': int(self.clears),
                'libref': sorted(self.lib_refuted)[:256]}

    def from_dict(self, d):
        n = 0
        try:
            for s, row in (d.get('trans') or {}).items():
                r = self.trans.setdefault(s, {})
                for cs, v in row.items():
                    c = key_ctl(cs)
                    if c not in r:
                        r[c] = WIN if v == 'WIN' else (OVER if v == 'OVER' else sys.intern(str(v)))
                        if v == 'WIN':
                            self.wins += 1
                        n += 1
            for s, k in (d.get('n_ctl') or {}).items():
                if s not in self.n_ctl:
                    self.n_ctl[s] = int(k)
            if d.get('root') and self.root is None:
                self.root = d['root']
            self.lives = max(self.lives, int(d.get('lives') or 0))
            self.clears = max(self.clears, int(d.get('clears') or 0))
            self.lib_refuted.update(str(x) for x in (d.get('libref') or []))
            if len(self.trans) > MAX_STATES:
                self._evict()
        except (TypeError, ValueError, AttributeError, IndexError):
            self.errors += 1
        return n
