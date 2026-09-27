"""GO SOMEWHERE YOU HAVE NOT BEEN.

Written 2026-08-27 for the user's no-premise experiment: mirror a human
playing ARC.  Every number below was measured before a line was written.

THE HUMAN LOOP, and what each piece cost to establish:

  1. WHICH THING AM I?  An object whose displacement is predicted by the
     action.  Findable in about half the games -- re86 +63.0% lift over
     a +0.8% null, cn04 +36.7% vs +2.7%, sp80 +62.7%.  Absent in ft09,
     m0r0, su15 (98-100% base rate: nothing moves).
     ! Measure LIFT over the modal displacement, never raw accuracy --
       (0,0) is modal, so raw accuracy scores 100% for every blob.

  2. WHAT DO MY ACTIONS DO?  Keyed on the EGOCENTRIC 5x5 patch:
         key      distinct   recurs   predicts my move
         board    56-1166    10-51%   100%
         ego5       3-50     98-100%  83-98%   (null 26-39%)
     The board is precise and never recurs; the glance recurs 13% and
     cannot say where.  The ego patch recurs essentially always and
     still earns.  That is the index the recurrence law asks for.

  3. WHERE HAVE I BEEN?  His avatar has touched 3-16% of the reachable
     box; 7-56 distinct cells in a single life.  So unvisited PLACES are
     everywhere, and 'go somewhere new' is a real destination.

  ! NOT option-frontier planning.  A board with an untried action is
    always underfoot (median distance 0 in every game).  That design was
    measured and refuted.  This one navigates SPACE, not option-space.

COST CONTROL: full connected-component extraction runs only while he is
still identifying himself (capped).  After that he is located by a single
colour scan, which is one pass over the grid.  85-99% of his CPU is
already the save path, so this must not add a second heavy loop.

Returns None whenever it does not know something -- no self, no model, no
unvisited target.  The caller then behaves exactly as it would without
this module, so it can only ever ADD an ordering.
"""
from collections import Counter, defaultdict, deque

IDENT_STEPS = 400      # frames spent identifying the self, per game
IDENT_MIN_N = 60       # observations before a self may be named
IDENT_MIN_LIFT = 0.25  # action must beat the modal displacement by this
IDENT_MIN_SIZE = 2     # a 1-cell 'self' of the background colour is an
                       # artifact: ar25/ka59/tr87 all named colour 0 size 1
EGO_R = 2              # egocentric patch radius (5x5)
MAX_CELLS = 20000      # visited-cell cap per (game, level)

# A SELF NAMED ON 400 FRAMES MUST STILL EARN ON 6,000.
# Persisting self_id turned a marginal self from INERT (self_id None =>
# every method returns None and the caller behaves exactly as it would
# without this module) into one he ACTS on for good.  Measured over the
# five selves he had actually named, n=4,991-6,884 each:
#     ar25 (5,40)  +36.2    tr87 (0,7)   +28.3
#     ls20 (12,10) +20.1    sc25 (10,8)  +14.0
#     g50t (9,24)   -0.2  <-- the action explains NOTHING
# One in five earned nothing at all, and three of five could not
# reproduce the 0.25 they were named at.  A self is named on at most
# IDENT_STEPS frames; if on EARN_MIN_N more it cannot show even a fifth
# of IDENT_MIN_LIFT, the naming did not replicate and he goes back to
# behaving as though he had no self -- which is what he did before.
EARN_MIN_N = 300               # displacements before a self is judged
EARN_MIN_LIFT = IDENT_MIN_LIFT / 5.0


def _SELFCHECK_OFF():
    """Kill switch, `touch /root/SELFCHECK_OFF`, no restart.

    A KILL SWITCH rather than the project's usual `<GATE>_ON` file: this
    is a correctness guard, not an experiment, and defaulting it off
    would ship a self he is known not to be entitled to act on.  It can
    only ever make him QUIETER -- every path it takes returns None, which
    is precisely what he did before a self was persisted -- so it cannot
    introduce a steer, only withhold one.
    """
    try:
        import os as _os
        return _os.path.exists("/root/SELFCHECK_OFF")
    except Exception:
        return False


def _components(flat, h, w, maxobj=300):
    """Connected same-colour regions, 4-neighbour, background = modal."""
    seen = bytearray(h * w)
    bg = Counter(flat).most_common(1)[0][0]
    out = []
    for i in range(h * w):
        if seen[i] or flat[i] == bg:
            continue
        col = flat[i]
        stack = [i]
        seen[i] = 1
        cells = []
        while stack:
            j = stack.pop()
            cells.append(j)
            y, x = divmod(j, w)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    k = ny * w + nx
                    if not seen[k] and flat[k] == col:
                        seen[k] = 1
                        stack.append(k)
        ys = [c // w for c in cells]
        xs = [c % w for c in cells]
        out.append((col, len(cells),
                    (sum(ys) / len(ys), sum(xs) / len(xs))))
        if len(out) >= maxobj:
            break
    return out


class Explorer(object):
    """One per game.  Knows who he is, what his actions do, where he has
    been, and which way is somewhere new."""

    def __init__(self):
        self.self_id = None          # (colour, size) once identified
        self.ident_n = 0
        self._ident = defaultdict(lambda: defaultdict(Counter))
        self.model = defaultdict(Counter)   # (ego, action) -> displacement
        self.visited = defaultdict(set)     # level -> {(y, x)}
        self.here = None
        self.disp_all = Counter()            # displacement, since naming
        self.disp_by = defaultdict(Counter)  # action -> displacement
        self.steps = 0
        self.unearned = 0
        # THE ONE PREDICTION ON RECORD IS UNTESTABLE WITHOUT THIS.
        # [[p_the_no_premise_experiment]] put it in writing: the
        # egocentric key should help MORE on L1 than on L0 -- the reverse
        # of every other organ, because board keys recur 0.0% across the
        # L0->L1 boundary (0/432) while the ego patch is translation
        # invariant.  Nothing recorded a single explorer event by level,
        # so the claim has never been checkable in either direction.
        # level -> Counter(obs, no_self, ask, target, unusable, no_move,
        #                  all_visited)
        self.by_level = defaultdict(Counter)
        self.targets = 0
        self.plans = 0
        self.no_self = 0

    # ---- perception -------------------------------------------------
    def locate(self, flat, h, w):
        """Where am I?  A colour scan once known; components until then."""
        if self.self_id is None:
            return None
        col, size = self.self_id
        ys = xs = n = 0
        for i, v in enumerate(flat):
            if v == col:
                ys += i // w
                xs += i % w
                n += 1
        if not n:
            return None
        return (ys / float(n), xs / float(n))

    def ego(self, flat, cy, cx, h, w):
        out = []
        iy, ix = int(round(cy)), int(round(cx))
        for dy in range(-EGO_R, EGO_R + 1):
            for dx in range(-EGO_R, EGO_R + 1):
                y, x = iy + dy, ix + dx
                out.append(flat[y * w + x] if 0 <= y < h and 0 <= x < w
                           else 255)
        return bytes(out)

    # ---- learning ---------------------------------------------------
    def observe(self, flat, h, w, action, level):
        """One step.  Learn who I am, what that action did, where I am."""
        self.steps += 1
        if self.self_id is None and self.ident_n < IDENT_STEPS:
            self.ident_n += 1
            obs = _components(flat, h, w)
            prev = getattr(self, '_prev_obs', None)
            if prev is not None and action is not None:
                pm = {}
                for c, s, ctr in prev:
                    pm.setdefault((c, s), []).append(ctr)
                for c, s, ctr in obs:
                    cand = pm.get((c, s))
                    if not cand:
                        continue
                    p0 = min(cand, key=lambda q: abs(q[0] - ctr[0])
                             + abs(q[1] - ctr[1]))
                    d = (int(round(ctr[0] - p0[0])),
                         int(round(ctr[1] - p0[1])))
                    self._ident[(c, s)][action][d] += 1
            self._prev_obs = obs
            self._try_name_self()
            return
        self.by_level[int(level)]['obs'] += 1
        pos = self.locate(flat, h, w)
        if pos is None:
            self.no_self += 1
            self.by_level[int(level)]['no_self'] += 1
            return
        cell = (int(round(pos[0])), int(round(pos[1])))
        vis = self.visited[level]
        if len(vis) < MAX_CELLS:
            vis.add(cell)
        prev = self.here
        self.here = (pos, self.ego(flat, pos[0], pos[1], h, w))
        if prev is not None and action is not None:
            d = (int(round(pos[0] - prev[0][0])),
                 int(round(pos[1] - prev[0][1])))
            self.model[(prev[1], action)][d] += 1
            # Same arithmetic the naming used, on every frame since.
            self.disp_all[d] += 1
            self.disp_by[action][d] += 1

    def _try_name_self(self):
        """The object whose displacement the ACTION explains best."""
        best = None
        for oid, acts in self._ident.items():
            allc = Counter()
            for cc in acts.values():
                allc.update(cc)
            n = sum(allc.values())
            if n < IDENT_MIN_N or oid[1] < IDENT_MIN_SIZE:
                continue
            base = allc.most_common(1)[0][1] / float(n)
            cond = sum(cc.most_common(1)[0][1]
                       for cc in acts.values()) / float(n)
            if best is None or (cond - base) > best[1]:
                best = (oid, cond - base)
        if best is not None and best[1] >= IDENT_MIN_LIFT:
            self.self_id = best[0]
            self._ident = defaultdict(lambda: defaultdict(Counter))
            self._prev_obs = None

    # ---- deciding ---------------------------------------------------
    def toward_new(self, n_actions, level):
        """An action that takes me somewhere I have not been, or None.

        Greedy over the learned model: for each action, where does it put
        me?  Prefer a cell never stood on.  None when he has no self, no
        model for here, or every predicted landing is already visited --
        the caller then decides exactly as it would have.
        """
        _lv = self.by_level[int(level)]
        _lv['ask'] += 1
        if self.here is None or not self.usable():
            _lv['unusable'] += 1
            return None
        pos, eg = self.here
        vis = self.visited[level]
        best = None
        _had_model = False
        for a in range(int(n_actions)):
            cc = self.model.get((eg, a))
            if not cc:
                continue
            _had_model = True
            d = cc.most_common(1)[0][0]
            if d == (0, 0):
                continue
            land = (int(round(pos[0] + d[0])), int(round(pos[1] + d[1])))
            if land in vis:
                continue
            conf = cc.most_common(1)[0][1] / float(sum(cc.values()))
            if best is None or conf > best[1]:
                best = (a, conf)
        if best is None:
            # Distinguish the three silences: no model for this patch at
            # all, versus a model whose every landing is already stood on.
            _lv['no_model' if not _had_model else 'all_visited'] += 1
            return None
        _lv['target'] += 1
        self.targets += 1
        return best[0]

    def earns(self):
        """How much knowing the ACTION beats the modal displacement.

        None until EARN_MIN_N displacements have been seen -- an
        unjudged self is used, exactly as it was before.
        """
        n = sum(self.disp_all.values())
        if n < EARN_MIN_N:
            return None
        base = self.disp_all.most_common(1)[0][1] / float(n)
        cond = sum(c.most_common(1)[0][1]
                   for c in self.disp_by.values()) / float(n)
        return cond - base

    def usable(self):
        """A named self he is still entitled to act on."""
        if self.self_id is None:
            return False
        lift = self.earns()
        if lift is not None and lift < EARN_MIN_LIFT:
            self.unearned += 1
            return bool(_SELFCHECK_OFF())
        return True

    def moves_me(self, action, min_n=4):
        """Does this action move me from where I stand?

        Returns None when unknown, else the fraction of observed
        outcomes with a NON-zero displacement.  Keyed on the
        egocentric patch, which recurs 98-100% and predicts
        displacement 83-98% -- so it speaks about boards never
        seen, which board memory cannot.
        """
        if self.here is None or not self.usable():
            return None
        try:
            cc = self.model.get((self.here[1], int(action)))
            if not cc:
                return None
            tot = sum(cc.values())
            if tot < int(min_n):
                return None
            still = cc.get((0, 0), 0)
            return 1.0 - (still / float(tot))
        except Exception:
            return None

    def stats(self):
        return {
            'self_id': list(self.self_id) if self.self_id else None,
            'ident_n': self.ident_n,
            'ego_states': len({k[0] for k in self.model}),
            'model_entries': len(self.model),
            'visited_cells': sum(len(v) for v in self.visited.values()),
            'steps': self.steps,
            'targets': self.targets,
            'no_self': self.no_self,
            'by_level': {str(k): dict(v)
                         for k, v in self.by_level.items() if v},
            'earns': self.earns(),
            'usable': self.usable(),
            'unearned': self.unearned,
        }

    # ---- persistence -------------------------------------------------
    # HE FORGOT WHICH THING HE WAS, ON EVERY RESTART (2026-08-28).
    # ARCWorld._explorers is a class dict and NOTHING saved or loaded it,
    # while IDENT_STEPS=400 frames must be spent per game before a self
    # can be named at all.  Measured live: 8,508 arc steps over 25 games
    # in 1h13m = 340 frames/game against that 400-frame budget, and
    # today's restart intervals were 333s, 817s, 922s, 1695s, 2974s,
    # 2982s, 5327s, 5918s.  He was usually killed before he finished
    # working out which thing he was, and then began again from nothing.
    # Where he DOES name himself the model is worth having: sp80 predicts
    # its displacement at 85.8% against a 32.1% modal null (+53.7, held
    # out, n=374).
    # _ident is deliberately NOT carried.  A single-action game has
    # cond == base by construction, so its lift is identically 0 and no
    # amount of extra evidence can name a self there -- measured +0.000
    # in lp85/r11l/vc33/cd82 across three independent windows.  Carrying
    # a doomed accumulator would only make the failure permanent.
    def to_dict(self):
        """Serialisable state.  Only what cost frames to learn."""
        return {
            'self_id': list(self.self_id) if self.self_id else None,
            'ident_n': int(self.ident_n),
            'model': [
                [eg.hex(), int(a),
                 [[int(d[0]), int(d[1]), int(c)] for d, c in cc.items()]]
                for (eg, a), cc in self.model.items() if cc
            ],
            'visited': {str(lv): [[int(y), int(x)] for (y, x) in cells]
                        for lv, cells in self.visited.items() if cells},
            'steps': int(self.steps),
            'targets': int(self.targets),
            'by_level': {str(k): dict(v)
                         for k, v in self.by_level.items() if v},
            'disp_all': [[int(d[0]), int(d[1]), int(c)]
                         for d, c in self.disp_all.items()],
            'disp_by': [[int(a), [[int(d[0]), int(d[1]), int(c)]
                                  for d, c in cc.items()]]
                        for a, cc in self.disp_by.items()],
        }

    @classmethod
    def from_dict(cls, d):
        """Rebuild from to_dict().  A fresh Explorer on anything
        malformed -- a bad file must cost him nothing but the frames he
        would have spent anyway."""
        e = cls()
        try:
            sid = d.get('self_id')
            e.self_id = (int(sid[0]), int(sid[1])) if sid else None
            # A game that has NOT named a self resumes with ident_n = 0,
            # so it gets exactly the fresh attempt it gets today and
            # persistence can never lock one into permanent failure.
            e.ident_n = int(d.get('ident_n', 0)) if e.self_id else 0
            for eg, a, cnt in (d.get('model') or []):
                cc = e.model[(bytes.fromhex(eg), int(a))]
                for dy, dx, c in cnt:
                    cc[(int(dy), int(dx))] += int(c)
            for lv, cells in (d.get('visited') or {}).items():
                s = e.visited[int(lv)]
                for y, x in cells:
                    if len(s) < MAX_CELLS:
                        s.add((int(y), int(x)))
            e.steps = int(d.get('steps', 0))
            e.targets = int(d.get('targets', 0))
            for lv, cnt in (d.get('by_level') or {}).items():
                for k, v in (cnt or {}).items():
                    e.by_level[int(lv)][str(k)] += int(v)
            for dy, dx, c in (d.get('disp_all') or []):
                e.disp_all[(int(dy), int(dx))] += int(c)
            for a, cnt in (d.get('disp_by') or []):
                for dy, dx, c in cnt:
                    e.disp_by[int(a)][(int(dy), int(dx))] += int(c)
        except Exception:
            return cls()
        return e
