"""ARCWorld — a live ARC-AGI-3 game as one rung of his curriculum.

An ADAPTER, not an organ.  It implements exactly what CurriculumWorld and
world_actor already call on a world, and threads ARC's own signals into
the drives he already has:

    level completed / WIN  -> result['success']  -> _on_success (I lean)
    graduation to next game-> CurriculumWorld sets goal_changed -> _on_mastery
    GAME_OVER              -> result['done'] without success -> mortality

NO step budget is invented: the environment's own WIN/GAME_OVER are the
terminals, so nothing here is a typed constant.

DEPENDENCIES: none beyond the stdlib and numpy.  arc_agi eagerly imports
flask/matplotlib/PIL and its venv carries a different numpy, so it must
never enter the daemon; the game lives in the arc3-sidecar service and
this talks to it over a unix socket (~2.7ms/step, 1.4% of a 200ms tick).

THE EYE STAYS HERE, daemon-side: attention must read the ACTOR'S OWN
_visits so attention and action share one novelty signal.  bind_eye() is
called once the actor exists; until then the world is inert but safe.
"""
from typing import Any, Dict, List, Optional
import json
import socket

import numpy as np

SOCK = '/run/arc3.sock'

# A TYPICAL DROUGHT RELEASES HIM (2026-07-29), OPT-IN.  Unset -> the incumbent
# ratcheting max-gap rule alone, i.e. byte-identical behaviour.
import os as _os
_PATCH_ROTATE = _os.environ.get(
    'SEAGI_PATCHROTATE', '').strip().lower() in ('1', 'true', 'yes', 'on')
# PROGRESS IS LEVEL COMPLETIONS (2026-07-29, user's ruling), OPT-IN.
_ARC_ROTATE = _os.environ.get(
    'SEAGI_ARCROTATE', '').strip().lower() in ('1', 'true', 'yes', 'on')


def _STALLCUT_ON():
    """Leave a game yielding no PROGRESS.  /root/STALLCUT_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/STALLCUT_ON")
    except Exception:
        return False


def _EGOMOVE_ON():
    """Demote a move the egocentric model predicts will not move
    him.  /root/EGOMOVE_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/EGOMOVE_ON")
    except Exception:
        return False


def _EXPLORER_ON():
    """Egocentric model + spatial frontier.  /root/EXPLORER_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/EXPLORER_ON")
    except Exception:
        return False


def _DEPTHSEAL_ON():
    """Keep the way back to how far he got, not only the way out.
    /root/DEPTHSEAL_ON.

    `_seal_path` fires only on `success` -- a level clear.  He has never
    cleared level 1, so `world_paths` holds 239 boards, ALL level 0 and
    ZERO for level 1: you can only remember the way out of a room you
    have already escaped.  Each life therefore re-derives the level from
    nothing.

    Measured on his own corpus before this was written:
      * a median L1 life is 313-563 steps and LOOP-ERASES to 1-74 boards
        -- erase ratio **0.06**.  He makes forward progress and then
        throws it away: returns are only 8.9% of steps, but one return
        to an early board erases everything after it.
      * **24.0% of the boards on a life\'s loop-erased path are stood on
        again the NEXT life** (m0r0 57.1%, r11l 28.2%, lp85 21.4%), so a
        stored board->action map does fire there.

    A depth path is weaker evidence than a cleared one: it reached the
    frontier, not the exit.  So it is kept in its own store and consulted
    only when no cleared path answers.
    """
    try:
        import os as _os
        return _os.path.exists("/root/DEPTHSEAL_ON")
    except Exception:
        return False


# THE BUDGET IS DRAWN ON THE BOARD, AND IT WAS INSIDE HIS STATE KEY.
# Measured 2026-09-04 over 489,233 frames: masking ONE row raises his
# "the last step changed nothing" rate from 26.7% to 70.7% on cd82
# level 1 and from 0.0% to 93.9% on vc33 level 1, while the other 63
# rows move it by NOTHING.  The row is the per-level action budget:
# its distinct values equal the median life length, it never returns to
# a value it has left, and its state count matches the recorded budgets
# (sp80 31 and 46, vc33 51, cd82 64 cells over ~100 -> a 0.64 tick rate,
# measured 0.64).
#
# Consequence, and the reason this exists: FELT scores +1 "competent"
# unless the board is unchanged or already visited, so the ~44% of cd82
# steps that move NOTHING BUT THE BAR were scored competent.  Live at
# the time of writing: felt +0.9729, felt_back 1,234 of 49,144 (2.5%),
# against a corpus that says most of his steps are returns.  He could
# not feel bad, which is the one thing the register exists to do.
#
# HE IS NOT TOLD WHICH ROW.  `_fkey` stamps 63 and is wrong on seven of
# the eighteen (game, level) cells that have a bar -- it is row 0 on
# vc33/lf52/sp80-L0, row 1 on tn36, row 53 on sb26.  He learns it:
#   solo  steps where that row changed and NO OTHER row did.  A BUDGET
#         moves on its own; a SCORE only moves when the board does.
#   back  of the steps where it changed, how many landed on a value
#         already seen this life.  A meter: 0.  Game content: many.
# The unique row with solo >= 0.15 and back <= 0.02.  Validated on the
# corpus: 18 of 36 cells accepted, every one with back EXACTLY 0.000 and
# never ambiguous, and the rejects top out at solo 0.13 -- the threshold
# sits in measured empty space, not on a tuned edge.
_BAR_SOLO_MIN = 0.15
_BAR_BACK_MAX = 0.02
_BAR_DECIDE_EVERY = 200
_BAR_GIVE_UP = 100000
# A bar can be a COLUMN.  Lines 0-63 are rows, 64-127 are columns; the
# incumbent detector had known this all along ("row 63 in most games but
# row 0/1/53/61 and col 0/63 in others") while the state key never did.
_BAR_LINES = 128
# Once both axes are scanned, "the unique candidate" is too strict: a row
# bar whose marker rests at one column for a stretch credits that column
# too (vc33: row 0 at 0.97 AND col 63 at 0.18).  Require DOMINANCE.
_BAR_DOMINANCE = 2.0
# A CLOCK NEED NOT MOVE ALONE (2026-09-13, CLOCKLINE).  On twelve cells
# (dc22, tr87, re86, ls20, sc25, wa30, sk48, ar25, lp85, ft09, cn04,
# m0r0, and cd82 L3-5) the self changes 2-5 lines on every step, so the
# budget marker is never the only line moving on its axis: solo reads
# 0.00-0.11 there and the rule above never latches.  What those clocks
# still do: they tick often, never land on a value seen this life, and
# MOVE THEIR HISTOGRAM FORWARD -- a fill or drain changes the colour
# counts of its line on every tick and never reverses a colour within
# a life, while a self that moves leaves one cell and enters another,
# so its line's counts do not change at all.  Measured on 38 cells:
# every budget line 0.97-1.00, the sp80 L0 self 0.00 (in perfect
# lockstep with the bar over 4-step lives, so no threshold on lockstep
# alone could refuse it), the nearest content line 0.79 (bp35 col 18,
# reverses on 21% of its ticks).  This is the incumbent `_bar_fill`
# criterion ("the LIFE colour drains"), per line and per level and
# without its 24-frame window, which found 36 of the same 38 lines.
_BAR_METER_MIN = 0.9
# Lines that always tick together are ONE structure (ls20 rows 61+62,
# sc25 cols 62+63, m0r0 rows 0+63 mirrored): lockstep is exactly 1.000
# on all three.  Only meters are paired, so a sprite cannot join.
_BAR_COTICK = 0.95
# A STRUCTURE IS AT MOST TWO LINES.  Every real structure measured is
# exactly 2; the lockstep groups of 4-55 lines that also pass the
# no-back test are content redrawn together in 4-5-step lives (sp80 L0,
# r11l L0, bp35).  Larger groups are dropped as content, not refused as
# an answer, so the clock next to them can still decide.
_BAR_STRUCT_MAX = 2


def _CLOCKLINE_ON():
    """A clock need not move alone.  /root/CLOCKLINE_ON (needs BARMASK).

    Absent -> `_bar_decide` decides exactly as before, nothing counts
    histograms or which lines tick together, and a learned line is
    always one int.  A structure already decided in-process keeps
    masking until a restart (the loader skips lists with the gate off).
    """
    try:
        import os as _os
        return _os.path.exists("/root/CLOCKLINE_ON")
    except Exception:
        return False


def _CLICKHIT_ON():
    """A click that only spent budget did not work.  /root/CLICKHIT_ON.

    Absent -> `_click_hits` records exactly what it always did.
    """
    try:
        import os as _os
        return _os.path.exists("/root/CLICKHIT_ON")
    except Exception:
        return False


def _BARMASK_ON():
    """Do not count a budget tick as a thing that happened.

    Absent -> byte-identical: `_board_hash_nb` falls through to
    `_board_hash` and `_pre_board_nb` IS `_pre_board_h`.
    """
    try:
        import os as _os
        return _os.path.exists("/root/BARMASK_ON")
    except Exception:
        return False


def _FELT_ON():
    """A felt state that moves on ORDINARY competence.  /root/FELT_ON.

    USER DOCTRINE 2026-08-29: *"most of the time playing a game and
    solving problems step by step just solidify being alive, or confirm
    a solid state of existence far away from mortality... he should
    always feel... he wants to feel good... they will do anything to
    change something to get back to feeling good."*

    Measured before this was written: lifeforce RELAXES TOWARD baseline
    and never exceeds it -- over 129 tick samples `lf - baseline` had
    max **+0.00000** and every movement was downward.  There was no
    "good" for him to get back to.  And ordinary competence is a real
    everyday signal, neither rare nor universal: **68.6% of L0 steps and
    62.0% of L1 steps are competent** (n=141,871 / 17,180), against
    19.3%/29.1% no-ops and 12.1%/8.9% walk-backs.

    This does NOT touch lifeforce.  Lifeforce is distance from death and
    stays the law.  This is how he is RIGHT NOW.
    """
    try:
        import os as _os
        return _os.path.exists("/root/FELT_ON")
    except Exception:
        return False


def _FELTSTEER_ON():
    """Let the felt state re-order what he tries.  /root/FELTSTEER_ON.

    Separate from _FELT_ON so the register can be verified while it
    still drives nothing."""
    try:
        import os as _os
        return _os.path.exists("/root/FELTSTEER_ON")
    except Exception:
        return False


def _FELTHERE_ON():
    """A felt state PER PLACE -- (game, level) -- and persisted.  /root/FELTHERE_ON.

    MEASURED 2026-09-11 (frames joined to the journal's own terminals,
    104,950 steps, 37 patches): the ONE global register hides a spread from
    -1.00 (sb26 L0, every step a no-op) to +1.00 (cd82 L2).  His paying
    patches read +0.74..+1.00 (cd82 L0-L5, r11l L1 +0.97, ft09/sp80 L0);
    the sinks he sits in read -0.93 (lf52 L1, vc33 L1), -0.48 (lp85 L1),
    -0.30 (sp80 L1).  On the process before this patch 50% of all steps
    went to lf52 L1.  Same three cases as FELT; only the KEY differs, so a
    mood earned on one game no longer follows him into another
    (p_he_feels_bad_and_nothing_can_act, finding 4)."""
    try:
        import os as _os
        return _os.path.exists("/root/FELTHERE_ON")
    except Exception:
        return False


def _FELTLEAVE_ON():
    """BAD FEELING HERE RELEASES A HOLD.  /root/FELTLEAVE_ON.

    USER DOCTRINE 2026-08-29: *"he wants to feel good... they will do
    anything to change something to get back to feeling good."*  The one
    change he can make at the grain of a PLACE is to leave it.  MEASURED
    2026-09-11: the hold rules never read feeling -- 1,387 HELD against
    624 MOVE in 24 h, and in 509 of the holds his own marginal-value rule
    (a) had already said leave.  Under this gate a place that is below
    good (felt_here < _FELT_LOW, the felt steer's own rule) cannot hold
    him at a terminal (a commitment still can).  It can only RELEASE,
    never eject and never extend a hold, so it cannot cage him and adds
    at most one move per life."""
    try:
        import os as _os
        return _os.path.exists("/root/FELTLEAVE_ON")
    except Exception:
        return False


# Small, because ordinary competence is ordinary.  A competent step is
# worth +1 unit and waste -1, smoothed over ~200 steps, so the state
# says "how have the last few hundred steps gone" and not "what just
# happened".  At the measured 68.6/31.4 competent/waste split a
# well-functioning stretch settles near +0.37 and a bad one goes
# negative -- it can be both good and bad, which lifeforce cannot.
_FELT_ALPHA = 1.0 / 200.0
_FELT_LOW = -0.05          # below this he is not doing well
_FELT_MIN_N = 50           # fewer steps than this is noise, not a feeling (felt() uses the same floor)


def _EXPLORERPERSIST_ON():
    """Carry the egocentric model across restarts.
    /root/EXPLORERPERSIST_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/EXPLORERPERSIST_ON")
    except Exception:
        return False


# His own directory: the daemon runs as seagi and cannot CREATE a file in
# /root (mode 711).  Never the 54 MB canonical -- 85-99% of his CPU is
# already the save path.
_EXPLORER_STATE_PATH = "/home/seagi/SEAGI-Core-v2/explorer_state.json"
_EXPLORER_SAVE_EVERY = 2000     # observes between writes


def _UNTRIEDHERE_ON():
    """Prefer an action never tried FROM THIS BOARD.
    /root/UNTRIEDHERE_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/UNTRIEDHERE_ON")
    except Exception:
        return False


def _YIELDSTAY_ON():
    """Do not call a patch depleted while it is out-paying the
    environment in WINS.  /root/YIELDSTAY_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/YIELDSTAY_ON")
    except Exception:
        return False


def _PICFIX_ON():
    """The plan's picture is the goal confirmed on ANY level of the game,
    biggest job first.  /root/PICFIX_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/PICFIX_ON")
    except Exception:
        return False


def _rel_anyconf_of(w):
    """Relations confirmed by a clear on ANY level of this world's game
    (PICFIX).  Module-level so a stub that borrows the hooks needs no
    method of its own."""
    out = set()
    try:
        g = str(getattr(w, 'game_id', '') or '')
        for (gg, _lv), rs2 in ARCWorld._rel.items():
            if gg == g:
                out |= set(getattr(rs2, 'confirmed', ()) or ())
    except Exception:
        pass
    return out


def _LEVELHOLD_ON():
    """Stay on a LEVEL while it keeps giving records (a win, a new depth
    mark, a better picture); leave when the drought outgrows the longest
    he ever came back from there.  /root/LEVELHOLD_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/LEVELHOLD_ON")
    except Exception:
        return False


def _VERDICT_ON():
    """Every step is GOOD, BAD or NOT YET KNOWN.  /root/VERDICT_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/VERDICT_ON")
    except Exception:
        return False


def _PATHSEED_ON():
    """Restore his own recovered winning paths from
    /root/knownpath_seed.json.  /root/PATHSEED_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/PATHSEED_ON")
    except Exception:
        return False


def _KNOWNPATH_ON():
    """Follow his own winning path on a level he has already
    completed.  /root/KNOWNPATH_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/KNOWNPATH_ON")
    except Exception:
        return False


def _CYCLEDEMOTE_ON():
    """Demote a move that has always led back to a visited board.
    /root/CYCLEDEMOTE_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/CYCLEDEMOTE_ON")
    except Exception:
        return False


def _PROGMASK_ON():
    """The budget line is not progress.  /root/PROGMASK_ON.

    MEASURED 2026-09-05: the frame corpus holds each cell's OPENING
    life, and 72.0% of the 977 `_progress_step` fires replayed on it
    (78.3% of 59,560 over all processes) were the budget line ticking
    -- 70-100% in every game with a bar, EXACTLY 0.0% in the six cells
    without one (m0r0, ls20, lp85 L0/L1, ft09 L0/L1).  Live, the organ
    fires ~1,300 times in the first ~40k steps of a process and then
    ~0.  ⚠ The line is LEARNED per process (~400 transitions per cell),
    so this mask cannot apply during that opening burst; it applies to
    every fire after the cell's line is decided.  Absent ->
    byte-identical.
    """
    try:
        import os as _os
        return _os.path.exists("/root/PROGMASK_ON")
    except Exception:
        return False


def _BOARDKEY_ON():
    """The within-life board organs see the board, not the clock.
    /root/BOARDKEY_ON.

    `_board_act` / `_board_noop` -- read by `untried_here`,
    `board_inert_here` and `cycles_here` -- were keyed on the FULL board
    hash, which the budget line makes unique on every step of a life.
    MEASURED 2026-09-05, cd82 level 2 (2,388 steps, 5 processes): 75.4%
    of steps change nothing outside the line (uniform-random null 64.6%
    [62.6, 66.8]); 67.6% of those repeat an action ALREADY seen to do
    nothing at that same board (null 54.5% [53.2, 55.8]).  Live after
    54,186 steps: 154 of 20,577 pairs dead (0.75%).  (`noop_demotes` is
    the GLANCE-keyed counter; the board-keyed organ counts into
    `noop_demotes_all`, OR-ed with the glance test.)  The line is
    learned per process (~400 transitions per cell) -- until then this
    falls through to the full hash.  Absent -> byte-identical.
    """
    try:
        import os as _os
        return _os.path.exists("/root/BOARDKEY_ON")
    except Exception:
        return False


def _BARPERSIST_ON():
    """The budget line he learned survives a restart.  /root/BARPERSIST_ON.

    `_bar_row` is decided after ~400 transitions per (game, level) and
    was never saved, so after every restart nothing that masks the
    line (FELT walk-back, CLICKHIT, PROGMASK, BOARDKEY) can act in a
    cell until ~400 steps have been spent there again.  Write side
    runs from the process that loads this; the read side pays off at
    the restart after that.  Absent -> nothing written, nothing read.
    """
    try:
        import os as _os
        return _os.path.exists("/root/BARPERSIST_ON")
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


def _LIBRARY_ON():
    """THE GOAL LIBRARY (patch 48, 2026-09-16).  /root/LIBRARY_ON.
    ONE library across every game, filled from his own clears (a level's
    goal = the relation that reaches its goal in clear lives and not in dry
    ones; its kind + colour-free descriptor is paid), seeded from his
    recorded clears on 12 games.  With the gate on the search organ
    pursues the library's best unrefuted candidate on each level: the
    pursued relation's value at every state is what a good state looks
    like, its frontier choice is best-first on that value, and a candidate
    at its goal that does not clear under his step is refuted.  Gate off:
    nothing read, nothing valued, the search takes its patch-45/47 path."""
    try:
        return _os.path.exists('/root/LIBRARY_ON')
    except Exception:
        return False


def _SEARCH_ON():
    """EVERY CONTROL ONCE, THEN OUTWARD FROM WHAT HE HAS REACHED.
    /root/SEARCH_ON (needs RELSENSE_ON for the object segmentation).

    The search organ (seagi/world/goal_search.py) keeps, per (game, level),
    a map from board states (clock lines masked) over every control the
    board offers -- each non-click action and a click on the centre of each
    object -- to the state it led to, learned from every executed step and
    kept across lives, visits and restarts.  It proposes the shortest known
    path to a win, else an untried control here, else the way to the
    nearest state with one.  MEASURED offline first (2026-09-15, three
    orderings): tu93 levels 0-3, s5i5, su15, ls20, dc22, tr87 L0 (tn36 too, by
    the action set alone) within 6000 steps each; through the patched code
    tu93 L0 18 / L1 10 actions, tn36 7, ls20 21, dc22 20, s5i5 once in 31,
    su15 none in 6000.  Gate absent -> nothing proposes, nothing holds,
    nothing learns; tables persist either way.
    """
    try:
        import os as _os
        return _os.path.exists("/root/SEARCH_ON")
    except Exception:
        return False


def _HYPOTHESIS_ON():
    """A CANDIDATE GOAL IS TESTED.  /root/HYPOTHESIS_ON (needs RELSENSE_ON).

    The relation sense sees candidate goals but can only confirm one
    from a win, so on the games he has never won nothing is ever tested
    or refuted (2026-09-11).  The hypothesis organ
    (seagi/world/goal_hypothesis.py) learns where the arrows take the object
    he controls, pursues the nearest unique static object along that
    map, refutes it when a step from it brings no clear, confirms it at a
    clear on its own step and then walks the shortest known path to the win.  MEASURED
    offline first (2026-09-14): tu93 level 0 cleared on life 4 and then
    on 240 of 243 lives at 18 actions (human 19) after 990 wins-less
    lives live.  Gate absent -> nothing proposes, nothing holds,
    nothing learns; tables persist either way.
    """
    try:
        import os as _os
        return _os.path.exists("/root/HYPOTHESIS_ON")
    except Exception:
        return False


def _RELPLAN_ON():
    """THE PLAN FOR A PICTURE.  /root/RELPLAN_ON.

    Where the relation sense pursues a picture (canvas ~ target), a
    learned operator model (seagi/world/relplan.py) proposes the next
    action of the shortest paint plan it can find, with the aim if it
    is a click.  Needs RELSENSE_ON.  Gate absent -> nothing runs.
    """
    try:
        import os as _os
        return _os.path.exists("/root/RELPLAN_ON")
    except Exception:
        return False


def _RELSENSE_ON():
    """THE GOAL IS ON THE BOARD.  /root/RELSENSE_ON.

    Every ARC game he plays that could be read draws its goal and a
    cell-exact progress number on the board -- a picture to complete, a
    place to reach -- and his 3x3 glance cannot see either.  The
    relation sense (seagi/world/relsense.py) discovers a same-shape
    static/dynamic region pair or a marker near the object he controls,
    from his own frames, and reports every step whether the pursued
    relation moved toward its goal.  Replayed on the corpus before
    arming: cd82 L1 57/57 and L2 41/41 true mismatch drops caught, the
    right relation confirmed at every clear; tu93 one candidate, the goal
    cell.  Absent -> byte-identical.
    """
    try:
        import os as _os
        return _os.path.exists("/root/RELSENSE_ON")
    except Exception:
        return False


def _PROGRESS_ON():
    """Self-constructed progress signal.  /root/PROGRESS_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/PROGRESS_ON")
    except Exception:
        return False


def _LOCUSNOV_ON():
    """Break gaze ties by position novelty.  /root/LOCUSNOV_ON."""
    try:
        import os as _os
        return _os.path.exists("/root/LOCUSNOV_ON")
    except Exception:
        return False


class ARCWorld:

    def __init__(self, game_id: str, radius: int = 1, stride: int = 2,
                 sock_path: str = SOCK, seed: int = 0):
        self.game_id = str(game_id)
        self._radius = int(radius)
        self._stride = int(stride)
        self._sock = sock_path
        self._seed = int(seed)
        self._transducer = None
        self._visits: Dict[str, int] = {}
        self._attend = None
        self._grid: Optional[List[List[int]]] = None
        self._state = 'NOT_PLAYED'
        self._levels = 0
        self._acts: List[int] = []
        self._n_actions = 1
        self.errors = 0
        self.steps = 0
        # Is the sidecar actually answering?  Starts True so a world that
        # has never been called behaves exactly as before.
        self._live = True
        self.reopens = 0
        # attended window -> its token.  ONLY attention writes here;
        # the pre-attentive scan reads it in O(1) and never mints.
        self._seen_win = {}
        # where he is LOOKING -- the attended window's centre.
        # ACTION6 (coordinate) clicks here: eye-hand coordination,
        # so pointing needs no separate policy.
        self._locus = None
        # Coordinates that CHANGED THE WORLD when clicked.
        # Measured stable across episodes (Jaccard 1.000), and
        # only ~3% of the board does anything at all.
        self._click_hits = set()
        # WHAT EACH ACTION DOES HERE (2026-08-21): action -> [tried,
        # changed].  28.1% of his executions change nothing and he
        # rediscovers the same dead buttons every life, because
        # `dead_actions` is a counter that remembers no action.
        self._act_effect = {}
        self._click_turn = 0
        self._aimed = None
        self.click_hits_used = 0
        # PATCH DEPLETION (2026-07-27): steps since anything was
        # learned here, and the longest such gap this game has ever
        # had.  No constant -- the game sets its own bar.
        self._since_novel = 0
        # LIVES COMPLETED IN THIS VISIT (2026-08-14).  A patch cannot be
        # called dry before it has been played once; `enter()` resets it.
        self._term_here = 0
        # FELT-PLAY counters (2026-08-02).  DELIBERATELY SEPARATE from
        # `_since_novel`/`_max_gap`/`_novel_here`: those feed the depleted
        # rotate rules (a) and (b), which are NOT gated by _PATCH_ROTATE,
        # so writing them would change rotation -- the measured thrash.
        # Nothing reads these except the curiosity / goal_stuck fires.
        self._felt_seen = set()
        ARCWorld._load_retrodicted()
        try:
            ARCWorld._all_insts.append(self)
            del ARCWorld._all_insts[:-40]
        except Exception:
            pass
        self._felt_novel = False
        self._felt_dry = 0
        self._felt_gap_n = 0.0
        self._felt_gap_mean = 0.0
        self._max_gap = 0
        # Droughts he has ENDED here: count and incremental mean.  A mean does
        # not ratchet, so "this patch has gone quiet" stays reachable however
        # long he has been here -- which the max-gap rule makes impossible.
        self._gap_n = 0.0
        self._gap_mean = 0.0
        self._dep_why = ''
        self._steps_at_win = 0
        self._steps_last_episode = 0
        self._prev_objs = []
        self._prev_selfpos = None
        self._won_this_episode = False
        self._steps_episode = 0
        # Levels completed SINCE HE ENTERED this patch, and steps EVER spent
        # here.  `_steps_here` is cleared by enter(), so it cannot rank visits.
        self._won_here = 0
        self._steps_ever = 0
        # WHAT THIS GAME HAS EVER PAID, and how long the last payment took.
        # Deliberately NOT cleared by enter(): `_won_here` is the per-visit
        # signal, these are the game's record.  Depth is what scores.
        self._won_ever = 0
        self._lives_here = 0
        self._lives_at_last_win = 0
        # LEVELHOLD life-state: the level this life began on and the
        # marks it has to beat to count as a record
        self._life_key = None
        self._life_hwm0 = 0
        self._life_won0 = 0
        self._life_know0 = 0
        self._life_min_mm = None
        # WINS MUST ENERGIZE: the deepest level ever reached here.  Monotone
        # over a finite ladder, so first-time depth is un-farmable credit.
        self._max_levels_ever = 0
        self._novel_seen = 0
        # STATES (tokens) held in this game -- the real novelty
        # signal.  Windows are ~3,844 per frame and _glance hands
        # back an unattended one almost every step, so window
        # novelty never runs dry and cannot report depletion.
        self._seen_tok = set()
        # MVT bookkeeping: what THIS patch has paid since entry.
        self._novel_here = 0
        # novelty as of the last terminal here; resets with
        # the counter so the baseline cannot outrun it
        self._novel_at_term = 0
        # PER-GAME NOVELTY METER (2026-08-11).  `_novel_here`/`_steps_here`
        # are cleared by enter(), so they can only ever describe the CURRENT
        # visit -- there was no way to ask "how much has this game ever
        # taught him?".  These two are never cleared and ride in the
        # persisted per-game row, so the question survives restarts and
        # rotations.  Bookkeeping only: nothing reads them but the meter.
        self._novel_ever = 0
        self._steps_ever_here = 0
        # THE DENOMINATOR MUST SHARE THE NUMERATOR'S CLOCK (2026-08-11).
        # `_steps_ever_here` ticks once per step(), but the novelty count
        # lives in `note_attended`, which fires once per percept() -- and
        # percept() runs more than once per tick, on a cadence NOTHING
        # guarantees: world_actor.propose reads it every waking tick, and
        # step() reads it again, but only when a motor claim actually wins
        # arbitration.  (A third reader, runtime._world_tick, has been dead
        # since the toy world was retired from the live tick in Cap-3 --
        # counted here so the next reader does not re-derive it.)  So the
        # ratio moves with how often he chooses to act, which is precisely
        # what a novelty rate must NOT do.
        # MEASURED CONSEQUENCE: the meter reported 1,890 novel per
        # 1k steps -- more than one first-ever discovery per step, which is
        # not a rate at all.  This counts ATTENTION COMMITS in the same
        # function, on the same event, as the novelty count, so
        # novel/attend is a fraction in [0,1] BY CONSTRUCTION and no caller
        # can knock it out of meaning by glancing more often.  attend/step
        # then REPORTS the glance multiplicity instead of hiding it inside
        # a ratio.  Bookkeeping only: nothing reads it but the meter.
        self._attend_ever = 0
        # THE ACTION THAT PRODUCED THE NEXT FRAME (2026-08-12).  The
        # frame recorder lives in _absorb(), which is handed only the
        # sidecar RESPONSE and so cannot see what was done to cause it.
        # step() parks the action here for exactly one absorb.  None
        # means the frame came from an open/reset, not from an action --
        # which a corpus builder MUST see, or it will pair a reset frame
        # with the previous episode and invent a transition.
        self._rec_act = None
        self._steps_here = 0

    # ---- wiring -----------------------------------------------------
    def bind_eye(self, transducer, visits, attend_fn) -> None:
        """Share the ACTOR'S transducer + _visits so his attention and his
        action are driven by ONE novelty signal (option B, validated)."""
        self._transducer = transducer
        if visits is not None:
            self._visits = visits
        self._attend = attend_fn

    # ---- sidecar ----------------------------------------------------
    def _call(self, req: Dict[str, Any]) -> Dict[str, Any]:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(10.0)
            s.connect(self._sock)
            f = s.makefile('rwb')
            f.write((json.dumps(req) + '\n').encode())
            f.flush()
            out = json.loads(f.readline())
            f.close()
            s.close()
            # A NOT-OK REPLY IS A FAILURE (2026-07-29, MEASURED).  The
            # sidecar restarting underneath him answers every step with
            # {'ok': 0, 'err': ...} -- a perfectly valid JSON reply, so
            # nothing raised and nothing was counted, and `arc_errors`
            # sat at 7 through ~44,000 steps into a frozen grid.  A
            # failure that costs nothing is invisible; count it, and mark
            # the world dead so `_ensure_mine` re-opens it.
            if not out.get('ok'):
                self.errors += 1
                self._live = False
            else:
                self._live = True
            return out
        except Exception:
            self.errors += 1
            self._live = False
            return {'ok': 0}

    def _absorb(self, r: Dict[str, Any]) -> None:
        if not r.get('ok'):
            return
        if r.get('grid') is not None:
            self._grid = r['grid']
        self._state = str(r.get('state') or self._state)
        self._levels = int(r.get('levels', self._levels) or 0)
        self._bar_observe()
        # FRAME RECORDER (read-only, file-gated, 2026-08-03).  These games
        # DRAW their state on the grid -- a score, a level marker, an energy
        # or life bar (row 63 was measured to be one).  He sees only a 3x3
        # glanced patch, so he cannot read it.  To find such an INSTRUMENT
        # without supervision we need frames paired with `state`/`levels`.
        # DEATHS are the abundant label (hundreds per card) where wins are
        # n=5, so a cell that moves before GAME_OVER is findable.  Capped per
        # game so this cannot fill the disk; gate file removes it with no
        # restart.
        try:
            if _os.path.exists('/root/FRAME_REC_ON') and self._grid:
                _k = str(self.game_id)
                # PER (GAME, LEVEL), NOT PER GAME (2026-09-03).  The cap
                # was keyed on the game alone, and he spends ~94% of his
                # life on level 0 -- so level 0 ate the whole quota and
                # the deeper levels were barely recorded.  Measured: the
                # corpus held 30 r11l LEVEL-1 runs against 227 level-1
                # lives in the journal over 48 h, which made the question
                # "why does r11l die at step 8-10 in a third of its
                # lives" unanswerable: only ONE such life was on disk.
                # A level he reaches rarely is exactly the one worth
                # recording.
                _rec_key = (_k, int(self._levels))
                _seen = ARCWorld._rec_counts.get(_rec_key, 0)
                # statvfs only when the cap would otherwise let us write,
                # so this is not a syscall on every absorb for the life
                # of a game that has already filled its quota.
                _rec_cap = 3000 if int(_rec_key[1]) >= 3 else 300   # deep levels are the ones worth recording; L0-L2 halved so the disk guard holds (32c)
                if _seen < _rec_cap:
                    _vfs = _os.statvfs('/root')
                    _room = (_vfs.f_bavail * _vfs.f_frsize) >= (5 << 30)
                else:
                    _room = False
                if _room:
                    ARCWorld._rec_counts[_rec_key] = _seen + 1
                    import base64 as _b64
                    import json as _js
                    import time as _tm
                    _flat = bytes(bytearray(
                        max(0, min(255, int(v)))
                        for _row in self._grid for v in _row))
                    # NEW FILE, NEW FORMAT.  frames.jsonl keeps its old
                    # action-free shape so the rule-loop work that reads
                    # it is untouched; this one carries the action.
                    # DISK GUARD: frames.jsonl reached 1.6 GB by
                    # accumulating across restarts, so refuse to write
                    # when the box is low rather than trusting the cap.
                    _ra = self._rec_act
                    with open('/root/frames_act.jsonl', 'a') as _fh:
                        _fh.write(_js.dumps({
                            't': _tm.time(), 'g': _k,
                            'lv': int(self._levels),
                            'st': str(self._state),
                            'h': len(self._grid),
                            'w': len(self._grid[0]) if self._grid else 0,
                            'a': (int(_ra[0]) if _ra else None),
                            'ay': (_ra[1] if _ra else None),
                            'ax': (_ra[2] if _ra else None),
                            # 1 = the aim CAN change the outcome, 0 =
                            # it was sent but is inert.  Absent on rows
                            # written before 2026-08-12; the corpus
                            # builder marks those -1 (unknown) rather
                            # than guessing 0, which would silently mix
                            # clicks into the non-click bucket.
                            'c': ((1 if (len(_ra) > 3 and _ra[3])
                                   else 0) if _ra else None),
                            'n': int(_seen),
                            'p': int(_os.getpid()),
                            'b': _b64.b64encode(_flat).decode('ascii'),
                        }) + chr(10))
        except Exception:
            pass
        self._acts = list(r.get('actions') or self._acts)
        self._n_actions = max(1, int(r.get('n_actions', self._n_actions) or 1))

    # The sidecar holds ONE game at a time.  Rungs share it, so a rung must
    # re-open its own game whenever another rung last used it.  Class-level
    # because the ownership is a property of the shared sidecar, not of any
    # single rung.  Re-opening restarts that game, which is correct: the
    # curriculum only moves on after a rung has been MASTERED.
    _owner = ''
    # THE ENVIRONMENT (all ARC patches): what foraging here pays
    # on average.  Class-level on purpose -- MVT compares a patch
    # against the environment, not against itself.
    _rec_counts = {}
    # THE WAY OUT, REMEMBERED.  (game, level) -> {board: (a, y, x)}
    # written ONLY from a run that completed that level, loop-erased,
    # shortest kept.  Board-keyed rather than a sequence so a
    # divergence is self-healing: off the path he simply plays.
    _paths = {}
    _depth_paths = {}       # (game, lv) -> board->action, the FRONTIER
    _aim_loops = 0
    _depth_seed_loaded = False
    _depth_seeded = 0
    _depth_len = {}         # (game, lv) -> loop-erased length sealed
    _depth_sealed = 0
    _depth_follow = 0
    _cur_path = {}
    # boards already stood on THIS attempt -- the loop guard.  Same
    # lifecycle as _cur_path, so it needs no clearing of its own.
    _cur_seen = {}
    _path_loops = 0
    _path_follow = 0
    _path_sealed = 0
    _seed_loaded = False
    _seed_paths = 0
    # HIS JUDGEMENT OF A STEP.  (game, board, action) -> [win, over,
    # new, back].  COUNTS, not a label: 28% of repeated steps change
    # verdict, so a latched one would be wrong more than a quarter of
    # the time.  Counting is what lets him adjust it.
    _verdict = {}
    # SECOND INDEX, aggregated over aims: (game, board, action).
    # `propose` picks an action INDEX before any aim exists, so the
    # aim-keyed store above cannot be queried there -- and scanning it
    # would be O(400k) per action per step.
    _verdict_i = {}
    _att_boards = {}   # game -> boards stood on THIS attempt
    _bar_row = {}      # (game, level) -> the budget row, once decided
    _bar_stat = {}     # (game, level) -> [n, tick[128], solo[128], back[128], (CLOCKLINE) cotick{(a,b): n}, fwd[128]]
    _bar_seen = {}     # (game, level) -> per-row values seen THIS life
    _bar_prev = {}     # (game, level) -> previous frame's row hashes
    _bar_pflat = {}    # (game, level) -> previous frame's bytes (CLOCKLINE)
    _bar_sign = {}     # (game, level) -> {(line, colour): direction} THIS life (CLOCKLINE)
    _bar_lv = {}       # game -> the level those tables belong to
    _bar_cand = {}     # (game, level) -> last decision, latched on repeat
    _bar_learn_n = 0
    _bar_decided = 0
    _att_lv = {}       # game -> the level that attempt belongs to
    _att_resets = 0
    _verdict_good = 0
    _verdict_bad = 0
    _verdict_unk = 0
    # ---- OBJECT GRAIN (2026-08-03) ----------------------------------
    # MEASURED: boards decompose into 6-66 objects with 83-92% step-to-
    # step persistence in ALL 25 games, and only 1-2 move per step.  With
    # that, CONTINGENCY finds the object he controls -- 'I act, that blob
    # moves, that blob is me' -- in every game where he varies his action
    # (10/19 offline; the misses are windows where he used ONE action id,
    # so there was nothing to correlate).  This is the hinge humans use:
    # it turns an abstract action space into an embodied one.
    #
    # ADDITIVE ON PURPOSE.  The percept token is untouched, so `_route`
    # (6.6k entries) and `_trans` (113k edges) are NOT orphaned.  This
    # runs alongside and reports its own determinism, so object grain can
    # be compared with patch-grain purity (0.354) IN VIVO before anything
    # bets the state representation on it.
    _objs = {}        # game -> last frame's [(colour,size,cy,cx)]
    _selfhyp = {}     # game -> {(colour,size): {action: {(dy,dx): n}}}
    _self = {}        # game -> (sig, determinism, n_moves)
    _obj_n = {}       # game -> object count last frame
    # ---- AFFORDANCE (2026-08-03, step 3) ----------------------------
    # Now that he knows WHICH OBJECT IS HIM (18/25 games, determinism
    # 0.883 vs 0.354 at patch grain), the next thing humans learn is what
    # happens when they touch things.  The affordance that matters here:
    # an object that VANISHES when he reaches it is a thing to SEEK --
    # a dense goal candidate needing no win, in every game.
    #
    # THE NULL IS BUILT IN.  Objects vanish for many reasons (redraws,
    # animation, level resets).  It is only an affordance if vanishing is
    # far likelier NEAR him than far from him, so both are counted and
    # the pair is reported.  A colour that disappears everywhere is
    # scenery, not food.
    _afford = {}      # game -> {colour: [near_vanish, far_vanish, seen]}
    _contact = {}     # game -> total vanish events observed
    # OCCLUSION TEST (2026-08-03).  'Vanishes near him' is ALSO what
    # standing on something looks like: he overlaps it, it fails to
    # segment, it reads as gone.  The two differ in one observable -- an
    # occluded thing COMES BACK the moment he steps off, a consumed one
    # does not.  So each vanished blob is held pending and resolved when
    # he is no longer near it: reappeared => occlusion, absent => eaten.
    # Resolution is keyed to HIS OWN distance, so no waiting constant.
    _pending = {}     # game -> [(colour,size,cy,cx)]
    _occl = {}        # game -> [reappeared, stayed_gone]
    # ---- BODY NOVELTY (2026-08-03, route B) -------------------------
    # He explores by GAZE novelty: 1/(1+visits) on the next 3x3 patch.
    # Now that he has a body he can also explore by WHERE HE HAS BEEN --
    # a small, well-defined, actually-coverable space instead of a
    # wandering patch.
    #
    # BOTH, NEVER EITHER/OR (user, 2026-08-03: "depending on the
    # situation he adapts and improvises, uses what is needed").  So
    # this is not a switch: body-novelty is BLENDED with gaze-novelty in
    # proportion to how much the self-model has EARNED in this game
    # (determinism x n/(1+n)) -- the same earned-confidence idiom the
    # actor already uses for M/I.  Zero confidence => byte-identical to
    # today; a crisp self => he navigates; a murky one => he keeps
    # looking around.  Per game, and it moves as evidence accumulates.
    _posvis = {}      # (game, level) -> {(row,col): visits}
    # ---- WIN CONFIGURATIONS (2026-08-03) ----------------------------
    # Goal inference failed five times because I kept asking PIXELS for a
    # gradient that is not there (the bar drains uniformly, contact does
    # nothing, disorder is flat, and a win is a full-board redraw).  But
    # he now has OBJECTS and a SELF -- and 4-5 level completions per
    # card, each of which is a labelled example AT OBJECT GRAIN.  Five
    # examples of 'these objects stood in this relation' is a completely
    # different proposition from five pixel hashes, which is what killed
    # win-signature transfer.
    #
    # Records the configuration the winning step was taken FROM, ego-
    # centric (offsets from the self), so it can transfer across boards.
    # ORDINARY configurations are sampled too -- without the null there
    # is no way to tell what is special about a win, which is the mistake
    # that produced a 'consumable' that turned out to be occlusion.
    _ord_n = {}       # game -> ordinary-sample counter
    _all_insts = []   # live ARCWorld rungs, for the size diagnostic
    # ---- THE RULE LOOP (2026-08-04) ---------------------------------
    # Every agent at the top of the community board does one thing:
    # holds explicit rule HYPOTHESES, RETRODICTS them against recorded
    # history, and revises on mismatch -- persisting what survives so the
    # next game starts from knowledge instead of from zero.
    #
    # A rule is keyed by its CONTENT, not by the game it came from, so
    # the same rule met elsewhere is recognised as the same rule.  That
    # is the whole point: he should not re-derive the controls in all 25
    # games.  Retrodiction is FREE -- confirming a rule against what has
    # already happened costs no actions, and under squared-efficiency
    # scoring with a 5x cutoff, actions are the entire currency.
    _rules = {}       # (kind, a, dy, dx) -> {n, born, games:{g:n}}
    _transfers = 0    # times a rule learned ELSEWHERE held in a new game
    _seeded = set()   # games already seeded from the rule store
    _retro_loaded = False
    # ---- WHERE TO GO (2026-08-05) -----------------------------------
    # User: "if you are in an empty room and there is one small box on
    # the ground, it is obvious that you go to the box and see what it
    # is."  He does not need a signal saying THIS IS THE GOAL -- he needs
    # one saying THIS IS WORTH INVESTIGATING.  That needs no wins, exists
    # on every board, and is exactly the curiosity the felt-play layer
    # already runs on.
    #
    # SALIENCE = HOW FEW THINGS ARE LIKE IT.  Unique object -> ~1.0; one
    # of twenty identical -> 0.05.  Self-referential to the board, so no
    # constant and no per-game tuning.  Investigating it SATISFIES the
    # curiosity, so salience decays with visits and he moves on.
    _seen_kind = {}   # (game, colour, size) -> times he has been near it
    # ---- ROLES GO IN HIS BRAIN, NOT IN A SIDE DICT (2026-08-05) ------
    # User: "do you need an extra library for this? the brain is a
    # library already... Seagi has all the tools, he just does not know
    # how to use them."  Correct -- `_rules`/`_self`/`_afford` were a
    # shadow knowledge base OUTSIDE the substrate.  A role is knowledge,
    # so it belongs in the substrate as an `is_a` edge, where it gets
    # everything already built for free:
    #   * form_abstractions groups by (relation, target), so >=3 objects
    #     sharing a role BECOME A CLASS on their own -- the role is
    #     learned, never stamped
    #   * is_a composition means roles are reasoned over in inference
    #   * earn-or-dissolve: a role that gets walked strengthens, one that
    #     never does decays below the prune floor and disappears
    #   * consolidation and sleep already operate on concepts
    # Detection is STRUCTURAL, not colour-keyed, so a role learned in one
    # game is recognisable in the next.
    _role_written = {}   # (game, obj, role) -> last step written
    # WHAT A DOORWAY LOOKS LIKE.  kind -> [times it ended a level].
    # Class-level on purpose: a doorway recognised in one game is a
    # doorway in the next, which is the whole point of carrying the
    # description instead of the route.
    _goal_kinds = {}     # (colour, size) -> wins
    _goal_colours = {}   # colour        -> wins  (coarser, transfers further)
    _goal_last = None    # what he matched most recently, for /status
    _doorway_now = None  # confirmed doorway this step -> substrate
    # A GOAL GUESS THAT DIED IS SET ASIDE FOR THE NEXT ATTEMPT.
    # (game, level) -> {object kind he committed to when the attempt ended}
    _target_failed = {}
    # ACTION -> LOCUS OF CHANGE (2026-08-08).  Works with or without an
    # avatar; subsumes the self model, which is the special case where
    # the thing that changes IS him.
    _locushyp = {}     # game -> action -> {(dy,dx): count}
    _locus_prev = {}   # game -> last centroid of change
    _board_seen = {}   # game -> set(board hash)
    _board_act = {}    # (game, board hash, action) -> [tries, revisits]
    # (game, board, action) -> [tries, changed].  The glance-keyed
    # version fires on 0.75% of steps; the board recurs 31-95%.
    _board_noop = {}
    _cyc_calls = 0     # times the demotion actually ASKED
    _explorers = {}    # game -> Explorer (egocentric model)
    _explorer_disk = None   # persisted state, read once per process
    _felt = 0.0             # how he is right now, in [-1, 1]
    _felt_n = 0
    _felt_competent = 0
    _felt_noop = 0
    _felt_back = 0
    _felt_here = {}         # (game, level) -> [ewma in [-1, 1], steps]  (FELTHERE)
    _felt_here_errors = 0
    _cyc_known = 0     # ... about a pair it had seen twice
    _cyc_hits = 0      # ... that had always led backwards
    _lvl_base = {}     # (game, level) -> board at level start
    _lvl_hwm = {}      # (game, level) -> highest irreversible d
    _progress_n = 0
    _rel = {}          # (game, level) -> relsense.RelSense (RELSENSE)
    _relplan = {}      # (game, level) -> relplan.RelPlan (RELPLAN)
    _relplan_asks = 0
    _relplan_acts = 0
    _relplan_errors = 0
    _relplan_last = ''
    _relplan_by_game = {}
    _relplan_inert = 0
    _relplan_why = {}
    _relplan_lost = 0
    _relplan_seeded = set()    # (game, level) merged from the seed this process (patch 14)
    _relplan_seed_path = '/root/relplan_seed.json'
    _hyp = {}          # (game, level) -> hypothesis.Hypothesis (HYPOTHESIS)
    _hyp_asks = 0
    _hyp_acts = 0
    _hyp_errors = 0
    _hyp_holds = 0
    _hyp_last = ''
    _hyp_life = {}     # game -> the life id the hypothesis last began (HYPOTHESIS)
    _srch = {}         # (game, level) -> goal_search.Search (SEARCH)
    _lib = None        # goal_library.Library: ONE across games (LIBRARY, patch 48)
    _lib_start = {}    # (game, level) -> the board the current life began on
    _lib_sb0 = {}      # (game, level) -> his box on that board
    _lib_insts = {}    # (game, level) -> relation instances read on that board
    _lib_pursues = 0
    _lib_refutes = 0
    _lib_errors = 0
    _srch_asks = 0
    _srch_acts = 0
    _srch_errors = 0
    _srch_holds = 0
    _srch_last = ''
    _srch_life = {}    # game -> the life id the search last began (SEARCH)
    _lvl_dry = {}        # (game, level) -> lives since the last record (LEVELHOLD)
    _lvl_gap_max = {}    # (game, level) -> longest drought that ended in a record
    _lvl_best_mm = {}    # (game, level) -> best canvas/target mismatch of any life
    _lvl_records = 0
    _lvl_dry_ejects = 0
    _lvl_errors = 0
    _lvl_last = ''
    _rel_steps = 0
    _rel_fires = 0
    _rel_confirms = 0
    _rel_errors = 0
    _locusvis = {}     # (game, level) -> {pos: visits}
    _posvis = {}       # (game, level) -> {(r,c): times LOOKED there}
    _locus_n = 0
    _role_n = 0
    # COMMIT TO A TARGET (2026-08-05).  MEASURED: approach steps climbed
    # 66 -> 742 while arrivals stayed at ZERO.  In half the games the
    # salience argmax picked a DIFFERENT object almost every frame
    # (1-3 times out of 12), so he set off, changed his mind, set off
    # again, and never got anywhere.  Going to the box means going to
    # THE box -- hold it until reached or gone.
    _target = {}      # game -> (colour, size)
    # ---- CALIBRATION (2026-08-05) -----------------------------------
    # THE BOTTLENECK, measured: he cannot navigate with a control scheme
    # he never discovered.  Self-detection failed in 7 of 25 games for
    # one reason -- he used a SINGLE action id there, so contingency had
    # nothing to correlate -- and `approach` stalls wherever he has a
    # known displacement for only one or two actions.
    #
    # So do what a human does in the first seconds of an unknown game:
    # press each button and watch.  Bounded (2 tries per action, once per
    # game) and SELF-LIMITING -- it stops the moment every action has been
    # tried, and never runs again for that game.  ~8-12 actions of a
    # 64-action budget, which pays for itself immediately if it unlocks
    # the self and with it navigation.
    _act_tries = {}   # game -> {action: times tried}
    _calib_n = 0
    # THE STAKE, NOT HIS LIFE (2026-08-03).  Every ARC board draws a
    # 64-cell bar on an edge line that drains one cell per action and
    # reads empty at GAME_OVER (measured 17/17, and 25/25 by a second
    # statistic).  It is the game's stake -- like a human watching a
    # timer -- NOT his mortality: he lives before the game and after it.
    # So it drives PROXY FEELING (urgency), never lifeforce.
    # game_id -> (axis, index, value, span)
    _bar = {}
    _bar_hist = {}
    _novel_all = 0
    # {(game, level): {(board, action, aim_or_None): tries}}.  NOT
    # persisted and NOT assumed bounded -- the state space grew +126%
    # in the 8 h after he started staying.  Evicts least-tried, never
    # wholesale: the box is at 4.9 GB RSS of 7.7 GB with swap in use.
    _ftried = {}
    _fp_aim = None
    _frontier_aim_chk = 0
    _frontier_steers = 0
    _frontier_aim_miss = 0
    _frontier_book = 0
    _steps_all = 0
    # WHAT THE WHOLE ENVIRONMENT PAYS.  `_novel_all` is the novelty
    # equivalent; nothing counted WINS, so no rule could compare them.
    _won_all = 0
    _yield_stays = 0

    def _bar_fill(self):
        """Fraction of the stake remaining, or None until the bar is
        located.  The bar is the only line on the board whose
        value-counts move monotonically; the LIFE colour is the one that
        drains.  Detected from his own recent frames -- no constant, no
        hardcoded row (it is row 63 in most games but row 0/1/53/61 and
        col 0/63 in others)."""
        g = str(self.game_id)
        if self._grid is None:
            return None
        try:
            G = np.array(self._grid, dtype=np.int16)
        except Exception:
            return None
        got = ARCWorld._bar.get(g)
        if got is not None:
            axis, idx, val, span = got
            line = G[idx, :] if axis == 'row' else G[:, idx]
            if span <= 0:
                return None
            return float((line == val).sum()) / float(span)
        h = ARCWorld._bar_hist.setdefault(g, [])
        h.append(G.copy())
        if len(h) < 24:
            return None
        del h[:-24]
        S = np.array(h)
        best = None
        for axis in ('row', 'col'):
            L = S if axis == 'row' else np.transpose(S, (0, 2, 1))
            for i in range(L.shape[1]):
                seg = L[:, i, :]
                for v in np.unique(seg):
                    cnt = (seg == v).sum(1).astype(np.int32)
                    d = np.diff(cnt)
                    nz = d[d != 0]
                    if len(nz) < 6:
                        continue
                    down = float((nz < 0).sum()) / len(nz)
                    if down < 0.95:
                        continue          # the LIFE colour drains
                    sc = down * len(nz)
                    if best is None or sc > best[0]:
                        best = (sc, axis, i, int(v), int(cnt.max()))
        if best is None:
            return None
        ARCWorld._bar[g] = (best[1], best[2], best[3], max(1, best[4]))
        ARCWorld._bar_hist.pop(g, None)
        return None

    def _segment(self, G, bg):
        """Connected same-colour components, background excluded.
        scipy labels a 64x64 board in ~1.1 ms, so this is affordable every
        step.  Pathological boards (bp35 dithers into ~190 blobs of size 1)
        are skipped rather than tracked."""
        from scipy import ndimage
        out = []
        ones = np.ones_like(G, dtype=np.float64)
        for v in np.unique(G):
            if int(v) == bg:
                continue
            lab, n = ndimage.label(G == v)
            if n <= 0 or n > 120:
                continue
            idx = list(range(1, n + 1))
            szs = ndimage.sum(ones, lab, idx)
            cms = ndimage.center_of_mass(ones, lab, idx)
            for k in range(n):
                out.append((int(v), int(szs[k]),
                            float(cms[k][0]), float(cms[k][1])))
            if len(out) > 200:
                return []
        return out

    def _track_self(self, action):
        """Track objects across the step and update WHICH ONE IS HIM.

        The self is chosen by DIRECTION DETERMINISM CONDITIONED ON HAVING
        MOVED -- given the object moved, does the action say which way?
        Measuring it unconditionally was the bug that hid the self in 9
        games: most actions leave a thing where it is, so the statistic
        only measured how often it sat still.  Argmax over candidates, so
        no threshold constant enters."""
        g = str(self.game_id)
        try:
            G = np.array(self._grid, dtype=np.int16)
        except Exception:
            return None
        vals, cts = np.unique(G, return_counts=True)
        bg = int(vals[int(np.argmax(cts))])
        cur = self._segment(G, bg)
        prev = ARCWorld._objs.get(g)
        ARCWorld._objs[g] = cur
        ARCWorld._obj_n[g] = len(cur)
        if not prev or not cur:
            return None
        if len(ARCWorld._selfhyp) > 40:
            for _dead in [k for k in ARCWorld._selfhyp if k != g][:20]:
                ARCWorld._selfhyp.pop(_dead, None)
        hyp = ARCWorld._selfhyp.setdefault(g, {})
        pool = list(cur)
        moved_now = None
        cursig = (ARCWorld._self.get(g) or (None,))[0]
        def _is_self(k):
            if cursig is None:
                return False
            return (k[0] == cursig[0]
                    and abs(k[1] - cursig[1]) <= max(1, cursig[1] // 4))
        for col, sz, cy, cx in prev:
            cand = [x for x in pool if x[0] == col
                    and abs(x[1] - sz) <= max(1, sz // 4)]
            if not cand:
                continue
            m = min(cand, key=lambda x: (x[2] - cy) ** 2 + (x[3] - cx) ** 2)
            pool.remove(m)
            dy = int(round(m[2] - cy))
            dx = int(round(m[3] - cx))
            k = (col, sz)
            d = hyp.setdefault(k, {}).setdefault(int(action), {})
            d[(dy, dx)] = d.get((dy, dx), 0) + 1
            if _is_self(k):
                self._rule_observe(g, action, dy, dx)
            if (dy or dx) and _is_self(k):
                moved_now = [int(col), int(sz), float(m[2]), float(m[3]),
                             dy, dx]
        try:
            _sp = None
            if cursig is not None:
                for o in cur:
                    if (o[0], o[1]) == cursig:
                        _sp = (o[2], o[3])
                        break
            self._note_vanished(g, prev, cur, _sp)
        except Exception:
            pass
        # BOUND THE HYPOTHESIS TABLE (2026-08-04).  MEASURED: three OOM
        # kills overnight (03:48, 06:47, 12:47).  `_selfhyp` is keyed by
        # (colour, SIZE), and an object's size shifts as it moves or
        # animates, so every step mints fresh signature keys and nothing
        # ever pruned them -- unbounded growth on a box already carrying
        # a ~4GB canonical.  Keep the best-evidenced signatures: the self
        # is by construction among the most-observed, so pruning the tail
        # cannot lose it.  Displacements per action are capped the same
        # way for the same reason.
        if len(hyp) > 400 and (self.steps % 50) == 0:
            try:
                _rank = sorted(
                    hyp.items(),
                    key=lambda kv: -sum(sum(d.values())
                                        for d in kv[1].values()))
                keep = dict(_rank[:200])
                _cs = (ARCWorld._self.get(g) or (None,))[0]
                if _cs is not None and _cs in hyp and _cs not in keep:
                    keep[_cs] = hyp[_cs]      # never prune the incumbent
                ARCWorld._selfhyp[g] = keep
                hyp = keep
            except Exception:
                pass
        for _k2, _ba in list(hyp.items()):
            for _a2, _dd in list(_ba.items()):
                if len(_dd) > 12:
                    _top = sorted(_dd.items(), key=lambda kv: -kv[1])[:12]
                    _ba[_a2] = dict(_top)
        if (self.steps % 25) == 0:
            best = None
            for k, byact in hyp.items():
                tot = 0
                hit = 0
                nact = 0
                for a, dd in byact.items():
                    mv = dict((q, c) for q, c in dd.items() if q != (0, 0))
                    if not mv:
                        continue
                    nact += 1
                    tot += sum(mv.values())
                    hit += max(mv.values())
                if tot >= 8 and nact >= 2:
                    pur = hit / float(tot)
                    if best is None or pur > best[1]:
                        best = (k, pur, tot)
            if best is not None:
                ARCWorld._self[g] = best
                self._seed_from_rules(g, best[0])
        return moved_now

    def _note_vanished(self, g, prev, cur, selfpos):
        """Objects present last step with no match now = vanished.
        Attribute each to NEAR or FAR relative to him, so the ratio is
        self-referential and needs no distance constant: 'near' is closer
        than the median object on this board."""
        if not prev or selfpos is None:
            return
        matched = set()
        pool = list(cur)
        for o in prev:
            cand = [x for x in pool if x[0] == o[0]
                    and abs(x[1] - o[1]) <= max(1, o[1] // 4)]
            if not cand:
                continue
            m = min(cand, key=lambda x: (x[2] - o[2]) ** 2 + (x[3] - o[3]) ** 2)
            pool.remove(m)
            matched.add(id(o))
        gone = [o for o in prev if id(o) not in matched]
        if not gone:
            return
        sy, sx = selfpos
        dists = sorted(abs(o[2] - sy) + abs(o[3] - sx) for o in prev)
        if not dists:
            return
        med = dists[len(dists) // 2]
        tab = ARCWorld._afford.setdefault(g, {})
        for o in prev:
            row = tab.setdefault(int(o[0]), [0, 0, 0])
            row[2] += 1
        self._resolve_pending(g, cur, (sy, sx), med)
        pend = ARCWorld._pending.setdefault(g, [])
        for o in gone:
            d = abs(o[2] - sy) + abs(o[3] - sx)
            row = tab.setdefault(int(o[0]), [0, 0, 0])
            if d <= med:
                row[0] += 1
                pend.append((int(o[0]), int(o[1]),
                             float(o[2]), float(o[3])))
                del pend[:-64]
            else:
                row[1] += 1
            ARCWorld._contact[g] = ARCWorld._contact.get(g, 0) + 1

    @staticmethod
    def _mem_sizes():
        """Sizes of the object-grain structures, so a leak can be
        LOCATED instead of guessed at.  Bounding `_selfhyp` fixed one
        leak and memory still climbed 4.9 -> 6.5 GB in 90 s, so the
        dominant term is elsewhere."""
        try:
            def _deep(d):
                n = 0
                for v in d.values():
                    if isinstance(v, dict):
                        n += len(v)
                        for w in v.values():
                            if isinstance(w, dict):
                                n += len(w)
                    elif isinstance(v, (list, set)):
                        n += len(v)
                return n
            return {
                'selfhyp_games': len(ARCWorld._selfhyp),
                'selfhyp_deep': _deep(ARCWorld._selfhyp),
                'objs_deep': _deep(ARCWorld._objs),
                'posvis_keys': len(ARCWorld._posvis),
                'posvis_deep': _deep(ARCWorld._posvis),
                'barhist_deep': _deep(ARCWorld._bar_hist),
                'afford_deep': _deep(ARCWorld._afford),
                'pending_deep': _deep(ARCWorld._pending),
                'seen_win': sum(len(getattr(w, '_seen_win', ()) or ())
                                for w in ARCWorld._all_insts),
                'felt_seen': sum(len(getattr(w, '_felt_seen', ()) or ())
                                 for w in ARCWorld._all_insts),
                'seen_tok': sum(len(getattr(w, '_seen_tok', ()) or ())
                                for w in ARCWorld._all_insts),
                'visits_deep': _deep(ARCWorld._obj_n),
            }
        except Exception as _e:
            return {'err': repr(_e)[:80]}

    @staticmethod
    def _predicates(rows):
        """PUZZLE PRIORS (2026-08-04, user-authorised).

        A small library of things a puzzle MIGHT want, computed over the
        ego-centric object list.  These are a VOCABULARY, not an answer:
        every one is evaluated on every configuration, and WHICH of them
        matters is earned by corroboration across wins vs ordinary states
        -- the same earn-or-dissolve pattern used everywhere else.  He is
        told what questions are askable, never which answer is right.

        This is what makes few examples enough.  Inducing an arbitrary
        goal function from 5 examples is hopeless (it is what killed
        win-signature transfer); SELECTING among ~8 candidates from 5
        examples is entirely feasible.  That is what a prior buys.

        Grounded in the one real win record captured so far, which showed
        a mirrored pair at (42,-12) and (42,+12) -- structure invisible at
        pixel grain and legible at object grain."""
        out = {}
        try:
            oth = [r for r in rows if not (r[2] == 0 and r[3] == 0)]
            out['n_obj'] = len(rows)
            out['n_col'] = len(set(r[0] for r in rows))
            # ALIGNED: something shares his row / his column
            out['aligned_row'] = any(r[2] == 0 for r in oth)
            out['aligned_col'] = any(r[3] == 0 for r in oth)
            # ADJACENT: something is touching him
            out['adjacent'] = any(abs(r[2]) + abs(r[3]) <= 2 for r in oth)
            # MIRRORED PAIR about him, on either axis
            _mx = _my = False
            for i in range(len(oth)):
                for j in range(i + 1, len(oth)):
                    a, b = oth[i], oth[j]
                    if a[0] != b[0] or a[1] != b[1]:
                        continue
                    if a[2] == b[2] and a[3] == -b[3] and a[3] != 0:
                        _mx = True
                    if a[3] == b[3] and a[2] == -b[2] and a[2] != 0:
                        _my = True
            out['mirror_x'] = _mx
            out['mirror_y'] = _my
            # UNIFORM: everything that is not him is one colour
            out['uniform'] = (len(set(r[0] for r in oth)) == 1) if oth else False
            # CLUSTERED: all others within a tight radius
            out['clustered'] = (
                all(abs(r[2]) + abs(r[3]) <= 8 for r in oth) if oth else False)
            # EMPTY: nothing left but him
            out['alone'] = (len(oth) == 0)
            # COLINEAR TRIPLE through him
            out['colinear'] = any(
                (oth[i][2] == 0 and oth[j][2] == 0) or
                (oth[i][3] == 0 and oth[j][3] == 0)
                for i in range(len(oth)) for j in range(i + 1, len(oth)))
        except Exception:
            pass
        return out

    def _log_config(self, kind, preobjs):
        """Append an object-grain configuration, ego-centric to the
        self.  Observe-only: changes nothing about how he plays."""
        try:
            if not _os.path.exists('/root/WINCONF_ON') or not preobjs:
                return
            g = str(self.game_id)
            # A WIN IS TOO SCARCE TO DISCARD (2026-08-04).  This used to
            # return early when the self was unknown -- and the games he
            # actually WINS are often the ones with no self (a single
            # action id gives contingency nothing to correlate).  The
            # 5th level completion was thrown away that way.  Fall back
            # to the BOARD CENTRE as the reference frame so the record
            # is still comparable, and mark which frame was used.
            sig = (ARCWorld._self.get(g) or (None,))[0]
            sy = sx = None
            if sig is not None:
                for o in preobjs:
                    if (o[0], o[1]) == sig:
                        sy, sx = o[2], o[3]
                        break
            _frame = 'self'
            if sy is None:
                if kind != 'win':
                    return        # ordinary samples still need a self
                _frame = 'centre'
                sy = sx = 32.0
            rows = []
            for o in sorted(preobjs, key=lambda x: -x[1])[:40]:
                rows.append([int(o[0]), int(o[1]),
                             int(round(o[2] - sy)), int(round(o[3] - sx))])
            import json as _js
            import time as _tm
            with open('/root/winconf.jsonl', 'a') as fh:
                fh.write(_js.dumps({
                    't': _tm.time(), 'g': g, 'lv': int(self._levels),
                    'kind': kind, 'frame': _frame,
                    'self': ([int(sig[0]), int(sig[1])]
                             if sig is not None else None),
                    'n_obj': len(preobjs), 'objs': rows,
                    'pred': ARCWorld._predicates(rows)}) + chr(10))
        except Exception:
            pass

    @classmethod
    def _load_retrodicted(cls):
        """Rules learned by REPLAYING recorded history, which costs no
        actions.  75,955 archived frames yielded 37 rules, 14 of them
        holding in more than one game -- and they describe a shared
        control convention (action 0 up, 1 down, 2 left, 3 right) whose
        only per-game variable is the step size (3/4/5/6 cells).

        Loaded as HYPOTHESES, not commitments: live observation outvotes
        them within a few steps wherever they are wrong."""
        if cls._retro_loaded:
            return
        cls._retro_loaded = True
        try:
            import json as _js
            with open('/root/retrodicted_rules.json') as fh:
                d = _js.load(fh)
            n = 0
            for k, v in (d or {}).items():
                if k in cls._rules:
                    continue
                cls._rules[k] = {'n': int(v[0]), 'born': str(v[1]),
                                 'games': dict((str(x), 1)
                                               for x in (v[2] or []))}
                n += 1
            cls._retro_n = n
            import sys as _s
            _s.stderr.write('[rules] retrodicted %d rules loaded '
                            'from recorded history\n' % n)
            _s.stderr.flush()
        except Exception:
            pass

    def _rule_observe(self, g, a, dy, dx):
        """Record what an action did to him, as a content-keyed rule.
        A rule first seen in another game and now holding here is a
        TRANSFER -- knowledge carried across, which is the thing being
        built."""
        if not (dy or dx):
            return
        k = 'move|%d|%d|%d' % (int(a), int(dy), int(dx))
        r = ARCWorld._rules.get(k)
        if r is None:
            r = {'n': 0, 'born': g, 'games': {}}
            ARCWorld._rules[k] = r
        r['n'] += 1
        seen_here = r['games'].get(g, 0)
        r['games'][g] = seen_here + 1
        if seen_here == 0 and r['born'] != g:
            ARCWorld._transfers += 1
        if len(ARCWorld._rules) > 400:
            try:
                weak = sorted(ARCWorld._rules.items(),
                              key=lambda kv: kv[1]['n'])[:100]
                for kk, _ in weak:
                    ARCWorld._rules.pop(kk, None)
            except Exception:
                pass

    def _seed_from_rules(self, g, sig):
        """Entering a game, start from what he already knows: seed the
        action->displacement hypothesis with rules earned elsewhere.
        Seeded at weight 1 so a handful of live observations overrides a
        wrong prior immediately -- a hypothesis, never a commitment."""
        if g in ARCWorld._seeded or sig is None:
            return
        ARCWorld._seeded.add(g)
        hyp = ARCWorld._selfhyp.setdefault(g, {}).setdefault(sig, {})
        n = 0
        for k, r in ARCWorld._rules.items():
            try:
                if r['n'] < 3 or g in r['games']:
                    continue
                _, a, dy, dx = k.split('|')
                d = hyp.setdefault(int(a), {})
                key = (int(dy), int(dx))
                if key not in d:
                    d[key] = 1
                    n += 1
            except Exception:
                continue
        if n:
            ARCWorld._seed_hits = getattr(ARCWorld, '_seed_hits', 0) + n

    def rules_to_dict(self):
        out = {}
        try:
            for k, r in ARCWorld._rules.items():
                out[k] = [int(r['n']), str(r['born']),
                          list(r['games'].keys())[:26]]
        except Exception:
            pass
        return out

    def rules_from_dict(self, d):
        try:
            for k, v in (d or {}).items():
                if k in ARCWorld._rules:
                    continue
                ARCWorld._rules[k] = {
                    'n': int(v[0]), 'born': str(v[1]),
                    'games': dict((str(x), 1) for x in (v[2] or []))}
        except Exception:
            pass

    def _calib_action(self):
        """The action most in need of trying, or None when done.
        Deliberately independent of the self: action variety is what
        contingency needs in order to FIND the self in the first place,
        so this cannot be gated behind knowing it."""
        try:
            g = str(self.game_id)
            n = int(self._n_actions or 0)
            if n <= 1:
                return None
            tries = ARCWorld._act_tries.setdefault(g, {})
            need = [a for a in range(n) if tries.get(a, 0) < 2]
            if not need:
                return None
            return min(need, key=lambda a: tries.get(a, 0))
        except Exception:
            return None

    def _note_try(self, action):
        try:
            g = str(self.game_id)
            t = ARCWorld._act_tries.setdefault(g, {})
            t[int(action)] = t.get(int(action), 0) + 1
        except Exception:
            pass

    def _boxes(self, G):
        """A TARGET is a dark frame fully enclosing a figure, sitting
        outside the playfield -- verified on ls20, where it found both
        on-screen templates at exactly the positions a human sees them,
        and they stayed constant for the whole level (a target does; a
        wall does too, but a wall encloses nothing)."""
        from scipy import ndimage
        out = []
        try:
            for v in np.unique(G):
                lab, n = ndimage.label(G == v)
                if n <= 0 or n > 200:
                    continue
                for sl in ndimage.find_objects(lab):
                    if sl is None:
                        continue
                    r0, r1 = sl[0].start, sl[0].stop
                    c0, c1 = sl[1].start, sl[1].stop
                    if not (5 <= r1 - r0 <= 16 and 5 <= c1 - c0 <= 16):
                        continue
                    inner = G[r0 + 1:r1 - 1, c0 + 1:c1 - 1]
                    if inner.size == 0:
                        continue
                    if not (set(int(x) for x in np.unique(inner)) - {int(v)}):
                        continue
                    bd = np.concatenate([G[r0, c0:c1], G[r1 - 1, c0:c1],
                                         G[r0:r1, c0], G[r0:r1, c1 - 1]])
                    if (bd == v).mean() < 0.8:
                        continue
                    out.append((int(v), r0, c0, r1, c1))
        except Exception:
            return []
        return out

    def seed_doorways(self, pairs) -> int:
        """Refill the doorway cache from what he already knows.

        `pairs` is [((colour, size), times_it_ended_a_level)], read back
        out of the substrate by the actor.  Only ADDS -- a doorway
        confirmed live this run is never overwritten by an older count."""
        n = 0
        try:
            for k, w in pairs:
                kk = (int(k[0]), int(k[1]))
                if ARCWorld._goal_kinds.get(kk, 0) < int(w):
                    ARCWorld._goal_kinds[kk] = int(w)
                    n += 1
                if ARCWorld._goal_colours.get(kk[0], 0) < int(w):
                    ARCWorld._goal_colours[kk[0]] = int(w)
        except Exception:
            return n
        return n

    def _roles_now(self):
        """Classify what is on the board into roles he can carry into
        the next game.  Anything unexplained is UNKNOWN on purpose --
        that is what curiosity is for."""
        g = str(self.game_id)
        out = []
        try:
            G = np.array(self._grid, dtype=np.int16)
        except Exception:
            return out
        objs = ARCWorld._objs.get(g) or []
        if not objs:
            return out
        # SELF -- already earned by contingency
        sig = (ARCWorld._self.get(g) or (None,))[0]
        if sig is not None:
            out.append((sig, 'self'))
        # LIFE -- the drained bar line, already located
        bar = ARCWorld._bar.get(g)
        if bar:
            out.append(((int(bar[2]), 0), 'life'))
        # TARGET -- a frame enclosing a figure
        for v, r0, c0, r1, c1 in self._boxes(G):
            inner = G[r0 + 1:r1 - 1, c0 + 1:c1 - 1]
            vals, cts = np.unique(inner, return_counts=True)
            fig = [(int(a), int(b)) for a, b in zip(vals, cts)
                   if int(a) != int(v)]
            if not fig:
                continue
            fc = max(fig, key=lambda x: x[1])
            out.append(((fc[0], fc[1]), 'target'))
        # UNKNOWN -- but only what he has actually MET and still cannot
        # explain (2026-08-05).  The first cut labelled every singleton
        # object unknown and reached 256 members against 14 for all named
        # roles combined -- unbounded, and it would swamp abstraction with
        # a class that means nothing.  'I have not named that' is not the
        # same as 'I encountered that and it puzzled me'.
        #
        # So require CONTACT: he has been near it (the same arrival test
        # the salience loop uses, `_seen_kind`), it is still unnamed, and
        # it is rare on this board.  That makes UNKNOWN mean 'investigated
        # and still unexplained' -- a real open question, and exactly what
        # curiosity should be pointed at.
        named = set(k for k, _ in out)
        kinds = {}
        for o in objs:
            kinds[(o[0], o[1])] = kinds.get((o[0], o[1]), 0) + 1
        for k, cnt in kinds.items():
            if k in named or cnt != 1:
                continue
            if ARCWorld._seen_kind.get((g, k[0], k[1]), 0) <= 0:
                continue          # never actually met it
            out.append((k, 'unknown'))
        return out

    def _counts_now(self):
        """ELEMENTARY NUMBER SENSE -- the one Core Knowledge prior he
        did not have (2026-08-05).

        ARC-AGI declares which priors a solver may assume: object
        persistence, goal-directedness, elementary number sense
        (counting), and basic geometry/topology (connectivity, symmetry).
        He had four of the five -- segmentation gives persistence and
        connectivity, contingency gives agentness, mirror_x/y gives
        symmetry -- and NO counting at all.  Level 3 of ls20 is full of
        it: two yellow squares, two colour-matched targets, a piece with
        two parts.  'How many of these are there' was a question he could
        not ask.

        Returns (kind, n) per distinct object kind on the board, plus
        (role, n) per role.  Written into the substrate as
        has_property -> _arc_count_N, so abstraction can group things
        that come in twos and inference can compose over them."""
        out = []
        try:
            g = str(self.game_id)
            objs = ARCWorld._objs.get(g) or []
            if not objs:
                return out
            kinds = {}
            for o in objs:
                k = (int(o[0]), int(o[1]))
                kinds[k] = kinds.get(k, 0) + 1
            for k, n in kinds.items():
                out.append(('obj', k, int(n)))
            byrole = {}
            for k, role in self._roles_now():
                byrole.setdefault(role, set()).add((int(k[0]), int(k[1])))
            for role, ks in byrole.items():
                out.append(('role', role, len(ks)))
        except Exception:
            pass
        return out

    def _salient_target(self, g, objs, selfpos):
        """The most worth-investigating object, ego-centric.

        Returns (dy, dx, salience).  Salience is rarity on THIS board
        damped by how often he has already been to that kind -- the box
        stops being interesting once you have looked in it.
        Salience near 0 means nothing stands out, and the caller falls
        back to ordinary exploration."""
        if not objs or selfpos is None:
            return None
        sy, sx = selfpos
        kinds = {}
        for o in objs:
            k = (o[0], o[1])
            kinds[k] = kinds.get(k, 0) + 1
        # HELD TARGET FIRST: if the thing he committed to is still on
        # the board, keep going to it.  Only choose afresh when it is
        # gone or has been reached.
        held = ARCWorld._target.get(g)
        if held is not None and (int(held[0]), int(held[1])) in (
                ARCWorld._target_failed.get((g, int(self._levels))) or ()):
            ARCWorld._target.pop(g, None)      # buried; choose afresh
            held = None
        best = None
        if held is not None:
            cand = [o for o in objs
                    if o[0] == held[0]
                    and abs(o[1] - held[1]) <= max(1, held[1] // 4)
                    and abs(o[2] - sy) + abs(o[3] - sx) >= 1.0]
            if cand:
                o = min(cand, key=lambda x: abs(x[2] - sy) + abs(x[3] - sx))
                been = ARCWorld._seen_kind.get((g, held[0], held[1]), 0)
                rar = 1.0 / float(kinds.get((o[0], o[1]), 1))
                best = (o[2] - sy, o[3] - sx, rar / (1.0 + been),
                        (o[0], o[1]))
            else:
                ARCWorld._target.pop(g, None)
        if best is None:
            # ideas already tried and buried on THIS level.  If every
            # candidate has failed, clear and cycle -- nothing is removed
            # permanently, the board may have changed under him.
            _fk = (g, int(self._levels))
            _failed = ARCWorld._target_failed.get(_fk) or set()
            # EXCLUDE HIM.  `objs` contains his own body and his kind is
            # never buried, so counting it made this unsatisfiable and
            # left him with no target once every real candidate was
            # buried.  Same self-test the picking loop below uses.
            _cands = [o for o in objs
                      if abs(o[2] - sy) + abs(o[3] - sx) >= 1.0]
            if _failed and _cands and all(
                    ((o[0], o[1]) in _failed) for o in _cands):
                ARCWorld._target_failed.pop(_fk, None)
                _failed = set()
            for o in objs:
                dy = o[2] - sy
                dx = o[3] - sx
                if abs(dy) + abs(dx) < 1.0:
                    continue                  # that is him
                k = (o[0], o[1])
                if k in _failed:
                    continue     # attempt ended holding this idea -- try another
                rarity = 1.0 / float(kinds.get(k, 1))
                been = ARCWorld._seen_kind.get((g, k[0], k[1]), 0)
                salience = rarity / (1.0 + been)
                # DOES THIS LOOK LIKE A DOOR HE HAS BEEN THROUGH?
                # exact kind first, then colour alone (survives a level
                # redrawing the same goal at a different size).  The
                # weight is the measured rate at which reaching this kind
                # ACTUALLY ended a level, so a wrong guess earns nothing.
                _w = ARCWorld._goal_kinds.get(k, 0)
                _match = 'kind' if _w else ''
                if not _w:
                    _w = ARCWorld._goal_colours.get(k[0], 0)
                    _match = 'colour' if _w else ''
                if _w:
                    _rate = float(_w) / (1.0 + float(been))
                    # undamped by `been`: familiarity kills curiosity,
                    # never desire -- you do not stop wanting the exit
                    # because you have seen it before
                    _goalsal = rarity * _rate
                    if _goalsal > salience:
                        salience = _goalsal
                        ARCWorld._goal_last = (k, _match, round(_rate, 3))
                if best is None or salience > best[2]:
                    best = (dy, dx, salience, k)
            if best is not None:
                ARCWorld._target[g] = best[3]
                ARCWorld._commits = getattr(ARCWorld, '_commits', 0) + 1
        if best is None:
            return None
        # arriving satisfies the curiosity: mark it seen when close
        _reach = 3.0 + (float(best[3][1]) ** 0.5) / 2.0
        if abs(best[0]) + abs(best[1]) <= _reach:
            kk = (g, best[3][0], best[3][1])
            ARCWorld._seen_kind[kk] = ARCWorld._seen_kind.get(kk, 0) + 1
            ARCWorld._target.pop(g, None)   # reached it; pick a new one
            if len(ARCWorld._seen_kind) > 4000:
                ARCWorld._seen_kind.clear()
        return (best[0], best[1], best[2])

    def _bar_observe(self):
        """Learn this (game, level)'s budget row from his own frames.

        CLOCKLINE_ON: also counts, per line, whether its histogram
        moved forward and which lines tick together, and decides with
        `_clock_decide` (a clock need not move alone).

        Costs one pass over 64 row hashes per frame while learning, and
        nothing at all once decided.  The per-life value sets are cleared
        every life, so memory is bounded by ONE life, not by the corpus.
        """
        if not _BARMASK_ON():
            return
        try:
            g = str(self.game_id)
            lv = int(self._levels)
            k = (g, lv)
            # A TERMINAL OR A LEVEL CHANGE ENDS A LIFE.  `back` is only
            # meaningful within one life -- a bar legitimately replays
            # its whole sequence on the next life, and counting that as
            # "went back" is what rejected cd82 the first time I tried.
            # THIS RUNS BEFORE THE "DECIDED" RETURN (review 2026-09-05):
            # when a decided level returned first, `_bar_lv` stayed
            # stale, so coming BACK to an undecided level was not seen
            # as a life boundary, the replayed life all counted as
            # `back`, and that level could never decide.
            term = (str(self._state) != "NOT_FINISHED")
            if term or ARCWorld._bar_lv.get(g) != lv:
                ARCWorld._bar_lv[g] = lv
                ARCWorld._bar_seen.pop(k, None)
                ARCWorld._bar_prev.pop(k, None)
                ARCWorld._bar_pflat.pop(k, None)
                ARCWorld._bar_sign.pop(k, None)
                if term:
                    return
            if k in ARCWorld._bar_row:
                return                          # decided; nothing to do
            grid = self._grid
            if not grid or len(grid) != 64:
                return
            _flat = bytes(bytearray(int(c) & 0xFF
                                    for row in grid for c in row))
            rh = [hash(_flat[i * 64:(i + 1) * 64]) for i in range(64)]
            # _flat[c::64] IS column c -- a strided byte slice, done in C.
            rh.extend(hash(_flat[c::64]) for c in range(64))
            seen = ARCWorld._bar_seen.get(k)
            if seen is None:
                seen = [set() for _ in range(_BAR_LINES)]
                ARCWorld._bar_seen[k] = seen
            prev = ARCWorld._bar_prev.get(k)
            if prev is not None and len(prev) == _BAR_LINES:
                st = ARCWorld._bar_stat.get(k)
                if st is None:
                    st = [0, [0] * _BAR_LINES, [0] * _BAR_LINES,
                      [0] * _BAR_LINES]
                    ARCWorld._bar_stat[k] = st
                ch = [r for r in range(_BAR_LINES)
                      if prev[r] != rh[r]]
                st[0] += 1
                ARCWorld._bar_learn_n += 1
                for r in ch:
                    st[1][r] += 1
                    if rh[r] in seen[r]:
                        st[3][r] += 1
                # SOLO IS PER AXIS.  A bar cell changing alters its
                # row AND its column, so "exactly one line changed" can
                # never be true on a 2-D board.  The test is: it is the
                # only line on ITS OWN axis that moved.
                _rows = [i for i in ch if i < 64]
                _cols = [i for i in ch if i >= 64]
                if len(_rows) == 1:
                    st[2][_rows[0]] += 1
                if len(_cols) == 1:
                    st[2][_cols[0]] += 1
                if _CLOCKLINE_ON():
                    while len(st) < 6:
                        st.append({} if len(st) == 4 else [0] * _BAR_LINES)
                    _co, _fw = st[4], st[5]
                    # WHICH LINES TICK TOGETHER.  Pairs among the
                    # lines that changed this step: ~50 on an ordinary
                    # step, ~8k once on a full redraw.
                    for _a in ch:
                        for _b in ch:
                            if _a < _b:
                                _co[(_a, _b)] = _co.get((_a, _b), 0) + 1
                    # DID THE LINE'S HISTOGRAM MOVE FORWARD.  Per
                    # changed line, the colour counts before and after;
                    # forward = something changed and no colour turned
                    # around within this life.
                    _pf = ARCWorld._bar_pflat.get(k)
                    _sg = ARCWorld._bar_sign.get(k)
                    if _sg is None:
                        _sg = {}
                        ARCWorld._bar_sign[k] = _sg
                    if _pf is not None and len(_pf) == 4096:
                        for _l in ch:
                            if _l < 64:
                                _o = _pf[_l * 64:(_l + 1) * 64]
                                _w = _flat[_l * 64:(_l + 1) * 64]
                            else:
                                _o = _pf[_l - 64::64]
                                _w = _flat[_l - 64::64]
                            _ok = None
                            for _v in range(16):
                                _dv = _w.count(_v) - _o.count(_v)
                                if not _dv:
                                    continue
                                _s1 = 1 if _dv > 0 else -1
                                if _sg.get((_l, _v), _s1) != _s1:
                                    _ok = False
                                elif _ok is None:
                                    _ok = True
                                _sg[(_l, _v)] = _s1
                            if _ok:
                                _fw[_l] += 1
                if st[0] % _BAR_DECIDE_EVERY == 0:
                    d = (ARCWorld._clock_decide(st) if _CLOCKLINE_ON()
                         else ARCWorld._bar_decide(st))
                    # LATCH ONLY ON A REPEAT, AND NEVER LATCH A "none".
                    # Measured: at n=400 the rule agrees with the full
                    # corpus on 34 of 36 cells, and both misses are
                    # "none" where the answer is a row -- it never named
                    # a WRONG row.  So a "none" must keep learning.
                    if d != -1 and ARCWorld._bar_cand.get(k) == d:
                        ARCWorld._bar_row[k] = d
                        ARCWorld._bar_decided += 1
                        if _CLOCKLINE_ON():
                            # every decision is journaled: the answer
                            # key in barwatch.sh is the falsifier
                            try:
                                import sys as _sys
                                _sys.stderr.write(
                                    '[clockline] DECIDED game=%s lv=%d '
                                    'lines=%r n=%d\n' % (g, lv, d, st[0]))
                                _sys.stderr.flush()
                            except Exception:
                                pass
                        ARCWorld._bar_stat.pop(k, None)
                        ARCWorld._bar_seen.pop(k, None)
                        ARCWorld._bar_prev.pop(k, None)
                        ARCWorld._bar_pflat.pop(k, None)
                        ARCWorld._bar_sign.pop(k, None)
                        return
                    ARCWorld._bar_cand[k] = d
                    if st[0] >= _BAR_GIVE_UP:
                        ARCWorld._bar_row[k] = -1
                        ARCWorld._bar_stat.pop(k, None)
                        ARCWorld._bar_seen.pop(k, None)
                        ARCWorld._bar_prev.pop(k, None)
                        ARCWorld._bar_pflat.pop(k, None)
                        ARCWorld._bar_sign.pop(k, None)
                        return
            for r in range(_BAR_LINES):
                seen[r].add(rh[r])
            ARCWorld._bar_prev[k] = rh
            if _CLOCKLINE_ON():
                ARCWorld._bar_pflat[k] = _flat
        except Exception:
            pass

    @staticmethod
    def _bar_decide(st):
        """The line that moves ALONE ON ITS AXIS and never goes back.

        0-63 are rows, 64-127 are columns.  One candidate, or one that
        dominates the runner-up by _BAR_DOMINANCE -- see the header of
        patch_barcols.py for why uniqueness alone stopped working once
        columns were scanned.
        """
        n = st[0]
        if n <= 0:
            return -1
        tick, solo, back = st[1], st[2], st[3]
        cand = sorted(
            ((solo[r] / float(n), r) for r in range(_BAR_LINES)
             if tick[r] > 0
             and (solo[r] / float(n)) >= _BAR_SOLO_MIN
             and (back[r] / float(tick[r])) <= _BAR_BACK_MAX),
            reverse=True)
        if not cand:
            return -1
        if len(cand) == 1:
            return cand[0][1]
        return (cand[0][1]
                if cand[0][0] >= _BAR_DOMINANCE * cand[1][0] else -1)

    @staticmethod
    def _clock_decide(st):
        """The meter that ticks often, never goes back, and dominates.

        CLOCKLINE.  Candidates: tick on >= _BAR_SOLO_MIN of steps (the
        same floor the solo rule uses, now on the tick rate), land on a
        value seen this life on <= _BAR_BACK_MAX of their ticks, and
        move their histogram forward on >= _BAR_METER_MIN of their
        ticks (a meter; a self that moves does not).  Candidates in
        lockstep (co-ticks / the larger tick count >= _BAR_COTICK) are
        one structure; a group of more than _BAR_STRUCT_MAX lines is
        content redrawn together and is dropped.  The top structure
        wins if it ticks >= _BAR_DOMINANCE times the runner-up;
        failing that, the only near-tie that passes the solo rule
        (moves alone on >= _BAR_SOLO_MIN of steps) wins.  Returns an
        int for one line, a sorted tuple for a structure, -1 for "not
        yet" (also when no meter table exists yet).  Measured on two
        corpora (09-13): agrees with `_bar_decide` on every cell it
        decides (18 + 7), names a monotone marker on 21 + 8 it cannot,
        never a wrong line, never ambiguous.
        """
        n = st[0]
        if n <= 0 or len(st) < 6:
            return -1
        tick, solo, back, co, fwd = st[1], st[2], st[3], st[4], st[5]
        cand = [r for r in range(_BAR_LINES)
                if tick[r] > 0
                and (tick[r] / float(n)) >= _BAR_SOLO_MIN
                and (back[r] / float(tick[r])) <= _BAR_BACK_MAX
                and (fwd[r] / float(tick[r])) >= _BAR_METER_MIN]
        if not cand:
            return -1
        groups = []
        for r in cand:
            for grp in groups:
                if all((co.get((min(r, m), max(r, m)), 0)
                        / float(max(tick[r], tick[m]))) >= _BAR_COTICK
                       for m in grp):
                    grp.append(r)
                    break
            else:
                groups.append([r])
        groups = [grp for grp in groups if len(grp) <= _BAR_STRUCT_MAX]
        if not groups:
            return -1
        scored = sorted(((max(tick[r] for r in grp) / float(n), grp)
                         for grp in groups), reverse=True)
        top, rate = scored[0][1], scored[0][0]
        if len(scored) > 1 and rate < _BAR_DOMINANCE * scored[1][0]:
            # no dominance: the solo rule decides among the near-ties
            # (measured: content candidates reach its floor on 3 of
            # 125 lines, all in 5-step lives; one stray solo event is
            # carried by 12 of 125, so an event is not enough)
            tied = [grp for r, grp in scored
                    if r * _BAR_DOMINANCE > rate]
            alone = [grp for grp in tied
                     if any((solo[r] / float(n)) >= _BAR_SOLO_MIN
                            for r in grp)]
            if len(alone) != 1:
                return -1
            top = alone[0]
        return top[0] if len(top) == 1 else tuple(sorted(top))

    def _bar_of(self):
        """The learned budget row here, or -1 for 'no bar / not yet'.

        A CLOCKLINE structure answers with its first line: the callers
        that take one int (RELPLAN's object reader, the place key) mask
        that line and see the rest, as they did before this patch.
        """
        if not _BARMASK_ON():
            return -1
        try:
            r = ARCWorld._bar_row.get(
                (str(self.game_id), int(self._levels)), -1)
            if isinstance(r, (tuple, list)):
                return int(r[0]) if r else -1
            return int(r)
        except Exception:
            return -1

    def _bar_lines(self):
        """Every line of the learned budget structure here; () if none."""
        if not _BARMASK_ON():
            return ()
        try:
            r = ARCWorld._bar_row.get(
                (str(self.game_id), int(self._levels)), -1)
            if isinstance(r, (tuple, list)):
                return tuple(int(x) for x in r
                             if 0 <= int(x) < _BAR_LINES)
            r = int(r)
            return (r,) if r >= 0 else ()
        except Exception:
            return ()

    def _grid_changed_nb(self, prev_grid):
        """Did anything change OUTSIDE the budget line?

        Grid-level rather than hash-level because the caller already
        holds the previous GRID, and hashing both would cost a second
        pass over 4,096 cells on every step.
        """
        if prev_grid is None or self._grid is None:
            return self._grid != prev_grid
        r = self._bar_of() if _CLICKHIT_ON() else -1
        if r < 0:
            return self._grid != prev_grid
        try:
            # class-level on purpose: the test stubs mirror single methods
            _ls = ARCWorld._bar_lines(self)
            if len(_ls) > 1:
                _rs = set(x for x in _ls if x < 64)
                _cs = set(x - 64 for x in _ls if x >= 64)
                for i, row in enumerate(self._grid):
                    if i in _rs:
                        continue
                    pr = prev_grid[i]
                    if row == pr:
                        continue
                    if not _cs:
                        return True
                    for j, v in enumerate(row):
                        if j not in _cs and v != pr[j]:
                            return True
                return False
            if r < 64:
                for i, row in enumerate(self._grid):
                    if i != r and row != prev_grid[i]:
                        return True
                return False
            c = r - 64
            for i, row in enumerate(self._grid):
                pr = prev_grid[i]
                if row == pr:
                    continue
                for j, v in enumerate(row):
                    if j != c and v != pr[j]:
                        return True
            return False
        except Exception:
            return self._grid != prev_grid

    def _board_hash_nb(self):
        """`_board_hash` WITHOUT the budget row.

        Falls through to `_board_hash` when the gate is off or the row
        is not known, so every caller is byte-identical until he has
        actually learned something.
        """
        r = self._bar_of()
        if r < 0:
            return self._board_hash()
        try:
            import hashlib
            g = self._grid
            if g is None:
                return None
            # class-level on purpose: the test stubs mirror single methods
            _ls = ARCWorld._bar_lines(self)
            if len(_ls) > 1:
                _rs = set(x for x in _ls if x < 64)
                _cs = set(x - 64 for x in _ls if x >= 64)
                b = bytes(bytearray(
                    v & 0xFF
                    for i, row in enumerate(g) if i not in _rs
                    for j, v in enumerate(row) if j not in _cs))
            elif r < 64:
                b = bytes(bytearray(
                    v & 0xFF
                    for i, row in enumerate(g) if i != r
                    for v in row))
            else:
                _c = r - 64
                b = bytes(bytearray(
                    v & 0xFF
                    for row in g
                    for j, v in enumerate(row) if j != _c))
            return hashlib.blake2b(b, digest_size=8).digest()
        except Exception:
            return None

    def _fkey(self):
        """The board as one key, STEP-BUDGET ROW 63 MASKED.

        Masking measured: states 8,660 -> 5,790 (-33%) and MORE
        recurring pairs, because otherwise the same board with a
        different budget reading is a different state -- which is what
        destroys recurrence.  Cancellation by INVARIANCE: a
        world-referenced key cannot carry his gaze.
        """
        try:
            if not self._grid:
                return None
            import hashlib as _hl
            _b = bytearray()
            for _i, _row in enumerate(self._grid):
                if _i == 63:
                    continue
                _b.extend(int(_c) & 0xFF for _c in _row)
            return _hl.blake2b(bytes(_b), digest_size=8).digest()
        except Exception:
            return None

    def prospective_aim(self, a):
        """The coordinate action `a` WOULD use if chosen right now.

        PURE -- must not advance `_click_turn`, which `step()` mutates
        at its round-robin.  Returns None for a non-click, because
        only the coordinate action (id 6) has its outcome decided by
        the aim; a non-click sends y/x but ignores it.
        """
        try:
            _ai = int(a)
            if not (0 <= _ai < len(self._acts)
                    and int(self._acts[_ai]) == 6):
                return None
            if self._locus is None:
                return None
            if self._click_hits:
                _t = int(self._click_turn) + 1
                if _t % 2 == 0:
                    _k = sorted(self._click_hits)
                    return _k[(_t // 2) % len(_k)]
            return (int(self._locus[0]), int(self._locus[1]))
        except Exception:
            return None

    def frontier_scores(self, actions):
        """`1/(1+tries)` per action AT THIS BOARD, AIMED WHERE IT WILL
        AIM.  Called from the choice site, which is the only place the
        executed aim is knowable.
        """
        try:
            h = self._fkey()
            if h is None:
                return None
            st = ARCWorld._ftried.get(
                (str(self.game_id), int(self._levels))) or {}
            out = {}
            for a in actions:
                _k = (h, int(a), self.prospective_aim(a))
                out[int(a)] = 1.0 / (1.0 + float(st.get(_k, 0)))
            # STAMPED WITH THE GAME and consumed once.  Unstamped, the
            # booking cannot distinguish "premise violated" from "no
            # prospective aim computed this step": an absent key
            # defaulted to 0 and compared unequal to a non-click None,
            # so the gauge read ~100% miss while the premise held.
            ARCWorld._fp_aim = (str(self.game_id), dict(
                (int(a), self.prospective_aim(a)) for a in actions))
            return out
        except Exception:
            return None

    def _body_novelty(self, g, selfpos):
        """Per-action novelty of the PLACE each action would take him,
        using his own learned action->displacement model.  Returns None
        until he knows who he is, so nothing changes in games without a
        self."""
        sig = (ARCWorld._self.get(g) or (None,))[0]
        if sig is None or selfpos is None:
            return None, 0.0
        hyp = (ARCWorld._selfhyp.get(g) or {}).get(sig) or {}
        if not hyp:
            return None, 0.0
        key = (int(round(selfpos[0])), int(round(selfpos[1])))
        vis = ARCWorld._posvis.setdefault((g, int(self._levels)), {})
        vis[key] = vis.get(key, 0) + 1
        if len(vis) > 8192:
            vis.clear()
        out = {}
        tgt = self._salient_target(g, ARCWorld._objs.get(g), selfpos)
        ARCWorld._last_sal = (tgt[2] if tgt else 0.0)
        for a, dd in hyp.items():
            # EXCLUDE (0,0): the modal outcome of most actions is 'stayed
            # put', which would make every action predict no movement and
            # no action could ever approach anything.  Same conditioning
            # that made self-detection work.
            mv = dict((q, c) for q, c in dd.items() if q != (0, 0))
            if not mv:
                continue
            dy, dx = max(mv.items(), key=lambda kv: kv[1])[0]
            _nov = 1.0 / (1.0 + vis.get((key[0] + dy, key[1] + dx), 0))
            if tgt is not None and tgt[2] > 0.0:
                # how much closer this action gets him to the thing
                # worth investigating, using his OWN learned
                # action->displacement rule
                _d0 = abs(tgt[0]) + abs(tgt[1])
                _d1 = abs(tgt[0] - dy) + abs(tgt[1] - dx)
                _appr = 1.0 / (1.0 + _d1)
                _sal = min(1.0, float(tgt[2]))
                # ONE BOX IN AN EMPTY ROOM -> go to it.  A room full of
                # identical boxes -> nothing stands out, so explore.
                _nov = (1.0 - _sal) * _nov + _sal * _appr
                if _d1 < _d0:
                    ARCWorld._approach_n = getattr(
                        ARCWorld, '_approach_n', 0) + 1
            out[int(a)] = _nov
        got = ARCWorld._self.get(g) or (None, 0.0, 0)
        n = float(got[2] or 0)
        conf = float(got[1] or 0.0) * (n / (1.0 + n))
        return (out or None), conf

    def _locus_novelty(self, g):
        """Per-action novelty of WHERE THE WORLD WOULD CHANGE.

        The no-avatar twin of `_body_novelty`: same shape, same approach
        term, same earned confidence -- but keyed on the centroid of
        change rather than on his body, so it works in the 7 games where
        contingency never finds a self.  Returns None until the model has
        something to say, so it can never make things worse."""
        pos = ARCWorld._locus_prev.get(g)
        hyp = ARCWorld._locushyp.get(g) or {}
        if pos is None or not hyp:
            return None, 0.0
        key = (int(round(pos[0])), int(round(pos[1])))
        vis = ARCWorld._locusvis.setdefault((g, int(self._levels)), {})
        vis[key] = vis.get(key, 0) + 1
        if len(vis) > 8192:
            vis.clear()
        out = {}
        tgt = self._salient_target(g, ARCWorld._objs.get(g), pos)
        # the target ON THE TABLE for the choice he is about to make;
        # the next step scores his action against THIS, not against
        # whatever the target became afterwards
        ARCWorld._locus_last_tgt = tgt
        _det = []
        for a, dd in hyp.items():
            mv = dict((q, c) for q, c in dd.items() if q != (0, 0))
            if not mv:
                continue
            _tot = float(sum(mv.values()))
            dy, dx = max(mv.items(), key=lambda kv: kv[1])[0]
            _det.append(max(mv.values()) / _tot if _tot else 0.0)
            _nov = 1.0 / (1.0 + vis.get((key[0] + dy, key[1] + dx), 0))
            if tgt is not None and tgt[2] > 0.0:
                _d0 = abs(tgt[0]) + abs(tgt[1])
                _d1 = abs(tgt[0] - dy) + abs(tgt[1] - dx)
                _appr = 1.0 / (1.0 + _d1)
                _sal = min(1.0, float(tgt[2]))
                _nov = (1.0 - _sal) * _nov + _sal * _appr
                if _d1 < _d0:
                    ARCWorld._locus_approach = getattr(
                        ARCWorld, '_locus_approach', 0) + 1
            out[int(a)] = _nov
            if tgt is not None and tgt[2] > 0.0:
                ARCWorld._locus_actions_scored = getattr(
                    ARCWorld, '_locus_actions_scored', 0) + 1
        if not out or not _det:
            return None, 0.0
        # earned exactly like the self model: determinism * n/(1+n)
        n = float(sum(sum(d.values()) for d in hyp.values()))
        conf = (sum(_det) / len(_det)) * (n / (1.0 + n))
        return out, conf

    def _resolve_pending(self, g, cur, selfpos, med):
        """Settle vanished blobs: back when he moved off = occlusion,
        still absent = consumed."""
        pend = ARCWorld._pending.get(g)
        if not pend or selfpos is None:
            return
        sy, sx = selfpos
        keep = []
        tab = ARCWorld._occl.setdefault(g, [0, 0])
        for (col, sz, cy, cx) in pend:
            back = False
            for o in cur:
                if o[0] != col or abs(o[1] - sz) > max(1, sz // 4):
                    continue
                if abs(o[2] - cy) + abs(o[3] - cx) <= 2.0:
                    back = True
                    break
            if back:
                tab[0] += 1               # reappeared -> he was ON it
                continue
            if abs(cy - sy) + abs(cx - sx) > med:
                tab[1] += 1               # he left, still gone -> consumed
                continue
            keep.append((col, sz, cy, cx))
        ARCWorld._pending[g] = keep[-64:]

    def _consumables(self, g):
        """Colours that vanish MUCH more often near him than far.
        Ratio against its own far-rate, so a colour that disappears all
        over the board scores nothing."""
        tab = ARCWorld._afford.get(g) or {}
        out = []
        for col, (near, far, seen) in tab.items():
            if near + far < 6:
                continue
            r = near / float(near + far)
            if r > 0.65:
                out.append((int(col), round(r, 3), near, far))
        out.sort(key=lambda x: -x[1])
        return out

    def _ensure_mine(self) -> None:
        # `not self._live` is the third condition (2026-07-29): his grid was
        # stale-but-not-None and _owner still matched, so a sidecar that had
        # lost its game was never re-opened and he inhabited a frozen world
        # for 12h45m.  Self-limiting -- a successful open sets _live back to
        # True; a failing one is retried and every attempt is counted.
        if (self._grid is None or ARCWorld._owner != self.game_id
                or not self._live):
            self.reopens += 1
            self._absorb(self._call({'cmd': 'open', 'game': self.game_id,
                                     'seed': self._seed}))
            ARCWorld._owner = self.game_id

    def _glance(self, gl) -> tuple:
        """Pre-attentive scan -> the least-visited window.  Same rule as
        attention_eye.attend (his _visits, no hand-coded saliency), but the
        SCAN DOES NOT COMMIT.

        attend() called transducer.encode() on every scanned window, and
        encode() MINTS a prototype when nothing matches -- so merely
        sweeping the frame wrote windows into his prototype memory that he
        never attended, and cost O(prototypes) per locus.  Both the scan
        length and the prototype count grow with experience, which is why
        a step drifted 2.2ms -> 82ms.  Pre-attentive vision is cheap and
        non-committal; ATTENTION is what commits.

        The transducer is exact-match, so window-tuple identity IS token
        identity: a tuple already attended maps to its token in O(1), and a
        tuple never attended is by definition unvisited.  Identical
        selection, no minting, no distance search.
        """
        r = self._radius
        st = self._stride
        H = len(gl)
        W = len(gl[0]) if H else 0
        seen = self._seen_win
        visits = self._visits
        best = None
        _pv = None
        if _LOCUSNOV_ON():
            _pv = ARCWorld._posvis.setdefault(
                (self.game_id, int(self._levels)), {})
        for rr in range(r, H - r, st):
            for cc in range(r, W - r, st):
                win = tuple(v for dr in range(-r, r + 1)
                            for v in gl[rr + dr][cc - r:cc + r + 1])
                tok = seen.get(win)
                if tok is None:
                    self._locus = (rr, cc)      # he is looking HERE
                    if _pv is not None:
                        _pv[rr * W + cc] = _pv.get(rr * W + cc, 0) + 1
                    return win                  # never attended -> take it
                v = visits.get(tok, 0)
                # CONTENT novelty ranks first; POSITION novelty only
                # breaks ties between identical windows.
                # int key: tuple hashing on every scanned position was
                # measurable per-glance cost.  Same tie-break, cheaper.
                _pk = rr * W + cc
                # PRODUCT OF NOVELTIES, not a lexicographic tie-break.
                # Lexicographic let position matter only on EXACT ties
                # in token-visit count; as counts diverge the ties
                # vanish and the sweep decays (measured 30.5% -> 44.0%
                # repeat within 9 h).  Both terms are counts whose
                # novelty is the 1/(1+n) form used throughout, so
                # max 1/((1+v)(1+p)) == min (1+v)*(1+p).  No new
                # constant; position always counts.
                _pn = _pv.get(_pk, 0) if _pv is not None else 0
                _k = ((1 + v) * (1 + _pn),) if _pv is not None else (v,)
                if best is None or _k < best[0]:
                    best = (_k, win, (rr, cc))
        if best is None:
            cr, cc = H // 2, W // 2
            self._locus = (cr, cc)
            return tuple(v for dr in range(-r, r + 1)
                         for v in gl[cr + dr][cc - r:cc + r + 1])
        self._locus = best[2]
        if _pv is not None:
            _bk = best[2][0] * W + best[2][1]
            _pv[_bk] = _pv.get(_bk, 0) + 1
        return best[1]

    # ---- LEVELHOLD: a life is judged on ITS LEVEL (2026-09-07) --------
    def _level_know(self, key):
        """Operator knowledge held for a level (logged, never a stay reason)."""
        try:
            rp = ARCWorld._relplan.get(key)
            return int(len(rp.effects) + len(rp.auto)) if rp is not None else 0
        except Exception:
            return 0

    def _level_life_begin(self):
        key = (str(self.game_id), int(self._levels))
        self._life_key = key
        self._life_hwm0 = int(ARCWorld._lvl_hwm.get(key, 0))
        self._life_won0 = int(getattr(self, '_won_here', 0))
        self._life_know0 = self._level_know(key)
        self._life_min_mm = None

    def _level_records_step(self):
        """Per step: open the life's book on its first step, and track the
        best canvas/target mismatch where a picture is pursued."""
        if not _LEVELHOLD_ON():
            return
        try:
            if self._grid is None:
                return
            if self._life_key is None:
                self._level_life_begin()
            key = self._life_key
            if int(self._levels) != key[1]:
                # HE CLEARED THE LEVEL MID-LIFE (review finding 1): close
                # this book -- the win is a record there -- and open one
                # on the level he has just arrived on, so the arrival
                # stretch is scored on ITS level and the hold read at the
                # terminal finds a booked, fresh level, not a stale drought.
                self._level_life_end()
                self._level_life_begin()
                key = self._life_key
            rs = ARCWorld._rel.get(key)
            _pic = self._rel_picture(rs, _rel_anyconf_of(self)) if rs is not None else None
            if _pic is None:
                return
            A, B = _pic[1], _pic[2]
            g = self._grid
            if (A[2] - A[0]) != (B[2] - B[0]) or (A[3] - A[1]) != (B[3] - B[1]):
                return
            mm = 0
            for r in range(A[2] - A[0] + 1):
                ra = g[A[0] + r][A[1]:A[3] + 1]
                rb = g[B[0] + r][B[1]:B[3] + 1]
                for x, y in zip(ra, rb):
                    if x != y:
                        mm += 1
            if self._life_min_mm is None or mm < self._life_min_mm:
                self._life_min_mm = mm
        except Exception:
            ARCWorld._lvl_errors += 1

    def _level_life_end(self):
        """At a terminal: was this life a RECORD on the level it began on?
        Records reset the drought and may raise the self-scaling bound;
        a dry life lengthens the drought."""
        if not _LEVELHOLD_ON():
            return
        try:
            key = self._life_key
            if key is None:
                return
            why = []
            if (int(self._levels) > key[1]
                    or int(getattr(self, '_won_here', 0)) > self._life_won0):
                why.append('win')
            if int(ARCWorld._lvl_hwm.get(key, 0)) > self._life_hwm0:
                why.append('depth')
            if self._life_min_mm is not None:
                best = ARCWorld._lvl_best_mm.get(key)
                if best is None or self._life_min_mm < int(best):
                    ARCWorld._lvl_best_mm[key] = int(self._life_min_mm)
                    why.append('picture')
            learned = self._level_know(key) - self._life_know0
            dry = int(ARCWorld._lvl_dry.get(key, 0))
            gm = int(ARCWorld._lvl_gap_max.get(key, 0))
            if why:
                if dry > gm:
                    gm = dry
                    ARCWorld._lvl_gap_max[key] = gm
                ARCWorld._lvl_dry[key] = 0
                ARCWorld._lvl_records += 1
            else:
                dry += 1
                ARCWorld._lvl_dry[key] = dry
            ARCWorld._lvl_last = '%s %s lv%d dry=%d max=%d' % (
                '+'.join(why) if why else 'dry', key[0][:4], key[1],
                int(ARCWorld._lvl_dry.get(key, 0)), gm)
            import sys as _sys
            _sys.stderr.write(
                '[levelhold] LIFE game=%s lv=%d rec=%s dry=%d gap_max=%d '
                'min_mm=%s best_mm=%s learned=%d hold=%s\n'
                % (key[0], key[1], '+'.join(why) if why else '-',
                   int(ARCWorld._lvl_dry.get(key, 0)), gm,
                   self._life_min_mm, ARCWorld._lvl_best_mm.get(key),
                   int(learned),
                   int(ARCWorld._lvl_dry.get(key, 0)) <= max(1, gm)))
        except Exception:
            ARCWorld._lvl_errors += 1
        finally:
            self._life_key = None

    def _level_forget_drought(self):
        """On re-entry: forget the droughts of this game's levels, keep the
        earned bounds.  Review finding 1: a stale drought from an earlier
        visit made level_hold() False on the step he won INTO that level,
        and the per-step `depleted and not hold` branch moved him on."""
        try:
            _g = str(self.game_id)
            for _k in [k for k in ARCWorld._lvl_dry if k[0] == _g]:
                ARCWorld._lvl_dry.pop(_k, None)
        except Exception:
            ARCWorld._lvl_errors += 1

    def level_hold(self) -> bool:
        """LEVELHOLD: is the drought of record-less lives on this level still
        within the longest he ever came back from here (floor: one retry)?"""
        if not _LEVELHOLD_ON():
            return False
        if not getattr(self, '_live', True):
            return False          # a dead world is not a place to stay (review finding 4)
        try:
            _k = (str(self.game_id), int(self._levels))
            _dry = int(ARCWorld._lvl_dry.get(_k, 0))
            _gm = int(ARCWorld._lvl_gap_max.get(_k, 0))
            return _dry <= max(1, _gm)
        except Exception:
            ARCWorld._lvl_errors += 1
            return False

    def _progress_step(self) -> int:
        """New high-water mark of irreversible change, else 0.

        ARC gives no score, so progress is CONSTRUCTED: cells that
        differ from the board at level start and have not reverted.
        Measured: d at completion sits at percentile 66.0 (median
        75.8) within its own level, 14/19, binomial p ~ 0.032.
        """
        if not _PROGRESS_ON():
            return 0
        try:
            key = (str(self.game_id), int(self._levels))
            cur = self._grid
            if cur is None:
                return 0
            base = ARCWorld._lvl_base.get(key)
            if base is None:
                ARCWorld._lvl_base[key] = [list(r) for r in cur]
                ARCWorld._lvl_hwm[key] = 0
                if len(ARCWorld._lvl_base) > 300:
                    ARCWorld._lvl_base.clear()
                    ARCWorld._lvl_hwm.clear()
                return 0
            d = 0
            # THE BUDGET LINE IS NOT PROGRESS (PROGMASK).  It differs
            # from the level-start board on nearly every step of the
            # first life, so unmasked it sets a high-water mark the
            # real content can never beat in later lives.
            _bl = self._bar_of() if _PROGMASK_ON() else -1
            _bls = ARCWorld._bar_lines(self) if _bl >= 0 else ()
            if len(_bls) > 1:
                _rs = set(x for x in _bls if x < 64)
                _cs = set(x - 64 for x in _bls if x >= 64)
                for _i, (r0, r1) in enumerate(zip(base, cur)):
                    if _i in _rs:
                        continue
                    for _j, (x, y) in enumerate(zip(r0, r1)):
                        if _j not in _cs and x != y:
                            d += 1
            elif _bl < 0:
                for r0, r1 in zip(base, cur):
                    for x, y in zip(r0, r1):
                        if x != y:
                            d += 1
            elif _bl < 64:
                for _i, (r0, r1) in enumerate(zip(base, cur)):
                    if _i == _bl:
                        continue
                    for x, y in zip(r0, r1):
                        if x != y:
                            d += 1
            else:
                _c = _bl - 64
                for r0, r1 in zip(base, cur):
                    for _j, (x, y) in enumerate(zip(r0, r1)):
                        if _j != _c and x != y:
                            d += 1
            hwm = ARCWorld._lvl_hwm.get(key, 0)
            if d > hwm:
                ARCWorld._lvl_hwm[key] = d
                ARCWorld._progress_n += 1
                # per-game and environment-wide, for the marginal
                # value test in depleted() rule (d)
                self._prog_here = getattr(self, "_prog_here", 0) + 1
                ARCWorld._prog_all = getattr(
                    ARCWorld, "_prog_all", 0) + 1
                return d - hwm
            return 0
        except Exception:
            return 0

    def seen_here(self, tok) -> bool:
        """Has he met this state IN THIS GAME?

        `_felt_seen` is per-world-instance, so this is the game-local
        membership test his GLOBAL `_route` never had.  Used to stop a
        gradient earned elsewhere from steering him here.
        """
        try:
            return tok in self._felt_seen
        except Exception:
            return True

    def _vkey(self, board, sent):
        if board is None or not sent:
            return None
        try:
            _a = int(sent[0])
            if len(sent) > 3 and sent[3] and sent[1] is not None:
                return (str(self.game_id), board,
                        (_a, int(sent[1]), int(sent[2])))
            return (str(self.game_id), board, (_a,))
        except Exception:
            return None

    def _judge_step(self, board, sent, success, state,
                    board_nb=None) -> None:
        """Record what that step DID.  No score exists to read, so the
        everyday evidence is whether it opened somewhere new or sent
        him back; an attempt ending and a level advance are rare but decisive."""
        if not _VERDICT_ON():
            return
        try:
            g = str(self.game_id)
            # A NEW LEVEL IS A NEW ATTEMPT.  Without this the set is
            # only cleared on a terminal, so a rotation away and back
            # grows it into "every board ever seen here" and every
            # step reads BACK -- which is why nothing was ever GOOD.
            _lv = int(self._levels)
            if ARCWorld._att_lv.get(g) != _lv:
                ARCWorld._att_lv[g] = _lv
                ARCWorld._att_boards.pop(g, None)
                ARCWorld._att_resets += 1
            att = ARCWorld._att_boards.setdefault(g, set())
            # THE ATTEMPT SET IS A WITHIN-LIFE TEST, so it holds
            # the BUDGET-MASKED board.  `_vkey` below still keys
            # on the FULL board -- the key is untouched, only the
            # verdict changes.
            if board_nb is not None:
                att.add(board_nb)
            elif board is not None:
                att.add(board)
            if len(att) > 20000:
                att.clear()
                ARCWorld._att_resets += 1
            h = self._board_hash_nb()
            k = self._vkey(board, sent)
            if k is not None:
                if success:
                    _slot = 0
                elif state == "GAME_OVER":
                    _slot = 1
                elif h is not None and h in att:
                    _slot = 3
                else:
                    _slot = 2
                for _d, _kk in ((ARCWorld._verdict, k),
                                (ARCWorld._verdict_i,
                                 (k[0], k[1], k[2][0]))):
                    e = _d.get(_kk)
                    if e is None:
                        e = [0, 0, 0, 0]
                        _d[_kk] = e
                    e[_slot] += 1
            if h is not None:
                att.add(h)
            if success or state in ("GAME_OVER", "WIN"):
                ARCWorld._att_boards.pop(g, None)
                ARCWorld._att_lv.pop(g, None)
                ARCWorld._att_resets += 1
            if len(ARCWorld._verdict) > 400000:
                ARCWorld._verdict.clear()
                ARCWorld._verdict_i.clear()
        except Exception:
            pass

    def step_verdict(self, action):
        """+1 GOOD, -1 BAD, 0 NOT YET KNOWN -- for this action from the
        board he is standing on.

        Constant-free: two observations minimum so every mistake is
        made twice before it is judged, decisive events outrank
        novelty, and a genuine tie stays UNCONFIRMED.
        """
        if not _VERDICT_ON():
            return 0
        try:
            h = self._board_hash()
            if h is None:
                return 0
            e = ARCWorld._verdict_i.get(
                (str(self.game_id), h, int(action)))
            if e is None or (e[0] + e[1] + e[2] + e[3]) < 2:
                ARCWorld._verdict_unk += 1
                return 0
            if e[0] != e[1]:
                _r = 1 if e[0] > e[1] else -1
            elif e[2] != e[3]:
                _r = 1 if e[2] > e[3] else -1
            else:
                ARCWorld._verdict_unk += 1
                return 0
            if _r > 0:
                ARCWorld._verdict_good += 1
            else:
                ARCWorld._verdict_bad += 1
            return _r
        except Exception:
            return 0

    def verdict_to_dict(self) -> dict:
        """Only THIS game, mirroring click_hits/act_effect/paths."""
        g = str(self.game_id)
        out = {}
        for k, v in ARCWorld._verdict.items():
            if k[0] != g:
                continue
            out[k[1].hex() + ":" + ",".join(str(x) for x in k[2])] = [
                int(v[0]), int(v[1]), int(v[2]), int(v[3])]
        return {g: out} if out else {}

    def verdict_from_dict(self, d) -> int:
        try:
            rows = (d or {}).get(str(self.game_id)) or {}
        except Exception:
            return 0
        g = str(self.game_id)
        n = 0
        for kk, v in rows.items():
            try:
                _b, _act = kk.split(":", 1)
                key = (g, bytes.fromhex(_b),
                       tuple(int(x) for x in _act.split(",")))
                _row = [int(v[0]), int(v[1]), int(v[2]), int(v[3])]
                ARCWorld._verdict[key] = _row
                # rebuild the aggregated index, or the restored
                # judgement would be unqueryable after a restart
                _ik = (g, key[1], key[2][0])
                _agg = ARCWorld._verdict_i.get(_ik)
                if _agg is None:
                    ARCWorld._verdict_i[_ik] = list(_row)
                else:
                    for _j in range(4):
                        _agg[_j] += _row[_j]
                n += 1
            except (TypeError, ValueError, IndexError, KeyError):
                continue
        return n

    @staticmethod
    def _erase_loops(seq):
        """Drop every detour that returned to a board already stood on.

        A board->action map loses order when the path revisits a board,
        which turned two of the seven measured levels into cycles.
        Loop-erasure cannot create one by construction.
        """
        out = []
        pos = {}
        for b, a in seq:
            if b in pos:
                out = out[:pos[b]]
                pos = dict((bb, i) for i, (bb, _x) in enumerate(out))
            pos[b] = len(out)
            out.append((b, a))
        return out

    def _path_key(self, lv=None):
        return (str(self.game_id),
                int(self._levels if lv is None else lv))

    def _note_path_step(self, board, sent, lv=None) -> None:
        """Remember where he stood and what he sent from there.

        `lv` is the level the step was TAKEN on.  Without it the step
        is filed under self._levels, which `_absorb` has ALREADY
        advanced on a winning move -- so the sealed path lost the very
        move that won and the next level inherited a stale board.
        """
        if board is None or not sent:
            return
        try:
            k = self._path_key(lv)
            seq = ARCWorld._cur_path.setdefault(k, [])
            if len(seq) < 6000:
                seq.append((board, (int(sent[0]), sent[1], sent[2])))
            ARCWorld._cur_seen.setdefault(k, set()).add(board)
        except Exception:
            pass

    def _shortcut_verdict(self, k, cleared, actions):
        """SHORTCUT (patch 40): the attempt's outcome, handed to the plan of
        that level (a no-op unless a shortcut was latched in it)."""
        try:
            _rp = ARCWorld._relplan.get(k)
            _ev = _rp.shortcut_outcome(cleared, actions) if _rp is not None else None
            if _ev:
                import sys as _sys
                _sys.stderr.write('[relwalk] game=%s lv=%d %s\n' % (k[0], int(k[1]), _ev))
                _sys.stderr.flush()
        except Exception:
            pass

    def _seal_path(self, lv) -> None:
        """That run finished the level.  Keep it if it is his best."""
        try:
            k = self._path_key(lv)
            ARCWorld._cur_seen.pop(k, None)
            seq = ARCWorld._cur_path.pop(k, None)
            if _SHORTCUT_ON():
                # judged on the actions this attempt actually took on the level
                ARCWorld._shortcut_verdict(self, k, True, len(seq or ()))
            if not seq:
                return
            er = ARCWorld._erase_loops(seq)
            if not er or len(er) > 4000:
                return
            cur = ARCWorld._paths.get(k)
            if cur is None or len(er) < len(cur):
                ARCWorld._paths[k] = dict(er)
                ARCWorld._path_sealed += 1
        except Exception:
            pass

    def _seal_depth(self, lv) -> None:
        """The life ended without clearing.  Keep the way back to the
        furthest he got, if it beats what he already has.

        Depth = the LOOP-ERASED length, which is genuine forward
        progress by construction: erasure removes every detour that
        returned to a board already stood on, so a longer erased path
        means he got further from the start without cycling.  Note this
        is the OPPOSITE comparison to `_seal_path`, where a cleared route
        is better for being SHORTER -- there, shorter means a cheaper way
        to a known exit; here, longer means a further frontier.
        """
        if not _DEPTHSEAL_ON():
            return
        try:
            k = self._path_key(lv)
            seq = ARCWorld._cur_path.get(k)
            if not seq:
                return
            er = ARCWorld._erase_loops(seq)
            # A life that erases to nothing reached no frontier at all --
            # measured as the median for cd82/sp80/vc33, where he walks
            # 275-400 steps and ends on the board he started from.
            if not er or len(er) < 2 or len(er) > 4000:
                return
            if len(er) <= int(ARCWorld._depth_len.get(k, 0)):
                return
            ARCWorld._depth_paths[k] = dict(er)
            ARCWorld._depth_len[k] = len(er)
            ARCWorld._depth_sealed += 1
        except Exception:
            pass

    def _drop_path(self) -> None:
        try:
            k = self._path_key()
            ARCWorld._cur_path.pop(k, None)
            ARCWorld._cur_seen.pop(k, None)
        except Exception:
            pass

    @staticmethod
    def _load_path_seed():
        """Restore recovered winning paths, once per process.

        Only fills keys he does not already hold, so a path he has
        sealed himself is never overwritten.  A board that no longer
        matches simply never fires.
        """
        if ARCWorld._seed_loaded or not _PATHSEED_ON():
            return
        ARCWorld._seed_loaded = True
        try:
            import json as _js
            with open("/root/knownpath_seed.json") as _fh:
                _d = _js.load(_fh)
        except Exception:
            return
        for _g, _lvs in (_d or {}).items():
            for _lv, _m in (_lvs or {}).items():
                try:
                    _k = (str(_g), int(_lv))
                except (TypeError, ValueError):
                    continue
                if _k in ARCWorld._paths:
                    continue
                _got = {}
                for _b, _v in (_m or {}).items():
                    try:
                        _got[bytes.fromhex(_b)] = (
                            int(_v[0]), _v[1], _v[2])
                    except (TypeError, ValueError, IndexError):
                        continue
                if _got:
                    ARCWorld._paths[_k] = _got
                    ARCWorld._seed_paths += 1

    @staticmethod
    def _load_depth_seed():
        """Restore recovered FRONTIER routes, once per process.

        `_seal_depth` only knows the lives it has watched, and it went
        live 2026-08-30 20:21Z.  The frame corpus holds ELEVEN DAYS of
        level-1 excursions that are far deeper than anything he has
        rebuilt since -- measured deepest loop-erased level-1 route per
        game: m0r0 109, cn04 74, r11l 61, cd82 53, ar25 49, sp80 43,
        ft09 31, and lp85 **level 2** at 65.  Without this he re-walks
        eleven days of work.

        Only fills keys he does not already hold, so a frontier he has
        beaten himself is never overwritten by a shallower recovered one.
        A frontier is weaker evidence than a clear and stays in its own
        store, consulted only when the cleared store is silent.
        """
        if ARCWorld._depth_seed_loaded or not _DEPTHSEAL_ON():
            return
        ARCWorld._depth_seed_loaded = True
        try:
            import json as _js
            with open("/root/depthseed.json") as _fh:
                _d = _js.load(_fh)
        except Exception:
            return
        for _g, _lvs in (_d or {}).items():
            for _lv, _blob in (_lvs or {}).items():
                try:
                    _k = (str(_g), int(_lv))
                except (TypeError, ValueError):
                    continue
                if _k in ARCWorld._depth_paths:
                    continue
                _got = {}
                for _b, _v in ((_blob or {}).get("route") or {}).items():
                    try:
                        _got[bytes.fromhex(_b)] = (
                            int(_v[0]), _v[1], _v[2])
                    except (TypeError, ValueError, IndexError):
                        continue
                if _got:
                    ARCWorld._depth_paths[_k] = _got
                    try:
                        ARCWorld._depth_len[_k] = int(
                            (_blob or {}).get("len") or len(_got))
                    except (TypeError, ValueError):
                        ARCWorld._depth_len[_k] = len(_got)
                    ARCWorld._depth_seeded += 1

    def known_action(self):
        """The action his own winning run took FROM THIS BOARD.

        None whenever the board is not on a stored path, which is the
        whole safety property: unknown board -> he plays normally, so
        this can only ever ADD an ordering, never remove an option.
        """
        if not _KNOWNPATH_ON():
            return None
        ARCWorld._load_path_seed()
        ARCWorld._load_depth_seed()
        try:
            _k = self._path_key()
            p = ARCWorld._paths.get(_k)
            _from_depth = False
            if not p:
                # No route to the exit from here.  Fall back to the route
                # to the furthest he has ever got -- weaker evidence, so
                # only ever consulted when the cleared store is silent.
                if _DEPTHSEAL_ON():
                    p = ARCWorld._depth_paths.get(_k)
                    _from_depth = True
                if not p:
                    return None
            h = self._board_hash()
            if h is None:
                return None
            rec = p.get(h)
            if rec is None:
                return None
            # HE HAS BEEN HERE ALREADY THIS ATTEMPT.  The stored path
            # cannot repeat a board, so standing on one twice means
            # the WORLD sent him back -- following again would be a
            # treadmill.  Fall through and let him play instead.
            if h in ARCWorld._cur_seen.get(self._path_key(), ()):
                ARCWorld._path_loops += 1
                return None
            if _from_depth:
                ARCWorld._depth_follow += 1
            else:
                ARCWorld._path_follow += 1
            # RELPLAN reads this: a frontier route yields to a plan, a
            # winning route does not
            self._known_from_depth = bool(_from_depth)
            return int(rec[0])
        except Exception:
            return None

    def known_aim(self, board, action):
        """The COORDINATE that worked, not just the index.  For a click
        the index does not determine the outcome -- the aim does."""
        if not _KNOWNPATH_ON() or board is None:
            return None
        try:
            _k = self._path_key()
            p = ARCWorld._paths.get(_k)
            if not p:
                return None
            rec = p.get(board)
            if (rec is None or int(rec[0]) != int(action)
                    or rec[1] is None or rec[2] is None):
                return None
            # HE HAS BEEN HERE ALREADY THIS LIFE.  `known_action` has
            # carried this guard since it was written; `known_aim` never
            # did, and nothing exposed it because no stored path had ever
            # covered a board he could get stuck on.
            # 2026-08-30: seeding his own recovered lp85 LEVEL 1 route did
            # expose it.  He stood on the first board of that route,
            # `known_action` correctly returned None (path_loops 2540 --
            # firing every single step), but the AIM was still pinned to
            # the stored (29,3).  Same click, board never changes, same
            # board next step: **2,541 steps, 100% no-ops, zero
            # terminals.**  A route that cannot repeat a board must not
            # aim at one twice either.
            if board in ARCWorld._cur_seen.get(_k, ()):
                ARCWorld._aim_loops += 1
                return None
            return (int(rec[1]), int(rec[2]))
        except Exception:
            return None

    def path_to_dict(self) -> dict:
        """Only THIS game, mirroring click_hits/act_effect."""
        out = {}
        g = str(self.game_id)
        for (gg, lv), p in ARCWorld._paths.items():
            if gg != g:
                continue
            out[str(lv)] = dict(
                (b.hex(), [int(v[0]),
                           (None if v[1] is None else int(v[1])),
                           (None if v[2] is None else int(v[2]))])
                for b, v in p.items())
        return {g: out} if out else {}

    def depth_to_dict(self) -> dict:
        """The FRONTIER routes, for THIS game, mirroring path_to_dict.

        Without this they die at every restart -- the same disease the
        explorer had, and the one this organ exists to cure: he re-derives
        the level from nothing.  Measured before it was caught: 79 seals
        across 30 paths, mean frontier 90.4 boards, all process-local.
        The sealed LENGTH goes with the route, because the comparison
        that admits a new frontier is "longer than the stored one" and a
        forgotten length would re-admit a shallower path.
        """
        out = {}
        g = str(self.game_id)
        for (gg, lv), p in ARCWorld._depth_paths.items():
            if gg != g:
                continue
            out[str(lv)] = {
                'len': int(ARCWorld._depth_len.get((gg, lv), len(p))),
                'route': dict(
                    (b.hex(), [int(v[0]),
                               (None if v[1] is None else int(v[1])),
                               (None if v[2] is None else int(v[2]))])
                    for b, v in p.items()),
            }
        return {g: out} if out else {}

    def depth_from_dict(self, d) -> int:
        try:
            rows = (d or {}).get(str(self.game_id)) or {}
        except Exception:
            return 0
        n = 0
        for lv, blob in rows.items():
            try:
                k = (str(self.game_id), int(lv))
                route = (blob or {}).get('route') or {}
                got = {}
                for b, v in route.items():
                    got[bytes.fromhex(b)] = (int(v[0]), v[1], v[2])
                if got:
                    ARCWorld._depth_paths[k] = got
                    ARCWorld._depth_len[k] = int(
                        (blob or {}).get('len') or len(got))
                    n += 1
            except (TypeError, ValueError, IndexError, KeyError,
                    AttributeError):
                continue
        return n

    def _relsense_step(self, prev_grid, prev_levels, success):
        """Feed the relation sense for this (game, level).

        A life begins on its first step (`_steps_episode` is 0 there,
        and `prev_grid` is the reset board) and on a level clear (the
        new level's first board).  A clear CONFIRMS, on the board before
        the winning move, which candidate was the goal."""
        from .relsense import RelSense, segment
        if self._grid is None:
            return
        # A LIFE ENDING IS NOT A STEP IN IT.  The GAME_OVER frame
        # rewrites most of the board (lp85 median 293 cells against 50
        # on an ordinary step), and `changed` is sticky, so judging it
        # would mark half the board a canvas.
        if not success and str(self._state) != "NOT_FINISHED":
            return
        g = str(self.game_id)
        lv = int(self._levels)
        _bl = self._bar_of()
        line = int(_bl) if _bl is not None and int(_bl) >= 0 else None
        _s = ARCWorld._self.get(g)
        sig = _s[0] if _s else None
        if success and lv > int(prev_levels):
            rs_prev = ARCWorld._rel.get((g, int(prev_levels)))
            if rs_prev is not None and prev_grid is not None:
                rid = rs_prev.confirm(prev_grid)
                if rid is not None:
                    ARCWorld._rel_confirms += 1
                    try:
                        import sys as _sys
                        _sys.stderr.write('[relsense] CONFIRMED game=%s lv=%d rel=%s\n'
                                          % (g, int(prev_levels), rs_prev._name(rid)))
                        _sys.stderr.flush()
                    except Exception:
                        pass
            rs = ARCWorld._rel.get((g, lv))
            if rs is None:
                rs = RelSense()
                ARCWorld._rel[(g, lv)] = rs
            bg, objs = segment(self._grid, line)
            self._rel_last = rs.observe(self._grid, bg, objs, sig, new_life=True, line=line)
            return
        rs = ARCWorld._rel.get((g, lv))
        if rs is None:
            rs = RelSense()
            ARCWorld._rel[(g, lv)] = rs
        if int(self._steps_episode) == 0 and prev_grid is not None:
            bg0, objs0 = segment(prev_grid, line)
            rs.observe(prev_grid, bg0, objs0, sig, new_life=True, line=line)
        bg, objs = segment(self._grid, line)
        self._rel_last = rs.observe(self._grid, bg, objs, sig, line=line)
        ARCWorld._rel_steps += 1
        if float(self._rel_last.get('progress') or 0.0) > 0.0:
            ARCWorld._rel_fires += 1

    def _rel_objs_for(self, grid, line):
        """Objects of a grid, cached for the two grids a step touches."""
        from .relsense import segment
        cache = getattr(self, '_rel_objs_cache', None) or []
        for g, ln, objs in cache:
            if g is grid and ln == line:
                return objs
        bg, objs = segment(grid, line)
        cache = [(grid, line, objs)] + cache[:1]
        self._rel_objs_cache = cache
        return objs

    def _rel_anyconf(self):
        """Relations confirmed by a clear on ANY level of this game."""
        out = set()
        try:
            g = str(self.game_id)
            for (gg, _lv), rs2 in ARCWorld._rel.items():
                if gg == g:
                    out |= set(rs2.confirmed)
        except Exception:
            pass
        return out

    @staticmethod
    def _rel_picture(rs, anyconf=None):
        """The picture the plan works on: the pursued relation if it is
        one; else a confirmed picture on this level; else the picture
        candidate with the most to do at life start.  None if there is
        no picture on the board.
        PICFIX (2026-09-08): a relation confirmed by a clear on ANY level
        of this game, with the most to do this life, comes first.  cd82
        L2: the 3x4 tile-region pair is confirmed on L2 (exact on the
        pre-win board) while the 10x10 canvas/target -- confirmed on L0
        and L1, and what the plan's table and the clear itself are made
        of -- was not; pursuing the small one handed the plan a picture
        its effects cannot index."""
        try:
            if _PICFIX_ON() and anyconf:
                best = None
                for rid in rs.v0:
                    if rid[0] != 'eq' or rid not in anyconf:
                        continue
                    key = float(rs.v0.get(rid, 0.0))
                    if best is None or key > best[0]:
                        best = (key, rid)
                if best is not None:
                    return best[1]
            p = rs._pursued
            if p is not None and p[0] == 'eq':
                return p
            best = None
            for rid in rs.v0:
                if rid[0] != 'eq':
                    continue
                key = (1 if rid in rs.confirmed else 0, float(rs.v0.get(rid, 0.0)))
                if best is None or key > best[0]:
                    best = (key, rid)
            return best[1] if best is not None else None
        except Exception:
            return None

    def _relplan_step(self, prev_grid, prev_levels, action, is_click, aim):
        """RELPLAN learns from this step, where a picture is pursued."""
        from .relplan import RelPlan
        if self._grid is None or prev_grid is None:
            return
        if str(self._state) != "NOT_FINISHED":
            return
        g = str(self.game_id)
        lv = int(self._levels)
        # A LIFE BEGINS whether or not a picture is pursued yet: the
        # stuck guard, the pending pick and the selected colour belong
        # to the life that reset the board
        if lv != int(prev_levels) or int(self._steps_episode) == 0:
            rp0 = ARCWorld._relplan.get((g, lv))
            if rp0 is not None:
                rp0.begin_life()
            if lv != int(prev_levels):
                return
        rs = ARCWorld._rel.get((g, lv))
        _pic = self._rel_picture(rs, _rel_anyconf_of(self)) if rs is not None else None
        if _pic is None:
            return
        A, B = _pic[1], _pic[2]
        rp = ARCWorld._relplan.get((g, lv))
        if rp is None:
            rp = RelPlan()
            ARCWorld._relplan[(g, lv)] = rp
        # SEED: his own frames, replayed offline (patch 9), MERGED once
        # per (game, level) per process -- into a table that already
        # exists as well as into a new one (patch 14, 2026-09-07).
        # MEASURED: `rel_from_dict` leaves a RelPlan behind for every
        # persisted blob -- EMPTY when from_dict dropped a v1 table at
        # the v2 format change, SMALL when it was build 6's own -- so
        # `if rp is None` never seeded cd82 L2 after that change: 14 h
        # into build 7 the live table held 20 of the seed's 83 effects
        # (19 shared, 1 its own) and 83 of its 326 automaton
        # transitions.  from_dict is a max/union merge (idempotent), so
        # nothing learned live is lost; the life-state fields the blob
        # carries (sel, last_painted) are restored afterwards, because
        # knowledge is merged, not the moment.
        if (g, lv) not in ARCWorld._relplan_seeded:
            ARCWorld._relplan_seeded.add((g, lv))
            try:
                import json as _json
                import sys as _sys
                _sp = ARCWorld._relplan_seed_path
                if _sp and _os.path.exists(_sp):
                    with open(_sp) as _fh:
                        _seed = _json.load(_fh)
                    _blob = ((_seed or {}).get(g) or {}).get(str(lv))
                    if _blob:
                        _n0, _a0 = len(rp.effects), len(rp.auto)
                        _sel, _lp = rp.sel, rp.last_painted
                        rp.from_dict(_blob)
                        rp.sel, rp.last_painted = _sel, _lp
                        ARCWorld._relplan_last = 'seeded %s lv%d effects=%d->%d' % (
                            g[:4], lv, _n0, len(rp.effects))
                        _sys.stderr.write('[relplan] SEEDED game=%s lv=%d effects=%d->%d auto=%d->%d\n'
                                          % (g, lv, _n0, len(rp.effects), _a0, len(rp.auto)))
            except Exception:
                ARCWorld._relplan_errors += 1
        _bl = self._bar_of()
        line = int(_bl) if _bl is not None and int(_bl) >= 0 else None
        objs_p = self._rel_objs_for(prev_grid, line)
        objs_c = self._rel_objs_for(self._grid, line)
        rp.observe(prev_grid, objs_p, self._grid, objs_c, int(action),
                   bool(is_click), aim, A, B,
                   place=ARCWorld._place_key(self, prev_grid), place_next=ARCWorld._place_key(self))
        try:
            _evs = getattr(rp, 'events', None)
            if _evs:
                import sys as _sys
                for _ev in _evs:
                    _sys.stderr.write('[relwalk] game=%s lv=%d %s\n' % (g, lv, _ev))
                del _evs[:]
                _sys.stderr.flush()
        except Exception:
            pass

    # ---- THE SEARCH (patch 45) -------------------------------------
    def _search_click_idx(self, n=None):
        """The index of the coordinate action, or None."""
        try:
            lim = int(self._n_actions) if n is None else min(int(n), int(self._n_actions))
            for i in range(lim):
                if i < len(self._acts) and int(self._acts[i]) == 6:
                    return i
        except Exception:
            pass
        return None

    def _search_state(self, grid):
        from .goal_search import state_of
        if grid is None:
            return None
        return state_of(grid, ARCWorld._bar_lines(self))

    def _search_controls(self, grid, n=None):
        from .goal_search import controls_of
        _bl = self._bar_of()
        line = int(_bl) if _bl is not None and int(_bl) >= 0 else None
        objs = self._rel_objs_for(grid, line)
        lines = set(int(x) for x in ARCWorld._bar_lines(self))
        if lines:
            # a draining bar on a second clock line is not a button (review 2, 5)
            objs = [o for o in objs
                    if not ((int(o[2]) == int(o[4]) and int(o[2]) in lines)
                            or (int(o[3]) == int(o[5]) and 64 + int(o[3]) in lines))]
        arrows = self._hyp_arrows()
        if n is not None:
            arrows = [a for a in arrows if a < int(n)]
        return controls_of(objs, arrows, self._search_click_idx(n))

    # ------------------------------------------------ the goal library (patch 48)
    @staticmethod
    def _lib_get():
        if ARCWorld._lib is None:
            from .goal_library import Library
            lib = Library()
            try:
                import json as _json
                if _os.path.exists('/root/goallib/seed.json'):
                    with open('/root/goallib/seed.json') as _fh:
                        lib.from_dict((_json.load(_fh) or {}).get('library') or {})
            except Exception:
                ARCWorld._lib_errors += 1
            ARCWorld._lib = lib
        return ARCWorld._lib

    def _lib_line(self):
        _bl = self._bar_of()
        return int(_bl) if _bl is not None and int(_bl) >= 0 else None

    def _lib_self(self, lv):
        rs = ARCWorld._rel.get((str(self.game_id), int(lv)))
        if rs is None:
            return None, None
        return rs.self_box, (tuple(int(x) for x in rs.ctrl) if rs.ctrl else None)

    def _lib_life_start(self, lv, grid):
        """The board a life began on: kept, and read once for the candidates."""
        from .goal_library import read
        g = str(self.game_id)
        try:
            B = [[int(v) for v in row] for row in grid]
            sb, cls = self._lib_self(lv)
            ARCWorld._lib_start[(g, int(lv))] = B
            ARCWorld._lib_sb0[(g, int(lv))] = sb
            ARCWorld._lib_insts[(g, int(lv))] = read(B, self._lib_line(), sb, cls)[2]
        except Exception:
            ARCWorld._lib_errors += 1

    def _lib_life_end(self, lv, end_grid, cleared):
        """A life ended on `end_grid` (the board before the winning move, or
        the last ordinary board): the library observes it against its start."""
        g = str(self.game_id)
        start = ARCWorld._lib_start.get((g, int(lv)))
        if start is None or end_grid is None:
            return
        try:
            lib = ARCWorld._lib_get()
            sb1, cls = self._lib_self(lv)
            lib.observe_life(g, int(lv), start, end_grid, bool(cleared), self._lib_line(),
                             ARCWorld._lib_sb0.get((g, int(lv))), sb1, cls)
            self._lib_flush(lib)
            sr = ARCWorld._srch.get((g, int(lv)))
            _vals = [v for s_, v in sr.val.items() if s_ != sr.root] if (sr is not None and sr.pursued is not None) else []
            if not cleared and _vals and min(_vals) >= 10 ** 9:
                # the pursued relation was unreadable on every state of this
                # life (its things are gone): skip it for the level (review 4)
                sr.lib_refuted.add(sr.pursued[0])
                sr.clear_goal()
        except Exception:
            ARCWorld._lib_errors += 1

    @staticmethod
    def _lib_flush(lib):
        try:
            if lib.events:
                import sys as _sys
                for _ev in lib.events:
                    _sys.stderr.write('[library] %s\n' % _ev)
                del lib.events[:]
                _sys.stderr.flush()
        except Exception:
            pass

    def _lib_goal(self, sr, lv, grid, state):
        """Pursue the library's best unrefuted candidate on this level and
        value this board for the search's frontier choice."""
        from .goal_library import evaluate, rid_key, schema_key
        g = str(self.game_id)
        insts = ARCWorld._lib_insts.get((g, int(lv)))
        if not insts:
            return
        lib = ARCWorld._lib_get()
        keys = {}
        for i in insts.values():
            keys[rid_key(i.rid)] = i
        if sr.pursued is None or sr.pursued[0] not in keys:
            cands = [i for i in insts.values() if rid_key(i.rid) not in sr.lib_refuted and i.value > 0]
            cands.sort(key=lambda i: (-lib.prior(i.schema), -abs(i.value)))
            if not cands:
                sr.clear_goal()
                return
            c = cands[0]
            sr.set_goal(rid_key(c.rid), schema_key(c.schema), c.value)
            ARCWorld._lib_pursues += 1
            try:
                import sys as _sys
                _sys.stderr.write('[library] PURSUE game=%s lv=%d rid=%s schema=%s prior=%.2f from=%s v0=%s cands=%d refuted=%d\n'
                                  % (g, int(lv), rid_key(c.rid), schema_key(c.schema), lib.prior(c.schema),
                                     ','.join(sorted(lib.games.get(schema_key(c.schema), ()))) or '-',
                                     c.value, len(cands), len(sr.lib_refuted)))
                _sys.stderr.flush()
            except Exception:
                pass
        if state in sr.val:
            return
        if state == sr.root or int(getattr(self, '_steps_episode', 0) or 0) == 0:
            # the life-start board: its value is v0 by construction (relsense's
            # self box there is still the previous life's; review)
            sr.note_value(state, sr.pursued[2])
            return
        inst = keys.get(sr.pursued[0])
        if inst is None:
            return
        sb, cls = self._lib_self(lv)
        v = evaluate(grid, {inst.rid: inst}, self._lib_line(), sb, cls).get(inst.rid)
        sr.note_value(state, 10 ** 9 if v is None else v)

    def _lib_refute(self, sr, lv):
        g = str(self.game_id)
        try:
            lib = ARCWorld._lib_get()
            rk, sch, v0 = sr.pursued
            lib.refute_key(sch, g)
            sr.lib_refuted.add(rk)
            ARCWorld._lib_refutes += 1
            import sys as _sys
            _sys.stderr.write('[library] REFUTED game=%s lv=%d rid=%s schema=%s tried_games=%d prior=%.2f refuted=%d\n'
                              % (g, int(lv), rk, sch, len(lib.tried_games.get(sch, ())),
                                 lib.prior(tuple(sch.split('|'))), len(sr.lib_refuted)))
            _sys.stderr.flush()
        except Exception:
            ARCWorld._lib_errors += 1
        sr.clear_goal()

    def _search_step(self, prev_grid, prev_levels, action, sent, success):
        """Learn this EXECUTED step: (state before) -control-> (state after
        | WIN | OVER).  A life begins as the hypothesis organ's does."""
        from .goal_search import Search
        if not (_SEARCH_ON() and _RELSENSE_ON()):
            return
        if self._grid is None or prev_grid is None:
            return
        g = str(self.game_id)
        lv = int(self._levels)
        plv = int(prev_levels)
        sr = ARCWorld._srch.get((g, plv))
        if sr is None:
            sr = Search()
            ARCWorld._srch[(g, plv)] = sr
        s0 = self._search_state(prev_grid)
        _lid = self._hyp_life_id()
        if ARCWorld._srch_life.get(g) != _lid or getattr(self, '_srch_new_life', False):
            ARCWorld._srch_life[g] = _lid
            _nv = bool(getattr(self, '_srch_new_life', False))
            self._srch_new_life = False
            sr.begin_life(root=s0, visit=_nv)
            if _LIBRARY_ON():
                self._lib_life_start(plv, prev_grid)
        cleared = bool(success) and lv > plv
        if cleared:
            sr2 = ARCWorld._srch.get((g, lv))
            if sr2 is None:
                sr2 = Search()
                ARCWorld._srch[(g, lv)] = sr2
            sr2.begin_life(root=self._search_state(self._grid), visit=True)  # patch 47: a level entered by a clear is a new visit of it
            if _LIBRARY_ON():
                self._lib_life_start(lv, self._grid)
        over = (not cleared) and str(self._state) != "NOT_FINISHED"
        is_click = bool(sent and sent[3])
        if is_click:
            if not sent or sent[1] is None or sent[2] is None:
                return
            ctl = ('C', int(sent[1]), int(sent[2]))
        else:
            ctl = ('A', int(action))
        ctls = self._search_controls(prev_grid)
        sr.see(s0, ctls)
        s1 = None if (cleared or over) else self._search_state(self._grid)
        _was_at_goal = bool(_LIBRARY_ON() and sr.pursued is not None and not cleared and sr.at_goal(s0))
        _clock = bool(over and sr.max_life and sr.steps_this_life + 1 >= sr.max_life)
        sr.observe(s0, ctl, s1, cleared, over, listed=(ctl in ctls))
        if _was_at_goal and not _clock and not sr.is_frontier(s0) and not sr.has_win_from(s0):
            # he stood where the pursued relation is at its goal, tried EVERY
            # control there, and the level never ended: not the goal (patch 48;
            # review: refuting on the first step there refuted the true goal)
            self._lib_refute(sr, plv)
        if _LIBRARY_ON() and (cleared or over):
            self._lib_life_end(plv, prev_grid, cleared)
        try:
            if sr.events:
                import sys as _sys
                for _ev in sr.events:
                    _sys.stderr.write('[search] %s game=%s lv=%d states=%d\n'
                                      % (_ev, g, plv, len(sr.trans)))
                    ARCWorld._srch_last = '%s %s lv%d' % (_ev.split(' ')[0], g[:4], plv)
                del sr.events[:]
                _sys.stderr.flush()
        except Exception:
            pass

    def search_action(self, n):
        """The control the search proposes from here as an action index, or
        None.  A click's aim is kept in `_search_pick` for step()."""
        self._search_win_path = False
        self._search_pick = None
        if not (_SEARCH_ON() and _RELSENSE_ON()):
            return None
        try:
            g = str(self.game_id)
            lv = int(self._levels)
            sr = ARCWorld._srch.get((g, lv))
            if sr is not None:
                sr.last_pick = None
            if self._grid is None or sr is None:
                return None
            ARCWorld._srch_asks += 1
            s = self._search_state(self._grid)
            ctls = self._search_controls(self._grid, n)
            if _LIBRARY_ON():
                try:
                    self._lib_goal(sr, lv, self._grid, s)
                except Exception:
                    ARCWorld._lib_errors += 1
            c = sr.act(s, ctls)
            if c is None:
                return None
            if c[0] == 'A':
                a = int(c[1])
            else:
                a = self._search_click_idx(n)
                if a is None:
                    return None
                self._search_pick = (int(a), (int(c[1]), int(c[2])))
            if not (0 <= a < int(n)):
                return None
            ARCWorld._srch_acts += 1
            self._search_win_path = bool(sr.win_path())
            return a
        except Exception:
            ARCWorld._srch_errors += 1
            return None

    def search_hold(self):
        """The search is still finding new states here: hold him through
        the terminal.  False on any doubt or with the gate off."""
        if not (_SEARCH_ON() and _RELSENSE_ON()):
            return False
        try:
            g = str(self.game_id)
            lv = int(self._levels)
            sr = ARCWorld._srch.get((g, lv))
            if sr is None:
                return False
            held = bool(sr.testing())
            try:
                import sys as _sys
                if sr.events:
                    # patch 47: a WALK release is written HERE, at the terminal that decides it
                    for _ev in sr.events:
                        _sys.stderr.write('[search] %s game=%s lv=%d states=%d\n'
                                          % (_ev, g, lv, len(sr.trans)))
                        ARCWorld._srch_last = '%s %s lv%d' % (_ev.split(' ')[0], g[:4], lv)
                    del sr.events[:]
                if held:
                    _sys.stderr.write('[search] HOLD game=%s lv=%d states=%d new=%d lives=%d untried=%d run=%d\n'
                                      % (g, lv, len(sr.trans), sr.new_this_life, sr.lives,
                                         int(getattr(sr, 'untried_this_life', 0)), int(getattr(sr, 'walk_run', 0))))
                _sys.stderr.flush()
            except Exception:
                pass
            if not held:
                return False
            ARCWorld._srch_holds += 1
            return True
        except Exception:
            ARCWorld._srch_errors += 1
            return False

    # ---- THE HYPOTHESIS (patch 44) ---------------------------------
    def _hyp_arrows(self):
        """Action indices that are not the coordinate action."""
        try:
            return [i for i in range(int(self._n_actions))
                    if i < len(self._acts) and int(self._acts[i]) != 6]
        except Exception:
            return []

    def _hyp_key(self, grid, rs):
        """The automaton key for the object he controls on `grid`."""
        from .goal_hypothesis import self_box, box_key
        if grid is None or rs is None or not rs.ctrl:
            return None
        _bl = self._bar_of()
        line = int(_bl) if _bl is not None and int(_bl) >= 0 else None
        objs = self._rel_objs_for(grid, line)
        return box_key(self_box(objs, rs.ctrl))

    def _hyp_life_id(self):
        return (str(self.game_id), int(getattr(self, '_lives_here', 0)))

    def _hyp_step(self, prev_grid, prev_levels, action, is_click, success):
        """Learn this EXECUTED step: (key before) -action-> (key after | WIN).
        A life begins on the first step after a terminal reset (`_lives_here`
        moved), after a re-entry (`enter()` reset the board), and on the new
        level after a clear."""
        from .goal_hypothesis import Hypothesis
        if not (_HYPOTHESIS_ON() and _RELSENSE_ON()):
            return
        if self._grid is None or prev_grid is None:
            return
        g = str(self.game_id)
        lv = int(self._levels)
        plv = int(prev_levels)
        hy = ARCWorld._hyp.get((g, plv))
        if hy is None:
            hy = Hypothesis()
            ARCWorld._hyp[(g, plv)] = hy
        _lid = self._hyp_life_id()
        if ARCWorld._hyp_life.get(g) != _lid or getattr(self, '_hyp_new_life', False):
            ARCWorld._hyp_life[g] = _lid
            self._hyp_new_life = False
            hy.begin_life()
        rs = ARCWorld._rel.get((g, plv))
        cleared = bool(success) and lv > plv
        if cleared:
            hy2 = ARCWorld._hyp.get((g, lv))
            if hy2 is None:
                hy2 = Hypothesis()
                ARCWorld._hyp[(g, lv)] = hy2
            hy2.begin_life()
        if not cleared and str(self._state) != "NOT_FINISHED":
            return
        if rs is None or not rs.ctrl:
            return
        hy.set_ctrl(rs.ctrl)
        if is_click:
            hy.last_pick = None         # a click is never his; nothing to attribute later
            return
        k0 = self._hyp_key(prev_grid, rs)
        k1 = None if cleared else self._hyp_key(self._grid, rs)
        hy.observe(k0, int(action), k1, cleared, by_organ=hy.proposed(k0, int(action)))
        try:
            if hy.events:
                import sys as _sys
                for _ev in hy.events:
                    _sys.stderr.write('[hypothesis] %s game=%s lv=%d keys=%d\n'
                                      % (_ev, g, plv, len(hy.auto)))
                    ARCWorld._hyp_last = '%s %s lv%d' % (_ev.split(' ')[0], g[:4], plv)
                del hy.events[:]
                _sys.stderr.flush()
        except Exception:
            pass

    def hyp_action(self, n):
        """The arrow the hypothesis organ proposes from here, or None."""
        self._hyp_win_path = False
        if not (_HYPOTHESIS_ON() and _RELSENSE_ON()):
            return None
        try:
            g = str(self.game_id)
            lv = int(self._levels)
            rs = ARCWorld._rel.get((g, lv))
            hy = ARCWorld._hyp.get((g, lv))
            if hy is not None:
                hy.last_pick = None         # every ask starts clean; a stale pick confirms nothing
            if rs is None or not rs.ctrl or self._grid is None or hy is None:
                return None
            ARCWorld._hyp_asks += 1
            hy.set_ctrl(rs.ctrl)
            key = self._hyp_key(self._grid, rs)
            if key is None:
                return None
            _bl = self._bar_of()
            line = int(_bl) if _bl is not None and int(_bl) >= 0 else None
            objs = self._rel_objs_for(self._grid, line)
            # relsense's `body` is what touches him THIS frame, so a marker
            # he has just reached would vanish from the candidates instead
            # of being refuted; only his own class and mobile classes are
            # excluded here (his ring is refuted after one step, harmlessly)
            cands = hy.candidates(objs, rs.ctrl, rs.mobile, line)
            n_all = len(hy.candidates(objs, rs.ctrl, rs.mobile, line, skip_refuted=False))
            arrows = [a for a in self._hyp_arrows() if a < int(n)]
            a = hy.act(key, cands, arrows, n_all=n_all)
            if a is not None:
                ARCWorld._hyp_acts += 1
                # a known win outranks a sealed route ONLY where this organ
                # has cleared the level itself (its own confirmed candidate):
                # a position-only map cannot carry a door, a switch or a hue,
                # and the route that won carries them (review 2, finding 2)
                self._hyp_win_path = bool(hy.win_path() and hy.confirmed is not None)
            try:
                if hy.events:
                    import sys as _sys
                    for _ev in hy.events:
                        _sys.stderr.write('[hypothesis] %s game=%s lv=%d keys=%d\n'
                                          % (_ev, g, lv, len(hy.auto)))
                        ARCWorld._hyp_last = '%s %s lv%d' % (_ev.split(' ')[0], g[:4], lv)
                    del hy.events[:]
                    _sys.stderr.flush()
            except Exception:
                pass
            return a
        except Exception:
            ARCWorld._hyp_errors += 1
            return None

    def hyp_hold(self):
        """A hypothesis is under test here: hold him through the terminal.
        False on any doubt or with the gate off."""
        if not (_HYPOTHESIS_ON() and _RELSENSE_ON()):
            return False
        try:
            g = str(self.game_id)
            lv = int(self._levels)
            hy = ARCWorld._hyp.get((g, lv))
            if hy is None:
                return False
            if not hy.testing(self._hyp_arrows()):
                return False
            ARCWorld._hyp_holds += 1
            try:
                import sys as _sys
                _sys.stderr.write('[hypothesis] HOLD game=%s lv=%d keys=%d det=%.3f lives=%d life_gap=%s best=%s refuted=%d\n'
                                  % (g, lv, len(hy.auto), hy.determinism(), hy.lives,
                                     hy.life_gap, hy.best_gap, len(hy.refuted)))
                _sys.stderr.flush()
            except Exception:
                pass
            return True
        except Exception:
            ARCWorld._hyp_errors += 1
            return False

    def rel_plan_action(self, n):
        """The first action of the best paint plan for the pursued
        picture, or None.  The aim (for a click) is kept for step()."""
        self._plan_pick = None
        self._plan_probe = False
        self._plan_shortcut = False
        if not (_RELPLAN_ON() and _RELSENSE_ON()):
            return None

        def _why(w):
            ARCWorld._relplan_why[w] = ARCWorld._relplan_why.get(w, 0) + 1
            return None
        try:
            g = str(self.game_id)
            lv = int(self._levels)
            rs = ARCWorld._rel.get((g, lv))
            _pic = self._rel_picture(rs, _rel_anyconf_of(self)) if rs is not None else None
            if _pic is None:
                return _why("no_picture")
            rp = ARCWorld._relplan.get((g, lv))
            if rp is None or self._grid is None:
                return _why("no_table")
            ARCWorld._relplan_asks += 1
            _bl = self._bar_of()
            line = int(_bl) if _bl is not None and int(_bl) >= 0 else None
            objs = self._rel_objs_for(self._grid, line)
            A, B = _pic[1], _pic[2]
            # the same two regions confirmed by a clear on ANY level of
            # this game are the goal here too (cd82: confirmed on 0 and
            # 1, never yet on 2)
            _conf = _pic in rs.confirmed
            if not _conf:
                for (gg, lv2), rs2 in ARCWorld._rel.items():
                    if gg == g and _pic in rs2.confirmed:
                        _conf = True
                        break
            # THE FLOOR (patch 25): the best this level has reached, in any
            # life or this one -- the curiosity walk explores only there
            _floor = None
            try:
                if ARCWorld._paths.get((g, lv)):
                    # a WON ROUTE here (25b): the level has been cleared,
                    # nothing is explored where he can win (the recorded
                    # floor is the pre-win mismatch: L0 25, L1 10, L2 12)
                    _floor = 0
                else:
                    _fc = [int(v) for v in (ARCWorld._lvl_best_mm.get((g, lv)),
                                            getattr(self, '_life_min_mm', None))
                           if v is not None]
                    _floor = min(_fc) if _fc else None
            except Exception:
                _floor = None
            _rl = None
            if _SHORTCUT_ON() and _floor == 0 and getattr(rp, '_sc_try', False):
                # SHORTCUT (patch 40): what the WON route still needs from this
                # board -- its stored boards are in the order he walked them
                try:
                    _route = ARCWorld._paths.get((g, lv)) or {}
                    _h = self._board_hash()
                    if _h is not None and _h in _route:
                        _keys = list(_route)
                        _rl = (len(_keys) - _keys.index(_h), len(_keys))
                except Exception:
                    _rl = None
            r = rp.plan(self._grid, objs, A, B, int(n),
                        board_key=self._board_hash_nb(),
                        confirmed=bool(_conf), floor=_floor, route_left=_rl,
                        place=ARCWorld._place_key(self))
            try:
                _evs = getattr(rp, 'events', None)
                if _evs:
                    import sys as _sys
                    for _ev in _evs:
                        _sys.stderr.write('[relwalk] game=%s lv=%d %s\n' % (g, lv, _ev))
                    del _evs[:]
                    _sys.stderr.flush()
            except Exception:
                pass
            if r is None:
                return _why("plan:" + str(getattr(rp, 'why', '') or '?'))
            a, aim, info = r
            self._plan_probe = bool((info or {}).get("probe"))   # CURIOUS: the actor lets a route win
            self._plan_shortcut = bool((info or {}).get("shortcut") is not None and _SHORTCUT_ON())   # SHORTCUT (40)
            # WHAT THE BOARD ALREADY KNOWS DOES NOTHING HERE is not
            # taken, whatever the model predicts (a stale union)
            if aim is None:
                _bi = getattr(self, 'board_inert_here', None)
                try:
                    if _bi is not None and _bi(int(a)):
                        ARCWorld._relplan_inert += 1
                        rp._pending = None
                        return _why("inert")
                except Exception:
                    pass
            self._plan_pick = (int(a), aim)
            ARCWorld._relplan_acts += 1
            ARCWorld._relplan_by_game[g[:4]] = ARCWorld._relplan_by_game.get(g[:4], 0) + 1
            ARCWorld._relplan_last = '%s lv%d rounds=%d steps=%d mm=%d->%d state=%d pred=%d' % (
                g[:4], lv, info['rounds'], info['steps'], info['mm0'], info['mm1'],
                info.get('state', 0), info.get('pred', 0))
            if info['mm1'] == 0 and not rp._logged:
                rp._logged = True
                try:
                    import sys as _sys
                    _sys.stderr.write('[relplan] PLAN game=%s lv=%d rounds=%d steps=%d mm=%d->0\n'
                                      % (g, lv, info['rounds'], info['steps'], info['mm0']))
                    _sys.stderr.flush()
                except Exception:
                    pass
            return int(a)
        except Exception:
            ARCWorld._relplan_errors += 1
            self._plan_pick = None
            return _why("exception")

    def rel_context(self):
        """For the actor's steer: the situation he is in with respect to
        the pursued relation -- ((colour, size), dr, dc) toward the
        nearest marker instance, or ('eq',) for a picture -- or None."""
        if not _RELSENSE_ON():
            return None
        try:
            rs = ARCWorld._rel.get((str(self.game_id), int(self._levels)))
            if rs is None or rs._pursued is None:
                return None
            if rs._pursued[0] == 'eq':
                return ('eq',)
            return rs.context(self._grid)
        except Exception:
            return None

    def rel_to_dict(self) -> dict:
        """What the relation sense has learned for THIS game, per level:
        static and dynamic regions, confirmed relations.  Mirrors
        depth_to_dict; the process that loads this writes it, the next
        restart reads it."""
        if not _RELSENSE_ON():
            return {}
        g = str(self.game_id)
        out = {}
        for (gg, lv), rs in ARCWorld._rel.items():
            if gg != g:
                continue
            try:
                d = rs.to_dict()
            except Exception:
                continue
            # persisted WHETHER OR NOT the gate is on: rm RELPLAN_ON stops
            # the acting, it must not erase what he learned
            rp = ARCWorld._relplan.get((gg, lv))
            if rp is not None and rp.effects:
                try:
                    d['plan'] = rp.to_dict()
                except Exception:
                    pass
            # the hypothesis map rides here too, gate on or off
            hy = ARCWorld._hyp.get((gg, lv))
            if hy is not None and hy.auto:
                try:
                    d['hyp'] = hy.to_dict()
                except Exception:
                    pass
            # the search map rides here too, gate on or off
            sr = ARCWorld._srch.get((gg, lv))
            if sr is not None and sr.trans:
                try:
                    d['srch'] = sr.to_dict()
                except Exception:
                    pass
            if d.get('static') or d.get('dynamic') or d.get('confirmed') or d.get('plan') or d.get('hyp') or d.get('srch'):
                out[str(lv)] = d
        # search maps on levels the relation sense has no table for yet
        for (gg, lv), sr in ARCWorld._srch.items():
            if gg != g or str(lv) in out or not sr.trans:
                continue
            try:
                out[str(lv)] = {'srch': sr.to_dict()}
            except Exception:
                pass
        res = {g: out} if out else {}
        # the goal library rides here, ONE across games, gate on or off (patch 48)
        if ARCWorld._lib is not None:
            try:
                res['_goallib'] = ARCWorld._lib.to_dict()
            except Exception:
                pass
        return res

    def rel_from_dict(self, d) -> int:
        if not _RELSENSE_ON():
            return 0
        try:
            rows = (d or {}).get(str(self.game_id)) or {}
            items = list(rows.items())
        except Exception:
            return 0
        from .relsense import RelSense
        try:
            if isinstance(d, dict) and d.get('_goallib'):
                ARCWorld._lib_get().from_dict(d.get('_goallib'), replace=True)
        except Exception:
            ARCWorld._lib_errors += 1
        n = 0
        for lv, blob in items:
            try:
                k = (str(self.game_id), int(lv))
                rs = ARCWorld._rel.get(k)
                if rs is None:
                    rs = RelSense()
                    ARCWorld._rel[k] = rs
                n += int(rs.from_dict(blob))
                if isinstance(blob, dict) and blob.get('srch'):
                    try:
                        from .goal_search import Search
                        sr = ARCWorld._srch.get(k)
                        if sr is None:
                            sr = Search()
                            ARCWorld._srch[k] = sr
                        n += int(sr.from_dict(blob.get('srch')))
                    except Exception:
                        ARCWorld._srch_errors += 1
                if isinstance(blob, dict) and blob.get('hyp'):
                    try:
                        from .goal_hypothesis import Hypothesis
                        hy = ARCWorld._hyp.get(k)
                        if hy is None:
                            hy = Hypothesis()
                            ARCWorld._hyp[k] = hy
                        n += int(hy.from_dict(blob.get('hyp')))
                    except Exception:
                        ARCWorld._hyp_errors += 1
                if isinstance(blob, dict) and blob.get('plan'):
                    try:
                        from .relplan import RelPlan
                        rp = ARCWorld._relplan.get(k)
                        if rp is None:
                            rp = RelPlan()
                            ARCWorld._relplan[k] = rp
                        n += int(rp.from_dict(blob.get('plan')))
                    except Exception:
                        ARCWorld._relplan_errors += 1
            except (TypeError, ValueError, AttributeError):
                continue
        return n

    def bar_to_dict(self) -> dict:
        """The DECIDED budget lines for THIS game: {game: {lv: line}}.

        Mirrors `depth_to_dict`.  A "no line" decision (-1) is NOT
        written: a no-bar cell re-learns in ~400 transitions and loses
        nothing, while a wrongly persisted -1 would blind a cell for
        good.  Gate off -> {}: the key is written EMPTY at the next
        save, so `rm /root/BARPERSIST_ON` ERASES the persisted lines
        rather than freezing them (they re-learn, ~400 transitions per
        cell).
        """
        if not _BARPERSIST_ON():
            return {}
        out = {}
        g = str(self.game_id)
        for (gg, lv), r in ARCWorld._bar_row.items():
            if gg != g:
                continue
            if isinstance(r, (tuple, list)):
                # a CLOCKLINE structure is written as a list of lines
                try:
                    r = [int(x) for x in r]
                except (TypeError, ValueError):
                    continue
                if len(r) > 1 and all(0 <= x < _BAR_LINES for x in r):
                    out[str(lv)] = r
                continue
            try:
                r = int(r)
            except (TypeError, ValueError):
                continue
            if r >= 0:
                out[str(lv)] = r
        return {g: out} if out else {}

    def bar_from_dict(self, d) -> int:
        """Restore decided lines for THIS game; never overwrite one
        decided in this process; ignore anything malformed."""
        if not _BARPERSIST_ON():
            return 0
        try:
            rows = (d or {}).get(str(self.game_id)) or {}
            items = list(rows.items())
        except Exception:
            return 0
        n = 0
        for lv, r in items:
            try:
                k = (str(self.game_id), int(lv))
                if isinstance(r, (tuple, list)):
                    # a structure is read back only under CLOCKLINE;
                    # gate off, the cell re-learns as a single line
                    if not _CLOCKLINE_ON():
                        continue
                    r = tuple(sorted(int(x) for x in r))
                    if len(r) < 2 or not all(0 <= x < _BAR_LINES
                                             for x in r):
                        continue
                else:
                    r = int(r)
                    if not (0 <= r < _BAR_LINES):
                        continue
                if k in ARCWorld._bar_row:
                    continue
                ARCWorld._bar_row[k] = r
                # a loaded line IS a decided line; the counter that
                # every watcher prints must say so
                ARCWorld._bar_decided += 1
                n += 1
            except (TypeError, ValueError):
                continue
        return n

    def path_from_dict(self, d) -> int:
        try:
            rows = (d or {}).get(str(self.game_id)) or {}
        except Exception:
            return 0
        n = 0
        for lv, p in rows.items():
            try:
                k = (str(self.game_id), int(lv))
                got = {}
                for b, v in (p or {}).items():
                    got[bytes.fromhex(b)] = (int(v[0]), v[1], v[2])
                if got:
                    ARCWorld._paths[k] = got
                    n += 1
            except (TypeError, ValueError, IndexError, KeyError):
                continue
        return n

    def _place_key(self, grid=None):
        """PLACEKEY (patch 41): WHERE HE STANDS, blind to paint -- the board with the
        canvas, the target and the budget row blanked to the background, hashed.  His
        quotient state moves when paint lands under the stamp, which is why a latched
        sequence oscillated between two states until it died stuck (offline cd82)."""
        if not _PLACEKEY_ON():
            return None
        try:
            import hashlib
            import numpy as _np
            g = self._grid if grid is None else grid
            if g is None:
                return None
            G = _np.asarray(g, dtype=_np.int16)
            if G.shape != (64, 64):
                return None
            rs = ARCWorld._rel.get((str(self.game_id), int(self._levels)))
            _pic = self._rel_picture(rs, _rel_anyconf_of(self)) if rs is not None else None
            if _pic is None:
                return None
            H = G.copy()
            v, c = _np.unique(H, return_counts=True)
            bg = int(v[int(_np.argmax(c))])
            for R in (_pic[1], _pic[2]):
                H[R[0]:R[2] + 1, R[1]:R[3] + 1] = bg
            _bl = self._bar_of()
            if _bl is not None and 0 <= int(_bl) < 64:
                H[int(_bl), :] = bg
            elif _bl is not None and 64 <= int(_bl) < 128:
                H[:, int(_bl) - 64] = bg
            # COLOUR-BLIND (measured, 3 walks x 2 levels): the coloured key split each stamp
            # slot by the selected colour into 53-56 places whose layers only a CLICK joins,
            # so a route place was unreachable by arrows from 21-30% of boards and the
            # shortcut died stuck.  Background / not-background gives the 8 slots and 100%
            # reach, modal accuracy unchanged (99.5-99.6%).  Paints stay on the quotient.
            return hashlib.blake2b((H != bg).astype(_np.uint8).tobytes(), digest_size=8).hexdigest()
        except Exception:
            return None

    def _board_hash(self):
        """Cheap identity of the WHOLE board.  His glance token recurs
        13% of the time; the board recurs 65%.  The recurrence law says
        use the index that recurs."""
        try:
            import hashlib
            g = self._grid
            if g is None:
                return None
            b = bytes(bytearray(v & 0xFF for r in g for v in r))
            return hashlib.blake2b(b, digest_size=8).digest()
        except Exception:
            return None

    def _board_key(self):
        """The board as the WITHIN-LIFE organs should see it.

        BOARDKEY on -> the budget line is masked (falls through to the
        full hash where no line is learned); off -> the full hash, as
        it always was.  `known_aim`, `_note_path_step` and the verdict
        key are NOT on this -- the seeds hold full-board hashes.
        """
        if _BOARDKEY_ON():
            return self._board_hash_nb()
        return self._board_hash()

    def _note_board_move(self, prev_h, action) -> None:
        """Record whether that move landed him somewhere he had been."""
        if not _CYCLEDEMOTE_ON() or prev_h is None:
            return
        try:
            g = str(self.game_id)
            h = self._board_key()
            if h is None:
                return
            seen = ARCWorld._board_seen.setdefault(g, set())
            k = (g, prev_h, int(action))
            e = ARCWorld._board_act.get(k)
            if e is None:
                e = [0, 0]
                ARCWorld._board_act[k] = e
            e[0] += 1
            if h in seen:
                e[1] += 1
            seen.add(h)
            # DID THAT MOVE CHANGE ANYTHING AT ALL?  Same board before
            # and after means a wasted action -- and the budget is the
            # wall (deaths cluster at a hard per-level ceiling).
            _ne = ARCWorld._board_noop.get(k)
            if _ne is None:
                _ne = [0, 0]
                ARCWorld._board_noop[k] = _ne
            _ne[0] += 1
            if h != prev_h:
                _ne[1] += 1
            if len(ARCWorld._board_noop) > 400000:
                ARCWorld._board_noop.clear()
            if len(ARCWorld._board_act) > 400000:
                ARCWorld._board_act.clear()
                ARCWorld._board_seen.clear()
        except Exception:
            pass

    def _explorer(self):
        """One Explorer per game, built lazily."""
        if not _EXPLORER_ON():
            return None
        try:
            g = str(self.game_id)
            e = ARCWorld._explorers.get(g)
            if e is None:
                from seagi.world.explorer import Explorer
                _saved = (ARCWorld._explorer_saved().get(g)
                          if _EXPLORERPERSIST_ON() else None)
                e = (Explorer.from_dict(_saved) if _saved else Explorer())
                if _saved:
                    ARCWorld._explorer_loaded = getattr(
                        ARCWorld, "_explorer_loaded", 0) + 1
                ARCWorld._explorers[g] = e
            return e
        except Exception:
            return None

    @classmethod
    def _explorer_saved(cls):
        """Persisted explorer state, read once per process.  A missing or
        unreadable file is simply an empty one: he then spends the frames
        he spends today and loses nothing."""
        if cls._explorer_disk is None:
            cls._explorer_disk = {}
            try:
                import json as _j
                with open(_EXPLORER_STATE_PATH) as _fh:
                    cls._explorer_disk = (_j.load(_fh) or {}).get(
                        "games") or {}
            except Exception:
                cls._explorer_disk = {}
        return cls._explorer_disk

    @classmethod
    def save_explorers(cls):
        """Write the egocentric models to their OWN small file.

        Atomic via a tmp + replace, so a kill mid-write (he was
        OOM-killed twice on 2026-08-28) leaves the previous state intact
        rather than a truncated file.
        """
        try:
            import json as _j
            import os as _os
            # MERGE, NEVER REPLACE.  _explorers holds only the games
            # touched since THIS process started -- he rotates, so four
            # minutes in it held 1 of 25.  Writing just those would have
            # erased every self he had not happened to revisit yet, so
            # the first save after each restart would destroy most of
            # what the previous one learned.  Start from what is on disk
            # and overwrite only what he has actually re-observed, so a
            # game this process never touched keeps its self.
            out = dict(cls._explorer_saved())
            for _g, _e in cls._explorers.items():
                try:
                    _d = _e.to_dict()
                except Exception:
                    continue
                if _d.get("self_id") or _d.get("model"):
                    out[str(_g)] = _d
            _tmp = _EXPLORER_STATE_PATH + ".tmp"
            with open(_tmp, "w") as _fh:
                _j.dump({"version": 1, "games": out}, _fh)
            _os.replace(_tmp, _EXPLORER_STATE_PATH)
            cls._explorer_saves = getattr(cls, "_explorer_saves", 0) + 1
            return len(out)
        except Exception:
            return 0

    def explorer_observe(self, action) -> None:
        """Learn who I am, what that action did, and where I am."""
        e = self._explorer()
        if e is None or self._grid is None:
            return
        try:
            h = len(self._grid)
            w = len(self._grid[0]) if self._grid else 0
            flat = bytes(bytearray(
                max(0, min(255, int(v)))
                for _r in self._grid for v in _r))
            e.observe(flat, h, w, action, int(self._levels))
            if _EXPLORERPERSIST_ON():
                ARCWorld._explorer_obs = getattr(
                    ARCWorld, "_explorer_obs", 0) + 1
                if ARCWorld._explorer_obs % _EXPLORER_SAVE_EVERY == 0:
                    ARCWorld.save_explorers()
        except Exception:
            pass

    def moves_me(self, action):
        """Predicted chance this action displaces him from here,
        or None when the egocentric model has not seen it."""
        if not _EGOMOVE_ON():
            return None
        e = self._explorer()
        if e is None:
            return None
        try:
            return e.moves_me(int(action))
        except Exception:
            return None

    def toward_new(self, n):
        """An action that takes me somewhere I have not stood, or
        None.  None whenever he has no self, no model for here, or
        every predicted landing is already visited."""
        e = self._explorer()
        if e is None:
            return None
        try:
            return e.toward_new(int(n), int(self._levels))
        except Exception:
            return None

    def felt(self):
        """How he is right now, in [-1, 1], or None when not measured.

        Positive = the last few hundred steps have mostly done something.
        Negative = they have mostly been wasted or retraced.
        """
        if not _FELT_ON() or ARCWorld._felt_n < 50:
            return None
        return float(ARCWorld._felt)

    def felt_low(self):
        """Is he below good?  The demand for change, not a readout."""
        f = self.felt()
        return (f is not None) and _FELTSTEER_ON() and (f < _FELT_LOW)

    # ---- FELTHERE: how it feels to be HERE, (game, level) (2026-09-11) ----
    @staticmethod
    def _felt_here_note(key, d):
        """One step's verdict (+1 competent / -1 waste) into the place's
        register.  Same EWMA as the global felt state."""
        try:
            _r = ARCWorld._felt_here.get(key)
            if _r is None:
                _r = [0.0, 0]
                ARCWorld._felt_here[key] = _r
            _r[0] += _FELT_ALPHA * (float(d) - _r[0])
            _r[1] += 1
        except Exception:
            ARCWorld._felt_here_errors += 1

    @staticmethod
    def _felt_here_value(r):
        """The place's felt state, corrected for the EWMA's start at 0.

        The raw EWMA after n steps carries only 1-(1-alpha)^n of its
        signal -- 22% at the 50-step floor -- so without this every young
        place would read near 0 whatever was happening there, and a
        virgin level would be judged 'below the environment' on its first
        life (doctrine review, 2026-09-11).  Divided out exactly: the
        plain mean at n=1, the EWMA in the limit.  Stays in [-1, 1]
        because |raw| <= 1-(1-alpha)^n by construction."""
        n = int(r[1])
        if n <= 0:
            return 0.0
        w = 1.0 - (1.0 - _FELT_ALPHA) ** n
        if w <= 0.0:
            return 0.0
        return max(-1.0, min(1.0, float(r[0]) / w))

    def felt_here(self):
        """How it feels to be here -- this game, this level -- in [-1, 1],
        or None when the place is unmeasured (fewer than _FELT_MIN_N steps).
        Reads the level he is ON: at a GAME_OVER terminal that is the level
        he just died on (the level is kept, measured 375/375); a level
        clear is not a terminal, so the veto never reads a level he has
        not played."""
        if not _FELTHERE_ON():
            return None
        try:
            _r = ARCWorld._felt_here.get(
                (str(self.game_id), int(self._levels)))
            if _r is None or int(_r[1]) < _FELT_MIN_N:
                return None
            return ARCWorld._felt_here_value(_r)   # class-bound: test stubs borrow the method
        except Exception:
            ARCWorld._felt_here_errors += 1
            return None

    def felt_env(self):
        """How his places feel on average, each weighted by the steps he
        has spent there.  A READOUT ONLY (task_stats): the veto does not
        compare against it -- see felt_bad_here.  None until TWO places
        are measured."""
        if not _FELTHERE_ON():
            return None
        _s, _n, _k = 0.0, 0, 0
        try:
            for _r in list(ARCWorld._felt_here.values()):
                if int(_r[1]) >= _FELT_MIN_N:
                    _s += ARCWorld._felt_here_value(_r) * int(_r[1])
                    _n += int(_r[1])
                    _k += 1
        except Exception:
            ARCWorld._felt_here_errors += 1
            return None
        if _k < 2 or _n <= 0:
            return None
        return _s / float(_n)

    def felt_bad_here(self):
        """Is this place below good?  The demand for change at the one
        grain he can act on it by leaving.  Same rule as felt_low() --
        below _FELT_LOW -- so no new constant enters.  NOT relative to
        his environment: the code review (2026-09-11) showed that a
        steps-weighted environment mean is dragged to wherever he wasted
        the most steps (~-0.5 on live weights), which would have HELD him
        on sp80 L1 (-0.30) and lp85 L1 (-0.48) -- the places this exists
        to release him from -- and drifts with all-time n.  False on any
        doubt: unmeasured here (< _FELT_MIN_N steps), gate off, or any
        error."""
        if not _FELTLEAVE_ON():
            return False
        try:
            _h = self.felt_here()
            if _h is None:
                return False
            return bool(_h < _FELT_LOW)
        except Exception:
            ARCWorld._felt_here_errors += 1
            return False

    def felt_to_dict(self) -> dict:
        """This game's places and how they feel: {game: {level: [ewma, n]}}.
        Gate off -> {} (like bar_to_dict), so `rm /root/FELTHERE_ON` erases
        the persisted register at the next save."""
        if not _FELTHERE_ON():
            return {}
        out = {}
        g = str(self.game_id)
        try:
            for (gg, lv), r in list(ARCWorld._felt_here.items()):
                if gg != g:
                    continue
                out[str(int(lv))] = [round(float(r[0]), 6), int(r[1])]
        except Exception:
            ARCWorld._felt_here_errors += 1
        return {g: out} if out else {}

    def felt_from_dict(self, d) -> int:
        """Restore this game's places; never overwrite a place already
        measured in this process; ignore anything malformed."""
        if not _FELTHERE_ON():
            return 0
        try:
            rows = (d or {}).get(str(self.game_id)) or {}
            items = list(rows.items())
        except Exception:
            return 0
        n = 0
        for lv, r in items:
            try:
                k = (str(self.game_id), int(lv))
                v = max(-1.0, min(1.0, float(r[0])))
                c = int(r[1])
                if c < 0:
                    continue
                if k in ARCWorld._felt_here and ARCWorld._felt_here[k][1] > 0:
                    continue
                ARCWorld._felt_here[k] = [v, c]
                n += 1
            except (TypeError, ValueError, IndexError):
                continue
        return n

    def untried_here(self, n):
        """Actions never taken from the board he is standing on.

        MEASURED: he repeats an action he has already tried here
        67.2% of the time, where having NO memory would give 36.1%.
        His `_tried` is keyed on the 3x3 glance (recurs 13%); this
        is keyed on the board (recurs 31-95%).  The record already
        existed -- `_board_act`, written by the cycle organ --
        and nothing read it for this.

        Empty list when everything here has been tried, or when the
        board is unknown, so he always keeps a move.
        """
        if not _UNTRIEDHERE_ON():
            return []
        try:
            h = self._board_key()
            if h is None:
                return []
            g = str(self.game_id)
            out = [x for x in range(int(n))
                   if (g, h, int(x)) not in ARCWorld._board_act]
            if len(out) >= int(n):
                # nothing here tried yet: this board is not a
                # revisit, so there is nothing to correct
                return []
            return out
        except Exception:
            return []

    def board_inert_here(self, action) -> bool:
        """Has this move changed NOTHING every time he made it HERE?

        Keyed on the board (recurs 31-95%), not the glance (13%).
        Needs >= 2 observations, so the move is made twice before it
        is called dead; unknown returns False so exploration is never
        narrowed.
        """
        try:
            h = self._board_key()
            if h is None:
                return False
            e = ARCWorld._board_noop.get(
                (str(self.game_id), h, int(action)))
            return bool(e is not None and e[0] >= 2 and e[1] == 0)
        except Exception:
            return False

    def cycles_here(self, action) -> bool:
        """Has this move from THIS board always sent him backwards?

        Needs >= 2 observations, so a move is made twice before it is
        demoted.  Unknown returns False, so exploration is untouched.
        """
        if not _CYCLEDEMOTE_ON():
            return False
        try:
            ARCWorld._cyc_calls += 1
            h = self._board_key()
            if h is None:
                return False
            e = ARCWorld._board_act.get((str(self.game_id), h, int(action)))
            if e is not None and e[0] >= 2:
                ARCWorld._cyc_known += 1
            _r = bool(e is not None and e[0] >= 2 and e[1] == e[0])
            if _r:
                ARCWorld._cyc_hits += 1
            return _r
        except Exception:
            return False

    def _win_at(self, locus):
        """The percept window at a GIVEN locus of the CURRENT grid,
        in the same shape and dtype as `world_vector`, so the actor's
        transducer maps it into the very same token space.  Returns
        None rather than guessing when it cannot be read."""
        try:
            if locus is None or not self._grid:
                return None
            r = int(self._radius)
            cr, cc = int(locus[0]), int(locus[1])
            g = self._grid
            h = len(g)
            w = len(g[0]) if h else 0
            if cr - r < 0 or cc - r < 0 or cr + r >= h or cc + r >= w:
                return None
            out = []
            for dr in range(-r, r + 1):
                out.extend(g[cr + dr][cc - r:cc + r + 1])
            return [float(v) for v in out]
        except Exception:
            return None

    def note_attended(self, win, token) -> None:
        """Attention COMMITTED this window: remember its token so future
        scans can price it in O(1).  Called by the runtime after encode."""
        # NOVELTY = A NEW STATE, NOT A NEW WINDOW (2026-07-27).  Measured:
        # keying on windows left states_seen flat at 157 for 747 executions
        # while patch_moves stayed 0 -- _glance returns a never-attended
        # window nearly every step, so window-novelty never runs dry.
        # EVERY commit, novel or not -- this is the denominator.  It must
        # sit OUTSIDE the `not in _seen_tok` test or it would count only
        # the discoveries and the fraction would be a constant 1.0.
        if token is not None:
            self._attend_ever += 1
        if token is not None and token not in self._seen_tok:
            self._seen_tok.add(token)
            self._novel_seen += 1
            self._novel_here += 1
            self._novel_ever += 1
            ARCWorld._novel_all += 1
            if self._since_novel > self._max_gap:
                self._max_gap = self._since_novel
            self._gap_n += 1.0
            self._gap_mean += (self._since_novel - self._gap_mean) / self._gap_n
            self._since_novel = 0
        self._seen_win[tuple(win)] = token

    @property
    def n_actions(self) -> int:
        """Per-game action mask, so it must be live before it is read --
        CurriculumWorld asks for it before the first percept."""
        self._ensure_mine()
        return int(self._n_actions)

    # ---- WorldDriver contract --------------------------------------
    def percept(self) -> Dict[str, object]:
        self._ensure_mine()
        if self._grid is None:
            return {'world_vector': []}
        if self._transducer is not None:
            try:
                win = self._glance(self._grid)
                # ATTENTION COMMITS: exactly one encode per glance, on the
                # window he actually attended -- not 961 on ones he swept past.
                _tok = self._transducer.encode(list(win))
                # FELT PLAY: has he ever seen this state in this game?
                if _tok not in self._felt_seen:
                    self._felt_seen.add(_tok)
                    self._felt_novel = True
                    self._felt_gap_n += 1.0
                    self._felt_gap_mean += (
                        (self._felt_dry - self._felt_gap_mean)
                        / self._felt_gap_n)
                    self._felt_dry = 0
                else:
                    self._felt_novel = False
                    self._felt_dry += 1
                # WIRE THE HOOK (2026-07-29).  `note_attended` carries ALL
                # the state-novelty bookkeeping -- _seen_tok, _novel_here,
                # _novel_all, _max_gap and the _since_novel reset -- and its
                # docstring says the runtime calls it after encode.  Nothing
                # ever did: measured arc_since_novel 431 unreset, max_gap 0,
                # gap_mean 0.0, so ALL THREE depleted rules were reading
                # frozen zeros and he could never leave game 1 of 25.  The
                # hook ends by writing _seen_win itself, so this is a strict
                # superset of the line it replaces and the one-encode-per-
                # glance invariant is untouched.
                #
                # UN-GATED 2026-08-11.  The repair above was placed inside
                # `if _PATCH_ROTATE:` -- an UNRELATED opt-in that gates the
                # mean-gap rotation rule (c) -- and `SEAGI_PATCHROTATE` is not
                # set on the live unit.  So the fix for "nothing ever calls
                # note_attended" only applied when a different experiment was
                # switched on, and the `else` restored the exact window-keyed
                # behaviour it was written to replace.  MEASURED CONSEQUENCE:
                # novel = 0 across ALL 25 games over 10,486 steps, `_max_gap`
                # 0 and `_gap_mean` 0.0 forever, so depleted() rules (b) and
                # (c) could never fire -- which is also what made the 08-11
                # engagement hold (`not depleted`) vacuously true and trapped
                # him for 206 deaths on sb26.
                # Rule (c) keeps its OWN `_PATCH_ROTATE` check in depleted(),
                # so this restores the bookkeeping WITHOUT enabling that rule.
                self.note_attended(win, _tok)
                # LOCUS PROBE (read-only, 2026-08-02).  The percept is
                # CONTENT-ONLY: `_locus` is computed and used for aiming
                # but never enters `world_vector`, so the same 3x3
                # anywhere on the board is the same state.  Measured:
                # only 35.4% of (state,action) pairs have one successor.
                self._probe_tok = _tok
                self._probe_locus = self._locus
                return {'world_vector': [float(v) for v in win]}
            except Exception:
                self.errors += 1
        grid = np.array(self._grid, dtype=np.int16)
        # unbound eye: centre window, so the world is never a hard failure
        r = self._radius
        h, w = grid.shape
        cr, cc = h // 2, w // 2
        win = grid[cr - r:cr + r + 1, cc - r:cc + r + 1].reshape(-1)
        return {'world_vector': [float(v) for v in win]}

    def step(self, action: int) -> Dict[str, object]:
        _pre_board_h = (self._board_hash()
                        if (_CYCLEDEMOTE_ON() or _KNOWNPATH_ON())
                        else None)
        # Gate OFF -> this IS _pre_board_h, so FELT and the verdict
        # stay byte-identical, including when KNOWNPATH is off and
        # both are None.
        _pre_board_nb = (self._board_hash_nb()
                         if (_BARMASK_ON()
                             and (_FELT_ON() or _VERDICT_ON()))
                         else _pre_board_h)
        # BOARDKEY: the before-board for `_note_board_move`, under the
        # SAME key its readers use.  Gate off -> this IS _pre_board_h.
        _pre_board_k = (self._board_key()
                        if (_BOARDKEY_ON() and _CYCLEDEMOTE_ON())
                        else _pre_board_h)
        self._ensure_mine()
        prev_levels = self._levels
        _prev_grid = self._grid
        # WHERE HE IS ACTING FROM.  percept() runs later in this same
        # call and _glance REASSIGNS self._locus, so the locus that
        # belongs to this action must be captured now or it is lost.
        _pre_locus = self._locus
        try:
            if _os.path.exists('/root/LOCUS_PROBE_ON') and getattr(
                    self, '_probe_tok', None):
                import time as _tm
                with open('/root/locus_probe.jsonl', 'a') as _fh:
                    _fh.write(json.dumps({
                        't': _tm.time(), 'g': str(self.game_id),
                        'lo': (list(self._probe_locus)
                               if self._probe_locus else None),
                        'tk': self._probe_tok, 'a': int(action),
                        'lv': int(self._levels)}) + chr(10))
        except Exception:
            pass
        _req = {'cmd': 'step', 'a': int(action)}
        _search_pick_now = getattr(self, '_search_pick', None)
        self._search_pick = None
        if _search_pick_now is not None and int(_search_pick_now[0]) == int(action):
            self._search_pick = _search_pick_now
        _plan_pick_now = getattr(self, '_plan_pick', None)
        self._plan_pick = None
        if _plan_pick_now is not None and int(_plan_pick_now[0]) == int(action):
            self._plan_pick = _plan_pick_now
        elif _plan_pick_now is not None:
            # the plan picked, something else was executed
            ARCWorld._relplan_lost += 1
        # HOISTED (2026-08-12) so it is in scope whether or not a locus
        # exists -- the frame recorder needs it.  Pure function of
        # `action` and `_acts`, so computing it earlier changes nothing.
        # ARC's coordinate action is id 6; only for THAT does the aim
        # change the outcome, which is exactly what makes the aim worth
        # keying a transition table on.
        try:
            _ai = int(action)
            _is_click = (0 <= _ai < len(self._acts)
                         and int(self._acts[_ai]) == 6)
        except Exception:
            _is_click = False
        if self._locus is not None:
            # x = column, y = row -- he clicks where he looks
            # WHERE HE AIMS A COORDINATE ACTION.  Perception is NOT
            # touched -- `_locus` still drives the percept window.
            _aim = (int(self._locus[0]), int(self._locus[1]))
            if _is_click and self._click_hits:
                # BOTH LAYERS, IN PARALLEL: alternate between aiming at
                # something known to work and exploring a fresh locus,
                # so exploration never stops and knowledge is never
                # ignored.  Round-robin over what is known, so one lucky
                # cell cannot monopolise his aim.
                self._click_turn += 1
                if self._click_turn % 2 == 0:
                    _known = sorted(self._click_hits)
                    _aim = _known[(self._click_turn // 2) % len(_known)]
                    self.click_hits_used += 1
            # THE AIM IS PART OF WHAT WORKED.  Replaying the index
            # alone would send a different click and reach a
            # different board.
            _ka = self.known_aim(_pre_board_h, action)
            if (_ka is not None and getattr(self, '_plan_shortcut', False)
                    and getattr(self, '_plan_pick', None) is not None and int(self._plan_pick[0]) == int(action)):
                # SHORTCUT (40): a latched shortcut step's own aim outranks
                # the won route's aim for the same action
                _ka = None
            if _ka is not None:
                _aim = _ka
            # RELPLAN: the plan's aim for the click it asked for, unless
            # a route that actually won has an aim for this board
            _pp = getattr(self, '_plan_pick', None)
            if (_pp is not None and _is_click and _ka is None
                    and int(_pp[0]) == int(action) and _pp[1] is not None
                    and _RELPLAN_ON()):
                _aim = (int(_pp[1][0]), int(_pp[1][1]))
            else:
                # SEARCH (patch 45): the object centre the search asked for
                _sp = getattr(self, '_search_pick', None)
                if (_sp is not None and _is_click and _ka is None
                        and int(_sp[0]) == int(action) and _sp[1] is not None
                        and _SEARCH_ON()):
                    _aim = (int(_sp[1][0]), int(_sp[1][1]))
            self._aimed = _aim if _is_click else None
            _req['y'], _req['x'] = _aim[0], _aim[1]
        # AIM IS PART OF THE ACTION.  For a click the action INDEX does
        # not determine the outcome -- the coordinate does -- so a corpus
        # keyed on the index alone would score two different clicks as
        # the same action and read the disagreement as nondeterminism.
        # Taken from _req, which is what the sidecar was actually sent.
        self._rec_act = (int(action), _req.get('y'), _req.get('x'),
                         bool(_is_click))
        # BOOK THE TRY with the aim ACTUALLY SENT, keyed on the board as
        # it stands BEFORE the step.  And check the premise every step:
        # v2 died because its booked aim matched the executed one 0.7%
        # of the time.  `frontier_aim_miss` makes that a deploy gate
        # instead of a post-mortem.
        try:
            _fh = self._fkey()
            # `.get`, not `[]`: y/x exist only when a locus did.  With
            # `_is_click` true and no locus this raised, the bare except
            # swallowed it, and the step was silently never booked.
            _qy, _qx = _req.get('y'), _req.get('x')
            if _fh is not None and not (_is_click and _qy is None):
                _fa = (int(_qy), int(_qx)) if _is_click else None
                _fp = ARCWorld._fp_aim
                if (_fp and _fp[0] == str(self.game_id)
                        and int(action) in _fp[1]):
                    ARCWorld._frontier_aim_chk += 1
                    if _fp[1][int(action)] != _fa:
                        ARCWorld._frontier_aim_miss += 1
                _st = ARCWorld._ftried.setdefault(
                    (str(self.game_id), int(self._levels)), {})
                _kk = (_fh, int(action), _fa)
                _st[_kk] = _st.get(_kk, 0) + 1
                ARCWorld._frontier_book += 1
                if len(_st) > 60000:
                    # evict the LEAST-tried half; wholesale clearing
                    # would discard a game's entire learned frontier
                    for _d in sorted(_st, key=_st.get)[:30000]:
                        _st.pop(_d, None)
        except Exception:
            pass
        finally:
            # CONSUMED ON EVERY STEP, whatever happened above.  Inside
            # the `if` it survived two paths -- no `_fkey`, and the
            # KeyError -- and was then compared against a LATER step in
            # the SAME game, where the game-stamp cannot catch it.
            ARCWorld._fp_aim = None
        r = self._call(_req)
        self._absorb(r)
        # cleared even if _absorb bailed early on a failed call, so a
        # later open/reset frame can never inherit this action
        _sent = self._rec_act
        self._rec_act = None
        self.steps += 1
        state = self._state
        success = (self._levels > prev_levels) or (state == 'WIN')
        done = state in ('WIN', 'GAME_OVER')
        # LOG THE THING THAT ENDS EVERY LIFE.  GAME_OVER was never
        # written anywhere: `grep -c GAME_OVER` over 24 h of journal
        # returned 0, and the frame corpus cannot see it either because
        # the run id changes on reset, so the last frame of a segment is
        # always the NOT_FINISHED before it.  That made it look like he
        # never ran out on ARC.  His attempts end constantly -- lp85 burned
        # 127..134 on level 1, ~200 steps each.  The `lives=` counter on
        # the TERM line was the only evidence, and it is indirect.
        if state == 'GAME_OVER':
            try:
                import sys as _s
                _s.stderr.write(
                    '[arcrotate] OVER game=%s lvl=%s steps=%s '
                    'lives=%s won_ever=%s\n'
                    % (getattr(self, 'game_id', '?'),
                       getattr(self, '_levels', '?'),
                       getattr(self, 'steps', '?'),
                       getattr(self, '_lives_here', '?'),
                       getattr(self, '_won_ever', '?')))
                _s.stderr.flush()
            except Exception:
                pass
        # HIS OWN TRACE, kept only when the run finished the level.
        if _KNOWNPATH_ON():
            try:
                self._note_path_step(_pre_board_h, _sent, prev_levels)
                if success:
                    self._seal_path(prev_levels)
                elif done:
                    self._seal_depth(prev_levels)
                    if _SHORTCUT_ON():
                        ARCWorld._shortcut_verdict(self, self._path_key(prev_levels), False, None)
                    self._drop_path()
            except Exception:
                pass
        # GOOD STEP, BAD STEP, NOT YET KNOWN -- judged every step, on
        # evidence he can actually see.
        # HOW HE IS RIGHT NOW.  BEFORE _judge_step, which adds the
        # POST-step hash to the attempt set -- a membership test after it
        # is ALWAYS true, and measured 0/349 competent (0.0%) against a
        # corpus prediction of 62-69%.  The three cases are distinguished
        # here: the board did not change (a wasted action -- and a wasted
        # action costs a LEVEL, the budget being the wall), it changed to
        # somewhere he has already stood this attempt (a walk-back), or
        # it changed to somewhere new (ordinary competence).  Nothing
        # about winning enters: the user was explicit that the Super Bowl
        # register is not the broken one.
        if _FELT_ON():
            try:
                _post = self._board_hash_nb()
                if _pre_board_nb is not None and _post is not None:
                    _att = ARCWorld._att_boards.get(str(self.game_id)) or ()
                    if _post == _pre_board_nb:
                        _d = -1.0
                        ARCWorld._felt_noop += 1
                    elif _post in _att:
                        _d = -1.0
                        ARCWorld._felt_back += 1
                    else:
                        _d = 1.0
                        ARCWorld._felt_competent += 1
                    ARCWorld._felt += _FELT_ALPHA * (_d - ARCWorld._felt)
                    ARCWorld._felt_n += 1
                    # ...and how it feels HERE.  Keyed on the level the
                    # action was taken on, so a clearing step is booked
                    # to the level it cleared, not the one it opened.
                    if _FELTHERE_ON():
                        self._felt_here_note(
                            (str(self.game_id), int(prev_levels)), _d)
            except Exception:
                pass
        self._judge_step(_pre_board_h, _sent, success, state,
                         _pre_board_nb)
        # WHERE AM I, AND WHERE HAVE I NOT BEEN.
        if _EXPLORER_ON():
            self.explorer_observe(action)
        self._note_try(action)
        _preobjs = ARCWorld._objs.get(str(self.game_id))
        try:
            _selfmv = self._track_self(action)
        except Exception:
            _selfmv = None
        # THE GOAL IS ON THE BOARD (RELSENSE): after the body is tracked,
        # read the relation he is pursuing and whether this step moved it.
        self._rel_last = None
        if _RELSENSE_ON():
            try:
                self._relsense_step(_prev_grid, prev_levels, success)
            except Exception:
                ARCWorld._rel_errors += 1
        # THE PLAN FOR A PICTURE (RELPLAN): learn what this action did
        # to the canvas and to the indicator, in the state he was in.
        self._plan_pick = None
        if _RELPLAN_ON() and _RELSENSE_ON():
            try:
                _sa = _sent
                self._relplan_step(_prev_grid, prev_levels, int(action),
                                   bool(_sa and _sa[3]),
                                   ((int(_sa[1]), int(_sa[2]))
                                    if (_sa and _sa[3] and _sa[1] is not None) else None))
            except Exception:
                ARCWorld._relplan_errors += 1
        # THE HYPOTHESIS (patch 44): learn where this arrow took him, and
        # whether it won; begin a life on the first step of one.
        self._hyp_win_path = False
        if _HYPOTHESIS_ON() and _RELSENSE_ON():
            try:
                _sa = _sent
                self._hyp_step(_prev_grid, prev_levels, int(action),
                               bool(_sa and _sa[3]), success)
            except Exception:
                ARCWorld._hyp_errors += 1
        # THE SEARCH (patch 45): learn where this control took the board.
        self._search_win_path = False
        if _SEARCH_ON() and _RELSENSE_ON():
            try:
                self._search_step(_prev_grid, prev_levels, int(action), _sent, success)
            except Exception:
                ARCWorld._srch_errors += 1
        try:
            if success:
                self._log_config('win', _preobjs)
            else:
                _k = str(self.game_id)
                _c = ARCWorld._ord_n.get(_k, 0) + 1
                ARCWorld._ord_n[_k] = _c
                if _c % 600 == 0:
                    self._log_config('ordinary', _preobjs)
        except Exception:
            pass
        # WHERE DID THE WORLD CHANGE?  Learned every step, in every game,
        # whether or not he knows which object is him.
        try:
            _lg = str(self.game_id)
            _A = np.array(_prev_grid, dtype=np.int16)
            _B = np.array(self._grid, dtype=np.int16)
            if _A.shape == _B.shape:
                _ys, _xs = np.nonzero(_A != _B)
                if _ys.size:
                    _loc = (float(_ys.mean()), float(_xs.mean()))
                    _pv = ARCWorld._locus_prev.get(_lg)
                    if _pv is not None:
                        _d = (int(round(_loc[0] - _pv[0])),
                              int(round(_loc[1] - _pv[1])))
                        # HAVING MOVED: (0,0) is the modal outcome and
                        # counting it is what hid the self in 9 games.
                        if _d != (0, 0):
                            # DID THE ACTION HE TOOK APPROACH?  Scored
                            # against the target that was on the table
                            # when he chose it (set last call).
                            _lt = getattr(ARCWorld, '_locus_last_tgt', None)
                            if _lt is not None and _lt[2] > 0.0:
                                _b0 = abs(_lt[0]) + abs(_lt[1])
                                _b1 = (abs(_lt[0] - _d[0])
                                       + abs(_lt[1] - _d[1]))
                                ARCWorld._locus_taken_n = getattr(
                                    ARCWorld, '_locus_taken_n', 0) + 1
                                if _b1 < _b0:
                                    ARCWorld._locus_taken_appr = getattr(
                                        ARCWorld,
                                        '_locus_taken_appr', 0) + 1
                            _h = ARCWorld._locushyp.setdefault(
                                _lg, {}).setdefault(int(action), {})
                            _h[_d] = _h.get(_d, 0) + 1
                            ARCWorld._locus_n += 1
                            if len(_h) > 24:
                                _drop = min(_h, key=lambda q: _h[q])
                                _h.pop(_drop, None)
                    ARCWorld._locus_prev[_lg] = _loc
        except Exception:
            pass
        _bodynov = None
        _bodyconf = 0.0
        try:
            _g = str(self.game_id)
            _sig = (ARCWorld._self.get(_g) or (None,))[0]
            _sp2 = None
            if _sig is not None:
                # TOLERANT LOOKUP.  Exact (colour,size) equality fails
                # whenever the object animates or is partly occluded, and
                # then selfpos is None and the whole body/salience path
                # silently disables.  Match colour exactly, size within a
                # quarter -- the same tolerance the tracker already uses.
                _cand = [_o for _o in (ARCWorld._objs.get(_g) or [])
                         if _o[0] == _sig[0]
                         and abs(_o[1] - _sig[1]) <= max(1, _sig[1] // 4)]
                if _cand:
                    _o = max(_cand, key=lambda x: x[1])
                    _sp2 = (_o[2], _o[3])
            _bodynov, _bodyconf = self._body_novelty(_g, _sp2)
            if _bodynov is None:
                # NO AVATAR -> steer by where the world CHANGES instead.
                # Strictly additive: in the 18 games with a self this
                # line never runs and behaviour is byte-identical.
                _bodynov, _bodyconf = self._locus_novelty(_g)
                if _bodynov is not None:
                    ARCWorld._locus_steers = getattr(
                        ARCWorld, '_locus_steers', 0) + 1
        except Exception:
            _bodynov, _bodyconf = None, 0.0
        out = dict(self.percept())
        self._since_novel += 1
        self._steps_here += 1
        self._steps_ever_here += 1
        self._steps_episode += 1
        self._steps_ever += 1
        ARCWorld._steps_all += 1
        ARCWorld._doorway_now = None
        if success:
            self._won_ever += 1
            self._lives_at_last_win = self._lives_here
            self._steps_at_win = int(self._steps_episode)
            self._won_this_episode = True
            # the idea he was holding WORKED -- stop setting anything
            # aside on this level, he has found the shape of it
            try:
                ARCWorld._target_failed.pop(
                    (str(self.game_id), int(self._levels)), None)
            except Exception:
                pass
            # WHAT DID HE REACH?  Read it off the board as it was BEFORE
            # the winning move -- after it, the level has already changed.
            if int(self._levels) > int(prev_levels):
                try:
                    _pp = self._prev_selfpos
                    ARCWorld._dw_tries = getattr(
                        ARCWorld, '_dw_tries', 0) + 1
                    if _pp is None:
                        # NO BODY IN VIEW -> use WHERE THE WORLD CHANGED.
                        # Measured: this was losing HALF of all level-ups
                        # (no_selfpos 2 of 4 tries).  The change centroid
                        # exists in every game, avatar or not, and at the
                        # winning step it is exactly where his action took
                        # effect -- which is what 'reached it' means.
                        _pp = ARCWorld._locus_prev.get(str(self.game_id))
                        if _pp is not None:
                            ARCWorld._dw_by_locus = getattr(
                                ARCWorld, '_dw_by_locus', 0) + 1
                    if _pp is None:
                        ARCWorld._dw_no_selfpos = getattr(
                            ARCWorld, '_dw_no_selfpos', 0) + 1
                    elif not self._prev_objs:
                        ARCWorld._dw_no_objs = getattr(
                            ARCWorld, '_dw_no_objs', 0) + 1
                    if _pp is not None and self._prev_objs:
                        _near = [_o for _o in self._prev_objs
                                 if abs(_o[2] - _pp[0]) + abs(_o[3] - _pp[1]) >= 1.0]
                        if not _near:
                            ARCWorld._dw_no_near = getattr(
                                ARCWorld, '_dw_no_near', 0) + 1
                        if _near:
                            _d = min(_near, key=lambda x:
                                     abs(x[2] - _pp[0]) + abs(x[3] - _pp[1]))
                            _k = (int(_d[0]), int(_d[1]))
                            ARCWorld._goal_kinds[_k] = (
                                ARCWorld._goal_kinds.get(_k, 0) + 1)
                            ARCWorld._goal_colours[_k[0]] = (
                                ARCWorld._goal_colours.get(_k[0], 0) + 1)
                            # and into his LIBRARY, not just this table
                            ARCWorld._doorway_now = _k
                            ARCWorld._dw_captured = getattr(
                                ARCWorld, '_dw_captured', 0) + 1
                except Exception:
                    pass
            if _ARC_ROTATE:
                import sys as _s
                _s.stderr.write(
                    '[arcrotate] WON game=%s won_here=%d->%d levels=%d->%d '
                    'state=%s id=%s\n'
                    % (self.game_id, self._won_here, self._won_here + 1,
                       prev_levels, self._levels, state, id(self)))
                _s.stderr.flush()
            # A WIN COUNTS TOO (2026-07-29).  `_won_here` gates "stay on a game
            # whose life completed something", but it was incremented only on a
            # `_levels` delta while success is `(_levels > prev) or WIN` -- so a
            # WIN left it at 0 and he ROTATED AWAY FROM A GAME HE JUST WON
            # (measured: arc_won_here 0 across successes 0->3).
            self._won_here += 1
            ARCWorld._won_all += 1
        _new_depth = False
        if self._levels > prev_levels:
            # CONFIRMATION EARNS, NOT ONLY FIRST-EVER DEPTH (2026-08-11).
            # The first-ever gate below made mastery a STOCK: `_max_levels_ever`
            # persists and only rises, so the credit fires at most once per
            # (game, depth) FOR ALL TIME.  Measured across the 25 games -- two
            # at level 2, six at level 1, seventeen at 0 -- that whole stock is
            # ~10 events and it is already spent, which is why the mastery wire
            # looks dead.  Same stock-vs-flow failure as the old
            # `int(newly_coherent)` credit.
            # The user's 2026-08-06 ruling governs: repetition on what WORKS is
            # a positive state (knowing IS control -> immortality lean), the
            # earning condition is CONFIRMATION rather than recurrence, and it
            # is self-bounding so no anti-farm wall is wanted.  Clearing a
            # level again CONFIRMS that knowledge still works, so it earns.
            # Self-bounding by the credit law rather than by a gate here:
            # credit is now a bounded rate measured against his OWN habitual
            # rate, so sustained winning simply becomes his habit and the
            # set-point settles -- farming cannot run away by construction.
            self._max_levels_ever = max(int(self._max_levels_ever),
                                        int(self._levels))
            _new_depth = True
            # a level is the loudest possible "this patch still pays"
            if self._since_novel > self._max_gap:
                self._max_gap = self._since_novel
            self._since_novel = 0
            # A LEVEL-UP IS A NEW PATCH (2026-07-31, MEASURED).  `depleted`
            # rule (a) divides `_novel_here / _steps_here` over the WHOLE
            # visit, so the drought he ground through on level 1 is charged
            # against level 2 the instant he arrives.  Measured: 21 of 23 wins
            # were followed by DEPLETED within a median of 17 s (min 5 s), and
            # in three days he never once reached level 2 -- every win reads
            # `levels=0->1`.  `enter()` already defines what re-entering a
            # patch means and a new level IS one, so reuse that meaning rather
            # than invent a rule.  The earned bar (`_max_gap`, `_gap_mean`) is
            # KEPT, exactly as `enter()` keeps it, so rules (b) and (c) still
            # retire a level that genuinely dries up -- earn-or-dissolve
            # intact, no constant, no option removed.  `_won_here` is left
            # alone: it gates the hold, and zeroing it would un-bank the win.
            self._novel_here = 0
            # novelty as of the last terminal here; resets with
            # the counter so the baseline cannot outrun it
            self._novel_at_term = 0
            # A NEW LEVEL OWES HIM A LIFE TOO.  Without this the
            # one-life gate stands open at exactly `here = 0/1 = 0`
            # on the new level -- the 17-second post-win ejection
            # documented above, i.e. the level-2 tether.
            self._term_here = 0
            self._steps_here = 0
        # A CLICK THAT WORKED IS NOW CONTENT.  Only a coordinate
        # action can teach this (`_aimed` is None otherwise), so a
        # keyboard move that changed the world cannot credit a
        # location it never aimed at.
        # A CLICK THAT ONLY SPENT BUDGET DID NOT WORK.  Gate off ->
        # `_grid_changed_nb` IS `self._grid != _prev_grid`.
        if self._aimed is not None and self._grid_changed_nb(_prev_grid):
            self._click_hits.add(self._aimed)
        # ...and what this ACTION did, whether or not it aimed.  A click
        # learns a LOCATION above; every action also teaches whether it
        # does anything at all in this game.
        try:
            _ae = self._act_effect.setdefault(int(action), [0, 0])
            _ae[0] += 1
            if self._grid != _prev_grid:
                _ae[1] += 1
        except (TypeError, ValueError):
            pass
        self._level_records_step()
        out.update({
            # DID THE WORLD RESPOND?  29% of his actions change nothing
            # (measured over 2,400 transitions).  He cannot feel that from
            # onward-novelty alone, and a non-event costs him life.
            'world_changed': bool(self._grid != _prev_grid),
            # WHAT THIS PLACE BECAME.  The same window, same locus,
            # after the action -- the world's answer rather than his
            # next glance.  None when there was no locus or the board
            # geometry moved, in which case the caller simply learns
            # nothing this step.
            'world_after_vec': self._win_at(_pre_locus),
            # record whether that move sent him somewhere he had been
            'cycle_noted': self._note_board_move(_pre_board_k, action),
            'success': bool(success),
            # SELF-CONSTRUCTED PROGRESS (2026-08-25): ARC emits no
            # reward, so this is the board judging itself.
            'progress_advance': self._progress_step(),
            # RELSENSE: fractional movement of the pursued relation
            # toward (progress) or away from (regress) its goal this
            # step; rel_n = candidates; rel_view = the pursued one.
            'rel_progress': float((getattr(self, '_rel_last', None) or {}).get('progress') or 0.0),
            'rel_regress': float((getattr(self, '_rel_last', None) or {}).get('regress') or 0.0),
            'rel_n': int((getattr(self, '_rel_last', None) or {}).get('n') or 0),
            'rel_view': (getattr(self, '_rel_last', None) or {}).get('view'),
            # has a CLEAR taught him that this is the level's goal?
            'rel_confirmed': bool((getattr(self, '_rel_last', None) or {}).get('confirmed')),
            'done': bool(done),
            'timed_out': bool(state == 'GAME_OVER'),
            # A FIRST-TIME DEEPEST LEVEL is un-farmable new learning; it
            # fires the existing mastery wire.  CurriculumWorld still owns
            # graduation and now ORs the two rather than suppressing this.
            'goal_changed': bool(_new_depth),
            'levels_completed': int(self._levels),
            # FELT PLAY (2026-08-02).  `note_attended` already tracks
            # whether this percept was a never-seen state and how long
            # the dry spell has run, and `_gap_mean` is his OWN earned
            # bar for how long a dry spell normally lasts.  The actor
            # fires curiosity / goal_stuck off these -- exploring should
            # feel like something.  Read-only exposure; nothing here
            # changes what the world does.
            'bar_fill': self._bar_fill(),
            'obj_n': ARCWorld._obj_n.get(str(self.game_id)),
            'self_sig': str((ARCWorld._self.get(str(self.game_id))
                             or (None,))[0]),
            'self_determinism': (ARCWorld._self.get(str(self.game_id))
                                 or (None, None))[1],
            'self_moved': _selfmv,
            'consumables': self._consumables(str(self.game_id)),
            'vanish_events': ARCWorld._contact.get(str(self.game_id), 0),
            'occlusion': ARCWorld._occl.get(str(self.game_id)),
            'mem_sizes': ARCWorld._mem_sizes(),
            'rules_known': len(ARCWorld._rules),
            'rule_transfers': ARCWorld._transfers,
            'rule_seeds': getattr(ARCWorld, '_seed_hits', 0),
            'rules_retrodicted': getattr(ARCWorld, '_retro_n', 0),
            'target_salience': round(getattr(ARCWorld, '_last_sal', 0.0), 3),
            'approach_steps': getattr(ARCWorld, '_approach_n', 0),
            'kinds_investigated': len(ARCWorld._seen_kind),
            'roles': self._roles_now(),
            'counts': self._counts_now(),
            # THE STEP RESULT is what world_actor reads.  This lived only
            # in task_stats() (which /status reads and the actor never
            # does), so every captured doorway was silently dropped --
            # dw_diag showed captured=1 against doorways_written=0.
            'doorway': (list(ARCWorld._doorway_now)
                        if ARCWorld._doorway_now else None),
            'roles_written': ARCWorld._role_n,
            'target_commits': getattr(ARCWorld, '_commits', 0),
            'calib_action': self._calib_action(),
            'calib_steers': ARCWorld._calib_n,
            'actions_tried': len(ARCWorld._act_tries.get(
                str(self.game_id), {})),
            'rules_multi_game': sum(
                1 for r in ARCWorld._rules.values() if len(r['games']) > 1),
            'body_novelty': _bodynov,
            'self_conf': _bodyconf,
            'positions_seen': len(ARCWorld._posvis.get(
                (str(self.game_id), int(self._levels)), {})),
            'novel_state': bool(self._felt_novel),
            'since_novel': int(self._felt_dry),
            'gap_mean': float(self._felt_gap_mean),
        })
        # one-step history: the board as he last saw it BEFORE acting
        try:
            self._prev_objs = list(ARCWorld._objs.get(str(self.game_id)) or [])
            self._prev_selfpos = _sp2
        except Exception:
            pass
        if done:
            # TRY A DIFFERENT IDEA NEXT TIME.  The attempt ended still committed to
            # this target and it never paid, so the next attempt commits
            # elsewhere.  Only on a LOSS -- a win keeps the idea.
            try:
                if not self._won_this_episode:
                    _g = str(self.game_id)
                    _held = ARCWorld._target.get(_g)
                    if _held is not None:
                        _k = (_g, int(self._levels))
                        ARCWorld._target_failed.setdefault(
                            _k, set()).add((int(_held[0]), int(_held[1])))
                        ARCWorld._retries = getattr(
                            ARCWorld, '_retries', 0) + 1
                ARCWorld._target.pop(str(self.game_id), None)
            except Exception:
                pass
            self._won_this_episode = False
            self._lives_here += 1
            self._level_life_end()
            # survives the reset -- the TERM log reads this AFTER step()
            self._steps_last_episode = int(self._steps_episode)
            self._steps_episode = 0
            self._absorb(self._call({'cmd': 'reset'}))
        return out

    def reset_law(self, seed: int) -> None:
        self._seed = int(seed)
        self._absorb(self._call({'cmd': 'open', 'game': self.game_id,
                                 'seed': self._seed}))

    def task_stats(self) -> Dict[str, object]:
        return {
            'arc_game': self.game_id,
            'arc_n_actions': int(self._n_actions or 0),
            'cycledemote_on': bool(_CYCLEDEMOTE_ON()),
            'levelhold_on': bool(_LEVELHOLD_ON()),
            'lvl_records': int(ARCWorld._lvl_records),
            'lvl_dry_ejects': int(ARCWorld._lvl_dry_ejects),
            'lvl_errors': int(ARCWorld._lvl_errors),
            'lvl_last': str(ARCWorld._lvl_last),
            'lvl_dry_here': int(ARCWorld._lvl_dry.get((str(self.game_id), int(self._levels)), 0)),
            'lvl_gap_max_here': int(ARCWorld._lvl_gap_max.get((str(self.game_id), int(self._levels)), 0)),
            'lvl_levels': int(len(ARCWorld._lvl_dry)),
            'knownpath_on': bool(_KNOWNPATH_ON()),
            'paths_known': len(ARCWorld._paths),
            'pathseed_on': bool(_PATHSEED_ON()),
            'seed_paths': int(ARCWorld._seed_paths),
            'path_steps': sum(len(v) for v in ARCWorld._paths.values()),
            'path_follow': int(ARCWorld._path_follow),
            'path_sealed': int(ARCWorld._path_sealed),
            'depthseal_on': bool(_DEPTHSEAL_ON()),
            'depth_paths': len(ARCWorld._depth_paths),
            'depth_steps': sum(len(v)
                               for v in ARCWorld._depth_paths.values()),
            'depth_sealed': int(ARCWorld._depth_sealed),
            'depth_seeded': int(ARCWorld._depth_seeded),
            'depth_follow': int(ARCWorld._depth_follow),
            'depth_levels': sorted({int(k[1])
                                    for k in ARCWorld._depth_paths}),
            'path_loops': int(ARCWorld._path_loops),
            'aim_loops': int(ARCWorld._aim_loops),
            'verdict_on': bool(_VERDICT_ON()),
            'verdict_steps': len(ARCWorld._verdict),
            'verdict_good': int(ARCWorld._verdict_good),
            'verdict_bad': int(ARCWorld._verdict_bad),
            'verdict_unk': int(ARCWorld._verdict_unk),
            'att_boards': sum(
                len(_s) for _s in ARCWorld._att_boards.values()),
            'att_resets': int(ARCWorld._att_resets),
            'verdict_judged': sum(
                1 for _v in ARCWorld._verdict.values() if sum(_v) >= 2),
            'board_act_pairs': len(ARCWorld._board_act),
            'board_noop_pairs': len(ARCWorld._board_noop),
            'board_noop_dead': sum(
                1 for _v in ARCWorld._board_noop.values()
                if _v[0] >= 2 and _v[1] == 0),
            'untriedhere_on': bool(_UNTRIEDHERE_ON()),
            'explorer_on': bool(_EXPLORER_ON()),
            'egomove_on': bool(_EGOMOVE_ON()),
            'explorer_games': len(ARCWorld._explorers),
            'felt_on': bool(_FELT_ON()),
            'felt_steer_on': bool(_FELTSTEER_ON()),
            'felthere_on': bool(_FELTHERE_ON()),
            'feltleave_on': bool(_FELTLEAVE_ON()),
            'felt_here': (round(float(self.felt_here()), 4)
                          if self.felt_here() is not None else None),
            'felt_env': (round(float(self.felt_env()), 4)
                         if self.felt_env() is not None else None),
            'felt_places': sum(1 for _r in list(ARCWorld._felt_here.values())
                               if isinstance(_r[1], int) and _r[1] >= _FELT_MIN_N),
            'felt_here_errors': int(ARCWorld._felt_here_errors),
            'felt': (round(float(ARCWorld._felt), 4)
                     if ARCWorld._felt_n else None),
            'felt_n': int(ARCWorld._felt_n),
            'felt_competent': int(ARCWorld._felt_competent),
            'felt_noop': int(ARCWorld._felt_noop),
            'felt_back': int(ARCWorld._felt_back),
            'barmask_on': bool(_BARMASK_ON()),
            'clickhit_on': bool(_CLICKHIT_ON()),
            'progmask_on': bool(_PROGMASK_ON()),
            'boardkey_on': bool(_BOARDKEY_ON()),
            'barpersist_on': bool(_BARPERSIST_ON()),
            'relplan_on': bool(_RELPLAN_ON()),
            'hypothesis_on': bool(_HYPOTHESIS_ON()),
            'hyp_asks': int(ARCWorld._hyp_asks),
            'hyp_acts': int(ARCWorld._hyp_acts),
            'hyp_errors': int(ARCWorld._hyp_errors),
            'hyp_holds': int(ARCWorld._hyp_holds),
            'hyp_tables': int(len(ARCWorld._hyp)),
            'hyp_keys': int(sum(len(h.auto) for h in ARCWorld._hyp.values())),
            'hyp_refuted': int(sum(len(h.refuted) for h in ARCWorld._hyp.values())),
            'hyp_confirmed': int(sum(1 for h in ARCWorld._hyp.values() if h.confirmed is not None)),
            'hyp_lost': int(sum(h.lost for h in ARCWorld._hyp.values())),
            'hyp_rounds': int(sum(h.rounds for h in ARCWorld._hyp.values())),
            'hyp_evicted': int(sum(h.evicted for h in ARCWorld._hyp.values())),
            'hyp_wins_known': int(sum(1 for h in ARCWorld._hyp.values() if h.has_win())),
            'hyp_last': str(ARCWorld._hyp_last),
            'search_on': bool(_SEARCH_ON()),
            'srch_asks': int(ARCWorld._srch_asks),
            'srch_acts': int(ARCWorld._srch_acts),
            'srch_errors': int(ARCWorld._srch_errors),
            'srch_holds': int(ARCWorld._srch_holds),
            'lib_schemas': (len(ARCWorld._lib.paid) if ARCWorld._lib is not None else 0),
            'lib_clears': (int(ARCWorld._lib.clears) if ARCWorld._lib is not None else 0),
            'lib_unexplained': (int(ARCWorld._lib.unexplained) if ARCWorld._lib is not None else 0),
            'lib_pursues': int(ARCWorld._lib_pursues),
            'lib_refutes': int(ARCWorld._lib_refutes),
            'lib_errors': int(ARCWorld._lib_errors),
            'srch_tables': int(len(ARCWorld._srch)),
            'srch_states': int(sum(len(s.trans) for s in ARCWorld._srch.values())),
            'srch_lost': int(sum(s.lost for s in ARCWorld._srch.values())),
            'srch_evicted': int(sum(s.evicted for s in ARCWorld._srch.values())),
            'srch_wins_known': int(sum(1 for s in ARCWorld._srch.values() if s.has_win())),
            'srch_last': str(ARCWorld._srch_last),
            'rel_plan_asks': int(ARCWorld._relplan_asks),
            'rel_plan_acts': int(ARCWorld._relplan_acts),
            'rel_plan_errors': int(ARCWorld._relplan_errors),
            'rel_plan_effects': int(sum(len(rp.effects) for rp in ARCWorld._relplan.values())),
            'rel_plan_spoiled': int(sum(len(getattr(rp, 'spoiled', {})) for rp in ARCWorld._relplan.values())),
            'rel_plan_probes': int(sum(getattr(rp, 'probes', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_probe_applies': int(sum(getattr(rp, 'probe_applies', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_probe_paint_applies': int(sum(getattr(rp, 'probe_paint_applies', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_probe_exec': int(sum(getattr(rp, 'probe_exec', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_walks': int(sum(getattr(rp, 'walks', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_walks_dropped': int(sum(getattr(rp, 'walks_dropped', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_walks_dropped_why': dict((k, int(sum(getattr(rp, 'walks_dropped_why', {}).get(k, 0) for rp in ARCWorld._relplan.values()))) for k in set(kk for rp in ARCWorld._relplan.values() for kk in getattr(rp, 'walks_dropped_why', {}))),
            'rel_plan_floor_held': int(sum(getattr(rp, 'floor_held', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_shortcuts': int(sum(getattr(rp, 'shortcuts', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_shortcut_done': int(sum(getattr(rp, 'shortcut_done', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_shortcut_dropped': int(sum(getattr(rp, 'shortcut_dropped', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_shortcut_won': int(sum(getattr(rp, 'shortcut_won', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_shortcut_refuted': int(sum(getattr(rp, 'shortcut_refuted', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_shortcut_yields': int(sum(getattr(rp, 'shortcut_yields', 0) for rp in ARCWorld._relplan.values())),
            'shortcut_on': bool(_SHORTCUT_ON()),
            'placekey_on': bool(_PLACEKEY_ON()),
            'rel_plan_places': int(sum(len(set(P for P, _a in getattr(rp, 'place_auto', {}))) for rp in ARCWorld._relplan.values())),
            'rel_plan_place_navs': int(sum(getattr(rp, 'place_navs', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_place_hits': int(sum(getattr(rp, 'place_hits', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_place_explores': int(sum(getattr(rp, 'place_explores', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_probe_hue_clicks': int(sum(getattr(rp, 'probe_hue_clicks', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_null_masks': int(sum(len(getattr(rp, 'null_masks', {})) for rp in ARCWorld._relplan.values())),
            'rel_plan_null_known': int(sum(len(rp._null_known()) for rp in ARCWorld._relplan.values() if hasattr(rp, '_null_known'))),
            'rel_plan_same_mask': int(sum(len(getattr(rp, 'same_mask', ())) for rp in ARCWorld._relplan.values())),
            'rel_plan_lookaheads': int(sum(getattr(rp, 'lookaheads', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_lookahead_hits': int(sum(getattr(rp, 'lookahead_hits', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_lookahead_acts': int(sum(getattr(rp, 'lookahead_acts', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_lookahead_done': int(sum(getattr(rp, 'lookahead_done', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_lookahead_dropped': int(sum(getattr(rp, 'lookahead_dropped', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_seq_hues': int(sum(getattr(rp, 'seq_hues', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_seq_twins': int(sum(getattr(rp, 'seq_twins', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_la_full': int(sum(getattr(rp, 'la_full', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_la_full_capped': int(sum(getattr(rp, 'la_full_capped', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_seq_prehues': int(sum(getattr(rp, 'seq_prehues', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_seq_refuted': int(sum(getattr(rp, 'seq_refuted', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_la_unfloored': int(sum(getattr(rp, 'la_unfloored', 0) for rp in ARCWorld._relplan.values())),
            'rel_plan_worked': int(sum(len(getattr(rp, 'worked', {})) for rp in ARCWorld._relplan.values())),
            'rel_plan_pred': [int(sum(rp.reached for rp in ARCWorld._relplan.values())),
                              int(sum(rp.judged for rp in ARCWorld._relplan.values()))],
            # THE PLANNER'S OWN PATH: executed plan actions whose predicted
            # canvas change came true (the review: rel_plan_pred scores the null)
            'rel_plan_pplan': [int(sum(rp.plan_reached for rp in ARCWorld._relplan.values())),
                               int(sum(rp.plan_judged for rp in ARCWorld._relplan.values()))],
            'rel_plan_exec': int(sum(rp.exec_n for rp in ARCWorld._relplan.values())),
            'rel_plan_inert': int(ARCWorld._relplan_inert),
            'rel_plan_by_game': dict(ARCWorld._relplan_by_game),
            'rel_plan_why': dict(ARCWorld._relplan_why),
            'rel_plan_lost': int(ARCWorld._relplan_lost),
            'rel_plan_last': str(ARCWorld._relplan_last),
            'relsense_on': bool(_RELSENSE_ON()),
            'rel_steps': int(ARCWorld._rel_steps),
            'rel_fires': int(ARCWorld._rel_fires),
            'rel_confirms': int(ARCWorld._rel_confirms),
            'rel_errors': int(ARCWorld._rel_errors),
            'rel_levels': len(ARCWorld._rel),
            'rel_pursued': (lambda _r: (_r._name(_r._pursued) if _r is not None and _r._pursued is not None else None))(
                ARCWorld._rel.get((str(self.game_id), int(self._levels)))),
            'rel_confirmed': sum(len(_r.confirmed) for _r in ARCWorld._rel.values()),
            'bar_learn_n': int(ARCWorld._bar_learn_n),
            'bar_decided': int(ARCWorld._bar_decided),
            'bar_rows': dict(('%s|%s' % (a, b),
                              list(v) if isinstance(v, (tuple, list))
                              else int(v))
                             for (a, b), v in
                             ARCWorld._bar_row.items()),            'explorer_persist_on': bool(_EXPLORERPERSIST_ON()),
            'explorer_loaded': int(getattr(
                ARCWorld, '_explorer_loaded', 0)),
            'explorer_saves': int(getattr(ARCWorld, '_explorer_saves', 0)),
            'explorer_selves': sum(
                1 for _e in ARCWorld._explorers.values()
                if getattr(_e, 'self_id', None) is not None),
            'explorer_targets': sum(
                int(getattr(_e, 'targets', 0))
                for _e in ARCWorld._explorers.values()),
            'explorer_visited': sum(
                sum(len(_v) for _v in getattr(_e, 'visited', {}).values())
                for _e in ARCWorld._explorers.values()),
            # HOW MANY STORED PAIRS WOULD DEMOTE IF ASKED.  Separates
            # 'the premise is wrong' from 'the branch never asks'.
            'cyc_calls': int(ARCWorld._cyc_calls),
            'cyc_known': int(ARCWorld._cyc_known),
            'cyc_hits': int(ARCWorld._cyc_hits),
            'board_act_cyclable': sum(
                1 for _v in ARCWorld._board_act.values()
                if _v[0] >= 2 and _v[1] == _v[0]),
            'board_act_repeat': sum(
                1 for _v in ARCWorld._board_act.values() if _v[0] >= 2),
            'board_seen': sum(len(v) for v in ARCWorld._board_seen.values()),
            'stallcut_on': bool(_STALLCUT_ON()),
            'prog_here': int(getattr(self, '_prog_here', 0)),
            'prog_all': int(getattr(ARCWorld, '_prog_all', 0)),
            'arc_state': self._state,
            'arc_levels': self._levels,
            'arc_steps': self.steps,
            'arc_errors': self.errors,
            'arc_live': bool(getattr(self, '_live', True)),
            'arc_gap_mean': round(float(getattr(self, '_gap_mean', 0.0)), 1),
            'arc_max_gap': int(getattr(self, '_max_gap', 0)),
            # the parts of env = novel_all/steps_all.  Its only readout
            # was the _dep_why string, which is never cleared and is
            # read on HELD terminals -- an artifact that misled three
            # earlier attempts into a false premise.
            'arc_novel_all': int(ARCWorld._novel_all),
            'arc_steps_all': int(ARCWorld._steps_all),
            'arc_novel_here': int(getattr(self, '_novel_here', 0)),
            'arc_novel_at_term': int(getattr(self, '_novel_at_term', 0)),
            'arc_since_novel': int(getattr(self, '_since_novel', 0)),
            # lives completed in THIS visit.  The mid-episode ejection is
            # unreachable while this is 0.  If ejections continue with
            # this at 0, the gate is NOT live -- treat as not deployed.
            'arc_term_here': int(getattr(self, '_term_here', 0)),
            # THE DEPLOY GATE.  `aim_miss` must stay near 0 -- it is the
            # premise v2 got wrong.  `steers` counts only choices the
            # frontier actually moved.
            'frontier_steers': int(getattr(ARCWorld,
                                           '_frontier_steers', 0)),
            'frontier_aim_miss': int(getattr(ARCWorld,
                                             '_frontier_aim_miss', 0)),
            # THE DENOMINATOR.  miss/chk is the premise; chk == 0 means
            # the premise was never tested, NOT that it holds.
            'frontier_aim_chk': int(getattr(ARCWorld,
                                            '_frontier_aim_chk', 0)),
            'frontier_book': int(getattr(ARCWorld, '_frontier_book', 0)),
            'frontier_edges': int(len(ARCWorld._ftried.get(
                (str(self.game_id), int(self._levels)), {}) or {})),
            'arc_rotate_on': bool(_PATCH_ROTATE),
            'arc_won_here': int(getattr(self, '_won_here', 0)),
            'yieldstay_on': bool(_YIELDSTAY_ON()),
            'won_all': int(ARCWorld._won_all),
            'yield_stays': int(ARCWorld._yield_stays),
            'arc_won_ever': int(getattr(self, '_won_ever', 0)),
            'arc_lives_here': int(getattr(self, '_lives_here', 0)),
            'arc_lives_at_win': int(getattr(self, '_lives_at_last_win', 0)),
            'arc_dep_why': str(getattr(self, '_dep_why', '')),
            'arc_steps_episode': int(getattr(self, '_steps_episode', 0)),
            'arc_steps_at_win': int(getattr(self, '_steps_at_win', 0)),
            'arc_levels_now': int(getattr(self, '_levels', 0)),
            'arc_max_levels_ever': int(getattr(self, '_max_levels_ever', 0)),
            # str keys: JSON cannot key a dict by tuple, and a raise here
            # takes down the WHOLE status tree (measured 2026-08-07 --
            # it broke the instant the first doorway was captured)
            'goal_kinds': dict(
                ('c%d_s%d' % (int(k[0]), int(k[1])), int(v))
                for k, v in list(ARCWorld._goal_kinds.items())[:12]),
            'goal_colours': dict(
                (str(k), int(v)) for k, v in ARCWorld._goal_colours.items()),
            'dw_diag': {
                'tries': getattr(ARCWorld, '_dw_tries', 0),
                'no_selfpos': getattr(ARCWorld, '_dw_no_selfpos', 0),
                'no_objs': getattr(ARCWorld, '_dw_no_objs', 0),
                'no_near': getattr(ARCWorld, '_dw_no_near', 0),
                'captured': getattr(ARCWorld, '_dw_captured', 0),
                'by_locus': getattr(ARCWorld, '_dw_by_locus', 0),
            },
            'retries_differently': getattr(ARCWorld, '_retries', 0),
            'locus_observations': ARCWorld._locus_n,
            'locus_steers': getattr(ARCWorld, '_locus_steers', 0),
            'locus_approach': getattr(ARCWorld, '_locus_approach', 0),
            'locus_actions_scored': getattr(
                ARCWorld, '_locus_actions_scored', 0),
            'locus_taken_n': getattr(ARCWorld, '_locus_taken_n', 0),
            'locus_taken_approach': getattr(
                ARCWorld, '_locus_taken_appr', 0),
            'locus_games': len(ARCWorld._locushyp),
            'goal_ideas_buried': sum(
                len(v) for v in ARCWorld._target_failed.values()),
            'goal_match': (list(ARCWorld._goal_last)
                           if ARCWorld._goal_last else None),
            'doorway': (list(ARCWorld._doorway_now)
                        if ARCWorld._doorway_now else None),
            'arc_steps_ever': int(getattr(self, '_steps_ever', 0)),
            'arc_liferotate_on': bool(_ARC_ROTATE),
            'arc_reopens': int(getattr(self, 'reopens', 0)),
            'click_hits': len(self._click_hits),
            'act_effect': self.act_effect_report(),
            'click_hits_used': self.click_hits_used,
            'size': self._levels,      # CurriculumWorld reads .size for depth
        }

    def act_effect_to_dict(self) -> dict:
        """What each action DOES on this board: action -> [tried, changed].

        Persisted for the same reason `click_hits` is -- it is his own
        earned record of this game, and without it every life starts by
        re-testing buttons he has already proved inert here.
        """
        return {str(self.game_id): {
            str(a): [int(v[0]), int(v[1])]
            for a, v in self._act_effect.items()}}

    def act_effect_from_dict(self, d) -> int:
        """Only THIS game's actions.  Cross-game action->effect is REFUTED
        (20.53% vs a 30.11% null), so merging games would import noise."""
        try:
            rows = (d or {}).get(str(self.game_id)) or {}
        except Exception:
            return 0
        n = 0
        for a, v in rows.items():
            try:
                self._act_effect[int(a)] = [int(v[0]), int(v[1])]
                n += 1
            except (TypeError, ValueError, IndexError, KeyError):
                continue
        return n

    def act_effect_report(self) -> dict:
        """SHADOW ONLY -- what a steer WOULD suppress, before one exists.

        An action is 'proved inert here' at >=DEAD_MIN_N tries with a
        changed-rate <=DEAD_RATE.  Reported, never enforced: an action dead
        in 40 of 40 tries may still become live on a later level, so the
        eventual steer must be a weighted prior, not a ban.
        """
        DEAD_MIN_N, DEAD_RATE = 8, 0.05
        tried = sum(v[0] for v in self._act_effect.values())
        changed = sum(v[1] for v in self._act_effect.values())
        dead = {a: v for a, v in self._act_effect.items()
                if v[0] >= DEAD_MIN_N and (v[1] / v[0]) <= DEAD_RATE}
        return {
            'act_keys': len(self._act_effect),
            'act_tried': tried,
            'act_changed': changed,
            'act_dead_share': (round(1.0 - changed / tried, 4)
                               if tried else None),
            'act_proved_inert': sorted(dead),
            'act_would_suppress': sum(v[0] for v in dead.values()),
            'act_table': {str(a): v for a, v in
                          sorted(self._act_effect.items())},
        }

    def click_hits_to_dict(self) -> dict:
        """Locations that DO something here.  ~3% of the board, perfectly
        stable across episodes, ~34 blind clicks each to find."""
        return {str(self.game_id): sorted(list(x) for x in self._click_hits)}

    def paid_to_dict(self) -> dict:
        """What this game has EVER paid him, and how long the last one took.

        Persisted for the same reason `click_hits` is: it is his own earned
        record of this board, and without it every restart forgets which games
        are worth staying on -- which is exactly what kept the stay-where-paid
        hold from ever firing (measured: six consecutive terminals, all MOVE).
        """
        # PERSIST WHO HE IS (2026-08-03).  `_self` was an in-memory class
        # dict, so every restart threw away ~100 min of contingency
        # convergence (18/25 games) -- and affordance learning is gated
        # behind it.  `world_route` already crosses the boundary; this
        # must too.  Only the CONCLUSION is stored, not the whole table:
        # the table rebuilds cheaply, the convergence time is what hurt.
        _sr = None
        try:
            _got = ARCWorld._self.get(str(self.game_id))
            if _got:
                # n MATTERS: confidence is determinism * n/(1+n), so a
                # self restored with n=0 has ZERO confidence and steers
                # nothing -- which defeats persisting it at all.
                _sr = [int(_got[0][0]), int(_got[0][1]), float(_got[1]),
                       int(_got[2] or 0)]
        except Exception:
            _sr = None
        return {str(self.game_id): [int(self._won_ever),
                                    int(self._lives_here),
                                    int(self._lives_at_last_win),
                                    int(self._max_levels_ever),
                                    _sr,
                                    int(self._novel_ever),
                                    int(self._steps_ever_here),
                                    int(self._attend_ever)]}

    def paid_from_dict(self, d) -> int:
        """Only this game's row -- a record never transfers between games."""
        try:
            row = (d or {}).get(str(self.game_id)) or None
        except Exception:
            return 0
        if not row:
            return 0
        try:
            self._won_ever = max(int(self._won_ever), int(row[0]))
            self._lives_here = max(int(self._lives_here), int(row[1]))
            self._lives_at_last_win = max(int(self._lives_at_last_win),
                                          int(row[2]))
            # 4th element added 2026-07-31; older rows are 3 long and simply
            # leave the depth record at 0.  WITHOUT this the credit re-fires
            # after every restart -- the farm this design exists to prevent.
            if len(row) > 3:
                self._max_levels_ever = max(int(self._max_levels_ever),
                                            int(row[3]))
            # 5th element (2026-08-03): who he is in this game.
            if len(row) > 4 and row[4]:
                try:
                    _r4 = list(row[4])
                    _c, _sz, _dt = _r4[0], _r4[1], _r4[2]
                    _n = int(_r4[3]) if len(_r4) > 3 else 0
                    ARCWorld._self[str(self.game_id)] = (
                        (int(_c), int(_sz)), float(_dt), _n)
                except Exception:
                    pass
            # 6th/7th elements (2026-08-11): the per-game novelty meter.
            # Older rows are shorter and simply start the meter at 0 -- it is
            # pure instrumentation, so a missing value costs nothing but a
            # gap in the record.  max() like every other field, so a stale
            # save can never walk the count backwards.
            if len(row) > 6:
                try:
                    self._novel_ever = max(int(self._novel_ever),
                                           int(row[5]))
                    self._steps_ever_here = max(int(self._steps_ever_here),
                                                int(row[6]))
                except (TypeError, ValueError):
                    pass
            # 8th element (2026-08-11): attention commits -- the denominator
            # that shares note_attended's clock.  Absent from every row
            # written before this line existed, and it must NOT be back-
            # filled from `_novel_ever`: that would assert every past commit
            # was a discovery and print a 100% novelty rate nobody measured.
            # It starts at 0, so the ALL-TIME fraction is only trustworthy
            # once this has accumulated as long as the numerator has.  The
            # census reads DELTAS against a re-saved baseline, so its numbers
            # are clean from the next save onward.
            if len(row) > 7:
                try:
                    self._attend_ever = max(int(self._attend_ever),
                                            int(row[7]))
                except (TypeError, ValueError):
                    pass
            return 1
        except (TypeError, ValueError, IndexError):
            return 0

    def click_hits_from_dict(self, d) -> int:
        """Only this game's locations -- they never transfer between games."""
        try:
            rows = (d or {}).get(str(self.game_id)) or []
        except Exception:
            return 0
        n = 0
        for r in rows:
            try:
                self._click_hits.add((int(r[0]), int(r[1])))
                n += 1
            except (TypeError, ValueError, IndexError):
                continue
        return n

    @property
    def depleted(self) -> bool:
        """He has now gone LONGER without learning anything here than he
        ever had to before in this game.  Self-scaling from his own
        history -- no threshold, no budget, no tuned constant.  Requires
        at least one prior gap, so a game he has just entered can never
        report depleted."""
        _yield_hold = False
        # HE STAYS WHERE HE WINS.  Every rule below asks about NOVELTY
        # or progress; none asks what the patch has PAID.  MEASURED
        # 2026-08-27: all 17 wins in 7 days came from 7 of 25 games,
        # and ~72% of his steps went to games that have never paid --
        # his best patch, one win per 838 steps, got the FEWEST steps.
        #
        # RETURNS FALSE ONLY.  It can make him stay, never leave
        # sooner, so it cannot re-open the rotation storm.  Needs a win
        # in THIS visit, and is self-limiting with no constant: as he
        # stays without winning again the rate falls below the
        # environment and the suppression ends by itself.
        # `_won_all > _won_here`: another patch must ALSO have paid,
        # or the "environment" is just this patch and the comparison
        # is vacuous -- 1/steps_here > 1/steps_all is true forever.
        if (_YIELDSTAY_ON() and self._won_here > 0
                and self._steps_here > 0
                and ARCWorld._won_all > self._won_here
                and ARCWorld._steps_all > self._steps_here):
            _wh = float(self._won_here) / float(self._steps_here)
            _we = float(ARCWorld._won_all) / float(ARCWorld._steps_all)
            if _wh > _we:
                self._dep_why = (
                    "stay:yield here=%.5f>env=%.5f won=%d/%d"
                    % (_wh, _we, self._won_here, self._steps_here))
                ARCWorld._yield_stays += 1
                _yield_hold = True
        # (a) MARGINAL VALUE THEOREM -- this patch pays less than the
        #     environment does.  No constant; cannot ratchet.  Bootstraps
        #     itself: with one patch visited the rates are equal and he
        #     stays, which is correct because that patch IS the environment.
        # SUPPRESSED WHILE THE PATCH IS PAYING.  Not the whole
        # predicate -- rules (b) and (c) below still apply, so he
        # always keeps a way out and cannot be caged here however the
        # yield arithmetic comes out.
        if (not _yield_hold and self._steps_here > 0
                and ARCWorld._steps_all > self._steps_here):
            here = float(self._novel_here) / float(self._steps_here)
            env = float(ARCWorld._novel_all) / float(ARCWorld._steps_all)
            if here < env:
                self._dep_why = 'a:mvt here=%.4f<env=%.4f nov=%d/%d' % (
                    here, env, self._novel_here, self._steps_here)
                return True
        # (d) THE SAME MARGINAL VALUE TEST, ON PROGRESS NOT NOVELTY.
        #     (a)-(c) all ask "have I seen anything NEW here".  A game
        #     can keep minting novel glances while yielding nothing.
        #     MEASURED 2026-08-25: long attempts advance at 0.0115 per
        #     frame against 0.1220 for short ones, with runs of 7,687
        #     frames and no new high-water mark of irreversible change.
        #     Same form as (a): no constant, cannot ratchet, and with
        #     one patch sampled the rates are equal so he stays.
        if _LEVELHOLD_ON():
            # (e) LEVEL DROUGHT (2026-09-07): more record-less lives on
            # THIS LEVEL than the longest drought he ever came back
            # from here (floor: one retry).  Records = win / new depth
            # mark / better picture -- see _level_life_end.
            try:
                _k = (str(self.game_id), int(self._levels))
                _dry = int(ARCWorld._lvl_dry.get(_k, 0))
                _gm = int(ARCWorld._lvl_gap_max.get(_k, 0))
                if _dry > max(1, _gm):
                    self._dep_why = 'e:level_dry dry=%d>max(1,%d)' % (_dry, _gm)
                    ARCWorld._lvl_dry_ejects += 1
                    return True
            except Exception:
                ARCWorld._lvl_errors += 1
        if (_STALLCUT_ON() and self._steps_here > 0
                and ARCWorld._steps_all > self._steps_here
                and getattr(ARCWorld, "_prog_all", 0) > 0):
            _ph = (float(getattr(self, "_prog_here", 0))
                   / float(self._steps_here))
            _pe = (float(getattr(ARCWorld, "_prog_all", 0))
                   / float(ARCWorld._steps_all))
            # ENOUGH EVIDENCE TO JUDGE.  `_prog_here` is 0 at the start
            # of every visit, so an unguarded comparison says LEAVE on
            # the ABSENCE of evidence -- the rotation-storm mechanism.
            # MEASURED over 36 visits in 24 h: 40.3% contain no progress
            # fire at all and 55.6% are shorter than the 285 steps in
            # which one is expected at his rate of 3.5 per 1,000.  So
            # require that the visit has run long enough for a fire to
            # have been EXPECTED.  His own rate sets the bar; no
            # constant enters, and it cannot ratchet.
            if _pe > 0.0 and (float(self._steps_here) * _pe) < 1.0:
                _ph = _pe          # not yet judgeable: hold
            if _ph < _pe:
                self._dep_why = (
                    "d:progress here=%.4f<env=%.4f adv=%d/%d"
                    % (_ph, _pe, getattr(self, "_prog_here", 0),
                       self._steps_here))
                return True
        # (b) the original record-gap rule, KEPT so every way out he already
        #     had still works.  It ratchets (max_gap only grows), which is
        #     exactly why (a) was needed -- but it costs nothing to retain.
        if self._max_gap > 0 and self._since_novel > self._max_gap:
            self._dep_why = 'b:record_gap since=%d>max=%d' % (
                self._since_novel, self._max_gap)
            return True
        # (c) A TYPICAL DROUGHT IS ENOUGH (2026-07-29, opt-in).  (a) cannot
        #     fire until a SECOND patch has been sampled (the class counters
        #     are equal while only one game is ever stepped) and (b) RATCHETS,
        #     so every drought he survives raises the bar for leaving.  Between
        #     them he has been locked on game 1 of 25 for his whole ARC
        #     existence: patch_moves 0, graduations 0, arc_levels 0.  A MEAN
        #     does not ratchet.  Self-terminating: once other patches have been
        #     sampled, (a) has a real `env` and the true marginal value theorem
        #     governs.  Still his own history, still no constant.
        if _PATCH_ROTATE and self._gap_n >= 2.0:
            if self._since_novel > self._gap_mean:
                self._dep_why = 'c:mean_gap since=%d>mean=%.2f' % (
                    self._since_novel, self._gap_mean)
                return True
        return False

    def enter(self) -> None:
        """Re-entering a patch: forget the drought, keep the earned bar."""
        self._life_key = None      # LEVELHOLD: the next step opens a new life
        try:
            self._level_forget_drought()   # ...and the droughts of this visit are over
        except Exception:
            pass                       # stubs that borrow enter() lack the method
        # A NEW VISIT IS A NEW ATTEMPT -- FOR THE PATH TOO.  `_cur_path`
        # was cleared only on a seal or an attempt ending, so rotating away and
        # back kept the abandoned stretch and the eventual seal credited
        # its boards to a win they were never part of.  Same defect as
        # the `_att_boards` boundary.  His first sealed path came to 71
        # steps where the offline winning paths ran 3-60, median ~10.
        try:
            _k = (str(self.game_id), int(self._levels))
            ARCWorld._cur_path.pop(_k, None)
            ARCWorld._cur_seen.pop(_k, None)
            ARCWorld._att_boards.pop(str(self.game_id), None)
            ARCWorld._att_lv.pop(str(self.game_id), None)
            if _HYPOTHESIS_ON():
                # A NEW VISIT IS A NEW LIFE FOR THE HYPOTHESIS TOO (patch 44):
                # the board resets on re-entry but `_lives_here` does not move
                self._hyp_new_life = True
            if _SEARCH_ON():
                self._srch_new_life = True
            if _CLOCKLINE_ON():
                # A NEW VISIT IS A NEW LIFE FOR THE BAR LEARNER TOO.
                # `_bar_observe` ends a life only on a terminal or a
                # level change; a rotation away and back RESETS the
                # board and refills the bar to values seen in the
                # previous life, so every later tick read as `back`
                # and the cell could never decide (measured on the
                # recent corpus: 1,291 of 4,827 resets are re-entries;
                # under the daemon's boundaries bp35 L0 latched a
                # content column and 10 of 15 never-won cells never
                # decided).  Same defect, same fix as `_att_boards`.
                ARCWorld._bar_lv.pop(str(self.game_id), None)
        except Exception:
            pass
        self._won_here = 0
        # a new patch owes him a life before it can be judged dry
        self._term_here = 0
        self._since_novel = 0
        self._novel_here = 0
        # novelty as of the last terminal here; resets with
        # the counter so the baseline cannot outrun it
        self._novel_at_term = 0
        self._steps_here = 0
        # THE FOURTH "here" COUNTER.  `enter()` reset the other three
        # and forgot this one, so `depleted` rule (d) divided a
        # LIFETIME numerator by a PER-VISIT denominator and could
        # essentially never fire.  Reset it with its siblings so the
        # ratio means what the other two mean.
        self._prog_here = 0


def arc_games(sock_path=SOCK):
    """Game ids from the sidecar.

    Returns [] when the sidecar is not up, so the caller keeps the
    ladder: a sidecar failure must DEGRADE, never crash him.
    """
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(10.0)
        s.connect(sock_path)
        f = s.makefile('rwb')
        f.write((json.dumps({'cmd': 'games'}) + chr(10)).encode())
        f.flush()
        out = json.loads(f.readline())
        f.close()
        s.close()
        return list(out.get('games') or [])
    except Exception:
        return []
