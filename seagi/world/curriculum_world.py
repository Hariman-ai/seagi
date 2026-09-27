"""CurriculumWorld -- a LADDER of progressively harder worlds.

Presents the SAME single-world interface (percept / step / n_actions) to the
WorldActor, but GRADUATES to the next harder world once the agent MASTERS the
current one (mastery_threshold consecutive successes).  The agent is told
nothing: it simply keeps facing a harder PROBLEM each time it conquers one --
GoalWorld (open navigation) -> a walled maze -> the Towers-of-Hanoi planning
puzzle, each a genuinely different kind of problem.

Graduation fires goal_changed=True for that ONE step, so the actor's existing
_on_mastery (the un-farmable lifeforce credit) fires exactly once per rung.  A
FINITE ladder => finite credits => un-farmable by construction (the same
guarantee GoalWorld's goal-permutation gave, now carried by the rungs
themselves).  The current world's own goal_changed (e.g. GoalWorld permuting) is
SUPPRESSED, so the only mastery signal the actor ever sees is a real graduation.
No new agent machinery: the WorldActor is unchanged; the curriculum is just the
shape of the world it faces.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List


def _LEVELHOLD_ON():
    """Stay by LEVEL RECORDS, not by wins on the game.  /root/LEVELHOLD_ON."""
    try:
        return os.path.exists("/root/LEVELHOLD_ON")
    except Exception:
        return False


def _NOGRAD_ON():
    """A GAME IS NOT A RUNG.  /root/NOGRAD_ON: mastery graduation does not
    move him off a rotating game (a world that exposes `_won_here`)."""
    try:
        return os.path.exists("/root/NOGRAD_ON")
    except Exception:
        return False


def _YIELDPICK_ON():
    """WHERE HE GOES NEXT IS WHERE HE EXPECTS TO BE PAID.  /root/YIELDPICK_ON:
    the next game is drawn in proportion to its expected yield instead of
    being the game with the fewest steps."""
    try:
        return os.path.exists("/root/YIELDPICK_ON")
    except Exception:
        return False


def _OFFICIALYIELD_ON():
    """WHAT PAYS IS THE OFFICIAL SCORE (2026-09-12).  /root/OFFICIALYIELD_ON:
    the yield a game is expected to pay counts OFFICIAL levels cleared there
    (ARC Prize's record of his best run, /root/official_best.json) instead of
    his 'wins' counter, which counts every L0 re-clear, weighted by what the
    game can still pay (100 minus its official score); the +1 prior stays for
    every game, so a capped game keeps the mass of an unvisited one.  Needs
    YIELDPICK_ON."""
    try:
        return os.path.exists("/root/OFFICIALYIELD_ON")
    except Exception:
        return False


_OFFICIAL_BEST = "/root/official_best.json"


def _SEARCH_ON():
    """A SEARCH STILL FINDING NEW STATES HOLDS HIM.  /root/SEARCH_ON: at a
    terminal, while the life that just ended reached board states never
    seen here before and somewhere untried is still reachable from the
    level's root, he stays; a bad feeling does not release that hold.
    It ends by itself: a life that finds nothing new releases him."""
    try:
        return os.path.exists("/root/SEARCH_ON")
    except Exception:
        return False


def _SEARCHLIFE_ON():
    """THE SEARCH HOLD BUYS A LIFE, NOT A STEP.  /root/SEARCHLIFE_ON (needs
    SEARCH_ON).  MEASURED 2026-09-15 (13 h after patch 45 went live): 6 of
    29 search holds were followed within a second by a DEPLETED move on the
    same game, because the hold is asked only on the terminal step while
    the DEPLETED exit runs on every step, and `depleted()` was still True
    from the terminal.  Under this gate the hold decided at a terminal is
    latched on the curriculum for the life it bought -- keyed by the game's
    id -- and the DEPLETED exit reads it on every step of that life.  It
    ends by itself at the next terminal (where the search is asked again),
    on any change of game, and the moment the world stops answering (the
    dead-world fail-open is untouched).  Gate absent: byte-identical."""
    try:
        return os.path.exists("/root/SEARCHLIFE_ON")
    except Exception:
        return False


_SWEEP_GATE = "/root/SWEEP_ON"
_SWEEP_STATE = "/home/seagi/sweep_state.json"


def _SWEEP_ON():
    """THE FULL SWEEP (2026-09-25, user's order).  /root/SWEEP_ON: every ARC
    game gets ONE life on the current scorecard before any hold, organ or
    yield draw applies, so the card carries all 25 games (ARC's card score
    is the mean over the games PLAYED on it -- measured on card 0bf70e60:
    6 games, 8.02).  Once every game has had its life the sweep is done and
    `_allocate` runs exactly as before for the rest of the card.  The
    visited set lives in _SWEEP_STATE (seagi-writable); it resets when the
    gate file is newer than the state, so re-touching the gate starts a
    new sweep.  Gate absent: byte-identical."""
    try:
        return os.path.exists(_SWEEP_GATE)
    except Exception:
        return False


def _sweep_load():
    import json as _json
    import time as _time
    try:
        _gm = float(os.path.getmtime(_SWEEP_GATE))
    except Exception:
        _gm = 0.0
    try:
        _st = _json.load(open(_SWEEP_STATE))
        if (isinstance(_st, dict) and isinstance(_st.get('visited'), dict)
                and float(_st.get('started', 0.0)) >= _gm - 1.0):
            return _st
    except Exception:
        pass
    return {'started': _time.time(), 'visited': {}, 'done': None, 'lives': 0}


def _sweep_save(_st):
    import json as _json
    try:
        _tmp = _SWEEP_STATE + '.tmp'
        with open(_tmp, 'w') as _f:
            _json.dump(_st, _f, sort_keys=True)
        os.replace(_tmp, _SWEEP_STATE)
    except Exception:
        pass


def _ALLOC_ON():
    """ONE ALLOCATION DECISION.  /root/ALLOC_ON.  MEASURED 2026-09-15 over
    715 terminals in 24 h: nine stacked stay/leave rules held him on cd82
    -- official score 100, nothing left to pay -- for 322 of 328
    terminals (10 h in one stretch) and moved him mid-life 113 times,
    six of those off a search the organ was still running.  Under this
    gate `_allocate` is the only thing that decides where his next life
    goes: commitment, then an organ testing here, then official headroom,
    then his own record history released by a bad feeling, else the
    official-yield draw.  No mid-life exit but a dead world.  Replayed on
    that day: 83 stays; outside cd82 it differs on 17 of 387 terminals.
    Gate absent: the old rules run untouched, byte-identical."""
    try:
        return os.path.exists("/root/ALLOC_ON")
    except Exception:
        return False


def _HYPOTHESIS_ON():
    """A HYPOTHESIS UNDER TEST HOLDS HIM.  /root/HYPOTHESIS_ON: at a
    terminal, while the world says a candidate goal is still being tested
    here (arc_world `hyp_hold`: the life that just ended brought a
    candidate nearer than any life before -- or was the first of the
    process -- and somewhere untried is reachable), he stays; a bad
    feeling does not release that hold.  A known win does not hold by
    itself: it is walked from the first step of the next visit.  Offline the first tu93 clear took four lives; live
    FELTLEAVE gave him one."""
    try:
        return os.path.exists("/root/HYPOTHESIS_ON")
    except Exception:
        return False


def _FELTLEAVE_ON():
    """A PLACE THAT FEELS BAD CANNOT HOLD HIM.  /root/FELTLEAVE_ON: at a
    terminal, a hold that is not a commitment is released when the world
    says this place feels worse than his environment (arc_world
    `felt_bad_here`)."""
    try:
        return os.path.exists("/root/FELTLEAVE_ON")
    except Exception:
        return False


# One source of draws so a test can seed it.
_PICK_RNG = random.Random()

# PROGRESS IS LEVEL COMPLETIONS (2026-07-29, user's ruling), OPT-IN.
# The rotation rule lives in this
# module, and a flag read from another module's namespace is a NameError
# (it was -- 8 curriculum/game-integration tests, caught by the gate).
_ARC_ROTATE = os.environ.get(
    'SEAGI_ARCROTATE', '').strip().lower() in ('1', 'true', 'yes', 'on')


class CurriculumWorld:

    def __init__(self, worlds: List[Any], mastery_threshold: int = 5,
                 escalator=None, commitment_provider=None,
                 commitment_met=None, commitment_lapsed=None,
                 commitment_baseline=None):
        # TASK COMMITMENT (2026-08-10, user: *"a task given like go play the
        # game until you succeed should also make him go and play"*).
        # `commitment_provider() -> game_id or None` is the task he has been
        # ASKED to do; `commitment_met(game_id)` is called once he has done
        # it.  Both optional and getattr-guarded, so every world that does not
        # wire them is byte-identical.  This is the EXTRINSIC half of
        # persistence: engagement (intrinsic) already holds him while a patch
        # is paying, and a commitment holds him through a patch that has
        # stopped paying, because he said he would.
        if not worlds:
            raise ValueError('CurriculumWorld needs at least one world')
        self._commitment_provider = commitment_provider
        self._commitment_met = commitment_met
        self._commitment_lapsed = commitment_lapsed
        # Lives already spent on the target when the task was given, so the
        # budget measures THIS attempt and not his whole history there.
        self._commitment_baseline = commitment_baseline
        self.held_by_commitment = 0
        self.commitments_met = 0
        self.commitments_lapsed = 0
        self._worlds = list(worlds)
        self._idx = 0
        self.mastery_threshold = int(mastery_threshold)
        self._consec = 0
        self.graduations = 0
        self.escalations = 0
        # moves made because a patch stopped yielding.
        self.patch_moves = 0
        self.patch_move_errors = 0
        # escalator(top_world) -> a HARDER version of the last rung.  When given,
        # the top rung never ends: mastering it escalates its difficulty (e.g.
        # Hanoi n -> n+1, whose solution is twice as long).  None = the top rung
        # is fixed (mastered once and then just re-solved).
        self._escalator = escalator

    @property
    def _commit_lives0(self) -> int:
        """Lives already spent on the target when the task was given."""
        if self._commitment_baseline is None:
            return 0
        try:
            return int(self._commitment_baseline() or 0)
        except Exception:
            return 0

    @property
    def world(self) -> Any:
        return self._worlds[self._idx]

    @property
    def n_actions(self) -> int:
        return int(getattr(self.world, 'n_actions', 4))

    def percept(self) -> Dict[str, object]:
        return self.world.percept()

    def step(self, action: int) -> Dict[str, object]:
        r = self.world.step(action)
        graduated = False
        if r.get('success'):
            self._consec += 1
            if self._consec >= self.mastery_threshold:
                # A GAME IS NOT A RUNG (2026-09-10).  MEASURED on pid
                # 1078631: 14 cd82 visits cleared L0..L4; on the 12 with
                # no lost life on the way up, the FIFTH consecutive
                # `success` reached `mastery_threshold` and this branch
                # bumped `_idx` 1->2 in the same second as `levels=4->5`
                # -- no log line -- landing him on ft09, judged depleted
                # at once, then r11l.  `graduations` read 12.  He played
                # L5 ONLY on the 2 visits where he had lost a life (the
                # `timed_out` reset below).  Same class as the 07-30
                # depletion bug: winning silently ejected him from the
                # game he won on.  The game rungs are INDEPENDENT worlds,
                # not a difficulty ladder; a clean run to a virgin level
                # is the best reason to stay, and LEVELHOLD already
                # governs leaving.  Ladder worlds (no `_won_here`) are
                # byte-identical; the ARC world raises `goal_changed`
                # itself on EVERY level clear (arc_world `_new_depth`,
                # 2026-08-11 "confirmation earns"), so the OR at the end
                # of step() was already true: zero credit is lost.
                if _NOGRAD_ON() and hasattr(self.world, '_won_here'):
                    import sys as _s
                    _s.stderr.write(
                        '[arcrotate] MASTERY game=%s consec=%d -> STAY '
                        '(NOGRAD: a game is not a rung)\n'
                        % (getattr(self.world, 'game_id', '?'),
                           self._consec))
                    _s.stderr.flush()
                    self._consec = 0
                elif self._idx < len(self._worlds) - 1:
                    # MASTERED this rung -> climb to the next, harder one.
                    self._idx += 1
                    self._consec = 0
                    self.graduations += 1
                    graduated = True
                elif self._escalator is not None:
                    # TOP rung mastered -> ESCALATE its difficulty (e.g. Hanoi
                    # n -> n+1) so the ladder never ends.  Each step is a
                    # genuinely new, harder task => creditable + un-farmable
                    # (cannot re-bank an easier solve).
                    self._worlds[-1] = self._escalator(self._worlds[-1])
                    self._consec = 0
                    self.escalations += 1
                    graduated = True
        elif r.get('timed_out'):
            self._consec = 0
        # A DEPLETED PATCH RELEASES HIM (2026-07-27).  Mastery graduation
        # above is untouched; this is a SECOND, co-present way to change
        # patch, for worlds that are INDEPENDENT rather than a difficulty
        # ladder (the game rungs).  Worlds that do not report `depleted`
        # -- Hanoi, mazes, quests -- are unaffected by construction.
        # A LIFE THAT COMPLETED NOTHING MOVES HIM ON (2026-07-29, user's
        # ruling: PROGRESS IS LEVEL COMPLETIONS).  The clock is DEATH -- the
        # world's own terminal -- so no constant enters and there is at most one
        # rotation per life, against the 150-per-150 thrash that percept-novelty
        # depletion produced.  Target is the LEAST-VISITED patch by steps EVER
        # spent there, i.e. the count-based novelty he already uses, at patch
        # grain.  A game that yields completions KEEPS him, so he settles where
        # he can make progress; one that yields nothing releases him after a
        # single life.  Co-present with mastery graduation and the incumbent
        # `depleted` check below, both untouched; worlds that do not expose
        # `_won_here` are unaffected by construction.
        # STAY WHERE HE HAS BEEN PAID (2026-07-30).  Measured: he completed
        # level 1 on vc33 (27 actions) and sp80 (23 vs baseline 39) and was
        # rotated off BOTH after about one life, mid-way into level 2 -- while
        # the score diluted 0.2646 -> 0.2001 as `total_levels` grew with every
        # new environment visited.  Depth scores; breadth does not.  A game
        # holds him while it keeps paying and releases him when it stops, with
        # the grace period set by how many lives its LAST level took -- his own
        # history, no constant.
        _hold = False
        # A PROMISE OUTRANKS A PATCH.  Checked FIRST and independent of
        # `_ARC_ROTATE`/`_won_ever`, because a task he was given is not a
        # property of the game -- it is something he is holding.  "Until you
        # succeed" means the commitment ends when it is MET, not when the
        # game stops being interesting; that is the whole difference between
        # an intention and an appetite.
        _commit = None
        if self._commitment_provider is not None:
            try:
                _commit = self._commitment_provider()
            except Exception:
                _commit = None
        if _commit and _commit == getattr(self.world, 'game_id', None):
            # PERSISTENCE, NOT INDEFINITELY (2026-08-10, user: *"he should
            # explore and try to succeed. persistence yes, but not
            # indefinitely"*).  An unbounded commitment pointed at a game he
            # cannot win would pin him there forever.
            #
            # The budget is denominated in PAYMENTS, never attempts.  This
            # file already records what happens otherwise: a budget built
            # from `_lives_at_last_win` let FLAILING buy patience (sp80 won
            # after 91 lives of thrashing, which then bought 68 more dry
            # ones).  So the stretch a promise buys is his own total wins --
            # a quantity only success can raise -- and the `max(1, ...)`
            # form is the one this file already uses for grace.
            _spent = (int(getattr(self.world, '_lives_here', 0))
                      - int(self._commit_lives0 or 0))
            _budget = max(1, sum(int(getattr(_w, '_won_ever', 0))
                                 for _w in self._worlds))
            if int(getattr(self.world, '_won_ever', 0)) > 0:
                # DONE.  He was asked to succeed here and he has.  Releasing
                # it is what makes it a commitment rather than a cage -- and
                # it must be released, or "until you succeed" would pin him
                # on a game he has already beaten forever.
                self.commitments_met += 1
                if self._commitment_met is not None:
                    try:
                        self._commitment_met(_commit)
                    except Exception:
                        pass
            elif _spent > _budget:
                # LAPSED.  He tried for longer than every level he has ever
                # won put together and it did not come.  Trying is released,
                # not punished -- he goes back to being led by what is
                # actually paying, which is the whole point of the engagement
                # hold underneath this.
                self.commitments_lapsed += 1
                if self._commitment_lapsed is not None:
                    try:
                        self._commitment_lapsed(_commit)
                    except Exception:
                        pass
            else:
                _hold = True
                self.held_by_commitment += 1
        _hold_commit = bool(_hold)      # a promise outranks a feeling (FELTLEAVE)
        _hold_hyp = False
        _srch_latched = False           # SEARCHLIFE: this step is held by the latch
        # ONE ALLOCATION DECISION (2026-09-15, patch 46).  Everything
        # below this line -- level hold, engagement hold, hypothesis and
        # search holds, the SEARCHLIFE latch, FELTLEAVE, the terminal
        # move, the yield-stay inside depleted() and the mid-life
        # DEPLETED exit -- is the pile that held him 10 h on a capped
        # game and cut the search's lives.  Under the gate none of it
        # runs; `_allocate` decides, once, at the terminal.
        if _ARC_ROTATE and hasattr(self.world, '_won_ever') and _ALLOC_ON():
            r = self._allocate(r, _hold_commit)
            r['goal_changed'] = bool(graduated) or bool(r.get('goal_changed'))
            return r
        if _ARC_ROTATE and hasattr(self.world, '_won_ever'):
            _we = int(getattr(self.world, '_won_ever', 0))
            _lh = int(getattr(self.world, '_lives_here', 0))
            _lw = int(getattr(self.world, '_lives_at_last_win', 0))
            # GRACE COUNTS PAYMENTS, NOT FLAILING (2026-07-30).  MEASURED:
            # sp80 `won_ever=4 lives=159 at_win=91` -> grace 91, so he was held
            # for 68 dry lives and 2,566 actions on a level-2 he cannot reach
            # (44x baseline), because he had FLAILED for 91 lives before the
            # win.  The old budget `max(1, _lw)` grows with `_lives_here`, i.e.
            # with failure, and `paid_from_dict`'s max() merge ratchets it
            # across restarts.  Denominating the budget in LEVELS ACTUALLY PAID
            # keeps it his own history while making it impossible for failure to
            # buy patience.  `_lw` remains the dry-spell ORIGIN, which is right.
            if _LEVELHOLD_ON() and hasattr(self.world, 'level_hold'):
                # LEVEL RECORDS BUY PATIENCE (2026-09-07).  MEASURED on
                # build 7: 96% of a day's steps went to nine visits that
                # each opened with a cheap level-0 win and then sat 11-47
                # lives on a level-1 he never cleared -- the term below
                # held him for up to `won_ever` lives, so every cheap win
                # lengthened the next hold by one; cd82 L2, where he knows
                # most, got ONE life per visit.  Now: stay while the
                # drought of record-less lives on THIS LEVEL is within the
                # longest he ever came back from here; depleted() rule
                # (e) ejects past it.  The old term is not consulted.
                if self.world.level_hold():
                    _hold = True
                    self.held_by_level = getattr(self, 'held_by_level', 0) + 1
            elif _we > 0 and (_lh - _lw) <= max(1, _we):
                _hold = True
            # ENGAGEMENT BUYS PATIENCE (2026-08-10, user: *"he abandons any
            # game he's never won after a single loss. that should not
            # happen"*).  The rule above buys patience ONLY with a prior win,
            # which is rich-get-richer and lands exactly where it hurts: 21 of
            # 25 environments have never been won, so those are the ones he
            # can never earn patience for.  MEASURED: ~25 terminal ejections
            # an hour, each one moving him off a game he had never cracked,
            # however well it was going when it ended.
            #
            # A forager does not need to have previously found food in a patch
            # to keep working one that is still paying (Charnov); what governs
            # is the CURRENT return rate.  He already computes that -- it is
            # `depleted()` rule (a), his novelty rate here against his
            # environment-wide rate -- and the hold rule never asked it.  So
            # ask it: he stays while he is still learning here and leaves when
            # his own marginal-value rule says the patch is dry.
            #
            # Self-terminating and no constant: every existing exit still
            # governs (MVT, record gap, mean gap), and the branch is
            # unreachable unless _ARC_ROTATE is on and the world keeps a win
            # record, so Hanoi/mazes/quests are byte-identical.
            # ENGAGEMENT MUST BE POSITIVE EVIDENCE, NOT THE ABSENCE OF ITS
            # DENIAL (2026-08-11).  This read `not depleted`, which is
            # vacuously TRUE at bootstrap and TRAPPED HIM: measured on
            # sb26 -- `since_novel 1032` of `steps 1032` (nothing new, ever),
            # `max_gap 0` so rule (b) cannot fire, `gap_mean 0.0` so (c)
            # cannot, and (a) compares his rate here against an environment
            # average that IS this game once he stops sampling others.  So
            # `depleted()` could never return True, the hold never released,
            # and he sat for 206 deaths on a game he had never won.
            # `_novel_here` is what he has actually FOUND this visit, and
            # `enter()` resets it to 0, so the hold must be EARNED on arrival
            # and a patch that teaches him nothing cannot buy patience at
            # all.  `not depleted` still governs afterwards, once his own
            # gap history exists to make it meaningful.
            elif (int(getattr(self.world, '_novel_here', 0)) > 0
                    and not bool(getattr(self.world, 'depleted', False))):
                _hold = True
                self.held_by_engagement = getattr(
                    self, 'held_by_engagement', 0) + 1
                # SHADOW, no effect on _hold: how often a hold keyed on
                # novelty SINCE THE LAST TERMINAL would have fired.  The
                # live rule uses the CUMULATIVE counter, which cannot
                # release once anything has ever been found here.
                # The gap between these two counters is the release
                # rate the redesign needs and nobody has measured.
                if (int(getattr(self.world, '_novel_here', 0))
                        > int(getattr(self.world, '_novel_at_term', 0))):
                    self.would_hold_recent = getattr(
                        self, 'would_hold_recent', 0) + 1
        # A HYPOTHESIS UNDER TEST HOLDS HIM (2026-09-14, patch 44).  Every
        # hold above asks about novelty, records, wins or feeling; none
        # asks whether he is in the middle of testing a candidate goal.
        # MEASURED offline: the first tu93 clear needs ~4 lives of
        # mapping; live he got one life per visit for 990 lives.  The
        # world answers False on any doubt, and the hold ends by itself
        # when a life brings no candidate nearer than ever before or there
        # is nowhere untried left (wa30 offline: held 2 of 25 terminals).
        if (_HYPOTHESIS_ON() and (r.get('done') or r.get('timed_out'))):
            try:
                _hh = getattr(self.world, 'hyp_hold', None)
                if _hh is not None and bool(_hh()):
                    _hold = True
                    _hold_hyp = True
                    self.held_by_hypothesis = int(getattr(self, 'held_by_hypothesis', 0)) + 1
            except Exception:
                pass
        # A SEARCH STILL FINDING NEW STATES HOLDS HIM (2026-09-15, patch 45).
        # Offline the first s5i5 clear needed ~3,300 steps of systematic
        # sweep; a hold that asks only about records or feeling would have
        # released him after one life.  Same standing as the hypothesis
        # hold: a bad feeling does not release it; nothing new does.
        # THE HOLD BUYS A LIFE, NOT A STEP (2026-09-15, SEARCHLIFE).
        # MEASURED 13 h after patch 45: 6 of 29 search holds were
        # followed within a second by `DEPLETED -> MOVE` on the same
        # game -- the search block runs only on the terminal step, the
        # DEPLETED branch below runs on every step, and `depleted()`
        # was still True from the terminal.  The organ cannot be
        # re-asked mid-life (`begin_life` zeroes what it reads), so
        # the decision made at a terminal is latched for the life it
        # bought and spent at the next terminal, where it is made again.
        # Every terminal spends it, whatever the search gate says, so a
        # gate removed mid-life cannot leave a stale latch behind.
        if _SEARCHLIFE_ON() and (r.get('done') or r.get('timed_out')):
            self._srch_life_game = None
        if (_SEARCH_ON() and (r.get('done') or r.get('timed_out'))):
            try:
                _sh = getattr(self.world, 'search_hold', None)
                if _sh is not None and bool(_sh()):
                    _hold = True
                    _hold_hyp = True
                    self.held_by_search = int(getattr(self, 'held_by_search', 0)) + 1
                    if _SEARCHLIFE_ON():
                        _gid = getattr(self.world, 'game_id', None)
                        if _gid is not None:
                            self._srch_life_game = _gid
                            self.search_latches = int(getattr(self, 'search_latches', 0)) + 1
            except Exception:
                pass
        elif _SEARCH_ON() and _SEARCHLIFE_ON():
            # Mid-life on the game the search held him on: the hold
            # stands.  Never on a world that has stopped answering (it
            # would never reach the terminal that spends this, and the
            # dead-world fail-open below must keep working), and a
            # different game means he has already moved: the latch is
            # spent.  Nothing here reads the organ.
            _lg = getattr(self, '_srch_life_game', None)
            if _lg is not None:
                if (_lg == getattr(self.world, 'game_id', None)
                        and getattr(self.world, '_live', True)):
                    _hold = True
                    _hold_hyp = True
                    _srch_latched = True
                    self.held_by_search_life = int(getattr(self, 'held_by_search_life', 0)) + 1
                else:
                    self._srch_life_game = None
        # A PLACE THAT FEELS BAD CANNOT HOLD HIM (2026-09-11, FELTLEAVE).
        # USER DOCTRINE 2026-08-29: bad feeling is a DEMAND to change
        # something.  Every hold above asks about novelty, records or
        # wins; none asks how it feels here, and MEASURED 2026-09-11 the
        # answer separates his places cleanly (paying patches +0.74..+1.0,
        # the sinks he sits in -0.93).  So at a terminal a hold that is
        # not a commitment yields to a place that is below good
        # (arc_world felt_bad_here: felt_here < _FELT_LOW, the felt
        # steer's own rule).  RELEASE ONLY: it never ejects mid-life, never
        # extends a hold, and the world answers False on any doubt, so
        # with the gate off -- or for worlds without the method -- this
        # is byte-identical.  At most one move per life, like every
        # terminal move.
        if (_hold and not _hold_commit and not _hold_hyp and _FELTLEAVE_ON()
                and (r.get('done') or r.get('timed_out'))):
            _bad = False
            try:
                _fb = getattr(self.world, 'felt_bad_here', None)
                _bad = bool(_fb()) if _fb is not None else False
            except Exception:
                _bad = False
            if _bad:
                _hold = False
                self.felt_leaves = int(getattr(self, 'felt_leaves', 0)) + 1
                try:
                    import sys as _s
                    _fh = getattr(self.world, 'felt_here', None)
                    _fe = getattr(self.world, 'felt_env', None)
                    _hv = _fh() if _fh is not None else None
                    _ev = _fe() if _fe is not None else None
                    _s.stderr.write(
                        '[arcrotate] FELTLEAVE game=%s lvl=%s here=%s env=%s '
                        '-> released\n'
                        % (getattr(self.world, 'game_id', '?'),
                           getattr(self.world, '_levels', '?'),
                           ('%.3f' % _hv) if _hv is not None else '?',
                           ('%.3f' % _ev) if _ev is not None else '?'))
                    _s.stderr.flush()
                except Exception:
                    pass
        if (_ARC_ROTATE and len(self._worlds) > 1
                and (r.get('done') or r.get('timed_out')) and _hold):
            import sys as _s
            _s.stderr.write(
                '[arcrotate] TERM game=%s won_ever=%s lives=%s at_win=%s '
                'lvl=%s steps=%s at_win_steps=%s why=%s '
                '-> HELD (still paying)\n'
                % (getattr(self.world, 'game_id', '?'),
                   getattr(self.world, '_won_ever', '?'),
                   getattr(self.world, '_lives_here', '?'),
                   getattr(self.world, '_lives_at_last_win', '?'),
                   getattr(self.world, '_levels', '?'),
                   getattr(self.world, '_steps_last_episode', '?'),
                   getattr(self.world, '_steps_at_win', '?'),
                   getattr(self.world, '_dep_why', '') or 'none'))
            _s.stderr.flush()
        #  guard restored: without it EVERY world rotated on a
        # terminal, including the non-rotating ladders, which broke
        # test_failure_resets_consecutive_count.  Worlds that do not keep a
        #  record are out of scope by construction.
        # Record novelty as of THIS terminal, held or moved, so the next
        # life can be judged against it.  Read-only bookkeeping: nothing
        # consumes it yet.  Scoped to rotating worlds per the invariant below.
        if ((r.get('done') or r.get('timed_out'))
                and hasattr(self.world, '_won_ever')):
            try:
                self.world._novel_at_term = int(
                    getattr(self.world, '_novel_here', 0))
                # A LIFE HAS BEEN PLAYED HERE.  Counted on EVERY
                # terminal, held or moved, because what it licenses is
                # "this patch has been tried", not "he left".
                self.world._term_here = int(
                    getattr(self.world, '_term_here', 0)) + 1
            except Exception:
                pass
        if (_ARC_ROTATE and len(self._worlds) > 1
                and (r.get('done') or r.get('timed_out'))
                and hasattr(self.world, '_won_ever')
                and not _hold):
            _best, _bi = None, self._idx
            # SWITCHING GAMES IS FREE (2026-09-10, user: "he should not get
            # punished between games for moving from one game to another").
            # MEASURED on pid 1078631 (17.5 h, 188 wins): the least-steps
            # target below LEVELS step counts across all 25 games (every
            # game got 1,600-3,100 steps), so 61% of his steps went to the
            # 17 games that have never paid; cd82 got 2.7% of visits (14)
            # for 70 of the 188 wins; bp35 (0 wins ever, dies in ~20
            # steps) got 69 visits, the most of any game, because short
            # lives keep its count lowest.  Every step spent on a game
            # bought ~24 steps of exile: a productive long visit was
            # punished and a fast death rewarded.  Under the gate he goes
            # where he expects to be paid; the old rule stays as the
            # fallback while he has no win on record anywhere.
            _yp = self._yield_pick() if _YIELDPICK_ON() else None
            if _yp is not None:
                _bi = _yp
            else:
                for _j, _w in enumerate(self._worlds):
                    if _j == self._idx:
                        continue
                    _sv = int(getattr(_w, '_steps_ever', 0))
                    if _best is None or _sv < _best:
                        _best, _bi = _sv, _j
            import sys as _s
            _s.stderr.write(
                '[arcrotate] TERM game=%s won_here=%s -> %s (idx %s->%s)\n'
                % (getattr(self.world, 'game_id', '?'),
                   getattr(self.world, '_won_here', '?'),
                   'MOVE' if _bi != self._idx else 'stay',
                   self._idx, _bi))
            _s.stderr.flush()
            if _bi != self._idx:
                self._idx = _bi
                self._consec = 0
                self.patch_moves += 1
                try:
                    self.world.enter()
                except Exception:
                    self.patch_move_errors += 1
        # DEPLETION MUST ASK THE SAME QUESTION AS DEATH (2026-07-30).
        # MEASURED: he completed a level on sp80 at 03:44:34 and 64 s later was
        # on ar25 -- idx 13->14, i.e. exactly this branch's `(idx+1) % n`, with
        # NO terminal and NO log line, because this was the ONLY patch-change
        # path that never consulted `_hold`.  Stay-where-paid lived solely in
        # the terminal branch, so WINNING A LEVEL silently ejected him from the
        # game he won on; both wins ever (vc33, sp80) carry that signature.
        # `_hold` is initialised False unconditionally and only set True under
        # `_ARC_ROTATE and hasattr(self.world, '_won_ever')`, so with the gate
        # off -- and for Hanoi/mazes/quests -- this branch is byte-identical.
        # The log line is part of the fix: silence is what hid this.
        # HE MAY NOT LEAVE BEFORE HE HAS PLAYED ONE LIFE HERE
        # (2026-08-14).  `enter()` zeroes `_novel_here`/`_steps_here`, so
        # on the next step rule (a) reads `here = 0/1 = 0` -- below ANY
        # positive env -- while the engagement hold asks `_novel_here > 0`
        # and gets False.  One counter, one instant, makes "leave" true
        # and "stay" false together: 17,418 ejections/day on step 2, and
        # `TERM -> MOVE` exactly 0 for three days.  Measured: 99.91% of
        # ejections happen in a visit that never reached a terminal, while
        # the good regime rotated at terminals 525-to-6.  Nothing is
        # disarmed -- (a), (b) and (c) all still govern, from the first
        # terminal on, by which point the rate has a real denominator.
        # FAIL OPEN ON A DEAD WORLD.  `_call` returns ok=0 and `_absorb`
        # returns early, so `_state` is sticky and `done` can never become
        # true -- a game stuck not-ok would produce no terminal, leaving
        # `_term_here` at 0 and the exit unreachable forever.  A patch
        # that is not answering cannot have been "played once".
        _dep = (len(self._worlds) > 1
                and (int(getattr(self.world, '_term_here', 0)) >= 1
                     or not getattr(self.world, '_live', True))
                and getattr(self.world, 'depleted', False))
        # LOG THE EDGES, NOT EVERY STEP (2026-07-30).  This branch wrote one
        # line per step while depleted-and-held -- 783 identical lines in 50
        # min.  `_dep_held_n` is ephemeral logging state on the world (never
        # persisted, getattr-defaulted), so the hold DECISION is unchanged.
        if _dep and _hold:
            if _srch_latched:
                # SEARCHLIFE: DEPLETED was True on a mid-life step and
                # the latched search hold stood in the way (other holds
                # may have too).  This is the count that judges the patch.
                self.search_life_blocks = int(getattr(self, 'search_life_blocks', 0)) + 1
            _hn = int(getattr(self.world, '_dep_held_n', 0)) + 1
            try:
                self.world._dep_held_n = _hn
            except Exception:
                _hn = 1
            if _hn == 1:
                import sys as _s
                _s.stderr.write(
                    '[arcrotate] DEPLETED game=%s won_ever=%s lives=%s '
                    'at_win=%s -> HELD (still paying)\n'
                    % (getattr(self.world, 'game_id', '?'),
                       getattr(self.world, '_won_ever', '?'),
                       getattr(self.world, '_lives_here', '?'),
                       getattr(self.world, '_lives_at_last_win', '?')))
                _s.stderr.flush()
        else:
            _hn = int(getattr(self.world, '_dep_held_n', 0))
            if _hn > 0:
                import sys as _s
                _s.stderr.write(
                    '[arcrotate] DEPLETED game=%s -> HOLD ENDED '
                    'held_steps=%d\n'
                    % (getattr(self.world, 'game_id', '?'), _hn))
                _s.stderr.flush()
                try:
                    self.world._dep_held_n = 0
                except Exception:
                    pass
        if _dep and not _hold:
            import sys as _s
            _s.stderr.write(
                '[arcrotate] DEPLETED game=%s won_ever=%s -> MOVE (idx %s->%s)\n'
                % (getattr(self.world, 'game_id', '?'),
                   getattr(self.world, '_won_ever', '?'),
                   self._idx, (self._idx + 1) % len(self._worlds)))
            _s.stderr.flush()
            self._idx = (self._idx + 1) % len(self._worlds)
            self._consec = 0
            self.patch_moves += 1
            try:
                self.world.enter()
            except Exception:
                self.patch_move_errors += 1
        # The actor's _on_mastery (lifeforce credit) keys on goal_changed.
        # Graduation still fires it (finite rungs => un-farmable).  As of
        # 2026-07-31 the inner world may ALSO raise it, and a rung world may do so
        # only on a level deeper than any it has ever reached -- monotone over
        # a finite ladder, so it meets the same un-farmability bar that the
        # blanket suppression was protecting.  OR, not overwrite: suppressing
        # it here is why 23 wins credited zero lifeforce.
        r['goal_changed'] = bool(graduated) or bool(r.get('goal_changed'))
        return r

    # ---- persistence ----

    # HIS PLACE ON THE LADDER SURVIVES A RESTART (2026-08-18).  `_idx`
    # was built to 0 in __init__ and nothing anywhere restored it, so
    # every boot put him back on rung 0 and zeroed `graduations` -- 31
    # boots in the 7 days to 2026-08-18.  He re-climbed from the bottom
    # after every deploy.
    #
    # The rung index is not the whole of his place: `_consec` is his
    # run-up to mastery, and once he is escalating the top rung it is
    # that rung's DIFFICULTY that says where he stands.

    MAX_ESCALATION_REPLAY = 64

    def to_dict(self) -> Dict[str, Any]:
        return {
            'idx': int(self._idx),
            'consec': int(self._consec),
            'graduations': int(self.graduations),
            'escalations': int(self.escalations),
            'patch_moves': int(self.patch_moves),
            'patch_move_errors': int(self.patch_move_errors),
            'held_by_engagement': int(
                getattr(self, 'held_by_engagement', 0)),
            # Persisted WITH its minuend, never without: the measurement
            # is the DIFFERENCE of these two, so an all-time minuend
            # against a process-local subtrahend is a lying diagnostic.
            'would_hold_recent': int(
                getattr(self, 'would_hold_recent', 0)),
            'held_by_level': int(getattr(self, 'held_by_level', 0)),
            'held_by_commitment': int(
                getattr(self, 'held_by_commitment', 0)),
            'commitments_met': int(getattr(self, 'commitments_met', 0)),
            'commitments_lapsed': int(
                getattr(self, 'commitments_lapsed', 0)),
            # Diagnostic only -- the ladder is rebuilt from code, so a
            # save from a longer ladder must clamp, never extend.
            'ladder_size': len(self._worlds),
        }

    def load_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return

        # THE LADDER MUST BE THE SAME LADDER (2026-08-19, adversarial
        # review).  `ladder_size` was recorded and never checked.  His
        # saved position is a 16-rung MAZE ladder; ARC is 25 games.
        # Restoring one onto the other clamps to a valid index and
        # carries graduations/_consec that describe a different world --
        # silently, which is the worst kind.
        _saved_n = state.get('ladder_size')
        if isinstance(_saved_n, int) and _saved_n != len(self._worlds):
            return

        def _int(key: str, current: int) -> int:
            try:
                return int(state[key])
            except (KeyError, TypeError, ValueError):
                return current

        # RE-ESCALATE FIRST.  Restoring the index onto an un-escalated
        # top rung would silently hand him an easier task after every
        # restart -- the ladder would look right and be wrong.  Applied
        # as a DELTA so a second load_dict is a no-op.
        esc = max(0, _int('escalations', int(self.escalations)))
        need = esc - int(self.escalations)
        applied = 0
        if self._escalator is not None and need > 0:
            for _ in range(min(need, self.MAX_ESCALATION_REPLAY)):
                try:
                    self._worlds[-1] = self._escalator(self._worlds[-1])
                    applied += 1
                except Exception:
                    break
        # Report what is IN EFFECT, not what the save claimed.  With no
        # escalator (or a replay that failed) the top rung is NOT
        # escalated, and a counter that disagrees with top_difficulty is
        # a diagnostic that lies.
        self.escalations = int(self.escalations) + applied

        idx = _int('idx', self._idx)
        self._idx = max(0, min(idx, len(self._worlds) - 1))
        self._consec = max(0, _int('consec', self._consec))
        self.graduations = _int('graduations', self.graduations)
        self.patch_moves = _int('patch_moves', self.patch_moves)
        self.patch_move_errors = _int('patch_move_errors',
                                      self.patch_move_errors)
        self.held_by_engagement = _int(
            'held_by_engagement', getattr(self, 'held_by_engagement', 0))
        self.would_hold_recent = _int(
            'would_hold_recent', getattr(self, 'would_hold_recent', 0))
        self.held_by_commitment = _int(
            'held_by_commitment', getattr(self, 'held_by_commitment', 0))
        self.commitments_met = _int(
            'commitments_met', getattr(self, 'commitments_met', 0))
        self.commitments_lapsed = _int(
            'commitments_lapsed', getattr(self, 'commitments_lapsed', 0))

    # ---- ONE ALLOCATION DECISION (patch 46) --------------------------
    def _headroom(self):
        """(100 - official score) / 100 for the current game, from ARC
        Prize's record of his best run; 1.0 when no record is readable
        (an unknown game is assumed able to pay)."""
        try:
            _rec = self._official_record()
            if not _rec:
                return 1.0
            _g = str(getattr(self.world, 'game_id', ''))[:4]
            if _g not in _rec:
                return 1.0
            _sc = float(_rec[_g][2])
            return min(1.0, max(0.0, (100.0 - _sc) / 100.0))
        except Exception:
            return 1.0

    def _alloc_move(self, why, target, head=None):
        """Move him to `target` (None: next in the ring).  One move, the
        [alloc] line, and the legacy line every instrument reads
        (sessions.py, the storm watchers) in its unchanged format."""
        _from = self._idx
        if target is None or target == self._idx:
            target = (self._idx + 1) % len(self._worlds)
        _w = self.world
        import sys as _s
        _s.stderr.write(
            '[alloc] game=%s lv=%s why=%s head=%s rec_age=%s steps=%s '
            '-> MOVE (idx %s->%s) target=%s\n'
            % (getattr(_w, 'game_id', '?'), getattr(_w, '_levels', '?'), why,
               ('%.2f' % head) if head is not None else '?',
               int(getattr(self, '_official_rec_age', -1)),
               getattr(_w, '_steps_last_episode', '?'),
               _from, target,
               str(getattr(self._worlds[target], 'game_id', '?'))[:4]))
        if why == 'dead':
            _s.stderr.write(
                '[arcrotate] DEPLETED game=%s won_ever=%s -> MOVE (idx %s->%s)\n'
                % (getattr(_w, 'game_id', '?'), getattr(_w, '_won_ever', '?'),
                   _from, target))
        else:
            _s.stderr.write(
                '[arcrotate] TERM game=%s won_here=%s -> MOVE (idx %s->%s)\n'
                % (getattr(_w, 'game_id', '?'), getattr(_w, '_won_here', '?'),
                   _from, target))
        _s.stderr.flush()
        self._idx = target
        self._consec = 0
        self.patch_moves += 1
        self.alloc_moves = int(getattr(self, 'alloc_moves', 0)) + 1
        _aw = getattr(self, 'alloc_why', None)
        if not isinstance(_aw, dict):
            _aw = self.alloc_why = {}
        _aw[why] = int(_aw.get(why, 0)) + 1
        try:
            self.world.enter()
        except Exception:
            self.patch_move_errors += 1

    def _allocate(self, r, hold_commit):
        """Where does his next life go?  Decided ONCE, at the terminal:
        commitment > an organ testing here > official headroom > his own
        record history (released by a bad feeling) > the official-yield
        draw.  Mid-life nothing moves him but a world that stopped
        answering.  Every terminal writes one [alloc] line (with the
        oracle's headroom and the record's age) and the legacy TERM line.
        The organs are asked BEFORE the terminal bookkeeping, as before,
        and each under its own gate; both are asked (the hypothesis
        organ's ratchet moves when it is asked)."""
        _w = self.world
        _term = bool(r.get('done') or r.get('timed_out'))
        if not _term:
            if not getattr(_w, '_live', True) and len(self._worlds) > 1:
                # FAIL OPEN ON A DEAD WORLD: no terminal will ever come.
                self.alloc_dead_moves = int(getattr(self, 'alloc_dead_moves', 0)) + 1
                self._alloc_move('dead', None)
            return r
        # SEARCHLIFE's latch is not consulted under this gate: spend it
        # here so a gate flip mid-life cannot revive a stale one.
        self._srch_life_game = None
        _head = self._headroom()
        _why = None
        _leave = 'dry'
        if hold_commit:
            _why = 'commit'
        else:
            _holders = []
            for _name, _gate, _ctr in (('hyp_hold', _HYPOTHESIS_ON, 'held_by_hypothesis'),
                                       ('search_hold', _SEARCH_ON, 'held_by_search')):
                if not _gate():
                    continue
                _f = getattr(_w, _name, None)
                if _f is None:
                    continue
                try:
                    if bool(_f()):
                        setattr(self, _ctr, int(getattr(self, _ctr, 0)) + 1)
                        _holders.append(_name.split('_')[0])
                except Exception:
                    pass
            if _holders:
                _why = 'organ:' + '+'.join(_holders)
            else:
                _lh = False
                try:
                    _lf = getattr(_w, 'level_hold', None)
                    _lh = bool(_lf()) if _lf is not None else False
                except Exception:
                    _lh = False
                _bad = False
                try:
                    _fb = getattr(_w, 'felt_bad_here', None)
                    _bad = bool(_fb()) if _fb is not None else False
                except Exception:
                    _bad = False
                if _head <= 0.0:
                    _leave = 'capped'
                elif _lh and not _bad:
                    _why = 'record'
                    self.held_by_level = int(getattr(self, 'held_by_level', 0)) + 1
                elif _lh and _bad:
                    # a hold he had, released by how it feels here
                    _leave = 'felt_bad'
                    self.felt_leaves = int(getattr(self, 'felt_leaves', 0)) + 1
                else:
                    _leave = 'dry'
        # A LIFE HAS BEEN PLAYED HERE -- the bookkeeping every terminal
        # did before, in the old order: after the organs were asked.
        try:
            _w._novel_at_term = int(getattr(_w, '_novel_here', 0))
            _w._term_here = int(getattr(_w, '_term_here', 0)) + 1
        except Exception:
            pass
        # THE FULL SWEEP (2026-09-25): one life on every ARC game first.
        # Outranks commitment, organs and holds while games remain; after
        # the last one it is done and everything below runs unchanged.
        if _SWEEP_ON():
            _st = _sweep_load()
            if not _st.get('done'):
                import time as _t
                _g4 = str(getattr(_w, 'game_id', '?'))[:4]
                _st['visited'][_g4] = int(_st['visited'].get(_g4, 0)) + 1
                _st['lives'] = int(_st.get('lives', 0)) + 1
                _todo = [_j for _j, _o in enumerate(self._worlds)
                         if hasattr(_o, '_won_ever')
                         and str(getattr(_o, 'game_id', '?'))[:4] not in _st['visited']]
                import sys as _s
                if _todo:
                    _sweep_save(_st)
                    self.sweep_moves = int(getattr(self, 'sweep_moves', 0)) + 1
                    _s.stderr.write('[sweep] game=%s lives=%d visited=%d/%d -> MOVE to %s\n'
                                    % (_g4, _st['lives'], len(_st['visited']),
                                       sum(1 for _o in self._worlds if hasattr(_o, '_won_ever')),
                                       str(getattr(self._worlds[_todo[0]], 'game_id', '?'))[:4]))
                    _s.stderr.flush()
                    self._alloc_move('sweep', _todo[0], _head)
                    return r
                _st['done'] = _t.time()
                _sweep_save(_st)
                _s.stderr.write('[sweep] COMPLETE: %d games, %d lives; normal allocation resumes\n'
                                % (len(_st['visited']), _st['lives']))
                _s.stderr.flush()
        if _why is not None:
            self.alloc_stays = int(getattr(self, 'alloc_stays', 0)) + 1
            _aw = getattr(self, 'alloc_why', None)
            if not isinstance(_aw, dict):
                _aw = self.alloc_why = {}
            _aw[_why] = int(_aw.get(_why, 0)) + 1
            import sys as _s
            _s.stderr.write(
                '[alloc] game=%s lv=%s why=%s head=%.2f rec_age=%s steps=%s -> STAY (lives=%s)\n'
                % (getattr(_w, 'game_id', '?'), getattr(_w, '_levels', '?'),
                   _why, _head, int(getattr(self, '_official_rec_age', -1)),
                   getattr(_w, '_steps_last_episode', '?'),
                   getattr(_w, '_lives_here', '?')))
            _s.stderr.write(
                '[arcrotate] TERM game=%s won_ever=%s lives=%s at_win=%s '
                'lvl=%s steps=%s at_win_steps=%s why=%s '
                '-> HELD (still paying)\n'
                % (getattr(_w, 'game_id', '?'), getattr(_w, '_won_ever', '?'),
                   getattr(_w, '_lives_here', '?'), getattr(_w, '_lives_at_last_win', '?'),
                   getattr(_w, '_levels', '?'), getattr(_w, '_steps_last_episode', '?'),
                   getattr(_w, '_steps_at_win', '?'), 'alloc:' + _why))
            _s.stderr.flush()
            return r
        if len(self._worlds) < 2:
            return r
        _target = self._yield_pick() if _YIELDPICK_ON() else None
        if _target is None:
            # no payout on record anywhere: the least-visited game, as before
            _best, _bi = None, self._idx
            for _j, _o in enumerate(self._worlds):
                if _j == self._idx:
                    continue
                _sv = int(getattr(_o, '_steps_ever', 0))
                if _best is None or _sv < _best:
                    _best, _bi = _sv, _j
            _target = _bi
        self._alloc_move(_leave, _target, _head)
        return r

    def _yield_pick(self):
        """Index of the next game, drawn in proportion to what he expects it
        to pay; None when he has no win on record anywhere (the caller then
        falls back to the least-steps sweep, byte-identical to before).

        Expected yield of game j = (won_ever_j + 1) / (steps_ever_here_j + C)
        with C = (sum of steps_ever_here) / (sum of won_ever) over ALL
        games, the current one included: the environment's own measured
        cost of one win.  A game he has never visited is expected to pay
        exactly the environment average, a game played for thousands of
        steps without a win is expected to pay little, and a game that pays
        keeps paying in expectation.  The +1 is the rule-of-succession
        prior (one imagined win at the measured cost); nothing is tuned by
        hand.  Both counters are his PERSISTED all-time record on that game
        (`world_paid` rows: won_ever row[0], steps_ever_here row[6]), so
        the choice survives a restart.  Reads and writes nothing of
        lifeforce: a proxy-layer choice driven by game events.  The
        current game is excluded, as under the old rule: LEVELHOLD has
        just released him from it.  Every draw is logged with the whole
        share vector so the choice can be audited from the journal.
        """
        _wr = self._yield_weights()
        if _wr is None:
            return None
        _ws, _cost, _off = _wr
        _tot = sum(_e for _j, _e in _ws)
        if _tot <= 0.0:
            # every game he could move to is capped: nothing to expect
            # anywhere, so the old least-steps sweep decides.
            return None
        _r = _PICK_RNG.random() * _tot
        _acc = 0.0
        _pick, _pe = _ws[-1]
        for _j, _e in _ws:
            _acc += _e
            if _r <= _acc:
                _pick, _pe = _j, _e
                break
        self.yield_picks = int(getattr(self, 'yield_picks', 0)) + 1
        try:
            import sys as _s
            _s.stderr.write(
                ('[arcrotate] PICK game=%s share=%.1f%% cost=%.0f '
                 'shares=%s%s\n')
                % (getattr(self._worlds[_pick], 'game_id', '?'),
                   100.0 * _pe / _tot, _cost,
                   ','.join('%s:%.1f' % (
                       str(getattr(self._worlds[_j], 'game_id', '?'))[:4],
                       100.0 * _e / _tot)
                       for _j, _e in sorted(_ws, key=lambda t: -t[1])),
                   (' off=1 rec_age=%ds' % int(getattr(self, '_official_rec_age', -1)))
                   if _off else ''))
            _s.stderr.flush()
        except Exception:
            pass
        return _pick

    def _official_record(self):
        """{game4: (levels_cleared, levels_total, official_score)} from ARC
        Prize's record of his best run per game.  The cron rewrites the file
        in place every 15 min, so a read can land on a half-written file: the
        last good read is kept and returned instead; None until one read
        succeeded.  `_official_rec_age` = seconds since the file was written,
        printed on every PICK line so a dead cron is visible in the journal."""
        try:
            import json as _json
            import time as _time
            with open(_OFFICIAL_BEST) as _f:
                _d = _json.load(_f)
            _rec = {}
            for _g, _v in (_d or {}).items():
                _v = _v or {}
                _rec[str(_g)[:4]] = (int(_v.get('levels', 0) or 0),
                                     len(_v.get('baselines') or []),
                                     float(_v.get('score', 0.0) or 0.0))
            if _rec:
                self._official_rec = _rec
                try:
                    self._official_rec_age = int(
                        _time.time() - os.path.getmtime(_OFFICIAL_BEST))
                except Exception:
                    self._official_rec_age = -1
        except Exception:
            pass
        return getattr(self, '_official_rec', None)

    def _yield_weights(self):
        """[(index, expected yield)] for every game but the current one, the
        measured cost of one payout, and whether the official record was
        used; None when he has no payout on record anywhere.

        Gate off: yield_j = (won_ever_j + 1) / (steps_ever_here_j + C),
        C = total steps / total won_ever -- unchanged since patch 36.
        OFFICIALYIELD_ON: the payout is an OFFICIAL level, not a 'win'.
        yield_j = (headroom_j * levels_j + 1) / (steps_ever_here_j + C),
        C = total steps / total official levels (the environment's measured
        cost of one official level), headroom_j = (100 - official score_j)
        / 100: what the game can still pay, read from ARC Prize's own score
        (a complete game is capped at 100).  The +1 is the same rule-of-
        succession prior as before -- one imagined level at the measured
        cost -- and it is kept for EVERY game, so no game ever has zero
        mass: a capped game is not exiled, it is expected to pay only what
        an unvisited one would.  MEASURED 2026-09-12 on his persisted
        counters: the gate-off vector gave cd82 (capped) 40% of picks and
        the 15 never-won games 0.9% combined; this vector gives cd82 ~2%,
        the 15 never-won ~50% and the 9 scored-uncapped ~48%, near-uniform,
        because every game's steps are small against the ~120k steps one
        official level has cost.  No constant is set by hand.  This is a
        stand-in for a worldview that cannot yet tell places apart (his
        chemistry reads flat across all places, 09-11), not his decision.
        """
        _ts = 0
        for _w in self._worlds:
            _ts += int(getattr(_w, '_steps_ever_here', 0) or 0)
        _rec = self._official_record() if _OFFICIALYIELD_ON() else None
        if _rec:
            _tl = 0
            for _w in self._worlds:
                _g = str(getattr(_w, 'game_id', ''))[:4]
                _tl += int(_rec.get(_g, (0, 0))[0])
            if _tl <= 0 or _ts <= 0:
                return None
            _cost = float(_ts) / float(_tl)
            _ws = []
            for _j, _w in enumerate(self._worlds):
                if _j == self._idx:
                    continue
                _g = str(getattr(_w, 'game_id', ''))[:4]
                _lv, _n, _sc = _rec.get(_g, (0, 0, 0.0))
                _head = min(1.0, max(0.0, (100.0 - _sc) / 100.0))
                _e = ((_head * _lv + 1.0)
                      / (int(getattr(_w, '_steps_ever_here', 0) or 0)
                         + _cost))
                _ws.append((_j, _e))
            if not _ws:
                return None
            return _ws, _cost, True
        _tw = 0
        for _w in self._worlds:
            _tw += int(getattr(_w, '_won_ever', 0) or 0)
        if _tw <= 0 or _ts <= 0:
            return None
        _cost = float(_ts) / float(_tw)
        _ws = []
        for _j, _w in enumerate(self._worlds):
            if _j == self._idx:
                continue
            _e = ((int(getattr(_w, '_won_ever', 0) or 0) + 1.0)
                  / (int(getattr(_w, '_steps_ever_here', 0) or 0) + _cost))
            _ws.append((_j, _e))
        if not _ws:
            return None
        return _ws, _cost, False

    def task_stats(self) -> Dict[str, object]:
        try:
            s = dict(self.world.task_stats())
        except Exception:
            s = {}
        s.update({
            'ladder_index': self._idx,
            'ladder_size': len(self._worlds),
            'ladder_world': type(self.world).__name__,
            'graduations': self.graduations,
            'escalations': self.escalations,
            'patch_moves': self.patch_moves,
            'felt_leaves': int(getattr(self, 'felt_leaves', 0)),
            # How often engagement -- not a prior win -- kept him on a game.
            # Without this the new hold branch is unmeasurable, and an
            # unmeasurable change cannot be shown to work or to have stopped.
            'held_by_engagement': getattr(self, 'held_by_engagement', 0),
            # SHADOW: what a hold keyed on novelty SINCE THE LAST
            # TERMINAL would have held.  held_by_engagement minus this
            # is how often the live CUMULATIVE rule held him on
            # evidence that was already spent.
            'would_hold_recent': getattr(self, 'would_hold_recent', 0),
            'held_by_level': getattr(self, 'held_by_level', 0),
            # Patch 45 / 45b: terminals the search (the hypothesis) held;
            # lives the latch bought; mid-life steps under a latched hold;
            # and DEPLETED steps the latch stood against (moves prevented).
            # Process counters, not persisted: they restart at 0 with him.
            'held_by_search': int(getattr(self, 'held_by_search', 0)),
            'held_by_hypothesis': int(getattr(self, 'held_by_hypothesis', 0)),
            'search_latches': int(getattr(self, 'search_latches', 0)),
            'held_by_search_life': int(getattr(self, 'held_by_search_life', 0)),
            'search_life_blocks': int(getattr(self, 'search_life_blocks', 0)),
            'search_life_game': getattr(self, '_srch_life_game', None),
            'held_by_commitment': getattr(self, 'held_by_commitment', 0),
            # Patch 46 ALLOC: the one decision, counted by its reason.
            'alloc_stays': int(getattr(self, 'alloc_stays', 0)),
            'alloc_moves': int(getattr(self, 'alloc_moves', 0)),
            'alloc_dead_moves': int(getattr(self, 'alloc_dead_moves', 0)),
            'alloc_why': dict(getattr(self, 'alloc_why', None) or {}),
            'commitments_met': getattr(self, 'commitments_met', 0),
            'commitments_lapsed': getattr(self, 'commitments_lapsed', 0),
            'commitment_budget': max(1, sum(int(getattr(_w, '_won_ever', 0))
                                            for _w in self._worlds)),
            # None when nothing is committed.  It previously reported raw
            # `_lives_here`, so with no commitment it read e.g. "budget 104,
            # spent 203" -- which looks exactly like an expired promise and
            # is a diagnostic that lies.
            'commitment_spent': (
                None if not (self._commitment_provider
                             and self._commitment_provider())
                else (int(getattr(self.world, '_lives_here', 0))
                      - self._commit_lives0)),
            'commitment': (self._commitment_provider()
                           if self._commitment_provider is not None
                           else None),
            'patch_move_errors': self.patch_move_errors,
            'top_difficulty': getattr(self._worlds[-1], 'disks',
                                      getattr(self._worlds[-1], 'size', None)),
            'consec_success': self._consec,
            'yield_picks': int(getattr(self, 'yield_picks', 0)),
            'official_yield_on': bool(_OFFICIALYIELD_ON()),
            'official_record_games': len(getattr(self, '_official_rec', None) or {}),
        })
        return s

    def __getattr__(self, name: str) -> Any:
        # Delegate any other attribute access to the CURRENT rung so callers
        # that read a world-specific field still work.  Only reached for attrs
        # not defined above; guarded against access before __init__ completes.
        try:
            worlds = self.__dict__['_worlds']
            idx = self.__dict__['_idx']
        except KeyError:
            raise AttributeError(name)
        return getattr(worlds[idx], name)
