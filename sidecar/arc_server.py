#!/usr/bin/env python3
"""ARC sidecar — owns the live ARC-AGI-3 games, speaks a tiny JSON protocol.

Run under /root/arc3_venv/bin/python.  The DAEMON must not import arc_agi:
it eagerly pulls flask, matplotlib, PIL, dotenv and requests, and the venv
carries numpy 2.5.1 against the daemon's 2.4.3.  Forcing that into a live
process is precisely the risk BUILD PRUDENTLY exists to avoid.  So the
game lives here and the daemon holds a dependency-free client.

His EYE stays daemon-side: this process ships raw frames only.  Attention
must read his own _visits, so windowing happens in his head, not here.

Protocol: newline-delimited JSON over a Unix socket.
  {"cmd":"games"}                    -> {"ok":1,"games":[...]}
  {"cmd":"open","game":"sk48","seed":0}
  {"cmd":"reset"}
  {"cmd":"step","a":<int index into `actions`>}
    -> {"ok":1,"grid":[[...]],"state":"NOT_FINISHED|WIN|GAME_OVER",
        "levels":<int>,"actions":[<int>,...],"game":"sk48"}
"""
import json, os, socket, sys, time, traceback

SOCK = '/run/arc3.sock'

import arc_agi
from arcengine.enums import GameAction

_NONRESET = [ga for ga in GameAction if ga.name != 'RESET']


class Games:
    # Give a failing card a chance to settle before concluding it is dead.
    # HOW LONG A CARD MAY KEEP FAILING BEFORE IT COUNTS AS DEAD.
    #
    # MEASURED: a restart-race transient runs ~14 s; a genuinely closed card
    # failed 1,856 consecutive times over ~50 min (~3,000 s).  600 s sits 40x
    # above the transient and 5x below the death signature -- a range, not a
    # tuned number.  The asymmetry picks the side to err on: waiting costs
    # nothing but time, while renewing wrongly DISCARDS THE BANKED SCORE (that
    # is how a healthy card carrying 1,075 actions was binned at 18:31).
    #
    # ELAPSED, NOT SLEPT.  `arc_world._call` uses `s.settimeout(10.0)`, so any
    # blocking settle longer than 10 s just makes the CALLER time out, count an
    # error and reconnect -- and this server handles one connection at a time,
    # so those retries queue behind the sleep.  `_ensure_mine` already retries
    # continuously, so it supplies the cadence and we only track how long the
    # failure has persisted.  Every open() returns promptly.
    _SETTLE_SECONDS = 600.0

    def __init__(self):
        self.arc = arc_agi.Arcade(arc_api_key=os.environ.get('ARC_KEY', ''))
        self.env = None
        self.frame = None
        self.game = None
        self.seed = 0
        self.actions = []
        # game_id -> server-side run guid, so a re-open RESUMES that run
        # instead of starting a new one at level 1 (see open()).
        self.guids = {}
        # RECORD HIS PLAY (2026-07-29).  make() takes a scorecard_id and this
        # never passed one, so nothing he has ever done was scored
        # (total_actions stayed 0).  One card for the whole process; if this
        # fails, card stays None and everything behaves exactly as before.
        self.card = None
        # KEEP THE CARD (2026-07-30).  This opened a NEW card on every restart,
        # so restarting the sidecar threw the running score away (the live card
        # held 0.5028 / 4 levels after 26 h).  Reuse it when the server still
        # knows it; any failure falls through to the original path below.
        _CARDF = '/root/arc_card.json'
        try:
            _prev = (json.load(open(_CARDF)) or {}).get('card_id')
        except Exception:
            _prev = None
        # NO CROSS-PROCESS REUSE -- REUSE IS A BLACK HOLE (2026-07-31).
        # MEASURED, with a control ruling out propagation lag:
        #     A opens card, plays 5             -> total_actions 5
        #     B (other session) reuses it, plays 10
        #     A re-reads at +0/+15/+30/+60 s    -> 5, 5, 5, 5
        #     A plays 2 more                    -> 7   (recording IS live)
        # A reused card ACCEPTS play -- reset() and step() both succeed -- and
        # records NOTHING.  Scoring is bound to the session that OPENED the
        # card, and get_scorecard 404s for everyone else, so the loss is
        # silent too.  Reuse is therefore worse than the broken liveness gate
        # it replaced: the gate wasted a card, reuse plays into one that
        # scores nothing.  A restart costs the score because the API enforces
        # it, not because of anything we can fix here.
        # The file is kept as an AUDIT RECORD of the previous card only.
        if _prev:
            sys.stderr.write('arc_server: previous card was %s -- NOT reused '
                             '(a reused card records nothing); opening a '
                             'fresh one\n' % _prev)
            sys.stderr.flush()
        for _m in (() if self.card else ('open_scorecard', 'create_scorecard')):
            try:
                _f = getattr(self.arc, _m, None)
                if _f is None:
                    continue
                self.card = _f(tags=['seagi', 'agent'])
                sys.stderr.write('arc_server: scorecard %s via %s\n'
                                 % (self.card, _m))
                sys.stderr.flush()
                break
            except Exception as _e:
                sys.stderr.write('arc_server: %s failed: %r\n' % (_m, _e))
                sys.stderr.flush()
        self._cardf = _CARDF
        # ONE renewal per dead-world episode; cleared by a real frame.
        self._renewed = False
        # When the current unbroken run of failed opens began (None = healthy).
        self._fail_since = None
        self.renewals = 0
        self._persist_card()

    def _persist_card(self):
        # SILENT NO MORE.  This used to be `except Exception: pass`, which is
        # why /root/arc_card.json never existed and the reuse path could never
        # fire.  A persistence failure is now visible in the log.
        if not self.card:
            return
        try:
            with open(self._cardf, 'w') as fh:
                json.dump({'card_id': str(self.card)}, fh)
        except Exception as _e:
            sys.stderr.write('arc_server: card persist FAILED: %r\n' % _e)
            sys.stderr.flush()

    def _renew(self):
        """The card closed underneath him -- rebuild the whole client.

        The scorecard auto-closes on idle, and the session/cookie jar that
        owned the server-side runs dies with it.  Nothing short of a new
        Arcade can talk to the API again, which is why a restart used to be
        the only cure.  Runs do not survive, so the remembered guids go too.
        """
        self.renewals += 1
        sys.stderr.write('arc_server: RENEWING client (attempt %d)\n'
                         % self.renewals)
        sys.stderr.flush()
        try:
            self.arc = arc_agi.Arcade(
                arc_api_key=os.environ.get('ARC_KEY', ''))
        except Exception as _e:
            sys.stderr.write('arc_server: Arcade rebuild failed: %r\n' % _e)
            sys.stderr.flush()
            return False
        self.card = None
        self.guids = {}
        for _m in ('open_scorecard', 'create_scorecard'):
            try:
                _f = getattr(self.arc, _m, None)
                if _f is None:
                    continue
                self.card = _f(tags=['seagi', 'agent'])
                sys.stderr.write('arc_server: renewed scorecard %s via %s\n'
                                 % (self.card, _m))
                sys.stderr.flush()
                break
            except Exception as _e:
                sys.stderr.write('arc_server: %s failed: %r\n' % (_m, _e))
                sys.stderr.flush()
        self._persist_card()
        return True

    def list_games(self):
        for attr in ('available_environments', 'get_environments'):
            try:
                v = getattr(self.arc, attr)
                v = v() if callable(v) else v
                out = []
                for g in (v or []):
                    out.append(str(getattr(g, 'game_id', None)
                                   or getattr(g, 'name', None) or g))
                if out:
                    return sorted(set(out))
            except Exception:
                continue
        return []

    def _acts(self):
        avail = getattr(self.frame, 'available_actions', None) or []
        acts = [ga for ga in _NONRESET if ga.value in avail]
        return acts or list(_NONRESET)

    def _make(self, game, seed):
        if self.card:
            return self.arc.make(game, seed=int(seed), render_mode=None,
                                 scorecard_id=self.card)
        return self.arc.make(game, seed=int(seed), render_mode=None)

    def _attempt(self, game, seed, guid=None):
        """One make+reset.  Returns the frame, or None if anything failed.

        A dead scorecard surfaces EITHER as `reset()` returning None OR as
        `make()` raising, depending on where the API refuses.  Both are the
        same fact -- no frame -- so both must reach the renewal path.  Letting
        a raise escape to the `serve()` handler was a hole with the identical
        shape to the bug that killed the world: `_ensure_mine` would re-open
        forever and `_renew()` would never run.
        """
        try:
            self.env = self._make(game, seed)
            if guid:
                try:
                    self.env._guid = guid
                except Exception:
                    pass
            return self.env.reset()
        except Exception as _e:
            sys.stderr.write('arc_server: open attempt failed for %s: %r\n'
                             % (game, _e))
            sys.stderr.flush()
            return None

    def open(self, game, seed=0):
        # RESUME THE RUN (2026-07-30).  MEASURED: banking across DEATHS already
        # works (runs show up to 44 resets while level_actions[0] stays at 12),
        # but `make()` returns a wrapper with `_guid = None`, so every re-open
        # starts a NEW server-side run at level 1 -- 866 guids for 1,732 runs,
        # and 0 runs ever started past level 1.  ROTATION, not death, is what
        # un-banks him, and `arc_world.py` re-opens on every owner change.
        # Putting the remembered guid back before the first reset makes the
        # library's own resume path (`if self._guid: payload["guid"]`) continue
        # the existing run, which server-side takes `level_reset()` and keeps
        # his level and score.
        self.game, self.seed = game, int(seed)
        _g = self.guids.get(game)
        self.frame = self._attempt(game, seed, _g)
        if self.frame is None and _g:
            # Stale or expired run -- fall back to exactly the old behaviour.
            sys.stderr.write('arc_server: resume failed for %s, fresh run\n'
                             % game)
            sys.stderr.flush()
            self.guids.pop(game, None)
            self.frame = self._attempt(game, seed)
        if self.frame is None:
            # SETTLE BEFORE CONDEMNING.  A couple of failed resets is what a
            # restart race looks like, not what a dead card looks like.  Start
            # (or continue) the failure streak and return promptly -- the
            # caller retries, and only a streak that outlives the whole window
            # is treated as death.
            _now = time.time()
            if self._fail_since is None:
                self._fail_since = _now
                sys.stderr.write('arc_server: open failing for %s -- settling, '
                                 'will renew if it persists %.0fs\n'
                                 % (game, self._SETTLE_SECONDS))
                sys.stderr.flush()
            _held = _now - self._fail_since
            if _held >= self._SETTLE_SECONDS and not self._renewed:
                # THE CARD REALLY IS GONE.  It has failed continuously for the
                # whole window, which is the measured signature of a closed
                # scorecard -- and no amount of re-opening fixes that from the
                # old client.  Renew once; the latch clears only when a frame
                # actually arrives, so this can never spam the API.
                sys.stderr.write('arc_server: card %s has failed continuously '
                                 'for %.0fs -- RENEWING\n'
                                 % (self.card, _held))
                sys.stderr.flush()
                self._renewed = True
                if self._renew():
                    self.frame = self._attempt(game, seed)
        if self.frame is not None:
            if self._fail_since is not None:
                sys.stderr.write('arc_server: card %s recovered after %.0fs '
                                 '-- NOT renewing\n'
                                 % (self.card, time.time() - self._fail_since))
                sys.stderr.flush()
                self._fail_since = None
            self._renewed = False
            # ONLY on a real frame.  A failed open leaves `self.env` pointing
            # at the PREVIOUS game, so filing its guid here would poison this
            # game's resume with another game's run.
            try:
                _n = getattr(self.env, '_guid', None)
                if _n:
                    self.guids[game] = _n
            except Exception:
                pass
        self.actions = self._acts()
        return self.snap()

    def reset(self):
        self.frame = self.env.reset()
        self.actions = self._acts()
        return self.snap()

    def step(self, a, x=None, y=None):
        i = int(a)
        if not self.actions:
            self.actions = self._acts()
        i = max(0, min(i, len(self.actions) - 1))
        act = self.actions[i]
        # ACTION6 is the COORDINATE action: without data it is a silent
        # no-op (verified: it returned a byte-identical grid).  The caller
        # supplies where he is LOOKING, so pointing costs him no new
        # decision.  Other actions take no data.
        if act.name == 'ACTION6' and x is not None and y is not None:
            self.frame = self.env.step(act, data={'x': int(x), 'y': int(y)})
        else:
            self.frame = self.env.step(act)
        self.actions = self._acts()
        return self.snap()

    def snap(self):
        if self.frame is None:
            # A failure that reports itself beats one that raises
            # AttributeError inside the request handler.
            return {'ok': 0, 'err': 'no frame', 'game': self.game}
        grid = self.frame._frame[-1]
        grid = [[int(v) for v in row] for row in grid]
        st = getattr(self.frame, 'state', None)
        return {
            'ok': 1,
            'game': self.game,
            'grid': grid,
            'state': getattr(st, 'name', str(st)),
            'levels': int(getattr(self.frame, 'levels_completed', 0) or 0),
            'actions': [int(x.value) for x in self.actions],
            'n_actions': len(self.actions),
        }


def serve():
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    g = Games()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    os.chmod(SOCK, 0o666)
    srv.listen(4)
    sys.stderr.write('arc_server ready on %s\n' % SOCK)
    sys.stderr.flush()
    while True:
        conn, _ = srv.accept()
        f = conn.makefile('rwb')
        try:
            for line in f:
                if not line.strip():
                    continue
                try:
                    req = json.loads(line)
                    c = req.get('cmd')
                    if c == 'games':
                        out = {'ok': 1, 'games': g.list_games()}
                    elif c == 'open':
                        out = g.open(req['game'], req.get('seed', 0))
                    elif c == 'reset':
                        out = g.reset()
                    elif c == 'step':
                        out = g.step(req.get('a', 0),
                                     req.get('x'), req.get('y'))
                    elif c == 'ping':
                        out = {'ok': 1}
                    elif c == 'card':
                        out = {'ok': 1, 'id': g.card,
                               'renewals': int(getattr(g, 'renewals', 0))}
                    elif c == 'close':
                        # CLOSE THE CARD (2026-09-25, the full sweep): finalise
                        # it on ARC's side while it is still alive (a card
                        # that idles out unclosed reads back 404 -- measured
                        # on every SEAGI3 card), keep the returned scorecard
                        # under /root/closed_cards/, then renew so play goes
                        # on with a fresh card.  The stale env is dropped so
                        # the next step fails fast and `_ensure_mine` reopens.
                        _sid = g.card
                        _card = None
                        try:
                            _c = g.arc.close_scorecard(_sid)
                            _card = (_c.model_dump()
                                     if hasattr(_c, 'model_dump')
                                     else (_c.dict()
                                           if hasattr(_c, 'dict')
                                           else str(_c)))
                        except Exception as _e:
                            _card = {'unavailable': '%s: %s'
                                     % (type(_e).__name__, _e)}
                        _ok = bool(isinstance(_card, dict)
                                   and 'unavailable' not in _card)
                        try:
                            os.makedirs('/root/closed_cards', exist_ok=True)
                            with open('/root/closed_cards/%s.json' % _sid,
                                      'w') as _f:
                                json.dump({'id': _sid, 'ok': _ok,
                                           'closed': time.strftime(
                                               '%Y-%m-%dT%H:%M:%SZ',
                                               time.gmtime()),
                                           'card': _card}, _f)
                        except Exception:
                            pass
                        sys.stderr.write('arc_server: CLOSE card %s ok=%s\n'
                                         % (_sid, _ok))
                        sys.stderr.flush()
                        if _ok:
                            g._renew()
                            g.env = None
                            g.frame = None
                        out = {'ok': 1 if _ok else 0, 'id': _sid,
                               'card': _card, 'new_card': g.card,
                               'url': 'https://three.arcprize.org/scorecards/%s'
                                      % _sid}
                    elif c == 'scorecard':
                        # READ-ONLY diagnostic.  Never let it break serving.
                        _sid = g.card or getattr(
                            g.arc, '_default_scorecard_id', None)
                        _card = None
                        try:
                            _c = g.arc.get_scorecard(_sid)
                            if _c is not None:
                                _card = (_c.model_dump()
                                         if hasattr(_c, 'model_dump')
                                         else (_c.dict()
                                               if hasattr(_c, 'dict')
                                               else str(_c)))
                        except Exception as _e:
                            _card = {'unavailable': '%s: %s'
                                     % (type(_e).__name__, _e)}
                        out = {'ok': 1, 'id': _sid, 'card': _card,
                               'renewals': int(getattr(g, 'renewals', 0)),
                               'key_tail': str(
                                   getattr(g.arc, 'arc_api_key', ''))[-6:]}
                    else:
                        out = {'ok': 0, 'err': 'bad cmd'}
                except Exception as e:
                    out = {'ok': 0, 'err': '%s: %s' % (type(e).__name__, e),
                           'tb': traceback.format_exc()[-400:]}
                f.write((json.dumps(out) + '\n').encode())
                f.flush()
        except Exception:
            pass
        finally:
            try:
                f.close(); conn.close()
            except Exception:
                pass


if __name__ == '__main__':
    serve()
