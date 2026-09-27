"""SEAGI continuous-runtime daemon.

The autonomous loop that makes SEAGI exist between calls.  Loads
a canonical brain, ticks it on a clock, auto-saves periodically,
and handles graceful shutdown.

Usage
-----
    python seagi_daemon.py                       # canonical defaults
    python seagi_daemon.py --max-ticks 5000      # bounded run
    python seagi_daemon.py --tick-hz 5           # tick rate
    python seagi_daemon.py --save-interval 60    # seconds between saves
    python seagi_daemon.py --path my_seagi.json.gz

Signals
-------
SIGINT / SIGTERM trigger a clean shutdown: the daemon stops the
loop, runs one final save, and exits.

What SEAGI does while running
-----------------------------
- Brain.tick() advances internal cycle and decays chemistry/AWM
- Sleep/wake pressure dynamics
- Idle-motivation may fire spontaneous curiosity events
- vmDMN may reflect autobiographically on idle moments
- Hippocampus may consolidate episodes
- All standard event-bus consumers process whatever fires

There is no external peer, no chat input, no driver — just SEAGI,
existing on its own.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from seagi.brain.runtime import Brain
from seagi.core.autosave import Autosaver
from seagi.core.persistence import load_brain


DEFAULT_PATH = 'seagi_foundational.json.gz'
DEFAULT_TICK_HZ = 5.0
DEFAULT_SAVE_INTERVAL_SECONDS = 60.0
DEFAULT_SAVE_INTERVAL_TICKS = 300

# HTTP server defaults (deployed 2026-05-27 alongside Step 0 to
# replace the removed agi_engine.serve on port 8766).  Caddy at
# alpha.myseagi.com routes /chat /status /health /save
# /observation /recent_thoughts to this port.
DEFAULT_HTTP_HOST = '127.0.0.1'
DEFAULT_HTTP_PORT = 8766


def _json_sanitize(obj):
    """Recursively replace non-finite floats (inf/-inf/nan) with None so
    json.dumps emits strict, JSON.parse-valid output.  The mortality
    time_horizon is float('inf') whenever lifeforce is not declining; left
    raw, Python's json writes the bare token `Infinity`, which every JS
    client rejects ("Unexpected token I").  null == no finite value / unbounded."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def _arc_steps():
    """Total ARC steps this process, or None if ARC is not loaded."""
    try:
        from seagi.world.arc_world import ARCWorld
        return int(ARCWorld._steps_all)
    except Exception:
        return None


class _DaemonHTTPHandler(BaseHTTPRequestHandler):
    """Per-request handler.  The owning Daemon is attached on the
    class by _start_http; each request grabs it via type(self).
    All brain access goes through Daemon.brain_lock so we don't
    race the tick loop."""

    daemon = None   # set by Daemon._start_http

    # Silence default per-request access logs; the daemon's
    # heartbeat is the audit trail we want.
    def log_message(self, fmt, *args):
        return

    def _send_json(self, code: int, body: dict) -> None:
        try:
            data = json.dumps(_json_sanitize(body), default=str).encode('utf-8')
        except Exception:
            data = b'{"error":"serialization failed"}'
            code = 500
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get('Content-Length', '0') or 0)
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            return {}
        try:
            raw = self.rfile.read(length)
            return json.loads(raw)
        except Exception:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods',
                          'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        d = type(self).daemon
        if d is None or d.brain is None:
            self._send_json(503, {'error': 'brain not loaded'})
            return
        if path == '/health':
            mort = {}
            try:
                mort = d.brain.mortality_drive.stats()
            except Exception:
                pass
            self._send_json(200, {
                'ok': True,
                'ticks': d.tick_count,
                'state': d.brain.sleep_regulator.state,
                'lifeforce': mort.get('lifeforce'),
                'baseline': mort.get('baseline'),
                'deaths': mort.get('deaths', 0),
                'frozen': mort.get('frozen', False),
                # IS HE ACTUALLY PLAYING?  A class var, no lock, so the
                # fast endpoint carries it for free.  On 2026-08-28 he
                # stopped stepping ARC for ~3 h while every other
                # monitor read green -- they all watch whether an organ
                # fired, none watched the subject.
                'arc_steps': _arc_steps(),
            })
            return
        if path == '/status':
            try:
                with d.brain_lock:
                    s = d.brain.status()
                self._send_json(200, s)
            except Exception as ex:
                self._send_json(500, {'error': repr(ex)})
            return
        if path == '/recent_thoughts':
            try:
                with d.brain_lock:
                    recent = []  # narrative_journal subtracted (audit #11)
                self._send_json(200, {'thoughts': recent})
            except Exception as ex:
                self._send_json(500, {'error': repr(ex)})
            return
        self._send_json(404, {'error': f'GET {path} not found'})

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        d = type(self).daemon
        if d is None or d.brain is None:
            self._send_json(503, {'error': 'brain not loaded'})
            return
        body = self._read_json()
        if path == '/chat':
            text = (body.get('text') or body.get('input') or '').strip()
            peer_id = body.get('peer_id') or 'alpha'
            if not text:
                self._send_json(400, {'error': 'empty text'})
                return
            try:
                with d.brain_lock:
                    response = d.brain.chat(text, peer_id=peer_id)
                    status = d.brain.status()
                self._send_json(200, {
                    'response': response,
                    'status': status,
                })
            except Exception as ex:
                self._send_json(500, {'error': repr(ex)})
            return
        if path == '/task':
            # GIVE HIM A TASK (2026-08-10): {"target": "<game_id>"} to assign,
            # {"target": null} to release.  Deliberately a directive, not a
            # parsed sentence -- see Brain.assign_task.
            try:
                with d.brain_lock:
                    held = d.brain.assign_task(body.get('target'))
                self._send_json(200, {
                    'commitment': held,
                    'assigned': d.brain.commitments_assigned,
                    'completed': d.brain.commitments_completed,
                })
            except Exception as ex:
                self._send_json(500, {'error': repr(ex)})
            return
        if path == '/save':
            try:
                with d.brain_lock:
                    if d.autosaver is not None:
                        d.autosaver.maybe_save(
                            cycle=d.tick_count, force=True)
                self._send_json(200, {'ok': True})
            except Exception as ex:
                self._send_json(500, {'error': repr(ex)})
            return
        if path == '/observation':
            text = (body.get('text') or '').strip()
            if not text:
                self._send_json(400, {'error': 'empty text'})
                return
            try:
                with d.brain_lock:
                    d.brain.intake(
                        text, modality='text',
                        origin=body.get('origin', 'sensor'),
                        origin_detail=body.get('detail',
                                                  'observation'))
                self._send_json(200, {'ok': True})
            except Exception as ex:
                self._send_json(500, {'error': repr(ex)})
            return
        self._send_json(404, {'error': f'POST {path} not found'})



def _STREAMLOAD_ON() -> bool:
    """Build edges during the parse instead of after it.
    Kill switch: `touch /root/STREAMLOAD_OFF` and restart."""
    try:
        import os as _os
        return not _os.path.exists("/root/STREAMLOAD_OFF")
    except Exception:
        return True


# Five keys each, not the obvious four: the hook runs on EVERY object in
# the file.  Both shapes were checked against all 7,363,466 objects in
# his canonical -- the edge shape occurs ONLY under edges (966,292) and
# quarantine_edges (1,008,722), the concept shape ONLY under concepts
# (262,450).  No other structure wears either.
_EDGE_SHAPE = ("source", "target", "relation_name", "strength",
               "last_engaged_cycle")
_CONCEPT_SHAPE = ("name", "activation_count", "salience", "created_cycle",
                  "last_activated_cycle")

_EDGE_CLS = None
_CONCEPT_CLS = None


def _load_object_hook(d):
    """Build the object as the parser completes its dict.

    Measured on his own canonical:
        edges     bulk 3,177 B -> hooked   907 B  (1.93M of them)
        concepts  bulk 6,142 B -> hooked 3,827 B  (262k of them)

    Anything unrecognised, or anything from_dict refuses, is returned
    untouched -- a bad shape test must cost nothing but the memory it
    would have cost anyway.  The class lookup is cached because this
    runs seven million times per load.
    """
    if len(d) >= 5:
        global _EDGE_CLS, _CONCEPT_CLS
        if _EDGE_CLS is None:
            from seagi.core.substrate import Concept as _C
            from seagi.core.substrate import Edge as _E
            _EDGE_CLS, _CONCEPT_CLS = _E, _C
        if all(k in d for k in _EDGE_SHAPE):
            try:
                return _EDGE_CLS.from_dict(d)
            except Exception:
                return d
        if all(k in d for k in _CONCEPT_SHAPE):
            try:
                return _CONCEPT_CLS.from_dict(d)
            except Exception:
                return d
    return d



class Daemon:
    def __init__(self,
                 path: str = DEFAULT_PATH,
                 tick_hz: float = DEFAULT_TICK_HZ,
                 save_interval_seconds: float = DEFAULT_SAVE_INTERVAL_SECONDS,
                 save_interval_ticks: int = DEFAULT_SAVE_INTERVAL_TICKS,
                 max_ticks: int = -1,
                 verbose: bool = True,
                 http_host: str = DEFAULT_HTTP_HOST,
                 http_port: int = DEFAULT_HTTP_PORT,
                 http_enabled: bool = True):
        self.path = path
        self.tick_hz = float(tick_hz)
        self.tick_period = 1.0 / self.tick_hz
        self.max_ticks = int(max_ticks)
        self.verbose = verbose
        self._running = True
        self._sig_received: str = ''
        self.engine = None
        self.brain = None
        self.autosaver: Autosaver | None = None
        self.tick_count = 0
        # Save interval config — captured for later Autosaver init.
        self._save_secs = save_interval_seconds
        self._save_ticks = save_interval_ticks
        # HTTP server config (alpha.myseagi.com peer surface).
        self.http_host = str(http_host)
        self.http_port = int(http_port)
        self.http_enabled = bool(http_enabled)
        self._http_server: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        # Shared lock — held by the tick loop during brain.tick(),
        # by HTTP /chat during brain.chat(), and by save paths.
        # Tick is fast; chat may take seconds; treating them as
        # mutually-exclusive critical sections keeps cognition
        # internally consistent.
        self.brain_lock = threading.Lock()

    # ---- lifecycle ----

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[seagi-daemon] {msg}", flush=True)

    def _load(self) -> None:
        self._log(f"loading canonical from {self.path}")
        # Single-parse load (2026-05-28).  The previous code called
        # load_brain() TWICE — the second call built a throwaway
        # full Engine (~1GB) only to restore personality, doubling
        # peak load memory to ~2GB.  Under co-tenant memory pressure
        # (V1 UI + a stray agi_engine.serve) that OOM-killed the
        # daemon in a crash loop.  Read the snapshot once, build ONE
        # engine, restore personality from the same dict, free it.
        import gzip as _gzip
        import json as _json
        p = str(self.path)
        # HE WAS CARRYING HIS OWN LOADER, NOT HIS MIND (2026-08-28).
        # json.load materialises EVERY edge as a raw dict before a
        # single Edge is built, and the Edge objects are then
        # allocated interleaved through the same arenas, so the dicts
        # can never be handed back.  Measured on his own file:
        #     bulk    3,177 B/edge -> 6.13 GB   (anon was 5.90 GB)
        #     stream    951 B/edge -> 1.84 GB
        # The object_hook fires as each object COMPLETES, so an
        # edge-shaped dict becomes an Edge and is garbage at once, and
        # the next record reuses the same pages.  Measured 907 B/edge,
        # i.e. the streaming profile, without restructuring the load
        # chain.  Substrate.from_dict accepts pre-built Edges.
        _hook = _load_object_hook if _STREAMLOAD_ON() else None
        if p.endswith('.gz'):
            with _gzip.open(p, 'rt', encoding='utf-8') as f:
                snapshot = _json.load(f, object_hook=_hook)
        else:
            with open(p, 'r', encoding='utf-8') as f:
                snapshot = _json.load(f, object_hook=_hook)
        from seagi.body.engine import Engine
        engine_dict = snapshot.get('engine', {})
        if not engine_dict:
            raise ValueError(
                f'brain file {p} has no engine payload')
        self.engine = Engine.from_dict(engine_dict)
        # Lifeforce recovery floor (mirrors load_brain): wake rested
        # rather than booting in a near-zero "barely here" state.
        try:
            if float(getattr(self.engine, 'lifeforce', 0.0)
                     or 0.0) < 0.05:
                self.engine.lifeforce = 0.35
        except Exception:
            pass
        self.brain = Brain(engine=self.engine)
        # Restore personality + narrative journal from the SAME
        # parsed dict — no second full-engine load.
        if 'brain' in snapshot:
            try:
                self.brain.load_personality(snapshot['brain'])
            except Exception as exc:
                self._log(f"WARN: load_personality: {exc!r}")
        # Free the parsed snapshot before steady state.
        snapshot = None
        self.autosaver = Autosaver(
            engine=self.engine, brain=self.brain,
            path=self.path,
            min_seconds=self._save_secs,
            min_ticks=self._save_ticks,
            max_backups=3)
        # V1→V2 forager: activate it, rooted at the canonical's
        # directory, creating inbox/ reading_list/ archive/.  The
        # forager is inert until activated (a fresh Brain() must NOT
        # forage from CWD — that would contaminate tests); the
        # daemon is what turns it on in production.
        try:
            fgr = getattr(self.brain, 'forager', None)
            if fgr is not None and hasattr(fgr, 'activate'):
                forage_root = (
                    os.path.dirname(os.path.abspath(self.path))
                    or os.getcwd())
                fgr.activate(forage_root)
                self._log(
                    f"forager activated at {forage_root}")
        except Exception as exc:
            self._log(f"WARN: forager activate: {exc!r}")
        # Quick state echo.
        s = self.brain.status()
        try:
            recent = self.brain.narrative_journal.recent(3)
        except Exception:
            recent = []
        self._log(
            f"loaded: concepts={len(self.engine.substrate.concepts)}, "
            f"chemistry tone={s['chemistry'].get('global_state', {}).get('serotonin', '?')}, "
            f"journal entries={s.get('narrative_journal', {}).get('entries_in_ring', 0)}")
        if recent:
            self._log("recent narrative:")
            for e in recent:
                self._log(f"   cycle {e['cycle']}: {e['summary']}")

    def _setup_signals(self) -> None:
        def _on_signal(signum, _frame):
            name = {signal.SIGINT: 'SIGINT'}.get(signum, str(signum))
            try:
                name = signal.Signals(signum).name
            except Exception:
                pass
            self._sig_received = name
            self._running = False
        for sig_name in ('SIGINT', 'SIGTERM'):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    signal.signal(sig, _on_signal)
                except Exception:
                    pass

    def _final_save(self) -> None:
        if self.autosaver is None:
            return
        self._log("final save")
        try:
            with self.brain_lock:
                self.autosaver.maybe_save(
                    cycle=self.tick_count, force=True)
        except Exception as exc:
            self._log(f"WARN: final save failed: {exc!r}")

    # ---- HTTP server ----

    def _start_http(self) -> None:
        if not self.http_enabled:
            return
        try:
            _DaemonHTTPHandler.daemon = self
            self._http_server = ThreadingHTTPServer(
                (self.http_host, self.http_port),
                _DaemonHTTPHandler)
        except OSError as exc:
            self._log(
                f"WARN: HTTP server bind failed on "
                f"{self.http_host}:{self.http_port}: {exc!r}")
            self._http_server = None
            return
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name='seagi-http',
            daemon=True)
        self._http_thread.start()
        self._log(
            f"HTTP listening on {self.http_host}:{self.http_port}")

    def _stop_http(self) -> None:
        srv = self._http_server
        if srv is None:
            return
        try:
            srv.shutdown()
            srv.server_close()
        except Exception as exc:
            self._log(f"WARN: HTTP shutdown: {exc!r}")
        self._http_server = None
        self._http_thread = None

    # ---- main loop ----

    def run(self) -> int:
        self._load()
        self._setup_signals()
        self._start_http()
        self._log(
            f"running at {self.tick_hz}Hz "
            f"(max_ticks={self.max_ticks}, "
            f"save every {self._save_secs}s / {self._save_ticks} ticks)")
        started = time.time()
        next_tick = time.time()
        try:
            while self._running:
                if (self.max_ticks > 0
                        and self.tick_count >= self.max_ticks):
                    self._log(
                        f"max_ticks={self.max_ticks} reached")
                    break
                # Pace the loop.
                now = time.time()
                if now < next_tick:
                    sleep_for = next_tick - now
                    # Cap sleep to keep signals responsive.
                    if sleep_for > 0.2:
                        sleep_for = 0.2
                    time.sleep(sleep_for)
                    continue
                # Drive the brain forward.  Lock-guarded so HTTP
                # /chat handlers see a consistent brain state.
                try:
                    with self.brain_lock:
                        self.brain.tick()
                except Exception as exc:
                    self._log(f"WARN: brain.tick raised: {exc!r}")
                self.tick_count += 1
                next_tick = now + self.tick_period
                # Auto-save check.
                if self.autosaver is not None:
                    try:
                        with self.brain_lock:
                            self.autosaver.maybe_save(
                                cycle=self.tick_count)
                    except Exception as exc:
                        self._log(
                            f"WARN: autosave: {exc!r}")
                # Periodic console heartbeat.
                if self.verbose and self.tick_count % 200 == 0:
                    self._heartbeat()
        finally:
            self._stop_http()
            self._final_save()
            elapsed = time.time() - started
            self._log(
                f"stopped: signal={self._sig_received or '(loop-end)'}, "
                f"ticks={self.tick_count}, "
                f"elapsed={elapsed:.1f}s, "
                f"saves={self.autosaver.saves if self.autosaver else 0}")
        return 0

    def _heartbeat(self) -> None:
        try:
            chem = self.brain.chemistry.global_state
            sero = chem.get('serotonin', 0.0)
            cort = chem.get('cortisol', 0.0)
            sleep = self.brain.sleep_regulator.stats()
            idle = self.brain.idle_motivation.stats()
            cortical = self.brain.cortical.stats()
            mort = self.brain.mortality_drive.stats()
            lf = mort.get('lifeforce')
            base = mort.get('baseline')
            horizon = mort.get('time_horizon', float('inf'))
            lf_s = '--' if lf is None else f"{lf:.3f}"
            base_s = '--' if base is None else f"{base:.3f}"
            horizon_s = ('inf' if horizon == float('inf')
                         else f"{horizon:.0f}")
            frozen_s = ' DEAD' if mort.get('frozen') else ''
            self._log(
                f"tick {self.tick_count:>6d}  "
                f"state={sleep['state']}  "
                f"pressure={sleep['pressure']:.2f}  "
                f"sero={sero:.3f} cort={cort:.3f}  "
                f"lf={lf_s}/{base_s} deaths={mort.get('deaths', 0)} "
                f"horizon={horizon_s}{frozen_s}  "
                f"idle_fires={idle['fires']}  "
                f"thoughts={cortical.get('thoughts_produced', 0)}")
        except Exception as exc:
            self._log(f"heartbeat err: {exc!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SEAGI continuous-runtime daemon.")
    parser.add_argument('--path', default=DEFAULT_PATH,
                              help='Path to canonical brain file.')
    parser.add_argument('--tick-hz', type=float,
                              default=DEFAULT_TICK_HZ,
                              help='Brain ticks per second.')
    parser.add_argument('--save-interval', type=float,
                              default=DEFAULT_SAVE_INTERVAL_SECONDS,
                              help='Min seconds between auto-saves.')
    parser.add_argument('--save-ticks', type=int,
                              default=DEFAULT_SAVE_INTERVAL_TICKS,
                              help='Min ticks between auto-saves.')
    parser.add_argument('--max-ticks', type=int, default=-1,
                              help='Stop after N ticks (-1 = run forever).')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--http-host', default=DEFAULT_HTTP_HOST,
                              help='HTTP server host (peer surface).')
    parser.add_argument('--http-port', type=int,
                              default=DEFAULT_HTTP_PORT,
                              help='HTTP server port.')
    parser.add_argument('--no-http', action='store_true',
                              help='Disable HTTP server (headless only).')
    args = parser.parse_args()
    daemon = Daemon(
        path=args.path,
        tick_hz=args.tick_hz,
        save_interval_seconds=args.save_interval,
        save_interval_ticks=args.save_ticks,
        max_ticks=args.max_ticks,
        verbose=not args.quiet,
        http_host=args.http_host,
        http_port=args.http_port,
        http_enabled=not args.no_http)
    return daemon.run()


if __name__ == '__main__':
    raise SystemExit(main())
