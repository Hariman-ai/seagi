"""Persistence — single save_brain / load_brain wrapping engine state.

Wraps Engine.to_dict / Engine.from_dict with:
    - Atomic write (temp file + rename)
    - Versioning (SCHEMA_VERSION)
    - Optional gzip compression
    - Tolerant load (warns on missing fields, doesn't crash)
    - Companion metadata (.meta.json with cycle, lifeforce, save time)

Usage:
    from seagi.core.persistence import save_brain, load_brain

    save_brain(engine, 'brain.json')
    engine = load_brain('brain.json')

For deployment, a small `BrainCheckpointer` schedules periodic saves
in a background thread without blocking the tick loop.
"""

from __future__ import annotations
import gzip
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any,  Any, Dict, Optional

# Engine imported lazily inside functions to avoid
# circular import via the agi_engine.mi_value shim chain.


# Bumped when on-disk schema changes incompatibly. Older saves attempt
# to load; missing fields are skipped with a warning.
SCHEMA_VERSION = 1


@dataclass
class BrainMetadata:
    """Lightweight companion to a brain save — fast to read without
    loading the full substrate."""
    schema_version: int
    saved_at_iso: str
    cycle: int
    lifeforce: float
    concepts: int
    edges: int
    episodes: int

    def to_dict(self) -> dict:
        return {
            'schema_version': self.schema_version,
            'saved_at_iso': self.saved_at_iso,
            'cycle': self.cycle,
            'lifeforce': self.lifeforce,
            'concepts': self.concepts,
            'edges': self.edges,
            'episodes': self.episodes,
        }


# ---------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------

def _STREAMSAVE_ON() -> bool:
    """Build each edge/concept dict as the encoder reaches it, instead of
    materialising all 2.25M first.  Kill switch: `touch
    /root/STREAMSAVE_OFF` (read per save, no restart)."""
    try:
        return not os.path.exists("/root/STREAMSAVE_OFF")
    except Exception:
        return True


def _save_default(o):
    """`default` for json.dump under a lazy snapshot.

    ONLY Edge and Concept are converted -- everything else keeps the
    previous `default=str` behaviour exactly, so nothing that reached
    `str()` before starts serialising differently now.
    """
    try:
        from seagi.core.substrate import Concept, Edge
        if isinstance(o, (Edge, Concept)):
            return o.to_dict()
    except Exception:
        pass
    return str(o)


def save_brain(engine: 'Any',
                path: str,
                gzipped: Optional[bool] = None,
                brain: Optional['Any'] = None) -> BrainMetadata:
    """Atomically write `engine` to `path`. Returns metadata.

    gzipped: None = infer from extension (.gz → True); else explicit.
    Atomic: write to path.tmp first, then rename.
    Side-effect: a companion `<path>.meta.json` is also written for
    fast inspection without loading the full brain.

    Phase H.1: when `brain` is supplied, its M/I-weighted
    personality state (identity, goals, action credit, skills,
    discovered laws, confirmed hypotheses, schemas) is also
    written into the payload under the 'brain' key.  Each
    registry filters by its own M/I-weight threshold so only
    items that crossed the personality floor persist.
    """
    path = str(path)
    if gzipped is None:
        gzipped = path.endswith('.gz')

    snapshot: Dict[str, Any] = {
        'schema_version': SCHEMA_VERSION,
        'saved_at_iso': _now_iso(),
        'engine': None,
    }
    # STREAMED, NOT MATERIALISED, PART TWO (2026-08-29).  json.dumps was
    # fixed on 08-20 so the JSON text is no longer built as one string --
    # but `engine.to_dict()` still built EVERY edge and concept as a raw
    # dict before the encoder walked any of them: 1,010 B/edge, a 2.00 GB
    # transient that took RSS from 3,541,700 kB to 5,911,144 every 600 s
    # on a 7.75 GB box.  With `lazy=True` the objects go in whole and
    # `_save_default` converts each one as it is reached.
    _lazy = _STREAMSAVE_ON()
    try:
        snapshot['engine'] = (engine.to_dict(lazy=True) if _lazy
                              else engine.to_dict())
    except TypeError:
        # an engine that predates the flag
        _lazy = False
        snapshot['engine'] = engine.to_dict()
    if brain is not None:
        try:
            snapshot['brain'] = brain.to_dict()
        except Exception as exc:
            print(f'WARN: brain.to_dict failed ({exc!r}); '
                  f'saving engine only', file=sys.stderr)

    # Atomic write.
    tmp = path + '.tmp'
    # STREAMED, NOT MATERIALISED (2026-08-20, profiled).  `json.dumps`
    # built the entire ~945 MB payload as one string before writing a
    # byte -- a transient allocation against a 5 GB RSS on a 7.75 GB box,
    # and the likely cause of the OOM kills.  `json.dump` writes straight
    # into the file.  compresslevel=1 because level 9 (the default) spent
    # 44% of his ENTIRE CPU squeezing a file rewritten every minute; he
    # was living 8% of the time and saving 92%.
    if gzipped:
        with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=1) as f:
            json.dump(snapshot, f, separators=(',', ':'),
                      default=_save_default if _lazy else str)
    else:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, separators=(',', ':'),
                      default=_save_default if _lazy else str)
    os.replace(tmp, path)

    # Companion metadata.
    meta = BrainMetadata(
        schema_version=SCHEMA_VERSION,
        saved_at_iso=snapshot['saved_at_iso'],
        cycle=int(engine.hierarchy.cycle_counter),
        lifeforce=float(engine.lifeforce),
        concepts=len(engine.substrate.concepts),
        edges=len(engine.substrate.edges),
        episodes=len(engine.substrate.episodes),
    )
    meta_path = path + '.meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta.to_dict(), f, indent=2)
    return meta


def load_brain(path: str,
                seed: Optional[int] = None,
                strict_version: bool = False,
                brain: Optional['Any'] = None) -> 'Any':
    """Read engine state from `path` and reconstruct an Engine.

    seed: passed to Engine for deterministic reseeding.
    strict_version: when True, raise on schema_version mismatch;
    when False (default), warn to stderr and proceed (tolerant load).

    Phase H.1: when `brain` is supplied AND the saved payload
    contains a 'brain' key (saved with brain= via save_brain),
    the personality state is restored INTO the given Brain via
    `brain.load_personality(...)`.  Older payloads without a
    'brain' key are accepted silently — backward compatible.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'brain file not found: {path}')
    is_gzip = path.endswith('.gz')
    if is_gzip:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            snapshot = json.load(f)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            snapshot = json.load(f)

    file_version = int(snapshot.get('schema_version', 0))
    if file_version != SCHEMA_VERSION:
        msg = (f'brain schema mismatch: file={file_version} '
                f'current={SCHEMA_VERSION}')
        if strict_version:
            raise ValueError(msg)
        print(f'WARN: {msg}; loading tolerantly', file=sys.stderr)

    engine_dict = snapshot.get('engine', {})
    if not engine_dict:
        raise ValueError(f'brain file {path} has no engine payload')
    from seagi.body.engine import Engine
    engine = Engine.from_dict(engine_dict, seed=seed)
    # Brain-recovery floor: if the saved brain had lifeforce
    # at (or near) 0, give it a starting boost on load.  The
    # alternative is the agent boots already in "barely here"
    # state and takes many quiet cycles to recover.  Brain
    # analog: waking up rested.  Only applies on COLD load
    # (one-time on startup), not during runtime.
    try:
        if float(getattr(engine, 'lifeforce', 0.0) or 0.0) < 0.05:
            engine.lifeforce = 0.35
            print('serve: lifeforce boost on load (was near 0)',
                    file=sys.stderr)
    except Exception:
        pass

    # Phase H.1: restore personality state into the supplied
    # Brain if both payload and caller provided one.
    if brain is not None and 'brain' in snapshot:
        try:
            brain.load_personality(snapshot['brain'])
        except Exception as exc:
            print(f'WARN: brain.load_personality failed '
                  f'({exc!r}); engine loaded but personality '
                  f'left at defaults', file=sys.stderr)

    return engine


def read_brain_metadata(path: str) -> Optional[BrainMetadata]:
    """Cheap inspection: load only the companion .meta.json."""
    meta_path = str(path) + '.meta.json'
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return BrainMetadata(
            schema_version=int(d.get('schema_version', 0)),
            saved_at_iso=str(d.get('saved_at_iso', '')),
            cycle=int(d.get('cycle', 0)),
            lifeforce=float(d.get('lifeforce', 0.7)),
            concepts=int(d.get('concepts', 0)),
            edges=int(d.get('edges', 0)),
            episodes=int(d.get('episodes', 0)),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------
# Round-trip self-check (defensive — used by daemons)
# ---------------------------------------------------------------------

def quick_self_check(engine: 'Any') -> bool:
    """Lightweight roundtrip: serialize, deserialize, compare a few
    invariants. Returns True if the engine survives a save+load."""
    try:
        d = engine.to_dict()
        from seagi.body.engine import Engine
        restored = Engine.from_dict(d, seed=0)
        return (
            len(restored.substrate.concepts)
                == len(engine.substrate.concepts)
            and len(restored.substrate.edges)
                == len(engine.substrate.edges)
            and abs(restored.lifeforce - engine.lifeforce) < 1e-9
            and (restored.hierarchy.cycle_counter
                 == engine.hierarchy.cycle_counter)
        )
    except Exception:
        return False


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _now_iso() -> str:
    """Current UTC time in ISO format. No tz dependency."""
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
