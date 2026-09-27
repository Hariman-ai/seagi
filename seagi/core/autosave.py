"""Autosaver — periodic state persistence for continuous runtime.

Wraps `save_brain` with interval logic and an optional backup
rotation.  Intended for use by the autonomous daemon (Phase C.1.d)
to keep the brain's state durable across crashes / restarts /
power events.

Usage
-----
    autosaver = Autosaver(
        engine=engine, brain=brain,
        path='seagi_live.json.gz',
        min_seconds=60,           # save at most every 60s
        min_ticks=300,            # OR every 300 brain ticks
        max_backups=3)

    # Called from the daemon loop:
    if autosaver.maybe_save(cycle=current_cycle):
        # a save just happened
        ...

The maybe_save method returns True when a save actually wrote.
It debounces — back-to-back calls within the interval are no-ops.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Any, Optional

from .persistence import save_brain


class Autosaver:
    """Periodic save_brain with interval gating + backup rotation."""

    def __init__(self,
                 engine: Any,
                 brain: Any,
                 path: str,
                 min_seconds: float = 60.0,
                 min_ticks: int = 300,
                 max_backups: int = 3):
        self.engine = engine
        self.brain = brain
        self.path = str(path)
        self.min_seconds = float(min_seconds)
        self.min_ticks = int(min_ticks)
        self.max_backups = int(max_backups)
        # Track when we last saved so we can debounce.
        self._last_save_time: float = 0.0
        self._last_save_cycle: int = -1
        # Diagnostics.
        self.saves: int = 0
        self.last_path: str = ''

    def maybe_save(self,
                       cycle: Optional[int] = None,
                       force: bool = False) -> bool:
        """Save if min_seconds OR min_ticks have elapsed since
        last save (whichever is sooner).  Returns True if a save
        actually happened.  Pass force=True to bypass gating."""
        now = time.time()
        cycle = int(cycle) if cycle is not None else -1
        if not force:
            secs_since = now - self._last_save_time
            ticks_since = (cycle - self._last_save_cycle) \
                if cycle >= 0 and self._last_save_cycle >= 0 \
                else self.min_ticks + 1
            if (secs_since < self.min_seconds
                and ticks_since < self.min_ticks):
                return False
        # Rotate existing path → .bak.1 .bak.2 ...
        self._rotate_backups()
        # Write the new save.
        try:
            save_brain(self.engine, self.path, brain=self.brain)
        except Exception as exc:
            print(f"WARN: autosave failed: {exc!r}")
            return False
        self._last_save_time = now
        self._last_save_cycle = cycle
        self.saves += 1
        self.last_path = self.path
        return True

    def _rotate_backups(self) -> None:
        """Move .bak.N → .bak.(N+1), drop the oldest."""
        if not os.path.exists(self.path):
            return
        if self.max_backups <= 0:
            return
        # Walk backwards from the oldest backup so we don't
        # overwrite anything.
        for i in range(self.max_backups, 0, -1):
            src = self.path if i == 1 else (
                self.path + f'.bak.{i - 1}')
            dst = self.path + f'.bak.{i}'
            if os.path.exists(src):
                # If we're at the top of the rotation and the
                # oldest slot exists, drop it.
                if i == self.max_backups and os.path.exists(dst):
                    try:
                        os.remove(dst)
                    except OSError:
                        pass
                try:
                    shutil.copy2(src, dst)
                except OSError:
                    pass

    def stats(self) -> dict:
        return {
            'saves': self.saves,
            'last_save_time': self._last_save_time,
            'last_save_cycle': self._last_save_cycle,
            'last_path': self.last_path,
            'path': self.path,
        }
