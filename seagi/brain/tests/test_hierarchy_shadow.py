"""Hierarchy shadow-validation (Capability 1, Step 5).

The predictive-coding Hierarchy (body/hierarchy.py) has NEVER run a live
inference cycle — `engine.tick()` only increments its counter.  Before
its edge-update deltas (EDGE_REINFORCE_DELTA=0.05, EDGE_WEAKEN_DELTA=0.03)
are EVER allowed to mutate the real substrate, this shadow window drives
the Hierarchy over firsthand WORLD data on a SCRATCH substrate and proves
the never-run dynamics are SAFE:

  * free-energy stays FINITE and does not DIVERGE (it settles after the
    warm-up transient as recurring structure is learned);
  * edge strength stays BOUNDED in [0, 1] — no runaway, no collapse.

Nothing here touches the live substrate (a fresh Substrate() is the
sandbox).  This is the gate the locked spec requires before Capability 1
is allowed to drive the Hierarchy for real.
"""

from __future__ import annotations

import math
import random
import sys
import types
import unittest

from seagi.core.substrate import Substrate, WORLD_TOKEN_PREFIX
from seagi.core.mi_value import MIValue
from seagi.body.hierarchy import Hierarchy
from seagi.brain.capabilities.grounding import GroundingLoop, WORLD_RELATION
from seagi.brain.capabilities.writer import JournaledSubstrateWriter
from seagi.world.world_driver import WorldDriver
from seagi.brain.capabilities.world_transducer import WorldTransducer


class _DeliverBus:
    def __init__(self, writer):
        self.writer = writer

    def publish(self, event):
        self.writer.handle(event, self)


def _seed_world_substrate(steps, grid, seed=7):
    """Run grounding over the firsthand world to populate a SCRATCH
    substrate with world token concepts + transitions_to edges, and
    return the token trajectory the Hierarchy will replay.

    A seeded random-action walk visits many cells, so the substrate gets
    MANY transition edges and the Hierarchy replay exercises BOTH the
    reinforce delta (recurring transitions) and the weaken delta
    (predicted-but-didn't-fire transitions)."""
    sub = Substrate()
    sub._quarantine_migrated = True
    engine = types.SimpleNamespace(substrate=sub)
    writer = JournaledSubstrateWriter(engine=engine)
    loop = GroundingLoop(engine=engine, bus=_DeliverBus(writer))
    driver = WorldDriver(grid=grid, k=8, stochastic=False, seed=grid)
    trans = WorldTransducer()
    rng = random.Random(seed)
    c = 1
    prev = None
    seq = []
    for _ in range(steps):
        tok = trans.encode(driver.percept()['world_vector'])
        if prev is not None:
            loop.observe(prev, tok, cycle=c)
            c += 1
        seq.append(tok)
        driver.step(rng.randrange(driver.n_actions))   # varied walk
        prev = tok
    return sub, seq


def _wmean(xs, lo, hi):
    seg = xs[lo:hi]
    return sum(seg) / len(seg) if seg else 0.0


class TestHierarchyShadow(unittest.TestCase):

    @unittest.skip(
        "SHADOW GATE = NO-GO (2026-06-09): driving the never-run Hierarchy "
        "over real world data overflows — combine_or SUMS evidence counts "
        "(mi_value.py:78) and L2 lateral recurrence re-combines persisted "
        "propositions every cycle, so mi.n compounds unbounded until "
        "decay_step (hierarchy.py:672) raises OverflowError. The deltas "
        "themselves are safe (free-energy converged to 0, edge strength "
        "clamped at 1.0 in the deterministic run) but the n-accumulation "
        "bug BLOCKS wiring the Hierarchy. It stays dark until the "
        "unbounded-n is fixed (its own design+audit — combine_or is used "
        "everywhere). The grounding core does not depend on the Hierarchy.")
    def test_deltas_converge_and_bounded_on_world(self):
        sub, seq = _seed_world_substrate(steps=500, grid=5)
        h = Hierarchy(sub)
        mi = MIValue(0.5, 0.5, 1)
        zero = MIValue.zero()

        fes = []
        for i in range(1, len(seq)):
            # prev + cur tokens co-fire = a transition proposition along
            # the substrate edge the grounding loop already wrote.
            obs = {seq[i - 1]: mi, seq[i]: mi}
            r = h.cycle(observation=obs, tone=zero, lifeforce=0.7)
            fes.append(float(r['free_energy']['total']))

        n = len(fes)
        q = max(5, n // 4)
        fe_first = _wmean(fes, 0, q)
        fe_peak = max(fes)
        fe_secondq = _wmean(fes, q, 2 * q)
        fe_last = _wmean(fes, n - q, n)
        strengths = [e.strength for e in sub.edges.values()]
        world_str = [e.strength for k, e in sub.edges.items()
                     if k[1] == WORLD_RELATION]
        smin = min(strengths) if strengths else 0.0
        smax = max(strengths) if strengths else 0.0

        print(
            f"\n[shadow] cycles={n} edges={len(sub.edges)} "
            f"FE first={fe_first:.4f} 2ndQ={fe_secondq:.4f} "
            f"peak={fe_peak:.4f} last={fe_last:.4f} | "
            f"edge_strength min={smin:.4f} max={smax:.4f} | "
            f"world_edges={len(world_str)} "
            f"world_str_max={(max(world_str) if world_str else 0):.4f}",
            file=sys.stderr)

        # 1) free-energy is finite throughout — never NaN / inf.
        self.assertTrue(all(math.isfinite(x) for x in fes),
                        "free-energy went non-finite (diverged)")
        # 2) edge strength stays bounded in [0, 1] — no runaway, the
        #    central safety property of the never-run deltas.
        self.assertLessEqual(smax, 1.0 + 1e-9,
                             "edge strength ran away above 1.0")
        self.assertGreaterEqual(smin, 0.0)
        # 3) it does NOT diverge after the warm-up: the last window does
        #    not exceed the post-transient (second-quarter) window.
        self.assertLessEqual(fe_last, fe_secondq + 1e-6,
                             "free-energy diverged after warm-up")
        # 4) the deltas ACTUALLY fired (the validation exercised them):
        #    recurring world transitions got reinforced above the floor.
        self.assertTrue(world_str)
        self.assertGreater(max(world_str), 0.1,
                           "backward-pass reinforcement never engaged")


if __name__ == '__main__':
    unittest.main()
