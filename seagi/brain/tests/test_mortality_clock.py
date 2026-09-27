"""Tests for the LIVE MortalityClock (promoted shadow, 2026-07-18) + the
subtractive advance law in MortalityDrive + the substrate reinforce
observer / origin threading.

The twelve audited-flip acceptance tests:
 1. stable-digest determinism across processes (hash-salting landmine)
 2. ledger round-trip (no second credit after persist/load)
 3. crash-window reconcile (pre-seed MARKS without CREDIT)
 4. once-per-fact crediting
 5. Option-A fade-gated revival (world facts once more; fcc never)
 6. freeze-proof + wall clamp at both ends
 7. cf folds into o (the missing-term landmine — dropping cf would
    roughly halve aging)
 8. origin threading lands in the right split buckets; maintenance
    settlings STILL credit E
 9. death: distillate record written, E + life split reset, ledger intact
10. dead-save load (no 'mortality_clock' key) loads clean
11. wall-migration sanity (externally-set wall survives load)
12. not-a-scheduler: zero settlings → wall strictly increases every tick
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

from seagi.core.substrate import (
    Edge,
    EDGE_PRUNE_FLOOR,
    EDGE_STRENGTH_DECAY_PER_CYCLE,
    register_reinforce_observer,
    unregister_reinforce_observer,
    reinforce_observer_errors,
)
from seagi.brain import EventBus
from seagi.brain.capabilities.mortality_clock import (
    MortalityClock,
    SATURATION_CEILING,
    _composable_result_relations,
)
from seagi.brain.capabilities.mortality_drive import (
    MortalityDrive,
    BASELINE_FLOOR,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# A relation name that is certainly NOT a composition result (world-class,
# never fcc-eligible).
_WORLD_REL = 'zz_world_probe_rel'


class _DriveStub:
    """Minimal read-only stand-in for MortalityDrive (the clock only reads
    deaths / revivals / wall / obstruction_ema / _lf_get)."""

    def __init__(self):
        self.deaths = 0
        self.revivals = 0
        self.wall = 0.1
        self.obstruction_ema = 0.0
        self._lf_get = lambda: 0.7
        self._last_wall_advance = 0.0


class _SubStub:
    """Minimal substrate stand-in: two edge pools + fade backlog field."""

    def __init__(self):
        self.edges = {}
        self.quarantine_edges = {}
        self._last_fade_backlog = 0


class _Body:
    def __init__(self, lifeforce=0.7):
        self.lifeforce = lifeforce


def _drive(body, cycle_box):
    bus = EventBus()
    md = MortalityDrive(
        bus=bus,
        cycle_provider=lambda: cycle_box[0],
        lifeforce_get=lambda: body.lifeforce,
        lifeforce_set=lambda v: setattr(body, 'lifeforce', v))
    return bus, md


def _clock(drive=None, sub=None, **kw):
    """Construct a clock with logging disabled unless paths are given."""
    kw.setdefault('log_path', '')
    kw.setdefault('death_log_path', '')
    return MortalityClock(
        mortality_drive=(drive if drive is not None else _DriveStub()),
        substrate_provider=((lambda: sub) if sub is not None else None),
        **kw)


def _fcc_edge(s, t, fcc=5, strength=0.5):
    return Edge(source=s, target=t, relation_name='causes',
                strength=strength, first_coherent_cycle=fcc)


def _world_edge(s, t, strength=0.977):
    return Edge(source=s, target=t, relation_name=_WORLD_REL,
                strength=strength, first_coherent_cycle=0)


def _n(clock, cls, origin=None):
    buf = clock._settle_alltime[cls]
    if origin is not None:
        return int(buf[origin]['n'])
    return int(sum(buf[oc]['n'] for oc in buf))


class _ClockTestBase(unittest.TestCase):

    def tearDown(self):
        unregister_reinforce_observer()


# =====================================================================
# 1. stable-digest determinism
# =====================================================================
class TestStableDigest(_ClockTestBase):

    def test_digest_stable_across_hashseed_processes(self):
        script = (
            "from seagi.brain.capabilities.mortality_clock import "
            "MortalityClock; "
            "print(MortalityClock._khash(('alpha', 'causes', 'beta')))")
        outs = []
        for seed in ('0', '424242'):
            env = dict(os.environ)
            env['PYTHONHASHSEED'] = seed
            r = subprocess.run(
                [sys.executable, '-c', script],
                capture_output=True, text=True, env=env, cwd=_REPO_ROOT,
                timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            outs.append(r.stdout.strip())
        self.assertEqual(outs[0], outs[1])
        # And equal to the in-process digest (same keying everywhere).
        self.assertEqual(
            int(outs[0]),
            MortalityClock._khash(('alpha', 'causes', 'beta')))

    def test_digest_differs_for_different_facts(self):
        self.assertNotEqual(
            MortalityClock._khash(('a', 'r', 'b')),
            MortalityClock._khash(('a', 'r', 'c')))


# =====================================================================
# 2. ledger round-trip
# =====================================================================
class TestLedgerRoundTrip(_ClockTestBase):

    def test_no_second_credit_after_persist_load(self):
        c1 = _clock()
        e = _fcc_edge('a', 'b')
        e.reinforce(1, origin='cognition')
        self.assertEqual(_n(c1, 'internal_fcc'), 1)
        snap = c1.to_dict()
        self.assertEqual(snap['schema'], 1)
        self.assertEqual(snap['key_digest'], 'blake2b8/fam2i')
        # JSON round-trip (what persistence actually does).
        snap = json.loads(json.dumps(snap))

        c2 = _clock()          # fresh instance, replaces the observer
        c2.load_dict(snap)
        self.assertIn(MortalityClock._khash(('a', 'causes', 'b')),
                      c2._settled)
        e.reinforce(2, origin='cognition')
        # Already settled: NO second credit.
        self.assertEqual(_n(c2, 'internal_fcc'), 1)
        self.assertEqual(c2._tick_settle_earn, 0.0)
        self.assertEqual(c2._settlings_this_tick, 0)

    def test_mismatched_key_digest_starts_fresh(self):
        c1 = _clock()
        c1._settled.add(12345)
        snap = c1.to_dict()
        snap['key_digest'] = 'builtin_salted'
        c2 = _clock()
        c2.load_dict(snap)
        self.assertEqual(len(c2._settled), 0)


# =====================================================================
# 3. crash-window reconcile
# =====================================================================
class TestCrashWindowReconcile(_ClockTestBase):

    def test_preseed_marks_without_credit(self):
        sub = _SubStub()
        e1 = _fcc_edge('f1', 'x')
        sub.edges[('f1', 'causes', 'x')] = e1
        c1 = _clock(sub=sub)
        e1.reinforce(1, origin='cognition')
        snap = json.loads(json.dumps(c1.to_dict()))
        snap_counts = 1
        self.assertEqual(_n(c1, 'internal_fcc'), snap_counts)

        # AFTER the snapshot (the crash window), MORE facts settle.
        e2 = _fcc_edge('f2', 'x')
        e3 = _fcc_edge('f3', 'x')
        sub.edges[('f2', 'causes', 'x')] = e2
        sub.edges[('f3', 'causes', 'x')] = e3
        e2.reinforce(2, origin='cognition')
        e3.reinforce(2, origin='cognition')

        # Fresh boot from the OLD snapshot: pre-seed reconciles.
        c2 = _clock(sub=sub)
        c2.load_dict(snap)
        c2.run(3)
        kh2 = MortalityClock._khash(('f2', 'causes', 'x'))
        kh3 = MortalityClock._khash(('f3', 'causes', 'x'))
        # MARKED...
        self.assertIn(kh2, c2._settled)
        self.assertIn(kh3, c2._settled)
        self.assertEqual(c2._preseeded_count, 2)
        # ...without CREDIT: counts still the snapshot's, E untouched.
        self.assertEqual(_n(c2, 'internal_fcc'), snap_counts)
        self.assertEqual(c2.E_settle, 0.0)
        # And a re-reinforce of a reconciled fact credits nothing.
        e2.reinforce(4, origin='cognition')
        self.assertEqual(_n(c2, 'internal_fcc'), snap_counts)
        self.assertEqual(c2._tick_settle_earn, 0.0)


# =====================================================================
# 4. once-per-fact
# =====================================================================
class TestOncePerFact(_ClockTestBase):

    def test_n_reconfirms_credit_exactly_once(self):
        c = _clock()
        e = _fcc_edge('p', 'q')
        for i in range(1, 11):
            e.reinforce(i, origin='cognition')
        self.assertEqual(_n(c, 'internal_fcc'), 1)
        self.assertEqual(c._settlings_this_tick, 1)
        self.assertEqual(c.reinforces_seen, 10)


# =====================================================================
# 5. Option-A fade-gated revival
# =====================================================================
class TestOptionARevival(_ClockTestBase):

    def test_world_fact_revives_exactly_once_after_genuine_fade(self):
        comp = _composable_result_relations()
        if comp is not None:
            self.assertNotIn(_WORLD_REL, comp)
        c = _clock()
        e = _world_edge('w', 'x')
        e.reinforce(1)                       # 0.977→~0.982: crosses 0.98
        self.assertEqual(_n(c, 'world_098'), 1)
        self.assertEqual(_n(c, 'revival'), 0)

        # GENUINE fade far below the prune floor → dormant on next touch.
        far = 300000
        e.reinforce(far)                     # pre_eff 0 ≤ floor → dormant
        kh = MortalityClock._khash(('w', _WORLD_REL, 'x'))
        self.assertIn(kh, c._dormant)
        self.assertEqual(_n(c, 'revival'), 0)   # climbing back ≠ credit

        # The structural ~196-confirm re-climb back to the door.
        for _ in range(400):
            e.reinforce(far)
        self.assertEqual(_n(c, 'revival'), 1)   # exactly ONE revival credit
        self.assertNotIn(kh, c._dormant)
        # Continued re-confirms at the ceiling: nothing more.
        for _ in range(10):
            e.reinforce(far)
        self.assertEqual(_n(c, 'revival'), 1)

    def test_fcc_settled_fact_never_revives(self):
        c = _clock()
        e = _fcc_edge('m', 'n')
        e.reinforce(1)                       # settled internal_fcc
        self.assertEqual(_n(c, 'internal_fcc'), 1)
        e.reinforce(300000)                  # faded — but fcc is permanent
        kh = MortalityClock._khash(('m', 'causes', 'n'))
        self.assertNotIn(kh, c._dormant)
        for _ in range(400):
            e.reinforce(300000)              # re-climb past 0.98
        self.assertEqual(_n(c, 'revival'), 0)
        self.assertEqual(_n(c, 'internal_fcc'), 1)


# =====================================================================
# 6. freeze-proof + clamp
# =====================================================================
class TestFreezeProofAndClamp(_ClockTestBase):

    def test_advance_at_zero_E_always_at_least_D(self):
        D = EDGE_STRENGTH_DECAY_PER_CYCLE
        prev = 0.0
        for o in (0.0, 0.5, 1.0, 5.0):
            adv = MortalityDrive.compute_pertick_advance(o, 0.0)
            self.assertGreaterEqual(adv, D)
            self.assertGreater(adv, 0.0)
            self.assertGreater(adv, prev - 1e-15)   # monotone in o
            prev = adv

    def test_wall_clamped_at_ceiling(self):
        body = _Body(0.9)
        _, md = _drive(body, [0])
        md.set_clock_providers(E_provider=lambda: 0.0,
                               cf_provider=lambda: 1e6)
        md.tick()
        md.wall = 1.0
        md.tick()
        self.assertLessEqual(md.wall, 1.0)

    def test_wall_clamped_at_floor(self):
        body = _Body(0.9)
        _, md = _drive(body, [0])
        md.set_clock_providers(E_provider=lambda: 1e6,
                               cf_provider=lambda: 0.0)
        md.tick()
        md.tick()
        self.assertGreaterEqual(md.wall, BASELINE_FLOOR)
        self.assertEqual(md.wall, BASELINE_FLOOR)


# =====================================================================
# 7. cf folds into o (the missing-term landmine)
# =====================================================================
class TestCfInO(_ClockTestBase):

    def test_drive_advance_reflects_obstruction_plus_cf(self):
        D = EDGE_STRENGTH_DECAY_PER_CYCLE
        body = _Body(0.9)
        _, md = _drive(body, [0])
        md.set_clock_providers(E_provider=lambda: 0.0,
                               cf_provider=lambda: 0.5)
        md.tick()
        # obstruction_ema is 0 (no ledger) → o must be exactly cf.
        self.assertAlmostEqual(md._last_o, md.obstruction_ema + 0.5)
        self.assertAlmostEqual(md._last_wall_advance,
                               MortalityDrive.compute_pertick_advance(
                                   md.obstruction_ema + 0.5, 0.0))
        # Dropping cf would halve this tick's aging: assert it does NOT.
        self.assertAlmostEqual(md._last_wall_advance, D * 1.5)
        self.assertGreater(md._last_wall_advance, D * 1.0)

    def test_clock_cf_is_unswept_fraction_of_backlog(self):
        sub = _SubStub()
        sub._last_fade_backlog = 10
        c = _clock(sub=sub, qcap_provider=lambda: 4)
        self.assertAlmostEqual(c.cf(), 0.6)
        sub._last_fade_backlog = 0
        self.assertAlmostEqual(c.cf(), 0.0)


# =====================================================================
# 8. origin threading
# =====================================================================
class TestOriginThreading(_ClockTestBase):

    def test_each_origin_lands_in_its_bucket(self):
        c = _clock()
        _fcc_edge('o1', 'x').reinforce(1, origin='cognition')
        _fcc_edge('o2', 'x').reinforce(1, origin='maintenance')
        _fcc_edge('o3', 'x').reinforce(1)                  # unlabeled
        _fcc_edge('o4', 'x').reinforce(1, origin='weird_label')
        self.assertEqual(_n(c, 'internal_fcc', 'cognition'), 1)
        self.assertEqual(_n(c, 'internal_fcc', 'maintenance'), 1)
        self.assertEqual(_n(c, 'internal_fcc', 'unknown'), 2)

    def test_maintenance_settling_still_credits_E(self):
        c = _clock()
        _fcc_edge('o5', 'x').reinforce(1, origin='maintenance')
        self.assertGreater(c._tick_settle_earn, 0.0)
        c.run(1)
        self.assertGreater(c.E(), 0.0)

    def test_observer_errors_visible_not_swallowed_silently(self):
        before = reinforce_observer_errors()
        register_reinforce_observer(
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
        e = _fcc_edge('o6', 'x')
        e.reinforce(1, origin='cognition')     # must NOT raise
        self.assertEqual(reinforce_observer_errors(), before + 1)
        # The reinforce itself still happened.
        self.assertGreater(e.strength, 0.5)


# =====================================================================
# 9. death: distillate + resets + ledger persistence
# =====================================================================
class TestDeathSuccession(_ClockTestBase):

    def test_death_record_written_and_state_reset(self):
        tmp = tempfile.mkdtemp()
        clog = os.path.join(tmp, 'clock.jsonl')
        dlog = os.path.join(tmp, 'deaths.jsonl')
        stub = _DriveStub()
        sub = _SubStub()
        c = _clock(drive=stub, sub=sub, log_path=clog, death_log_path=dlog)
        c.run(1)                               # life begins
        e = _fcc_edge('d1', 'x')
        e.reinforce(2, origin='cognition')
        c.run(2)                               # settle drains into E
        self.assertGreater(c.E(), 0.0)
        self.assertEqual(_n(c, 'internal_fcc'), 1)
        kh = MortalityClock._khash(('d1', 'causes', 'x'))

        stub.deaths = 1                        # the drive registers a death
        c.run(3)

        # Death record with the distillate, in BOTH files.
        self.assertTrue(os.path.exists(dlog))
        with open(dlog, 'r', encoding='utf-8') as f:
            recs = [json.loads(ln) for ln in f if ln.strip()]
        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec['kind'], 'death')
        dist = rec['distillate']
        for k in ('life_split', 'E_at_death', 'wall', 'lifeforce',
                  'ledger_size', 'dormant_count', 'deaths_total'):
            self.assertIn(k, dist)
        self.assertGreater(dist['E_at_death'], 0.0)
        self.assertEqual(dist['deaths_total'], 1)
        self.assertEqual(
            dist['life_split']['internal_fcc']['cognition']['n'], 1)
        with open(clog, 'r', encoding='utf-8') as f:
            kinds = [json.loads(ln).get('kind') for ln in f if ln.strip()]
        self.assertIn('death', kinds)

        # E reset, per-life split reset, LEDGER INTACT.
        self.assertEqual(c.E_settle, 0.0)
        self.assertEqual(int(sum(
            c._settle_life[cls][oc]['n']
            for cls in c._settle_life for oc in c._settle_life[cls])), 0)
        self.assertIn(kh, c._settled)
        self.assertEqual(_n(c, 'internal_fcc'), 1)   # all-time survives
        self.assertEqual(c.deaths_observed, 1)

    def test_boot_with_historical_deaths_writes_no_phantom_record(self):
        """Regression (caught by the promotion smoke 2026-07-18): the clock
        is constructed BEFORE load_personality restores the drive's
        persisted deaths counter; the first run must adopt that history as
        baseline, NOT register a phantom death into the succession chain."""
        tmp = tempfile.mkdtemp()
        dlog = os.path.join(tmp, 'deaths.jsonl')
        stub = _DriveStub()
        c = _clock(drive=stub, death_log_path=dlog)
        stub.deaths = 11             # a loaded save restores history
        stub.revivals = 4
        c.run(1)                     # boot: must NOT register a death
        self.assertEqual(c.deaths_observed, 0)
        self.assertEqual(c.revives_observed, 0)
        self.assertFalse(os.path.exists(dlog))
        # A LIVE death after boot still registers exactly once.
        stub.deaths = 12
        c.run(2)
        self.assertEqual(c.deaths_observed, 1)
        self.assertTrue(os.path.exists(dlog))
        with open(dlog, 'r', encoding='utf-8') as f:
            recs = [json.loads(ln) for ln in f if ln.strip()]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]['distillate']['deaths_total'], 12)


# =====================================================================
# 10. dead-save load (no 'mortality_clock' key)
# =====================================================================
class TestOldSaveLoads(_ClockTestBase):

    def test_brain_load_without_clock_key_is_clean(self):
        tmp = tempfile.mkdtemp()
        os.environ['SEAGI_CLOCK_LOG'] = os.path.join(tmp, 'clock.jsonl')
        os.environ['SEAGI_DEATH_LOG'] = os.path.join(tmp, 'deaths.jsonl')
        try:
            from seagi.body.engine import Engine
            from seagi.brain.runtime import Brain
            brain = Brain(engine=Engine())
            self.assertIsNotNone(brain.mortality_clock)
            brain.load_personality({})           # old save: no clock key
            self.assertEqual(len(brain.mortality_clock._settled), 0)
            d = brain.to_dict()
            self.assertIn('mortality_clock', d)
            self.assertEqual(d['mortality_clock']['key_digest'],
                             'blake2b8/fam2i')
            # And the status block surfaces the clock + observer_errors.
            st = brain.status()
            self.assertIn('mortality_clock', st)
            self.assertIn('observer_errors', st['mortality_clock'])
            self.assertNotIn('mortality_clock_shadow', st)
        finally:
            os.environ.pop('SEAGI_CLOCK_LOG', None)
            os.environ.pop('SEAGI_DEATH_LOG', None)


# =====================================================================
# 11. wall-migration sanity
# =====================================================================
class TestWallMigration(_ClockTestBase):

    def test_externally_set_wall_survives_load(self):
        body = _Body(0.7)
        _, md1 = _drive(body, [0])
        md1.tick()
        md1.wall = 0.4321          # externally set (migration)
        d = json.loads(json.dumps(md1.to_dict()))
        _, md2 = _drive(_Body(0.7), [0])
        md2.load_dict(d)
        self.assertAlmostEqual(md2.wall, 0.4321)
        self.assertFalse(md2.frozen)


# =====================================================================
# 12. not-a-scheduler
# =====================================================================
class TestNotAScheduler(_ClockTestBase):

    def test_wall_strictly_increases_with_zero_settlings(self):
        body = _Body(0.9)
        cyc = [0]
        _, md = _drive(body, cyc)
        clock = _clock(drive=md)
        md.set_clock_providers(E_provider=clock.E, cf_provider=clock.cf)
        md.tick()                   # seed the wall
        walls = [md.wall]
        for i in range(100):
            cyc[0] += 1
            md.tick()
            clock.run(cyc[0])       # zero settlings every tick
            walls.append(md.wall)
        for a, b in zip(walls, walls[1:]):
            self.assertGreater(b, a)


if __name__ == '__main__':
    unittest.main()
