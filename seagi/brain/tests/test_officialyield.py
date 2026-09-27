"""Patch 42 OFFICIALYIELD: the pick yield reads official levels + headroom."""
import json
import os
from types import SimpleNamespace as NS

import pytest

import seagi.world.curriculum_world as cw
from seagi.world.curriculum_world import CurriculumWorld

# his persisted counters as measured 2026-09-12 (won_ever, steps_ever_here)
# and the official record of the same hour (levels cleared, levels total,
# official score).
ROWS = {
    'cd82': (2530, 198525, 6, 6, 100.0), 'lp85': (60, 140611, 1, 8, 2.78),
    'lf52': (43, 123786, 1, 10, 0.07), 'ft09': (252, 101332, 1, 6, 4.76),
    'm0r0': (22, 83890, 1, 6, 0.24), 'sk48': (0, 82979, 0, 8, 0.0),
    'vc33': (303, 79155, 1, 7, 0.21), 'r11l': (381, 77960, 1, 6, 4.76),
    'sp80': (418, 74686, 1, 6, 4.76), 'sb26': (0, 71223, 0, 8, 0.0),
    'cn04': (28, 64007, 1, 6, 2.17), 'ar25': (12, 56531, 1, 8, 0.48),
    'tr87': (0, 56308, 0, 6, 0.0), 'wa30': (0, 55873, 0, 9, 0.0),
    'ls20': (0, 54909, 0, 7, 0.0), 'su15': (0, 53429, 0, 9, 0.0),
    'g50t': (0, 53370, 0, 7, 0.0), 'ka59': (0, 52322, 0, 7, 0.0),
    'tn36': (0, 50553, 0, 7, 0.0), 'dc22': (0, 50503, 0, 6, 0.0),
    're86': (0, 50332, 0, 8, 0.0), 'sc25': (0, 49007, 0, 6, 0.0),
    'tu93': (0, 48634, 0, 9, 0.0), 's5i5': (0, 47222, 0, 8, 0.0),
    'bp35': (0, 46878, 0, 9, 0.0),
}
ORDER = list(ROWS)


def _cw(idx=0, rows=ROWS):
    c = object.__new__(CurriculumWorld)
    c._worlds = [NS(game_id=g + '-abcd1234', _won_ever=r[0],
                    _steps_ever_here=r[1]) for g, r in rows.items()]
    c._idx = idx
    return c


def _record(tmp_path, rows=ROWS):
    p = tmp_path / 'official_best.json'
    p.write_text(json.dumps({g: {'levels': r[2], 'baselines': [1] * r[3],
                                 'score': r[4]}
                             for g, r in rows.items()}))
    return str(p)


def _shares(ws, order=ORDER):
    t = sum(e for _, e in ws)
    return {order[j]: 100.0 * e / t for j, e in ws}


def test_gate_off_is_the_patch_36_formula(monkeypatch):
    monkeypatch.setattr(cw, '_OFFICIALYIELD_ON', lambda: False)
    c = _cw(idx=ORDER.index('bp35'))
    ws, cost, off = c._yield_weights()
    assert off is False
    tw = sum(r[0] for r in ROWS.values())
    ts = sum(r[1] for r in ROWS.values())
    assert cost == pytest.approx(ts / tw)
    for j, e in ws:
        g = ORDER[j]
        assert e == pytest.approx((ROWS[g][0] + 1.0) / (ROWS[g][1] + cost))
    assert all(j != ORDER.index('bp35') for j, _ in ws)
    sh = _shares(ws)
    assert 35 < sh['cd82'] < 45          # the capped game still dominates


def test_gate_off_log_line_is_unchanged(monkeypatch, capsys):
    monkeypatch.setattr(cw, '_OFFICIALYIELD_ON', lambda: False)
    monkeypatch.setattr(cw, '_PICK_RNG', NS(random=lambda: 0.5))
    c = _cw(idx=ORDER.index('bp35'))
    c._yield_pick()
    line = capsys.readouterr().err.strip().splitlines()[-1]
    assert line.startswith('[arcrotate] PICK game=')
    assert ' off=' not in line
    assert line.count('shares=') == 1 and not line.endswith(',')


def test_gate_on_capped_game_keeps_only_the_prior(monkeypatch, tmp_path):
    monkeypatch.setattr(cw, '_OFFICIALYIELD_ON', lambda: True)
    monkeypatch.setattr(cw, '_OFFICIAL_BEST', _record(tmp_path))
    c = _cw(idx=ORDER.index('bp35'))
    ws, cost, off = c._yield_weights()
    assert off is True
    ts = sum(r[1] for r in ROWS.values())
    assert cost == pytest.approx(ts / 15.0)        # 15 official levels
    # cd82: headroom 0 -> (0*6 + 1)/(steps + C): the prior alone
    j = ORDER.index('cd82')
    e_cd82 = dict(ws)[j]
    assert e_cd82 == pytest.approx(1.0 / (ROWS['cd82'][1] + cost))
    assert all(e > 0 for _, e in ws)               # no game has zero mass
    sh = _shares(ws)
    assert 1.0 < sh['cd82'] < 3.0
    never = [g for g, r in ROWS.items() if r[2] == 0 and g != 'bp35']
    assert 40 < sum(sh[g] for g in never) < 55
    assert sh['sk48'] > 2.0
    # a scored game keeps a (headroom*levels+1) edge over a never-won one
    assert sh['sp80'] > sh['sk48']
    # sp80 headroom is (100-4.76)/100
    j = ORDER.index('sp80')
    assert dict(ws)[j] == pytest.approx(
        ((100 - 4.76) / 100.0 * 1 + 1.0) / (ROWS['sp80'][1] + cost))


def test_gate_on_pick_lands_on_cd82_about_two_percent(monkeypatch, tmp_path):
    monkeypatch.setattr(cw, '_OFFICIALYIELD_ON', lambda: True)
    monkeypatch.setattr(cw, '_OFFICIAL_BEST', _record(tmp_path))
    c = _cw(idx=ORDER.index('bp35'))
    picks = []
    for k in range(400):
        monkeypatch.setattr(cw, '_PICK_RNG',
                            NS(random=lambda k=k: (k + 0.5) / 400.0))
        picks.append(ORDER[c._yield_pick()])
    assert 2 <= picks.count('cd82') <= 14        # ~2% of evenly spaced draws
    assert 'sk48' in picks and 'sp80' in picks
    assert len(set(picks)) == 24                 # every other game reachable
    assert c.yield_picks == 400


def test_gate_on_log_line_carries_off_and_age(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cw, '_OFFICIALYIELD_ON', lambda: True)
    monkeypatch.setattr(cw, '_OFFICIAL_BEST', _record(tmp_path))
    monkeypatch.setattr(cw, '_PICK_RNG', NS(random=lambda: 0.5))
    c = _cw(idx=ORDER.index('bp35'))
    c._yield_pick()
    line = capsys.readouterr().err.strip().splitlines()[-1]
    assert ' off=1 rec_age=' in line
    assert line.index('shares=') < line.index(' off=1')
    assert c._official_rec_age >= 0


def test_all_capped_every_game_keeps_the_prior(monkeypatch, tmp_path):
    rows = {'cd82': (10, 100, 6, 6, 100.0), 'ft09': (5, 300, 6, 6, 100.0),
            'bp35': (0, 50, 0, 9, 0.0)}
    monkeypatch.setattr(cw, '_OFFICIALYIELD_ON', lambda: True)
    monkeypatch.setattr(cw, '_OFFICIAL_BEST', _record(tmp_path, rows))
    c = _cw(idx=2, rows=rows)
    ws, cost, off = c._yield_weights()
    assert off is True
    assert [e for _, e in ws] == pytest.approx(
        [1.0 / (100 + cost), 1.0 / (300 + cost)])
    assert c._yield_pick() is not None


def test_half_written_record_keeps_the_last_good_read(monkeypatch, tmp_path):
    monkeypatch.setattr(cw, '_OFFICIALYIELD_ON', lambda: True)
    p = _record(tmp_path)
    monkeypatch.setattr(cw, '_OFFICIAL_BEST', p)
    c = _cw(idx=ORDER.index('bp35'))
    ws1, _, off1 = c._yield_weights()
    with open(p, 'w') as f:
        f.write('{"cd82": {"lev')            # the cron mid-write
    ws2, _, off2 = c._yield_weights()
    assert off1 is off2 is True
    assert ws1 == ws2


def test_no_record_ever_read_uses_the_old_formula(monkeypatch, tmp_path):
    monkeypatch.setattr(cw, '_OFFICIALYIELD_ON', lambda: True)
    monkeypatch.setattr(cw, '_OFFICIAL_BEST', str(tmp_path / 'missing.json'))
    c = _cw(idx=ORDER.index('bp35'))
    ws, cost, off = c._yield_weights()
    assert off is False
    assert _shares(ws)['cd82'] > 35


def test_no_official_level_anywhere_is_none(monkeypatch, tmp_path):
    rows = {'bp35': (3, 100, 0, 9, 0.0), 'dc22': (0, 50, 0, 6, 0.0)}
    monkeypatch.setattr(cw, '_OFFICIALYIELD_ON', lambda: True)
    monkeypatch.setattr(cw, '_OFFICIAL_BEST', _record(tmp_path, rows))
    c = _cw(idx=0, rows=rows)
    assert c._yield_weights() is None


def test_seagi_can_read_the_live_record():
    if not os.path.exists(cw._OFFICIAL_BEST):
        pytest.skip('no live record on this host')
    c = _cw()
    rec = c._official_record()
    # cd82 is his one complete game (09-11): every level cleared, score at the cap
    lv, n, sc = rec['cd82']
    assert n > 0 and lv == n and sc >= 100.0
    assert len(rec) == 25
    assert all(n > 0 and 0 <= lv <= n for lv, n, _ in rec.values())
