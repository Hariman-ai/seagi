"""A STREAMING SAVE MUST WRITE THE SAME BYTES.

`json.dumps` was fixed on 2026-08-20 so the JSON text is no longer built
as one string.  But `engine.to_dict()` still built EVERY edge and concept
as a raw dict before the encoder walked any of them.  Measured on his own
edges: **1,010 B/edge materialised, 0 streamed** -- a 2.00 GB transient
that took RSS from 3,541,700 kB to 5,911,144 every 600 s on a 7.75 GB
box.  Before the load fix, that same transient needed ~8.3 GB and could
not fit at all.

Memory is not a reason to write a different file.  These pin byte
equality, not merely "it still loads".
"""
import gzip
import json

import pytest

from seagi.core import persistence
from seagi.core.substrate import Concept, Edge, Substrate


def _edge(src, tgt, rel="is_a"):
    return {
        "source": src, "target": tgt, "relation_name": rel,
        "strength": 0.02, "mi": {"m": 0.0, "i": 0.0, "n": 0},
        "transmitter_trace": None, "bubbles": [],
        "evidence": {"hits": 1, "misses": 2, "inconclusive": 0},
        "last_reinforced_cycle": 7, "first_coherent_cycle": 3,
        "last_engaged_cycle": 11,
    }


def _concept(name, acts=7):
    return {
        "name": name, "mi": {"m": 0.0, "i": 0.0, "n": 0},
        "transmitters": None, "bubbles": [],
        "activation_count": acts, "last_activated_cycle": 14504273,
        "knowledge": [], "embedding": None, "created_cycle": 0,
        "synthetic": False, "salience": 0.0015,
        "salience_last_update_cycle": 14504273,
    }


def _sub():
    return Substrate.from_dict({
        "concepts": {n: _concept(n, i + 1)
                     for i, n in enumerate(("a", "b", "c"))},
        "edges": [_edge("a", "b"), _edge("b", "c", "transitions_to")],
        "quarantine_edges": [_edge("c", "a", "has_role")],
    })


def _dump(d, lazy):
    return json.dumps(d, separators=(",", ":"),
                      default=persistence._save_default if lazy else str)


def test_lazy_and_materialised_serialise_to_identical_bytes():
    sub = _sub()
    assert _dump(sub.to_dict(lazy=True), True) == \
           _dump(sub.to_dict(lazy=False), False)


def test_lazy_leaves_objects_in_place():
    """If it quietly materialised anyway the memory win would be a lie."""
    d = _sub().to_dict(lazy=True)
    assert all(isinstance(x, Edge) for x in d["edges"])
    assert all(isinstance(x, Edge) for x in d["quarantine_edges"])
    assert all(isinstance(v, Concept) for v in d["concepts"].values())


def test_the_default_is_still_materialised():
    """Every other caller must be byte-identical to before."""
    d = _sub().to_dict()
    assert all(isinstance(x, dict) for x in d["edges"])
    assert all(isinstance(v, dict) for v in d["concepts"].values())


def test_a_lazy_snapshot_round_trips_through_from_dict():
    sub = _sub()
    raw = json.loads(_dump(sub.to_dict(lazy=True), True))
    back = Substrate.from_dict(raw)
    assert set(back.concepts) == set(sub.concepts)
    assert set(back.edges) == set(sub.edges)
    assert set(back.quarantine_edges) == set(sub.quarantine_edges)
    for k, e in sub.edges.items():
        assert back.edges[k].strength == e.strength
        assert back.edges[k].first_coherent_cycle == e.first_coherent_cycle


def test_save_default_only_touches_edges_and_concepts():
    """Anything that reached str() before must still reach it."""
    class Odd:
        def to_dict(self):
            return {"should": "not be used"}

        def __str__(self):
            return "odd!"

    assert persistence._save_default(Odd()) == "odd!"
    assert isinstance(persistence._save_default(
        Edge.from_dict(_edge("a", "b"))), dict)
    assert isinstance(persistence._save_default(
        Concept.from_dict(_concept("a"))), dict)


def test_kill_switch(monkeypatch):
    monkeypatch.setattr(persistence.os.path, "exists", lambda p: False)
    assert persistence._STREAMSAVE_ON() is True
    monkeypatch.setattr(persistence.os.path, "exists",
                        lambda p: p == "/root/STREAMSAVE_OFF")
    assert persistence._STREAMSAVE_ON() is False


def test_save_brain_writes_identical_files_either_way(tmp_path,
                                                      monkeypatch):
    """End to end, through the real save_brain, gz and all."""
    from seagi.body.engine import Engine
    eng = Engine(substrate=_sub(), lifeforce=0.7, seed=0)

    monkeypatch.setattr(persistence, "_STREAMSAVE_ON", lambda: True)
    a = tmp_path / "a.json.gz"
    persistence.save_brain(eng, str(a))

    monkeypatch.setattr(persistence, "_STREAMSAVE_ON", lambda: False)
    b = tmp_path / "b.json.gz"
    persistence.save_brain(eng, str(b))

    with gzip.open(str(a), "rt", encoding="utf-8") as f:
        da = json.load(f)
    with gzip.open(str(b), "rt", encoding="utf-8") as f:
        db = json.load(f)
    da.pop("saved_at_iso", None)
    db.pop("saved_at_iso", None)
    assert da == db
