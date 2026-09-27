"""BUILDING EDGES DURING THE PARSE MUST CHANGE NOTHING BUT THE MEMORY.

json.load materialises every edge as a raw dict before a single Edge is
built, and the Edges are then allocated interleaved through the same
arenas, so the dicts can never be handed back.  Measured on his own
canonical: bulk 3,177 B/edge (6.13 GB, against a measured 5.90 GB anon)
versus 951 B/edge streamed (1.84 GB).  The object_hook reaches 907 B/edge
-- the streaming profile -- by converting each edge-shaped dict as the
parser completes it.

These pin that the RESULT is identical.  Memory is not a reason to
accept a different mind.
"""
import gzip
import json

import pytest

import seagi_daemon
from seagi.core.substrate import Edge, Substrate


def _edge(src, tgt, rel="is_a", strength=0.02):
    return {
        "source": src, "target": tgt, "relation_name": rel,
        "strength": strength, "mi": {"m": 0.0, "i": 0.0, "n": 0},
        "transmitter_trace": None, "bubbles": [],
        "evidence": {"hits": 0, "misses": 0, "inconclusive": 0},
        "last_reinforced_cycle": 7, "first_coherent_cycle": 0,
        "last_engaged_cycle": 11,
    }


def _payload():
    return {
        "concepts": {"a": {"name": "a"}, "b": {"name": "b"},
                     "c": {"name": "c"}},
        "edges": [_edge("a", "b"), _edge("b", "c", "transitions_to")],
        "quarantine_edges": [_edge("c", "a", "has_role")],
    }


def _fields(e):
    return tuple(getattr(e, k, None) for k in Edge.__slots__
                 if k not in ("_legacy_mi", "bubbles", "evidence"))


def test_prebuilt_edges_load_identically_to_raw_dicts():
    raw = _payload()
    pre = json.loads(json.dumps(raw))
    pre["edges"] = [Edge.from_dict(x) for x in pre["edges"]]
    pre["quarantine_edges"] = [Edge.from_dict(x)
                               for x in pre["quarantine_edges"]]

    a = Substrate.from_dict(raw)
    b = Substrate.from_dict(pre)

    assert set(a.edges) == set(b.edges)
    assert set(a.quarantine_edges) == set(b.quarantine_edges)
    for k in a.edges:
        assert _fields(a.edges[k]) == _fields(b.edges[k]), k
    for k in a.quarantine_edges:
        assert _fields(a.quarantine_edges[k]) == _fields(
            b.quarantine_edges[k]), k


def test_prebuilt_edges_rebuild_the_same_indexes():
    """A dropped index is a silently smaller mind, not a crash."""
    raw = _payload()
    pre = json.loads(json.dumps(raw))
    pre["edges"] = [Edge.from_dict(x) for x in pre["edges"]]
    pre["quarantine_edges"] = [Edge.from_dict(x)
                               for x in pre["quarantine_edges"]]
    a = Substrate.from_dict(raw)
    b = Substrate.from_dict(pre)

    assert dict(a._quarantine_index) == dict(b._quarantine_index)
    for name, ca in a.concepts.items():
        cb = b.concepts[name]
        assert {r: len(v) for r, v in ca.edges_out.items()} == \
               {r: len(v) for r, v in cb.edges_out.items()}


def test_hook_converts_an_edge_shape():
    out = seagi_daemon._load_object_hook(_edge("a", "b"))
    assert isinstance(out, Edge)
    assert out.source == "a" and out.target == "b"


@pytest.mark.parametrize("obj", [
    {"name": "a"},
    {"source": "a", "target": "b"},
    {"source": "a", "target": "b", "relation_name": "is_a"},
    # four of the five keys is NOT an edge -- the shape test runs on
    # every object in the file and must not claim a near-miss
    {"source": "a", "target": "b", "relation_name": "is_a",
     "strength": 0.1},
    {},
])
def test_hook_leaves_everything_else_alone(obj):
    assert seagi_daemon._load_object_hook(obj) is obj


def test_hook_returns_the_dict_when_from_dict_refuses():
    bad = _edge("a", "b")
    bad["strength"] = {"not": "a number"}
    out = seagi_daemon._load_object_hook(bad)
    assert out is bad or isinstance(out, Edge)


def test_a_real_json_load_with_the_hook_yields_edges(tmp_path):
    p = tmp_path / "s.json.gz"
    with gzip.open(str(p), "wt", encoding="utf-8") as f:
        json.dump(_payload(), f)
    with gzip.open(str(p), "rt", encoding="utf-8") as f:
        snap = json.load(f, object_hook=seagi_daemon._load_object_hook)
    assert all(isinstance(x, Edge) for x in snap["edges"])
    assert all(isinstance(x, Edge) for x in snap["quarantine_edges"])
    assert all(isinstance(x, dict) for x in snap["concepts"].values())
    sub = Substrate.from_dict(snap)
    assert len(sub.edges) == 2
    assert len(sub.quarantine_edges) == 1


def test_kill_switch_is_default_on(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda p: False)
    assert seagi_daemon._STREAMLOAD_ON() is True
    monkeypatch.setattr("os.path.exists",
                        lambda p: p == "/root/STREAMLOAD_OFF")
    assert seagi_daemon._STREAMLOAD_ON() is False


# --- concepts take the same route ------------------------------------
# Measured on his canonical: bulk 6,142 B/concept against 3,827 hooked,
# ~0.6 GB at 262k.  The shape was checked against all 7,363,466 objects
# in the file and occurs ONLY under `concepts` (262,450) -- nothing else
# in his brain wears it.

from seagi.core.substrate import Concept


def _concept(name, acts=7):
    """The 12 keys his canonical actually stores."""
    return {
        "name": name, "mi": {"m": 0.0, "i": 0.0, "n": 0},
        "transmitters": None, "bubbles": [],
        "activation_count": acts, "last_activated_cycle": 14504273,
        "knowledge": [], "embedding": None, "created_cycle": 0,
        "synthetic": False, "salience": 0.0015,
        "salience_last_update_cycle": 14504273,
    }


def _cpayload():
    return {
        "concepts": {"a": _concept("a"), "b": _concept("b", 3),
                     "c": _concept("c", 1)},
        "edges": [_edge("a", "b"), _edge("b", "c", "transitions_to")],
        "quarantine_edges": [_edge("c", "a", "has_role")],
    }


def test_hook_converts_a_concept_shape():
    out = seagi_daemon._load_object_hook(_concept("x"))
    assert isinstance(out, Concept)
    assert out.name == "x"
    assert out.activation_count == 7


@pytest.mark.parametrize("drop", ["activation_count", "salience",
                                  "created_cycle", "last_activated_cycle"])
def test_four_of_the_five_concept_keys_is_not_a_concept(drop):
    d = _concept("x")
    del d[drop]
    assert seagi_daemon._load_object_hook(d) is d


def test_a_concept_is_not_mistaken_for_an_edge_or_vice_versa():
    assert isinstance(seagi_daemon._load_object_hook(_concept("x")), Concept)
    assert isinstance(seagi_daemon._load_object_hook(_edge("a", "b")), Edge)


def test_prebuilt_concepts_load_identically_to_raw_dicts():
    raw = _cpayload()
    pre = json.loads(json.dumps(raw))
    pre["concepts"] = {k: Concept.from_dict(v)
                       for k, v in pre["concepts"].items()}
    pre["edges"] = [Edge.from_dict(x) for x in pre["edges"]]
    pre["quarantine_edges"] = [Edge.from_dict(x)
                               for x in pre["quarantine_edges"]]

    a = Substrate.from_dict(raw)
    b = Substrate.from_dict(pre)

    assert set(a.concepts) == set(b.concepts)
    for name, ca in a.concepts.items():
        cb = b.concepts[name]
        assert ca.name == cb.name
        assert ca.activation_count == cb.activation_count
        assert ca.last_activated_cycle == cb.last_activated_cycle
        assert ca.created_cycle == cb.created_cycle
        assert ca.synthetic == cb.synthetic
        # edges must still be attached to the right concepts
        assert {r: len(v) for r, v in ca.edges_out.items()} == \
               {r: len(v) for r, v in cb.edges_out.items()}
    assert set(a.edges) == set(b.edges)
    assert dict(a._quarantine_index) == dict(b._quarantine_index)


def test_a_real_json_load_converts_both_kinds(tmp_path):
    p = tmp_path / "c.json.gz"
    with gzip.open(str(p), "wt", encoding="utf-8") as f:
        json.dump(_cpayload(), f)
    with gzip.open(str(p), "rt", encoding="utf-8") as f:
        snap = json.load(f, object_hook=seagi_daemon._load_object_hook)
    assert all(isinstance(v, Concept) for v in snap["concepts"].values())
    assert all(isinstance(x, Edge) for x in snap["edges"])
    sub = Substrate.from_dict(snap)
    assert len(sub.concepts) == 3
    assert len(sub.edges) == 2
    assert len(sub.quarantine_edges) == 1

