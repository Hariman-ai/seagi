"""Scale indexes — fast lookups when the substrate has millions of
concepts/edges.

Three indexes:

  TokenIndex
      Inverted index: token → set of concept names whose names
      contain that token. Replaces O(N) "find concepts mentioned
      in this text" scans with O(tokens × matches).

  RelationIndex
      relation_name → list of edges. Replaces the O(E) full-scan
      that EdgesQuery does when filtering by relation alone.

  LSHIndex
      Random-projection locality-sensitive hashing for embedding
      cosine similarity. Replaces the O(N) cosine sweep in
      `attend_to_text` with O(buckets) candidate set followed by
      exact rescoring on the candidates.

These are PURE engineering — no ML libraries, no heuristics that
would change reasoning behavior. Same answers as full-scan, just
faster. The substrate maintains them as side-effects on
add_concept/add_edge and uses them in query paths.

Per the M/I vision: scale is a precondition for AGI but not the
intelligence itself. These indexes don't introduce learning, just
let the M/I machinery operate over a much larger substrate without
the per-tick cost growing linearly with substrate size.
"""

from __future__ import annotations
import math
import re
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------
# Tokenization — shared by TokenIndex
# ---------------------------------------------------------------------

# Tokens are alphanumeric runs lowercased. Underscores in concept
# names ("ice_cream") split into ["ice", "cream"] so a query
# mentioning "ice" still finds the compound.
_TOKEN_RE = re.compile(r'[A-Za-z0-9]+')


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens. Empty list on empty input."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# ---------------------------------------------------------------------
# TokenIndex — inverted index over concept names
# ---------------------------------------------------------------------

class TokenIndex:
    """token → set(concept_name).

    Built incrementally as concepts are added. Lookup by token or
    by free-form text (which tokenizes the input and unions matches).
    """

    def __init__(self) -> None:
        self._index: Dict[str, Set[str]] = {}
        # Reverse: name → its tokens, so removal is O(tokens) not O(N).
        self._name_to_tokens: Dict[str, List[str]] = {}

    def add(self, name: str) -> None:
        """Index a concept name. Idempotent."""
        if not name:
            return
        if name in self._name_to_tokens:
            return
        toks = tokenize(name)
        # Always include the full name as a token too (in case it
        # contains characters tokenize() drops, like dashes).
        full = name.lower()
        if full and full not in toks:
            toks = list(toks) + [full]
        self._name_to_tokens[name] = toks
        for t in toks:
            self._index.setdefault(t, set()).add(name)

    def remove(self, name: str) -> None:
        """Drop a concept from the index."""
        toks = self._name_to_tokens.pop(name, None)
        if not toks:
            return
        for t in toks:
            bucket = self._index.get(t)
            if bucket is None:
                continue
            bucket.discard(name)
            if not bucket:
                self._index.pop(t, None)

    def find_by_token(self, token: str) -> Set[str]:
        """All concept names containing the given token. Empty
        set if no match."""
        return set(self._index.get(token.lower(), ()))

    def find_by_text(self, text: str,
                       max_results: int = 1000) -> Set[str]:
        """Union of matches for each token in `text`. Capped to
        `max_results` to bound worst-case cost on common tokens."""
        out: Set[str] = set()
        for t in tokenize(text):
            bucket = self._index.get(t)
            if not bucket:
                continue
            out.update(bucket)
            if len(out) >= max_results:
                break
        return out

    def __len__(self) -> int:
        return len(self._name_to_tokens)


# ---------------------------------------------------------------------
# RelationIndex — relation_name → edges
# ---------------------------------------------------------------------

class RelationIndex:
    """relation_name → list of Edge objects, PLUS a
    (relation_name, target) → set(source) group index.

    EdgesQuery currently scans Substrate.edges.values() unconditionally;
    when the query filters by relation alone, this index gives a direct
    list. When filtering by source, callers should prefer
    `concept.edges_out[relation]` — that's already O(1).

    The (relation, target) → sources map (`_by_rel_target`, wake-
    consolidation refactor 2026-07-13) is the O(1) group lookup Phase-S
    abstraction formation needs when it runs SCOPED to the dirty set:
    `form_abstractions` groups sources by (relation, target), so given
    one dirty member's (relation, target) edge it can read the whole
    co-group in O(1) instead of re-scanning every edge. Maintained on
    the same add/remove call-sites as `_by_rel` (add_edge, _reap_edge,
    from_dict rebuild), so it stays consistent for free. PURE
    engineering — no new learning, no tunable constant.
    """

    def __init__(self) -> None:
        self._by_rel: Dict[str, List] = {}
        self._by_rel_target: Dict[Tuple[str, str], Set[str]] = {}

    def add(self, edge) -> None:
        rel = getattr(edge, 'relation_name', None)
        if not rel:
            return
        self._by_rel.setdefault(rel, []).append(edge)
        src = getattr(edge, 'source', None)
        tgt = getattr(edge, 'target', None)
        if src is not None and tgt is not None:
            self._by_rel_target.setdefault(
                (rel, tgt), set()).add(src)

    def remove(self, edge) -> None:
        rel = getattr(edge, 'relation_name', None)
        src = getattr(edge, 'source', None)
        tgt = getattr(edge, 'target', None)
        if rel is not None and src is not None and tgt is not None:
            bucket = self._by_rel_target.get((rel, tgt))
            if bucket is not None:
                bucket.discard(src)
                if not bucket:
                    self._by_rel_target.pop((rel, tgt), None)
        bucket = self._by_rel.get(rel) if rel else None
        if not bucket:
            return
        try:
            bucket.remove(edge)
        except ValueError:
            return
        if not bucket:
            self._by_rel.pop(rel, None)

    def get(self, relation_name: str) -> List:
        return list(self._by_rel.get(relation_name, ()))

    def iter_edges(self, relation_name: str) -> List:
        """Live (uncopied) edge list for a relation — read-only, for
        hot scans (analogy partner enumeration) that must not pay an
        O(bucket) copy per dirty concept."""
        return self._by_rel.get(relation_name, ())

    def bucket_size(self, relation_name: str) -> int:
        """O(1) count of edges under a relation (analogy picks the
        rarest relation in a skeleton to bound the partner scan)."""
        return len(self._by_rel.get(relation_name, ()))

    def sources_for_rel_target(self, relation_name: str,
                                 target: str) -> Set[str]:
        """The set of source concepts sharing (relation, target) — the
        abstraction co-group. Returns a COPY, so callers may mutate.

        NOTE the copy is O(n), not O(1) as this docstring used to claim.
        Read-only hot paths should use sources_view() instead."""
        return set(self._by_rel_target.get((relation_name, target), ()))

    def sources_view(self, relation_name: str, target: str):
        """READ-ONLY view of the same set — no copy, genuinely O(1).

        The facet paths (ignition, dock_observe, frontier ordering) call
        this once per facet-lock per percept and only iterate.  Once
        descriptions are written on ENCOUNTER a common lock carries
        thousands of sources, so copying per call dominates (measured:
        ignition suite 57s -> >400s).  CALLERS MUST NOT MUTATE the
        returned set.
        """
        return self._by_rel_target.get((relation_name, target), frozenset())

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_rel.values())


# ---------------------------------------------------------------------
# LSHIndex — random-projection LSH for cosine similarity
# ---------------------------------------------------------------------

class LSHIndex:
    """Random-projection LSH for L2-normalized vectors.

    Approach:
      - K hyperplanes per band, B bands. Each band hashes a vector
        to a K-bit signature; two vectors share a bucket in a band
        iff their K signs all match.
      - A query collects every concept that shares ANY band-bucket
        with it. That candidate set is then rescored with exact
        cosine to return top_k.
      - Recall improves with B, precision/cost with K.

    For substrates where concepts have semantic embeddings of varying
    dim, the index is dim-keyed. Concepts are indexed under the dim
    of their embedding; queries lookup against the matching dim.

    No ML libs — just Python lists and a deterministic RNG.
    """

    def __init__(self,
                 num_bands: int = 8,
                 hashes_per_band: int = 10,
                 seed: int = 0):
        self.num_bands = int(num_bands)
        self.hashes_per_band = int(hashes_per_band)
        self.seed = int(seed)
        # dim → planes (list of B lists of K vectors)
        self._planes: Dict[int, List[List[List[float]]]] = {}
        # dim → buckets per band: List[Dict[signature_int, List[name]]]
        self._buckets: Dict[int, List[Dict[int, List[str]]]] = {}
        # name → (dim, list of B signatures) so we can remove
        # cleanly even after vector changes.
        self._name_to_sig: Dict[str, Tuple[int, List[int]]] = {}

    # ---- internals ----

    def _ensure_planes(self, dim: int) -> None:
        if dim in self._planes:
            return
        rng = random.Random(self.seed ^ dim ^ 0x9E3779B1)
        bands: List[List[List[float]]] = []
        for _b in range(self.num_bands):
            band: List[List[float]] = []
            for _k in range(self.hashes_per_band):
                # Gaussian random hyperplane normal.
                v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
                norm = math.sqrt(sum(x * x for x in v)) or 1.0
                band.append([x / norm for x in v])
            bands.append(band)
        self._planes[dim] = bands
        self._buckets[dim] = [
            {} for _ in range(self.num_bands)]

    def _signatures(self, dim: int,
                      vec: Sequence[float]) -> List[int]:
        """K-bit sign signatures across B bands for `vec`."""
        bands = self._planes[dim]
        sigs: List[int] = []
        for band in bands:
            sig = 0
            for h in band:
                # sign of dot product → 1 bit.
                dot = 0.0
                for a, b in zip(vec, h):
                    dot += a * b
                if dot >= 0.0:
                    sig = (sig << 1) | 1
                else:
                    sig = (sig << 1)
            sigs.append(sig)
        return sigs

    # ---- public API ----

    def add(self, name: str, vec: Sequence[float]) -> None:
        """Index a vector under `name`. If the name was previously
        indexed, the old entry is replaced."""
        if not vec:
            return
        # Remove previous if any.
        if name in self._name_to_sig:
            self.remove(name)
        dim = len(vec)
        self._ensure_planes(dim)
        sigs = self._signatures(dim, vec)
        bucks = self._buckets[dim]
        for b, sig in enumerate(sigs):
            bucks[b].setdefault(sig, []).append(name)
        self._name_to_sig[name] = (dim, sigs)

    def remove(self, name: str) -> None:
        entry = self._name_to_sig.pop(name, None)
        if entry is None:
            return
        dim, sigs = entry
        bucks = self._buckets.get(dim)
        if bucks is None:
            return
        for b, sig in enumerate(sigs):
            bucket = bucks[b].get(sig)
            if not bucket:
                continue
            try:
                bucket.remove(name)
            except ValueError:
                pass
            if not bucket:
                bucks[b].pop(sig, None)

    def candidates(self, vec: Sequence[float]) -> Set[str]:
        """Names sharing at least one band-bucket with `vec`."""
        if not vec:
            return set()
        dim = len(vec)
        bands = self._planes.get(dim)
        if bands is None:
            return set()
        sigs = self._signatures(dim, vec)
        out: Set[str] = set()
        bucks = self._buckets[dim]
        for b, sig in enumerate(sigs):
            bucket = bucks[b].get(sig)
            if bucket:
                out.update(bucket)
        return out

    def __len__(self) -> int:
        return len(self._name_to_sig)


# ---------------------------------------------------------------------
# Helpers — exact cosine rescore over candidate set
# ---------------------------------------------------------------------

def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two L2-normalized vectors. If
    either is zero-norm or dims differ, returns 0."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
