"""MiniLM + n-gram fallback — semantic embedding backend.

Two-backend design (mirrors current SEAGI's embedding.py):

  MiniLM (preferred): real semantic encoder via sentence-transformers
    `all-MiniLM-L6-v2` (384-dim). "fire" and "flame" become neighbors;
    "fire" and "banana" don't. Loads lazily on first encode.

  N-gram (fallback): character 3-grams + word tokens hashed into a
    256-dim vector. Surface-form similarity only (fuzzy matching on
    morphology). No semantic understanding. Used when sentence-
    transformers isn't installed.

Downstream code calls `encode_text(text)` and `cosine(a, b)` without
knowing which backend is live. `is_minilm_available()` and
`backend_info()` let callers introspect.

Vector format: plain Python list of floats, L2-normalized. Bounded
in-memory cache (~20k entries) keeps repeated encodings free.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import math
import threading


# Backend dimensions.
_MINILM_DIM = 384
_NGRAM_DIM = 256

# Backend state.
_BACKEND: str = 'ngram'  # promoted to 'minilm' on successful load
_MINILM_MODEL = None
_LOAD_ATTEMPTED = False
_LOAD_ERROR: Optional[str] = None
_BACKEND_LOCK = threading.Lock()

# Bounded encoding cache.
_ENCODE_CACHE: Dict[str, List[float]] = {}
_ENCODE_CACHE_MAX = 20_000


# ---------------------------------------------------------------------
# Backend selection — lazy load
# ---------------------------------------------------------------------

def _load_minilm():
    """Lazy-load sentence-transformers MiniLM. Idempotent. Returns
    the model on success, None on failure (caller falls back to n-gram).
    """
    global _MINILM_MODEL, _LOAD_ATTEMPTED, _LOAD_ERROR, _BACKEND
    if _MINILM_MODEL is not None:
        return _MINILM_MODEL
    if _LOAD_ATTEMPTED:
        return None
    with _BACKEND_LOCK:
        if _MINILM_MODEL is not None:
            return _MINILM_MODEL
        if _LOAD_ATTEMPTED:
            return None
        _LOAD_ATTEMPTED = True
        try:
            from sentence_transformers import SentenceTransformer
            _MINILM_MODEL = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2')
            _BACKEND = 'minilm'
        except Exception as ex:
            _LOAD_ERROR = f'{type(ex).__name__}: {ex}'
            _MINILM_MODEL = None
    return _MINILM_MODEL


def is_minilm_available() -> bool:
    """True iff MiniLM has loaded successfully."""
    _load_minilm()
    return _MINILM_MODEL is not None


def get_dim() -> int:
    """Live backend's embedding dimension."""
    _load_minilm()
    return _MINILM_DIM if _BACKEND == 'minilm' else _NGRAM_DIM


def backend_info() -> Dict[str, Any]:
    """Introspection: which backend is live, dim, cache stats."""
    _load_minilm()
    return {
        'backend': _BACKEND,
        'dim': get_dim(),
        'cache_size': len(_ENCODE_CACHE),
        'cache_max': _ENCODE_CACHE_MAX,
        'load_error': _LOAD_ERROR,
    }


def reset_for_test() -> None:
    """Clear the cache (useful in tests). Backend remains as-is."""
    _ENCODE_CACHE.clear()


# ---------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------

def _zeros(dim: int) -> List[float]:
    return [0.0] * dim


def _normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n <= 0:
        return v
    return [x / n for x in v]


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity. Inputs assumed L2-normalized; this is a
    plain dot product. Length mismatch returns 0."""
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------

_PUNCT_STRIP = '.,!?;:"\'()[]{}'

_CHAR_N = 3
_WORD_WEIGHT = 2.0
_CHAR_WEIGHT = 1.0


def _hash_bucket(token: str, dim: int) -> int:
    """Stable in-process murmur-ish fold; reproducible within a
    session. Cross-session stability not needed since we recompute
    embeddings on substrate load."""
    h = 0
    for ch in token:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h % dim


def _ngram_encode(cleaned: str, dim: int) -> List[float]:
    """Stdlib fallback — character 3-grams + word tokens hashed
    into a normalized `dim`-vector."""
    vec = _zeros(dim)
    for word in cleaned.replace('\n', ' ').split():
        word = word.strip(_PUNCT_STRIP)
        if len(word) < 2:
            continue
        idx = _hash_bucket(f'w:{word}', dim)
        vec[idx] += _WORD_WEIGHT
    padded = f' {cleaned} '
    for i in range(len(padded) - _CHAR_N + 1):
        gram = padded[i:i + _CHAR_N]
        idx = _hash_bucket(f'c:{gram}', dim)
        vec[idx] += _CHAR_WEIGHT
    return _normalize(vec)


def encode_text(text: str) -> List[float]:
    """Encode text to a normalized vector via the live backend.
    Empty / None → zero vector."""
    if not text:
        return _zeros(get_dim())
    cleaned = str(text).lower().strip()
    if not cleaned:
        return _zeros(get_dim())

    cached = _ENCODE_CACHE.get(cleaned)
    if cached is not None:
        return cached

    model = _load_minilm()
    if model is not None and _BACKEND == 'minilm':
        try:
            vec_np = model.encode(
                cleaned, normalize_embeddings=True,
                show_progress_bar=False)
            vec = vec_np.tolist() if hasattr(vec_np, 'tolist') else list(vec_np)
        except Exception:
            vec = _ngram_encode(cleaned, _NGRAM_DIM)
    else:
        vec = _ngram_encode(cleaned, _NGRAM_DIM)

    # Bounded LRU-ish cache: drop half on overflow (insertion-ordered dict).
    if len(_ENCODE_CACHE) >= _ENCODE_CACHE_MAX:
        to_drop = _ENCODE_CACHE_MAX // 2
        for i, k in enumerate(list(_ENCODE_CACHE.keys())):
            if i >= to_drop:
                break
            del _ENCODE_CACHE[k]
    _ENCODE_CACHE[cleaned] = vec
    return vec


def encode_batch(texts: List[str],
                    batch_size: int = 64) -> List[List[float]]:
    """Batch-encode many texts at once.  10x+ faster than calling
    `encode_text` in a loop because the model processes a tensor
    of multiple inputs per forward pass instead of one at a time.

    Caches results so subsequent calls hit the cache.  Skips
    empty inputs (returns zero vectors).
    """
    if not texts:
        return []
    cleaned_list = [
        (str(t).lower().strip() if t else '') for t in texts]
    # Build the list of items we need to actually encode
    # (skip empties, skip cache hits).
    needed_idx: List[int] = []
    needed_clean: List[str] = []
    for i, c in enumerate(cleaned_list):
        if not c:
            continue
        if c in _ENCODE_CACHE:
            continue
        needed_idx.append(i)
        needed_clean.append(c)

    if needed_clean:
        model = _load_minilm()
        if model is not None and _BACKEND == 'minilm':
            try:
                # sentence-transformers batches efficiently.
                vecs_np = model.encode(
                    needed_clean,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                    convert_to_numpy=True)
                for j, c in enumerate(needed_clean):
                    v = vecs_np[j]
                    vec = v.tolist() if hasattr(v, 'tolist') else list(v)
                    _ENCODE_CACHE[c] = vec
            except Exception:
                # Fall back per-item to n-gram.
                for c in needed_clean:
                    _ENCODE_CACHE[c] = _ngram_encode(
                        c, _NGRAM_DIM)
        else:
            for c in needed_clean:
                _ENCODE_CACHE[c] = _ngram_encode(c, _NGRAM_DIM)

    # Assemble final output (in original order, including
    # empties → zero vectors and cache hits).
    out: List[List[float]] = []
    dim = get_dim()
    for c in cleaned_list:
        if not c:
            out.append(_zeros(dim))
        else:
            out.append(_ENCODE_CACHE.get(c, _zeros(dim)))
    # Cap cache as in encode_text.
    if len(_ENCODE_CACHE) >= _ENCODE_CACHE_MAX:
        to_drop = _ENCODE_CACHE_MAX // 2
        for i, k in enumerate(list(_ENCODE_CACHE.keys())):
            if i >= to_drop:
                break
            del _ENCODE_CACHE[k]
    return out
