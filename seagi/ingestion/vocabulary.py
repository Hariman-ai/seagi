"""Vocabulary builder — pass 1 of the D.3 substrate rebuild.

The vocabulary-aware lemmatizer only collapses a surface form to
a base when that base is itself an attested word (so 'computer'
never becomes 'comput').  "Attested" means: appears somewhere in
the corpus being ingested.

This module does pass 1 — scan the whole corpus, tokenize, and
return the set of every content token observed (raw surface
forms, NOT lemmatized — we want to know exactly what the corpus
contains).  Pass 2 (re-ingestion) then constructs a
`Lemmatizer(vocabulary=...)` from this set.

Cheap: one linear scan, tokenization only, no brain involved.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Set, Tuple

from seagi.core.text_io import tokenize, _looks_substantive


# Same sentence-irrelevant content gate the parser uses: a token
# is vocabulary-worthy when it is substantive and >= 3 chars.
# Function words are intentionally KEPT in the raw vocab scan —
# they're filtered later at ingest time; here we only want an
# honest picture of what surface forms exist.
_MIN_VOCAB_LEN = 3


def _iter_corpus_text(corpus_dir: Path,
                          file_pattern: str = '**/*.txt') -> Any:
    """Yield the text of every corpus file (txt + md), skipping
    sidecar marker files."""
    seen = set()
    for pat in (file_pattern, '**/*.md'):
        for path in sorted(corpus_dir.glob(pat)):
            if path in seen:
                continue
            seen.add(path)
            name = path.name
            if name.endswith('.read') or '.read' in name:
                continue
            try:
                raw = path.read_text(
                    encoding='utf-8', errors='replace')
            except Exception:
                continue
            # Strip HTML if it looks like markup (mirror driver).
            if re.search(r'<[A-Za-z/]', raw):
                from seagi.ingestion.driver import _strip_html
                raw = _strip_html(raw)
            yield raw


def build_vocabulary(corpus_dir: Any,
                         file_pattern: str = '**/*.txt'
                         ) -> Tuple[Set[str], Dict[str, int]]:
    """Pass 1.  Scan `corpus_dir` and return:

      (vocabulary, frequency)

    `vocabulary` — the set of content tokens (raw surface forms,
       >= 3 chars, substantive shape) that appear in the corpus.
    `frequency`  — token → occurrence count, for diagnostics and
       for tie-breaking if a caller wants it.

    The vocabulary is what the vocab-aware Lemmatizer collapses
    TO: 'designed' becomes 'design' only if 'design' is in here.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory does not exist: {corpus_dir}")
    freq: Dict[str, int] = {}
    for text in _iter_corpus_text(corpus_dir, file_pattern):
        for tok in tokenize(text):
            if len(tok) < _MIN_VOCAB_LEN:
                continue
            if not _looks_substantive(tok):
                continue
            freq[tok] = freq.get(tok, 0) + 1
    vocab = set(freq.keys())
    return vocab, freq


def vocabulary_summary(vocab: Set[str],
                            freq: Dict[str, int]) -> Dict[str, Any]:
    """Compact stats for logging a pass-1 scan."""
    if not freq:
        return {'unique_tokens': 0, 'total_tokens': 0}
    total = sum(freq.values())
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:15]
    return {
        'unique_tokens': len(vocab),
        'total_tokens': total,
        'top_15': top,
    }
