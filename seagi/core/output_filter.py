"""Concept-name appropriateness filter.

Tells the thalamic gate (and any future surface-output code)
whether a given concept name is suitable to treat as a focal /
surface to the user.  The substrate may legitimately contain
stopwords, HTML fragments, numeric tokens, and contraction
fragments as concept nodes (they're indexable, edges still
form), but they should NOT become attention focals or appear
in voice — that produces word salad.

Extracted from agi_engine/meaning_activation.py during the
v1/v2 unification (2026-05-14, Session 6).  The original
meaning_activation.py was dead cognitive code; this filter
was the one piece v2 brain genuinely needed.
"""

from __future__ import annotations


# Filler words: legitimate English but never useful as
# substrate focals.  Includes question-words, articles,
# pronouns, common verbs, discourse connectives, contraction
# fragments, and so on.
_FILLER_WORDS = frozenset({
    'what', 'who', 'where', 'when', 'why', 'how',
    'which', 'whose', 'is', 'are', 'do', 'does', 'did',
    'tell', 'told', 'tells', 'telling',
    'describe', 'explain', 'show', 'give', 'given',
    'know', 'knew', 'known', 'understand', 'think',
    'thinks', 'thought', 'thinking', 'about',
    'me', 'you', 'us', 'them', 'i', 'we', 'he', 'she', 'they',
    'a', 'an', 'the', 'this', 'that', 'these', 'those',
    'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to', 'for',
    'be', 'been', 'being',
    'near', 'with', 'without', 'against', 'beside',
    'vs', 'versus',
    'around', 'between', 'inside', 'outside', 'over', 'under',
    'just', 'like', 'even', 'still', 'only', 'also', 'too',
    'actually', 'really', 'very', 'quite', 'rather',
    'normal', 'potentially', 'possibly', 'maybe', 'perhaps',
    'something', 'nothing', 'anything', 'everything',
    'someone', 'anyone', 'everyone', 'no one', 'nobody',
    'somewhere', 'anywhere', 'everywhere',
    'somehow', 'anyhow',
    'thing', 'things', 'stuff', 'way', 'ways', 'lot', 'lots',
    'one', 'two', 'three', 'first', 'second', 'last', 'next',
    'much', 'many', 'few', 'most', 'some', 'any', 'all',
    'every', 'each', 'both', 'either', 'neither',
    'such', 'same', 'other', 'another',
    'will', 'would', 'shall', 'should', 'can', 'could',
    'may', 'might', 'must', 'have', 'has', 'had',
    'was', 'were', 'am',
    'so', 'because', 'since', 'while', 'until', 'before',
    'after', 'during', 'though', 'although', 'however',
    'therefore', 'thus', 'here', 'there', 'now', 'then',
    'today', 'yesterday', 'tomorrow', 'never', 'always',
    'often', 'sometimes', 'usually', 'rarely', 'ever',
    'no', 'yes', 'yeah', 'okay', 'ok',
    'hi', 'hello', 'hey', 'bye', 'goodbye',
    'morning', 'evening', 'afternoon', 'night',
    'please', 'thanks', 'thank',
    'self', 'myself', 'yourself', 'himself', 'herself',
    'itself', 'themselves', 'ourselves',
    'my', 'mine', 'your', 'yours', 'his', 'hers', 'its',
    'our', 'ours', 'their', 'theirs',
    'get', 'got', 'getting', 'gets',
    'say', 'says', 'said', 'saying',
    'make', 'made', 'makes', 'making',
    'take', 'took', 'taken', 'takes', 'taking',
    'come', 'came', 'comes', 'coming',
    'go', 'went', 'gone', 'goes', 'going',
    'want', 'wants', 'wanted',
    'put', 'puts', 'putting',
    'use', 'used', 'uses', 'using',
    'try', 'tried', 'tries', 'trying',
    'work', 'works', 'worked', 'working',
    'seem', 'seems', 'seemed',
    'look', 'looks', 'looked',
    'feel', 'feels', 'felt',
    'far', 'close', 'back', 'forward',
    'long', 'short', 'big', 'small', 'high', 'low',
    'good', 'bad', 'better', 'worse', 'best', 'worst',
    'don', 'won', 'ain', 'isn', 'aren', 'wasn', 'weren',
    'didn', 'doesn', 'hasn', 'haven', 'hadn',
    'couldn', 'wouldn', 'shouldn', 'mustn', 'mightn',
    'let', 'lets', "let's",
})


# Short tokens that ARE legitimate (3-char names, etc.) —
# the length-3 cutoff below would otherwise reject them.
_SHORT_TOKEN_WHITELIST = frozenset({
    'm', 'i',
    'ai',
})


# Concept names produced by procedural / forager paths that
# look like real concepts but shouldn't surface.
_PROCEDURAL_NAME_PREFIXES = (
    'formed_bundle_',
    'formed_cluster_',
    'bundle_',
    'cluster_',
    '_invented_',
    'invented_',
    'compound_',
    'auto_',
    'tmp_',
)


def _is_procedural_name(name: str) -> bool:
    """True for concept names that look procedural / generated /
    HTML-fragment.  Used to filter out junk concepts from being
    treated as focals."""
    if not name:
        return True
    if name.startswith('_'):
        return True
    for p in _PROCEDURAL_NAME_PREFIXES:
        if name.startswith(p):
            return True
    for ch in ('<', '>', '&'):
        if ch in name:
            return True
    if name.count(';') >= 2:
        return True
    if len(name) > 40:
        return True
    if name:
        non_alnum = sum(
            1 for c in name if not c.isalnum() and c != '_')
        if non_alnum / len(name) > 0.30:
            return True
    return False


def _is_output_appropriate(name: str) -> bool:
    """Whether a concept name is appropriate to surface or
    treat as a focal.  Combines procedural-name, filler-word,
    digit, and short-token filters."""
    if not name:
        return False
    s = str(name).strip()
    if not s:
        return False
    lowered = s.lower()
    if _is_procedural_name(s):
        return False
    if lowered in _FILLER_WORDS:
        return False
    if s.isdigit():
        return False
    if len(s) <= 6:
        digit_ratio = (
            sum(1 for c in s if c.isdigit())
            / max(1, len(s)))
        if digit_ratio > 0.5:
            return False
    if len(s) < 3 and lowered not in _SHORT_TOKEN_WHITELIST:
        return False
    if not any(c.isalpha() for c in s):
        return False
    return True
