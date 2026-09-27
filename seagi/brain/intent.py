"""Intent classification — speech-act detection for incoming text.

Phase F.2 (2026-05-16).  Doctrine: "everything needs to be wired."
The intent of a peer utterance shapes which cognitive handler the
brain dispatches.  Currently `Brain.chat()` sends every input
through the same cortical `_think_about` path, which works well
for "what is X" but produces awkward output for introspective,
counterfactual, yes/no, and statement intents.

What this module is
-------------------
A small, regex-based classifier.  Not ML — we don't need it.  The
universal-survival speech acts have stable surface patterns that
a sub-100-line classifier handles correctly.  Future versions can
swap in learning-based intent classification without changing the
caller contract.

Speech-act taxonomy
-------------------
QUESTION_FACTUAL          "what is X", "tell me about X"
QUESTION_INTROSPECTIVE    "how do you feel", "what are you thinking"
QUESTION_COUNTERFACTUAL   "what if X", "what would happen if..."
QUESTION_WHY              "why X", "why does Y..."
QUESTION_HOW              "how X", "how does Y..."
QUESTION_YES_NO           "is X Y?", "do you Y?"
STATEMENT                 "X is Y", "Y happened"
GREETING                  "hi", "hello"
AGREEMENT                 "yes", "right"
DISAGREEMENT              "no", "wrong"
UNKNOWN                   uncategorizable input

The classifier returns an `IntentResult` carrying the kind plus
extracted focal/target hints.  Downstream handlers consume the
hints to focus their substrate walks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


class Intent:
    QUESTION_FACTUAL = 'question_factual'
    QUESTION_INTROSPECTIVE = 'question_introspective'
    QUESTION_COUNTERFACTUAL = 'question_counterfactual'
    QUESTION_WHY = 'question_why'
    QUESTION_HOW = 'question_how'
    QUESTION_YES_NO = 'question_yes_no'
    STATEMENT = 'statement'
    GREETING = 'greeting'
    AGREEMENT = 'agreement'
    DISAGREEMENT = 'disagreement'
    UNKNOWN = 'unknown'


@dataclass
class IntentResult:
    kind: str
    focal: str = ''        # primary subject of the utterance
    target: str = ''       # secondary subject (for yes/no, relations)
    raw: str = ''          # original input (lowercased, stripped)


# Stopwords that get filtered when extracting focal hints.
_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'do', 'does', 'did', 'doing',
    'have', 'has', 'had', 'having',
    'will', 'would', 'should', 'could', 'shall', 'may', 'might', 'must',
    'can', 'cannot',
    'about', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
    'as', 'so', 'and', 'or', 'but', 'if', 'because',
    'me', 'you', 'i', 'we', 'us', 'they', 'them', 'he', 'she', 'it',
    'my', 'your', 'our', 'their', 'his', 'her', 'its',
    'tell', 'know', 'understand', 'explain', 'describe',
    'this', 'that', 'these', 'those',
    'some', 'any', 'all', 'much', 'many', 'few',
})


# Surface-form sentinels.  Order matters: more-specific patterns
# checked before more-general.
_GREETING = frozenset({
    'hi', 'hello', 'hey', 'greetings', 'good morning',
    'good afternoon', 'good evening',
})
_AGREEMENT = frozenset({
    'yes', 'yeah', 'yep', 'right', 'correct', 'exactly', 'sure',
})
_DISAGREEMENT = frozenset({
    'no', 'nope', 'wrong', 'incorrect', 'never',
})

_INTROSPECTIVE_PATTERNS = (
    r'^how (do|are) you',
    r'^what are you (thinking|feeling)',
    r'^what do you feel',
    r'^what.+(your|you)\s+(mood|state|feeling|thought|tone)',
    r'^do you feel',
)

_COUNTERFACTUAL_PATTERNS = (
    r'^what if',
    r'^what would happen if',
    r'^suppose ',
    r'^imagine ',
)

_YES_NO_LEADS = ('is', 'are', 'was', 'were', 'do', 'does', 'did',
                    'can', 'could', 'will', 'would', 'should',
                    'have', 'has', 'had', 'am')


def classify_intent(text: str) -> IntentResult:
    """Classify an input string by speech act.  Returns an
    IntentResult with `kind`, optional `focal`, and `target` hints.
    """
    if not text:
        return IntentResult(kind=Intent.UNKNOWN, raw='')
    t = text.strip().lower().rstrip('?.! ')
    raw = t
    if not t:
        return IntentResult(kind=Intent.UNKNOWN, raw=text)

    # Greetings (whole-utterance match)
    if t in _GREETING:
        return IntentResult(kind=Intent.GREETING, raw=raw)
    # Multi-word greetings ('good morning' etc.) — check prefix
    for g in _GREETING:
        if ' ' in g and t.startswith(g):
            return IntentResult(kind=Intent.GREETING, raw=raw)

    # Single-word agreement/disagreement
    if t in _AGREEMENT:
        return IntentResult(kind=Intent.AGREEMENT, raw=raw)
    if t in _DISAGREEMENT:
        return IntentResult(kind=Intent.DISAGREEMENT, raw=raw)

    # Introspective questions ("how do you feel" etc.)
    for pat in _INTROSPECTIVE_PATTERNS:
        if re.match(pat, t):
            return IntentResult(
                kind=Intent.QUESTION_INTROSPECTIVE,
                focal='self', raw=raw)

    # Counterfactual questions ("what if...")
    for pat in _COUNTERFACTUAL_PATTERNS:
        if re.match(pat, t):
            focal = _extract_focal_after_marker(
                t, markers=('what if', 'suppose', 'imagine',
                              'what would happen if'))
            return IntentResult(
                kind=Intent.QUESTION_COUNTERFACTUAL,
                focal=focal, raw=raw)

    # Why / how questions
    if t.startswith('why '):
        focal = _extract_focal_after_marker(t, markers=('why',))
        return IntentResult(
            kind=Intent.QUESTION_WHY, focal=focal, raw=raw)
    if t.startswith('how '):
        focal = _extract_focal_after_marker(t, markers=('how',))
        return IntentResult(
            kind=Intent.QUESTION_HOW, focal=focal, raw=raw)

    # Yes/no questions — leading auxiliary verb + ends with '?'
    # or no inverted-statement form.
    looks_yes_no = (
        '?' in text
        and any(t.startswith(lead + ' ') for lead in _YES_NO_LEADS))
    if looks_yes_no:
        focal, target = _extract_yes_no_pair(t)
        return IntentResult(
            kind=Intent.QUESTION_YES_NO,
            focal=focal, target=target, raw=raw)

    # Factual question patterns ("what is X", "tell me about X")
    factual_prefixes = (
        'what is', "what's", 'what are', 'what was',
        'tell me about', 'describe', 'what do you know about',
    )
    for prefix in factual_prefixes:
        if t.startswith(prefix):
            focal = _extract_focal_after_marker(t, markers=(prefix,))
            return IntentResult(
                kind=Intent.QUESTION_FACTUAL,
                focal=focal, raw=raw)
    # Bare "what" / "where" / "when" / "who" / "which" questions
    if '?' in text and t.split()[0] in (
            'what', 'where', 'when', 'who', 'which'):
        focal = _extract_focal_after_marker(t, markers=(t.split()[0],))
        return IntentResult(
            kind=Intent.QUESTION_FACTUAL, focal=focal, raw=raw)

    # Otherwise — declarative statement.
    focal, target = _extract_statement_pair(t)
    return IntentResult(
        kind=Intent.STATEMENT,
        focal=focal, target=target, raw=raw)


# ---------------------------------------------------------------------
# Focal extraction helpers
# ---------------------------------------------------------------------


def _extract_focal_after_marker(text: str,
                                  markers: Tuple[str, ...]) -> str:
    """Return the first content word after any of `markers`.
    Strips stopwords and punctuation."""
    for m in markers:
        if text.startswith(m):
            rest = text[len(m):].strip()
            for w in _tokenize(rest):
                if w.lower() not in _STOPWORDS:
                    return w
    # Fallback — first non-stopword anywhere
    for w in _tokenize(text):
        if w.lower() not in _STOPWORDS:
            return w
    return ''


def _extract_yes_no_pair(text: str) -> Tuple[str, str]:
    """Best-effort (focal, target) extraction from a yes/no
    question.  E.g. "is fire hot?" → ('fire', 'hot').
    """
    words = _tokenize(text)
    if not words:
        return ('', '')
    # Skip leading auxiliary verb.
    if words and words[0].lower() in _YES_NO_LEADS:
        words = words[1:]
    content = [w for w in words if w.lower() not in _STOPWORDS]
    if len(content) >= 2:
        return (content[0], content[1])
    if content:
        return (content[0], '')
    return ('', '')


def _extract_statement_pair(text: str) -> Tuple[str, str]:
    """Naive (subject, attribute) extraction from a statement.
    Returns (focal, target) — first two content words by default.
    """
    content = [w for w in _tokenize(text)
                 if w.lower() not in _STOPWORDS]
    if len(content) >= 2:
        return (content[0], content[1])
    if content:
        return (content[0], '')
    return ('', '')


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation split.  Lowercase."""
    out = []
    for raw in re.split(r'[\s,.;:!?\'"]+', text.strip().lower()):
        if raw:
            out.append(raw)
    return out
