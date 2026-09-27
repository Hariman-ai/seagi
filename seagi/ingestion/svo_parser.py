"""SVO triple parser — relational ingestion.

Extracts (subject, relation, object) triples from English
sentences during Layer 3 corpus ingestion.  The relations
become typed edges in substrate (causes, is_a, has_property,
enables, etc.), which cortical reasoning then walks during
thinking.

Why this matters
----------------
The foundational ingestion before this wiring produced 62K
concepts and 85K edges — but those edges were almost entirely
`co_occurs` (from hippocampal focal-centric consolidation).
Cortical reasoning walks `causes`, `leads_to`, `is_a`,
`has_property`, `requires`, etc.  Without those relations in
substrate, cortical falls through to "X sits in my substrate
but I have no strong thought to offer" — Seagi has rich
chemistry but no graph to compose thoughts from.

The SVO parser is the fix.  Every ingested sentence produces
typed edges in addition to the chemistry imprint.

What is NOT here
----------------
- v1's `M_SEED_WORDS` / `I_SEED_WORDS` seed lexicon is
  intentionally OMITTED.  The doctrine (see
  [[feedback_seagi_mortality_baseline]]) forbids innate
  concept tags on substrate.  Innate machinery responses
  live in `seagi.brain.innate_lexicon` and fire chemistry,
  not labels.  Substrate stays neutral; trace accumulates
  from chemistry.
- This is a PURE parser.  It does not mutate substrate.
  Mutation happens via SubstrateWriteQueuedEvent which the
  JournaledSubstrateWriter consumes.

Phase D.2c (2026-05-20) — substrate-quality upgrade
---------------------------------------------------
The substrate audit found the parser was the root cause of two
quality problems: 28% morphological redundancy (every surface
form became its own concept) and function-word hub pollution
(only 97 function-word concepts but they touched 27.5% of
edges).  Three fixes, all at parse time so re-ingestion produces
a clean substrate:

  1. LEMMATIZE subject / object / promoted-relation — surface
     forms collapse to base lemmas (running → run).
  2. FILTER function words — FUNCTION_WORDS (the comprehensive
     set) keeps 'therefore' / 'both' / 'upon' out of subject /
     object slots, so they never become concepts.
  3. REJECT junk — `_looks_substantive` drops HTML / markup /
     URL fragments before they pollute the substrate.

`parse_sentence` accepts an optional `lemmatizer`; re-ingestion
passes a vocabulary-aware one (D.3).  Default is the safe
vocabulary-free lemmatizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from seagi.core.text_io import (
    tokenize, _STOPWORDS, FUNCTION_WORDS, _looks_substantive)
from seagi.ingestion.lemmatizer import Lemmatizer, lemmatize as _safe_lemmatize


# Verb-to-canonical-relation mapping.  Single-token verbs only.
# Unknown verbs encountered during parsing get promoted to new
# relation types (the parser flags them in
# `ParseResult.promoted_relation`).
VERB_RELATION_MAP: dict = {
    # Identity / classification
    'is': 'is_a',
    'are': 'is_a',
    'was': 'is_a',
    'were': 'is_a',
    # Possession / property
    'has': 'has_property',
    'have': 'has_property',
    'had': 'has_property',
    'contains': 'has_part',
    'holds': 'has_part',
    # Causation
    'causes': 'causes',
    'cause': 'causes',
    'caused': 'causes',
    'creates': 'creates',
    'create': 'creates',
    'created': 'creates',
    'makes': 'creates',
    'make': 'creates',
    'produces': 'produces',
    'produce': 'produces',
    'produced': 'produces',
    'enables': 'enables',
    'enable': 'enables',
    'helps': 'enables',
    'help': 'enables',
    'leads': 'leads_to',
    'lead': 'leads_to',
    # Capability
    'can': 'can_do',
    'requires': 'requires',
    'require': 'requires',
    'needs': 'requires',
    'need': 'requires',
    # Threat / harm
    'kills': 'kills',
    'kill': 'kills',
    'killed': 'kills',
    'destroys': 'destroys',
    'destroy': 'destroys',
    'destroyed': 'destroys',
    'damages': 'damages',
    'damage': 'damages',
    'damaged': 'damages',
    'threatens': 'threatens',
    'threaten': 'threatens',
    'hurts': 'damages',
    'hurt': 'damages',
    # Growth / preservation
    'grows': 'grows',
    'grow': 'grows',
    'preserves': 'preserves',
    'preserve': 'preserves',
    'extends': 'extends',
    'extend': 'extends',
    'protects': 'preserves',
    'protect': 'preserves',
    # Cognitive
    'knows': 'knows',
    'know': 'knows',
    'understands': 'understands',
    'understand': 'understands',
    'remembers': 'remembers',
    'remember': 'remembers',
    'thinks': 'thinks',
    'think': 'thinks',
    # Lacks (negation marker)
    'lacks': 'lacks',
    'lack': 'lacks',
    # Similarity / opposition
    'resembles': 'similar',
    'resemble': 'similar',
    'opposes': 'opposite_of',
    'oppose': 'opposite_of',
}

# Articles to skip when finding the object after "is a X".
_ARTICLES = frozenset({'a', 'an', 'the'})

# Minimum length for a token to become a substrate concept.
_MIN_CONCEPT_LEN = 3


def _is_concept_candidate(token: str) -> bool:
    """Whether a token may fill a subject / object slot — i.e.
    become a substrate concept.  Rejects function words, junk,
    and too-short tokens.  Phase D.2c."""
    if not token or len(token) < _MIN_CONCEPT_LEN:
        return False
    if token in FUNCTION_WORDS:
        return False
    if not _looks_substantive(token):
        return False
    return True


@dataclass
class ParseResult:
    """One extracted (subject, relation, object) triple.

    `promoted_relation` is True when the verb wasn't in
    VERB_RELATION_MAP — the parser proposed the verb lemma
    itself as the relation name.  Downstream substrate code
    creates this relation on demand.
    """
    subject: str
    relation: str
    object: str
    raw_verb: str
    promoted_relation: bool = False


def parse_sentence(sentence: str,
                       lemmatizer: Optional[Lemmatizer] = None
                       ) -> List[ParseResult]:
    """Extract (subject, relation, object) triples from one
    sentence.

    Strategy:
      - Tokenize the sentence.
      - For each token that looks like a verb (in
        VERB_RELATION_MAP or has a verb-like suffix), find:
          - subject: last concept-candidate token before verb
          - object: first concept-candidate token after verb
      - "is/are/was/were X" with article ("is a X") →
        relation 'is_a'.
      - "is X" without article → relation 'has_property'.
      - Other known verbs → canonical mapped relation.
      - Other suspect-verbs → relation = lemmatized verb
        (promoted_relation = True).

    Phase D.2c: subject / object / promoted relation are
    lemmatized; function words and junk never fill a concept
    slot.  `lemmatizer` defaults to the safe vocabulary-free
    lemmatizer; re-ingestion passes a vocabulary-aware one.

    Returns possibly multiple triples per sentence (one per
    valid verb).
    """
    tokens = tokenize(sentence)
    if not tokens:
        return []
    lem = lemmatizer.lemmatize if lemmatizer is not None \
        else _safe_lemmatize
    triples: List[ParseResult] = []
    for i, tok in enumerate(tokens):
        if (tok not in VERB_RELATION_MAP
                and not _looks_verb_like(tok)):
            continue
        subj = _find_concept_token_before(tokens, i)
        if subj is None:
            continue
        canonical_rel = VERB_RELATION_MAP.get(tok, tok)
        obj, rel = _find_object_and_resolve_relation(
            tokens, i, canonical_rel)
        if obj is None:
            continue
        # Lemmatize the concept slots only.  Relations are a
        # separate, small namespace whose names follow an
        # inflected-verb convention (VERB_RELATION_MAP canonicals
        # and INVERSE_RELATIONS keys are both -s forms); lemmatizing
        # them would break that convention for no substrate-quality
        # gain (the audit's 28% redundancy was in concept slots,
        # not relations).
        subj_lemma = lem(subj)
        obj_lemma = lem(obj)
        promoted = tok not in VERB_RELATION_MAP
        # Post-lemma guards: a lemma can land on a function word
        # or fall below the concept-length floor.
        if not _is_concept_candidate(subj_lemma):
            continue
        if not _is_concept_candidate(obj_lemma):
            continue
        # Skip degenerate self-triples (after lemmatization —
        # 'dogs chase dog' would collapse).
        if subj_lemma == obj_lemma:
            continue
        triples.append(ParseResult(
            subject=subj_lemma,
            relation=rel,
            object=obj_lemma,
            raw_verb=tok,
            promoted_relation=promoted,
        ))
    return triples


def _looks_verb_like(token: str) -> bool:
    """Heuristic: tokens that probably function as verbs in
    'X verb Y' constructions.  Suffix-driven; misses some,
    catches some non-verbs.  Real POS tagging would refine
    but isn't required for the architectural fix.

    Phase D.2c: function words are never verbs — this blocks
    'upon' / 'thus' / 'whereas' from spawning garbage relations."""
    if len(token) < 3:
        return False
    if token in FUNCTION_WORDS:
        return False
    return (token.endswith('s')
            or token.endswith('ed')
            or token.endswith('ing'))


def _find_concept_token_before(
        tokens: List[str], i: int) -> Optional[str]:
    """Last concept-candidate token before index i — skips
    function words, articles, junk.  Returns None if none found.
    Phase D.2c (was _find_content_token_before, _STOPWORDS-only)."""
    for j in range(i - 1, -1, -1):
        t = tokens[j]
        if t in _ARTICLES:
            continue
        if _is_concept_candidate(t):
            return t
    return None


def _find_object_and_resolve_relation(
        tokens: List[str],
        verb_idx: int,
        canonical_rel: str
        ) -> Tuple[Optional[str], str]:
    """Resolve (object, final_relation) for a verb at
    position verb_idx.

    Special handling for 'is/are/was/were':
      'is a X' / 'is an X' / 'is the X' → ('X', 'is_a')
      'is X' (no article) → ('X', 'has_property')
    Other verbs: object is the next content token; relation
    stays canonical_rel.
    """
    j = verb_idx + 1
    saw_article = False
    while j < len(tokens):
        t = tokens[j]
        if t in _ARTICLES:
            saw_article = True
            j += 1
            continue
        # Phase D.2c: skip function words / junk; the first real
        # concept-candidate token is the object.
        if not _is_concept_candidate(t):
            j += 1
            continue
        if canonical_rel == 'is_a' and not saw_article:
            return t, 'has_property'
        return t, canonical_rel
    return None, canonical_rel
