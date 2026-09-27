"""Text I/O — encoder (text → engine) and decoder (engine → text).

Phase 1 surface:
    - tokenize: word-level, lowercase, strip punctuation
    - text_to_observation: tokenize → look up/create concepts in
      substrate → return observation dict ready for Engine.tick
    - proposition_to_text: render an (subject, relation, object) tuple
      as natural-ish language
    - action_to_text: render an Action — say/ask/acknowledge produce
      text; internal actions (rest/attend/probe/stay_silent) return None
    - classify_register: simple punctuation-driven register hint
      ('asking' / 'emphatic' / 'casual') for Layer 5 peer updates

This module deliberately uses NO knowledge tables, NO templates that
fake understanding. The audit forbade those (`_OPPOSITES`,
`_KNOWN_PROPERTIES`, etc. in meaning.py). What's here is pure
formatting: the engine commits to a proposition; the formatter renders
what was decided.

Phase 1 simplifications:
    - No semantic embedding lookup — token-string matches concept name
      directly. Phase 2 wires MiniLM for fuzzy matching.
    - No grammar inflection ("is a animal" not "is an animal"). Phase 2
      can add an articles/agreement pass.
    - No multi-sentence input — each text input is one observation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Hashable, Set, Tuple
import re

from .mi_value import MIValue
from .substrate import Substrate, Concept, EdgeKey


# Stopwords filtered from concept activations. We don't drop them
# from the surface text — only from what activates substrate. This
# is conservative; Phase 2 can tune.
_STOPWORDS: Set[str] = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'do', 'does', 'did', 'have',
    'has', 'had', 'will', 'would', 'shall', 'should', 'can', 'could',
    'may', 'might', 'must', 'i', 'you', 'he', 'she', 'it', 'we',
    'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
    'its', 'our', 'their', 'this', 'that', 'these', 'those', 'to',
    'of', 'in', 'on', 'at', 'by', 'for', 'with', 'as', 'so', 'not',
    'no', 'yes',
}

# Phase D.2b (2026-05-20): comprehensive function-word set.
# _STOPWORDS above is the small ingestion-era set (~60 words).
# The substrate audit found that function words SEAGI's reverie
# and cortical walks kept surfacing ('therefore', 'both', 'up',
# 'also', 'upon', 'whereas') were NOT in _STOPWORDS — so they
# became first-class substrate concepts and over-connected hub
# nodes (27.5% edge pollution).  FUNCTION_WORDS is the superset:
# articles, pronouns, auxiliaries, prepositions, conjunctions,
# determiners, quantifiers, conjunctive adverbs, particles.  Used
# by the improved SVO parser (D.2c) to keep function words out
# of the substrate at ingestion time.
FUNCTION_WORDS: Set[str] = _STOPWORDS | {
    # wh-words
    'which', 'what', 'who', 'whom', 'whose', 'when', 'where',
    'why', 'how', 'whether', 'whatever', 'whoever', 'whenever',
    'wherever', 'however',
    # demonstratives / locatives
    'there', 'here', 'thus', 'hence', 'therefore', 'whereas',
    'thereby', 'herein', 'thereof', 'therein', 'whereby',
    # subordinating / coordinating conjunctions
    'though', 'although', 'while', 'because', 'since', 'unless',
    'until', 'till', 'whereas', 'nor', 'either', 'neither',
    'whilst',
    # prepositions
    'upon', 'unto', 'into', 'onto', 'within', 'without', 'about',
    'above', 'below', 'beneath', 'between', 'among', 'amongst',
    'through', 'throughout', 'during', 'before', 'after',
    'against', 'toward', 'towards', 'beyond', 'behind', 'beside',
    'besides', 'despite', 'amid', 'amidst', 'per', 'via', 'down',
    'off', 'over', 'under', 'across', 'along', 'around', 'near',
    'onto', 'out', 'up',
    # quantifiers / determiners
    'both', 'all', 'some', 'many', 'much', 'more', 'most', 'few',
    'fewer', 'less', 'least', 'such', 'any', 'every', 'each',
    'other', 'another', 'same', 'own', 'enough', 'several',
    'various', 'certain', 'whole', 'half',
    # conjunctive / degree adverbs
    'again', 'also', 'too', 'yet', 'still', 'only', 'just',
    'even', 'ever', 'never', 'always', 'now', 'soon', 'well',
    'very', 'quite', 'rather', 'almost', 'indeed', 'perhaps',
    'maybe', 'else', 'otherwise', 'instead', 'meanwhile',
    'moreover', 'furthermore', 'nevertheless', 'nonetheless',
    'anyway', 'anyhow', 'somewhat', 'somehow', 'somewhere',
    'everywhere', 'nowhere', 'altogether', 'already', 'often',
    'sometimes', 'usually', 'merely', 'simply', 'truly',
    # number words
    'one', 'two', 'three', 'four', 'five', 'six', 'seven',
    'eight', 'nine', 'ten', 'first', 'second', 'third', 'last',
    'next', 'once', 'twice',
    # particles / misc function words
    'away', 'back', 'forth', 'aside', 'apart', 'together',
    'cannot', 'shall', 'ought', 'let', 'thy', 'thee', 'thou',
    'hath', 'doth', 'ye',
}

# Word minimum length for substrate activation — drops "a", "I", etc.
_MIN_TOKEN_LEN = 2

# Hard upper bound on a token's length when becoming a substrate
# concept.  Real words top out around 25 chars (e.g.
# 'antidisestablishmentarianism').  Anything longer is HTML, code,
# or a URL; reject it before it pollutes the substrate.
_MAX_TOKEN_LEN = 30

_PUNCT_STRIP = '.,;:!?"\'()[]{}'

# Token shapes that should never become a substrate concept.  These
# patterns matched the bulk of the pollution observed in the live
# substrate after weeks of forager activity: HTML tag fragments,
# MathJax markup, URLs, rich-text remnants.
_URL_PREFIXES = ('http://', 'https://', 'http', 'https',
                  'www.', 'ftp://', 'ftp', '//', './')

# Allowed non-alphanumeric characters in a substrate concept name.
# Lets through compound English words ('real-life', "doesn't") and
# canonical multi-token concept names ('co_occurs_with') while
# rejecting markup chars.
_TOKEN_ALLOWED_PUNCT = "_-'"


def _looks_substantive(token: str) -> bool:
    """Whether a token is a candidate to become a substrate concept.

    Rejects HTML / markup / URL / code-shaped tokens before they
    pollute the substrate.  Empirical pollution observed live:
    `</span></li>`, `>good`, `://example.com/path`, `mjx-line`,
    `</mjx-c></mjx-mi></mjx-texatom>`.  Each fails one of the
    checks below.
    """
    if not token:
        return False
    if len(token) > _MAX_TOKEN_LEN:
        return False
    # HTML / markup characters.
    for ch in ('<', '>', '&'):
        if ch in token:
            return False
    # URL-shaped tokens.
    for p in _URL_PREFIXES:
        if token.startswith(p):
            return False
    if '://' in token or token.startswith('//'):
        return False
    # Punctuation density: a real word has mostly alphanumerics,
    # plus a small number of allowed compound-word marks.  HTML
    # fragments and code have many slashes / colons / quotes.
    non_word = sum(
        1 for c in token
        if not c.isalnum() and c not in _TOKEN_ALLOWED_PUNCT)
    if non_word / len(token) > 0.30:
        return False
    return True


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Word-level tokenization. Lowercase, strip punctuation, split
    on whitespace. Empty tokens dropped."""
    if not text:
        return []
    cleaned = text.lower().strip()
    # Replace common punctuation with spaces so contractions don't merge.
    for ch in _PUNCT_STRIP:
        cleaned = cleaned.replace(ch, ' ')
    tokens = [t for t in cleaned.split() if t]
    return tokens


# Phase D.3b (2026-05-20): ingestion-lemmatizer hook.  When set,
# `content_tokens` lemmatizes each token AND filters the full
# FUNCTION_WORDS set — so a clean re-ingestion produces a
# substrate of base lemmas with no function-word concepts.  When
# None (the default, and always the case for live chat / forager
# / normal intake), `content_tokens` behaves EXACTLY as before —
# zero blast radius on runtime behavior.  The re-ingestion driver
# (D.3d) sets this around the rebuild pass and clears it after.
_ingestion_lemmatizer = None  # object with a .lemmatize(str)->str


def set_ingestion_lemmatizer(lem) -> None:
    """Install the lemmatizer used by `content_tokens` during a
    substrate rebuild.  Pass an object exposing
    `lemmatize(word) -> str` (seagi.ingestion.lemmatizer.Lemmatizer)."""
    global _ingestion_lemmatizer
    _ingestion_lemmatizer = lem


def clear_ingestion_lemmatizer() -> None:
    """Restore default `content_tokens` behaviour (no lemmatize,
    no FUNCTION_WORDS filtering)."""
    global _ingestion_lemmatizer
    _ingestion_lemmatizer = None


def content_tokens(tokens: List[str]) -> List[str]:
    """Filter to content words: not stopwords, length above floor,
    and substantive shape (no HTML / URL / markup fragments).  This
    is the choke point that protects the substrate from junk
    concepts regardless of input source.

    Phase D.3b: when an ingestion lemmatizer is installed, each
    surviving token is additionally lemmatized to its base form
    and the full FUNCTION_WORDS set is filtered — this is what
    makes a re-ingested substrate clean (base lemmas, no function
    words).  Default (lemmatizer unset): behaviour unchanged."""
    lem = _ingestion_lemmatizer
    if lem is None:
        return [t for t in tokens
                if len(t) >= _MIN_TOKEN_LEN
                and t not in _STOPWORDS
                and _looks_substantive(t)]
    # Re-ingestion path: lemmatize + full function-word filter.
    out: List[str] = []
    for t in tokens:
        if len(t) < _MIN_TOKEN_LEN:
            continue
        if t in FUNCTION_WORDS:
            continue
        if not _looks_substantive(t):
            continue
        lemma = lem.lemmatize(t)
        # The lemma can itself be a function word or fall below
        # the length floor — re-check after collapsing.
        if (not lemma or len(lemma) < _MIN_TOKEN_LEN
                or lemma in FUNCTION_WORDS):
            continue
        out.append(lemma)
    return out


# ---------------------------------------------------------------------
# Encoder — text to observation dict
# ---------------------------------------------------------------------

def text_to_observation(text: str,
                        substrate: Substrate,
                        create_missing: bool = True,
                        default_mi: Optional[MIValue] = None,
                        cycle: int = 0,
                        similarity_lookup: bool = False,
                        similarity_threshold: float = 0.7,
                        ) -> Dict[Hashable, MIValue]:
    """Convert input text to an observation dict for Engine.tick.

    Each content token activates a concept (creating it if missing
    and `create_missing` is True). The activation MI uses the
    concept's substrate MI if known; otherwise `default_mi` (None →
    MIValue.zero with n=0).

    `similarity_lookup`: when True AND a token has no exact match
    in the substrate, query the MiniLM bridge for the closest
    existing concept above `similarity_threshold`. If found, that
    existing concept is activated INSTEAD of creating a new one.
    This is how 'flame' resolves to 'fire' when fire is in substrate
    — the cutover criterion of comprehension across phrasings.

    Returns {concept_name: MIValue}. Empty dict if text has no
    content tokens.
    """
    tokens = content_tokens(tokenize(text))
    if not tokens:
        return {}
    obs: Dict[Hashable, MIValue] = {}
    for tok in tokens:
        existing = substrate.concepts.get(tok)
        if existing is not None:
            obs[tok] = existing.mi
            existing.activate(cycle)
            continue

        # Fuzzy match via semantic similarity if enabled.
        if similarity_lookup:
            matches = substrate.attend_to_text(
                tok, top_k=1, threshold=similarity_threshold)
            if matches:
                match_name, _sim = matches[0]
                match_concept = substrate.concepts.get(match_name)
                if match_concept is not None:
                    obs[match_name] = match_concept.mi
                    match_concept.activate(cycle)
                    continue

        if create_missing:
            new_concept = Concept(name=tok,
                                   mi=default_mi or MIValue.zero(),
                                   created_cycle=cycle)
            new_concept.activate(cycle)
            substrate.add_concept(new_concept)
            obs[tok] = new_concept.mi
    return obs


# ---------------------------------------------------------------------
# Decoder — proposition / action to text
# ---------------------------------------------------------------------

# Vowel-starting letters for a/an article agreement. Heuristic only —
# doesn't handle silent-h ("an honest answer") or u-as-y ("a unicorn"),
# but covers ~95% of cases for Phase 1.
_VOWEL_STARTS = {'a', 'e', 'i', 'o', 'u'}


def _article_for(word: str) -> str:
    """Pick 'a' or 'an' based on first letter of `word`."""
    if not word:
        return 'a'
    return 'an' if word[0].lower() in _VOWEL_STARTS else 'a'


# Per-relation surface phrasing. {obj} is always the object; {article}
# (when present) gets a/an based on the object's first letter. Unknown
# relations fall through to a humanized rendering ('is_a' → 'is a').
#
# This table is GRAMMAR, not knowledge. The engine has already decided
# what proposition to assert via EFE over substrate; this just renders
# it cleanly. It does not introduce information — different from the
# audit-flagged anti-pattern of `_OPPOSITES` etc. in meaning.py, which
# encoded WHAT to say. Here we only encode HOW to say what was decided.
_RELATION_PHRASE: dict = {
    'is_a':         'is {article} {obj}',
    'has_property': 'is {obj}',         # X is Y; Y is adjective
    'similar':      'is similar to {obj}',
    'opposite':     'is the opposite of {obj}',
    'part_of':      'is part of {obj}',
    'has_part':     'has {obj}',
    'causes':       'causes {obj}',
    'caused_by':    'is caused by {obj}',
    'used_for':     'is used for {obj}',
    'can_do':       'can {obj}',
    'enables':      'enables {obj}',
    'precedes':     'precedes {obj}',
    'follows':      'follows {obj}',
    'threatens':    'threatens {obj}',
    'destroys':     'destroys {obj}',
    'kills':        'kills {obj}',
    'damages':      'damages {obj}',
    'creates':      'creates {obj}',
    'grows':        'grows {obj}',
    'preserves':    'preserves {obj}',
    'extends':      'extends {obj}',
    'knows':        'knows {obj}',
    'understands':  'understands {obj}',
    'remembers':    'remembers {obj}',
    'lacks':        'lacks {obj}',
}


def _phrase_for_relation(rel: str, obj: str) -> str:
    """Render a relation's surface phrase with object inserted.
    Unknown relations (those promoted via scaffold extension) fall
    through to the humanized name."""
    template = _RELATION_PHRASE.get(rel)
    if template is None:
        return f"{rel.replace('_', ' ')} {obj}"
    if '{article}' in template:
        return template.format(article=_article_for(obj), obj=obj)
    return template.format(obj=obj)


def proposition_to_text(prop: EdgeKey,
                        question: bool = False) -> str:
    """Render a (subject, relation, object) tuple as a sentence.

    question=False → declarative ending in '.'
    question=True  → ending in '?'

    Examples:
        ('fire', 'causes', 'heat')        → 'Fire causes heat.'
        ('water', 'has_property', 'wet')  → 'Water is wet.'
        ('cat', 'is_a', 'animal')         → 'Cat is an animal.'
        ('healing', 'opposite', 'wound')  → 'Healing is the opposite of wound.'
    """
    if not isinstance(prop, tuple) or len(prop) != 3:
        return ''
    subj, rel, obj = prop
    if not subj or not obj:
        return ''
    phrase = _phrase_for_relation(str(rel), str(obj))
    body = f"{subj} {phrase}".strip()
    if not body:
        return ''
    body = body[0].upper() + body[1:]
    return body + ('?' if question else '.')


def action_to_text(action,
                   substrate: Optional[Substrate] = None
                   ) -> Optional[str]:
    """Convert an Action to its natural-language surface, if any.

    say / ask / acknowledge → text.
    rest / stay_silent / attend / probe / tools → None
        (these are internal or non-vocal; callers can render their
         own diagnostic text if desired).
    """
    kind = action.kind
    params = action.params

    if kind == 'say':
        prop = params.get('proposition')
        if prop is None:
            return None
        return proposition_to_text(prop, question=False)

    if kind == 'ask':
        prop = params.get('proposition')
        if prop is None:
            return None
        return proposition_to_text(prop, question=True)

    if kind == 'acknowledge':
        return 'Acknowledged.'

    # Internal / silent / rest / tool actions emit no surface speech
    # in Phase 1.
    return None


# ---------------------------------------------------------------------
# Register classification — light heuristic from punctuation
# ---------------------------------------------------------------------

def classify_register(text: str) -> Optional[str]:
    """Phase 1 register hint based on surface signals.

    Returns one of:
        'asking'   — text ends with '?'
        'emphatic' — text ends with '!'
        'casual'   — short, lowercase, no formal markers
        None       — no clear hint

    Layer 5 PeerModel.update_from_utterance accepts this as a hint.
    """
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.endswith('?'):
        return 'asking'
    if stripped.endswith('!'):
        return 'emphatic'
    if len(stripped) <= 60 and stripped == stripped.lower():
        return 'casual'
    return None
