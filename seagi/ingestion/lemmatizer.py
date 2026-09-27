"""Rule-based lemmatizer — substrate-quality normalization.

Phase D.2a (2026-05-20).  The substrate audit found 28% of
content concepts are morphological variants of another concept
that already exists (dimensions/dimension, wronged/wrong,
slayers/slayer).  The SVO parser ingested every surface form as
its own concept.  This lemmatizer collapses surface forms to a
base lemma so re-ingestion produces ONE concept per lemma.

Design — why rule-based, not nltk/spaCy
---------------------------------------
- No new dependency; ingestion of 111K sentences stays fast.
- Doctrine: tight, not extravagant.
- It does not need to be linguistically perfect — it needs to
  collapse the obvious morphological redundancy the audit found.

Over-stripping guard — the vocabulary
-------------------------------------
Pure suffix-stripping over-strips: 'computer' → 'comput'.  The
fix is to make collapsing CORPUS-AWARE: a surface form is only
collapsed to a base when that base is itself a known word.

  Lemmatizer()                  — irregulars + SAFE suffix rules
                                  only (regular plurals); used
                                  when no vocabulary is available.
  Lemmatizer(vocabulary=vocab)  — full suffix rules, but a strip
                                  is accepted only when the
                                  resulting base is in `vocab`.

Re-ingestion (D.3) does a pass-1 vocabulary scan, then a pass-2
ingest with a vocabulary-aware lemmatizer.
"""

from __future__ import annotations

from typing import Dict, Optional, Set


# ---------------------------------------------------------------
# Irregular forms — surface form → lemma.  Always applied,
# vocabulary or not.  Covers the common irregular verbs + noun
# plurals that suffix rules can't reach.
# ---------------------------------------------------------------

IRREGULAR: Dict[str, str] = {
    # be / have / do
    'is': 'be', 'are': 'be', 'was': 'be', 'were': 'be',
    'been': 'be', 'being': 'be', 'am': 'be',
    'has': 'have', 'had': 'have', 'having': 'have',
    'does': 'do', 'did': 'do', 'done': 'do', 'doing': 'do',
    # irregular verbs (past / participle → base)
    'ran': 'run', 'running': 'run',
    'went': 'go', 'gone': 'go', 'going': 'go',
    'said': 'say', 'saying': 'say',
    'made': 'make', 'making': 'make',
    'came': 'come', 'coming': 'come',
    'took': 'take', 'taken': 'take', 'taking': 'take',
    'saw': 'see', 'seen': 'see', 'seeing': 'see',
    'knew': 'know', 'known': 'know', 'knowing': 'know',
    'thought': 'think', 'thinking': 'think',
    'gave': 'give', 'given': 'give', 'giving': 'give',
    'found': 'find', 'finding': 'find',
    'told': 'tell', 'telling': 'tell',
    'became': 'become', 'becoming': 'become',
    'felt': 'feel', 'feeling': 'feel',
    'left': 'leave', 'leaving': 'leave',
    'brought': 'bring', 'bringing': 'bring',
    'began': 'begin', 'begun': 'begin', 'beginning': 'begin',
    'kept': 'keep', 'keeping': 'keep',
    'held': 'hold', 'holding': 'hold',
    'stood': 'stand', 'standing': 'stand',
    'heard': 'hear', 'hearing': 'hear',
    'meant': 'mean', 'meaning': 'mean',
    'met': 'meet', 'meeting': 'meet',
    'set': 'set', 'setting': 'set',
    'put': 'put', 'putting': 'put',
    'lost': 'lose', 'losing': 'lose',
    'paid': 'pay', 'paying': 'pay',
    'led': 'lead', 'leading': 'lead',
    'understood': 'understand',
    'spoke': 'speak', 'spoken': 'speak', 'speaking': 'speak',
    'grew': 'grow', 'grown': 'grow', 'growing': 'grow',
    'wrote': 'write', 'written': 'write', 'writing': 'write',
    'drew': 'draw', 'drawn': 'draw', 'drawing': 'draw',
    'gave': 'give',
    'fell': 'fall', 'fallen': 'fall', 'falling': 'fall',
    'rose': 'rise', 'risen': 'rise', 'rising': 'rise',
    'died': 'die', 'dying': 'die',
    'lay': 'lie', 'lying': 'lie',
    'sent': 'send', 'sending': 'send',
    'built': 'build', 'building': 'build',
    'bought': 'buy', 'buying': 'buy',
    'caught': 'catch', 'catching': 'catch',
    'taught': 'teach', 'teaching': 'teach',
    'fought': 'fight', 'fighting': 'fight',
    'sought': 'seek', 'seeking': 'seek',
    'won': 'win', 'winning': 'win',
    'chose': 'choose', 'chosen': 'choose', 'choosing': 'choose',
    # irregular noun plurals
    'children': 'child', 'men': 'man', 'women': 'woman',
    'people': 'person', 'feet': 'foot', 'teeth': 'tooth',
    'mice': 'mouse', 'geese': 'goose', 'lives': 'life',
    'leaves': 'leaf', 'wives': 'wife', 'wolves': 'wolf',
    'knives': 'knife', 'selves': 'self', 'shelves': 'shelf',
    'thieves': 'thief', 'halves': 'half', 'calves': 'calf',
    'loaves': 'loaf', 'elves': 'elf',
}

# Comparative / superlative irregulars.
IRREGULAR.update({
    'better': 'good', 'best': 'good',
    'worse': 'bad', 'worst': 'bad',
    'more': 'much', 'most': 'much',
    'less': 'little', 'least': 'little',
    'further': 'far', 'furthest': 'far',
})


# Suffix rules: (suffix, replacement, is_safe).  "Safe" rules
# are regular noun-plural transforms that almost never
# over-strip; they apply even without a vocabulary.  Unsafe
# rules require a vocabulary check on the resulting base.
# Ordered longest-suffix-first so 'ies' beats 's'.
_SUFFIX_RULES = [
    # (suffix, replacement, safe)
    ('ies',  'y',  True),    # studies → study
    ('sses', 'ss', True),    # glasses → glass  (keep 'ss')
    ('xes',  'x',  True),    # boxes → box
    ('ches', 'ch', True),    # churches → church
    ('shes', 'sh', True),    # wishes → wish
    ('zzes', 'zz', True),
    ('ied',  'y',  False),   # studied → study
    ('ing',  '',   False),   # walking → walk
    ('ings', '',   False),
    ('edly', '',   False),
    ('ed',   '',   False),   # walked → walk
    ('est',  '',   False),   # fastest → fast
    ('ly',   '',   False),   # quickly → quick
    ('er',   '',   False),   # faster → fast
    ('es',   '',   False),   # passes → pass
    ('s',    '',   True),    # cats → cat
]

# Minimum length of a lemma after stripping.  Guards against
# turning short words into fragments.
_MIN_LEMMA_LEN = 3


class Lemmatizer:
    """Collapses surface forms to base lemmas.

    Stateless apart from the optional vocabulary.  Construct once,
    reuse across the whole ingestion.
    """

    def __init__(self, vocabulary: Optional[Set[str]] = None):
        # When set, an unsafe suffix strip is accepted only if the
        # resulting base is in this set.  When None, only safe
        # (regular-plural) rules are applied.
        self._vocab = vocabulary
        # Small memo so repeated tokens aren't re-derived.
        self._cache: Dict[str, str] = {}

    def lemmatize(self, word: str) -> str:
        """Return the base lemma of `word`.  Lowercased.  Returns
        the word unchanged when no rule applies."""
        if not word:
            return word
        low = word.lower()
        cached = self._cache.get(low)
        if cached is not None:
            return cached
        result = self._derive(low)
        self._cache[low] = result
        return result

    def _derive(self, low: str) -> str:
        # 1. Irregular table wins outright.
        irr = IRREGULAR.get(low)
        if irr is not None:
            return irr
        # 2. Too short to strip meaningfully.
        if len(low) <= _MIN_LEMMA_LEN:
            return low
        # 3. Suffix rules, longest-suffix first.
        for suffix, repl, safe in _SUFFIX_RULES:
            if not low.endswith(suffix):
                continue
            base = low[:-len(suffix)] + repl
            if len(base) < _MIN_LEMMA_LEN or base == low:
                continue
            # Candidate base forms — for -ing / -ed, English
            # orthography means the true lemma may need a silent
            # 'e' restored ('determining' → 'determine', not
            # 'determin') or a doubled final consonant collapsed
            # ('running' → 'run').  Offer all plausible bases;
            # _pick_base chooses the vocabulary-backed one.
            candidates = [base]
            if suffix in ('ing', 'ed'):
                candidates.append(base + 'e')   # determine, notice
                if (len(base) >= 2 and base[-1] == base[-2]
                        and base[-1] not in 'aeiou'):
                    dedoubled = base[:-1]
                    if len(dedoubled) >= _MIN_LEMMA_LEN:
                        candidates.append(dedoubled)
            if safe:
                # Safe rule (regular plural) — first candidate.
                return candidates[0]
            # Unsafe rule: require vocabulary backing.
            if self._vocab is None:
                # No vocabulary → don't risk over-stripping.
                continue
            chosen = self._pick_base(low, candidates)
            if chosen != low:
                return chosen
        return low

    def _pick_base(self, original: str, candidates) -> str:
        """Choose the first candidate that is vocabulary-backed.
        With no vocabulary, return the first candidate (used only
        by safe rules / dedoubling, where over-strip risk is low).
        Falls back to `original` when nothing qualifies."""
        if self._vocab is None:
            return candidates[0]
        for c in candidates:
            if c in self._vocab:
                return c
        return original


# Module-level convenience: a vocabulary-free lemmatizer for
# callers that just want safe normalization.
_DEFAULT = Lemmatizer()


def lemmatize(word: str) -> str:
    """Safe (vocabulary-free) lemmatization via a shared
    instance.  Applies irregulars + regular-plural rules only."""
    return _DEFAULT.lemmatize(word)
