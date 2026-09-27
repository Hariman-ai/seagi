"""Seagi ingestion — Layer 3 corpus pipeline.

Sentence-by-sentence reading under live chemistry.  Forges
personality through accumulated NT-tag trace per concept.
"""

from .driver import (
    CorpusIngester,
    split_sentences,
    split_paragraphs,
    DEFAULT_TICKS_BETWEEN_SENTENCES,
    DEFAULT_REFLECT_EVERY_N_SENTENCES,
)
from .vocabulary import build_vocabulary, vocabulary_summary
from .lemmatizer import Lemmatizer, lemmatize

__all__ = [
    'CorpusIngester',
    'split_sentences',
    'split_paragraphs',
    'DEFAULT_TICKS_BETWEEN_SENTENCES',
    'DEFAULT_REFLECT_EVERY_N_SENTENCES',
    'build_vocabulary',
    'vocabulary_summary',
    'Lemmatizer',
    'lemmatize',
]
