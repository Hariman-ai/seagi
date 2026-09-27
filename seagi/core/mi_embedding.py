"""MIEmbedding — semantic vector + parallel M/I head.

Per AGI_ENGINE_DESIGN.md §4.2.6 and audit module 10. Embeddings entering
attention must carry M/I; pure semantic embeddings (MiniLM) are M/I-blind
by definition. The split keeps the semantic vector pure (good for
similarity tasks where M/I shouldn't dominate) and adds a parallel M/I
head; downstream scoring weights both per Layer 0.3.5.3.

Brain analog: cortical sensory feature extraction (semantic) is
structurally distinct from limbic valence assignment (M/I); they meet
at downstream integration sites (attention, decision).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple
import math

from seagi.core.mi_value import MIValue


@dataclass
class MIEmbedding:
    """Semantic vector + M/I head.

    `semantic` is a fixed-length list of floats (typically 384 from
    MiniLM, 256 from the n-gram fallback). It is L2-normalized at
    construction so cosine similarity is just the dot product.

    `mi` carries the M/I valuation aligned with the semantic content
    (e.g. for a concept embedding, this is the concept's MIValue;
    for a query embedding, it's derived from current tone).
    """
    semantic: List[float]
    mi: MIValue

    def __post_init__(self):
        # Normalize semantic vector at construction. If zero-vector,
        # leave as-is (norm-0 means "no signal").
        n = math.sqrt(sum(x * x for x in self.semantic))
        if n > 0:
            self.semantic = [x / n for x in self.semantic]

    @property
    def dim(self) -> int:
        return len(self.semantic)

    def cosine(self, other: 'MIEmbedding') -> float:
        """Cosine similarity in semantic space. Both vectors are
        already normalized so this is just the dot product."""
        if self.dim != other.dim:
            return 0.0
        return sum(a * b for a, b in zip(self.semantic, other.semantic))

    def mi_alignment(self, other: 'MIEmbedding') -> float:
        """How aligned this embedding's M/I is with another's. Ranges
        in [-1, 1]: +1 means both lean the same polarity strongly,
        -1 means opposite polarities. Used in M/I-aware scoring."""
        my_polarity = self.mi.polarity   # i - m
        other_polarity = other.mi.polarity
        # Magnitude-weighted polarity match.
        magnitude = max(self.mi.magnitude * other.mi.magnitude, 1e-6)
        # Same-sign polarity → positive; opposite → negative.
        return my_polarity * other_polarity / magnitude

    def score_against(self,
                      other: 'MIEmbedding',
                      tone_m: float = 0.0,
                      tone_i: float = 0.0,
                      lambda_m: float = 1.0,
                      lambda_i: float = 1.0
                      ) -> float:
        """Layer 0.3.5.3 attention score: cosine similarity scaled by
        M/I-channel precision under current tone.

            score = cosine(self, other)
                  * (1 + λ_M · tone_m · self.mi.m · other.mi.m)
                  * (1 + λ_I · tone_i · self.mi.i · other.mi.i)

        Replaces the audit's '70/30 anti-pattern' (cosine + small bias)
        with a multiplicative form where M/I parameterizes the score.
        """
        base = self.cosine(other)
        m_factor = 1 + lambda_m * tone_m * self.mi.m * other.mi.m
        i_factor = 1 + lambda_i * tone_i * self.mi.i * other.mi.i
        return base * m_factor * i_factor

    def to_dict(self) -> dict:
        return {
            'semantic': list(self.semantic),
            'mi': self.mi.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MIEmbedding':
        return cls(
            semantic=list(d.get('semantic', [])),
            mi=MIValue.from_dict(d.get('mi', {})),
        )

    @classmethod
    def zero(cls, dim: int = 384) -> 'MIEmbedding':
        return cls(semantic=[0.0] * dim, mi=MIValue.zero())


# ---------------------------------------------------------------------
# Encoders — bridge MIEmbedding to the MiniLM backend
# ---------------------------------------------------------------------

def encode_text_with_mi(text: str,
                        mi: Optional[MIValue] = None) -> 'MIEmbedding':
    """Encode arbitrary text into an MIEmbedding via the live
    backend (MiniLM if available, n-gram fallback otherwise).

    `mi` is the M/I head — typically derived from current tone for
    query embeddings, or from the source concept's M/I for concept
    embeddings. Defaults to MIValue.zero (neutral)."""
    from seagi.core.minilm_backend import encode_text
    semantic = encode_text(text)
    return MIEmbedding(
        semantic=semantic,
        mi=mi if mi is not None else MIValue.zero(),
    )


def encode_concept_with_mi(concept) -> 'MIEmbedding':
    """Encode a substrate Concept into an MIEmbedding using its
    name + key properties + top knowledge as the semantic surface,
    and its `concept.mi` as the M/I head.

    The semantic surface is intentionally rich enough that "fire"
    with knowledge "fire is hot and bright" embeds differently
    from a bare "fire" — the embedding reflects what the concept
    HAS COME TO MEAN in this engine, not just its lemma."""
    if concept is None:
        return MIEmbedding.zero()
    parts: List[str] = []
    name = getattr(concept, 'name', None)
    if name:
        parts.append(str(name))
    # Top 3 knowledge text snippets (if present).
    knowledge = getattr(concept, 'knowledge', None) or []
    if knowledge:
        ranked = sorted(
            knowledge,
            key=lambda k: float(getattr(k, 'confidence', 0.0) or 0.0),
            reverse=True)
        for item in ranked[:3]:
            text = getattr(item, 'text', None)
            if text:
                parts.append(str(text)[:120])
    text = ' '.join(parts).strip()
    return encode_text_with_mi(text, mi=getattr(concept, 'mi', None))


def _concept_to_encoding_text(concept) -> str:
    """Same surface that `encode_concept_with_mi` builds —
    extracted as a helper so a batch path can produce many
    texts at once and call the batch encoder."""
    if concept is None:
        return ''
    parts: List[str] = []
    name = getattr(concept, 'name', None)
    if name:
        parts.append(str(name))
    knowledge = getattr(concept, 'knowledge', None) or []
    if knowledge:
        ranked = sorted(
            knowledge,
            key=lambda k: float(getattr(k, 'confidence', 0.0) or 0.0),
            reverse=True)
        for item in ranked[:3]:
            text = getattr(item, 'text', None)
            if text:
                parts.append(str(text)[:120])
    return ' '.join(parts).strip()


def bulk_embed_concepts(substrate, batch_size: int = 64) -> int:
    """Bulk-encode every substrate concept that doesn't yet
    have a current-dim embedding.  Uses the batch MiniLM API
    so ~14K concepts encode in tens of seconds instead of
    minutes.  Returns the count of newly-embedded concepts.

    Called from `serve.run` AFTER MiniLM warmup but BEFORE
    the API starts, so the first /chat doesn't pay the
    lazy-encode cost.  Idempotent — concepts that already
    have current-dim embeddings are skipped.
    """
    if substrate is None:
        return 0
    from seagi.core.minilm_backend import encode_batch, get_dim
    target_dim = get_dim()
    # Collect concepts needing fresh embeddings.
    needs: List[Tuple[str, Any]] = []
    for name, c in (getattr(
            substrate, 'concepts', None) or {}).items():
        emb = getattr(c, 'embedding', None)
        if emb is None or getattr(emb, 'dim', 0) != target_dim:
            needs.append((str(name), c))
    if not needs:
        return 0
    # Build texts in the same shape encode_concept_with_mi
    # produces, so the resulting embeddings are equivalent.
    texts = [_concept_to_encoding_text(c) for _, c in needs]
    # One batch call replaces N individual encode_text calls.
    vecs = encode_batch(texts, batch_size=batch_size)
    # Stamp the embeddings back onto concepts AND populate LSH.
    lsh = getattr(substrate, '_lsh_index', None)
    for (name, concept), vec in zip(needs, vecs):
        mi = getattr(concept, 'mi', None) or MIValue.zero()
        emb = MIEmbedding(semantic=vec, mi=mi)
        concept.embedding = emb
        if lsh is not None and vec:
            try:
                lsh.add(name, vec)
            except Exception:
                pass
    return len(needs)
