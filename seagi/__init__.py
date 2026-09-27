"""SEAGI — the unified architecture.

This package contains the doctrine-aligned implementation of
Seagi (Self-Evolving AGI).  It replaces the v1/v2 split in
`agi_engine/`.

Subpackages
-----------
  seagi.core       — data layer (substrate, concepts, edges,
                       bubbles, MI value, persistence)
  seagi.body       — body machinery (lifeforce, embodiment,
                       hierarchy / cycle counter)
  seagi.brain      — cognitive engine (chemistry, AWM,
                       cortical reasoning, sentinels, BG,
                       hippocampus, DMN, etc.)
  seagi.ingestion  — Layer 3 corpus ingestion (sentence-by-
                       sentence pipeline, SVO triple parser,
                       autonomous forager)

The legacy `agi_engine/` package is preserved during the
transition.  Once everything seagi/ needs has been migrated,
agi_engine/ is moved to archive/ and the unified seagi/ tree
becomes the single source of truth.
"""
