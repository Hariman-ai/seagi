"""Seagi core — the data layer.

Substrate, concepts, edges, bubbles, M/I values, persistence.
These are what PERSIST on disk and define what Seagi knows.

This __init__ is kept minimal to avoid eager loading of every
submodule (which would chain through agi_engine shims and
trigger circular imports while v1 cognitive modules still
exist).  Import specific names from their submodules:

    from seagi.core.bubble import Bubble
    from seagi.core.substrate import Substrate, Concept
    from seagi.core.mi_value import MIValue, TransmitterState
    from seagi.core.persistence import save_brain, load_brain
"""
