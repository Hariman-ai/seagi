"""Brain v2 capabilities — interconnected functional pieces.

Each capability is a module here.  The names follow brain
anatomy because that's the user-approved organizing metaphor.
Each module has a header docstring naming its brain analog,
its Phase status, and what it reads/writes/emits on the event
bus.

Layout:

    sensory.py          Phase 1 — raw input decoding
    thalamic_gate.py    Phase 1 — salience filter (FULL)
    awm.py              Phase 2 — active working memory + chemistry TS
    lts.py              Phase 2+ — long-term substrate query API
    chemistry.py        Phase 2 — NT + bubble enrichment + endocrine
    cerebellum.py       Phase 4 — prediction + timing (lateral + vermis)
    basal_ganglia.py    Phase 4b — arbitration + striatum sub-loops
    cortical.py         Phase 3 — event-triggered reasoning
    dmn.py              Phase 4a — dorsomedial + ventromedial reflection
    hippocampus.py      Phase 4a — episode formation + consolidation
    amygdala.py         Phase 4b — threat detection + override
    insula.py           Phase 4a — interoception (felt body)
    acc.py              Phase 4 — conflict / mismatch detection
    anterior_pfc.py     Phase 4 — metacognition + prospective intent
    value_landscape.py  Phase 4 — mesolimbic / orbitofrontal map
    body_schema.py      Phase 4 — operational envelope
    source_monitor.py   Phase 4 — self/other distinction (TPJ + mPFC)
    lc.py               Phase 4 — NE alertness modulator
    vta.py              Phase 4 — dopamine RPE
    time_perception.py  Phase 4 — multi-scale time
    corpus_callosum.py  Phase 4 — parallel novel + familiar streams
    motor_speech.py     Phase 5 — single-voice coherence layer
"""
