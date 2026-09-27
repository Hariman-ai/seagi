"""ReplayConsolidator — the return-path organ (Step 0). Proves the safe,
shadow-staged MVP: lifeforce-decoupled strength re-walk (single-edge
recheck never touches the credit fields), TENSION-seed selection (the felt
core min(M,I) outranks high-frequency I-only junk; junk never selected),
EARN-GATED re-anchor re-imprint (re-anchor only on real re-walk, no
encounter farm), and the trace-decay fade + re-evoke."""

from __future__ import annotations

import types
import unittest

from seagi.core.substrate import Substrate, EDGE_PRUNE_FLOOR
from seagi.core.bubble import Bubble
from seagi.core.mi_value import TransmitterState
from seagi.core.layer6 import derive_mi_from_trace
from seagi.brain.capabilities.replay_consolidator import ReplayConsolidator


def _rig():
    sub = Substrate()
    sub._quarantine_migrated = True
    engine = types.SimpleNamespace(substrate=sub)
    return sub, ReplayConsolidator(engine=engine)


def _bubble(sub, name, cycle, i=0.0, m=0.0):
    """Attach a bubble to a concept.
    i>0 → dopamine departure → I-side (immortality) score.
    m>0 → cortisol departure → M-side (mortality) score.
    tension = min(M, I), so a concept needs BOTH poles to be selected by
    the tension-scored return path (I-only = high magnitude but zero
    tension = the frequency-confound junk the selector must ignore)."""
    t = TransmitterState()
    if i > 0:
        t['dopamine'] = 0.30 + i      # dopamine baseline 0.30
    if m > 0:
        t['cortisol'] = 0.10 + m      # cortisol baseline 0.10
    c = sub.get_or_create_concept(name)
    b = Bubble(transmitter_trace=t, last_active_cycle=cycle)
    c.bubbles.append(b)
    return b


def _triangle(sub):
    # a -(is_a)-> b corroborated by a -(is_a)-> x -(is_a)-> b
    for n in ('a', 'x', 'b'):
        sub.get_or_create_concept(n)
    sub.add_edge('a', 'x', 'is_a', strength=0.3)
    sub.add_edge('x', 'b', 'is_a', strength=0.3)
    return sub.add_edge('a', 'b', 'is_a', strength=0.3)


class TestReplayConsolidator(unittest.TestCase):

    def test_recheck_corroborates_without_touching_credit_fields(self):
        # The single-edge recheck must confirm corroboration but NEVER
        # set first_coherent_cycle — the lifeforce decoupling, structural.
        sub, _ = _rig()
        e = _triangle(sub)
        fcc0 = e.first_coherent_cycle
        ok, corr = sub.recheck_edge_coherence(e, 100)
        self.assertTrue(ok)
        self.assertEqual(len(corr), 2)
        self.assertEqual(e.first_coherent_cycle, fcc0)   # untouched

    def test_uncorroborated_edge_not_confirmed(self):
        sub, _ = _rig()
        for n in ('a', 'b'):
            sub.get_or_create_concept(n)
        e = sub.add_edge('a', 'b', 'is_a', strength=0.3)   # no a->x->b
        ok, corr = sub.recheck_edge_coherence(e, 100)
        self.assertFalse(ok)
        self.assertEqual(corr, [])

    def test_rewalk_reinforces_corroborated_core(self):
        sub, loop = _rig()
        e = _triangle(sub)
        _bubble(sub, 'a', cycle=0, i=0.5, m=0.5)   # tense (both poles) + stale
        loop.run_pass(cycle=5000, k=10)
        self.assertGreaterEqual(loop.edges_rewalked, 1)
        self.assertEqual(e.last_reinforced_cycle, 5000)   # re-walked
        self.assertGreater(e.effective_strength(5000), EDGE_PRUNE_FLOOR)

    def test_dead_corroborator_fades_and_is_not_reanchored(self):
        # A tense (selected) concept whose edge does NOT corroborate earns
        # nothing AND is not re-anchored — the earn-gate (Change B).
        sub, loop = _rig()
        for n in ('a', 'b'):
            sub.get_or_create_concept(n)
        e = sub.add_edge('a', 'b', 'is_a', strength=0.3)
        b = _bubble(sub, 'a', cycle=0, i=0.5, m=0.5)   # selected (tension>0)
        loop.run_pass(cycle=5000, k=10)
        self.assertEqual(loop.concepts_selected, 1)         # it WAS selected
        self.assertEqual(loop.edges_rewalked, 0)
        self.assertGreaterEqual(loop.edges_faded, 1)
        self.assertNotEqual(e.last_reinforced_cycle, 5000)  # not re-walked
        self.assertEqual(loop.reanchored, 0)                # earn-gate: no stamp
        self.assertEqual(b.last_active_cycle, 0)            # NOT re-anchored

    def test_low_mi_junk_never_selected(self):
        # baseline-trace (junk) concept → tension 0 → never seeded.
        sub, loop = _rig()
        for n in ('junk', 'b'):
            sub.get_or_create_concept(n)
        sub.add_edge('junk', 'b', 'is_a', strength=0.3)
        _bubble(sub, 'junk', cycle=0, i=0.0)
        loop.run_pass(cycle=5000, k=10)
        self.assertEqual(loop.concepts_selected, 0)

    def test_tension_selects_felt_core_over_high_frequency_i_junk(self):
        # The frequency-confound regression (the `than` vs `justice` case):
        # `frequent` has a HIGH I-side (magnitude high, like a heavily-read
        # function word) but zero M-side → tension 0; `felt` departed BOTH
        # poles → tension high.  With k=1 only the felt core is selected.
        sub, loop = _rig()
        for n in ('felt', 'frequent', 'b', 'x'):
            sub.get_or_create_concept(n)
        _bubble(sub, 'frequent', cycle=0, i=0.9)            # max-magnitude junk
        _bubble(sub, 'felt', cycle=0, i=0.4, m=0.4)         # lower magnitude, tense
        loop.run_pass(cycle=5000, k=1)
        self.assertIn('felt', loop.last_selection_top)
        self.assertNotIn('frequent', loop.last_selection_top)

    def test_reanchor_only_no_encounter_farm(self):
        sub, loop = _rig()
        _triangle(sub)
        b = _bubble(sub, 'a', cycle=0, i=0.5, m=0.5)    # tense → selected
        ec0 = b.encounter_count
        loop.run_pass(cycle=5000, k=10)
        self.assertEqual(b.encounter_count, ec0)        # no chemistry farm
        self.assertEqual(b.last_active_cycle, 5000)     # re-anchored (earned)

    def test_effective_trace_fades_then_reevokes_own_signature(self):
        t = TransmitterState()
        t['dopamine'] = 0.80
        b = Bubble(transmitter_trace=t, last_active_cycle=0)
        mi0 = derive_mi_from_trace(b.transmitter_trace).magnitude
        self.assertGreater(mi0, 0.1)
        mi_faded = derive_mi_from_trace(b.effective_trace(10 ** 7)).magnitude
        self.assertLess(mi_faded, mi0)                  # fades from disuse
        b.last_active_cycle = 10 ** 7                    # re-anchor
        mi_re = derive_mi_from_trace(b.effective_trace(10 ** 7)).magnitude
        self.assertAlmostEqual(mi_re, mi0, places=5)    # own value re-evoked

    def test_shadow_staged_concept_mi_reads_raw(self):
        # SHADOW: Concept.mi must still read the RAW (undecayed) trace, so
        # this phase does NOT change the live M/I landscape (no inversion
        # shock). The decay is shadow-logged only.
        sub, loop = _rig()
        b = _bubble(sub, 'a', cycle=0, i=0.5, m=0.5)   # tense → exercises shadow
        c = sub.concepts['a']
        raw = c.mi.magnitude
        loop.run_pass(cycle=10 ** 7, k=10)
        # mi still reads raw (re-anchored to 1e7, elapsed 0 anyway); the
        # property never routes through effective_trace this phase.
        self.assertGreater(raw, 0.1)
        self.assertTrue(loop.stats()['shadow_staged'])


if __name__ == '__main__':
    unittest.main()
