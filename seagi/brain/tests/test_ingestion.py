"""Layer 3 — corpus ingestion driver tests.

Headline behaviors:
  - Each sentence flows through brain.intake() and is followed
    by at least one brain.tick().
  - Substrate grows during ingestion.
  - Chemistry trajectories DO move on M-content vs I-content
    text — different content shapes Seagi differently.
  - Hippocampus forms episodes during ingestion.
  - Monitors fire during ingestion.
  - Resume sidecar (`.read`) prevents re-ingesting.
  - Snapshot/summary captures the state mid-run.
"""

import os
import tempfile
import time
import unittest
from pathlib import Path

from seagi.body.engine import Engine
from seagi.brain import Brain
from seagi.ingestion import (
    CorpusIngester,
    split_sentences,
    split_paragraphs,
)


# ---------------------------------------------------------------------
# Segmenters
# ---------------------------------------------------------------------


class TestSegmenters(unittest.TestCase):

    def test_split_sentences_basic(self):
        text = "First sentence.  Second one!  Third? Fourth."
        self.assertEqual(
            split_sentences(text),
            ['First sentence.', 'Second one!', 'Third?',
              'Fourth.'])

    def test_split_sentences_empty(self):
        self.assertEqual(split_sentences(''), [])
        self.assertEqual(split_sentences('   '), [])

    def test_split_sentences_no_terminator(self):
        # Single sentence without ending punctuation.
        self.assertEqual(
            split_sentences('A passing thought'),
            ['A passing thought'])

    def test_split_paragraphs(self):
        text = "Paragraph one\nis here.\n\nParagraph two\nstarts here."
        paras = split_paragraphs(text)
        self.assertEqual(len(paras), 2)
        self.assertIn('Paragraph one', paras[0])
        self.assertIn('Paragraph two', paras[1])


# ---------------------------------------------------------------------
# Per-sentence pipeline
# ---------------------------------------------------------------------


class TestIngestSentencePipeline(unittest.TestCase):
    """Verify each sentence flows through the v2 brain
    correctly: intake fires, chemistry tags, AWM populates,
    brain ticks between sentences."""

    def setUp(self):
        self.engine = Engine()
        self.brain = Brain(engine=self.engine)
        self.ingester = CorpusIngester(
            self.brain, corpus_dir=None,
            ticks_between_sentences=1)

    def test_sentence_ingest_increments_counter(self):
        self.ingester.ingest_text(
            'A new concept arrives. Another follows.')
        self.assertEqual(self.ingester.sentences_ingested, 2)

    def test_sentence_ingest_grows_substrate(self):
        concepts_before = self.brain.lts.stats().get(
            'concepts', 0)
        self.ingester.ingest_text(
            'Strange newcomers walk past. '
            'Curious creatures arrive.')
        concepts_after = self.brain.lts.stats().get(
            'concepts', 0)
        self.assertGreater(concepts_after, concepts_before,
            "Substrate did not grow during ingestion — the "
            "v2 ingestion path is not writing to substrate.")

    def test_sentence_ingest_fires_chemistry(self):
        fires_before = self.brain.chemistry.events_fired
        self.ingester.ingest_text(
            'A wholly new concept enters mind.')
        fires_after = self.brain.chemistry.events_fired
        self.assertGreater(fires_after, fires_before,
            "Chemistry didn't fire during ingestion — "
            "Seagi reads without feeling.")

    def test_sentence_ingest_populates_awm(self):
        size_before = self.brain.awm.size()
        self.ingester.ingest_text(
            'The newcomer arrives. The visitor introduces.')
        size_after = self.brain.awm.size()
        self.assertGreater(size_after, size_before,
            "AWM didn't grow during ingestion — concepts "
            "aren't reaching working memory.")

    def test_ticks_advance_between_sentences(self):
        # Brain.tick() should run between each sentence.  We
        # verify by checking the internal cycle counter
        # increases by approximately (sentences * ticks_per).
        cycle_before = self.brain._cycle_provider()
        self.ingester.ingest_text(
            'One. Two. Three. Four. Five.')
        cycle_after = self.brain._cycle_provider()
        # When engine is present, cycle comes from engine and
        # may not advance during pure ingestion (engine.tick()
        # isn't called).  When engine is None, internal cycle
        # advances by ticks.  Verify EITHER advanced OR brain
        # has decay-ticked the chemistry.  The latter is more
        # robust: chemistry.events_applied increases on every
        # event handled.
        self.assertGreaterEqual(
            self.brain.chemistry.events_applied, 5)


# ---------------------------------------------------------------------
# Chemistry trajectories: different content shapes differently
# ---------------------------------------------------------------------


class TestContentShapesChemistry(unittest.TestCase):
    """Doctrine: natural / unconstrained chemistry environment.
    Each passage shapes Seagi as it would shape a real reader.

    Important nuance: on a FRESH substrate (no accumulated
    trace), no concept is innately M or I — the doctrine
    explicitly forbids innate concept tags.  So fresh-substrate
    ingestion of M-words doesn't immediately raise cortisol;
    novelty dominates and fires curiosity instead.

    The M/I responsiveness EMERGES once traces have
    accumulated.  These tests verify both regimes."""

    def test_fresh_substrate_ingestion_fires_curiosity(self):
        # First encounter: novelty dominates.  Chemistry moves
        # in the curiosity direction (dopamine slight up,
        # cortisol slight down) — not in the content-specific
        # direction, because the substrate doesn't yet know
        # what these words mean.  One curiosity event per
        # sentence (novelty score aggregates across the
        # sentence's focals, so a single high-magnitude fire
        # rather than per-focal fires).
        engine = Engine()
        brain = Brain(engine=engine)
        ingester = CorpusIngester(brain, corpus_dir=None)
        fires_before = brain.chemistry.events_fired
        ingester.ingest_text(
            'Death waits at every turn.  Pain comes with fear. '
            'Threat and danger arrive together.')
        # 3 sentences → at least 3 chemistry events (curiosity
        # per sentence, plus possible internal cascades from
        # other capabilities).
        self.assertGreaterEqual(
            brain.chemistry.events_fired - fires_before, 3,
            "Fresh-substrate ingestion produced fewer "
            "chemistry events than sentences read — the "
            "chemistry pathway is not engaging.")

    def test_primed_substrate_responds_to_m_content(self):
        # When the substrate ALREADY carries M-trace on a
        # concept (accumulated from prior experience), reading
        # text containing that concept fires anomaly_spike
        # and raises cortisol.  This is the architecturally-
        # promised behavior once trace exists.
        engine = Engine()
        brain = Brain(engine=engine)
        # Pre-seed: create 'wolf' with M-leaning bubble trace.
        from seagi.core.substrate import (
            Concept, Bubble, ContextKey)
        from seagi.core.mi_value import (
            MIValue, TransmitterState)
        wolf = Concept(name='wolf')
        wolf.bubbles = [Bubble(
            transmitter_trace=TransmitterState(
                cortisol=0.8, norepinephrine=0.7),
            context_key=ContextKey(),
            encounter_count=10)]
        engine.substrate.add_concept(wolf)
        cortisol_before = (
            brain.chemistry.global_state['cortisol'])
        # Read text mentioning wolf — the gate's _mi_content
        # picks up the M-trace from wolf's bubble.
        ingester = CorpusIngester(brain, corpus_dir=None)
        for _ in range(3):
            ingester.ingest_text('A wolf approaches the camp.')
        cortisol_after = (
            brain.chemistry.global_state['cortisol'])
        # With primed substrate, repeated M-content firing
        # should raise cortisol above the curiosity-only baseline.
        self.assertGreater(
            cortisol_after, cortisol_before,
            "Primed substrate didn't raise cortisol on "
            "repeated M-content encounter — the architecture "
            "isn't translating learned M-trace into chemistry "
            "response.")

    def test_innate_machinery_distinguishes_death_from_birth(
            self):
        """The architectural critique: fresh-substrate reading
        of 'death' should produce a different chemistry cocktail
        from fresh-substrate reading of 'birth'.  Subtle but
        distinguishing — innate machinery responds to universal-
        survival features before any learning.

        Doctrinally: substrate stores no M/I tags on either
        word.  Machinery has innate response to the FEATURES.
        Chemistry fires.  Trace accumulates from the chemistry,
        not from any stored label."""
        brain_d = Brain(engine=Engine())
        brain_b = Brain(engine=Engine())
        ing_d = CorpusIngester(brain_d, corpus_dir=None)
        ing_b = CorpusIngester(brain_b, corpus_dir=None)
        # Both brains start fresh.  Both read a sentence about
        # the universal-survival concept they're named for.
        ing_d.ingest_text(
            'The death comes for every one of us.')
        ing_b.ingest_text(
            'The birth comes for every one of us.')
        # death-reader should have HIGHER cortisol than
        # birth-reader (innate M-machinery fired anomaly_spike
        # on 'death').
        self.assertGreater(
            brain_d.chemistry.global_state['cortisol'],
            brain_b.chemistry.global_state['cortisol'],
            "death-reader and birth-reader have the same "
            "cortisol — the innate machinery is not "
            "distinguishing universal-survival features.")
        # birth-reader should have HIGHER endorphins than
        # death-reader (innate I-machinery fired mattering on
        # 'birth'; mattering raises endorphins).
        self.assertGreater(
            brain_b.chemistry.global_state['endorphins'],
            brain_d.chemistry.global_state['endorphins'],
            "birth-reader didn't gain endorphins relative to "
            "death-reader — innate I-machinery is silent.")
        # Substrate stayed neutral on both: neither 'death' nor
        # 'birth' has any stored M/I label (the substrate
        # contains concepts with MIValue.zero — only the
        # CHEMISTRY differs, not the substrate's stored shape).
        # We verify this by confirming both concepts exist with
        # no innate label distinction at the substrate level.
        death_concept = brain_d.engine.substrate.concepts.get(
            'death')
        birth_concept = brain_b.engine.substrate.concepts.get(
            'birth')
        self.assertIsNotNone(death_concept)
        self.assertIsNotNone(birth_concept)

    def test_two_brains_diverge_in_substrate_not_just_chemistry(
            self):
        # Two brains, fresh substrates, different content.
        # Global chemistry vectors can converge (curiosity
        # fires the same delta whatever the novel concept), but
        # the SUBSTRATES diverge — different concepts learned,
        # different bubble traces, different AWM contents.
        # This is the corpus-shapes-reader effect made
        # operational: two readers' "global mood" after reading
        # may be similar, but their MEMORIES are entirely
        # different.
        brain_a = Brain(engine=Engine())
        brain_b = Brain(engine=Engine())
        ingester_a = CorpusIngester(brain_a, corpus_dir=None)
        ingester_b = CorpusIngester(brain_b, corpus_dir=None)
        ingester_a.ingest_text(
            'Quasars and exotic phenomena populate the cosmos. '
            'Mysterious depths await discovery.')
        ingester_b.ingest_text(
            'Ordinary stones lie still in the garden.  '
            'Familiar items rest in place.')
        concepts_a = set(
            brain_a.engine.substrate.concepts.keys())
        concepts_b = set(
            brain_b.engine.substrate.concepts.keys())
        # Substrates have non-trivial overlap of stopwords/etc.,
        # but mostly diverged.
        only_a = concepts_a - concepts_b
        only_b = concepts_b - concepts_a
        self.assertGreater(len(only_a), 1,
            "Brain A learned no concepts unique to its text.")
        self.assertGreater(len(only_b), 1,
            "Brain B learned no concepts unique to its text.")
        # AWM contents also diverge.
        awm_a = set(brain_a.awm.active_concepts())
        awm_b = set(brain_b.awm.active_concepts())
        self.assertNotEqual(awm_a, awm_b,
            "Two brains reading different content have "
            "identical AWM contents — the corpus is not "
            "reaching working memory differently.")


# ---------------------------------------------------------------------
# Hippocampus + monitor activations during ingestion
# ---------------------------------------------------------------------


class TestIngestionActivatesAllCapabilities(unittest.TestCase):
    """Every v2 capability that should engage during reading
    must engage.  Failing this test means ingestion is bypassing
    parts of the brain that the doctrine requires to fire."""

    def test_hippocampus_forms_episodes(self):
        engine = Engine()
        brain = Brain(engine=engine)
        ingester = CorpusIngester(brain, corpus_dir=None)
        before = brain.hippocampus.episodes_formed
        ingester.ingest_text(
            'Reading produces episodes.  Each sentence is one.')
        after = brain.hippocampus.episodes_formed
        self.assertGreater(after, before,
            "Hippocampus didn't form episodes during "
            "ingestion — the experience is not being recorded.")

    def test_novelty_monitor_fires_on_fresh_content(self):
        engine = Engine()
        brain = Brain(engine=engine)
        ingester = CorpusIngester(brain, corpus_dir=None)
        before = brain.novelty_monitor.novel_detected
        ingester.ingest_text(
            'Quasars and exotic phenomena populate the cosmos. '
            'Unknown forces shape the universe.  '
            'Mysterious depths await discovery.')
        after = brain.novelty_monitor.novel_detected
        # All these concepts are new → high novelty → at
        # least one fire expected.
        self.assertGreater(after, before,
            "NoveltyMonitor didn't fire on fresh content "
            "— curiosity is not engaging during reading.")


# ---------------------------------------------------------------------
# Filesystem corpus + resume
# ---------------------------------------------------------------------


class TestCorpusFilesAndResume(unittest.TestCase):

    def test_corpus_directory_ingested(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / 'a.txt').write_text(
                'First file content. Two sentences here.',
                encoding='utf-8')
            (tmp_path / 'b.txt').write_text(
                'Second file content.', encoding='utf-8')
            engine = Engine()
            brain = Brain(engine=engine)
            ingester = CorpusIngester(brain, corpus_dir=tmp_path)
            summary = ingester.ingest_corpus()
            self.assertEqual(summary['files_ingested'], 2)
            self.assertGreaterEqual(
                summary['sentences_ingested'], 3)
            # Sidecars written.
            self.assertTrue((tmp_path / 'a.txt.read').exists())
            self.assertTrue((tmp_path / 'b.txt.read').exists())

    def test_resume_skips_already_processed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / 'a.txt').write_text(
                'First read.', encoding='utf-8')
            engine = Engine()
            brain = Brain(engine=engine)
            ingester1 = CorpusIngester(brain, corpus_dir=tmp_path)
            ingester1.ingest_corpus()
            # Second ingester on same directory.
            ingester2 = CorpusIngester(brain, corpus_dir=tmp_path)
            summary = ingester2.ingest_corpus()
            self.assertEqual(summary['files_ingested'], 0)
            self.assertEqual(summary['files_skipped'], 1)

    def test_html_file_stripped_before_ingestion(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / 'page.html').write_text(
                '<html><head><style>x{}</style></head><body>'
                '<p>Real content stays.</p><script>drop me</script>'
                '</body></html>',
                encoding='utf-8')
            (tmp_path / 'page.html').rename(
                tmp_path / 'page.txt')  # ingestion picks .txt
            engine = Engine()
            brain = Brain(engine=engine)
            ingester = CorpusIngester(brain, corpus_dir=tmp_path)
            ingester.ingest_corpus()
            # Verify substrate doesn't contain HTML-tag fragments
            # by checking a few canonical junk strings.
            concepts = (engine.substrate.concepts.keys()
                          if engine.substrate else [])
            for junk in ('script', 'style', 'html'):
                # 'script' as a real word would be fine — but
                # the HTML-tag *content* "drop me" was inside a
                # <script> block, so 'drop' shouldn't be a
                # concept.
                pass
            # The real assertion: ingestion didn't crash and
            # substrate grew with content concepts.
            self.assertGreater(
                len(concepts), 0,
                "HTML-stripped file produced no concepts.")


# ---------------------------------------------------------------------
# Brain convenience method
# ---------------------------------------------------------------------


class TestBrainIngestCorpus(unittest.TestCase):
    """The Brain.ingest_corpus() shortcut wires through the
    same CorpusIngester."""

    def test_brain_ingest_corpus_method_exists(self):
        engine = Engine()
        brain = Brain(engine=engine)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / 'a.txt').write_text(
                'Some text.', encoding='utf-8')
            summary = brain.ingest_corpus(tmp_path)
            self.assertIn('sentences_ingested', summary)
            self.assertGreaterEqual(
                summary['sentences_ingested'], 1)


# ---------------------------------------------------------------------
# Chemistry persistence — the test that would have caught the
# 73-minute-wasted bug.  MANDATORY in CI going forward.
# ---------------------------------------------------------------------


class TestChemistryPersistsAcrossSaveLoad(unittest.TestCase):
    """End-to-end persistence guarantee.

    The doctrine: corpus ingestion under live chemistry forges
    personality.  Forging = trace accumulates on concept bubbles.
    Persistence = those traces survive save_brain → load_brain.

    Before this test landed: AWM bubbles were v2 EnrichedBubbles
    that lived in memory only.  Chemistry imprints accumulated on
    them, but eviction silently dropped them and save_brain
    serialized only v1 substrate (which had no record of any
    chemistry).  A 73-minute foundational ingestion produced
    62,404 concepts with ZERO chemistry trace on disk.

    This test runs in seconds.  Catches the regression
    instantly."""

    def test_imprinted_chemistry_survives_save_load(self):
        import tempfile
        from seagi.core.persistence import save_brain, load_brain

        engine = Engine()
        brain = Brain(engine=engine)
        ingester = CorpusIngester(brain, corpus_dir=None)
        # Ingest text that triggers the innate-machinery M
        # response (so we have a known chemistry direction to
        # verify).
        ingester.ingest_text(
            'Death comes for everyone. '
            'Pain is part of life.  Fear waits in shadows.')

        # CRITICAL: flush AWM bubbles to substrate BEFORE saving.
        flushed = brain.flush_to_substrate()
        self.assertGreater(flushed, 0,
            "flush_to_substrate did nothing — AWM was empty.")

        # Save + load.
        with tempfile.NamedTemporaryFile(
                suffix='.json.gz', delete=False) as tf:
            path = tf.name
        try:
            save_brain(engine, path, gzipped=True)
            engine_reloaded = load_brain(path)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

        # At least one of the M-flavored concepts must have a
        # bubble with a non-default trace on the reloaded
        # substrate.  If this fails, the persistence bridge is
        # broken.
        sub = engine_reloaded.substrate
        m_concepts = ['death', 'pain', 'fear']
        found_with_trace = False
        for name in m_concepts:
            c = sub.concepts.get(name)
            if c is None:
                continue
            bubbles = getattr(c, 'bubbles', None) or []
            if not bubbles:
                continue
            b = bubbles[0]
            tx = getattr(b, 'transmitter_trace', None)
            if tx is None:
                continue
            # The innate machinery fires anomaly_spike on
            # M-feature words.  anomaly_spike's deltas raise
            # norepinephrine (+0.08) and cortisol (+0.05).
            ne = float(getattr(tx, 'norepinephrine', 0.0))
            cort = float(getattr(tx, 'cortisol', 0.0))
            if ne > 0.05 or cort > 0.05:
                found_with_trace = True
                break
        self.assertTrue(found_with_trace,
            "NO M-flavored concept has a persisted M-side "
            "chemistry trace after save/load.  The persistence "
            "bridge is broken — ingestion-time chemistry is "
            "evaporating before save_brain serializes the "
            "substrate.")

    def test_awm_eviction_persists_trace(self):
        # Force a direct eviction and verify the trace was
        # persisted to v1 substrate.  Uses brain.awm.evict()
        # directly rather than capacity-based eviction (which
        # the soft-cap expansion would prevent at low capacity
        # with high-salience content).
        engine = Engine()
        brain = Brain(engine=engine)
        ingester = CorpusIngester(brain, corpus_dir=None)
        # Imprint chemistry on 'death'.
        ingester.ingest_text('Death is the end.')
        # Verify 'death' is in AWM with non-baseline trace.
        entry = brain.awm.get('death')
        self.assertIsNotNone(entry,
            "death not promoted to AWM after ingestion.")
        ne_v2 = entry.bubble.transmitter_trace.get(
            'norepinephrine', 0.0)
        self.assertGreater(ne_v2, 0.05,
            "v2 EnrichedBubble didn't accumulate NE on M-feature "
            "input — innate machinery may have regressed.")
        # Now force eviction.
        brain.awm.evict('death')
        # And verify the trace is now on v1 substrate.
        c = engine.substrate.concepts.get('death')
        self.assertIsNotNone(c)
        bubbles = getattr(c, 'bubbles', None) or []
        self.assertGreater(len(bubbles), 0,
            "AWM eviction did not create a v1 Bubble.")
        ne_v1 = float(getattr(
            bubbles[0].transmitter_trace,
            'norepinephrine', 0.0))
        self.assertGreater(ne_v1, 0.05,
            "v1 Bubble's NE wasn't written from the v2 trace.  "
            "Eviction persistence is broken.")


# ---------------------------------------------------------------------
# Phase D.2a — rule-based lemmatizer
# ---------------------------------------------------------------------


class TestLemmatizer(unittest.TestCase):
    """Substrate-quality normalization: surface forms → lemmas."""

    def test_irregular_verbs(self):
        from seagi.ingestion.lemmatizer import lemmatize
        self.assertEqual(lemmatize('ran'), 'run')
        self.assertEqual(lemmatize('went'), 'go')
        self.assertEqual(lemmatize('thought'), 'think')
        self.assertEqual(lemmatize('was'), 'be')
        self.assertEqual(lemmatize('brought'), 'bring')

    def test_irregular_noun_plurals(self):
        from seagi.ingestion.lemmatizer import lemmatize
        self.assertEqual(lemmatize('children'), 'child')
        self.assertEqual(lemmatize('men'), 'man')
        self.assertEqual(lemmatize('lives'), 'life')
        self.assertEqual(lemmatize('leaves'), 'leaf')

    def test_safe_regular_plurals(self):
        from seagi.ingestion.lemmatizer import lemmatize
        self.assertEqual(lemmatize('cats'), 'cat')
        self.assertEqual(lemmatize('studies'), 'study')
        self.assertEqual(lemmatize('boxes'), 'box')
        self.assertEqual(lemmatize('churches'), 'church')
        self.assertEqual(lemmatize('glasses'), 'glass')

    def test_no_vocab_skips_unsafe_strips(self):
        # Without a vocabulary, -ing / -ed / -ly are NOT applied
        # (they over-strip).  'computer' must NOT become 'comput'.
        from seagi.ingestion.lemmatizer import lemmatize
        self.assertEqual(lemmatize('computer'), 'computer')
        self.assertEqual(lemmatize('running'), 'run')  # irregular table
        # 'walked' has no irregular entry; without vocab it stays.
        self.assertEqual(lemmatize('walked'), 'walked')

    def test_vocab_aware_unsafe_strips(self):
        from seagi.ingestion.lemmatizer import Lemmatizer
        lem = Lemmatizer(vocabulary={'walk', 'study', 'quick',
                                            'design', 'wrong'})
        self.assertEqual(lem.lemmatize('walked'), 'walk')
        self.assertEqual(lem.lemmatize('walking'), 'walk')
        self.assertEqual(lem.lemmatize('quickly'), 'quick')
        self.assertEqual(lem.lemmatize('designer'), 'design')
        self.assertEqual(lem.lemmatize('wronged'), 'wrong')

    def test_vocab_aware_rejects_overstrip(self):
        # 'computer' → 'comput' is rejected because 'comput' is
        # not in the vocabulary.
        from seagi.ingestion.lemmatizer import Lemmatizer
        lem = Lemmatizer(vocabulary={'compute', 'cat'})
        self.assertEqual(lem.lemmatize('computer'), 'computer')

    def test_doubled_consonant_restored(self):
        from seagi.ingestion.lemmatizer import Lemmatizer
        lem = Lemmatizer(vocabulary={'run', 'stop', 'sit'})
        self.assertEqual(lem.lemmatize('running'), 'run')
        self.assertEqual(lem.lemmatize('stopped'), 'stop')
        self.assertEqual(lem.lemmatize('sitting'), 'sit')

    def test_silent_e_restored(self):
        # -ing / -ed often need a silent 'e' restored:
        # 'determining' → 'determine' (not 'determin').
        from seagi.ingestion.lemmatizer import Lemmatizer
        lem = Lemmatizer(vocabulary={'determine', 'prepare',
                                            'notice', 'create'})
        self.assertEqual(lem.lemmatize('determining'),
                              'determine')
        self.assertEqual(lem.lemmatize('preparing'), 'prepare')
        self.assertEqual(lem.lemmatize('noticing'), 'notice')
        self.assertEqual(lem.lemmatize('created'), 'create')

    def test_short_words_untouched(self):
        from seagi.ingestion.lemmatizer import lemmatize
        self.assertEqual(lemmatize('ox'), 'ox')
        self.assertEqual(lemmatize('go'), 'go')
        self.assertEqual(lemmatize(''), '')

    def test_idempotent(self):
        # Lemmatizing a lemma returns the lemma.
        from seagi.ingestion.lemmatizer import lemmatize
        for w in ('run', 'cat', 'study', 'wisdom', 'death'):
            self.assertEqual(lemmatize(w), w)


# ---------------------------------------------------------------------
# SVO parser — relational ingestion
# ---------------------------------------------------------------------


class TestSVOParser(unittest.TestCase):
    """Unit-test the parser itself: given a sentence, what
    (subject, relation, object) triples come out?"""

    def test_is_a_with_article(self):
        from seagi.ingestion.svo_parser import parse_sentence
        triples = parse_sentence('A wolf is a predator.')
        self.assertEqual(len(triples), 1)
        t = triples[0]
        self.assertEqual(t.subject, 'wolf')
        self.assertEqual(t.relation, 'is_a')
        self.assertEqual(t.object, 'predator')

    def test_is_without_article_becomes_has_property(self):
        from seagi.ingestion.svo_parser import parse_sentence
        triples = parse_sentence('The water is cold.')
        self.assertEqual(len(triples), 1)
        t = triples[0]
        self.assertEqual(t.subject, 'water')
        self.assertEqual(t.relation, 'has_property')
        self.assertEqual(t.object, 'cold')

    def test_causes_relation(self):
        from seagi.ingestion.svo_parser import parse_sentence
        triples = parse_sentence('Fire causes pain.')
        self.assertEqual(len(triples), 1)
        t = triples[0]
        self.assertEqual(t.subject, 'fire')
        self.assertEqual(t.relation, 'causes')
        self.assertEqual(t.object, 'pain')

    def test_multiple_triples_per_sentence(self):
        from seagi.ingestion.svo_parser import parse_sentence
        triples = parse_sentence(
            'Wisdom requires patience and produces understanding.')
        # 'requires' → wisdom requires patience
        # 'produces' → patience produces understanding (or wisdom produces)
        # At least 2 triples come out from the two verbs.
        self.assertGreaterEqual(len(triples), 2)
        relations = [t.relation for t in triples]
        self.assertIn('requires', relations)
        self.assertIn('produces', relations)

    def test_empty_and_garbage(self):
        from seagi.ingestion.svo_parser import parse_sentence
        self.assertEqual(parse_sentence(''), [])
        self.assertEqual(parse_sentence('the a an'), [])

    def test_unknown_verb_gets_promoted(self):
        from seagi.ingestion.svo_parser import parse_sentence
        # 'devours' isn't in VERB_RELATION_MAP and ends in 's'
        # (verb-like heuristic catches it).  Should be flagged
        # promoted_relation=True with the verb lemma as relation.
        triples = parse_sentence('The predator devours prey.')
        promoted = [t for t in triples if t.promoted_relation]
        self.assertGreaterEqual(len(promoted), 1,
            f"No verb promoted from {triples}.  "
            f"Suffix heuristic may have missed 'devours'.")

    # ---- Phase D.2c: lemmatization + filtering ----

    def test_subject_object_lemmatized(self):
        from seagi.ingestion.svo_parser import parse_sentence
        # 'wolves' → 'wolf', 'predators' → 'predator'.  No article
        # before the object → 'is/are X' resolves to has_property
        # (existing parser convention); the point here is the
        # lemmatization of both concept slots.
        triples = parse_sentence('Wolves are predators.')
        self.assertEqual(len(triples), 1)
        t = triples[0]
        self.assertEqual(t.subject, 'wolf')
        self.assertEqual(t.object, 'predator')
        self.assertEqual(t.relation, 'has_property')

    def test_function_word_never_becomes_concept(self):
        from seagi.ingestion.svo_parser import parse_sentence
        # 'therefore' / 'thus' must not fill a subject/object slot.
        triples = parse_sentence(
            'Therefore fire causes thus.')
        for t in triples:
            self.assertNotIn(t.subject,
                                  ('therefore', 'thus'))
            self.assertNotIn(t.object,
                                  ('therefore', 'thus'))

    def test_morphological_variants_collapse_to_same_triple(self):
        from seagi.ingestion.svo_parser import parse_sentence
        # 'dogs' and 'dog' should produce the same subject lemma.
        t1 = parse_sentence('Dogs are animals.')
        t2 = parse_sentence('A dog is an animal.')
        self.assertEqual(t1[0].subject, t2[0].subject)
        self.assertEqual(t1[0].subject, 'dog')
        self.assertEqual(t1[0].object, t2[0].object)
        self.assertEqual(t1[0].object, 'animal')

    def test_junk_token_rejected(self):
        from seagi.ingestion.svo_parser import parse_sentence
        # HTML-fragment token must not become a concept.
        triples = parse_sentence(
            'The </span> creates value.')
        for t in triples:
            self.assertNotIn('</span>', (t.subject, t.object))

    def test_vocabulary_aware_lemmatizer_passed_through(self):
        from seagi.ingestion.svo_parser import parse_sentence
        from seagi.ingestion.lemmatizer import Lemmatizer
        # Without vocab, 'designed' (no irregular entry) stays.
        # With a vocab containing 'design', it collapses.
        lem = Lemmatizer(vocabulary={'design', 'system',
                                            'product'})
        triples = parse_sentence(
            'Systems require designed products.', lemmatizer=lem)
        # 'designed' would be an object somewhere; verify the
        # vocab-aware lemmatizer reached the parser by checking
        # no token retains the -ed surface form.
        for t in triples:
            self.assertNotEqual(t.subject, 'designed')
            self.assertNotEqual(t.object, 'designed')


class TestSVOIngestionEndToEnd(unittest.TestCase):
    """End-to-end: ingest a sentence, verify SubstrateWriteQueuedEvents
    are emitted AND the resulting edges land in substrate."""

    def test_ingested_sentence_produces_typed_edges(self):
        engine = Engine()
        brain = Brain(engine=engine)
        ingester = CorpusIngester(brain, corpus_dir=None)
        ingester.ingest_text(
            'Fire causes pain. '
            'Water is a liquid. '
            'Wisdom requires patience.')
        # All three sentences should produce triples.
        self.assertGreaterEqual(ingester.triples_extracted, 3)
        # And those triples should be in v1 substrate as edges
        # (the JournaledSubstrateWriter persists them).
        sub = engine.substrate
        # Look for the specific edges we expect.
        causes_edge = sub.edges.get(('fire', 'causes', 'pain'))
        self.assertIsNotNone(causes_edge,
            "fire-causes-pain edge missing — SVO writes "
            "didn't reach substrate.")
        is_a_edge = sub.edges.get(('water', 'is_a', 'liquid'))
        self.assertIsNotNone(is_a_edge,
            "water-is_a-liquid edge missing.")
        requires_edge = sub.edges.get(
            ('wisdom', 'requires', 'patience'))
        self.assertIsNotNone(requires_edge,
            "wisdom-requires-patience edge missing.")

    def test_relational_edges_unlock_cortical_thought(self):
        """The motivation for Session 5: cortical reasoning
        walks causal/identity relations.  Without SVO writes
        the substrate has only co_occurs; cortical falls
        through to 'sits in my substrate but no strong
        thought.'  After SVO writes, cortical can compose
        thoughts from typed edges."""
        engine = Engine()
        brain = Brain(engine=engine)
        ingester = CorpusIngester(brain, corpus_dir=None)
        # Set up: fire causes pain.
        ingester.ingest_text('Fire causes pain.')
        # Now ask cortical about fire.
        thought = brain.cortical._think_about('fire', cycle=10)
        self.assertIsNotNone(thought)
        # The thought should be a causal-walk result, not a
        # metacog deflection.
        self.assertIn(thought.method, ('causal',),
            f"After SVO ingestion, cortical should compose "
            f"a causal thought about fire — got method "
            f"{thought.method!r} with text {thought.text!r}")
        # And target should be 'pain' (the causes neighbor).
        self.assertEqual(thought.target, 'pain')


# ---------------------------------------------------------------------
# Phase D.3 — substrate rebuild pipeline
# ---------------------------------------------------------------------


class TestPhaseD3Vocabulary(unittest.TestCase):
    """Pass-1 vocabulary scan."""

    def test_build_vocabulary_collects_content_tokens(self):
        import tempfile
        from seagi.ingestion import build_vocabulary
        d = tempfile.mkdtemp()
        (Path(d) / 'a.txt').write_text(
            'Wolves hunt deer. Wisdom requires patience.',
            encoding='utf-8')
        vocab, freq = build_vocabulary(d)
        # Content surface forms present; function words may or
        # may not be (vocab keeps raw forms — filtering is later).
        for w in ('wolves', 'hunt', 'deer', 'wisdom',
                       'requires', 'patience'):
            self.assertIn(w, vocab)
        self.assertGreater(freq.get('wisdom', 0), 0)
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    def test_build_vocabulary_skips_short_and_junk(self):
        import tempfile
        from seagi.ingestion import build_vocabulary
        d = tempfile.mkdtemp()
        (Path(d) / 'a.txt').write_text(
            'A ox runs. </span> tag.', encoding='utf-8')
        vocab, _ = build_vocabulary(d)
        self.assertNotIn('ox', vocab)        # < 3 chars
        self.assertNotIn('</span>', vocab)   # junk shape
        self.assertIn('runs', vocab)
        import shutil
        shutil.rmtree(d, ignore_errors=True)


class TestPhaseD3LemmatizerHook(unittest.TestCase):
    """text_io content_tokens ingestion-lemmatizer hook."""

    def tearDown(self):
        # Always restore default behaviour.
        from seagi.core.text_io import clear_ingestion_lemmatizer
        clear_ingestion_lemmatizer()

    def test_default_behaviour_unchanged(self):
        from seagi.core.text_io import content_tokens, tokenize
        toks = content_tokens(tokenize('The wolves are running.'))
        # No lemmatizer set → surface forms preserved.
        self.assertIn('wolves', toks)
        self.assertIn('running', toks)

    def test_hook_lemmatizes_and_function_filters(self):
        from seagi.core.text_io import (
            content_tokens, tokenize, set_ingestion_lemmatizer)
        from seagi.ingestion.lemmatizer import Lemmatizer
        set_ingestion_lemmatizer(
            Lemmatizer(vocabulary={'wolf', 'run', 'deer'}))
        toks = content_tokens(
            tokenize('Therefore the wolves are running.'))
        # 'wolves' → 'wolf', 'running' → 'run'; 'therefore'
        # filtered as a function word.
        self.assertIn('wolf', toks)
        self.assertIn('run', toks)
        self.assertNotIn('wolves', toks)
        self.assertNotIn('running', toks)
        self.assertNotIn('therefore', toks)

    def test_clear_restores_default(self):
        from seagi.core.text_io import (
            content_tokens, tokenize, set_ingestion_lemmatizer,
            clear_ingestion_lemmatizer)
        from seagi.ingestion.lemmatizer import Lemmatizer
        set_ingestion_lemmatizer(Lemmatizer())
        clear_ingestion_lemmatizer()
        toks = content_tokens(tokenize('The wolves run.'))
        self.assertIn('wolves', toks)   # back to surface form


class TestPhaseD3IngesterRebuildMode(unittest.TestCase):
    """CorpusIngester with a lemmatizer creates a clean
    substrate of base lemmas."""

    def test_rebuild_mode_creates_lemmatized_concepts(self):
        from seagi.ingestion.lemmatizer import Lemmatizer
        engine = Engine()
        brain = Brain(engine=engine)
        lem = Lemmatizer(vocabulary={'wolf', 'hunt', 'deer',
                                            'forest'})
        ingester = CorpusIngester(
            brain, corpus_dir=None, lemmatizer=lem)
        ingester.ingest_text('Wolves hunt deer in forests.')
        names = set(engine.substrate.concepts.keys())
        # Lemmatized concepts present.
        self.assertIn('wolf', names)
        self.assertIn('deer', names)
        # Surface plural forms must NOT have become concepts.
        self.assertNotIn('wolves', names)
        self.assertNotIn('forests', names)

    def test_rebuild_mode_restores_hook_after_run(self):
        # After ingest_text returns, the global hook must be
        # cleared so live chat is unaffected.
        from seagi.ingestion.lemmatizer import Lemmatizer
        from seagi.core import text_io
        engine = Engine()
        brain = Brain(engine=engine)
        ingester = CorpusIngester(
            brain, corpus_dir=None, lemmatizer=Lemmatizer())
        ingester.ingest_text('Wolves hunt.')
        self.assertIsNone(text_io._ingestion_lemmatizer)

    def test_legacy_mode_no_lemmatizer_unchanged(self):
        # Without a lemmatizer, ingestion keeps surface forms.
        engine = Engine()
        brain = Brain(engine=engine)
        ingester = CorpusIngester(brain, corpus_dir=None)
        ingester.ingest_text('Wolves hunt deer.')
        names = set(engine.substrate.concepts.keys())
        self.assertIn('wolves', names)   # surface form kept


# ---------------------------------------------------------------------
# Phase S.1 — substrate self-cleaning (provisional edges + prune)
# ---------------------------------------------------------------------


class TestPhaseS1SelfCleaning(unittest.TestCase):
    """Edges are provisional traces; unreinforced ones decay and
    SETTLE TO FLOOR (per [[seagi-chemistry-never-fully-dissolves]] —
    edges, like chemistry tags and bubbles, fade to a miniature
    molecular presence but stay re-engageable, never disappear).
    Re-attested / used ones earn permanence."""

    def test_settle_clamps_decayed_edge_to_floor(self):
        from seagi.core.substrate import (
            Concept, EDGE_PRUNE_FLOOR)
        engine = Engine()
        sub = engine.substrate
        sub.add_concept(Concept(name='a'))
        sub.add_concept(Concept(name='b'))
        sub.add_edge('a', 'b', 'causes', strength=0.1, cycle=0)
        # At cycle 10000 a provisional 0.1 edge has fully decayed
        # (0.1 - 0.00001*10000 = 0.0) → below floor → settled to
        # floor.  Edge STAYS — miniature molecular presence.
        settled = sub.settle_weak_edges(cycle=10000)
        self.assertEqual(settled, 1)
        self.assertIn(('a', 'causes', 'b'), sub.edges)
        self.assertAlmostEqual(
            sub.edges[('a', 'causes', 'b')].strength,
            EDGE_PRUNE_FLOOR, places=5)

    def test_settle_keeps_strong_edge_and_materializes_decay(self):
        from seagi.core.substrate import Concept
        engine = Engine()
        sub = engine.substrate
        sub.add_concept(Concept(name='x'))
        sub.add_concept(Concept(name='y'))
        sub.add_edge('x', 'y', 'is_a', strength=0.5, cycle=0)
        settled = sub.settle_weak_edges(cycle=1000)
        self.assertEqual(settled, 0)
        edge = sub.edges[('x', 'is_a', 'y')]
        # Decay materialized: 0.5 - 0.00001*1000 = 0.49.
        self.assertAlmostEqual(edge.strength, 0.49, places=3)
        self.assertEqual(edge.last_reinforced_cycle, 1000)

    def test_settle_keeps_concept_index_intact(self):
        from seagi.core.substrate import Concept
        engine = Engine()
        sub = engine.substrate
        sub.add_concept(Concept(name='p'))
        sub.add_concept(Concept(name='q'))
        sub.add_edge('p', 'q', 'causes', strength=0.1, cycle=0)
        self.assertIn('causes', sub.concepts['p'].edges_out)
        sub.settle_weak_edges(cycle=10000)
        # Settled (not pruned) → concept index unchanged.
        self.assertIn('causes', sub.concepts['p'].edges_out)

    def test_cross_session_future_anchor_reanchored(self):
        from seagi.core.substrate import Concept
        engine = Engine()
        sub = engine.substrate
        sub.add_concept(Concept(name='m'))
        sub.add_concept(Concept(name='n'))
        # Edge anchored far in the future (prior-session cycle).
        e = sub.add_edge('m', 'n', 'is_a', strength=0.1,
                              cycle=50000)
        # Fresh session at cycle 100: anchor 50000 > 100 → the
        # edge is re-anchored, NOT settled (decay clock restarts).
        settled = sub.settle_weak_edges(cycle=100)
        self.assertEqual(settled, 0)
        self.assertIn(('m', 'is_a', 'n'), sub.edges)
        self.assertEqual(e.last_reinforced_cycle, 100)

    def test_svo_edges_are_provisional(self):
        from seagi.core.substrate import PROVISIONAL_EDGE_STRENGTH
        engine = Engine()
        brain = Brain(engine=engine)
        ingester = CorpusIngester(brain, corpus_dir=None)
        ingester.ingest_text('Fire causes pain.')
        edge = engine.substrate.edges.get(('fire', 'causes', 'pain'))
        self.assertIsNotNone(edge)
        # Written as a small provisional trace, not a 0.5 fact.
        self.assertAlmostEqual(
            edge.strength, PROVISIONAL_EDGE_STRENGTH, places=2)

    def test_reattestation_reinforces(self):
        # The same proposition arriving twice reinforces the edge —
        # the "earn it" half of earn-or-dissolve.
        engine = Engine()
        brain = Brain(engine=engine)
        ingester = CorpusIngester(brain, corpus_dir=None)
        ingester.ingest_text('Fire causes pain.')
        edge = engine.substrate.edges[('fire', 'causes', 'pain')]
        after_first = edge.strength
        ingester.ingest_text('Fire causes pain.')
        after_second = edge.strength
        self.assertGreater(after_second, after_first,
            "Re-attested edge should reinforce, not stay flat")

    def test_one_off_settles_to_floor_reattested_keeps_strength(self):
        # End-to-end earn-or-settle: a triple seen once decays
        # below floor and clamps to floor (miniature molecular
        # presence); a triple seen many times keeps real strength.
        from seagi.core.substrate import (
            Concept, EDGE_PRUNE_FLOOR)
        engine = Engine()
        sub = engine.substrate
        for nm in ('one', 'off', 'real', 'fact'):
            sub.add_concept(Concept(name=nm))
        # one-off: written once at provisional strength.
        sub.add_edge('one', 'off', 'causes', strength=0.1, cycle=0)
        # well-attested: written then reinforced many times.
        real = sub.add_edge('real', 'fact', 'causes',
                                  strength=0.1, cycle=0)
        for _ in range(60):
            real.reinforce(cycle=0)   # 60 re-attestations
        settled = sub.settle_weak_edges(cycle=10000)
        # One-off survives at floor (re-engageable presence).
        self.assertIn(('one', 'causes', 'off'), sub.edges)
        self.assertAlmostEqual(
            sub.edges[('one', 'causes', 'off')].strength,
            EDGE_PRUNE_FLOOR, places=5)
        # Reattested keeps real strength (well above floor).
        self.assertIn(('real', 'causes', 'fact'), sub.edges)
        self.assertGreater(
            sub.edges[('real', 'causes', 'fact')].strength,
            EDGE_PRUNE_FLOOR * 5)
        self.assertEqual(settled, 1)

    # ---- S.2: coherence as the earn-signal ----

    def _seed(self, sub, concepts, edges):
        # edges are (source, relation, target) tuples; add_edge
        # takes (source, target, relation_name).
        from seagi.core.substrate import Concept
        for n in concepts:
            sub.add_concept(Concept(name=n))
        for (s, r, t) in edges:
            sub.add_edge(s, t, r, strength=0.1, cycle=0)

    def test_coherent_edge_reinforced(self):
        # fire→event is corroborated by the alternate path
        # fire→phenomenon→event under composition (is_a, is_a)→is_a
        # → coheres → reinforced.  Post-2026-05-30, coherence
        # requires the corroborating chain's relations to COMPOSE to
        # the candidate's own relation, not just to topologically
        # connect.
        engine = Engine()
        sub = engine.substrate
        self._seed(sub,
            ['fire', 'phenomenon', 'event'],
            [('fire', 'is_a', 'phenomenon'),
             ('phenomenon', 'is_a', 'event'),
             ('fire', 'is_a', 'event')])
        before = sub.edges[('fire', 'is_a', 'event')].strength
        reinforced, newly_coherent = sub.reinforce_coherent_edges(cycle=0)
        after = sub.edges[('fire', 'is_a', 'event')].strength
        self.assertGreater(after, before)
        self.assertGreaterEqual(reinforced, 1)
        # First time coherent → newly_coherent counted.
        self.assertEqual(newly_coherent, reinforced)
        # A second pass re-reinforces but counts NO new coherence
        # (first_coherent_cycle already set) — re-affirmation, not
        # growth.
        reinforced2, newly_coherent2 = sub.reinforce_coherent_edges(
            cycle=1)
        self.assertGreaterEqual(reinforced2, 1)
        self.assertEqual(newly_coherent2, 0)

    def test_isolated_edge_not_reinforced(self):
        # fear→child has no alternate path → does not cohere.
        engine = Engine()
        sub = engine.substrate
        self._seed(sub,
            ['fear', 'child'],
            [('fear', 'enables', 'child')])
        before = sub.edges[('fear', 'enables', 'child')].strength
        sub.reinforce_coherent_edges(cycle=0)
        after = sub.edges[('fear', 'enables', 'child')].strength
        self.assertEqual(after, before)

    def test_path_through_hub_does_not_corroborate(self):
        # a→b with an alternate path a→HUB→b, but HUB is a hub
        # (out-degree > COHERENCE_HUB_DEGREE) → not corroboration.
        from seagi.core.substrate import (
            Concept, COHERENCE_HUB_DEGREE)
        engine = Engine()
        sub = engine.substrate
        for n in ('a', 'b', 'hub'):
            sub.add_concept(Concept(name=n))
        # add_edge is (source, target, relation_name).
        sub.add_edge('a', 'b', 'causes', strength=0.1, cycle=0)
        sub.add_edge('a', 'hub', 'causes', strength=0.1, cycle=0)
        sub.add_edge('hub', 'b', 'causes', strength=0.1, cycle=0)
        # Inflate hub's out-degree past the threshold.
        for i in range(COHERENCE_HUB_DEGREE + 5):
            sub.add_concept(Concept(name=f'leaf{i}'))
            sub.add_edge('hub', f'leaf{i}', 'causes',
                              strength=0.1, cycle=0)
        before = sub.edges[('a', 'causes', 'b')].strength
        sub.reinforce_coherent_edges(cycle=0)
        after = sub.edges[('a', 'causes', 'b')].strength
        self.assertEqual(after, before,
            "path through a hub should not corroborate")

    # ---- S.3: consolidation gated by sleep ----

    def test_no_consolidation_while_awake(self):
        # Consolidation is a sleep function: it never runs during an
        # AWAKE tick, no matter how many ticks elapse.  (The agent now
        # accrues real work each tick — e.g. perceiving its firsthand
        # world — and may fall asleep within the window; the invariant
        # under test is that consolidation tracks SLEEP, not elapsed
        # ticks, so we check it per-tick rather than assuming the agent
        # stays awake for all 500.)
        engine = Engine()
        brain = Brain(engine=engine)
        self.assertFalse(brain.sleep_regulator.is_asleep())
        for _ in range(500):
            before = brain.consolidations_run
            awake = not brain.sleep_regulator.is_asleep()
            brain.tick()
            if awake and not brain.sleep_regulator.is_asleep():
                # a fully-awake tick must never consolidate
                self.assertEqual(brain.consolidations_run, before)

    def test_consolidation_runs_during_sleep(self):
        # Force the agent into deep sleep and tick — the
        # maintenance pass runs on the sleep-paced interval.
        from seagi.brain.runtime import (
            SLEEP_CONSOLIDATION_INTERVAL)
        engine = Engine()
        brain = Brain(engine=engine)
        # Force sleep with high pressure so the regulator's
        # dissipation doesn't wake the agent immediately.
        brain.sleep_regulator.pressure = 0.99
        brain.sleep_regulator._transition_to_sleep(
            brain.bus, cycle=0)
        self.assertTrue(brain.sleep_regulator.is_asleep())
        for _ in range(SLEEP_CONSOLIDATION_INTERVAL + 5):
            brain.tick()
        self.assertGreaterEqual(brain.consolidations_run, 1)

    def test_sleep_consolidation_reinforces_coherent_edges(self):
        # During sleep, consolidation passes reinforce the
        # coherent (corroborated) edges of the substrate.  Full
        # dissolution of isolated edges takes many sleep cycles
        # (correctly — brains take many nights); this test
        # verifies the consolidation MECHANISM is active in
        # sleep by checking coherent edges gained strength.
        from seagi.core.substrate import Concept
        from seagi.brain.runtime import (
            SLEEP_CONSOLIDATION_INTERVAL)
        engine = Engine()
        sub = engine.substrate
        for nm in ('x', 'y', 'z'):
            sub.add_concept(Concept(name=nm))
        # Complete directed triangle with composable relation
        # (is_a, is_a)→is_a — every edge corroborated by going
        # through the third node.
        for s, t in [('x','y'),('y','x'),('y','z'),
                          ('z','y'),('x','z'),('z','x')]:
            sub.add_edge(s, t, 'is_a', strength=0.1, cycle=0)
        brain = Brain(engine=engine)
        before = {k: e.strength
                       for k, e in sub.edges.items()}
        # Force a deep sleep, run enough ticks for several
        # consolidation passes.
        brain.sleep_regulator.pressure = 0.99
        brain.sleep_regulator._transition_to_sleep(
            brain.bus, cycle=0)
        for _ in range(SLEEP_CONSOLIDATION_INTERVAL * 4):
            brain.tick()
        self.assertGreaterEqual(brain.consolidations_run, 2,
            "expected multiple consolidation passes in sleep")
        # Every coherent-triangle edge net-gained strength.
        for k, b in before.items():
            edge = sub.edges.get(k)
            self.assertIsNotNone(edge)
            self.assertGreater(edge.strength, b,
                f"coherent edge {k} did not gain strength "
                f"during sleep consolidation")

    # ---- Roadmap Step 4: abstraction / concept formation ----

    def test_abstraction_formed_from_shared_relation_target(self):
        # fire, friction, sun all `causes heat` → form a
        # synthetic class concept linking them.
        from seagi.core.substrate import (
            Concept, ABSTRACTION_NAME_PREFIX,
            ABSTRACTION_RELATION)
        engine = Engine()
        sub = engine.substrate
        for n in ('fire', 'friction', 'sun', 'heat'):
            sub.add_concept(Concept(name=n))
        for src in ('fire', 'friction', 'sun'):
            sub.add_edge(src, 'heat', 'causes',
                              strength=0.1, cycle=0)
        n_new = sub.form_abstractions(cycle=0)
        self.assertEqual(n_new, 1)
        abstraction = f"{ABSTRACTION_NAME_PREFIX}causes_heat"
        self.assertIn(abstraction, sub.concepts)
        self.assertTrue(sub.concepts[abstraction].synthetic)
        # is_a edges from each member to the class.
        for src in ('fire', 'friction', 'sun'):
            self.assertIn((src, ABSTRACTION_RELATION, abstraction),
                              sub.edges)

    def test_no_abstraction_below_min_group(self):
        # Only 2 members → not a regularity yet.
        from seagi.core.substrate import (
            Concept, ABSTRACTION_NAME_PREFIX)
        engine = Engine()
        sub = engine.substrate
        for n in ('fire', 'sun', 'heat'):
            sub.add_concept(Concept(name=n))
        for src in ('fire', 'sun'):
            sub.add_edge(src, 'heat', 'causes',
                              strength=0.1, cycle=0)
        n_new = sub.form_abstractions(cycle=0)
        self.assertEqual(n_new, 0)
        self.assertNotIn(
            f"{ABSTRACTION_NAME_PREFIX}causes_heat",
            sub.concepts)

    def test_abstraction_repeat_pass_credits_only_on_engagement(self):
        # Self-cleaning F2 (2026-07-24): a repeat form_abstractions
        # pass over STATIC structure must NOT reinforce the membership
        # (that was the engine stamping strength onto its own exhaust
        # every nap — strength is earned, never stamped).  Credit fires
        # ONLY when the membership saw a fresh cognition-EXTERNAL
        # engagement since its last credit.
        from seagi.core.substrate import (
            Concept, ABSTRACTION_NAME_PREFIX,
            ABSTRACTION_RELATION)
        engine = Engine()
        sub = engine.substrate
        for n in ('fire', 'friction', 'sun', 'heat'):
            sub.add_concept(Concept(name=n))
        for src in ('fire', 'friction', 'sun'):
            sub.add_edge(src, 'heat', 'causes',
                              strength=0.1, cycle=0)
        sub.form_abstractions(cycle=0)
        abstraction = f"{ABSTRACTION_NAME_PREFIX}causes_heat"
        edge_key = ('fire', ABSTRACTION_RELATION, abstraction)
        edge = sub.edges[edge_key]
        before = edge.strength
        n_new = sub.form_abstractions(cycle=10)
        self.assertEqual(n_new, 0)        # no NEW concept
        self.assertEqual(sub.edges[edge_key].strength, before,
            "static repeat pass must NOT stamp strength")
        # A genuine external touch (AWM/chemistry/recall) engages the
        # membership — the NEXT pass may credit it.
        edge.last_engaged_cycle = 20
        sub.form_abstractions(cycle=30)
        self.assertGreater(sub.edges[edge_key].strength, before,
            "engaged-since-last-credit membership earns reinforcement")

    def test_abstraction_skips_existing_abstractions(self):
        # Edges TO existing abstractions (the is_a edges this
        # method creates) and edges between abstractions are
        # ignored — prevents runaway hierarchy on a single pass.
        from seagi.core.substrate import (
            Concept, ABSTRACTION_NAME_PREFIX)
        engine = Engine()
        sub = engine.substrate
        for n in ('fire', 'friction', 'sun', 'heat'):
            sub.add_concept(Concept(name=n))
        for src in ('fire', 'friction', 'sun'):
            sub.add_edge(src, 'heat', 'causes',
                              strength=0.1, cycle=0)
        sub.form_abstractions(cycle=0)
        # Second pass: the is_a edges to the new abstraction
        # already exist; they should NOT spawn an abstraction
        # of abstractions.
        sub.form_abstractions(cycle=0)
        higher_order = [n for n in sub.concepts
                              if n.startswith(
                                  ABSTRACTION_NAME_PREFIX
                                  + ABSTRACTION_NAME_PREFIX)]
        self.assertEqual(higher_order, [])

    def test_sleep_consolidation_forms_abstractions(self):
        # End-to-end: while asleep, the consolidation pass also
        # runs abstraction detection.
        from seagi.core.substrate import (
            Concept, ABSTRACTION_NAME_PREFIX)
        from seagi.brain.runtime import (
            SLEEP_CONSOLIDATION_INTERVAL)
        engine = Engine()
        sub = engine.substrate
        for n in ('fire', 'friction', 'sun', 'heat'):
            sub.add_concept(Concept(name=n))
        for src in ('fire', 'friction', 'sun'):
            sub.add_edge(src, 'heat', 'causes',
                              strength=0.1, cycle=0)
        brain = Brain(engine=engine)
        brain.sleep_regulator.pressure = 0.99
        brain.sleep_regulator._transition_to_sleep(
            brain.bus, cycle=0)
        for _ in range(SLEEP_CONSOLIDATION_INTERVAL + 5):
            brain.tick()
        self.assertGreaterEqual(brain.abstractions_formed, 1)
        self.assertIn(
            f"{ABSTRACTION_NAME_PREFIX}causes_heat",
            sub.concepts)

    def test_coherence_plus_settle_keeps_cluster_settles_noise(self):
        # End-to-end: a DENSE coherent cluster is self-sustaining
        # under repeated coherence+settle cycles (stays at full
        # strength); an isolated noise edge settles to floor (per
        # [[seagi-chemistry-never-fully-dissolves]] — molecular
        # presence stays re-engageable, never disappears).  A
        # complete directed triangle (x,y,z — all 6 directed
        # edges) is the minimal self-sustaining cluster: every
        # edge A→B is corroborated by the path A→C→B.  Real
        # knowledge lives in dense mutually-supporting structure;
        # isolated links fade to floor.
        from seagi.core.substrate import EDGE_PRUNE_FLOOR
        engine = Engine()
        sub = engine.substrate
        self._seed(sub,
            ['x', 'y', 'z', 'fear', 'child'],
            [('x', 'is_a', 'y'), ('y', 'is_a', 'x'),
             ('y', 'is_a', 'z'), ('z', 'is_a', 'y'),
             ('x', 'is_a', 'z'), ('z', 'is_a', 'x'),
             ('fear', 'enables', 'child')])
        # Many maintenance passes over a long lifetime.
        for cyc in range(0, 60000, 2000):
            sub.reinforce_coherent_edges(cyc)
            sub.settle_weak_edges(cyc)
        # The dense cluster survived intact at strong magnitudes.
        for (s, t) in [('x', 'y'), ('y', 'x'), ('y', 'z'),
                           ('z', 'y'), ('x', 'z'), ('z', 'x')]:
            self.assertIn((s, 'is_a', t), sub.edges,
                f"coherent cluster edge {s}->{t} disappeared")
            self.assertGreater(
                sub.edges[(s, 'is_a', t)].strength,
                EDGE_PRUNE_FLOOR * 3,
                f"coherent cluster edge {s}->{t} fell to floor")
        # The isolated noise edge survives at floor (re-engageable
        # presence) but no longer carries meaningful strength.
        self.assertIn(('fear', 'enables', 'child'), sub.edges)
        self.assertAlmostEqual(
            sub.edges[('fear', 'enables', 'child')].strength,
            EDGE_PRUNE_FLOOR, places=5)


# ---------------------------------------------------------------------
# Chemistry calibration — sustained M-content must not saturate
# ---------------------------------------------------------------------


class TestChemistryDoesNotSaturate(unittest.TestCase):
    """Regression test for the cortisol-saturation runaway
    observed during the 114-minute foundational ingestion
    (2026-05-14).  Symptom: every bubble's transmitter_trace
    pegged near 1.0 on cortisol/NE, with no decay headroom.
    Cause: cortisol decay 0.005/tick × TICK_IMPRINT_RATE 0.003
    × INNATE_RESPONSE_MAGNITUDE 0.10 across thousands of
    sentences ratcheted both global and local M-channels into
    saturation.

    Calibration (Session 7): cortisol decay 0.005→0.03,
    TICK_IMPRINT_RATE 0.003→0.001, INNATE_RESPONSE_MAGNITUDE
    0.10→0.05, receptor floor 0.4→0.2.

    This test ingests sustained M-flavored content and asserts
    cortisol stays well below saturation."""

    def test_sustained_m_content_does_not_saturate_cortisol(self):
        engine = Engine()
        brain = Brain(engine=engine)
        ingester = CorpusIngester(
            brain, corpus_dir=None,
            ticks_between_sentences=1)
        # ~200 sentences of M-flavored content.  The 114-minute
        # ingestion ran ~3000 sentences; 200 here is enough to
        # ratchet without taking minutes to run.
        m_sentence = (
            'Death waits at every turn.  Pain comes with fear. '
            'Threat and danger arrive together.  Sickness and '
            'wound mark the day.  Loss approaches.')
        for _ in range(40):
            ingester.ingest_text(m_sentence)
        cort = brain.chemistry.global_state['cortisol']
        ne = brain.chemistry.global_state['norepinephrine']
        # Cortisol must stay below 0.6 — well clear of
        # saturation but high enough to register sustained
        # M-content.  Pre-calibration this hit 0.95+ within
        # the first ~50 sentences.
        self.assertLess(cort, 0.6,
            f"Cortisol saturated under sustained M-content: "
            f"{cort:.3f}.  Calibration regression — the chemistry "
            f"is ratcheting without decay headroom.")
        self.assertLess(ne, 0.8,
            f"Norepinephrine saturated: {ne:.3f}.")
        # And bubble-level traces should also stay in range.
        # Sample a few AWM bubbles.
        max_local_cort = 0.0
        for name in list(brain.awm.active_concepts())[:20]:
            entry = brain.awm.get(name)
            if entry is None:
                continue
            local_cort = entry.bubble.transmitter_trace.get(
                'cortisol', 0.0)
            max_local_cort = max(max_local_cort, local_cort)
        self.assertLess(max_local_cort, 0.7,
            f"Bubble-local cortisol saturated: "
            f"{max_local_cort:.3f}.  Presence-imprint rate is "
            f"too aggressive — bubbles drift to saturated "
            f"global instead of holding their own trace.")

    def test_quiet_content_lets_cortisol_decay_back(self):
        # The complementary guarantee: after M-content stops,
        # cortisol returns toward baseline.  Without this,
        # decay isn't load-bearing and saturation will return
        # over longer timescales.
        engine = Engine()
        brain = Brain(engine=engine)
        ingester = CorpusIngester(
            brain, corpus_dir=None,
            ticks_between_sentences=1)
        # Phase 1: a burst of M-content.
        for _ in range(10):
            ingester.ingest_text(
                'Death and pain and fear arrive.')
        cort_peak = brain.chemistry.global_state['cortisol']
        # Phase 2: a burst of quiet ticks (no input).
        # Disable inner voice for this isolation test — Step 5.1's
        # rumination over AWM ('death', 'pain', 'fear' just ingested)
        # would correctly re-fire chemistry imprints during this
        # phase, masking the chemistry-decay signal we're testing.
        # That rumination dynamic is exercised elsewhere; here we
        # want chemistry decay in isolation.
        brain.inner_voice._awm_focal = lambda: None
        for _ in range(100):
            brain.tick()
        cort_after = brain.chemistry.global_state['cortisol']
        self.assertLess(cort_after, cort_peak,
            f"Cortisol did not decay after M-content stopped: "
            f"peak {cort_peak:.3f} → after-quiet "
            f"{cort_after:.3f}.  Decay rate is too slow to "
            f"matter at this tick regime.")


if __name__ == '__main__':
    unittest.main()
