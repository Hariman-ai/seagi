"""Layer 3 — foundational knowledge ingestion driver.

The doctrine: corpus ingestion under live chemistry forges
personality.  Two SEAGI instances fed the same corpus in
different chemistry states become different people.  This
module is the mechanism.

Design commitments
------------------
1. SENTENCE-BY-SENTENCE pacing.  Each sentence flows through
   `brain.intake()` separately.  Between sentences, the brain
   ticks at its natural rate — chemistry decays, AWM-resident
   bubbles imprint current global chemistry, insula samples
   body, basal ganglia arbitrates.  Whatever Seagi was FEELING
   when he read a particular sentence is what stamps that
   sentence's concepts.

2. NATURAL / UNCONSTRAINED chemistry environment.  No
   modulation, no fixed mood.  Chemistry responds to the
   content as it comes.  The corpus shapes Seagi as it would
   shape a real reader: an M-leaning passage produces cortisol
   and norepinephrine rises; an I-leaning passage produces
   dopamine and oxytocin rises; the next paragraph reads
   AGAINST the trailing chemistry of the previous one.

3. NO PROGRESS WITHOUT IMPRINT.  The pipeline cannot be
   "rushed."  Sentence intake is followed by at least one full
   `brain.tick()` so the chemistry can do its work before the
   next sentence arrives.

4. RESUME-CAPABLE.  Each file processed gets a sidecar `.read`
   marker (timestamp + cycle when finished).  Re-running the
   ingester picks up where it left off; already-read files are
   skipped.  Useful for very large corpora and interruption
   recovery.

5. OBSERVABLE.  `snapshot()` returns the current state of all
   the things that should be moving during ingestion:
   substrate growth, AWM size, chemistry trajectory, episodes
   formed/consolidated, monitor activations.  A callback hook
   lets external code log/checkpoint mid-ingestion.

What this driver does NOT do
----------------------------
- It does not decide what to read.  The corpus is supplied
  by the operator.  What Seagi reads is a personality-forming
  decision and the architecture refuses to make it implicitly.
- It does not rewrite or annotate text.  Sentences are fed
  raw, exactly as written.
- It does not impose any quality threshold beyond the standard
  thalamic gate's attenuation (every sentence reaches
  chemistry; salience scales response).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


# Sentence boundary pattern.  Splits after `.`, `!`, `?` when
# followed by whitespace.  Reasonably accurate for prose;
# accepts occasional sub-sentence chunks at edge cases like
# "Dr." / "U.S.A." — those misses don't break chemistry imprint,
# they just produce slightly shorter "sentences" than perfect
# segmentation would.
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

# Paragraph break pattern — two-or-more newlines.  Used by the
# paragraph-level pacing mode (less brain-correct than sentence
# pacing but available for callers who explicitly want it).
_PARAGRAPH_SPLIT = re.compile(r'\n\s*\n+')


def split_sentences(text: str) -> List[str]:
    """Public sentence segmenter.  Returns non-empty stripped
    sentences in source order."""
    out: List[str] = []
    for chunk in _SENTENCE_SPLIT.split(text or ''):
        s = chunk.strip()
        if s:
            out.append(s)
    return out


def split_paragraphs(text: str) -> List[str]:
    """Public paragraph segmenter."""
    out: List[str] = []
    for chunk in _PARAGRAPH_SPLIT.split(text or ''):
        p = ' '.join(chunk.split())  # collapse internal whitespace
        if p:
            out.append(p)
    return out


# Default pacing — how many `brain.tick()` calls between
# sentences.  1 is the brain-correct minimum: chemistry gets
# one cycle of decay + presence-imprint before the next
# sentence lands.  Operators can increase this for longer
# inter-sentence pauses (simulating slower reading + more
# rumination between sentences).
DEFAULT_TICKS_BETWEEN_SENTENCES = 1

# How often to fire a reflection cycle during long ingestion
# runs.  The brain's idle driver fires reflection naturally
# every ~50 cycles of quiet, but ingestion is NOT quiet —
# percepts keep arriving — so the natural driver never fires.
# We trigger it manually every N sentences so the hippocampus
# can consolidate accumulated episodes and the vmDMN can run
# its narrative pass.
DEFAULT_REFLECT_EVERY_N_SENTENCES = 200


class CorpusIngester:
    """Sentence-by-sentence corpus reader.  Feeds text through
    the v2 brain's intake pipeline at brain-correct pace so
    chemistry forges personality during ingestion.

    Usage
    -----
    >>> ingester = CorpusIngester(brain, 'corpus_pilot/')
    >>> ingester.ingest_corpus()

    Or pass text directly (used by tests and ad-hoc pieces):
    >>> ingester.ingest_text('A short passage.', source='ad-hoc')
    """

    def __init__(self,
                 brain: Any,
                 corpus_dir: Optional[Any] = None,
                 *,
                 ticks_between_sentences: int = (
                     DEFAULT_TICKS_BETWEEN_SENTENCES),
                 reflect_every_n_sentences: int = (
                     DEFAULT_REFLECT_EVERY_N_SENTENCES),
                 sidecar_suffix: str = '.read',
                 lemmatizer: Optional[Any] = None):
        """
        brain: the v2 Brain instance (from
            agi_engine.brain.Brain).
        corpus_dir: directory containing .txt / .md files to
            ingest.  Walked recursively.  None means inline
            text-only mode (ingest_text directly).
        ticks_between_sentences: how many brain.tick() calls
            run between each sentence.  1 is brain-correct
            minimum; higher values simulate slower reading.
        reflect_every_n_sentences: trigger a reflection cycle
            (REFLECTION_FIRED) every N sentences so the
            hippocampus consolidates and vmDMN narrates.
        sidecar_suffix: appended to processed file names to
            mark them as read (e.g. `book.txt.read`).  Enables
            resume.
        lemmatizer: Phase D.3 — when supplied, the ingester runs
            in substrate-rebuild mode.  Concepts created during
            intake are lemmatized + function-word-filtered (via
            text_io's ingestion-lemmatizer hook) and SVO triples
            use the same lemmatizer.  None = legacy behaviour.
        """
        self.brain = brain
        self.corpus_dir = (
            Path(corpus_dir) if corpus_dir is not None else None)
        self.ticks_between_sentences = max(
            1, int(ticks_between_sentences))
        self.reflect_every_n_sentences = max(
            1, int(reflect_every_n_sentences))
        self.sidecar_suffix = sidecar_suffix
        self.lemmatizer = lemmatizer
        # Stats.
        self.sentences_ingested: int = 0
        self.files_ingested: int = 0
        self.files_skipped: int = 0
        self.triples_extracted: int = 0
        # Hook for callers to checkpoint / log mid-run.
        # Called with (ingester, current_file, sentence_idx,
        # total_sentences) after each sentence.
        self.progress_callback: Optional[Callable] = None

    # ---- D.3: rebuild-mode lemmatizer scoping ----

    def _install_lemmatizer(self) -> None:
        """Install the rebuild lemmatizer into text_io's
        content_tokens hook, so concept creation during intake
        lemmatizes + function-filters.  No-op in legacy mode."""
        if self.lemmatizer is None:
            return
        from seagi.core.text_io import set_ingestion_lemmatizer
        set_ingestion_lemmatizer(self.lemmatizer)

    def _uninstall_lemmatizer(self) -> None:
        """Restore default content_tokens behaviour."""
        if self.lemmatizer is None:
            return
        from seagi.core.text_io import clear_ingestion_lemmatizer
        clear_ingestion_lemmatizer()

    # ---- public API ----

    def ingest_corpus(self,
                          file_pattern: str = '**/*.txt'
                          ) -> Dict[str, Any]:
        """Walk corpus_dir, ingesting every text file not yet
        marked as processed.  Returns a summary dict at the end.
        """
        if self.corpus_dir is None:
            raise ValueError(
                "ingest_corpus requires corpus_dir to be set "
                "at construction time")
        if not self.corpus_dir.exists():
            raise FileNotFoundError(
                f"Corpus directory does not exist: "
                f"{self.corpus_dir}")
        files = sorted(self.corpus_dir.glob(file_pattern))
        # Also accept .md files by default.
        md_files = sorted(self.corpus_dir.glob('**/*.md'))
        all_files = list(files) + [
            f for f in md_files if f not in files]
        self._install_lemmatizer()
        try:
            for path in all_files:
                if path.name.endswith(self.sidecar_suffix):
                    continue
                if self._already_processed(path):
                    self.files_skipped += 1
                    continue
                self._ingest_file(path)
                self._mark_processed(path)
                self.files_ingested += 1
        finally:
            self._uninstall_lemmatizer()
        return self.summary()

    def ingest_text(self,
                       text: str,
                       *,
                       source: str = 'inline') -> None:
        """Ingest a string directly, sentence-by-sentence.  Used
        for tests, ad-hoc pieces, or programmatic reading.
        Does not check or write sidecar files."""
        sentences = split_sentences(text)
        self._install_lemmatizer()
        try:
            for i, sentence in enumerate(sentences):
                self._ingest_sentence(sentence, source)
                if self.progress_callback is not None:
                    self.progress_callback(
                        self, source, i, len(sentences))
                # Periodic reflection.
                if (self.sentences_ingested
                        % self.reflect_every_n_sentences == 0
                        and self.sentences_ingested > 0):
                    self._fire_reflection()
        finally:
            self._uninstall_lemmatizer()

    # ---- per-file processing ----

    def _ingest_file(self, path: Path) -> None:
        text = self._read_file_text(path)
        sentences = split_sentences(text)
        for i, sentence in enumerate(sentences):
            self._ingest_sentence(sentence, path.name)
            if self.progress_callback is not None:
                self.progress_callback(
                    self, path, i, len(sentences))
            if (self.sentences_ingested
                    % self.reflect_every_n_sentences == 0
                    and self.sentences_ingested > 0):
                self._fire_reflection()

    def _read_file_text(self, path: Path) -> str:
        """Read text.  Strip HTML tags if the file looks like
        HTML (presence of `<` followed by an alphanumeric or
        `/`).  Otherwise return raw."""
        raw = path.read_text(encoding='utf-8', errors='replace')
        if re.search(r'<[A-Za-z/]', raw):
            return _strip_html(raw)
        return raw

    # ---- per-sentence pipeline ----

    def _ingest_sentence(self,
                                sentence: str,
                                source: str) -> None:
        """One sentence:
          1. send through brain.intake (chemistry tagging,
             AWM, episode formation, all the v2 brain stuff)
          2. extract SVO triples and emit substrate writes
             for each (relational graph)
          3. run N brain.tick()s so chemistry settles
             before the next sentence
        """
        sentence = sentence.strip()
        if not sentence:
            return
        self.brain.intake(
            sentence,
            modality='text',
            origin='ingestion',
            origin_detail=source)
        # Relational ingestion — emit substrate writes for
        # extracted (subject, relation, object) triples.  This
        # is what gives cortical reasoning real edges to walk
        # (causes, is_a, has_property, etc.) rather than just
        # co_occurs from hippocampal consolidation.
        self._emit_svo_triples(sentence, source)
        for _ in range(self.ticks_between_sentences):
            self.brain.tick()
        self.sentences_ingested += 1

    def _emit_svo_triples(self,
                                  sentence: str,
                                  source: str) -> None:
        """Parse the sentence into (subject, relation, object)
        triples and publish a SubstrateWriteQueuedEvent for
        each.  JournaledSubstrateWriter persists them.

        Concepts referenced by triples are already in
        substrate (text_to_observation created them during
        intake), so the writer just adds edges; it won't
        create orphan concepts."""
        # Lazy import to avoid a circular import at module
        # load time (svo_parser imports from seagi.core.text_io
        # which transitively touches seagi.core.bubble's lazy
        # chemistry import).
        from seagi.ingestion.svo_parser import parse_sentence
        from seagi.brain.events import (
            EventKind, SubstrateWriteQueuedEvent)
        import time as _time
        # D.3: in rebuild mode the same vocab-aware lemmatizer
        # used for concept creation also lemmatizes SVO concept
        # slots, so edges and concepts agree on base lemmas.
        triples = parse_sentence(sentence, lemmatizer=self.lemmatizer)
        if not triples:
            return
        cycle = self.brain._cycle_provider()
        # Phase S.1: a parsed triple is ONE encounter — it enters
        # substrate as a PROVISIONAL trace (small strength), not a
        # half-certain fact.  It earns permanence only by being
        # re-attested / walked / coherence-reinforced; otherwise
        # settle_weak_edges fades it to floor (re-engageable
        # presence per [[seagi-chemistry-never-fully-dissolves]]).
        # Doctrine rules 4 + 5.
        from seagi.core.substrate import PROVISIONAL_EDGE_STRENGTH
        for t in triples:
            self.brain.bus.publish(SubstrateWriteQueuedEvent(
                kind=EventKind.SUBSTRATE_WRITE_QUEUED,
                cycle=cycle,
                timestamp=_time.time(),
                source_capability='ingestion',
                origin='ingestion',
                origin_detail=source,
                subject=t.subject,
                relation=t.relation,
                object=t.object,
                strength=PROVISIONAL_EDGE_STRENGTH,
                write_reason='svo_ingest'))
            self.triples_extracted += 1

    # ---- reflection trigger ----

    def _fire_reflection(self) -> None:
        """Manually fire a reflection cycle.  Ingestion keeps
        the brain busy with percepts, so the natural idle-driver
        never fires; this gives hippocampus a chance to
        consolidate episodes accumulated during the corpus and
        vmDMN a chance to narrate."""
        try:
            self.brain.fire_reflection(
                trigger='ingestion_reflect',
                reflection_kind='general')
        except Exception:
            # If fire_reflection isn't available on this brain
            # variant, skip — not load-bearing.
            pass

    # ---- resume tracking ----

    def _sidecar_path(self, path: Path) -> Path:
        return path.with_name(path.name + self.sidecar_suffix)

    def _already_processed(self, path: Path) -> bool:
        return self._sidecar_path(path).exists()

    def _mark_processed(self, path: Path) -> None:
        cycle = 0
        try:
            cycle = int(self.brain._cycle_provider())
        except Exception:
            pass
        sidecar = self._sidecar_path(path)
        sidecar.write_text(
            f"processed at cycle {cycle} ts {time.time()}\n",
            encoding='utf-8')

    # ---- diagnostics ----

    def snapshot(self) -> Dict[str, Any]:
        """Snapshot the brain state mid-ingestion.  Useful for
        logging trajectories or verifying the pipeline is
        actually moving things that should be moving."""
        brain = self.brain
        lts_stats = brain.lts.stats() if brain.lts else {}
        return {
            'sentences_ingested': self.sentences_ingested,
            'files_ingested': self.files_ingested,
            'files_skipped': self.files_skipped,
            'triples_extracted': self.triples_extracted,
            'awm_size': brain.awm.size(),
            'awm_capacity': brain.awm.capacity,
            'substrate_concepts': lts_stats.get('concepts', 0),
            'substrate_edges': lts_stats.get('edges', 0),
            'substrate_episodes': lts_stats.get('episodes', 0),
            'chemistry_state': dict(
                brain.chemistry.global_state),
            'm_polarity': brain.chemistry.m_polarity(),
            'i_polarity': brain.chemistry.i_polarity(),
            'arousal_modulator': (
                brain.chemistry.arousal_modulator()),
            'episodes_formed': (
                brain.hippocampus.episodes_formed),
            'episodes_consolidated': (
                brain.hippocampus.episodes_consolidated),
            'amygdala_threats': brain.amygdala.threats_detected,
            'nacc_opportunities': (
                brain.nucleus_accumbens.opportunities_detected),
            'novelty_detected': (
                brain.novelty_monitor.novel_detected),
            'uncertainties_recorded': (
                brain.uncertainty_monitor.uncertainties_recorded),
            'gate_attenuated': brain.gate.attenuated_count,
            'gate_attended': brain.gate.attended_count,
            'reflections_fired': brain.reflections_fired,
        }

    def summary(self) -> Dict[str, Any]:
        """Final summary at the end of a corpus run."""
        snap = self.snapshot()
        snap['_run_complete'] = True
        return snap


# ---- HTML stripping helper ----


def _strip_html(html: str) -> str:
    """Stdlib-only HTML → text.  Used when an ingested file
    looks like HTML.  Conservative: drop content of
    <script>/<style>/<head>; keep everything else as plain
    text with entities decoded."""
    from html.parser import HTMLParser

    class _Extractor(HTMLParser):
        _DROP = frozenset({'script', 'style', 'head'})

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts: List[str] = []
            self.drop_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag in self._DROP:
                self.drop_depth += 1

        def handle_endtag(self, tag):
            if tag in self._DROP and self.drop_depth > 0:
                self.drop_depth -= 1

        def handle_data(self, data):
            if self.drop_depth == 0:
                self.parts.append(data)

    ex = _Extractor()
    try:
        ex.feed(html)
    except Exception:
        return html
    return ''.join(ex.parts)
