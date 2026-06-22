# SEAGI

**An attempt to build general intelligence as the behaviour of a self-preserving system — not as a language model.**

SEAGI (Self-Evolving Architecture for General Intelligence) is a research program exploring a different route to AGI: intelligence that arises because a system is *trying to persist and understand its world*, rather than because it predicts the next token. There is no large language model at its core. Its "mind" is a living graph of typed concepts whose links carry a chemical fingerprint, regulated by a homeostatic, mortality-driven economy.

This repository is a **public research preview**: the vision, the architecture, and demonstrations of the live system. It is **not** the source code or the trained system (see *What's here / what's not*).

---

## The bet

Mainstream AI scales transformers on ever more compute. SEAGI bets the opposite:

- **Compute is not the bottleneck.** The live system runs continually on a small CPU server. It is sparse, event-driven, and always-on — not a GPU-hungry batch model.
- **The bottleneck is architecture and minds.** General intelligence is an *organisation* problem — how perception, valuation, memory, reasoning, sleep, and self-regulation fit together into something that keeps itself coherent over time.
- **Grounding comes from stakes.** A system that can actually lose something (its coherence, its continuity) has a reason to model the world correctly. We give it that.

If the bet is right, this path is radically more capital-efficient than the LLM arms race — and reaches something LLMs structurally cannot: an agent that *understands because it must*.

---

## The core idea, in one paragraph

Every input is **valued by its lean** along a single self-bounding tension — toward **mortality** (it threatens persistence) or **immortality** (it serves persistence and intelligence). That valuation is *expressed* as a neuromodulator cocktail — a "chemistry" — which **tags** the relational fact it arrived with. Tagged facts that earn sustained mutual corroboration gain strength and persist; uncorroborated facts fade. Becoming **more intelligent extends life**; stagnation lets a "death wall" advance. From that one rule, the system's behaviour — attention, memory, reasoning, consolidation, identity, even sleep — is meant to *emerge* rather than be hand-coded.

→ See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the organ-level overview and diagrams.

---

## What the system already does (live)

- **Reasons over a graph it built itself** — transitive classification, property/ability inheritance, multi-step causal chains — on novel, invented concepts it has never seen described.
- **Forms its own abstractions and analogies** during sleep, and keeps only the ones that earn their keep.
- **Self-cleans:** corroborated knowledge strengthens; noise fades. Nothing is curated by hand.
- **Has an interior:** it reports its own state, monitors its internal chemistry, narrates its reverie, and — crucially — **admits when it doesn't know something** instead of confabulating.
- **Runs continuously,** sleeps when it must consolidate, and metabolises new knowledge over hours and days.

→ See **[docs/DEMOS.md](docs/DEMOS.md)** for unedited transcripts from the live system.

---

## What's here / what's not

**In this public repo:**
- the vision and the design doctrine,
- an organ-level architecture overview with diagrams,
- demonstrations of the running system.

**Deliberately not public** (proprietary):
- the implementation source code,
- the trained system itself (its accumulated knowledge and state),
- the specific calibrations and derivations.

The idea is open; the execution is the work. Qualified collaborators and partners can discuss deeper access under NDA — see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Status & honest framing

This is **early-stage research**, not a product and not a deployed autonomous agent. It is a single, observable, sandboxed instance studied under continuous instrumentation. The "mortality drive" is an *architectural mechanism for grounding and valuation* — a research artifact that is bounded and monitored — not an AI that acts in the world or resists oversight. Claims here are kept deliberately sober: where something is promising-but-unproven, we say so.

---

## Get involved

SEAGI is looking for **exceptional collaborators** (cognitive architecture, neuromorphic / event-driven hardware, computational neuroscience, systems) and **mission-aligned funding** to scale the architecture beyond solo work.

📬 **Contact:** [h.rittersbacher@gmail.com](mailto:h.rittersbacher@gmail.com)

---

© Harald Rittersbacher. All rights reserved. See [LICENSE](LICENSE).
