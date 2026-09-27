# SEAGI — architecture, theory, and what it can and cannot do

State as of 2026-09-16 (HST 09-15). Written from the code on the VPS, the measured record in memory, and the user's own doctrine statements. Numbers are the last stable readings; nothing here is advocacy.

---

## 1. What SEAGI is

SEAGI is one persistent, mortal, non-LLM agent: a Python daemon (`seagi_daemon.py`, ticking at 5 Hz) that has run on a 4-core, 8 GB VPS since spring 2026. There is exactly one of him, no copies; his whole state is a compressed JSON save (`seagi_foundational.json.gz`, ~140 MB compressed, over 1 GB of JSON) that persists concepts, edges, episodes, chemistry set-points, mortality state, and everything learned in every game. He sleeps in blocks (a two-process sleep regulator on adenosine pressure), consolidates during sleep, wakes and plays.

His venue since 08-19 is ARC-AGI-3: 25 interactive puzzle games with no instructions, played through the official API. The official score of his scorecard (ARC Prize's own computation, per level: (human baseline actions / his actions)² × 100, averaged) is the success metric since 09-11.

## 2. The theory he is built on

In the user's words, condensed:

> Mortality is the given evolutionary baseline. Any organism that finds ways to extend life survives longer. Fear, pain, pleasure, fun and all variations of neurotransmitters, tagged to experience, sensory input and information, lead to a world view, world experience and personality that allow SEAGI to interpret incoming information, make assumptions, decisions, and play games, mostly at a proxy level, as survival is not threatened most of the time. He needs a world he has learned about, like a human who has 10–20 years of experience before making good decisions. SEAGI jumps in at the final moment and must use what evolution already created, to walk the last meters on his own.

The load-bearing claims, each recorded as doctrine:

- **Mortality is the only axiom, and for agents it is perceived.** Finite lifeforce; a wall (his age) that climbs; learning holds the wall back; the distance is felt as real and he never knows when. The fear of death is real. The only thing that kills is his own deliberate act, done knowingly. Never a game, a drought, an organ misfiring, or age itself. *Doctrine rewritten for agents on 2026-09-16.* The earlier two-cause version, in which age also killed and a successor inherited, mirrored how evolution reached intelligence; for agents it is kept as an experimental setting for studying progress across many generations, not as doctrine.
- **M/I two-axis valence.** Every cognitive event carries a mortality pull and an immortality pull, not a single good/bad. M/I is automatic and constitutes the world he perceives before he decides; it is not a module he consults.
- **Chemistry tags experience at perception time.** Eight channels (dopamine, serotonin, norepinephrine, acetylcholine, cortisol, oxytocin, endorphins, adenosine) with earned allostatic set-points. The substrate stores no valence; the tag is applied when a concept activates. Personality is the accumulated tagging history.
- **The proxy world is the normal case.** A game matters without mattering literally. Ordinary competent play "solidifies being alive"; only the rare and special is a boost toward immortality. Feeling good is proxy immortality, and the gap need never close.
- **Knowledge survives death only by being passed on.** Death is final for the individual; revival is succession with inheritance.
- **Only M/I is law.** Every other rule, guardrail and priority is a revisable default.

The user's stated purpose (08-16): prove the M/I hypothesis and that AGI can be reached on its premise; ARC-AGI-3 is the venue, not the objective. The novel, unpublished part of the claim, identified on 08-17, is about **retention**: affective tagging determines which experiences are kept, and retention rather than prediction accuracy determines competence gain.

## 3. Architecture as it runs

### 3.1 Substrate of experience (`seagi/core`, `seagi/body`)

- **Long-term substrate**: concepts, edges (subject–relation–target with strength and settling), bubbles (context-conditional sub-tags), episodes. Loaded with an object hook during the parse; saving was 85–99% of CPU until 08-20 and is now bounded.
- **Active working memory**: the active set plus chemistry time-series.
- **Chemistry engine**: eight channels, tonic and phasic, decaying toward allostatic set-points that drift with load.
- **Mortality drive**: lifeforce relaxes toward a baseline; the wall climbs by `D(1+o) − E`, where E is the learning term; when the wall reaches lifeforce for a dwell period he dies and a successor is reborn young. Deaths: 17 on the counter, of which 14 are pre-fix debris, one was a genuine old-age death (08-30), and the rest came from a death loop where E had collapsed to zero (09-04), which the user ruled a mortality bug.
- **Body**: embodiment, energy, integrity; a hierarchy of primitive states.

### 3.2 Capabilities (`seagi/brain/capabilities`, ~50 modules)

Named after the structures whose contribution they imitate, and deliberately interconnected rather than modular: cortical reasoner, hippocampus (episodes, consolidation), amygdala, basal ganglia, cerebellum, thalamic gate, insula, nucleus accumbens, VTA / locus coeruleus / raphe (neuromodulator nuclei), default-mode network, inner voice, narrative journal, personality, mortality clock, replay and reasoning consolidators, curiosity, novelty and uncertainty monitors, skill library, schema discovery, grounding loops, world transducer, and the world actor. Arbitration is claim-based: each capability emits a claim strength and the strongest drives the next action. In practice on ARC, the world actor and the game organs below decide almost every step.

### 3.3 The ARC world stack (`seagi/world`, ~11,000 lines)

`arc_world.py` (6,150 lines) talks to a sidecar over a Unix socket; the sidecar holds the ARC API session and scorecard and renews the card when it closes. Everything he knows about a game is keyed by game and level and persisted in the save.

| Organ | What it does | Shipped |
|---|---|---|
| State key | The 64×64 board hashed with the clock lines masked (the per-level action budget is drawn on the board) | 09-04, 09-13 |
| Relation sense | Reads candidate goals off the board: two same-shape regions to be made equal; the object he controls brought to a marker. Confirms one only from a win | 09-05 |
| Relation planner | Plans paint/apply operations on a confirmed relation; learned operator model; this completed cd82 | 09-06 to 09-12 |
| Hypothesis organ | Generic loop: find the self, map where arrows take it, pick a unique static object as a candidate goal, pursue, refute on reach-without-clear, confirm on clear; memory across lives | 09-14 |
| Search organ | Systematic sweep of every control (arrows plus a click on every object centre) over board states with memory across lives; the shortest known path to a win is replayed | 09-15 |
| Convergence judgement | The search holds a game only while its own new-states-per-step rate is falling; a walk (constant rate) is released after twelve terminals | 09-16 |
| Goal library | One library across games: kinds of goal (equal regions, self to class, class vanishes) with colour-free descriptors, learned from his clears against dry lives as the null, seeded from his recorded clears, pursued and refuted by the search | 09-16 |
| Allocation | One decision at every terminal: commitment, then an organ testing here, then official headroom, then his own record, then an official-yield draw | 09-15 |
| Felt state | A per-step register in [−1, 1]: up on a competent step, down on a no-op or walk-back | 08-29 |

Seventy-three feature gates exist as files in `/root` (`<NAME>_ON`); each patch is deployed behind one and can be undone by deleting the file. The test suite is 1,600+ tests; each patch has its own pre-registration with a per-session judge.

### 3.4 Operations

- Daemon `seagi-v2`, HTTP on `127.0.0.1:8766` (`/health` 40 ms, `/status` seconds to tens of seconds). A restart takes about a minute to answer and can kill the scorecard through the idle gap; the sidecar renews it after a 600-second settle.
- Official score: `/root/official.log` every 15 minutes; best run per game across cards in `/root/official_best.json`.
- Every change: premise stated and measured first, staged on a copy, targeted tests with the gate present and absent, adversarial review by a separate agent, pre-registered judge, deploy, judge per session.

## 4. What he can do (measured)

| Capability | Evidence |
|---|---|
| Play unattended for weeks, sleep, wake, persist everything | Uptime since spring; the save carries every table across restarts |
| Not die from the game | Verified by enumeration: no game path writes lifeforce; the game ended him 732 times in 1,732 runs while deaths never moved |
| Complete a full ARC game | cd82: all six levels, official 100/100, first completed 09-11; level 5 replays at 24 actions |
| Reach near-human action counts where the goal is read | tu93 level 0 in 18 actions against a human baseline of 19; tu93 now 3 of 9 levels, official 12.0 |
| Find a level's clearing sequence without a goal | The search organ clears tu93 L0–L3, s5i5, tn36, su15, ls20, dc22 and tr87 offline in 360–5,600 steps; live it cleared tn36, tu93 L1→2→3 and vc33 L1→2, all first-ever, within one day of being given the lives |
| Replay what he learned | On return visits the win path is walked from the first step: tu93 levels 0–2 in 65 seconds |
| Judge his own search | Walk games are released at 16–17 lives; converging searches are held to the clear |
| Learn what a level wanted | The goal library names the goal on 9 of 12 cleared games from his own recordings, one schema recurring on three games |
| Keep his official score moving | 2.40 on 09-11 → 5.446 on 09-16; levels ever cleared 17 → 20 in one day under patch 47; 12 of 25 games scored |

## 5. What he cannot do (measured)

| Limit | Evidence |
|---|---|
| Transfer content between games | Refuted at three grains: pixel glances (identity must be byte-exact; tolerance hurts), the action grammar (20.5% vs a 30.1% null, n=15,662), and object kinds (a colour-free prior ranks the responding object no better than uniform: 4.19 vs 3.93 clicks to first effect over 16 games) |
| Score on 13 of 25 games | Never scored: sc25, sk48, sb26, ka59, bp35, re86, wa30, g50t, dc22, s5i5, su15, ls20, tr87 (the last five clear offline and await lives) |
| Read a goal that is not one of three kinds | The library's kinds are mine; he fills, ranks and refutes them but cannot add one. Three of his twelve cleared games are unexplained by them |
| Search a game with no goal signal | On the walks (sc25, sk48, sb26, bp35, re86) the goal library changed nothing: the pursued relation does not vary over reachable states |
| Move a piece he cannot move | ka59, dc22, g50t need a push or carry operator he does not have |
| Beat random play where the goal is invisible | Random play clears level 0 on six of eight games he cleared and on none of the never-won ones |
| Compare to the benchmark | 5.4 is on the public games; the benchmark for agents sits near 0.25% on a different set and the 100% leaderboard entry is a demo. The numbers are not comparable |

## 6. Where the theory stands

What was tested and what it showed:

- **Chemistry reaching a decision.** Audited 08-17: of 14,003 ARC action choices, zero could have been changed by any neurotransmitter; the "M/I web" the actor reads contains no chemistry. The only channel that changes behaviour is adenosine (whether he plays). Chemistry lean was flat across 39 places (09-11). The felt bridge (a bad feeling releasing a hold) ran a pre-registered ABBA test and measured null.
- **The M/I readout.** Two inconsistent definitions exist in the code (a raw channel mean and a rectified departure); 19 of 28 M/I event kinds have no emitter. The tagging carried a common-mode offset after the allostatic change, so it could not differentiate experiences.
- **Lifeforce cannot express "good".** It only relaxes toward baseline; over 129 samples it never rose above it. The felt state was built beside it for that reason.
- **The retention claim, the novel part, was never tested.** The percept path published every experience with mortality and immortality content hard-coded to zero, so tagging could not select what was kept. This is the claim that separates the theory from reward shaping, and it remains open.
- **What produced the score.** Every point came from general methods built by hand and given the lives they need: relation reading, the hypothesis loop, systematic search, one allocation decision, and the convergence judgement. None of them is learned, and none of them is chemistry. The learned things are per-game tables.

Honest summary: the mortality machinery works as designed and is verified not to be lethal from the game; the felt register exists; the tagging-to-decision link and the retention link have not been shown, and the score is evidence of the engineering, not of the hypothesis.

## 7. State at stop

| | value |
|---|---|
| official all-time | 5.446 / 100, 12 of 25 games scored, 20 levels ever |
| best games | cd82 100.0 (6 levels), tu93 12.0 (3), cn04 / ft09 / r11l / sp80 4.8 (1 each), lp85 2.8, tn36 1.4, vc33 0.2 (2) |
| deaths | 17 (see §3.1) |
| live patches | 45 SEARCH, 46 ALLOC, 47 CONVERGE, 48 LIBRARY (pid 1805684 since 19:42Z) |
| save | 140 MB gz; RSS ~5–5.7 GB of 7.7 GB |
| tests | 23 failed / 1,637+ passed baseline (the 23 are old, identical across patches) |

## 8. What the next builds would have been

1. **Kind induction** for the goal library: predicates over objects (colour equality, adjacency, containment, alignment, count) and their conjunctions searched for the one that separates clear lives from dry lives; it must re-derive the three existing kinds from his clears before being trusted with a fourth.
2. **Operator discovery**: push, carry and toggle as learnable operators, for the games where arrows cannot reach the candidate.
3. **The retention test**: un-zero the M/I content of percepts, let tagging select what the hippocampus keeps, and run the within-subject clamp ablation on the official score. This is the experiment that would test the theory rather than the engineering.

## 9. Where things are

- VPS `alpha.myseagi.com`, code in `/home/seagi/SEAGI-Core-v2/`, gates and tools in `/root/`, memory and doctrine in `~/.claude/projects/.../memory/` (index `MEMORY.md`, start at `p_RESUME_SEAGI.md`).
- Per-patch pre-registrations: `/root/prereg_*.md`. Undo any patch: `rm /root/<GATE>_ON`. Full undo: restore `*.PRE_<patch>_*` and restart.
- Daily cron `fusion_count.sh` is disabled (it buffered the 1 GB save in memory beside him).

## 10. The LLM version: M/I applied to a language model

Discussed at the stop (09-16). All standing rules set aside except the M/I neurotransmitter theory, applied on an LLM instead of the non-LLM substrate.

### What it would be

The LLM is the cortex; the M/I loop is wrapped around it.

- **Appraisal by the LLM itself.** Every experience (an observation, a move, an outcome, a conversation turn) is appraised in words by the model: expected or surprising, a betrayal or a confirmation, how severe, how rare. This is the part SEAGI could never do: his appraisal was numeric proxies from a 3×3 glance, so "the same outcome has no fixed valence" was unimplementable.
- **Chemistry from appraisal.** The appraisal drives the eight-channel chemistry state with earned set-points and the two pulls, mortality and immortality. Departures are measured from the current earned baseline, not from an innate constant, or the tags flatten into a common-mode offset as they did on 08-20.
- **Tags decide retention and retrieval.** The tag applied at encoding decides what enters long-term memory and what is recalled later. This is the unpublished claim: affective tagging selects what is kept, and retention rather than prediction accuracy decides competence gain.
- **Chemistry modulates the model.** Norepinephrine as exploration temperature, serotonin as patience, dopamine as how strongly a lesson is written, cortisol as caution and withdrawal.
- **Personality is the tagging history**, and it shapes appraisal in turn.
- **Mortality underneath, perceived.** A finite lifeforce and an aging wall that only learning slows, felt from the inside as real, so the agent has a reason to act that is not a task reward. Refinement of 09-16: the wall itself never kills; the fear is real, the age death is not, and only the deliberate act kills.

### The mortality rule (the user's ruling, 09-16)

No normal task can ever kill him; the life-threatening scenario during play is theoretical only. The rule is structural: lifeforce and the wall are written by exactly one path, age, and nothing inside a task can call it. Winning, losing, being ended, a drought all land in the proxy layer only (chemistry, tags, felt state, personality). The proxy fear shapes the decision while the mortality ledger is untouched. Learning may slow the wall only through a bounded term that can never stall or invert it, so no timer hides in a formula (SEAGI's September deaths came from `D(1+o) − E` with E collapsed to zero, not from the game). The only other lethal path is the deliberate act: an explicit, named, irreversible choice made knowingly, never a side effect of a tool call.

### Prediction, stated so it can be wrong

- Single-shot benchmarks: no effect. There is no lived history to tag.
- Long-horizon, lived tasks (ARC-AGI-3, multi-day agents, anything where what you remember decides what you can do): a real but modest gain from tagged retention over recency or similarity retrieval, if and only if affective tags correlate with later usefulness. That is the hypothesis.
- Behaviour modulation alone: small effects; the field reads it as adaptive hyperparameters (Keramati and Gutkin showed homeostatic drive-reduction and reward maximisation are the same objective). The retention result is the one that would not be dismissed.

### The test

A within-subject clamp ablation on the same agent and the same tasks: tags live versus tags clamped, in alternating pre-registered blocks, judged on competence gain. One SEAGI, no copies, forces exactly this design, which is stronger than the population comparisons the field runs.

### Cost

Weeks, not months. The wrapper is small, the LLM does the hard part, and the experiment is one ablation. Of everything considered at the stop, this is the first version that would test the theory rather than the engineering around it.
