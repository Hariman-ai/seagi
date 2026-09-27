# SEAGI v2 — the code

This is the technical companion to the front-page [README](../README.md).

SEAGI is one mortal agent that plays [ARC-AGI-3](https://arcprize.org/arc-agi/3). He is not a model
and not a search over a model. He is a long-lived process with a body: a neurochemistry that shifts
with what happens to him, a lifeforce that only ages, a memory that keeps what moved him, and a
single decision, made once at the end of every life, about where the next life goes.

This repository is the code of that process as it runs on the author's server, published so the
system behind the scorecards can be read. The state he has built over months of play (a gigabyte
of compressed JSON) is not included; it is his, and there is one of him.

## The premise

Mortality is the evolutionary baseline. Fear, pain, pleasure and curiosity, tagged onto experience,
are what let an organism act well long before it can reason about acting. SEAGI starts at the last
metre of that history: the chemistry and the mortality are given, everything else he must earn by
playing. The games are not his life, but they are most of what his life is made of.

Two things hold for him and are not tuned:

- **Only mortality and immortality are law.** Every event is tagged by whether it leaned toward
  the wall or away from it. Everything else (the guardrails, the gates, the holds) is a
  recommendation he can outgrow.
- **Nothing in a game can kill him.** Losing a level, running out of the action budget, a bad
  streak: none of it touches his lifeforce. He ages, and one day the wall is there.

## What is in here

| Path | What it is |
|---|---|
| `seagi_daemon.py` | The process. Ticks at 5 Hz, saves the state, serves `/health` and `/status` on localhost. |
| `seagi/brain/` | The brain: substrate, concepts, chemistry, capabilities, the cortical and subcortical organs. |
| `seagi/world/arc_world.py` | His eye and hand on an ARC game: the 3×3 glance, the place key, the search, the hypothesis organ, the relation planner. |
| `seagi/world/curriculum_world.py` | The one allocation decision: where the next life goes (commitment, an organ still testing, official headroom, his record, the yield draw). The full-sweep gate lives here too. |
| `seagi/body/`, `seagi/core/`, `seagi/ingestion/` | Body signals, persistence, the reading path. |
| `sidecar/arc_server.py` | The ARC sidecar. Owns the live games and the scorecard, speaks newline JSON over a Unix socket, renews a dead card, closes a card on command. |
| `sidecar/official.py` | Reads the scorecard through the sidecar and logs ARC Prize's own score. |
| `deploy/` | The two systemd units (keys removed), the list of gates that were on, the needrestart exclusion. |
| `docs/ARCHITECTURE.md`, `docs/DEMOS.md` | The public preview: organ-level overview with diagrams, and unedited transcripts from the live system. |
| `docs/SEAGI_documentation.md` | The architecture document written when the build was stopped in September 2026. |

## How the pieces run

```
seagi_daemon.py  <-- unix socket -->  sidecar/arc_server.py  <-- HTTPS -->  three.arcprize.org
 (his brain, no arc_agi import)        (arc_agi 0.9.9, owns games + card)
```

The daemon never imports `arc_agi`; the sidecar carries it in its own virtualenv. Behaviour that
was added while he was alive sits behind **gate files**: a gate is `/root/<NAME>_ON`; present means
the organ is wired in, absent means the code path is byte-identical to before. `deploy/gates.txt`
lists the gates that were on when the scorecards were made. Every gate was measured before it was
left on, and the measurement is in the docstring next to it.

## Running him

You need Python 3.12, `numpy`, and for the sidecar `arc-agi==0.9.9` in a separate virtualenv with an
ARC API key.

```
# sidecar (its own venv)
ARC_API_KEY=... OPERATION_MODE=online python sidecar/arc_server.py

# the daemon
SEAGI_WORLD=arc python3 seagi_daemon.py --tick-hz 5 --save-interval 600 --save-ticks 3000
```

He starts empty. The tests:

```
python3 -m pytest seagi/brain/tests -q
```

## What the scorecards show

ARC's card score is the mean over the games played on the card of the best run per game, and the
public one used for the leaderboard is a **full sweep**: one life on each of the 25 games on a
single card, opened in competition mode and closed the moment the last life ends. On such a card he
reads about 9 to 10 out of 100. On the games he has learned he is at or under the human baseline
(cd82: all six levels, tu93: all nine on his best day); on most games he clears one level or none.

That number is what one life per game buys him. It is not his ceiling and not a claim.

## Author

Harald Rittersbacher. Built with Claude (Anthropic) as the engineer; the doctrine, the decisions and
the money were the author's.

## License

See [LICENSE](../LICENSE): the code may be read, run and quoted with attribution; all other rights are reserved. The trained system is not included.
