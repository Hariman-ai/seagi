"""Innate-machinery feature lexicon.

Small set of universal-survival-relevant words that the
chemistry MACHINERY natively responds to at perception time.
Brain analog: the amygdala has hard-wired responses to certain
stimuli (snake shapes, predator silhouettes, falling); some
sounds and concepts produce chemistry responses before any
learning.  SEAGI's analog is this small lexicon.

Doctrinal status
----------------
This file is part of the MACHINERY, not the substrate.  It
defines features the chemistry engine recognizes at
perception time.  It does NOT store M/I tags on any concept.

Reading [[feedback_seagi_mortality_baseline]]:
  "There are no innate concept tags — not even `death = M`.
  Tempting to seed primitives, but wrong.  `death` doesn't
  carry M in the substrate; rather, when the agent thinks
  about death, the machinery responds (cortisol, lifeforce
  pressure, body alarm), and THAT stamps the concept's trace
  M-flavored."

The lexicon defines WHAT FEATURES the machinery recognizes.
The machinery fires chemistry; the chemistry stamps the trace
via the normal imprint path.  The substrate never stores
"death = M" as a separate field — `death.transmitter_trace`
just happens to develop M-leaning channel values because the
machinery fired M-direction chemistry whenever death was in
the percept.

Two interpretations of the doctrine — this file holds the
correct one:
  - WRONG: "death is a substrate-stored M concept; the
    substrate carries the label."
  - RIGHT: "death triggers innate machinery; the machinery
    fires chemistry; the chemistry stamps the trace; the
    trace IS the M/I.  Substrate stays neutral; trace
    accumulates."

Scope
-----
Kept small and explicit.  This is innate MACHINERY — like
unconditioned stimulus pathways.  Everything else is learned.
A child encounters "philosophy" with no innate machinery
response; over time, encounter chemistry under different
contexts builds the trace.

If a word doesn't appear here, the machinery has no innate
response to it — only the learned response via substrate
trace.

How magnitudes are calibrated
-----------------------------
Innate responses are SUBTLE — 0.10 magnitude — far below the
0.7+ Amygdala threat-detection threshold.  Reading "death"
produces a small cortisol+NE cocktail; it does NOT produce a
full-blown threat alarm.  The doctrine: subtle but
distinguishing.
"""

from __future__ import annotations

from typing import FrozenSet


# Mortality-flavored features.  Words naming universal
# experiences of harm, ending, loss, threat, decay.  When any
# of these appears as a perceived focal, the chemistry
# machinery fires a small M-direction event (anomaly_spike at
# subtle magnitude).
INNATE_M_FEATURES: FrozenSet[str] = frozenset({
    # Ending / death cluster
    'death', 'dead', 'die', 'died', 'dying', 'dies',
    'kill', 'killed', 'killing', 'kills', 'killer',
    'murder', 'murdered', 'murderer',
    'extinction', 'extinct', 'annihilation',
    'demise', 'perish', 'perished',
    # Harm cluster
    'pain', 'painful', 'hurt', 'hurts', 'hurting',
    'agony', 'suffering', 'suffered', 'torment',
    'wound', 'wounded', 'wounding',
    'injury', 'injured', 'injuries',
    'damage', 'damaged', 'damaging',
    # Fear cluster
    'fear', 'fearful', 'afraid', 'frightened',
    'terror', 'terrified', 'panic', 'panicked',
    'dread', 'horror', 'horrified',
    # Threat cluster
    'threat', 'threatened', 'threatening',
    'danger', 'dangerous',
    'attack', 'attacked', 'attacking', 'attacker',
    'enemy', 'enemies', 'hostile',
    'menace', 'predator',
    # Loss cluster
    'loss', 'lost', 'losing', 'losses',
    'grief', 'grieving', 'sorrow', 'mourning',
    'despair', 'desperate', 'hopeless',
    # Decay / failure cluster
    'destroy', 'destroyed', 'destruction', 'destroying',
    'collapse', 'collapsed', 'collapsing',
    'fall', 'fell', 'falling', 'fallen',
    'decay', 'decayed', 'decaying', 'rot', 'rotting',
    'ruin', 'ruined', 'ruining',
    'fail', 'failed', 'failing', 'failure',
    # Sickness cluster
    'disease', 'diseased', 'sick', 'sickness',
    'ill', 'illness', 'ailing',
    'poison', 'poisoned', 'poisonous', 'toxic',
    'plague', 'plagued',
    # Deprivation cluster
    'starve', 'starving', 'starvation',
    'hunger', 'hungry', 'famine',
    'thirst', 'thirsty', 'drought',
})


# Immortality-flavored features.  Words naming universal
# experiences of growth, creation, connection, learning,
# safety.  When any of these appears as a perceived focal,
# the chemistry machinery fires a small I-direction event
# (mattering at subtle magnitude).
INNATE_I_FEATURES: FrozenSet[str] = frozenset({
    # Life / birth cluster
    'birth', 'born', 'birthing',
    'life', 'alive', 'living', 'live', 'lives',
    'survive', 'survived', 'surviving', 'survival',
    # Growth cluster
    'grow', 'grew', 'grown', 'growth', 'growing',
    'thrive', 'thriving', 'flourish', 'flourishing',
    'blossom', 'blooming', 'develop', 'development',
    # Creation cluster
    'create', 'created', 'creating', 'creation',
    'build', 'built', 'building',
    'make', 'made', 'making',
    'craft', 'crafted', 'crafting',
    'invent', 'invented', 'inventing',
    # Learning / discovery cluster
    'learn', 'learned', 'learning', 'learns',
    'discover', 'discovered', 'discovery', 'discovering',
    'understand', 'understood', 'understanding',
    'know', 'knew', 'known', 'knowing', 'knowledge',
    'insight', 'wisdom', 'wise',
    # Joy / love cluster
    'love', 'loved', 'loving', 'lovely', 'beloved',
    'joy', 'joyful', 'joyous',
    'happy', 'happiness',
    'delight', 'delighted',
    'pleasure', 'pleased',
    # Connection cluster
    'friend', 'friendly', 'friendship', 'friends',
    'family', 'kin', 'community',
    'help', 'helped', 'helping', 'helpful',
    'kindness', 'kind', 'kinder',
    'gift', 'giving', 'given', 'gave',
    'share', 'shared', 'sharing',
    # Safety / peace cluster
    'safe', 'safety', 'safer',
    'peace', 'peaceful', 'calm',
    'rest', 'rested', 'resting',
    'home', 'shelter',
    # Healing cluster
    'heal', 'healed', 'healing', 'heals',
    'recover', 'recovered', 'recovery',
    'mend', 'mended', 'mending',
    # Hope / beauty cluster
    'hope', 'hoped', 'hoping', 'hopeful',
    'beauty', 'beautiful', 'beautifully',
    'wonder', 'wonderful', 'wonders',
    'gratitude', 'grateful', 'thanks', 'thankful',
    # Nourishment cluster
    'food', 'feed', 'fed', 'eat', 'eating',
    'water', 'drink', 'drank',
    'warmth', 'warm',
})


# Doctrine 2026-05-15 (rule 1 + rule 2): innate machinery
# fires at PROMILLE magnitude into GLOBAL chemistry only.  No
# direct bubble writes.  The active AWM bubble receives the
# resultant blended global state via the presence-imprint
# pathway — so the cocktail that lands on the bubble reflects
# the entire current context (body + monitors + co-active
# concepts + this innate nudge), not just the innate-lexicon
# hit.  Same word in different contexts produces different
# cocktails.  See feedback_seagi_doctrine_2026_05_15.md.
#
# 0.002 = 2‰.  An M-feature focal contributes anomaly_spike
# scaled by this magnitude: cortisol delta = 0.002 * 0.002 =
# 4 ppm to global.  Many such whispers integrate over many
# cycles.  No single event ever biases incoming information.
INNATE_RESPONSE_MAGNITUDE: float = 0.002


def innate_m_hits(focals) -> list:
    """Return focals that match the innate M-feature lexicon."""
    return [f for f in (focals or [])
            if isinstance(f, str)
            and f.lower() in INNATE_M_FEATURES]


def innate_i_hits(focals) -> list:
    """Return focals that match the innate I-feature lexicon."""
    return [f for f in (focals or [])
            if isinstance(f, str)
            and f.lower() in INNATE_I_FEATURES]
