"""Curiosity — autonomous question generation.

Phase F.12 (2026-05-16).  v1 had `curiosity.py`; v2 lost it.
Doctrine ([[project_seagi_curiosity_2026_05_08]]): the agent
ASKS the user when uncertain.  Hold-verdict, thin-substrate,
or contradiction → "Could you tell me which fits?"  This is
bidirectional dialog, not just one-way Q&A.

What this module is
-------------------
A small templated question generator.  Given a focal + the
reason curiosity fired, returns a first-person question for
MotorSpeech to surface.  Doctrine alignment: the generator
is RULE-BASED, not template-padded — every question is
substrate-justified by an actual gap (thin substrate, open
learn_about goal, contradiction flag).

Triggers
--------
- THIN_SUBSTRATE: cortical produced a metacog/thin thought →
   "Could you tell me more about X?"
- LEARN_ABOUT_GOAL: goal_tracker has an open learn_about goal
   on the current focal → "What does X mean to you?"
- CONTRADICTION: F.6 detected substrate or self-model conflict
   that wasn't decisive → "I hold X and also Y — which fits?"
- HOLD_VERDICT: cortical couldn't pick between two paths →
   same as contradiction but worded as a choice request.

Doctrine alignment
------------------
- Curiosity questions are at most ONE per turn — agent asks
  one focused question, not a barrage.
- Questions emerge from genuine gaps in substrate / identity,
  not as a politeness layer.
- The curiosity-question append never blocks an answer; it
  augments existing response output.
"""

from __future__ import annotations

from typing import Optional


# Question kinds — what triggered the curiosity.
CURIOSITY_THIN = 'thin_substrate'
CURIOSITY_GOAL = 'learn_about_goal'
CURIOSITY_CONTRADICTION = 'contradiction'
CURIOSITY_HOLD = 'hold_verdict'


def generate_question(focal: str,
                          kind: str = CURIOSITY_THIN,
                          *,
                          contradiction_alt: str = '') -> str:
    """Compose a first-person curiosity question.

    Args:
      focal: the concept the agent is uncertain about.
      kind: which trigger fired (see constants above).
      contradiction_alt: the alternative claim for
        CURIOSITY_CONTRADICTION / CURIOSITY_HOLD.

    Returns an empty string if no clean question can be composed
    (e.g. empty focal).  Caller appends the question to its
    response.
    """
    if not focal:
        return ''
    if kind == CURIOSITY_THIN:
        # Shorter wording — the cortical thin-substrate thought
        # already explains why; the question just asks for help.
        return f'Could you tell me more about {focal}?'
    if kind == CURIOSITY_GOAL:
        return (f'What does {focal} mean to you?  I have '
                  f'been holding an open question about it.')
    if kind == CURIOSITY_CONTRADICTION:
        if contradiction_alt:
            return (f'I hold {focal} and also {contradiction_alt}'
                      f' — which fits better?')
        return (f'I hold conflicting views about {focal} — '
                  f'which fits better?')
    if kind == CURIOSITY_HOLD:
        if contradiction_alt:
            return (f'Is it {focal} or {contradiction_alt}?  '
                      f'I cannot decide between them.')
        return (f'Could you help me decide about {focal}?')
    return ''
