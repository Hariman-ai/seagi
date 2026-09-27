"""Conversation memory — bounded ring buffer of recent turns.

Phase F.3 (2026-05-16).  Doctrine: dialog is multi-turn.  The
v1 substrate had a conversation.py ring buffer; the v2
consolidation dropped it.  Brain.chat() became stateless per
turn — every reply composed in isolation, no awareness of what
was just said.

This module restores that:

  - `ConversationTurn` carries one turn: role, peer_id,
     utterance, intent kind + focal/target, cycle, and a
     compact chemistry snapshot at the moment of the turn.
  - `Conversation` is a ring buffer (default 16 turns) with
     lookup by peer_id, recent-N queries, and topic-mentioned
     helpers.

Used by:
  - `Brain.chat()` records both the peer's input AND the
     agent's response on every turn.
  - The introspective intent handler reads recent turns for
     "we just talked about X" framing.
  - Future: reference resolution ("that one" / "it" mapped
     back to the most-recent salient concept), repetition
     detection (avoid saying the same thing twice in a row),
     dmDMN peer-model updates.

Bounded by default — chat conversations are typically short
in any single session.  The full history is not durable; the
substrate is the long-term record via episodes consolidated
by Hippocampus.  Conversation memory is working-memory for
dialog flow only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


# Default ring size.  16 turns covers ~8 back-and-forth exchanges
# which is enough for the immediate dialog context.  Beyond that,
# the substrate's episodic memory + hippocampal consolidation
# carries the longer arc.
DEFAULT_BUFFER_SIZE = 16


@dataclass
class ConversationTurn:
    """One turn in the dialog."""
    role: str                # 'peer' or 'agent'
    peer_id: str
    utterance: str
    intent_kind: str = ''    # from Intent.* (e.g. 'question_factual')
    focal: str = ''          # intent's primary subject
    target: str = ''         # intent's secondary subject
    cycle: int = 0
    # Compact chemistry snapshot — store a 2-tuple (m_polarity,
    # i_polarity) for the dialog-state diagnostics that don't
    # need the full 8-channel state.  Full state lives in
    # chemistry.global_history if needed.
    m_polarity: float = 0.0
    i_polarity: float = 0.0
    tone_label: str = ''     # 'flat', 'warm', 'somber', etc.


class Conversation:
    """Bounded ring of recent turns across all peers.  Indexed
    by peer for fast per-peer lookup."""

    def __init__(self, buffer_size: int = DEFAULT_BUFFER_SIZE):
        self.buffer_size = int(buffer_size)
        self._turns: Deque[ConversationTurn] = deque(
            maxlen=self.buffer_size)

    def __len__(self) -> int:
        return len(self._turns)

    def is_empty(self) -> bool:
        return not self._turns

    def record_peer(self,
                       peer_id: str,
                       utterance: str,
                       *,
                       intent_kind: str = '',
                       focal: str = '',
                       target: str = '',
                       cycle: int = 0,
                       m_polarity: float = 0.0,
                       i_polarity: float = 0.0,
                       tone_label: str = '') -> ConversationTurn:
        """Record a peer's utterance."""
        t = ConversationTurn(
            role='peer',
            peer_id=peer_id,
            utterance=utterance,
            intent_kind=intent_kind,
            focal=focal,
            target=target,
            cycle=cycle,
            m_polarity=m_polarity,
            i_polarity=i_polarity,
            tone_label=tone_label,
        )
        self._turns.append(t)
        return t

    def record_agent(self,
                        peer_id: str,
                        utterance: str,
                        *,
                        intent_kind: str = '',
                        focal: str = '',
                        target: str = '',
                        cycle: int = 0,
                        m_polarity: float = 0.0,
                        i_polarity: float = 0.0,
                        tone_label: str = '') -> ConversationTurn:
        """Record the agent's response."""
        t = ConversationTurn(
            role='agent',
            peer_id=peer_id,
            utterance=utterance,
            intent_kind=intent_kind,
            focal=focal,
            target=target,
            cycle=cycle,
            m_polarity=m_polarity,
            i_polarity=i_polarity,
            tone_label=tone_label,
        )
        self._turns.append(t)
        return t

    # ---- queries ----

    def recent(self, n: int = 4) -> List[ConversationTurn]:
        """Last N turns in chronological order (oldest first)."""
        if n <= 0:
            return []
        if n >= len(self._turns):
            return list(self._turns)
        return list(self._turns)[-n:]

    def recent_for_peer(self,
                              peer_id: str,
                              n: int = 4) -> List[ConversationTurn]:
        """Last N turns involving a specific peer."""
        out = [t for t in self._turns if t.peer_id == peer_id]
        if n <= 0:
            return []
        if n >= len(out):
            return out
        return out[-n:]

    def last_focal(self,
                       peer_id: Optional[str] = None) -> str:
        """Most-recent focal seen in conversation (any role).
        If peer_id given, filter to turns involving that peer."""
        for t in reversed(self._turns):
            if peer_id is not None and t.peer_id != peer_id:
                continue
            if t.focal:
                return t.focal
        return ''

    def last_agent_utterance(self,
                                   peer_id: Optional[str] = None
                                   ) -> str:
        """Most-recent thing the agent said (for repetition check)."""
        for t in reversed(self._turns):
            if t.role != 'agent':
                continue
            if peer_id is not None and t.peer_id != peer_id:
                continue
            return t.utterance
        return ''

    def topics_recent(self,
                            n: int = 4,
                            peer_id: Optional[str] = None) -> List[str]:
        """Recent focal concepts (de-duplicated, most-recent first).
        Useful for self-state composition: "we have been talking
        about X, Y, Z"."""
        seen: List[str] = []
        for t in reversed(self._turns):
            if peer_id is not None and t.peer_id != peer_id:
                continue
            if t.focal and t.focal not in seen and t.focal != 'self':
                seen.append(t.focal)
            if len(seen) >= n:
                break
        return seen

    def mentioned(self, concept: str) -> bool:
        """True if `concept` appears as focal or target in any
        recent turn."""
        c = concept.lower()
        for t in self._turns:
            if t.focal.lower() == c or t.target.lower() == c:
                return True
        return False

    def turn_count(self) -> Tuple[int, int]:
        """(peer_turn_count, agent_turn_count)."""
        p = sum(1 for t in self._turns if t.role == 'peer')
        a = sum(1 for t in self._turns if t.role == 'agent')
        return (p, a)
