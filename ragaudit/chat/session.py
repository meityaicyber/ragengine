"""Conversation state.

The brief describes a chatbot, which means follow-up questions. The previous
pipeline had no session concept at all: every call was standalone, so "what
about the second one?" could not work (D06).

Sessions are held in memory behind a narrow interface. A multi-process
deployment swaps in a shared store without touching callers.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock

from ragaudit.domain.models import Answer


@dataclass
class Turn:
    question: str
    answer: Answer
    asked_at: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    report_scope: list[str] | None = None
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)

    def history_pairs(self, max_turns: int) -> list[tuple[str, str]]:
        """Recent history as (role, text) pairs for prompting."""
        out: list[tuple[str, str]] = []
        for turn in self.turns[-max_turns:]:
            out.append(("User", turn.question))
            # Truncate prior answers: history is for resolving references, not
            # for re-supplying evidence, and full answers crowd the window.
            text = turn.answer.text
            out.append(("Assistant", text if len(text) <= 400 else text[:400] + "..."))
        return out

    def last_finding_ids(self) -> list[str]:
        """Finding ids cited in the most recent answer.

        Used to resolve ordinal references such as "the second one" without
        another model call.
        """
        if not self.turns:
            return []
        seen: list[str] = []
        for citation in self.turns[-1].answer.citations:
            if citation.finding_id and citation.finding_id not in seen:
                seen.append(citation.finding_id)
        return seen


class SessionStore:
    """Bounded, TTL-expiring session store."""

    def __init__(self, ttl_s: int = 3600, max_sessions: int = 10_000) -> None:
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = RLock()

    def create(self, report_scope: list[str] | None = None) -> Session:
        with self._lock:
            self._evict()
            session = Session(session_id=uuid.uuid4().hex, report_scope=report_scope)
            self._sessions[session.session_id] = session
            return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if time.time() - session.last_seen_at > self.ttl_s:
                del self._sessions[session_id]
                return None
            session.last_seen_at = time.time()
            self._sessions.move_to_end(session_id)
            return session

    def get_or_create(self, session_id: str | None, report_scope: list[str] | None = None) -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                return existing
        return self.create(report_scope)

    def append(self, session: Session, turn: Turn) -> None:
        with self._lock:
            session.turns.append(turn)
            session.last_seen_at = time.time()

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict(self) -> None:
        now = time.time()
        stale = [sid for sid, s in self._sessions.items() if now - s.last_seen_at > self.ttl_s]
        for sid in stale:
            del self._sessions[sid]
        while len(self._sessions) >= self.max_sessions:
            self._sessions.popitem(last=False)

    def __len__(self) -> int:
        return len(self._sessions)
