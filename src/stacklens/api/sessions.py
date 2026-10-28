"""Conversation storage for the chat endpoint.

Two backends behind one interface. In-memory is the default and needs nothing
running. Redis is what makes the service horizontally scalable: the session a
client opened against one replica is readable by the next, and a restart does
not forget it.

The store never builds a session itself. The factory is injected, because
constructing one reaches the model provider, and a store that cannot be
exercised without credentials is a store that goes untested.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain_core.messages import messages_from_dict, messages_to_dict

from stacklens.agent import ChatSession

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], ChatSession]

MAX_SESSIONS = 100
KEY_PREFIX = "stacklens:session:"


class SessionStore(ABC):
    """Where conversations live between requests."""

    def __init__(self, factory: SessionFactory | None = None) -> None:
        self._factory: SessionFactory = factory or ChatSession

    @abstractmethod
    def get_or_create(self, session_id: str | None) -> tuple[str, ChatSession]:
        """Return an existing conversation, or open a new one.

        An unknown id opens a new conversation rather than failing: a client
        holding an id from before a restart should keep working.
        """

    @abstractmethod
    def drop(self, session_id: str) -> bool:
        """Forget one conversation. Returns whether it existed."""

    @abstractmethod
    def __len__(self) -> int:
        """How many conversations are held right now."""

    def _new_id(self, session_id: str | None) -> str:
        return session_id or uuid4().hex


class InMemorySessionStore(SessionStore):
    """Keeps the most recent conversations in this process, evicting the oldest.

    The cap exists because every session holds its whole message history: an
    unbounded dictionary is a memory leak with a friendly name.
    """

    def __init__(
        self,
        max_sessions: int = MAX_SESSIONS,
        factory: SessionFactory | None = None,
    ) -> None:
        super().__init__(factory)
        self._sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self._max_sessions = max_sessions

    def __len__(self) -> int:
        return len(self._sessions)

    def get_or_create(self, session_id: str | None) -> tuple[str, ChatSession]:
        if session_id and session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return session_id, self._sessions[session_id]

        new_id = self._new_id(session_id)
        self._sessions[new_id] = self._factory()
        self._evict_oldest()

        logger.info("Opened session %s (%d active)", new_id, len(self._sessions))
        return new_id, self._sessions[new_id]

    def drop(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def _evict_oldest(self) -> None:
        while len(self._sessions) > self._max_sessions:
            evicted, _ = self._sessions.popitem(last=False)
            logger.info("Evicted session %s (cap %d)", evicted, self._max_sessions)


class RedisSessionStore(SessionStore):
    """Keeps conversations in Redis, shared across replicas and restarts.

    Sessions expire on their own after `ttl_seconds`. A conversation nobody
    returns to should not occupy memory forever, and Redis expiring keys is
    the mechanism built for that.
    """

    def __init__(
        self,
        client: Any,
        ttl_seconds: int,
        factory: SessionFactory | None = None,
    ) -> None:
        super().__init__(factory)
        self._client = client
        self._ttl = ttl_seconds

    def __len__(self) -> int:
        return len(self._client.keys(f"{KEY_PREFIX}*"))

    def get_or_create(self, session_id: str | None) -> tuple[str, ChatSession]:
        if session_id:
            stored = self._load(session_id)
            if stored is not None:
                return session_id, stored

        new_id = self._new_id(session_id)
        session = self._factory()
        self.save(new_id, session)

        logger.info("Opened session %s in Redis", new_id)
        return new_id, session

    def save(self, session_id: str, session: ChatSession) -> None:
        """Persist a conversation and refresh its expiry.

        Only the messages are stored. The agent graph holds closures that
        cannot be serialised, and it does not need to be: it is rebuilt from
        configuration on every request. Storing the conversation as JSON also
        means a session written by one deployment stays readable by the next.
        """
        payload = json.dumps({"messages": messages_to_dict(session.messages)})
        self._client.setex(_key(session_id), self._ttl, payload)

    def drop(self, session_id: str) -> bool:
        return bool(self._client.delete(_key(session_id)))

    def _load(self, session_id: str) -> ChatSession | None:
        raw = self._client.get(_key(session_id))
        if raw is None:
            return None

        try:
            stored = json.loads(raw)
            messages = messages_from_dict(stored["messages"])
        except Exception:
            # Unreadable data beats a 500 on every request from then on.
            logger.warning("Discarding unreadable session %s", session_id, exc_info=True)
            self.drop(session_id)
            return None

        session = self._factory()
        session.restore(messages, session_id)
        return session


def _key(session_id: str) -> str:
    return f"{KEY_PREFIX}{session_id}"
