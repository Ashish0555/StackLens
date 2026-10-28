"""HTTP layer: the agent exposed as a service.

It exists because the orchestration already lives in `agent.service`. Adding
an interface is a wrapper, not a rewrite -- which was the reason for keeping
that layer free of terminal concerns in the first place.
"""

from stacklens.api.feedback import FeedbackStore
from stacklens.api.sessions import InMemorySessionStore, RedisSessionStore, SessionStore

__all__ = [
    "FeedbackStore",
    "InMemorySessionStore",
    "RedisSessionStore",
    "SessionStore",
    "create_app",
]


def create_app():
    from stacklens.api.app import create_app as factory

    return factory()
