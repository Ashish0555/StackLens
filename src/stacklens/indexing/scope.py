"""Collection scope is carried per request; it never mutates cached settings."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from stacklens.config import get_settings

_collection: ContextVar[str | None] = ContextVar("stacklens_collection", default=None)


def current_collection() -> str:
    return _collection.get() or get_settings().collection_name


@contextmanager
def collection_scope(name: str) -> Iterator[None]:
    token = _collection.set(name)
    try:
        yield
    finally:
        _collection.reset(token)


def user_collection(user_id: str) -> str:
    return f"stacklens_user_{user_id}"
