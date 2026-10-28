"""The Postgres connection: how it is described, and how it fails.

The schema statements are exercised by the integration suite, which needs a
real database. What is worth testing without one is the part a reader depends
on when something is wrong: the message, and what it does not leak.
"""

from __future__ import annotations

import pytest

from stacklens.config import get_settings
from stacklens.indexing import database


@pytest.fixture(autouse=True)
def fresh_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """No pool survives between tests, since each one repoints the URL.

    The connect timeout is dropped to a second. Five is the right production
    default and it is five seconds of waiting per test for a connection that
    is never going to happen; what these tests check is the message, not how
    patient the driver is.
    """
    monkeypatch.setenv("DATABASE_CONNECT_TIMEOUT", "1")
    database.forget_engine()


class TestDescription:
    def test_it_hides_the_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """This string reaches logs, the status command and the API response."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://stacklens:s3cr3t@db:5432/rag")
        get_settings.cache_clear()

        assert "s3cr3t" not in database.describe_database()

    def test_it_names_the_host_and_the_database(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "DATABASE_URL", "postgresql+psycopg://stacklens:pw@db.internal:5433/corpus"
        )
        get_settings.cache_clear()

        described = database.describe_database()

        assert "db.internal" in described
        assert "5433" in described
        assert "corpus" in described


class TestConnection:
    def test_an_unreachable_database_raises_a_named_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://stacklens:pw@127.0.0.1:1/rag")
        get_settings.cache_clear()

        with pytest.raises(database.DatabaseUnavailableError):
            database.verify_connection()

    def test_the_message_names_the_command_that_fixes_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The driver's own error quotes a port and names no remedy."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://stacklens:pw@127.0.0.1:1/rag")
        get_settings.cache_clear()

        with pytest.raises(database.DatabaseUnavailableError) as raised:
            database.verify_connection()

        message = str(raised.value)

        assert "docker compose" in message
        assert "DATABASE_URL" in message

    def test_the_message_does_not_leak_the_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://stacklens:s3cr3t@127.0.0.1:1/rag")
        get_settings.cache_clear()

        with pytest.raises(database.DatabaseUnavailableError) as raised:
            database.verify_connection()

        assert "s3cr3t" not in str(raised.value)


class TestPool:
    def test_the_engine_is_opened_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pool per call would defeat the point of pooling."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://stacklens:pw@db:5432/rag")
        get_settings.cache_clear()

        assert database.get_engine() is database.get_engine()

    def test_forgetting_it_opens_a_new_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://stacklens:pw@db:5432/rag")
        get_settings.cache_clear()

        first = database.get_engine()
        database.forget_engine()

        assert database.get_engine() is not first

    def test_the_pool_size_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A replica count and a pool size multiply into the connection limit."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://stacklens:pw@db:5432/rag")
        monkeypatch.setenv("DATABASE_POOL_SIZE", "20")
        get_settings.cache_clear()

        assert database.get_engine().pool.size() == 20
