"""Optional tracing: off by default, and never able to break an answer."""

from __future__ import annotations

from typing import Any

import pytest

from stacklens.config import get_settings
from stacklens.observability import tracing as observability


class FakeLangfuse:
    """Stands in for the Langfuse client without reaching the network."""

    def __init__(self, *, authenticates: bool = True, flushes: bool = True, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.authenticates = authenticates
        self.flushes = flushes
        self.flush_count = 0

    def auth_check(self) -> bool:
        return self.authenticates

    def flush(self) -> None:
        self.flush_count += 1
        if not self.flushes:
            msg = "langfuse is down"
            raise ConnectionError(msg)


@pytest.fixture(autouse=True)
def _fresh_client() -> None:
    observability.reset()


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.test")
    get_settings.cache_clear()


def install(monkeypatch: pytest.MonkeyPatch, client: FakeLangfuse | None) -> None:
    """Replace the Langfuse class the module imports lazily."""
    import sys
    import types

    def factory(**kwargs: Any) -> FakeLangfuse:
        # Record how the client was configured, since the instance itself is
        # built before the module asks for it.
        client.kwargs.update(kwargs)
        return client

    chosen = factory if client else _explode
    module = types.ModuleType("langfuse")
    module.Langfuse = chosen  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)


def _explode(**_: Any) -> Any:
    msg = "cannot reach langfuse"
    raise RuntimeError(msg)


class TestDisabledByDefault:
    def test_no_keys_means_no_tracing(self) -> None:
        assert get_settings().tracing_enabled is False

    def test_one_key_alone_is_not_enough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        get_settings.cache_clear()

        assert get_settings().tracing_enabled is False

    def test_no_callbacks_when_off(self) -> None:
        assert observability.build_callbacks() == []

    def test_flush_is_a_no_op_when_off(self) -> None:
        observability.flush()

    def test_run_config_still_names_the_run(self) -> None:
        config = observability.build_run_config(session_id="abc")

        assert config["run_name"] == observability.RUN_NAME
        assert config["callbacks"] == []
        assert "metadata" not in config


class TestEnabled:
    def test_keys_turn_it_on(self, keys: None) -> None:
        assert get_settings().tracing_enabled is True

    def test_client_receives_the_configured_host(
        self, keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = FakeLangfuse()
        install(monkeypatch, client)

        observability._client()

        assert client.kwargs["base_url"] == "https://langfuse.test"
        assert client.kwargs["public_key"] == "pk-lf-test"

    def test_run_config_carries_session_and_settings(
        self, keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeLangfuse())

        metadata = observability.build_run_config(session_id="session-1")["metadata"]

        assert metadata["langfuse_session_id"] == "session-1"
        assert metadata["chat_model"] == get_settings().chat_model
        assert metadata["knowledge_domain"] == get_settings().knowledge_domain
        assert "stacklens" in metadata["langfuse_tags"]

    def test_the_client_is_built_once(self, keys: None, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeLangfuse())

        assert observability._client() is observability._client()


class TestNeverBreaksTheAnswer:
    """Observability that can take the application down is worse than none."""

    def test_rejected_keys_disable_tracing(
        self, keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, FakeLangfuse(authenticates=False))

        assert observability._client() is None
        assert observability.build_callbacks() == []

    def test_an_unreachable_langfuse_disables_tracing(
        self, keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, None)

        assert observability._client() is None

    def test_a_failing_flush_is_swallowed(
        self, keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = FakeLangfuse(flushes=False)
        install(monkeypatch, client)

        observability.flush()

        assert client.flush_count == 1

    def test_run_config_survives_a_broken_client(
        self, keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, None)

        config = observability.build_run_config(session_id="abc")

        assert config["callbacks"] == []
        assert config["run_name"] == observability.RUN_NAME
