"""Settings contract: defaults, environment reading and boot-time validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stacklens.config import Settings


def test_defaults_are_production_ready() -> None:
    settings = Settings()

    assert settings.chat_model == "gpt-4o-mini"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.temperature == 0.0
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.retrieval_k == 8


def test_reads_values_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_MODEL", "gpt-4o")
    monkeypatch.setenv("CHUNK_SIZE", "500")

    settings = Settings()

    assert settings.chat_model == "gpt-4o"
    assert settings.chunk_size == 500


def test_api_key_never_leaks_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secreto")

    settings = Settings()

    assert "sk-super-secreto" not in repr(settings)
    assert settings.openai_api_key.get_secret_value() == "sk-super-secreto"


def test_github_token_never_leaks_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-super-secreto")

    settings = Settings()

    assert "ghp-super-secreto" not in repr(settings)
    assert settings.github_token.get_secret_value() == "ghp-super-secreto"


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("TEMPERATURE", "3.0"),
        ("TEMPERATURE", "-1"),
        ("CHUNK_SIZE", "0"),
        ("CHUNK_OVERLAP", "-5"),
        ("RETRIEVAL_K", "0"),
    ],
)
def test_invalid_configuration_fails_at_boot(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings()


def test_unknown_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOMETHING_UNRELATED", "x")

    assert Settings().chat_model == "gpt-4o-mini"
