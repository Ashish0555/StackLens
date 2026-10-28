"""Shared fixtures for the whole test suite."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Keep the developer's own environment from leaking into the tests.

    Three separate leaks are closed here:

    * the environment variables themselves;
    * the .env file, which pydantic-settings reads from the working directory
      regardless of what the environment says -- so the tests run somewhere
      that has none;
    * the settings singleton cache, which would otherwise let the first test
      to call get_settings() freeze its values for every test after it.
    """
    from stacklens.config import get_settings

    monkeypatch.chdir(tmp_path)

    for variable in (
        "APP_DB_PATH",
        "UPLOAD_DIR",
        "INGESTION_MODE",
        "COOKIE_SECURE",
        "REGISTRATION_ENABLED",
        "VALIDATION_RETRIES",
        "EVIDENCE_JUDGE_ENABLED",
        "OPENAI_API_KEY",
        "CHAT_MODEL",
        "EMBEDDING_MODEL",
        "TEMPERATURE",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "CHUNK_STRATEGY",
        "ARTICLE_MAX_CHARS",
        "RETRIEVAL_K",
        "SEARCH_STRATEGY",
        "RERANK_STRATEGY",
        "RERANK_MODEL",
        "RERANK_CANDIDATES",
        "MAX_SEARCHES_PER_TURN",
        "GUARDRAILS_ENABLED",
        "GUARDRAIL_SCANNER",
        "MAX_QUESTION_CHARS",
        "MAX_ANSWER_TOKENS",
        "INJECTION_MODEL",
        "SCAN_CORPUS_FOR_INJECTION",
        "DATA_DIR",
        "COLLECTION_NAME",
        "DATABASE_URL",
        "DATABASE_POOL_SIZE",
        "DATABASE_CONNECT_TIMEOUT",
        "DATABASE_MAX_OVERFLOW",
        "EMBEDDING_DIMENSIONS",
        "KNOWLEDGE_DOMAIN",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "PROMPT_LABEL",
        "PROMPT_CACHE_SECONDS",
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "SESSION_BACKEND",
        "REDIS_URL",
        "SESSION_TTL_SECONDS",
        "API_KEY",
        "RATE_LIMIT",
        "RATE_LIMIT_STORAGE",
        "API_WORKERS",
        "MAX_RETRIES",
        "REQUEST_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(variable, raising=False)

    get_settings.cache_clear()
    _forget_caches()
    yield
    get_settings.cache_clear()
    _forget_caches()


def _forget_caches() -> None:
    """Drop what is built from the settings and is cheap to build again.

    **The model caches are deliberately not dropped here.** Clearing them
    between tests means every test that reaches a guardrail reloads weights,
    and the suite went from two minutes to nine when this file did that. A
    test that changes `INJECTION_MODEL` or `GUARDRAIL_SCANNER` clears the one
    it needs itself; the rest inherit a loaded model, which is what production
    does anyway.

    The reranker stays here because building it is a constructor call with no
    weights in it: the model loads on first use, not on selection.
    """
    from stacklens.indexing.reranker import forget_reranker

    forget_reranker()


@pytest.fixture
def knowledge_base(tmp_path: Path) -> Path:
    """A minimal knowledge base on disk, one file per subject."""
    root = tmp_path / "data"
    root.mkdir()

    (root / "pricing.md").write_text(
        "# Planos\n\nO plano Growth custa R$ 890 por mes.\n",
        encoding="utf-8",
    )
    (root / "limits.txt").write_text(
        "Tamanho maximo de um evento de log: 256 KB.\n",
        encoding="utf-8",
    )
    (root / "ignored.png").write_bytes(b"\x89PNG\r\n")
    (root / "empty.md").write_text("   \n", encoding="utf-8")

    return root


def postgres_is_up() -> bool:
    """Whether the test database is reachable.

    The store no longer has an embedded mode, so anything that touches it
    needs a running Postgres. Tests that do are skipped rather than failed
    when there is none, which keeps the unit suite runnable with no
    infrastructure at all.
    """
    from sqlalchemy import create_engine, text

    from stacklens.config import Settings

    try:
        engine = create_engine(Settings().database_url, pool_pre_ping=True)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
    except Exception:
        return False
    return True


requires_postgres = pytest.mark.skipif(
    not postgres_is_up(),
    reason="needs a running Postgres (docker compose up -d postgres)",
)


@pytest.fixture
def temporary_index(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Point the vector store at a throwaway collection.

    A collection per test rather than a database per test: the rows are
    scoped by collection anyway, and creating one costs a single insert
    against a database that is already up.
    """
    from stacklens.config import get_settings
    from stacklens.indexing import forget_engine

    name = f"test_{uuid4().hex}"
    monkeypatch.setenv("COLLECTION_NAME", name)
    get_settings.cache_clear()
    forget_engine()

    yield name

    from stacklens.indexing import reset_index

    with suppress(Exception):
        reset_index()
    forget_engine()
