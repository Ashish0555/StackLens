"""Optional LLM tracing through Langfuse.

The local metrics in `types.RunMetrics` say what a run cost in total. A trace
says where it went: each model call, each tool, each retrieved passage, with
its own latency and token count.

Tracing is off unless both Langfuse keys are configured, and a broken or
unreachable Langfuse never breaks an answer -- observability that can take the
application down with it is worse than none.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from stacklens.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _client() -> Any | None:
    """Create the Langfuse client once, or None when tracing is off."""
    settings = get_settings()

    if not settings.tracing_enabled:
        logger.debug("Tracing disabled: Langfuse keys are not configured.")
        return None

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_host,
        )
        if not client.auth_check():
            logger.warning("Langfuse rejected the configured keys. Tracing disabled.")
            return None
    except Exception:
        logger.warning("Could not start Langfuse. Tracing disabled.", exc_info=True)
        return None

    logger.info("Tracing enabled: %s", settings.langfuse_host)
    return client


RUN_NAME = "rag.ask"

# Without a cap, a question the agent cannot satisfy makes it search again and
# again, each round adding passages to the history until the context window
# overflows and the whole call dies. Ten steps is well past any legitimate
# search-then-answer loop.
RECURSION_LIMIT = 10


def client_or_none() -> Any | None:
    """The Langfuse client, or None when tracing is not configured.

    Exposed because datasets and experiments need the same connection that
    carries the traces, rather than a second one.
    """
    return _client()


def build_callbacks() -> list[Any]:
    """Callbacks to hand to the agent. Empty when tracing is off."""
    if _client() is None:
        return []

    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception:
        logger.warning("Could not build the Langfuse callback.", exc_info=True)
        return []


def build_run_config(*, session_id: str) -> dict[str, Any]:
    """The config for one agent invocation, tracing included when it is on.

    Without a run name every trace lands in the dashboard called "LangGraph",
    and without a session id the turns of one conversation are unrelated rows.
    The metadata records which settings produced the run, so a trace from last
    week still explains itself.
    """
    config: dict[str, Any] = {
        "callbacks": build_callbacks(),
        "run_name": RUN_NAME,
        "recursion_limit": RECURSION_LIMIT,
    }

    if _client() is None:
        return config

    settings = get_settings()
    config["metadata"] = {
        "langfuse_session_id": session_id,
        "langfuse_tags": ["stacklens", settings.search_strategy.value],
        "knowledge_domain": settings.knowledge_domain,
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "retrieval_k": settings.retrieval_k,
    }
    return config


def record_score(*, trace_id: str, name: str, value: bool, comment: str | None = None) -> bool:
    """Attach a score to a trace. Returns whether it was sent.

    This is the platform's own mechanism for human judgement on a run, so the
    verdict lands next to the prompt, the retrieved passages and the cost
    rather than in a file nobody correlates with them.
    """
    client = _client()
    if client is None or not trace_id:
        return False

    try:
        client.create_score(
            name=name,
            value=value,
            trace_id=trace_id,
            data_type="BOOLEAN",
            comment=comment,
        )
    except Exception:
        logger.warning("Could not record the score in Langfuse.", exc_info=True)
        return False

    return True


def fetch_prompt(name: str, *, label: str, fallback: str) -> tuple[str, int | None]:
    """Read a prompt from Langfuse, or fall back to the text shipped in code.

    Returns the template and the version it came from, with None meaning the
    fallback was used. The SDK caches for `prompt_cache_seconds`, so a request
    does not pay a round trip, and it takes the fallback itself when the
    platform cannot be reached: a prompt store that can stop the agent from
    answering is worse than no prompt store.
    """
    client = _client()
    if client is None:
        return fallback, None

    try:
        prompt = client.get_prompt(
            name,
            label=label,
            fallback=fallback,
            cache_ttl_seconds=get_settings().prompt_cache_seconds,
        )
    except Exception:
        logger.warning("Could not fetch the prompt %r from Langfuse.", name, exc_info=True)
        return fallback, None

    # is_fallback tells the two apart: the SDK returns the fallback as a prompt
    # object, so a truthy result is not proof that the platform answered.
    if getattr(prompt, "is_fallback", False):
        return fallback, None

    return str(prompt.prompt), prompt.version


def publish_prompt(name: str, template: str, *, label: str, commit_message: str) -> int | None:
    """Publish a prompt and give it the label the agent reads.

    Returns the version created, or None when tracing is off.
    """
    client = _client()
    if client is None:
        return None

    try:
        prompt = client.create_prompt(
            name=name,
            prompt=template,
            labels=[label],
            type="text",
            commit_message=commit_message,
        )
    except Exception:
        logger.warning("Could not publish the prompt %r.", name, exc_info=True)
        return None

    return int(prompt.version)


def langsmith_enabled() -> bool:
    """Whether LangChain is also exporting traces to LangSmith.

    Nothing here turns it on. LangChain instruments itself when LANGSMITH_
    variables are present, so the setting is reported rather than applied.
    """
    tracing = os.environ.get("LANGSMITH_TRACING", "").lower() in {"true", "1"}
    return tracing and bool(os.environ.get("LANGSMITH_API_KEY"))


def flush() -> None:
    """Send whatever is still buffered.

    The SDK batches in the background. A CLI process is short-lived enough to
    exit before that batch leaves, so the traces of the very run you wanted to
    inspect would be the ones lost.
    """
    client = _client()
    if client is None:
        return

    try:
        client.flush()
    except Exception:
        logger.warning("Could not flush traces to Langfuse.", exc_info=True)


def reset() -> None:
    """Drop the cached client. For tests, and for settings changed at runtime."""
    _client.cache_clear()
