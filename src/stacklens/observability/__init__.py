"""Knowing what the agent did, what it cost, and writing it down.

Three concerns that share one question: after a run, what happened? Tracing
sends the timeline to Langfuse, pricing turns token counts into money, and
logging puts the same story on the console and in a file.

The Langfuse connection lives in `tracing` and serves three purposes at once,
which is why prompt fetching sits there too: the platform holds the prompt
registry on the same client that carries the traces, and opening a second
connection to keep the naming tidy would cost more than it explains.
"""

from stacklens.observability.logging_setup import setup_logging
from stacklens.observability.pricing import MODEL_PRICING_USD_PER_MILLION, estimate_cost_usd
from stacklens.observability.tracing import (
    RECURSION_LIMIT,
    RUN_NAME,
    build_callbacks,
    build_run_config,
    client_or_none,
    fetch_prompt,
    flush,
    langsmith_enabled,
    publish_prompt,
    record_score,
    reset,
)

__all__ = [
    "MODEL_PRICING_USD_PER_MILLION",
    "RECURSION_LIMIT",
    "RUN_NAME",
    "build_callbacks",
    "build_run_config",
    "client_or_none",
    "estimate_cost_usd",
    "fetch_prompt",
    "flush",
    "langsmith_enabled",
    "publish_prompt",
    "record_score",
    "reset",
    "setup_logging",
]
