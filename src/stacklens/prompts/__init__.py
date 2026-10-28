"""Where the agent's instructions come from, and how they are rendered.

The text lives in `prompt_templates.py`. This module decides where to read it
from: Langfuse when tracing is configured, the templates otherwise. Publishing
the prompts turns them into versioned assets, so changing how the agent behaves
becomes a label move in a UI rather than a commit, a build and a deploy.

An unreachable Langfuse must never stop the agent from answering, so every
path here falls back to the shipped text.
"""

from __future__ import annotations

import logging

from stacklens.config import get_settings
from stacklens.observability import fetch_prompt
from stacklens.prompts.templates import (
    CALCULATOR_DESCRIPTION_TEMPLATE,
    CALCULATOR_TOOL_PROMPT_NAME,
    PUBLISHED_PROMPTS,
    SEARCH_TOOL_DESCRIPTION_TEMPLATE,
    SEARCH_TOOL_PROMPT_NAME,
    SYSTEM_PROMPT_NAME,
    SYSTEM_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

DOMAIN_PLACEHOLDER = "{{domain}}"

__all__ = [
    "CALCULATOR_TOOL_PROMPT_NAME",
    "PUBLISHED_PROMPTS",
    "SEARCH_TOOL_PROMPT_NAME",
    "SYSTEM_PROMPT_NAME",
    "SYSTEM_PROMPT_TEMPLATE",
    "build_calculator_description",
    "build_search_tool_description",
    "build_system_prompt",
    "describe_source",
]


def build_system_prompt() -> str:
    """Render the system prompt for the configured knowledge domain."""
    return _render(SYSTEM_PROMPT_NAME, SYSTEM_PROMPT_TEMPLATE)


def build_search_tool_description() -> str:
    """Render the search tool's description for the configured domain.

    The model never reads the tool's body, only this text. It is therefore the
    contract that decides whether the tool gets called at all.
    """
    return _render(SEARCH_TOOL_PROMPT_NAME, SEARCH_TOOL_DESCRIPTION_TEMPLATE)


def build_calculator_description() -> str:
    """Render the calculator's description.

    The same kind of contract as the search tool: the model reads it to decide
    whether a question needs arithmetic, so it is tuned like a prompt and
    versioned with the prompts.
    """
    return _render(CALCULATOR_TOOL_PROMPT_NAME, CALCULATOR_DESCRIPTION_TEMPLATE)


def describe_source() -> tuple[str, int | None]:
    """Where the active system prompt came from, and which version.

    Recorded next to an evaluation score, because a run graded against one
    prompt version cannot be compared to a run graded against another.
    """
    _, version = fetch_prompt(
        SYSTEM_PROMPT_NAME,
        label=get_settings().prompt_label,
        fallback=SYSTEM_PROMPT_TEMPLATE,
    )
    return ("langfuse" if version is not None else "local", version)


def _render(name: str, fallback: str) -> str:
    settings = get_settings()
    template, _ = fetch_prompt(name, label=settings.prompt_label, fallback=fallback)

    return template.replace(DOMAIN_PLACEHOLDER, settings.knowledge_domain)
