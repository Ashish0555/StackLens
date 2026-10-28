"""Domain types shared across layers.

These exist so results travel as typed objects instead of loose dictionaries
that every caller has to unpack and cast by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One retrieved chunk and how far it sits from the question."""

    document: Document
    distance: float

    @property
    def source(self) -> str:
        return str(self.document.metadata.get("source", "desconhecida"))

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def article(self) -> str | None:
        """The article this passage came from, when the chunking found one."""
        article = self.document.metadata.get("article")
        return str(article) if article else None

    @property
    def citation(self) -> str:
        """How the passage should be cited: file, plus article when known."""
        return f"{self.source}, {self.article}" if self.article else self.source


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool the agent decided to invoke, with the arguments it chose."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """What one run of the agent consumed.

    Collected locally from the provider's own usage reporting, so latency and
    cost are visible without signing up for anything.
    """

    latency_seconds: float
    input_tokens: int
    output_tokens: int
    tool_calls: int
    model: str
    estimated_cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class AnswerResult:
    """The agent's final answer plus the reasoning trail that produced it."""

    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)
    metrics: RunMetrics | None = None
    # What the output guardrails noticed. Findings rather than failures: the
    # answer already exists by the time these are known, and a correct refusal
    # legitimately cites nothing.
    findings: list[Any] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        return [call.name for call in self.tool_calls]

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)
