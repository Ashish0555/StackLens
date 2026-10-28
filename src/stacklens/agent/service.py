"""Application service: what the interfaces call, whatever the interface is.

Keeping this layer free of terminal concerns is what makes an HTTP endpoint,
a scheduled job or a bot a thin wrapper instead of a rewrite.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from stacklens.config import get_settings
from stacklens.guardrails import check_answer, check_question
from stacklens.observability import build_run_config, estimate_cost_usd
from stacklens.types import AnswerResult, RunMetrics, ToolCall

logger = logging.getLogger(__name__)


def build_agent() -> Any:
    from stacklens.agent.workflow import build_workflow

    return build_workflow()


def ask(question: str) -> AnswerResult:
    """Ask a one-off question and return the answer with its reasoning trail."""
    return ChatSession().send(question)


class ChatSession:
    """A conversation that remembers what was already said.

    The model itself is stateless: this accumulated message list *is* the
    memory, resent in full on every turn.
    """

    def __init__(self) -> None:
        self._agent = build_agent()
        self._messages: list[BaseMessage] = []
        # Groups every turn of this conversation under one session in tracing.
        self._session_id = uuid4().hex

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    @property
    def session_id(self) -> str:
        return self._session_id

    def restore(self, messages: list[BaseMessage], session_id: str | None = None) -> None:
        """Reload a conversation that was persisted elsewhere.

        Only the messages travel. The graph is rebuilt from configuration on
        every request, so it never needs to be stored, and a session written
        by an older deployment stays readable by a newer one.
        """
        self._messages = list(messages)
        if session_id:
            self._session_id = session_id

    def send(self, question: str) -> AnswerResult:
        """Send a question, updating the conversation history in place.

        The guardrails run here rather than in the CLI or the API, so every
        interface is covered by construction and a new one cannot forget.
        """
        # Before the history grows: a refused question must leave the
        # conversation exactly as it was, or the next turn would carry a
        # message the model never answered.
        check_question(question)

        self._messages = _bound_history(self._messages)
        self._messages.append(HumanMessage(question))
        # Everything after this mark belongs to the current turn. Measuring
        # from here is what keeps earlier turns from being counted twice.
        # Rebuilt every turn, which is what gives the search budget a fresh
        # count. Construction is local object assembly with no network in it,
        # so paying for it once per question is cheaper than the plumbing that
        # would carry a reset into the tool.
        self._agent = build_agent()

        started = time.perf_counter()
        try:
            state = self._agent.invoke(
                {"messages": self._messages},
                config=build_run_config(session_id=self._session_id),
            )
        except Exception:
            self._messages.pop()  # Roll back the unanswered user turn.
            raise
        elapsed = time.perf_counter() - started

        self._messages = _bound_history(state["messages"])
        last_user = max(
            (i for i, message in enumerate(self._messages) if isinstance(message, HumanMessage)),
            default=-1,
        )
        produced = self._messages[last_user + 1 :]

        result = _to_result(produced, elapsed)
        result.findings.extend(
            check_answer(
                result.answer,
                total_tokens=result.metrics.total_tokens if result.metrics else 0,
                retrieved=result.used_tools,
            )
        )
        logger.info(
            "Question length=%d used %d tool(s) in %.2fs, %d token(s)",
            len(question),
            len(result.tool_calls),
            elapsed,
            result.metrics.total_tokens if result.metrics else 0,
        )
        return result


def _to_result(
    produced: list[BaseMessage],
    elapsed: float,
) -> AnswerResult:
    """Turn the raw message list into the typed result callers expect."""
    tool_calls = _extract_tool_calls(produced)

    return AnswerResult(
        answer=str(produced[-1].content) if produced else "",
        tool_calls=tool_calls,
        messages=list(produced),
        metrics=_collect_metrics(produced, tool_calls, elapsed),
    )


def _bound_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep recent complete turns within configured count and character budgets."""
    settings = get_settings()
    human_positions = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    start = (
        human_positions[max(0, len(human_positions) - settings.max_conversation_turns)]
        if human_positions
        else 0
    )
    recent = messages[start:]
    selected: list[BaseMessage] = []
    chars = 0
    for message in reversed(recent):
        size = len(str(message.content))
        if selected and chars + size > settings.max_conversation_chars:
            break
        selected.append(message)
        chars += size
    return list(reversed(selected))


def _extract_tool_calls(messages: list[BaseMessage]) -> list[ToolCall]:
    return [
        ToolCall(name=call["name"], arguments=call["args"])
        for message in messages
        if isinstance(message, AIMessage) and message.tool_calls
        for call in message.tool_calls
    ]


def _collect_metrics(
    produced: list[BaseMessage],
    tool_calls: list[ToolCall],
    elapsed: float,
) -> RunMetrics:
    """Sum the provider's own usage reporting across this turn's model calls."""
    input_tokens = 0
    output_tokens = 0

    for message in produced:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)

    model = get_settings().chat_model

    return RunMetrics(
        latency_seconds=elapsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=len(tool_calls),
        model=model,
        estimated_cost_usd=estimate_cost_usd(model, input_tokens, output_tokens),
    )
