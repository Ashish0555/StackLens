"""Reasoning trail rendering: every message kind reaches the reader."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from stacklens.agent import format_trace


def test_shows_the_user_question() -> None:
    trace = format_trace([HumanMessage("quanto custa?")])

    assert "[VOCÊ] quanto custa?" in trace


def test_shows_each_tool_the_agent_decided_to_call() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_documentation", "args": {"question": "preço"}, "id": "1"},
            {"name": "calculate", "args": {"expression": "890*12"}, "id": "2"},
        ],
    )

    trace = format_trace([message])

    assert "search_documentation" in trace
    assert "calculate" in trace


def test_shows_the_final_answer() -> None:
    trace = format_trace([AIMessage(content="R$ 890 por mês.")])

    assert "[AGENTE responde] R$ 890 por mês." in trace


def test_long_tool_output_is_truncated() -> None:
    message = ToolMessage(content="x" * 500, tool_call_id="1", name="search_documentation")

    trace = format_trace([message])

    assert trace.endswith("...")
    assert len(trace) < 200


def test_short_tool_output_is_not_truncated() -> None:
    message = ToolMessage(content="R$ 890", tool_call_id="1", name="search_documentation")

    assert format_trace([message]).endswith("R$ 890")


def test_multiline_tool_output_stays_on_one_line() -> None:
    message = ToolMessage(content="linha 1\nlinha 2", tool_call_id="1", name="search_documentation")

    assert "\n" not in format_trace([message])


def test_empty_conversation_renders_nothing() -> None:
    assert format_trace([]) == ""
