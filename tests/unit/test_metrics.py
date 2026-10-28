"""Run metrics: what each answer cost, measured locally with no external service."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from stacklens.agent import service
from stacklens.agent.service import ChatSession
from stacklens.config import get_settings
from stacklens.observability import MODEL_PRICING_USD_PER_MILLION, estimate_cost_usd


def usage(input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class MeteredAgent:
    """A fake graph that reports token usage the way the real provider does."""

    def __init__(self, per_turn: list[list[BaseMessage]]) -> None:
        self.per_turn = per_turn
        self.turn = 0

    def invoke(self, state: dict[str, Any], **_: Any) -> dict[str, Any]:
        produced = self.per_turn[min(self.turn, len(self.per_turn) - 1)]
        self.turn += 1
        return {"messages": list(state["messages"]) + list(produced)}


def install(monkeypatch: pytest.MonkeyPatch, per_turn: list[list[BaseMessage]]) -> MeteredAgent:
    agent = MeteredAgent(per_turn)
    monkeypatch.setattr(service, "build_agent", lambda: agent)
    return agent


class TestCostEstimation:
    def test_known_model_is_priced(self) -> None:
        cost = estimate_cost_usd("gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)

        assert cost == pytest.approx(0.15)

    def test_input_and_output_are_priced_separately(self) -> None:
        cost = estimate_cost_usd("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)

        assert cost == pytest.approx(0.75)

    def test_unknown_model_yields_no_estimate(self) -> None:
        assert estimate_cost_usd("some-unreleased-model", 1000, 1000) is None

    def test_zero_usage_costs_nothing(self) -> None:
        assert estimate_cost_usd("gpt-4o-mini", 0, 0) == 0.0

    def test_every_listed_price_has_input_and_output(self) -> None:
        assert all(len(price) == 2 for price in MODEL_PRICING_USD_PER_MILLION.values())


class TestCollection:
    def test_sums_usage_across_the_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(
            monkeypatch,
            [
                [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "calculate", "args": {}, "id": "1"}],
                        usage_metadata=usage(100, 20),
                    ),
                    AIMessage(content="pronto", usage_metadata=usage(150, 30)),
                ]
            ],
        )

        metrics = service.ask("pergunta").metrics

        assert metrics is not None
        assert metrics.input_tokens == 250
        assert metrics.output_tokens == 50
        assert metrics.total_tokens == 300

    def test_counts_the_tool_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(
            monkeypatch,
            [
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "search_documentation", "args": {}, "id": "1"},
                            {"name": "calculate", "args": {}, "id": "2"},
                        ],
                        usage_metadata=usage(10, 5),
                    ),
                    AIMessage(content="pronto", usage_metadata=usage(10, 5)),
                ]
            ],
        )

        metrics = service.ask("pergunta").metrics

        assert metrics is not None
        assert metrics.tool_calls == 2

    def test_records_the_configured_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, [[AIMessage(content="ok", usage_metadata=usage(1, 1))]])

        metrics = service.ask("pergunta").metrics

        assert metrics is not None
        assert metrics.model == get_settings().chat_model

    def test_measures_latency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, [[AIMessage(content="ok", usage_metadata=usage(1, 1))]])

        metrics = service.ask("pergunta").metrics

        assert metrics is not None
        assert metrics.latency_seconds >= 0

    def test_survives_a_provider_that_reports_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, [[AIMessage(content="sem usage")]])

        metrics = service.ask("pergunta").metrics

        assert metrics is not None
        assert metrics.total_tokens == 0


class TestConversationAccounting:
    """The bug this guards against: re-counting the whole history every turn."""

    def test_each_turn_reports_only_its_own_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(
            monkeypatch,
            [
                [AIMessage(content="primeira", usage_metadata=usage(100, 10))],
                [AIMessage(content="segunda", usage_metadata=usage(200, 20))],
            ],
        )
        session = ChatSession()

        first = session.send("um")
        second = session.send("dois")

        assert first.metrics is not None
        assert second.metrics is not None
        assert first.metrics.total_tokens == 110
        assert second.metrics.total_tokens == 220

    def test_tool_calls_do_not_accumulate_across_turns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(
            monkeypatch,
            [
                [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "calculate", "args": {}, "id": "1"}],
                        usage_metadata=usage(10, 1),
                    ),
                    AIMessage(content="primeira", usage_metadata=usage(10, 1)),
                ],
                [AIMessage(content="segunda", usage_metadata=usage(10, 1))],
            ],
        )
        session = ChatSession()

        session.send("um")
        second = session.send("dois")

        assert second.metrics is not None
        assert second.metrics.tool_calls == 0
