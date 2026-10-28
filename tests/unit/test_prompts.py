"""Domain decoupling: the corpus is configuration, never a name written in code.

These tests exist to stop a regression that already happened once — a product
name hardcoded in the system prompt and in the search tool's description,
which made the agent claim a domain the indexed documents did not have.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stacklens.config import get_settings
from stacklens.prompts import build_search_tool_description, build_system_prompt
from stacklens.tools import build_search_tool, build_tools

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def domain(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure a domain that could never be a leftover literal in the code."""
    value = "regulação do mercado de capitais brasileiro"
    monkeypatch.setenv("KNOWLEDGE_DOMAIN", value)
    get_settings.cache_clear()
    return value


class TestSystemPrompt:
    def test_carries_the_configured_domain(self, domain: str) -> None:
        assert domain in build_system_prompt()

    def test_follows_the_domain_when_it_changes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KNOWLEDGE_DOMAIN", "normas contábeis")
        get_settings.cache_clear()
        first = build_system_prompt()

        monkeypatch.setenv("KNOWLEDGE_DOMAIN", "políticas de segurança")
        get_settings.cache_clear()
        second = build_system_prompt()

        assert first != second
        assert "normas contábeis" in first
        assert "políticas de segurança" in second

    def test_keeps_the_grounding_rules(self, domain: str) -> None:
        prompt = build_system_prompt()

        assert "search_documentation" in prompt
        assert "I could not find this in the documentation" in prompt
        assert "calculate" in prompt

    def test_leaves_no_unfilled_placeholder(self, domain: str) -> None:
        assert "{" not in build_system_prompt()


class TestSearchToolDescription:
    def test_carries_the_configured_domain(self, domain: str) -> None:
        assert domain in build_search_tool_description()

    def test_reaches_the_tool_the_model_actually_sees(self, domain: str) -> None:
        assert domain in build_search_tool().description

    def test_reaches_the_registry(self, domain: str) -> None:
        descriptions = " ".join(tool.description for tool in build_tools())

        assert domain in descriptions

    def test_leaves_no_unfilled_placeholder(self, domain: str) -> None:
        assert "{" not in build_search_tool_description()


class TestRegistry:
    def test_registers_both_tools(self, domain: str) -> None:
        assert {tool.name for tool in build_tools()} == {"search_documentation", "calculate"}

    def test_every_tool_describes_itself(self, domain: str) -> None:
        assert all(tool.description.strip() for tool in build_tools())


class TestNoHardcodedDomain:
    """The literal that caused the regression must never come back."""

    def test_source_never_names_the_old_product(self) -> None:
        offenders = [
            path.relative_to(SOURCE_ROOT)
            for path in SOURCE_ROOT.rglob("*.py")
            if "nimbus" in path.read_text(encoding="utf-8").lower()
        ]

        assert offenders == []

    def test_default_domain_is_generic(self) -> None:
        assert (
            get_settings().knowledge_domain
            == "engineering documentation, API contracts, architecture and operational runbooks"
        )


class TestSearchBudget:
    """A vector search always returns its k nearest chunks, however far.

    It can therefore never report finding nothing, and on a question the
    corpus cannot answer the agent rewords the query until the graph runs out
    of steps and the call dies with no answer at all. The budget turns that
    into a refusal, which is the truthful outcome.
    """

    def search_count(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Replace the retrieval, recording every query that reaches it."""
        from stacklens.tools import documentation

        asked: list[str] = []

        def fake_search(question: str, k: int | None = None) -> list:
            asked.append(question)
            return []

        monkeypatch.setattr(documentation, "search", fake_search)
        return asked

    def test_it_searches_while_within_budget(
        self, monkeypatch: pytest.MonkeyPatch, domain: str
    ) -> None:
        monkeypatch.setenv("MAX_SEARCHES_PER_TURN", "3")
        get_settings.cache_clear()
        asked = self.search_count(monkeypatch)

        tool = build_search_tool()
        for term in ("a", "b", "c"):
            tool.invoke({"question": term})

        assert asked == ["a", "b", "c"]

    def test_beyond_the_budget_it_stops_searching(
        self, monkeypatch: pytest.MonkeyPatch, domain: str
    ) -> None:
        monkeypatch.setenv("MAX_SEARCHES_PER_TURN", "2")
        get_settings.cache_clear()
        asked = self.search_count(monkeypatch)

        tool = build_search_tool()
        for term in ("a", "b", "c", "d"):
            tool.invoke({"question": term})

        assert asked == ["a", "b"]

    def test_it_tells_the_model_to_stop_rather_than_failing(
        self, monkeypatch: pytest.MonkeyPatch, domain: str
    ) -> None:
        """Raising would end the run; the model has to be told to conclude."""
        monkeypatch.setenv("MAX_SEARCHES_PER_TURN", "1")
        get_settings.cache_clear()
        self.search_count(monkeypatch)

        tool = build_search_tool()
        tool.invoke({"question": "primeira"})
        answer = tool.invoke({"question": "segunda"})

        assert "refuse" in answer
        assert "Search budget exhausted" in answer

    def test_each_tool_carries_its_own_budget(
        self, monkeypatch: pytest.MonkeyPatch, domain: str
    ) -> None:
        """A fresh tool per turn is what resets the count."""
        monkeypatch.setenv("MAX_SEARCHES_PER_TURN", "1")
        get_settings.cache_clear()
        asked = self.search_count(monkeypatch)

        build_search_tool().invoke({"question": "turno-1"})
        build_search_tool().invoke({"question": "turno-2"})

        assert asked == ["turno-1", "turno-2"]
