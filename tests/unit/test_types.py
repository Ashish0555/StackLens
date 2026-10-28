"""Domain types: the contract that replaced the untyped result dictionaries."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from stacklens.types import AnswerResult, SearchHit, ToolCall


class TestSearchHit:
    def test_exposes_the_source_of_the_chunk(self) -> None:
        hit = SearchHit(Document(page_content="texto", metadata={"source": "produto.md"}), 0.12)

        assert hit.source == "produto.md"

    def test_falls_back_when_the_source_is_missing(self) -> None:
        hit = SearchHit(Document(page_content="texto"), 0.12)

        assert hit.source == "desconhecida"

    def test_exposes_the_chunk_content(self) -> None:
        hit = SearchHit(Document(page_content="texto"), 0.12)

        assert hit.content == "texto"

    def test_is_immutable(self) -> None:
        hit = SearchHit(Document(page_content="texto"), 0.12)

        with pytest.raises(AttributeError):
            hit.distance = 0.9  # type: ignore[misc]


class TestAnswerResult:
    def test_lists_the_tools_that_were_used(self) -> None:
        result = AnswerResult(
            answer="R$ 890",
            tool_calls=[
                ToolCall("search_documentation", {"question": "preço"}),
                ToolCall("calculate", {"expression": "890*12"}),
            ],
        )

        assert result.tool_names == ["search_documentation", "calculate"]
        assert result.used_tools is True

    def test_reports_when_no_tool_was_used(self) -> None:
        result = AnswerResult(answer="oi")

        assert result.tool_names == []
        assert result.used_tools is False

    def test_defaults_do_not_leak_between_instances(self) -> None:
        first = AnswerResult(answer="a")
        second = AnswerResult(answer="b")

        first.tool_calls.append(ToolCall("calculate", {}))

        assert second.tool_calls == []
