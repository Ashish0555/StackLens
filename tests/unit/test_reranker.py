"""The second pass, and the fact that it is optional.

The model itself is not exercised here: loading a cross-encoder costs seconds
and a download, and what can go wrong in this code is the wiring around it.
What is tested is that it reorders by score, that it is chosen by
configuration, that it never runs by default, and that asking for it without
the dependency says which command installs it.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from stacklens.config import RerankStrategy, get_settings
from stacklens.indexing import reranker as module
from stacklens.indexing.reranker import (
    CrossEncoderReranker,
    PassThroughReranker,
    RerankerUnavailableError,
    get_reranker,
    reranking_enabled,
)


def chunk(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "doc.pdf"})


class FakeCrossEncoder:
    """Scores a pair by how much of the question the passage repeats."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls = 0

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls += 1
        return [
            len(set(query.lower().split()) & set(passage.lower().split()))
            for query, passage in pairs
        ]


@pytest.fixture
def cross_encoder(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand in for sentence_transformers without importing it."""
    built: dict[str, Any] = {}

    def build(model_name: str) -> FakeCrossEncoder:
        built["model"] = FakeCrossEncoder(model_name)
        return built["model"]

    monkeypatch.setattr(
        CrossEncoderReranker,
        "_load",
        lambda self: built.get("model") or build(self.model_name),
    )
    return built


class TestPassThrough:
    def test_it_keeps_the_order_it_was_given(self) -> None:
        candidates = [chunk("primeiro"), chunk("segundo"), chunk("terceiro")]

        reranked = PassThroughReranker().rerank("pergunta", candidates, limit=3)

        assert [document.page_content for document in reranked] == [
            "primeiro",
            "segundo",
            "terceiro",
        ]

    def test_it_cuts_to_the_limit(self) -> None:
        candidates = [chunk(str(number)) for number in range(10)]

        assert len(PassThroughReranker().rerank("pergunta", candidates, limit=4)) == 4

    def test_it_handles_an_empty_pool(self) -> None:
        assert PassThroughReranker().rerank("pergunta", [], limit=5) == []


class TestCrossEncoder:
    def test_it_reorders_by_score(self, cross_encoder: dict[str, Any]) -> None:
        """The point of the second pass: the pool arrives in the wrong order."""
        candidates = [
            chunk("assunto totalmente diferente"),
            chunk("o prazo de suspensao da oferta"),
        ]

        reranked = CrossEncoderReranker("fake").rerank(
            "qual o prazo de suspensao", candidates, limit=2
        )

        assert reranked[0].page_content == "o prazo de suspensao da oferta"

    def test_it_returns_only_the_limit(self, cross_encoder: dict[str, Any]) -> None:
        """A reranker is a funnel: it reads many and hands back few."""
        candidates = [chunk(f"trecho {number} prazo") for number in range(20)]

        assert len(CrossEncoderReranker("fake").rerank("prazo", candidates, limit=5)) == 5

    def test_it_scores_every_candidate_in_one_call(self, cross_encoder: dict[str, Any]) -> None:
        """The cost is the forward pass, and batching is what the model is for."""
        candidates = [chunk(f"trecho {number}") for number in range(12)]

        CrossEncoderReranker("fake").rerank("pergunta", candidates, limit=3)

        assert cross_encoder["model"].calls == 1

    def test_an_empty_pool_never_loads_the_model(self, cross_encoder: dict[str, Any]) -> None:
        """Loading costs seconds, and there is nothing to score."""
        assert CrossEncoderReranker("fake").rerank("pergunta", [], limit=5) == []
        assert "model" not in cross_encoder

    def test_the_model_is_loaded_once(self, cross_encoder: dict[str, Any]) -> None:
        reranker = CrossEncoderReranker("fake")

        reranker.rerank("pergunta", [chunk("a")], limit=1)
        reranker.rerank("outra", [chunk("b")], limit=1)

        assert cross_encoder["model"].calls == 2


class TestMissingDependency:
    def test_it_names_the_command_that_installs_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """torch is two gigabytes, so it is not installed by default."""
        import builtins

        real_import = builtins.__import__

        def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "sentence_transformers":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)

        with pytest.raises(RerankerUnavailableError) as raised:
            CrossEncoderReranker("fake").rerank("pergunta", [chunk("a")], limit=1)

        message = str(raised.value)

        assert "pip install" in message
        assert "rerank" in message
        assert "RERANK_STRATEGY=none" in message


class TestSelection:
    def test_it_is_off_by_default(self) -> None:
        """It fixes precision, and what was wrong on this corpus was recall."""
        assert get_settings().rerank_strategy is RerankStrategy.NONE
        assert reranking_enabled() is False
        assert isinstance(get_reranker(), PassThroughReranker)

    def test_configuration_turns_it_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RERANK_STRATEGY", "cross_encoder")
        get_settings.cache_clear()
        module.forget_reranker()

        assert reranking_enabled() is True
        assert isinstance(get_reranker(), CrossEncoderReranker)

    def test_the_model_comes_from_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RERANK_STRATEGY", "cross_encoder")
        monkeypatch.setenv("RERANK_MODEL", "BAAI/bge-reranker-base")
        get_settings.cache_clear()
        module.forget_reranker()

        built = get_reranker()

        assert isinstance(built, CrossEncoderReranker)
        assert built.model_name == "BAAI/bge-reranker-base"

    def test_it_is_built_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rebuilding would reload the weights, which costs seconds."""
        monkeypatch.setenv("RERANK_STRATEGY", "cross_encoder")
        get_settings.cache_clear()
        module.forget_reranker()

        assert get_reranker() is get_reranker()

    def test_an_unknown_strategy_is_refused_at_boot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo is a startup failure, not a silently disabled second pass."""
        from pydantic import ValidationError

        monkeypatch.setenv("RERANK_STRATEGY", "crossencoder")
        get_settings.cache_clear()

        with pytest.raises(ValidationError):
            get_settings()
