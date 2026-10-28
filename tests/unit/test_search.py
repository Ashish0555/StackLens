"""How the two stages fit together: what is retrieved, and what survives.

Retrieval decides what is in the pool. Reranking decides what comes out of
it. The width of the pool is therefore not a constant, it depends on whether
anything is going to narrow it afterwards, and getting that wrong is either
wasted work or a reranker with nothing to choose between.

Neither the database nor a model is involved here. Both are faked, because
what is being checked is the arithmetic and the order of the stages.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from stacklens.config import get_settings
from stacklens.indexing import vector_store as module
from stacklens.indexing.hybrid import FUSION_POOL


def chunk(text: str, source: str = "doc.pdf") -> Document:
    return Document(page_content=text, metadata={"source": source})


class FakeStore:
    """Records how wide it was asked to search."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.asked_for: list[int] = []

    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
        self.asked_for.append(k)
        return [(document, 0.1 * rank) for rank, document in enumerate(self.documents[:k])]


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore([chunk(f"trecho {number}") for number in range(500)])
    monkeypatch.setattr(module, "get_vector_store", lambda: fake)
    monkeypatch.setattr(module, "keyword_search", lambda query, limit: [])
    return fake


@pytest.fixture
def reranking(monkeypatch: pytest.MonkeyPatch) -> list[list[Document]]:
    """Turn the second pass on, with a reranker that reverses the pool."""
    seen: list[list[Document]] = []

    class ReversingReranker:
        def rerank(self, query: str, candidates: list[Document], limit: int) -> list[Document]:
            seen.append(candidates)
            return list(reversed(candidates))[:limit]

    monkeypatch.setattr(module, "reranking_enabled", lambda: True)
    monkeypatch.setattr(module, "get_reranker", ReversingReranker)
    return seen


class TestPoolWidth:
    def test_without_reranking_the_pool_is_what_the_caller_asked_for(
        self, store: FakeStore
    ) -> None:
        """Retrieving wider would be work thrown away: the pool is the answer."""
        module.search("pergunta", k=8)

        assert store.asked_for == [8 * FUSION_POOL]

    def test_with_reranking_the_pool_widens_to_the_candidate_count(
        self, store: FakeStore, reranking: list[list[Document]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reranker only earns its latency when handed more than it returns."""
        monkeypatch.setenv("RERANK_CANDIDATES", "24")
        get_settings.cache_clear()

        module.search("pergunta", k=8)

        assert store.asked_for == [24 * FUSION_POOL]

    def test_the_pool_never_narrows_below_what_was_asked_for(
        self, store: FakeStore, reranking: list[list[Document]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fewer candidates than results would leave the caller short."""
        monkeypatch.setenv("RERANK_CANDIDATES", "4")
        get_settings.cache_clear()

        module.search("pergunta", k=8)

        assert store.asked_for == [8 * FUSION_POOL]

    def test_the_vector_only_strategy_skips_the_fusion_multiplier(
        self, store: FakeStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """There is no second list to fuse, so there is nothing to fuse deeper."""
        monkeypatch.setenv("SEARCH_STRATEGY", "vector")
        get_settings.cache_clear()

        module.search("pergunta", k=8)

        assert store.asked_for == [8]


class TestStageOrder:
    def test_the_reranker_reads_the_candidates_and_decides_the_result(
        self, store: FakeStore, reranking: list[list[Document]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RERANK_CANDIDATES", "12")
        get_settings.cache_clear()

        hits = module.search("pergunta", k=3)

        assert len(reranking[0]) == 12
        assert [hit.document.page_content for hit in hits] == [
            "trecho 11",
            "trecho 10",
            "trecho 9",
        ]

    def test_without_reranking_the_retrieved_order_survives(self, store: FakeStore) -> None:
        hits = module.search("pergunta", k=3)

        assert [hit.document.page_content for hit in hits] == [
            "trecho 0",
            "trecho 1",
            "trecho 2",
        ]


class TestDistances:
    def test_a_hit_the_vector_search_ranked_carries_its_own_distance(
        self, store: FakeStore
    ) -> None:
        assert module.search("pergunta", k=2)[0].distance == pytest.approx(0.0)

    def test_a_keyword_only_hit_reports_the_worst_distance_seen(
        self, store: FakeStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It has no distance of its own, and inventing one would read as a score."""
        stranger = chunk("só a busca por palavra achou isto")
        monkeypatch.setattr(module, "keyword_search", lambda query, limit: [stranger])

        hits = module.search("pergunta", k=40)
        found = next(hit for hit in hits if hit.document.page_content == stranger.page_content)

        assert found.distance == max(hit.distance for hit in hits)
