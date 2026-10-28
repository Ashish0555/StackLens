"""The store against a real Postgres, with a fake embedding.

Every function here writes SQL, and SQL is the one thing a mock cannot check.
A `DELETE` with a wrong `WHERE` passes every unit test and empties the index
in production.

The embedding is faked and deterministic, so these need a database but no API
key and no network. That is what lets them run in CI, which is the only place
the SQL is exercised on a machine nobody has already set up by hand.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pytest
from langchain_core.documents import Document

from stacklens.config import get_settings
from tests.conftest import requires_postgres

pytestmark = [pytest.mark.integration, requires_postgres]


class FakeEmbeddings:
    """Stable vectors derived from the text, at the configured width.

    Deterministic on purpose: the same chunk embeds to the same vector on
    every run, so a test that asserts an order is not asserting luck.
    """

    def __init__(self, width: int) -> None:
        self.width = width

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeated = (digest * (self.width // len(digest) + 1))[: self.width]
        return [byte / 255 for byte in repeated]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


CORPUS = [
    Document(
        page_content="Art. 70. A SRE pode suspender ou cancelar a oferta a qualquer tempo.",
        metadata={"source": "resolucao-160.pdf", "article": "Art. 70"},
    ),
    Document(
        page_content="§ 2º O prazo de suspensão da oferta não pode ser superior a 30 dias.",
        metadata={"source": "resolucao-160.pdf", "article": "Art. 70"},
    ),
    Document(
        page_content="Art. 12. O lote suplementar não pode ultrapassar 15% da quantidade.",
        metadata={"source": "resolucao-35.pdf", "article": "Art. 12"},
    ),
]


@pytest.fixture
def store(temporary_index: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A throwaway collection, with the provider replaced by arithmetic."""
    from stacklens.indexing import vector_store

    monkeypatch.setenv("OPENAI_API_KEY", "nao-usado")
    monkeypatch.setattr(
        vector_store,
        "build_embeddings",
        lambda: FakeEmbeddings(get_settings().embedding_dimensions),
    )
    # The classifier would download two gigabytes to say nothing about three
    # sentences of regulation.
    monkeypatch.setattr(vector_store, "_warn_about_injection", lambda chunks: None)

    yield


@pytest.fixture
def indexed(store: None) -> None:
    from stacklens.indexing import index_documents

    index_documents(CORPUS)


class TestIndexing:
    def test_it_reports_what_it_wrote(self, store: None) -> None:
        from stacklens.indexing import index_documents

        assert index_documents(CORPUS) == 3

    def test_the_count_matches(self, indexed: None) -> None:
        from stacklens.indexing import count_documents

        assert count_documents() == 3

    def test_nothing_to_index_is_not_an_error(self, store: None) -> None:
        from stacklens.indexing import index_documents

        assert index_documents([]) == 0

    def test_indexing_twice_does_not_duplicate(self, indexed: None) -> None:
        """The id is a hash of collection, source and content, so a repeat
        overwrites.

        This is what would make ingestion safe to retry from a queue, where
        the same message can be delivered more than once.
        """
        from stacklens.indexing import count_documents, index_documents

        index_documents(CORPUS)

        assert count_documents() == 3

    def test_the_indexes_exist_after_the_first_write(self, indexed: None) -> None:
        """Neither can be created before langchain-postgres makes its tables."""
        from sqlalchemy import text

        from stacklens.indexing import get_engine

        with get_engine().connect() as connection:
            names = {
                row[0]
                for row in connection.execute(
                    text(
                        "SELECT indexname FROM pg_indexes "
                        "WHERE tablename = 'langchain_pg_embedding'"
                    )
                ).all()
            }

        assert "stacklens_document_fts" in names
        assert "stacklens_embedding_hnsw" in names


class TestCollectionIsolation:
    """Two collections must be able to hold the same chunk.

    The id is the primary key of a table every collection shares. Derived from
    the content alone, writing a chunk into a second collection silently moves
    the row out of the first, and the second reports nothing was written. This
    is invisible with one collection and wrong the moment there are two, which
    is what a second corpus, a tenant or an A/B of a chunking strategy is.
    """

    def test_the_same_chunk_can_live_in_two_collections(
        self, store: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from stacklens.indexing import count_documents, index_documents, reset_index

        first = get_settings().collection_name
        index_documents(CORPUS)

        second = f"{first}_vizinha"
        monkeypatch.setenv("COLLECTION_NAME", second)
        get_settings.cache_clear()
        try:
            assert index_documents(CORPUS) == 3
            assert count_documents() == 3

            monkeypatch.setenv("COLLECTION_NAME", first)
            get_settings.cache_clear()

            assert count_documents() == 3
        finally:
            monkeypatch.setenv("COLLECTION_NAME", second)
            get_settings.cache_clear()
            reset_index()
            monkeypatch.setenv("COLLECTION_NAME", first)
            get_settings.cache_clear()

    def test_removing_from_one_leaves_the_other(
        self, store: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from stacklens.indexing import count_documents, delete_source, index_documents, reset_index

        first = get_settings().collection_name
        index_documents(CORPUS)

        second = f"{first}_vizinha"
        monkeypatch.setenv("COLLECTION_NAME", second)
        get_settings.cache_clear()
        try:
            index_documents(CORPUS)
            delete_source("resolucao-160.pdf")

            monkeypatch.setenv("COLLECTION_NAME", first)
            get_settings.cache_clear()

            assert count_documents() == 3
        finally:
            monkeypatch.setenv("COLLECTION_NAME", second)
            get_settings.cache_clear()
            reset_index()
            monkeypatch.setenv("COLLECTION_NAME", first)
            get_settings.cache_clear()


class TestCounting:
    def test_an_empty_collection_counts_zero(self, store: None) -> None:
        from stacklens.indexing import count_documents

        assert count_documents() == 0

    def test_it_counts_only_its_own_collection(
        self, indexed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Collections share the tables, so a wrong WHERE would count both."""
        from stacklens.indexing import count_documents

        monkeypatch.setenv("COLLECTION_NAME", "uma_colecao_que_nao_existe")
        get_settings.cache_clear()

        assert count_documents() == 0


class TestSources:
    def test_it_groups_by_document(self, indexed: None) -> None:
        """590 chunks is not something a person can act on. A file name is."""
        from stacklens.indexing import list_sources

        found = {source.name: source.chunks for source in list_sources()}

        assert found == {"resolucao-160.pdf": 2, "resolucao-35.pdf": 1}

    def test_an_empty_collection_lists_nothing(self, store: None) -> None:
        from stacklens.indexing import list_sources

        assert list_sources() == []

    def test_they_come_back_in_a_stable_order(self, indexed: None) -> None:
        from stacklens.indexing import list_sources

        names = [source.name for source in list_sources()]

        assert names == sorted(names)


class TestRemoval:
    def test_it_removes_only_the_named_document(self, indexed: None) -> None:
        from stacklens.indexing import delete_source, list_sources

        removed = delete_source("resolucao-160.pdf")

        assert removed == 2
        assert [source.name for source in list_sources()] == ["resolucao-35.pdf"]

    def test_removing_what_is_not_there_returns_zero(self, indexed: None) -> None:
        """Idempotent: a retry is not an error."""
        from stacklens.indexing import count_documents, delete_source

        assert delete_source("nao-existe.pdf") == 0
        assert count_documents() == 3

    def test_removing_twice_is_not_an_error(self, indexed: None) -> None:
        from stacklens.indexing import delete_source

        delete_source("resolucao-35.pdf")

        assert delete_source("resolucao-35.pdf") == 0

    def test_reset_empties_the_collection(self, indexed: None) -> None:
        from stacklens.indexing import count_documents, reset_index

        reset_index()

        assert count_documents() == 0


class TestSearching:
    def test_a_search_returns_hits_with_their_source(self, indexed: None) -> None:
        from stacklens.indexing import search

        hits = search("prazo de suspensão", k=2)

        assert hits
        assert all(hit.source for hit in hits)

    def test_the_keyword_half_finds_what_the_words_name(self, indexed: None) -> None:
        """A question written without the accent must still find the text.

        This is the whole reason for the portuguese_unaccent configuration:
        the stock one stems "suspensão" and "suspensao" to different words.
        """
        from stacklens.indexing import keyword_search

        found = keyword_search("prazo de suspensao da oferta", 1)

        assert found
        assert "30 dias" in found[0].page_content

    def test_it_finds_an_article_number(self, indexed: None) -> None:
        """Identifiers have identity rather than meaning, and an embedding
        blurs exactly those."""
        from stacklens.indexing import keyword_search

        found = keyword_search("lote suplementar 15%", 1)

        assert found
        assert "lote suplementar" in found[0].page_content

    def test_a_question_with_no_usable_words_finds_nothing(self, indexed: None) -> None:
        from stacklens.indexing import keyword_search

        assert keyword_search("!!! ???", 5) == []

    def test_it_never_returns_more_than_asked_for(self, indexed: None) -> None:
        from stacklens.indexing import search

        assert len(search("oferta", k=2)) <= 2

    def test_searching_an_empty_collection_is_not_an_error(self, store: None) -> None:
        from stacklens.indexing import search

        assert search("qualquer coisa", k=3) == []


class TestLocation:
    def test_it_names_the_database_without_the_password(self, store: None) -> None:
        """This string reaches logs, `stacklens status` and the API status response."""
        from stacklens.indexing import describe_location

        described = describe_location()

        assert "postgresql" in described
        assert "stacklens:rag@" not in described
