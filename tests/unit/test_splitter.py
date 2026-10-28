"""Cutting documents into chunks, and the article strategy in particular.

Chunking is the decision that moved the evaluation most: 93% by characters
against 97% by article on the same dataset. It is also the one nothing else
would catch breaking. A regex that stops matching headings does not raise; it
quietly produces one enormous chunk per document, and the score falls weeks
later for reasons nobody can attribute.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from stacklens.config import ChunkStrategy
from stacklens.indexing.splitter import split_documents


def page(text: str, source: str = "resolucao.pdf", number: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": number})


def article(number: int, body: str = "") -> str:
    """One article long enough to survive the footnote filter."""
    return f"Art. {number}. {body or 'A oferta observará o disposto nesta Resolução, sem prejuízo das demais regras aplicáveis.'}"


def by_article(documents: list[Document], **kwargs: int) -> list[Document]:
    settings = {"chunk_size": 1000, "chunk_overlap": 200} | kwargs
    return split_documents(documents, strategy=ChunkStrategy.ARTICLES, **settings)  # type: ignore[arg-type]


class TestValidation:
    def test_an_overlap_at_least_the_chunk_size_is_refused(self) -> None:
        """The window would never advance, and the split would not terminate."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            split_documents([page("texto")], chunk_size=100, chunk_overlap=100)

    def test_the_message_names_both_numbers(self) -> None:
        with pytest.raises(ValueError) as raised:
            split_documents([page("texto")], chunk_size=100, chunk_overlap=150)

        assert "150" in str(raised.value)
        assert "100" in str(raised.value)

    def test_nothing_in_yields_nothing_out(self) -> None:
        assert split_documents([], chunk_size=1000, chunk_overlap=200) == []
        assert by_article([]) == []


class TestByCharacters:
    def test_a_long_document_is_cut(self) -> None:
        chunks = split_documents([page("frase. " * 400)], chunk_size=200, chunk_overlap=50)

        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 200 for chunk in chunks)

    def test_a_short_document_stays_whole(self) -> None:
        assert len(split_documents([page("curto")], chunk_size=1000, chunk_overlap=200)) == 1

    def test_the_source_survives(self) -> None:
        chunks = split_documents([page("frase. " * 400)], chunk_size=200, chunk_overlap=50)

        assert all(chunk.metadata["source"] == "resolucao.pdf" for chunk in chunks)


class TestArticleDetection:
    def test_each_article_becomes_its_own_chunk(self) -> None:
        """The point of the strategy: a rule keeps its own paragraphs."""
        text = "\n".join(article(n) for n in (1, 2, 3, 4))

        chunks = by_article([page(text)])

        assert len(chunks) == 4

    def test_the_heading_is_recorded_as_metadata(self) -> None:
        """It is the better citation: a page number belongs to the page."""
        text = "\n".join(article(n) for n in (70, 71, 72))

        articles = [chunk.metadata.get("article") for chunk in by_article([page(text)])]

        assert articles == ["Art. 70", "Art. 71", "Art. 72"]

    def test_a_lowercase_reference_does_not_open_an_article(self) -> None:
        """In the shipped corpus there are 138 such references against 106
        headings. Splitting on them would cut a rule in half at exactly the
        point where it names the exception that qualifies it."""
        text = "\n".join(
            [
                article(
                    1,
                    "A oferta observará o disposto no art. 36 desta Resolução, "
                    "ressalvado o previsto no art. 40, que trata das dispensas.",
                ),
                article(2),
                article(3),
            ]
        )

        chunks = by_article([page(text)])

        assert len(chunks) == 3
        assert "art. 36" in chunks[0].page_content

    def test_a_heading_with_no_space_still_counts(self) -> None:
        text = "\n".join(
            f"Art.{n}. A oferta observará o disposto nesta Resolução." for n in (1, 2, 3)
        )

        assert len(by_article([page(text)])) == 3

    def test_scraps_shorter_than_a_rule_are_dropped(self) -> None:
        """A stray heading in a table of contents is not an article."""
        text = "Art. 1\nArt. 2\n" + "\n".join(article(n) for n in (10, 11, 12))

        chunks = by_article([page(text)])

        assert all(len(chunk.page_content) >= 40 for chunk in chunks)


class TestFallback:
    def test_a_document_with_too_few_headings_is_cut_by_characters(self) -> None:
        """The strategy is a setting for the whole corpus. One plain README in
        the folder must not be mangled by it."""
        text = "Este documento não é uma resolução. " * 60 + article(1)

        chunks = by_article([page(text)], chunk_size=200, chunk_overlap=50)

        assert len(chunks) > 1
        assert all(chunk.metadata.get("article") is None for chunk in chunks)

    def test_prose_with_no_headings_at_all_still_splits(self) -> None:
        chunks = by_article([page("frase comum. " * 200)], chunk_size=200, chunk_overlap=50)

        assert len(chunks) > 1

    def test_two_headings_are_not_enough(self) -> None:
        """Three is the threshold: two could be a coincidence in prose."""
        text = "\n".join(article(n) for n in (1, 2))

        chunks = by_article([page(text)])

        assert all(chunk.metadata.get("article") is None for chunk in chunks)


class TestPagesAreJoined:
    def test_an_article_spanning_a_page_break_stays_whole(self) -> None:
        """The loader emits one document per PDF page, and an article routinely
        crosses one. Splitting page by page would cut exactly the rules this
        strategy exists to keep together."""
        first = page("Art. 70. A SRE poderá suspender a oferta", number=1)
        second = page(
            " quando verificar irregularidade, pelo prazo de 30 dias.\n"
            + article(71)
            + "\n"
            + article(72),
            number=2,
        )

        chunks = by_article([first, second])

        opening = next(c for c in chunks if c.metadata.get("article") == "Art. 70")
        assert "30 dias" in opening.page_content

    def test_documents_from_different_sources_are_not_merged(self) -> None:
        pages = [
            page("\n".join(article(n) for n in (1, 2, 3)), source="a.pdf"),
            page("\n".join(article(n) for n in (1, 2, 3)), source="b.pdf"),
        ]

        sources = {chunk.metadata["source"] for chunk in by_article(pages)}

        assert sources == {"a.pdf", "b.pdf"}

    def test_the_page_number_is_dropped(self) -> None:
        """It belongs to the page, not to the article."""
        text = "\n".join(article(n) for n in (1, 2, 3))

        chunks = by_article([page(text, number=7)])

        assert all("page" not in chunk.metadata for chunk in chunks)

    def test_sources_keep_the_order_they_arrived_in(self) -> None:
        pages = [
            page("\n".join(article(n) for n in (1, 2, 3)), source="segundo.pdf"),
            page("\n".join(article(n) for n in (1, 2, 3)), source="primeiro.pdf"),
        ]

        seen = [chunk.metadata["source"] for chunk in by_article(pages)]

        assert seen[0] == "segundo.pdf"


class TestOversizedBlocks:
    def test_a_huge_article_is_split_further(self) -> None:
        """Annexes and tables carry no headings, so everything after the last
        article arrives as one enormous block. Left whole it would swamp the
        context window on its own."""
        huge = article(1, "cláusula. " * 900)
        text = "\n".join([huge, article(2), article(3)])

        chunks = by_article([page(text)], chunk_size=500, chunk_overlap=50)

        assert len(chunks) > 3
        assert all(len(chunk.page_content) <= 500 for chunk in chunks)

    def test_the_pieces_keep_the_article_they_came_from(self) -> None:
        huge = article(1, "cláusula. " * 900)
        text = "\n".join([huge, article(2), article(3)])

        chunks = by_article([page(text)], chunk_size=500, chunk_overlap=50)
        pieces = [c for c in chunks if c.metadata.get("article") == "Art. 1"]

        assert len(pieces) > 1

    def test_an_article_under_the_cap_is_left_alone(self) -> None:
        text = "\n".join(article(n) for n in (1, 2, 3))

        chunks = by_article([page(text)], chunk_size=100, chunk_overlap=20)

        assert len(chunks) == 3


class TestChunkIndex:
    def test_every_chunk_is_numbered_in_order(self) -> None:
        text = "\n".join(article(n) for n in (1, 2, 3, 4, 5))

        indexes = [chunk.metadata["chunk_index"] for chunk in by_article([page(text)])]

        assert indexes == list(range(5))

    def test_the_character_strategy_numbers_them_too(self) -> None:
        chunks = split_documents([page("frase. " * 200)], chunk_size=200, chunk_overlap=50)

        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
