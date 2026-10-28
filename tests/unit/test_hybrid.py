"""Fusing two rankings into one.

An embedding spreads a long article's signal across everything the article
discusses, so one sentence stating a deadline ranks below the article's main
subject. Keyword search does not have that problem, and cannot follow a
paraphrase. Both run, and this is where the two rankings are merged.
"""

from __future__ import annotations

from langchain_core.documents import Document

from stacklens.indexing.hybrid import RRF_CONSTANT, fuse, tokenise


def chunk(text: str, source: str = "doc.pdf") -> Document:
    return Document(page_content=text, metadata={"source": source})


class TestTokenise:
    def test_it_lowercases(self) -> None:
        assert tokenise("PRAZO Máximo") == ["prazo", "maximo"]

    def test_it_folds_accents(self) -> None:
        """Portuguese writes the same word both ways often enough to matter."""
        assert tokenise("suspensão") == tokenise("suspensao")

    def test_it_drops_punctuation(self) -> None:
        assert tokenise("Art. 70, §2º:") == ["art", "70", "2"]

    def test_it_keeps_numbers(self) -> None:
        """The deadlines and percentages are the point of this corpus."""
        assert "30" in tokenise("prazo de 30 dias")

    def test_empty_text_yields_nothing(self) -> None:
        assert tokenise("") == []


class TestFusion:
    def test_a_document_in_both_rankings_outranks_one_in_either(self) -> None:
        """The whole idea: agreement between retrievers is the signal."""
        both = chunk("both")
        only_first = chunk("only first")
        only_second = chunk("only second")

        fused = fuse([[only_first, both], [only_second, both]], limit=3)

        assert fused[0].page_content == "both"

    def test_a_higher_rank_scores_better(self) -> None:
        first = chunk("first")
        second = chunk("second")

        assert fuse([[first, second]], limit=2)[0].page_content == "first"

    def test_it_deduplicates_the_same_chunk(self) -> None:
        repeated = chunk("mesmo texto")

        assert len(fuse([[repeated], [repeated]], limit=5)) == 1

    def test_two_chunks_with_the_same_text_from_different_files_are_distinct(self) -> None:
        """Identity is the source plus the text, as it is for the stored ids."""
        assert len(fuse([[chunk("igual", "a.pdf"), chunk("igual", "b.pdf")]], limit=5)) == 2

    def test_it_cuts_to_the_limit(self) -> None:
        many = [chunk(str(n)) for n in range(20)]

        assert len(fuse([many], limit=4)) == 4

    def test_fusing_nothing_yields_nothing(self) -> None:
        assert fuse([], limit=5) == []

    def test_a_deep_hit_in_one_list_can_beat_a_shallow_miss_in_the_other(self) -> None:
        """Why the pool is wider than the number of passages wanted."""
        deep = chunk("deep")
        shallow_only = chunk("shallow")

        first = [chunk(f"v{n}") for n in range(3)] + [shallow_only]
        second = [chunk(f"k{n}") for n in range(2)] + [deep]

        fused = fuse([first, second], limit=8)

        assert fused.index(deep) < fused.index(shallow_only)

    def test_the_constant_is_the_published_one(self) -> None:
        assert RRF_CONSTANT == 60
