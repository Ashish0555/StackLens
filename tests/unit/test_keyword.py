"""Turning a question into something the database's text search understands."""

from __future__ import annotations

from stacklens.indexing.keyword import build_terms


class TestBuildTerms:
    def test_it_joins_terms_with_or(self) -> None:
        """A question is a sentence and a passage is a fragment.

        Requiring every word to appear would answer nothing for most real
        questions; ranking is what decides which partial match is worth
        reading.
        """
        assert build_terms("prazo de suspensão") == "prazo | de | suspensao"

    def test_it_folds_accents(self) -> None:
        """The stored side is folded too, by unaccent in the configuration."""
        assert build_terms("suspensão") == build_terms("suspensao")

    def test_it_drops_the_operators_tsquery_understands(self) -> None:
        """The only characters that survive are letters and digits.

        This is what keeps a question from reaching the query parser as
        syntax. `to_tsquery` raises on a malformed expression, so a stray
        ampersand would be an error rather than a poor result.
        """
        assert build_terms("prazo & (suspensão | !cancelamento)") == (
            "prazo | suspensao | cancelamento"
        )

    def test_it_keeps_article_numbers(self) -> None:
        """The identifiers are exactly what the embedding is worst at."""
        assert build_terms("Art. 70, §2º") == "art | 70 | 2"

    def test_a_question_with_no_words_yields_nothing(self) -> None:
        """The caller skips the query entirely rather than sending garbage."""
        assert build_terms("!!! ???") == ""
