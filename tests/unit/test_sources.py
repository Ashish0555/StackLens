"""Listing and removing indexed documents.

Re-ingesting overwrites a chunk whose text is unchanged, so it cannot undo a
deletion by itself. A document taken out of the source folder, or a paragraph
removed in a revision, would otherwise stay in the index for good and keep
being retrieved against questions it no longer answers.

The SQL runs against a real database in the integration suite. What is checked
here is the command around it: that it names what will disappear, that it
refuses a name it does not know, and that it does not delete without consent.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from stacklens.cli import app
from stacklens.indexing import IndexedSource

runner = CliRunner()


def patch(monkeypatch: pytest.MonkeyPatch, name: str, value: object) -> None:
    """Replace a name in every cli submodule that imported it.

    The commands live in one module each and import what they need by name, so
    a fake has to land wherever the name was bound. Patching by search rather
    than by path means a command moving between modules does not silently stop
    being faked.
    """
    import sys

    import stacklens.cli  # noqa: F401  imports every submodule

    replaced = [
        module
        for path, module in list(sys.modules.items())
        if path.startswith("stacklens.cli") and hasattr(module, name)
    ]
    for module in replaced:
        monkeypatch.setattr(module, name, value)

    assert replaced, f"{name} is not imported by any cli module"


INDEXED = [
    IndexedSource(name="resolucao-160.pdf", chunks=416),
    IndexedSource(name="resolucao-35.pdf", chunks=131),
]


@pytest.fixture
def indexed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fake the store, and record what was asked to be deleted."""
    deleted: list[str] = []

    def delete(source: str) -> int:
        deleted.append(source)
        return next(entry.chunks for entry in INDEXED if entry.name == source)

    patch(monkeypatch, "list_sources", lambda: list(INDEXED))
    patch(monkeypatch, "delete_source", delete)
    return deleted


@pytest.fixture
def empty(monkeypatch: pytest.MonkeyPatch) -> None:
    patch(monkeypatch, "list_sources", list)


class TestListing:
    def test_it_shows_each_document_and_its_chunk_count(self, indexed: list[str]) -> None:
        """590 chunks is not something a person can act on. A file name is."""
        result = runner.invoke(app, ["sources"])

        assert result.exit_code == 0
        assert "resolucao-160.pdf" in result.stdout
        assert "416" in result.stdout

    def test_it_totals_the_documents_and_the_chunks(self, indexed: list[str]) -> None:
        result = runner.invoke(app, ["sources"])

        assert "2 documento(s)" in result.stdout
        assert "547" in result.stdout

    def test_an_empty_index_names_the_command_that_fills_it(self, empty: None) -> None:
        result = runner.invoke(app, ["sources"])

        assert result.exit_code == 0
        assert "stacklens ingest" in result.stdout


class TestRemoval:
    def test_it_removes_the_named_document(self, indexed: list[str]) -> None:
        result = runner.invoke(app, ["sources", "--remove", "resolucao-35.pdf", "--yes"])

        assert result.exit_code == 0
        assert indexed == ["resolucao-35.pdf"]

    def test_it_reports_how_many_chunks_went(self, indexed: list[str]) -> None:
        result = runner.invoke(app, ["sources", "--remove", "resolucao-35.pdf", "--yes"])

        assert "131" in result.stdout

    def test_it_asks_before_deleting(self, indexed: list[str]) -> None:
        """Nothing re-ingests this on its own, so the prompt is the safety net."""
        result = runner.invoke(app, ["sources", "--remove", "resolucao-35.pdf"], input="n\n")

        assert indexed == []
        assert result.exit_code == 0

    def test_confirming_at_the_prompt_deletes(self, indexed: list[str]) -> None:
        runner.invoke(app, ["sources", "--remove", "resolucao-35.pdf"], input="y\n")

        assert indexed == ["resolucao-35.pdf"]

    def test_the_prompt_says_how_much_will_disappear(self, indexed: list[str]) -> None:
        result = runner.invoke(app, ["sources", "--remove", "resolucao-160.pdf"], input="n\n")

        assert "416" in result.stdout

    def test_an_unknown_document_fails_and_lists_what_exists(self, indexed: list[str]) -> None:
        """A typo in a file name must not look like a document with no chunks."""
        result = runner.invoke(app, ["sources", "--remove", "resolucao-99.pdf", "--yes"])

        assert result.exit_code == 1
        assert indexed == []
        assert "resolucao-160.pdf" in result.stdout
