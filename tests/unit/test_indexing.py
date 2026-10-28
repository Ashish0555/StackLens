"""Document loading and splitting: pure logic, no network."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from stacklens.indexing import load_documents, split_documents


class TestLoadDocuments:
    def test_loads_every_supported_file(self, knowledge_base: Path) -> None:
        documents = load_documents(knowledge_base)

        sources = {doc.metadata["source"] for doc in documents}
        assert sources == {"pricing.md", "limits.txt"}

    def test_skips_unsupported_extensions(self, knowledge_base: Path) -> None:
        sources = {doc.metadata["source"] for doc in load_documents(knowledge_base)}

        assert "ignored.png" not in sources

    def test_skips_empty_files(self, knowledge_base: Path) -> None:
        sources = {doc.metadata["source"] for doc in load_documents(knowledge_base)}

        assert "empty.md" not in sources

    def test_walks_subdirectories(self, knowledge_base: Path) -> None:
        nested = knowledge_base / "nested"
        nested.mkdir()
        (nested / "deep.md").write_text("Conteudo aninhado.", encoding="utf-8")

        sources = {doc.metadata["source"] for doc in load_documents(knowledge_base)}

        assert "deep.md" in sources

    def test_order_is_deterministic(self, knowledge_base: Path) -> None:
        first = [doc.metadata["source"] for doc in load_documents(knowledge_base)]
        second = [doc.metadata["source"] for doc in load_documents(knowledge_base)]

        assert first == second

    def test_missing_directory_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_documents(tmp_path / "nao-existe")

    def test_every_document_carries_its_source(self, knowledge_base: Path) -> None:
        for document in load_documents(knowledge_base):
            assert document.metadata.get("source")


class TestSplitDocuments:
    def test_long_document_is_broken_into_several_chunks(self) -> None:
        document = Document(page_content="palavra " * 500, metadata={"source": "a.md"})

        chunks = split_documents([document], chunk_size=200, chunk_overlap=50)

        assert len(chunks) > 1

    def test_short_document_stays_whole(self) -> None:
        document = Document(page_content="texto curto", metadata={"source": "a.md"})

        chunks = split_documents([document], chunk_size=1000, chunk_overlap=200)

        assert len(chunks) == 1
        assert chunks[0].page_content == "texto curto"

    def test_source_metadata_survives_the_split(self) -> None:
        document = Document(page_content="palavra " * 500, metadata={"source": "a.md"})

        chunks = split_documents([document], chunk_size=200, chunk_overlap=50)

        assert all(chunk.metadata["source"] == "a.md" for chunk in chunks)

    def test_chunks_are_numbered_in_order(self) -> None:
        document = Document(page_content="palavra " * 500, metadata={"source": "a.md"})

        chunks = split_documents([document], chunk_size=200, chunk_overlap=50)

        assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))

    def test_overlap_greater_than_chunk_is_rejected(self) -> None:
        document = Document(page_content="texto", metadata={"source": "a.md"})

        with pytest.raises(ValueError, match="chunk_overlap"):
            split_documents([document], chunk_size=100, chunk_overlap=100)

    def test_empty_input_yields_empty_output(self) -> None:
        assert split_documents([], chunk_size=1000, chunk_overlap=200) == []
