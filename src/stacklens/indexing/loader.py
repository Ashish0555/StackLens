"""Reading documents from disk into LangChain Documents."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = frozenset({".md", ".txt", ".markdown", ".rst"})
PDF_SUFFIXES = frozenset({".pdf"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


def load_documents(data_dir: Path) -> list[Document]:
    """Walk the directory recursively and load every supported file.

    Unsupported extensions are skipped silently: any real folder carries
    images and system files that are not part of the knowledge base.
    """
    if not data_dir.exists():
        msg = f"Pasta de dados não encontrada: {data_dir}"
        raise FileNotFoundError(msg)

    documents: list[Document] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue

        loaded = _load_pdf(path) if suffix in PDF_SUFFIXES else _load_text(path)
        logger.debug("Loaded %s (%d document(s))", path.name, len(loaded))
        documents.extend(loaded)

    logger.info("Loaded %d document(s) from %s", len(documents), data_dir)
    return documents


def _load_text(path: Path) -> list[Document]:
    """Read a plain text file as a single Document."""
    content = path.read_text(encoding="utf-8", errors="ignore")
    if not content.strip():
        return []
    return [Document(page_content=content, metadata={"source": path.name})]


def _load_pdf(path: Path) -> list[Document]:
    """Read a PDF as one Document per page that actually holds text."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": page_number},
            )
        )

    return documents
