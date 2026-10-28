"""Breaking documents into indexable chunks.

Two strategies. Characters is the general one. Articles exists because legal
and regulatory texts are already divided by their author, and cutting every
1000 characters splits a rule away from the exception that qualifies it.
"""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from stacklens.config import ChunkStrategy

logger = logging.getLogger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Only a capitalised "Art." opens an article. Lowercase "art. 36" is a
# cross-reference inside a sentence, and splitting there would cut a rule in
# half -- in this corpus there are 138 such references against 106 headings.
_ARTICLE_HEADING = re.compile(r"\bArt\.\s*\d+")
_ARTICLE_BOUNDARY = re.compile(r"(?=\bArt\.\s*\d+)")

# Below this many headings a document is prose, not a regulation.
_MIN_ARTICLES = 3

# Shorter than this, a block is a footnote or a stray heading, not a rule.
_MIN_BLOCK_CHARS = 40


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    strategy: ChunkStrategy = ChunkStrategy.CHARACTERS,
    article_max_chars: int = 4000,
) -> list[Document]:
    """Split documents into chunks, by the configured strategy."""
    if chunk_overlap >= chunk_size:
        msg = (
            f"chunk_overlap ({chunk_overlap}) deve ser menor que "
            f"chunk_size ({chunk_size}), senão a quebra não avança."
        )
        raise ValueError(msg)

    if strategy is ChunkStrategy.ARTICLES:
        chunks = _split_by_article(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            article_max_chars=article_max_chars,
        )
    else:
        chunks = _split_by_characters(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    logger.info(
        "Split %d document(s) into %d chunk(s) by %s",
        len(documents),
        len(chunks),
        strategy.value,
    )
    return chunks


def _split_by_characters(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Cut every chunk_size characters, preferring paragraph boundaries.

    The overlap keeps an idea whole in at least one chunk when it falls across
    a boundary.
    """
    return _character_splitter(chunk_size, chunk_overlap).split_documents(documents)


def _split_by_article(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    article_max_chars: int,
) -> list[Document]:
    """Give each article its own chunk, so a rule keeps its own paragraphs.

    Pages are joined back together first. The loader emits one document per
    PDF page, and an article routinely spans a page break -- splitting page by
    page would cut exactly the rules this strategy exists to keep whole.

    A source with no articles falls back to character splitting: the strategy
    is a setting for the whole corpus, and one plain README in the folder must
    not be mangled by it.
    """
    chunks: list[Document] = []

    for document in _join_pages(documents):
        blocks = _article_blocks(document.page_content)

        if len(blocks) < _MIN_ARTICLES:
            chunks.extend(
                _split_by_characters([document], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            )
            continue

        for block in blocks:
            chunks.extend(
                _block_to_chunks(
                    block,
                    document=document,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    article_max_chars=article_max_chars,
                )
            )

    return chunks


def _join_pages(documents: list[Document]) -> list[Document]:
    """Merge every document sharing a source back into one, in order.

    The page number is dropped: it belongs to the page, not to the article,
    and the article heading is the better citation anyway.
    """
    merged: dict[str, list[str]] = {}
    order: list[str] = []

    for document in documents:
        source = str(document.metadata.get("source", ""))
        if source not in merged:
            merged[source] = []
            order.append(source)
        merged[source].append(document.page_content)

    return [
        Document(page_content="\n".join(merged[source]), metadata={"source": source})
        for source in order
    ]


def _article_blocks(text: str) -> list[str]:
    """Cut the text at every article heading, dropping footnote-sized scraps."""
    return [
        block.strip()
        for block in _ARTICLE_BOUNDARY.split(text)
        if len(block.strip()) >= _MIN_BLOCK_CHARS
    ]


def _block_to_chunks(
    block: str,
    *,
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
    article_max_chars: int,
) -> list[Document]:
    """Turn one article into chunks, splitting further only when it is huge.

    Annexes and tables carry no headings, so everything after the last article
    arrives here as a single enormous block. Left whole it would swamp the
    context window on its own.
    """
    metadata = dict(document.metadata)
    heading = _ARTICLE_HEADING.match(block)
    if heading:
        metadata["article"] = heading.group(0)

    if len(block) <= article_max_chars:
        return [Document(page_content=block, metadata=metadata)]

    oversized = Document(page_content=block, metadata=metadata)
    return _character_splitter(chunk_size, chunk_overlap).split_documents([oversized])


def _character_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
    )
