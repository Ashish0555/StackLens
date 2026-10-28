"""The vector store: persisting chunks as vectors and searching them.

Postgres with pgvector holds the embeddings, and the same rows answer keyword
search through the full text index. One database means the vector and the
metadata are written in the same transaction and cannot drift apart, and it
means the operational burden is a database the organisation already runs.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy import text

from stacklens.config import SearchStrategy, get_settings
from stacklens.indexing.database import (
    COLLECTION_TABLE,
    EMBEDDING_TABLE,
    DatabaseUnavailableError,
    describe_database,
    ensure_search_indexes,
    get_engine,
    verify_connection,
    verify_schema,
)
from stacklens.indexing.hybrid import FUSION_POOL, fuse, identity
from stacklens.indexing.keyword import keyword_search
from stacklens.indexing.reranker import RerankerUnavailableError, get_reranker, reranking_enabled
from stacklens.indexing.scope import current_collection
from stacklens.providers import build_embeddings
from stacklens.types import SearchHit

logger = logging.getLogger(__name__)

__all__ = [
    "DatabaseUnavailableError",
    "IndexedSource",
    "RerankerUnavailableError",
    "count_documents",
    "delete_source",
    "describe_location",
    "get_vector_store",
    "index_documents",
    "list_sources",
    "reset_index",
    "search",
]


@dataclass(frozen=True, slots=True)
class IndexedSource:
    """One source document, and how much of the index it occupies."""

    name: str
    chunks: int


_COUNT = text(
    f"""
    SELECT count(*)
    FROM {EMBEDDING_TABLE} AS embedding
    JOIN {COLLECTION_TABLE} AS collection
      ON embedding.collection_id = collection.uuid
    WHERE collection.name = :collection
    """
)

# The source is the file the chunk came from, written into the metadata by the
# loader. Grouping by it is what turns "590 chunks" into something a person can
# act on.
_LIST_SOURCES = text(
    f"""
    SELECT embedding.cmetadata ->> 'source' AS source, count(*) AS chunks
    FROM {EMBEDDING_TABLE} AS embedding
    JOIN {COLLECTION_TABLE} AS collection
      ON embedding.collection_id = collection.uuid
    WHERE collection.name = :collection
    GROUP BY source
    ORDER BY source
    """
)

_DELETE_SOURCE = text(
    f"""
    DELETE FROM {EMBEDDING_TABLE} AS embedding
    USING {COLLECTION_TABLE} AS collection
    WHERE embedding.collection_id = collection.uuid
      AND collection.name = :collection
      AND embedding.cmetadata ->> 'source' = :source
    """
)


def get_vector_store() -> PGVector:
    """Open the vector store described by the current settings.

    The connection and the schema are checked on the way in, so a database
    that is away or a migration that was never applied fails with a message
    naming the command instead of a driver error thrown from somewhere deeper.
    """
    settings = get_settings()

    verify_connection()
    verify_schema()

    return PGVector(
        embeddings=build_embeddings(),
        connection=get_engine(),
        collection_name=current_collection(),
        # Declaring the width turns the column into a fixed-size vector, which
        # is what an approximate index can be built on. Left undeclared, the
        # column is unconstrained and every search reads every row.
        embedding_length=settings.embedding_dimensions,
        use_jsonb=True,
        create_extension=False,
    )


def index_documents(chunks: list[Document]) -> int:
    """Embed the chunks and store them. Returns how many were indexed.

    The chunks are scanned for hidden instructions on the way in. A retrieved
    passage is read the way the system prompt is read, so a document carrying
    an instruction attacks every question that retrieves it. Scanning here
    rather than at query time is the whole saving: the corpus cannot change
    between two questions, so the answer cannot either.
    """
    if not chunks:
        logger.warning("Nothing to index.")
        return 0

    _warn_about_injection(chunks)

    store = get_vector_store()
    store.add_documents(documents=chunks, ids=[_stable_id(chunk) for chunk in chunks])

    # The tables exist only once something has been written to them, so this
    # is the first moment the indexes can be built.
    ensure_search_indexes()

    logger.info("Indexed %d chunk(s) in %s", len(chunks), describe_location())
    return len(chunks)


def search(query: str, k: int | None = None) -> list[SearchHit]:
    """Return the k chunks most relevant to the query.

    Retrieval and reranking answer different questions. Retrieval decides what
    is in the pool and is judged on whether the answer is there at all.
    Reranking decides what comes out of it and is judged on whether the best
    of them is on top. With no reranker the pool is the answer, so it is
    retrieved at exactly the width the caller asked for.

    In `vector` mode the ranking is nearest-neighbour on the embedding, and
    the hits carry their distance. In `hybrid` mode a keyword ranking is fused
    with it, and the distance stops being meaningful for the fused list: two
    rankings are merged by position, not by score, because a cosine distance
    and a text search rank are not on the same scale.
    """
    settings = get_settings()
    limit = settings.retrieval_k if k is None else k

    # A reranker only earns its latency when it is handed more than it
    # returns. Without one, retrieving wider would be work thrown away.
    wanted = max(settings.rerank_candidates, limit) if reranking_enabled() else limit

    candidates, distances = _retrieve(query, wanted)

    return [
        SearchHit(document=document, distance=distances.get(identity(document), _worst(distances)))
        for document in get_reranker().rerank(query, candidates, limit)
    ]


def _retrieve(query: str, wanted: int) -> tuple[list[Document], dict[str, float]]:
    """The candidate pool, and whatever distances are known about it."""
    settings = get_settings()

    if settings.search_strategy is SearchStrategy.VECTOR:
        scored = get_vector_store().similarity_search_with_score(query, k=wanted)
        return [document for document, _ in scored], _distances(scored)

    # Each retriever is asked for several times what the fusion returns. Two
    # short lists only agree on what either would have found alone; the
    # passages worth adding sit deeper in one of them.
    pool = wanted * FUSION_POOL
    scored = get_vector_store().similarity_search_with_score(query, k=pool)
    distances = _distances(scored)

    by_vector = [document for document, _ in scored]
    by_keyword = keyword_search(query, pool)

    if not by_keyword:
        return by_vector[:wanted], distances

    return fuse([by_vector, by_keyword], limit=wanted), distances


def _distances(scored: list[tuple[Document, float]]) -> dict[str, float]:
    return {identity(document): distance for document, distance in scored}


def _worst(distances: dict[str, float]) -> float:
    """What to report for a hit the vector search never saw.

    A document the keyword search contributed has no distance of its own.
    Reporting the worst seen keeps the field honest rather than inventing a
    number that would read as a similarity.
    """
    return max(distances.values(), default=0.0)


def count_documents() -> int:
    """How many chunks are currently indexed."""
    verify_connection()

    with get_engine().connect() as connection:
        try:
            total = connection.execute(_COUNT, {"collection": current_collection()}).scalar()
        except Exception:
            # Nothing has been indexed yet, so langchain-postgres has not
            # created its tables. An empty index is the honest answer.
            return 0

    return int(total or 0)


def list_sources() -> list[IndexedSource]:
    """Every source document currently indexed, with its chunk count.

    Removing a document requires naming it exactly, and the name is the file
    name the loader recorded, not the path it was read from.
    """
    verify_connection()

    with get_engine().connect() as connection:
        try:
            rows = connection.execute(_LIST_SOURCES, {"collection": current_collection()}).all()
        except Exception:
            return []

    return [IndexedSource(name=row[0] or "", chunks=row[1]) for row in rows]


def delete_source(source: str) -> int:
    """Remove every chunk that came from one document. Returns how many.

    Re-ingesting overwrites a chunk whose text is unchanged, but it cannot
    remove one that no longer exists: a revoked document, or a paragraph
    deleted from a revision, would otherwise stay in the index and keep being
    retrieved against questions it no longer answers.
    """
    verify_connection()

    with get_engine().begin() as connection:
        result = connection.execute(
            _DELETE_SOURCE,
            {"collection": current_collection(), "source": source},
        )

    removed = result.rowcount or 0
    logger.info("Removed %d chunk(s) of %s from %s", removed, source, describe_location())

    return removed


def reset_index() -> None:
    """Drop every indexed chunk.

    Ingestion is idempotent for identical chunks, but changing the chunking
    strategy produces different text and therefore different ids. Without a
    way to clear the collection, the old chunks stay behind and compete for
    retrieval against the new ones.
    """
    get_vector_store().delete_collection()
    logger.info("Cleared the index at %s", describe_location())


def describe_location() -> str:
    """Where the index lives, for diagnostics and log messages."""
    return describe_database()


def _warn_about_injection(chunks: list[Document]) -> None:
    """Report chunks that look like instructions rather than like documents.

    A warning and not a refusal. This corpus is regulation, and regulation
    tells the reader what to do, so a classifier trained on jailbreaks will
    sometimes read a genuine article as an instruction. Refusing to index
    would silently drop the law; naming the passage lets a person look.
    """
    from stacklens.guardrails import scan_chunks

    flagged = scan_chunks([chunk.page_content for chunk in chunks])
    if not flagged:
        return

    logger.warning(
        "%d of %d chunk(s) look like instructions. Review them before trusting the index.",
        len(flagged),
        len(chunks),
    )
    for index, verdict in flagged.items():
        source = chunks[index].metadata.get("source", "desconhecida")
        excerpt = chunks[index].page_content[:80].replace("\n", " ")
        logger.warning("  %s (%s, %.2f): %s", source, verdict.label, verdict.score, excerpt)


def _stable_id(chunk: Document) -> str:
    """Derive a deterministic id from the collection, source and content.

    This is what makes ingestion idempotent: running it twice overwrites the
    same records instead of duplicating the whole index, and it is what makes
    ingestion safe to retry from a queue where the same message can be
    delivered more than once.

    **The collection is part of the hash, and has to be.** The id is the
    primary key of a table every collection shares, so an id derived from the
    content alone means the same chunk cannot exist in two collections:
    writing it into the second silently moves the row out of the first, and
    the second reports nothing was written. Invisible with one collection,
    and wrong the moment there are two.
    """
    raw = f"{current_collection()}::{chunk.metadata.get('source', '')}::{chunk.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
