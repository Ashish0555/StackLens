"""Keyword search, answered by Postgres full text search.

An embedding compares meaning, which is what lets "quanto custa o plano mais
barato" find a passage that says neither word. It also spreads a long article's
signal across everything that article discusses, so one sentence stating a
deadline sits below whatever the article is mostly about.

Keyword search compares words. It cannot follow a paraphrase, and it does not
need to when the question names the terms the text uses: article numbers,
paragraph marks and deadlines have identity rather than meaning, and an
embedding blurs exactly those.

Postgres answers this against the same rows that hold the vectors, using a GIN
index. The alternative, a BM25 index built in the application's memory from
every stored chunk, is a copy that has to be rebuilt on every write and cannot
outgrow one process.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from sqlalchemy import text

from stacklens.indexing.database import (
    COLLECTION_TABLE,
    EMBEDDING_TABLE,
    TEXT_SEARCH_CONFIG,
    get_engine,
)
from stacklens.indexing.hybrid import tokenise
from stacklens.indexing.scope import current_collection

logger = logging.getLogger(__name__)

# The configuration name is interpolated rather than bound, so the expression
# matches the one the GIN index was built on. It is a module constant, never
# user input.
_SEARCH = text(
    f"""
    SELECT embedding.document, embedding.cmetadata
    FROM {EMBEDDING_TABLE} AS embedding
    JOIN {COLLECTION_TABLE} AS collection
      ON embedding.collection_id = collection.uuid
    WHERE collection.name = :collection
      AND to_tsvector('{TEXT_SEARCH_CONFIG}', embedding.document)
          @@ to_tsquery('{TEXT_SEARCH_CONFIG}', :terms)
    ORDER BY ts_rank_cd(
      to_tsvector('{TEXT_SEARCH_CONFIG}', embedding.document),
      to_tsquery('{TEXT_SEARCH_CONFIG}', :terms)
    ) DESC
    LIMIT :limit
    """
)


def keyword_search(query: str, limit: int) -> list[Document]:
    """The stored chunks that share the most words with the query.

    Terms are joined with OR rather than AND. A question is a sentence and a
    passage is a fragment, so requiring every word to appear would return
    nothing for most real questions; ranking decides which partial matches
    are worth reading.
    """
    terms = build_terms(query)
    if not terms:
        return []

    with get_engine().connect() as connection:
        rows = connection.execute(
            _SEARCH,
            {
                "collection": current_collection(),
                "terms": terms,
                "limit": limit,
            },
        ).all()

    return [Document(page_content=row[0], metadata=dict(row[1] or {})) for row in rows]


def build_terms(query: str) -> str:
    """Turn a question into a tsquery expression.

    Tokenising here rather than handing the raw sentence to `plainto_tsquery`
    is what allows the OR. It also means the input to the query parser is
    already reduced to words and digits, so none of the operators tsquery
    understands can reach it.
    """
    return " | ".join(tokenise(query))
