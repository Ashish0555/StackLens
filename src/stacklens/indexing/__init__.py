"""Offline pipeline: load documents, split them, store them as vectors."""

from stacklens.indexing.database import (
    DatabaseUnavailableError,
    SchemaOutOfDateError,
    describe_database,
    ensure_search_indexes,
    forget_engine,
    get_engine,
    verify_connection,
    verify_schema,
)
from stacklens.indexing.hybrid import fuse, tokenise
from stacklens.indexing.keyword import keyword_search
from stacklens.indexing.loader import SUPPORTED_SUFFIXES, load_documents
from stacklens.indexing.reranker import (
    Reranker,
    RerankerUnavailableError,
    forget_reranker,
    get_reranker,
    reranking_enabled,
)
from stacklens.indexing.splitter import split_documents
from stacklens.indexing.vector_store import (
    IndexedSource,
    count_documents,
    delete_source,
    describe_location,
    get_vector_store,
    index_documents,
    list_sources,
    reset_index,
    search,
)

__all__ = [
    "SUPPORTED_SUFFIXES",
    "DatabaseUnavailableError",
    "IndexedSource",
    "Reranker",
    "RerankerUnavailableError",
    "SchemaOutOfDateError",
    "count_documents",
    "delete_source",
    "describe_database",
    "describe_location",
    "ensure_search_indexes",
    "forget_engine",
    "forget_reranker",
    "fuse",
    "get_engine",
    "get_reranker",
    "get_vector_store",
    "index_documents",
    "keyword_search",
    "list_sources",
    "load_documents",
    "reranking_enabled",
    "reset_index",
    "search",
    "split_documents",
    "tokenise",
    "verify_connection",
    "verify_schema",
]
