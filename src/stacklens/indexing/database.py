"""The Postgres connection and the schema the retrieval layer depends on.

Two things live here that the rest of the package assumes are already true:
the extensions, and the text search configuration used by keyword search.

Postgres ships a `portuguese` configuration that stems but does not fold
accents, so "suspensão" and "suspensao" stem to different words and a question
written without the accent finds nothing. Folding them is a matter of putting
`unaccent` ahead of the stemmer in a configuration of our own, which is the
standard recipe and the reason this module exists at all.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text

from stacklens.config import get_settings

logger = logging.getLogger(__name__)

# The text search configuration built below. Both the stored column and the
# query must be parsed with the same one, or the stems will not match.
TEXT_SEARCH_CONFIG = "stacklens_english"

# langchain-postgres owns these tables. Naming them here keeps every raw
# statement in one file, so an upgrade that renames them breaks in one place.
EMBEDDING_TABLE = "langchain_pg_embedding"
COLLECTION_TABLE = "langchain_pg_collection"

_FTS_INDEX = "stacklens_document_fts"
_VECTOR_INDEX = "stacklens_embedding_hnsw"


class DatabaseUnavailableError(RuntimeError):
    """The configured Postgres could not be reached."""


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """The connection pool, opened once per process.

    `pool_pre_ping` costs one round trip per checkout and buys immunity to the
    connection a database restart or an idle timeout left dead in the pool.
    """
    settings = get_settings()

    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        # Fail fast rather than waiting on the operating system. An address
        # that never answers takes over two minutes to give up on by default,
        # which is longer than any caller is willing to wait and long enough
        # for a readiness probe to hang instead of reporting not ready.
        connect_args={"connect_timeout": settings.database_connect_timeout},
    )


def forget_engine() -> None:
    """Dispose of the pool, so the next caller opens a new one."""
    if get_engine.cache_info().currsize:
        get_engine().dispose()
    get_engine.cache_clear()


def verify_connection() -> None:
    """Fail early, and name the knob to turn.

    The driver's own error quotes a host and a port and says nothing about
    which command brings that host up.
    """
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as error:
        msg = (
            f"Não foi possível conectar ao Postgres em {describe_database()}. "
            f"Suba o banco (docker compose up -d postgres) ou ajuste DATABASE_URL."
        )
        raise DatabaseUnavailableError(msg) from error


class SchemaOutOfDateError(RuntimeError):
    """The database is reachable but has not had the migrations applied."""


def verify_schema() -> None:
    """Fail early when the migrations have not been applied, and name the command.

    Applying migrations from inside the application was the previous design and
    it is an anti-pattern at more than one replica: several processes starting
    together race to create the same objects, and a long migration blocks every
    boot rather than one deployment step. Alembic owns the schema now, and this
    only checks that it ran.

    The text search configuration is the marker because it is the object this
    project owns that nothing else creates.
    """
    with get_engine().connect() as connection:
        exists = connection.execute(
            text("SELECT 1 FROM pg_ts_config WHERE cfgname = :name"),
            {"name": TEXT_SEARCH_CONFIG},
        ).first()

    if exists:
        return

    msg = (
        f"O banco em {describe_database()} não tem as migrações aplicadas. "
        f"Rode: alembic upgrade head"
    )
    raise SchemaOutOfDateError(msg)


def ensure_search_indexes() -> None:
    """Build the indexes both halves of the search depend on.

    Neither can exist before langchain-postgres has created its tables, which
    is why this runs after the first write rather than at startup.

    The GIN index is what keeps keyword search from parsing every stored
    document on every query. The HNSW index is the approximate nearest
    neighbour index for the vectors; without it pgvector is exact, which is
    correct and reads every row. Both are the difference between a corpus of
    hundreds and one of millions.
    """
    statements = (
        f"CREATE INDEX IF NOT EXISTS {_FTS_INDEX} ON {EMBEDDING_TABLE} "
        f"USING GIN (to_tsvector('{TEXT_SEARCH_CONFIG}', document))",
        f"CREATE INDEX IF NOT EXISTS {_VECTOR_INDEX} ON {EMBEDDING_TABLE} "
        f"USING HNSW (embedding vector_cosine_ops)",
    )

    with get_engine().begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def describe_database() -> str:
    """The database location with the password removed, for logs and status."""
    from sqlalchemy.engine import make_url

    return make_url(get_settings().database_url).render_as_string(hide_password=True)
