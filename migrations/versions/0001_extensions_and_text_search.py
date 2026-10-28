"""Extensions and the Portuguese text search configuration.

What this project owns in the database. The tables belong to
langchain-postgres, which creates them on first write, and the two search
indexes are built after that because they cannot exist before the table does.

Postgres ships a `portuguese` configuration that stems but does not fold
accents, so "suspensão" stems to `suspensã` while "suspensao" stems to
`suspensa`, and a question written without the accent finds nothing. Measured
against a running database, that is the difference between the keyword half of
the search working and not working at all.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TEXT_SEARCH_CONFIG = "stacklens_english"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")

    # CREATE TEXT SEARCH CONFIGURATION has no IF NOT EXISTS, so existence is
    # checked rather than the duplicate-object error swallowed. A migration
    # that cannot be re-run is a migration that fails a retry.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_ts_config WHERE cfgname = '{TEXT_SEARCH_CONFIG}'
            ) THEN
                CREATE TEXT SEARCH CONFIGURATION {TEXT_SEARCH_CONFIG} (COPY = english);
                ALTER TEXT SEARCH CONFIGURATION {TEXT_SEARCH_CONFIG}
                    ALTER MAPPING FOR hword, hword_part, word
                    WITH unaccent, english_stem;
            END IF;
        END
        $$
        """
    )


def downgrade() -> None:
    """Drop the configuration, and leave the extensions alone.

    `vector` and `unaccent` are database-wide and something else may be using
    them. Dropping an extension to undo one application's migration is how a
    rollback takes down a neighbour.
    """
    op.execute(f"DROP TEXT SEARCH CONFIGURATION IF EXISTS {TEXT_SEARCH_CONFIG}")
