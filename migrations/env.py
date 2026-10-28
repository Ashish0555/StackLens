"""Alembic's entry point, pointed at this project's settings.

The URL is not written in alembic.ini. It comes from `DATABASE_URL` through the
same settings object the application reads, so there is one place that knows
where the database is and no way for the two to disagree.

There is no `target_metadata` and no autogenerate. The tables belong to
langchain-postgres, which creates them itself; what this project owns is the
extensions and the text search configuration, and those are written by hand
because autogenerate cannot see them.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from stacklens.config import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = None


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it.

    This is what produces a script for a DBA to review, which is how a change
    reaches a database nobody is allowed to connect to from a laptop.
    """
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and run the migrations."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
