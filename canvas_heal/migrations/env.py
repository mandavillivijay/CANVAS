"""Alembic migration environment for canvas-heal PostgreSQL backend.

Usage::

    export CANVAS_STORE_URL=postgresql://user:pass@localhost/canvas
    alembic upgrade head

The ``CANVAS_STORE_URL`` environment variable must be a PostgreSQL URL.
SQLite databases are self-migrating and do not use Alembic.
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

_STORE_URL = os.environ.get("CANVAS_STORE_URL", "")


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to the database."""
    if not _STORE_URL:
        raise RuntimeError("CANVAS_STORE_URL environment variable is not set.")
    context.configure(
        url=_STORE_URL,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live database connection."""
    if not _STORE_URL:
        raise RuntimeError("CANVAS_STORE_URL environment variable is not set.")
    try:
        from sqlalchemy import engine_from_config, pool
    except ImportError as exc:
        raise ImportError(
            "sqlalchemy is required to run online Alembic migrations. "
            "Install it with: pip install canvas-heal[postgres]"
        ) from exc

    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _STORE_URL

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
