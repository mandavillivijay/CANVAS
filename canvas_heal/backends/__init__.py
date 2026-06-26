from __future__ import annotations

from canvas_heal.backends.base import IntentStoreBackend
from canvas_heal.backends.sqlite import SQLiteBackend

__all__ = ["IntentStoreBackend", "SQLiteBackend", "open_backend"]


def open_backend(
    url: str,
    team_id: str = "",
    project_id: str = "",
    **kwargs,
) -> IntentStoreBackend:
    """Create the appropriate backend from a store URL.

    Supported schemes:
    - ``sqlite:///path/to/file.db``
    - ``sqlite:///:memory:``
    - ``postgresql://user:pass@host:5432/dbname``

    Extra keyword arguments are forwarded to the backend constructor
    (e.g. ``min_pool``/``max_pool`` for PostgreSQL).
    """
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///"):]
        if not path:
            path = ":memory:"
        return SQLiteBackend(path, team_id=team_id, project_id=project_id)

    if url.startswith("postgresql://") or url.startswith("postgres://"):
        from canvas_heal.backends.postgres import PostgreSQLBackend  # lazy import
        return PostgreSQLBackend(url, team_id=team_id, project_id=project_id, **kwargs)

    raise ValueError(
        f"Unsupported store URL: {url!r}. "
        "Use 'sqlite:///path' or 'postgresql://user:pass@host/db'."
    )
