from __future__ import annotations

import logging
from typing import Optional, Tuple

from canvas_heal.backends.base import IntentStoreBackend

_log = logging.getLogger("canvas_heal.backends.postgres")

_DDL_INTENTS = """
CREATE TABLE IF NOT EXISTS intents (
    id          SERIAL PRIMARY KEY,
    name        TEXT    NOT NULL,
    team_id     TEXT    NOT NULL DEFAULT '',
    project_id  TEXT    NOT NULL DEFAULT '',
    selector    TEXT    NOT NULL,
    descriptor  TEXT    NOT NULL,
    embedding   BYTEA   NOT NULL,
    model_name  TEXT    NOT NULL DEFAULT '',
    page_url    TEXT    NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, team_id, project_id)
)
"""

_DDL_VERSIONS = """
CREATE TABLE IF NOT EXISTS intent_versions (
    id          SERIAL PRIMARY KEY,
    intent_name TEXT    NOT NULL,
    team_id     TEXT    NOT NULL DEFAULT '',
    project_id  TEXT    NOT NULL DEFAULT '',
    selector    TEXT    NOT NULL,
    descriptor  TEXT    NOT NULL,
    embedding   BYTEA   NOT NULL,
    model_name  TEXT    NOT NULL DEFAULT '',
    page_url    TEXT    NOT NULL DEFAULT '',
    recorded_by TEXT    NOT NULL DEFAULT '',
    recorded_at TIMESTAMPTZ DEFAULT NOW()
)
"""

_DDL_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_versions_ns "
    "ON intent_versions (intent_name, team_id, project_id)"
)


class PostgreSQLBackend(IntentStoreBackend):
    """PostgreSQL backend with connection pooling via psycopg3.

    Install the optional extra to use this backend::

        pip install canvas-heal[postgres]

    The ``dsn`` argument accepts any libpq connection string or URI, e.g.::

        postgresql://user:pass@host:5432/mydb
    """

    def __init__(
        self,
        dsn: str,
        team_id: str = "",
        project_id: str = "",
        min_pool: int = 1,
        max_pool: int = 10,
    ) -> None:
        try:
            from psycopg_pool import ConnectionPool  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "psycopg[pool] is required for the PostgreSQL backend. "
                "Install it with: pip install canvas-heal[postgres]"
            ) from exc

        self.team_id = team_id
        self.project_id = project_id
        self._pool = ConnectionPool(dsn, min_size=min_pool, max_size=max_pool)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(_DDL_INTENTS)
            conn.execute(_DDL_VERSIONS)
            conn.execute(_DDL_INDEX)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(
        self,
        name: str,
        selector: str,
        descriptor_json: str,
        embedding_blob: bytes,
        model_name: str,
        page_url: str,
        recorded_by: str,
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO intents (name, team_id, project_id, selector, descriptor, embedding, model_name, page_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name, team_id, project_id) DO UPDATE SET
                    selector   = EXCLUDED.selector,
                    descriptor = EXCLUDED.descriptor,
                    embedding  = EXCLUDED.embedding,
                    model_name = EXCLUDED.model_name,
                    page_url   = EXCLUDED.page_url
                """,
                (name, self.team_id, self.project_id, selector, descriptor_json,
                 embedding_blob, model_name, page_url),
            )
            conn.execute(
                """
                INSERT INTO intent_versions
                    (intent_name, team_id, project_id, selector, descriptor, embedding, model_name, page_url, recorded_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (name, self.team_id, self.project_id, selector, descriptor_json,
                 embedding_blob, model_name, page_url, recorded_by),
            )

    def rollback(self, name: str, version_id: int, recorded_by: str) -> bool:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT selector, descriptor, embedding, model_name, page_url "
                "FROM intent_versions WHERE id = %s AND intent_name = %s AND team_id = %s AND project_id = %s",
                (version_id, name, self.team_id, self.project_id),
            ).fetchone()
            if row is None:
                return False
            selector, desc_json, blob, model_name, page_url = row
            conn.execute(
                """
                INSERT INTO intents (name, team_id, project_id, selector, descriptor, embedding, model_name, page_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name, team_id, project_id) DO UPDATE SET
                    selector   = EXCLUDED.selector,
                    descriptor = EXCLUDED.descriptor,
                    embedding  = EXCLUDED.embedding,
                    model_name = EXCLUDED.model_name,
                    page_url   = EXCLUDED.page_url
                """,
                (name, self.team_id, self.project_id, selector, desc_json, blob, model_name, page_url),
            )
            conn.execute(
                """
                INSERT INTO intent_versions
                    (intent_name, team_id, project_id, selector, descriptor, embedding, model_name, page_url, recorded_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (name, self.team_id, self.project_id, selector, desc_json, blob, model_name, page_url, recorded_by),
            )
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Tuple[str, str, bytes, str]]:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT selector, descriptor, embedding, page_url FROM intents "
                "WHERE name = %s AND team_id = %s AND project_id = %s",
                (name, self.team_id, self.project_id),
            ).fetchone()
        if row is None:
            return None
        selector, descriptor_json, blob, page_url = row
        return selector, descriptor_json, bytes(blob), page_url

    def all(self) -> list[tuple[str, str]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT name, selector FROM intents "
                "WHERE team_id = %s AND project_id = %s ORDER BY name",
                (self.team_id, self.project_id),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def count(self) -> int:
        with self._pool.connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM intents WHERE team_id = %s AND project_id = %s",
                (self.team_id, self.project_id),
            ).fetchone()[0]

    def get_version_history(self, name: str) -> list[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, intent_name, selector, descriptor, model_name, page_url, recorded_by, recorded_at "
                "FROM intent_versions "
                "WHERE intent_name = %s AND team_id = %s AND project_id = %s "
                "ORDER BY recorded_at DESC, id DESC",
                (name, self.team_id, self.project_id),
            ).fetchall()
        return [
            {
                "id": r[0],
                "intent_name": r[1],
                "selector": r[2],
                "descriptor_json": r[3],
                "model_name": r[4],
                "page_url": r[5],
                "recorded_by": r[6],
                "recorded_at": str(r[7]),
            }
            for r in rows
        ]

    def close(self) -> None:
        self._pool.close()
