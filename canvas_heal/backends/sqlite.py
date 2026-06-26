from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Tuple

from canvas_heal.backends.base import IntentStoreBackend


class SQLiteBackend(IntentStoreBackend):
    """SQLite storage backend supporting both file-based and :memory: databases."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        team_id: str = "",
        project_id: str = "",
    ) -> None:
        self.team_id = team_id
        self.project_id = project_id
        self._conn = sqlite3.connect(str(db_path))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS intents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                team_id     TEXT    NOT NULL DEFAULT '',
                project_id  TEXT    NOT NULL DEFAULT '',
                selector    TEXT    NOT NULL,
                descriptor  TEXT    NOT NULL,
                embedding   BLOB    NOT NULL,
                model_name  TEXT    NOT NULL DEFAULT '',
                page_url    TEXT    NOT NULL DEFAULT '',
                created_at  TEXT    DEFAULT (datetime('now')),
                UNIQUE(name, team_id, project_id)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS intent_versions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                intent_name TEXT    NOT NULL,
                team_id     TEXT    NOT NULL DEFAULT '',
                project_id  TEXT    NOT NULL DEFAULT '',
                selector    TEXT    NOT NULL,
                descriptor  TEXT    NOT NULL,
                embedding   BLOB    NOT NULL,
                model_name  TEXT    NOT NULL DEFAULT '',
                page_url    TEXT    NOT NULL DEFAULT '',
                recorded_by TEXT    NOT NULL DEFAULT '',
                recorded_at TEXT    DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_versions_ns "
            "ON intent_versions (intent_name, team_id, project_id)"
        )
        self._conn.commit()

        # Migrate existing databases that pre-date team/project columns.
        for table, cols in [
            ("intents", ["team_id TEXT NOT NULL DEFAULT ''", "project_id TEXT NOT NULL DEFAULT ''"]),
            ("intent_versions", ["team_id TEXT NOT NULL DEFAULT ''", "project_id TEXT NOT NULL DEFAULT ''"]),
        ]:
            for col_def in cols:
                try:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
                    self._conn.commit()
                except sqlite3.OperationalError:
                    pass  # column already exists

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
        self._conn.execute(
            "INSERT OR REPLACE INTO intents "
            "(name, team_id, project_id, selector, descriptor, embedding, model_name, page_url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, self.team_id, self.project_id, selector, descriptor_json, embedding_blob, model_name, page_url),
        )
        self._conn.execute(
            "INSERT INTO intent_versions "
            "(intent_name, team_id, project_id, selector, descriptor, embedding, model_name, page_url, recorded_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, self.team_id, self.project_id, selector, descriptor_json, embedding_blob, model_name, page_url, recorded_by),
        )
        self._conn.commit()

    def rollback(self, name: str, version_id: int, recorded_by: str) -> bool:
        row = self._conn.execute(
            "SELECT selector, descriptor, embedding, model_name, page_url "
            "FROM intent_versions WHERE id = ? AND intent_name = ? AND team_id = ? AND project_id = ?",
            (version_id, name, self.team_id, self.project_id),
        ).fetchone()
        if row is None:
            return False
        selector, desc_json, blob, model_name, page_url = row
        self._conn.execute(
            "INSERT OR REPLACE INTO intents "
            "(name, team_id, project_id, selector, descriptor, embedding, model_name, page_url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, self.team_id, self.project_id, selector, desc_json, blob, model_name, page_url),
        )
        self._conn.execute(
            "INSERT INTO intent_versions "
            "(intent_name, team_id, project_id, selector, descriptor, embedding, model_name, page_url, recorded_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, self.team_id, self.project_id, selector, desc_json, blob, model_name, page_url, recorded_by),
        )
        self._conn.commit()
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Tuple[str, str, bytes, str]]:
        row = self._conn.execute(
            "SELECT selector, descriptor, embedding, page_url FROM intents "
            "WHERE name = ? AND team_id = ? AND project_id = ?",
            (name, self.team_id, self.project_id),
        ).fetchone()
        if row is None:
            return None
        selector, descriptor_json, blob, page_url = row
        return selector, descriptor_json, bytes(blob), page_url

    def all(self) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT name, selector FROM intents "
            "WHERE team_id = ? AND project_id = ? ORDER BY name",
            (self.team_id, self.project_id),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM intents WHERE team_id = ? AND project_id = ?",
            (self.team_id, self.project_id),
        ).fetchone()[0]

    def get_version_history(self, name: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, intent_name, selector, descriptor, model_name, page_url, recorded_by, recorded_at "
            "FROM intent_versions "
            "WHERE intent_name = ? AND team_id = ? AND project_id = ? "
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
                "recorded_at": r[7],
            }
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
