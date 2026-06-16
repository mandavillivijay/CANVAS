from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from canvas.descriptor import SemanticDescriptor
from canvas.embedder import IntentEmbedder

THRESHOLD_AUTO_HEAL = 0.92
THRESHOLD_CONFIRM = 0.75


class Resolution(Enum):
    HEALED = "healed"
    NEEDS_CONFIRMATION = "needs_confirmation"
    FAILED = "failed"


@dataclass
class ResolverResult:
    status: Resolution
    selector: Optional[str]
    confidence: float
    descriptor: Optional[SemanticDescriptor]
    message: str


_DEFAULT_DB = Path(__file__).parent.parent / "canvas_intents.db"


class IntentStore:
    """SQLite-backed store for intent fingerprints (descriptor + embedding)."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = _DEFAULT_DB
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS intents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                selector    TEXT    NOT NULL,
                descriptor  TEXT    NOT NULL,
                embedding   BLOB    NOT NULL,
                model_name  TEXT    NOT NULL DEFAULT '',
                created_at  TEXT    DEFAULT (datetime('now'))
            )
        """)
        self._conn.commit()
        try:
            self._conn.execute("ALTER TABLE intents ADD COLUMN model_name TEXT NOT NULL DEFAULT ''")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    def store(
        self,
        name: str,
        selector: str,
        descriptor: SemanticDescriptor,
        embedding: np.ndarray,
        model_name: str = "",
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO intents (name, selector, descriptor, embedding, model_name) VALUES (?, ?, ?, ?, ?)",
            (name, selector, json.dumps(descriptor.to_dict()), embedding.astype(np.float32).tobytes(), model_name),
        )
        self._conn.commit()

    def get(self, name: str) -> Optional[Tuple[str, SemanticDescriptor, np.ndarray]]:
        row = self._conn.execute(
            "SELECT selector, descriptor, embedding FROM intents WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        selector, descriptor_json, blob = row
        d = json.loads(descriptor_json)
        descriptor = SemanticDescriptor(
            tag=d["tag"], role=d["role"], label=d["label"],
            element_type=d["element_type"], placeholder=d["placeholder"],
            text_content=d["text_content"], parent_tag=d["parent_tag"],
            parent_role=d["parent_role"], section_heading=d["section_heading"],
            landmark=d["landmark"],
        )
        return selector, descriptor, np.frombuffer(blob, dtype=np.float32).copy()

    def all(self) -> list[tuple[str, str]]:
        """Return list of (name, selector) for all stored intents."""
        rows = self._conn.execute("SELECT name, selector FROM intents ORDER BY name").fetchall()
        return [(row[0], row[1]) for row in rows]

    def count(self) -> int:
        """Return the number of stored intents."""
        return self._conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> IntentStore:
        return self

    def __exit__(self, *_) -> None:
        self.close()


class ConfidenceGatedResolver:
    """
    At record time: embed a SemanticDescriptor and persist it in the IntentStore.
    At resolve time: embed all candidates, find the nearest match, and gate the
    decision into HEALED / NEEDS_CONFIRMATION / FAILED based on cosine similarity.
    """

    def __init__(
        self,
        store: IntentStore,
        embedder: Optional[IntentEmbedder] = None,
        threshold_auto: float = THRESHOLD_AUTO_HEAL,
        threshold_confirm: float = THRESHOLD_CONFIRM,
    ) -> None:
        self._store = store
        self._embedder = embedder or IntentEmbedder.get()
        self.threshold_auto = threshold_auto
        self.threshold_confirm = threshold_confirm

    def record(self, name: str, selector: str, descriptor: SemanticDescriptor) -> None:
        """Embed and persist an intent fingerprint under the given name."""
        embedding = self._embedder.embed_descriptor(descriptor)
        self._store.store(name, selector, descriptor, embedding, model_name=self._embedder.MODEL_NAME)

    def precompute_candidates(
        self, candidates: list[tuple[str, "SemanticDescriptor"]]
    ) -> list[tuple[str, "SemanticDescriptor", np.ndarray]]:
        """Pre-embed a candidate list once, to reuse across multiple resolve() calls."""
        descriptors = [desc for _, desc in candidates]
        embeddings = self._embedder.batch_embed_descriptors(descriptors)
        return [(sel, desc, emb) for (sel, desc), emb in zip(candidates, embeddings)]

    def resolve(
        self,
        name: str,
        candidates: List[Tuple[str, SemanticDescriptor]],
    ) -> ResolverResult:
        """
        Find the best-matching candidate for a stored intent.

        Returns HEALED if confidence >= threshold_auto,
                NEEDS_CONFIRMATION if >= threshold_confirm,
                FAILED otherwise.
        """
        stored = self._store.get(name)
        if stored is None:
            return ResolverResult(Resolution.FAILED, None, 0.0, None, f"No intent stored for '{name}'")

        if not candidates:
            return ResolverResult(Resolution.FAILED, None, 0.0, None, "No candidates provided")

        _, _, stored_embedding = stored
        best_selector: Optional[str] = None
        best_descriptor: Optional[SemanticDescriptor] = None
        best_score = -1.0

        for item in candidates:
            if len(item) == 3:
                selector, descriptor, candidate_embedding = item
            else:
                selector, descriptor = item
                candidate_embedding = self._embedder.embed_descriptor(descriptor)

            score = IntentEmbedder.cosine_similarity(stored_embedding, candidate_embedding)
            if score > best_score:
                best_score, best_selector, best_descriptor = score, selector, descriptor

        if best_score >= self.threshold_auto:
            return ResolverResult(
                Resolution.HEALED, best_selector, best_score, best_descriptor,
                f"Auto-healed to '{best_selector}' (confidence {best_score:.3f})",
            )
        if best_score >= self.threshold_confirm:
            return ResolverResult(
                Resolution.NEEDS_CONFIRMATION, best_selector, best_score, best_descriptor,
                f"Needs confirmation: best match '{best_selector}' (confidence {best_score:.3f})",
            )
        return ResolverResult(
            Resolution.FAILED, None, best_score, best_descriptor,
            f"No confident match found (best confidence {best_score:.3f})",
        )
