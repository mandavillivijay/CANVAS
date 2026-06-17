from __future__ import annotations

import json
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from canvas_heal.descriptor import SemanticDescriptor
from canvas_heal.embedder import IntentEmbedder

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


@dataclass
class HealingEvent:
    intent_name: str
    status: Resolution
    confidence: float
    original_selector: str
    resolved_selector: str | None
    page_url: str
    timestamp: str  # ISO 8601 format


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
                page_url    TEXT    NOT NULL DEFAULT '',
                created_at  TEXT    DEFAULT (datetime('now'))
            )
        """)
        self._conn.commit()
        try:
            self._conn.execute("ALTER TABLE intents ADD COLUMN model_name TEXT NOT NULL DEFAULT ''")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        try:
            self._conn.execute("ALTER TABLE intents ADD COLUMN page_url TEXT NOT NULL DEFAULT ''")
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
        page_url: str = "",
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO intents (name, selector, descriptor, embedding, model_name, page_url) VALUES (?, ?, ?, ?, ?, ?)",
            (name, selector, json.dumps(descriptor.to_dict()), embedding.astype(np.float32).tobytes(), model_name, page_url),
        )
        self._conn.commit()

    def get(self, name: str) -> Optional[Tuple[str, SemanticDescriptor, np.ndarray, str]]:
        row = self._conn.execute(
            "SELECT selector, descriptor, embedding, page_url FROM intents WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        selector, descriptor_json, blob, page_url = row
        d = json.loads(descriptor_json)
        descriptor = SemanticDescriptor(
            tag=d["tag"], role=d["role"], label=d["label"],
            element_type=d["element_type"], placeholder=d["placeholder"],
            text_content=d["text_content"], parent_tag=d["parent_tag"],
            parent_role=d["parent_role"], section_heading=d["section_heading"],
            landmark=d["landmark"],
        )
        return selector, descriptor, np.frombuffer(blob, dtype=np.float32).copy(), page_url

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
        self._audit_log: list[HealingEvent] = []

    def record(
        self,
        name: str,
        selector: str,
        descriptor: SemanticDescriptor,
        page_url: str = "",
    ) -> None:
        """Embed and persist an intent fingerprint under the given name."""
        embedding = self._embedder.embed_descriptor(descriptor)
        self._store.store(
            name, selector, descriptor, embedding,
            model_name=self._embedder.MODEL_NAME, page_url=page_url,
        )

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
        skip_hidden: bool = True,
    ) -> ResolverResult:
        """
        Find the best-matching candidate for a stored intent.

        Returns HEALED if confidence >= threshold_auto,
                NEEDS_CONFIRMATION if >= threshold_confirm,
                FAILED otherwise.

        When skip_hidden is True, candidates that are hidden (is_visible=False)
        or disabled (is_disabled=True) are skipped before scoring.
        """
        stored = self._store.get(name)
        if stored is None:
            result = ResolverResult(Resolution.FAILED, None, 0.0, None, f"No intent stored for '{name}'")
            self._log_event(name, result, original_selector="", page_url="")
            return result

        original_selector, _, stored_embedding, page_url = stored

        if not candidates:
            result = ResolverResult(Resolution.FAILED, None, 0.0, None, "No candidates provided")
            self._log_event(name, result, original_selector, page_url)
            return result

        best_selector: Optional[str] = None
        best_descriptor: Optional[SemanticDescriptor] = None
        best_score = -1.0

        for item in candidates:
            if len(item) == 3:
                selector, descriptor, candidate_embedding = item
            else:
                selector, descriptor = item
                candidate_embedding = None

            if skip_hidden:
                if getattr(descriptor, "is_visible", True) == False:
                    continue
                if getattr(descriptor, "is_disabled", False) == True:
                    continue

            if candidate_embedding is None:
                candidate_embedding = self._embedder.embed_descriptor(descriptor)

            score = IntentEmbedder.cosine_similarity(stored_embedding, candidate_embedding)
            if score > best_score:
                best_score, best_selector, best_descriptor = score, selector, descriptor

        if best_score >= self.threshold_auto:
            result = ResolverResult(
                Resolution.HEALED, best_selector, best_score, best_descriptor,
                f"Auto-healed to '{best_selector}' (confidence {best_score:.3f})",
            )
        elif best_score >= self.threshold_confirm:
            result = ResolverResult(
                Resolution.NEEDS_CONFIRMATION, best_selector, best_score, best_descriptor,
                f"Needs confirmation: best match '{best_selector}' (confidence {best_score:.3f})",
            )
        else:
            result = ResolverResult(
                Resolution.FAILED, None, best_score, best_descriptor,
                f"No confident match found (best confidence {best_score:.3f})",
            )

        self._log_event(name, result, original_selector, page_url)
        return result

    def _log_event(
        self,
        name: str,
        result: ResolverResult,
        original_selector: str,
        page_url: str,
    ) -> None:
        self._audit_log.append(HealingEvent(
            intent_name=name,
            status=result.status,
            confidence=result.confidence,
            original_selector=original_selector,
            resolved_selector=result.selector,
            page_url=page_url,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

    def get_audit_log(self) -> list[HealingEvent]:
        return self._audit_log

    def clear_audit_log(self) -> None:
        self._audit_log = []

    def export_junit_xml(self, path: str | Path, suite_name: str = "canvas-heal") -> None:
        n = len(self._audit_log)
        failures = sum(1 for e in self._audit_log if e.status == Resolution.FAILED)
        testsuite = ET.Element("testsuite", {
            "name": suite_name,
            "tests": str(n),
            "failures": str(failures),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        for event in self._audit_log:
            testcase = ET.SubElement(testsuite, "testcase", {
                "name": event.intent_name,
                "classname": "canvas_heal",
            })
            if event.status == Resolution.NEEDS_CONFIRMATION:
                system_out = ET.SubElement(testcase, "system-out")
                system_out.text = (
                    f"Needs confirmation: confidence={event.confidence:.3f}, "
                    f"resolved to {event.resolved_selector}"
                )
            elif event.status == Resolution.FAILED:
                failure = ET.SubElement(testcase, "failure", {
                    "message": f"{event.intent_name} could not be resolved",
                    "type": "CanvasHealFailure",
                })
                failure.text = f"confidence={event.confidence:.3f}"

        tree = ET.ElementTree(testsuite)
        tree.write(path, encoding="unicode", xml_declaration=True)
