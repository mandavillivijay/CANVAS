from __future__ import annotations

import json
import logging
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_log = logging.getLogger("canvas_heal.resolver")

from canvas_heal import _telemetry
from canvas_heal.backends import IntentStoreBackend, SQLiteBackend, open_backend
from canvas_heal.descriptor import SemanticDescriptor
from canvas_heal.embedder import IntentEmbedder

THRESHOLD_AUTO_HEAL = 0.92

# Default PII regex patterns applied when store_raw_text=False
_PII_PATTERNS: list[re.Pattern] = [
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),  # email
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # phone
]

# Descriptor dict keys whose values may contain raw user text
_TEXT_FIELDS = frozenset({"text_content", "label", "placeholder", "section_heading", "text"})


def _scrub(value: str, patterns: list[re.Pattern]) -> str:
    for p in patterns:
        value = p.sub("[REDACTED]", value)
    return value


def _scrub_descriptor_dict(d: dict, patterns: list[re.Pattern]) -> dict:
    return {
        k: _scrub(v, patterns) if k in _TEXT_FIELDS and isinstance(v, str) else v
        for k, v in d.items()
    }


def _get_git_email() -> str:
    try:
        result = subprocess.run(
            ["git", "config", "user.email"],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip()
    except Exception:
        return ""


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
class IntentVersion:
    id: int
    intent_name: str
    selector: str
    descriptor_text: str
    model_name: str
    page_url: str
    recorded_by: str
    recorded_at: str


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
    """Facade over a storage backend for intent fingerprints.

    Accepts either a legacy ``db_path`` (SQLite) or a pre-built backend
    via the ``_backend`` parameter (or use :func:`open_store` for URL-based
    construction).  PII scrubbing and numpy serialization live here; the
    backend stores raw bytes and JSON strings.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        store_raw_text: bool = True,
        pii_patterns: list[re.Pattern] | None = None,
        team_id: str = "",
        project_id: str = "",
        _backend: IntentStoreBackend | None = None,
    ) -> None:
        self._store_raw_text = store_raw_text
        self._pii_patterns = pii_patterns if pii_patterns is not None else _PII_PATTERNS
        self._recorded_by = _get_git_email()

        if _backend is not None:
            self._backend: IntentStoreBackend = _backend
        else:
            if db_path is None:
                db_path = _DEFAULT_DB
            self._backend = SQLiteBackend(str(db_path), team_id=team_id, project_id=project_id)

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def store(
        self,
        name: str,
        selector: str,
        descriptor: SemanticDescriptor,
        embedding: np.ndarray,
        model_name: str = "",
        page_url: str = "",
    ) -> None:
        desc_dict = descriptor.to_dict()
        stored_url = page_url
        if not self._store_raw_text:
            desc_dict = _scrub_descriptor_dict(desc_dict, self._pii_patterns)
            stored_url = _scrub(page_url, self._pii_patterns)
            _log.debug("stored intent=%r with PII scrubbing applied", name)
        blob = embedding.astype(np.float32).tobytes()
        desc_json = json.dumps(desc_dict)
        self._backend.store(name, selector, desc_json, blob, model_name, stored_url, self._recorded_by)

    def rollback(self, name: str, version_id: int) -> bool:
        """Restore the intent to a previously recorded version. Returns True on success."""
        ok = self._backend.rollback(name, version_id, f"rollback:{self._recorded_by}")
        if ok:
            _log.info("rolled back intent=%r to version_id=%d", name, version_id)
        return ok

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Tuple[str, SemanticDescriptor, np.ndarray, str]]:
        row = self._backend.get(name)
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
        return self._backend.all()

    def count(self) -> int:
        return self._backend.count()

    def get_version_history(self, name: str) -> list[IntentVersion]:
        rows = self._backend.get_version_history(name)
        versions = []
        for r in rows:
            d = json.loads(r["descriptor_json"])
            versions.append(IntentVersion(
                id=r["id"],
                intent_name=r["intent_name"],
                selector=r["selector"],
                descriptor_text=d.get("text", ""),
                model_name=r["model_name"],
                page_url=r["page_url"],
                recorded_by=r["recorded_by"],
                recorded_at=r["recorded_at"],
            ))
        return versions

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> IntentStore:
        return self

    def __exit__(self, *_) -> None:
        self.close()


def open_store(
    url: str,
    store_raw_text: bool = True,
    pii_patterns: list[re.Pattern] | None = None,
    team_id: str = "",
    project_id: str = "",
    **backend_kwargs,
) -> IntentStore:
    """Create an :class:`IntentStore` from a URL.

    Examples::

        store = open_store("sqlite:///:memory:")
        store = open_store("sqlite:///canvas_intents.db", team_id="ci", project_id="my-app")
        store = open_store("postgresql://user:pass@localhost/canvas", team_id="ci")
    """
    backend = open_backend(url, team_id=team_id, project_id=project_id, **backend_kwargs)
    return IntentStore(
        store_raw_text=store_raw_text,
        pii_patterns=pii_patterns,
        _backend=backend,
    )


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
        with _telemetry.span("canvas_heal.record", {
            "canvas.intent_name": name,
            "canvas.selector": selector,
            "canvas.model_name": self._embedder.MODEL_NAME,
        }):
            _log.debug("recording intent=%r selector=%r url=%r", name, selector, page_url)
            embedding = self._embedder.embed_descriptor(descriptor)
            self._store.store(
                name, selector, descriptor, embedding,
                model_name=self._embedder.MODEL_NAME, page_url=page_url,
            )
            _telemetry.record_record(name, self._embedder.MODEL_NAME)
            _log.debug("recorded intent=%r descriptor=%r", name, descriptor.to_text())

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
        with _telemetry.span("canvas_heal.resolve", {
            "canvas.intent_name": name,
            "canvas.candidate_count": len(candidates),
        }) as _span:
            stored = self._store.get(name)
            if stored is None:
                _log.error("FAILED intent=%r reason='no intent stored'", name)
                result = ResolverResult(Resolution.FAILED, None, 0.0, None, f"No intent stored for '{name}'")
                _span.set_attribute("canvas.resolution", result.status.value)
                _span.set_attribute("canvas.confidence", result.confidence)
                self._log_event(name, result, original_selector="", page_url="")
                return result

            original_selector, _, stored_embedding, page_url = stored

            if not candidates:
                _log.error("FAILED intent=%r reason='no candidates provided'", name)
                result = ResolverResult(Resolution.FAILED, None, 0.0, None, "No candidates provided")
                _span.set_attribute("canvas.resolution", result.status.value)
                _span.set_attribute("canvas.confidence", result.confidence)
                self._log_event(name, result, original_selector, page_url)
                return result

            _log.debug("resolving intent=%r candidates=%d", name, len(candidates))

            best_selector: Optional[str] = None
            best_descriptor: Optional[SemanticDescriptor] = None
            best_score = -1.0
            top_scores: list[float] = []

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
                top_scores.append(score)
                if score > best_score:
                    best_score, best_selector, best_descriptor = score, selector, descriptor

            top3 = sorted(top_scores, reverse=True)[:3]
            _log.debug("intent=%r top3_scores=%s best=%.3f", name, [f"{s:.3f}" for s in top3], best_score)

            if best_score >= self.threshold_auto:
                result = ResolverResult(
                    Resolution.HEALED, best_selector, best_score, best_descriptor,
                    f"Auto-healed to '{best_selector}' (confidence {best_score:.3f})",
                )
                _log.info("HEALED intent=%r selector=%r confidence=%.3f", name, best_selector, best_score)
            elif best_score >= self.threshold_confirm:
                result = ResolverResult(
                    Resolution.NEEDS_CONFIRMATION, best_selector, best_score, best_descriptor,
                    f"Needs confirmation: best match '{best_selector}' (confidence {best_score:.3f})",
                )
                _log.warning("NEEDS_CONFIRMATION intent=%r selector=%r confidence=%.3f", name, best_selector, best_score)
            else:
                result = ResolverResult(
                    Resolution.FAILED, None, best_score, best_descriptor,
                    f"No confident match found (best confidence {best_score:.3f})",
                )
                _log.error("FAILED intent=%r best_confidence=%.3f", name, best_score)

            _span.set_attribute("canvas.resolution", result.status.value)
            _span.set_attribute("canvas.confidence", result.confidence)
            if result.selector:
                _span.set_attribute("canvas.resolved_selector", result.selector)
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
        _telemetry.record_resolve(result.status.value, result.confidence, name)

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
