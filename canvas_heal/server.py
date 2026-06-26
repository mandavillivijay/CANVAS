"""CANVAS-HEAL REST service.

Runs a shared embedding/resolve server so CI agents can skip loading the
90 MB model locally.  Start with::

    canvas-heal serve --store-url sqlite:///canvas_intents.db --port 8000

Or with PostgreSQL::

    canvas-heal serve --store-url postgresql://user:pass@pg/canvas --port 8000

Secure with an API key::

    CANVAS_API_KEY=secret canvas-heal serve ...

Clients set ``X-Canvas-Api-Key: secret`` on every request, or configure the
pytest option ``--canvas-api-key``.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

_log = logging.getLogger("canvas_heal.server")


# ---------------------------------------------------------------------------
# Per-app state helpers (avoids global dict, supports multiple app instances)
# ---------------------------------------------------------------------------

def _get_canvas_state(app: FastAPI) -> dict:
    try:
        return app.state.canvas_state
    except AttributeError:
        raise HTTPException(status_code=503, detail="Server not initialised")


def _set_canvas_state(app: FastAPI, resolver: Any, store: Any, embedder: Any) -> None:
    app.state.canvas_state = {
        "resolver": resolver,
        "store": store,
        "embedder": embedder,
    }


def _clear_canvas_state(app: FastAPI) -> None:
    if hasattr(app.state, "canvas_state"):
        del app.state.canvas_state


# ---------------------------------------------------------------------------
# Auth dependency factory
# ---------------------------------------------------------------------------

def _make_auth_dep(api_key: str | None):
    """Return a FastAPI dependency that enforces API key auth.

    ``api_key=None`` means read from the ``CANVAS_API_KEY`` environment
    variable at request time.  ``api_key=""`` means no auth required.
    """
    def _check_api_key(x_canvas_api_key: str = Header(default="")) -> None:
        expected = api_key if api_key is not None else os.environ.get("CANVAS_API_KEY", "")
        if expected and x_canvas_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return _check_api_key


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DescriptorDict(BaseModel):
    tag: str = ""
    role: str = ""
    label: str = ""
    element_type: str = ""
    placeholder: str = ""
    text_content: str = ""
    parent_tag: str = ""
    parent_role: str = ""
    section_heading: str = ""
    landmark: str = ""


class RecordRequest(BaseModel):
    selector: str
    descriptor: DescriptorDict
    page_url: str = ""


class RecordResponse(BaseModel):
    intent_name: str
    selector: str
    model_name: str


class CandidateItem(BaseModel):
    selector: str
    descriptor: DescriptorDict


class ResolveRequest(BaseModel):
    candidates: list[CandidateItem]
    skip_hidden: bool = True


class ResolveResponse(BaseModel):
    status: str
    selector: Optional[str]
    confidence: float
    message: str


class IntentSummary(BaseModel):
    name: str
    selector: str


class AuditEntry(BaseModel):
    intent_name: str
    status: str
    confidence: float
    original_selector: str
    resolved_selector: Optional[str]
    page_url: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model: str
    intents_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _desc_from_dict(d: DescriptorDict):
    from canvas_heal.descriptor import SemanticDescriptor
    return SemanticDescriptor(
        tag=d.tag, role=d.role, label=d.label,
        element_type=d.element_type, placeholder=d.placeholder,
        text_content=d.text_content, parent_tag=d.parent_tag,
        parent_role=d.parent_role, section_heading=d.section_heading,
        landmark=d.landmark,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_app(
    model: str = "all-MiniLM-L6-v2",
    store_url: str | None = None,
    db_path: str | None = None,
    team_id: str = "",
    project_id: str = "",
    threshold_auto: float = 0.92,
    threshold_confirm: float = 0.75,
    api_key: str | None = None,
) -> FastAPI:
    # None → read CANVAS_API_KEY env var at request time (production default)
    # ""   → no auth required
    # str  → enforce that specific key
    _auth_dep = _make_auth_dep(api_key)

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        # Only run full lifespan initialisation when canvas_state is not
        # already injected (e.g. by _init_for_testing).
        if hasattr(app.state, "canvas_state"):
            yield
            return

        from canvas_heal.embedder import IntentEmbedder
        from canvas_heal.resolver import ConfidenceGatedResolver, IntentStore, open_store

        _log.info("loading embedding model %r …", model)
        embedder = IntentEmbedder.get(model)

        if store_url:
            store = open_store(store_url, team_id=team_id, project_id=project_id)
        else:
            store = IntentStore(db_path, team_id=team_id, project_id=project_id)

        resolver = ConfidenceGatedResolver(
            store, embedder,
            threshold_auto=threshold_auto,
            threshold_confirm=threshold_confirm,
        )
        _set_canvas_state(app, resolver, store, embedder)
        _log.info("canvas-heal server ready")
        yield
        store.close()
        _clear_canvas_state(app)

    app = FastAPI(
        title="CANVAS-HEAL",
        description="Shared semantic self-healing locator service",
        version="0.3.0",
        lifespan=_lifespan,
    )

    # ---------------------------------------------------------------
    # Routes
    # ---------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request):
        try:
            state = _get_canvas_state(request.app)
        except HTTPException:
            return HealthResponse(status="starting", model="", intents_count=0)
        return HealthResponse(
            status="ok",
            model=state["embedder"].MODEL_NAME,
            intents_count=state["store"].count(),
        )

    @app.post(
        "/intents/{name}/record",
        response_model=RecordResponse,
        status_code=201,
        tags=["intents"],
        dependencies=[Depends(_auth_dep)],
    )
    def record_intent(name: str, body: RecordRequest, request: Request):
        state = _get_canvas_state(request.app)
        descriptor = _desc_from_dict(body.descriptor)
        state["resolver"].record(name, body.selector, descriptor, page_url=body.page_url)
        return RecordResponse(
            intent_name=name,
            selector=body.selector,
            model_name=state["embedder"].MODEL_NAME,
        )

    @app.post(
        "/intents/{name}/resolve",
        response_model=ResolveResponse,
        tags=["intents"],
        dependencies=[Depends(_auth_dep)],
    )
    def resolve_intent(name: str, body: ResolveRequest, request: Request):
        state = _get_canvas_state(request.app)
        candidates = [
            (item.selector, _desc_from_dict(item.descriptor))
            for item in body.candidates
        ]
        result = state["resolver"].resolve(name, candidates, skip_hidden=body.skip_hidden)
        return ResolveResponse(
            status=result.status.value,
            selector=result.selector,
            confidence=result.confidence,
            message=result.message,
        )

    @app.get(
        "/intents/",
        response_model=list[IntentSummary],
        tags=["intents"],
        dependencies=[Depends(_auth_dep)],
    )
    def list_intents(request: Request):
        state = _get_canvas_state(request.app)
        return [IntentSummary(name=n, selector=s) for n, s in state["store"].all()]

    @app.get(
        "/audit",
        response_model=list[AuditEntry],
        tags=["ops"],
        dependencies=[Depends(_auth_dep)],
    )
    def audit(request: Request):
        state = _get_canvas_state(request.app)
        return [
            AuditEntry(
                intent_name=e.intent_name,
                status=e.status.value,
                confidence=e.confidence,
                original_selector=e.original_selector,
                resolved_selector=e.resolved_selector,
                page_url=e.page_url,
                timestamp=e.timestamp,
            )
            for e in state["resolver"].get_audit_log()
        ]

    return app


def _init_for_testing(app: FastAPI, resolver: Any, store: Any, embedder: Any) -> None:
    """Inject pre-built objects into a specific app — for testing only."""
    _set_canvas_state(app, resolver, store, embedder)
