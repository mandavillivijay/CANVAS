"""Tests for the canvas-heal REST service.

Uses httpx.AsyncClient with ASGITransport (httpx 0.28+ compatible) and
manually seeds app state to avoid triggering the lifespan.
"""
from __future__ import annotations

import httpx
import pytest

from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import ConfidenceGatedResolver, IntentStore
from canvas_heal.server import _init_for_testing, create_app

pytestmark = pytest.mark.anyio

# ---------------------------------------------------------------------------
# Shared descriptor
# ---------------------------------------------------------------------------

_DESC = {
    "tag": "button", "role": "button", "label": "Submit", "element_type": "submit",
    "placeholder": "", "text_content": "Submit", "parent_tag": "form",
    "parent_role": "", "section_heading": "", "landmark": "",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


def _make_seeded_app(api_key: str = ""):
    """Return (app, store) with resolver pre-injected (no lifespan needed)."""
    store = IntentStore(":memory:")
    embedder = IntentEmbedder.get()
    resolver = ConfidenceGatedResolver(store, embedder)
    app = create_app(api_key=api_key)
    _init_for_testing(app, resolver, store, embedder)
    return app, store


@pytest.fixture(scope="module")
async def client():
    app, store = _make_seeded_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c
    store.close()


@pytest.fixture(scope="module")
async def secured_client():
    app, store = _make_seeded_app(api_key="test-secret")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c
    store.close()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

async def test_health_returns_200(client):
    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_health_body(client):
    data = (await client.get("/health")).json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "intents_count" in data


async def test_health_unauthenticated_allowed(secured_client):
    # /health is always public
    resp = await secured_client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

async def test_record_returns_201(client):
    resp = await client.post(
        "/intents/submit-btn/record",
        json={"selector": "#submit", "descriptor": _DESC, "page_url": "http://localhost/form"},
    )
    assert resp.status_code == 201


async def test_record_response_body(client):
    resp = await client.post(
        "/intents/save-btn/record",
        json={"selector": "#save", "descriptor": _DESC},
    )
    data = resp.json()
    assert data["intent_name"] == "save-btn"
    assert data["selector"] == "#save"
    assert "model_name" in data


async def test_record_increments_count(client):
    before = (await client.get("/health")).json()["intents_count"]
    await client.post(
        "/intents/counter-btn/record",
        json={"selector": "#c", "descriptor": _DESC},
    )
    after = (await client.get("/health")).json()["intents_count"]
    assert after >= before + 1


async def test_record_requires_api_key(secured_client):
    resp = await secured_client.post(
        "/intents/btn/record",
        json={"selector": "#btn", "descriptor": _DESC},
    )
    assert resp.status_code == 401


async def test_record_with_correct_api_key(secured_client):
    resp = await secured_client.post(
        "/intents/btn/record",
        json={"selector": "#btn", "descriptor": _DESC},
        headers={"X-Canvas-Api-Key": "test-secret"},
    )
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------

async def test_resolve_healed(client):
    await client.post(
        "/intents/login-btn/record",
        json={"selector": "#login", "descriptor": _DESC},
    )
    resp = await client.post(
        "/intents/login-btn/resolve",
        json={"candidates": [{"selector": "#login", "descriptor": _DESC}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healed"
    assert data["confidence"] >= 0.92


async def test_resolve_failed_no_candidates(client):
    await client.post(
        "/intents/nc-btn/record",
        json={"selector": "#nc", "descriptor": _DESC},
    )
    resp = await client.post(
        "/intents/nc-btn/resolve",
        json={"candidates": []},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failed"


async def test_resolve_unknown_intent(client):
    resp = await client.post(
        "/intents/ghost-intent/resolve",
        json={"candidates": [{"selector": "#x", "descriptor": _DESC}]},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failed"


async def test_resolve_confidence_in_range(client):
    await client.post(
        "/intents/conf-btn/record",
        json={"selector": "#cf", "descriptor": _DESC},
    )
    resp = await client.post(
        "/intents/conf-btn/resolve",
        json={"candidates": [{"selector": "#cf", "descriptor": _DESC}]},
    )
    data = resp.json()
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# List intents
# ---------------------------------------------------------------------------

async def test_list_returns_array(client):
    resp = await client.get("/intents/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_list_contains_recorded_intent(client):
    await client.post(
        "/intents/list-btn/record",
        json={"selector": "#lb", "descriptor": _DESC},
    )
    resp = await client.get("/intents/")
    names = [item["name"] for item in resp.json()]
    assert "list-btn" in names


async def test_list_entry_fields(client):
    resp = await client.get("/intents/")
    for item in resp.json():
        assert "name" in item
        assert "selector" in item


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

async def test_audit_returns_array(client):
    await client.post(
        "/intents/audit-btn/record",
        json={"selector": "#ab", "descriptor": _DESC},
    )
    await client.post(
        "/intents/audit-btn/resolve",
        json={"candidates": [{"selector": "#ab", "descriptor": _DESC}]},
    )
    resp = await client.get("/audit")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_audit_entry_shape(client):
    # Make sure audit-btn resolve happened above (same module-scoped client)
    resp = await client.get("/audit")
    entries = [e for e in resp.json() if e["intent_name"] == "audit-btn"]
    assert entries
    e = entries[0]
    for field in ("status", "confidence", "timestamp", "original_selector"):
        assert field in e
