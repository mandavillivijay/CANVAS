import pytest


def pytest_addoption(parser):
    group = parser.getgroup("canvas-heal")
    group.addoption(
        "--canvas-db",
        default=":memory:",
        help="Path to canvas-heal SQLite database (default: :memory:). "
             "Superseded by --canvas-store-url when both are provided.",
    )
    group.addoption(
        "--canvas-store-url",
        default=None,
        help=(
            "Store URL for canvas-heal (overrides --canvas-db). "
            "Examples: sqlite:///canvas_intents.db, postgresql://user:pass@host/db"
        ),
    )
    group.addoption(
        "--canvas-team-id",
        default="",
        help="Team namespace for the canvas-heal intent store (multi-tenant).",
    )
    group.addoption(
        "--canvas-project-id",
        default="",
        help="Project namespace for the canvas-heal intent store (multi-tenant).",
    )
    group.addoption(
        "--canvas-server-url",
        default=None,
        help=(
            "URL of a running canvas-heal server. When set, CI agents skip local "
            "model loading and proxy record/resolve calls over HTTP. "
            "Example: http://canvas-heal:8000"
        ),
    )
    group.addoption(
        "--canvas-api-key",
        default="",
        help="API key sent as X-Canvas-Api-Key header to the canvas-heal server.",
    )
    group.addoption(
        "--canvas-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name for intent embedding",
    )
    group.addoption(
        "--canvas-scrub-text",
        action="store_true",
        default=False,
        help="Scrub PII (emails, phone numbers) from text stored in the intent database",
    )


@pytest.fixture(scope="session")
def canvas_store(request):
    from canvas_heal.resolver import IntentStore, open_store

    store_url = request.config.getoption("--canvas-store-url")
    team_id = request.config.getoption("--canvas-team-id")
    project_id = request.config.getoption("--canvas-project-id")
    scrub = request.config.getoption("--canvas-scrub-text")

    if store_url:
        store = open_store(
            store_url,
            store_raw_text=not scrub,
            team_id=team_id,
            project_id=project_id,
        )
    else:
        db = request.config.getoption("--canvas-db")
        store = IntentStore(
            db,
            store_raw_text=not scrub,
            team_id=team_id,
            project_id=project_id,
        )
    yield store
    store.close()


@pytest.fixture(scope="session")
def canvas_embedder(request):
    from canvas_heal.embedder import IntentEmbedder
    model = request.config.getoption("--canvas-model")
    return IntentEmbedder.get(model)


@pytest.fixture(scope="session")
def canvas_resolver(request, canvas_store, canvas_embedder):
    server_url = request.config.getoption("--canvas-server-url")
    if server_url:
        from canvas_heal.client import RemoteCanvasResolver
        api_key = request.config.getoption("--canvas-api-key")
        resolver = RemoteCanvasResolver(server_url, api_key=api_key)
        yield resolver
        resolver.close()
    else:
        from canvas_heal.resolver import ConfidenceGatedResolver
        yield ConfidenceGatedResolver(canvas_store, canvas_embedder)
