import pytest


def pytest_addoption(parser):
    group = parser.getgroup("canvas-heal")
    group.addoption(
        "--canvas-db",
        default=":memory:",
        help="Path to canvas-heal intent database (default: :memory:)",
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
    from canvas_heal.resolver import IntentStore
    db = request.config.getoption("--canvas-db")
    scrub = request.config.getoption("--canvas-scrub-text")
    store = IntentStore(db, store_raw_text=not scrub)
    yield store
    store.close()


@pytest.fixture(scope="session")
def canvas_embedder(request):
    from canvas_heal.embedder import IntentEmbedder
    model = request.config.getoption("--canvas-model")
    return IntentEmbedder.get(model)


@pytest.fixture(scope="session")
def canvas_resolver(canvas_store, canvas_embedder):
    from canvas_heal.resolver import ConfidenceGatedResolver
    return ConfidenceGatedResolver(canvas_store, canvas_embedder)
