import pytest
from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import ConfidenceGatedResolver, IntentStore


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


@pytest.fixture(scope="session")
def canvas_store(request):
    db = request.config.getoption("--canvas-db")
    store = IntentStore(db)
    yield store
    store.close()


@pytest.fixture(scope="session")
def canvas_embedder(request):
    model = request.config.getoption("--canvas-model")
    return IntentEmbedder.get(model)


@pytest.fixture(scope="session")
def canvas_resolver(canvas_store, canvas_embedder):
    return ConfidenceGatedResolver(canvas_store, canvas_embedder)
