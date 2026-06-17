import numpy as np
import pytest
from canvas_heal.embedder import IntentEmbedder


@pytest.fixture(scope="module")
def embedder():
    return IntentEmbedder.get()


def test_embed_shape_and_norm(embedder):
    vec = embedder.embed("submit button in checkout form")
    assert vec.shape == (384,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


def test_similar_intents_high_similarity(embedder):
    a = embedder.embed("button labeled 'Submit Order' inside form under heading 'Checkout'")
    b = embedder.embed("button labeled 'Place Order' inside form under heading 'Checkout'")
    sim = IntentEmbedder.cosine_similarity(a, b)
    assert sim > 0.80, f"Expected high similarity for paraphrase, got {sim:.3f}"


def test_dissimilar_intents_lower_similarity(embedder):
    a = embedder.embed("button labeled 'Submit Order' inside form under heading 'Checkout'")
    b = embedder.embed("link with text 'Privacy Policy' inside footer")
    sim = IntentEmbedder.cosine_similarity(a, b)
    assert sim < 0.80, f"Expected lower similarity for unrelated intents, got {sim:.3f}"


def test_identical_text_near_one(embedder):
    text = "textbox placeholder 'Search products' inside nav"
    a = embedder.embed(text)
    b = embedder.embed(text)
    assert IntentEmbedder.cosine_similarity(a, b) > 0.99


def test_singleton_identity():
    assert IntentEmbedder.get() is IntentEmbedder.get()


def test_embed_descriptor(embedder):
    from bs4 import BeautifulSoup
    from canvas_heal.descriptor import extract_from_tag
    el = BeautifulSoup('<button>Add to Cart</button>', "html.parser").find("button")
    desc = extract_from_tag(el)
    vec = embedder.embed_descriptor(desc)
    assert vec.shape == (384,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


def test_custom_model_name_stored():
    assert IntentEmbedder.get("all-MiniLM-L6-v2").MODEL_NAME == "all-MiniLM-L6-v2"


def test_different_model_names_are_different_instances():
    from unittest.mock import patch
    with patch("sentence_transformers.SentenceTransformer"):
        other = IntentEmbedder.get("some-other-model")
    assert IntentEmbedder.get("all-MiniLM-L6-v2") is not other


def test_same_model_name_returns_singleton():
    assert IntentEmbedder.get("all-MiniLM-L6-v2") is IntentEmbedder.get("all-MiniLM-L6-v2")


def test_multilingual_constant_exists():
    from canvas_heal.embedder import MULTILINGUAL_MODEL
    assert isinstance(MULTILINGUAL_MODEL, str)
