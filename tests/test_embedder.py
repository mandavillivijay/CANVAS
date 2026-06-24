import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from canvas_heal.embedder import (
    IntentEmbedder,
    ModelIntegrityError,
    KNOWN_MODEL_HASHES,
    MULTILINGUAL_MODEL,
    _find_weights_file,
    _sha256_path,
)


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
    with patch("sentence_transformers.SentenceTransformer"):
        other = IntentEmbedder.get("some-other-model")
    assert IntentEmbedder.get("all-MiniLM-L6-v2") is not other


def test_same_model_name_returns_singleton():
    assert IntentEmbedder.get("all-MiniLM-L6-v2") is IntentEmbedder.get("all-MiniLM-L6-v2")


def test_multilingual_constant_exists():
    assert isinstance(MULTILINGUAL_MODEL, str)


# --- ModelIntegrityError ---

def test_model_integrity_error_is_exception():
    with pytest.raises(ModelIntegrityError):
        raise ModelIntegrityError("test")


def test_known_model_hashes_dict_exists():
    assert isinstance(KNOWN_MODEL_HASHES, dict)
    assert "all-MiniLM-L6-v2" in KNOWN_MODEL_HASHES
    assert "paraphrase-multilingual-MiniLM-L12-v2" in KNOWN_MODEL_HASHES


def test_integrity_check_raises_on_wrong_hash(tmp_path):
    """IntentEmbedder raises ModelIntegrityError when hash doesn't match."""
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"fake model weights")

    fake_model = MagicMock()
    fake_model._first_module.return_value.auto_model.config._name_or_path = str(tmp_path)

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        with pytest.raises(ModelIntegrityError, match="integrity check failed"):
            IntentEmbedder("__hash_test_model__", model_sha256="deadbeef" * 8)


def test_integrity_check_passes_on_correct_hash(tmp_path):
    """IntentEmbedder loads without error when the hash matches."""
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"fake model weights")
    correct_hash = _sha256_path(weights)

    fake_model = MagicMock()
    fake_model._first_module.return_value.auto_model.config._name_or_path = str(tmp_path)
    fake_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        embedder = IntentEmbedder("__hash_pass_model__", model_sha256=correct_hash)
    assert embedder.MODEL_NAME == "__hash_pass_model__"


def test_canvas_model_cache_dir_passed_to_sentence_transformers(tmp_path, monkeypatch):
    """CANVAS_MODEL_CACHE_DIR is forwarded as cache_folder to SentenceTransformer."""
    monkeypatch.setenv("CANVAS_MODEL_CACHE_DIR", str(tmp_path))

    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_st.return_value._first_module.return_value.auto_model.config._name_or_path = str(tmp_path)
        IntentEmbedder("__cache_dir_test__")
        mock_st.assert_called_once_with("__cache_dir_test__", cache_folder=str(tmp_path))


def test_find_weights_file_prefers_safetensors(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "pytorch_model.bin").write_bytes(b"")
    result = _find_weights_file(tmp_path)
    assert result.name == "model.safetensors"


def test_find_weights_file_falls_back_to_bin(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"")
    result = _find_weights_file(tmp_path)
    assert result.name == "pytorch_model.bin"


def test_find_weights_file_searches_subdirs(tmp_path):
    snap = tmp_path / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"")
    result = _find_weights_file(tmp_path)
    assert result is not None
    assert result.name == "model.safetensors"


def test_find_weights_file_returns_none_when_missing(tmp_path):
    assert _find_weights_file(tmp_path) is None


def test_sha256_path(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world")
    digest = _sha256_path(f)
    import hashlib
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert digest == expected
