from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import numpy as np

_log = logging.getLogger("canvas_heal.embedder")

MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# SHA-256 of the primary weights file (model.safetensors or pytorch_model.bin).
# Empty string means "not yet verified" — hash check is skipped for that model.
# To populate: python -c "from canvas_heal.embedder import compute_model_hash; print(compute_model_hash('all-MiniLM-L6-v2'))"
KNOWN_MODEL_HASHES: dict[str, str] = {
    "all-MiniLM-L6-v2": "",
    "paraphrase-multilingual-MiniLM-L12-v2": "",
}


class ModelIntegrityError(Exception):
    """Raised when a model's SHA-256 hash does not match the expected value."""


def _sha256_path(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_weights_file(model_dir: Path) -> Path | None:
    """Return the primary weights file inside model_dir (searches recursively)."""
    for name in ("model.safetensors", "pytorch_model.bin"):
        direct = model_dir / name
        if direct.exists():
            return direct
    # HF hub snapshot layout: .../snapshots/<commit-hash>/<weights>
    for name in ("model.safetensors", "pytorch_model.bin"):
        matches = sorted(model_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _get_model_path(model) -> Path | None:
    """Best-effort: return the local cache directory for a loaded SentenceTransformer."""
    try:
        path_str = model._first_module().auto_model.config._name_or_path
        p = Path(path_str)
        if p.exists():
            return p
    except (AttributeError, TypeError, OSError):
        pass
    return None


def compute_model_hash(model_name: str, cache_dir: str | None = None) -> str:
    """Load model_name and return the SHA-256 hex digest of its primary weights file.

    Use this to populate KNOWN_MODEL_HASHES entries or verify a bundled model.
    """
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, cache_folder=cache_dir)
    model_path = _get_model_path(m)
    if model_path is None:
        raise ModelIntegrityError(f"Cannot locate weights directory for {model_name!r}")
    weights = _find_weights_file(model_path)
    if weights is None:
        raise ModelIntegrityError(f"No weights file found in {model_path}")
    return _sha256_path(weights)


class IntentEmbedder:
    """Wraps sentence-transformers to produce normalized intent vectors."""

    MODEL_NAME = "all-MiniLM-L6-v2"
    _instances: dict[str, "IntentEmbedder"] = {}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model_sha256: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer
        self.MODEL_NAME = model_name
        cache_dir = os.environ.get("CANVAS_MODEL_CACHE_DIR")

        _log.debug("loading model %r cache_dir=%r", model_name, cache_dir)
        t0 = time.monotonic()
        self._model = SentenceTransformer(model_name, cache_folder=cache_dir)
        _log.debug("model %r loaded in %.2fs", model_name, time.monotonic() - t0)

        expected = model_sha256 or KNOWN_MODEL_HASHES.get(model_name, "")
        if expected:
            self._verify_integrity(model_name, expected)

    def _verify_integrity(self, model_name: str, expected_hash: str) -> None:
        model_path = _get_model_path(self._model)
        if model_path is None:
            _log.warning("integrity check skipped for %r: model path could not be determined", model_name)
            return
        weights = _find_weights_file(model_path)
        if weights is None:
            _log.warning("integrity check skipped for %r: no weights file found in %s", model_name, model_path)
            return
        _log.debug("verifying integrity of %r using %s", model_name, weights.name)
        actual = _sha256_path(weights)
        if actual != expected_hash:
            raise ModelIntegrityError(
                f"Model integrity check failed for {model_name!r}: "
                f"expected SHA-256 {expected_hash!r}, got {actual!r}"
            )
        _log.info("model %r integrity verified (SHA-256 OK)", model_name)

    @classmethod
    def get(cls, model_name: str = "all-MiniLM-L6-v2", model_sha256: str | None = None) -> "IntentEmbedder":
        """Return the process-wide singleton for a model, loading it on first call."""
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name, model_sha256=model_sha256)
        return cls._instances[model_name]

    def embed(self, text: str) -> np.ndarray:
        """Embed a text string into a normalized float32 vector (dim=384)."""
        t0 = time.monotonic()
        vec = self._model.encode(text, normalize_embeddings=True)
        _log.debug("embed latency=%.3fs text_len=%d", time.monotonic() - t0, len(text))
        return vec.astype(np.float32)

    def embed_descriptor(self, descriptor) -> np.ndarray:
        """Embed a SemanticDescriptor by calling its to_text() method."""
        return self.embed(descriptor.to_text())

    def batch_embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of texts in one batched call. Much faster than calling embed() in a loop."""
        if not texts:
            return []
        t0 = time.monotonic()
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        _log.debug("batch_embed candidates=%d latency=%.3fs", len(texts), time.monotonic() - t0)
        return [v.astype(np.float32) for v in vecs]

    def batch_embed_descriptors(self, descriptors: list) -> list[np.ndarray]:
        """Batch-embed a list of SemanticDescriptors."""
        return self.batch_embed([d.to_text() for d in descriptors])

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two pre-normalized vectors (dot product)."""
        return float(np.dot(a, b))
