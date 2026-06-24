from __future__ import annotations

import logging
import time

import numpy as np

_log = logging.getLogger("canvas_heal.embedder")

MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class IntentEmbedder:
    """Wraps sentence-transformers to produce normalized intent vectors."""

    MODEL_NAME = "all-MiniLM-L6-v2"
    _instances: dict[str, "IntentEmbedder"] = {}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self.MODEL_NAME = model_name
        _log.debug("loading model %r", model_name)
        t0 = time.monotonic()
        self._model = SentenceTransformer(self.MODEL_NAME)
        _log.debug("model %r loaded in %.2fs", model_name, time.monotonic() - t0)

    @classmethod
    def get(cls, model_name: str = "all-MiniLM-L6-v2") -> "IntentEmbedder":
        """Return the process-wide singleton for a model, loading it on first call."""
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name)
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
