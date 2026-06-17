from __future__ import annotations

import numpy as np


class IntentEmbedder:
    """Wraps sentence-transformers to produce normalized intent vectors."""

    MODEL_NAME = "all-MiniLM-L6-v2"
    _instance: IntentEmbedder | None = None

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.MODEL_NAME)

    @classmethod
    def get(cls) -> IntentEmbedder:
        """Return the process-wide singleton, loading the model on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, text: str) -> np.ndarray:
        """Embed a text string into a normalized float32 vector (dim=384)."""
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_descriptor(self, descriptor) -> np.ndarray:
        """Embed a SemanticDescriptor by calling its to_text() method."""
        return self.embed(descriptor.to_text())

    def batch_embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of texts in one batched call. Much faster than calling embed() in a loop."""
        if not texts:
            return []
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.astype(np.float32) for v in vecs]

    def batch_embed_descriptors(self, descriptors: list) -> list[np.ndarray]:
        """Batch-embed a list of SemanticDescriptors."""
        return self.batch_embed([d.to_text() for d in descriptors])

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two pre-normalized vectors (dot product)."""
        return float(np.dot(a, b))
