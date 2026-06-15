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

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two pre-normalized vectors (dot product)."""
        return float(np.dot(a, b))
