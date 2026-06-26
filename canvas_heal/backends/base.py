from __future__ import annotations

from typing import Optional, Tuple


class IntentStoreBackend:
    """Abstract base class for intent storage backends.

    Backends store raw serialized data (descriptor JSON + embedding bytes).
    Numpy conversion, PII scrubbing, and descriptor parsing live in IntentStore.
    All methods are namespaced by ``team_id`` and ``project_id`` set at construction.
    """

    team_id: str
    project_id: str

    def store(
        self,
        name: str,
        selector: str,
        descriptor_json: str,
        embedding_blob: bytes,
        model_name: str,
        page_url: str,
        recorded_by: str,
    ) -> None:
        raise NotImplementedError

    def get(self, name: str) -> Optional[Tuple[str, str, bytes, str]]:
        """Return (selector, descriptor_json, embedding_blob, page_url) or None."""
        raise NotImplementedError

    def all(self) -> list[tuple[str, str]]:
        """Return (name, selector) pairs for every stored intent in the namespace."""
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def get_version_history(self, name: str) -> list[dict]:
        """Return version rows newest-first as plain dicts."""
        raise NotImplementedError

    def rollback(self, name: str, version_id: int, recorded_by: str) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> IntentStoreBackend:
        return self

    def __exit__(self, *_) -> None:
        self.close()
