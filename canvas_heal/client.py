"""HTTP client for canvas-heal REST service.

CI agents that set ``--canvas-server-url`` use this client instead of loading
the 90 MB embedding model locally.  The server loads the model once; all
agents send lightweight descriptor dicts over HTTP.

Usage (direct)::

    from canvas_heal.client import RemoteCanvasResolver
    resolver = RemoteCanvasResolver("http://canvas-heal:8000", api_key="secret")
    resolver.record("submit-btn", "#submit", descriptor)
    result = resolver.resolve("submit-btn", candidates)

Usage (pytest) — set ``--canvas-server-url`` and the plugin handles the rest.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

_log = logging.getLogger("canvas_heal.client")


class RemoteCanvasResolver:
    """Drop-in replacement for :class:`~canvas_heal.resolver.ConfidenceGatedResolver`
    that delegates embedding and storage to a running canvas-heal server.

    Only ``httpx`` is required (already a fastapi dependency); no sentence-
    transformers or numpy are needed on the CI agent side.
    """

    def __init__(self, server_url: str, api_key: str = "", timeout: float = 30.0) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for the remote client. "
                "Install it with: pip install canvas-heal[server]"
            ) from exc

        self._base = server_url.rstrip("/")
        self._headers = {"X-Canvas-Api-Key": api_key} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=self._headers)
        self._audit_log: list[Any] = []

    # ------------------------------------------------------------------
    # Core API (matches ConfidenceGatedResolver)
    # ------------------------------------------------------------------

    def record(
        self,
        name: str,
        selector: str,
        descriptor: Any,
        page_url: str = "",
    ) -> None:
        """Record an intent fingerprint on the server."""
        payload = {
            "selector": selector,
            "descriptor": descriptor.to_dict() if hasattr(descriptor, "to_dict") else descriptor,
            "page_url": page_url,
        }
        resp = self._client.post(f"{self._base}/intents/{name}/record", json=payload)
        resp.raise_for_status()

    def resolve(
        self,
        name: str,
        candidates: List[Tuple[str, Any]],
        skip_hidden: bool = True,
    ) -> Any:
        """Resolve an intent against a candidate list via the server."""
        from canvas_heal.resolver import Resolution, ResolverResult

        candidate_dicts = [
            {
                "selector": sel,
                "descriptor": desc.to_dict() if hasattr(desc, "to_dict") else desc,
            }
            for sel, desc in candidates
        ]
        payload = {"candidates": candidate_dicts, "skip_hidden": skip_hidden}
        resp = self._client.post(f"{self._base}/intents/{name}/resolve", json=payload)
        resp.raise_for_status()
        data = resp.json()

        result = ResolverResult(
            status=Resolution(data["status"]),
            selector=data["selector"],
            confidence=data["confidence"],
            descriptor=None,
            message=data["message"],
        )
        self._audit_log.append(result)
        return result

    def get_audit_log(self) -> list[Any]:
        return self._audit_log

    def clear_audit_log(self) -> None:
        self._audit_log = []

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def health(self) -> dict:
        resp = self._client.get(f"{self._base}/health")
        resp.raise_for_status()
        return resp.json()

    def list_intents(self) -> list[dict]:
        resp = self._client.get(f"{self._base}/intents/")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RemoteCanvasResolver:
        return self

    def __exit__(self, *_) -> None:
        self.close()
