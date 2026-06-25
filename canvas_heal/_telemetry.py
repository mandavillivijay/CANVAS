"""Optional OpenTelemetry instrumentation for canvas-heal.

If opentelemetry-api is not installed, all operations are silent no-ops.
Tracers and meters are acquired lazily so tests can configure providers
before the first CANVAS call.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

try:
    import opentelemetry.metrics as _ot_metrics
    import opentelemetry.trace as _ot_trace

    _OTEL = True
except ImportError:
    _OTEL = False
    _ot_trace = None  # type: ignore[assignment]
    _ot_metrics = None  # type: ignore[assignment]

_VERSION = "0.3.0"
_SCHEMA = "https://opentelemetry.io/schemas/1.26.0"


class _NoopSpan:
    """Drop-in replacement when OTel is unavailable or a span is not active."""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: ARG002
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        pass


_NOOP = _NoopSpan()


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Generator[Any, None, None]:
    """Context manager that starts an OTel span when available, else yields a no-op."""
    if not _OTEL:
        yield _NOOP
        return
    tracer = _ot_trace.get_tracer("canvas_heal", _VERSION, schema_url=_SCHEMA)
    with tracer.start_as_current_span(name, attributes=attributes or {}) as s:
        yield s


def _meter():
    if not _OTEL:
        return None
    return _ot_metrics.get_meter("canvas_heal", _VERSION, schema_url=_SCHEMA)


def record_resolve(status: str, confidence: float, intent_name: str) -> None:
    m = _meter()
    if m is None:
        return
    attrs = {"canvas.resolution": status, "canvas.intent_name": intent_name}
    m.create_histogram(
        "canvas_heal.resolve.confidence",
        description="Cosine similarity confidence score per resolve() call",
        unit="1",
    ).record(confidence, attrs)
    m.create_counter(
        "canvas_heal.resolve.total",
        description="Number of resolve() calls by resolution status",
    ).add(1, attrs)


def record_record(intent_name: str, model_name: str) -> None:
    m = _meter()
    if m is None:
        return
    m.create_counter(
        "canvas_heal.record.total",
        description="Number of record() calls",
    ).add(1, {"canvas.intent_name": intent_name, "canvas.model_name": model_name})


def record_embed(duration_s: float, model_name: str, batch_size: int = 1) -> None:
    m = _meter()
    if m is None:
        return
    m.create_histogram(
        "canvas_heal.embed.duration_seconds",
        description="Duration of embedding operations",
        unit="s",
    ).record(duration_s, {"canvas.model_name": model_name, "canvas.batch_size": batch_size})
