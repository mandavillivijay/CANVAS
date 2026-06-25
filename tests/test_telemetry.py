"""Tests for OpenTelemetry instrumentation in canvas-heal.

Requires opentelemetry-sdk (included in dev extras).

OTel's global providers can only be set once per process, so we use a
session-scoped fixture that installs them once, and clear the span exporter
between tests.
"""
from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

pytest.importorskip("opentelemetry", reason="opentelemetry-api not installed")

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import opentelemetry.metrics as otel_metrics
import opentelemetry.trace as otel_trace

from canvas_heal.descriptor import extract_from_tag
from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import ConfidenceGatedResolver, IntentStore, Resolution


def _tag(html: str):
    return BeautifulSoup(html, "html.parser").find(True)


# ---------------------------------------------------------------------------
# Session-scoped provider setup — OTel only allows one global provider per process
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _otel_session():
    span_exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(span_exp))
    otel_trace.set_tracer_provider(tp)

    metric_reader = InMemoryMetricReader()
    mp = MeterProvider(metric_readers=[metric_reader])
    otel_metrics.set_meter_provider(mp)

    return span_exp, metric_reader


@pytest.fixture()
def otel_providers(_otel_session):
    """Yield (span_exporter, metric_reader) with a clean span slate per test."""
    span_exp, metric_reader = _otel_session
    span_exp.clear()
    yield span_exp, metric_reader


def _fresh_resolver():
    return ConfidenceGatedResolver(IntentStore(":memory:"), IntentEmbedder.get())


def _metric_names(metric_reader: InMemoryMetricReader) -> set[str]:
    data = metric_reader.get_metrics_data()
    if data is None:
        return set()
    return {
        m.name
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    }


# ---------------------------------------------------------------------------
# Span presence tests
# ---------------------------------------------------------------------------

def test_record_emits_span(otel_providers):
    span_exp, _ = otel_providers
    res = _fresh_resolver()
    res.record("cart", "#cart", extract_from_tag(_tag("<button>Add to Cart</button>")))

    names = [s.name for s in span_exp.get_finished_spans()]
    assert "canvas_heal.record" in names


def test_resolve_emits_span(otel_providers):
    span_exp, _ = otel_providers
    res = _fresh_resolver()
    desc = extract_from_tag(_tag("<button>Submit</button>"))
    res.record("submit", "#submit", desc)
    span_exp.clear()

    res.resolve("submit", [("#submit", desc)])
    names = [s.name for s in span_exp.get_finished_spans()]
    assert "canvas_heal.resolve" in names


def test_embed_emits_span(otel_providers):
    span_exp, _ = otel_providers
    IntentEmbedder.get().embed("checkout button")

    names = [s.name for s in span_exp.get_finished_spans()]
    assert "canvas_heal.embed" in names


def test_batch_embed_emits_span(otel_providers):
    span_exp, _ = otel_providers
    IntentEmbedder.get().batch_embed(["login button", "submit form"])

    names = [s.name for s in span_exp.get_finished_spans()]
    assert "canvas_heal.batch_embed" in names


# ---------------------------------------------------------------------------
# Span attribute tests
# ---------------------------------------------------------------------------

def test_record_span_attributes(otel_providers):
    span_exp, _ = otel_providers
    res = _fresh_resolver()
    res.record("pay", "#pay-btn", extract_from_tag(_tag("<button>Pay Now</button>")))

    spans = {s.name: s for s in span_exp.get_finished_spans()}
    s = spans["canvas_heal.record"]
    assert s.attributes["canvas.intent_name"] == "pay"
    assert s.attributes["canvas.selector"] == "#pay-btn"
    assert "canvas.model_name" in s.attributes


def test_resolve_span_attributes_healed(otel_providers):
    span_exp, _ = otel_providers
    res = _fresh_resolver()
    desc = extract_from_tag(_tag("<button>Add to Cart</button>"))
    res.record("cart", "#cart", desc)
    span_exp.clear()

    res.resolve("cart", [("#cart", desc)])
    spans = {s.name: s for s in span_exp.get_finished_spans()}
    s = spans["canvas_heal.resolve"]
    assert s.attributes["canvas.intent_name"] == "cart"
    assert s.attributes["canvas.resolution"] == Resolution.HEALED.value
    assert s.attributes["canvas.confidence"] > 0.99
    assert s.attributes.get("canvas.resolved_selector") == "#cart"


def test_resolve_span_attributes_failed(otel_providers):
    span_exp, _ = otel_providers
    res = _fresh_resolver()
    res.resolve("ghost", [])  # no intent recorded → FAILED

    spans = {s.name: s for s in span_exp.get_finished_spans()}
    s = spans["canvas_heal.resolve"]
    assert s.attributes["canvas.resolution"] == Resolution.FAILED.value
    assert s.attributes["canvas.confidence"] == 0.0


def test_embed_span_attributes(otel_providers):
    span_exp, _ = otel_providers
    IntentEmbedder.get().embed("search box")

    spans = {s.name: s for s in span_exp.get_finished_spans()}
    s = spans["canvas_heal.embed"]
    assert "canvas.model_name" in s.attributes
    assert s.attributes["canvas.text_length"] == len("search box")


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

def test_resolve_confidence_metric(otel_providers):
    _, reader = otel_providers
    res = _fresh_resolver()
    desc = extract_from_tag(_tag("<button>Login</button>"))
    res.record("login", "#login", desc)
    res.resolve("login", [("#login", desc)])

    assert "canvas_heal.resolve.confidence" in _metric_names(reader)


def test_resolve_total_metric(otel_providers):
    _, reader = otel_providers
    res = _fresh_resolver()
    desc = extract_from_tag(_tag("<button>Logout</button>"))
    res.record("logout", "#logout", desc)
    res.resolve("logout", [("#logout", desc)])

    assert "canvas_heal.resolve.total" in _metric_names(reader)


def test_record_total_metric(otel_providers):
    _, reader = otel_providers
    res = _fresh_resolver()
    res.record("save", "#save", extract_from_tag(_tag("<button>Save</button>")))

    assert "canvas_heal.record.total" in _metric_names(reader)


def test_embed_duration_metric(otel_providers):
    _, reader = otel_providers
    IntentEmbedder.get().embed("buy button")

    assert "canvas_heal.embed.duration_seconds" in _metric_names(reader)


# ---------------------------------------------------------------------------
# Graceful degradation — noop when OTel flag is off
# ---------------------------------------------------------------------------

def test_noop_fallback_without_otel(monkeypatch):
    import canvas_heal._telemetry as tel

    monkeypatch.setattr(tel, "_OTEL", False)
    with tel.span("canvas_heal.test") as s:
        s.set_attribute("k", "v")  # _NoopSpan — must not raise

    tel.record_resolve("healed", 0.95, "intent")
    tel.record_record("intent", "model")
    tel.record_embed(0.01, "model", 1)
