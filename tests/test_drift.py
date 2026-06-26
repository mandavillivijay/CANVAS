"""Tests for confidence drift detection and alerting (Issue #10)."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from canvas_heal.cli import _sparkline, _cmd_drift
from canvas_heal.descriptor import extract_from_tag
from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import ConfidenceGatedResolver, IntentStore, Resolution


def _tag(html: str):
    return BeautifulSoup(html, "html.parser").find(True)


def _fresh(drift_threshold=0.85, drift_window=10, drift_webhook_url=None):
    store = IntentStore(":memory:")
    res = ConfidenceGatedResolver(
        store,
        IntentEmbedder.get(),
        drift_threshold=drift_threshold,
        drift_window=drift_window,
        drift_webhook_url=drift_webhook_url,
    )
    return store, res


# ---------------------------------------------------------------------------
# confidence_history logging
# ---------------------------------------------------------------------------

def test_confidence_logged_on_resolve():
    store, res = _fresh()
    desc = extract_from_tag(_tag("<button>Submit</button>"))
    res.record("submit", "#submit", desc)
    res.resolve("submit", [("#submit", desc)])

    rows = store._conn.execute(
        "SELECT confidence, resolution FROM confidence_history WHERE intent_name = 'submit'"
    ).fetchall()
    assert len(rows) == 1
    confidence, resolution = rows[0]
    assert confidence > 0.99
    assert resolution == Resolution.HEALED.value


def test_confidence_logged_for_failed_resolve():
    store, res = _fresh()
    res.resolve("ghost", [])

    rows = store._conn.execute(
        "SELECT confidence FROM confidence_history WHERE intent_name = 'ghost'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 0.0


def test_multiple_resolves_accumulate():
    store, res = _fresh()
    desc = extract_from_tag(_tag("<button>Login</button>"))
    res.record("login", "#login", desc)

    for _ in range(3):
        res.resolve("login", [("#login", desc)])

    rows = store._conn.execute(
        "SELECT COUNT(*) FROM confidence_history WHERE intent_name = 'login'"
    ).fetchone()
    assert rows[0] == 3


# ---------------------------------------------------------------------------
# get_confidence_trend
# ---------------------------------------------------------------------------

def test_trend_empty_for_unknown_intent():
    store, _ = _fresh()
    trend = store.get_confidence_trend("unknown")
    assert trend["values"] == []
    assert trend["rolling_avg"] is None
    assert trend["sample_count"] == 0


def test_trend_rolling_avg():
    store, _ = _fresh()
    for c in [0.9, 0.8, 0.7]:
        store.log_confidence("x", c, "healed")

    trend = store.get_confidence_trend("x", window=3)
    assert trend["rolling_avg"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert trend["sample_count"] == 3


def test_trend_window_limits_values():
    store, _ = _fresh()
    for c in [0.95, 0.90, 0.85, 0.80, 0.75]:
        store.log_confidence("y", c, "healed")

    trend = store.get_confidence_trend("y", window=3)
    assert trend["sample_count"] == 3
    # most recent 3 values, oldest first
    assert trend["values"] == pytest.approx([0.85, 0.80, 0.75])


def test_get_all_confidence_trends():
    store, _ = _fresh()
    store.log_confidence("a", 0.9, "healed")
    store.log_confidence("b", 0.6, "failed")

    trends = store.get_all_confidence_trends(window=10)
    names = {t["intent_name"] for t in trends}
    assert "a" in names
    assert "b" in names


# ---------------------------------------------------------------------------
# Drift detection — WARNING emitted
# ---------------------------------------------------------------------------

def test_no_warning_before_window_full(caplog):
    store, res = _fresh(drift_threshold=0.85, drift_window=5)
    for c in [0.7, 0.7, 0.7, 0.7]:   # 4 calls, window=5
        store.log_confidence("x", c, "healed")
    res._check_drift("x")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "DRIFT" in r.message]
    assert len(warnings) == 0


def test_warning_when_window_full_and_below_threshold(caplog):
    store, res = _fresh(drift_threshold=0.85, drift_window=5)
    for c in [0.7, 0.7, 0.7, 0.7, 0.7]:  # 5 calls, all below 0.85
        store.log_confidence("btn", c, "healed")

    with caplog.at_level(logging.WARNING, logger="canvas_heal.resolver"):
        res._check_drift("btn")

    warnings = [r for r in caplog.records if "DRIFT" in r.message]
    assert len(warnings) == 1
    assert "btn" in warnings[0].message


def test_no_warning_when_above_threshold(caplog):
    store, res = _fresh(drift_threshold=0.85, drift_window=3)
    for c in [0.95, 0.96, 0.97]:
        store.log_confidence("good", c, "healed")

    with caplog.at_level(logging.WARNING, logger="canvas_heal.resolver"):
        res._check_drift("good")

    warnings = [r for r in caplog.records if "DRIFT" in r.message]
    assert len(warnings) == 0


def test_drift_triggered_via_resolve(caplog):
    store, res = _fresh(drift_threshold=0.92, drift_window=3)
    desc = extract_from_tag(_tag("<button>Checkout</button>"))
    res.record("checkout", "#checkout", desc)

    low_desc = extract_from_tag(_tag('<a href="/privacy">Privacy Policy</a>'))
    with caplog.at_level(logging.WARNING, logger="canvas_heal.resolver"):
        for _ in range(3):
            res.resolve("checkout", [("#other", low_desc)])

    warnings = [r for r in caplog.records if "DRIFT" in r.message]
    assert len(warnings) >= 1


# ---------------------------------------------------------------------------
# Slack webhook — fires once per intent per session
# ---------------------------------------------------------------------------

def test_webhook_fires_on_drift():
    store, res = _fresh(drift_threshold=0.85, drift_window=3, drift_webhook_url="http://fake.webhook/")
    for c in [0.7, 0.7, 0.7]:
        store.log_confidence("pay", c, "healed")

    with patch.object(res, "_fire_webhook") as mock_fire:
        res._check_drift("pay")
        assert mock_fire.call_count == 1
        mock_fire.assert_called_once_with("pay", pytest.approx(0.7))


def test_webhook_fires_only_once_per_session():
    store, res = _fresh(drift_threshold=0.85, drift_window=3, drift_webhook_url="http://fake.webhook/")
    for c in [0.7, 0.7, 0.7, 0.7, 0.7]:
        store.log_confidence("pay", c, "healed")

    with patch.object(res, "_fire_webhook") as mock_fire:
        res._check_drift("pay")
        res._check_drift("pay")
        assert mock_fire.call_count == 1


def test_webhook_not_fired_without_url():
    store, res = _fresh(drift_threshold=0.85, drift_window=3, drift_webhook_url=None)
    for c in [0.7, 0.7, 0.7]:
        store.log_confidence("x", c, "healed")

    with patch.object(res, "_fire_webhook") as mock_fire:
        res._check_drift("x")
        mock_fire.assert_not_called()


def test_webhook_failure_does_not_raise():
    store, res = _fresh(drift_threshold=0.85, drift_window=3, drift_webhook_url="http://invalid.local/")
    for c in [0.7, 0.7, 0.7]:
        store.log_confidence("x", c, "healed")
    res._fire_webhook("x", 0.7)   # should not raise


# ---------------------------------------------------------------------------
# Sparkline helper
# ---------------------------------------------------------------------------

def test_sparkline_empty():
    assert _sparkline([]) == ""


def test_sparkline_uniform():
    s = _sparkline([0.9, 0.9, 0.9])
    assert len(s) == 3
    assert len(set(s)) == 1   # all same char


def test_sparkline_ascending():
    s = _sparkline([0.5, 0.7, 0.9])
    assert s[0] < s[-1]   # chars increase (Unicode code point)


def test_sparkline_length():
    values = [0.8] * 7
    assert len(_sparkline(values)) == 7


# ---------------------------------------------------------------------------
# CLI drift command
# ---------------------------------------------------------------------------

def test_cmd_drift_empty_db(tmp_path, capsys):
    db = str(tmp_path / "test.db")
    store = IntentStore(db)
    store.close()

    args = MagicMock()
    args.db = db
    args.window = 10
    args.threshold = 0.85

    rc = _cmd_drift(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "No confidence history" in out


def test_cmd_drift_shows_intents(tmp_path, capsys):
    db = str(tmp_path / "test.db")
    store = IntentStore(db)
    for c in [0.9, 0.95, 0.92]:
        store.log_confidence("login", c, "healed")
    store.close()

    args = MagicMock()
    args.db = db
    args.window = 10
    args.threshold = 0.85

    rc = _cmd_drift(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "login" in out
    assert "ok" in out   # above threshold


def test_cmd_drift_shows_drift_flag(tmp_path, capsys):
    db = str(tmp_path / "test.db")
    store = IntentStore(db)
    for c in [0.6, 0.65, 0.6] * 5:
        store.log_confidence("checkout", c, "healed")
    store.close()

    args = MagicMock()
    args.db = db
    args.window = 10
    args.threshold = 0.85

    _cmd_drift(args)
    out = capsys.readouterr().out
    assert "YES" in out
