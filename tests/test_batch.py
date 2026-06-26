"""Tests for bulk re-record workflow (Issue #11)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from bs4 import BeautifulSoup

from canvas_heal.batch import (
    BulkRerecorder,
    PageEntry,
    RerecordResult,
    load_page_map,
    print_report,
)
from canvas_heal.cli import _cmd_rerecord_all
from canvas_heal.descriptor import SemanticDescriptor, extract_from_tag
from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import ConfidenceGatedResolver, IntentStore


def _tag(html: str):
    return BeautifulSoup(html, "html.parser").find(True)


def _desc(html: str) -> SemanticDescriptor:
    return extract_from_tag(_tag(html))


def _fresh_store() -> IntentStore:
    return IntentStore(":memory:")


def _make_extractor(html_map: dict[str, str]):
    """Return a mock descriptor_extractor that returns descriptors based on selector."""
    def extractor(page, selector: str) -> SemanticDescriptor:
        if selector not in html_map:
            raise ValueError(f"selector not found: {selector}")
        return _desc(html_map[selector])
    return extractor


# ---------------------------------------------------------------------------
# load_page_map
# ---------------------------------------------------------------------------

def test_load_page_map_valid(tmp_path):
    data = [
        {"url": "https://app/login", "intents": [{"name": "login", "selector": "#login"}]},
        {"url": "https://app/cart", "intents": [{"name": "cart", "selector": "#cart"}]},
    ]
    p = tmp_path / "map.json"
    p.write_text(json.dumps(data))

    entries = load_page_map(p)
    assert len(entries) == 2
    assert entries[0].url == "https://app/login"
    assert entries[0].intents[0]["name"] == "login"
    assert entries[1].url == "https://app/cart"


def test_load_page_map_missing_file():
    with pytest.raises(FileNotFoundError):
        load_page_map("/does/not/exist.json")


# ---------------------------------------------------------------------------
# RerecordResult.is_regression
# ---------------------------------------------------------------------------

def test_is_regression_below_threshold():
    r = RerecordResult("x", "#x", "http://e", old_similarity=0.5, status="updated")
    assert r.is_regression is True


def test_is_regression_above_threshold():
    r = RerecordResult("x", "#x", "http://e", old_similarity=0.95, status="updated")
    assert r.is_regression is False


def test_is_regression_no_prior():
    r = RerecordResult("x", "#x", "http://e", old_similarity=None, status="new")
    assert r.is_regression is False


# ---------------------------------------------------------------------------
# BulkRerecorder.rerecord_page — new intent
# ---------------------------------------------------------------------------

def test_rerecord_new_intent():
    store = _fresh_store()
    rec = BulkRerecorder(store, IntentEmbedder.get())
    page = MagicMock()
    entry = PageEntry(url="https://app/login", intents=[{"name": "login", "selector": "#login"}])
    extractor = _make_extractor({"#login": "<button>Login</button>"})

    results = rec.rerecord_page(page, entry, extractor)

    assert len(results) == 1
    assert results[0].status == "new"
    assert results[0].old_similarity is None
    assert store.get("login") is not None


def test_rerecord_updates_existing_intent():
    store = _fresh_store()
    rec = BulkRerecorder(store, IntentEmbedder.get())
    page = MagicMock()
    entry = PageEntry(url="https://app/login", intents=[{"name": "login", "selector": "#login"}])

    # Pre-record
    desc_v1 = _desc("<button>Sign In</button>")
    store.store("login", "#login", desc_v1, IntentEmbedder.get().embed_descriptor(desc_v1))

    extractor = _make_extractor({"#login": "<button>Login</button>"})
    results = rec.rerecord_page(page, entry, extractor)

    assert results[0].status == "updated"
    assert results[0].old_similarity is not None
    assert 0 <= results[0].old_similarity <= 1.0


def test_rerecord_identical_element_high_similarity():
    store = _fresh_store()
    rec = BulkRerecorder(store, IntentEmbedder.get())
    page = MagicMock()
    entry = PageEntry(url="https://app", intents=[{"name": "btn", "selector": "#btn"}])
    desc = _desc("<button>Add to Cart</button>")
    store.store("btn", "#btn", desc, IntentEmbedder.get().embed_descriptor(desc))

    extractor = _make_extractor({"#btn": "<button>Add to Cart</button>"})
    results = rec.rerecord_page(page, entry, extractor)

    assert results[0].old_similarity > 0.99


def test_rerecord_unrelated_element_flags_regression():
    store = _fresh_store()
    rec = BulkRerecorder(store, IntentEmbedder.get(), regression_threshold=0.75)
    page = MagicMock()
    entry = PageEntry(url="https://app", intents=[{"name": "btn", "selector": "#btn"}])
    desc = _desc("<button>Checkout</button>")
    store.store("btn", "#btn", desc, IntentEmbedder.get().embed_descriptor(desc))

    extractor = _make_extractor({"#btn": '<a href="/privacy">Privacy Policy</a>'})
    results = rec.rerecord_page(page, entry, extractor)

    assert results[0].is_regression is True


# ---------------------------------------------------------------------------
# BulkRerecorder.rerecord_page — error handling
# ---------------------------------------------------------------------------

def test_rerecord_error_on_missing_selector():
    store = _fresh_store()
    rec = BulkRerecorder(store, IntentEmbedder.get())
    page = MagicMock()
    entry = PageEntry(url="https://app", intents=[{"name": "x", "selector": "#missing"}])
    extractor = _make_extractor({})  # no selectors → raises

    results = rec.rerecord_page(page, entry, extractor)

    assert results[0].status == "error"
    assert results[0].error is not None


def test_rerecord_continues_after_error():
    store = _fresh_store()
    rec = BulkRerecorder(store, IntentEmbedder.get())
    page = MagicMock()
    entry = PageEntry(url="https://app", intents=[
        {"name": "bad", "selector": "#missing"},
        {"name": "good", "selector": "#btn"},
    ])
    extractor = _make_extractor({"#btn": "<button>Submit</button>"})

    results = rec.rerecord_page(page, entry, extractor)

    assert len(results) == 2
    assert results[0].status == "error"
    assert results[1].status == "new"


def test_rerecord_multiple_intents_on_one_page():
    store = _fresh_store()
    rec = BulkRerecorder(store, IntentEmbedder.get())
    page = MagicMock()
    entry = PageEntry(url="https://app", intents=[
        {"name": "login", "selector": "#login"},
        {"name": "signup", "selector": "#signup"},
    ])
    extractor = _make_extractor({
        "#login": "<button>Login</button>",
        "#signup": "<button>Sign Up</button>",
    })

    results = rec.rerecord_page(page, entry, extractor)

    assert len(results) == 2
    assert all(r.status == "new" for r in results)
    assert store.get("login") is not None
    assert store.get("signup") is not None


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

def test_dry_run_does_not_write_to_store():
    store = _fresh_store()
    rec = BulkRerecorder(store, IntentEmbedder.get(), dry_run=True)
    page = MagicMock()
    entry = PageEntry(url="https://app", intents=[{"name": "btn", "selector": "#btn"}])
    extractor = _make_extractor({"#btn": "<button>Pay</button>"})

    results = rec.rerecord_page(page, entry, extractor)

    assert results[0].status == "dry-run"
    assert store.get("btn") is None   # nothing written


def test_dry_run_shows_similarity_for_existing():
    store = _fresh_store()
    desc = _desc("<button>Checkout</button>")
    store.store("checkout", "#checkout", desc, IntentEmbedder.get().embed_descriptor(desc))

    rec = BulkRerecorder(store, IntentEmbedder.get(), dry_run=True)
    page = MagicMock()
    entry = PageEntry(url="https://app", intents=[{"name": "checkout", "selector": "#checkout"}])
    extractor = _make_extractor({"#checkout": "<button>Checkout</button>"})

    results = rec.rerecord_page(page, entry, extractor)

    assert results[0].status == "dry-run"
    assert results[0].old_similarity is not None


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------

def test_print_report_counts(capsys):
    results = [
        RerecordResult("a", "#a", "http://p", old_similarity=0.99, status="updated"),
        RerecordResult("b", "#b", "http://p", old_similarity=None, status="new"),
        RerecordResult("c", "#c", "http://p", old_similarity=0.40, status="updated"),
        RerecordResult("d", "#d", "http://p", old_similarity=None, status="error", error="boom"),
    ]
    updated, new_count, regressions, errors = print_report(results, regression_threshold=0.75)

    assert new_count == 1
    assert len(regressions) == 1
    assert regressions[0].intent_name == "c"
    assert len(errors) == 1


def test_print_report_dry_run_shows_flag(capsys):
    results = [
        RerecordResult("x", "#x", "http://p", old_similarity=0.40, status="dry-run"),
    ]
    print_report(results, regression_threshold=0.75)
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "would regress" in out


# ---------------------------------------------------------------------------
# CLI rerecord-all command
# ---------------------------------------------------------------------------

def test_cmd_rerecord_all_dry_run(tmp_path, capsys):
    db = str(tmp_path / "test.db")
    map_file = tmp_path / "map.json"
    map_file.write_text(json.dumps([
        {"url": "https://app", "intents": [{"name": "btn", "selector": "#btn"}]}
    ]))

    store_for_pre = IntentStore(db)
    desc = _desc("<button>Submit</button>")
    store_for_pre.store("btn", "#btn", desc, IntentEmbedder.get().embed_descriptor(desc))
    store_for_pre.close()

    # Patch run() to avoid real Playwright
    mock_results = [RerecordResult("btn", "#btn", "https://app", old_similarity=0.99, status="dry-run")]

    args = MagicMock()
    args.page_map = str(map_file)
    args.db = db
    args.model = "all-MiniLM-L6-v2"
    args.threshold = 0.75
    args.dry_run = True
    args.scrub_text = False

    from canvas_heal import batch as batch_mod
    original_run = BulkRerecorder.run
    BulkRerecorder.run = lambda self, pm, de=None: mock_results
    try:
        rc = _cmd_rerecord_all(args)
    finally:
        BulkRerecorder.run = original_run

    assert rc == 0
    out = capsys.readouterr().out
    assert "btn" in out


def test_cmd_rerecord_all_missing_page_map(tmp_path, capsys):
    args = MagicMock()
    args.page_map = str(tmp_path / "nonexistent.json")
    args.db = str(tmp_path / "test.db")
    args.model = "all-MiniLM-L6-v2"
    args.threshold = 0.75
    args.dry_run = False
    args.scrub_text = False

    rc = _cmd_rerecord_all(args)
    assert rc == 1
