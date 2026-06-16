# Using CANVAS in a pytest suite

Reference for wiring CANVAS self-healing locators into a real pytest test suite.

## Session fixture

Load the model once per run via a session-scoped fixture. The intent DB lives at a
project-level path so the recorded baseline travels with the suite.

```python
# conftest.py
import pytest
from canvas.embedder import IntentEmbedder
from canvas.resolver import ConfidenceGatedResolver, IntentStore

@pytest.fixture(scope="session")
def canvas_resolver(tmp_path_factory):
    db = tmp_path_factory.mktemp("canvas") / "intents.db"
    store = IntentStore(db)
    embedder = IntentEmbedder.get()
    return ConfidenceGatedResolver(store, embedder)
```

For a committed baseline, point `IntentStore` at a fixed path (e.g.
`tests/intents.db`) instead of `tmp_path_factory`.

## Recording pass

Run a one-off script against a live URL to capture intents. Re-run only when the
baseline UI changes.

```python
from playwright.sync_api import sync_playwright
from canvas.descriptor import extract_from_playwright
from canvas.resolver import ConfidenceGatedResolver, IntentStore
from canvas.embedder import IntentEmbedder

resolver = ConfidenceGatedResolver(IntentStore("tests/intents.db"), IntentEmbedder.get())
with sync_playwright() as pw:
    page = pw.chromium.launch(headless=True).new_page()
    page.goto("https://app.example.com/checkout")
    resolver.record("checkout_submit", "#btn-submit",
                    extract_from_playwright(page, "#btn-submit"))
```

## Resolving in a test

Call `resolve()` with the live candidates and branch on `Resolution` status.

```python
def test_checkout(canvas_resolver, page):
    candidates = [(sel, extract_from_playwright(page, sel))
                  for sel in page.eval_on_selector_all("button,a", "els => els.map(e => ...)")]
    result = canvas_resolver.resolve("checkout_submit", candidates)

    if result.status == Resolution.HEALED:
        selector = result.selector  # use the healed selector
    elif result.status == Resolution.NEEDS_CONFIRMATION:
        pytest.skip(f"manual review: {result.message} (conf={result.confidence:.3f})")
    else:  # FAILED
        raise AssertionError(f"intent unresolved: {result.message}")

    page.click(selector)
```

## Performance

When resolving multiple intents against the same V2 DOM, call
`resolver.precompute_candidates(candidates)` once per page load to embed the
candidate set a single time, then `resolve()` each intent against the cache.

## Committing the baseline

Commit the `.db` file alongside the suite so the recorded baseline is versioned
with the tests. CI then resolves against a known-good reference rather than
re-recording on every run.
