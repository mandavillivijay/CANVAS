"""
Adversarial test suite for CANVAS semantic intent-based self-healing locators.

Each test:
  1. Records an intent from an element in HTML_V1 (original DOM).
  2. Presents candidates extracted from HTML_V2 (modified DOM).
  3. Asserts the resolver heals / rejects correctly.
"""
from __future__ import annotations

import pytest
from playwright.sync_api import sync_playwright

from canvas.descriptor import extract_from_playwright
from canvas.embedder import IntentEmbedder
from canvas.resolver import ConfidenceGatedResolver, IntentStore, Resolution


# ---------------------------------------------------------------------------
# Module-scoped fixtures — model and browser load only once for the whole file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def resolver():
    store = IntentStore(":memory:")
    embedder = IntentEmbedder.get()
    yield ConfidenceGatedResolver(store, embedder)


@pytest.fixture(scope="module")
def _browser():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def browser_page(_browser):
    page = _browser.new_page()
    yield page
    page.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _load(page, html: str) -> None:
    page.set_content(html)
    page.wait_for_load_state("domcontentloaded")


# ---------------------------------------------------------------------------
# Scenario 1 — ID and class rename
# ---------------------------------------------------------------------------

HTML_S1_V1 = """
<form id="checkout-form">
  <h2>Checkout</h2>
  <button id="btn-submit" class="btn-primary">Place Order</button>
</form>
"""

HTML_S1_V2 = """
<form id="checkout-form">
  <h2>Checkout</h2>
  <button id="cta-confirm" class="cta-button">Place Order</button>
</form>
"""


def test_id_class_rename_heals(browser_page, resolver):
    """Button with same text/context but renamed id + class must heal."""
    _load(browser_page, HTML_S1_V1)
    desc_v1 = extract_from_playwright(browser_page, "#btn-submit")
    resolver.record("s1_submit_btn", "#btn-submit", desc_v1)

    _load(browser_page, HTML_S1_V2)
    candidate_desc = extract_from_playwright(browser_page, "#cta-confirm")
    candidates = [("#cta-confirm", candidate_desc)]

    result = resolver.resolve("s1_submit_btn", candidates)
    assert result.status == Resolution.HEALED, (
        f"Expected HEALED, got {result.status} (confidence={result.confidence:.3f}): {result.message}"
    )


# ---------------------------------------------------------------------------
# Scenario 2 — Label paraphrase
# ---------------------------------------------------------------------------

HTML_S2_V1 = """
<form>
  <h2>Account Settings</h2>
  <button aria-label="Save your changes">Save</button>
</form>
"""

HTML_S2_V2 = """
<form>
  <h2>Account Settings</h2>
  <button aria-label="Apply and save changes">Update</button>
</form>
"""


def test_label_paraphrase_resolves(browser_page, resolver):
    """Paraphrased aria-label must resolve to HEALED or NEEDS_CONFIRMATION."""
    _load(browser_page, HTML_S2_V1)
    desc_v1 = extract_from_playwright(browser_page, "button[aria-label='Save your changes']")
    resolver.record("s2_save_btn", "button[aria-label='Save your changes']", desc_v1)

    _load(browser_page, HTML_S2_V2)
    candidate_desc = extract_from_playwright(
        browser_page, "button[aria-label='Apply and save changes']"
    )
    candidates = [("button[aria-label='Apply and save changes']", candidate_desc)]

    result = resolver.resolve("s2_save_btn", candidates)
    assert result.status in (Resolution.HEALED, Resolution.NEEDS_CONFIRMATION), (
        f"Expected HEALED or NEEDS_CONFIRMATION, got {result.status} "
        f"(confidence={result.confidence:.3f}): {result.message}"
    )


# ---------------------------------------------------------------------------
# Scenario 3 — DOM restructuring (element moved to modal)
# ---------------------------------------------------------------------------

HTML_S3_V1 = """
<main>
  <h2>Shopping Cart</h2>
  <button id="checkout-btn">Proceed to Checkout</button>
</main>
"""

HTML_S3_V2 = """
<div role="dialog" aria-label="Confirm order">
  <button id="checkout-modal-btn">Proceed to Checkout</button>
</div>
"""


def test_dom_restructuring_heals(browser_page, resolver):
    """Button with identical text moved to a dialog must heal or need confirmation."""
    _load(browser_page, HTML_S3_V1)
    desc_v1 = extract_from_playwright(browser_page, "#checkout-btn")
    resolver.record("s3_checkout_btn", "#checkout-btn", desc_v1)

    _load(browser_page, HTML_S3_V2)
    candidate_desc = extract_from_playwright(browser_page, "#checkout-modal-btn")
    candidates = [("#checkout-modal-btn", candidate_desc)]

    result = resolver.resolve("s3_checkout_btn", candidates)
    assert result.status in (Resolution.HEALED, Resolution.NEEDS_CONFIRMATION), (
        f"Expected HEALED or NEEDS_CONFIRMATION, got {result.status} "
        f"(confidence={result.confidence:.3f}): {result.message}"
    )


# ---------------------------------------------------------------------------
# Scenario 4 — False positive rejection
# ---------------------------------------------------------------------------

HTML_S4_V1 = """
<form>
  <h2>Login</h2>
  <button type="submit">Sign In</button>
</form>
"""

HTML_S4_V2 = """
<a href="/terms">Terms and Conditions</a>
"""


def test_false_positive_rejected(browser_page, resolver):
    """Completely unrelated element must NOT auto-heal (confidence < 0.92)."""
    _load(browser_page, HTML_S4_V1)
    desc_v1 = extract_from_playwright(browser_page, "button[type=submit]")
    resolver.record("s4_signin_btn", "button[type=submit]", desc_v1)

    _load(browser_page, HTML_S4_V2)
    candidate_desc = extract_from_playwright(browser_page, "a[href='/terms']")
    candidates = [("a[href='/terms']", candidate_desc)]

    result = resolver.resolve("s4_signin_btn", candidates)
    assert result.status != Resolution.HEALED, (
        f"Expected NOT HEALED, got {result.status} (confidence={result.confidence:.3f})"
    )
    assert result.confidence < 0.92, (
        f"Confidence too high for unrelated element: {result.confidence:.3f}"
    )


# ---------------------------------------------------------------------------
# Scenario 5 — Best match from mixed candidates
# ---------------------------------------------------------------------------

HTML_S5_V1 = """
<nav>
  <a href="/dashboard">Go to Dashboard</a>
</nav>
"""

HTML_S5_V2 = """
<footer><a href="/legal">© 2026</a></footer>
<nav><a href="/app/dashboard">Dashboard</a></nav>
<button>Help</button>
"""


def test_best_match_from_mixed_candidates(browser_page, resolver):
    """Resolver must pick the dashboard link over footer and help button."""
    _load(browser_page, HTML_S5_V1)
    desc_v1 = extract_from_playwright(browser_page, "nav a")
    resolver.record("s5_dashboard_link", "nav a", desc_v1)

    _load(browser_page, HTML_S5_V2)
    candidates = [
        ("footer a[href='/legal']", extract_from_playwright(browser_page, "footer a[href='/legal']")),
        ("nav a[href='/app/dashboard']", extract_from_playwright(browser_page, "nav a[href='/app/dashboard']")),
        ("button", extract_from_playwright(browser_page, "button")),
    ]

    result = resolver.resolve("s5_dashboard_link", candidates)
    # The key guarantee: correct candidate wins. Whether it auto-heals or needs
    # confirmation depends on exact similarity scores near the 0.92 boundary.
    assert result.status in (Resolution.HEALED, Resolution.NEEDS_CONFIRMATION), (
        f"Expected HEALED or NEEDS_CONFIRMATION, got {result.status} "
        f"(confidence={result.confidence:.3f}): {result.message}"
    )
    assert result.selector == "nav a[href='/app/dashboard']", (
        f"Expected dashboard link to win, got: {result.selector!r} (confidence={result.confidence:.3f})"
    )
    assert result.confidence > 0.85, (
        f"Confidence for correct candidate too low: {result.confidence:.3f}"
    )
