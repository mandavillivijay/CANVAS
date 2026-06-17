"""
CANVAS Demo — end-to-end pipeline: Record → Simulate redesign → Resolve

Run with:
    python demo.py

No live server needed. Both UI versions are served via page.set_content().
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

from canvas_heal.descriptor import extract_from_playwright
from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import ConfidenceGatedResolver, IntentStore, Resolution

# ---------------------------------------------------------------------------
# Inline HTML — V1 and V2 of the same UI, different structure / IDs
# ---------------------------------------------------------------------------

HTML_V1 = """
<html><body>
  <main>
    <h2>Checkout</h2>
    <form id="checkout-form">
      <input type="email" id="email" placeholder="Your email address" aria-label="Email">
      <button id="btn-place-order" class="btn-primary" aria-label="Place your order">
        Place Order
      </button>
    </form>
  </main>
</body></html>
"""

HTML_V2 = """
<html><body>
  <div role="dialog" aria-label="Order confirmation">
    <h2>Confirm Your Order</h2>
    <input type="email" id="email-field" placeholder="Enter your email" aria-label="Email address">
    <button id="cta-confirm" class="cta-button confirm-action">
      Confirm &amp; Place Order
    </button>
  </div>
</body></html>
"""

# Intent names — used as stable keys across UI versions
INTENTS = [
    ("order_button", "#btn-place-order"),
    ("email_input", "#email"),
]

SEPARATOR = "=" * 64


def _print_separator(title: str = "") -> None:
    if title:
        pad = (64 - len(title) - 2) // 2
        print(f"\n{'=' * pad} {title} {'=' * (64 - pad - len(title) - 2)}\n")
    else:
        print(f"\n{SEPARATOR}\n")


def main() -> None:
    embedder = IntentEmbedder.get()  # load model once; singleton

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()

        # ------------------------------------------------------------------ #
        # STEP A — Record phase (V1 of the UI)
        # ------------------------------------------------------------------ #
        _print_separator("STEP A — RECORD PHASE (V1 UI)")

        page.set_content(HTML_V1)

        store = IntentStore(":memory:")
        resolver = ConfidenceGatedResolver(store, embedder=embedder)

        for name, selector in INTENTS:
            descriptor = extract_from_playwright(page, selector)
            resolver.record(name, selector, descriptor)
            print(f"[RECORDED]  intent='{name}'")
            print(f"            selector='{selector}'")
            print(f"            descriptor: {descriptor.to_text()}")
            print()

        # ------------------------------------------------------------------ #
        # STEP B — Simulate UI redesign (V2 of the UI)
        # ------------------------------------------------------------------ #
        _print_separator("STEP B — UI REDESIGN SIMULATION")

        print("Simulating UI redesign — IDs and structure changed.")
        print("V1 selectors (#btn-place-order, #email) no longer exist in V2.")
        print()

        page.set_content(HTML_V2)

        # Build candidate list: all buttons and inputs in V2
        candidate_elements = page.query_selector_all(
            "a, button, input, select, textarea, "
            "[role='button'], [role='link'], [role='checkbox'], "
            "[role='menuitem'], [role='tab'], [role='option']"
        )
        candidates = []
        for el in candidate_elements:
            sel = el.evaluate("""e => {
    if (e.id) return '#' + e.id;
    const tag = e.tagName.toLowerCase();
    const parent = e.parentElement;
    if (!parent) return tag;
    const siblings = Array.from(parent.children).filter(c => c.tagName === e.tagName);
    if (siblings.length === 1) return tag;
    const idx = siblings.indexOf(e) + 1;
    return tag + ':nth-of-type(' + idx + ')';
}""")
            desc = extract_from_playwright(page, sel)
            candidates.append((sel, desc))
            print(f"[CANDIDATE] selector='{sel}'  ->  {desc.to_text()}")

        # ------------------------------------------------------------------ #
        # STEP C — Resolve phase
        # ------------------------------------------------------------------ #
        _print_separator("STEP C — RESOLVE PHASE")

        any_failed = False

        for name, _ in INTENTS:
            result = resolver.resolve(name, candidates)

            status_label = result.status.value.upper()
            print(f"Intent      : '{name}'")
            print(f"Status      : {status_label}")
            print(f"Confidence  : {result.confidence:.3f}")
            print(f"Message     : {result.message}")
            if result.selector:
                print(f"Resolved to : '{result.selector}'")
            print()

            if result.status == Resolution.FAILED:
                any_failed = True

        _print_separator()

        browser.close()

    if any_failed:
        print("[RESULT] One or more intents could not be resolved. Exiting with code 1.")
        sys.exit(1)
    else:
        print("[RESULT] All intents resolved successfully.")


if __name__ == "__main__":
    main()
