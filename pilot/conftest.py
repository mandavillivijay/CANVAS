import pytest
from playwright.sync_api import sync_playwright

from canvas_heal.descriptor import extract_from_playwright
from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import ConfidenceGatedResolver, IntentStore

from pilot.html import HTML_V1, HTML_V2, INTENTS_V1


@pytest.fixture(scope="session")
def _pw():
    with sync_playwright() as pw:
        yield pw


@pytest.fixture(scope="session")
def browser(_pw):
    b = _pw.chromium.launch(headless=True)
    yield b
    b.close()


@pytest.fixture(scope="session")
def resolver(browser):
    """Record all V1 intents once for the whole session."""
    store = IntentStore(":memory:")
    embedder = IntentEmbedder.get()
    res = ConfidenceGatedResolver(store, embedder)

    page = browser.new_page()
    page.set_content(HTML_V1)
    page.wait_for_load_state("domcontentloaded")

    for name, selector in INTENTS_V1:
        desc = extract_from_playwright(page, selector)
        res.record(name, selector, desc, page_url="https://shopeasy.example/checkout")
        print(f"  [recorded] {name} → {desc.to_text()}")

    page.close()
    return res


@pytest.fixture(scope="session")
def v2_candidates(browser, resolver):
    """Extract and pre-embed all interactive V2 elements once for the session."""
    page = browser.new_page()
    page.set_content(HTML_V2)
    page.wait_for_load_state("domcontentloaded")

    elements = page.query_selector_all(
        "a, button, input, select, textarea, [role='button'], [role='link']"
    )
    raw = []
    for el in elements:
        sel = el.evaluate("""e => {
            if (e.id) return '#' + e.id;
            const cls = Array.from(e.classList)[0];
            if (cls) return '.' + cls;
            return e.tagName.toLowerCase();
        }""")
        desc = extract_from_playwright(page, sel)
        raw.append((sel, desc))

    page.close()
    return resolver.precompute_candidates(raw)


@pytest.fixture(scope="session")
def v2_page(browser):
    """A V2 page for targeted single-element extractions."""
    page = browser.new_page()
    page.set_content(HTML_V2)
    page.wait_for_load_state("domcontentloaded")
    yield page
    page.close()
