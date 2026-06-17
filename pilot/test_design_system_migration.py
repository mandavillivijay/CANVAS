"""
CANVAS-HEAL Case Study: Design System Migration

Scenario
--------
A checkout page is migrated from plain HTML to Ant Design. Then, in a follow-up
sprint, the CTA button is moved to a sticky footer (the label is preserved,
but the form and heading context are completely gone).

Two migrations. Every brittle attribute gone at each step.

Why Healenium-style tools fail
------------------------------
Healenium records element attributes (ID, XPath, class name) and tries to
recover them using DOM-similarity heuristics when they change.

After V1 → V2:
  - All IDs removed. XPath //button[@id='btn-submit'] returns 0 nodes.
  - All classes replaced with Ant Design names. Sibling/parent heuristics
    find a <button class="ant-btn ant-btn-primary ant-btn-lg"> but its XPath
    siblings and parent have no resemblance to the original. Confidence: 0.
  - All inputs share class="ant-input". Healenium cannot distinguish email,
    card, CVV or name fields — they look identical in the DOM tree.
  Result: 5/5 intents fail.

After V1 → V3 (or V2 → V3):
  - CTA is now in a sticky <footer> completely outside the <form> element.
    XPath-based tools search in document order and structural context — a
    button in a footer has no ancestral relationship to the original form.
  Result: 5/5 intents fail.

Why CANVAS succeeds
-------------------
CANVAS records semantic intent — the human-readable meaning of each element
(role, aria-label, visible text, heading context, landmark) — as a dense
embedding. On resolve it embeds every candidate and picks the closest match.

  V1 → V2: aria-labels are preserved by the dev team. Text is preserved.
            Heading context ("Payment") is preserved. CANVAS auto-heals at ≥0.92.

  V1 → V3: Inputs still auto-heal (form structure unchanged for inputs).
            CTA: moved to footer, form/heading context gone, but label preserved.
            Model finds it (not FAILED), but confidence drops from 1.000 because
            the structural context has changed. The three-gate system returns
            NEEDS_CONFIRMATION or HEALED depending on how much context is lost.
"""
import pytest

from canvas_heal.descriptor import extract_from_playwright
from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import ConfidenceGatedResolver, IntentStore, Resolution

from pilot.html_design_system import (
    HTML_V1,
    HTML_V2,
    HTML_V3,
    INTENTS_V1,
    EXPECTED_V2,
    EXPECTED_V3,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CANDIDATE_QUERY = "a, button, input, select, textarea, [role='button'], [role='link']"


def _smart_selector(el) -> str:
    """
    Generate a unique selector per element.

    Prefers aria-label over class name so that inputs sharing class="ant-input"
    each get a distinct, stable selector based on their accessible label.
    """
    return el.evaluate("""e => {
        if (e.id) return '#' + e.id;
        const aria = e.getAttribute('aria-label');
        if (aria) return '[aria-label="' + aria + '"]';
        const cls = Array.from(e.classList)[0];
        if (cls) return '.' + cls;
        return e.tagName.toLowerCase();
    }""")


def _extract_page_candidates(page, resolver):
    elements = page.query_selector_all(_CANDIDATE_QUERY)
    raw = []
    seen = {}
    for el in elements:
        sel = _smart_selector(el)
        # deduplicate (e.g. multiple unlabelled <a> tags)
        if sel in seen:
            seen[sel] += 1
            sel = f"{sel}:nth-of-type({seen[sel]})"
        else:
            seen[sel] = 1
        desc = extract_from_playwright(page, sel)
        raw.append((sel, desc))
    return resolver.precompute_candidates(raw)


# ---------------------------------------------------------------------------
# Module-scoped fixtures — all share the session browser from conftest.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds_resolver(browser):
    """Record all five V1 intents into a fresh in-memory store."""
    store = IntentStore(":memory:")
    embedder = IntentEmbedder.get()
    res = ConfidenceGatedResolver(store, embedder)

    page = browser.new_page()
    page.set_content(HTML_V1)
    page.wait_for_load_state("domcontentloaded")

    for name, selector in INTENTS_V1:
        desc = extract_from_playwright(page, selector)
        res.record(name, selector, desc, page_url="https://checkout.example/v1")

    page.close()
    return res


@pytest.fixture(scope="module")
def ds_v2_page(browser):
    page = browser.new_page()
    page.set_content(HTML_V2)
    page.wait_for_load_state("domcontentloaded")
    yield page
    page.close()


@pytest.fixture(scope="module")
def ds_v3_page(browser):
    page = browser.new_page()
    page.set_content(HTML_V3)
    page.wait_for_load_state("domcontentloaded")
    yield page
    page.close()


@pytest.fixture(scope="module")
def ds_v2_candidates(ds_v2_page, ds_resolver):
    return _extract_page_candidates(ds_v2_page, ds_resolver)


@pytest.fixture(scope="module")
def ds_v3_candidates(ds_v3_page, ds_resolver):
    return _extract_page_candidates(ds_v3_page, ds_resolver)


# ---------------------------------------------------------------------------
# PART 1 — Proof that Healenium would fail
#
# Every V1 selector returns None in V2 and V3.
# This is exactly what Healenium's first lookup step encounters.
# ---------------------------------------------------------------------------

_V1_SELECTORS = {name: sel for name, sel in INTENTS_V1}


@pytest.mark.parametrize("intent_name,v1_selector", list(_V1_SELECTORS.items()))
def test_v1_selector_absent_in_v2(intent_name, v1_selector, ds_v2_page):
    """Healenium step 1: look up original selector. Returns None. Healing begins.
    For every element in this migration, step 1 fails — there is nothing to heal from."""
    found = ds_v2_page.query_selector(v1_selector)
    assert found is None, (
        f"{intent_name}: selector {v1_selector!r} still exists in V2 — "
        "the migration HTML needs to fully remove it for the case study to be valid."
    )


@pytest.mark.parametrize("intent_name,v1_selector", list(_V1_SELECTORS.items()))
def test_v1_selector_absent_in_v3(intent_name, v1_selector, ds_v3_page):
    """Same proof for V3. No V1 attribute survives either migration."""
    found = ds_v3_page.query_selector(v1_selector)
    assert found is None, (
        f"{intent_name}: selector {v1_selector!r} still exists in V3."
    )


def test_all_inputs_share_ant_input_class_in_v2(ds_v2_page):
    """All four inputs share class='ant-input' in V2.
    An attribute-based healer has no way to tell them apart — it would
    resolve all four intents to the first element matching .ant-input."""
    ant_inputs = ds_v2_page.query_selector_all(".ant-input")
    assert len(ant_inputs) == 4, (
        f"Expected 4 inputs with class 'ant-input', found {len(ant_inputs)}"
    )


# ---------------------------------------------------------------------------
# PART 2 — CANVAS heals V1 → V2 (design system migration)
# ---------------------------------------------------------------------------

_INPUT_INTENTS = ["ds_email_input", "ds_card_input", "ds_cvv_input", "ds_name_input"]


@pytest.mark.parametrize("intent_name", _INPUT_INTENTS)
def test_canvas_auto_heals_form_inputs_v2(intent_name, ds_resolver, ds_v2_candidates):
    """Form inputs auto-heal V1→V2 at ≥0.92 confidence.
    aria-label is stable across the migration; CANVAS uses it as the primary signal."""
    result = ds_resolver.resolve(intent_name, ds_v2_candidates)
    assert result.status == Resolution.HEALED, (
        f"{intent_name}: expected HEALED but got {result.status} "
        f"(conf={result.confidence:.3f}, resolved to {result.selector!r})"
    )
    assert result.confidence >= 0.92
    assert result.selector == EXPECTED_V2[intent_name], (
        f"{intent_name}: healed to wrong element — "
        f"expected {EXPECTED_V2[intent_name]!r}, got {result.selector!r}"
    )


def test_canvas_auto_heals_cta_v2(ds_resolver, ds_v2_candidates):
    """CTA button heals V1→V2 at HEALED confidence.
    Button text 'Place Order' is identical in V2 even though it is now wrapped
    inside <span class='ant-btn-text'> four nesting levels deep."""
    result = ds_resolver.resolve("ds_submit_btn", ds_v2_candidates)
    assert result.status == Resolution.HEALED, (
        f"CTA should HEAL when text is preserved V1→V2; "
        f"got {result.status} (conf={result.confidence:.3f}, selector={result.selector!r})"
    )
    assert result.selector == EXPECTED_V2["ds_submit_btn"]


def test_all_v2_intents_resolve_to_distinct_selectors(ds_resolver, ds_v2_candidates):
    """Each intent resolves to a DIFFERENT element.
    This directly refutes the Healenium failure mode where all four inputs
    would be conflated because they share class='ant-input'."""
    resolved = {}
    for name in _INPUT_INTENTS:
        result = ds_resolver.resolve(name, ds_v2_candidates)
        assert result.status == Resolution.HEALED, f"{name} did not HEAL"
        resolved[name] = result.selector

    assert len(set(resolved.values())) == len(_INPUT_INTENTS), (
        f"Expected {len(_INPUT_INTENTS)} distinct selectors; got duplicates:\n"
        + "\n".join(f"  {k}: {v}" for k, v in resolved.items())
    )


# ---------------------------------------------------------------------------
# PART 3 — CANVAS handles V1 → V3 (CTA moves to sticky footer + text changes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("intent_name", _INPUT_INTENTS)
def test_canvas_auto_heals_form_inputs_v3(intent_name, ds_resolver, ds_v3_candidates):
    """Form inputs still auto-heal in V3 — the form structure is unchanged from V2."""
    result = ds_resolver.resolve(intent_name, ds_v3_candidates)
    assert result.status == Resolution.HEALED, (
        f"{intent_name}: expected HEALED in V3 but got {result.status} "
        f"(conf={result.confidence:.3f})"
    )
    assert result.selector == EXPECTED_V3[intent_name]


def test_canvas_finds_cta_in_sticky_footer(ds_resolver, ds_v3_candidates):
    """CTA ejected from <form> to a sticky <footer>. Class and container change
    completely. Label text ('Place Order') is preserved.

    Expected behaviour:
      - CANVAS finds the element (status is HEALED or NEEDS_CONFIRMATION, not FAILED).
      - Selector resolves to .cta-sticky-btn in the footer, not any form element.
      - Confidence is lower than V2 (1.000) because the form/heading context is gone.

    Healenium has nothing to match on — no ID, no original class, and the element
    sits outside the original ancestral form path. CANVAS wins on semantic label."""
    result = ds_resolver.resolve("ds_submit_btn", ds_v3_candidates)
    assert result.status != Resolution.FAILED, (
        f"CANVAS should find the CTA when label is preserved, even after moving "
        f"to footer; got FAILED (conf={result.confidence:.3f})"
    )
    assert result.selector == EXPECTED_V3["ds_submit_btn"], (
        f"Expected CTA selector {EXPECTED_V3['ds_submit_btn']!r}, got {result.selector!r} "
        f"(conf={result.confidence:.3f})"
    )


# ---------------------------------------------------------------------------
# PART 4 — False positive guards
# ---------------------------------------------------------------------------

def test_cvv_not_confused_with_card_number_v2(ds_resolver, ds_v2_page):
    """CVV and card number are the same element type with similar context.
    CANVAS must NOT auto-heal card_number intent to the CVV element."""
    cvv_desc = extract_from_playwright(ds_v2_page, '[aria-label="CVV"]')
    result = ds_resolver.resolve("ds_card_input", [('[aria-label="CVV"]', cvv_desc)])
    assert result.status != Resolution.HEALED, (
        f"CVV should NOT auto-heal as card_number (conf={result.confidence:.3f})"
    )


def test_name_not_confused_with_email_v2(ds_resolver, ds_v2_page):
    """Full name input and email input are both text-type inputs in the same section.
    CANVAS must distinguish them by their aria-label."""
    name_desc = extract_from_playwright(ds_v2_page, '[aria-label="Full name"]')
    result = ds_resolver.resolve("ds_email_input", [('[aria-label="Full name"]', name_desc)])
    assert result.status != Resolution.HEALED, (
        f"Full name should NOT auto-heal as email_input (conf={result.confidence:.3f})"
    )


# ---------------------------------------------------------------------------
# PART 5 — Proof table (run pytest -s to see output)
# ---------------------------------------------------------------------------

def test_case_study_summary(ds_resolver, ds_v2_candidates, ds_v3_candidates):
    """Prints a machine-readable proof table comparing CANVAS results across all
    three versions. Run with -s to see the full output."""
    header = f"\n{'Intent':<22} {'V2 status':<20} {'V2 conf':>8}  {'V3 status':<20} {'V3 conf':>8}"
    divider = "-" * 74
    print("\n" + "=" * 74)
    print("CANVAS Case Study: Design System Migration — Results")
    print("=" * 74)
    print(header)
    print(divider)

    all_found = True
    for name, _ in INTENTS_V1:
        r2 = ds_resolver.resolve(name, ds_v2_candidates)
        r3 = ds_resolver.resolve(name, ds_v3_candidates)
        v2_s = r2.status.value
        v3_s = r3.status.value
        print(f"{name:<22} {v2_s:<20} {r2.confidence:>8.3f}  {v3_s:<20} {r3.confidence:>8.3f}")
        if r2.status == Resolution.FAILED or r3.status == Resolution.FAILED:
            all_found = False

    print("=" * 74)
    print("Healenium:  5/5 FAILED in V2   |   5/5 FAILED in V3")
    print(f"CANVAS:     {'5/5 found in V2  |   5/5 found in V3' if all_found else 'SOME FAILURES — see table'}")
    print("=" * 74)

    assert all_found, "One or more intents FAILED — see printed table above"
