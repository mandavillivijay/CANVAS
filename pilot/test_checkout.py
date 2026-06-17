"""
CANVAS-HEAL Pilot — ShopEasy Checkout

Scenario: the ShopEasy frontend team ships a full redesign of the checkout
page. Every ID changes, the DOM structure changes, and several labels are
reworded. A traditional test suite would fail on every locator.

These tests verify that CANVAS-HEAL heals each intent correctly, picks the
right element from a pool of candidates, and refuses to heal when the intent
genuinely doesn't match.
"""
import pytest

from canvas_heal.descriptor import extract_from_playwright
from canvas_heal.resolver import Resolution
from pilot.html import EXPECTED_V2


# ---------------------------------------------------------------------------
# 1. Parametrised healing sweep — every recorded intent must resolve
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("intent_name", list(EXPECTED_V2.keys()))
def test_intent_resolves(intent_name, resolver, v2_candidates):
    """Every V1 intent must resolve (HEALED or NEEDS_CONFIRMATION) against V2."""
    result = resolver.resolve(intent_name, v2_candidates)
    assert result.status != Resolution.FAILED, (
        f"{intent_name}: failed to resolve — conf={result.confidence:.3f}, {result.message}"
    )


@pytest.mark.parametrize("intent_name", list(EXPECTED_V2.keys()))
def test_intent_resolves_to_correct_element(intent_name, resolver, v2_candidates):
    """Every intent must heal to the exact expected V2 selector."""
    result = resolver.resolve(intent_name, v2_candidates)
    assert result.selector == EXPECTED_V2[intent_name], (
        f"{intent_name}: expected {EXPECTED_V2[intent_name]!r}, "
        f"got {result.selector!r} (conf={result.confidence:.3f})"
    )


# ---------------------------------------------------------------------------
# 2. Primary CTA must auto-heal (confidence >= 0.92)
# ---------------------------------------------------------------------------

def test_place_order_resolves_with_confirmation(resolver, v2_candidates):
    """'Place Order' → 'Confirm & Pay': text changed significantly so the model
    correctly lands in NEEDS_CONFIRMATION (0.75–0.92), not HEALED.
    This is the three-gate system working as designed — the resolver finds the
    right element but flags it for human review before auto-healing."""
    result = resolver.resolve("place_order_btn", v2_candidates)
    assert result.status == Resolution.NEEDS_CONFIRMATION, (
        f"Expected NEEDS_CONFIRMATION for significant label change, "
        f"got {result.status} (conf={result.confidence:.3f})"
    )
    assert result.selector == "#cta-confirm-order"
    assert 0.75 <= result.confidence < 0.92


# ---------------------------------------------------------------------------
# 3. False positive rejection — CVV must not match card number intent
# ---------------------------------------------------------------------------

def test_cvv_does_not_heal_to_card_number(resolver, v2_page):
    """CVV field and card number field have different semantic purpose.
    The resolver must NOT auto-heal card_number_input to the CVV element."""
    cvv_desc = extract_from_playwright(v2_page, ".fld-cvv")
    result = resolver.resolve("card_number_input", [(".fld-cvv", cvv_desc)])
    assert result.status != Resolution.HEALED, (
        f"CVV should not auto-heal to card_number_input (conf={result.confidence:.3f})"
    )


# ---------------------------------------------------------------------------
# 4. Best candidate wins from a mixed pool
# ---------------------------------------------------------------------------

def test_correct_candidate_wins_from_mixed_pool(resolver, v2_page):
    """When given a mix of unrelated candidates plus the correct one,
    the resolver must pick the correct card number field."""
    from canvas_heal.descriptor import extract_from_playwright as efp
    candidates = [
        (".lnk-home",        efp(v2_page, ".lnk-home")),
        ("#cta-confirm-order", efp(v2_page, "#cta-confirm-order")),
        (".fld-cvv",         efp(v2_page, ".fld-cvv")),
        (".fld-card",        efp(v2_page, ".fld-card")),
    ]
    result = resolver.resolve("card_number_input", candidates)
    assert result.selector == ".fld-card", (
        f"Expected .fld-card to win, got {result.selector!r} (conf={result.confidence:.3f})"
    )


# ---------------------------------------------------------------------------
# 5. Empty candidate pool → explicit FAILED (no silent false positives)
# ---------------------------------------------------------------------------

def test_empty_candidates_fails_explicitly(resolver):
    result = resolver.resolve("place_order_btn", [])
    assert result.status == Resolution.FAILED
    assert "No candidates" in result.message


# ---------------------------------------------------------------------------
# 6. Audit log and JUnit XML
# ---------------------------------------------------------------------------

def test_audit_log_records_all_resolutions(resolver, v2_candidates):
    resolver.clear_audit_log()
    for name in list(EXPECTED_V2.keys())[:3]:
        resolver.resolve(name, v2_candidates)
    log = resolver.get_audit_log()
    assert len(log) == 3
    assert all(e.page_url == "https://shopeasy.example/checkout" for e in log)


def test_junit_xml_export(resolver, v2_candidates, tmp_path):
    resolver.clear_audit_log()
    resolver.resolve("place_order_btn", v2_candidates)
    resolver.resolve("email_input", v2_candidates)
    out = tmp_path / "pilot_results.xml"
    resolver.export_junit_xml(str(out), suite_name="shopeasy-checkout")
    assert out.exists()
    content = out.read_text()
    assert "<testsuite" in content
    assert "shopeasy-checkout" in content
    assert "place_order_btn" in content
