import json

import pytest
from bs4 import BeautifulSoup

from canvas_heal.descriptor import SemanticDescriptor, extract_from_tag
from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import (
    ConfidenceGatedResolver,
    IntentStore,
    Resolution,
    _scrub,
    _scrub_descriptor_dict,
    _PII_PATTERNS,
)


def _tag(html: str):
    return BeautifulSoup(html, "html.parser").find(True)


@pytest.fixture(scope="module")
def resolver():
    store = IntentStore(":memory:")
    return ConfidenceGatedResolver(store, IntentEmbedder.get())


def test_exact_match_heals(resolver):
    desc = extract_from_tag(_tag("<button>Add to Cart</button>"))
    resolver.record("add_to_cart", "button.add-to-cart", desc)

    result = resolver.resolve("add_to_cart", [
        ("button.add-to-cart", extract_from_tag(_tag("<button>Add to Cart</button>"))),
    ])
    assert result.status == Resolution.HEALED
    assert result.confidence > 0.99


def test_semantic_paraphrase_resolves(resolver):
    desc = extract_from_tag(_tag('<button aria-label="Submit your order">Confirm</button>'))
    resolver.record("submit_order", "#submit-btn", desc)

    result = resolver.resolve("submit_order", [
        ("#submit-v2", extract_from_tag(_tag('<button aria-label="Submit your order now">Go</button>'))),
    ])
    assert result.status in (Resolution.HEALED, Resolution.NEEDS_CONFIRMATION)
    assert result.confidence > 0.70


def test_best_of_multiple_candidates_wins(resolver):
    desc = extract_from_tag(_tag('<input type="search" placeholder="Search products">'))
    resolver.record("search_box", "input[type=search]", desc)

    result = resolver.resolve("search_box", [
        ("#login-btn", extract_from_tag(_tag("<button>Login</button>"))),
        ("#search", extract_from_tag(_tag('<input type="search" placeholder="Search products">'))),
        ("#about", extract_from_tag(_tag('<a href="/about">About Us</a>'))),
    ])
    assert result.status == Resolution.HEALED
    assert result.selector == "#search"


def test_unrelated_element_does_not_auto_heal(resolver):
    desc = extract_from_tag(_tag("<button>Checkout</button>"))
    resolver.record("checkout_btn", "#checkout", desc)

    result = resolver.resolve("checkout_btn", [
        ("#privacy", extract_from_tag(_tag('<a href="/privacy">Privacy Policy</a>'))),
    ])
    assert result.status != Resolution.HEALED
    assert result.confidence < 0.92


def test_missing_intent_fails(resolver):
    result = resolver.resolve("nonexistent_intent", [])
    assert result.status == Resolution.FAILED
    assert "No intent stored" in result.message


def test_empty_candidates_fails(resolver):
    desc = extract_from_tag(_tag("<button>Login</button>"))
    resolver.record("login_btn", "#login", desc)

    result = resolver.resolve("login_btn", [])
    assert result.status == Resolution.FAILED
    assert "No candidates" in result.message


def _fresh_resolver():
    store = IntentStore(":memory:")
    return ConfidenceGatedResolver(store, IntentEmbedder.get())


def test_page_url_stored_and_retrieved():
    store = IntentStore(":memory:")
    res = ConfidenceGatedResolver(store, IntentEmbedder.get())
    desc = extract_from_tag(_tag("<button>Save</button>"))
    res.record("save_btn", "#save", desc, page_url="https://app.example/settings")

    stored = store.get("save_btn")
    assert stored is not None
    selector, descriptor, embedding, page_url = stored
    assert page_url == "https://app.example/settings"


def test_record_with_page_url():
    res = _fresh_resolver()
    desc = extract_from_tag(_tag("<button>Delete</button>"))
    res.record("delete_btn", "#delete", desc, page_url="https://app.example/list")


def test_audit_log_captures_healed_event():
    res = _fresh_resolver()
    desc = extract_from_tag(_tag("<button>Add to Cart</button>"))
    res.record("cart", "button.add-to-cart", desc, page_url="https://shop/cart")

    res.resolve("cart", [
        ("button.add-to-cart", extract_from_tag(_tag("<button>Add to Cart</button>"))),
    ])
    log = res.get_audit_log()
    assert len(log) == 1
    assert log[0].status == Resolution.HEALED
    assert log[0].intent_name == "cart"
    assert log[0].original_selector == "button.add-to-cart"
    assert log[0].page_url == "https://shop/cart"


def test_audit_log_captures_failed_event():
    res = _fresh_resolver()
    desc = extract_from_tag(_tag("<button>Checkout</button>"))
    res.record("checkout", "#checkout", desc)

    res.resolve("checkout", [])
    log = res.get_audit_log()
    assert len(log) == 1
    assert log[0].status == Resolution.FAILED


def test_clear_audit_log():
    res = _fresh_resolver()
    desc = extract_from_tag(_tag("<button>Login</button>"))
    res.record("login", "#login", desc)
    res.resolve("login", [])
    assert len(res.get_audit_log()) == 1

    res.clear_audit_log()
    assert res.get_audit_log() == []


def test_export_junit_xml_creates_file(tmp_path):
    res = _fresh_resolver()
    desc = extract_from_tag(_tag("<button>Add to Cart</button>"))
    res.record("cart", "button.add-to-cart", desc)
    res.resolve("cart", [
        ("button.add-to-cart", extract_from_tag(_tag("<button>Add to Cart</button>"))),
    ])

    out = tmp_path / "results.xml"
    res.export_junit_xml(str(out))
    assert out.exists()
    assert "<testsuite" in out.read_text()


def test_skip_hidden_candidate():
    res = _fresh_resolver()
    desc = extract_from_tag(_tag("<button>Submit</button>"))
    res.record("submit", "#submit", desc)

    hidden = extract_from_tag(_tag("<button>Submit</button>"))
    hidden.is_visible = False

    result = res.resolve("submit", [("#submit", hidden)], skip_hidden=True)
    assert result.status == Resolution.FAILED


# --- PII Scrubbing ---

def test_scrub_replaces_email():
    result = _scrub("Contact john.doe@example.com for help", _PII_PATTERNS)
    assert "john.doe@example.com" not in result
    assert "[REDACTED]" in result


def test_scrub_replaces_phone():
    result = _scrub("Call us at 555-867-5309", _PII_PATTERNS)
    assert "555-867-5309" not in result
    assert "[REDACTED]" in result


def test_scrub_leaves_safe_text_intact():
    text = "Submit Order button inside checkout form"
    assert _scrub(text, _PII_PATTERNS) == text


def test_scrub_descriptor_dict_cleans_text_fields():
    d = {
        "tag": "button",
        "role": "button",
        "text_content": "Email: test@example.com",
        "label": "Call 555-123-4567",
        "placeholder": "Enter email@domain.com",
        "section_heading": "Contact",
        "text": "button with text 'test@example.com'",
        "is_visible": True,
        "bounding_box": None,
    }
    result = _scrub_descriptor_dict(d, _PII_PATTERNS)
    assert "test@example.com" not in result["text_content"]
    assert "555-123-4567" not in result["label"]
    assert "email@domain.com" not in result["placeholder"]
    assert result["tag"] == "button"          # structural fields untouched
    assert result["is_visible"] is True


def test_store_raw_text_false_scrubs_stored_data():
    store = IntentStore(":memory:", store_raw_text=False)
    res = ConfidenceGatedResolver(store, IntentEmbedder.get())

    desc = extract_from_tag(_tag('<input placeholder="Enter email@example.com">'))
    res.record("email_input", "#email", desc, page_url="https://app.com/profile?user=bob@test.com")

    row = store._conn.execute(
        "SELECT descriptor, page_url FROM intents WHERE name = 'email_input'"
    ).fetchone()
    stored_desc = json.loads(row[0])
    stored_url = row[1]

    assert "email@example.com" not in stored_desc.get("placeholder", "")
    assert "bob@test.com" not in stored_url


def test_store_raw_text_false_resolution_still_works():
    store = IntentStore(":memory:", store_raw_text=False)
    res = ConfidenceGatedResolver(store, IntentEmbedder.get())

    desc = extract_from_tag(_tag('<button aria-label="Submit Order">Submit</button>'))
    res.record("submit", "#submit", desc)

    result = res.resolve("submit", [
        ("#submit-v2", extract_from_tag(_tag('<button aria-label="Submit Order">Place Order</button>'))),
    ])
    # Embedding was computed before scrubbing, so resolution still succeeds
    assert result.status in (Resolution.HEALED, Resolution.NEEDS_CONFIRMATION)
    assert result.confidence > 0.70


def test_store_raw_text_true_preserves_data():
    store = IntentStore(":memory:", store_raw_text=True)
    res = ConfidenceGatedResolver(store, IntentEmbedder.get())

    desc = extract_from_tag(_tag('<input placeholder="user@example.com">'))
    res.record("email_input", "#email", desc, page_url="https://app.com?user=test@example.com")

    row = store._conn.execute(
        "SELECT descriptor, page_url FROM intents WHERE name = 'email_input'"
    ).fetchone()
    stored_desc = json.loads(row[0])
    stored_url = row[1]

    assert "user@example.com" in stored_desc.get("placeholder", "")
    assert "test@example.com" in stored_url
