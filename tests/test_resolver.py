import pytest
from bs4 import BeautifulSoup

from canvas_heal.descriptor import SemanticDescriptor, extract_from_tag
from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import ConfidenceGatedResolver, IntentStore, Resolution


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
