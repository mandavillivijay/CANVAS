import pytest
from bs4 import BeautifulSoup

from canvas.descriptor import extract_from_tag
from canvas.embedder import IntentEmbedder
from canvas.resolver import ConfidenceGatedResolver, IntentStore, Resolution


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
