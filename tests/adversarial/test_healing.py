from bs4 import BeautifulSoup
from canvas_heal.descriptor import extract_from_tag
from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import ConfidenceGatedResolver, IntentStore, Resolution
import pytest


def _tag(html):
    return BeautifulSoup(html, "html.parser").find(True)


def _candidate(selector, html):
    return (selector, extract_from_tag(_tag(html)))


@pytest.fixture(scope="module")
def resolver():
    store = IntentStore(":memory:")
    return ConfidenceGatedResolver(store, IntentEmbedder.get())


def test_dom_restructuring_container_change(resolver):
    recorded = _tag('<form><button id="place">Place Order</button></form>')
    resolver.record("place_order_restructure", "#place", extract_from_tag(recorded))

    result = resolver.resolve(
        "place_order_restructure",
        [_candidate("#place2", '<section><button id="place2">Place Order</button></section>')],
    )

    assert result.status in (Resolution.HEALED, Resolution.NEEDS_CONFIRMATION)
    assert result.status is not Resolution.FAILED
    assert result.selector == "#place2"


def test_id_and_class_churn(resolver):
    recorded = _tag('<button id="btn-primary" class="old-style hero">Place Order</button>')
    resolver.record("id_churn", "#btn-primary", extract_from_tag(recorded))

    result = resolver.resolve(
        "id_churn",
        [_candidate("#cta-final", '<button id="cta-final" class="brand-new shiny">Place Order</button>')],
    )

    assert result.status is Resolution.HEALED
    assert result.selector == "#cta-final"


def test_label_paraphrase(resolver):
    recorded = _tag('<button aria-label="Add item to cart">+</button>')
    resolver.record("label_paraphrase", "#add", extract_from_tag(recorded))

    result = resolver.resolve(
        "label_paraphrase",
        [_candidate("#add2", '<button aria-label="Add to cart">+</button>')],
    )

    assert result.status is Resolution.HEALED


def test_section_heading_change(resolver):
    recorded = _tag(
        '<section><h2>Shipping Address</h2>'
        '<input aria-label="Street" placeholder="Street address"></section>'
    )
    target = recorded.find("input")
    resolver.record("heading_change", "#street", extract_from_tag(target))

    candidate_root = _tag(
        '<section><h2>Delivery Details</h2>'
        '<input aria-label="Street" placeholder="Street address"></section>'
    )
    candidate = candidate_root.find("input")

    result = resolver.resolve("heading_change", [("#street2", extract_from_tag(candidate))])

    assert result.status in (Resolution.HEALED, Resolution.NEEDS_CONFIRMATION)
    assert result.status is not Resolution.FAILED


def test_landmark_change(resolver):
    recorded = _tag('<nav><input type="search" aria-label="Search products"></nav>')
    target = recorded.find("input")
    resolver.record("landmark_change", "#search", extract_from_tag(target))

    candidate_root = _tag('<section><input type="search" aria-label="Search products"></section>')
    candidate = candidate_root.find("input")

    result = resolver.resolve("landmark_change", [("#search2", extract_from_tag(candidate))])

    assert result.status in (Resolution.HEALED, Resolution.NEEDS_CONFIRMATION)
    assert result.status is not Resolution.FAILED


def test_false_positive_dissimilar_element(resolver):
    recorded = _tag('<button id="submit">Submit Order</button>')
    resolver.record("false_positive_link", "#submit", extract_from_tag(recorded))

    result = resolver.resolve(
        "false_positive_link",
        [_candidate("#privacy", '<a href="/privacy">Privacy Policy</a>')],
    )

    assert result.status is not Resolution.HEALED


def test_false_positive_same_type_different_purpose(resolver):
    recorded = _tag('<input type="email" aria-label="Email address">')
    resolver.record("false_positive_input", "#email", extract_from_tag(recorded))

    result = resolver.resolve(
        "false_positive_input",
        [_candidate("#coupon", '<input type="text" aria-label="Coupon code">')],
    )

    assert result.status is not Resolution.HEALED


def test_best_candidate_wins_in_pool(resolver):
    recorded = _tag('<form><button id="checkout">Checkout</button></form>')
    resolver.record("checkout_pool", "#checkout", extract_from_tag(recorded))

    candidates = [
        _candidate("#login", '<button id="login">Log in</button>'),
        _candidate("#privacy", '<a href="/privacy">Privacy Policy</a>'),
        _candidate("#footernav", '<footer><a href="/about">About Us</a></footer>'),
        _candidate("#checkout2", '<form><button id="checkout2" class="new-cta">Checkout</button></form>'),
    ]

    result = resolver.resolve("checkout_pool", candidates)

    assert result.status is Resolution.HEALED
    assert result.selector == "#checkout2"


def test_complete_semantic_removal(resolver):
    recorded = _tag('<button id="ghost">Ghost Button</button>')
    resolver.record("removed_element", "#ghost", extract_from_tag(recorded))

    result = resolver.resolve("removed_element", [])

    assert result.status is Resolution.FAILED
    assert "No candidates" in result.message


def test_threshold_override_blocks_exact_match():
    store = IntentStore(":memory:")
    strict = ConfidenceGatedResolver(store, IntentEmbedder.get(), threshold_auto=0.99)

    recorded = _tag('<button id="exact">Place Order</button>')
    strict.record("threshold_override", "#exact", extract_from_tag(recorded))

    candidate = [_candidate("#exact2", '<button id="exact2" class="different">Place Order!</button>')]

    result = strict.resolve("threshold_override", candidate)
    assert result.status is Resolution.NEEDS_CONFIRMATION
    assert result.confidence >= 0.92

    default = ConfidenceGatedResolver(strict._store, IntentEmbedder.get())
    assert default.resolve("threshold_override", candidate).status is Resolution.HEALED


def test_threshold_override_lenient_auto_heals():
    store = IntentStore(":memory:")
    lenient = ConfidenceGatedResolver(store, IntentEmbedder.get(), threshold_auto=0.50)

    recorded = _tag('<input type="email" aria-label="Email address">')
    lenient.record("lenient_override", "#email", extract_from_tag(recorded))

    result = lenient.resolve(
        "lenient_override",
        [_candidate("#contact", '<input type="email" aria-label="Contact email">')],
    )

    assert result.status is Resolution.HEALED
