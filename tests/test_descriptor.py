from unittest import mock

import pytest
from bs4 import BeautifulSoup
from canvas_heal.descriptor import extract_from_tag


def _parse(html: str):
    return BeautifulSoup(html, "html.parser").find(True)


def test_button_role_and_text():
    desc = extract_from_tag(_parse("<button>Submit Order</button>"))
    assert desc.tag == "button"
    assert desc.role == "button"
    assert desc.text_content == "Submit Order"
    assert "button" in desc.to_text()
    assert "Submit Order" in desc.to_text()


def test_input_email_descriptor():
    desc = extract_from_tag(_parse(
        '<input type="email" placeholder="Enter your email" aria-label="Email address">'
    ))
    assert desc.tag == "input"
    assert desc.element_type == "email"
    assert desc.placeholder == "Enter your email"
    assert desc.label == "Email address"
    text = desc.to_text()
    assert "textbox" in text
    assert "Email address" in text


def test_input_checkbox_role():
    desc = extract_from_tag(_parse('<input type="checkbox" aria-label="Accept terms">'))
    assert desc.role == "checkbox"


def test_input_search_role():
    desc = extract_from_tag(_parse('<input type="search" placeholder="Search products">'))
    assert desc.role == "searchbox"


def test_link_descriptor():
    desc = extract_from_tag(_parse('<a href="/checkout">Proceed to Checkout</a>'))
    assert desc.role == "link"
    assert "Proceed to Checkout" in desc.to_text()


def test_explicit_role_overrides_tag():
    desc = extract_from_tag(_parse('<div role="button" aria-label="Open menu">☰</div>'))
    assert desc.role == "button"
    assert desc.label == "Open menu"


def test_section_heading_extracted():
    html = """
    <div>
        <h2>Billing Details</h2>
        <form>
            <input type="text" placeholder="Full name">
        </form>
    </div>
    """
    el = BeautifulSoup(html, "html.parser").find("input")
    desc = extract_from_tag(el)
    assert desc.section_heading == "Billing Details"
    assert "Billing Details" in desc.to_text()


def test_landmark_extracted():
    html = "<nav><a href='/'>Home</a></nav>"
    el = BeautifulSoup(html, "html.parser").find("a")
    desc = extract_from_tag(el)
    assert desc.landmark == "nav"
    assert "inside nav" in desc.to_text()


def test_form_landmark():
    html = "<form><button type='submit'>Save</button></form>"
    el = BeautifulSoup(html, "html.parser").find("button")
    desc = extract_from_tag(el)
    assert desc.landmark == "form"


def test_to_dict_has_text_key():
    desc = extract_from_tag(_parse('<button aria-label="Close dialog">X</button>'))
    d = desc.to_dict()
    assert d["role"] == "button"
    assert d["label"] == "Close dialog"
    assert "text" in d
    assert d["text"] == desc.to_text()


def test_disabled_attribute_detected():
    desc = extract_from_tag(_parse("<button disabled>Submit</button>"))
    assert desc.is_disabled is True


def test_aria_disabled_detected():
    desc = extract_from_tag(_parse('<div role="button" aria-disabled="true">Go</div>'))
    assert desc.is_disabled is True


def test_enabled_element_not_disabled():
    desc = extract_from_tag(_parse("<button>Submit</button>"))
    assert desc.is_disabled is False


def test_bounding_box_none_from_tag():
    desc = extract_from_tag(_parse("<button>Submit</button>"))
    assert desc.bounding_box is None


def test_is_visible_true_from_tag():
    desc = extract_from_tag(_parse("<button>Submit</button>"))
    assert desc.is_visible is True


def test_extract_from_selenium_import_error():
    from canvas_heal.descriptor import extract_from_selenium

    blocked = {
        "selenium": None,
        "selenium.webdriver": None,
        "selenium.webdriver.common": None,
        "selenium.webdriver.common.by": None,
        "selenium.webdriver.remote": None,
        "selenium.webdriver.remote.webelement": None,
    }
    with mock.patch.dict("sys.modules", blocked):
        with pytest.raises(ImportError):
            extract_from_selenium(None, "button")


def test_selenium_js_constant_exists():
    from canvas_heal.descriptor import _SELENIUM_JS

    assert isinstance(_SELENIUM_JS, str)


def test_extract_from_playwright_frame_function_exists():
    from canvas_heal.descriptor import extract_from_playwright_frame

    assert callable(extract_from_playwright_frame)
