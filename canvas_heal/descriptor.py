from __future__ import annotations

import logging
from dataclasses import dataclass

from bs4 import Tag

_log = logging.getLogger("canvas_heal.descriptor")

_HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
_LANDMARK_TAGS = frozenset({"nav", "main", "aside", "footer", "header", "form", "section", "article"})

_IMPLICIT_ROLES: dict[str, str] = {
    "button": "button",
    "a": "link",
    "select": "listbox",
    "textarea": "textbox",
    "nav": "navigation",
    "main": "main",
    "aside": "complementary",
    "footer": "contentinfo",
    "header": "banner",
    "form": "form",
    "h1": "heading", "h2": "heading", "h3": "heading",
    "h4": "heading", "h5": "heading", "h6": "heading",
    "img": "img",
    "table": "table",
    "li": "listitem",
    "ul": "list", "ol": "list",
    "dialog": "dialog",
}

_INPUT_TYPE_ROLES: dict[str, str] = {
    "checkbox": "checkbox",
    "radio": "radio",
    "button": "button",
    "submit": "button",
    "reset": "button",
    "range": "slider",
    "search": "searchbox",
}

# JavaScript run against a Playwright element handle to extract element properties
_PLAYWRIGHT_JS = """(el) => {
    if (!el) return null;

    function nearestHeading(node) {
        let current = node;
        while (current && current !== document.body) {
            let sib = current.previousElementSibling;
            while (sib) {
                if (/^H[1-6]$/.test(sib.tagName))
                    return (sib.innerText || '').trim().substring(0, 80);
                const h = sib.querySelector('h1,h2,h3,h4,h5,h6');
                if (h) return (h.innerText || '').trim().substring(0, 80);
                sib = sib.previousElementSibling;
            }
            current = current.parentElement || (current.getRootNode && current.getRootNode().host) || null;
        }
        return '';
    }

    function nearestLandmark(node) {
        const lm = new Set(['nav','main','aside','footer','header','form','section','article']);
        let current = node.parentElement || (node.getRootNode && node.getRootNode().host) || null;
        while (current) {
            if (lm.has(current.tagName.toLowerCase()))
                return current.getAttribute('role') || current.tagName.toLowerCase();
            current = current.parentElement || (current.getRootNode && current.getRootNode().host) || null;
        }
        return '';
    }

    const parent = el.parentElement;
    const style = getComputedStyle(el);
    return {
        tag: el.tagName.toLowerCase(),
        explicit_role: el.getAttribute('role') || '',
        label: el.getAttribute('aria-label') || el.getAttribute('title') || '',
        element_type: (el.getAttribute('type') || '').toLowerCase(),
        placeholder: el.getAttribute('placeholder') || '',
        text_content: (el.innerText || el.textContent || '').trim().substring(0, 120),
        parent_tag: parent ? parent.tagName.toLowerCase() : '',
        parent_explicit_role: parent ? (parent.getAttribute('role') || '') : '',
        section_heading: nearestHeading(el),
        landmark: nearestLandmark(el),
        is_visible: el.offsetParent !== null && style.visibility !== 'hidden' && style.display !== 'none' && !el.hidden,
        is_disabled: el.disabled === true || el.getAttribute('aria-disabled') === 'true',
        bounding_box: el.getBoundingClientRect ? {
            x: Math.round(el.getBoundingClientRect().x),
            y: Math.round(el.getBoundingClientRect().y),
            width: Math.round(el.getBoundingClientRect().width),
            height: Math.round(el.getBoundingClientRect().height)
        } : null,
    };
}"""


_SELENIUM_JS = """
var el = arguments[0];
var parent = el.parentElement;
function nearestHeading(node) {
    var current = node;
    while (current && current !== document.body) {
        var sib = current.previousElementSibling;
        while (sib) {
            if (/^H[1-6]$/.test(sib.tagName)) return (sib.innerText || '').trim().substring(0, 80);
            var h = sib.querySelector('h1,h2,h3,h4,h5,h6');
            if (h) return (h.innerText || '').trim().substring(0, 80);
            sib = sib.previousElementSibling;
        }
        current = current.parentElement;
    }
    return '';
}
function nearestLandmark(node) {
    var lm = ['nav','main','aside','footer','header','form','section','article'];
    var current = node.parentElement;
    while (current) {
        if (lm.indexOf(current.tagName.toLowerCase()) !== -1)
            return current.getAttribute('role') || current.tagName.toLowerCase();
        current = current.parentElement;
    }
    return '';
}
var rect = el.getBoundingClientRect();
return {
    tag: el.tagName.toLowerCase(),
    explicit_role: el.getAttribute('role') || '',
    label: el.getAttribute('aria-label') || el.getAttribute('title') || '',
    element_type: (el.getAttribute('type') || '').toLowerCase(),
    placeholder: el.getAttribute('placeholder') || '',
    text_content: (el.innerText || el.textContent || '').trim().substring(0, 120),
    parent_tag: parent ? parent.tagName.toLowerCase() : '',
    parent_explicit_role: parent ? (parent.getAttribute('role') || '') : '',
    section_heading: nearestHeading(el),
    landmark: nearestLandmark(el),
    is_visible: el.offsetParent !== null && getComputedStyle(el).visibility !== 'hidden' && getComputedStyle(el).display !== 'none' && !el.hidden,
    is_disabled: el.disabled === true || el.getAttribute('aria-disabled') === 'true',
    bounding_box: { x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height) }
};
"""


def _infer_role(tag: str, element_type: str, explicit_role: str) -> str:
    if explicit_role:
        return explicit_role
    if tag == "input":
        return _INPUT_TYPE_ROLES.get(element_type, "textbox")
    return _IMPLICIT_ROLES.get(tag, tag)


@dataclass
class SemanticDescriptor:
    tag: str
    role: str
    label: str
    element_type: str
    placeholder: str
    text_content: str
    parent_tag: str
    parent_role: str
    section_heading: str
    landmark: str
    is_visible: bool = True
    is_disabled: bool = False
    bounding_box: dict | None = None

    def to_text(self) -> str:
        parts = [self.role or self.tag]
        if self.label:
            parts.append(f"labeled '{self.label}'")
        elif self.text_content:
            parts.append(f"with text '{self.text_content}'")
        if self.placeholder:
            parts.append(f"placeholder '{self.placeholder}'")
        if self.element_type and self.element_type not in ("", "text", "submit", "button"):
            parts.append(f"type {self.element_type}")
        if self.landmark:
            parts.append(f"inside {self.landmark}")
        if self.section_heading:
            parts.append(f"under heading '{self.section_heading}'")
        if self.parent_tag and self.parent_tag not in ("", "body", "div", "span") and self.parent_tag != self.landmark:
            parts.append(f"within {self.parent_tag}")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "role": self.role,
            "label": self.label,
            "element_type": self.element_type,
            "placeholder": self.placeholder,
            "text_content": self.text_content,
            "parent_tag": self.parent_tag,
            "parent_role": self.parent_role,
            "section_heading": self.section_heading,
            "landmark": self.landmark,
            "is_visible": self.is_visible,
            "is_disabled": self.is_disabled,
            "bounding_box": self.bounding_box,
            "text": self.to_text(),
        }


def extract_from_tag(el: Tag) -> SemanticDescriptor:
    """Extract a SemanticDescriptor from a BeautifulSoup Tag."""
    tag = el.name or ""
    explicit_role = el.get("role", "")
    element_type = el.get("type", "").lower()
    role = _infer_role(tag, element_type, explicit_role)
    label = (el.get("aria-label") or el.get("title") or "").strip()
    placeholder = el.get("placeholder", "").strip()
    text_content = el.get_text(strip=True)[:120]

    parent = el.parent
    parent_tag = ""
    parent_role = ""
    if parent and hasattr(parent, "name") and parent.name and parent.name != "[document]":
        parent_tag = parent.name
        parent_role = _infer_role(parent_tag, "", parent.get("role", ""))

    heading_tag = el.find_previous(_HEADING_TAGS)
    section_heading = heading_tag.get_text(strip=True)[:80] if heading_tag else ""

    landmark = ""
    node = el.parent
    while node and hasattr(node, "name"):
        if node.name in _LANDMARK_TAGS:
            landmark = node.get("role", node.name)
            break
        node = node.parent

    is_disabled = el.get("disabled") is not None or el.get("aria-disabled") == "true"

    return SemanticDescriptor(
        tag=tag,
        role=role,
        label=label,
        element_type=element_type,
        placeholder=placeholder,
        text_content=text_content,
        parent_tag=parent_tag,
        parent_role=parent_role,
        section_heading=section_heading,
        landmark=landmark,
        is_visible=True,
        is_disabled=is_disabled,
        bounding_box=None,
    )


def extract_from_playwright(page, selector: str) -> SemanticDescriptor:
    """Extract a SemanticDescriptor from a live Playwright page by CSS selector."""
    _log.debug("extracting descriptor selector=%r", selector)
    handle = page.query_selector(selector)
    if handle is None:
        _log.error("element not found selector=%r", selector)
        raise ValueError(f"Element not found for selector: {selector!r}")
    info = handle.evaluate(_PLAYWRIGHT_JS)

    tag = info["tag"]
    element_type = info["element_type"]
    role = _infer_role(tag, element_type, info["explicit_role"])
    parent_role = _infer_role(info["parent_tag"], "", info["parent_explicit_role"])

    return SemanticDescriptor(
        tag=tag,
        role=role,
        label=info["label"].strip(),
        element_type=element_type,
        placeholder=info["placeholder"].strip(),
        text_content=info["text_content"],
        parent_tag=info["parent_tag"],
        parent_role=parent_role,
        section_heading=info["section_heading"],
        landmark=info["landmark"],
        is_visible=info["is_visible"],
        is_disabled=info["is_disabled"],
        bounding_box=info["bounding_box"],
    )


def extract_from_selenium(driver, selector: str) -> SemanticDescriptor:
    """Extract a SemanticDescriptor from a live Selenium driver by CSS selector."""
    _log.debug("extracting descriptor selector=%r (selenium)", selector)
    try:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.remote.webelement import WebElement  # noqa: F401
    except ImportError:
        raise ImportError("selenium is required: pip install canvas-heal[selenium]")

    el = driver.find_element(By.CSS_SELECTOR, selector)
    info = driver.execute_script(_SELENIUM_JS, el)

    tag = info["tag"]
    element_type = info["element_type"]
    role = _infer_role(tag, element_type, info["explicit_role"])
    parent_role = _infer_role(info["parent_tag"], "", info["parent_explicit_role"])

    return SemanticDescriptor(
        tag=tag,
        role=role,
        label=info["label"].strip(),
        element_type=element_type,
        placeholder=info["placeholder"].strip(),
        text_content=info["text_content"],
        parent_tag=info["parent_tag"],
        parent_role=parent_role,
        section_heading=info["section_heading"],
        landmark=info["landmark"],
        is_visible=info.get("is_visible", True),
        is_disabled=info.get("is_disabled", False),
        bounding_box=info.get("bounding_box"),
    )


def extract_from_playwright_frame(page, frame_selector: str, element_selector: str) -> SemanticDescriptor:
    """Extract a SemanticDescriptor from an element inside an iframe."""
    frame = page.frame_locator(frame_selector)
    handle = frame.locator(element_selector).element_handle()
    if handle is None:
        raise ValueError(f"Element not found: frame={frame_selector!r}, selector={element_selector!r}")
    info = handle.evaluate(_PLAYWRIGHT_JS)
    if info is None:
        raise ValueError(f"Element not found in frame: {element_selector!r}")
    tag = info["tag"]
    element_type = info["element_type"]
    role = _infer_role(tag, element_type, info["explicit_role"])
    parent_role = _infer_role(info["parent_tag"], "", info["parent_explicit_role"])
    return SemanticDescriptor(
        tag=tag,
        role=role,
        label=info["label"].strip(),
        element_type=element_type,
        placeholder=info["placeholder"].strip(),
        text_content=info["text_content"],
        parent_tag=info["parent_tag"],
        parent_role=parent_role,
        section_heading=info["section_heading"],
        landmark=info["landmark"],
        is_visible=info.get("is_visible", True),
        is_disabled=info.get("is_disabled", False),
        bounding_box=info.get("bounding_box"),
    )


async def extract_from_playwright_frame_async(page, frame_selector: str, element_selector: str) -> SemanticDescriptor:
    """Extract a SemanticDescriptor from an element inside an iframe (async)."""
    frame = page.frame_locator(frame_selector)
    handle = await frame.locator(element_selector).element_handle()
    if handle is None:
        raise ValueError(f"Element not found: frame={frame_selector!r}, selector={element_selector!r}")
    info = await handle.evaluate(_PLAYWRIGHT_JS)
    if info is None:
        raise ValueError(f"Element not found in frame: {element_selector!r}")
    tag = info["tag"]
    element_type = info["element_type"]
    role = _infer_role(tag, element_type, info["explicit_role"])
    parent_role = _infer_role(info["parent_tag"], "", info["parent_explicit_role"])
    return SemanticDescriptor(
        tag=tag,
        role=role,
        label=info["label"].strip(),
        element_type=element_type,
        placeholder=info["placeholder"].strip(),
        text_content=info["text_content"],
        parent_tag=info["parent_tag"],
        parent_role=parent_role,
        section_heading=info["section_heading"],
        landmark=info["landmark"],
        is_visible=info.get("is_visible", True),
        is_disabled=info.get("is_disabled", False),
        bounding_box=info.get("bounding_box"),
    )


async def extract_from_playwright_async(page, selector: str) -> SemanticDescriptor:
    """Extract a SemanticDescriptor from an async Playwright page by CSS selector."""
    handle = await page.query_selector(selector)
    if handle is None:
        raise ValueError(f"Element not found for selector: {selector!r}")
    info = await handle.evaluate(_PLAYWRIGHT_JS)

    tag = info["tag"]
    element_type = info["element_type"]
    role = _infer_role(tag, element_type, info["explicit_role"])
    parent_role = _infer_role(info["parent_tag"], "", info["parent_explicit_role"])

    return SemanticDescriptor(
        tag=tag,
        role=role,
        label=info["label"].strip(),
        element_type=element_type,
        placeholder=info["placeholder"].strip(),
        text_content=info["text_content"],
        parent_tag=info["parent_tag"],
        parent_role=parent_role,
        section_heading=info["section_heading"],
        landmark=info["landmark"],
        is_visible=info["is_visible"],
        is_disabled=info["is_disabled"],
        bounding_box=info["bounding_box"],
    )
