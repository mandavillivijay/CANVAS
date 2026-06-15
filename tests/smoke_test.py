from playwright.sync_api import sync_playwright


def run_smoke_test():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://example.com")

        h1 = page.query_selector("h1")
        if h1 is None:
            print("FAIL: no h1 found on page")
            browser.close()
            return

        text = h1.inner_text()
        tag = h1.evaluate("el => el.tagName.toLowerCase()")

        aria_attrs = h1.evaluate("""el => {
            const attrs = {};
            for (const attr of el.attributes) {
                if (attr.name.startsWith('aria-') || attr.name === 'role') {
                    attrs[attr.name] = attr.value;
                }
            }
            return attrs;
        }""")

        print(f"tag:        {tag}")
        print(f"text:       {text}")
        print(f"aria attrs: {aria_attrs if aria_attrs else '(none)'}")
        print("PASS: Playwright + Chromium toolchain verified")

        browser.close()


if __name__ == "__main__":
    run_smoke_test()
