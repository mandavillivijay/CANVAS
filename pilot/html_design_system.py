"""
Design system migration case study HTML fixtures.

Simulates a team migrating a raw-HTML checkout page to Ant Design.

V1 — Original raw HTML: semantic IDs, flat DOM, simple classes.
     This is what the test suite was written against.

V2 — Ant Design migration: no IDs, deeply nested wrappers, generated class
     names (ant-btn, ant-input, ant-form-item, ...), button text inside <span>.
     Healenium fails here: no ID, XPath/sibling heuristics find nothing.

V3 — Extreme: same Ant Design structure BUT the CTA is moved to a sticky
     footer outside the form entirely, and its text changes to "Complete Purchase".
     Healenium fails completely. CANVAS finds it (with confirmation flag).
"""

# ---------------------------------------------------------------------------
# V1 — Raw HTML, clean and semantic
# ---------------------------------------------------------------------------
HTML_V1 = """<!DOCTYPE html>
<html>
<body>
  <main>
    <h1>Checkout</h1>
    <form id="checkout-form">
      <h2>Shipping</h2>
      <input id="inp-name"  type="text"  aria-label="Full name"      placeholder="Full name">
      <input id="inp-email" type="email" aria-label="Email address"  placeholder="Email">
      <h2>Payment</h2>
      <input id="inp-card"  type="text"  aria-label="Card number"    placeholder="Card number">
      <input id="inp-cvv"   type="text"  aria-label="CVV"            placeholder="CVV">
      <button id="btn-submit" class="btn-primary" type="submit">Place Order</button>
    </form>
  </main>
</body>
</html>"""

# ---------------------------------------------------------------------------
# V2 — Ant Design migration
#
# What changed (everything Healenium relies on):
#   - All IDs gone
#   - All classes replaced with generated Ant Design names
#   - Button text wrapped in <span class="ant-btn-text">
#   - DOM nesting depth: 1 level (V1) → 4 levels (V2) per element
#   - Inputs all share class="ant-input" — indistinguishable by attribute
#
# What stayed the same (what CANVAS relies on):
#   - aria-label on every input (preserved by the dev team)
#   - Visible text of the button ("Place Order")
#   - Form landmark
#   - Heading context ("Shipping", "Payment")
# ---------------------------------------------------------------------------
HTML_V2 = """<!DOCTYPE html>
<html>
<body>
  <main>
    <h1>Checkout</h1>
    <form class="ant-form ant-form-vertical">
      <h2>Shipping</h2>
      <div class="ant-form-item">
        <div class="ant-form-item-control">
          <div class="ant-input-affix-wrapper">
            <input class="ant-input" type="text"  aria-label="Full name"     placeholder="Enter your full name">
          </div>
        </div>
      </div>
      <div class="ant-form-item">
        <div class="ant-form-item-control">
          <div class="ant-input-affix-wrapper">
            <input class="ant-input" type="email" aria-label="Email address" placeholder="Enter your email">
          </div>
        </div>
      </div>
      <h2>Payment</h2>
      <div class="ant-form-item">
        <div class="ant-form-item-control">
          <div class="ant-input-affix-wrapper">
            <input class="ant-input" type="text"  aria-label="Card number"   placeholder="Enter card number">
          </div>
        </div>
      </div>
      <div class="ant-form-item">
        <div class="ant-form-item-control">
          <div class="ant-input-affix-wrapper">
            <input class="ant-input" type="text"  aria-label="CVV"           placeholder="Security code">
          </div>
        </div>
      </div>
      <div class="ant-form-item ant-form-item-submit">
        <button type="submit" class="ant-btn ant-btn-primary ant-btn-lg ant-btn-block">
          <span class="ant-btn-icon"></span>
          <span class="ant-btn-text">Place Order</span>
        </button>
      </div>
    </form>
  </main>
</body>
</html>"""

# ---------------------------------------------------------------------------
# V3 — Ant Design + CTA ejected to sticky footer
#
# Additional changes vs V2:
#   - Submit button removed from inside the form entirely
#   - A sticky <footer> added at the bottom of the page
#   - CTA is now inside <footer> — completely outside <form>
#   - CTA class changed: ant-btn → cta-sticky-btn
#   - CTA text PRESERVED: still "Place Order"
#
# Why text is kept the same: this isolates the structural change alone.
# The heading context ("Payment") is gone. The form landmark is gone.
# The class is completely different. But the human-readable label survives.
#
# Healenium: total failure. No ID, no original class, footer not in
#            the original DOM ancestry tree — nothing to recover from.
# CANVAS: heading/form context lost → confidence drops below 0.92;
#         but label matches → resolves to NEEDS_CONFIRMATION. The three-gate
#         system correctly flags this for human review rather than auto-healing.
#
# Confidence note: if text ALSO changes (e.g. "Complete Purchase"), the
# model drops to ~0.62 — an honest FAILED result, not a silent wrong match.
# ---------------------------------------------------------------------------
HTML_V3 = """<!DOCTYPE html>
<html>
<body>
  <main>
    <h1>Checkout</h1>
    <form class="ant-form ant-form-vertical">
      <h2>Shipping</h2>
      <div class="ant-form-item">
        <div class="ant-form-item-control">
          <div class="ant-input-affix-wrapper">
            <input class="ant-input" type="text"  aria-label="Full name"     placeholder="Enter your full name">
          </div>
        </div>
      </div>
      <div class="ant-form-item">
        <div class="ant-form-item-control">
          <div class="ant-input-affix-wrapper">
            <input class="ant-input" type="email" aria-label="Email address" placeholder="Enter your email">
          </div>
        </div>
      </div>
      <h2>Payment</h2>
      <div class="ant-form-item">
        <div class="ant-form-item-control">
          <div class="ant-input-affix-wrapper">
            <input class="ant-input" type="text"  aria-label="Card number"   placeholder="Enter card number">
          </div>
        </div>
      </div>
      <div class="ant-form-item">
        <div class="ant-form-item-control">
          <div class="ant-input-affix-wrapper">
            <input class="ant-input" type="text"  aria-label="CVV"           placeholder="Security code">
          </div>
        </div>
      </div>
    </form>
  </main>
  <footer class="sticky-footer">
    <span class="order-total">Total: $54.98</span>
    <button type="button" class="cta-sticky-btn">
      <span>Place Order</span>
    </button>
  </footer>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Intents to record from V1
# ---------------------------------------------------------------------------
INTENTS_V1 = [
    ("ds_submit_btn",   "#btn-submit"),
    ("ds_email_input",  "#inp-email"),
    ("ds_card_input",   "#inp-card"),
    ("ds_cvv_input",    "#inp-cvv"),
    ("ds_name_input",   "#inp-name"),
]

# Expected healed selectors in V2.
# The smart extractor returns the first class for elements without aria-label;
# the V2 button's first class is "ant-btn" (shared namespace, but unique in this page).
EXPECTED_V2 = {
    "ds_submit_btn":  ".ant-btn",
    "ds_email_input": '[aria-label="Email address"]',
    "ds_card_input":  '[aria-label="Card number"]',
    "ds_cvv_input":   '[aria-label="CVV"]',
    "ds_name_input":  '[aria-label="Full name"]',
}

# Expected healed selectors in V3
EXPECTED_V3 = {
    "ds_submit_btn":  ".cta-sticky-btn",
    "ds_email_input": '[aria-label="Email address"]',
    "ds_card_input":  '[aria-label="Card number"]',
    "ds_cvv_input":   '[aria-label="CVV"]',
    "ds_name_input":  '[aria-label="Full name"]',
}
