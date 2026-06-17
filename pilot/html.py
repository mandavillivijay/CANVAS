"""
HTML fixtures for the CANVAS-HEAL pilot.

V1 = original ShopEasy checkout UI (the "before" state, recorded baseline).
V2 = redesigned UI shipped by the dev team (different IDs, structure, labels).

All element IDs, class names, and placeholders changed in V2.
Semantic intent (what each element does) is preserved.
"""

HTML_V1 = """<!DOCTYPE html>
<html>
<body>
  <nav id="main-nav">
    <a id="nav-home" href="/">Home</a>
    <a id="nav-cart" href="/cart">Cart (3 items)</a>
    <a id="nav-account" href="/account">My Account</a>
  </nav>
  <main>
    <h1>Checkout</h1>
    <section id="order-summary">
      <h2>Order Summary</h2>
      <p>Widget Pro x2 — $49.98</p>
      <strong>Total: $54.98</strong>
    </section>
    <form id="checkout-form">
      <h2>Shipping Details</h2>
      <input id="full-name"    type="text"  placeholder="Full name"       aria-label="Full name">
      <input id="email"        type="email" placeholder="Email address"   aria-label="Email address">
      <input id="street"       type="text"  placeholder="Street address"  aria-label="Street address">
      <h2>Payment</h2>
      <input id="card-number"  type="text"  placeholder="Card number"     aria-label="Card number">
      <input id="card-expiry"  type="text"  placeholder="MM/YY"           aria-label="Card expiry">
      <input id="card-cvv"     type="text"  placeholder="CVV"             aria-label="CVV">
      <button id="btn-place-order" type="submit" class="btn-primary">Place Order</button>
    </form>
    <a id="link-back-to-cart" href="/cart">Back to cart</a>
  </main>
</body>
</html>"""

HTML_V2 = """<!DOCTYPE html>
<html>
<body>
  <header>
    <nav role="navigation">
      <a class="lnk-home"    href="/">Home</a>
      <a class="lnk-cart"    href="/cart">Shopping Cart (3)</a>
      <a class="lnk-account" href="/account">Account</a>
    </nav>
  </header>
  <div role="main">
    <h1>Secure Checkout</h1>
    <aside class="order-panel">
      <h2>Your Order</h2>
      <p>Widget Pro x2 — $49.98</p>
      <strong>Order Total: $54.98</strong>
    </aside>
    <form class="checkout-wrapper">
      <h2>Delivery Information</h2>
      <input class="fld-name"   type="text"  placeholder="Your full name"              aria-label="Full name">
      <input class="fld-email"  type="email" placeholder="Your email"                  aria-label="Email address">
      <input class="fld-street" type="text"  placeholder="Street"                      aria-label="Street address">
      <h2>Payment</h2>
      <input class="fld-card"   type="text"  placeholder="Credit or debit card number" aria-label="Card number">
      <input class="fld-expiry" type="text"  placeholder="Expiry MM/YY"                aria-label="Card expiry">
      <input class="fld-cvv"    type="text"  placeholder="Security code"               aria-label="CVV">
      <button id="cta-confirm-order" class="cta-button">Confirm &amp; Pay</button>
    </form>
    <a class="lnk-return" href="/cart">Return to cart</a>
  </div>
</body>
</html>"""

# Intents to record from V1: (intent_name, css_selector)
INTENTS_V1 = [
    ("place_order_btn",   "#btn-place-order"),
    ("email_input",       "#email"),
    ("card_number_input", "#card-number"),
    ("card_cvv_input",    "#card-cvv"),
    ("full_name_input",   "#full-name"),
    ("back_to_cart_link", "#link-back-to-cart"),
    ("nav_home_link",     "#nav-home"),
]

# Expected healed selectors in V2 for each intent
EXPECTED_V2 = {
    "place_order_btn":   "#cta-confirm-order",
    "email_input":       ".fld-email",
    "card_number_input": ".fld-card",
    "card_cvv_input":    ".fld-cvv",
    "full_name_input":   ".fld-name",
    "back_to_cart_link": ".lnk-return",
    "nav_home_link":     ".lnk-home",
}
