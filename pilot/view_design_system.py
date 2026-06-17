"""
Visual demo for the design system migration case study.

Run:   python pilot/view_design_system.py

Opens a browser window showing V1, V2, and V3 side by side with annotations
explaining what changed at each step and how CANVAS / Healenium respond.
"""
import pathlib
import tempfile
import webbrowser

from pilot.html_design_system import HTML_V1, HTML_V2, HTML_V3

# Strip the outer <html>/<body> tags so we can embed each version inline
def _body(html: str) -> str:
    import re
    m = re.search(r"<body>(.*?)</body>", html, re.DOTALL)
    return m.group(1).strip() if m else html

V1_BODY = _body(HTML_V1)
V2_BODY = _body(HTML_V2)
V3_BODY = _body(HTML_V3)

PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CANVAS Case Study — Design System Migration</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f0f2f5;
      color: #222;
    }}

    /* ── top bar ── */
    header {{
      background: #1a1a2e;
      color: #fff;
      padding: 18px 32px;
    }}
    header h1 {{ font-size: 1.2rem; font-weight: 600; letter-spacing: .02em; }}
    header p  {{ font-size: .85rem; opacity: .7; margin-top: 4px; }}

    /* ── layout ── */
    .columns {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 20px;
      padding: 24px 32px;
      align-items: start;
    }}

    /* ── version card ── */
    .card {{
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,.08);
      overflow: hidden;
    }}
    .card-header {{
      padding: 14px 18px 10px;
      border-bottom: 1px solid #eee;
    }}
    .card-header .version {{
      font-size: .7rem;
      font-weight: 700;
      letter-spacing: .1em;
      text-transform: uppercase;
      color: #888;
    }}
    .card-header h2 {{
      font-size: 1rem;
      margin-top: 2px;
    }}
    .card-header .subtitle {{
      font-size: .8rem;
      color: #555;
      margin-top: 4px;
    }}

    /* ── chips ── */
    .chips {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }}
    .chip {{
      font-size: .7rem;
      padding: 2px 8px;
      border-radius: 20px;
      font-weight: 600;
    }}
    .chip-pass  {{ background: #e6f4ea; color: #137333; }}
    .chip-warn  {{ background: #fff3cd; color: #856404; }}
    .chip-fail  {{ background: #fce8e6; color: #c5221f; }}

    /* ── preview iframe ── */
    .preview {{
      width: 100%;
      height: 420px;
      border: none;
      display: block;
    }}

    /* ── diff annotations ── */
    .diff-section {{
      padding: 12px 18px;
      border-top: 1px solid #eee;
      font-size: .8rem;
    }}
    .diff-section h3 {{
      font-size: .75rem;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: #888;
      margin-bottom: 6px;
    }}
    .diff-list {{ list-style: none; }}
    .diff-list li {{ padding: 2px 0; }}
    .diff-list li::before {{ content: "▸ "; color: #aaa; }}
    .diff-list .changed::before {{ content: "✎ "; color: #e6a817; }}
    .diff-list .added::before   {{ content: "+ "; color: #137333; font-weight: 700; }}
    .diff-list .removed::before {{ content: "− "; color: #c5221f; font-weight: 700; }}
    .diff-list .same::before    {{ content: "✓ "; color: #888; }}

    /* ── legend ── */
    .legend {{
      margin: 0 32px 24px;
      background: #fff;
      border-radius: 10px;
      padding: 16px 20px;
      font-size: .82rem;
      border-left: 4px solid #1a1a2e;
      box-shadow: 0 2px 8px rgba(0,0,0,.06);
    }}
    .legend h3 {{ font-size: .9rem; margin-bottom: 8px; }}
    .legend p   {{ color: #444; line-height: 1.5; }}
    .legend code {{
      background: #f0f2f5; border-radius: 3px;
      padding: 1px 5px; font-size: .8rem;
    }}
  </style>
</head>
<body>
  <header>
    <h1>CANVAS Case Study — Design System Migration</h1>
    <p>Three snapshots of the same checkout page. Every attribute Healenium relies on is gone by V2.</p>
  </header>

  <div class="legend">
    <h3>How to read this</h3>
    <p>
      V1 is the original page — what the test suite was written against.
      V2 simulates migrating to Ant Design: all IDs removed, all inputs share
      <code>class="ant-input"</code>, button text is buried inside nested <code>&lt;span&gt;</code> tags.
      V3 goes further: the CTA leaves the form entirely and moves to a sticky footer, and its text changes.
      <br><br>
      <strong>Healenium</strong> records element IDs and XPath fingerprints.
      Every one of its recorded attributes is absent from V2 and V3 — it fails on all 5 elements.
      <br>
      <strong>CANVAS</strong> records semantic intent (aria-label, visible text, heading context, landmark).
      That information survives both migrations.
    </p>
  </div>

  <div class="columns">

    <!-- V1 -->
    <div class="card">
      <div class="card-header">
        <div class="version">Version 1</div>
        <h2>Raw HTML — baseline</h2>
        <div class="subtitle">Original checkout. Tests written against this.</div>
        <div class="chips">
          <span class="chip chip-pass">Healenium: records OK</span>
          <span class="chip chip-pass">CANVAS: records OK</span>
        </div>
      </div>
      <iframe class="preview" srcdoc="{HTML_V1.replace('"', '&quot;')}"></iframe>
      <div class="diff-section">
        <h3>Selectors available</h3>
        <ul class="diff-list">
          <li class="same"><code>#btn-submit</code> — CTA button</li>
          <li class="same"><code>#inp-email</code> — email</li>
          <li class="same"><code>#inp-card</code> — card number</li>
          <li class="same"><code>#inp-cvv</code> — CVV</li>
          <li class="same"><code>#inp-name</code> — full name</li>
        </ul>
      </div>
    </div>

    <!-- V2 -->
    <div class="card">
      <div class="card-header">
        <div class="version">Version 2</div>
        <h2>Ant Design migration</h2>
        <div class="subtitle">All IDs gone. Inputs indistinguishable by class.</div>
        <div class="chips">
          <span class="chip chip-fail">Healenium: 5/5 FAILED</span>
          <span class="chip chip-pass">CANVAS: 5/5 HEALED (≥0.92)</span>
        </div>
      </div>
      <iframe class="preview" srcdoc="{HTML_V2.replace('"', '&quot;')}"></iframe>
      <div class="diff-section">
        <h3>What changed</h3>
        <ul class="diff-list">
          <li class="removed">All element IDs removed</li>
          <li class="changed">All classes → Ant Design names</li>
          <li class="changed">Button text wrapped in <code>&lt;span class="ant-btn-text"&gt;</code></li>
          <li class="changed">DOM nesting: 1 level → 4 levels per element</li>
          <li class="removed">All inputs share <code>class="ant-input"</code></li>
          <li class="same">aria-labels preserved (dev best practice)</li>
          <li class="same">Button text unchanged ("Place Order")</li>
          <li class="same">Heading context preserved ("Payment")</li>
        </ul>
      </div>
    </div>

    <!-- V3 -->
    <div class="card">
      <div class="card-header">
        <div class="version">Version 3</div>
        <h2>CTA → sticky footer</h2>
        <div class="subtitle">CTA moves out of form. Text changes. Ant Design still in place.</div>
        <div class="chips">
          <span class="chip chip-fail">Healenium: 5/5 FAILED</span>
          <span class="chip chip-pass">Inputs: HEALED (≥0.92)</span>
          <span class="chip chip-warn">CTA: found (lower conf)</span>
        </div>
      </div>
      <iframe class="preview" srcdoc="{HTML_V3.replace('"', '&quot;')}"></iframe>
      <div class="diff-section">
        <h3>Additional changes vs V2</h3>
        <ul class="diff-list">
          <li class="removed">CTA removed from inside &lt;form&gt;</li>
          <li class="added">CTA added to sticky <code>&lt;footer&gt;</code></li>
          <li class="same">Text preserved: "Place Order"</li>
          <li class="changed">Class: <code>.ant-btn</code> → <code>.cta-sticky-btn</code></li>
          <li class="removed">Form context &amp; "Payment" heading gone for CTA</li>
          <li class="same">Form inputs unchanged from V2</li>
        </ul>
      </div>
    </div>

  </div>
</body>
</html>"""

if __name__ == "__main__":
    out = pathlib.Path(tempfile.mktemp(suffix=".html"))
    out.write_text(PAGE, encoding="utf-8")
    print(f"Opening {out}")
    webbrowser.open(out.as_uri())
