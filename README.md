[![PyPI version](https://img.shields.io/pypi/v/canvas-heal.svg)](https://pypi.org/project/canvas-heal/) [![Python](https://img.shields.io/pypi/pyversions/canvas-heal.svg)](https://pypi.org/project/canvas-heal/) [![Coverage](https://img.shields.io/badge/coverage-≥80%25%20branch-brightgreen)](https://github.com/mandavillivijay/CANVAS/actions)

# CANVAS
Context-Aware Navigation and Visual Anchoring System for Selectors

An open source research prototype for semantic intent-based self-healing test locators,
built for the QE and test automation community.

## Why CANVAS

Existing self-healing tools use attribute-based matching — CSS selectors, XPath, element IDs,
class names. These break whenever a UI undergoes structural redesign, even when the element's
function is unchanged.

CANVAS takes a different approach: **semantic intent anchoring**. At record time it encodes
*what an element does* — its role, accessible label, workflow position, and surrounding context —
into a dense vector using a local sentence-transformers model. At runtime it embeds live DOM
candidates and finds the nearest semantic match, then gates the decision:

- **Auto-heal** if confidence is high enough
- **Request human confirmation** if confidence is in a grey zone
- **Fail explicitly** if no confident match exists

This means a "Submit Order" button that moves from a sidebar to a modal, or gets its ID changed
from `#btn-submit` to `#cta-primary`, can still be found — because its *intent* is preserved
even when its *address* is not.

## What's inside (v0.2.0)

- **Semantic intent anchoring** — encodes role, accessible label, heading context, landmark, and visible text into a dense vector
- **Shadow DOM piercing** — `extract_from_playwright` uses the Playwright ElementHandle approach, natively reaching into shadow roots
- **Visibility and disabled-state filtering** — the resolver skips hidden and disabled candidates automatically
- **Bounding box capture** — `(x, y, width, height)` geometry to disambiguate duplicate-intent elements
- **Page/flow context** — intents are scoped to a URL, so a checkout `Submit` is never confused with a login `Submit`
- **Healing audit log + JUnit XML export** — structured healing decisions for CI dashboards
- **Multilingual model support** — swap in `paraphrase-multilingual-MiniLM-L12-v2` (or any sentence-transformers model)
- **Re-record CLI** — `canvas-heal rerecord / list / audit`

## Status

All phases complete. **v0.2.0 published** to PyPI as [`canvas-heal`](https://pypi.org/project/canvas-heal/).

## Roadmap

| Phase | Name | Status |
|-------|------|--------|
| 1 | Semantic Element Descriptor | Complete |
| 2 | Intent Embedding | Complete |
| 3 | Confidence-Gated Resolver | Complete |
| 4 | Adversarial Testing | Complete |

## Structure

```
canvas_heal/        — core source code
  descriptor.py     — DOM element descriptor extraction (+ shadow DOM, bounding box)
  embedder.py       — sentence-transformers embedding pipeline
  resolver.py       — confidence-gated semantic resolver + SQLite intent store
  cli.py            — canvas-heal command-line interface
tests/              — unit tests and adversarial test scenarios
docs/               — technical documentation
```

## Quick Start

```bash
pip install canvas-heal
python -m playwright install chromium
python -m pytest tests/ -v
# Re-record an intent against a live URL
canvas-heal rerecord --url https://yourapp.com/checkout --selector "#submit-btn" --name checkout_submit --db tests/intents.db
```

## CLI reference

The `canvas-heal` command provides three subcommands:

- **`canvas-heal rerecord`** — capture a fresh intent from a live page.
  Flags: `--url <page>`, `--selector <css>`, `--name <intent-name>`, `--db <path>`.
- **`canvas-heal list`** — list recorded intents in a database.
  Flags: `--db <path>`.
- **`canvas-heal audit`** — print the healing audit log.
  Flags: `--db <path>`.

## Contributing

CANVAS is in early prototype stage. Community feedback, issues, and pull requests are
welcome once Phase 4 (adversarial testing) is complete. Watch the repo or open a discussion
if you want to get involved earlier.

## License

MIT — see [LICENSE](LICENSE)
