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

## Status

Phase 3 complete — Confidence-Gated Resolver implemented and tested (22/22 tests passing).
Phase 4 (Adversarial Testing) in progress.

## Roadmap

| Phase | Name | Status |
|-------|------|--------|
| 1 | Semantic Element Descriptor | Complete |
| 2 | Intent Embedding | Complete |
| 3 | Confidence-Gated Resolver | Complete |
| 4 | Adversarial Testing | In progress |

## Structure

```
canvas/         — core source code
  descriptor.py — DOM element descriptor extraction
  embedder.py   — sentence-transformers embedding pipeline
  resolver.py   — confidence-gated semantic resolver + SQLite intent store
tests/          — unit tests and adversarial test scenarios
docs/           — technical documentation
```

## Quick Start

```bash
pip install -r requirements.txt
python -m playwright install chromium
python -m pytest tests/ -v
```

## Contributing

CANVAS is in early prototype stage. Community feedback, issues, and pull requests are
welcome once Phase 4 (adversarial testing) is complete. Watch the repo or open a discussion
if you want to get involved earlier.

## License

MIT — see [LICENSE](LICENSE)
