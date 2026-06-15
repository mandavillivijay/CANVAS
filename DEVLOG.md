# CANVAS Development Log

A running record of design decisions, implementation progress, and lessons learned
as CANVAS is built as an open source prototype for the QE and test automation community.

---

## Session 001 — 2026-06-12

### What was done
Initialized project repository and folder structure.

### Design decisions
Chose Python as primary language for access to sentence-transformers and Playwright bindings without needing additional tooling.

Chose local-only embedding model (sentence-transformers) to ensure CANVAS is self-contained and does not depend on any external API.

### Next session
Install dependencies: playwright, sentence-transformers, numpy, beautifulsoup4. Verify toolchain end to end.

---

## Session 002 — 2026-06-12

### What was done
Verified full toolchain end to end.

- Confirmed all dependencies already present in the Python 3.13 environment: playwright 1.55.0, sentence-transformers 5.1.1, numpy 2.3.3, beautifulsoup4 4.12.2, pytest 9.0.1.
- Ran `python -m playwright install chromium` — Chromium browser binary installed successfully (exit 0).
- Created `tests/smoke_test.py`: launches headless Chromium, opens https://example.com, extracts h1 tag name, inner text, and any aria/role attributes, prints results, closes browser.
- Ran smoke test — output confirmed:
  - tag: h1
  - text: Example Domain
  - aria attrs: (none)
  - PASS: Playwright + Chromium toolchain verified

### Issues encountered
None. All packages were pre-installed in the environment; no dependency conflicts observed.

### Design decisions
Using Python 3.13 (vs. planned 3.11+) — no compatibility issues observed with any package. Will proceed on 3.13.

### Next session
Begin implementing core modules:
- `canvas/descriptor.py` — DOM element descriptor extraction (role, label, structural context, workflow position)
- `canvas/embedder.py` — sentence-transformers embedding pipeline (all-MiniLM-L6-v2)
- `canvas/resolver.py` — confidence-gated semantic resolver with heal / confirm / fail decision logic

---

## Session 003 — 2026-06-12

### What was done
Implemented all three core modules and their unit test suites. 22/22 tests pass.

**canvas/descriptor.py**
- `SemanticDescriptor` dataclass: tag, role, label, element_type, placeholder, text_content, parent_tag, parent_role, section_heading, landmark
- `to_text()` produces a natural-language intent string (e.g. "button labeled 'Submit Order' inside form under heading 'Checkout'") used as embedding input
- `extract_from_tag(el)` — BeautifulSoup extractor for unit testing and offline use
- `extract_from_playwright(page, selector)` — live DOM extractor via JS evaluation; handles implicit ARIA role inference, nearest heading traversal, and landmark detection

**canvas/embedder.py**
- `IntentEmbedder` singleton wrapping `all-MiniLM-L6-v2`
- `embed(text)` → normalized float32 vector (dim=384)
- `embed_descriptor(descriptor)` → calls `to_text()` then embeds
- `cosine_similarity(a, b)` — dot product of pre-normalized vectors

**canvas/resolver.py**
- `IntentStore` — SQLite-backed store (supports `:memory:` for tests); persists selector, descriptor JSON, and embedding blob
- `ConfidenceGatedResolver` — record-time embedding + resolve-time nearest-neighbour lookup with three gates:
  - ≥ 0.92: HEALED (auto-heal)
  - ≥ 0.75: NEEDS_CONFIRMATION
  - < 0.75: FAILED

### Test results
```
tests/test_descriptor.py   10 passed  0.51s
tests/test_embedder.py      6 passed
tests/test_resolver.py      6 passed
Total: 22 passed in ~150s (model load dominates)
```

### Issues encountered
HuggingFace hub symlink warning on Windows (non-functional, cosmetic only). Suppressed via `HF_HUB_DISABLE_SYMLINKS_WARNING` env var if needed.

### Design decisions
- `to_text()` template omits generic parent tags (div, span, body) to keep intent strings semantically meaningful
- Threshold constants (`THRESHOLD_AUTO_HEAL=0.92`, `THRESHOLD_CONFIRM=0.75`) are module-level for easy tuning
- `IntentEmbedder` uses a class-level singleton to avoid reloading the 90MB model across test runs
- SQLite `:memory:` path supported in `IntentStore` to enable fast, isolated unit tests

### Next session
Phase 4: Adversarial test scenarios — DOM restructuring, attribute changes, complete redesigns, false-positive detection
