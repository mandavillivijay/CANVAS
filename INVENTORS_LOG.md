# CANVAS Inventors Log

## Session 001 — 2026-06-12

### What was done
Initialized project repository and folder structure.

### Design decisions
Chose Python as primary language for access to sentence-transformers and Playwright bindings without needing additional tooling.

Chose local-only embedding model (sentence-transformers) to ensure the invention is self-contained and not dependent on any external API.

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
