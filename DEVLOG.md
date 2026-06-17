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

---

## Session 004 — 2026-06-12

### What was built

**demo.py** (project root) — self-contained end-to-end pipeline demonstration. Runs the full CANVAS lifecycle in a single `python demo.py` invocation: Record two element intents from a V1 checkout UI, simulate a UI redesign (different IDs, different DOM structure), collect V2 candidates, and resolve each intent through `ConfidenceGatedResolver`. Prints step-labelled output for each phase and exits with code 1 if any intent reaches FAILED status, making it CI-friendly.

**docs/architecture.md** — technical reference for QE engineers new to embeddings. Covers: ASCII flow diagram of the Record → IntentStore → Resolve pipeline; field-by-field breakdown of `SemanticDescriptor` and the `to_text()` template with a concrete example; rationale for `all-MiniLM-L6-v2` and what dim=384 normalized vectors give us; confidence gate thresholds with tuning guidance; and the SQLite schema with notes on binary blob storage and idempotent upsert.

### Key design decision

`demo.py` uses `page.set_content()` with inline HTML strings rather than loading any external URL. This makes the demo fully self-contained — no live server, no network access, no fixture setup — so community members can clone the repo and run it immediately. V1 and V2 are defined as module-level string constants, making the "UI redesign" simulation explicit and readable without any test fixtures.

### Next session
Adversarial test suite — Phase 4 complete: DOM restructuring scenarios, attribute-only changes, complete semantic redesigns, and false-positive detection across dissimilar elements.

---

## Session 005 — 2026-06-16

### What was built

**tests/adversarial/test_healing.py** — the Phase 4 adversarial test suite. It exercises the full Record → resolve lifecycle against deliberately hostile UI changes to prove that healing is driven by semantic intent rather than brittle addressing, and — just as important — that it refuses to heal when intent genuinely differs.

Ten scenarios cover both the heal path and the guardrails:

1. **DOM restructuring** — the target element moves to a different container (form → section); intent is preserved, so it must still heal.
2. **ID/class churn** — the same button reappears with entirely different brittle attributes (id/class).
3. **Label paraphrase** — an `aria-label` is reworded but carries the same intent.
4. **Section heading rename** — the surrounding heading changes (shipping → delivery).
5. **Landmark change** — the enclosing landmark changes (nav → main).
6. **False positive guard** — a completely dissimilar element must NOT auto-heal.
7. **False positive guard** — same element type, different purpose (email input vs. coupon-code input) must not be confused.
8. **Best candidate wins** — in a pool of four adversarial candidates, the resolver picks the single best semantic match.
9. **Complete semantic removal** — when candidates are empty, resolution returns FAILED rather than forcing a match.
10. **Threshold override** — a custom `threshold_auto` is respected and changes the gate decision.

### Design decisions

- **`scope="module"` fixture for the embedder/resolver setup.** The `all-MiniLM-L6-v2` model is ~90MB and dominates runtime on load. Loading it once per module (rather than per-test) keeps the adversarial suite fast while still isolating it from the other test modules.
- **BeautifulSoup, not Playwright, for the adversarial DOM.** These scenarios are about semantic matching logic, not live-browser behaviour, so candidates are built from HTML strings via `extract_from_tag`. This keeps the suite hermetic and fast — no Chromium launch, no network, no async — while exercising the same descriptor → embedding → gate path. Live-DOM extraction stays covered by the smoke test and demo.
- **Threshold-override test drives a borderline pair.** Rather than asserting on a hard-coded similarity number (brittle against model updates), the test uses a candidate whose similarity sits between two thresholds and shows that flipping `threshold_auto` flips the decision — proving the override is wired through the gate rather than checking a specific float.
- **False-positive scenarios are first-class.** Half the value of a self-healing locator is knowing when *not* to heal, so the dissimilar-element and same-type/different-purpose cases assert the resolver declines (FAILED / NEEDS_CONFIRMATION) rather than only testing the happy path.

### Next session

Phase 4 complete. Project is now ready for community release. Consider: publishing to PyPI, adding a CONTRIBUTING.md, writing a blog post for the QE community.

---

## Session 006 — 2026-06-17

### What was done

Community release and v0.2.0 enhancements shipped.

**Release tasks (all complete):**
- Renamed Python package from `canvas` to `canvas_heal` to avoid namespace collision with Canvas Medical SDK on PyPI
- Added `pyproject.toml` (hatchling build, `canvas-heal` package name) and `CONTRIBUTING.md`
- Published v0.1.0 to PyPI: `pip install canvas-heal`
- Announced on Reddit (r/softwaretesting, r/selenium) and LinkedIn

**v0.2.0 enhancements (54/54 tests passing):**
- **Shadow DOM support** — `extract_from_playwright` now uses Playwright ElementHandle instead of raw `document.querySelector`, natively piercing shadow roots
- **Visibility/disabled detection** — `is_visible` and `is_disabled` fields added to `SemanticDescriptor`; resolver skips hidden/disabled candidates by default
- **Bounding box capture** — `bounding_box` (x, y, width, height) stored per element for disambiguation of duplicate-intent elements
- **Page/flow context** — `page_url` stored in IntentStore and passed through `record()`; prevents cross-contamination between identical elements on different pages
- **Healing audit log** — `HealingEvent` dataclass; `get_audit_log()` / `clear_audit_log()` on resolver
- **JUnit XML export** — `export_junit_xml()` for CI dashboard integration (stdlib only)
- **Multilingual model support** — `IntentEmbedder.get(model_name)` singleton keyed by model; `MULTILINGUAL_MODEL` constant for non-English UIs
- **Re-record CLI** — `canvas-heal rerecord / list / audit` commands via `canvas_heal/cli.py`
- Published v0.2.0 to PyPI

---

## Session 007 — 2026-06-17

### What was done

Ecosystem, adoption, and v0.3.0 shipped.

**Documentation and community:**
- README fully overhauled — PyPI/Python badges, "What's inside v0.2.0" capabilities section, updated status, structure tree with `canvas_heal/cli.py`, Quick Start with `pip install canvas-heal` and rerecord example, full CLI reference
- `CHANGELOG.md` created at project root covering v0.1.0 and v0.2.0
- GitHub releases created for v0.1.0, v0.2.0, and v0.3.0 with release notes

**v0.3.0 enhancements (57/57 tests passing):**
- **GitHub Actions CI** — `.github/workflows/ci.yml` runs the full test suite on every push and PR across Python 3.11, 3.12, 3.13
- **pytest plugin** — `canvas_heal/pytest_plugin.py` registered as `pytest11` entry point; provides `canvas_store`, `canvas_embedder`, and `canvas_resolver` session-scoped fixtures with `--canvas-db` and `--canvas-model` CLI options — zero boilerplate for teams integrating CANVAS-HEAL
- **Selenium adapter** — `extract_from_selenium(driver, selector)` added to `descriptor.py`; selenium is an optional dependency (`pip install canvas-heal[selenium]`); guarded import with a clear error message if not installed
- **iframe support** — `extract_from_playwright_frame(page, frame_selector, element_selector)` and async variant added; unblocks elements inside payment forms (Stripe, Braintree) and embedded widgets
- Published v0.3.0 to PyPI
