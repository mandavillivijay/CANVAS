# CANVAS Architecture

Technical reference for contributors and QE engineers integrating CANVAS into a test suite.

---

## 1. System Overview

CANVAS runs in two phases: **record** (baseline capture at test authoring time) and **resolve** (automatic selector recovery at test runtime when the DOM has changed).

```
  RECORD TIME
  ──────────────────────────────────────────────────────
  Playwright page  ──►  extract_from_playwright()
                           │
                           ▼
                      SemanticDescriptor
                           │
                    IntentEmbedder.embed_descriptor()
                           │
                           ▼  (float32[384] blob)
                      IntentStore (SQLite)
                        intents table
                           │
                     stored under name
                       e.g. "order_button"


  RESOLVE TIME
  ──────────────────────────────────────────────────────
  V2 DOM candidates  ──►  extract_from_playwright() × N
                                │
                         embed each candidate
                                │
                    cosine_similarity(stored, candidate)
                                │
                     ConfidenceGatedResolver
                      │          │          │
                   ≥ 0.92     0.75–0.92   < 0.75
                   HEALED   NEEDS_CONFIRM  FAILED
```

---

## 2. SemanticDescriptor

`SemanticDescriptor` (`canvas/descriptor.py`) captures the **semantic identity** of a DOM element — not its brittle CSS selector or XPath, but what it *means* and where it sits in the page.

| Field | Source | Why it matters |
|---|---|---|
| `tag` | element tag name | coarse type signal (`button`, `input`) |
| `role` | explicit `role=""` or inferred from tag | ARIA role is more stable than tag alone |
| `label` | `aria-label` or `title` | author-provided human name; very stable |
| `element_type` | `type=""` attribute | distinguishes `email` from `text` input |
| `placeholder` | `placeholder=""` | secondary label signal when aria-label absent |
| `text_content` | `innerText` (first 120 chars) | button copy; changes less often than IDs |
| `parent_tag` | immediate parent tag | structural proximity |
| `parent_role` | inferred role of parent | helps distinguish button-in-nav from button-in-form |
| `section_heading` | nearest preceding h1–h6 | workflow position ("Checkout", "Confirm Your Order") |
| `landmark` | nearest landmark ancestor (`form`, `main`, `nav`, …) | page region |

`to_text()` serializes these fields into a natural-language string used as embedding input:

```
button labeled 'Place your order' inside form under heading 'Checkout'
```

Fields are omitted when empty or generic (e.g. `div`, `span`, `body` are skipped as parent tags because they carry no semantic weight). This keeps the embedding focused on meaningful signal.

---

## 3. Embedding

**Model:** `all-MiniLM-L6-v2` from the `sentence-transformers` library.

Reasons for this choice:
- 384-dimensional output — small enough to fit thousands of intents in memory, large enough to capture rich semantics
- Pre-trained on 1B+ sentence pairs; strong on short descriptive phrases
- Ships as a single ~90 MB download; no API key, no network dependency at resolve time
- Well-benchmarked on semantic textual similarity (STS) tasks, which is exactly what CANVAS needs

Vectors are L2-normalized at embed time (`normalize_embeddings=True`). This makes cosine similarity equivalent to a dot product, reducing the similarity computation to a single `np.dot()` call and keeping resolve time O(n) in the number of candidates.

The 384-float32 vector for each intent occupies 1,536 bytes, stored as a raw binary blob in SQLite. For a test suite with 500 intents that is ~750 KB — negligible.

---

## 4. Confidence Gates

After computing cosine similarity between the stored intent vector and each V2 candidate, `ConfidenceGatedResolver` bins the best score into three zones:

| Zone | Threshold | Meaning | Recommended action |
|---|---|---|---|
| **HEALED** | score ≥ 0.92 | High confidence — same semantic intent, just restructured | Proceed automatically; log the new selector for review |
| **NEEDS_CONFIRMATION** | 0.75 ≤ score < 0.92 | Plausible match but uncertain | Pause test; surface to engineer for manual confirmation; optionally update the stored fingerprint |
| **FAILED** | score < 0.75 | No credible match found | Fail the test explicitly; the element may be genuinely absent or the UI semantics changed too much |

**Tuning guidance:**

- The thresholds are module-level constants (`THRESHOLD_AUTO_HEAL = 0.92`, `THRESHOLD_CONFIRM = 0.75` in `canvas/resolver.py`) and can be overridden per resolver instance via constructor arguments.
- Lower `threshold_auto` to heal more aggressively (accept more false positives).
- Raise `threshold_confirm` to widen the FAILED zone (reject more uncertain matches).
- A reasonable first experiment: run the resolver in dry-run mode on a known-good redesign, observe the score distribution, then set `threshold_auto` just below the 5th-percentile score of true-positive matches.

---

## 5. SQLite Intent Store

**Schema:**

```sql
CREATE TABLE IF NOT EXISTS intents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    selector    TEXT    NOT NULL,
    descriptor  TEXT    NOT NULL,   -- JSON of SemanticDescriptor fields
    embedding   BLOB    NOT NULL,   -- raw float32[384] bytes (1536 bytes)
    created_at  TEXT    DEFAULT (datetime('now'))
);
```

**Design notes:**

- `name` is the stable logical key (e.g. `"order_button"`). It is the only thing the test suite needs to pass at resolve time.
- `descriptor` is stored as JSON for human readability and debugging; the embedding is stored as a raw binary blob for zero-overhead deserialization (`np.frombuffer(blob, dtype=np.float32)`).
- `INSERT OR REPLACE` makes `store()` idempotent: re-recording an intent after a deliberate UI update overwrites the old fingerprint cleanly.
- Pass `":memory:"` as `db_path` to `IntentStore` for in-process, zero-disk tests. This is what the unit test suite and `demo.py` use.
- For production use, commit the `.db` file alongside the test suite in version control so the baseline travels with the code.
