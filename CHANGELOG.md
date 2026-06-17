# Changelog

All notable changes to canvas-heal are documented here.

## [0.2.0] — 2026-06-17

### Added
- Shadow DOM support: `extract_from_playwright` now uses Playwright ElementHandle, natively piercing shadow roots
- `is_visible` and `is_disabled` fields on `SemanticDescriptor`; resolver skips hidden/disabled candidates by default (`skip_hidden=True`)
- `bounding_box` field (x, y, width, height) on `SemanticDescriptor` for disambiguation
- `page_url` parameter on `record()` and stored in `IntentStore` to scope intents to a specific page
- Healing audit log: `get_audit_log()`, `clear_audit_log()` on `ConfidenceGatedResolver`
- JUnit XML export: `export_junit_xml(path)` for CI dashboard integration
- Multilingual model support: `IntentEmbedder.get(model_name)` accepts any sentence-transformers model; `MULTILINGUAL_MODEL` constant provided
- CLI entry point `canvas-heal` with subcommands: `rerecord`, `list`, `audit`
- `CONTRIBUTING.md` for community onboarding

### Changed
- Python package renamed from `canvas` to `canvas_heal` to avoid namespace collision with Canvas Medical SDK

## [0.1.0] — 2026-06-16

### Added
- Initial release
- Semantic element descriptor extraction (`SemanticDescriptor`, `extract_from_tag`, `extract_from_playwright`)
- Intent embedding via `all-MiniLM-L6-v2` (sentence-transformers, local, no API key)
- Confidence-gated resolver: HEALED (≥0.92) / NEEDS_CONFIRMATION (≥0.75) / FAILED (<0.75)
- SQLite-backed `IntentStore`
- Adversarial test suite: DOM restructuring, attribute churn, label paraphrase, false-positive detection
- Self-contained `demo.py`
- `docs/architecture.md` and `docs/pytest_integration.md`
