# Contributing to CANVAS-HEAL

Thanks for your interest in contributing. CANVAS-HEAL is an early-stage open source prototype — feedback, bug reports, and pull requests are all welcome.

## Getting started

```bash
git clone https://github.com/mandavillivijay/CANVAS.git
cd canvas-heal
pip install -e ".[dev]"
python -m playwright install chromium
```

## Running the tests

```bash
pytest tests/ -v
```

The full suite takes ~2–3 minutes on first run because the `all-MiniLM-L6-v2` model (~90MB) is downloaded and loaded. Subsequent runs are faster as the model is cached locally.

## Project structure

```
canvas/         — core library (descriptor, embedder, resolver)
tests/          — unit tests and adversarial scenarios
docs/           — technical documentation
demo.py         — self-contained end-to-end demo
```

## What to work on

- **Bug reports** — open a GitHub issue with a minimal reproduction
- **Threshold tuning** — if you find real-world cases where the 0.92 / 0.75 gates misfire, open an issue with the similarity scores
- **New descriptor fields** — `canvas/descriptor.py` extracts semantic context from the DOM; additional signals (e.g. computed styles, tab order) are welcome
- **Framework integrations** — the resolver is framework-agnostic; adapters for Selenium, Cypress, or other runners are in scope
- **Performance** — candidate pre-embedding (`resolver.precompute_candidates`) is the main lever; ideas for batching or caching at scale are welcome

## Pull request guidelines

- Keep PRs focused — one concern per PR
- Include or update tests for any behaviour change
- Run `pytest tests/ -v` and confirm all tests pass before opening a PR
- Update `DEVLOG.md` with a brief session note if the change is non-trivial

## Code style

Standard Python — no formatter is enforced yet, but match the style of the surrounding code. Type annotations are used throughout; please keep new code annotated.

## License

By contributing you agree that your contributions will be licensed under the MIT License.
