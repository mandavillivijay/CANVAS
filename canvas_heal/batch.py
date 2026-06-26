"""Bulk re-record workflow for CANVAS-HEAL (Issue #11).

Processes a page-map JSON file, re-records all intents across multiple pages
in a single Playwright session, and produces a similarity diff report.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import IntentStore

_log = logging.getLogger("canvas_heal.batch")

DEFAULT_REGRESSION_THRESHOLD = 0.75


@dataclass
class PageEntry:
    url: str
    intents: list[dict]  # each: {"name": str, "selector": str}


@dataclass
class RerecordResult:
    intent_name: str
    selector: str
    url: str
    old_similarity: Optional[float]
    status: str  # "updated" | "new" | "dry-run" | "error"
    error: Optional[str] = None

    @property
    def is_regression(self) -> bool:
        return (
            self.old_similarity is not None
            and self.old_similarity < DEFAULT_REGRESSION_THRESHOLD
        )


def load_page_map(path: str | Path) -> list[PageEntry]:
    """Parse a page-map JSON file into a list of PageEntry objects."""
    with open(path) as f:
        data = json.load(f)
    return [PageEntry(url=p["url"], intents=p["intents"]) for p in data]


def print_report(
    results: list[RerecordResult],
    regression_threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> tuple[int, int, list[RerecordResult], list[RerecordResult]]:
    """Print a diff report and return (updated, new, regressions, errors)."""
    by_url: dict[str, list[RerecordResult]] = {}
    for r in results:
        by_url.setdefault(r.url, []).append(r)

    updated = new_count = 0
    regressions: list[RerecordResult] = []
    errors: list[RerecordResult] = []

    for url, page_results in by_url.items():
        print(f"\nPage: {url}")
        for r in page_results:
            sim_str = f"similarity={r.old_similarity:.3f}" if r.old_similarity is not None else "new"
            if r.status == "error":
                flag = f"  ✗ error: {r.error}"
                errors.append(r)
            elif r.status == "dry-run":
                flag = "  (dry-run)"
                if r.old_similarity is not None and r.old_similarity < regression_threshold:
                    flag += f"  ⚠ would regress (< {regression_threshold})"
            elif r.is_regression:
                flag = f"  ⚠ regression (< {regression_threshold})"
                regressions.append(r)
                updated += 1
            else:
                flag = "  ✓"
                if r.status == "new":
                    new_count += 1
                else:
                    updated += 1
            print(f"  {r.intent_name:<30}  {r.selector:<25}  {sim_str}{flag}")

    print(f"\nSummary: {updated} updated, {new_count} new, {len(regressions)} regressions, {len(errors)} errors")

    if regressions:
        print(f"\nRegressions (similarity < {regression_threshold} — verify manually):")
        for r in regressions:
            print(f"  {r.intent_name}  similarity={r.old_similarity:.3f}  ({r.url})")

    return updated, new_count, regressions, errors


class BulkRerecorder:
    """Re-records many intents across multiple pages in a single browser session."""

    def __init__(
        self,
        store: IntentStore,
        embedder: Optional[IntentEmbedder] = None,
        regression_threshold: float = DEFAULT_REGRESSION_THRESHOLD,
        dry_run: bool = False,
    ) -> None:
        self._store = store
        self._embedder = embedder or IntentEmbedder.get()
        self.regression_threshold = regression_threshold
        self.dry_run = dry_run

    def rerecord_page(
        self,
        page,
        entry: PageEntry,
        descriptor_extractor: Optional[Callable] = None,
    ) -> list[RerecordResult]:
        """Re-record all intents on one page. Accepts an injectable extractor for testing."""
        if descriptor_extractor is None:
            from canvas_heal.descriptor import extract_from_playwright
            descriptor_extractor = extract_from_playwright

        results = []
        for intent in entry.intents:
            name = intent["name"]
            selector = intent["selector"]
            try:
                descriptor = descriptor_extractor(page, selector)
                new_embedding = self._embedder.embed_descriptor(descriptor)

                existing = self._store.get(name)
                old_sim: Optional[float] = None
                if existing is not None:
                    _, _, old_embedding, _ = existing
                    old_sim = IntentEmbedder.cosine_similarity(old_embedding, new_embedding)

                if self.dry_run:
                    _log.info(
                        "dry-run intent=%r selector=%r old_sim=%s",
                        name, selector, f"{old_sim:.3f}" if old_sim is not None else "new",
                    )
                    results.append(RerecordResult(
                        intent_name=name, selector=selector, url=entry.url,
                        old_similarity=old_sim, status="dry-run",
                    ))
                else:
                    self._store.store(
                        name, selector, descriptor, new_embedding,
                        model_name=self._embedder.MODEL_NAME, page_url=entry.url,
                    )
                    status = "new" if existing is None else "updated"
                    _log.info(
                        "re-recorded intent=%r selector=%r status=%r old_sim=%s",
                        name, selector, status, f"{old_sim:.3f}" if old_sim is not None else "n/a",
                    )
                    results.append(RerecordResult(
                        intent_name=name, selector=selector, url=entry.url,
                        old_similarity=old_sim, status=status,
                    ))
            except Exception as exc:
                _log.error("error re-recording intent=%r selector=%r: %s", name, selector, exc)
                results.append(RerecordResult(
                    intent_name=name, selector=selector, url=entry.url,
                    old_similarity=None, status="error", error=str(exc),
                ))
        return results

    def run(
        self,
        page_map: list[PageEntry],
        descriptor_extractor: Optional[Callable] = None,
    ) -> list[RerecordResult]:
        """Run bulk re-record across all pages using one Playwright browser session."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "playwright is required for bulk re-record: pip install playwright"
            )

        all_results: list[RerecordResult] = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                for entry in page_map:
                    _log.info("processing page url=%r intents=%d", entry.url, len(entry.intents))
                    page = browser.new_page()
                    try:
                        page.goto(entry.url)
                        results = self.rerecord_page(page, entry, descriptor_extractor)
                        all_results.extend(results)
                    finally:
                        page.close()
            finally:
                browser.close()
        return all_results
