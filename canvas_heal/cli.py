from __future__ import annotations

import argparse
import sys

from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import ConfidenceGatedResolver, IntentStore


def _cmd_rerecord(args: argparse.Namespace) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Error: Playwright is not installed. Install it with 'pip install playwright' and run 'playwright install'.", file=sys.stderr)
        return 1

    from canvas_heal.descriptor import extract_from_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(args.url)
            descriptor = extract_from_playwright(page, args.selector)

            store = IntentStore(args.db, store_raw_text=not args.scrub_text)
            resolver = ConfidenceGatedResolver(store, IntentEmbedder.get(args.model))
            resolver.record(args.name, args.selector, descriptor)
            store.close()

            print(f"Recorded: {args.name}")
            print(f"Selector: {args.selector}")
            print(f"Descriptor: {descriptor.to_text()}")
            print(f"Model: {args.model}")
        finally:
            browser.close()
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    store = IntentStore(args.db)
    try:
        intents = store.all()
        if not intents:
            print("No intents stored.")
            return 0
        for name, selector in intents:
            print(f"{name}  →  {selector}")
    finally:
        store.close()
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    store = IntentStore(args.db)
    try:
        print(f"Total intents: {store.count()}")
        for name, selector in store.all():
            print(f"{name}  →  {selector}")
    finally:
        store.close()
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    store = IntentStore(args.db)
    try:
        versions = store.get_version_history(args.name)
        if not versions:
            print(f"No version history found for intent '{args.name}'.")
            return 0
        print(f"Version history for '{args.name}' ({len(versions)} versions):\n")
        for v in versions:
            print(f"  [{v.id:>4}]  {v.recorded_at}  by {v.recorded_by or '(unknown)'}")
            print(f"          selector: {v.selector}")
            print(f"          model:    {v.model_name}")
            print(f"          text:     {v.descriptor_text[:80]}")
            print()
    finally:
        store.close()
    return 0


def _cmd_rollback(args: argparse.Namespace) -> int:
    store = IntentStore(args.db)
    try:
        ok = store.rollback(args.name, args.version_id)
        if not ok:
            print(f"Error: version {args.version_id} not found for intent '{args.name}'.", file=sys.stderr)
            return 1
        print(f"Rolled back '{args.name}' to version {args.version_id}.")
    finally:
        store.close()
    return 0


def _cmd_rerecord_all(args: argparse.Namespace) -> int:
    from canvas_heal.batch import BulkRerecorder, load_page_map, print_report
    try:
        page_map = load_page_map(args.page_map)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error reading page map: {exc}", file=sys.stderr)
        return 1

    store = IntentStore(args.db, store_raw_text=not args.scrub_text)
    rerecorder = BulkRerecorder(
        store,
        IntentEmbedder.get(args.model),
        regression_threshold=args.threshold,
        dry_run=args.dry_run,
    )
    try:
        results = rerecorder.run(page_map)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    _, _, regressions, errors = print_report(results, regression_threshold=args.threshold)
    return 1 if (errors or regressions) else 0


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1e-9
    return "".join(
        _SPARK_CHARS[min(int((v - lo) / span * (len(_SPARK_CHARS) - 1)), len(_SPARK_CHARS) - 1)]
        for v in values
    )


def _cmd_drift(args: argparse.Namespace) -> int:
    store = IntentStore(args.db)
    try:
        trends = store.get_all_confidence_trends(window=args.window)
        if not trends:
            print("No confidence history found. Run some heal() calls first.")
            return 0
        threshold = args.threshold
        print(f"{'Intent':<30}  {'Avg':>6}  {'N':>4}  {'Drift?':>6}  Sparkline")
        print("-" * 72)
        for t in trends:
            avg = t["rolling_avg"]
            drifting = avg is not None and t["sample_count"] >= args.window and avg < threshold
            avg_str = f"{avg:.3f}" if avg is not None else "  n/a"
            drift_str = "⚠ YES" if drifting else "  ok"
            spark = _sparkline(t["values"])
            print(f"{t['intent_name']:<30}  {avg_str:>6}  {t['sample_count']:>4}  {drift_str:>6}  {spark}")
    finally:
        store.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="canvas-heal", description="CANVAS-HEAL semantic self-healing locator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_rerecord = subparsers.add_parser("rerecord", help="Re-record an intent fingerprint from a live page")
    p_rerecord.add_argument("--url", required=True, help="Page URL to navigate to")
    p_rerecord.add_argument("--selector", required=True, help="CSS selector of the target element")
    p_rerecord.add_argument("--name", required=True, help="Intent name to store under")
    p_rerecord.add_argument("--db", required=True, help="Path to the intent store database")
    p_rerecord.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    p_rerecord.add_argument("--scrub-text", action="store_true", help="Scrub PII (emails, phone numbers) from stored text fields")
    p_rerecord.set_defaults(func=_cmd_rerecord)

    p_list = subparsers.add_parser("list", help="List all stored intents")
    p_list.add_argument("--db", required=True, help="Path to the intent store database")
    p_list.set_defaults(func=_cmd_list)

    p_audit = subparsers.add_parser("audit", help="Show stored intent count and details")
    p_audit.add_argument("--db", required=True, help="Path to the intent store database")
    p_audit.set_defaults(func=_cmd_audit)

    p_history = subparsers.add_parser("history", help="Show version history for a recorded intent")
    p_history.add_argument("--name", required=True, help="Intent name")
    p_history.add_argument("--db", required=True, help="Path to the intent store database")
    p_history.set_defaults(func=_cmd_history)

    p_rollback = subparsers.add_parser("rollback", help="Restore an intent to a previous version")
    p_rollback.add_argument("--name", required=True, help="Intent name")
    p_rollback.add_argument("--version-id", required=True, type=int, dest="version_id", help="Version ID from 'canvas-heal history'")
    p_rollback.add_argument("--db", required=True, help="Path to the intent store database")
    p_rollback.set_defaults(func=_cmd_rollback)

    p_rerecord_all = subparsers.add_parser("rerecord-all", help="Bulk re-record intents from a page-map JSON file")
    p_rerecord_all.add_argument("--page-map", required=True, dest="page_map", help="Path to page-map JSON file")
    p_rerecord_all.add_argument("--db", required=True, help="Path to the intent store database")
    p_rerecord_all.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    p_rerecord_all.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold for regression flagging (default: 0.75)")
    p_rerecord_all.add_argument("--dry-run", action="store_true", dest="dry_run", help="Show what would change without writing to the database")
    p_rerecord_all.add_argument("--scrub-text", action="store_true", dest="scrub_text", help="Scrub PII from stored text fields")
    p_rerecord_all.set_defaults(func=_cmd_rerecord_all)

    p_drift = subparsers.add_parser("drift", help="Show confidence drift trend for all intents")
    p_drift.add_argument("--db", required=True, help="Path to the intent store database")
    p_drift.add_argument("--window", type=int, default=30, help="Number of recent resolve calls to analyse (default: 30)")
    p_drift.add_argument("--threshold", type=float, default=0.85, help="Drift threshold (default: 0.85)")
    p_drift.set_defaults(func=_cmd_drift)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
