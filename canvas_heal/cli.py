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

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
