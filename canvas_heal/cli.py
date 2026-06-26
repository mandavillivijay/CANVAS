from __future__ import annotations

import argparse
import sys

from canvas_heal.embedder import IntentEmbedder
from canvas_heal.resolver import ConfidenceGatedResolver, IntentStore, open_store


def _open_store(args: argparse.Namespace) -> IntentStore:
    """Build an IntentStore from CLI args (supports both --db and --store-url)."""
    store_url: str | None = getattr(args, "store_url", None)
    team_id: str = getattr(args, "team_id", "")
    project_id: str = getattr(args, "project_id", "")
    scrub: bool = getattr(args, "scrub_text", False)

    if store_url:
        return open_store(
            store_url,
            store_raw_text=not scrub,
            team_id=team_id,
            project_id=project_id,
        )
    return IntentStore(
        args.db,
        store_raw_text=not scrub,
        team_id=team_id,
        project_id=project_id,
    )


def _add_store_args(parser: argparse.ArgumentParser, *, require_db: bool = True) -> None:
    """Attach the standard store location / namespace arguments to a subcommand."""
    group = parser.add_mutually_exclusive_group(required=require_db)
    group.add_argument("--db", default=None, help="Path to SQLite intent database")
    group.add_argument(
        "--store-url",
        dest="store_url",
        default=None,
        metavar="URL",
        help="Store URL (sqlite:///path or postgresql://user:pass@host/db)",
    )
    parser.add_argument("--team-id", dest="team_id", default="", help="Team namespace")
    parser.add_argument("--project-id", dest="project_id", default="", help="Project namespace")


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

            store = _open_store(args)
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
    store = _open_store(args)
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
    store = _open_store(args)
    try:
        print(f"Total intents: {store.count()}")
        for name, selector in store.all():
            print(f"{name}  →  {selector}")
    finally:
        store.close()
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    store = _open_store(args)
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
    store = _open_store(args)
    try:
        ok = store.rollback(args.name, args.version_id)
        if not ok:
            print(f"Error: version {args.version_id} not found for intent '{args.name}'.", file=sys.stderr)
            return 1
        print(f"Rolled back '{args.name}' to version {args.version_id}.")
    finally:
        store.close()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed. Install it with 'pip install canvas-heal[server]'.", file=sys.stderr)
        return 1

    from canvas_heal.server import create_app

    store_url: str | None = getattr(args, "store_url", None)
    app = create_app(
        model=args.model,
        store_url=store_url,
        db_path=None if store_url else getattr(args, "db", None),
        team_id=getattr(args, "team_id", ""),
        project_id=getattr(args, "project_id", ""),
        threshold_auto=args.threshold_auto,
        threshold_confirm=args.threshold_confirm,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="canvas-heal", description="CANVAS-HEAL semantic self-healing locator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_rerecord = subparsers.add_parser("rerecord", help="Re-record an intent fingerprint from a live page")
    p_rerecord.add_argument("--url", required=True, help="Page URL to navigate to")
    p_rerecord.add_argument("--selector", required=True, help="CSS selector of the target element")
    p_rerecord.add_argument("--name", required=True, help="Intent name to store under")
    _add_store_args(p_rerecord)
    p_rerecord.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    p_rerecord.add_argument("--scrub-text", action="store_true", dest="scrub_text", help="Scrub PII from stored text fields")
    p_rerecord.set_defaults(func=_cmd_rerecord)

    p_list = subparsers.add_parser("list", help="List all stored intents")
    _add_store_args(p_list)
    p_list.set_defaults(func=_cmd_list)

    p_audit = subparsers.add_parser("audit", help="Show stored intent count and details")
    _add_store_args(p_audit)
    p_audit.set_defaults(func=_cmd_audit)

    p_history = subparsers.add_parser("history", help="Show version history for a recorded intent")
    p_history.add_argument("--name", required=True, help="Intent name")
    _add_store_args(p_history)
    p_history.set_defaults(func=_cmd_history)

    p_rollback = subparsers.add_parser("rollback", help="Restore an intent to a previous version")
    p_rollback.add_argument("--name", required=True, help="Intent name")
    p_rollback.add_argument("--version-id", required=True, type=int, dest="version_id", help="Version ID from 'canvas-heal history'")
    _add_store_args(p_rollback)
    p_rollback.set_defaults(func=_cmd_rollback)

    p_serve = subparsers.add_parser("serve", help="Run canvas-heal as a shared REST embedding service")
    _add_store_args(p_serve, require_db=False)
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_serve.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    p_serve.add_argument("--threshold-auto", type=float, dest="threshold_auto", default=0.92)
    p_serve.add_argument("--threshold-confirm", type=float, dest="threshold_confirm", default=0.75)
    p_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
