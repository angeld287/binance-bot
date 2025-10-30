"""CLI helper to execute the reports job locally."""

from __future__ import annotations

import argparse
import json
from typing import Any

from core.reports.job import resolve_orchestrator


def _build_event(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if args.from_:
        payload["from"] = args.from_
    if args.to:
        payload["to"] = args.to
    if args.symbols:
        payload["symbols"] = args.symbols
    if args.dry_run:
        payload["dry_run"] = True
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reports/export job")
    parser.add_argument("--from", dest="from_", help="Start date (ISO8601)")
    parser.add_argument("--to", dest="to", help="End date (ISO8601)")
    parser.add_argument("--symbols", nargs="*", help="Symbols to export")
    parser.add_argument("--dry-run", action="store_true", help="Run without persistence")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    event = _build_event(args)
    orchestrator, path = resolve_orchestrator()
    result = orchestrator(event or None, now=None)
    print(json.dumps({"path": path, "result": result}, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    raise SystemExit(main())
