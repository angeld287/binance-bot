"""AWS Lambda handler for the reports/export job."""

from __future__ import annotations

import logging
from typing import Any

from core.reports.job import resolve_orchestrator

logger = logging.getLogger(__name__)


def _extract_params(event: dict | None) -> dict[str, Any]:
    event = event or {}
    return {
        "from": event.get("from"),
        "to": event.get("to"),
        "symbols": event.get("symbols"),
        "dry_run": event.get("dry_run"),
    }


def handler(event=None, context=None):  # pragma: no cover - entry point
    """AWS Lambda entry point for the reporting job."""

    orchestrator, path = resolve_orchestrator()
    params = _extract_params(event)
    logger.info("reports.handler.start path=%s params=%s", path, params)
    return orchestrator(event, now=None)
