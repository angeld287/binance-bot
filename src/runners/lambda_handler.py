from __future__ import annotations

import logging

from core.application.execution import run_iteration

logger = logging.getLogger(__name__)


def handler(event=None, context=None):  # pragma: no cover - entry point
    """AWS Lambda entry point that delegates to :func:`run_iteration`."""
    try:
        result = run_iteration()
        return {"statusCode": 200, "body": result}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Iteration failed: %s", exc)
        return {"statusCode": 500, "body": {"error": str(exc)}}
