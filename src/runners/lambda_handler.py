from __future__ import annotations

import logging
import os

from config.logging import setup_logging
from config.settings import load_settings
from core.application.execution import run_iteration

logger = logging.getLogger(__name__)


def handler(event=None, context=None):  # pragma: no cover - entry point
    """AWS Lambda entry point that delegates to :func:`run_iteration`."""
    settings = load_settings()
    setup_logging(level=settings.LOG_LEVEL, mode="plain")

    logger.info(
        "══════════════════ 🚀🚀🚀 INICIO EJECUCIÓN LAMBDA 🚀🚀🚀 ═══════════════════"
    )

    if os.getenv("USE_PROXY"):
        logger.info("Proxy: ENABLED (HTTP proxy)")
    else:
        logger.info("Proxy: DISABLED (NAT)")

    try:
        result = run_iteration(event_in=event)
        status = 200
        body = result
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Fallo en iteración: %s", exc)
        status = 500
        body = {"error": str(exc)}
    finally:
        logger.info(
            "══════════════════ 🛑🛑🛑 FIN EJECUCIÓN LAMBDA 🛑🛑🛑 ═══════════════════"
        )

    return {"statusCode": status, "body": body}
