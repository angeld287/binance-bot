from __future__ import annotations

from datetime import datetime
import logging

from config.settings import load_settings

logger = logging.getLogger(__name__)


def run_iteration(now: datetime | None = None) -> dict[str, object]:
    """Execute a single iteration of the bot orchestration."""
    current_time = now or datetime.utcnow()
    settings = load_settings()
    logger.info("Running iteration for %s at %s", settings.STRATEGY_NAME, current_time.isoformat())
    return {"ok": True, "strategy": settings.STRATEGY_NAME}
