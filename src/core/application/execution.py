from __future__ import annotations

from datetime import datetime
import logging

from adapters.brokers.binance import make_broker
from adapters.data_providers.binance import make_market_data
from config.settings import load_settings, Settings
from strategies import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


def _resolve_market_data(settings: Settings):
    if settings.FEATURE_DATASOURCE == "binance":
        return make_market_data(settings)
    raise ValueError(f"Unsupported datasource: {settings.FEATURE_DATASOURCE}")


def _resolve_broker(settings: Settings):
    if settings.FEATURE_BROKER == "binance":
        return make_broker(settings)
    raise ValueError(f"Unsupported broker: {settings.FEATURE_BROKER}")


def run_iteration(now: datetime | None = None) -> dict[str, object]:
    """Execute a single iteration of the bot orchestration."""

    current_time = now or datetime.utcnow()
    settings = load_settings()
    logger.info(
        "Running iteration for %s at %s", settings.STRATEGY_NAME, current_time.isoformat()
    )

    market_data = _resolve_market_data(settings)
    broker = _resolve_broker(settings)

    strategy_cls = STRATEGY_REGISTRY[settings.STRATEGY_NAME]
    strategy = strategy_cls(
        market_data=market_data, broker=broker, settings=settings
    )

    signal = strategy.generate_signal(current_time)

    return {"ok": True, "strategy": settings.STRATEGY_NAME, "signal": signal}
