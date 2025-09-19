from __future__ import annotations

from datetime import datetime
import logging
import os
import time

from adapters.brokers.binance import make_broker
from adapters.data_providers.binance import make_market_data
from config.settings import load_settings, Settings
from strategies import STRATEGY_REGISTRY
from core.ports.settings import get_symbol

logger = logging.getLogger("bot.exec")


def _resolve_market_data(settings: Settings):
    if settings.FEATURE_DATASOURCE == "binance":
        return make_market_data(settings)
    raise ValueError(f"Unsupported datasource: {settings.FEATURE_DATASOURCE}")


def _resolve_broker(settings: Settings):
    if settings.FEATURE_BROKER == "binance":
        return make_broker(settings)
    raise ValueError(f"Unsupported broker: {settings.FEATURE_BROKER}")


def run_iteration(event_in: dict | None = None, now: datetime | None = None) -> dict:
    """Execute a single iteration of the bot orchestration."""

    current_time = now or datetime.utcnow()
    settings = load_settings()
    logger.info(
        "Running iteration for %s at %s", settings.STRATEGY_NAME, current_time.isoformat()
    )
    logger.info(
        "Active config: %s",
        {
            "STRATEGY_NAME": settings.STRATEGY_NAME,
            "FEATURE_BROKER": settings.FEATURE_BROKER,
            "SYMBOL": settings.SYMBOL,
            "INTERVAL": settings.INTERVAL,
        },
    )
    symbol = get_symbol(settings)
    logger.info("sym_norm=%s", symbol)

    market_data = _resolve_market_data(settings)
    try:
        server_ms = market_data.get_server_time_ms()
        local_ms = int(time.time() * 1000)
        drift_ms = local_ms - server_ms
        safety_ms = int(os.getenv("SAFETY_MS", "300"))
        offset_ms = safety_ms - drift_ms
        logger.info(
            "Binance timing: serverTime=%d localTime=%d drift_ms=%+d safety_ms=%d offset_ms=%+d",
            server_ms,
            local_ms,
            drift_ms,
            safety_ms,
            offset_ms,
        )
    except Exception as exc:  # pragma: no cover - network failures or unsupported
        logger.debug("Unable to compute timing drift: %s", exc)
    try:
        price = market_data.get_price(symbol)
        logger.info("Current price for %s: %f", symbol, price)
    except Exception as exc:  # pragma: no cover - network failures
        logger.warning("Failed to fetch current price: %s", exc)
    broker = _resolve_broker(settings)

    strategy_cls = STRATEGY_REGISTRY[settings.STRATEGY_NAME]

    # Constructor compatible: intenta inyectar dependencias (legacy); si no, sin args (nuevo)
    try:
        strategy = strategy_cls(
            market_data=market_data,
            broker=broker,
            settings=settings,
        )
    except TypeError:
        strategy = strategy_cls()

    # Método compatible: usa .run si existe; si no, cae a generate_signal (legacy)
    if hasattr(strategy, "run"):
        result = strategy.run(
            exchange=broker,
            market_data=market_data,
            settings=settings,
            now_utc=current_time,
            event=event_in,  # pasa el event para estrategias como liquidity-sweep
        )
        return result
    else:
        signal = strategy.generate_signal(current_time)
        return {"ok": True, "strategy": settings.STRATEGY_NAME, "signal": signal}
