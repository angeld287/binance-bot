from __future__ import annotations

from functools import lru_cache
from typing import Any
import logging
import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    STRATEGY_NAME: str = Field(
        default="breakout",
        validation_alias=AliasChoices("STRATEGY", "STRATEGY_NAME"),
    )
    FEATURE_BROKER: str = "binance"
    FEATURE_DATASOURCE: str = "binance"

    SYMBOL: str = "BTCUSDT"
    RISK_PCT: float = 0.003
    TIMEOUT_NO_FILL_MIN: int = 20
    MICROBUFFER_PCT_MIN: float = 0.0002
    MICROBUFFER_ATR1M_MULT: float = 0.25
    BUFFER_SL_PCT_MIN: float = 0.0005
    BUFFER_SL_ATR1M_MULT: float = 0.5
    STOP_LOSS_PCT: float | None = None
    TAKE_PROFIT_PCT: float | None = None
    TP_POLICY: str = "STRUCTURAL_OR_1_8R"
    MAX_LOOKBACK_MIN: int = 60
    INTERVAL: str = "1h"
    RR_FILTER_ENABLED: bool | None = None
    RR_MIN: float | None = None
    STRICT_ROUNDING: bool = True
    MIN_NOTIONAL_BUFFER_PCT: float = 0.03
    MIN_NOTIONAL_BUFFER_USD: float = 0.10

    BINANCE_API_KEY: str | None = None
    BINANCE_API_SECRET: str | None = None

    LOG_LEVEL: str = "INFO"
    BINANCE_TESTNET: bool = False
    PAPER_TRADING: bool | None = None  # Deprecated; fallback to BINANCE_TESTNET

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def get(self, key: str, default: Any | None = None) -> Any:
        """Return configuration value for ``key`` with ``default`` fallback."""
        return getattr(self, key, default)


@lru_cache
def load_settings() -> Settings:
    """Factory function to load settings from environment or .env file.

    Note: This function is cached. In AWS Lambda, environment variable
    changes are picked up only on a cold start (new deployment or
    container restart).
    """
    settings = Settings()

    # Fallback: map legacy PAPER_TRADING to BINANCE_TESTNET if the new variable
    # is not explicitly provided.
    if os.getenv("BINANCE_TESTNET") is None and os.getenv("PAPER_TRADING") is not None:
        settings.BINANCE_TESTNET = bool(settings.PAPER_TRADING)

    testnet = settings.BINANCE_TESTNET
    domain = (
        "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
    )
    logging.getLogger(__name__).info(
        "BINANCE_TESTNET=%s domain=%s", testnet, domain
    )

    return settings


def get_stop_loss_pct(settings: Settings) -> float | None:
    """Return STOP_LOSS_PCT if set and positive."""
    val = getattr(settings, "STOP_LOSS_PCT", None)
    if val is None and hasattr(settings, "get"):
        val = settings.get("STOP_LOSS_PCT")
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def get_take_profit_pct(settings: Settings) -> float | None:
    """Return TAKE_PROFIT_PCT if set and positive."""
    val = getattr(settings, "TAKE_PROFIT_PCT", None)
    if val is None and hasattr(settings, "get"):
        val = settings.get("TAKE_PROFIT_PCT")
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None
