from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    STRATEGY_NAME: str = "breakout"
    FEATURE_BROKER: str = "binance"
    FEATURE_DATASOURCE: str = "binance"

    SYMBOL: str = "BTCUSDT"
    RISK_PCT: float = 0.003
    TIMEOUT_NO_FILL_MIN: int = 20
    MICROBUFFER_PCT_MIN: float = 0.0002
    MICROBUFFER_ATR1M_MULT: float = 0.25
    BUFFER_SL_PCT_MIN: float = 0.0005
    BUFFER_SL_ATR1M_MULT: float = 0.5
    TP_POLICY: str = "STRUCTURAL_OR_1_8R"
    MAX_LOOKBACK_MIN: int = 60
    INTERVAL: str = "1h"

    BINANCE_API_KEY: str | None = None
    BINANCE_API_SECRET: str | None = None

    LOG_LEVEL: str = "INFO"
    PAPER_TRADING: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def get(self, key: str, default: Any | None = None) -> Any:
        """Return configuration value for ``key`` with ``default`` fallback."""
        return getattr(self, key, default)


@lru_cache
def load_settings() -> Settings:
    """Factory function to load settings from environment or .env file."""
    return Settings()
