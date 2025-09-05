from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    STRATEGY_NAME: str = "breakout"
    FEATURE_BROKER: str = "binance"
    FEATURE_DATASOURCE: str = "binance"

    SYMBOL: str = "BTCUSDT"
    INTERVAL: str = "1h"

    BINANCE_API_KEY: str | None = None
    BINANCE_API_SECRET: str | None = None

    LOG_LEVEL: str = "INFO"
    PAPER_TRADING: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def load_settings() -> Settings:
    """Factory function to load settings from environment or .env file."""
    return Settings()
