"""Typed settings using pydantic-settings."""

from __future__ import annotations

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application configuration."""

    strategy_name: str = Field(default="breakout", alias="STRATEGY_NAME")
    feature_broker: str = Field(default="binance", alias="FEATURE_BROKER")
    feature_datasource: str = Field(default="binance", alias="FEATURE_DATASOURCE")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
