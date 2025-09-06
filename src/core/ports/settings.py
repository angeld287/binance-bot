from __future__ import annotations

from typing import Any, Protocol


class SettingsProvider(Protocol):
    """Generic provider for configuration values."""

    def get(self, key: str, default: Any | None = None) -> Any:
        ...


# ---------------------------------------------------------------------------
# Helper accessors with defaults


def get_symbol(settings: SettingsProvider) -> str:
    return settings.get("SYMBOL", "BTCUSDT")


def get_risk_pct(settings: SettingsProvider) -> float:
    return float(settings.get("RISK_PCT", 0.003))


def get_timeout_no_fill_min(settings: SettingsProvider) -> int:
    return int(settings.get("TIMEOUT_NO_FILL_MIN", 20))


def get_microbuffer_pct_min(settings: SettingsProvider) -> float:
    return float(settings.get("MICROBUFFER_PCT_MIN", 0.0002))


def get_microbuffer_atr1m_mult(settings: SettingsProvider) -> float:
    return float(settings.get("MICROBUFFER_ATR1M_MULT", 0.25))


def get_buffer_sl_pct_min(settings: SettingsProvider) -> float:
    return float(settings.get("BUFFER_SL_PCT_MIN", 0.0005))


def get_buffer_sl_atr1m_mult(settings: SettingsProvider) -> float:
    return float(settings.get("BUFFER_SL_ATR1M_MULT", 0.5))


def get_tp_policy(settings: SettingsProvider) -> str:
    return settings.get("TP_POLICY", "STRUCTURAL_OR_1_8R")


def get_max_lookback_min(settings: SettingsProvider) -> int:
    return int(settings.get("MAX_LOOKBACK_MIN", 60))


def get_log_level(settings: SettingsProvider) -> str:
    return settings.get("LOG_LEVEL", "INFO")
