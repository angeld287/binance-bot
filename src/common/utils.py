"""Helper functions shared across modules."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo
import math
import os
import re
from typing import Any, Dict, List, Optional

__all__ = [
    "sanitize_client_order_id",
    "is_in_blackout",
    "inspect_risk_notional_env",
]


_CID_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]")


def sanitize_client_order_id(raw: str) -> str:
    """Sanitize ``raw`` to meet Binance client order id requirements.

    Any character outside ``[A-Za-z0-9_-]`` is replaced with ``-`` and the
    resulting string is truncated to 36 characters. The transformation is
    deterministic and does not generate new identifiers.
    """

    return _CID_SANITIZE_RE.sub("-", str(raw))[:36]


def is_in_blackout(now_utc: datetime, tz: str, windows_str: str) -> bool:
    """Return ``True`` if ``now_utc`` falls within any blackout window.

    Parameters
    ----------
    now_utc:
        Current time in UTC (naive or timezone-aware).
    tz:
        IANA timezone string to convert ``now_utc`` into.
    windows_str:
        Comma-separated list of ranges in ``HH:MM-HH:MM`` format. If empty or
        ``None``, no blackout is applied.
    """

    if not windows_str:
        return False

    utc = now_utc if now_utc.tzinfo else now_utc.replace(tzinfo=ZoneInfo("UTC"))
    local_time = utc.astimezone(ZoneInfo(tz)).time()

    for window in windows_str.split(","):
        window = window.strip()
        if not window:
            continue
        try:
            start_str, end_str = window.split("-")
            sh, sm = map(int, start_str.split(":"))
            eh, em = map(int, end_str.split(":"))
            start = time(sh, sm)
            end = time(eh, em)
        except ValueError:
            continue

        if start <= local_time <= end:
            return True

    return False


_RISK_ENV_KEYS: List[str] = [
    "RISK_NOTIONAL_USDT",
    "RISK_NOTIONAL_USD",
    "RISK_NOTIONAL",
    "RISKNATIONALUSDT",
    "RISK_NATIONAL_USDT",
    "RISK_NOTIONAL_USDT_PCT",
]


def inspect_risk_notional_env(current_value: Any) -> Dict[str, Any]:
    """Inspect environment overrides for ``RISK_NOTIONAL_USDT`` diagnostics."""

    checked: List[str] = list(_RISK_ENV_KEYS)
    env_key_used: Optional[str] = None
    env_value_raw: Optional[str] = None
    env_pct_conflict: Optional[str] = None
    env_numeric = math.nan

    for key in _RISK_ENV_KEYS:
        raw_val = os.getenv(key)
        if raw_val is None:
            continue
        if key == "RISK_NOTIONAL_USDT_PCT":
            env_pct_conflict = raw_val
            continue
        env_key_used = key
        env_value_raw = raw_val
        try:
            env_numeric = float(raw_val)
        except (TypeError, ValueError):
            env_numeric = math.nan
        break

    try:
        current_numeric = float(current_value)
    except (TypeError, ValueError):
        current_numeric = math.nan

    source_of_defaults = env_key_used is None or math.isnan(env_numeric)

    return {
        "env_raw_keys_checked": checked,
        "env_key_used": env_key_used,
        "env_value_raw": env_value_raw,
        "env_value_numeric": env_numeric,
        "env_pct_conflict": env_pct_conflict,
        "risk_notional_usdt": current_numeric,
        "source_of_defaults": source_of_defaults,
    }
