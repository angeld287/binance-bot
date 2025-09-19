"""Helper functions shared across modules."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo
import re

__all__ = ["sanitize_client_order_id", "is_in_blackout"]


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
