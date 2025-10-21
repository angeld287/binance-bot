"""Helpers to emit optional rounding diagnostics for Binance Futures orders."""

from __future__ import annotations

import json
import logging
import os
from decimal import InvalidOperation
from typing import Any, Mapping

from common.precision import format_decimal, to_decimal

__all__ = [
    "is_rounding_diag_enabled",
    "format_rounding_diag_number",
    "emit_rounding_diag",
]


_ENV_FLAG = "ROUNDING_DIAG"
_DEFAULT_LOGGER = logging.getLogger("bot.rounding_diag")


def is_rounding_diag_enabled() -> bool:
    """Return ``True`` when diagnostics are enabled via ``ROUNDING_DIAG``."""

    value = os.getenv(_ENV_FLAG)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def format_rounding_diag_number(value: Any) -> str | None:
    """Format ``value`` as a human readable string for diagnostic payloads."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            dec = to_decimal(text)
        except (InvalidOperation, ValueError):
            return text
        return format_decimal(dec)
    try:
        dec = to_decimal(value)
    except (InvalidOperation, ValueError, TypeError):
        return str(value)
    return format_decimal(dec)


def emit_rounding_diag(
    payload: Mapping[str, Any], *, logger: logging.Logger | None = None
) -> None:
    """Serialize and emit ``payload`` if diagnostics are enabled."""

    if not is_rounding_diag_enabled():
        return
    target = logger or _DEFAULT_LOGGER
    target.info(json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")))

