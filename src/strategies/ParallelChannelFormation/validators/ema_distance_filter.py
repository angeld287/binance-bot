"""EMA distance validation helper for the Parallel Channel strategy."""

from __future__ import annotations

import logging
from typing import Any, Mapping


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_ema_distance(
    *,
    symbol: str,
    side: str,
    mark_price: float | None,
    ema_fast: float | None,
    enabled: bool,
    threshold_pct: float,
    logger: logging.Logger | None = None,
) -> tuple[str, Mapping[str, Any] | None]:
    """Validate the percentage distance between price and the fast EMA."""

    if not enabled:
        return "ok", None

    ema_value = _safe_float(ema_fast)
    price_value = _safe_float(mark_price)
    if ema_value is None or ema_value <= 0:
        return "ok", None
    if price_value is None:
        return "ok", None

    distance_pct = abs(price_value - ema_value) / ema_value * 100.0
    side_norm = side.upper() if side else ""

    if distance_pct > threshold_pct:
        if logger is not None:
            logger.info(
                "Rejected by EMA Distance Filter: distance_pct=%.2f > threshold=%.2f | symbol=%s | side=%s",
                distance_pct,
                threshold_pct,
                symbol,
                side_norm or "",
            )
        return (
            "reject",
            {
                "reason": "ema_distance_filter",
                "distance_pct": distance_pct,
                "threshold_pct": threshold_pct,
                "ema_fast": ema_value,
                "mark_price": price_value,
                "side": side_norm,
            },
        )

    return "ok", None


__all__ = ["check_ema_distance"]
