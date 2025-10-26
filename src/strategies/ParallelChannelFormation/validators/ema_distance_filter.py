"""EMA distance validation helper for the Parallel Channel strategy."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from ..config.env_loader import ChannelEnv


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_config(
    env_config: ChannelEnv | Mapping[str, Any] | None,
) -> tuple[bool, float]:
    """Extract EMA distance configuration from the provided environment object."""

    if isinstance(env_config, ChannelEnv):
        return (
            bool(getattr(env_config, "ema_distance_filter_enabled", True)),
            float(getattr(env_config, "ema_distance_threshold_pct", 0.8)),
        )

    if isinstance(env_config, Mapping):
        enabled_raw = env_config.get("ema_distance_filter_enabled", True)
        threshold_raw = env_config.get("ema_distance_threshold_pct", 0.8)
        try:
            enabled = bool(enabled_raw)
        except Exception:  # pragma: no cover - defensive
            enabled = True
        try:
            threshold = float(threshold_raw)
        except (TypeError, ValueError):
            threshold = 0.8
        return enabled, threshold

    return True, 0.8


def check_ema_distance(
    *,
    symbol: str,
    side: str,
    mark_price: float | None,
    ema_fast: float | None,
    env_config: ChannelEnv | Mapping[str, Any] | None,
    logger: logging.Logger | None = None,
) -> tuple[str, Mapping[str, Any] | None]:
    """Validate the percentage distance between price and the fast EMA."""

    enabled, threshold_pct = _resolve_config(env_config)

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
                "Rejected by EMA Distance Filter: distance_pct=%s > threshold=%s | symbol=%s | side=%s",
                f"{distance_pct:.2f}",
                f"{threshold_pct:.2f}",
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
