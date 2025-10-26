"""EMA transition filter helper for the Parallel Channel strategy."""

from __future__ import annotations

import logging
import os
from typing import Iterable, Sequence

from config.utils import parse_bool

logger = logging.getLogger("bot.strategy.parallel_channel")

_FILTER_ENABLED_ENV = "EMA_TransitionFilter_Enabled"
_SLOPE_THRESHOLD_ENV = "EMA_Fast_Slope_Threshold"

_DEFAULT_FILTER_ENABLED = True
_DEFAULT_SLOPE_THRESHOLD = 0.0005


def _get_bool_env(key: str, default: bool) -> bool:
    return parse_bool(os.getenv(key), default=default)


def _get_float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value != value:  # NaN check
        return default
    return value


def calculate_ema_fast_slope(ema_fast_history: Sequence[float] | Iterable[float] | None) -> float | None:
    """Return the approximate slope for the fast EMA history."""

    if ema_fast_history is None:
        return None
    history = list(ema_fast_history)
    if len(history) < 3:
        return None
    latest = float(history[-1])
    older = float(history[-3])
    return (latest - older) / 2.0


def ema_transition_filter_allows_entry(
    price_current: float | None,
    ema_fast_value: float | None,
    ema_slow_value: float | None,
    ema_fast_slope: float | None,
) -> bool:
    """Return ``True`` when the EMA transition filter allows a new entry."""

    enabled = _get_bool_env(_FILTER_ENABLED_ENV, _DEFAULT_FILTER_ENABLED)
    if not enabled:
        return True

    slope_threshold = _get_float_env(_SLOPE_THRESHOLD_ENV, _DEFAULT_SLOPE_THRESHOLD)

    if (
        price_current is not None
        and ema_fast_value is not None
        and ema_slow_value is not None
        and (
            (price_current <= ema_fast_value and price_current >= ema_slow_value)
            or (price_current >= ema_fast_value and price_current <= ema_slow_value)
        )
    ):
        logger.info(
            (
                "ENTRY BLOCKED by EMA Transition Filter: price between fast/slow EMA | "
                "price_current=%.8f ema_fast=%.8f ema_slow=%.8f ema_fast_slope=%s threshold=%.8f"
            ),
            price_current,
            ema_fast_value,
            ema_slow_value,
            ema_fast_slope,
            slope_threshold,
        )
        return False

    if ema_fast_slope is not None and abs(ema_fast_slope) < slope_threshold:
        logger.info(
            (
                "ENTRY BLOCKED by EMA Transition Filter: low fast EMA slope (flat momentum) | "
                "ema_fast_slope=%.8f threshold=%.8f price_current=%s ema_fast=%s ema_slow=%s"
            ),
            ema_fast_slope,
            slope_threshold,
            price_current,
            ema_fast_value,
            ema_slow_value,
        )
        return False

    return True


__all__ = [
    "calculate_ema_fast_slope",
    "ema_transition_filter_allows_entry",
]

