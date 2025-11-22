"""Compression filter helper for the Parallel Channel strategy.

The filter blocks entries when price and EMAs show short-term compression
conditions to avoid false breakouts after EMA crosses.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Mapping, Sequence

from config.utils import parse_bool

logger = logging.getLogger("bot.strategy.parallel_channel")

_ENABLE_ENV = "ENABLE_COMPRESSION_FILTER"
_LOOKBACK_ENV = "COMPRESSION_LOOKBACK"
_MAX_SLOPE_ENV = "COMPRESSION_MAX_EMA_SLOPE_PCT"
_MAX_DISTANCE_ENV = "COMPRESSION_MAX_EMA_DISTANCE_PCT"
_MIN_FLIPS_ENV = "COMPRESSION_MIN_SIDE_FLIPS"

_DEFAULT_ENABLED = False
_DEFAULT_LOOKBACK = 8
_DEFAULT_MAX_SLOPE = 0.002
_DEFAULT_MAX_DISTANCE = 0.0015
_DEFAULT_MIN_FLIPS = 3


def _get_int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


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


def _ema_series(values: Sequence[float], period: int) -> list[float]:
    if not values or period <= 0:
        return []
    multiplier = 2 / (period + 1)
    ema_values: list[float] = []
    ema_current = float(values[0])
    ema_values.append(ema_current)
    for value in values[1:]:
        ema_current = (float(value) - ema_current) * multiplier + ema_current
        ema_values.append(ema_current)
    return ema_values


def _count_side_flips(closes: Sequence[float], ema_slow_series: Sequence[float]) -> int:
    side_flips = 0
    previous_sign = None
    for close, ema_value in zip(closes, ema_slow_series):
        diff = close - ema_value
        if diff == 0:
            current_sign = previous_sign
        else:
            current_sign = 1 if diff > 0 else -1
        if previous_sign is not None and current_sign is not None and current_sign != previous_sign:
            side_flips += 1
        previous_sign = current_sign
    return side_flips


def compression_filter_allows_entry(
    *,
    enabled: bool,
    closes: Sequence[float] | Iterable[float] | None,
    ema_fast_length: int,
    ema_slow_length: int,
    ema_fast_value: float | None,
    ema_slow_value: float | None,
    symbol: str | None = None,
    logger_override: logging.Logger | None = None,
) -> tuple[bool, Mapping[str, float | int | str] | None]:
    """Return filter decision and meta information for compression detection."""

    active_logger = logger_override or logger
    enabled = parse_bool(os.getenv(_ENABLE_ENV), default=enabled)
    if not enabled:
        return True, None

    lookback = _get_int_env(_LOOKBACK_ENV, _DEFAULT_LOOKBACK)
    max_slope_pct = _get_float_env(_MAX_SLOPE_ENV, _DEFAULT_MAX_SLOPE)
    max_distance_pct = _get_float_env(_MAX_DISTANCE_ENV, _DEFAULT_MAX_DISTANCE)
    min_side_flips = _get_int_env(_MIN_FLIPS_ENV, _DEFAULT_MIN_FLIPS)

    closes_seq = list(closes or [])
    if len(closes_seq) <= lookback:
        active_logger.debug(
            {
                "strategy": "ParallelChannelFormation",
                "symbol": symbol or "",
                "state": "compression_filter",
                "reason": "insufficient_data",
            }
        )
        return True, None

    fast_series = _ema_series(closes_seq, ema_fast_length)
    slow_series = _ema_series(closes_seq, ema_slow_length)

    if len(slow_series) <= lookback or len(fast_series) <= lookback:
        active_logger.debug(
            {
                "strategy": "ParallelChannelFormation",
                "symbol": symbol or "",
                "state": "compression_filter",
                "reason": "insufficient_data",
            }
        )
        return True, None

    slow_now = slow_series[-1]
    slow_past = slow_series[-(lookback + 1)]
    if slow_past == 0:
        active_logger.debug(
            {
                "strategy": "ParallelChannelFormation",
                "symbol": symbol or "",
                "state": "compression_filter",
                "reason": "insufficient_data",
            }
        )
        return True, None
    ema_slow_slope_pct = (slow_now - slow_past) / slow_past
    ema_slow_slope_abs = abs(ema_slow_slope_pct)

    distance_values: list[float] = []
    for fast, slow in zip(fast_series[-lookback:], slow_series[-lookback:]):
        if slow == 0:
            continue
        distance_values.append(abs(fast - slow) / slow)
    if not distance_values:
        active_logger.debug(
            {
                "strategy": "ParallelChannelFormation",
                "symbol": symbol or "",
                "state": "compression_filter",
                "reason": "insufficient_data",
            }
        )
        return True, None
    avg_ema_distance_pct = sum(distance_values) / len(distance_values)

    closes_tail = closes_seq[-lookback:]
    slow_tail = slow_series[-lookback:]
    side_flips = _count_side_flips(closes_tail, slow_tail)

    distance_ok = avg_ema_distance_pct <= max_distance_pct
    slope_ok = ema_slow_slope_abs <= max_slope_pct
    flips_ok = side_flips >= min_side_flips
    is_compression = slope_ok and distance_ok and flips_ok

    log_payload = {
        "strategy": "ParallelChannelFormation",
        "symbol": symbol or "",
        "state": "compression_filter",
        "passed": not is_compression,
        "metrics": {
            "avg_ema_distance_pct": avg_ema_distance_pct,
            "ema_slow_slope_pct": ema_slow_slope_pct,
            "ema_slow_slope_abs": ema_slow_slope_abs,
            "side_flips": side_flips,
        },
        "conditions": {
            "distance_ok": distance_ok,
            "slope_ok": slope_ok,
            "flips_ok": flips_ok,
        },
        "thresholds": {
            "COMPRESSION_MAX_EMA_DISTANCE_PCT": max_distance_pct,
            "COMPRESSION_MAX_EMA_SLOPE_PCT": max_slope_pct,
            "COMPRESSION_MIN_SIDE_FLIPS": min_side_flips,
        },
        "context": {
            "ema_fast_length": ema_fast_length,
            "ema_slow_length": ema_slow_length,
            "lookback": lookback,
        },
    }

    if is_compression:
        active_logger.info(log_payload)

    if is_compression:
        active_logger.info(
            (
                "entry blocked by CompressionFilter: symbol=%s, "
                "emaSlowSlopePct=%.6f, avgEmaDistancePct=%.6f, sideFlips=%d"
            ),
            symbol or "",
            ema_slow_slope_pct,
            avg_ema_distance_pct,
            side_flips,
        )
        return False, {
            "reason": "compression_filter",
            "ema_slow_slope_pct": ema_slow_slope_pct,
            "avg_ema_distance_pct": avg_ema_distance_pct,
            "side_flips": side_flips,
        }

    active_logger.debug(log_payload)
    return True, None


__all__ = ["compression_filter_allows_entry"]
