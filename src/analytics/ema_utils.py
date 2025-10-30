"""Exponential moving average helpers."""

from __future__ import annotations

from typing import List, Sequence, Tuple


def parse_interval_to_ms(interval: str) -> int:
    """Convert Binance interval strings (``1m``, ``15m``, ``1h``) to milliseconds."""

    interval = (interval or "").strip().lower()
    if not interval:
        raise ValueError("Interval must be a non-empty string")

    unit = interval[-1]
    try:
        value = int(interval[:-1])
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid interval: {interval}") from exc

    factors = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    if unit not in factors:
        raise ValueError(f"Unsupported interval unit: {unit}")
    return value * factors[unit]


def compute_ema_series(values: Sequence[float], period: int) -> List[float | None]:
    """Return EMA values for ``values`` using ``period``.

    The returned list matches the input length. Elements preceding the first
    complete EMA window are set to ``None``.
    """

    if period <= 0:
        raise ValueError("EMA period must be positive")
    if not values:
        return []

    alpha = 2 / (period + 1)
    ema_values: List[float | None] = [None] * len(values)

    ema_prev = values[0]
    ema_values[0] = ema_prev
    for idx in range(1, len(values)):
        price = values[idx]
        ema_prev = (price - ema_prev) * alpha + ema_prev
        ema_values[idx] = ema_prev
    return ema_values


def compute_ema_map(klines: Sequence[Sequence], period: int) -> List[Tuple[int, float | None]]:
    """Return ``[(open_time_ms, ema_value), ...]`` for ``klines``.

    ``klines`` is expected to follow Binance's format with the open time at
    index 0 and the close price at index 4.
    """

    closes: List[float] = [float(candle[4]) for candle in klines]
    ema_values = compute_ema_series(closes, period)
    result: List[Tuple[int, float | None]] = []
    for candle, ema in zip(klines, ema_values):
        result.append((int(candle[0]), ema))
    return result


def find_candle_index_for_timestamp(
    klines: Sequence[Sequence], target_ms: int, interval_ms: int
) -> int | None:
    """Return the index of the candle whose window contains ``target_ms``."""

    if not klines:
        return None

    last_idx = len(klines) - 1
    for idx, candle in enumerate(klines):
        open_time = int(candle[0])
        close_time = open_time + interval_ms
        if open_time <= target_ms < close_time:
            return idx
    # If target is beyond the last candle range, use the last candle
    if target_ms >= int(klines[last_idx][0]):
        return last_idx
    return None
