"""Helpers to compute MFE/MAE metrics for a roundtrip."""

from __future__ import annotations

from typing import Sequence, Tuple


def _percent(value: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (value / base) * 100


def _extract_high_low(candle: Sequence) -> Tuple[float, float]:
    return float(candle[2]), float(candle[3])


def compute_mfe_mae(
    candles: Sequence[Sequence], direction: str, entry_price: float
) -> tuple[float, float, int | None]:
    """Compute MFE/MAE percentages and timestamp for the favourable move."""

    if not candles or entry_price <= 0:
        return 0.0, 0.0, None

    direction_norm = (direction or "").upper()
    if direction_norm in ("SELL", "SHORT"):
        direction_norm = "SHORT"
    else:
        direction_norm = "LONG"

    mfe_pct = 0.0
    mae_pct = 0.0
    mfe_ts: int | None = None

    for candle in candles:
        open_time = int(candle[0])
        high, low = _extract_high_low(candle)

        if direction_norm == "LONG":
            favorable = max(0.0, _percent(high - entry_price, entry_price))
            adverse = max(0.0, _percent(entry_price - low, entry_price))
        else:
            favorable = max(0.0, _percent(entry_price - low, entry_price))
            adverse = max(0.0, _percent(high - entry_price, entry_price))

        if favorable >= mfe_pct:
            mfe_pct = favorable
            mfe_ts = open_time
        if adverse >= mae_pct:
            mae_pct = adverse

    return mfe_pct, mae_pct, mfe_ts
