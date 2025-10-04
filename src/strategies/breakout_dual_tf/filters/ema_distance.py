"""Utility helpers to compute price distance to EMAs.

This module keeps the logic self-contained so it can be plugged into the
breakout dual timeframe strategy without touching the main flow yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Sequence

try:  # pragma: no cover - thin wrapper, behaviour tested via caller
    from ..core import _ema as _core_ema
except Exception:  # pragma: no cover - fallback for safety in isolation
    _core_ema = None


def ema(values: Sequence[float], period: int) -> float:
    """Re-export the core EMA helper so callers can reuse it here.

    Parameters
    ----------
    values:
        Sequence of prices used to derive the EMA.
    period:
        Number of periods for the exponential moving average.

    Returns
    -------
    float
        The computed EMA. Falls back to ``0.0`` if the core helper is not
        available.
    """

    if _core_ema is None:
        if not values:
            return 0.0
        multiplier = 2 / (period + 1)
        ema_val = float(values[0])
        for value in values[1:]:
            ema_val = (float(value) - ema_val) * multiplier + ema_val
        return ema_val
    return _core_ema(values, period)


@dataclass(slots=True)
class EmaDistanceResult:
    """Container with EMA distance information.

    Attributes
    ----------
    tf:
        Timeframe label (from the ``INTERVAL`` environment variable).
    side:
        Trade direction (``"LONG"`` or ``"SHORT"``). Empty when not defined.
    use_wick:
        Whether wicks were considered to choose the reference price.
    price_ref:
        Price used to compute distances against the EMAs.
    ema7:
        Value of the EMA with period 7.
    ema25:
        Value of the EMA with period 25.
    nearest_ema_label:
        Label of the EMA closer to ``price_ref`` (``"ema7"`` or ``"ema25"``).
    dist_to_ema7_pct:
        Absolute percentage distance between ``price_ref`` and ``ema7``.
    dist_to_ema25_pct:
        Absolute percentage distance between ``price_ref`` and ``ema25``.
    dist_to_nearest_pct:
        Absolute percentage distance to the nearest EMA.
    dist_to_avg_pct:
        Absolute percentage distance versus the simple average of both EMAs.
    ok:
        Placeholder flag for future threshold checks. Always ``True`` for now.
    reason:
        Textual reason/annotation about the calculation outcome.
    """

    tf: str
    side: str
    use_wick: bool
    price_ref: float
    ema7: float
    ema25: float
    nearest_ema_label: str
    dist_to_ema7_pct: float
    dist_to_ema25_pct: float
    dist_to_nearest_pct: float
    dist_to_avg_pct: float
    ok: bool
    reason: str


def compute_ema_distance(
    ohlc: Mapping[str, float],
    ema7: float | None,
    ema25: float | None,
    *,
    side: str = "",
    use_wick: bool = True,
    tf: str | None = None,
) -> EmaDistanceResult:
    """Compute price distance to EMA7 and EMA25 for the provided candle.

    Examples
    --------
    >>> ohlc = {"open": 100, "high": 105, "low": 98, "close": 102}
    >>> result = compute_ema_distance(ohlc, 100.0, 101.0, side="LONG")
    >>> result.nearest_ema_label
    'ema7'
    """

    tf_value = tf or os.getenv("INTERVAL", "unknown")
    side_norm = side.upper()

    close_price = float(ohlc.get("close", 0.0))
    low_price = float(ohlc.get("low", close_price))
    high_price = float(ohlc.get("high", close_price))

    if side_norm == "LONG":
        price_ref = low_price if use_wick else close_price
    elif side_norm == "SHORT":
        price_ref = high_price if use_wick else close_price
    else:
        price_ref = close_price

    if ema7 is None or ema25 is None:
        return EmaDistanceResult(
            tf=tf_value,
            side=side_norm,
            use_wick=use_wick,
            price_ref=price_ref,
            ema7=float(ema7) if ema7 is not None else 0.0,
            ema25=float(ema25) if ema25 is not None else 0.0,
            nearest_ema_label="",
            dist_to_ema7_pct=0.0,
            dist_to_ema25_pct=0.0,
            dist_to_nearest_pct=0.0,
            dist_to_avg_pct=0.0,
            ok=True,
            reason="ema_unavailable",
        )

    ema7_val = float(ema7)
    ema25_val = float(ema25)

    if ema7_val <= 0 or ema25_val <= 0:
        return EmaDistanceResult(
            tf=tf_value,
            side=side_norm,
            use_wick=use_wick,
            price_ref=price_ref,
            ema7=ema7_val,
            ema25=ema25_val,
            nearest_ema_label="",
            dist_to_ema7_pct=0.0,
            dist_to_ema25_pct=0.0,
            dist_to_nearest_pct=0.0,
            dist_to_avg_pct=0.0,
            ok=True,
            reason="ema_invalid",
        )

    dist_to_ema7 = abs(price_ref - ema7_val) / ema7_val
    dist_to_ema25 = abs(price_ref - ema25_val) / ema25_val

    if dist_to_ema7 <= dist_to_ema25:
        nearest_label = "ema7"
        dist_to_nearest = dist_to_ema7
    else:
        nearest_label = "ema25"
        dist_to_nearest = dist_to_ema25

    ema_avg = (ema7_val + ema25_val) / 2.0
    if ema_avg > 0:
        dist_to_avg = abs(price_ref - ema_avg) / ema_avg
        avg_reason = "computed"
    else:
        dist_to_avg = 0.0
        avg_reason = "ema_avg_invalid"

    reason = "computed" if avg_reason == "computed" else avg_reason

    return EmaDistanceResult(
        tf=tf_value,
        side=side_norm,
        use_wick=use_wick,
        price_ref=price_ref,
        ema7=ema7_val,
        ema25=ema25_val,
        nearest_ema_label=nearest_label,
        dist_to_ema7_pct=round(dist_to_ema7, 6),
        dist_to_ema25_pct=round(dist_to_ema25, 6),
        dist_to_nearest_pct=round(dist_to_nearest, 6),
        dist_to_avg_pct=round(dist_to_avg, 6),
        ok=True,
        reason=reason,
    )


def is_far_from_ema(result: EmaDistanceResult, *, max_dist_pct: float) -> tuple[bool, str]:
    """Check whether the nearest EMA is farther than a provided threshold.

    Parameters
    ----------
    result:
        The :class:`EmaDistanceResult` to be evaluated.
    max_dist_pct:
        Maximum allowed distance in percentage (expressed as a decimal value).

    Returns
    -------
    tuple[bool, str]
        Tuple with the boolean outcome and a textual reason.
    """

    if max_dist_pct <= 0:
        return False, "max_dist_pct_invalid"

    if result.dist_to_nearest_pct > max_dist_pct:
        return True, "dist_above_threshold"
    return False, "within_threshold"


if __name__ == "__main__":  # pragma: no cover - manual smoke test only
    candle = {"open": 100, "high": 105, "low": 98, "close": 102}
    demo = compute_ema_distance(candle, 100.0, 101.0, side="LONG", use_wick=True)
    print(demo)
    print(is_far_from_ema(demo, max_dist_pct=0.05))
