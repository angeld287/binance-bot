"""Utility helpers to compute price distance to EMAs.

This module keeps the logic self-contained so it can be plugged into the
breakout dual timeframe strategy without touching the main flow yet.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:  # pragma: no cover - thin wrapper, behaviour tested via caller
    from ..core import _ema as _core_ema
except Exception:  # pragma: no cover - fallback for safety in isolation
    _core_ema = None


logger = logging.getLogger("bot.strategy.breakout_dual_tf")


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


def _ema_series(values: Sequence[float], period: int) -> list[float]:
    if not values:
        return []
    multiplier = 2 / (period + 1)
    ema_values: list[float] = []
    ema_current = float(values[0])
    ema_values.append(ema_current)
    for value in values[1:]:
        ema_current = (float(value) - ema_current) * multiplier + ema_current
        ema_values.append(ema_current)
    return ema_values


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


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        logger.warning("[EmaDistance] invalid_float_env %s=%s", name, raw)
        return default


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        logger.warning("[EmaDistance] invalid_int_env %s=%s", name, raw)
        return default


def _normalise_tf_label(default_tf: str) -> str:
    tf_env = os.getenv("INTERVAL", default_tf) or default_tf
    tf = str(tf_env).lower()
    if tf in {"15m", "0:15", "quarter_hour", "15"}:
        return "15m"
    if tf in {"1h", "60m", "hour", "1"}:
        return "1h"
    if tf.endswith("m") and tf[:-1].isdigit():
        return f"{int(tf[:-1])}m"
    if tf.endswith("h") and tf[:-1].isdigit():
        return f"{int(tf[:-1])}h"
    return default_tf


def _max_dist_threshold(tf_label: str, policy: str, *, atr_norm: float | None) -> tuple[float, str]:
    tf_key = "1h" if tf_label == "1h" else "15m"
    if policy == "adaptive":
        atr_value = atr_norm or 0.0
        if atr_value <= 0:
            logger.warning("[EmaDistance] fallback_to_fixed: reason=atr_unavailable")
            policy = "fixed"
        else:
            k_env = (
                _parse_float_env("EMA_ADAPTIVE_K_1H", 1.0)
                if tf_key == "1h"
                else _parse_float_env("EMA_ADAPTIVE_K_15M", 1.0)
            )
            threshold = atr_value * k_env
            if threshold > 0:
                return threshold, "adaptive"
            logger.warning("[EmaDistance] fallback_to_fixed: reason=adaptive_threshold_invalid")
            policy = "fixed"

    max_pct = (
        _parse_float_env("EMA_MAX_DIST_PCT_1H", 0.018)
        if tf_key == "1h"
        else _parse_float_env("EMA_MAX_DIST_PCT_15M", 0.012)
    )
    return max_pct, "fixed"


def _format_candle(candle: Sequence[float]) -> Mapping[str, float]:
    return {
        "open": float(candle[1]),
        "high": float(candle[2]),
        "low": float(candle[3]),
        "close": float(candle[4]),
    }


def ema_distance_filter(ctx: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Evaluate whether the EMA distance filter allows proceeding.

    Parameters
    ----------
    ctx:
        Context dictionary containing the latest candle (``ohlc`` or ``candle``),
        EMA values, recent history and auxiliary metadata like ``direction`` or
        ``atr``.

    Returns
    -------
    tuple[bool, dict[str, Any]]
        Tuple with the boolean outcome and diagnostic metadata.
    """

    meta: dict[str, Any] = {
        "policy": os.getenv("EMA_DISTANCE_POLICY", "adaptive").strip().lower(),
        "tf_label": _normalise_tf_label(str(ctx.get("exec_tf", "unknown"))),
        "use_wick": bool(ctx.get("use_wick", True)),
    }

    ema7 = ctx.get("ema7")
    ema25 = ctx.get("ema25")
    ohlc = ctx.get("ohlc") or ctx.get("candle")
    if ohlc is None and ctx.get("candle_exec") is not None:
        ohlc = _format_candle(ctx["candle_exec"])
    if ohlc is None and isinstance(ctx.get("candles"), Sequence):
        candles_seq = ctx["candles"]
        if candles_seq:
            ohlc = _format_candle(candles_seq[-1])

    if ohlc is None or ema7 is None or ema25 is None:
        logger.warning("[EmaDistance] data_unavailable: reason=ema_or_ohlc_missing")
        meta["reason"] = "ema_or_ohlc_missing"
        return True, meta

    result = compute_ema_distance(
        ohlc,
        float(ema7),
        float(ema25),
        side=str(ctx.get("direction", "")),
        use_wick=meta["use_wick"],
        tf=meta["tf_label"],
    )
    meta["result"] = result
    meta["nearest"] = result.nearest_ema_label
    meta["dist"] = result.dist_to_nearest_pct

    if result.reason != "computed":
        logger.warning("[EmaDistance] data_unavailable: reason=%s", result.reason)
        meta["reason"] = result.reason
        return True, meta

    ema_ref = result.ema7 if result.nearest_ema_label == "ema7" else result.ema25
    if ema_ref <= 0:
        logger.warning("[EmaDistance] data_unavailable: reason=ema_ref_invalid")
        meta["reason"] = "ema_ref_invalid"
        return True, meta

    atr_value = float(ctx.get("atr") or ctx.get("atr_exec") or 0.0)
    atr_norm = atr_value / ema_ref if atr_value > 0 else 0.0
    meta["atr_norm"] = atr_norm

    max_allowed, effective_policy = _max_dist_threshold(meta["tf_label"], meta["policy"], atr_norm=atr_norm)
    meta["policy"] = effective_policy
    meta["max_allowed"] = max_allowed

    if max_allowed <= 0:
        logger.warning("[EmaDistance] data_unavailable: reason=max_allowed_invalid")
        meta["reason"] = "max_allowed_invalid"
        return True, meta

    is_far, _ = is_far_from_ema(result, max_dist_pct=max_allowed)
    meta["is_far"] = is_far

    candles_seq = ctx.get("candles") or ctx.get("exec_candles") or []
    lookback = max(1, _parse_int_env("EMA_REATTACH_LOOKBACK", 5))
    min_touches = max(1, _parse_int_env("EMA_REATTACH_MIN_TOUCHES", 2))
    tol_pct = max(0.0, _parse_float_env("EMA_REATTACH_TOL_PCT", 0.0015))
    touch_k = max(0.0, _parse_float_env("EMA_REATTACH_TOUCH_K", 0.25))

    meta["lookback"] = lookback
    meta["min_touches"] = min_touches

    reattach_touch_tol = tol_pct
    if atr_norm > 0:
        reattach_touch_tol = max(tol_pct, touch_k * atr_norm)
    meta["reattach_touch_tol"] = reattach_touch_tol

    recent_far = is_far
    touches = 0
    allow_by_touches = False
    allow_by_hysteresis = False

    if candles_seq and len(candles_seq) >= 2:
        closes = [float(c[4]) for c in candles_seq]
        ema7_series = _ema_series(closes, 7)
        ema25_series = _ema_series(closes, 25)
        start_idx = max(0, len(candles_seq) - lookback)
        for idx in range(start_idx, len(candles_seq)):
            ema7_val = ema7_series[idx] if idx < len(ema7_series) else float(ema7)
            ema25_val = ema25_series[idx] if idx < len(ema25_series) else float(ema25)
            candle = candles_seq[idx]
            candle_map = _format_candle(candle)
            hist_result = compute_ema_distance(
                candle_map,
                ema7_val,
                ema25_val,
                side=str(ctx.get("direction", "")),
                use_wick=False,
                tf=meta["tf_label"],
            )
            if hist_result.dist_to_nearest_pct > max_allowed:
                recent_far = True
            if hist_result.dist_to_nearest_pct <= reattach_touch_tol:
                touches += 1
        meta["touches"] = touches
        if touches >= min_touches:
            allow_by_touches = True
        allow_by_hysteresis = result.dist_to_nearest_pct <= max_allowed * 0.6
    else:
        meta["touches"] = None
        allow_by_hysteresis = result.dist_to_nearest_pct <= max_allowed * 0.6
        if candles_seq:
            logger.warning("[EmaDistance] insufficient_history_for_reattach len=%s", len(candles_seq))

    meta["allow_by_touches"] = allow_by_touches
    meta["allow_by_hysteresis"] = allow_by_hysteresis
    meta["recent_far"] = recent_far
    meta["hysteresis_limit"] = max_allowed * 0.6

    if not recent_far:
        return not is_far, meta

    if allow_by_touches:
        meta["reattach_pass"] = True
        return True, meta
    if allow_by_hysteresis:
        meta["reattach_pass"] = True
        return True, meta

    meta["reattach_blocked"] = True
    return False, meta


if __name__ == "__main__":  # pragma: no cover - manual smoke test only
    candle = {"open": 100, "high": 105, "low": 98, "close": 102}
    demo = compute_ema_distance(candle, 100.0, 101.0, side="LONG", use_wick=True)
    print(demo)
    print(is_far_from_ema(demo, max_dist_pct=0.05))
