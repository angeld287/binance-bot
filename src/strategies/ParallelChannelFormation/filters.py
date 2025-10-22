"""Filter wrappers leveraging core helpers for the channel strategy."""

from __future__ import annotations

from typing import Any, Mapping

from strategies.breakout_dual_tf.filters.ema_distance import compute_ema_distance


def apply_filters(
    *,
    rr: float | None,
    confidence_threshold: float,
    ema_fast: float | None,
    ema_slow: float | None,
    volume_avg: float | None,
    atr: float | None,
    meta: Mapping[str, Any],
    side: str | None,
) -> tuple[bool, str | None]:
    """Apply RR/EMA/volatility filters returning decision and reason."""

    if rr is not None and rr < confidence_threshold:
        return False, "rr_filter"

    side_norm = (side or "").upper()
    ohlc = meta.get("ohlc") if isinstance(meta, Mapping) else None
    if ohlc and isinstance(ohlc, Mapping):
        ema_result = compute_ema_distance(
            ohlc,
            ema_fast,
            ema_slow,
            side=side_norm or "LONG",
        )
        if not ema_result.ok and ema_result.reason:
            return False, "ema_filter"
        if side_norm == "LONG" and ema_result.ema7 > ema_result.ema25:
            pass
        elif side_norm == "SHORT" and ema_result.ema7 < ema_result.ema25:
            pass
        elif ema_fast is not None and ema_slow is not None:
            if side_norm == "LONG" and ema_fast < ema_slow:
                return False, "ema_filter"
            if side_norm == "SHORT" and ema_fast > ema_slow:
                return False, "ema_filter"

    if volume_avg is not None and volume_avg <= 0:
        return False, "volume_filter"

    if atr is not None and atr <= 0:
        return False, "atr_filter"

    return True, None


__all__ = ["apply_filters"]
