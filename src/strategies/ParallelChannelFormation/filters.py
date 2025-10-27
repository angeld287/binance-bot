"""Filter wrappers leveraging core helpers for the channel strategy."""

from __future__ import annotations

import os

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
) -> tuple[bool, str | None, Mapping[str, Any] | None]:
    """Apply RR/EMA/volatility filters returning decision and reason."""

    if rr is not None and rr < confidence_threshold:
        return False, "rr_filter", None

    if os.getenv("PCF_EMA_FILTER_ENABLED", "1") == "0":
        return True, "ema_filter_skipped", None

    side_norm = (side or "").upper()
    ohlc = meta.get("ohlc") if isinstance(meta, Mapping) else None
    if ohlc and isinstance(ohlc, Mapping):
        ema_result = compute_ema_distance(
            ohlc,
            ema_fast,
            ema_slow,
            side=side_norm or "LONG",
        )

        def _build_ema_meta() -> Mapping[str, Any]:
            return {
                "side_norm": side_norm,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "ema7": ema_result.ema7,
                "ema25": ema_result.ema25,
                "price_ref": ema_result.price_ref,
                "dist_to_ema7_pct": ema_result.dist_to_ema7_pct,
                "dist_to_ema25_pct": ema_result.dist_to_ema25_pct,
                "dist_to_avg_pct": ema_result.dist_to_avg_pct,
                "ema_result_reason": ema_result.reason,
            }
        if not ema_result.ok and ema_result.reason:
            return False, "ema_filter", _build_ema_meta()
        if side_norm == "LONG" and ema_result.ema7 > ema_result.ema25:
            pass
        elif side_norm == "SHORT" and ema_result.ema7 < ema_result.ema25:
            pass
        elif ema_fast is not None and ema_slow is not None:
            if side_norm == "LONG" and ema_fast < ema_slow:
                return False, "ema_filter", _build_ema_meta()
            if side_norm == "SHORT" and ema_fast > ema_slow:
                return False, "ema_filter", _build_ema_meta()

    if volume_avg is not None and volume_avg <= 0:
        return False, "volume_filter", None

    if atr is not None and atr <= 0:
        return False, "atr_filter", None

    return True, None, None


__all__ = ["apply_filters"]
