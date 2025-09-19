"""False breakout validator for the breakout strategy."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Sequence
import logging
import math
import os

logger = logging.getLogger("bot.strategy.breakout")


# ---------------------------------------------------------------------------
# Environment helpers

def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Settings container


@dataclass(frozen=True)
class FalseBreakoutSettings:
    """Configuration for the false breakout validator."""

    level_lookback: int = 48
    min_touches: int = 2
    touch_tolerance_pct: float = 0.10
    close_buffer_pct: float = 0.10
    close_buffer_atr: float = 0.2
    vol_ma_mult: float = 1.3
    vol_ma_n: int = 20
    wick_max_ratio: float = 1.2
    retest_wait: int = 3
    time_window: int = 6
    use_wick_touch: bool = True
    atr_period: int = 14

    def to_log_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


BREAKOUT_FBV_ENABLED = _to_bool(os.getenv("BREAKOUT_FBV_ENABLED"), False)

_FBV_SETTINGS = FalseBreakoutSettings(
    level_lookback=_to_int(os.getenv("FBV_LEVEL_LOOKBACK"), 48),
    min_touches=_to_int(os.getenv("FBV_MIN_TOUCHES"), 2),
    touch_tolerance_pct=_to_float(os.getenv("FBV_TOUCH_TOLERANCE_PCT"), 0.10),
    close_buffer_pct=_to_float(os.getenv("FBV_CLOSE_BUFFER_PCT"), 0.10),
    close_buffer_atr=_to_float(os.getenv("FBV_CLOSE_BUFFER_ATR"), 0.2),
    vol_ma_mult=_to_float(os.getenv("FBV_VOL_MA_MULT"), 1.3),
    vol_ma_n=_to_int(os.getenv("FBV_VOL_MA_N"), 20),
    wick_max_ratio=_to_float(os.getenv("FBV_WICK_MAX_RATIO"), 1.2),
    retest_wait=_to_int(os.getenv("FBV_RETEST_WAIT"), 3),
    time_window=_to_int(os.getenv("FBV_TIME_WINDOW"), 6),
    use_wick_touch=_to_bool(os.getenv("FBV_USE_WICK_TOUCH"), True),
)

logger.info("fbv.settings %s", {"enabled": BREAKOUT_FBV_ENABLED, **_FBV_SETTINGS.to_log_dict()})


def get_false_breakout_settings() -> FalseBreakoutSettings:
    """Return immutable validator settings."""

    return _FBV_SETTINGS


# ---------------------------------------------------------------------------
# Candle helpers


def _get_value(source: Any, index: int, keys: Sequence[str]) -> float:
    if isinstance(source, dict):
        for key in keys:
            if key in source:
                return float(source[key])
        raise KeyError(f"Missing keys {keys!r} in candle: {source!r}")
    return float(source[index])


def _normalize_candle(raw: Any) -> dict[str, float]:
    open_time: float
    if isinstance(raw, dict):
        for key in ("open_time", "openTime", "time", "t"):
            if key in raw:
                open_time = float(raw[key])
                break
        else:
            open_time = float(raw.get(0, 0.0))
    else:
        open_time = float(raw[0])
    return {
        "open_time": open_time,
        "open": _get_value(raw, 1, ("open", "o")),
        "high": _get_value(raw, 2, ("high", "h")),
        "low": _get_value(raw, 3, ("low", "l")),
        "close": _get_value(raw, 4, ("close", "c")),
        "volume": _get_value(raw, 5, ("volume", "v")),
    }


def _true_range(curr: dict[str, float], prev_close: float) -> float:
    return max(
        curr["high"] - curr["low"],
        abs(curr["high"] - prev_close),
        abs(curr["low"] - prev_close),
    )


def _compute_atr(candles: Sequence[dict[str, float]], period: int) -> float | None:
    if len(candles) <= period:
        return None
    trs: list[float] = []
    for offset in range(1, period + 1):
        candle = candles[-offset]
        prev = candles[-offset - 1]
        trs.append(_true_range(candle, prev["close"]))
    if not trs:
        return None
    return sum(trs) / len(trs)


def _moving_average(values: Sequence[float], length: int) -> float | None:
    if length <= 0:
        return None
    if len(values) < length:
        return None
    window = values[-length:]
    return sum(window) / len(window)


def _touch_tolerance(level: float, pct: float) -> float:
    if pct <= 0:
        return 0.0
    return abs(level) * (pct / 100.0)


def _close_buffer(level: float, pct: float, atr: float | None, atr_mult: float) -> float:
    buffer_pct = abs(level) * (pct / 100.0) if pct > 0 else 0.0
    buffer_atr = (atr or 0.0) * atr_mult if atr is not None and atr_mult > 0 else 0.0
    return max(buffer_pct, buffer_atr, 0.0)


def _candle_touches(
    candle: dict[str, float],
    side: str,
    level: float,
    tolerance: float,
    use_wick: bool,
) -> bool:
    if side == "BUY":
        if use_wick:
            return candle["high"] >= level - tolerance
        return candle["close"] >= level - tolerance or candle["open"] >= level - tolerance
    else:
        if use_wick:
            return candle["low"] <= level + tolerance
        return candle["close"] <= level + tolerance or candle["open"] <= level + tolerance


def _close_confirms(
    candle: dict[str, float],
    side: str,
    level: float,
    buffer_value: float,
) -> bool:
    if buffer_value <= 0:
        return side == "BUY" and candle["close"] > level or side == "SELL" and candle["close"] < level
    if side == "BUY":
        return candle["close"] >= level + buffer_value
    return candle["close"] <= level - buffer_value


def _wick_ratio(candle: dict[str, float], side: str) -> float:
    body = abs(candle["close"] - candle["open"])
    if side == "BUY":
        wick = candle["high"] - max(candle["close"], candle["open"])
    else:
        wick = min(candle["close"], candle["open"]) - candle["low"]
    if body <= 0:
        return math.inf if wick > 0 else 0.0
    return wick / body


def _find_level_index(
    candles: Sequence[dict[str, float]],
    side: str,
    level: float,
    tolerance: float,
    start: int,
    end: int,
) -> int:
    best_idx: int | None = None
    best_dist: float | None = None
    for idx in range(start, max(start, end)):
        price = candles[idx]["high"] if side == "BUY" else candles[idx]["low"]
        dist = abs(price - level)
        if dist <= tolerance:
            current_best = best_dist if best_dist is not None else float("inf")
            if best_idx is None or dist < current_best or (
                math.isclose(dist, current_best) and idx < best_idx
            ):
                best_idx = idx
                best_dist = dist
    if best_idx is None:
        best_idx = max(start, end - 1)
    return best_idx


# ---------------------------------------------------------------------------
# Public API


def validate_false_breakout(
    ctx: dict[str, Any],
    side: str,
    level: float,
    timeframe: str,
    klines: Sequence[Any],
    now: datetime,
    params: FalseBreakoutSettings | dict[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Validate a breakout signal to filter potential false breakouts."""

    if isinstance(params, FalseBreakoutSettings):
        settings = params
    elif isinstance(params, dict):
        merged = _FBV_SETTINGS.to_log_dict()
        merged.update(params)
        settings = FalseBreakoutSettings(**merged)
    else:
        settings = _FBV_SETTINGS

    candles = [_normalize_candle(c) for c in klines]
    if len(candles) < 2:
        metrics = {"insufficient_candles": True}
        log_data = {
            "allowed": True,
            "reason": "insufficient_data",
            "level": level,
            "touch_age": None,
            "metrics": metrics,
        }
        logger.info("fbv.result %s", log_data)
        return True, "insufficient_data", {"metrics": metrics}

    last_idx = len(candles) - 1
    tolerance = _touch_tolerance(level, settings.touch_tolerance_pct)
    atr = _compute_atr(candles, settings.atr_period)
    buffer_value = _close_buffer(level, settings.close_buffer_pct, atr, settings.close_buffer_atr)
    last = candles[-1]

    volumes = [c["volume"] for c in candles[:-1]]
    vol_ma = _moving_average(volumes, settings.vol_ma_n)
    last_volume = last["volume"]
    vol_ratio = (last_volume / vol_ma) if vol_ma and vol_ma > 0 else None
    volume_ok = vol_ratio is None or vol_ratio >= settings.vol_ma_mult

    wick_ratio = _wick_ratio(last, side)
    wick_ok = math.isinf(wick_ratio) or wick_ratio <= settings.wick_max_ratio

    if math.isinf(wick_ratio):
        wick_ok = False

    close_distance = last["close"] - level if side == "BUY" else level - last["close"]
    close_distance_pct = (close_distance / abs(level) * 100.0) if level else None

    start_idx = max(0, last_idx - settings.level_lookback)
    level_idx = _find_level_index(candles, side, level, tolerance, start_idx, last_idx)

    touch_indices: list[int] = []
    first_touch_idx: int | None = None
    first_confirm_idx: int | None = None

    for idx in range(level_idx + 1, last_idx):
        candle = candles[idx]
        touched = _candle_touches(candle, side, level, tolerance, settings.use_wick_touch)
        confirmed = _close_confirms(candle, side, level, buffer_value)
        if touched:
            touch_indices.append(idx)
            if first_touch_idx is None and not confirmed:
                first_touch_idx = idx
        if confirmed and first_confirm_idx is None:
            first_confirm_idx = idx

    if first_touch_idx is None and touch_indices:
        first_touch_idx = touch_indices[0]

    touch_count = len(touch_indices)

    touch_age = None
    if first_touch_idx is not None:
        touch_age = last_idx - first_touch_idx

    time_window_ok = True
    if touch_age is not None and settings.time_window > 0:
        time_window_ok = touch_age <= settings.time_window

    min_touches_ok = touch_count >= max(0, settings.min_touches)

    last_confirm = _close_confirms(last, side, level, buffer_value)

    retest_ok = False
    if not last_confirm and first_confirm_idx is not None and settings.retest_wait > 0:
        if last_idx - first_confirm_idx <= settings.retest_wait:
            if _candle_touches(last, side, level, tolerance, True):
                retest_ok = True

    close_confirm_ok = last_confirm or retest_ok

    reason = "ok"
    allowed = True

    if not min_touches_ok:
        reason = "touches"
        allowed = False
    elif not time_window_ok:
        reason = "time_window"
        allowed = False
    elif not close_confirm_ok:
        reason = "close_buffer"
        allowed = False
    else:
        failed_checks = []
        if not wick_ok:
            failed_checks.append("wick_ratio")
        if not volume_ok:
            failed_checks.append("vol_confirm")
        if failed_checks:
            reason = "/".join(failed_checks)
            allowed = False

    metrics = {
        "atr": atr,
        "buffer_value": buffer_value,
        "close_distance": close_distance,
        "close_distance_pct": close_distance_pct,
        "volume": last_volume,
        "vol_ma": vol_ma,
        "vol_ratio": vol_ratio,
        "wick_ratio": wick_ratio,
        "touch_count": touch_count,
        "first_touch_idx": first_touch_idx,
        "first_confirm_idx": first_confirm_idx,
        "last_confirm": last_confirm,
        "retest_ok": retest_ok,
        "time_window_ok": time_window_ok,
        "min_touches_ok": min_touches_ok,
        "close_confirm_ok": close_confirm_ok,
    }

    log_data = {
        "allowed": allowed,
        "reason": reason,
        "level": level,
        "touch_age": touch_age,
        "metrics": metrics,
    }
    logger.info("fbv.result %s", log_data)
    if not allowed:
        logger.info("fbv.blocked_by=%s", reason)

    return allowed, reason, {"metrics": metrics}

