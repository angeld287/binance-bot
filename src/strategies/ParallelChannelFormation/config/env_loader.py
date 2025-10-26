"""Environment loader for the Parallel Channel Formation strategy."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

from config.utils import parse_bool


@dataclass(frozen=True)
class ChannelEnv:
    tolerance_slope: float
    min_touches: int
    min_vertical_gap_pct: float
    min_duration_bars: int
    confidence_threshold: float
    ema_distance_filter_enabled: bool
    ema_distance_threshold_pct: float
    tp_mode: str
    sl_enabled: bool
    fixed_sl_pct: float
    price_tick_override: float | None
    qty_step_override: float | None
    min_notional_buffer_pct: float
    max_trades_per_channel: int
    exit_buffer_pct: float


def _load_defaults() -> MutableMapping[str, Mapping[str, object]]:
    path = Path(__file__).with_name("env_variables.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _get_value(
    key: str,
    *,
    defaults: Mapping[str, Mapping[str, object]],
    coerce=float,
    allow_none: bool = False,
) -> float | int | None:
    raw = os.getenv(key)
    if raw is None:
        default_entry = defaults.get(key, {})
        default_value = default_entry.get("default")
        if default_value is None:
            return None
        raw = str(default_value)
    try:
        return coerce(raw)
    except (TypeError, ValueError):
        return None if allow_none else coerce(defaults.get(key, {}).get("default", 0))


def load_env(*, settings=None) -> ChannelEnv:
    defaults = _load_defaults()

    tol = _get_value("CHANNEL_TOLERANCE_SLOPE", defaults=defaults, coerce=float) or 0.05
    touches = int(_get_value("CHANNEL_MIN_TOUCHES", defaults=defaults, coerce=int) or 3)
    gap_pct = (
        _get_value("CHANNEL_MIN_VERTICAL_DISTANCE_PCT", defaults=defaults, coerce=float)
        or 0.01
    )
    duration = int(
        _get_value("CHANNEL_MIN_DURATION_BARS", defaults=defaults, coerce=int) or 10
    )
    confidence = (
        _get_value("CHANNEL_CONFIDENCE_THRESHOLD", defaults=defaults, coerce=float) or 0.0
    )
    ema_distance_enabled = parse_bool(
        os.getenv("PARALLEL_CHANNEL_EMA_DISTANCE_FILTER_ENABLED"),
        default=True,
    )
    ema_distance_threshold_raw = _get_value(
        "PARALLEL_CHANNEL_EMA_DISTANCE_THRESHOLD_PCT",
        defaults=defaults,
        coerce=float,
    )
    ema_distance_threshold = (
        float(ema_distance_threshold_raw)
        if ema_distance_threshold_raw is not None
        else 0.8
    )
    tp_mode = str(
        os.getenv("TP_MODE")
        or defaults.get("TP_MODE", {}).get("default")
        or "opuesto_inmediato"
    )
    sl_enabled = parse_bool(
        os.getenv("SL_ENABLED"),
        default=parse_bool(defaults.get("SL_ENABLED", {}).get("default"), default=False),
    )
    price_override = _get_value(
        "PRICE_TICK_OVERRIDE", defaults=defaults, coerce=float, allow_none=True
    )
    qty_override = _get_value(
        "QTY_STEP_OVERRIDE", defaults=defaults, coerce=float, allow_none=True
    )
    buffer_pct = (
        _get_value("MIN_NOTIONAL_BUFFER_PCT", defaults=defaults, coerce=float) or 0.03
    )
    fixed_sl_pct = (
        _get_value("CHANNEL_FIXED_SL_PCT", defaults=defaults, coerce=float)
        or 1.0
    )
    exit_buffer_pct = (
        _get_value("EXIT_BUFFER_PCT", defaults=defaults, coerce=float, allow_none=True)
        or 0.0
    )
    max_trades_channel_raw = _get_value(
        "CHANNEL_MAX_TRADES_PER_CHANNEL", defaults=defaults, coerce=int
    )
    try:
        max_trades_channel = int(max_trades_channel_raw or 1)
    except (TypeError, ValueError):
        max_trades_channel = 1
    if max_trades_channel <= 0:
        max_trades_channel = 1

    return ChannelEnv(
        tolerance_slope=float(tol),
        min_touches=int(touches),
        min_vertical_gap_pct=float(gap_pct),
        min_duration_bars=int(duration),
        confidence_threshold=float(confidence),
        ema_distance_filter_enabled=bool(ema_distance_enabled),
        ema_distance_threshold_pct=float(ema_distance_threshold),
        tp_mode=tp_mode,
        sl_enabled=bool(sl_enabled),
        fixed_sl_pct=float(fixed_sl_pct),
        price_tick_override=float(price_override) if price_override else None,
        qty_step_override=float(qty_override) if qty_override else None,
        min_notional_buffer_pct=float(buffer_pct),
        max_trades_per_channel=int(max_trades_channel),
        exit_buffer_pct=float(exit_buffer_pct),
    )


__all__ = ["ChannelEnv", "load_env"]
