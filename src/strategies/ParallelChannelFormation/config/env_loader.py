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
    tp_mode: str
    sl_enabled: bool
    price_tick_override: float | None
    qty_step_override: float | None
    min_notional_buffer_pct: float
    tp_store_path: str


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

    tp_store_default = defaults.get("TP_STORE_PATH", {}).get("default")
    if tp_store_default:
        tp_store_path = str(tp_store_default)
    else:
        tp_store_path = str(
            Path(__file__).resolve().parent.parent / "state" / "tp_store.json"
        )
    tp_store_path = os.getenv("TP_STORE_PATH", tp_store_path)

    return ChannelEnv(
        tolerance_slope=float(tol),
        min_touches=int(touches),
        min_vertical_gap_pct=float(gap_pct),
        min_duration_bars=int(duration),
        confidence_threshold=float(confidence),
        tp_mode=tp_mode,
        sl_enabled=bool(sl_enabled),
        price_tick_override=float(price_override) if price_override else None,
        qty_step_override=float(qty_override) if qty_override else None,
        min_notional_buffer_pct=float(buffer_pct),
        tp_store_path=tp_store_path,
    )


__all__ = ["ChannelEnv", "load_env"]
