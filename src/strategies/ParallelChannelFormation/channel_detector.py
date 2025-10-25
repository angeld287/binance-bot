"""Parallel channel formation detector and execution flow."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Mapping, Sequence

from common.precision import format_decimal, round_to_step, to_decimal
from common.utils import sanitize_client_order_id
from core.ports.broker import BrokerPort
from core.ports.market_data import MarketDataPort
from core.ports.settings import SettingsProvider, get_symbol
from strategies.wedge_formation.strategy import (
    SymbolFilters,
    apply_qty_guards,
    compute_order_precision,
    get_symbol_filters,
    OrderPrecisionError,
    WedgeFormationStrategy,
)
from utils.tp_store_s3 import load_symbol_channel, load_tp_entry, persist_symbol_channel, persist_tp_value

from .geometry_utils import (
    Line,
    are_parallel,
    find_pivots,
    fit_line,
    has_min_touches,
    vertical_gap_pct,
)
from .config.env_loader import ChannelEnv, load_env
from . import filters as channel_filters

logger = logging.getLogger("bot.strategy.parallel_channel")

STRATEGY_NAME = "ParallelChannelFormation"

LOG_CHANNEL_META = os.getenv("LOG_CHANNEL_META", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CHANNEL_SLOPE_EPSILON = 1e-5
CHANNEL_BREAK_TOLERANCE = 0.0005
DEFAULT_MAX_TRADES_PER_CHANNEL = 1


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if math.isnan(value):  # type: ignore[arg-type]
        return default
    return value


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return value


CHANNEL_GEOM_SLOPE_TOLERANCE = _env_float("CHANNEL_SLOPE_TOLERANCE", 0.02)
CHANNEL_GEOM_WIDTH_TOLERANCE = _env_float("CHANNEL_WIDTH_TOLERANCE", 0.02)
CHANNEL_GEOM_EDGE_TOLERANCE_PCT = _env_float("CHANNEL_EDGE_TOLERANCE_PCT", 0.002)
CHANNEL_GEOM_ANCHOR_TOLERANCE_MS = _env_int("CHANNEL_ANCHOR_TOLERANCE_MS", 120_000)

UTC_MINUS_FOUR = timezone(timedelta(hours=-4))

TIMEFRAME_BOUNDS: dict[str, tuple[int, int]] = {
    "1m": (20, 60),
    "5m": (30, 100),
    "15m": (40, 150),
}


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _channel_trade_limit(env: ChannelEnv) -> int:
    raw_limit = getattr(env, "max_trades_per_channel", DEFAULT_MAX_TRADES_PER_CHANNEL)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = DEFAULT_MAX_TRADES_PER_CHANNEL
    if limit <= 0:
        limit = DEFAULT_MAX_TRADES_PER_CHANNEL
    return limit


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):  # type: ignore[arg-type]
        return default
    return number


def _relative_close(a: float, b: float, tolerance: float) -> bool:
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= abs(tolerance) * scale


def _anchors_close(a: Any, b: Any, tolerance_ms: int) -> bool:
    first = _coerce_float(a)
    second = _coerce_float(b)
    if first is None or second is None:
        return False
    return abs(first - second) <= float(abs(tolerance_ms))


def _load_active_symbol_channel(symbol: str) -> dict[str, Any] | None:
    stored = load_symbol_channel(symbol)
    if isinstance(stored, Mapping):
        return dict(stored)
    return None


def _persist_symbol_channel_state(symbol: str, payload: Mapping[str, Any]) -> None:
    persist_symbol_channel(symbol, payload)


def _normalize_channel_payload(
    *,
    symbol: str,
    side: str,
    slope: float,
    width: float,
    anchor_start_ts: int,
    anchor_end_ts: int,
    high_level: float,
    low_level: float,
    entry_price: float,
    lifetime_trades_opened: int,
    max_trades_allowed: int,
    tp_price: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "slope": float(slope),
        "width": float(width),
        "anchor_start_ts": int(anchor_start_ts),
        "anchor_end_ts": int(anchor_end_ts),
        "anchor_start_hm": _format_anchor_hm(anchor_start_ts),
        "anchor_end_hm": _format_anchor_hm(anchor_end_ts),
        "high_level": float(high_level),
        "low_level": float(low_level),
        "entry_price": float(entry_price),
        "lifetime_trades_opened": int(lifetime_trades_opened),
        "max_trades_allowed": int(max_trades_allowed),
    }
    if tp_price is not None:
        payload["tp_price"] = float(tp_price)
    if extra:
        for key, value in extra.items():
            if key in {"channel_id", "client_order_id"}:
                continue
            payload[key] = value
    return payload


def _is_same_channel_geometry(
    stored: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
) -> bool:
    if not stored:
        return False

    stored_slope = _coerce_float(stored.get("slope"))
    candidate_slope = _coerce_float(candidate.get("slope"))
    if stored_slope is None or candidate_slope is None:
        return False
    if not _relative_close(stored_slope, candidate_slope, CHANNEL_GEOM_SLOPE_TOLERANCE):
        return False

    stored_width = _coerce_float(stored.get("width"))
    candidate_width = _coerce_float(candidate.get("width"))
    if stored_width is None or candidate_width is None:
        return False
    if not _relative_close(stored_width, candidate_width, CHANNEL_GEOM_WIDTH_TOLERANCE):
        return False

    if not _anchors_close(
        stored.get("anchor_start_ts"), candidate.get("anchor_start_ts"), CHANNEL_GEOM_ANCHOR_TOLERANCE_MS
    ):
        return False
    if not _anchors_close(
        stored.get("anchor_end_ts"), candidate.get("anchor_end_ts"), CHANNEL_GEOM_ANCHOR_TOLERANCE_MS
    ):
        return False

    stored_high = _coerce_float(stored.get("high_level"))
    stored_low = _coerce_float(stored.get("low_level"))
    candidate_high = _coerce_float(candidate.get("high_level"))
    candidate_low = _coerce_float(candidate.get("low_level"))
    if None in {stored_high, stored_low, candidate_high, candidate_low}:
        return False

    if not _relative_close(stored_high, candidate_high, CHANNEL_GEOM_EDGE_TOLERANCE_PCT):
        return False
    if not _relative_close(stored_low, candidate_low, CHANNEL_GEOM_EDGE_TOLERANCE_PCT):
        return False

    return True


@dataclass(slots=True)
class MarketSnapshot:
    candles: Sequence[Sequence[float]]
    timeframe: str
    atr: float
    ema_fast: float | None
    ema_slow: float | None
    volume_avg: float | None


def _timeframe_to_seconds(timeframe: str | None) -> float:
    tf = (timeframe or "").strip().lower()
    if not tf:
        return 0.0
    unit = tf[-1]
    value_raw = tf[:-1] if unit.isalpha() else tf
    try:
        value = float(value_raw)
    except (TypeError, ValueError):
        return 0.0
    multiplier = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
    }.get(unit, 0)
    if multiplier <= 0:
        return 0.0
    return value * multiplier


def _format_anchor_hm(timestamp_raw: Any) -> str:
    try:
        timestamp_value = float(timestamp_raw)
    except (TypeError, ValueError):
        return ""

    if math.isnan(timestamp_value):  # type: ignore[arg-type]
        return ""

    # Handle both millisecond and second precision inputs gracefully.
    seconds = timestamp_value / 1000.0 if timestamp_value > 10**10 else timestamp_value
    try:
        dt_value = datetime.fromtimestamp(seconds, tz=UTC_MINUS_FOUR)
    except (OverflowError, OSError, ValueError):
        return ""
    return dt_value.strftime("%H:%M")


def _build_channel_meta(
    *,
    symbol: str,
    channel: Mapping[str, Any],
    upper_line: Line,
    lower_line: Line,
    candles: Sequence[Sequence[float]],
    timeframe: str | None,
    thresholds: Mapping[str, Any] | None,
    order_response: Mapping[str, Any] | None,
    client_id: str,
) -> dict[str, Any] | None:
    if not candles:
        return None

    entry_index = len(candles) - 1
    entry_candle_ts = float(candles[-1][0])
    entry_ts = datetime.utcnow().timestamp()

    lower_at_entry = float(lower_line.value_at(entry_index))
    upper_at_entry = float(upper_line.value_at(entry_index))
    slope_value = float(upper_line.slope)
    mid_at_entry = (upper_at_entry + lower_at_entry) / 2.0
    intercept_mid = mid_at_entry - slope_value * float(entry_index)
    width = abs(upper_at_entry - lower_at_entry) / 2.0

    pivots_high, pivots_low = find_pivots(candles)
    combined = sorted(pivots_high + pivots_low, key=lambda item: item[0])
    if combined:
        anchor_start_idx = combined[0][0]
        anchor_end_idx = combined[-1][0]
        try:
            anchor_start_ts = float(candles[anchor_start_idx][0])
        except (IndexError, TypeError, ValueError):
            anchor_start_ts = entry_candle_ts
        try:
            anchor_end_ts = float(candles[anchor_end_idx][0])
        except (IndexError, TypeError, ValueError):
            anchor_end_ts = entry_candle_ts
    else:
        anchor_start_ts = entry_candle_ts
        anchor_end_ts = entry_candle_ts

    timeframe_sec = _timeframe_to_seconds(timeframe)
    tolerance_pct = None
    if thresholds:
        tolerance_pct = thresholds.get("CHANNEL_TOLERANCE_PCT")

    meta: dict[str, Any] = {
        "side": channel.get("side"),
        "anchor_start_ts": int(anchor_start_ts),
        "anchor_end_ts": int(anchor_end_ts),
        "anchor_start_hm": _format_anchor_hm(anchor_start_ts),
        "anchor_end_hm": _format_anchor_hm(anchor_end_ts),
        "slope": slope_value,
        "intercept_mid": intercept_mid,
        "width": width,
        "lower_at_entry": lower_at_entry,
        "upper_at_entry": upper_at_entry,
        "entry_ts": entry_ts,
        "entry_index": entry_index,
        "entry_candle_ts": int(entry_candle_ts),
        "timeframe": timeframe,
        "timeframe_sec": timeframe_sec,
        "break_logged": False,
        "tolerance_pct": tolerance_pct,
        "break_tolerance": CHANNEL_BREAK_TOLERANCE,
    }

    if isinstance(order_response, Mapping):
        order_id = order_response.get("orderId") or order_response.get("order_id")
        if order_id is not None:
            meta["order_id"] = order_id

    try:
        meta["entry_price"] = float(channel.get("entry_price", 0.0) or meta["lower_at_entry"])
    except (TypeError, ValueError):
        meta["entry_price"] = meta["lower_at_entry"]

    return meta


def _log(payload: Mapping[str, Any]) -> None:
    try:
        logger.info(json.dumps(payload, default=str))
    except Exception:  # pragma: no cover - defensive
        logger.info(str(payload))


def _safe_float_env(key: str) -> float | None:
    raw = os.getenv(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _safe_int_env(key: str) -> int | None:
    raw = os.getenv(key)
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _iter_position_entries(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries: list[Mapping[str, Any]] = []
    positions = payload.get("open_positions")
    if isinstance(positions, Sequence) and not isinstance(positions, (str, bytes)):
        for item in positions:
            if isinstance(item, Mapping):
                entries.append(item)
    if not entries and isinstance(payload, Mapping):
        entries.append(payload)
    return entries


def _is_position_active(entry: Mapping[str, Any]) -> bool:
    status_raw = entry.get("status")
    if status_raw is not None:
        status_norm = str(status_raw).strip().upper()
        if status_norm in {"CLOSED", "FILLED", "CANCELED", "CANCELLED", "STOPPED"}:
            return False
    is_open = entry.get("is_open")
    if isinstance(is_open, bool) and not is_open:
        return False
    if entry.get("closed_at") or entry.get("closedAt"):
        return False
    channel_meta = entry.get("channel_meta")
    if isinstance(channel_meta, Mapping):
        structure_exit = channel_meta.get("structure_exit")
        if isinstance(structure_exit, Mapping):
            exit_status = structure_exit.get("status")
            exit_status_norm = str(exit_status or "").strip().upper()
            if exit_status_norm in {"SUCCESS", "FILLED", "CLOSED"}:
                return False
            if structure_exit.get("time") and not exit_status_norm:
                return False
    return True


def _channel_thresholds(env: ChannelEnv) -> dict[str, float | int | None]:
    min_touches = _safe_int_env("CHANNEL_MIN_TOUCHES_PER_SIDE")
    min_bars = _safe_int_env("CHANNEL_MIN_BARS")
    min_width_pct = _safe_float_env("CHANNEL_MIN_WIDTH_PCT")
    min_width_atr = _safe_float_env("CHANNEL_MIN_WIDTH_ATR")
    max_slope_diff_pct = _safe_float_env("CHANNEL_MAX_SLOPE_DIFF_PCT")
    tolerance_pct = _safe_float_env("CHANNEL_TOLERANCE_PCT")

    return {
        "CHANNEL_MIN_TOUCHES_PER_SIDE": min_touches if min_touches is not None else env.min_touches,
        "CHANNEL_MIN_BARS": min_bars if min_bars is not None else env.min_duration_bars,
        "CHANNEL_MIN_WIDTH_PCT": (
            min_width_pct if min_width_pct is not None else env.min_vertical_gap_pct
        ),
        "CHANNEL_MIN_WIDTH_ATR": min_width_atr,
        "CHANNEL_MAX_SLOPE_DIFF_PCT": (
            max_slope_diff_pct if max_slope_diff_pct is not None else env.tolerance_slope
        ),
        "CHANNEL_TOLERANCE_PCT": (
            tolerance_pct if tolerance_pct is not None else env.tolerance_slope
        ),
    }


def _count_touches(line: Line, pivots: Sequence[tuple[int, float]], *, tolerance: float) -> int:
    touches = 0
    tol = abs(float(tolerance))
    for idx, price in pivots:
        expected = line.value_at(idx)
        if abs(expected - price) <= tol:
            touches += 1
    return touches


def _apply_overrides(filters: SymbolFilters, env: ChannelEnv) -> SymbolFilters:
    tick = filters.tick_size
    step = filters.step_size
    min_notional = filters.min_notional
    min_qty = filters.min_qty

    if env.price_tick_override is not None and env.price_tick_override > 0:
        tick = to_decimal(env.price_tick_override)
    if env.qty_step_override is not None and env.qty_step_override > 0:
        step = to_decimal(env.qty_step_override)

    buffer_multiplier = Decimal("1") + to_decimal(env.min_notional_buffer_pct)
    if buffer_multiplier > 1:
        min_notional = min_notional * buffer_multiplier

    return SymbolFilters(
        tick_size=tick,
        step_size=step,
        min_notional=min_notional,
        min_qty=min_qty,
    )


def _load_tp_entry(symbol: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    payload = load_tp_entry(symbol)
    if not payload:
        _log(
            {
                "strategy": STRATEGY_NAME,
                "symbol": symbol,
                "state": "tp_store",
                "action": "load_tp",
                "storage_backend": "s3",
                "result": "miss",
                "reason": "tp_not_found_in_store",
            }
        )
        return None

    tp_raw = payload.get("tp_value")
    try:
        tp_value = float(tp_raw)
    except (TypeError, ValueError):
        _log(
            {
                "strategy": STRATEGY_NAME,
                "symbol": symbol,
                "state": "tp_store",
                "action": "load_tp",
                "storage_backend": "s3",
                "result": "miss",
                "reason": "invalid_tp_value",
            }
        )
        return None

    entry: dict[str, Any] = {
        "strategy": STRATEGY_NAME,
        "tp_price": tp_value,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "source": payload.get("source", "detector"),
    }
    if "side" in payload:
        entry["side"] = payload["side"]
    if "qty" in payload:
        entry["qty"] = payload["qty"]
    channel_meta = payload.get("channel_meta")
    if isinstance(channel_meta, Mapping):
        entry["channel_meta"] = channel_meta

    _log(
        {
            "strategy": STRATEGY_NAME,
            "symbol": symbol,
            "state": "tp_store",
            "action": "load_tp",
            "storage_backend": "s3",
            "result": "hit",
        }
    )
    return entry, payload


def _persist_tp_entry(symbol: str, payload: Mapping[str, Any]) -> None:
    timestamp_raw = payload.get("timestamp")
    ts_epoch: float
    if isinstance(timestamp_raw, (int, float)):
        ts_epoch = float(timestamp_raw)
    elif isinstance(timestamp_raw, str):
        try:
            ts_epoch = datetime.fromisoformat(timestamp_raw).timestamp()
        except ValueError:
            ts_epoch = datetime.utcnow().timestamp()
    else:
        ts_epoch = datetime.utcnow().timestamp()

    try:
        tp_price = float(payload.get("tp_price", 0.0))
    except (TypeError, ValueError):
        tp_price = 0.0

    extra: dict[str, Any] = {}
    for key in (
        "strategy",
        "side",
        "qty",
        "source",
        "status",
        "opened_at",
        "closed_at",
    ):
        value = payload.get(key)
        if value is not None:
            extra[key] = value

    persisted = persist_tp_value(
        symbol,
        tp_price,
        ts_epoch,
        extra=extra if extra else None,
    )
    _log(
        {
            "strategy": STRATEGY_NAME,
            "symbol": symbol,
            "state": "tp_store",
            "action": "save_tp",
            "storage_backend": "s3",
            "result": "persisted" if persisted else "error",
        }
    )


def _persist_channel_meta(
    symbol: str,
    tp_entry: Mapping[str, Any],
    channel_meta: Mapping[str, Any] | None,
) -> None:
    if not isinstance(channel_meta, Mapping):
        return

    existing_payload = load_tp_entry(symbol) or {}
    extra_payload: dict[str, Any] = {
        k: v
        for k, v in existing_payload.items()
        if k not in {"symbol", "tp_value", "timestamp"}
    }

    for key in (
        "strategy",
        "side",
        "qty",
        "source",
        "status",
        "opened_at",
        "closed_at",
    ):
        value = tp_entry.get(key)
        if value is not None:
            extra_payload[key] = value

    extra_payload["channel_meta"] = dict(channel_meta)

    timestamp_value = channel_meta.get("entry_ts")
    if timestamp_value is None:
        timestamp_value = existing_payload.get("timestamp")
    if timestamp_value is None:
        timestamp_value = datetime.utcnow().timestamp()
    try:
        timestamp_float = float(timestamp_value)
    except (TypeError, ValueError):
        timestamp_float = datetime.utcnow().timestamp()

    tp_value = tp_entry.get("tp_price")
    if tp_value is None:
        tp_value = existing_payload.get("tp_value", 0.0)
    try:
        tp_float = float(tp_value)
    except (TypeError, ValueError):
        tp_float = 0.0

    persist_tp_value(
        symbol,
        tp_float,
        timestamp_float,
        extra=extra_payload,
    )


def _maybe_log_channel_break(
    *,
    symbol: str,
    side: str,
    current_price: float | None,
    candle_close: float | None = None,
    store_payload: Mapping[str, Any],
    position: Mapping[str, Any] | None,
    exchange: BrokerPort | None = None,
    qty: float | None = None,
    filters: SymbolFilters | None = None,
) -> None:
    channel_meta = store_payload.get("channel_meta")
    if not isinstance(channel_meta, Mapping):
        return
    if channel_meta.get("break_logged"):
        return
    price_reference = candle_close if candle_close is not None else current_price
    if price_reference is None:
        return

    try:
        slope_value = float(channel_meta.get("slope", 0.0))
        intercept_mid = float(channel_meta.get("intercept_mid", 0.0))
        width = abs(float(channel_meta.get("width", 0.0)))
    except (TypeError, ValueError):
        return

    try:
        entry_index = float(channel_meta.get("entry_index", 0.0))
    except (TypeError, ValueError):
        entry_index = 0.0

    entry_ts_raw = channel_meta.get("entry_ts")
    timeframe_sec = channel_meta.get("timeframe_sec")
    if timeframe_sec in (None, 0, 0.0):
        timeframe_sec = _timeframe_to_seconds(channel_meta.get("timeframe"))
    try:
        timeframe_sec = float(timeframe_sec)
    except (TypeError, ValueError):
        timeframe_sec = 0.0

    try:
        entry_ts = float(entry_ts_raw) if entry_ts_raw is not None else None
    except (TypeError, ValueError):
        entry_ts = None

    now_dt = datetime.utcnow()
    now_ts = now_dt.timestamp()

    bars_elapsed = 0.0
    if entry_ts is not None and timeframe_sec and timeframe_sec > 0:
        bars_elapsed = max((now_ts - entry_ts) / timeframe_sec, 0.0)

    current_index = entry_index + bars_elapsed
    mid_now = slope_value * current_index + intercept_mid
    lower_now = mid_now - width
    upper_now = mid_now + width

    tolerance = channel_meta.get("break_tolerance", CHANNEL_BREAK_TOLERANCE)
    try:
        tolerance = abs(float(tolerance))
    except (TypeError, ValueError):
        tolerance = CHANNEL_BREAK_TOLERANCE

    side_norm = str(side or "").upper()
    broke_long = side_norm == "LONG" and price_reference < lower_now * (1 - tolerance)
    broke_short = side_norm == "SHORT" and price_reference > upper_now * (1 + tolerance)
    if not (broke_long or broke_short):
        return

    entry_price = None
    if isinstance(position, Mapping):
        try:
            entry_price = float(position.get("entryPrice"))
        except (TypeError, ValueError):
            entry_price = None
        if entry_price is None:
            raw_entry = position.get("entry_price")
            try:
                entry_price = float(raw_entry)
            except (TypeError, ValueError, AttributeError):
                entry_price = None
    if entry_price is None:
        try:
            entry_price = float(channel_meta.get("entry_price", 0.0))
        except (TypeError, ValueError):
            entry_price = None

    position_id = None
    if isinstance(position, Mapping):
        position_id = position.get("positionId") or position.get("position_id")
    if position_id is None:
        position_id = channel_meta.get("order_id")

    close_status: str | None = None
    close_error: str | None = None
    close_qty: float | None = None

    qty_active: float | None = None
    if isinstance(position, Mapping):
        try:
            qty_active = abs(float(position.get("positionAmt", 0.0)))
        except (TypeError, ValueError):
            qty_active = None
    if (qty_active is None or qty_active <= 0) and qty is not None:
        try:
            qty_active = abs(float(qty))
        except (TypeError, ValueError):
            qty_active = None

    if exchange is not None and filters is not None:
        qty_for_close = qty_active if qty_active is not None else 0.0
        if qty_for_close <= 0:
            close_status = "skipped"
            close_error = "qty_non_positive"
        else:
            success, error_reason, qty_used = _close_position_market(
                exchange=exchange,
                symbol=symbol,
                position_side=side_norm,
                qty=qty_for_close,
                filters=filters,
            )
            close_status = "success" if success else "failed"
            close_error = error_reason
            close_qty = qty_used
    elif exchange is not None and filters is None:
        close_status = "skipped"
        close_error = "filters_missing"

    if close_status is not None:
        close_payload: dict[str, Any] = {
            "action": "sl_structure_close",
            "strategy": STRATEGY_NAME,
            "symbol": symbol,
            "position_id": position_id,
            "side": side_norm,
            "reason": "channel_break",
            "status": close_status,
            "time": now_dt.isoformat(timespec="seconds"),
        }
        if close_qty is not None:
            close_payload["qty"] = close_qty
        if close_error:
            close_payload["error"] = close_error
        _log(close_payload)

    log_payload = {
        "action": "channel_break",
        "strategy": STRATEGY_NAME,
        "symbol": symbol,
        "position_id": position_id,
        "side": side_norm,
        "entry_price": entry_price,
        "time": now_dt.isoformat(timespec="seconds"),
        "channel": {
            "slope": slope_value,
            "width": width,
            "lower_at_entry": channel_meta.get("lower_at_entry"),
            "upper_at_entry": channel_meta.get("upper_at_entry"),
        },
        "now": {
            "price": price_reference,
            "lower": lower_now,
            "upper": upper_now,
        },
    }
    _log(log_payload)

    new_meta = dict(channel_meta)
    new_meta["break_logged"] = True
    new_meta["break_logged_at"] = now_dt.isoformat(timespec="seconds")
    if close_status is not None:
        structure_exit: dict[str, Any] = {
            "status": close_status,
            "time": now_dt.isoformat(timespec="seconds"),
            "reason": "channel_break",
        }
        if close_qty is not None:
            structure_exit["qty"] = close_qty
        if close_error:
            structure_exit["error"] = close_error
        new_meta["structure_exit"] = structure_exit

    extra_payload: dict[str, Any] = {
        k: v
        for k, v in store_payload.items()
        if k not in {"symbol", "tp_value", "timestamp"}
    }
    extra_payload["channel_meta"] = new_meta
    if close_status == "success":
        extra_payload["status"] = "CLOSED"
        extra_payload["closed_at"] = now_dt.isoformat(timespec="seconds")

    timestamp_value = store_payload.get("timestamp")
    if timestamp_value is None:
        timestamp_value = new_meta.get("entry_ts") or now_ts
    try:
        timestamp_float = float(timestamp_value)
    except (TypeError, ValueError):
        timestamp_float = now_ts

    tp_value_raw = store_payload.get("tp_value", 0.0)
    try:
        tp_value = float(tp_value_raw)
    except (TypeError, ValueError):
        tp_value = 0.0

    persist_tp_value(
        symbol,
        tp_value,
        timestamp_float,
        extra=extra_payload,
    )

def _build_client_id(*parts: Any) -> str:
    raw = "_".join(str(p) for p in parts if p is not None)
    return sanitize_client_order_id(raw)


def _precision_with_retry(
    *,
    price: Decimal,
    qty: Decimal,
    side: str,
    order_type: str,
    filters: SymbolFilters,
    exchange: BrokerPort,
    symbol: str,
) -> tuple[Decimal, Decimal]:
    attempt_qty = qty
    last_error: str | None = None
    for attempt in range(2):
        try:
            precision = compute_order_precision(
                price_requested=price,
                qty_requested=attempt_qty,
                stop_requested=None,
                side=side,
                order_type=order_type,
                filters=filters,
                exchange=exchange,
                symbol=symbol,
            )
        except OrderPrecisionError as exc:
            last_error = exc.reason
            attempt_qty = attempt_qty + filters.step_size
            continue

        if precision.price_adjusted is None or precision.qty_adjusted is None:
            last_error = "precision_missing"
            attempt_qty = attempt_qty + filters.step_size
            continue

        guard = apply_qty_guards(
            symbol=symbol,
            side=side,
            order_type=order_type,
            price_dec=precision.price_adjusted,
            qty_dec=precision.qty_adjusted,
            filters=filters,
        )
        if guard.success and guard.qty is not None:
            return precision.price_adjusted, guard.qty

        last_error = guard.reason or "qty_guard_failed"
        attempt_qty = (guard.qty or precision.qty_adjusted) + filters.step_size

    raise OrderPrecisionError("ORDER_PRECISION_FAILED", last_error or "precision_failed")


def _place_fixed_stop_loss(
    *,
    exchange: BrokerPort,
    symbol: str,
    position_side: str,
    entry_price: float,
    qty: Decimal,
    filters: SymbolFilters,
    env: ChannelEnv,
    position_id: Any | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if not env.sl_enabled:
        return None, "sl_disabled"

    pct = float(env.fixed_sl_pct)
    if pct <= 0:
        return None, "sl_pct_non_positive"

    side_norm = str(position_side or "").upper()
    exit_side = "SELL" if side_norm == "LONG" else "BUY"

    entry_dec = to_decimal(entry_price)
    pct_dec = Decimal(str(pct)) / Decimal("100")
    if exit_side == "SELL":
        target_stop = entry_dec * (Decimal("1") - pct_dec)
    else:
        target_stop = entry_dec * (Decimal("1") + pct_dec)

    if target_stop <= 0:
        return None, "sl_price_non_positive"

    try:
        precision = compute_order_precision(
            price_requested=None,
            qty_requested=qty,
            stop_requested=target_stop,
            side=exit_side,
            order_type="STOP_MARKET",
            filters=filters,
            exchange=exchange,
            symbol=symbol,
        )
    except OrderPrecisionError as exc:
        return None, exc.reason

    stop_adjusted = precision.stop_adjusted
    qty_adjusted = precision.qty_adjusted
    if stop_adjusted is None or qty_adjusted is None:
        return None, "precision_missing"

    guard = apply_qty_guards(
        symbol=symbol,
        side=exit_side,
        order_type="STOP_MARKET",
        price_dec=entry_dec,
        qty_dec=qty_adjusted,
        filters=filters,
        allow_increase=False,
    )
    if not guard.success or guard.qty is None:
        return None, guard.reason or "qty_guard_failed"

    client_id = _build_client_id(
        "PCF",
        symbol,
        "SL",
        int(datetime.utcnow().timestamp()),
    )

    stop_price = float(stop_adjusted)
    qty_value = float(guard.qty)

    try:
        exchange.place_stop_reduce_only(
            symbol=symbol,
            side=exit_side,
            stopPrice=stop_price,
            qty=qty_value,
            clientOrderId=client_id,
        )
    except Exception as exc:  # pragma: no cover - network failures
        return None, str(exc)

    payload = {
        "price": stop_price,
        "qty": qty_value,
        "client_order_id": client_id,
        "position_id": position_id,
    }
    return payload, None


def _close_position_market(
    *,
    exchange: BrokerPort,
    symbol: str,
    position_side: str,
    qty: float,
    filters: SymbolFilters,
) -> tuple[bool, str | None, float | None]:
    qty_value = abs(float(qty))
    if qty_value <= 0:
        return False, "qty_non_positive", None

    qty_dec = to_decimal(qty_value)
    step = filters.step_size
    if step > 0:
        qty_dec = round_to_step(qty_dec, step, rounding=ROUND_DOWN)
        if qty_dec <= 0 and step > 0:
            qty_dec = step

    qty_final = float(qty_dec)
    exit_side = "SELL" if str(position_side or "").upper() == "LONG" else "BUY"
    client_id = _build_client_id(
        "PCF",
        symbol,
        "EXIT",
        int(datetime.utcnow().timestamp()),
    )

    try:
        exchange.place_entry_market(
            symbol=symbol,
            side=exit_side,
            qty=qty_final,
            clientOrderId=client_id,
            reduceOnly=True,
        )
    except Exception as exc:  # pragma: no cover - network failures
        return False, str(exc), qty_final

    return True, None, qty_final


def place_tp_if_missing(
    *,
    exchange: BrokerPort,
    symbol: str,
    position_side: str,
    tp_price: float,
    qty: float,
    filters: SymbolFilters,
    env: ChannelEnv,
    store_entry: Mapping[str, Any],
) -> tuple[bool, str | None, bool]:
    tp_price_dec = to_decimal(tp_price)
    qty_dec = to_decimal(qty)
    filters_adj = _apply_overrides(filters, env)

    price_final: Decimal | None = None
    qty_final: Decimal | None = None
    reason: str | None = None
    retried = False

    for attempt in range(2):
        try:
            price_final, qty_final = _precision_with_retry(
                price=tp_price_dec,
                qty=qty_dec,
                side="SELL" if position_side == "LONG" else "BUY",
                order_type="LIMIT",
                filters=filters_adj,
                exchange=exchange,
                symbol=symbol,
            )
            break
        except OrderPrecisionError as exc:
            reason = exc.reason
            if attempt == 0:
                qty_dec = qty_dec + filters_adj.step_size
                retried = True
                continue
            _log(
                {
                    "strategy": STRATEGY_NAME,
                    "symbol": symbol,
                    "state": "tp_precision_failed",
                    "action": "reject",
                    "reason": reason,
                }
            )
            return False, reason, retried

    assert price_final is not None and qty_final is not None

    timestamp_raw = store_entry.get("timestamp")
    try:
        ts_dt = datetime.fromisoformat(str(timestamp_raw))
        ts_id = int(ts_dt.timestamp())
    except Exception:
        ts_id = int(datetime.utcnow().timestamp())

    client_id = _build_client_id("PCF", symbol, "TP", ts_id)
    side = "SELL" if position_side == "LONG" else "BUY"

    precision_log = {
        "tick": format_decimal(filters_adj.tick_size),
        "step": format_decimal(filters_adj.step_size),
        "minNotional": format_decimal(filters_adj.min_notional),
        "appliedBufferPct": float(env.min_notional_buffer_pct),
        "finalPrice": format_decimal(price_final),
        "finalQty": format_decimal(qty_final),
    }

    try:
        exchange.place_tp_reduce_only(
            symbol=symbol,
            side=side,
            tpPrice=float(price_final),
            qty=float(qty_final),
            clientOrderId=client_id,
        )
    except Exception as exc:  # pragma: no cover - network failures
        reason = str(exc)
        _log(
            {
                "strategy": STRATEGY_NAME,
                "symbol": symbol,
                "state": "tp_place_error",
                "action": "reject",
                "reason": reason,
                "precision_check": precision_log,
            }
        )
        return False, reason, retried

    _log(
        {
            "strategy": STRATEGY_NAME,
            "symbol": symbol,
            "state": "tp_placed",
            "action": "monitor_tp",
            "precision_check": precision_log,
        }
    )
    return True, None, retried


def _tp_exists(
    open_orders: Sequence[Mapping[str, Any]],
    target_price: float,
    filters: SymbolFilters,
) -> bool:
    if target_price <= 0:
        return False
    tolerance = float(filters.tick_size) if filters.tick_size > 0 else abs(target_price) * 0.0005
    for order in open_orders:
        status = str(order.get("status", "")).upper()
        if status not in {"NEW", "PARTIALLY_FILLED"}:
            continue
        price_val = order.get("price") or order.get("stopPrice")
        try:
            price = float(price_val)
        except (TypeError, ValueError):
            continue
        if abs(price - target_price) <= tolerance:
            return True
    return False


def _resolve_timeframe_bounds(env: ChannelEnv, snapshot: MarketSnapshot) -> tuple[int, int]:
    fallback_min_bars = int(env.min_duration_bars)
    fallback_max_bars = fallback_min_bars * 4
    timeframe_raw = (snapshot.timeframe or "").strip().lower()
    timeframe_key = timeframe_raw
    bounds = TIMEFRAME_BOUNDS.get(timeframe_key)
    if bounds is None:
        timeframe_seconds_map = {60: "1m", 300: "5m", 900: "15m"}
        timeframe_seconds = int(_timeframe_to_seconds(timeframe_raw) or 0)
        alias = timeframe_seconds_map.get(timeframe_seconds)
        if alias is not None:
            bounds = TIMEFRAME_BOUNDS.get(alias)
    if bounds is None:
        bounds = (fallback_min_bars, fallback_max_bars)
    return bounds


def _resolve_scan_window_bounds(
    env: ChannelEnv,
    snapshot: MarketSnapshot,
    total_candles: int,
) -> tuple[int, int]:
    min_duration_bars, max_duration_bars = _resolve_timeframe_bounds(env, snapshot)
    min_window = _safe_int_env("CHANNEL_MIN_BARS")
    max_window = _safe_int_env("CHANNEL_MAX_BARS")

    scan_min = min_window if min_window is not None else min_duration_bars
    scan_max = max_window if max_window is not None else max_duration_bars

    if scan_min <= 0:
        scan_min = 1
    if scan_max <= 0:
        scan_max = scan_min
    if scan_max < scan_min:
        scan_max = scan_min

    total = max(int(total_candles), 0)
    if total > 0:
        scan_max = min(scan_max, total)
        scan_min = min(scan_min, total)
        if scan_min <= 0:
            scan_min = 1
        if scan_max < scan_min:
            scan_max = scan_min
    else:
        scan_min = 1
        scan_max = 1

    return scan_min, scan_max


def _channel_pattern(
    candles: Sequence[Sequence[float]],
    env: ChannelEnv,
    snapshot: MarketSnapshot,
) -> tuple[
    tuple[Line, Line] | None,
    dict[str, Any],
    dict[str, float | int | None],
    dict[str, Any] | None,
]:
    pivots_high, pivots_low = find_pivots(candles)
    thresholds = _channel_thresholds(env)

    combined_pivots = list(pivots_high + pivots_low)
    bars_span = (
        combined_pivots[-1][0] - combined_pivots[0][0]
        if len(combined_pivots) >= 2
        else 0
    )

    min_duration_bars, max_duration_bars = _resolve_timeframe_bounds(env, snapshot)

    last_close = float(candles[-1][4]) if candles else 0.0
    atr_value = float(snapshot.atr or 0.0)

    metrics: dict[str, Any] = {
        "touches_top": 0,
        "touches_bottom": 0,
        "slope_top": None,
        "slope_bottom": None,
        "slope_diff_pct": None,
        "width_pct": None,
        "width_atr": None,
        "bars_span": bars_span,
        "overshoot_pct": 0.0,
        "tolerance_pct": thresholds.get("CHANNEL_TOLERANCE_PCT"),
        "atr": atr_value,
        "tick_size": env.price_tick_override,
        "price": last_close,
    }
    metrics.update({"min_bars": min_duration_bars, "max_bars": max_duration_bars})

    reason_detail: dict[str, Any] | None = None

    upper = fit_line(pivots_high)
    lower = fit_line(pivots_low)
    if upper is None or lower is None:
        reason_detail = {
            "reason": "insufficient_pivots",
            "measured_top": len(pivots_high),
            "measured_bottom": len(pivots_low),
            "min_pivots": 2,
        }
        return None, metrics, thresholds, reason_detail

    slope_top = float(upper.slope)
    slope_bottom = float(lower.slope)
    slope_diff = abs(slope_top - slope_bottom)
    tolerance_pct = thresholds.get("CHANNEL_TOLERANCE_PCT")
    if tolerance_pct is None:
        tolerance_pct = env.tolerance_slope
    overshoot_pct = max(0.0, slope_diff - tolerance_pct)

    metrics.update(
        {
            "slope_top": slope_top,
            "slope_bottom": slope_bottom,
            "slope_diff_pct": slope_diff,
            "overshoot_pct": overshoot_pct,
            "tolerance_pct": tolerance_pct,
        }
    )

    if not are_parallel(upper.slope, lower.slope, env.tolerance_slope):
        reason_detail = {
            "reason": "slope_diff_above_max",
            "measured": slope_diff,
            "max_pct": thresholds.get("CHANNEL_MAX_SLOPE_DIFF_PCT", env.tolerance_slope),
        }
        return None, metrics, thresholds, reason_detail

    gap_pct = vertical_gap_pct(upper, lower, last_close, len(candles) - 1)
    width_abs = abs(upper.value_at(len(candles) - 1) - lower.value_at(len(candles) - 1))
    width_atr = (width_abs / atr_value) if atr_value > 0 else None
    metrics.update({"width_pct": gap_pct, "width_atr": width_atr})

    min_width_pct = thresholds.get("CHANNEL_MIN_WIDTH_PCT", env.min_vertical_gap_pct)
    min_width_atr = thresholds.get("CHANNEL_MIN_WIDTH_ATR")
    min_width_pct_value = float(min_width_pct or env.min_vertical_gap_pct)
    meets_width_pct = gap_pct >= min_width_pct_value
    meets_width_atr = (
        width_atr is not None
        and min_width_atr is not None
        and width_atr >= float(min_width_atr)
    )
    if not (meets_width_pct or meets_width_atr):
        reason_detail = {
            "reason": "width_below_min",
            "measured": gap_pct,
            "min_pct": min_width_pct_value,
            "measured_atr": width_atr,
            "min_atr": float(min_width_atr) if min_width_atr is not None else None,
        }
        return None, metrics, thresholds, reason_detail

    atr_tolerance = atr_value
    touches_top = _count_touches(upper, pivots_high, tolerance=atr_tolerance)
    touches_bottom = _count_touches(lower, pivots_low, tolerance=atr_tolerance)
    metrics.update({"touches_top": touches_top, "touches_bottom": touches_bottom})

    min_touches = thresholds.get("CHANNEL_MIN_TOUCHES_PER_SIDE", env.min_touches)
    if not has_min_touches(
        upper, pivots_high, tolerance=atr_tolerance, min_touches=env.min_touches
    ):
        reason_detail = {
            "reason": "touches_top_below_min",
            "measured": touches_top,
            "min_touches": min_touches,
        }
        return None, metrics, thresholds, reason_detail
    if not has_min_touches(
        lower, pivots_low, tolerance=atr_tolerance, min_touches=env.min_touches
    ):
        reason_detail = {
            "reason": "touches_bottom_below_min",
            "measured": touches_bottom,
            "min_touches": min_touches,
        }
        return None, metrics, thresholds, reason_detail

    # Duration span validation is now handled by the scanning logic in ``run``.

    return (upper, lower), metrics, thresholds, None


def compute_channel_entry_tp(
    candles: Sequence[Sequence[float]],
    *,
    lines: tuple[Line, Line],
    snapshot: MarketSnapshot,
) -> dict[str, Any]:
    upper, lower = lines
    last_idx = len(candles) - 1
    last_close = float(candles[-1][4])
    lower_val = lower.value_at(last_idx)
    upper_val = upper.value_at(last_idx)

    distance_to_lower = abs(last_close - lower_val)
    distance_to_upper = abs(upper_val - last_close)

    if distance_to_lower < distance_to_upper:
        side = "LONG"
        entry_price = lower_val
        tp_price = upper_val
    else:
        side = "SHORT"
        entry_price = upper_val
        tp_price = lower_val

    rr = 0.0
    atr = snapshot.atr or 0.0
    if atr > 0:
        rr = abs(tp_price - entry_price) / atr if atr else 0.0

    return {
        "side": side,
        "entry_price": entry_price,
        "tp_price": tp_price,
        "rr": rr,
    }


def _check_open_orders(
    exchange: BrokerPort, symbol: str
) -> tuple[bool, int, list[Mapping[str, Any]]]:
    open_orders = exchange.open_orders(symbol)
    pending = []
    for order in open_orders:
        status = str(order.get("status", "")).upper()
        if status not in {"NEW", "PARTIALLY_FILLED"}:
            continue
        client_id = str(order.get("clientOrderId", "")).upper()
        if "_TP" in client_id:
            continue
        pending.append(order)
    _log(
        {
            "strategy": STRATEGY_NAME,
            "symbol": symbol,
            "state": "concurrency_orders",
            "action": "skip" if pending else "proceed",
            "concurrency_check": {
                "openOrdersCount": len(pending),
                "totalOpenOrders": len(open_orders),
            },
        }
    )
    return bool(pending), len(pending), list(open_orders)


def _check_position(exchange: BrokerPort, symbol: str) -> tuple[bool, float, dict[str, Any] | None]:
    position = exchange.get_position(symbol)
    if not position:
        _log(
            {
                "strategy": STRATEGY_NAME,
                "symbol": symbol,
                "state": "concurrency_position",
                "action": "proceed",
                "concurrency_check": {"hasOpenPosition": False},
            }
        )
        return False, 0.0, None
    try:
        amt = float(position.get("positionAmt", 0.0))
    except (TypeError, ValueError):
        amt = 0.0
    has_position = not math.isclose(amt, 0.0, abs_tol=1e-9)
    _log(
        {
            "strategy": STRATEGY_NAME,
            "symbol": symbol,
            "state": "concurrency_position",
            "action": "skip" if has_position else "proceed",
            "concurrency_check": {"hasOpenPosition": has_position, "positionAmt": amt},
        }
    )
    return has_position, amt, position


def run(
    symbol: str,
    market_data: MarketSnapshot,
    indicators: Mapping[str, Any] | None,
    env: ChannelEnv,
    *,
    exchange: BrokerPort,
) -> dict[str, Any]:
    pending, count, open_orders = _check_open_orders(exchange, symbol)
    if pending:
        return {"action": "reject", "reason": "open_order_exists", "meta": {"open_orders": count}}

    has_position, amt, position_raw = _check_position(exchange, symbol)
    if has_position:
        loaded_entry = _load_tp_entry(symbol)
        if not loaded_entry:
            _log(
                {
                    "strategy": STRATEGY_NAME,
                    "symbol": symbol,
                    "state": "tp_monitor",
                    "action": "monitor_tp",
                    "reason": "tp_not_found_in_store",
                    "tp_monitor": {"tpFoundInStore": False},
                }
            )
            return {"action": "monitor_tp", "reason": "tp_not_found_in_store"}

        store_entry, store_payload = loaded_entry
        store_entry = dict(store_entry)
        store_entry.setdefault("side", "LONG" if amt > 0 else "SHORT")
        store_entry.setdefault("qty", abs(amt))

        filters_raw = get_symbol_filters(exchange, symbol)
        filters = SymbolFilters(
            tick_size=to_decimal(filters_raw.tick_size),
            step_size=to_decimal(filters_raw.step_size),
            min_notional=to_decimal(filters_raw.min_notional),
            min_qty=to_decimal(filters_raw.min_qty),
        )
        filters = _apply_overrides(filters, env)

        qty_target = store_entry.get("qty")
        try:
            qty_value = float(qty_target) if qty_target is not None else abs(amt)
        except (TypeError, ValueError):
            qty_value = abs(amt)

        current_price = None
        if market_data.candles:
            try:
                current_price = float(market_data.candles[-1][4])
            except (TypeError, ValueError):
                current_price = None
        _maybe_log_channel_break(
            symbol=symbol,
            side=store_entry.get("side", "LONG" if amt > 0 else "SHORT"),
            current_price=current_price,
            candle_close=current_price,
            store_payload=store_payload,
            position=position_raw,
            exchange=exchange,
            qty=qty_value,
            filters=filters,
        )

        tp_price = float(store_entry.get("tp_price") or 0.0)
        tp_exists = _tp_exists(open_orders, tp_price, filters)
        if tp_exists:
            _log(
                {
                    "strategy": STRATEGY_NAME,
                    "symbol": symbol,
                    "state": "monitoring",
                    "action": "monitor_tp",
                    "tp_monitor": {
                        "tpFoundInStore": True,
                        "tpAlreadyOnExchange": True,
                        "placed": False,
                        "retriedOnce": False,
                    },
                }
            )
            return {"action": "monitor_tp", "state": "monitoring"}

        placed, reason, retried = place_tp_if_missing(
            exchange=exchange,
            symbol=symbol,
            position_side="LONG" if amt > 0 else "SHORT",
            tp_price=tp_price,
            qty=qty_value,
            filters=filters,
            env=env,
            store_entry=store_entry,
        )
        if placed:
            _log(
                {
                    "strategy": STRATEGY_NAME,
                    "symbol": symbol,
                    "state": "tp_monitor",
                    "action": "monitor_tp",
                    "tp_monitor": {
                        "tpFoundInStore": True,
                        "tpAlreadyOnExchange": False,
                        "placed": True,
                        "retriedOnce": retried,
                    },
                }
            )
            return {"action": "monitor_tp", "state": "tp_placed"}
        _log(
            {
                "strategy": STRATEGY_NAME,
                "symbol": symbol,
                "state": "tp_monitor",
                "action": "reject",
                "reason": reason or "precision_error_on_tp",
                "tp_monitor": {
                    "tpFoundInStore": True,
                    "tpAlreadyOnExchange": False,
                    "placed": False,
                    "retriedOnce": retried,
                },
            }
        )
        return {"action": "reject", "reason": reason or "precision_error_on_tp"}

    candles_all = list(market_data.candles or [])
    pattern_metrics: dict[str, Any] = {}
    pattern_thresholds = _channel_thresholds(env)
    pattern_reason: dict[str, Any] | None = None

    selected_candles: Sequence[Sequence[float]] | None = None
    lines: tuple[Line, Line] | None = None

    if not candles_all:
        (
            lines,
            pattern_metrics,
            pattern_thresholds,
            pattern_reason,
        ) = _channel_pattern(candles_all, env, market_data)
        if lines is not None:
            selected_candles = candles_all
    else:
        scan_min, scan_max = _resolve_scan_window_bounds(
            env, market_data, len(candles_all)
        )
        first_metrics: dict[str, Any] | None = None
        first_thresholds: dict[str, float | int | None] | None = None
        first_reason: dict[str, Any] | None = None

        for window_size in range(scan_max, scan_min - 1, -1):
            if window_size <= 0 or window_size > len(candles_all):
                continue
            start_limit = len(candles_all) - window_size
            for start_idx in range(start_limit, -1, -1):
                window = candles_all[start_idx : start_idx + window_size]
                (
                    candidate_lines,
                    metrics,
                    thresholds,
                    reason,
                ) = _channel_pattern(window, env, market_data)

                if first_metrics is None:
                    first_metrics = metrics
                    first_thresholds = thresholds
                    first_reason = reason

                pattern_metrics = metrics
                pattern_thresholds = thresholds
                pattern_reason = reason

                if candidate_lines is not None:
                    lines = candidate_lines
                    selected_candles = window
                    break
            if lines is not None:
                break

        if lines is None and first_metrics is not None:
            pattern_metrics = first_metrics
            pattern_thresholds = first_thresholds or pattern_thresholds
            pattern_reason = first_reason

    if lines is None or selected_candles is None:
        log_payload: dict[str, Any] = {
            "strategy": STRATEGY_NAME,
            "symbol": symbol,
            "state": "pattern_invalid",
            "action": "reject",
            "reason": "pattern_invalid",
            "pattern_metrics": pattern_metrics,
            "pattern_thresholds": pattern_thresholds,
        }
        if pattern_reason:
            log_payload["pattern_invalid_detail"] = pattern_reason
        _log(log_payload)
        return {"action": "reject", "reason": "pattern_invalid"}

    channel = compute_channel_entry_tp(selected_candles, lines=lines, snapshot=market_data)

    last_idx = len(selected_candles) - 1
    last_close = float(selected_candles[-1][4]) if selected_candles else 0.0
    upper_line, lower_line = lines
    lower_val = lower_line.value_at(last_idx) if selected_candles else 0.0
    upper_val = upper_line.value_at(last_idx) if selected_candles else 0.0
    distance_to_lower = abs(last_close - lower_val)
    distance_to_upper = abs(upper_val - last_close)
    slope_value = float(upper_line.slope)
    if slope_value > CHANNEL_SLOPE_EPSILON:
        channel_direction = "bullish"
    elif slope_value < -CHANNEL_SLOPE_EPSILON:
        channel_direction = "bearish"
    else:
        channel_direction = "flat"
    if channel["side"] == "LONG":
        edge = "bottom"
        distance_to_edge = distance_to_lower
    else:
        edge = "top"
        distance_to_edge = distance_to_upper
    distance_to_edge_pct = None
    if last_close:
        distance_to_edge_pct = distance_to_edge / abs(last_close)
    width_pct = pattern_metrics.get("width_pct")

    last_candle = selected_candles[-1] if selected_candles else None
    ohlc_meta = {}
    if last_candle:
        ohlc_meta = {
            "ohlc": {
                "open": float(last_candle[1]),
                "high": float(last_candle[2]),
                "low": float(last_candle[3]),
                "close": float(last_candle[4]),
            }
        }

    last_open_time = (
        int(candles_all[-1][0]) if candles_all else int(datetime.utcnow().timestamp())
    )
    client_id = _build_client_id("PCF", symbol, last_open_time)

    channel_meta_initial = _build_channel_meta(
        symbol=symbol,
        channel={
            "side": channel["side"],
            "entry_price": float(channel.get("entry_price", 0.0)),
        },
        upper_line=upper_line,
        lower_line=lower_line,
        candles=selected_candles,
        timeframe=market_data.timeframe,
        thresholds=pattern_thresholds,
        order_response=None,
        client_id=client_id,
    )

    max_trades_allowed = _channel_trade_limit(env)
    stored_channel = _load_active_symbol_channel(symbol)
    channel_state_payload: dict[str, Any] | None = None
    channel_state_same_geometry = False
    lifetime_trades = 0
    channel_limit_active = max_trades_allowed

    if isinstance(channel_meta_initial, Mapping):
        extra_fields = {
            key: channel_meta_initial[key]
            for key in (
                "intercept_mid",
                "entry_ts",
                "entry_index",
                "entry_candle_ts",
                "timeframe",
                "timeframe_sec",
                "break_logged",
                "tolerance_pct",
                "break_tolerance",
                "lower_at_entry",
                "upper_at_entry",
            )
            if key in channel_meta_initial
        }
        normalized_payload = _normalize_channel_payload(
            symbol=symbol,
            side=str(channel.get("side", "")),
            slope=_coerce_float(channel_meta_initial.get("slope"), slope_value) or slope_value,
            width=_coerce_float(channel_meta_initial.get("width"), 0.0) or 0.0,
            anchor_start_ts=int(channel_meta_initial.get("anchor_start_ts", 0)),
            anchor_end_ts=int(channel_meta_initial.get("anchor_end_ts", 0)),
            high_level=_coerce_float(
                channel_meta_initial.get("upper_at_entry"), upper_val
            )
            or upper_val,
            low_level=_coerce_float(
                channel_meta_initial.get("lower_at_entry"), lower_val
            )
            or lower_val,
            entry_price=_coerce_float(
                channel_meta_initial.get("entry_price"), channel.get("entry_price")
            )
            or float(channel.get("entry_price", lower_val)),
            lifetime_trades_opened=0,
            max_trades_allowed=max_trades_allowed,
            tp_price=_coerce_float(channel.get("tp_price")),
            extra=extra_fields,
        )
    else:
        normalized_payload = _normalize_channel_payload(
            symbol=symbol,
            side=str(channel.get("side", "")),
            slope=slope_value,
            width=float(abs(upper_val - lower_val) / 2.0),
            anchor_start_ts=int(selected_candles[0][0]) if selected_candles else 0,
            anchor_end_ts=int(selected_candles[-1][0]) if selected_candles else 0,
            high_level=float(upper_val),
            low_level=float(lower_val),
            entry_price=float(channel.get("entry_price", lower_val)),
            lifetime_trades_opened=0,
            max_trades_allowed=max_trades_allowed,
            tp_price=_coerce_float(channel.get("tp_price")),
            extra=None,
        )

    if normalized_payload:
        if _is_same_channel_geometry(stored_channel, normalized_payload):
            channel_state_same_geometry = True
            channel_state_payload = dict(stored_channel or {})
        else:
            channel_state_payload = dict(normalized_payload)
            channel_state_payload["lifetime_trades_opened"] = 0
            channel_state_payload["max_trades_allowed"] = max_trades_allowed
            _persist_symbol_channel_state(symbol, channel_state_payload)
        if channel_state_payload:
            lifetime_trades = _coerce_int(channel_state_payload.get("lifetime_trades_opened"), 0)
            channel_limit_active = _coerce_int(
                channel_state_payload.get("max_trades_allowed"), max_trades_allowed
            )
            if channel_limit_active <= 0:
                channel_limit_active = max_trades_allowed
            channel_state_payload["max_trades_allowed"] = channel_limit_active
            if channel_state_same_geometry and channel_state_payload is not None:
                if lifetime_trades >= channel_limit_active:
                    anchors_label = (
                        f"{channel_state_payload.get('anchor_start_hm', '')}-"
                        f"{channel_state_payload.get('anchor_end_hm', '')}"
                    )
                    _log(
                        {
                            "action": "reject",
                            "reason": "channel_trade_limit",
                            "strategy": STRATEGY_NAME,
                            "symbol": symbol,
                            "side": channel_state_payload.get("side"),
                            "lifetime": lifetime_trades,
                            "max": channel_limit_active,
                            "slope": channel_state_payload.get("slope"),
                            "width": channel_state_payload.get("width"),
                            "anchors": anchors_label,
                        }
                    )
                    return {"action": "reject", "reason": "channel_trade_limit"}
    else:
        channel_state_payload = None

    filters_result, filter_reason = channel_filters.apply_filters(
        rr=channel.get("rr"),
        confidence_threshold=env.confidence_threshold,
        ema_fast=market_data.ema_fast,
        ema_slow=market_data.ema_slow,
        volume_avg=market_data.volume_avg,
        atr=market_data.atr,
        meta={**(indicators or {}), **ohlc_meta},
        side=channel.get("side"),
    )
    if LOG_CHANNEL_META:
        rr_value = channel.get("rr")
        ema_ok = filter_reason != "ema_filter"
        if filters_result:
            ema_ok = True

        pivots_high, pivots_low = find_pivots(selected_candles)
        combined_pivots = sorted(pivots_high + pivots_low, key=lambda item: item[0])
        anchor_start_idx = combined_pivots[0][0] if combined_pivots else 0
        anchor_end_idx = combined_pivots[-1][0] if combined_pivots else last_idx

        def _safe_ts(idx: int) -> float | None:
            try:
                return float(selected_candles[idx][0])
            except (IndexError, TypeError, ValueError):
                return None

        anchor_start_ts_raw = _safe_ts(anchor_start_idx)
        anchor_end_ts_raw = _safe_ts(anchor_end_idx)
        anchor_start_ts = int(anchor_start_ts_raw) if anchor_start_ts_raw is not None else None
        anchor_end_ts = int(anchor_end_ts_raw) if anchor_end_ts_raw is not None else None

        upper_meta = {
            "slope": float(upper_line.slope),
            "intercept": float(upper_line.intercept),
            "value_at_last": float(upper_val),
        }
        lower_meta = {
            "slope": float(lower_line.slope),
            "intercept": float(lower_line.intercept),
            "value_at_last": float(lower_val),
        }

        log_payload = {
            "strategy": STRATEGY_NAME,
            "symbol": symbol,
            "state": "signal_eval",
            "channel": {
                "slope": float(slope_value),
                "direction": channel_direction,
                "width_pct": float(width_pct) if width_pct is not None else None,
                "anchor_start_idx": int(anchor_start_idx),
                "anchor_end_idx": int(anchor_end_idx),
                "anchor_start_ts": anchor_start_ts,
                "anchor_end_ts": anchor_end_ts,
                "upper_line": upper_meta,
                "lower_line": lower_meta,
                "pivots": {
                    "high": [(int(idx), float(price)) for idx, price in pivots_high],
                    "low": [(int(idx), float(price)) for idx, price in pivots_low],
                },
            },
            "edge": edge,
            "distance_to_edge_pct": float(distance_to_edge_pct) if distance_to_edge_pct is not None else None,
            "filters": {
                "ema_ok": bool(ema_ok),
                "rr": float(rr_value) if rr_value is not None else None,
                "rr_min": float(env.confidence_threshold) if env.confidence_threshold is not None else None,
            },
            "side_decision": channel.get("side"),
        }
        _log(log_payload)
    if not filters_result:
        _log(
            {
                "strategy": STRATEGY_NAME,
                "symbol": symbol,
                "state": "filter_reject",
                "action": "reject",
                "reason": filter_reason,
            }
        )
        return {"action": "reject", "reason": filter_reason or "filter_reject"}

    filters_raw = get_symbol_filters(exchange, symbol)
    filters = SymbolFilters(
        tick_size=filters_raw.tick_size,
        step_size=filters_raw.step_size,
        min_notional=filters_raw.min_notional,
        min_qty=filters_raw.min_qty,
    )
    filters = _apply_overrides(filters, env)

    entry_price = to_decimal(channel["entry_price"])
    qty_estimate = to_decimal(indicators.get("qty") if indicators else 0)
    if qty_estimate <= 0:
        qty_estimate = Decimal("1")

    try:
        price_final, qty_final = _precision_with_retry(
            price=entry_price,
            qty=qty_estimate,
            side="BUY" if channel["side"] == "LONG" else "SELL",
            order_type="LIMIT",
            filters=filters,
            exchange=exchange,
            symbol=symbol,
        )
    except OrderPrecisionError as exc:
        _log(
            {
                "strategy": STRATEGY_NAME,
                "symbol": symbol,
                "state": "precision_error_entry",
                "action": "reject",
                "reason": exc.reason,
            }
        )
        return {"action": "reject", "reason": "precision_error_on_entry"}

    precision_log = {
        "tick": format_decimal(filters.tick_size),
        "step": format_decimal(filters.step_size),
        "minNotional": format_decimal(filters.min_notional),
        "appliedBufferPct": float(env.min_notional_buffer_pct),
        "finalPrice": format_decimal(price_final),
        "finalQty": format_decimal(qty_final),
    }

    opened_at_iso = datetime.utcnow().isoformat(timespec="seconds")
    tp_entry = {
        "strategy": STRATEGY_NAME,
        "tp_price": channel["tp_price"],
        "side": channel["side"],
        "timestamp": opened_at_iso,
        "source": "detector",
        "qty": float(qty_final),
        "status": "OPEN",
        "opened_at": opened_at_iso,
    }
    _persist_tp_entry(symbol, tp_entry)

    try:
        order_response = exchange.place_entry_limit(
            symbol=symbol,
            side="BUY" if channel["side"] == "LONG" else "SELL",
            price=float(price_final),
            qty=float(qty_final),
            clientOrderId=client_id,
        )
    except Exception as exc:  # pragma: no cover - exchange failure
        _log(
            {
                "strategy": STRATEGY_NAME,
                "symbol": symbol,
                "state": "order_error",
                "action": "reject",
                "reason": str(exc),
                "precision_check": precision_log,
            }
        )
        return {"action": "reject", "reason": "order_error"}

    channel_meta: Mapping[str, Any] | None
    if isinstance(channel_meta_initial, Mapping):
        channel_meta = dict(channel_meta_initial)
        channel_meta["entry_price"] = float(price_final)
    else:
        channel_meta = _build_channel_meta(
            symbol=symbol,
            channel={"side": channel["side"], "entry_price": float(price_final)},
            upper_line=upper_line,
            lower_line=lower_line,
            candles=selected_candles,
            timeframe=market_data.timeframe,
            thresholds=pattern_thresholds,
            order_response=None,
            client_id=client_id,
        )
        channel_meta = dict(channel_meta) if isinstance(channel_meta, Mapping) else None

    position_id = None
    if isinstance(channel_meta, Mapping):
        if isinstance(order_response, Mapping):
            order_id = order_response.get("orderId") or order_response.get("order_id")
            if order_id is not None:
                channel_meta["order_id"] = order_id
        position_id = channel_meta.get("order_id")
    if position_id is None and isinstance(order_response, Mapping):
        position_id = order_response.get("orderId")

    sl_info, sl_error = _place_fixed_stop_loss(
        exchange=exchange,
        symbol=symbol,
        position_side=channel["side"],
        entry_price=float(price_final),
        qty=qty_final,
        filters=filters,
        env=env,
        position_id=position_id,
    )

    if channel_meta is not None:
        channel_meta = dict(channel_meta)
        if sl_info:
            channel_meta["fixed_sl"] = {
                "price": sl_info.get("price"),
                "qty": sl_info.get("qty"),
                "client_order_id": sl_info.get("client_order_id"),
                "pct": float(env.fixed_sl_pct),
            }
        elif env.sl_enabled:
            channel_meta.setdefault("fixed_sl", {})

    now_iso = datetime.utcnow().isoformat(timespec="seconds")
    if sl_info:
        _log(
            {
                "action": "fixed_sl_set",
                "strategy": STRATEGY_NAME,
                "symbol": symbol,
                "position_id": sl_info.get("position_id") or position_id,
                "side": channel["side"],
                "entry_price": float(price_final),
                "sl_price": sl_info.get("price"),
                "pct": float(env.fixed_sl_pct),
                "qty": sl_info.get("qty"),
                "time": now_iso,
            }
        )
    elif env.sl_enabled:
        action = "fixed_sl_skipped" if sl_error in {"sl_disabled", "sl_pct_non_positive"} else "fixed_sl_failed"
        _log(
            {
                "action": action,
                "strategy": STRATEGY_NAME,
                "symbol": symbol,
                "position_id": position_id,
                "side": channel["side"],
                "entry_price": float(price_final),
                "pct": float(env.fixed_sl_pct),
                "reason": sl_error,
                "time": now_iso,
            }
        )

    if channel_state_payload is not None:
        updated_channel_state = dict(channel_state_payload)
        updated_channel_state["entry_price"] = float(price_final)
        updated_channel_state["lifetime_trades_opened"] = lifetime_trades + 1
        updated_channel_state["max_trades_allowed"] = channel_limit_active
        if sl_info:
            updated_channel_state["fixed_sl"] = {
                "price": sl_info.get("price"),
                "qty": sl_info.get("qty"),
                "client_order_id": sl_info.get("client_order_id"),
                "pct": float(env.fixed_sl_pct),
            }
        elif env.sl_enabled:
            updated_channel_state.setdefault("fixed_sl", {})

        channel_state_payload = updated_channel_state
        lifetime_after = updated_channel_state["lifetime_trades_opened"]
        _persist_symbol_channel_state(symbol, updated_channel_state)

        anchors_label = (
            f"{updated_channel_state.get('anchor_start_hm', '')}-"
            f"{updated_channel_state.get('anchor_end_hm', '')}"
        )
        _log(
            {
                "action": "open",
                "reason": "channel_entry",
                "strategy": STRATEGY_NAME,
                "symbol": symbol,
                "side": channel["side"],
                "lifetime_after": lifetime_after,
                "max": updated_channel_state.get("max_trades_allowed"),
                "slope": updated_channel_state.get("slope"),
                "width": updated_channel_state.get("width"),
                "anchors": anchors_label,
            }
        )

    _persist_channel_meta(symbol, tp_entry, channel_meta)

    _log(
        {
            "strategy": STRATEGY_NAME,
            "symbol": symbol,
            "state": "placed_order",
            "action": "place_order",
            "precision_check": precision_log,
        }
    )

    return {
        "action": "place_order",
        "side": channel["side"],
        "entry_price": float(price_final),
        "tp1": channel["tp_price"],
        "tp_mode": env.tp_mode,
        "meta": {"precision": precision_log},
    }


class ParallelChannelFormationStrategy:
    """High-level AWS Lambda compatible strategy wrapper."""

    def __init__(
        self,
        market_data: MarketDataPort,
        broker: BrokerPort,
        settings: SettingsProvider,
    ) -> None:
        self._market_data = market_data
        self._broker = broker
        self._settings = settings

    def run(
        self,
        exchange: BrokerPort | None = None,
        market_data: MarketDataPort | None = None,
        settings: SettingsProvider | None = None,
        now_utc: datetime | None = None,
        event: Any | None = None,
    ) -> dict[str, Any]:
        exch = exchange or self._broker
        md = market_data or self._market_data
        settings = settings or self._settings
        symbol = get_symbol(settings)
        timeframe = str(settings.get("INTERVAL", "15m"))

        env = load_env(settings=settings)

        candles = md.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=200)
        atr = WedgeFormationStrategy._compute_atr(candles)

        ema_fast = None
        ema_slow = None
        closes = [float(c[4]) for c in candles]
        if len(closes) >= 25:
            ema_fast = sum(closes[-7:]) / 7
            ema_slow = sum(closes[-25:]) / 25

        volume_avg = None
        volumes = [float(c[5]) for c in candles]
        if volumes:
            volume_avg = sum(volumes[-20:]) / min(len(volumes), 20)

        snapshot = MarketSnapshot(
            candles=candles,
            timeframe=timeframe,
            atr=atr,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            volume_avg=volume_avg,
        )

        indicators = {"qty": 1.0}

        return run(
            symbol,
            snapshot,
            indicators,
            env,
            exchange=exch,
        )


__all__ = ["ParallelChannelFormationStrategy", "STRATEGY_NAME", "run"]
