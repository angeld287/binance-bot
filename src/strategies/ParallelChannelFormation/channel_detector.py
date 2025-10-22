"""Parallel channel formation detector and execution flow."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.precision import format_decimal, to_decimal
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

from .geometry_utils import (
    Line,
    are_parallel,
    find_pivots,
    fit_line,
    has_min_duration,
    has_min_touches,
    vertical_gap_pct,
)
from .config.env_loader import ChannelEnv, load_env
from . import filters as channel_filters

logger = logging.getLogger("bot.strategy.parallel_channel")

STRATEGY_NAME = "ParallelChannelFormation"


@dataclass(slots=True)
class MarketSnapshot:
    candles: Sequence[Sequence[float]]
    timeframe: str
    atr: float
    ema_fast: float | None
    ema_slow: float | None
    volume_avg: float | None


def _log(payload: Mapping[str, Any]) -> None:
    try:
        logger.info(json.dumps(payload, default=str))
    except Exception:  # pragma: no cover - defensive
        logger.info(str(payload))


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


def _read_tp_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def load_tp_from_store(symbol: str, path: Path) -> dict[str, Any] | None:
    data = _read_tp_store(path)
    entry = data.get(symbol)
    if not isinstance(entry, dict):
        return None
    if entry.get("strategy") != STRATEGY_NAME:
        return None
    return entry


def save_tp_to_store(symbol: str, payload: Mapping[str, Any], path: Path) -> None:
    store = _read_tp_store(path)
    store[symbol] = dict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(store, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


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


def _channel_pattern(
    candles: Sequence[Sequence[float]],
    env: ChannelEnv,
    snapshot: MarketSnapshot,
) -> tuple[Line, Line] | None:
    pivots_high, pivots_low = find_pivots(candles)
    upper = fit_line(pivots_high)
    lower = fit_line(pivots_low)
    if upper is None or lower is None:
        return None

    if not are_parallel(upper.slope, lower.slope, env.tolerance_slope):
        return None

    last_close = float(candles[-1][4]) if candles else 0.0
    gap_pct = vertical_gap_pct(upper, lower, last_close, len(candles) - 1)
    if gap_pct < env.min_vertical_gap_pct:
        return None

    atr_tolerance = snapshot.atr or 0.0
    if not has_min_touches(upper, pivots_high, tolerance=atr_tolerance, min_touches=env.min_touches):
        return None
    if not has_min_touches(lower, pivots_low, tolerance=atr_tolerance, min_touches=env.min_touches):
        return None

    if not has_min_duration(pivots_high + pivots_low, min_bars=env.min_duration_bars):
        return None

    return upper, lower


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
    tp_store_path: Path,
) -> dict[str, Any]:
    pending, count, open_orders = _check_open_orders(exchange, symbol)
    if pending:
        return {"action": "reject", "reason": "open_order_exists", "meta": {"open_orders": count}}

    has_position, amt, _position_raw = _check_position(exchange, symbol)
    if has_position:
        store_entry = load_tp_from_store(symbol, tp_store_path)
        if not store_entry:
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

        filters_raw = get_symbol_filters(exchange, symbol)
        filters = SymbolFilters(
            tick_size=to_decimal(filters_raw.tick_size),
            step_size=to_decimal(filters_raw.step_size),
            min_notional=to_decimal(filters_raw.min_notional),
            min_qty=to_decimal(filters_raw.min_qty),
        )
        filters = _apply_overrides(filters, env)

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

        qty_target = store_entry.get("qty")
        try:
            qty_value = float(qty_target) if qty_target is not None else abs(amt)
        except (TypeError, ValueError):
            qty_value = abs(amt)

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

    lines = _channel_pattern(market_data.candles, env, market_data)
    if lines is None:
        _log(
            {
                "strategy": STRATEGY_NAME,
                "symbol": symbol,
                "state": "pattern_invalid",
                "action": "reject",
                "reason": "pattern_invalid",
            }
        )
        return {"action": "reject", "reason": "pattern_invalid"}

    channel = compute_channel_entry_tp(market_data.candles, lines=lines, snapshot=market_data)

    last_candle = market_data.candles[-1] if market_data.candles else None
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

    tp_entry = {
        "strategy": STRATEGY_NAME,
        "tp_price": channel["tp_price"],
        "side": channel["side"],
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "source": "detector",
        "qty": float(qty_final),
    }
    save_tp_to_store(symbol, tp_entry, tp_store_path)

    candles = market_data.candles
    last_open_time = int(candles[-1][0]) if candles else int(datetime.utcnow().timestamp())
    client_id = _build_client_id("PCF", symbol, last_open_time)

    try:
        exchange.place_entry_limit(
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
        tp_store_path = Path(env.tp_store_path)

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
            tp_store_path=tp_store_path,
        )


__all__ = ["ParallelChannelFormationStrategy", "STRATEGY_NAME", "run"]
