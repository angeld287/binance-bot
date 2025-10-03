"""Limit take-profit placement based on ATR anchors for breakout dual TF."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from importlib import import_module
from typing import Any, Iterable, Sequence
from uuid import uuid4

from common.utils import sanitize_client_order_id

logger = logging.getLogger("bot.strategy.breakout_dual_tf.tp_atr")


try:  # pragma: no cover - optional dependency in some environments
    from binance.exceptions import BinanceAPIException as APIError  # type: ignore
except Exception:  # pragma: no cover - fallback when Binance library is missing
    try:
        from binance.error import BinanceAPIException as APIError  # type: ignore
    except Exception:  # pragma: no cover - ultimate fallback
        class APIError(Exception):  # type: ignore
            """Fallback error used when Binance API exception is unavailable."""


def _interval_to_ms(interval: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    if not interval:
        return 60_000
    suffix = interval[-1]
    try:
        value = int(interval[:-1])
    except (TypeError, ValueError):
        return 60_000
    return value * units.get(suffix, 60_000)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _get_filters(broker: Any, symbol: str) -> dict[str, Any]:
    if broker is None:
        return {}
    if not hasattr(broker, "get_symbol_filters"):
        return {}
    try:
        return broker.get_symbol_filters(symbol) or {}
    except Exception:  # pragma: no cover - defensive
        return {}


def _round_price(price: float, *, broker: Any, symbol: str, side: str, tick_size: float) -> float:
    if broker is not None and hasattr(broker, "round_price_to_tick"):
        try:
            rounded = broker.round_price_to_tick(symbol, price)
            if rounded:
                price = float(rounded)
        except Exception:  # pragma: no cover - fallback to manual
            pass
    if tick_size > 0:
        if side.upper() == "BUY":
            price = math.floor(price / tick_size) * tick_size
        else:
            price = math.ceil(price / tick_size) * tick_size
    return float(price)


def _round_qty(qty: float, *, broker: Any, symbol: str, step_size: float) -> float:
    if broker is not None and hasattr(broker, "round_qty_to_step"):
        try:
            rounded = broker.round_qty_to_step(symbol, qty)
            if rounded:
                return float(rounded)
        except Exception:  # pragma: no cover - fallback
            pass
    if step_size > 0:
        qty = math.floor(qty / step_size) * step_size
    return float(qty)


def _fetch_candles(
    broker: Any,
    symbol: str,
    timeframe: str,
    *,
    limit: int,
) -> list[list[float]]:
    if broker is None:
        return []

    interval_ms = _interval_to_ms(timeframe)
    lookback_min = max(int(interval_ms / 60_000) * limit, limit)

    fetch_attempts = (
        ("fetch_ohlcv", {"symbol": symbol, "timeframe": timeframe, "limit": limit}),
        (
            "get_klines",
            {"symbol": symbol, "interval": timeframe, "limit": limit, "lookback_min": lookback_min},
        ),
        (
            "klines",
            {"symbol": symbol, "interval": timeframe, "limit": limit, "lookback_min": lookback_min},
        ),
    )

    candles: Sequence[Sequence[Any]] = []
    for attr, kwargs in fetch_attempts:
        if not hasattr(broker, attr):
            continue
        fetcher = getattr(broker, attr)
        try:
            candles = fetcher(**kwargs)
        except TypeError:  # pragma: no cover - positional signatures
            try:
                candles = fetcher(symbol, timeframe, limit)
            except Exception:
                continue
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning("tp_atr_candles_error %s", {"symbol": symbol, "error": str(exc)})
            continue
        if candles:
            break

    normalized: list[list[float]] = []
    for candle in candles or []:
        if len(candle) < 5:
            continue
        open_time = _to_float(candle[0])
        open_price = _to_float(candle[1])
        high = _to_float(candle[2])
        low = _to_float(candle[3])
        close = _to_float(candle[4])
        volume = _to_float(candle[5]) if len(candle) > 5 else 0.0
        normalized.append([open_time, open_price, high, low, close, volume])
    return normalized


def _compute_atr_proxy(candles: Sequence[Sequence[float]], period: int) -> float:
    if not candles:
        return 0.0
    compute = getattr(_compute_atr_proxy, "_core_fn", None)
    if compute is None:
        try:
            core_module = import_module("strategies.breakout_dual_tf.core")
            compute = getattr(core_module, "_compute_atr", None)
        except Exception:  # pragma: no cover - fallback when import fails
            compute = None
        setattr(_compute_atr_proxy, "_core_fn", compute)
    if callable(compute):
        return float(compute(candles, period=period))

    # Fallback simplified ATR (SMA of true range)
    if len(candles) < 2:
        return 0.0
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]
    trs: list[float] = []
    prev_close = closes[0]
    for high, low, close in zip(highs[1:], lows[1:], closes[1:]):
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        prev_close = close
    if not trs:
        return 0.0
    window = trs[-period:]
    if not window:
        return 0.0
    return float(sum(window) / len(window))


def _parse_timestamp(candidate: Any) -> int | None:
    ts = _to_float(candidate)
    if ts <= 0:
        return None
    if ts < 10_000_000_000:  # seconds
        ts *= 1000
    return int(ts)


def _find_fill_timestamp(position: dict[str, Any] | None, broker: Any, symbol: str, position_side: str | None) -> int | None:
    if position is not None:
        for key in ("fillTime", "fill_ts", "updateTime", "update_time", "time", "timestamp", "openTime", "open_time"):
            if key in position:
                ts = _parse_timestamp(position.get(key))
                if ts:
                    return ts

    fetch_methods = (
        "get_position_fill_ts",
        "get_position_fill_timestamp",
        "get_fill_timestamp",
    )
    for attr in fetch_methods:
        if not hasattr(broker, attr):
            continue
        try:
            ts_value = getattr(broker, attr)(symbol, position_side=position_side)
        except TypeError:  # pragma: no cover - positional signature fallback
            ts_value = getattr(broker, attr)(symbol)
        except Exception:  # pragma: no cover - best effort
            continue
        ts = _parse_timestamp(ts_value)
        if ts:
            return ts
    return None


def _select_position_entry(data: Any, position_side: str | None) -> dict[str, Any] | None:
    if data is None:
        return None
    if isinstance(data, dict):
        return data
    if not isinstance(data, Iterable):
        return None
    side_norm = (position_side or "").upper()
    selected: dict[str, Any] | None = None
    for item in data:
        if not isinstance(item, dict):
            continue
        if side_norm:
            pos_side = str(item.get("positionSide") or item.get("position_side") or "").upper()
            if pos_side and pos_side != side_norm:
                continue
        qty = _to_float(item.get("positionAmt") or item.get("position_amount") or 0.0)
        if side_norm and qty == 0.0 and item.get("positionSide"):
            # skip empty hedge entries for explicit side
            continue
        selected = item
        if side_norm:
            break
    return selected


def _get_position(broker: Any, symbol: str, position_side: str | None) -> dict[str, Any] | None:
    if broker is None:
        return None
    for attr in ("get_position", "futures_position_information", "position_information"):
        if not hasattr(broker, attr):
            continue
        try:
            data = getattr(broker, attr)(symbol)
        except Exception:  # pragma: no cover - ignore fetch errors
            continue
        position = _select_position_entry(data, position_side)
        if position:
            return position
    return None


def ensure_tp_limit_reduce_only_by_atr(
    *,
    broker: Any,
    symbol: str,
    position_side: str,
    entry_price: float,
    k_list: Sequence[float] | None,
    atr_period: int,
    timeframe: str,
) -> dict[str, Any]:
    """Ensure limit reduce-only take-profits exist based on ATR multiples.

    Parameters
    ----------
    broker:
        Exchange adapter implementing the :class:`BrokerPort` contract.
    symbol:
        Trading pair for the position.
    position_side:
        Direction of the position (``"LONG"`` or ``"SHORT"``).
    entry_price:
        Average entry price of the position.
    k_list:
        Multipliers applied to the ATR to derive TP prices.
    atr_period:
        Lookback used to compute the ATR.
    timeframe:
        Timeframe used to fetch candles when computing the ATR.

    Returns
    -------
    dict
        Summary containing placement results and computed levels.
    """

    result: dict[str, Any] = {"status": "skipped"}
    if broker is None:
        result["reason"] = "no_broker"
        return result

    position_side_norm = str(position_side or "").upper()
    if position_side_norm not in {"LONG", "SHORT"}:
        result["reason"] = "invalid_side"
        return result

    position = _get_position(broker, symbol, position_side_norm)
    if not position:
        logger.info("tp_atr_skip_no_position %s", {"symbol": symbol, "position_side": position_side_norm})
        result["reason"] = "no_position"
        return result

    pos_qty = _to_float(position.get("positionAmt") or position.get("position_amount") or 0.0)
    if position.get("positionSide"):
        pos_side_raw = str(position.get("positionSide") or "").upper()
        if pos_side_raw and pos_side_raw != position_side_norm:
            logger.info(
                "tp_atr_skip_side_mismatch %s",
                {
                    "symbol": symbol,
                    "expected": position_side_norm,
                    "actual": pos_side_raw,
                },
            )
            result["reason"] = "side_mismatch"
            return result
    else:
        if position_side_norm == "LONG" and pos_qty < 0:
            logger.info(
                "tp_atr_skip_side_mismatch %s",
                {"symbol": symbol, "expected": position_side_norm, "qty": pos_qty},
            )
            result["reason"] = "side_mismatch"
            return result
        if position_side_norm == "SHORT" and pos_qty > 0:
            logger.info(
                "tp_atr_skip_side_mismatch %s",
                {"symbol": symbol, "expected": position_side_norm, "qty": pos_qty},
            )
            result["reason"] = "side_mismatch"
            return result

    qty_abs = abs(pos_qty)
    if qty_abs <= 0:
        logger.info("tp_atr_skip_no_position %s", {"symbol": symbol, "position_side": position_side_norm})
        result["reason"] = "qty_zero"
        return result

    filters = _get_filters(broker, symbol)
    tick_size = _to_float(filters.get("PRICE_FILTER", {}).get("tickSize", 0.0))
    step_size = _to_float(filters.get("LOT_SIZE", {}).get("stepSize", 0.0))

    open_orders: Sequence[dict[str, Any]] = []
    if hasattr(broker, "open_orders"):
        try:
            open_orders = broker.open_orders(symbol) or []
        except Exception:  # pragma: no cover - ignore fetch failures
            open_orders = []

    exit_side = "SELL" if position_side_norm == "LONG" else "BUY"
    pending_reduce_qty = 0.0
    existing_tp_orders: list[dict[str, Any]] = []
    for order in open_orders:
        if not isinstance(order, dict):
            continue
        order_side = str(order.get("side") or "").upper()
        if order_side and order_side != exit_side:
            continue
        pos_side = str(order.get("positionSide") or order.get("position_side") or "").upper()
        if pos_side and pos_side not in {position_side_norm, "BOTH"}:
            continue
        reduce_only = bool(order.get("reduceOnly")) or bool(order.get("reduce_only"))
        if not reduce_only:
            continue
        order_type = str(
            order.get("type")
            or order.get("orderType")
            or order.get("origType")
            or ""
        ).upper()
        if order_type not in {"LIMIT", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT"}:
            continue
        existing_tp_orders.append(order)
        qty_candidate = (
            order.get("origQty")
            or order.get("orig_quantity")
            or order.get("quantity")
            or order.get("qty")
        )
        pending_reduce_qty += abs(_to_float(qty_candidate))

    if existing_tp_orders:
        logger.info(
            "tp_atr_skip_existing %s",
            {
                "symbol": symbol,
                "position_side": position_side_norm,
                "orders": len(existing_tp_orders),
            },
        )
        result["reason"] = "existing_tp"
        return result

    available_qty = max(qty_abs - pending_reduce_qty, 0.0)
    if available_qty <= 0:
        logger.info(
            "tp_atr_skip_qty %s",
            {
                "symbol": symbol,
                "position_side": position_side_norm,
                "available_qty": available_qty,
            },
        )
        result["reason"] = "no_available_qty"
        return result

    fill_ts = _find_fill_timestamp(position, broker, symbol, position_side_norm)
    anchor_used = fill_ts is not None
    if fill_ts is None:
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        fill_ts = now_ms

    hour_ms = 3_600_000
    anchor_open = (fill_ts // hour_ms) * hour_ms
    anchor_close = anchor_open + hour_ms

    fetch_limit = max(atr_period + 5, atr_period * 2)
    candles = _fetch_candles(broker, symbol, timeframe, limit=fetch_limit)
    candles_sorted = sorted(candles, key=lambda c: c[0])
    if anchor_used:
        filtered = [c for c in candles_sorted if c[0] < anchor_close]
    else:
        filtered = candles_sorted
    if len(filtered) < atr_period + 1:
        filtered = candles_sorted[-(atr_period + 1) :]
    atr_value = _compute_atr_proxy(filtered, period=max(atr_period, 1))

    logger.info(
        "tp_atr_anchor %s",
        {
            "symbol": symbol,
            "position_side": position_side_norm,
            "anchor_used": anchor_used,
            "fill_ts": fill_ts,
            "k_open": anchor_open,
            "atr_period": atr_period,
            "atr_value": atr_value,
        },
    )

    if atr_value <= 0 or not k_list:
        result["reason"] = "atr_unavailable"
        result["atr"] = atr_value
        return result

    levels: list[dict[str, Any]] = []
    exit_side_local = exit_side
    for k in k_list:
        k_float = _to_float(k)
        if position_side_norm == "LONG":
            target_price = entry_price + k_float * atr_value
        else:
            target_price = entry_price - k_float * atr_value
        target_price = _round_price(
            target_price,
            broker=broker,
            symbol=symbol,
            side=exit_side_local,
            tick_size=tick_size,
        )
        levels.append({"k": k_float, "price": target_price})

    logger.info("tp_atr_levels %s", {"symbol": symbol, "k_list": list(k_list), "levels": levels})

    qtys: list[float] = []
    remaining = available_qty
    splits = len(levels)
    if splits <= 0:
        result["reason"] = "no_levels"
        return result

    base_qty = remaining / splits if splits else remaining
    for index, level in enumerate(levels, start=1):
        if index < splits:
            qty = _round_qty(base_qty, broker=broker, symbol=symbol, step_size=step_size)
            qty = max(qty, 0.0)
            remaining = max(remaining - qty, 0.0)
        else:
            qty = max(remaining, 0.0)
            qty = _round_qty(qty, broker=broker, symbol=symbol, step_size=step_size)
        qtys.append(qty)

    qtys = [q for q in qtys if q > 0]
    if not qtys:
        result["reason"] = "qty_rounding"
        return result

    orders_placed: list[dict[str, Any]] = []
    for idx, (level, qty) in enumerate(zip(levels, qtys), start=1):
        price = level["price"]
        if price <= 0 or qty <= 0:
            continue
        cid = sanitize_client_order_id(
            f"bdtf-tp-{position_side_norm.lower()}-{idx}-{uuid4().hex[:6]}"
        )
        try:
            if hasattr(broker, "place_tp_reduce_only"):
                response = broker.place_tp_reduce_only(symbol, exit_side_local, price, qty, cid)
            else:
                response = broker.place_entry_limit(
                    symbol,
                    exit_side_local,
                    price,
                    qty,
                    cid,
                    timeInForce="GTC",
                    reduceOnly=True,
                )
        except APIError as exc:  # pragma: no cover - network error handling
            logger.warning(
                "tp_atr_error %s",
                {"symbol": symbol, "code": getattr(exc, "code", None), "msg": str(exc)},
            )
            continue
        except Exception as exc:  # pragma: no cover - generic defensive guard
            logger.warning(
                "tp_atr_error %s",
                {"symbol": symbol, "code": None, "msg": str(exc)},
            )
            continue

        logger.info(
            "tp_atr_placed %s",
            {
                "symbol": symbol,
                "position_side": position_side_norm,
                "price": price,
                "qty": qty,
                "reduceOnly": True,
                "clientOrderId": cid,
            },
        )
        orders_placed.append({"price": price, "qty": qty, "clientOrderId": cid, "response": response})

    if orders_placed:
        result.update(
            {
                "status": "placed",
                "levels": levels,
                "orders": orders_placed,
                "atr": atr_value,
                "anchor_used": anchor_used,
            }
        )
    else:
        result.update({"status": "skipped", "reason": "placement_failed", "levels": levels, "atr": atr_value})
    return result


__all__ = ["ensure_tp_limit_reduce_only_by_atr"]
