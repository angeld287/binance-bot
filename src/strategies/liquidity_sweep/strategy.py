"""Liquidity sweep strategy skeleton.

This module provides a minimal implementation of the Liquidity Sweep
strategy.  It intentionally keeps side effects out of module import time and
splits pure computations from IO operations.  The goal is to serve as a
starting point for a more complete implementation.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from math import ceil, floor
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo
import json
import logging
from decimal import Decimal


TIMEOUT_NO_FILL_MIN = 5  # minutes after NY open to keep processing ticks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helper functions


def compute_levels(candles_m1: Sequence[Sequence[float]], *args: Any, **kwargs: Any) -> Mapping[str, float]:
    """Compute support/resistance levels from recent 1m ``candles_m1``.

    The implementation follows the requirements described in ``Paso 7`` of the
    kata.  It performs the following steps:

    * Compute ATR for the 1m series and for the series aggregated to 15m.
    * Detect fractal highs/lows (two candles on each side) in the last hour of
      data and use them as support/resistance candidates.
    * Score each candidate based on the number of touches within a small buffer,
      confluence with the last 15m candle high/low and proximity to the current
      price.
    * Select the best support below and resistance above the current price,
      ensuring the resulting range roughly matches the 15m ATR.

    Parameters
    ----------
    candles_m1:
        Iterable of OHLCV candles in the typical Binance format
        ``[timestamp, open, high, low, close, volume]``.

    Returns
    -------
    Mapping[str, float]
        Dictionary with keys ``S``, ``R``, ``atr1m``, ``atr15m``,
        ``microbuffer`` and ``buffer_sl``.  Values are aligned to the provided
        ``TICK_SIZE`` if available in ``settings``.
    """

    settings = kwargs.get("settings")
    if not candles_m1:
        return {}

    # ------------------------------------------------------------------
    # Helpers
    def _atr(candles: Sequence[Sequence[float]]) -> float:
        trs: list[float] = []
        prev_close = float(candles[0][4])
        for _, _, high, low, close, *_ in candles:
            h = float(high)
            l = float(low)
            c = float(close)
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            trs.append(tr)
            prev_close = c
        return sum(trs) / len(trs) if trs else 0.0

    def _resample_15m(candles: Sequence[Sequence[float]]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(candles), 15):
            chunk = candles[i : i + 15]
            if not chunk:
                continue
            high = max(float(c[2]) for c in chunk)
            low = min(float(c[3]) for c in chunk)
            close = float(chunk[-1][4])
            ts = float(chunk[0][0])
            out.append([ts, 0.0, high, low, close, 0.0])
        return out

    def _round_to_tick(value: float) -> float:
        tick = getattr(settings, "TICK_SIZE", 0.0) if settings else 0.0
        if tick:
            return round(value / tick) * tick
        return value

    # ------------------------------------------------------------------
    # ATR calculations
    atr1m = _atr(candles_m1)
    candles_15m = _resample_15m(candles_m1)
    atr15m = _atr(candles_15m)

    # Current price from last close
    price_now = float(candles_m1[-1][4])

    # Buffers
    micro_pct = float(getattr(settings, "MICROBUFFER_PCT_MIN", 0.0002)) if settings else 0.0002
    micro_mult = float(getattr(settings, "MICROBUFFER_ATR1M_MULT", 0.25)) if settings else 0.25
    sl_pct = float(getattr(settings, "BUFFER_SL_PCT_MIN", 0.0005)) if settings else 0.0005
    sl_mult = float(getattr(settings, "BUFFER_SL_ATR1M_MULT", 0.5)) if settings else 0.5

    microbuffer = max(micro_pct * price_now, micro_mult * atr1m)
    buffer_sl = max(sl_pct * price_now, sl_mult * atr1m)

    # ------------------------------------------------------------------
    # Detect fractals as level candidates
    highs = [float(c[2]) for c in candles_m1]
    lows = [float(c[3]) for c in candles_m1]

    candidates: list[dict[str, float]] = []
    for i in range(2, len(candles_m1) - 2):
        h = highs[i]
        l = lows[i]
        if h > max(highs[i - 2 : i]) and h > max(highs[i + 1 : i + 3]):
            candidates.append({"type": "R", "price": h})
        if l < min(lows[i - 2 : i]) and l < min(lows[i + 1 : i + 3]):
            candidates.append({"type": "S", "price": l})

    if not candidates:
        return {
            "S": _round_to_tick(price_now - microbuffer),
            "R": _round_to_tick(price_now + microbuffer),
            "atr1m": atr1m,
            "atr15m": atr15m,
            "microbuffer": _round_to_tick(microbuffer),
            "buffer_sl": _round_to_tick(buffer_sl),
        }

    last15_high = float(candles_15m[-1][2]) if candles_15m else highs[-1]
    last15_low = float(candles_15m[-1][3]) if candles_15m else lows[-1]

    def _touches(price: float, side: str) -> int:
        count = 0
        buf = microbuffer
        if side == "R":
            for h in highs:
                if abs(h - price) <= buf:
                    count += 1
        else:
            for l in lows:
                if abs(l - price) <= buf:
                    count += 1
        return count

    def _score(price: float, side: str) -> float:
        touches = _touches(price, side)
        confluence = 0
        if side == "R" and abs(price - last15_high) <= microbuffer:
            confluence = 1
        if side == "S" and abs(price - last15_low) <= microbuffer:
            confluence = 1
        proximity = max(0.0, 1 - abs(price_now - price) / price_now)
        return touches + confluence + proximity

    scored = [dict(c, score=_score(c["price"], c["type"])) for c in candidates]

    supports = [c for c in scored if c["price"] < price_now]
    resistances = [c for c in scored if c["price"] > price_now]

    supports.sort(key=lambda c: c["score"], reverse=True)
    resistances.sort(key=lambda c: c["score"], reverse=True)

    atr_min = 0.5 * atr15m
    atr_max = 1.5 * atr15m

    S = supports[0]["price"] if supports else price_now - microbuffer
    R = resistances[0]["price"] if resistances else price_now + microbuffer
    best_score = -1.0

    for s in supports or [{"price": S, "score": 0.0}]:
        for r in resistances or [{"price": R, "score": 0.0}]:
            rng = r["price"] - s["price"]
            if atr_min <= rng <= atr_max:
                sc = s["score"] + r["score"]
                if sc > best_score:
                    best_score = sc
                    S, R = s["price"], r["price"]

    S = _round_to_tick(S)
    R = _round_to_tick(R)
    microbuffer = _round_to_tick(microbuffer)
    buffer_sl = _round_to_tick(buffer_sl)

    return {
        "S": S,
        "R": R,
        "atr1m": atr1m,
        "atr15m": atr15m,
        "microbuffer": microbuffer,
        "buffer_sl": buffer_sl,
    }

def _round_to_tick(value: float, tick: float) -> float:
    """Round ``value`` to the closest ``tick`` size."""

    if tick:
        return round(value / tick) * tick
    return value


def build_entry_prices(levels: Mapping[str, float], *args: Any, **kwargs: Any) -> Mapping[str, float]:
    """Derive entry prices from previously computed ``levels``.

    Parameters
    ----------
    levels:
        Mapping containing at least ``S`` (support), ``R`` (resistance) and
        ``microbuffer`` values.

    Returns
    -------
    Mapping[str, float]
        Dictionary with ``buy_px`` and ``sell_px`` aligned to the provided tick
        size (``settings.TICK_SIZE``).

    Notes
    -----
    This function is pure and performs no IO.
    """

    settings = kwargs.get("settings")
    tick = float(getattr(settings, "TICK_SIZE", 0.0)) if settings else 0.0

    try:
        S = float(levels["S"])
        R = float(levels["R"])
        micro = float(levels["microbuffer"])
    except Exception:
        return {}

    buy_px = _round_to_tick(S + micro, tick)
    sell_px = _round_to_tick(R - micro, tick)
    return {"buy_px": buy_px, "sell_px": sell_px}


def build_bracket(
    side: str,
    entry: float,
    S: float,
    R: float,
    microbuffer: float,
    buffer_sl: float,
    atr1m: float,
    tp_policy: str | None = None,
    *args: Any,
    **kwargs: Any,
) -> Mapping[str, float]:
    """Construct stop-loss and take-profit levels for a position.

    Parameters mirror the description in ``Paso 8`` of the kata.  ``side``
    accepts ``"BUY"``/``"LONG"`` for long positions and ``"SELL"``/``"SHORT"``
    for shorts.

    The function is pure and performs no IO.
    """

    settings = kwargs.get("settings")
    tick = float(getattr(settings, "TICK_SIZE", 0.0)) if settings else 0.0

    is_long = str(side).upper() in {"BUY", "LONG"}

    if is_long:
        sl = S - buffer_sl
        tp_struct = R - microbuffer
        risk = entry - sl
        reward = tp_struct - entry
        if risk > 0 and reward / risk >= 1.2:
            tp = tp_struct
        else:
            tp = entry + 1.8 * abs(entry - sl)
    else:
        sl = R + buffer_sl
        tp_struct = S + microbuffer
        risk = sl - entry
        reward = entry - tp_struct
        if risk > 0 and reward / risk >= 1.2:
            tp = tp_struct
        else:
            tp = entry - 1.8 * abs(entry - sl)

    sl = _round_to_tick(sl, tick)
    tp = _round_to_tick(tp, tick)
    return {"sl": sl, "tp": tp}


# ---------------------------------------------------------------------------
# Internal IO helpers


def _has_position_or_active_orders(exchange: Any, symbol: str) -> tuple[bool, list[Any]]:
    """Return ``True`` if there's an open position or active order for ``symbol``.

    The function attempts to reuse the same inspection logic employed during
    the tick phase: it queries open orders and checks their ``status`` and also
    inspects the current position size when the broker exposes a position
    endpoint.  A non-zero ``positionAmt`` or any order with a ``NEW``,
    ``PARTIALLY_FILLED`` or ``PENDING_NEW`` status triggers a ``True`` result.
    """

    open_orders: list[Any] = []
    try:
        open_orders = exchange.open_orders(symbol)
    except Exception:
        open_orders = []

    for o in open_orders:
        st = str(o.get("status", "")).upper()
        if st in {"NEW", "PARTIALLY_FILLED", "PENDING_NEW"}:
            return True, open_orders

    qty = 0.0
    try:
        if hasattr(exchange, "position_information"):
            info = exchange.position_information(symbol)
        elif hasattr(exchange, "futures_position_information"):
            info = exchange.futures_position_information(symbol)
        elif hasattr(exchange, "get_position"):
            info = exchange.get_position(symbol)
        else:
            info = None

        if info is not None:
            if isinstance(info, list):
                info = info[0] if info else {}
            qty = float(info.get("positionAmt", 0.0))
    except Exception:
        qty = 0.0

    if abs(qty) > 0:
        return True, open_orders

    return False, open_orders


def do_preopen(exchange: Any, market_data: Any, symbol: str, settings: Any) -> dict:
    """Perform pre-open IO actions.

    The function fetches recent candle data to compute support and resistance
    levels and places two idempotent LIMIT orders around those levels.  Existing
    orders are updated only if the new price differs by more than one tick.

    Parameters
    ----------
    exchange:
        Trading adapter used to inspect and manage orders.  It must expose
        ``open_orders`` to query current orders, ``place_entry_limit`` and
        ``cancel_order`` to manage them as well as helpers
        ``get_symbol_filters`` and ``round_price_to_tick``.
    market_data:
        Data provider exposing ``fetch_ohlcv`` or ``get_klines`` to retrieve
        candle information.
    symbol:
        Market symbol, e.g. ``"BTCUSDT"``.
    settings:
        Configuration container providing at least ``MAX_LOOKBACK_MIN``.

    Returns
    -------
    dict
        Summary of the pre-open step containing the computed prices and the
        ``trade_id`` used for idempotency.
    """

    lookback = getattr(settings, "MAX_LOOKBACK_MIN", 60)

    should_skip, open_orders = _has_position_or_active_orders(exchange, symbol)
    if should_skip:
        logger.info(json.dumps({"action": "preopen", "reason": "existing_position_or_orders"}))
        return {"status": "preopen_skipped"}

    # ------------------------------------------------------------------
    # Fetch 1m candles
    if hasattr(market_data, "fetch_ohlcv"):
        candles = market_data.fetch_ohlcv(symbol, timeframe="1m", limit=lookback)
    else:  # pragma: no cover - fallback for providers exposing get_klines only
        candles = market_data.get_klines(symbol=symbol, interval="1m", lookback_min=lookback)

    levels: Mapping[str, float] = compute_levels(candles, settings=settings) or {}
    S = float(levels.get("S", 0.0))
    R = float(levels.get("R", 0.0))
    microbuffer = float(levels.get("microbuffer", 0.0))
    buffer_sl = float(levels.get("buffer_sl", 0.0))
    atr1m = float(levels.get("atr1m", 0.0))

    buy_px = S + microbuffer
    sell_px = R - microbuffer

    # Determine tick size to align prices and to evaluate changes
    tick = 0.0
    try:
        filters = exchange.get_symbol_filters(symbol)
        tick = float(filters["PRICE_FILTER"]["tickSize"])
    except Exception:  # pragma: no cover - helper not available
        pass

    if hasattr(exchange, "round_price_to_tick"):
        buy_px = exchange.round_price_to_tick(symbol, buy_px)
        sell_px = exchange.round_price_to_tick(symbol, sell_px)
    elif tick:  # pragma: no cover - manual rounding when helper missing
        buy_px = floor(buy_px / tick) * tick
        sell_px = floor(sell_px / tick) * tick

    # ------------------------------------------------------------------
    # Build client order ids
    ny_now = datetime.now(tz=ZoneInfo("America/New_York"))
    trade_id = f"{symbol}-{ny_now.strftime('%Y%m%d')}-NY"
    cid_buy = f"{trade_id}:pre:buy"
    cid_sell = f"{trade_id}:pre:sell"

    # ------------------------------------------------------------------
    # Determine quantity meeting exchange minimums
    filters = {}
    try:
        filters = exchange.get_symbol_filters(symbol)
    except Exception:  # pragma: no cover - helper not available
        filters = {}

    lot = filters.get("LOT_SIZE", {})
    min_qty = float(lot.get("minQty", 0.0))
    step = float(lot.get("stepSize", 0.0))
    min_notional = float(
        filters.get("MIN_NOTIONAL", {}).get("notional")
        or filters.get("MIN_NOTIONAL", {}).get("minNotional", 0.0)
    )

    def _ceil_step(x: float) -> float:
        return ceil(x / step) * step if step else x

    def _floor_step(x: float) -> float:
        return floor(x / step) * step if step else x

    risk_notional = float(getattr(settings, "RISK_NOTIONAL_USDT", 0.0) or 0.0)

    if risk_notional > 0:
        price = min(buy_px, sell_px)
        qty_min = max(min_qty, _ceil_step(min_notional / price if price else 0.0))
        qty_budget = _floor_step(risk_notional / price if price else 0.0)
    else:
        price = 0.0
        try:
            if market_data and hasattr(market_data, "get_price"):
                price = float(market_data.get_price(symbol))
        except Exception:
            price = 0.0
        if not price:
            price = (buy_px + sell_px) / 2

        qty_min = max(min_qty, _ceil_step(min_notional / price if price else 0.0))

        balance = 0.0
        try:
            balance = float(exchange.get_available_balance_usdt())
        except Exception:
            balance = 0.0
        risk_pct = float(getattr(settings, "RISK_PCT", 0.01))
        br_long = build_bracket(
            "BUY",
            buy_px,
            S,
            R,
            microbuffer,
            buffer_sl,
            atr1m,
            settings=settings,
        )
        br_short = build_bracket(
            "SELL",
            sell_px,
            S,
            R,
            microbuffer,
            buffer_sl,
            atr1m,
            settings=settings,
        )
        sl_long = float(br_long.get("sl", 0.0))
        sl_short = float(br_short.get("sl", 0.0))
        risk = max(abs(buy_px - sl_long), abs(sl_short - sell_px))
        qty_from_risk = (risk_pct * balance) / risk if risk else 0.0
        if hasattr(exchange, "round_qty_to_step"):
            qty_budget = exchange.round_qty_to_step(symbol, qty_from_risk)
        else:
            qty_budget = _floor_step(qty_from_risk)
    if qty_budget < qty_min:
        logger.info(json.dumps({
            "action": "preopen",
            "reason": "budget_below_min_using_qty_min",
            "qty_min": qty_min,
            "qty_budget": qty_budget
        }))
        qty_final = qty_min
        # (no returns aquí; seguir)

    qty_final = max(qty_budget, qty_min)

    def _ensure_limit(side: str, price: float, cid: str) -> None:
        existing = next((o for o in open_orders if o.get("clientOrderId") == cid), None)
        if existing:
            try:
                current_price = float(existing.get("price", 0.0))
            except Exception:  # pragma: no cover
                current_price = 0.0
            if tick and abs(current_price - price) <= tick:
                return
            exchange.cancel_order(symbol, clientOrderId=cid)
        logger.info(
            json.dumps(
                {
                    "sym": symbol,
                    "price": price,
                    "qty_min": qty_min,
                    "qty_budget": qty_budget,
                    "qty_final": qty_final,
                }
            )
        )
        # Normalize quantity and price according to exchange filters
        if hasattr(exchange, "round_qty_to_step"):
            qty_norm = exchange.round_qty_to_step(symbol, qty_final)
        else:
            qty_norm = _floor_step(qty_final)

        if hasattr(exchange, "round_price_to_tick"):
            price_norm = exchange.round_price_to_tick(symbol, price)
        elif tick:
            price_norm = floor(price / tick) * tick
        else:
            price_norm = price

        step_decimals = max(0, -Decimal(str(step)).as_tuple().exponent) if step else 0
        tick_decimals = max(0, -Decimal(str(tick)).as_tuple().exponent) if tick else 0

        qty_str = (
            f"{qty_norm:.{step_decimals}f}" if step_decimals else f"{int(qty_norm)}"
        )
        price_str = (
            f"{price_norm:.{tick_decimals}f}" if tick_decimals else f"{int(price_norm)}"
        )

        logger.info(
            json.dumps(
                {
                    "sym": symbol,
                    "price_after_tick": price_str,
                    "qty_after_step": qty_str,
                }
            )
        )

        exchange.place_entry_limit(
            symbol,
            side,
            price_str,
            qty_str,
            cid,
            timeInForce="GTC",
        )

    _ensure_limit("BUY", buy_px, cid_buy)
    _ensure_limit("SELL", sell_px, cid_sell)
    logger.info(
        json.dumps(
            {
                "action": "preopen",
                "trade_id": trade_id,
                "side": None,
                "prices": {"entry": None, "sl": None, "tp": None},
                "clientOrderIds": {"buy": cid_buy, "sell": cid_sell},
                "status": "preopen_ok",
                "reason": None,
            }
        )
    )

    return {
        "status": "preopen_ok",
        "trade_id": trade_id,
        "S": S,
        "R": R,
        "buy_px": buy_px,
        "sell_px": sell_px,
    }


def do_tick(
    exchange: Any,
    symbol: str,
    settings: Any,
    market_data: Any | None = None,
    event: Any | None = None,
) -> dict:
    """Handle the tick phase after market open.

    The function inspects pre-open orders to detect fills and, when one is
    executed, cancels the opposite order and places a protective bracket
    (stop-loss and take-profit) using ``reduceOnly`` instructions.

    Parameters
    ----------
    exchange:
        Trading adaptor exposing ``open_orders``, ``get_order``,
        ``cancel_order``, ``place_stop_reduce_only`` and
        ``place_tp_reduce_only`` helpers as well as balance/rounding
        utilities.  Entry orders are submitted via ``place_entry_limit`` or
        ``place_entry_market``.
    symbol:
        Market symbol, e.g. ``"BTCUSDT"``.
    settings:
        Configuration container providing at least ``RISK_PCT``.
    market_data:
        Data provider, currently unused but kept for interface symmetry.
    event:
        Optional mapping/object carrying ``open_at_epoch_ms`` and level
        information (``S``, ``R``, ``microbuffer`` and ``buffer_sl``).

    Returns
    -------
    dict
        Summary of the tick processing.  Possible values include ``waiting``
        when pre-orders are still active, ``timeout`` when they expired and
        ``bracket_placed`` once the protective orders have been submitted.
    """

    # Basic fallback for legacy calls using the old signature.  ``run`` may
    # still call :func:`do_tick` with ``now_utc`` and ``event`` only; in that
    # scenario we simply return an empty result to avoid unexpected failures.
    if settings is None or not isinstance(symbol, str):  # pragma: no cover - legacy
        return {}

    # ------------------------------------------------------------------
    # Build identifiers and timing helpers
    tz_ny = ZoneInfo("America/New_York")
    ny_now = datetime.now(tz=tz_ny)

    open_at_epoch_ms = None
    if isinstance(event, Mapping):
        open_at_epoch_ms = event.get("open_at_epoch_ms")
    else:
        open_at_epoch_ms = getattr(event, "open_at_epoch_ms", None)

    if open_at_epoch_ms is not None:
        open_at_ny = datetime.fromtimestamp(open_at_epoch_ms / 1000, tz=ZoneInfo("UTC")).astimezone(tz_ny)
    else:
        open_at_ny = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)

    trade_id = f"{symbol}-{open_at_ny.strftime('%Y%m%d')}-NY"
    cid_buy = f"{trade_id}:pre:buy"
    cid_sell = f"{trade_id}:pre:sell"
    cid_sl = f"{trade_id}:sl"
    cid_tp = f"{trade_id}:tp"

    def _log(
        status: str,
        *,
        reason: str | None = None,
        side: str | None = None,
        entry: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        cids: Mapping[str, str] | None = None,
    ) -> None:
        logger.info(
            json.dumps(
                {
                    "action": "tick",
                    "trade_id": trade_id,
                    "side": side,
                    "prices": {"entry": entry, "sl": sl, "tp": tp},
                    "clientOrderIds": cids or {"buy": cid_buy, "sell": cid_sell},
                    "status": status,
                    "reason": reason,
                }
            )
        )

    # ------------------------------------------------------------------
    # Inspect current open orders
    open_orders = exchange.open_orders(symbol)
    pre_buy = next((o for o in open_orders if o.get("clientOrderId") == cid_buy), None)
    pre_sell = next((o for o in open_orders if o.get("clientOrderId") == cid_sell), None)

    if pre_buy and pre_sell:
        # Both pre-orders still alive
        if ny_now > open_at_ny + timedelta(minutes=TIMEOUT_NO_FILL_MIN):
            try:
                exchange.cancel_order(symbol, clientOrderId=cid_buy)
            except Exception:
                pass
            try:
                exchange.cancel_order(symbol, clientOrderId=cid_sell)
            except Exception:
                pass
            _log("timeout")
            return {"status": "done", "reason": "timeout"}
        _log("waiting")
        return {"status": "waiting"}

    # Helper to fetch a single order
    def _fetch(cid: str) -> Mapping[str, Any] | None:
        try:
            return exchange.get_order(symbol, clientOrderId=cid)
        except Exception:  # pragma: no cover - network failures
            return None

    # Helper executing the bracket placement once a side filled
    def _place_bracket(filled_cid: str, info: Mapping[str, Any]) -> dict:
        is_long = filled_cid == cid_buy
        side = "LONG" if is_long else "SHORT"

        entry = 0.0
        try:
            entry = float(info.get("avgPrice") or info.get("price") or 0.0)
        except Exception:
            entry = 0.0

        # Cancel opposite order if still active
        other_cid = cid_sell if is_long else cid_buy
        try:
            exchange.cancel_order(symbol, clientOrderId=other_cid)
        except Exception:
            pass

        # Extract levels from the event
        def _ev(name: str) -> float | None:
            if isinstance(event, Mapping):
                val = event.get(name)
            else:
                val = getattr(event, name, None)
            try:
                return float(val) if val is not None else None
            except Exception:  # pragma: no cover - casting failures
                return None

        S = _ev("S")
        R = _ev("R")
        micro = _ev("microbuffer") or 0.0
        buffer_sl = _ev("buffer_sl") or 0.0
        atr1m = _ev("atr1m") or 0.0
        tp_policy = None
        if isinstance(event, Mapping):
            tp_policy = event.get("tp_policy")
        else:
            tp_policy = getattr(event, "tp_policy", None)

        # Attempt to derive missing levels from ``market_data``
        if S is None or R is None:
            candles = None
            lookback = getattr(settings, "MAX_LOOKBACK_MIN", 60)
            try:
                if market_data and hasattr(market_data, "fetch_ohlcv"):
                    candles = market_data.fetch_ohlcv(symbol, timeframe="1m", limit=lookback)
                elif market_data and hasattr(market_data, "get_klines"):
                    candles = market_data.get_klines(symbol=symbol, interval="1m", lookback_min=lookback)
            except Exception:  # pragma: no cover - network failures
                candles = None
            if candles:
                levels = compute_levels(candles, settings=settings) or {}
                S = levels.get("S", S)
                R = levels.get("R", R)
                micro = levels.get("microbuffer", micro)
                buffer_sl = levels.get("buffer_sl", buffer_sl)
                atr1m = levels.get("atr1m", atr1m)

        if S is None or R is None:
            logger.info(json.dumps({"action": "skip_sl_missing_S/R"}))
            return {"status": "done", "reason": "skip_sl_missing_S/R"}

        bracket = build_bracket(
            "BUY" if is_long else "SELL",
            entry,
            S,
            R,
            micro,
            buffer_sl,
            atr1m,
            tp_policy=tp_policy,
            settings=settings,
        )
        tp = float(bracket.get("tp", 0.0))

        # ------------------------------------------------------------------
        # Buffer rules and guards
        buffer_raw = float(buffer_sl)
        max_pct = float(getattr(settings, "MAX_SL_BUFFER_PCT", 0.8))
        nivel = S if is_long else R
        buffer_eff = min(buffer_raw, abs(nivel) * max_pct) if nivel is not None else buffer_raw
        sl_prev = (S - buffer_eff) if is_long else (R + buffer_eff)
        rule_applied = "raw"
        if buffer_eff != buffer_raw:
            rule_applied = "cap"
        if sl_prev <= 0:
            dist_pct = float(getattr(settings, "MAX_SL_DIST_PCT", 0.02))
            sl_prev = entry * (1 - dist_pct if is_long else 1 + dist_pct)
            rule_applied = "dist_fallback"
        logger.info(
            json.dumps(
                {
                    "action": "sl_buffer_rules",
                    "buffer_raw": buffer_raw,
                    "buffer_eff": buffer_eff,
                    "rule_applied": rule_applied,
                }
            )
        )

        # Coherence guards before touching exchange
        if sl_prev is None or sl_prev <= 0:
            logger.info(json.dumps({"action": "skip_sl_invalid"}))
            return {"status": "done", "reason": "skip_sl_invalid"}
        if is_long and not sl_prev < entry:
            logger.info(json.dumps({"action": "skip_sl_side_mismatch"}))
            return {"status": "done", "reason": "skip_sl_side_mismatch"}
        if not is_long and not sl_prev > entry:
            logger.info(json.dumps({"action": "skip_sl_side_mismatch"}))
            return {"status": "done", "reason": "skip_sl_side_mismatch"}

        # ``sl_prev`` is the stop before precision snapping; reuse for risk calcs
        sl = sl_prev
        # Normalize symbol before loading filters to avoid precision errors
        if hasattr(exchange, "normalize_symbol"):
            try:
                symbol_n = exchange.normalize_symbol(symbol)
            except Exception:
                symbol_n = symbol.replace("/", "")
        else:
            symbol_n = symbol.replace("/", "") if "/" in symbol else symbol

        try:
            filters = exchange.get_symbol_filters(symbol_n)
        except Exception:  # pragma: no cover - helper not available
            filters = {}
        lot = filters.get("LOT_SIZE", {})
        min_qty = float(lot.get("minQty", 0.0))
        step = float(lot.get("stepSize", 0.0))
        price_filter = filters.get("PRICE_FILTER", {})
        tick = float(price_filter.get("tickSize", 0.0))
        min_notional = float(
            filters.get("MIN_NOTIONAL", {}).get("notional")
            or filters.get("MIN_NOTIONAL", {}).get("minNotional", 0.0)
        )

        def _ceil_step(x: float) -> float:
            return ceil(x / step) * step if step else x

        def _floor_step(x: float) -> float:
            return floor(x / step) * step if step else x

        qty_min = max(min_qty, _ceil_step(min_notional / entry if entry else 0.0))

        risk_notional = float(getattr(settings, "RISK_NOTIONAL_USDT", 0.0) or 0.0)
        if risk_notional > 0 and entry:
            qty_budget = _floor_step(risk_notional / entry)
        else:
            balance = 0.0
            try:
                balance = float(exchange.get_available_balance_usdt())
            except Exception:
                balance = 0.0
            risk_pct = float(getattr(settings, "RISK_PCT", 0.01))
            qty_from_risk = 0.0
            if sl and entry:
                diff = abs(entry - sl)
                qty_from_risk = (risk_pct * balance) / diff if diff else 0.0
            if hasattr(exchange, "round_qty_to_step"):
                qty_budget = exchange.round_qty_to_step(symbol_n, qty_from_risk)
            else:
                qty_budget = _floor_step(qty_from_risk)

        if qty_budget < qty_min:
            logger.info(json.dumps({
                "action": "tick",
                "reason": "budget_below_min_using_qty_min",
                "qty_min": qty_min,
                "qty_budget": qty_budget
            }))
            qty_final = qty_min

        qty_final = max(qty_budget, qty_min)

        # Snap stop to tick directionally and quantity to step
        qty_prev = qty_final
        if tick:
            if is_long:
                sl_final = floor(sl_prev / tick) * tick
            else:
                sl_final = ceil(sl_prev / tick) * tick
        else:
            sl_final = sl_prev
        if sl_final <= 0:
            logger.info(json.dumps({"action": "skip_sl_after_snap_zero"}))
            return {"status": "done", "reason": "skip_sl_after_snap_zero"}
        if hasattr(exchange, "round_qty_to_step"):
            qty_final = exchange.round_qty_to_step(symbol_n, qty_prev)
        else:
            qty_final = qty_prev

        logger.info(
            json.dumps(
                {
                    "action": "sl_precision",
                    "symbol": symbol_n,
                    "tickSize": tick,
                    "stepSize": step,
                    "sl_prev": sl_prev,
                    "sl_final": sl_final,
                    "qty_prev": qty_prev,
                    "qty_final": qty_final,
                }
            )
        )

        exit_side = "SELL" if is_long else "BUY"

        def _place_sl(sym: str, stop_price: float, qty: float) -> None:
            working_type = getattr(settings, "SL_WORKING_TYPE", "MARK_PRICE")
            if hasattr(exchange, "place_stop_market"):
                exchange.place_stop_market(
                    sym,
                    exit_side,
                    stopPrice=stop_price,
                    closePosition=True,
                    workingType=working_type,
                    clientOrderId=cid_sl,
                )
            elif hasattr(exchange, "place_stop_reduce_only"):
                exchange.place_stop_reduce_only(
                    sym,
                    exit_side,
                    stopPrice=stop_price,
                    qty=qty,
                    clientOrderId=cid_sl,
                )
            elif hasattr(exchange, "place_stop"):
                exchange.place_stop(
                    sym,
                    exit_side,
                    stopPrice=stop_price,
                    qty=qty,
                    clientOrderId=cid_sl,
                    reduceOnly=True,
                )
            else:
                err = "exchange adapter lacks place_stop_market/place_stop_reduce_only/place_stop"
                logger.error(err)
                raise AttributeError(err)

        try:
            _place_sl(symbol_n, sl_final, qty_final)
        except Exception as exc:
            logger.error("Failed to place SL order: %s", exc)
            raise

        if hasattr(exchange, "place_tp_reduce_only"):
            try:
                exchange.place_tp_reduce_only(symbol, exit_side, tp, qty_final, cid_tp)
            except Exception as exc:
                logger.error("Failed to place TP order: %s", exc)
                raise
        elif hasattr(exchange, "place_entry_limit"):
            try:  # pragma: no cover - generic fallback
                exchange.place_entry_limit(
                    symbol,
                    exit_side,
                    tp,
                    qty_final,
                    cid_tp,
                    timeInForce="GTC",
                    reduceOnly=True,
                )
            except Exception as exc:
                logger.error("Failed to place TP order: %s", exc)
                raise
        else:
            err = "exchange adapter lacks place_tp_reduce_only/place_entry_limit"
            logger.error(err)
            raise AttributeError(err)
        _log(
            "bracket_placed",
            side=side,
            entry=entry,
            sl=sl,
            tp=tp,
            cids={"entry": filled_cid, "sl": cid_sl, "tp": cid_tp},
        )

        return {
            "status": "done",
            "reason": "bracket_placed",
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
        }

    # ------------------------------------------------------------------
    # Cases depending on which orders are visible
    if pre_buy and not pre_sell:
        info = _fetch(cid_sell)
        if info and info.get("status") == "FILLED":
            return _place_bracket(cid_sell, info)
        if info and info.get("status") in {"CANCELED", "EXPIRED"}:
            _log("preorder_cancelled")
            return {"status": "done", "reason": "preorder_cancelled"}
        _log("waiting")
        return {"status": "waiting"}

    if pre_sell and not pre_buy:
        info = _fetch(cid_buy)
        if info and info.get("status") == "FILLED":
            return _place_bracket(cid_buy, info)
        if info and info.get("status") in {"CANCELED", "EXPIRED"}:
            _log("preorder_cancelled")
            return {"status": "done", "reason": "preorder_cancelled"}
        _log("waiting")
        return {"status": "waiting"}

    # None visible: query both
    info_buy = _fetch(cid_buy)
    info_sell = _fetch(cid_sell)
    if info_buy and info_buy.get("status") == "FILLED":
        return _place_bracket(cid_buy, info_buy)
    if info_sell and info_sell.get("status") == "FILLED":
        return _place_bracket(cid_sell, info_sell)

    if (info_buy and info_buy.get("status") in {"CANCELED", "EXPIRED"}) or (
        info_sell and info_sell.get("status") in {"CANCELED", "EXPIRED"}
    ):
        _log("preorder_cancelled")
        return {"status": "done", "reason": "preorder_cancelled"}

    _log("waiting")
    return {"status": "waiting"}


# ---------------------------------------------------------------------------
# Strategy


class LiquiditySweepStrategy:
    """High level strategy coordinator."""

    def run(
        self,
        exchange: Any,
        market_data: Any | None = None,
        settings: Any | None = None,
        now_utc: datetime | None = None,
        event: Any | None = None,
    ) -> dict:
        """Execute the strategy and return a result dictionary."""
        utc_now = now_utc or datetime.now(tz=ZoneInfo("UTC"))
        ny_now = utc_now.astimezone(ZoneInfo("America/New_York"))

        open_at_epoch_ms = None
        force_phase = None
        symbol = getattr(settings, "SYMBOL", None)
        if isinstance(event, Mapping):
            open_at_epoch_ms = event.get("open_at_epoch_ms")
            force_phase = event.get("force_phase")
            symbol = event.get("symbol", symbol)
            timeout_min = event.get("timeout_no_fill_min", TIMEOUT_NO_FILL_MIN)
        else:
            open_at_epoch_ms = getattr(event, "open_at_epoch_ms", None)
            force_phase = getattr(event, "force_phase", None)
            if isinstance(event, str) and event in {"preopen", "tick"}:
                force_phase = event
            symbol = getattr(event, "symbol", symbol)
            timeout_min = getattr(event, "timeout_no_fill_min", TIMEOUT_NO_FILL_MIN)

        if open_at_epoch_ms is not None:
            open_at_ny = datetime.fromtimestamp(open_at_epoch_ms / 1000, tz=ZoneInfo("UTC")).astimezone(
                ZoneInfo("America/New_York")
            )
        else:
            open_at_ny = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)

        preopen_start = open_at_ny - timedelta(minutes=5)
        preopen_end = open_at_ny
        tick_start = open_at_ny
        tick_end = open_at_ny + timedelta(minutes=timeout_min)

        if force_phase == "preopen" or preopen_start <= ny_now < preopen_end:
            resp = do_preopen(exchange, market_data, symbol, settings)
            resp.setdefault("status", "preopen_ok")
            return resp

        if force_phase == "tick" or tick_start <= ny_now < tick_end:
            resp = do_tick(exchange, symbol, settings, market_data, event)
            resp.setdefault("status", "waiting")
            return resp

        return {"status": "done", "reason": "out_of_window"}


# ---------------------------------------------------------------------------
# Legacy compatibility


def generateSignal(context: Mapping[str, Any]) -> dict:
    """Legacy compatibility wrapper.

    ``context`` is expected to provide ``exchange``, ``now_utc`` and ``event``
    items.  The function delegates to :meth:`LiquiditySweepStrategy.run`.
    """
    return LiquiditySweepStrategy().run(**context)
