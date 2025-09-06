"""Liquidity sweep strategy skeleton.

This module provides a minimal implementation of the Liquidity Sweep
strategy.  It intentionally keeps side effects out of module import time and
splits pure computations from IO operations.  The goal is to serve as a
starting point for a more complete implementation.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from math import floor
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo


TIMEOUT_NO_FILL_MIN = 5  # minutes after NY open to keep processing ticks


# ---------------------------------------------------------------------------
# Pure helper functions


def compute_levels(order_book: Any, *args: Any, **kwargs: Any) -> list:
    """Return estimated liquidity levels from ``order_book``.

    This function is a placeholder.  A real implementation should analyse
    ``order_book`` and return a mapping with support/resistance levels plus
    auxiliary values such as ATR and buffers.  The :func:`do_preopen`
    implementation expects a dictionary with at least the following keys::

        {
            "S": float,              # Support level
            "R": float,              # Resistance level
            "atr1m": float,
            "atr15m": float,
            "microbuffer": float,   # Entry buffer
            "buffer_sl": float,     # Stop loss buffer
        }

    For the scope of this kata the function simply returns an empty dict so
    that :func:`do_preopen` can be exercised in isolation.
    """

    return {}


def build_entry_prices(levels: Iterable[Any], *args: Any, **kwargs: Any) -> list:
    """Derive entry prices from previously computed ``levels``.

    This function is pure and performs no IO.
    """
    return []


def build_bracket(entry_price: float, stop_loss: float, take_profit: float, *args: Any, **kwargs: Any) -> dict:
    """Construct a bracket order description.

    Pure function that returns a dictionary describing the bracket
    configuration.
    """
    return {}


# ---------------------------------------------------------------------------
# Internal IO helpers


def do_preopen(exchange: Any, symbol: str, settings: Any) -> dict:
    """Perform pre-open IO actions.

    The function fetches recent candle data to compute support and resistance
    levels and places two idempotent LIMIT orders around those levels.  Existing
    orders are updated only if the new price differs by more than one tick.

    Parameters
    ----------
    exchange:
        Object providing market-data and trading capabilities.  It must expose
        ``get_klines``/``fetch_ohlcv`` for candles, ``open_orders`` to query
        current orders, ``place_limit`` and ``cancel_order`` to manage orders as
        well as helpers ``get_symbol_filters`` and ``round_price_to_tick``.
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

    # ------------------------------------------------------------------
    # Fetch 1m candles
    if hasattr(exchange, "get_klines"):
        candles = exchange.get_klines(symbol=symbol, interval="1m", lookback_min=lookback)
    else:  # pragma: no cover - fallback for exchanges exposing ccxt-like API
        candles = exchange.fetch_ohlcv(symbol, timeframe="1m", limit=lookback)  # type: ignore[attr-defined]

    levels: Mapping[str, float] = compute_levels(candles, settings=settings) or {}
    S = float(levels.get("S", 0.0))
    R = float(levels.get("R", 0.0))
    microbuffer = float(levels.get("microbuffer", 0.0))

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

    qty = 1.0
    if hasattr(exchange, "round_qty_to_step"):
        qty = exchange.round_qty_to_step(symbol, qty)

    open_orders = exchange.open_orders(symbol)

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
        exchange.place_limit(symbol, side, price, qty, cid, timeInForce="GTC")

    _ensure_limit("BUY", buy_px, cid_buy)
    _ensure_limit("SELL", sell_px, cid_sell)

    return {
        "status": "preopen_ok",
        "trade_id": trade_id,
        "S": S,
        "R": R,
        "buy_px": buy_px,
        "sell_px": sell_px,
    }


def do_tick(exchange: Any, now_utc: datetime | None = None, event: Any | None = None) -> dict:
    """Perform IO actions for a regular tick/event."""
    return {}


# ---------------------------------------------------------------------------
# Strategy


class LiquiditySweepStrategy:
    """High level strategy coordinator."""

    def run(
        self, exchange: Any, now_utc: datetime | None = None, event: Any | None = None
    ) -> dict:
        """Execute the strategy and return a result dictionary."""
        utc_now = now_utc or datetime.now(tz=ZoneInfo("UTC"))
        ny_now = utc_now.astimezone(ZoneInfo("America/New_York"))

        open_at_epoch_ms = None
        force_phase = None
        if isinstance(event, Mapping):
            open_at_epoch_ms = event.get("open_at_epoch_ms")
            force_phase = event.get("force_phase")
        else:
            open_at_epoch_ms = getattr(event, "open_at_epoch_ms", None)
            force_phase = getattr(event, "force_phase", None)
            if isinstance(event, str) and event in {"preopen", "tick"}:
                force_phase = event

        if open_at_epoch_ms is not None:
            open_at_ny = datetime.fromtimestamp(open_at_epoch_ms / 1000, tz=ZoneInfo("UTC")).astimezone(
                ZoneInfo("America/New_York")
            )
        else:
            open_at_ny = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)

        preopen_start = open_at_ny - timedelta(minutes=5)
        preopen_end = open_at_ny
        tick_start = open_at_ny
        tick_end = open_at_ny + timedelta(minutes=TIMEOUT_NO_FILL_MIN)

        if force_phase == "preopen" or preopen_start <= ny_now < preopen_end:
            return do_preopen(exchange, utc_now)

        if force_phase == "tick" or tick_start <= ny_now < tick_end:
            return do_tick(exchange, utc_now, event)

        return {"status": "idle", "reason": "out_of_window"}


# ---------------------------------------------------------------------------
# Legacy compatibility


def generateSignal(context: Mapping[str, Any]) -> dict:
    """Legacy compatibility wrapper.

    ``context`` is expected to provide ``exchange``, ``now_utc`` and ``event``
    items.  The function delegates to :meth:`LiquiditySweepStrategy.run`.
    """
    exchange = context.get("exchange")
    now_utc = context.get("now_utc")
    event = context.get("event")
    return LiquiditySweepStrategy().run(exchange, now_utc=now_utc, event=event)
