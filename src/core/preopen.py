import math
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict


def compute_levels(*args: Any, **kwargs: Any) -> Dict[str, float]:
    """Placeholder for level computation.
    Should return a dictionary with keys:
    S, R, atr1m, atr15m, microbuffer, buffer_sl.
    """
    raise NotImplementedError


def _get_tick_size(exchange, symbol: str) -> float:
    info = exchange.futures_exchange_info()
    sym = symbol.replace("/", "")
    for s in info.get("symbols", []):
        if s.get("symbol") == sym:
            for f in s.get("filters", []):
                if f.get("filterType") == "PRICE_FILTER":
                    try:
                        return float(f.get("tickSize", 1.0))
                    except (TypeError, ValueError):
                        return 1.0
    return 1.0


def _align_price(price: float, tick: float, side: str) -> float:
    if tick <= 0:
        return price
    steps = price / tick
    if side.upper() == "BUY":
        return round(math.ceil(steps) * tick, 8)
    return round(math.floor(steps) * tick, 8)


def do_preopen(exchange, symbol: str, settings: Dict[str, Any]):
    """Pre-open routine.

    Obtains S/R levels and places two LIMIT orders in an idempotent way.
    """
    lookback = settings.get("MAX_LOOKBACK_MIN", 60)
    candles = exchange.futures_klines(symbol=symbol, interval="1m", limit=lookback)
    levels = compute_levels(exchange, symbol, settings, candles)
    S = levels.get("S")
    R = levels.get("R")
    micro = levels.get("microbuffer", 0)
    buy_px = S + micro
    sell_px = R - micro
    tick = _get_tick_size(exchange, symbol)
    buy_px = _align_price(buy_px, tick, "BUY")
    sell_px = _align_price(sell_px, tick, "SELL")
    ny = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
    trade_id = f"{symbol.replace('/', '')}-{ny}-NY"
    cid_buy = f"{trade_id}:pre:buy"
    cid_sell = f"{trade_id}:pre:sell"
    qty = settings.get("quantity") or settings.get("qty") or 0
    open_orders = exchange.futures_get_open_orders(symbol=symbol)

    def sync(side: str, price: float, cid: str):
        existing = next((o for o in open_orders if o.get("clientOrderId") == cid), None)
        if existing:
            try:
                old = float(existing.get("price", 0))
            except (TypeError, ValueError):
                old = 0
            if abs(old - price) <= tick:
                return
            exchange.futures_cancel_order(symbol=symbol, origClientOrderId=cid)
        exchange.futures_create_order(
            symbol=symbol,
            side=side,
            type="LIMIT",
            timeInForce="GTC",
            quantity=qty,
            price=price,
            clientOrderId=cid,
        )

    sync("BUY", buy_px, cid_buy)
    sync("SELL", sell_px, cid_sell)
    return {
        "status": "preopen_ok",
        "trade_id": trade_id,
        "S": S,
        "R": R,
        "buy_px": buy_px,
        "sell_px": sell_px,
    }
