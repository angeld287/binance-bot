"""Estrategia aleatoria de demostración.

Decide entrar en LONG o SHORT de forma no determinista cuando no existe
una posición activa. Se provee únicamente con fines de prueba."""

import os
import random
from typing import Any, Dict, Optional

ANALYSIS_WINDOW = 12

_seeded = False


def generate_signal(
    exchange: Any, symbol: str, window: int = ANALYSIS_WINDOW, *args: Any, **kwargs: Any
):
    """Genera una señal aleatoria compatible con la estrategia original."""
    global _seeded
    if not _seeded:
        seed = os.getenv("RANDOM_STRATEGY_SEED")
        if seed:
            try:
                random.seed(int(seed))
            except ValueError:
                random.seed(seed)
        _seeded = True

    position_qty = 0.0
    try:
        ctx: Optional[Dict[str, Any]] = None
        for arg in args:
            if isinstance(arg, dict) and "position" in arg:
                ctx = arg
                break
        if ctx is None:
            ctx = kwargs.get("ctx")
        if ctx and isinstance(ctx, dict):
            pos = ctx.get("position") or {}
            qty_val = pos.get("qty") or pos.get("positionAmt") or 0
            position_qty = abs(float(qty_val))
    except Exception:
        position_qty = 0.0

    if position_qty > 0:
        print('{"event":"RANDOM_STRATEGY","decision":"FLAT","note":"active position"}')
        return None, None, [], (None, None)

    decision = "LONG" if random.random() < 0.5 else "SHORT"
    side = "buy" if decision == "LONG" else "sell"

    price: Optional[float] = None
    try:
        ticker = exchange.fetch_ticker(symbol)
        if ticker:
            last_price = ticker.get("last")
            price = float(last_price) if last_price is not None else None
    except Exception:
        try:
            mark = exchange.futures_mark_price(symbol=symbol.replace("/", ""))
            mark_price = mark.get("markPrice") if isinstance(mark, dict) else None
            price = float(mark_price) if mark_price is not None else None
        except Exception:
            price = None

    if price is not None and price > 0:
        price_range = (price * 0.99, price * 1.01)
    else:
        price_range = (None, None)

    print(f'{{"event":"RANDOM_STRATEGY","decision":"{decision}","note":"random pick"}}')
    return side, price, [], price_range
