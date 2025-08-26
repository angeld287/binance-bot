"""Common utilities shared by strategies.

This module centralises helpers that were previously scattered in the
``execution`` module.  The functions defined here are deliberately
side‑effect free so they can be reused by multiple strategies.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_UP
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time

# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def floor_to_step(x: float, step: float) -> float:
    """Floor ``x`` to the nearest multiple of ``step`` using ``Decimal`` precision."""
    step_dec = Decimal(str(step))
    x_dec = Decimal(str(x))
    return float((x_dec // step_dec) * step_dec)


def _ceil_to_step(x: float, step: float) -> float:
    step_dec = Decimal(str(step))
    x_dec = Decimal(str(x))
    return float((x_dec / step_dec).to_integral_value(rounding=ROUND_UP) * step_dec)


def get_symbol_filters(symbol: str, exchange_info: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Extract quantity/price limits for ``symbol`` from ``exchange_info``."""
    step_size = 1.0
    min_qty = 0.0
    max_qty = float("inf")
    min_notional = 0.0
    s_info = next((s for s in exchange_info.get("symbols", []) if s.get("symbol") == symbol), {})
    for f in s_info.get("filters", []):
        ftype = f.get("filterType")
        if ftype in ("LOT_SIZE", "MARKET_LOT_SIZE"):
            step_size = float(f.get("stepSize", step_size))
            min_qty = float(f.get("minQty", min_qty))
            max_qty = float(f.get("maxQty", max_qty))
        elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
            mn = f.get("minNotional") or f.get("notional")
            if mn is not None:
                min_notional = float(mn)
    return step_size, min_qty, max_qty, min_notional


def compute_qty_from_usdt(symbol: str, price: float, target_usdt: float, exchange_info: Dict[str, Any]) -> Tuple[float, float]:
    """Compute the order quantity so the notional is close to ``target_usdt``.

    The calculation respects the symbol filters obtained from
    ``exchange_info`` and will return a quantity and the resulting
    notional (qty * price).
    """
    step_size, min_qty, max_qty, min_notional = get_symbol_filters(symbol, exchange_info)
    if price <= 0 or target_usdt <= 0:
        return 0.0, 0.0

    qty = floor_to_step(target_usdt / price, step_size)
    notional = qty * price

    if notional < min_notional:
        qty = _ceil_to_step(min_notional / price, step_size)
        notional = qty * price
    if qty < min_qty:
        qty = _ceil_to_step(min_qty, step_size)
        notional = qty * price
    if qty > max_qty:
        qty = floor_to_step(max_qty, step_size)
        notional = qty * price

    diff = abs(notional - target_usdt) / target_usdt if target_usdt else 0.0
    if diff > 0.05 and notional > target_usdt:
        cand = floor_to_step(target_usdt / price, step_size)
        if cand * price >= min_notional:
            qty = cand
            notional = qty * price
    return qty, notional


# ---------------------------------------------------------------------------
# Position/order helpers (lightweight wrappers)
# ---------------------------------------------------------------------------

def get_position_mode(client) -> Optional[str]:  # pragma: no cover - thin wrapper
    """Return the position mode for the futures account if available."""
    try:
        resp = client.futures_get_position_mode()
        if resp and isinstance(resp, dict):
            return resp.get("dualSidePosition")
    except Exception:
        return None
    return None


def get_symbol_positions(client, symbol: str) -> List[Dict[str, Any]]:  # pragma: no cover
    try:
        return client.futures_position_information(symbol=symbol)
    except Exception:
        return []


def has_open_position(client, symbol: str) -> bool:  # pragma: no cover
    positions = get_symbol_positions(client, symbol)
    for p in positions:
        try:
            amt = float(p.get("positionAmt", 0))
        except Exception:
            amt = 0.0
        if amt != 0:
            return True
    return False


def list_open_orders(client, symbol: str) -> List[Dict[str, Any]]:  # pragma: no cover
    try:
        return client.futures_get_open_orders(symbol=symbol)
    except Exception:
        return []


def split_orders(open_orders: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split ``open_orders`` into entry and exit orders based on ``reduceOnly`` flag."""
    entries: List[Dict[str, Any]] = []
    exits: List[Dict[str, Any]] = []
    for o in open_orders:
        if o.get("reduceOnly") or o.get("closePosition"):
            exits.append(o)
        else:
            entries.append(o)
    return entries, exits


def cancel_orders(client, symbol: str, orders: Iterable[Dict[str, Any]]) -> None:  # pragma: no cover
    for o in orders:
        try:
            client.futures_cancel_order(symbol=symbol, orderId=o.get("orderId"))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Time helpers and guards (minimal implementations)
# ---------------------------------------------------------------------------

OFFSET_MS = 0


def now_ms() -> int:
    return int(time.time() * 1000) + OFFSET_MS


def sync_time(server_time_ms: int) -> None:
    global OFFSET_MS
    OFFSET_MS = server_time_ms - int(time.time() * 1000)


def notional_guard(min_notional: float, notional: float) -> bool:
    """Return ``True`` if ``notional`` is big enough."""
    return notional >= min_notional


def exclusive_entry_guard(flag: bool, has_position: bool) -> bool:
    """Basic exclusive-entry guard."""
    return not flag or not has_position


def recv_window_const(default: int = 5000) -> int:
    """Read RECV_WINDOW from env (light wrapper)."""
    import os
    return int(os.getenv("RECV_WINDOW", str(default)))


def detect_supports_resistances(*args: Any, **kwargs: Any):  # pragma: no cover
    """Placeholder for SR detection; real implementation lives elsewhere."""
    raise NotImplementedError

__all__ = [
    "floor_to_step",
    "compute_qty_from_usdt",
    "get_position_mode",
    "get_symbol_positions",
    "has_open_position",
    "list_open_orders",
    "split_orders",
    "cancel_orders",
    "now_ms",
    "sync_time",
    "OFFSET_MS",
    "notional_guard",
    "exclusive_entry_guard",
    "recv_window_const",
    "detect_supports_resistances",
]
