import logging
from uuid import uuid4

from binance.exceptions import BinanceAPIException

from common.utils import sanitize_client_order_id
from config.settings import get_stop_loss_pct, get_take_profit_pct


def run_cr_on_open_position(ctx, symbol: str, position: dict, logger=None) -> None:
    """Manage protective orders (SL/TP) for an open position."""

    logger = logger or logging.getLogger("bot.strategy.breakout")

    exchange = ctx.get("exchange") if ctx else None
    settings = ctx.get("settings") if ctx else None
    market_data = ctx.get("market_data") if ctx else None

    sl_pct = get_stop_loss_pct(settings) if settings else None
    tp_pct = get_take_profit_pct(settings) if settings else None
    logger.info("breakout.cr: pct_read {sl=%s, tp=%s}", sl_pct, tp_pct)
    if sl_pct is None or tp_pct is None:
        logger.info("breakout.cr: bracket_skipped {reason=pct_missing}")
        return None

    qty = float(position.get("positionAmt") or position.get("qty") or 0.0)
    entry = float(position.get("entryPrice") or position.get("entry") or 0.0)
    side = "LONG" if qty > 0 else "SHORT" if qty < 0 else None
    exit_side = "SELL" if qty > 0 else "BUY" if qty < 0 else None
    qty_abs = abs(qty)

    if not entry:
        try:
            entry = float(market_data.get_price(symbol)) if market_data else 0.0
        except Exception:  # pragma: no cover - defensive
            entry = 0.0
    if not entry:
        logger.warning("breakout.cr: bracket_skipped {reason=invalid_entry}")
        return None

    try:
        filters = exchange.get_symbol_filters(symbol) if exchange else {}
        min_qty = float(filters.get("LOT_SIZE", {}).get("minQty", 0.0))
    except Exception:  # pragma: no cover - defensive
        min_qty = 0.0
    if qty_abs < min_qty:
        logger.info("breakout.cr: bracket_skipped {reason=qty_below_min}")
        return None

    if qty > 0:  # long
        sl = entry * (1 - sl_pct / 100.0)
        tp = entry * (1 + tp_pct / 100.0)
    else:  # short
        sl = entry * (1 + sl_pct / 100.0)
        tp = entry * (1 - tp_pct / 100.0)

    if hasattr(exchange, "round_price_to_tick"):
        sl = exchange.round_price_to_tick(symbol, sl)
        tp = exchange.round_price_to_tick(symbol, tp)
    if hasattr(exchange, "round_qty_to_step"):
        qty_abs = exchange.round_qty_to_step(symbol, qty_abs)

    logger.info(
        "breakout.cr: targets {side=%s, entry=%s, sl=%s, tp=%s, qty=%s}",
        side,
        entry,
        sl,
        tp,
        qty_abs,
    )

    try:
        open_orders = exchange.open_orders(symbol) if exchange else []
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("breakout.cr: open_orders_fail {error=%s}", exc)
        open_orders = []

    opposite_orders = [
        o
        for o in open_orders
        if o.get("reduceOnly") and o.get("side") == exit_side
    ]
    sl_order = next((o for o in opposite_orders if o.get("type") == "STOP_MARKET"), None)
    tp_order = next(
        (o for o in opposite_orders if o.get("type") == "TAKE_PROFIT_MARKET"),
        None,
    )

    eps = entry * 0.001

    def _place(kind: str, price: float) -> None:
        cid = sanitize_client_order_id(f"brk-{kind}-{uuid4().hex[:8]}")
        if kind == "SL":
            if hasattr(exchange, "place_stop_reduce_only"):
                exchange.place_stop_reduce_only(symbol, exit_side, price, qty_abs, cid)
            else:  # pragma: no cover - generic fallback
                exchange._client.futures_create_order(  # type: ignore[attr-defined]
                    symbol=symbol,
                    side=exit_side,
                    type="STOP_MARKET",
                    stopPrice=price,
                    quantity=qty_abs,
                    reduceOnly=True,
                    newClientOrderId=cid,
                )
        else:  # TP
            if hasattr(exchange, "place_tp_reduce_only"):
                try:
                    exchange.place_tp_reduce_only(
                        symbol, exit_side, price, qty_abs, cid
                    )
                except BinanceAPIException as e:
                    if e.code == -2022:
                        logger.info(
                            "breakout.cr: tp_skip_no_remaining (ReduceOnly -2022)"
                        )
                        return
                    raise
            else:  # pragma: no cover - generic fallback
                exchange._client.futures_create_order(  # type: ignore[attr-defined]
                    symbol=symbol,
                    side=exit_side,
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=price,
                    quantity=qty_abs,
                    reduceOnly=True,
                    newClientOrderId=cid,
                )
        logger.info(
            "breakout.cr: bracket_placed {type=%s, stop=%s, qty=%s}",
            kind,
            price,
            qty_abs,
        )

    def _ensure(existing: dict | None, kind: str, price: float) -> None:
        if existing:
            try:
                existing_price = float(existing.get("stopPrice") or existing.get("price") or 0)
            except (TypeError, ValueError):
                existing_price = 0.0
            if abs(existing_price - price) <= eps:
                logger.info("breakout.cr: bracket_skipped {reason=exists, type=%s}", kind)
                return
            try:
                exchange.cancel_order(symbol, orderId=existing.get("orderId"))
            except Exception:  # pragma: no cover - defensive
                pass
        _place(kind, price)

    _ensure(sl_order, "SL", sl)
    _ensure(tp_order, "TP", tp)

    return None
