import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any


class LiquiditySweepStrategy:
    """Single entrypoint strategy with NY time gating."""

    def __init__(self) -> None:
        self.symbol = os.getenv("SYMBOL", "BTCUSDT")
        self.risk_pct = float(os.getenv("RISK_PCT", "0.003"))
        self.risk_notional_usdt = float(os.getenv("RISK_NOTIONAL_USDT", "0"))
        self.timeout_no_fill_min = int(os.getenv("TIMEOUT_NO_FILL_MIN", "20"))
        self.microbuffer_pct_min = float(os.getenv("MICROBUFFER_PCT_MIN", "0.0002"))
        self.microbuffer_atr1m_mult = float(os.getenv("MICROBUFFER_ATR1M_MULT", "0.25"))
        self.buffer_sl_pct_min = float(os.getenv("BUFFER_SL_PCT_MIN", "0.0005"))
        self.buffer_sl_atr1m_mult = float(os.getenv("BUFFER_SL_ATR1M_MULT", "0.5"))
        self.tp_policy = os.getenv("TP_POLICY", "STRUCTURAL_OR_1_8R")
        self.max_lookback_min = int(os.getenv("MAX_LOOKBACK_MIN", "60"))

    # ---------------- Lógica principal ----------------
    def run(self, exchange, now_utc=None, event: Dict[str, Any] | None = None) -> Dict[str, Any]:
        event = event or {}
        now_utc = now_utc or datetime.now(timezone.utc)
        ny = ZoneInfo("America/New_York")
        ny_now = now_utc.astimezone(ny)

        open_ms = event.get("open_at_epoch_ms")
        if open_ms is not None:
            open_at = datetime.fromtimestamp(open_ms / 1000, ny)
        else:
            open_at = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)

        force = event.get("force_phase")
        if force == "preopen":
            return self._do_preopen(exchange, self.symbol, event, ny_now)
        if force == "tick":
            return self._do_tick(exchange, self.symbol, event, ny_now)

        pre_start = open_at - timedelta(minutes=5)
        if pre_start <= ny_now < open_at:
            return self._do_preopen(exchange, self.symbol, event, ny_now)
        if ny_now >= open_at:
            return self._do_tick(exchange, self.symbol, event, ny_now)
        return {"status": "idle", "reason": "out_of_window"}

    # ---------------- Lógica pura ----------------
    @staticmethod
    def _round_to_tick(price: float, tick: float) -> float:
        if tick == 0:
            return price
        return round(price / tick) * tick

    @staticmethod
    def compute_levels(candles_m1, now_ts, max_lookback_min, price_tick, price) -> dict:
        highs = [float(c[2]) for c in candles_m1]
        lows = [float(c[3]) for c in candles_m1]
        S = min(lows)
        R = max(highs)
        atr1m = sum(h - l for h, l in zip(highs, lows)) / max(len(candles_m1), 1)
        atr15m = atr1m * 15
        microbuffer = max(price * float(os.getenv("MICROBUFFER_PCT_MIN", "0.0002")), float(os.getenv("MICROBUFFER_ATR1M_MULT", "0.25")) * atr1m)
        buffer_sl = max(price * float(os.getenv("BUFFER_SL_PCT_MIN", "0.0005")), float(os.getenv("BUFFER_SL_ATR1M_MULT", "0.5")) * atr1m)
        S = LiquiditySweepStrategy._round_to_tick(S, price_tick)
        R = LiquiditySweepStrategy._round_to_tick(R, price_tick)
        return {"S": S, "R": R, "atr1m": atr1m, "atr15m": atr15m, "microbuffer": microbuffer, "buffer_sl": buffer_sl}

    @staticmethod
    def build_entry_orders(symbol, S, R, microbuffer, round_price) -> dict:
        buy_px = round_price(symbol, S + microbuffer)
        sell_px = round_price(symbol, R - microbuffer)
        return {"buy_px": buy_px, "sell_px": sell_px}

    @staticmethod
    def build_bracket(side, entry, S, R, microbuffer, buffer_sl, atr1m, tp_policy, round_price) -> dict:
        if side == "LONG":
            sl = round_price(S - buffer_sl)
            structural = round_price(R - microbuffer)
            risk = entry - sl
            rr = (structural - entry) / risk if risk else 0
            if rr >= 1.2:
                tp = structural
            else:
                tp = round_price(entry + 1.8 * risk)
        else:
            sl = round_price(R + buffer_sl)
            structural = round_price(S + microbuffer)
            risk = sl - entry
            rr = (entry - structural) / risk if risk else 0
            if rr >= 1.2:
                tp = structural
            else:
                tp = round_price(entry - 1.8 * risk)
        return {"sl": sl, "tp": tp}

    # ---------------- Acciones IO ----------------
    def _do_preopen(self, exchange, symbol, event, now_ny) -> dict:
        candles = exchange.get_klines(symbol, "1m", limit=self.max_lookback_min)
        price = float(candles[-1][4])
        filters = exchange.get_symbol_filters(symbol)
        price_tick = float(filters.get("tickSize", 1))
        levels = self.compute_levels(candles, None, self.max_lookback_min, price_tick, price)
        entries = self.build_entry_orders(symbol, levels["S"], levels["R"], levels["microbuffer"], exchange.round_price_to_tick)

        ny = ZoneInfo("America/New_York")
        open_ms = event.get("open_at_epoch_ms")
        if open_ms is not None:
            open_at = datetime.fromtimestamp(open_ms / 1000, ny)
        else:
            open_at = now_ny
        trade_id = f"{symbol}-{open_at.strftime('%Y%m%d')}-NY"
        buy_id = f"{trade_id}:pre:buy"
        sell_id = f"{trade_id}:pre:sell"

        qty = 1.0
        for side, price in [("BUY", entries["buy_px"]), ("SELL", entries["sell_px"])]:
            cid = buy_id if side == "BUY" else sell_id
            existing = exchange.get_order(symbol, clientOrderId=cid)
            if existing:
                existing_price = float(existing.get("price", 0))
                tick = float(filters.get("tickSize", 1))
                if abs(existing_price - price) > tick:
                    exchange.cancel_order(symbol, clientOrderId=cid)
                    exchange.place_limit(symbol, side, price, qty, cid)
            else:
                exchange.place_limit(symbol, side, price, qty, cid)

        return {
            "status": "preopen_ok",
            "trade_id": trade_id,
            "buy_px": entries["buy_px"],
            "sell_px": entries["sell_px"],
            "S": levels["S"],
            "R": levels["R"],
        }

    def _do_tick(self, exchange, symbol, event, now_ny) -> dict:
        ny = ZoneInfo("America/New_York")
        open_ms = event.get("open_at_epoch_ms")
        if open_ms is not None:
            open_at = datetime.fromtimestamp(open_ms / 1000, ny)
        else:
            open_at = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)

        trade_id = f"{symbol}-{open_at.strftime('%Y%m%d')}-NY"
        buy_id = f"{trade_id}:pre:buy"
        sell_id = f"{trade_id}:pre:sell"
        sl_id = f"{trade_id}:sl"
        tp_id = f"{trade_id}:tp"

        orders = {o["clientOrderId"]: o for o in exchange.open_orders(symbol)}
        buy_open = orders.get(buy_id)
        sell_open = orders.get(sell_id)

        if buy_open and sell_open:
            if now_ny > open_at + timedelta(minutes=self.timeout_no_fill_min):
                exchange.cancel_order(symbol, clientOrderId=buy_id)
                exchange.cancel_order(symbol, clientOrderId=sell_id)
                return {"status": "done", "reason": "timeout"}
            return {"status": "waiting"}

        # Determine which order filled
        filled_id = None
        active_id = None
        side = None
        if not buy_open and sell_open:
            info = exchange.get_order(symbol, clientOrderId=buy_id)
            status = info.get("status") if info else None
            if status == "FILLED":
                filled_id = buy_id
                active_id = sell_id
                side = "LONG"
            elif status in {"CANCELED", "EXPIRED"}:
                return {"status": "done", "reason": "preorder_cancelled"}
        elif not sell_open and buy_open:
            info = exchange.get_order(symbol, clientOrderId=sell_id)
            status = info.get("status") if info else None
            if status == "FILLED":
                filled_id = sell_id
                active_id = buy_id
                side = "SHORT"
            elif status in {"CANCELED", "EXPIRED"}:
                return {"status": "done", "reason": "preorder_cancelled"}
        else:
            # none open
            return {"status": "waiting"}

        if not filled_id:
            return {"status": "waiting"}

        entry_order = exchange.get_order(symbol, clientOrderId=filled_id)
        entry = float(entry_order.get("price", 0))
        # Simplified brackets
        filters = exchange.get_symbol_filters(symbol)
        price_tick = float(filters.get("tickSize", 1))
        S = entry - 10
        R = entry + 10
        microbuffer = max(entry * self.microbuffer_pct_min, self.microbuffer_atr1m_mult)
        buffer_sl = max(entry * self.buffer_sl_pct_min, self.buffer_sl_atr1m_mult)
        bracket = self.build_bracket(side, entry, S, R, microbuffer, buffer_sl, 1, self.tp_policy, lambda px: exchange.round_price_to_tick(symbol, px))

        balance = exchange.get_available_balance_usdt()
        if balance:
            risk = entry - bracket["sl"] if side == "LONG" else bracket["sl"] - entry
            qty = (self.risk_pct * balance) / risk if risk else 0
        elif self.risk_notional_usdt > 0:
            qty = self.risk_notional_usdt / entry
        else:
            qty = 1
        qty = exchange.round_qty_to_step(symbol, qty)

        if active_id:
            exchange.cancel_order(symbol, clientOrderId=active_id)
        exchange.place_sl_reduce_only(symbol, side, bracket["sl"], qty, sl_id)
        exchange.place_tp_reduce_only(symbol, side, bracket["tp"], qty, tp_id)

        return {
            "status": "done",
            "reason": "bracket_placed",
            "side": side,
            "entry": entry,
            "sl": bracket["sl"],
            "tp": bracket["tp"],
        }


def generateSignal(context: Dict[str, Any]) -> Dict[str, Any]:
    strat = LiquiditySweepStrategy()
    exchange = context.get("exchange")
    now_utc = context.get("now_utc")
    event = context.get("event")
    return strat.run(exchange=exchange, now_utc=now_utc, event=event)
