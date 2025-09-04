import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from strategies.liquidity_sweep.strategy import LiquiditySweepStrategy


class DummyExchange:
    def __init__(self):
        self.filters = {"tickSize": 1.0, "stepSize": 1.0, "minNotional": 5}
        self.orders = {}

    def round_price_to_tick(self, symbol, px):
        tick = self.filters["tickSize"]
        return round(px / tick) * tick

    def round_qty_to_step(self, symbol, qty):
        step = self.filters["stepSize"]
        return round(qty / step) * step

    def get_symbol_filters(self, symbol):
        return self.filters

    def get_klines(self, symbol, interval, start_ms=None, end_ms=None, limit=None):
        limit = limit or 1
        data = []
        base = 100.0
        for _ in range(limit):
            data.append([0, 0, base + 1, base - 1, base, 0, 0, 0, 0, 0, 0, 0])
        return data

    def open_orders(self, symbol):
        return [o for o in self.orders.values() if o["status"] == "NEW"]

    def get_order(self, symbol, clientOrderId=None, orderId=None):
        return self.orders.get(clientOrderId)

    def place_limit(self, symbol, side, price, qty, clientOrderId, timeInForce="GTC"):
        self.orders[clientOrderId] = {
            "clientOrderId": clientOrderId,
            "price": price,
            "qty": qty,
            "side": side,
            "status": "NEW",
        }

    def cancel_order(self, symbol, orderId=None, clientOrderId=None):
        if clientOrderId in self.orders:
            self.orders[clientOrderId]["status"] = "CANCELED"

    def place_sl_reduce_only(self, symbol, side, stopPrice, qty, clientOrderId):
        self.orders[clientOrderId] = {
            "clientOrderId": clientOrderId,
            "price": stopPrice,
            "qty": qty,
            "side": side,
            "status": "NEW",
            "type": "SL",
        }

    def place_tp_reduce_only(self, symbol, side, tpPrice, qty, clientOrderId):
        self.orders[clientOrderId] = {
            "clientOrderId": clientOrderId,
            "price": tpPrice,
            "qty": qty,
            "side": side,
            "status": "NEW",
            "type": "TP",
        }

    def get_available_balance_usdt(self):
        return 1000.0


def ny_dt(y, m, d, hh, mm):
    ny = ZoneInfo("America/New_York")
    return datetime(y, m, d, hh, mm, tzinfo=ny).astimezone(timezone.utc)


def test_preopen_places_limit_orders():
    ex = DummyExchange()
    strat = LiquiditySweepStrategy()
    now = ny_dt(2023, 1, 1, 9, 27)
    res = strat.run(exchange=ex, now_utc=now, event={})
    assert res["status"] == "preopen_ok"
    assert any(k.endswith(":pre:buy") for k in ex.orders)
    assert any(k.endswith(":pre:sell") for k in ex.orders)


def test_tick_waiting_when_both_new():
    ex = DummyExchange()
    strat = LiquiditySweepStrategy()
    pre = ny_dt(2023, 1, 1, 9, 27)
    strat.run(exchange=ex, now_utc=pre, event={})
    tick = ny_dt(2023, 1, 1, 9, 32)
    res = strat.run(exchange=ex, now_utc=tick, event={})
    assert res["status"] == "waiting"


def test_tick_buy_filled_places_bracket():
    ex = DummyExchange()
    strat = LiquiditySweepStrategy()
    pre = ny_dt(2023, 1, 1, 9, 27)
    strat.run(exchange=ex, now_utc=pre, event={})
    # mark buy filled
    for cid, o in list(ex.orders.items()):
        if o["side"] == "BUY":
            o["status"] = "FILLED"
    tick = ny_dt(2023, 1, 1, 9, 32)
    res = strat.run(exchange=ex, now_utc=tick, event={})
    assert res["status"] == "done"
    assert res["reason"] == "bracket_placed"
    assert any(k.endswith(":sl") for k in ex.orders)
    assert any(k.endswith(":tp") for k in ex.orders)


def test_tick_timeout_cancels_orders():
    ex = DummyExchange()
    strat = LiquiditySweepStrategy()
    pre = ny_dt(2023, 1, 1, 9, 27)
    strat.run(exchange=ex, now_utc=pre, event={})
    late = ny_dt(2023, 1, 1, 9, 55)
    res = strat.run(exchange=ex, now_utc=late, event={})
    assert res["status"] == "done"
    assert res["reason"] == "timeout"
    assert all(o["status"] != "NEW" for o in ex.orders.values())
