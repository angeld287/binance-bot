import logging
from types import SimpleNamespace

from strategies.breakout.cr_hook import run_cr_on_open_position


class DummySettings:
    def __init__(self, data):
        self.data = data
    def get(self, key, default=None):
        return self.data.get(key, default)


class DummyBroker:
    def __init__(self, open_orders=None):
        self._open = list(open_orders or [])
        self.placed = []
        self.cancelled = []
    def open_orders(self, symbol):
        return list(self._open)
    def place_stop_reduce_only(self, symbol, side, stopPrice, qty, clientOrderId):
        self.placed.append(("SL", stopPrice, qty, side))
    def place_tp_reduce_only(self, symbol, side, tpPrice, qty, clientOrderId):
        self.placed.append(("TP", tpPrice, qty, side))
    def cancel_order(self, symbol, orderId=None, clientOrderId=None):
        self.cancelled.append(orderId or clientOrderId)
    def get_symbol_filters(self, symbol):
        return {"LOT_SIZE": {"minQty": "0.001"}}
    def round_price_to_tick(self, symbol, px):
        return px
    def round_qty_to_step(self, symbol, qty):
        return qty


class DummyMD:
    def get_price(self, symbol):
        return 100.0


def _ctx(broker, settings=None):
    return {
        "exchange": broker,
        "settings": settings or DummySettings({"STOP_LOSS_PCT": 1, "TAKE_PROFIT_PCT": 1.5}),
        "market_data": DummyMD(),
    }


def test_places_orders_when_missing():
    brk = DummyBroker()
    ctx = _ctx(brk)
    position = {"positionAmt": "0.03", "entryPrice": "100"}
    run_cr_on_open_position(ctx, "BTCUSDT", position)
    assert ("SL", 99.0, 0.03, "SELL") in brk.placed
    assert ("TP", 101.5, 0.03, "SELL") in brk.placed


def test_skips_when_existing_within_epsilon():
    existing = [
        {"type": "STOP_MARKET", "side": "SELL", "stopPrice": "99", "reduceOnly": True, "orderId": 1},
        {"type": "TAKE_PROFIT_MARKET", "side": "SELL", "stopPrice": "101.5", "reduceOnly": True, "orderId": 2},
    ]
    brk = DummyBroker(open_orders=existing)
    ctx = _ctx(brk)
    position = {"positionAmt": "0.03", "entryPrice": "100"}
    run_cr_on_open_position(ctx, "BTCUSDT", position)
    assert brk.placed == []
