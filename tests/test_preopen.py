import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from core.preopen import do_preopen
import core.preopen as preopen


class DummyExchange:
    def __init__(self, open_orders=None, tick=1.0):
        self.open_orders = open_orders or []
        self.tick = tick
        self.created = []
        self.cancelled = []

    def futures_klines(self, symbol, interval, limit):
        self.klines_params = (symbol, interval, limit)
        return []

    def futures_exchange_info(self):
        return {
            "symbols": [
                {
                    "symbol": "TESTUSDT",
                    "filters": [{"filterType": "PRICE_FILTER", "tickSize": str(self.tick)}],
                }
            ]
        }

    def futures_get_open_orders(self, symbol):
        return list(self.open_orders)

    def futures_cancel_order(self, symbol, origClientOrderId=None):
        self.cancelled.append(origClientOrderId)
        self.open_orders = [o for o in self.open_orders if o.get("clientOrderId") != origClientOrderId]

    def futures_create_order(self, **kwargs):
        self.created.append(kwargs)
        self.open_orders.append({
            "clientOrderId": kwargs.get("clientOrderId"),
            "price": str(kwargs.get("price")),
        })
        return {"orderId": len(self.created)}


def _levels():
    return {"S": 100, "R": 200, "atr1m": 0, "atr15m": 0, "microbuffer": 5, "buffer_sl": 0}


def test_creates_orders(monkeypatch):
    ex = DummyExchange()
    monkeypatch.setattr(preopen, "compute_levels", lambda *a, **k: _levels())
    res = do_preopen(ex, "TEST/USDT", {"MAX_LOOKBACK_MIN": 10, "quantity": 1})
    assert res["status"] == "preopen_ok"
    assert len(ex.created) == 2
    assert ex.created[0]["side"] == "BUY"
    assert ex.created[1]["side"] == "SELL"


def test_idempotent_same_price(monkeypatch):
    ny = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
    trade_id = f"TESTUSDT-{ny}-NY"
    orders = [
        {"clientOrderId": f"{trade_id}:pre:buy", "price": "105"},
        {"clientOrderId": f"{trade_id}:pre:sell", "price": "195"},
    ]
    ex = DummyExchange(open_orders=orders)
    monkeypatch.setattr(preopen, "compute_levels", lambda *a, **k: _levels())
    do_preopen(ex, "TEST/USDT", {"MAX_LOOKBACK_MIN": 10, "quantity": 1})
    assert ex.created == []
    assert ex.cancelled == []


def test_recreate_when_price_changes(monkeypatch):
    ny = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
    trade_id = f"TESTUSDT-{ny}-NY"
    orders = [
        {"clientOrderId": f"{trade_id}:pre:buy", "price": "100"},
        {"clientOrderId": f"{trade_id}:pre:sell", "price": "195"},
    ]
    ex = DummyExchange(open_orders=orders)
    monkeypatch.setattr(preopen, "compute_levels", lambda *a, **k: _levels())
    do_preopen(ex, "TEST/USDT", {"MAX_LOOKBACK_MIN": 10, "quantity": 1})
    assert ex.cancelled == [f"{trade_id}:pre:buy"]
    assert len(ex.created) == 1
    assert ex.created[0]["clientOrderId"].endswith(":pre:buy")
