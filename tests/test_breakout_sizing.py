import json
import math
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from strategies.breakout.impl import BreakoutStrategy, Signal


class DummyExchange:
    def __init__(self, *, price, step, tick, min_qty, min_notional, balance):
        self.price = price
        self.step = step
        self.tick = tick
        self.min_qty = min_qty
        self.min_notional = min_notional
        self.balance = balance
        self.last_order = None

    def get_available_balance_usdt(self):
        return self.balance

    def get_symbol_filters(self, symbol):
        return {
            "LOT_SIZE": {"stepSize": str(self.step), "minQty": str(self.min_qty)},
            "PRICE_FILTER": {"tickSize": str(self.tick)},
            "MIN_NOTIONAL": {"notional": str(self.min_notional)},
        }

    def round_price_to_tick(self, symbol, px):
        return math.floor(px / self.tick) * self.tick if self.tick else px

    def round_qty_to_step(self, symbol, qty):
        return math.floor(qty / self.step) * self.step if self.step else qty

    def place_entry_limit(self, symbol, side, price, qty, cid, timeInForce="GTC"):
        self.last_order = {"symbol": symbol, "side": side, "price": price, "qty": qty}
        return self.last_order


class DummySettings:
    def __init__(self, data):
        self.data = data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __getattr__(self, item):
        return self.data.get(item)


def _make_strategy(exchange, settings, price):
    strategy = BreakoutStrategy(market_data=None, broker=exchange, settings=settings)
    strategy.generate_signal = lambda now: Signal(action="BUY", price=price, time=now)
    return strategy


def test_breakout_uses_risk_notional(caplog):
    exchange = DummyExchange(price=0.22, step=1.0, tick=0.01, min_qty=1.0, min_notional=5.0, balance=0.0)
    settings = DummySettings({"SYMBOL": "DOGEUSDT", "RISK_NOTIONAL_USDT": 6.6, "RISK_PCT": 0})
    strategy = _make_strategy(exchange, settings, 0.22)

    with caplog.at_level("INFO", logger="bot.strategy.breakout"):
        strategy.run(exchange=exchange, settings=settings, now_utc=datetime.utcnow())

    assert exchange.last_order["qty"] == 30.0
    trace = json.loads(next(r.message for r in caplog.records if "sizing_trace" in r.message))
    trace = trace["sizing_trace"]
    assert trace["qty_target_src"] == "NOTIONAL"
    assert trace["notional"] >= trace["minNotional"]


def test_breakout_uses_risk_pct(caplog):
    exchange = DummyExchange(price=20.0, step=0.001, tick=0.1, min_qty=0.001, min_notional=5.0, balance=100.0)
    settings = DummySettings({"SYMBOL": "ETHUSDT", "RISK_NOTIONAL_USDT": 0.0, "RISK_PCT": 0.02})
    strategy = _make_strategy(exchange, settings, 20.0)

    with caplog.at_level("INFO", logger="bot.strategy.breakout"):
        strategy.run(exchange=exchange, settings=settings, now_utc=datetime.utcnow())

    assert exchange.last_order["qty"] == 0.25
    trace = json.loads(next(r.message for r in caplog.records if "sizing_trace" in r.message))
    trace = trace["sizing_trace"]
    assert trace["qty_target_src"] == "PCT"
    assert trace["notional"] >= trace["minNotional"]
