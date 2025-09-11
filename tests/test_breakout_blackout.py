import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from strategies.breakout.impl import BreakoutStrategy, Signal


class DummyBroker:
    pass


class DummySettings:
    def get(self, key, default=None):
        return default

    def __getattr__(self, item):
        return None


class DummyExchange:
    def __init__(self):
        self.last_order = None

    def get_available_balance_usdt(self):
        return 0.0

    def get_symbol_filters(self, symbol):
        return {"LOT_SIZE": {"stepSize": "1", "minQty": "1"}, "PRICE_FILTER": {"tickSize": "1"}, "MIN_NOTIONAL": {"notional": "1"}}

    def round_price_to_tick(self, symbol, px):
        return px

    def round_qty_to_step(self, symbol, qty):
        return qty

    def place_entry_limit(self, symbol, side, price, qty, cid, timeInForce="GTC"):
        self.last_order = {"symbol": symbol, "side": side, "price": price, "qty": qty}
        return self.last_order


class DummySettingsFull(DummySettings):
    def __init__(self, data):
        self.data = data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __getattr__(self, item):
        return self.data.get(item)


# ---------------------------------------------------------------------------

def test_breakout_skipped_in_blackout(monkeypatch):
    broker = DummyBroker()
    settings = DummySettings()
    strategy = BreakoutStrategy(market_data=None, broker=broker, settings=settings)

    monkeypatch.setenv("BLACKOUT_TZ", "America/New_York")
    monkeypatch.setenv("BLACKOUT_WINDOWS", "08:25-09:40")

    ny = ZoneInfo("America/New_York")
    now_local = datetime(2023, 1, 1, 8, 30, tzinfo=ny)
    now_utc = now_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

    result = strategy.run(now_utc=now_utc)

    assert result["status"] == "skipped_blackout"


def test_breakout_runs_outside_blackout(monkeypatch):
    exchange = DummyExchange()
    settings = DummySettingsFull({"SYMBOL": "BTCUSDT"})
    strategy = BreakoutStrategy(market_data=None, broker=exchange, settings=settings)
    strategy.generate_signal = lambda now: Signal(action="BUY", price=1.0, time=now)

    monkeypatch.setenv("BLACKOUT_TZ", "America/New_York")
    monkeypatch.setenv("BLACKOUT_WINDOWS", "08:25-09:40")

    ny = ZoneInfo("America/New_York")
    now_local = datetime(2023, 1, 1, 9, 45, tzinfo=ny)
    now_utc = now_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

    result = strategy.run(exchange=exchange, settings=settings, now_utc=now_utc)

    assert result["status"] == "order_placed"
    assert exchange.last_order is not None
