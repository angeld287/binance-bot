import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from execution import handle, strategy


class DummyClient:
    def __init__(self, klines):
        self._klines = klines

    def futures_klines(self, symbol, interval, limit):  # pragma: no cover - simple stub
        return self._klines


class LogCapture(list):
    def __call__(self, msg):
        self.append(msg)


def test_handle_invokes_strategy():
    # Klines crafted so that the last candle breaks above the previous high
    price = 110
    last_high = 100
    klines = [[0, 0, last_high, 90, price, 0]] * 51
    client = DummyClient(klines)
    logs = LogCapture()
    ctx = {"client": client, "symbol": "TESTUSDT", "log": logs}
    plan = handle(ctx)
    assert plan is not None
    assert any("[STRAT:breakout]" in m for m in logs)
